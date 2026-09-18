"""Render an animation of the fly module learning: Kenyon cells lighting up, dopamine per compartment, the fast
weights changing, and the agent's choices - across one episode with a mid-episode reversal.
usage: uv run python -m flycritic.viz --ckpt runs/ch_mb_nofeat_kc_s0/ckpt.pt --out report/demo_odor.mp4"""
import argparse, math, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib import animation
from .env import OdorChoice
from .mb import MushroomBody
from .model import Agent

COMP = ["g1","g2","g3","g4","g5","a1","a2","a3","b1","b2","a'1","a'2","a'3","b'1","b'2"]

def run_episode(ckpt, seed=7):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False); a = ck["args"]
    torch.manual_seed(seed)
    env = OdorChoice(1, T=a["T"], seed=seed)
    mb = MushroomBody(shuffle=(a["critic"] == "shuffled"), seed=a["seed"])
    ag = Agent(env.obs_dim, env.n_actions, critic=a["critic"], mb=mb, d=a["d"], layers=a["layers"], ctx=a["T"],
               no_feat=a.get("no_feat", False), no_plastic=a.get("no_plastic", False), pre=a.get("pre", "h"), prior=a.get("prior", False))
    ag.load_state_dict(ck["agent"]); ag.eval()
    obs = env.reset(); st = ag.init_state(1, "cpu"); dop_prev = torch.zeros(1, ag.plastic.C)
    frames = []
    with torch.no_grad():
        for t in range(env.T):
            odor_idx = int(env.cur[0]); roles = env.roles[0].clone()
            logits, value, pre, x = ag.act(st, obs, dop_prev)
            p = torch.softmax(logits, -1)[0]; act = int(torch.argmax(p))
            kc = st["mb"]["kc"][0].clone(); pn_in = obs["pn"][0].clone()
            obs, r, done, info = env.step(torch.tensor([act]))
            dop, _ = ag.dopamine(st, r, value); ag.plastic_update(st, pre, torch.tensor([act]), dop); dop_prev = dop
            H = st["H"][0]                                    # (C, out, K)
            frames.append(dict(t=t, odor=odor_idx, role=float(roles[odor_idx]), act=act, p_approach=float(p[1]), r=float(r[0]),
                               kc=kc.numpy(), pn=pn_in.numpy(), dop=dop[0].numpy(), H=H.sum(0).numpy(), post_rev=bool(info["post_reversal"][0]),
                               t_rev=int(env.t_rev[0])))
    return frames, mb

def render(frames, mb, out, fps=6):
    K = frames[0]["kc"].shape[0]; side = int(math.ceil(math.sqrt(K)))
    grid = np.zeros(side * side)
    fig = plt.figure(figsize=(13, 7.2), facecolor="#0e0f13")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], width_ratios=[1.1, 1, 1.2], hspace=0.45, wspace=0.35)
    axk = fig.add_subplot(gs[0, 0]); axd = fig.add_subplot(gs[0, 1]); axh = fig.add_subplot(gs[0, 2]); axb = fig.add_subplot(gs[1, :])
    for ax in (axk, axd, axh, axb): ax.set_facecolor("#0e0f13"); [s.set_color("#444") for s in ax.spines.values()]; ax.tick_params(colors="#bbb", labelsize=8)
    im = axk.imshow(grid.reshape(side, side), cmap="magma", vmin=0, vmax=1.5, interpolation="nearest"); axk.set_xticks([]); axk.set_yticks([])
    axk.set_title("Kenyon cells (2,045)\nthe odor's sparse code", color="#eee", fontsize=9)
    bars = axd.bar(range(15), np.zeros(15), color="#888"); axd.set_ylim(-1.05, 1.05); axd.set_xticks(range(15)); axd.set_xticklabels(COMP, fontsize=7)
    axd.axhline(0, color="#666", lw=0.6); axd.set_title("dopamine per compartment\n(+ reward, - punishment)", color="#eee", fontsize=9)
    H0 = frames[0]["H"]; hm = axh.imshow(np.zeros((3, 64)), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto", interpolation="nearest")
    axh.set_yticks(range(3)); axh.set_yticklabels(["avoid", "approach", "value"], fontsize=8); axh.set_xticks([])
    axh.set_title("fast weights to actions\n(top-64 Kenyon cells, summed over compartments)", color="#eee", fontsize=9)
    T = len(frames); ts = np.arange(T)
    axb.set_xlim(0, T); axb.set_ylim(-1.3, 1.3); axb.set_xlabel("trial", color="#bbb"); axb.set_yticks([-1, 0, 1]); axb.set_yticklabels(["punished", "declined", "rewarded"], fontsize=8)
    axb.axvline(frames[0]["t_rev"], color="#f5c542", lw=1.2, ls="--"); axb.text(frames[0]["t_rev"] + 0.6, 1.12, "rules flip", color="#f5c542", fontsize=9)
    sc = axb.scatter([], [], s=28); pl, = axb.plot([], [], color="#4fc3f7", lw=1.2, alpha=0.9); axb.set_title("what happened each trial  (blue line = P(approach) for the current odor)", color="#eee", fontsize=10)
    txt = fig.text(0.5, 0.955, "", ha="center", color="#eee", fontsize=13)
    sub = fig.text(0.5, 0.925, "a frozen transformer whose only teacher is dopamine from the fly's mushroom body circuit", ha="center", color="#999", fontsize=9)
    active_hist = np.zeros(K)
    def upd(i):
        f = frames[i]; kc = f["kc"]; grid[:] = 0; grid[:K] = np.clip(kc, 0, 1.5); im.set_data(grid.reshape(side, side))
        for b, v in zip(bars, f["dop"]): b.set_height(float(v)); b.set_color("#2ecc71" if v > 0 else "#e74c3c")
        nonlocal active_hist; active_hist = 0.9 * active_hist + kc; top = np.argsort(-active_hist)[:64]
        hm.set_data(f["H"][:, top])
        xs = ts[:i + 1]; ys = np.array([fr["r"] if fr["act"] == 1 else 0.0 for fr in frames[:i + 1]])
        cols = ["#2ecc71" if y > 0 else ("#e74c3c" if y < 0 else "#555") for y in ys]
        sc.set_offsets(np.c_[xs, ys]); sc.set_color(cols); pl.set_data(xs, [2 * fr["p_approach"] - 1 for fr in frames[:i + 1]])
        odor_name = ["odor A", "odor B", "odor C"][f["odor"]]; role = {1.0: "currently rewarding", -1.0: "currently punishing", 0.0: "neutral"}[f["role"]]
        txt.set_text(f"trial {f['t']+1:3d}   {odor_name} ({role})   agent {'APPROACHES' if f['act']==1 else 'declines'}   outcome {f['r']:+.0f}")
        return [im, hm, sc, pl, txt, *bars]
    ani = animation.FuncAnimation(fig, upd, frames=T, interval=1000 / fps, blit=False)
    ani.save(out, fps=fps, dpi=110, writer="ffmpeg", savefig_kwargs={"facecolor": "#0e0f13"}); print("->", out)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--out", default="report/demo_odor.mp4"); p.add_argument("--seed", type=int, default=7); p.add_argument("--fps", type=int, default=6)
    a = p.parse_args(); fr, mb = run_episode(a.ckpt, a.seed); print(f"episode: {sum(f['r'] for f in fr):+.0f} total reward, reversal at trial {fr[0]['t_rev']}"); render(fr, mb, a.out, a.fps)
