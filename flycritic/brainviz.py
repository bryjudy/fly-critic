"""Animate the real fly brain (MaleCNS v1.0 geometry) lighting up while the fly module learns: open on the whole brain,
zoom into the mushroom body, then play one episode.

Every dot is a real synapse position from the connectome (sampled). Kenyon cells flash when their odor code is
active, compartments glow green or red with the dopamine the model emits, and synapses that carry a fast weight stay
lit - cyan for approach, magenta for avoid. The mushroom body slowly rotates.

usage: uv run python -m flycritic.brainviz --ckpt runs/ch_mb_nofeat_kc_s0/ckpt.pt --out report/demo_brain.mp4 [--gif report/demo_brain.gif]
"""
import argparse, subprocess, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt; from matplotlib import animation
from .viz import run_episode, COMP

BG = "#050608"

def project(P, theta, center):
    """rotate about the vertical (z) axis by theta and return screen (x, y) plus depth."""
    Q = P - center; c, s = np.cos(theta), np.sin(theta)
    x = Q[:, 0] * c - Q[:, 1] * s; depth = Q[:, 0] * s + Q[:, 1] * c
    return x, Q[:, 2], depth

ODOR_STYLE = dict(title="a fruit fly's brain, as a transformer learns through it", flip="RULES FLIP",
                  intro="the mushroom body - where the fly learns what smells to trust")
TOOL_STYLE = dict(title="a fruit fly's brain, teaching a frozen LLM which tool to call", flip="TOOLS ROTATE",
                  intro="the mushroom body - the part of the fly that learns from reward")
CREDIT = "MaleCNS v1.0 connectome  |  github.com/bryjudy/fly-critic"

def caption_odor(f, T, glow_dop):
    odor_name = ["odor A", "odor B", "odor C"][f["odor"]]
    if f["act"] == 1: outc = {1: "+1 reward", -1: "-1 punishment", 0: "nothing"}[int(round(f["r"]))]; act = "approaches"
    else: outc = "no outcome"; act = "declines"
    return f"trial {f['t'] + 1}     {odor_name}     {act}     {outc}"

def caption_tool(f, T, glow_dop):
    tool = f["tools"][f["act"]] if f["act"] < len(f["tools"]) else "no tool"
    outc = "success" if f["r"] > 0 else ("failure" if f["r"] < 0 else "nothing")
    return f"trial {f['t'] + 1}     {f['qtype']} query     calls {tool}     {outc}"

def camera(P, center, el_deg, az_deg):
    """project 3D points: tilt el about the left-right axis (0 = frontal view, 90 = view from above), then spin az about
    the screen's vertical axis. Returns screen x, screen y."""
    Q = P - center; el, az = np.deg2rad(el_deg), np.deg2rad(az_deg)
    y = Q[:, 1] * np.cos(el) - Q[:, 2] * np.sin(el); z = Q[:, 1] * np.sin(el) + Q[:, 2] * np.cos(el); x = Q[:, 0]
    x2 = x * np.cos(az) + z * np.sin(az)
    return x2, -y

def ease(u): u = np.clip(u, 0, 1); return u * u * (3 - 2 * u)

def render(frames, geom_path, out, fps=15, sub=3, spin_deg=40.0, gif=None, style=ODOR_STYLE, caption=caption_odor,
           brain_path="data/brain_mesh.npz", intro_s=2.2, zoom_s=2.4):
    g = np.load(geom_path); b = np.load(brain_path)
    KP, kidx, kcomp = g["kc_pts"], g["kc_idx"], g["kc_comp"]; PP, pidx = g["pn_pts"], g["pn_idx"]; MP = g["mbon_pts"]
    BR, OL, KL = b["brain_pts"], b["ol_pts"], b["kcL_pts"]
    mb_all = np.concatenate([KP, PP, MP]); mb_c = (mb_all.min(0) + mb_all.max(0)) / 2
    sx, sy = camera(mb_all, mb_c, 90, 0); mb_span = 1.06 * max(np.abs(sx).max(), np.abs(sy).max() / 0.94)   # fits the whole mushroom body in the zoomed view
    brain_all = np.concatenate([BR, OL]); br_c = brain_all.mean(0); br_c[2] = mb_c[2]; br_span = np.abs(brain_all - br_c)[:, :2].max()
    K = frames[0]["kc"].shape[0]; n_pn = frames[0]["pn"].shape[0]; T = len(frames); t_rev = frames[0]["t_rev"]
    n_intro, n_zoom, n_ep = int(intro_s * fps), int(zoom_s * fps), T * sub; nfr = n_intro + n_zoom + n_ep
    fig = plt.figure(figsize=(9, 10), facecolor=BG); ax = fig.add_axes([0, 0.06, 1, 0.88]); ax.set_axis_off()
    fig.text(0.5, 0.962, style["title"], ha="center", color="#eee", fontsize=15)
    cap = fig.text(0.5, 0.036, "", ha="center", color="#ddd", fontsize=12.5)
    fig.text(0.985, 0.008, CREDIT, ha="right", color="#555", fontsize=7.5)
    flip = fig.text(0.5, 0.5, "", ha="center", va="center", color="#f5c542", fontsize=34, fontweight="bold", alpha=0.0)
    # inset: where the mushroom body sits in the whole brain (drawn once)
    ins = fig.add_axes([0.02, 0.07, 0.24, 0.16]); ins.set_facecolor(BG); ins.set_axis_off()
    for P, c in [(BR, "#3a4460"), (OL, "#3a5a48"), (KL, "#3a4460")]:
        x, y = camera(P, br_c, 0, 0); ins.scatter(x, y, s=0.15, c=c, alpha=0.6, lw=0)
    x, y = camera(KP, br_c, 0, 0); ins.scatter(x, y, s=0.3, c="#ffb347", alpha=0.9, lw=0); ins.set_aspect("equal")
    ins.set_xlim(-br_span, br_span); ins.set_ylim(-br_span * 0.75, br_span * 0.75); ins.set_visible(False)
    glow_kc = np.zeros(K); glow_pn = np.zeros(n_pn); glow_dop = np.zeros(15); flip_a = [0.0]
    D = np.stack([fr["dop"] for fr in frames]); dop_mean = D.mean(0); dmax = max(1e-6, float(np.percentile(np.abs(D - dop_mean), 97)))
    def upd(i):
        nonlocal glow_kc, glow_pn, glow_dop
        if i < n_intro: u = 0.0
        elif i < n_intro + n_zoom: u = ease((i - n_intro) / n_zoom)
        else: u = 1.0
        ep_i = i - n_intro - n_zoom
        center = br_c * (1 - u) + mb_c * u; half = br_span * (1 - u) + mb_span * u; el = 90 * u
        az = spin_deg / 2 * np.sin(2 * np.pi * max(ep_i, 0) / n_ep) if ep_i >= 0 else 0.0
        ax.cla(); ax.set_facecolor(BG); ax.set_axis_off(); ax.set_xlim(-half, half); ax.set_ylim(-half * 0.94, half * 0.94); ax.set_aspect("equal")
        ghost = 0.55 * (1 - u) + 0.18 * u; sz = 0.5 + 1.5 * u
        for P, c in [(BR, "#3a4460"), (OL, "#3a5a48"), (KL, "#3a4460")]:
            x, y = camera(P, center, el, az); ax.scatter(x, y, s=sz * 0.5, c=c, alpha=ghost, lw=0)
        x, y = camera(KP, center, el, az); xp, yp = camera(PP, center, el, az); xm, ym = camera(MP, center, el, az)
        if ep_i < 0:   # intro: the whole mushroom body glows so people can see where it is
            pulse = 0.55 + 0.35 * np.sin(2 * np.pi * i / (fps * 1.6))
            ax.scatter(x, y, s=6 * (1 + 2 * u), c="#ffb347", alpha=0.10 * pulse, lw=0); ax.scatter(x, y, s=0.8 * (1 + 2 * u), c="#ffd9a0", alpha=0.85 * pulse, lw=0)
            ax.scatter(xp, yp, s=0.8, c="#ffd9a0", alpha=0.6 * pulse, lw=0)
            cap.set_text(style["intro"] if u < 0.5 else "2,045 Kenyon cells, 15 dopamine compartments, every dot a real synapse")
            flip.set_alpha(0.0); ins.set_visible(False); return []
        ins.set_visible(True)
        f = frames[ep_i // sub]; first = (ep_i % sub == 0)
        if first:
            glow_kc = np.maximum(glow_kc * 0.35, np.clip(f["kc"], 0, 1.5)); glow_pn = np.maximum(glow_pn * 0.35, np.clip(f["pn"], 0, 3) / 3)
            glow_dop = np.clip((f["dop"] - dop_mean) / dmax, -1, 1)
            if f["t"] == t_rev: flip_a[0] = 1.0
        else:
            glow_kc *= 0.6; glow_pn *= 0.6; glow_dop *= 0.5
        flip_a[0] *= 0.93
        ax.scatter(x, y, s=1.0, c="#28406a", alpha=0.7, lw=0); ax.scatter(xp, yp, s=0.7, c="#2a2f45", alpha=0.45, lw=0); ax.scatter(xm, ym, s=0.8, c="#3a2d3a", alpha=0.35, lw=0)
        H = f["H"]; pref = H[1] - H[0]; mag = np.abs(pref); thr = max(0.05, np.percentile(mag, 90)); lit = mag[kidx] > thr
        if lit.any():
            col = np.where(pref[kidx][lit] > 0, "#19d3f3", "#ff3fd1"); a = np.clip(mag[kidx][lit] / (thr * 4), 0.15, 0.8)
            ax.scatter(x[lit], y[lit], s=9, c=col, alpha=0.12, lw=0); ax.scatter(x[lit], y[lit], s=3, c=col, alpha=float(a.mean()), lw=0)
        for c in range(15):
            v = glow_dop[c]
            if abs(v) < 0.03: continue
            m = kcomp == c; colr = "#2ecc71" if v > 0 else "#e74c3c"; a = float(np.clip(abs(v), 0, 1))
            ax.scatter(x[m], y[m], s=14, c=colr, alpha=0.10 * a, lw=0); ax.scatter(x[m], y[m], s=2.5, c=colr, alpha=0.55 * a, lw=0)
        gp = glow_pn[pidx]; mp = gp > 0.05
        if mp.any(): ax.scatter(xp[mp], yp[mp], s=6, c="#ffb347", alpha=float(np.clip(gp[mp].mean(), 0.15, 0.7)), lw=0)
        gk = glow_kc[kidx]; mk = gk > 0.05
        if mk.any():
            ax.scatter(x[mk], y[mk], s=22, c="#ffffff", alpha=0.08 * float(np.clip(gk[mk].mean(), 0, 1)), lw=0)
            ax.scatter(x[mk], y[mk], s=3.5, c="#fff8e1", alpha=float(np.clip(gk[mk].mean(), 0.2, 0.95)), lw=0)
        cap.set_text(caption(f, T, glow_dop)); flip.set_text(style["flip"] if f["t"] >= t_rev else ""); flip.set_alpha(float(flip_a[0]))
        return []
    ani = animation.FuncAnimation(fig, upd, frames=nfr, interval=1000 / fps, blit=False)
    ani.save(out, fps=fps, dpi=100, writer="ffmpeg", savefig_kwargs={"facecolor": BG}); print("->", out)
    if gif:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-vf", "fps=8,scale=380:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=48[p];[s1][p]paletteuse=dither=none", gif], check=True)
        print("->", gif)

def run_toolworld_episode(ckpt, seed=7):
    """Replay one ToolWorld episode from a checkpoint trained with --env toolworld (LLM features come from the cached table)."""
    import torch
    from .llm.adapter import ToolWorldEnv
    from .llm.toolworld import TOOLS, QUERY_TYPES, PARAPHRASES
    from .mb import MushroomBody
    from .model import Agent
    ck = torch.load(ckpt, map_location="cpu", weights_only=False); a = ck["args"]
    torch.manual_seed(seed)
    env = ToolWorldEnv(1, T=a["T"], seed=seed, llm=a["llm"])
    mb = MushroomBody(shuffle=(a["critic"] == "shuffled" or a.get("randkc", False)), seed=a["seed"], collapse=a.get("collapse", False), uniform=a.get("uniform", False))
    ag = Agent(env.obs_dim, env.n_actions, critic=a["critic"], mb=mb, d=a["d"], layers=a["layers"], heads=a.get("heads", 4), ctx=a["T"],
               no_feat=a.get("no_feat", False), no_plastic=a.get("no_plastic", False), pre=a.get("pre", "h"), prior=a.get("prior", False))
    ag.load_state_dict(ck["agent"]); ag.eval()
    obs = env.reset(); st = ag.init_state(1, "cpu"); dop_prev = torch.zeros(1, ag.plastic.C); frames = []
    with torch.no_grad():
        for t in range(env.T):
            tw = env.tw; q = int(tw.query_type[0]); query = PARAPHRASES[QUERY_TYPES[q]][int(tw.para[0])]; best = TOOLS[int(tw.best[0, q])]
            logits, value, pre, x = ag.act(st, obs, dop_prev)
            p = torch.softmax(logits, -1)[0]; act = int(torch.argmax(p))
            kc = st["mb"]["kc"][0].clone(); pn_in = obs["pn"][0].clone()
            obs, r, done, info = env.step(torch.tensor([act]))
            dop, _ = ag.dopamine(st, r, value); ag.plastic_update(st, pre, torch.tensor([act]), dop); dop_prev = dop
            frames.append(dict(t=t, qtype=QUERY_TYPES[q], query=query, best=best, tools=TOOLS, act=act, p_act=float(p[act]), r=float(r[0]),
                               kc=kc.numpy(), pn=pn_in.numpy(), dop=dop[0].numpy(), H=st["H"][0].sum(0).numpy(), post_rev=bool(info["post_reversal"][0]),
                               t_rev=int(env.t_rev[0])))
    return frames

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--geom", default="data/mb_geom.npz"); p.add_argument("--out", default="report/demo_brain.mp4")
    p.add_argument("--gif", default=None); p.add_argument("--seed", type=int, default=7); p.add_argument("--fps", type=int, default=15); p.add_argument("--sub", type=int, default=3); p.add_argument("--spin", type=float, default=40); p.add_argument("--brain", default="data/brain_mesh.npz"); p.add_argument("--env", default="odor", choices=["odor", "toolworld"])
    p.add_argument("--best_of", type=int, default=1, help="replay this many seeds (starting at --seed) and animate the highest-reward episode")
    a = p.parse_args()
    run = (lambda s: run_toolworld_episode(a.ckpt, s)) if a.env == "toolworld" else (lambda s: run_episode(a.ckpt, s)[0])
    style, cap = (TOOL_STYLE, caption_tool) if a.env == "toolworld" else (ODOR_STYLE, caption_odor)
    cands = [run(a.seed + k) for k in range(a.best_of)]
    fr = max(cands, key=lambda fs: sum(f["r"] for f in fs))
    print(f"episode: {sum(f['r'] for f in fr):+.2f} total reward, reversal at trial {fr[0]['t_rev']}, {sum(f['r'] > 0 for f in fr)} successes")
    render(fr, a.geom, a.out, a.fps, a.sub, a.spin, a.gif, style=style, caption=cap, brain_path=a.brain)
