"""Animate the real mushroom body (MaleCNS v1.0 synapse geometry) lighting up while the fly module learns.

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

def render(frames, geom_path, out, fps=15, sub=3, spin_deg=70.0, gif=None):
    g = np.load(geom_path)
    KP, kidx, kcomp = g["kc_pts"], g["kc_idx"], g["kc_comp"]
    PP, pidx = g["pn_pts"], g["pn_idx"]; MP = g["mbon_pts"]
    allp = np.concatenate([KP, PP, MP]); center = allp.mean(0); span = np.abs(allp - center).max()
    K = frames[0]["kc"].shape[0]; n_pn = frames[0]["pn"].shape[0]; T = len(frames); t_rev = frames[0]["t_rev"]
    fig = plt.figure(figsize=(9, 10), facecolor=BG); ax = fig.add_axes([0, 0.04, 1, 0.86]); ax.set_facecolor(BG); ax.set_axis_off()
    ax.set_xlim(-span * 0.62, span * 0.62); ax.set_ylim(-span * 0.9, span * 0.9); ax.set_aspect("equal")
    title = fig.text(0.5, 0.965, "the fly's mushroom body, as a transformer learns through it", ha="center", color="#eee", fontsize=15)
    subt = fig.text(0.5, 0.94, "every dot is a real synapse from the MaleCNS v1.0 connectome  -  right hemisphere, 2,045 Kenyon cells", ha="center", color="#8a8f99", fontsize=9.5)
    cap = fig.text(0.5, 0.075, "", ha="center", color="#ddd", fontsize=12)
    cap2 = fig.text(0.5, 0.045, "", ha="center", color="#9aa", fontsize=10)
    fig.text(0.5, 0.022, "white flash = active Kenyon cells      orange = odor input in the calyx      green / red = dopamine burst (reward / punishment)", ha="center", color="#667", fontsize=8.5)
    fig.text(0.5, 0.006, "cyan / magenta = synapses whose fast weight now favors approach / avoid", ha="center", color="#667", fontsize=8.5)
    flip = fig.text(0.5, 0.5, "", ha="center", va="center", color="#f5c542", fontsize=34, fontweight="bold", alpha=0.0)
    glow_kc = np.zeros(K); glow_pn = np.zeros(n_pn); glow_dop = np.zeros(15); flip_a = [0.0]
    nfr = T * sub
    def upd(i):
        f = frames[i // sub]; first = (i % sub == 0)
        nonlocal glow_kc, glow_pn, glow_dop
        if first:
            glow_kc = np.maximum(glow_kc * 0.35, np.clip(f["kc"], 0, 1.5)); glow_pn = np.maximum(glow_pn * 0.35, np.clip(f["pn"], 0, 3) / 3)
            dmax = max(1e-6, float(np.percentile(np.abs(np.stack([fr["dop"] for fr in frames])), 97)))
            glow_dop = np.clip(f["dop"] / dmax, -1, 1)                    # burst, then fades over the sub-frames
            if f["t"] == t_rev: flip_a[0] = 1.0
        else:
            glow_kc *= 0.6; glow_pn *= 0.6; glow_dop *= 0.5
        flip_a[0] *= 0.93
        theta = np.deg2rad(spin_deg * (i / nfr) - spin_deg / 2)
        ax.cla(); ax.set_facecolor(BG); ax.set_axis_off(); ax.set_xlim(-span * 0.62, span * 0.62); ax.set_ylim(-span * 0.9, span * 0.9); ax.set_aspect("equal")
        # base tissue
        x, y, d = project(KP, theta, center); ax.scatter(x, y, s=1.0, c="#28406a", alpha=0.7, lw=0)
        xp, yp, _ = project(PP, theta, center); ax.scatter(xp, yp, s=0.7, c="#2a2f45", alpha=0.45, lw=0)
        xm, ym, _ = project(MP, theta, center); ax.scatter(xm, ym, s=0.8, c="#3a2d3a", alpha=0.35, lw=0)
        # learned synapses: fast weights per Kenyon cell (approach minus avoid), persistent tint
        H = f["H"]; pref = H[1] - H[0]; mag = np.abs(pref); thr = max(0.05, np.percentile(mag, 90))
        lit = mag[kidx] > thr
        if lit.any():
            col = np.where(pref[kidx][lit] > 0, "#19d3f3", "#ff3fd1"); a = np.clip(mag[kidx][lit] / (thr * 4), 0.15, 0.8)
            ax.scatter(x[lit], y[lit], s=9, c=col, alpha=0.12, lw=0); ax.scatter(x[lit], y[lit], s=3, c=col, alpha=float(a.mean()), lw=0)
        # dopamine per compartment: tint that compartment's Kenyon-cell synapses
        for c in range(15):
            v = glow_dop[c]
            if abs(v) < 0.03: continue
            m = kcomp == c; colr = "#2ecc71" if v > 0 else "#e74c3c"; a = float(np.clip(abs(v), 0, 1))
            ax.scatter(x[m], y[m], s=14, c=colr, alpha=0.10 * a, lw=0); ax.scatter(x[m], y[m], s=2.5, c=colr, alpha=0.55 * a, lw=0)
        # odor input in the calyx
        gp = glow_pn[pidx]; mp = gp > 0.05
        if mp.any(): ax.scatter(xp[mp], yp[mp], s=6, c="#ffb347", alpha=float(np.clip(gp[mp].mean(), 0.15, 0.7)), lw=0)
        # active Kenyon cells flash
        gk = glow_kc[kidx]; mk = gk > 0.05
        if mk.any():
            ax.scatter(x[mk], y[mk], s=22, c="#ffffff", alpha=0.08 * float(np.clip(gk[mk].mean(), 0, 1)), lw=0)
            ax.scatter(x[mk], y[mk], s=3.5, c="#fff8e1", alpha=float(np.clip(gk[mk].mean(), 0.2, 0.95)), lw=0)
        odor_name = ["odor A", "odor B", "odor C"][f["odor"]]; role = {1.0: "rewarding right now", -1.0: "punishing right now", 0.0: "neutral"}[f["role"]]
        act = "APPROACHES" if f["act"] == 1 else "declines"; outc = {1: "+1 reward", -1: "-1 punishment", 0: "nothing"}[int(round(f["r"]))] if f["act"] == 1 else "no outcome"
        cap.set_text(f"trial {f['t'] + 1} / {T}      {odor_name}  ({role})      agent {act}      {outc}")
        dom = np.argmax(np.abs(glow_dop)); cap2.set_text(f"dopamine strongest in compartment {COMP[dom]}     P(approach) = {f['p_approach']:.2f}" if np.abs(glow_dop).max() > 0.03 else f"P(approach) = {f['p_approach']:.2f}")
        flip.set_text("RULES FLIP" if f["t"] >= t_rev else ""); flip.set_alpha(float(flip_a[0]))
        return []
    ani = animation.FuncAnimation(fig, upd, frames=nfr, interval=1000 / fps, blit=False)
    ani.save(out, fps=fps, dpi=100, writer="ffmpeg", savefig_kwargs={"facecolor": BG}); print("->", out)
    if gif:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out, "-vf", "fps=12,scale=540:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=3", gif], check=True)
        print("->", gif)

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--ckpt", required=True); p.add_argument("--geom", default="data/mb_geom.npz"); p.add_argument("--out", default="report/demo_brain.mp4")
    p.add_argument("--gif", default=None); p.add_argument("--seed", type=int, default=7); p.add_argument("--fps", type=int, default=15); p.add_argument("--sub", type=int, default=3); p.add_argument("--spin", type=float, default=70)
    a = p.parse_args(); fr, mb = run_episode(a.ckpt, a.seed); print(f"episode: {sum(f['r'] for f in fr):+.0f} total reward, reversal at trial {fr[0]['t_rev']}")
    render(fr, a.geom, a.out, a.fps, a.sub, a.spin, a.gif)
