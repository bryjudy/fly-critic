"""Figures for the report. usage: uv run python -m flycritic.figures  -> report/fig_*.png"""
import json, glob, re, os, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
os.makedirs("report", exist_ok=True)
def load(prefix):
    g = collections.defaultdict(list)
    for f in sorted(glob.glob(f"runs/{prefix}*/log.jsonl")):
        cfg = re.sub(r"_s\d+$", "", f.split("/")[1]); rows = [json.loads(l) for l in open(f)]
        if len(rows) >= 50: g[cfg].append(np.array([r["reward"] for r in rows]))
    return g
def curves(g, keys, labels, title, out, oracle=33.5, smooth=15):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, lab in zip(keys, labels):
        if k not in g: continue
        n = min(len(r) for r in g[k]); R = np.stack([r[:n] for r in g[k]]); m = R.mean(0); sd = R.std(0)
        ker = np.ones(smooth) / smooth; ms = np.convolve(m, ker, "valid"); x = np.arange(len(ms)) + smooth // 2
        ax.plot(x, ms, label=f"{lab} (n={len(R)})"); 
        if len(R) > 1: ax.fill_between(x, np.convolve(m - sd, ker, "valid"), np.convolve(m + sd, ker, "valid"), alpha=0.15)
    ax.axhline(oracle, color="k", ls=":", lw=0.8); ax.text(2, oracle + 0.4, "oracle", fontsize=8)
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("PPO iteration (64 episodes each)"); ax.set_ylabel("reward per episode")
    ax.set_title(title); ax.legend(fontsize=8, loc="center right"); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig); print("->", out)
def bars(g, keys, labels, title, out, last=100):
    fig, ax = plt.subplots(figsize=(7, 3.8)); xs = np.arange(len(keys)); means, sds = [], []
    for k in keys:
        v = [r[-last:].mean() for r in g.get(k, [])]; means.append(np.mean(v) if v else 0); sds.append(np.std(v) if len(v) > 1 else 0)
    ax.bar(xs, means, yerr=sds, capsize=3, color=["#3b6fb6" if m > 3 else "#b0b0b0" for m in means])
    ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8); ax.set_ylabel(f"reward/episode (last {last} iters)")
    ax.axhline(33.5, color="k", ls=":", lw=0.8); ax.set_title(title); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig); print("->", out)
g = load("")
main_keys = ["ch_mb_kc", "ch_mb_nofeat_kc", "ch_shuf_nofeat_kc", "ch_mb_nofeat", "ch_scalar", "ch_none"]
main_lab = ["MB critic, outputs visible", "MB dopamine only (KC fast weights)", "shuffled wiring, dopamine only",
            "MB dopamine only (hidden-state fast weights)", "scalar TD critic", "no critic"]
curves(g, main_keys, main_lab, "OdorChoice with reversal: learning curves", "report/fig1_learning_curves.png")
abl_keys = ["ch_mb_nofeat_kc", "ch_uniform_nofeat_kc", "ch_collapse_nofeat_kc"]
abl_lab = ["15 compartments,\nfly timescales", "15 compartments,\nuniform timescale", "1 compartment"]
bars(g, abl_keys, abl_lab, "Compartment ablation (dopamine-only agents)", "report/fig3_ablation.png")
curves(g, abl_keys, [l.replace("\n", " ") for l in abl_lab], "Compartment ablation: learning curves", "report/fig3b_ablation_curves.png")
if any(k.startswith("long_") for k in g):
    curves(g, ["long_mb_nofeat_kc"], ["MB dopamine only, 1200 iters"], "Dopamine-only agent: long training", "report/fig4_plateau.png")
cx_keys = ["cx2_mb_kc_ctx", "cx2_mb_nofeat_kc_ctx", "cx2_mb_kc", "cx2_mb_nofeat_kc", "cx2_none"]
cx_lab = ["outputs visible, context->KC", "dopamine only, context->KC", "outputs visible, context->transformer only",
          "dopamine only, context->transformer only", "no critic"]
if any(k.startswith("cx2_") for k in g):
    curves(g, cx_keys, cx_lab, "Context-dependent task (v2): learning curves", "report/fig5_context.png", oracle=66.3)
hh_keys = ["long_mb_nofeat_kc", "hh_hybrid_kc", "hh_learned_kc_prior", "hh_learned_randkc_prior", "hh_learned_kc", "hh_learned_h", "hh_none"]
hh_lab = ["connectome dopamine (fixed circuit), KC fast weights", "hybrid: connectome dopamine + learned residual",
          "learned modulator, KC fast weights, fly priors", "learned modulator, RANDOM expansion, fly priors",
          "learned modulator, KC fast weights", "learned modulator, hidden-state fast weights", "in-context RL (no plastic layer)"]
if any(k.startswith("hh_") for k in g):
    curves(g, hh_keys, hh_lab, "Head-to-head at 1200 iterations (OdorChoice)", "report/fig7_head_to_head.png")
    bars(g, hh_keys, [l.replace(", ", ",\n") for l in hh_lab], "Head-to-head: final reward (3 seeds each)", "report/fig7b_head_to_head_bars.png")
# step 1: external benchmarks
for env, oracle, title in [("bandit", 180.0, "10-arm non-stationary bandit (oracle 180, tuned UCB 140, Thompson 123)"),
                           ("darkroom", None, "Dark Room (AD)"), ("keydoor", None, "Dark Key-to-Door (AD)")]:
    keys = [f"b1_{env}_learned_kc_prior", f"b1_{env}_mb_nofeat_kc", f"b1_{env}_none", f"b1_{env}_none_hi", f"b1_{env}_none_big"]
    labs = ["learned modulator + KC fast weights (fly priors)", "fixed connectome dopamine + KC fast weights",
            "in-context RL (default)", "in-context RL (lr 1e-3, ent 0.03)", "in-context RL (6 layers)"]
    if any(k in g for k in keys):
        curves(g, keys, labs, f"Step 1: {title}", f"report/fig8_{env}.png", oracle=oracle if oracle else 1e9)
# step 2: frozen LLM tool world
tw_keys = ["tw_learned_kc_prior", "tw_mb_nofeat_kc", "tw_none", "tw_none_hi"]
tw_lab = ["frozen LLM + learned modulator + KC fast weights", "frozen LLM + fixed connectome dopamine",
          "frozen LLM + in-context transformer head", "frozen LLM + in-context head (lr 1e-3, ent 0.03)"]
if any(k in g for k in tw_keys):
    curves(g, tw_keys, tw_lab, "Step 2: ToolWorld with a frozen 0.5B LLM (oracle 105)", "report/fig9_toolworld.png", oracle=105.0)
# per-compartment learned rates from the seed-0 dopamine-only checkpoint
import torch
ck = "runs/ch_mb_nofeat_kc_s0/ckpt.pt"
if os.path.exists(ck):
    sd = torch.load(ck, map_location="cpu", weights_only=False)["agent"]
    comp = ["g1","g2","g3","g4","g5","a1","a2","a3","b1","b2","a'1","a'2","a'3","b'1","b'2"]
    alpha = sd["plastic.alpha"][:, :, 0].numpy(); tau = sd["plastic.log_tau"].exp().numpy()
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4)); x = np.arange(15)
    axs[0].bar(x - 0.2, alpha[:, 1], 0.4, label="approach"); axs[0].bar(x + 0.2, alpha[:, 0], 0.4, label="avoid")
    axs[0].set_xticks(x); axs[0].set_xticklabels(comp, fontsize=7); axs[0].set_title("learned fast-weight gain per compartment"); axs[0].legend(fontsize=8)
    axs[1].bar(x, tau, color="#888"); axs[1].set_yscale("log"); axs[1].set_xticks(x); axs[1].set_xticklabels(comp, fontsize=7); axs[1].set_title("learned memory lifetime tau (steps)")
    fig.tight_layout(); fig.savefig("report/fig6_compartments.png", dpi=140); print("-> report/fig6_compartments.png")

# POPGym: fly vs GRU MMER (completed runs), sorted by env
import glob as _glob, re as _re, json as _json, collections as _c
_res=_c.defaultdict(lambda: _c.defaultdict(list))
for f in _glob.glob("runs_popgym/popgym_*/log.jsonl"):
    try: rows=[_json.loads(l) for l in open(f)]
    except Exception: continue
    if not rows or rows[-1].get("steps",0) < 2_900_000: continue
    tag=f.split("/")[-2]; _,mem,rest=tag.split("_",2); env=_re.sub(r"_s\d$","",rest)
    _res[env][mem].append(max(r.get("mmer",-9) for r in rows))
if _res:
    envs=sorted(_res); fig,ax=plt.subplots(figsize=(11,4.2)); x=np.arange(len(envs)); w=0.38
    for k,(mem,lab,col) in enumerate([("fly","fly module","#3b6fb6"),("gru","GRU","#999")]):
        m=[np.mean(_res[e][mem]) if _res[e][mem] else np.nan for e in envs]; sd=[np.std(_res[e][mem]) if len(_res[e][mem])>1 else 0 for e in envs]
        ax.bar(x+(k-0.5)*w, m, w, yerr=sd, capsize=2, label=lab, color=col)
    ax.set_xticks(x); ax.set_xticklabels([e.replace("PositionOnly","PO").replace("Multiarmed","MA") for e in envs], rotation=60, ha="right", fontsize=7)
    ax.axhline(0,color="k",lw=0.5); ax.set_ylabel("MMER (3M steps, 8 epochs)"); ax.set_title("POPGym budget pass: fly module vs GRU under one PPO trainer"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("report/fig10_popgym.png", dpi=140); plt.close(fig); print("-> report/fig10_popgym.png")
