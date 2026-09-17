"""Compare trained agents: reward curves aligned to the reversal, learning speed after reversal,
punishment avoidance, and (for MB critics) what the circuit itself believes about each odour.
usage: uv run python -m flycritic.evaluate --tags mb_s0 shuffled_s0 scalar_s0 none_s0 --episodes 20
"""
import argparse, json, torch, numpy as np
from .env import OdorGrid, OdorChoice, OdorContext
from .bench import Bandit, DarkRoom
from .mb import MushroomBody
from .model import Agent
from .train import rollout

def load(tag, device="cpu"):
    ck = torch.load(f"runs/{tag}/ckpt.pt", map_location=device, weights_only=False); a = ck["args"]
    if a.get("env") == "toolworld":
        from .llm.adapter import ToolWorldEnv
        env = ToolWorldEnv(a["B"], T=a["T"], device=device, seed=1234, llm=a.get("llm", "0.5b"), cache_path=None)
    elif a.get("env") == "bandit": env = Bandit(a["B"], T=a["T"], device=device, seed=1234)
    elif a.get("env") in ("darkroom", "keydoor"): env = DarkRoom(a["B"], T=a["T"], device=device, seed=1234, key_door=(a["env"] == "keydoor"))
    elif a.get("env") == "context": env = OdorContext(a["B"], T=a["T"], device=device, seed=1234, ctx_to_kc=a.get("ctx_to_kc", False))
    else:
        Env = OdorChoice if a.get("env", "grid") == "choice" else OdorGrid
        env = Env(a["B"], T=a["T"], grid=a["grid"], device=device, seed=1234)
    extra = 2 if (a.get("env") == "context" and a.get("ctx_to_kc")) else 0
    mb = MushroomBody(shuffle=(a["critic"] == "shuffled" or a.get("randkc", False)), seed=a["seed"], collapse=a.get("collapse", False),
                      uniform=a.get("uniform", False), extra_pn=extra, ctx_frac=a.get("ctx_frac", 0.3),
                      ctx_gain=a.get("ctx_gain", 0.1)) if (a["critic"] in ("mb", "shuffled", "hybrid") or a.get("pre") == "kc") else None
    agent = Agent(env.obs_dim, env.n_actions, critic=a["critic"], mb=mb, d=a["d"], layers=a["layers"], heads=a.get("heads", 4), ctx=a["T"],
                  no_feat=a.get("no_feat", False), no_plastic=a.get("no_plastic", False), pre=a.get("pre", "h"), prior=a.get("prior", False))
    agent.load_state_dict(ck["agent"]); agent.eval()
    return agent, env, a

def evaluate(tag, episodes=20, device="cpu"):
    agent, env, a = load(tag, device)
    T = env.T; W = 30
    aligned_r = np.zeros(2 * W); aligned_n = np.zeros(2 * W)      # reward vs time since reversal
    pre_hits = post_hits = pre_pun = post_pun = 0.0; first_post_reward = []; first_post_punish = []
    for _ in range(episodes):
        b = rollout(agent, env, device)
        R = b["R"].cpu().numpy(); trev = env.t_rev.cpu().numpy()
        for i in range(env.B):
            for k in range(-W, W):
                t = trev[i] + k
                if 0 <= t < T: aligned_r[k + W] += R[i, t]; aligned_n[k + W] += 1
            pre, post = R[i, :trev[i]], R[i, trev[i]:]
            pre_hits += (pre > 0).sum() / len(pre); post_hits += (post > 0).sum() / len(post)
            pre_pun += (pre < 0).sum() / len(pre); post_pun += (post < 0).sum() / len(post)
            # steps after reversal until first punishment (walking into the old reward odour) and first reward
            p = np.where(post < 0)[0]; r = np.where(post > 0)[0]
            first_post_punish.append(p[0] if len(p) else len(post)); first_post_reward.append(r[0] if len(r) else len(post))
    n = episodes * env.B
    return dict(tag=tag, critic=a["critic"], reward_per_step_pre=pre_hits / n - pre_pun / n,
                reward_per_step_post=post_hits / n - post_pun / n,
                reward_rate_pre=pre_hits / n, reward_rate_post=post_hits / n,
                punish_rate_pre=pre_pun / n, punish_rate_post=post_pun / n,
                steps_to_first_post_reward=float(np.mean(first_post_reward)),
                steps_to_first_post_punish=float(np.mean(first_post_punish)),
                aligned=(aligned_r / np.maximum(aligned_n, 1)).tolist())

def main():
    p = argparse.ArgumentParser(); p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=20); p.add_argument("--out", default="runs/compare")
    a = p.parse_args()
    res = [evaluate(t, a.episodes) for t in a.tags]
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    print(f"{'tag':14s} {'critic':9s} {'R/step pre':>10s} {'R/step post':>11s} {'punish pre':>10s} {'punish post':>11s} {'steps->1st post R':>17s}")
    for r in res:
        print(f"{r['tag']:14s} {r['critic']:9s} {r['reward_per_step_pre']:10.3f} {r['reward_per_step_post']:11.3f} "
              f"{r['punish_rate_pre']:10.3f} {r['punish_rate_post']:11.3f} {r['steps_to_first_post_reward']:17.1f}")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(-30, 30)
        for r in res:
            y = np.convolve(r["aligned"], np.ones(5) / 5, mode="same"); ax.plot(x, y, label=r["tag"].replace("ch_","").replace("_s0",""))
        ax.axvline(0, color="k", lw=0.8, ls="--"); ax.set_xlabel("steps since contingency reversal"); ax.set_ylabel("reward per step")
        ax.legend(); ax.set_title("OdorChoice: recovery after reward/punishment swap (oracle 0.33)"); fig.tight_layout(); fig.savefig(a.out + ".png", dpi=130)
        print("plot ->", a.out + ".png")
    except Exception as e: print("plot skipped:", e)

if __name__ == "__main__":
    main()
