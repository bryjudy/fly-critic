"""Zero-shot / long-context / RAG baselines on ToolWorld with a frozen LLM (no training).
usage: uv run python -m flycritic.llm.eval_baselines --model 0.5b --B 16 --T 120 --episodes 2"""
import argparse, json, time, torch
from .toolworld import ToolWorld
from .features import FrozenLLM
from .baselines import evaluate_baseline
p = argparse.ArgumentParser(); p.add_argument("--model", default="0.5b"); p.add_argument("--B", type=int, default=16)
p.add_argument("--T", type=int, default=120); p.add_argument("--episodes", type=int, default=2); p.add_argument("--kinds", nargs="+", default=["zeroshot", "longctx", "rag"]); p.add_argument("--device", default="cpu"); p.add_argument("--chunk", type=int, default=16)
a = p.parse_args()
llm = FrozenLLM(model=a.model, device=a.device, chunk=a.chunk); env = ToolWorld(a.B, T=a.T, seed=1234)
print(f"model {a.model} | oracle expected reward/episode {env.expected_oracle_reward():.1f} | random ~{a.T*(1/7*0 + 6/7*(0.9/6*1.0+ (1-0.9/6)*-0.25)):.1f}")
res = {}
for kind in a.kinds:
    t0 = time.time(); r = evaluate_baseline(kind, env, llm, episodes=a.episodes); res[kind] = r
    print(f"{kind:9s} {json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})}  ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(f"runs/tw_baselines_{a.model.replace('/','_')}.json", "w"), indent=1)
