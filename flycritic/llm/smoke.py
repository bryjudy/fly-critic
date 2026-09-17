"""Laptop smoke test: ToolWorld x Qwen2.5-0.5B. usage: uv run python -m flycritic.llm.smoke [--model 0.5b]"""
import argparse, time, torch
from .toolworld import ToolWorld, TOOLS
from .features import FrozenLLM
from .bridge import Bridge
from .baselines import evaluate_baseline

def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", default="0.5b"); p.add_argument("--B", type=int, default=4); p.add_argument("--T", type=int, default=20)
    a = p.parse_args(); torch.manual_seed(0)
    env = ToolWorld(a.B, T=a.T, seed=0)
    print(f"ToolWorld B={a.B} T={a.T} | oracle expected reward/episode {env.expected_oracle_reward():.1f}")
    # random policy
    obs = env.reset(); tot = torch.zeros(a.B)
    for _ in range(a.T): obs, r, d, i = env.step(torch.randint(0, env.n_actions, (a.B,))); tot += r
    print(f"random policy reward/episode {tot.mean():.2f}")
    obs = env.reset(); tot = torch.zeros(a.B)
    for _ in range(a.T): obs, r, d, i = env.step(env.oracle_action()); tot += r
    print(f"oracle policy (sampled) reward/episode {tot.mean():.2f}")
    t0 = time.time(); llm = FrozenLLM(a.model); print(f"loaded {a.model} on {llm.device} in {time.time()-t0:.0f}s, d_model {llm.d_model}, tool first-tokens unique: {llm.first_unique}")
    obs = env.reset(); h, tl = llm(obs["text"]); bridge = Bridge(llm.d_model); pn = bridge(h)
    print(f"hidden {tuple(h.shape)} | tool_logits {tuple(tl.shape)} | pn {tuple(pn.shape)} norm {pn.norm(dim=1).mean():.2f} nonneg {bool((pn>=0).all())}")
    print("zero-shot tool picks for first obs:", [TOOLS[i] for i in tl.argmax(1).tolist()], "| query types:", obs["query_type"].tolist())
    h2, _ = llm(obs["text"]); print(f"cache hit -> identical: {torch.equal(h, h2)}; llm forward calls so far {llm.calls}, {llm.call_time/max(llm.calls,1)*1000:.0f} ms/call")
    for kind in ["zeroshot", "longctx", "rag"]:
        t0 = time.time(); res = evaluate_baseline(kind, env, llm, episodes=1)
        print(f"{kind:9s} reward/episode {res['reward']:6.2f} | post-reversal {res['reward_post']:6.2f} | optimal rate {res['optimal_rate']:.2f} | {time.time()-t0:.0f}s, {llm.calls} calls, {llm.call_time/max(llm.calls,1)*1000:.0f} ms/call avg")

if __name__ == "__main__":
    main()
