"""Zero-shot baselines that use the frozen LLM directly on ToolWorld.

LONG-CONTEXT prompt (exact template):
    <SYSTEM line>
    History so far (trial, query, tool, outcome):
    1. <query> -> <Tool>: <success|failure>
    ...
    Query: <current query>
    <ASK line>
RAG prompt: same, but "Most relevant past trials:" lists the k=8 past records whose query hidden-state is most
cosine-similar to the current query's hidden state (computed with the same frozen LLM on the bare query text).
Both pick argmax over the 6 tool-name first-token log-probs.
"""
import torch
from .toolworld import TOOLS, SYSTEM, ASK, PARAPHRASES, QUERY_TYPES

def _query(env, i): return PARAPHRASES[QUERY_TYPES[env.query_type[i]]][env.para[i]]

def long_context_prompt(hist, query):
    lines = [SYSTEM, "History so far (trial, query, tool, outcome):"]
    lines += [f"{n+1}. {q} -> {TOOLS[a]}: {'success' if ok else 'failure'}" for n, (q, a, ok) in enumerate(hist)] or ["(none yet)"]
    return "\n".join(lines + [f"Query: {query}", ASK])

def rag_prompt(records, query):
    lines = [SYSTEM, "Most relevant past trials (query, tool, outcome):"]
    lines += [f"- {q} -> {TOOLS[a]}: {'success' if ok else 'failure'}" for (q, a, ok) in records] or ["(none yet)"]
    return "\n".join(lines + [f"Query: {query}", ASK])

@torch.no_grad()
def evaluate_baseline(kind, env, llm, episodes=1, k=8, verbose=False):
    """kind in {'longctx','rag','zeroshot'}. Returns dict(reward, reward_post, optimal_rate)."""
    tot = 0.0; post = 0.0; opt = 0.0; n = 0
    for _ in range(episodes):
        obs = env.reset(); B = env.B
        hist = [[] for _ in range(B)]; embs = [[] for _ in range(B)]
        for t in range(env.T):
            qs = [_query(env, i) for i in range(B)]
            if kind == "rag":
                qh, _ = llm(qs)
                prompts = []
                for i in range(B):
                    if embs[i]:
                        E = torch.stack(embs[i]); sim = torch.nn.functional.cosine_similarity(E, qh[i][None], dim=1)
                        top = sim.topk(min(k, len(embs[i]))).indices.tolist(); recs = [hist[i][j] for j in top]
                    else: recs = []
                    prompts.append(rag_prompt(recs, qs[i]))
            elif kind == "longctx": prompts = [long_context_prompt(hist[i], qs[i]) for i in range(B)]
            else: prompts = obs["text"]
            _, tl = llm(prompts); a = tl.argmax(1)
            obs2, r, done, info = env.step(a)
            for i in range(B):
                hist[i].append((qs[i], a[i].item(), bool(r[i] > 0)))
                if kind == "rag": embs[i].append(qh[i])
            tot += r.sum().item(); post += (r * info["post_reversal"].float().to(r.device)).sum().item(); opt += info["optimal"].float().sum().item()
            n += B; obs = obs2
            if verbose and t % 20 == 0: print(f"  t={t} mean r {r.mean():+.2f}")
    eps = episodes * env.B
    return dict(reward=tot / eps, reward_post=post / eps, optimal_rate=opt / n)
