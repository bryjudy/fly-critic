"""Precompute frozen-LLM features for EVERY possible ToolWorld observation text (4 types x 4 paraphrases x
(1 + 6 tools x 2 outcomes) = 208 texts) so training processes never load the model.
usage: uv run python -m flycritic.llm.precompute --model 0.5b   -> data/tw_feats_<model>.pt"""
import argparse, torch, time
from .toolworld import TOOLS, QUERY_TYPES, PARAPHRASES, SYSTEM, ASK
from .features import FrozenLLM

def all_texts():
    prevs = ["Previous trial: none."] + [f"Previous trial: used {t} -> {o}." for t in TOOLS for o in ("success", "failure")]
    return [f"{SYSTEM}\n{p}\nQuery: {q}\n{ASK}" for p in prevs for qt in QUERY_TYPES for q in PARAPHRASES[qt]] + ["probe"]

class TableLLM:
    """Drop-in for FrozenLLM backed by a precomputed table (falls back to a real model only for unknown texts)."""
    def __init__(self, path, fallback_model=None):
        d = torch.load(path); self.tab = d["table"]; self.model_name = d["model"]; self.fallback = fallback_model
        self._llm = None
    def __call__(self, texts):
        miss = [t for t in texts if t not in self.tab]
        if miss:
            if self._llm is None: self._llm = FrozenLLM(model=self.fallback or self.model_name)
            h, tl = self._llm(miss)
            for i, t in enumerate(miss): self.tab[t] = (h[i], tl[i])
        return torch.stack([self.tab[t][0] for t in texts]), torch.stack([self.tab[t][1] for t in texts])

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--model", default="0.5b"); p.add_argument("--device", default="cpu"); a = p.parse_args()
    texts = all_texts(); llm = FrozenLLM(model=a.model, device=a.device); t0 = time.time(); tab = {}
    for i in range(0, len(texts), 16):
        h, tl = llm(texts[i:i + 16])
        for j, t in enumerate(texts[i:i + 16]): tab[t] = (h[j].cpu(), tl[j].cpu())
    out = f"data/tw_feats_{a.model.replace('/', '_')}.pt"; torch.save({"model": a.model, "table": tab}, out)
    print(f"{len(tab)} texts -> {out} in {time.time()-t0:.0f}s; hidden dim {h.shape[1]}")
