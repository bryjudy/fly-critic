"""ToolWorld as a flycritic environment: text observations -> frozen LLM (cached) -> hidden state + zero-shot tool
log-probs -> Bridge -> 124 'PN' channels. The agent (any flycritic.train config) sees
  vec = [pn(124), tool_logprobs(n_tools), prev_action_onehot(n_actions), prev_reward]
and pn drives the fly's KC expansion. The LLM is never trained; only the small policy / plastic module is."""
import torch, torch.nn.functional as F
from .toolworld import ToolWorld
from .features import FrozenLLM
from .precompute import TableLLM
from .bridge import Bridge

class ToolWorldEnv:
    def __init__(self, B, T=120, device="cpu", seed=0, llm="0.5b", llm_device=None, cache_path=None, **kw):
        self.tw = ToolWorld(B, T=T, device="cpu", seed=seed)
        self.B, self.T, self.device = B, T, device
        self.n_actions = self.tw.n_actions; self.n_tools = self.tw.n_tools
        import os
        table = f"data/tw_feats_{llm.replace('/', '_')}.pt"
        self.llm = TableLLM(table, fallback_model=llm) if os.path.exists(table) else FrozenLLM(model=llm, device=llm_device, cache_path=cache_path)
        d_model = self.llm(["probe"])[0].shape[1]
        self.bridge = Bridge(d_model, seed=seed); self.P = self.bridge.W.shape[1]
        self.obs_dim = self.P + self.n_tools + self.n_actions + 1
        self.prev_a = torch.zeros(B, dtype=torch.long); self.prev_r = torch.zeros(B)
        self.t_rev = None
    def _wrap(self, o):
        hidden, tl = self.llm(o["text"])
        pn = self.bridge(hidden.float())
        lp = F.log_softmax(tl.float(), 1)
        a1 = F.one_hot(self.prev_a, self.n_actions).float()
        vec = torch.cat([pn, lp, a1, self.prev_r[:, None]], 1)
        return {"pn": pn.to(self.device), "vec": vec.to(self.device), "text": o["text"]}
    def reset(self):
        self.prev_a = torch.zeros(self.B, dtype=torch.long); self.prev_r = torch.zeros(self.B)
        o = self.tw.reset(); self.t_rev = self.tw.t_rev; self.t = 0
        return self._wrap(o)
    def step(self, a):
        a = a.to("cpu"); o, r, done, info = self.tw.step(a); self.t = self.tw.t
        self.prev_a, self.prev_r = a, r.to("cpu")
        info = {k: v.to(self.device) for k, v in info.items()}
        return self._wrap(o), r.to(self.device), done, info
    def oracle_reward(self): return self.tw.expected_oracle_reward()
