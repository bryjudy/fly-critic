"""ToolWorld: vectorised TEXT contextual-bandit with non-stationarity, for a frozen LLM + plastic module.

B parallel episodes, T trials each. 6 named tools, 4 query types (each with several paraphrases). Per episode,
each query type has one BEST tool (success prob 0.9); other tools succeed with prob U(0.1, 0.3). Reliabilities
are re-drawn per episode. At a random trial in the middle third ("tool rotation"), the best tool for two of the
four query types is swapped with another tool. Reward: +1 success, -0.25 failure, 0 for action 6 = "none".

Observation text (exact template, one line per part):
    "You are a routing assistant. Available tools: WebSearch, Calculator, Translator, CodeRunner, Weather, Wiki."
    "Previous trial: used <Tool> -> <success|failure>."          (or "Previous trial: none." at trial 0)
    "Query: <paraphrase>"
    "Which tool should handle this query? Answer with the tool name only."
`pn` numeric features are provided by flycritic.llm.bridge from the LLM hidden state (not by the env); the env
exposes `query_type` (B,) so a bridge-free oracle/random policy can also be run.
"""
import torch

TOOLS = ["WebSearch", "Calculator", "Translator", "CodeRunner", "Weather", "Wiki"]
QUERY_TYPES = ["lookup", "math", "translate", "code"]
PARAPHRASES = {
    "lookup": ["Who won the 1998 World Cup final?", "What is the capital of Mongolia?",
               "When was the Eiffel Tower completed?", "Find the population of Lagos."],
    "math": ["What is 17.5% of 2,480?", "Compute 3^9 minus 4,112.", "Solve 12x + 7 = 103 for x.",
             "What is the square root of 7,569?"],
    "translate": ["Translate 'good morning, friend' into Portuguese.", "How do you say 'thank you very much' in Japanese?",
                  "Render 'the meeting is postponed' in German.", "Translate 'where is the station' into Italian."],
    "code": ["Run this Python: print(sum(range(50)))", "Execute: sorted([5,3,9,1])[::-1] in Python and show output.",
             "Evaluate the JavaScript expression [1,2,3].map(x=>x*x).", "Run a snippet that reverses the string 'plasticity'."],
}
SYSTEM = "You are a routing assistant. Available tools: " + ", ".join(TOOLS) + "."
ASK = "Which tool should handle this query? Answer with the tool name only."

class ToolWorld:
    def __init__(self, B, T=120, device="cpu", seed=0, p_best=0.9, p_other=(0.1, 0.3), fail_cost=-0.25, reversal=True):
        self.B, self.T, self.device, self.reversal = B, T, device, reversal
        self.n_tools, self.n_types = len(TOOLS), len(QUERY_TYPES)
        self.n_actions = self.n_tools + 1                      # action n_tools = "none"
        self.p_best, self.p_other, self.fail_cost = p_best, p_other, fail_cost
        self.g = torch.Generator(device="cpu").manual_seed(seed)

    def _rand(self, *shape): return torch.rand(*shape, generator=self.g)
    def _randint(self, lo, hi, shape): return torch.randint(lo, hi, shape, generator=self.g)

    def reset(self):
        B = self.B
        # success prob table (B, n_types, n_tools)
        lo, hi = self.p_other
        self.P = lo + (hi - lo) * self._rand(B, self.n_types, self.n_tools)
        self.best = torch.stack([torch.randperm(self.n_tools, generator=self.g)[: self.n_types] for _ in range(B)])  # (B, n_types) distinct best tools
        self.P.scatter_(2, self.best.unsqueeze(2), self.p_best)
        self.t = 0
        self.t_rev = self._randint(self.T // 3, 2 * self.T // 3, (B,)) if self.reversal else torch.full((B,), 10 ** 9)
        self.prev_tool = torch.full((B,), -1, dtype=torch.long); self.prev_ok = torch.zeros(B, dtype=torch.bool)
        self.prev_r = torch.zeros(B)
        return self._obs()

    def _obs(self):
        B = self.B
        self.query_type = self._randint(0, self.n_types, (B,))
        self.para = self._randint(0, 4, (B,))
        return {"text": self.obs_text(), "query_type": self.query_type.clone(), "t": self.t}

    def obs_text(self):
        out = []
        for i in range(self.B):
            prev = "Previous trial: none." if self.prev_tool[i] < 0 else \
                f"Previous trial: used {TOOLS[self.prev_tool[i]]} -> {'success' if self.prev_ok[i] else 'failure'}."
            q = PARAPHRASES[QUERY_TYPES[self.query_type[i]]][self.para[i]]
            out.append(f"{SYSTEM}\n{prev}\nQuery: {q}\n{ASK}")
        return out

    def step(self, a):
        """a: LongTensor (B,) in [0, n_tools]; n_tools == 'none'."""
        a = a.to("cpu").long(); B = self.B
        idx = torch.arange(B)
        is_tool = a < self.n_tools
        p = self.P[idx, self.query_type, a.clamp(max=self.n_tools - 1)]
        ok = (self._rand(B) < p) & is_tool
        r = torch.where(is_tool, torch.where(ok, torch.ones(B), torch.full((B,), self.fail_cost)), torch.zeros(B))
        best_now = self.best[idx, self.query_type]
        optimal = a == best_now
        self.t += 1
        flip = self.t == self.t_rev
        if flip.any():
            for i in torch.nonzero(flip).flatten().tolist():
                # rotate: the best tools of query types 0 and 1 are swapped with each other's (distinct) tools
                b = self.best[i].clone(); b0, b1 = b[0].item(), b[1].item()
                self.P[i, 0, b0] = lo_hi(self); self.P[i, 1, b1] = lo_hi(self)
                self.best[i, 0], self.best[i, 1] = b1, b0
                self.P[i, 0, b1] = self.p_best; self.P[i, 1, b0] = self.p_best
        self.prev_tool = torch.where(is_tool, a, torch.full_like(a, -1)); self.prev_ok = ok; self.prev_r = r
        done = self.t >= self.T
        info = {"hit_reward": r > 0, "hit_punish": r < 0, "post_reversal": self.t > self.t_rev, "optimal": optimal}
        return self._obs(), r.to(self.device), done, info

    # ---- reference policies -------------------------------------------------
    def oracle_action(self): return self.best[torch.arange(self.B), self.query_type]
    def expected_oracle_reward(self):
        """Expected reward/episode for always picking the current best tool."""
        return self.T * (self.p_best * 1.0 + (1 - self.p_best) * self.fail_cost)

def lo_hi(env):
    lo, hi = env.p_other; return lo + (hi - lo) * env._rand(1).item()
