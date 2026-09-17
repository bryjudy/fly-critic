"""External benchmarks for the head-to-head (step 1).
  Bandit    : 10-arm Bernoulli bandit, T=200, arm means re-drawn per episode, best arm flips at a random trial in the
              middle third (non-stationary). Tuned classical baselines: discounted Thompson sampling, sliding-window UCB.
  DarkRoom  : Algorithm Distillation (Laskin et al. 2022) — 9x9 grid, goal fixed for the meta-episode, agent sees only
              its (x,y); reward 1 every step it stands on the goal; agent respawns at center every 20 steps; T=200.
  KeyToDoor : AD's Dark Key-to-Door — 9x9, must reach key (+1 once) then door (+1 once) per 40-step episode; T=200.
All expose the same interface as flycritic/env.py (reset/step -> obs{'pn','vec'}, r, done, info{post_reversal,optimal})
so every agent config in flycritic.train runs unchanged. `pn` is a fixed random sparse expansion of the observation
to 124 channels (random-expansion result: the fly's specific PN wiring is not needed)."""
import torch, math

def _pn_proj(in_dim, n_pn, g):
    W = torch.randn(in_dim, n_pn, generator=g) * (1 / math.sqrt(in_dim))
    return W

class _Base:
    n_pn = 124
    def _pn(self, feat):                                   # (B, in) -> nonneg (B,124) with norm 3 like odors
        x = torch.nn.functional.softplus(feat @ self.Wpn)
        return x / (x.norm(dim=1, keepdim=True) + 1e-6) * 3.0

class Bandit(_Base):
    def __init__(self, B, n_arms=10, T=200, reversal=True, device="cpu", seed=0, **kw):
        self.B, self.K, self.T, self.reversal, self.device = B, n_arms, T, reversal, device
        self.n_actions = n_arms; g = torch.Generator().manual_seed(seed)
        self.Wpn = _pn_proj(n_arms + 1, self.n_pn, g).to(device)
        self.obs_dim = self.n_pn + n_arms + 1
    def reset(self):
        B, dev = self.B, self.device
        self.p = torch.rand(B, self.K, device=dev) * 0.5           # arm means in [0,0.5]
        best = torch.randint(0, self.K, (B,), device=dev); self.p[torch.arange(B), best] = 0.9
        self.t = 0
        self.t_rev = torch.randint(self.T // 3, 2 * self.T // 3, (B,), device=dev) if self.reversal else torch.full((B,), 10**9, device=dev)
        self.prev_a = torch.zeros(B, dtype=torch.long, device=dev); self.prev_r = torch.zeros(B, device=dev)
        return self._obs()
    def _obs(self):
        a1 = torch.nn.functional.one_hot(self.prev_a, self.K).float()
        feat = torch.cat([a1, self.prev_r[:, None]], 1)              # context = last action + last reward
        pn = self._pn(feat); return {"pn": pn, "vec": torch.cat([pn, feat], 1)}
    def step(self, a):
        B, dev = self.B, self.device
        r = (torch.rand(B, device=dev) < self.p[torch.arange(B), a]).float()
        opt = a == self.p.argmax(1)
        self.t += 1
        flip = self.t == self.t_rev
        if flip.any():   # new best arm: demote old best to 0.1, promote a random other arm to 0.9
            idx = flip.nonzero().squeeze(1); old = self.p[idx].argmax(1)
            new = (old + torch.randint(1, self.K, (len(idx),), device=dev)) % self.K
            self.p[idx, old] = 0.1; self.p[idx, new] = 0.9
        self.prev_a, self.prev_r = a, r
        info = {"hit_reward": r > 0, "hit_punish": r <= 0, "post_reversal": self.t > self.t_rev, "optimal": opt}
        return self._obs(), r, self.t >= self.T, info

class DarkRoom(_Base):
    def __init__(self, B, grid=9, T=200, ep_len=20, key_door=False, device="cpu", seed=0, **kw):
        self.B, self.G, self.T, self.L, self.kd, self.device = B, grid, T, ep_len if not key_door else 40, key_door, device
        self.n_actions = 5; g = torch.Generator().manual_seed(seed)
        in_dim = 2 * grid + self.n_actions + 1 + (1 if key_door else 0)
        self.Wpn = _pn_proj(in_dim, self.n_pn, g).to(device)
        self.obs_dim = self.n_pn + in_dim
        self.moves = torch.tensor([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]], device=device)
    def reset(self):
        B, dev = self.B, self.device
        self.goal = torch.randint(0, self.G, (B, 2), device=dev)
        self.key = torch.randint(0, self.G, (B, 2), device=dev)
        self.pos = torch.full((B, 2), self.G // 2, device=dev)
        self.has_key = torch.zeros(B, dtype=torch.bool, device=dev); self.got_door = torch.zeros(B, dtype=torch.bool, device=dev)
        self.t = 0; self.t_rev = torch.full((B,), 10**9, device=dev)   # no reversal in AD tasks
        self.prev_a = torch.zeros(B, dtype=torch.long, device=dev); self.prev_r = torch.zeros(B, device=dev)
        return self._obs()
    def _obs(self):
        oh = lambda v: torch.nn.functional.one_hot(v, self.G).float()
        parts = [oh(self.pos[:, 0]), oh(self.pos[:, 1]), torch.nn.functional.one_hot(self.prev_a, self.n_actions).float(), self.prev_r[:, None]]
        if self.kd: parts.append(self.has_key.float()[:, None])
        feat = torch.cat(parts, 1)
        pn = self._pn(feat); return {"pn": pn, "vec": torch.cat([pn, feat], 1)}
    def step(self, a):
        B, dev = self.B, self.device
        self.pos = (self.pos + self.moves[a]).clamp(0, self.G - 1)
        if self.kd:
            at_key = (self.pos == self.key).all(1) & ~self.has_key
            at_door = (self.pos == self.goal).all(1) & self.has_key & ~self.got_door
            r = at_key.float() + at_door.float(); self.has_key |= at_key; self.got_door |= at_door
            opt = at_key | at_door
        else:
            r = (self.pos == self.goal).all(1).float(); opt = r > 0
        self.t += 1
        if self.t % self.L == 0:                                     # respawn, goal/key fixed
            self.pos[:] = self.G // 2; self.has_key[:] = False; self.got_door[:] = False
        self.prev_a, self.prev_r = a, r
        info = {"hit_reward": r > 0, "hit_punish": torch.zeros(B, dtype=torch.bool, device=dev), "post_reversal": self.t > self.t_rev, "optimal": opt}
        return self._obs(), r, self.t >= self.T, info

# ---- classical bandit baselines ---------------------------------------------------------------------------------
def run_bandit_baseline(kind, B=256, T=200, seeds=3, gamma=0.95, window=30, c=1.0):
    """kind: 'thompson' (discounted Beta-Bernoulli TS), 'ucb' (sliding-window UCB1), 'oracle', 'random'."""
    tot = []
    for s in range(seeds):
        env = Bandit(B, T=T, seed=100 + s); env.reset(); K = env.K
        a_ = torch.ones(B, K); b_ = torch.ones(B, K); hist_a = []; hist_r = []; R = torch.zeros(B)
        for t in range(T):
            if kind == "oracle": a = env.p.argmax(1)
            elif kind == "random": a = torch.randint(0, K, (B,))
            elif kind == "thompson": a = torch.distributions.Beta(a_, b_).sample().argmax(1)
            else:  # sliding-window UCB
                if len(hist_a) == 0: a = torch.randint(0, K, (B,))
                else:
                    A = torch.stack(hist_a[-window:], 1); Rw = torch.stack(hist_r[-window:], 1)
                    cnt = torch.zeros(B, K).scatter_add_(1, A, torch.ones_like(Rw)); s_ = torch.zeros(B, K).scatter_add_(1, A, Rw)
                    mean = s_ / cnt.clamp_min(1); bonus = c * torch.sqrt(math.log(A.shape[1] + 1) / cnt.clamp_min(1)); bonus[cnt == 0] = 10
                    a = (mean + bonus).argmax(1)
            _, r, _, _ = env.step(a); R += r; hist_a.append(a); hist_r.append(r)
            if kind == "thompson":  # discounting -> tracks non-stationarity
                a_ = 1 + gamma * (a_ - 1); b_ = 1 + gamma * (b_ - 1)
                a_[torch.arange(B), a] += r; b_[torch.arange(B), a] += 1 - r
        tot.append(R.mean().item())
    return sum(tot) / len(tot)

if __name__ == "__main__":
    for kind in ["oracle", "random", "ucb"]: print(f"{kind:9s} {run_bandit_baseline(kind):6.1f}")
    for g in [0.9, 0.95, 0.98, 1.0]: print(f"thompson gamma={g:<5} {run_bandit_baseline('thompson', gamma=g):6.1f}")
