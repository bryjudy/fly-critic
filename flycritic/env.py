"""OdorGrid: vectorized gridworld where odors (patterns over the fly's 124 PN channels) predict reward or
punishment, and the contingency reverses mid-episode. Torch, batch-first, no gym dependency."""
import torch

class OdorGrid:
    def __init__(self, B, n_pn=124, grid=6, T=100, n_odors=16, lam=3.0, noise=0.05, reversal=True,
                 device="cpu", seed=0):
        self.B, self.P, self.G, self.T, self.lam, self.noise, self.reversal = B, n_pn, grid, T, lam, noise, reversal
        self.device = device; self.n_actions = 5
        g = torch.Generator().manual_seed(seed)
        # odor dictionary: each odor excites ~12% of glomeruli with graded intensity (fixed across training)
        mask = (torch.rand(n_odors, n_pn, generator=g) < 0.12).float()
        self.odors = (mask * torch.rand(n_odors, n_pn, generator=g)).to(device)
        self.odors = self.odors / self.odors.norm(dim=1, keepdim=True)
        self.n_odors = n_odors
        self.moves = torch.tensor([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]], device=device)
        self.obs_dim = n_pn + 2 + self.n_actions + 1

    def _place(self, n):
        return torch.randint(0, self.G, (self.B, n, 2), device=self.device)

    def reset(self):
        B, dev = self.B, self.device
        # 3 sources per env: index 0 = reward, 1 = punish, 2 = neutral (roles flip at reversal)
        self.src_odor = torch.stack([torch.randperm(self.n_odors, device=dev)[:3] for _ in range(B)])
        self.src_pos = self._place(3)
        self.pos = torch.randint(0, self.G, (B, 2), device=dev)
        self.roles = torch.tensor([1.0, -1.0, 0.0], device=dev).repeat(B, 1)   # (B, 3)
        self.t = 0
        self.t_rev = torch.randint(self.T // 3, 2 * self.T // 3, (B,), device=dev) if self.reversal \
            else torch.full((B,), 10 ** 9, device=dev)
        self.prev_a = torch.zeros(B, dtype=torch.long, device=dev); self.prev_r = torch.zeros(B, device=dev)
        return self._obs()

    def _odor_field(self):
        d = (self.src_pos - self.pos[:, None, :]).abs().sum(2).float()          # (B, 3) manhattan
        conc = torch.exp(-d / self.lam)                                          # (B, 3)
        pn = torch.einsum("bs,bsp->bp", conc, self.odors[self.src_odor])         # (B, P)
        return pn + self.noise * torch.rand_like(pn)

    def _obs(self):
        pn = self._odor_field()
        pos = self.pos.float() / (self.G - 1)
        a1 = torch.nn.functional.one_hot(self.prev_a, self.n_actions).float()
        return {"pn": pn, "vec": torch.cat([pn, pos, a1, self.prev_r[:, None]], 1)}

    def step(self, a):
        B, dev = self.B, self.device
        self.pos = (self.pos + self.moves[a]).clamp(0, self.G - 1)
        hit = (self.src_pos == self.pos[:, None, :]).all(2)                      # (B, 3)
        r = (hit.float() * self.roles).sum(1)
        # relocate consumed sources
        newpos = self._place(3)
        self.src_pos = torch.where(hit[:, :, None], newpos, self.src_pos)
        self.t += 1
        flip = (self.t == self.t_rev)
        if flip.any():
            self.roles[flip] = self.roles[flip][:, [1, 0, 2]]
        self.prev_a, self.prev_r = a, r
        done = self.t >= self.T
        info = {"hit_reward": r > 0, "hit_punish": r < 0,
                "post_reversal": self.t > self.t_rev}
        return self._obs(), r, done, info

    # distance of agent to the currently-rewarding source (for approach metrics)
    def dist_to_reward(self):
        d = (self.src_pos - self.pos[:, None, :]).abs().sum(2).float()
        return d[self.roles > 0].view(self.B)


class OdorChoice:
    """Go/no-go conditioning (the fly T-maze in disguise). Each trial one odor is presented; action 1 = approach
    (receive that odor's outcome: +1 / -1 / 0), action 0 = avoid (0). Three odors per episode; reward and
    punishment odors swap at a random trial in the middle third. Odors re-drawn every episode."""
    def __init__(self, B, n_pn=124, T=100, n_odors=16, noise=0.05, reversal=True, device="cpu", seed=0, **kw):
        self.B, self.P, self.T, self.noise, self.reversal, self.device = B, n_pn, T, noise, reversal, device
        self.n_actions = 2
        g = torch.Generator().manual_seed(seed)
        mask = (torch.rand(n_odors, n_pn, generator=g) < 0.12).float()
        self.odors = (mask * torch.rand(n_odors, n_pn, generator=g)).to(device)
        self.odors = self.odors / self.odors.norm(dim=1, keepdim=True) * 3.0
        self.n_odors = n_odors
        self.obs_dim = n_pn + self.n_actions + 1

    def reset(self):
        B, dev = self.B, self.device
        self.ep_odor = torch.stack([torch.randperm(self.n_odors, device=dev)[:3] for _ in range(B)])  # (B,3)
        self.roles = torch.tensor([1.0, -1.0, 0.0], device=dev).repeat(B, 1)
        self.t = 0
        self.t_rev = torch.randint(self.T // 3, 2 * self.T // 3, (B,), device=dev) if self.reversal \
            else torch.full((B,), 10 ** 9, device=dev)
        self.prev_a = torch.zeros(B, dtype=torch.long, device=dev); self.prev_r = torch.zeros(B, device=dev)
        return self._obs()

    def _obs(self):
        self.cur = torch.randint(0, 3, (self.B,), device=self.device)                    # which of the 3 odors
        pn = self.odors[self.ep_odor.gather(1, self.cur[:, None]).squeeze(1)]
        pn = pn + self.noise * torch.rand_like(pn)
        a1 = torch.nn.functional.one_hot(self.prev_a, self.n_actions).float()
        return {"pn": pn, "vec": torch.cat([pn, a1, self.prev_r[:, None]], 1)}

    def step(self, a):
        role = self.roles.gather(1, self.cur[:, None]).squeeze(1)
        r = role * (a == 1).float()
        self.t += 1
        flip = (self.t == self.t_rev)
        if flip.any(): self.roles[flip] = self.roles[flip][:, [1, 0, 2]]
        self.prev_a, self.prev_r = a, r
        info = {"hit_reward": r > 0, "hit_punish": r < 0, "post_reversal": self.t > self.t_rev,
                "optimal": (role > 0) == (a == 1)}
        return self._obs(), r, self.t >= self.T, info


class OdorContext(OdorChoice):
    """Context-dependent go/no-go. A binary context cue (e.g. light on/off) is shown each trial. In context 0 the
    roles are (odor A +1, odor B -1, C 0); in context 1 they are swapped (A -1, B +1). A pure odor->valence
    memory averages to zero here, so the agent must combine odor with context. Mid-episode reversal flips the
    mapping in BOTH contexts. ctx_to_kc: also feed the context to the mushroom body as 2 extra PN channels (like
    the fly's multimodal KC inputs), so KC codes become odor x context conjunctions."""
    def __init__(self, B, ctx_to_kc=False, **kw):
        super().__init__(B, **kw)
        self.ctx_to_kc = ctx_to_kc
        self.obs_dim = self.P + 2 + self.n_actions + 1                 # + 2 context dims
        self.P_mb = self.P + (2 if ctx_to_kc else 0)

    def _obs(self):
        self.cur = torch.randint(0, 3, (self.B,), device=self.device)
        self.ctx = torch.randint(0, 2, (self.B,), device=self.device)
        pn = self.odors[self.ep_odor.gather(1, self.cur[:, None]).squeeze(1)]
        pn = pn + self.noise * torch.rand_like(pn)
        c = torch.nn.functional.one_hot(self.ctx, 2).float() * 3.0
        a1 = torch.nn.functional.one_hot(self.prev_a, self.n_actions).float()
        pn_mb = torch.cat([pn, c], 1) if self.ctx_to_kc else pn
        return {"pn": pn_mb, "vec": torch.cat([pn, c / 3.0, a1, self.prev_r[:, None]], 1)}

    def step(self, a):
        role = self.roles.gather(1, self.cur[:, None]).squeeze(1)
        role = torch.where(self.ctx == 1, -role, role)                  # context 1 inverts +1/-1 (neutral stays 0)
        r = role * (a == 1).float()
        self.t += 1
        flip = (self.t == self.t_rev)
        if flip.any(): self.roles[flip] = self.roles[flip][:, [1, 0, 2]]
        self.prev_a, self.prev_r = a, r
        info = {"hit_reward": r > 0, "hit_punish": r < 0, "post_reversal": self.t > self.t_rev,
                "optimal": (role > 0) == (a == 1)}
        return self._obs(), r, self.t >= self.T, info
