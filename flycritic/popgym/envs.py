"""POPGym vectorized environments for fly-critic.

make_vec(env_name, B, seed) -> VecPOP: B synchronous POPGym envs with
  * observation flattening: each Discrete / MultiDiscrete component / Tuple part is one-hot; Box is passed through
    (scaled by the space bounds); the previous action (one-hot per component) and previous reward are appended;
  * factorized actions: Discrete -> one categorical; MultiDiscrete -> one categorical per component;
  * auto-reset with per-env `done` flags returned from step();
  * a fixed random sparse projection of the flattened obs to 124 "PN" channels (as in flycritic/bench.py), so the fly
    memory's Kenyon-cell expansion can be used unchanged.
Registry: 8 families x {Easy, Medium, Hard} = 24 envs.
"""
import math, numpy as np, torch, gymnasium as gym
import popgym.envs as PE

FAMILIES = ["MultiarmedBandit", "RepeatPrevious", "RepeatFirst", "CountRecall", "HigherLower", "Autoencode",
            "PositionOnlyCartPole", "NoisyPositionOnlyCartPole"]
LEVELS = ["Easy", "Medium", "Hard"]
ENVS = [f"{f}{l}" for f in FAMILIES for l in LEVELS]
# empirical episode lengths (random policy, 3 episodes; measured 2026-09-15). CartPole variants terminate early.
EPISODE_LEN = {"MultiarmedBandit": (200, 400, 600), "RepeatPrevious": (51, 103, 155), "RepeatFirst": (51, 415, 831),
               "CountRecall": (51, 103, 207), "HigherLower": (51, 103, 155), "Autoencode": (103, 207, 311),
               "PositionOnlyCartPole": (200, 200, 200), "NoisyPositionOnlyCartPole": (200, 200, 200)}

def _space_parts(space):
    """Return list of (kind, n_or_bounds) leaves for flattening."""
    if isinstance(space, gym.spaces.Discrete): return [("disc", int(space.n))]
    if isinstance(space, gym.spaces.MultiDiscrete): return [("disc", int(n)) for n in space.nvec]
    if isinstance(space, gym.spaces.Tuple): return sum([_space_parts(s) for s in space.spaces], [])
    if isinstance(space, gym.spaces.Box):
        lo, hi = np.asarray(space.low, dtype=np.float32).ravel(), np.asarray(space.high, dtype=np.float32).ravel()
        return [("box", (lo, hi))]
    raise NotImplementedError(space)

def _leaves(obs, space):
    if isinstance(space, gym.spaces.Discrete): return [int(obs)]
    if isinstance(space, gym.spaces.MultiDiscrete): return [int(v) for v in np.asarray(obs).ravel()]
    if isinstance(space, gym.spaces.Tuple): return sum([_leaves(o, s) for o, s in zip(obs, space.spaces)], [])
    if isinstance(space, gym.spaces.Box): return [np.asarray(obs, dtype=np.float32).ravel()]
    raise NotImplementedError(space)

class ObsFlattener:
    def __init__(self, space):
        self.parts = _space_parts(space)
        self.dim = sum(spec if k == "disc" else len(spec[0]) for k, spec in self.parts)
    def __call__(self, obs, space):
        out = np.zeros(self.dim, dtype=np.float32); i = 0
        for (k, spec), leaf in zip(self.parts, _leaves(obs, space)):
            if k == "disc":
                out[i + min(int(leaf), spec - 1)] = 1.0; i += spec
            else:
                lo, hi = spec; rng = np.where(hi - lo > 0, hi - lo, 1.0)
                out[i:i + len(lo)] = np.clip((leaf - lo) / rng * 2 - 1, -3, 3); i += len(lo)
        return out

class VecPOP:
    """Synchronous vector of POPGym envs with auto-reset. Batch-first torch tensors on `device`."""
    n_pn = 124
    def __init__(self, env_name, B, seed=0, device="cpu"):
        assert env_name in ENVS, env_name
        self.name, self.B, self.device = env_name, B, device
        self.envs = [getattr(PE, env_name)() for _ in range(B)]
        self.obs_space, self.act_space = self.envs[0].observation_space, self.envs[0].action_space
        self.flat = ObsFlattener(self.obs_space)
        if isinstance(self.act_space, gym.spaces.Discrete): self.act_nvec = [int(self.act_space.n)]
        elif isinstance(self.act_space, gym.spaces.MultiDiscrete): self.act_nvec = [int(n) for n in self.act_space.nvec]
        else: raise NotImplementedError(self.act_space)
        self.n_act_total = sum(self.act_nvec)
        self.obs_dim = self.flat.dim + self.n_act_total + 1                 # + prev action one-hots + prev reward
        g = torch.Generator().manual_seed(seed)
        self.Wpn = (torch.randn(self.obs_dim, self.n_pn, generator=g) / math.sqrt(self.obs_dim)).to(device)
        self.seed = seed
        self.prev_a = np.zeros((B, len(self.act_nvec)), dtype=np.int64); self.prev_r = np.zeros(B, dtype=np.float32)
        self.ep_ret = np.zeros(B, dtype=np.float32); self.ep_len = np.zeros(B, dtype=np.int64)
        self.n_pn_ = self.n_pn

    def _pn(self, x):
        h = torch.nn.functional.softplus(x @ self.Wpn)
        return h / (h.norm(dim=1, keepdim=True) + 1e-6) * 3.0

    def _pack(self, raw_obs):
        rows = []
        for i, o in enumerate(raw_obs):
            f = self.flat(o, self.obs_space)
            a1 = np.zeros(self.n_act_total, dtype=np.float32); off = 0
            for j, n in enumerate(self.act_nvec):
                if self.ep_len[i] > 0: a1[off + self.prev_a[i, j]] = 1.0
                off += n
            rows.append(np.concatenate([f, a1, [self.prev_r[i]]]))
        x = torch.tensor(np.stack(rows), device=self.device)
        return {"vec": x, "pn": self._pn(x)}

    def reset(self):
        raw = []
        for i, e in enumerate(self.envs):
            o, _ = e.reset(seed=self.seed * 1000 + i); raw.append(o)
        self.prev_a[:] = 0; self.prev_r[:] = 0; self.ep_ret[:] = 0; self.ep_len[:] = 0
        return self._pack(raw)

    def step(self, actions):
        """actions: LongTensor (B, n_components). Returns obs, reward (B,), done (B,), info with finished-episode returns."""
        a = actions.detach().cpu().numpy().astype(np.int64); raw = []
        r = np.zeros(self.B, dtype=np.float32); done = np.zeros(self.B, dtype=bool); finished = []
        for i, e in enumerate(self.envs):
            act = int(a[i, 0]) if len(self.act_nvec) == 1 else a[i]
            o, rew, term, trunc, _ = e.step(act)
            r[i] = rew; self.ep_ret[i] += rew; self.ep_len[i] += 1
            self.prev_a[i] = a[i]; self.prev_r[i] = rew
            if term or trunc:
                done[i] = True; finished.append((float(self.ep_ret[i]), int(self.ep_len[i])))
                o, _ = e.reset(); self.ep_ret[i] = 0; self.ep_len[i] = 0; self.prev_a[i] = 0; self.prev_r[i] = 0
            raw.append(o)
        obs = self._pack(raw)
        return obs, torch.tensor(r, device=self.device), torch.tensor(done, device=self.device), {"finished": finished}

def make_vec(env_name, B, seed=0, device="cpu"): return VecPOP(env_name, B, seed, device)

if __name__ == "__main__":
    import time
    for name in ["MultiarmedBanditEasy", "CountRecallMedium", "AutoencodeEasy", "PositionOnlyCartPoleEasy", "HigherLowerEasy"]:
        v = make_vec(name, 8); o = v.reset(); t0 = time.time(); fin = []
        for _ in range(300):
            a = torch.stack([torch.randint(0, n, (8,)) for n in v.act_nvec], 1)
            o, r, d, info = v.step(a); fin += info["finished"]
        print(f"{name:26s} obs_dim={v.obs_dim:3d} act={v.act_nvec} pn={tuple(o['pn'].shape)} {8*300/(time.time()-t0):6.0f} steps/s  episodes={len(fin)} mean_len={np.mean([l for _, l in fin]) if fin else 0:.0f}")
