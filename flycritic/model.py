"""Small causal transformer policy with a dopamine-gated plastic output layer.

  y_t = W h_t  +  sum_c  alpha_c * H^c_t h_t
  H^c_{t+1} = (1 - 1/tau_c) H^c_t + eta_c * d_{c,t} * post_t (x) h_t        (three-factor Hebbian)

d_{c,t} is per-compartment signed dopamine from the mushroom-body critic (mode 'mb' / 'shuffled'),
a scalar TD error (mode 'scalar', one compartment), or absent (mode 'none').
Slow weights (transformer, W, alpha, eta, tau) are trained by PPO across episodes; H is reset each episode
and updated online by the rule, so gradients flow through the plasticity (differentiable plasticity).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, d, heads, drop=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(self, x, mask):
        h = self.ln1(x); x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))

class PlasticHead(nn.Module):
    """Fast weights from a presynaptic code `pre` (either the transformer hidden state h, or the fly's sparse
    Kenyon-cell code) to the output. `norm` ~ E[pre . pre] keeps the fast term O(eta * dopamine)."""
    def __init__(self, d, out, n_comp, eta0, tau0, d_pre=None, norm=None):
        super().__init__()
        self.W = nn.Linear(d, out)
        self.d_pre = d_pre or d; self.norm = norm or self.d_pre
        self.alpha = nn.Parameter(torch.full((n_comp, out, 1), 0.5 / n_comp))
        self.log_eta = nn.Parameter(torch.tensor(eta0).log()); self.log_tau = nn.Parameter(torch.tensor(tau0).log())
        self.d, self.out, self.C = d, out, n_comp
    def init_state(self, B, device): return torch.zeros(B, self.C, self.out, self.d_pre, device=device)
    def forward(self, h, H, pre):                              # h (B,d), pre (B,d_pre), H (B,C,out,d_pre)
        fast = torch.einsum("bcok,bk->bco", H * self.alpha, pre).sum(1)
        return self.W(h) + fast
    def update(self, H, pre, post, dop):                       # post (B,out), dop (B,C) signed
        eta, tau = self.log_eta.exp(), self.log_tau.exp()
        hebb = torch.einsum("bo,bk->bok", post, pre) / self.norm
        H = H * (1 - 1 / tau)[None, :, None, None] + (eta[None] * dop)[:, :, None, None] * hebb[:, None]
        return H.clamp(-5, 5)

class Agent(nn.Module):
    def __init__(self, obs_dim, n_actions, critic="mb", mb=None, d=128, layers=3, heads=4, ctx=100,
                 no_feat=False, no_plastic=False, pre="h", n_mod=15, prior=False):
        super().__init__()
        self.critic, self.mb, self.ctx, self.n_actions = critic, mb, ctx, n_actions
        self.no_feat, self.no_plastic = no_feat, no_plastic
        assert pre == "h" or mb is not None, "pre='kc' needs a mushroom body"
        self.pre = pre
        feat = 0
        if no_feat: pass
        elif critic in ("mb", "shuffled"): feat = mb.M + 1 + mb.C          # MBON vector, valence, dopamine
        elif critic == "scalar": feat = 2                                  # value estimate, TD error
        self.inp = nn.Linear(obs_dim + feat, d)
        self.pos = nn.Parameter(torch.randn(1, ctx, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        out = n_actions + 1
        self.mod = None
        if critic == "none" or no_plastic:
            self.head = nn.Linear(d, out); self.plastic = None
        else:
            if critic in ("mb", "shuffled"):
                C = mb.C; eta0 = mb.log_eta.exp().detach().tolist(); tau0 = mb.log_tau.exp().detach().tolist()
            elif critic == "hybrid":    # connectome RPE routing as initialization + learned residual correction
                C = mb.C; eta0 = mb.log_eta.exp().detach().tolist(); tau0 = mb.log_tau.exp().detach().tolist()
                self.mod = nn.Sequential(nn.Linear(d + 2 + C, 64), nn.Tanh(), nn.Linear(64, C))
                nn.init.zeros_(self.mod[2].weight); nn.init.zeros_(self.mod[2].bias)
            elif critic == "learned":   # Backpropamine-style: the network emits its own n_mod-channel modulator
                C = n_mod
                if prior and mb is not None and mb.C == C:
                    eta0 = mb.log_eta.exp().detach().tolist(); tau0 = mb.log_tau.exp().detach().tolist()
                else: eta0 = [0.22] * C; tau0 = [900.0] * C
                self.mod = nn.Sequential(nn.Linear(d + 2, 64), nn.Tanh(), nn.Linear(64, C), nn.Tanh())
            else:
                C = 1; eta0 = [0.3]; tau0 = [500.0]
            if pre == "kc": self.plastic = PlasticHead(d, out, C, eta0, tau0, d_pre=mb.K, norm=float(mb.n_active))
            else: self.plastic = PlasticHead(d, out, C, eta0, tau0)
        self.d = d

    def trunk(self, x):                                       # x (B,L,in) -> (B,L,d)
        L = x.shape[1]
        h = self.inp(x) + self.pos[:, :L]
        mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        for b in self.blocks: h = b(h, mask)
        return self.ln(h)

    def init_state(self, B, device):
        st = {"H": self.plastic.init_state(B, device) if self.plastic else None,
              "hist": torch.zeros(B, 0, self.inp.in_features, device=device),
              "v_prev": torch.zeros(B, device=device)}
        if self.mb is not None: st["mb"] = self.mb.reset(B, device)
        return st

    def features(self, st, obs, dop_prev):
        """Extra input features for this step (critic-dependent)."""
        if self.critic in ("mb", "shuffled") or (self.mb is not None and self.pre == "kc"):
            st["mb"], mbon, v = self.mb.sense(st["mb"], obs["pn"])       # (for 'learned'/pre=kc this only computes the KC code)
            if self.no_feat or self.critic not in ("mb", "shuffled"): return torch.zeros(mbon.shape[0], 0, device=mbon.device)
            return torch.cat([mbon, v[:, None], dop_prev], 1)
        if self.no_feat: return torch.zeros(obs["vec"].shape[0], 0, device=obs["vec"].device)
        if self.critic == "scalar":
            return torch.stack([st["v_prev"], dop_prev[:, 0]], 1)
        return torch.zeros(obs["vec"].shape[0], 0, device=obs["vec"].device)

    def presyn(self, st, h):
        return st["mb"]["kc"] if self.pre == "kc" else h

    def act(self, st, obs, dop_prev):
        """One online step. Returns logits, value, presynaptic code, updated state."""
        x = torch.cat([obs["vec"], self.features(st, obs, dop_prev)], 1)
        st["hist"] = torch.cat([st["hist"], x[:, None]], 1)[:, -self.ctx:]
        h = self.trunk(st["hist"])[:, -1]
        pre = self.presyn(st, h)
        y = self.plastic(h, st["H"], pre) if self.plastic else self.head(h)
        logits, value = y[:, :-1], y[:, -1]
        st["v_prev"] = value.detach(); st["h"] = h
        return logits, value, pre, x

    def learned_dopamine(self, h, r, value, fixed=None):
        if self.critic == "hybrid":
            return (fixed + self.mod(torch.cat([h, r[:, None], value[:, None], fixed], 1))).clamp(-1.5, 1.5)
        return self.mod(torch.cat([h, r[:, None], value[:, None]], 1))

    def dopamine(self, st, r, value, value_next_est=None):
        """Compute signed per-compartment dopamine for the step that just produced reward r."""
        if self.critic in ("mb", "shuffled"):
            st["mb"], dop, delta = self.mb.learn(st["mb"], r)
            return dop, delta
        if self.critic == "scalar":
            delta = (r - value.detach())
            return delta[:, None], delta
        if self.critic == "learned":
            return self.learned_dopamine(st["h"], r, value).detach(), None
        if self.critic == "hybrid":
            st["mb"], fixed, delta = self.mb.learn(st["mb"], r)
            st["fixed"] = fixed
            return self.learned_dopamine(st["h"], r, value, fixed).detach(), delta
        return None, None

    def plastic_update(self, st, pre, a, dop):
        if self.plastic is None or dop is None: return
        post = torch.cat([F.one_hot(a, self.n_actions).float(), torch.ones(a.shape[0], 1, device=pre.device)], 1)
        st["H"] = self.plastic.update(st["H"], pre, post, dop)
