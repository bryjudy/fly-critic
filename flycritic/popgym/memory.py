"""Memory modules for the POPGym harness. Common interface:
    mem.init_state(B, device) -> state
    h, state = mem(x_t, state, done_t)          # x_t (B, d_in) one timestep; done_t (B,) bool resets that env's state
    mem.out_dim
Three modules:
  GRU  — nn.GRUCell, hidden 128 (POPGym uses a 256-d recurrent state budget; we keep 128 to match the Fly's GRU+fast
         weight budget in parameter count order-of-magnitude; see results.py for the published GRU numbers).
  FFM  — Fast and Forgetful Memory (Morad et al. 2023, arXiv 2310.04128), ported from github.com/proroklab/ffm
         (standalone/ffm/ffm.py + ffa.py). Recurrent form of the aggregator:
             s_t = gamma * s_{t-1} + (y_t * in_gate_t)           gamma = exp(a + i*b), a in R^m (decay, learned,
                                                                  init linspace so traces are 'forgotten' at max_period),
                                                                  b in R^c (oscillation, init 2*pi/linspace(min,max period))
             y_t, thru_t, gates_t = Linear(x_t)                    (m, out, m+out)
             z_t = Linear(view_as_real(s_t))                       (2*m*c -> out)
             h_t = LayerNorm(z_t) * out_gate_t + thru_t * (1 - out_gate_t)
         Defaults from the repo: memory_size m=32, context_size c=4, min_period 1, max_period 1024. State is complex.
  Fly  — the fly-critic module: 124-channel PN vector -> MushroomBody.kenyon() sparse code (2045 units, 5% active) ->
         compartmentalized fast weights H (15 blocks, eta/tau initialized from the fly compartment priors and learnable)
         read out to a 128-d output; H updated online by the three-factor rule
             H_c <- (1 - 1/tau_c) H_c + eta_c * d_c * (post (x) kc)
         where the 15-channel modulator d = tanh(MLP([h_gru, r_{t-1}, v_{t-1}])) is LEARNED (the winning config in the
         fly-critic head-to-head), post = the module's own output activity (bounded), kc = the sparse code.
         Plus a small GRUCell (hidden 64) on the raw observation for ordinary short-term state; its output is concatenated
         with the fast-weight readout. Fast weights AND eligibility reset per env on done. This is the method.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from ..mb import MushroomBody

class GRU(nn.Module):
    def __init__(self, d_in, hidden=128, **kw):
        super().__init__(); self.cell = nn.GRUCell(d_in, hidden); self.out_dim = hidden; self.hidden = hidden
    def init_state(self, B, device): return torch.zeros(B, self.hidden, device=device)
    def forward(self, x, state, done):
        state = state * (~done).float()[:, None]
        h = self.cell(x, state); return h, h

class FFM(nn.Module):
    def __init__(self, d_in, out=128, memory_size=32, context_size=4, min_period=1, max_period=1024, forgotten_at=0.01, **kw):
        super().__init__()
        self.m, self.c, self.out_dim = memory_size, context_size, out
        self.pre = nn.Linear(d_in, 2 * memory_size + 2 * out)
        self.mix = nn.Linear(2 * memory_size * context_size, out)
        self.ln = nn.LayerNorm(out, elementwise_affine=False)
        a_high = math.log(forgotten_at) / max_period            # decay so a trace is 1% at max_period
        self.a = nn.Parameter(torch.linspace(-1.0, a_high, memory_size))            # real (decay) per memory channel
        self.b = nn.Parameter(2 * math.pi / torch.linspace(min_period, max_period, context_size))  # imag (oscillation)
    def init_state(self, B, device): return torch.zeros(B, self.m, self.c, dtype=torch.complex64, device=device)
    def forward(self, x, state, done):
        state = state * (~done).to(state.dtype)[:, None, None]
        y, thru, gate = self.pre(x).split([self.m, self.out_dim, self.m + self.out_dim], -1)
        in_gate, out_gate = gate.sigmoid().split([self.m, self.out_dim], -1)
        a = self.a.clamp(max=-1e-6)
        gamma = torch.exp(torch.complex(a[:, None].expand(self.m, self.c), self.b[None, :].expand(self.m, self.c)))
        state = gamma[None] * state + (y * in_gate).to(torch.complex64)[:, :, None]
        z = self.mix(torch.view_as_real(state).reshape(x.shape[0], -1))
        h = self.ln(z) * out_gate + thru * (1 - out_gate)
        return h, state

class Fly(nn.Module):
    def __init__(self, d_in, n_pn=124, out=128, gru_hidden=120, mb_path="data/mb_R.npz", n_mod=15, kc_sub=512, fast_out=8, seed=0, **kw):
        """kc_sub: number of Kenyon cells used (a fixed random subset of the 2045 connectome KCs, 5% active) — the
        full 2045 x 15 x fast_out state was too costly for 15M-step RL; the fly-critic random-expansion result says the
        specific KC wiring is not what matters. fast_out: width of the fast-weight readout (concatenated with the GRU)."""
        super().__init__()
        self.mb = MushroomBody(mb_path, learn_rates=False)      # used for its PN->KC weights and eta/tau priors only
        for p in self.mb.parameters(): p.requires_grad_(False)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(self.mb.K, generator=g)[:kc_sub]
        self.register_buffer("W_pk", self.mb.W_pk[:, idx].clone())          # (124, kc_sub), each KC's input sums to 1
        self.K, self.C = kc_sub, n_mod; self.n_active = max(1, int(0.05 * kc_sub))
        self.gru = nn.GRUCell(d_in, gru_hidden)
        self.fast_out = fast_out; self.out_dim = gru_hidden + fast_out
        eta0 = self.mb.log_eta.exp().detach(); tau0 = self.mb.log_tau.exp().detach()
        assert len(eta0) == n_mod
        self.log_eta = nn.Parameter(eta0.log()); self.log_tau = nn.Parameter(tau0.log())
        self.alpha = nn.Parameter(torch.full((n_mod, 1, 1), 0.5 / n_mod))
        self.mod = nn.Sequential(nn.Linear(gru_hidden + 2, 64), nn.Tanh(), nn.Linear(64, n_mod), nn.Tanh())
        self.post_proj = nn.Linear(gru_hidden, self.fast_out)   # what gets written: bounded activity of the readout
        self.n_pn = n_pn
    def kenyon(self, x):                                     # same rule as MushroomBody.kenyon on the KC subset
        h = x @ self.W_pk; thr = h.topk(self.n_active, dim=1).values[:, -1:]
        kc = F.relu(h - thr + 1e-6); return kc / (kc.sum(1, keepdim=True) / self.n_active + 1e-6)
    def init_state(self, B, device):
        return {"g": torch.zeros(B, self.gru.hidden_size, device=device),
                "H": torch.zeros(B, self.C, self.fast_out, self.K, device=device),
                "kc": torch.zeros(B, self.K, device=device), "post": torch.zeros(B, self.fast_out, device=device),
                "r": torch.zeros(B, device=device), "v": torch.zeros(B, device=device)}
    def forward(self, x, state, done, pn=None, r_prev=None):
        """x: (B, d_in) obs incl. prev action/reward; pn: (B, 124) PN vector for the KC code; r_prev: (B,) last reward."""
        keep = (~done).float()
        g = state["g"] * keep[:, None]; H = state["H"] * keep[:, None, None, None]
        # 1) write: dopamine from the PREVIOUS step's context, applied to the previous step's eligibility (kc, post)
        d = self.mod(torch.cat([g, state["r"][:, None], state["v"][:, None]], 1)) * keep[:, None]      # (B, C)
        eta, tau = self.log_eta.exp(), self.log_tau.exp()
        hebb = torch.einsum("bo,bk->bok", state["post"], state["kc"]) / self.n_active
        H = H * (1 - 1 / tau)[None, :, None, None] + (eta[None] * d)[:, :, None, None] * hebb[:, None]
        H = H.clamp(-5, 5)
        # 2) read
        g = self.gru(x, g)
        kc = self.kenyon(pn)
        fast = torch.einsum("bcok,bk->bo", H * self.alpha, kc)
        post = torch.tanh(self.post_proj(g) + fast)
        h = torch.cat([g, post], 1)
        new = {"g": g, "H": H, "kc": kc, "post": post.detach(), "r": r_prev if r_prev is not None else state["r"], "v": state["v"]}
        return h, new

MEMORIES = {"gru": GRU, "ffm": FFM, "fly": Fly}
