"""Connectome-derived mushroom body critic.

Circuit (right hemisphere, MaleCNS v1.0):
  PN (124) --fixed--> KC (2045, sparse via APL) --plastic--> MBON (49) --> valence
  DAN (166: PAM reward / PPL1 punishment) release dopamine per compartment (15),
  gating depression of the KC->MBON synapses that were just active (three-factor rule).
Each compartment has its own learning rate and decay, giving short- and long-term memory.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# compartment priors (Aso et al. 2014; Hige 2015): gamma = fast/short, alpha'/beta' = mid, alpha/beta = slow/long
_PRIOR = {"g": (0.5, 150.0), "a'": (0.25, 800.0), "b'": (0.25, 800.0), "a": (0.08, 6000.0), "b": (0.08, 6000.0)}
def _prior(c):
    for k in ("a'", "b'", "g", "a", "b"):
        if c.startswith(k): return _PRIOR[k]

class MushroomBody(nn.Module):
    def __init__(self, path="data/mb_R.npz", shuffle=False, seed=0, sparsity=0.05,
                 learn_rates=True, feedback_gain=0.0, device="cpu", collapse=False, uniform=False, extra_pn=0,
                 ctx_frac=0.3, ctx_gain=1.0):
        """collapse: merge the 15 compartments into ONE (single eta/tau, pooled dopamine) — tests whether the
        compartment structure matters at all. uniform: keep 15 compartments but give them all the same eta/tau —
        tests whether the *diversity of timescales* matters. extra_pn: append this many extra input channels
        (e.g. context cues) that project onto KCs like the fly's multimodal KC inputs."""
        super().__init__()
        d = np.load(path, allow_pickle=True)
        rng = np.random.default_rng(seed)
        W_pk = d["W_pn_kc"].astype(np.float32)               # (P, K)
        M = d["W_kc_mbon"].astype(np.float32)                # (K, M, C)
        dan_comp, dan_fam = d["dan_comp_abs"].astype(np.float32), d["dan_family"]
        self.compartments = ["all"] if collapse else list(d["compartments"]); self.mbon_types = list(d["mbon_types"])
        self.pn_types = list(d["pn_types"])
        if shuffle:  # degree-preserving destruction of the specific wiring
            for k in range(W_pk.shape[1]): W_pk[:, k] = rng.permutation(W_pk[:, k])
            M = M[:, rng.permutation(M.shape[1])][:, :, rng.permutation(M.shape[2])]
            dan_comp = dan_comp[:, rng.permutation(dan_comp.shape[1])]
        # PN->KC: normalize each KC's olfactory input to unit sum FIRST ...
        W_pk = W_pk / np.maximum(W_pk.sum(0, keepdims=True), 1)
        if extra_pn:  # ... then add context channels (cf. the fly's visual/thermal KC inputs) as an ABSOLUTE weight
            # ctx_gain on a random ctx_frac of KCs, so context nudges near-threshold odor-driven KCs over the line
            # and the top-k code becomes an odor x context conjunction without swamping odor identity
            ext = (rng.random((extra_pn, W_pk.shape[1])) < ctx_frac).astype(np.float32) * ctx_gain
            W_pk = np.concatenate([W_pk, ext], 0)
        if collapse:
            M = M.sum(2, keepdims=True); dan_comp = dan_comp.sum(1, keepdims=True)
        self.collapse, self.uniform = collapse, uniform
        P, K = W_pk.shape; C = M.shape[2]; Mn = M.shape[1]
        self.P, self.K, self.M, self.C = P, K, Mn, C
        self.n_active = max(1, int(sparsity * K))
        self.register_buffer("W_pk", torch.tensor(W_pk))
        # KC->MBON baseline weight (K, M), each MBON's input normalized to 1; compartment fractions Fc (M, C)
        # scaled so a random KC code (n_active cells at ~1) drives each MBON to ~1 at baseline
        Wkm = M.sum(2); self.register_buffer("W0", torch.tensor(Wkm / np.maximum(Wkm.sum(0, keepdims=True), 1) / sparsity))
        Fc = M.sum(0); Fc = Fc / np.maximum(Fc.sum(1, keepdims=True), 1e-9)
        self.register_buffer("Fc", torch.tensor(Fc))
        # DAN -> compartment drive, split by family, normalized per compartment
        pam = (dan_comp * (dan_fam == 0)[:, None]).sum(0); ppl = (dan_comp * (dan_fam == 1)[:, None]).sum(0)
        tot = np.maximum(pam + ppl, 1e-9)
        self.register_buffer("pam_frac", torch.tensor((pam / tot).astype(np.float32)))  # (C,)
        # MBON valence sign: PPL1-dominated compartment -> approach MBON (+1); PAM-dominated -> avoidance MBON (-1)
        comp_sign = np.where(pam / tot > 0.5, -1.0, 1.0).astype(np.float32)
        self.register_buffer("mbon_sign", torch.tensor(Fc @ comp_sign))              # (M,) in [-1,1]
        # MBON -> DAN feedback (M, C) via DAN compartment drive
        fb = d["W_mbon_dan"].astype(np.float32) @ (dan_comp / np.maximum(dan_comp.sum(1, keepdims=True), 1))
        self.register_buffer("W_fb", torch.tensor(fb / max(fb.max(), 1e-9)))
        self.feedback_gain = feedback_gain
        if collapse or uniform:   # geometric mean of the compartment priors: eta ~0.22, tau ~900
            eta = torch.full((C,), 0.22); tau = torch.full((C,), 900.0)
        else:
            eta = torch.tensor([_prior(c)[0] for c in self.compartments]); tau = torch.tensor([_prior(c)[1] for c in self.compartments])
        self.log_eta = nn.Parameter(eta.log(), requires_grad=learn_rates)
        self.log_tau = nn.Parameter(tau.log(), requires_grad=learn_rates)
        self.to(device)

    # ---- state -------------------------------------------------------------
    def reset(self, B, device=None):
        device = device or self.W0.device
        return {"W": self.W0.unsqueeze(0).expand(B, -1, -1).clone(),
                "kc": torch.zeros(B, self.K, device=device), "v": torch.zeros(B, device=device),
                "mbon": torch.zeros(B, self.M, device=device)}

    def kenyon(self, x):
        h = x @ self.W_pk                                     # (B, K)
        thr = h.topk(self.n_active, dim=1).values[:, -1:]     # APL-style global inhibition -> ~5% active
        kc = F.relu(h - thr + 1e-6)
        return kc / (kc.sum(1, keepdim=True) / self.n_active + 1e-6)   # active cells average ~1

    def sense(self, state, x):
        """PN activity x (B, P) -> KC code, MBON vector, valence. Stores KC for eligibility."""
        kc = self.kenyon(x)
        mbon = torch.einsum("bk,bkm->bm", kc, state["W"])     # (B, M)
        v = (mbon * self.mbon_sign).sum(1) / (self.mbon_sign.abs().sum() + 1e-6)
        state = dict(state, kc=kc, mbon=mbon, v=v)
        return state, mbon, v

    def learn(self, state, r, punish=None):
        """Reward r (B,) arriving after the sensed stimulus -> dopamine per compartment -> depression of
        the KC->MBON synapses active at sense time. Returns new state and dopamine (B, C) (signed: +PAM, -PPL1)."""
        delta = r - state["v"].detach()                       # reward prediction error
        if self.feedback_gain > 0:                            # connectome MBON->DAN feedback
            delta = delta + self.feedback_gain * (state["mbon"] @ self.W_fb).mean(1)
        pam = F.relu(delta)[:, None] * self.pam_frac         # (B, C) reward dopamine per compartment
        ppl = F.relu(-delta)[:, None] * (1 - self.pam_frac)  # punishment dopamine per compartment
        dop = pam + ppl                                       # unsigned drive
        eta = self.log_eta.exp(); tau = self.log_tau.exp()
        g = (dop * eta) @ self.Fc.T                            # (B, M): dopamine-weighted lr at each MBON
        W = state["W"]
        W = W - W * state["kc"].unsqueeze(2) * g.unsqueeze(1)  # three-factor depression
        rec = (self.Fc / tau).sum(1)                           # (M,) recovery rate toward baseline
        W = W + (self.W0.unsqueeze(0) - W) * rec
        W = W.clamp_min(0)
        state = dict(state, W=W)
        signed = pam - ppl
        return state, signed, delta

    def features(self, state):
        return torch.cat([state["mbon"], state["v"].unsqueeze(1)], 1)     # (B, M+1)
