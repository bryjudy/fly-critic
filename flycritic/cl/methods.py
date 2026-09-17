"""Continual-learning classifiers on frozen 512-d features. Common interface:
    m.train_task(x, y, task_id, classes)   # x (N,512) y (N,) global class ids in [0,100); classes = this task's ids
    m.predict(x) -> logits (N, n_classes)
SGD methods (finetune / ewc / replay / offline) see each task's data for `epochs` epochs (default 5).
Hebbian methods (flymodel / flycritic) are strictly online: every example is seen exactly once, in order."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from ..mb import MushroomBody

# ---------------------------------------------------------------- SGD-family --------------------------------------
class Linear:
    def __init__(self, d, n_classes, epochs=5, lr=1e-3, batch=64, seed=0, **kw):
        torch.manual_seed(seed); self.head = nn.Linear(d, n_classes); self.epochs, self.lr, self.batch = epochs, lr, batch
        self.opt = torch.optim.Adam(self.head.parameters(), lr=lr)
    def _loss(self, xb, yb): return F.cross_entropy(self.head(xb), yb)
    def _after_task(self, x, y): pass
    def _extra(self, xb, yb): return xb, yb
    def train_task(self, x, y, task_id, classes):
        n = len(x)
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]; xb, yb = self._extra(x[idx], y[idx])
                loss = self._loss(xb, yb); self.opt.zero_grad(); loss.backward(); self.opt.step()
        self._after_task(x, y)
    @torch.no_grad()
    def predict(self, x): return self.head(x)

class Finetune(Linear): pass

class Offline(Linear):
    """Upper bound: accumulates all data and retrains jointly after each task."""
    def __init__(self, *a, **kw): super().__init__(*a, **kw); self.X, self.Y = [], []
    def train_task(self, x, y, task_id, classes):
        self.X.append(x); self.Y.append(y); X, Y = torch.cat(self.X), torch.cat(self.Y)
        self.head.reset_parameters(); self.opt = torch.optim.Adam(self.head.parameters(), lr=self.lr)
        super().train_task(X, Y, task_id, classes)

class EWC(Linear):
    def __init__(self, *a, lam=10.0, **kw): super().__init__(*a, **kw); self.lam = lam; self.anchors = []   # (params, fisher)
    def _loss(self, xb, yb):
        loss = F.cross_entropy(self.head(xb), yb)
        for params, fisher in self.anchors:
            for p, p0, f in zip(self.head.parameters(), params, fisher): loss = loss + (self.lam / 2) * (f * (p - p0) ** 2).sum()
        return loss
    def _after_task(self, x, y):
        fisher = [torch.zeros_like(p) for p in self.head.parameters()]
        for i in range(0, len(x), 64):
            self.head.zero_grad(); F.cross_entropy(self.head(x[i:i + 64]), y[i:i + 64]).backward()
            for f, p in zip(fisher, self.head.parameters()): f += p.grad.detach() ** 2 * len(x[i:i + 64])
        fisher = [f / len(x) for f in fisher]
        self.anchors.append(([p.detach().clone() for p in self.head.parameters()], fisher))

class Replay(Linear):
    def __init__(self, *a, buffer=200, **kw):
        super().__init__(*a, **kw); self.cap = buffer; self.bx, self.by, self.seen = [], [], 0
        self.g = torch.Generator().manual_seed(kw.get("seed", 0) + 1)
    def _extra(self, xb, yb):
        if self.bx:
            k = min(len(self.bx), len(xb)); idx = torch.randperm(len(self.bx), generator=self.g)[:k]
            xb = torch.cat([xb, torch.stack([self.bx[i] for i in idx])]); yb = torch.cat([yb, torch.stack([self.by[i] for i in idx])])
        return xb, yb
    def _after_task(self, x, y):   # reservoir sampling
        for xi, yi in zip(x, y):
            self.seen += 1
            if len(self.bx) < self.cap: self.bx.append(xi); self.by.append(yi)
            else:
                j = torch.randint(0, self.seen, (1,), generator=self.g).item()
                if j < self.cap: self.bx[j] = xi; self.by[j] = yi

# ---------------------------------------------------------------- Hebbian family ----------------------------------
class FlyModel:
    """Shen, Dasgupta & Navlakha (2021), 'Algorithmic insights on continual learning from fruit flies':
    sparse random binary expansion (each KC samples a random subset of inputs), top-k winner-take-all (~5%),
    associative Hebbian weights KC->class updated ONLY in the column of the current class ("partial freezing":
    weights of other classes' output units are never touched, so old associations cannot be overwritten).
    Prediction = argmax_c  code . W[:, c] / n_c  (frequency-normalized)."""
    def __init__(self, d, n_classes, K=2045, sparsity=0.05, fan_in=0.1, seed=0, **kw):
        g = torch.Generator().manual_seed(seed)
        self.proj = (torch.rand(d, K, generator=g) < fan_in).float()      # binary random projection
        self.k = max(1, int(sparsity * K)); self.W = torch.zeros(K, n_classes); self.n = torch.zeros(n_classes)
        self.mu = None
    def code(self, x):
        x = x - (self.mu if self.mu is not None else 0)                   # center features (running mean of seen data)
        h = x @ self.proj; thr = h.topk(self.k, dim=1).values[:, -1:]; return (h >= thr).float()
    def train_task(self, x, y, task_id, classes):
        self.mu = x.mean(0) if self.mu is None else 0.5 * (self.mu + x.mean(0))
        for xi, yi in zip(x, y):                                            # strictly online, one pass
            c = self.code(xi[None])[0]; self.W[:, yi] += c; self.n[yi] += 1
    @torch.no_grad()
    def predict(self, x): return self.code(x) @ (self.W / self.n.clamp_min(1))

class FlyCritic:
    """Our module: features -> fixed random projection to 124 'PN' channels (softplus, norm 3) -> MushroomBody.kenyon()
    (connectome PN->KC wiring, APL-style top-5%) -> 15 compartment blocks of fast weights KC->class, each with its
    own learning rate eta_c and lifetime tau_c (fly priors), updated ONLINE by a three-factor rule:
        H_c <- (1 - 1/tau_c) H_c + eta_c * outer(kc, d),   d = +1 at true class, -1 at the predicted (wrong) class.
    Lifetimes are given in the fly's units (steps of a ~100-trial episode); here one 'step' is one example, so we
    scale tau by (examples per task / 100): gamma compartments then forget within ~1.5 tasks, alpha/beta essentially
    never (~60 tasks). `collapse=True` -> a single compartment with the geometric-mean eta/tau (ablation).
    `fit_gain=True` fits per-compartment readout gains alpha_c by logistic regression on task 0 only."""
    def __init__(self, d, n_classes, seed=0, collapse=False, fit_gain=False, examples_per_task=5000, tau_scale=None, **kw):
        g = torch.Generator().manual_seed(seed)
        self.mb = MushroomBody(collapse=collapse); self.K, self.C = self.mb.K, self.mb.C
        self.Wpn = torch.randn(d, self.mb.P, generator=g) / math.sqrt(d)
        eta = self.mb.log_eta.exp().detach().clone(); tau = self.mb.log_tau.exp().detach().clone()
        scale = tau_scale if tau_scale is not None else examples_per_task / 100.0
        self.eta, self.tau = eta, tau * scale
        self.H = torch.zeros(self.C, self.K, n_classes); self.alpha = torch.full((self.C,), 1.0 / self.C)
        self.fit_gain, self.n_classes, self.mu = fit_gain, n_classes, None; self.t = 0
    def code(self, x):
        x = x - (self.mu if self.mu is not None else 0)
        pn = F.softplus(x @ self.Wpn); pn = pn / (pn.norm(dim=1, keepdim=True) + 1e-6) * 3.0
        return self.mb.kenyon(pn)
    def _logits_per_comp(self, kc): return torch.einsum("ckn,bk->cbn", self.H, kc)       # (C,B,N)
    def train_task(self, x, y, task_id, classes):
        self.mu = x.mean(0) if self.mu is None else 0.5 * (self.mu + x.mean(0))
        decay = (1 - 1 / self.tau)[:, None, None]
        for xi, yi in zip(x, y):
            kc = self.code(xi[None])                                       # (1,K)
            logits = (self.alpha[:, None, None] * self._logits_per_comp(kc)).sum(0)[0]
            pred = logits.argmax().item()
            d = torch.zeros(self.n_classes); d[yi] = 1.0
            if pred != yi.item(): d[pred] = -1.0
            self.H = self.H * decay + self.eta[:, None, None] * torch.einsum("k,n->kn", kc[0], d)[None]
            self.t += 1
        if self.fit_gain and task_id == 0: self._fit_alpha(x, y)
    def _fit_alpha(self, x, y):
        with torch.no_grad(): L = self._logits_per_comp(self.code(x))       # (C,N,classes)
        a = torch.nn.Parameter(torch.zeros(self.C)); opt = torch.optim.Adam([a], lr=0.05)
        for _ in range(200):
            loss = F.cross_entropy((F.softplus(a)[:, None, None] * L).sum(0), y); opt.zero_grad(); loss.backward(); opt.step()
        self.alpha = F.softplus(a).detach(); self.alpha = self.alpha / self.alpha.sum()
    @torch.no_grad()
    def predict(self, x): return (self.alpha[:, None, None] * self._logits_per_comp(self.code(x))).sum(0)

METHODS = {"finetune": Finetune, "offline": Offline, "ewc": EWC, "replay": Replay, "flymodel": FlyModel,
           "flycritic": FlyCritic, "flycritic_collapsed": lambda *a, **kw: FlyCritic(*a, collapse=True, **kw),
           "flycritic_gain": lambda *a, **kw: FlyCritic(*a, fit_gain=True, **kw)}
