"""LLM hidden state -> the fly critic's 124 'PN' channels.
Fixed random Gaussian projection (seeded), softplus, then rescaled to L2 norm 3.0 (the OdorChoice odour norm), so
MushroomBody.kenyon() can be reused unchanged. Nothing here is trained."""
import torch, torch.nn.functional as F
n_pn = 124

class Bridge:
    def __init__(self, d_model, seed=0, n_pn=n_pn, norm=3.0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(d_model, n_pn, generator=g) / d_model ** 0.5
        self.norm = norm
    def __call__(self, hidden):                       # (B, d_model) -> (B, n_pn), nonneg, norm 3
        h = hidden - hidden.mean(1, keepdim=True)
        x = F.softplus(h @ self.W.to(hidden.device))
        return x / (x.norm(dim=1, keepdim=True) + 1e-6) * self.norm
