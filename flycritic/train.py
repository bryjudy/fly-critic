"""PPO training of the plastic transformer on OdorGrid with a chosen critic.
usage: uv run python -m flycritic.train --critic mb --iters 1500 --tag mb_v1
"""
import argparse, json, os, time, torch, torch.nn.functional as F, numpy as np
from .env import OdorGrid, OdorChoice, OdorContext
from .bench import Bandit, DarkRoom
from .mb import MushroomBody
from .model import Agent

def rollout(agent, env, device):
    """Run one full episode online, storing everything needed for PPO. Plastic state is recomputed in
    the update pass, so we store inputs x, actions, rewards, values, logps, dopamine."""
    obs = env.reset(); B = env.B
    st = agent.init_state(B, device)
    dop_prev = torch.zeros(B, agent.plastic.C if agent.plastic else (agent.mb.C if agent.mb else 1), device=device)
    X, A, R, V, LP, DOP, INFO, PN, FIX = [], [], [], [], [], [], [], [], []
    with torch.no_grad():
        for t in range(env.T):
            PN.append(obs["pn"])
            logits, value, h, x = agent.act(st, obs, dop_prev)
            dist = torch.distributions.Categorical(logits=logits); a = dist.sample()
            obs, r, done, info = env.step(a)
            dop, _ = agent.dopamine(st, r, value)
            agent.plastic_update(st, h, a, dop)
            dop_prev = dop if dop is not None else dop_prev
            X.append(x); A.append(a); R.append(r); V.append(value); LP.append(dist.log_prob(a)); DOP.append(dop)
            FIX.append(st.get("fixed"))
            INFO.append(info)
    return dict(X=torch.stack(X, 1), A=torch.stack(A, 1), R=torch.stack(R, 1), V=torch.stack(V, 1),
                LP=torch.stack(LP, 1), DOP=torch.stack(DOP, 1) if DOP[0] is not None else None, INFO=INFO,
                PN=torch.stack(PN, 1), FIX=torch.stack(FIX, 1) if FIX[0] is not None else None)

def replay(agent, batch, idx):
    """Recompute logits/values along stored sequences (teacher-forced on stored inputs & dopamine) with
    gradients through the plastic recurrence."""
    X, A = batch["X"][idx], batch["A"][idx]
    B, T, _ = X.shape; device = X.device
    H = agent.plastic.init_state(B, device) if agent.plastic else None
    logits_all, values_all = [], []
    assert T <= agent.ctx, "episode must fit in context"
    hs = agent.trunk(X)                                  # (B,T,d) causal, identical to the online computation
    if agent.plastic and agent.pre == "kc":              # KC code is a deterministic function of the stored PN input
        with torch.no_grad(): KC = agent.mb.kenyon(batch["PN"][idx].reshape(B * T, -1)).view(B, T, -1)
    R = batch["R"][idx]
    for t in range(T):
        h = hs[:, t]
        pre = KC[:, t] if (agent.plastic and agent.pre == "kc") else h
        y = agent.plastic(h, H, pre) if agent.plastic else agent.head(h)
        logits_all.append(y[:, :-1]); values_all.append(y[:, -1])
        if agent.plastic:
            post = torch.cat([F.one_hot(A[:, t], agent.n_actions).float(), torch.ones(B, 1, device=device)], 1)
            # learned modulator is recomputed WITH gradient so PPO trains it through the plasticity (Backpropamine)
            if agent.critic == "learned": dop = agent.learned_dopamine(h, R[:, t], y[:, -1])
            elif agent.critic == "hybrid": dop = agent.learned_dopamine(h, R[:, t], y[:, -1], batch["FIX"][idx][:, t])
            else: dop = batch["DOP"][idx][:, t]
            H = agent.plastic.update(H, pre, post, dop)
    return torch.stack(logits_all, 1), torch.stack(values_all, 1)

def gae(R, V, gamma=0.97, lam=0.95):
    T = R.shape[1]; adv = torch.zeros_like(R); last = torch.zeros_like(R[:, 0])
    for t in reversed(range(T)):
        nv = V[:, t + 1] if t + 1 < T else torch.zeros_like(last)
        delta = R[:, t] + gamma * nv - V[:, t]
        last = delta + gamma * lam * last; adv[:, t] = last
    return adv, adv + V

def metrics(batch, env):
    R = batch["R"]; T = R.shape[1]
    post = torch.stack([i["post_reversal"] for i in batch["INFO"]], 1)
    opt_ = torch.stack([i["optimal"] for i in batch["INFO"]], 1).float() if "optimal" in batch["INFO"][0] else None
    return dict(reward=R.sum(1).mean().item(), optimal=opt_.mean().item() if opt_ is not None else None,
                reward_pre=(R * (~post)).sum(1).mean().item(), reward_post=(R * post).sum(1).mean().item(),
                punish_hits=(R < 0).float().sum(1).mean().item(), reward_hits=(R > 0).float().sum(1).mean().item())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--critic", default="mb", choices=["mb", "shuffled", "scalar", "none", "learned", "hybrid"])
    p.add_argument("--randkc", action="store_true", help="use a shuffled (random) PN->KC expansion of the same size")
    p.add_argument("--prior", action="store_true", help="learned critic: init eta/tau from the fly compartment priors")
    p.add_argument("--iters", type=int, default=1000); p.add_argument("--B", type=int, default=64)
    p.add_argument("--T", type=int, default=100); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=3); p.add_argument("--mb_lr", type=float, default=1e-3)
    p.add_argument("--mb_splits", type=int, default=2, help="PPO minibatches per epoch (raise to cut replay memory: fast-weight autograd state ~ B/splits x T x C x out x K)")
    p.add_argument("--tag", default=None); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu"); p.add_argument("--grid", type=int, default=6)
    p.add_argument("--layers", type=int, default=3); p.add_argument("--d", type=int, default=128)
    p.add_argument("--env", default="choice", choices=["choice", "grid", "context", "bandit", "darkroom", "keydoor", "toolworld"])
    p.add_argument("--llm", default="0.5b", help="toolworld: frozen LLM alias/model id")
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ctx_to_kc", action="store_true", help="context task: feed context cue into the mushroom body too")
    p.add_argument("--collapse", action="store_true", help="ablation: 15 compartments -> 1")
    p.add_argument("--uniform", action="store_true", help="ablation: 15 compartments, identical eta/tau")
    p.add_argument("--ctx_frac", type=float, default=0.3); p.add_argument("--ctx_gain", type=float, default=0.1)
    p.add_argument("--ent", type=float, default=0.01, help="entropy coefficient")
    p.add_argument("--no_feat", action="store_true", help="critic gates plasticity but its outputs are hidden from the transformer")
    p.add_argument("--no_plastic", action="store_true", help="critic outputs fed as features, no plastic head")
    p.add_argument("--pre", default="h", choices=["h", "kc"], help="presynaptic side of the fast weights")
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device(a.device)
    if a.env == "context": env = OdorContext(a.B, T=a.T, device=device, seed=a.seed, ctx_to_kc=a.ctx_to_kc)
    elif a.env == "bandit": env = Bandit(a.B, T=a.T, device=device, seed=a.seed)
    elif a.env == "toolworld":
        from .llm.adapter import ToolWorldEnv
        env = ToolWorldEnv(a.B, T=a.T, device=device, seed=a.seed, llm=a.llm, cache_path=None)
    elif a.env in ("darkroom", "keydoor"): env = DarkRoom(a.B, T=a.T, device=device, seed=a.seed, key_door=(a.env == "keydoor"))
    else:
        Env = OdorChoice if a.env == "choice" else OdorGrid
        env = Env(a.B, T=a.T, grid=a.grid, device=device, seed=a.seed)
    extra = 2 if (a.env == "context" and a.ctx_to_kc) else 0
    mb = MushroomBody(shuffle=(a.critic == "shuffled" or a.randkc), seed=a.seed, device=device, collapse=a.collapse,
                      uniform=a.uniform, extra_pn=extra, ctx_frac=a.ctx_frac, ctx_gain=a.ctx_gain) if (a.critic in ("mb", "shuffled", "hybrid") or a.pre == "kc") else None
    agent = Agent(env.obs_dim, env.n_actions, critic=a.critic, mb=mb, d=a.d, layers=a.layers, heads=a.heads, ctx=a.T,
                  no_feat=a.no_feat, no_plastic=a.no_plastic, pre=a.pre, prior=a.prior).to(device)
    params = [{"params": [q for n, q in agent.named_parameters() if not n.startswith("mb.")], "lr": a.lr}]
    if mb is not None and a.critic in ("mb", "shuffled", "hybrid"): params.append({"params": list(mb.parameters()), "lr": a.mb_lr})
    opt = torch.optim.Adam(params)
    tag = a.tag or f"{a.critic}_s{a.seed}"; os.makedirs(f"runs/{tag}", exist_ok=True)
    log = open(f"runs/{tag}/log.jsonl", "a"); t0 = time.time()
    n_params = sum(q.numel() for n, q in agent.named_parameters() if not n.startswith("mb."))
    print(f"[{tag}] agent params {n_params/1e6:.2f}M critic={a.critic} device={device}")
    for it in range(a.iters):
        agent.eval(); batch = rollout(agent, env, device); agent.train()
        adv, ret = gae(batch["R"], batch["V"]); adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for ep in range(a.epochs):
            perm = torch.randperm(a.B, device=device)
            for mb_idx in perm.split(max(1, a.B // a.mb_splits)):
                logits, values = replay(agent, batch, mb_idx)
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(batch["A"][mb_idx]); ratio = (lp - batch["LP"][mb_idx]).exp()
                ad = adv[mb_idx]
                pg = -torch.min(ratio * ad, ratio.clamp(0.8, 1.2) * ad).mean()
                vl = F.mse_loss(values, ret[mb_idx]); ent = dist.entropy().mean()
                loss = pg + 0.5 * vl - a.ent * ent
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0); opt.step()
        m = metrics(batch, env); m.update(it=it, loss=loss.item(), ent=ent.item(), t=time.time() - t0)
        if mb is not None: m["eta"] = mb.log_eta.exp().detach().cpu().round(decimals=3).tolist()
        log.write(json.dumps(m) + "\n"); log.flush()
        if it % 10 == 0:
            print(f"it {it:4d} R {m['reward']:6.2f} pre {m['reward_pre']:6.2f} post {m['reward_post']:6.2f} "
                  f"+hits {m['reward_hits']:.1f} -hits {m['punish_hits']:.1f} opt {m['optimal'] or 0:.2f} ent {m['ent']:.2f} {m['t']:.0f}s", flush=True)
        if it % 100 == 0 or it == a.iters - 1:
            torch.save({"agent": agent.state_dict(), "args": vars(a)}, f"runs/{tag}/ckpt.pt")

if __name__ == "__main__":
    main()
