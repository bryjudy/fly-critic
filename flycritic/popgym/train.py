"""PPO for POPGym (variable-length episodes) with a pluggable memory module.
Protocol (POPGym, Morad et al. 2023, Table 2) where we can match it: gamma 0.99, GAE lambda 1.0, PPO clip 0.3,
value-loss coef 1.0, value clip 0.3, entropy 0.0, lr 5e-5, 30 SGD iters/epoch, batch 65536 steps (B envs x L),
minibatch 8192, actor/critic = 2-layer MLPs of width 128 with LeakyReLU, obs pre-projected to 128. Deviations: we
use truncated BPTT of L=128 (they use infinity with max episode 1024) and carry memory state across chunks; the KL
penalty (target 0.01, coef 0.2) is replaced by the clip alone. MMER = max over epochs of the mean episodic reward of
episodes completed during that epoch (their definition).
usage: uv run python -m flycritic.popgym.train --env MultiarmedBanditEasy --memory fly --steps 3000000 --seed 0 --tag popgym_fly_MultiarmedBanditEasy_s0
"""
import argparse, json, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .envs import make_vec, ENVS
from .memory import MEMORIES, Fly

class Policy(nn.Module):
    def __init__(self, obs_dim, act_nvec, memory, d=128):
        super().__init__()
        self.pre = nn.Sequential(nn.Linear(obs_dim, d), nn.LeakyReLU())
        self.mem = MEMORIES[memory](d)
        self.post = nn.Sequential(nn.Linear(self.mem.out_dim, d), nn.LeakyReLU())
        self.pi = nn.Sequential(nn.Linear(d, d), nn.LeakyReLU(), nn.Linear(d, sum(act_nvec)))
        self.v = nn.Sequential(nn.Linear(d, d), nn.LeakyReLU(), nn.Linear(d, 1))
        self.act_nvec, self.is_fly = act_nvec, isinstance(self.mem, Fly)
    def init_state(self, B, device): return self.mem.init_state(B, device)
    def step(self, obs, state, done, r_prev):
        x = self.pre(obs["vec"])
        if self.is_fly: h, state = self.mem(x, state, done, pn=obs["pn"], r_prev=r_prev)
        else: h, state = self.mem(x, state, done)
        z = self.post(h); logits = self.pi(z); v = self.v(z).squeeze(1)
        if self.is_fly: state["v"] = v.detach()
        return logits, v, state
    def dists(self, logits):
        return [torch.distributions.Categorical(logits=l) for l in logits.split(self.act_nvec, -1)]

def detach_state(s):
    if isinstance(s, dict): return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in s.items()}
    return s.detach()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", required=True, choices=ENVS); p.add_argument("--memory", default="fly", choices=list(MEMORIES))
    p.add_argument("--steps", type=int, default=15_000_000); p.add_argument("--B", type=int, default=64); p.add_argument("--L", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5); p.add_argument("--epochs", type=int, default=30); p.add_argument("--mb", type=int, default=8192)
    p.add_argument("--gamma", type=float, default=0.99); p.add_argument("--lam", type=float, default=1.0); p.add_argument("--clip", type=float, default=0.3)
    p.add_argument("--ent", type=float, default=0.0); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default=None); p.add_argument("--log_every", type=int, default=1)
    a = p.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device(a.device)
    env = make_vec(a.env, a.B, seed=a.seed, device=dev)
    pol = Policy(env.obs_dim, env.act_nvec, a.memory).to(dev)
    opt = torch.optim.Adam([q for q in pol.parameters() if q.requires_grad], lr=a.lr)
    tag = a.tag or f"popgym_{a.memory}_{a.env}_s{a.seed}"; os.makedirs(f"runs/{tag}", exist_ok=True)
    log = open(f"runs/{tag}/log.jsonl", "a"); t0 = time.time()
    n_params = sum(q.numel() for q in pol.parameters() if q.requires_grad)
    print(f"[{tag}] {a.env} memory={a.memory} params={n_params/1e6:.2f}M B={a.B} L={a.L} device={dev}", flush=True)
    obs = env.reset(); state = pol.init_state(a.B, dev); done = torch.zeros(a.B, dtype=torch.bool, device=dev)
    r_prev = torch.zeros(a.B, device=dev); steps = 0; epoch = 0; mmer = -1e9; recent = []
    while steps < a.steps:
        # ---- rollout of L steps, memory state carried (detached) ------------------------------------------------
        S0 = detach_state(state); D0 = done.clone(); R0 = r_prev.clone()
        O, PN, A, LP, R, V, DN = [], [], [], [], [], [], []
        finished = []
        with torch.no_grad():
            for t in range(a.L):
                logits, v, state = pol.step(obs, state, done, r_prev)
                ds = pol.dists(logits); acts = torch.stack([d.sample() for d in ds], 1)
                lp = sum(d.log_prob(acts[:, i]) for i, d in enumerate(ds))
                O.append(obs["vec"]); PN.append(obs["pn"]); A.append(acts); LP.append(lp); V.append(v); DN.append(done.clone())
                obs, r, done, info = env.step(acts); R.append(r); r_prev = r; finished += info["finished"]
                steps += a.B
            _, v_last, _ = pol.step(obs, state, done, r_prev)
        R = torch.stack(R, 1); V = torch.stack(V, 1); DN = torch.stack(DN, 1)          # DN[t] = episode boundary BEFORE step t
        # GAE: an episode ends at step t if done_after_t; done_after_t == DN[t+1] (or current `done` for the last step)
        done_after = torch.cat([DN[:, 1:], done[:, None]], 1).float()
        adv = torch.zeros_like(R); last = torch.zeros(a.B, device=dev)
        for t in reversed(range(a.L)):
            nv = V[:, t + 1] if t + 1 < a.L else v_last
            nonterm = 1 - done_after[:, t]
            delta = R[:, t] + a.gamma * nv * nonterm - V[:, t]
            last = delta + a.gamma * a.lam * nonterm * last; adv[:, t] = last
        ret = adv + V
        # ---- PPO epochs, replaying the memory from S0 over the chunk (truncated BPTT) ----------------------------
        O = torch.stack(O, 1); PN = torch.stack(PN, 1); A = torch.stack(A, 1); LP = torch.stack(LP, 1)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        n_mb = max(1, (a.B * a.L) // a.mb); env_per_mb = max(1, a.B // n_mb)
        for ep in range(a.epochs):
            perm = torch.randperm(a.B, device=dev)
            for idx in perm.split(env_per_mb):
                st = {k: v[idx] for k, v in S0.items()} if isinstance(S0, dict) else S0[idx]
                dn = D0[idx]; rp = R0[idx]; logps, vals, ents = [], [], []
                for t in range(a.L):
                    logits, v, st = pol.step({"vec": O[idx, t], "pn": PN[idx, t]}, st, dn, rp)
                    ds = pol.dists(logits)
                    logps.append(sum(d.log_prob(A[idx, t, i]) for i, d in enumerate(ds))); vals.append(v)
                    ents.append(sum(d.entropy() for d in ds))
                    dn = torch.cat([DN[idx, t + 1:t + 2]], 1).squeeze(1) if t + 1 < a.L else done[idx]; rp = R[idx, t]
                logp = torch.stack(logps, 1); val = torch.stack(vals, 1); ent = torch.stack(ents, 1).mean()
                ratio = (logp - LP[idx]).exp(); ad = adv_n[idx]
                pg = -torch.min(ratio * ad, ratio.clamp(1 - a.clip, 1 + a.clip) * ad).mean()
                v_clip = V[idx] + (val - V[idx]).clamp(-a.clip, a.clip)
                vl = torch.max((val - ret[idx]) ** 2, (v_clip - ret[idx]) ** 2).mean()
                loss = pg + 1.0 * vl - a.ent * ent
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        epoch += 1
        if finished:
            mean_ep = float(np.mean([r for r, _ in finished])); mmer = max(mmer, mean_ep); recent = finished[-100:]
        else: mean_ep = float("nan")
        rec = dict(epoch=epoch, steps=steps, mean_ep_reward=mean_ep, mmer=mmer, n_episodes=len(finished),
                   mean_ep_len=float(np.mean([l for _, l in finished])) if finished else None, loss=loss.item(), t=time.time() - t0)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if epoch % a.log_every == 0:
            print(f"ep {epoch:4d} steps {steps/1e6:6.2f}M  mean_ep {mean_ep:7.3f}  MMER {mmer:7.3f}  eps {len(finished):4d}  {steps/(time.time()-t0):7.0f} steps/s", flush=True)
        if epoch % 20 == 0: torch.save({"pol": pol.state_dict(), "args": vars(a)}, f"runs/{tag}/ckpt.pt")
    torch.save({"pol": pol.state_dict(), "args": vars(a)}, f"runs/{tag}/ckpt.pt")
    print(f"done: MMER {mmer:.3f} in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
