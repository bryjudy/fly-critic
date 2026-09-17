"""Stage 2 harness: the connectome critic gating fast weights in a locomotion policy for DeepMind's `flybody`
MuJoCo fruit fly (walk_on_ball task). Design:
  proprioception (joints, gyro, accel, forces) --learned projection--> 124 "PN" channels --> mushroom body
  (KC sparse code, MBON valence, per-compartment dopamine from RPE on the task reward)
  policy: MLP on proprioception + plastic head  a = W h + sum_c alpha_c H^c KC, H^c updated by dopamine x (a (x) KC)
  outer loop: PPO on slow weights (continuous Gaussian policy).
This file is a runnable feasibility harness (smoke-trains for a few minutes); reaching stable walking needs
~1e8 environment steps of compute (see IDEA.md).
usage (flybody pins conflict with the project lock, so use an overlay env):
  uv run --with mujoco --with dm_control --with "git+https://github.com/TuragaLab/flybody" python -m flycritic.body --minutes 3
"""
import argparse, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .mb import MushroomBody
from .model import PlasticHead

def make_env():
    from flybody.fly_envs import walk_on_ball
    return walk_on_ball()

def flat_obs(ts):
    return np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in ts.observation.values()])

class BodyAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, mb, d=256):
        super().__init__()
        self.mb = mb
        self.to_pn = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(), nn.Linear(128, mb.P), nn.Softplus())  # Box 2
        self.trunk = nn.Sequential(nn.Linear(obs_dim + mb.M + 1, d), nn.Tanh(), nn.Linear(d, d), nn.Tanh())
        eta0 = mb.log_eta.exp().detach().tolist(); tau0 = mb.log_tau.exp().detach().tolist()
        self.head = PlasticHead(d, act_dim + 1, mb.C, eta0, tau0, d_pre=mb.K, norm=float(mb.n_active))  # mean actions + value
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0)); self.act_dim = act_dim
    def step(self, st, obs):
        pn = self.to_pn(obs) * 3.0
        st["mb"], mbon, v = self.mb.sense(st["mb"], pn)
        h = self.trunk(torch.cat([obs, mbon, v[:, None]], 1))
        y = self.head(h, st["H"], st["mb"]["kc"])
        return y[:, :-1], y[:, -1], st["mb"]["kc"]

def main():
    p = argparse.ArgumentParser(); p.add_argument("--minutes", type=float, default=3); p.add_argument("--n_env", type=int, default=4)
    p.add_argument("--horizon", type=int, default=200); a = p.parse_args()
    torch.manual_seed(0); envs = [make_env() for _ in range(a.n_env)]
    ts = [e.reset() for e in envs]; obs_dim = flat_obs(ts[0]).shape[0]; act_dim = envs[0].action_spec().shape[0]
    mb = MushroomBody(); agent = BodyAgent(obs_dim, act_dim, mb); opt = torch.optim.Adam(agent.parameters(), 3e-4)
    print(f"flybody walk_on_ball | obs {obs_dim} act {act_dim} | agent params {sum(q.numel() for q in agent.parameters())/1e6:.2f}M | MB {mb.K} KCs, {mb.C} compartments")
    t0 = time.time(); it = 0; steps = 0
    while time.time() - t0 < a.minutes * 60:
        st = {"mb": mb.reset(a.n_env), "H": agent.head.init_state(a.n_env, "cpu")}
        O, A, R, V, LP, DOP, KC = [], [], [], [], [], [], []
        with torch.no_grad():
            for t in range(a.horizon):
                obs = torch.tensor(np.stack([flat_obs(x) for x in ts]))
                mu, val, kc = agent.step(st, obs)
                dist = torch.distributions.Normal(mu, agent.log_std.exp()); act = dist.sample().clamp(-1, 1)
                ts = [e.step(act[i].numpy()) for i, e in enumerate(envs)]
                r = torch.tensor([float(x.reward or 0.0) for x in ts])
                st["mb"], dop, _ = mb.learn(st["mb"], r)
                post = torch.cat([act, torch.ones(a.n_env, 1)], 1)
                st["H"] = agent.head.update(st["H"], kc, post, dop)
                O.append(obs); A.append(act); R.append(r); V.append(val); LP.append(dist.log_prob(act).sum(1)); DOP.append(dop); KC.append(kc)
                steps += a.n_env
                for i, x in enumerate(ts):
                    if x.last(): ts[i] = envs[i].reset()
        # PPO update (one epoch, replaying the plastic recurrence with stored KC codes and dopamine)
        R = torch.stack(R, 1); V = torch.stack(V, 1); adv = torch.zeros_like(R); last = torch.zeros(a.n_env)
        for t in reversed(range(a.horizon)):
            nv = V[:, t + 1] if t + 1 < a.horizon else torch.zeros(a.n_env)
            delta = R[:, t] + 0.97 * nv - V[:, t]; last = delta + 0.97 * 0.95 * last; adv[:, t] = last
        ret = adv + V; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        st = {"mb": mb.reset(a.n_env), "H": agent.head.init_state(a.n_env, "cpu")}; loss_pg = loss_v = 0
        for t in range(a.horizon):
            obs = O[t]; pn = agent.to_pn(obs) * 3.0
            st["mb"], mbon, v = mb.sense(st["mb"], pn)
            h = agent.trunk(torch.cat([obs, mbon, v[:, None]], 1)); y = agent.head(h, st["H"], KC[t])
            mu, val = y[:, :-1], y[:, -1]; dist = torch.distributions.Normal(mu, agent.log_std.exp())
            ratio = (dist.log_prob(A[t]).sum(1) - LP[t]).exp()
            loss_pg = loss_pg - torch.min(ratio * adv[:, t], ratio.clamp(0.8, 1.2) * adv[:, t]).mean()
            loss_v = loss_v + F.mse_loss(val, ret[:, t])
            post = torch.cat([A[t], torch.ones(a.n_env, 1)], 1); st["H"] = agent.head.update(st["H"], KC[t], post, DOP[t])
        loss = (loss_pg + 0.5 * loss_v) / a.horizon
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0); opt.step(); it += 1
        print(f"it {it:3d} | reward/step {R.mean():.4f} | dopamine |d| {torch.stack(DOP).abs().mean():.3f} | {steps/(time.time()-t0):.0f} env steps/s | {time.time()-t0:.0f}s", flush=True)
    print(f"done: {steps} env steps in {time.time()-t0:.0f}s. Stable walking needs ~1e8 steps -> {1e8/(steps/(time.time()-t0))/3600:.0f} h at this rate on one laptop core set; parallelise across cores/machines.")

if __name__ == "__main__":
    main()
