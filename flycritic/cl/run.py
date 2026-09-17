"""Run one method on Split-CIFAR-100 (or synthetic) and log the accuracy matrix.
usage: uv run python -m flycritic.cl.run --method flycritic --seed 0 --tag cl_flycritic_s0
       uv run python -m flycritic.cl.run --method ewc --lam 100 --data synthetic --tag _smoke"""
import argparse, json, os, time, torch
from .data import load_features, synthetic_features, make_tasks, N_CLASSES
from .methods import METHODS

def evaluate(m, xte, yte, tasks, seen):
    """Returns (class-incremental acc per task, task-incremental acc per task) for tasks 0..seen."""
    with torch.no_grad(): logits = m.predict(xte)
    ci, ti = [], []
    for j in range(seen + 1):
        cls = tasks[j]; mask = torch.isin(yte, cls)
        if mask.sum() == 0: ci.append(float("nan")); ti.append(float("nan")); continue
        lg, yy = logits[mask], yte[mask]
        ci.append((lg.argmax(1) == yy).float().mean().item())
        sub = lg[:, cls]; ti.append((cls[sub.argmax(1)] == yy).float().mean().item())
    return ci, ti

def metrics(A):
    """A[t][j] = acc on task j after training task t (j<=t). ACC, forgetting (Chaudhry et al. 2018), BWT."""
    T = len(A); last = A[-1]
    acc = sum(last[:T]) / T
    forg = sum(max(A[t][j] for t in range(j, T - 1)) - last[j] for j in range(T - 1)) / max(T - 1, 1)
    bwt = sum(last[j] - A[j][j] for j in range(T - 1)) / max(T - 1, 1)
    return dict(ACC=acc, forgetting=forg, BWT=bwt)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True, choices=list(METHODS)); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", required=True); p.add_argument("--data", default="real", choices=["real", "synthetic"])
    p.add_argument("--n_tasks", type=int, default=10); p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lam", type=float, default=10.0); p.add_argument("--buffer", type=int, default=200)
    p.add_argument("--tau_scale", type=float, default=None)
    a = p.parse_args(); torch.manual_seed(a.seed)
    if a.data == "real":
        xtr, ytr = load_features("train"); xte, yte = load_features("test"); n_classes = N_CLASSES
    else:
        xtr, ytr, xte, yte = synthetic_features(n_tasks=a.n_tasks, seed=a.seed); n_classes = int(ytr.max()) + 1
    tasks = make_tasks(n_classes, a.n_tasks, a.seed)
    per_task = len(xtr) // a.n_tasks
    m = METHODS[a.method](xtr.shape[1], n_classes, epochs=a.epochs, seed=a.seed, lam=a.lam, buffer=a.buffer,
                          examples_per_task=per_task, tau_scale=a.tau_scale)
    A_ci, A_ti, t0 = [], [], time.time()
    for t, cls in enumerate(tasks):
        mask = torch.isin(ytr, cls); idx = mask.nonzero().squeeze(1); idx = idx[torch.randperm(len(idx), generator=torch.Generator().manual_seed(a.seed + t))]
        m.train_task(xtr[idx], ytr[idx], t, cls)
        ci, ti = evaluate(m, xte, yte, tasks, t); A_ci.append(ci); A_ti.append(ti)
        print(f"[{a.tag}] task {t}: class-inc acc on seen tasks {[round(v, 3) for v in ci]}  ({time.time()-t0:.0f}s)", flush=True)
    res = dict(args=vars(a), acc_matrix_ci=A_ci, acc_matrix_ti=A_ti, ci=metrics(A_ci), ti=metrics(A_ti), seconds=time.time() - t0)
    os.makedirs(f"runs/{a.tag}", exist_ok=True); json.dump(res, open(f"runs/{a.tag}/cl.json", "w"), indent=1)
    print(f"[{a.tag}] FINAL class-inc ACC {res['ci']['ACC']:.3f} forgetting {res['ci']['forgetting']:.3f} BWT {res['ci']['BWT']:.3f} | task-inc ACC {res['ti']['ACC']:.3f}")

if __name__ == "__main__":
    main()
