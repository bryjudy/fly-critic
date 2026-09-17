"""Local smoke test on random 512-d features: 3 tasks x 10 classes x 50 examples, every method. (<60 s)"""
import subprocess, sys, time
t0 = time.time()
for m, extra in [("finetune", []), ("offline", []), ("ewc", ["--lam", "10"]), ("replay", ["--buffer", "200"]),
                 ("flymodel", []), ("flycritic", []), ("flycritic_collapsed", []), ("flycritic_gain", [])]:
    out = subprocess.run([sys.executable, "-m", "flycritic.cl.run", "--method", m, "--data", "synthetic", "--n_tasks", "3",
                          "--epochs", "2", "--tag", f"_smoke_{m}"] + extra, capture_output=True, text=True)
    last = [l for l in out.stdout.splitlines() if "FINAL" in l]; print(last[-1] if last else out.stderr[-400:])
print(f"smoke total {time.time()-t0:.0f}s")
