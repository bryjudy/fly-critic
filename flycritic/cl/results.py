"""Aggregate Split-CIFAR-100 results across methods and seeds. usage: uv run python -m flycritic.cl.results"""
import json, glob, re, collections, numpy as np
g = collections.defaultdict(list)
for f in sorted(glob.glob("runs/cl_*/cl.json")):
    r = json.load(open(f)); tag = re.sub(r"_s\d+$", "", f.split("/")[1]); g[tag].append(r)
print(f"{'method':28s} {'seeds':>5s} {'class-inc ACC':>14s} {'forgetting':>11s} {'BWT':>7s} {'task-inc ACC':>13s}")
for tag, rs in sorted(g.items(), key=lambda kv: -np.mean([r["ci"]["ACC"] for r in kv[1]])):
    acc = np.array([r["ci"]["ACC"] for r in rs]); fg = np.mean([r["ci"]["forgetting"] for r in rs]); bwt = np.mean([r["ci"]["BWT"] for r in rs])
    ti = np.mean([r["ti"]["ACC"] for r in rs])
    print(f"{tag:28s} {len(rs):5d} {100*acc.mean():8.1f} +/-{100*acc.std():4.1f} {100*fg:11.1f} {100*bwt:7.1f} {100*ti:13.1f}")
print("""
Reference points (class-incremental Split-CIFAR-100, 10 tasks; APPROXIMATE, from the cited papers' tables; protocols
differ from ours in a key way: they train the ResNet-18 backbone from scratch, we use a FROZEN ImageNet ResNet-18 and
compare classifiers/memories on identical features, which makes all our numbers higher and the comparison fairer to
the Hebbian methods). Verify against the papers before quoting.
  SGD fine-tune ~8-9% ACC; online EWC ~8-9%; ER buffer 200 ~20-25%, buffer 5120 ~45-50%; DER++ buffer 5120 ~60%.
    (Buzzega et al. 2020, 'Dark Experience for General Continual Learning', NeurIPS; Boschini et al. 2022, Mammoth.)
  FlyModel-style sparse-coding + Hebbian on frozen/pretrained features: reported strong low-forgetting on Split-MNIST /
    Split-CIFAR-100 (Shen, Dasgupta & Navlakha 2021, PNAS 118(38)); exact CIFAR-100 numbers depend on their feature
    pipeline -- treat as qualitative.
Upper bound in OUR setting = 'offline' (joint linear head on frozen features), typically ~65-70% on CIFAR-100.""")
