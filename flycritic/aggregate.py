"""Aggregate runs across seeds: mean +/- sd of final-100-iteration training reward and post-reversal reward.
usage: uv run python -m flycritic.aggregate"""
import json, glob, re, collections, numpy as np
groups = collections.defaultdict(list)
for f in sorted(glob.glob("runs/*/log.jsonl")):
    tag = f.split("/")[1]; cfg = re.sub(r"_s\d+$", "", tag)
    rows = [json.loads(l) for l in open(f)]
    if len(rows) < 50: continue
    last = rows[-100:]
    groups[cfg].append(dict(seed=tag, iters=len(rows), R=np.mean([r["reward"] for r in last]),
                            pre=np.mean([r["reward_pre"] for r in last]), post=np.mean([r["reward_post"] for r in last]),
                            punish=np.mean([r["punish_hits"] for r in last])))
print(f"{'config':24s} {'seeds':>5s} {'iters':>6s} {'R/episode':>16s} {'pre':>7s} {'post':>7s} {'punish hits':>12s}")
for cfg, rs in sorted(groups.items(), key=lambda kv: -np.mean([r['R'] for r in kv[1]])):
    R = np.array([r["R"] for r in rs]); sd = f"+/-{R.std():.1f}" if len(R) > 1 else ""
    print(f"{cfg:24s} {len(rs):5d} {min(r['iters'] for r in rs):6d} {R.mean():9.1f} {sd:>7s} "
          f"{np.mean([r['pre'] for r in rs]):7.1f} {np.mean([r['post'] for r in rs]):7.1f} {np.mean([r['punish'] for r in rs]):12.1f}")
print("(oracle = 33.5/episode; final-100-iteration training averages; pre/post = reward before/after the reversal)")
