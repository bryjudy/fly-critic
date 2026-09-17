"""Aggregate POPGym MMER per (env, memory) across seeds and compare with published numbers.
usage: uv run python -m flycritic.popgym.results [--runs runs]

Reference numbers: POPGym paper (Morad et al., ICLR 2023, arXiv 2303.01859), Appendix B Table 3 — MMER mean (sd)
over 3 trials, 15M steps, RLlib PPO. GRU column = their GRU; BEST column = best of their 13 models (name).
Naming: the paper's StatelessCartPole / NoisyStatelessCartPole are the current popgym PositionOnlyCartPole /
NoisyPositionOnlyCartPole (popgym 1.0 rename; ASSUMED — verify against popgym changelog before publishing).
FFM per-env numbers are in the FFM paper (arXiv 2310.04128) appendix and are not transcribed here; FFM reports
+7% mean reward over GRU across POPGym, noticeably better on 17 tasks and worse on 4 (their Fig. 4).
"""
import argparse, json, glob, re, collections, numpy as np

PUBLISHED = {  # env: (GRU mu, GRU sd, best model, best mu)
    "AutoencodeEasy": (-0.283, 0.029, "GRU", -0.283), "AutoencodeMedium": (-0.425, 0.018, "IndRNN", -0.420), "AutoencodeHard": (-0.456, 0.009, "IndRNN", -0.448),
    "CountRecallEasy": (0.177, 0.005, "LSTM", 0.509), "CountRecallMedium": (-0.528, 0.001, "PosMLP", -0.519), "CountRecallHard": (-0.475, 0.006, "PosMLP/TCN", -0.470),
    "HigherLowerEasy": (0.529, 0.002, "GRU", 0.529), "HigherLowerMedium": (0.511, 0.002, "IndRNN", 0.513), "HigherLowerHard": (0.506, 0.001, "IndRNN", 0.509),
    "MultiarmedBanditEasy": (0.619, 0.007, "Elman", 0.631), "MultiarmedBanditMedium": (0.538, 0.036, "TCN", 0.598), "MultiarmedBanditHard": (0.516, 0.083, "TCN", 0.574),
    "NoisyPositionOnlyCartPoleEasy": (0.995, 0.000, "GRU", 0.995), "NoisyPositionOnlyCartPoleMedium": (0.642, 0.012, "IndRNN", 0.659), "NoisyPositionOnlyCartPoleHard": (0.390, 0.007, "IndRNN", 0.404),
    "RepeatFirstEasy": (1.000, 0.000, "GRU/others", 1.000), "RepeatFirstMedium": (1.000, 0.000, "GRU", 1.000), "RepeatFirstHard": (0.940, 0.012, "IndRNN", 0.969),
    "RepeatPreviousEasy": (1.000, 0.000, "GRU/others", 1.000), "RepeatPreviousMedium": (-0.315, 0.017, "LMU", 0.789), "RepeatPreviousHard": (-0.428, 0.002, "LMU", 0.191),
    "PositionOnlyCartPoleEasy": (1.000, 0.000, "GRU/others", 1.000), "PositionOnlyCartPoleMedium": (1.000, 0.000, "GRU/others", 1.000), "PositionOnlyCartPoleHard": (1.000, 0.000, "GRU/others", 1.000),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--complete_only", action="store_true"); p.add_argument("--complete_steps", type=int, default=2_900_000); p.add_argument("--runs", default="runs"); a = p.parse_args()
    g = collections.defaultdict(list)
    for f in glob.glob(f"{a.runs}/popgym_*/log.jsonl"):
        if a.complete_only:
            try:
                last=json.loads(open(f).read().strip().splitlines()[-1])
                if last.get("steps",0) < a.complete_steps: continue
            except Exception: continue
        tag = f.split("/")[-2]; m = re.match(r"popgym_(\w+?)_(\w+?)_s(\d+)$", tag)
        if not m: continue
        rows = [json.loads(l) for l in open(f)]
        if not rows: continue
        g[(m.group(2), m.group(1))].append((rows[-1]["mmer"], rows[-1]["steps"]))
    mems = sorted({k[1] for k in g}); envs = [e for e in PUBLISHED if any((e, mm) in g for mm in mems)]
    print(f"{'env':32s} " + " ".join(f"{mm+' (ours)':>16s}" for mm in mems) + f" {'GRU (paper)':>12s} {'best (paper)':>20s}")
    for e in envs:
        cells = []
        for mm in mems:
            v = g.get((e, mm), [])
            cells.append(f"{np.mean([x[0] for x in v]):6.3f}+/-{np.std([x[0] for x in v]):5.3f}n{len(v)}" if v else f"{'-':>16s}")
        gm, gs, bn, bm = PUBLISHED[e]
        print(f"{e:32s} " + " ".join(cells) + f" {('%6.3f' % gm) if gm is not None else '   n/a':>12s} {f'{bm:6.3f} ({bn})':>20s}")
    steps = {k: min(x[1] for x in v) for k, v in g.items()}
    print(f"\n(min steps per cell: {min(steps.values())/1e6:.1f}M .. {max(steps.values())/1e6:.1f}M; paper = 15M, 3 trials)")

if __name__ == "__main__":
    main()
