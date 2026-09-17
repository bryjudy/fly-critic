# Experiment log

Started 2026-09-11. This is the working log I kept while building fly-critic - dates, decisions, failures and what they taught me. Cleaned of account details but otherwise as written.

## The idea
Google/Janelia released MaleCNS v1.0 (2026-09-03): the full male fruit-fly CNS, 166k neurons, 125M synapses.
The fly's mushroom body is the best-understood reinforcement circuit in any animal: Kenyon cells (sparse code)
-> MBONs (approach/avoid readout), with ~170 dopamine neurons (PAM = reward, PPL1 = punishment) tiling 15
compartments, each with its own plasticity rate and memory lifetime. Dopamine there does not change "mood";
it decides **what gets learned**.

We take that circuit straight out of the connectome (right hemisphere: 124 PNs, 2045 KCs, 49 MBONs, 166 DANs,
compartment-resolved synapse counts, transmitter predictions) and use it as the critic for a small transformer
agent. The dopamine it emits, per compartment, gates a three-factor Hebbian fast-weight layer in the transformer.
Slow weights learn by PPO across episodes; fast weights learn online within an episode, only when dopamine says so.

## The test
`OdorGrid`: 7x7 grid, three odour sources whose smells are patterns over the fly's real 124 PN channels.
One odour = +1, one = -1, one neutral. Mid-episode the reward and punishment odours swap. Every episode the
odours are re-drawn, so the agent must learn *within* the episode which smell to chase.

Four agents, identical transformer (0.63M params), differing only in what gates the plastic layer:
| critic   | dopamine source |
|----------|-----------------|
| mb       | real connectome mushroom body, 15 compartments |
| shuffled | same neuron counts, wiring & compartments permuted (degree-preserving) |
| scalar   | ordinary TD error from the value head, one compartment |
| none     | no plastic layer; transformer must learn in-context only |

If `mb` beats `shuffled`, the specific wiring matters. If `mb` beats `scalar`, the compartment structure matters.
Metrics: reward/step before and after reversal, steps to first post-reversal reward, punishment avoidance.

## Layout
- `connectome/fetch_mb.py` — pulls the MB circuit from neuPrint (no token needed) -> `data/mb_R.npz`
- `flycritic/mb.py` — the critic (KC sparse code, compartment-gated depression, per-compartment eta/tau, RPE -> DANs)
- `flycritic/env.py` — OdorGrid (vectorised torch)
- `flycritic/model.py` — transformer + dopamine-gated plastic head
- `flycritic/train.py` — PPO; `flycritic/evaluate.py` — comparison + reversal-aligned plot
- `runs/<tag>/` — log.jsonl, ckpt.pt

## Run
```
uv run python connectome/fetch_mb.py
uv run python -m flycritic.train --critic mb --iters 800 --tag mb_s0      # also: shuffled | scalar | none
uv run python -m flycritic.evaluate --tags mb_s0 shuffled_s0 scalar_s0 none_s0
```

## Status
- 2026-09-11: circuit extracted; unit test shows one-shot valence learning, generalisation stays odour-specific,
  fast compartments forget over ~300 steps while slow ones hold.
- 2026-09-11 RESULT (OdorChoice, 600 iters, seed 0, eval on 10x64 fresh episodes; oracle = 0.33 R/step):

  | agent | R/step pre-reversal | R/step post | punish rate post | mechanism |
  |---|---|---|---|---|
  | mb (real connectome) | 0.28 | 0.27 | 0.03 | MB learns odour values online, feeds MBON/valence to transformer |
  | shuffled connectome | 0.32 | 0.28 | 0.02 | same |
  | scalar TD critic | 0.00 | 0.00 | 0.22 | chance |
  | none (in-context only) | 0.00 | 0.00 | 0.14 | chance |
  | mb, outputs hidden (dopamine gates fast weights only) | 0.01 | 0.00 | 0.13 | chance |

  Reading: the fly circuit as an *online associative critic* is the whole effect. Real vs shuffled wiring: tie
  (expected; PN->KC wiring is near-random in the fly too). The dopamine-gated fast-weight head inside the
  transformer contributed nothing in this setup (nofeat run at chance), so "dopamine gates learning inside the
  transformer" is NOT yet demonstrated. Plot: runs/compare.png.
- OdorGrid (navigation) parked: all agents random after 480 iters even with visible plumes; reward too sparse.

- 2026-09-11 evening, v2 (`--pre kc`): fast weights now pair the fly's sparse KC code (presynaptic) with the action
  taken (postsynaptic), gated per compartment by dopamine — i.e. a second, action-selecting mushroom body inside
  the transformer. With the critic's outputs HIDDEN (dopamine is the only channel), seed 0 climbs from chance to
  ~15.7/episode by iter 270 and keeps rising; post-reversal reward now exceeds pre-reversal (it relearns after the
  swap). The outer loop tuned compartments on its own: gamma (fast) compartments gained weight on "approach" and
  their tau shortened 150->~125; alpha/beta kept as slow memory. Real vs shuffled still ~tie (15.7 vs 13.9).
  Seeds 1,2 for {mb_nofeat_kc, shuf_nofeat_kc, mb_kc} running on AWS (see aws/).
- 2026-09-11 v2 EVAL, seed 0, 6x64 fresh episodes (R/step; oracle 0.33; plot runs/compare_v2.png):

  | agent | pre-reversal | post-reversal | punish rate post |
  |---|---|---|---|
  | mb (outputs visible, h fast weights) | 0.29 | 0.27 | 0.03 |
  | mb_kc (outputs visible, KC fast weights) | 0.29 | 0.24 | 0.02 |
  | **mb_nofeat_kc (dopamine-only, KC fast weights)** | **0.18** | **0.17** | 0.09 |
  | shuf_nofeat_kc (same, shuffled wiring) | 0.15 | 0.17 | 0.11 |
  | mb_nofeat (dopamine-only, h fast weights) | 0.00 | 0.00 | 0.13 |
  | scalar / none | 0.00 | 0.00 | 0.22 / 0.14 |

  The dopamine-only KC agents show a textbook reversal curve: ~0.2 before the swap, a *perseveration dip* to
  -0.2 right after (stale fast weights keep approaching the old reward odour), then recovery over ~20 trials.
  Visible-output agents recover in ~10. This is the first configuration where "dopamine gates what the model
  itself learns" is doing the work.

- 2026-09-12 MULTI-SEED (3 seeds, final-100-iter training avg, R/episode, oracle 33.5):

  | config | R/episode | punish hits |
  |---|---|---|
  | mb_nofeat_kc (real wiring, dopamine-only) | **17.3 +/- 0.1** | 11.7 |
  | shuf_nofeat_kc (shuffled wiring, dopamine-only) | 16.1 +/- 0.8 | 12.8 |
  | mb / shuffled with outputs visible (1 seed each) | ~30.5 | 1.7 |
  | scalar / none / mb_nofeat(h) | ~0 | 14-16 |

  Real wiring beat shuffled on all 3 seeds (17.8/15.6, 18.6/18.2, 18.7/17.4) but the margin is small; call it
  "consistent, modest, not yet conclusive". The robust finding is architectural: sparse KC code + compartment-
  gated dopamine plasticity lets a frozen transformer learn odour->action online with dopamine as the ONLY
  teaching signal, at ~half the ceiling after 400 PPO iterations (still rising at stop).

- 2026-09-12 COMPARTMENT ABLATION (dopamine-only, KC fast weights, 3 seeds each, R/episode; post = after reversal):

  | mushroom body variant | R/episode | post-reversal | punish hits |
  |---|---|---|---|
  | full: 15 compartments, fly eta/tau priors | **17.3 +/- 0.1** | 8.5 | 11.7 |
  | uniform: 15 compartments, identical eta/tau | 12.8 +/- 0.3 | 4.6 | 10.9 |
  | collapsed: 1 compartment | 6.3 +/- 0.2 | **-0.9** | 23.6 |

  Clean result. Compartment structure matters (6 -> 13) AND the diversity of timescales matters on top (13 -> 17).
  The single-compartment agent cannot relearn after the reversal at all (post-reversal reward negative: it keeps
  approaching the old reward odour). Multi-timescale memory is what buys reversal learning.

- 2026-09-12 CONTEXT TASK (OdorContext, 2 seeds each): **null result for every agent**, including the no-critic
  baseline and the ctx_to_kc variant. All at 0.0 R/episode after 400 iters; several PPO runs collapsed to
  "never approach" (a guaranteed-zero local optimum). (Interim monitor readings of 10-12 for this task were
  misattributed — alphabetical ordering shifted as queued runs launched; they were the uniform-ablation runs.)
  As designed the task is uninformative: nobody learned it. Likely causes: (a) context enters KCs too weakly
  (2 of 126 PN channels, then per-KC input normalisation) so KC codes for (odour, ctx0) and (odour, ctx1)
  overlap and the associations cancel; (b) 6 conjunctions x reversal in 100 trials is a lot; (c) meta-RL of an
  XOR-like structure is slow for the transformer too. Redesign before re-running: strong context drive into
  KCs (or a dedicated context KC population, as in the fly's visual/thermal KCs), T=200, and an approach bonus
  or entropy floor to kill the never-approach attractor.

- 2026-09-12 STAGE 2 FEASIBILITY (`flycritic/body.py`): DeepMind `flybody` (MuJoCo 3.13, walk_on_ball: obs 289,
  act 59) installs and runs; harness = learned proprioception->124 PN projection -> real MB critic -> per-compartment
  dopamine -> KC->motor fast weights inside a Gaussian PPO policy. Smoke-trained 2 min end-to-end (0.22M params).
  Throughput 49 env steps/s with 4 sequential envs on the M5 -> ~1e8 steps for walking = ~600 h here. Needs
  vectorised multiprocess envs on a 32-64 core box (~1-3 days) — NOT attempted. flybody's pins (numpy<2) conflict
  with the project lock: run via `uv run --with mujoco --with dm_control --with "git+https://github.com/TuragaLab/flybody"`.
- 2026-09-12 CONTEXT v2 + PLATEAU: relaunched on GPU box 2 (runs-gpu2/): cx2_* (T=200, ent 0.03, ctx_frac 0.3,
  ctx_gain 0.1 -> conjunctive KC code) 5-way x 2 seeds, plus long_mb_nofeat_kc 1200 iters x 3 seeds.
- 2026-09-13 FINAL: context v2 -> visible critic + ctx->KC 57.1 +/-0.0 (n=2) of 66; dopamine-only + ctx->KC 11.3+/-1.0
  (rising); ctx->transformer-only and no-critic = 0 (n=2 each). Plateau: dopamine-only 21.6+/-5.3 at 1200 iters, still
  rising (one seed partially collapsed at 14). All boxes terminated. Report FINAL: report/REPORT.md (6 figs).
- Report draft: report/REPORT.md (+ report/fig*.png via `uv run python -m flycritic.figures`).

- 2026-09-13 HEAD-TO-HEAD (1200 iters, 3 seeds, OdorChoice): learned 15-ch modulator + KC fast weights + fly priors
  **28.2+/-0.4** > learned + KC 26.8+/-0.6 > fixed connectome dopamine 21.6+/-5.3 >> learned modulator on hidden
  state 0.0 = in-context RL 0.0. VERDICT: the connectome's dopamine ROUTING is not the active ingredient (a learned
  modulator beats it); the fly's ARCHITECTURE is (KC expansion necessary: learned_h=0; compartments necessary:
  ablation; priors help: +4.7 @400). Report §3.7 rewritten accordingly. All boxes terminated.

- 2026-09-14 FOLLOW-UPS (1200 iters, 3 seeds): learned modulator + fly priors on a size-matched RANDOM expansion
  28.5 +/-0.2 (vs 28.2 real wiring: input wiring irrelevant once modulator is learned); HYBRID connectome-dopamine
  + learned residual 28.9 +/-0.4 (PPO repairs the fixed circuit 21.6 -> 28.9; fixed routing = usable init, no
  advantage). Report §3.7 + fig7 updated. All boxes terminated.

- 2026-09-14 STEP 1+2 (head-to-head on external benchmarks + frozen LLM): code in flycritic/bench.py (Bandit,
  DarkRoom/KeyToDoor a la Algorithm Distillation, classical baselines: tuned SW-UCB 140, discounted TS 123, oracle
  180, random 62) and flycritic/llm/ (ToolWorld text bandit; FrozenLLM Qwen2.5 features w/ precomputed table
  flycritic/llm/precompute.py -> data/tw_feats_<model>.pt; Bridge -> 124 PN; long-context + RAG baselines).
  RULE (9/14): run experiments on AWS, never locally. Box gpu6 = step-1 suite (trimmed to bandit + darkroom;
  first attempt OOM'd at 18 GB/run -> --mb_splits 8), box cpu7 = step-2 ToolWorld @0.5B; both have aws/watchdog.sh
  (self-terminate). Phase 3 (aws/launch_gpu8.sh: 1.5B/7B baselines + 1.5B training) waits for GPU quota.
  Lesson: 6 processes loading a 0.5B model on the Mac's MPS at once hung/crashed -> precompute table on CPU.

- 2026-09-14 eve: cpu7 (c7i.4xlarge, 30 GB) OOM-killed every plastic ToolWorld run: the fast-weight replay uses
  ~14 GB RSS per run ON CPU (vs ~5 GB on GPU) -> only one fits. Terminated cpu7 (partial mb_nofeat_kc_s0 log kept in
  S3 runs-cpu7/), relaunched on r7i.4xlarge (128 GB) as cpu9 with 4 plastic runs + longctx/RAG baselines.
  Standard on-demand quota is 16 vCPU TOTAL, so only one 16-vCPU CPU box at a time.

- 2026-09-15 STEP 1 FINAL: bandit (1500 it, 3 seeds) module 136.7+/-12.9 vs in-context 99.5+/-2.3 vs SW-UCB 140;
  Dark Room (1000 it) module 38.0+/-4.9 vs in-context 31.5+/-4.3. STEP 2 FINAL (0.5B, 600 it): module 24.1 / 23.0
  vs in-context head 17.6 vs RAG 21.0 vs long-context 14.8 vs zero-shot 12.5 (oracle 105). Both boxes
  self-terminated. Phase 3 (gpu8): 1.5B training x6 + 1.5B/7B baselines launched 9/15 morning.

- 2026-09-15 PHASE 3: 1.5B features -> module 28.0+/-3.2 (30/30/23) vs in-context head 17.5+/-0.1 vs zero-shot 19.7
  (oracle 105): the module's edge grows with LLM size (0.5B: 24 vs 17.6). 1.5B/7B long-context+RAG baselines OOM'd
  twice (batched long prompts) -> features.py `chunk` param; rerun on gpu9 with chunk 2/1, GPU to itself.

- 2026-09-15 NEXT PHASE : POPGym = standard RL memory suite (24 target envs:
  {MultiarmedBandit, RepeatPrevious, RepeatFirst, CountRecall, HigherLower, Autoencode, PositionOnlyCartPole,
  NoisyPositionOnlyCartPole} x {Easy, Medium, Hard}; metric MMER; incumbent FFM). Harness in flycritic/popgym/ with
  memory modules gru | ffm | fly under one variable-length PPO trainer; budget pass 3M steps x 3 seeds on AWS, then
  15M on the best subset. Split-CIFAR-100 (10x10, frozen ResNet-18 features) in flycritic/cl/: finetune / EWC /
  replay / FlyModel / flycritic(+collapsed ablation) / offline; ACC + forgetting, 3 seeds. Both built by forks 9/15.

- 2026-09-15 STEP 2 COMPLETE: 7B long-context 20.2 / RAG 15.3 / zero-shot 19.9 vs 1.5B+module 28.0 and 0.5B+module
  24.1. Prompt memory barely beats zero-shot at any size; retrieval degrades with size. §3.9 finalized.

- 2026-09-15 POPGym FIRST ATTEMPT FAILED (my launcher): each fly run needs ~7 GB GPU (fork estimated 1 GB); the queue's
  free-memory check (2.5 GB) let 66/72 runs OOM at launch, crashed runs freed memory instantly so the queue raced
  through all 72 in 30 min, wrote the marker, watchdog shut the box down (also killing Split-CIFAR, whose CIFAR
  download I had ALSO corrupted by racing a manual download against it). 3 bandit s0 runs completed. Fixes:
  launch_popgym.sh now has hard MAXPAR (3), NEEDMB (7.5 GB), SETTLE 90 s, OOM-detect-and-retry, ENVS override;
  fly runs split across two 4-vCPU GPU boxes (gpu11 flyA = bandit/repeat/countrecall, gpu12 flyB = the rest);
  GRU baseline on cpu11 (8 concurrent) unaffected. CIFAR-100 tar.gz to be staged in S3 to avoid slow/corrupt downloads.

- 2026-09-15 eve: flyA/flyB running 3 fly runs each (launcher's pgrep double-counts uv+python -> MAXPAR must be
  2x the intended concurrency; set 6). ~10 h for the 72 fly runs. CIFAR-100 tar.gz being staged to
  s3://$FLYCRITIC_BUCKET/datasets/; Split-CIFAR then runs on flyA at nice 10 (marker line stripped so
  it can't trigger the watchdog early).

- 2026-09-15 night: Split-CIFAR feature caching on flyA hit CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED (torch wheel cuDNN
  vs DLAMI driver on g5.xlarge) -> rerun with CUDA_VISIBLE_DEVICES= (CPU ResNet-18, ~1 h at nice 10). The first
  attempt's 33 classifier runs crashed (no features) and their .out files were swept into runs-popgym-A by the
  POPGym sync loop: ignore cl_* files under runs-popgym-A; real CL results go to runs-cl/.

- 2026-09-15 ~15:30 ROOT CAUSE of the mass process deaths: work started via `aws ssm send-command` dies when the SSM
  invocation ends (timeout or completion path) even with nohup; earlier boxes survived by luck. FIX: start
  launcher/watchdog/CL as systemd transient units (`systemd-run --uid=ubuntu -p KillMode=process
  -p EnvironmentFile=/home/ubuntu/launch.env ...`) -> aws/restart_systemd.sh. All three POPGym boxes restarted
  this way at 21:26 UTC (incomplete runs purged; completed ones kept). CL restarted as unit fly-cl (CPU mode).

- 2026-09-15 16:33 local: flyA (g5.xlarge, 16 GB RAM) lost its SSM agent (ConnectionLost) with CPU dropping 100->32%;
  likely RAM exhaustion from 3 fly runs + CL feature extraction (4 DataLoader workers). Rebooted; post-reboot chain
  re-runs restart_systemd.sh (incomplete runs purged) and restarts CL as unit fly-cl2 with MemoryMax=5G, Nice=15,
  1 thread. Lesson: cap memory of side jobs on small boxes (systemd -p MemoryMax).

- 2026-09-15 POPGym interim: GRU baseline DONE (72 runs, cpu11 self-terminated). fly seed 0 on ~15 envs: >= GRU
  everywhere, wins on RepeatPreviousEasy (+0.11), BanditEasy (+0.03), NoisyCartPole (+0.16/+0.27). Budget pass
  under-trains vs paper (bandits/CountRecall/RepeatFirst); only equal-budget fly-vs-GRU is meaningful. results.py
  gained --complete_only (>=2.9M steps).

- 2026-09-16 POPGym budget pass DONE (67/72 fly + 72 gru): ~draw at equal budget (fly wins RepeatFirst E/H, Bandit M;
  GRU wins RepeatPrevious E, Bandit E, CountRecall M; 15 ties). NO takedown; full 15M protocol needed for a real
  claim. BUG: watchdog.sh pgrep pattern "flycritic.train " never matched popgym/cl process names -> boxes shut down
  right after their launchers finished, killing 5 fly runs + 24 of 33 CL runs. Fixed pattern; cleanup box gpu13
  reruns the 5 fly runs + full CL suite (features cached in S3 data/).

- 2026-09-16 Split-CIFAR seed 2: NEGATIVE for our module (class-inc ACC 7-8%, forgetting 65% = chance / last-task
  only) vs FlyModel 42.4% / 11.7% forgetting and replay-2000 53.9%. Decaying compartments are a liability when nothing
  old becomes wrong; FlyModel's synapse freezing is the right CL design. Main 15-compartment config + seeds 0/1
  rerunning on the cleanup box; expect the same.

- 2026-09-16 17:30: cleanup box gpu13 running under systemd (fly-cleanup2 + fly-watchdog3 w/ fixed pattern): 5 fly
  reruns (3 concurrent) + full Split-CIFAR suite on S3-cached features. launch_box.sh now fetches helper scripts and
  starts launcher/watchdog via systemd-run (no more nohup-under-SSM).

- 2026-09-16 Split-CIFAR FINAL (3 seeds): offline 67.4 | replay2000 54.6 | FlyModel 42.6 (forgetting 15.2) | replay200
  27.2 | finetune/EWC 9.1 | fly-critic 7.4-8.4 (forgetting ~65). Clean negative; written into REPORT §3.11 and §4.

- 2026-09-16 22:00 ALL DONE: POPGym fly 72/72 (reruns folded in), GRU 72/72, Split-CIFAR 33/33; all boxes terminated.
  Report §3.10/3.11/§4 final. Whole project (9/11-9/16): odour tasks, ablations, head-to-heads, bandit/DarkRoom,
  ToolWorld 0.5B/1.5B/7B, POPGym, Split-CIFAR. AWS spend roughly $120-150 total incl. idle/crash waste.

## Infra
- AWS account (us-west-2). GPU quotas are ZERO (G/VT and P, on-demand + spot);
  increase to 16 vCPU G requested 9/11 (PENDING). Standard on-demand cap 16 vCPU, spot cap 32.
- Training box = c7i.8xlarge SPOT (32 vCPU, ~$0.58/h), Ubuntu 24.04, tag Name=flycritic-train. SSH from this
  Mac to the box TIMES OUT even with the SG open to our IP (cause unknown; opening 0.0.0.0/0 was refused by the
  harness) -> use SSM instead: IAM an SSM+S3 instance profile (SSMManagedInstanceCore + scoped S3),
  bucket s3://$FLYCRITIC_BUCKET, code shipped as tarball via presigned URLs, commands via
  `aws ssm send-command`. Agent only registered after a REBOOT following profile attach.
- aws/status.sh (progress), aws/pull_results.sh (runs/ back via S3), aws/terminate.sh (ALWAYS run when done).
- GPU: g5.2xlarge (A10G 24 GB) with "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)" works with
  plain `uv sync` (PyPI torch has CUDA). Each run takes ~2 GB GPU RAM (fast-weight autograd over 100 steps) ->
  max ~9-10 concurrent; 16 at once OOM'd (9 died at start). remote_launch_gpu.sh + flycritic.train_waiter.sh
  (queue that launches as memory frees). ~5 s/iter with 8-10 runs sharing the GPU.
- ⚠️ 9/11: first (SPOT) box was reclaimed by AWS at iter ~250 ("no Spot capacity") and its disk vanished — all
  6 runs lost. Second box = on-demand c7i.4xlarge + `aws s3 sync runs/` loop every 5 min (remote_launch2.sh).
  Rule: never run >30 min on spot without continuous result sync (or --resume support).

## Next
- Redesign OdorContext per the notes above, then re-run the 5-way comparison.
- Write-up: the two solid results are (1) dopamine-only online learning in a frozen transformer via
  connectome-derived compartment gating (17.3/33.5, 3 seeds, controls 0) and (2) the compartment ablation
  (17.3 > 12.8 > 6.3). Real-vs-shuffled wiring is a small consistent edge, not a headline.
- Stage 2 (body): DeepMind `flybody` MuJoCo model with this critic gating a locomotion policy.
- Stage 3 (agent): the critic gating memory writes / retry decisions in a long-running LLM agent.
- 3 seeds each; report mean +/- sd.  - Ablate compartment count (collapse 15 -> 1) with real wiring.
- Harder task where the MB feature alone is insufficient (e.g. context-dependent valence, or valence that must be
  combined with a non-odour cue) so the transformer has to *use* the critic rather than copy it.
- Stage 2: MuJoCo `flybody` (DeepMind) with the same critic gating a locomotion policy.
- Stage 3: the critic gating memory writes / retry decisions in a long-running LLM agent.
