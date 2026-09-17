#!/bin/bash
# STEP 1: external benchmarks (bandit / darkroom / keydoor), T=200, 1500 iters.
# Per env: in-context RL x3 variants (default | lr1e-3+ent0.03 | 6-layer) x 3 seeds, learned-modulator+KC+priors x3 seeds,
# fixed connectome dopamine x2 seeds  -> 14 runs/env, 42 total. Queue launches as GPU memory frees; S3 sync every 5 min.
cd ~/fly-critic; export PATH=$HOME/.local/bin:$PATH
uv sync -q --no-dev 2>&1 | tail -1; mkdir -p runs
nohup bash -c 'while true; do aws s3 sync ~/fly-critic/runs s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/runs-gpu6/ --only-show-errors --exclude "*/ckpt.pt"; sleep 300; done' > ~/sync.log 2>&1 &
launch() { OMP_NUM_THREADS=1 nohup uv run python -m flycritic.train --device cuda --T 200 --iters 1500 --mb_splits 8 "$@" > runs/$TAG.out 2>&1 & }
Q=()
for env in bandit darkroom keydoor; do
  for s in 0 1 2; do
    Q+=("b1_${env}_learned_kc_prior_s$s|--env $env --critic learned --pre kc --prior --seed $s")
    Q+=("b1_${env}_none_s$s|--env $env --critic none --seed $s")
    Q+=("b1_${env}_none_hi_s$s|--env $env --critic none --lr 1e-3 --ent 0.03 --seed $s")
    Q+=("b1_${env}_none_big_s$s|--env $env --critic none --layers 6 --seed $s")
  done
  for s in 0 1; do Q+=("b1_${env}_mb_nofeat_kc_s$s|--env $env --critic mb --no_feat --pre kc --seed $s"); done
done
for item in "${Q[@]}"; do
  TAG=${item%%|*}; ARGS=${item#*|}
  while true; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -gt 3500 ] && break; sleep 30; done
  launch $ARGS --tag $TAG; echo "$(date +%T) launched $TAG (free ${free}MiB)"; sleep 75
done
echo "all queued runs launched"
