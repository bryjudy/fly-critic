#!/bin/bash
# Follow-ups: (a) learned modulator + fly priors on a size-matched RANDOM expansion; (b) hybrid = connectome RPE
# routing as initialisation + learned residual. 1200 iters, 3 seeds each. Results synced to S3 every 5 min.
cd ~/fly-critic; export PATH=$HOME/.local/bin:$PATH
uv sync -q --no-dev 2>&1 | tail -1; mkdir -p runs
nohup bash -c 'while true; do aws s3 sync ~/fly-critic/runs s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/runs-gpu5/ --only-show-errors --exclude "*/ckpt.pt"; sleep 300; done' > ~/sync.log 2>&1 &
launch() { OMP_NUM_THREADS=1 nohup uv run python -m flycritic.train --device cuda --env choice --iters 1200 "$@" > runs/$TAG.out 2>&1 & }
Q=()
for s in 0 1 2; do
  Q+=("hh_learned_randkc_prior_s$s|--critic learned --pre kc --prior --randkc --seed $s")
  Q+=("hh_hybrid_kc_s$s|--critic hybrid --pre kc --no_feat --seed $s")
done
for item in "${Q[@]}"; do
  TAG=${item%%|*}; ARGS=${item#*|}
  while true; do free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "$free" -gt 4000 ] && break; sleep 30; done
  launch $ARGS --tag $TAG; echo "$(date +%T) launched $TAG (free ${free}MiB)"; sleep 120
done
echo "all queued runs launched"
