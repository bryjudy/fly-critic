#!/bin/bash
# Rerun the 5 fly POPGym runs killed by the watchdog bug (seed 2), 3 at a time on one GPU box.
cd ~/fly-critic; export PATH=$HOME/.local/bin:$PATH
uv sync -q --no-dev 2>&1 | tail -1; mkdir -p runs
nohup bash -c 'while true; do aws s3 sync ~/fly-critic/runs s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/runs-popgym-fix/ --only-show-errors --exclude "*/ckpt.pt"; sleep 300; done' > ~/sync.log 2>&1 &
launch() { OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup uv run python -m flycritic.popgym.train --device cuda --epochs 8 --steps 3000000 --env $1 --memory fly --seed 2 --tag popgym_fly_$1_s2 > runs/popgym_fly_$1_s2.out 2>&1 & }
for e in NoisyPositionOnlyCartPoleEasy NoisyPositionOnlyCartPoleMedium NoisyPositionOnlyCartPoleHard CountRecallMedium CountRecallHard; do
  while [ "$(pgrep -fc 'python -m flycritic.popgym.train')" -ge 6 ]; do sleep 30; done
  launch $e; echo "$(date +%T) launched popgym_fly_${e}_s2" >> /home/ubuntu/launcher.log; sleep 90
done
echo "all queued runs launched" >> /home/ubuntu/launcher.log
