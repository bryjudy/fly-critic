#!/bin/bash
# POPGym budget pass: {fly, gru} x 24 envs x 3 seeds at 3M env steps each (paper protocol is 15M) = 144 runs.
# Throughput measured locally (M5 CPU, B=16, 2 PPO epochs): fly ~1.5k steps/s, gru ~2.7k, ffm ~4.3k. Under the full
# protocol (B=64, L=128, 30 epochs -> ~3.8k replayed policy steps per 8.2k env steps) expect ~300-600 env steps/s per
# run on the A10G/L4 (python-overhead bound), i.e. ~1.5-3 h per 3M-step run. With 8 concurrent runs (8 vCPU box):
# 144 runs x ~2 h / 8 ~= 36 h wall-clock. Use a CPU-heavy box (c7i.16xlarge is over quota; g5.2xlarge has 8 vCPU) or
# split across boxes. Runs are launched as GPU memory frees (each ~1 GB) with a hard cap of MAXPAR concurrent.
# Results sync to s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/$PREFIX/ every 5 min; appends "all queued runs launched"
# to /home/ubuntu/launcher.log when the queue is empty (watchdog.sh then shuts the box down after the last run).
cd ~/fly-critic; export PATH=$HOME/.local/bin:$PATH
uv sync -q --no-dev 2>&1 | tail -1; mkdir -p runs
MAXPAR=${MAXPAR:-3}; NEEDMB=${NEEDMB:-7500}; SETTLE=${SETTLE:-90}; STEPS=${STEPS:-3000000}; MEMORIES=${MEMORIES:-"fly gru"}; EPOCHS=${EPOCHS:-8}; DEVICE=${DEVICE:-cuda}; PREFIX=${PREFIX:-runs-popgym}
nohup bash -c 'while true; do aws s3 sync ~/fly-critic/runs s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/$PREFIX/ --only-show-errors --exclude "*/ckpt.pt"; sleep 300; done' > ~/sync.log 2>&1 &
ENVS=${ENVS:-"MultiarmedBanditEasy MultiarmedBanditMedium MultiarmedBanditHard RepeatPreviousEasy RepeatPreviousMedium RepeatPreviousHard RepeatFirstEasy RepeatFirstMedium RepeatFirstHard CountRecallEasy CountRecallMedium CountRecallHard HigherLowerEasy HigherLowerMedium HigherLowerHard AutoencodeEasy AutoencodeMedium AutoencodeHard PositionOnlyCartPoleEasy PositionOnlyCartPoleMedium PositionOnlyCartPoleHard NoisyPositionOnlyCartPoleEasy NoisyPositionOnlyCartPoleMedium NoisyPositionOnlyCartPoleHard"}
launch() { OMP_NUM_THREADS=${OMP:-1} nohup uv run python -m flycritic.popgym.train --device $DEVICE --epochs $EPOCHS --steps $STEPS --env $1 --memory $2 --seed $3 --tag popgym_$2_$1_s$3 > runs/popgym_$2_$1_s$3.out 2>&1 & }
for s in 0 1 2; do for e in $ENVS; do for m in $MEMORIES; do
  [ -f runs/popgym_${m}_${e}_s$s.out ] && continue
  while true; do
    free=$( [ "$DEVICE" = cuda ] && nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 || echo 99999 )
    n=$(pgrep -fc "python -m flycritic.popgym.train" || true)
    [ "$free" -gt "$NEEDMB" ] && [ "$n" -lt "$MAXPAR" ] && break; sleep 30
  done
  launch $e $m $s; echo "$(date +%T) launched popgym_${m}_${e}_s$s (free ${free}MiB, running $n)" >> /home/ubuntu/launcher.log; sleep $SETTLE
  if grep -q "OutOfMemoryError" runs/popgym_${m}_${e}_s$s.out 2>/dev/null; then
    echo "$(date +%T) OOM on launch of popgym_${m}_${e}_s$s -> removing its log and pausing 10 min before retrying" >> /home/ubuntu/launcher.log
    rm -rf runs/popgym_${m}_${e}_s$s runs/popgym_${m}_${e}_s$s.out; sleep 600
    launch $e $m $s; echo "$(date +%T) relaunched popgym_${m}_${e}_s$s" >> /home/ubuntu/launcher.log; sleep $SETTLE
  fi
done; done; done
echo "all queued runs launched" >> /home/ubuntu/launcher.log
