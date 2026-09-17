#!/bin/bash
# STEP 2 on a CPU box (c7i.4xlarge, 16 vCPU): ToolWorld with frozen Qwen2.5-0.5B features (precomputed table),
# 6 training runs x 600 iters + zero-shot/long-context/RAG baselines (0.5B on CPU). S3 sync every 5 min.
cd ~/fly-critic; export PATH=$HOME/.local/bin:$PATH
uv sync -q --no-dev 2>&1 | tail -1; mkdir -p runs data
nohup bash -c 'while true; do aws s3 sync ~/fly-critic/runs s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/runs-cpu7/ --only-show-errors --exclude "*/ckpt.pt"; sleep 300; done' > ~/sync.log 2>&1 &
uv run python -m flycritic.llm.precompute --model 0.5b --device cpu 2>&1 | tail -1
launch() { OMP_NUM_THREADS=2 nohup uv run python -m flycritic.train --env toolworld --T 120 --iters 600 "$@" > runs/$TAG.out 2>&1 & }
TAG=tw_learned_kc_prior_s0; launch --critic learned --pre kc --prior --seed 0 --tag $TAG
TAG=tw_learned_kc_prior_s1; launch --critic learned --pre kc --prior --seed 1 --tag $TAG
TAG=tw_mb_nofeat_kc_s0;     launch --critic mb --no_feat --pre kc --seed 0 --tag $TAG
TAG=tw_none_s0;             launch --critic none --seed 0 --tag $TAG
TAG=tw_none_s1;             launch --critic none --seed 1 --tag $TAG
TAG=tw_none_hi_s0;          launch --critic none --lr 1e-3 --ent 0.03 --seed 0 --tag $TAG
OMP_NUM_THREADS=4 nohup uv run python -m flycritic.llm.eval_baselines --model 0.5b --B 16 --T 120 --episodes 2 --device cpu > runs/tw_baselines_0.5b.out 2>&1 &
echo "all queued runs launched"
