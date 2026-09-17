#!/bin/bash
# STEP 2 on r7i.4xlarge (16 vCPU, 128 GB): plastic ToolWorld runs (~14 GB RSS each on CPU) + frozen-LLM baselines.
cd ~/fly-critic; export PATH=$HOME/.local/bin:$PATH
uv sync -q --no-dev 2>&1 | tail -1; mkdir -p runs data
nohup bash -c 'while true; do aws s3 sync ~/fly-critic/runs s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/runs-cpu9/ --only-show-errors --exclude "*/ckpt.pt"; sleep 300; done' > ~/sync.log 2>&1 &
uv run python -m flycritic.llm.precompute --model 0.5b --device cpu 2>&1 | tail -1
launch() { OMP_NUM_THREADS=4 nohup uv run python -m flycritic.train --env toolworld --T 120 --iters 600 --mb_splits 4 "$@" > runs/$TAG.out 2>&1 & }
TAG=tw_learned_kc_prior_s0; launch --critic learned --pre kc --prior --seed 0 --tag $TAG
TAG=tw_learned_kc_prior_s1; launch --critic learned --pre kc --prior --seed 1 --tag $TAG
TAG=tw_mb_nofeat_kc_s0;     launch --critic mb --no_feat --pre kc --seed 0 --tag $TAG
TAG=tw_mb_nofeat_kc_s1;     launch --critic mb --no_feat --pre kc --seed 1 --tag $TAG
OMP_NUM_THREADS=4 nohup uv run python -m flycritic.llm.eval_baselines --model 0.5b --B 16 --T 120 --episodes 2 --device cpu --kinds longctx rag > runs/tw_baselines_0.5b_ctx.out 2>&1 &
echo "all queued runs launched"
