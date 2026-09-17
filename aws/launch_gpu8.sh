#!/bin/bash
# STEP 2b on a GPU box (after step 1 frees the quota): frozen-LLM baselines at 1.5B and 7B (zero-shot / long-context /
# RAG, CUDA), plus ToolWorld training on 1.5B features (precomputed table): learned modulator x3 seeds, in-context x3.
cd ~/fly-critic; export PATH=$HOME/.local/bin:$PATH
uv sync -q --no-dev 2>&1 | tail -1; mkdir -p runs data
nohup bash -c 'while true; do aws s3 sync ~/fly-critic/runs s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/runs-gpu8/ --only-show-errors --exclude "*/ckpt.pt"; sleep 300; done' > ~/sync.log 2>&1 &
uv run python -m flycritic.llm.precompute --model 1.5b --device cuda 2>&1 | tail -1
launch() { OMP_NUM_THREADS=1 nohup uv run python -m flycritic.train --device cuda --env toolworld --llm 1.5b --T 120 --iters 600 --mb_splits 4 "$@" > runs/$TAG.out 2>&1 & }
for s in 0 1 2; do
  TAG=tw15_learned_kc_prior_s$s; launch --critic learned --pre kc --prior --seed $s --tag $TAG
  TAG=tw15_none_s$s;             launch --critic none --seed $s --tag $TAG
done
# baselines need real LLM calls (long prompts): run sequentially so they don't fight the trainers for GPU memory
uv run python -m flycritic.llm.eval_baselines --model 1.5b --B 16 --T 120 --episodes 2 --device cuda > runs/tw_baselines_1.5b.out 2>&1
uv run python -m flycritic.llm.eval_baselines --model 7b   --B 16 --T 120 --episodes 2 --device cuda > runs/tw_baselines_7b.out 2>&1
echo "all queued runs launched"
