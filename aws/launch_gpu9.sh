#!/bin/bash
# Baselines only (GPU to itself): 1.5B and 7B zero-shot / long-context / RAG on ToolWorld, one prompt per forward.
cd ~/fly-critic; export PATH=$HOME/.local/bin:$PATH
uv sync -q --no-dev 2>&1 | tail -1; mkdir -p runs
nohup bash -c 'while true; do aws s3 sync ~/fly-critic/runs s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/runs-gpu9/ --only-show-errors; sleep 300; done' > ~/sync.log 2>&1 &
( exec -a "flycritic.train baselines" bash -c '
  cd ~/fly-critic; export PATH=$HOME/.local/bin:$PATH
  uv run python -m flycritic.llm.eval_baselines --model 1.5b --B 16 --T 120 --episodes 2 --device cuda --chunk 2 > runs/tw_baselines_1.5b.out 2>&1
  uv run python -m flycritic.llm.eval_baselines --model 7b   --B 16 --T 120 --episodes 2 --device cuda --chunk 1 > runs/tw_baselines_7b.out 2>&1
  aws s3 sync ~/fly-critic/runs s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/runs-gpu9/ --only-show-errors' ) &
echo "all queued runs launched"
