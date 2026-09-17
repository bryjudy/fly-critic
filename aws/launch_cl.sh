#!/bin/bash
# Split-CIFAR-100 (frozen ResNet-18 features) on a CPU box: cache features once, then all methods x 3 seeds in parallel.
cd ~/fly-critic; export PATH=$HOME/.local/bin:$PATH
uv sync -q --no-dev 2>&1 | tail -1; mkdir -p runs data
nohup bash -c 'while true; do aws s3 sync ~/fly-critic/runs s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/runs-cl/ --only-show-errors; sleep 300; done' > ~/sync.log 2>&1 &
mkdir -p data/cifar100; aws s3 cp s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/datasets/cifar-100-python.tar.gz data/cifar100/cifar-100-python.tar.gz --only-show-errors || echo "no staged CIFAR in S3; torchvision will download"
aws s3 cp s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/data/cifar100_r18_train.pt data/ --only-show-errors; aws s3 cp s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/data/cifar100_r18_test.pt data/ --only-show-errors; [ -f data/cifar100_r18_train.pt ] && [ -f data/cifar100_r18_test.pt ] && echo "features fetched from S3, skipping extraction" || uv run python -m flycritic.cl.data --cache 2>&1 | tail -2
aws s3 cp data/cifar100_r18_train.pt s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/data/cifar100_r18_train.pt --only-show-errors
aws s3 cp data/cifar100_r18_test.pt  s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/data/cifar100_r18_test.pt  --only-show-errors
run() { TAG=$1; shift; OMP_NUM_THREADS=2 nohup bash -c "exec -a 'flycritic.train cl' uv run python -m flycritic.cl.run --tag $TAG $*" > runs/$TAG.out 2>&1 & }
for s in 0 1 2; do
  run cl_finetune_s$s          --method finetune --seed $s
  run cl_offline_s$s           --method offline --seed $s
  run cl_ewc1_s$s              --method ewc --lam 1 --seed $s
  run cl_ewc10_s$s             --method ewc --lam 10 --seed $s
  run cl_ewc100_s$s            --method ewc --lam 100 --seed $s
  run cl_replay200_s$s         --method replay --buffer 200 --seed $s
  run cl_replay2000_s$s        --method replay --buffer 2000 --seed $s
  run cl_flymodel_s$s          --method flymodel --seed $s
  run cl_flycritic_s$s         --method flycritic --seed $s
  run cl_flycritic_collapsed_s$s --method flycritic_collapsed --seed $s
  run cl_flycritic_gain_s$s    --method flycritic_gain --seed $s
  sleep 5
done
echo "all queued runs launched" >> /home/ubuntu/launcher.log
