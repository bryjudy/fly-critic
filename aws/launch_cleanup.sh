#!/bin/bash
# Cleanup box: (a) the 5 fly POPGym runs lost to the watchdog bug, (b) the full Split-CIFAR suite on cached features.
cd /home/ubuntu; export PATH=$HOME/.local/bin:$PATH
export PREFIX=runs-popgym-fix
bash /home/ubuntu/launch_fix5.sh &
sleep 60
sed -i '/all queued runs launched/d' /home/ubuntu/launch_cl.sh
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 nice -n 5 bash /home/ubuntu/launch_cl.sh > /home/ubuntu/cl_launcher.log 2>&1
wait
echo "all queued runs launched" >> /home/ubuntu/launcher.log
