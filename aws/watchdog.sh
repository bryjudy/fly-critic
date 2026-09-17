#!/bin/bash
# Runs ON the box. Waits until the launcher has queued everything and no training process remains, then does a
# final S3 sync and powers off. With InstanceInitiatedShutdownBehavior=terminate this kills the instance with no
# dependence on any laptop session. usage: watchdog.sh <s3-prefix>
PREFIX=$1; cd /home/ubuntu/fly-critic
sleep 600
while true; do
  launched=$(grep -c "all queued runs launched" /home/ubuntu/launcher.log 2>/dev/null)
  n=$(pgrep -fc "flycritic\.(train |popgym\.train|cl\.run|cl\.data)" || true)
  if [ "$launched" -ge 1 ] && [ "$n" = "0" ]; then break; fi
  sleep 120
done
aws s3 sync runs s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/$PREFIX/ --only-show-errors --exclude "*/ckpt.pt"
echo "$(date) watchdog: all done, shutting down" >> /home/ubuntu/launcher.log
aws s3 cp /home/ubuntu/launcher.log s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/$PREFIX/launcher.log --only-show-errors
sudo shutdown -h now
