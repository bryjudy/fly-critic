#!/bin/bash
# Generic: launch an instance (retrying types/zones for capacity), wait for SSM, bootstrap with a launcher + watchdog.
# usage: aws/launch_box.sh <name> "<itype1 itype2>" <ami> <launcher-s3-key> <s3-prefix> "<ENV=val ENV2=val ...>" [userdata]
set -u; export AWS_PROFILE=${AWS_PROFILE:-default}
NAME=$1; ITYPES=$2; AMI=$3; LNCHKEY=$4; PREFIX=$5; ENVVARS=$6; UD=${7:-aws/userdata_gpu.sh}; HELPERS=${HELPERS:-"launch_fix5.sh launch_cl.sh"}   # extra S3 scripts to place in /home/ubuntu
cd "$(dirname "$0")/.."
launched=""
for attempt in $(seq 1 24); do
  for ITYPE in $ITYPES; do for SUB in ${FLYCRITIC_SUBNETS:?space-separated subnet ids}; do
    OUT=$(aws ec2 run-instances --image-id $AMI --instance-type $ITYPE --key-name ${FLYCRITIC_KEY:?key pair name} --security-group-ids ${FLYCRITIC_SG:?security group id} --subnet-id $SUB \
      --iam-instance-profile Name=${FLYCRITIC_INSTANCE_PROFILE:?instance profile with SSM and S3 access} --instance-initiated-shutdown-behavior terminate \
      --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=100,VolumeType=gp3,DeleteOnTermination=true}' \
      --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=owner,Value=fly-critic},{Key=autoterminate,Value=true}]" \
      --user-data file://$UD --query 'Instances[0].InstanceId' --output text 2>&1) && { IID=$OUT; echo "$IID	$ITYPE" > aws/instance_$NAME; echo "$(date +%T) launched $NAME $ITYPE in $SUB: $IID"; launched=1; break 2; }
  done; done
  [ -n "$launched" ] && break; echo "$(date +%T) no capacity/quota ($(echo $OUT | cut -c1-60)); retry in 5 min"; sleep 300
done
[ -z "$launched" ] && { echo "GAVE UP launching $NAME"; exit 1; }
for i in $(seq 1 60); do st=$(aws ssm describe-instance-information --filters Key=InstanceIds,Values=$IID --query 'InstanceInformationList[0].PingStatus' --output text 2>/dev/null); [ "$st" = "Online" ] && break; sleep 15; done
TGZ=$(aws s3 presign s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/flycritic.tgz --expires-in 7200); LNCH=$(aws s3 presign s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/$LNCHKEY --expires-in 7200); WD=$(aws s3 presign s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/watchdog.sh --expires-in 7200)
HCMDS=""; for h in $HELPERS; do u=$(aws s3 presign s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/$h --expires-in 7200 2>/dev/null) && HCMDS="$HCMDS\"curl -sSL -o $h '$u' || true\","; done
CMD=$(aws ssm send-command --instance-ids $IID --document-name AWS-RunShellScript --timeout-seconds 3600 \
  --parameters "commands=[\"cd /home/ubuntu\",\"curl -sSL -o flycritic.tgz '$TGZ'\",\"curl -sSL -o flycritic.train_launcher.sh '$LNCH'\",\"curl -sSL -o watchdog.sh '$WD'\",$HCMDS\"tar xzf flycritic.tgz 2>/dev/null || true\",\"chown -R ubuntu:ubuntu fly-critic flycritic.train_launcher.sh watchdog.sh\",\"until test -f /home/ubuntu/READY; do sleep 5; done\",\"touch /home/ubuntu/launcher.log && chown ubuntu:ubuntu /home/ubuntu/launcher.log\",\"printf '%s\\n' $ENVVARS > /home/ubuntu/launch.env; chown ubuntu:ubuntu /home/ubuntu/launch.env\",\"systemd-run --unit=fly-launcher --uid=ubuntu --gid=ubuntu -p KillMode=process -p EnvironmentFile=/home/ubuntu/launch.env -E HOME=/home/ubuntu -E PATH=/snap/bin:/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin --working-directory=/home/ubuntu bash -c 'bash /home/ubuntu/flycritic.train_launcher.sh >> /home/ubuntu/launcher.log 2>&1'\",\"systemd-run --unit=fly-watchdog --uid=ubuntu --gid=ubuntu -p KillMode=process -E HOME=/home/ubuntu -E PATH=/snap/bin:/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin --working-directory=/home/ubuntu bash -c 'bash /home/ubuntu/watchdog.sh $PREFIX > /home/ubuntu/watchdog.log 2>&1'\",\"sleep 240; tail -n 3 /home/ubuntu/launcher.log | cut -c1-100; pgrep -fc 'flycritic'\"]" --query 'Command.CommandId' --output text)
for i in $(seq 1 40); do st=$(aws ssm get-command-invocation --command-id $CMD --instance-id $IID --query Status --output text 2>/dev/null); [ "$st" = "Success" ] || [ "$st" = "Failed" ] && break; sleep 15; done
aws ssm get-command-invocation --command-id $CMD --instance-id $IID --query '[Status,StandardOutputContent,StandardErrorContent]' --output text | grep -vE "^$" | tail -6 | cut -c1-110
