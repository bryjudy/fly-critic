#!/bin/bash
# Restart a box's launcher + watchdog as systemd transient units (immune to SSM command lifecycle).
# usage: aws/restart_systemd.sh <instance-id> <s3-prefix> "<KEY=val lines separated by ;>"
set -u; export AWS_PROFILE=${AWS_PROFILE:-default}; IID=$1; PREFIX=$2; ENVLINES=$3
ENVB64=$(echo "$ENVLINES" | tr ';' '\n' | base64 | tr -d '\n')
CMD=$(aws ssm send-command --instance-ids $IID --document-name AWS-RunShellScript --timeout-seconds 300 --parameters "commands=[
\"pkill -f flycritic.train_launcher; pkill -f watchdog.sh; sleep 1\",
\"cd /home/ubuntu/fly-critic/runs 2>/dev/null && for f in popgym_*.out; do [ -f \\\"\$f\\\" ] || continue; grep -q '^done:' \\\"\$f\\\" || { rm -rf \\\"\${f%.out}\\\" \\\"\$f\\\"; echo removed-incomplete \$f; }; done | wc -l\",
\"echo $ENVB64 | base64 -d > /home/ubuntu/launch.env; chown ubuntu:ubuntu /home/ubuntu/launch.env; touch /home/ubuntu/launcher.log; chown ubuntu:ubuntu /home/ubuntu/launcher.log\",
\"systemctl stop fly-launcher fly-watchdog 2>/dev/null; systemctl reset-failed 2>/dev/null\",
\"systemd-run --unit=fly-launcher --uid=ubuntu --gid=ubuntu -p KillMode=process -p EnvironmentFile=/home/ubuntu/launch.env -E HOME=/home/ubuntu -E PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin --working-directory=/home/ubuntu bash -c 'bash /home/ubuntu/flycritic.train_launcher.sh >> /home/ubuntu/launcher.log 2>&1'\",
\"systemd-run --unit=fly-watchdog --uid=ubuntu --gid=ubuntu -p KillMode=process -E HOME=/home/ubuntu -E PATH=/home/ubuntu/.local/bin:/usr/local/bin:/usr/bin:/bin --working-directory=/home/ubuntu bash -c 'bash /home/ubuntu/watchdog.sh $PREFIX > /home/ubuntu/watchdog.log 2>&1'\",
\"sleep 5; systemctl is-active fly-launcher fly-watchdog; tail -n 1 /home/ubuntu/launcher.log | cut -c1-90\"
]" --query 'Command.CommandId' --output text)
for i in $(seq 1 20); do st=$(aws ssm get-command-invocation --command-id $CMD --instance-id $IID --query Status --output text 2>/dev/null); [ "$st" = "Success" ] || [ "$st" = "Failed" ] && break; sleep 5; done
aws ssm get-command-invocation --command-id $CMD --instance-id $IID --query '[Status,StandardOutputContent,StandardErrorContent]' --output text | grep -vE "^$" | cut -c1-120
