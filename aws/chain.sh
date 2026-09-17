#!/bin/bash
# usage: aws/chain.sh <instance-id> <s3-prefix>   — poll box until training procs = 0, sync results, terminate, aggregate.
export AWS_PROFILE=${AWS_PROFILE:-default}; IID=$1; B=s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}/$2/
cd "$(dirname "$0")/.."
while true; do
  state=$(aws ec2 describe-instances --instance-ids $IID --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
  case "$state" in
    terminated|stopped|shutting-down) echo "$(date +%H:%M) instance $state"; break;;
    running) ;;
    *) echo "$(date +%H:%M) transient: state='$state'"; sleep 120; continue;;
  esac
  CMD=$(aws ssm send-command --instance-ids $IID --document-name AWS-RunShellScript --parameters 'commands=["pgrep -fc flycritic.train || true; for f in /home/ubuntu/fly-critic/runs/*.out; do grep -E \"^it \" $f | tail -n 1 | cut -c1-16; done | tr \"\\n\" \" \""]' --query 'Command.CommandId' --output text 2>/dev/null)
  sleep 6; out=$(aws ssm get-command-invocation --command-id $CMD --instance-id $IID --query StandardOutputContent --output text 2>/dev/null); n=$(echo "$out" | head -1 | tr -d '[:space:]')
  echo "$(date +%H:%M) remote procs: $n | $(echo "$out" | tail -n +2 | tr -s ' ')"
  [ "$n" = "0" ] && break
  sleep 600
done
until aws s3 sync $B runs/ --only-show-errors; do sleep 60; done; echo "runs synced from S3"
for i in 1 2 3; do st=$(aws ec2 describe-instances --instance-ids $IID --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null); [ "$st" = "running" ] && { aws ec2 terminate-instances --instance-ids $IID --query 'TerminatingInstances[0].CurrentState.Name' --output text && echo "INSTANCE TERMINATED"; break; }; [ "$st" = "terminated" ] && break; sleep 60; done
echo "=== aggregate across seeds ==="; uv run python -m flycritic.aggregate
