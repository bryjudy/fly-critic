#!/bin/bash
# Pull runs/ from the EC2 training box via S3 (no inbound ports). Usage: aws/pull_results.sh [instance-id]
set -e
export AWS_PROFILE=${AWS_PROFILE:-default}
IID=${1:-$(cut -f1 "$(dirname "$0")/current_instance")}; B=s3://${FLYCRITIC_BUCKET:?set FLYCRITIC_BUCKET}
PUT=$(aws s3 presign "$B/results.tgz" --expires-in 3600 --http-method PUT 2>/dev/null || true)
CMD=$(aws ssm send-command --instance-ids $IID --document-name AWS-RunShellScript --timeout-seconds 600 \
  --parameters "commands=[\"cd /home/ubuntu/fly-critic && tar czf /tmp/results.tgz runs && curl -sS -X PUT -T /tmp/results.tgz '$PUT' && echo uploaded\"]" \
  --query 'Command.CommandId' --output text)
for i in $(seq 1 40); do st=$(aws ssm get-command-invocation --command-id $CMD --instance-id $IID --query Status --output text 2>/dev/null); [ "$st" = "Success" ] && break; [ "$st" = "Failed" ] && { aws ssm get-command-invocation --command-id $CMD --instance-id $IID --query StandardErrorContent --output text; exit 1; }; sleep 5; done
aws s3 cp $B/results.tgz /tmp/results.tgz --only-show-errors
cd "$(dirname "$0")/.." && tar xzf /tmp/results.tgz && echo "runs/ merged from EC2:" && ls runs | grep -E "_s[12]"
