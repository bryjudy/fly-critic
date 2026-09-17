#!/bin/bash
# Tail the last progress line of every run on the EC2 box.
export AWS_PROFILE=${AWS_PROFILE:-default}; IID=${1:-$(cut -f1 "$(dirname "$0")/current_instance")}
CMD=$(aws ssm send-command --instance-ids $IID --document-name AWS-RunShellScript --parameters 'commands=["for f in /home/ubuntu/fly-critic/runs/*.out; do printf \"%-24s \" $(basename $f .out); grep -E \"^it \" $f | tail -n 1 | cut -c1-70; done; uptime"]' --query 'Command.CommandId' --output text)
sleep 4; aws ssm get-command-invocation --command-id $CMD --instance-id $IID --query StandardOutputContent --output text
