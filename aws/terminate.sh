#!/bin/bash
# Kill the EC2 training box. ALWAYS run when done — it bills while up.
export AWS_PROFILE=${AWS_PROFILE:-default}; IID=${1:-$(cut -f1 "$(dirname "$0")/current_instance")}
aws ec2 terminate-instances --instance-ids $IID --query 'TerminatingInstances[0].CurrentState.Name' --output text
