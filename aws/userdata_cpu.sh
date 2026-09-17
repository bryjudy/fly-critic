#!/bin/bash
apt-get update -y >/dev/null 2>&1; snap install aws-cli --classic || true
su - ubuntu -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
touch /home/ubuntu/READY
