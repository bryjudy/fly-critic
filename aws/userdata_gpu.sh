#!/bin/bash
su - ubuntu -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
snap install aws-cli --classic || true
touch /home/ubuntu/READY
