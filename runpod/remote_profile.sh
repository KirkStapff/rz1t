#!/usr/bin/env bash
# Runs on the rented H100. Synthetic profile only. No OWT, no confirmation.
set -euo pipefail
mkdir -p /workspace /workspace/rz1t/results/h100-profile
tar -xzf /tmp/rz1t-src.tar.gz -C /workspace
cd /workspace/rz1t
nvidia-smi > /workspace/rz1t/results/h100-profile/nvidia-smi-boot.txt
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
bash runpod/pod_setup.sh --cuda
JAX_PLATFORMS=cuda uv run python -m rz1t.profile \
  --out results/h100-profile --batches 1 2 4 8 --warmup 3 --timed 8
echo PROFILE_OK
