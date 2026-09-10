#!/bin/bash
set -euo pipefail
cd /workspace/rz1t
export PATH="$HOME/.local/bin:$PATH"
export JAX_PLATFORMS=cuda
ROOT=/workspace/rz1t/results/estimate
mkdir -p "$ROOT"
wait_train() {
  while true; do
    if pgrep -f '[.]venv/bin/python -m rz1t.train' >/dev/null; then
      sleep 30
    else
      break
    fi
  done
}
bundle_if_needed() {
  local name=$1
  if [[ -f "$ROOT/${name}/final.json" && ! -f "$ROOT/${name}-metrics.tgz" ]]; then
    JAX_PLATFORMS=cpu .venv/bin/python -m runpod.u6_artifacts bundle --root "$ROOT" --name "$name"
  fi
}
run_one() {
  local arm=$1 seed=$2 slug=$3
  local name="${arm}-s${seed}"
  local cfg="configs/calibration/estimate-v1-${slug}-s${seed}.yaml"
  local out="$ROOT/$name"
  if [[ -f "$out/completion.json" || -f "$out/final.json" ]]; then
    echo "SKIP_COMPLETE $name"
    bundle_if_needed "$name"
    return 0
  fi
  if [[ -f "$out/identity.json" ]]; then
    echo "RESUME $name"
    .venv/bin/python -m rz1t.train --config "$cfg" --out "$out" --resume
  else
    echo "START $name"
    .venv/bin/python -m rz1t.train --config "$cfg" --out "$out"
  fi
  bundle_if_needed "$name"
}
echo CONTINUE_START
wait_train
bundle_if_needed U6-s4
bundle_if_needed R6x2-s4
run_one U6 4 u6
run_one R6x2 4 r6x2
run_one U12 4 u12
run_one U6 5 u6
run_one R6x2 5 r6x2
run_one U12 5 u12
run_one U6 6 u6
run_one R6x2 6 r6x2
run_one U12 6 u12
echo CONTINUE_DONE
ls -la "$ROOT"
