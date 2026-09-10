#!/usr/bin/env bash
# Local environment setup only. No provisioning, training, or paid resources.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE=cpu
VENV="$ROOT/.venv"
while (($#)); do
  case "$1" in
    --cuda) MODE=cuda; shift ;;
    --venv) VENV="${2:?--venv requires a path}"; shift 2 ;;
    -h|--help) echo 'Usage: bash runpod/pod_setup.sh [--cuda] [--venv PATH]'; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
command -v uv >/dev/null || { echo 'Install uv explicitly first: https://docs.astral.sh/uv/' >&2; exit 1; }
[[ -f "$ROOT/pyproject.toml" ]] || { echo 'Missing root package metadata' >&2; exit 1; }
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv --python 3.12 "$VENV"
fi
"$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 required"'
EXTRAS=(--extra dev --extra prepare)
if [[ "$MODE" == cuda ]]; then
  command -v nvidia-smi >/dev/null || { echo '--cuda requires NVIDIA driver/nvidia-smi' >&2; exit 1; }
  nvidia-smi
  EXTRAS+=(--extra cuda)
fi
# Root extras own dependency policy. Never silently select a CUDA wheel on CPU.
cd "$ROOT"
UV_PROJECT_ENVIRONMENT="$VENV" uv sync --locked "${EXTRAS[@]}"
mkdir -p "$ROOT/reference/environments"
uv pip freeze --python "$VENV/bin/python" > "$ROOT/reference/environments/installed-$MODE.txt"
"$VENV/bin/python" -c 'import jax, rz1t; print("JAX", jax.__version__, "devices:", jax.devices())'
printf 'Environment ready: %s/bin/python\nNo training started. Shut down idle provider allocations yourself.\n' "$VENV"
