"""1xH100: OWT *subset* prepare + one 1e16 U12/R6x2 development pair, then terminate.

Not full OpenWebText, not confirmation, not a sealed primary.

    python -m runpod.launch_owt_pair
    python -m runpod.launch_owt_pair --execute
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import urllib.error
import urllib.request

from runpod import launch_profile as lp

ROOT = Path(__file__).resolve().parents[1]
ENCODER_SHA = "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"
VOCAB_SHA = "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"


def execute(args):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set")
    quotes = [lp.quote_gpu(key, community=True), lp.quote_gpu(key, community=False)]
    chosen = next((q for q in quotes if q["available"] and q["price_per_hour"]), None)
    if chosen is None:
        fallback = quotes[1]
        listed = fallback["gpu"].get("securePrice") or fallback["gpu"].get("communityPrice")
        if listed:
            chosen = dict(fallback)
            chosen.update(available=True, price_per_hour=listed, stock=chosen.get("stock") or "unknown")
    existing = lp.list_pods(key)
    plan = {
        "quotes": quotes,
        "gpu_type": lp.GPU_TYPE,
        "image": lp.IMAGE,
        "max_minutes": args.max_minutes,
        "disk_gb": args.disk_gb,
        "n_docs": args.n_docs,
        "existing_pods": [{"id": p.get("id"), "name": p.get("name"), "status": p.get("desiredStatus"),
                           "costPerHr": p.get("costPerHr")} for p in existing],
        "purpose": "OWT-subset 1e16 U12 vs R6x2 development pair; terminate after",
        "not_confirmation": True,
        "not_full_owt": True,
    }
    if existing:
        plan["decision"] = "abort_existing_pods"
        return plan
    if chosen is None:
        plan["decision"] = "abort_no_stock"
        return plan
    estimated = chosen["price_per_hour"] * (args.max_minutes / 60)
    plan["chosen"] = {k: chosen[k] for k in ("community", "price_per_hour", "stock")}
    plan["estimated_usd_cap"] = estimated
    plan["account"] = chosen["account"]
    if estimated > args.spend_cap:
        plan["decision"] = "abort_over_cap"
        return plan
    balance = chosen["account"]["clientBalance"]
    if balance is not None and balance < estimated + 5:
        plan["decision"] = "abort_low_balance"
        return plan
    if not args.execute:
        plan["decision"] = "dry_run"
        return plan

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    key_path = work / "id_ed25519"
    if not key_path.exists():
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(key_path),
             "-C", "rz1t-owt-pair"],
            check=True, capture_output=True, text=True,
        )
    public = (work / "id_ed25519.pub").read_text().strip()
    archive = work / "rz1t-src.tar.gz"
    lp.pack_repo(archive)
    started = time.time()
    deadline = started + args.max_minutes * 60
    pod = None
    result = dict(plan)
    result["decision"] = "execute"
    try:
        pod = lp.create_pod(key, name=args.name, community=chosen["community"],
                            ssh_public=public, disk_gb=args.disk_gb)
        result["pod"] = pod
        (work / "pod.json").write_text(json.dumps(pod, indent=2) + "\n")
        gpu_name = str((pod.get("machine") or {}).get("gpuDisplayName") or pod.get("gpu") or "")
        if pod.get("gpuCount") not in (1, None) or (
                gpu_name and "H100" not in gpu_name.upper() and lp.GPU_TYPE not in str(pod)):
            raise RuntimeError(f"refusing unexpected GPU allocation: {pod}")
        pod_state, host, port = lp.wait_ssh(key, pod["id"], key_path, min(deadline, time.time() + 600))
        result["ssh"] = {"host": host, "port": port}
        subprocess.run(
            ["scp", "-i", str(key_path), "-P", str(port),
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", "UserKnownHostsFile=/dev/null",
             "-o", "IdentitiesOnly=yes",
             str(archive), f"root@{host}:/tmp/rz1t-src.tar.gz"],
            check=True, timeout=180,
        )
        remaining = max(60, deadline - time.time() - 90)
        remote = f"""
set -euo pipefail
mkdir -p /workspace
tar -xzf /tmp/rz1t-src.tar.gz -C /workspace
cd /workspace/rz1t
mkdir -p results/owt-dev-1e16 data
nvidia-smi > results/owt-dev-1e16/nvidia-smi-boot.txt
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
bash runpod/pod_setup.sh --cuda
uv pip install datasets
test -f reference/gpt2-assets/encoder.json
test -f reference/gpt2-assets/vocab.bpe
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
uv run python -m rz1t.fetch_owt_subset --n-docs {args.n_docs} --out data/owt-subset.jsonl
test -s data/owt-subset.jsonl
PYTHON=.venv/bin/python bash runpod/prepare_owt.sh \\
  data/owt-subset.jsonl data/owt-subset-256 \\
  reference/gpt2-assets/encoder.json reference/gpt2-assets/vocab.bpe \\
  {ENCODER_SHA} {VOCAB_SHA} 256
JAX_PLATFORMS=cuda uv run python -m rz1t.train \\
  --config configs/dev/owt-u12-1e16.yaml --out results/owt-dev-1e16/U12 \\
  | tee results/owt-dev-1e16/U12-stdout.json
JAX_PLATFORMS=cuda uv run python -m rz1t.train \\
  --config configs/dev/owt-r6x2-1e16.yaml --out results/owt-dev-1e16/R6x2 \\
  | tee results/owt-dev-1e16/R6x2-stdout.json
python - <<'PY'
import json
from pathlib import Path
root = Path('results/owt-dev-1e16')
rows = []
for arm in ('U12', 'R6x2'):
    rec = json.loads((root / arm / 'analysis_records.json').read_text())
    rows.extend(rec)
(root / 'pair.json').write_text(json.dumps(rows, indent=2) + '\\n')
print(json.dumps(rows, indent=2))
PY
"""
        completed = lp.run_ssh(key_path, host, port, remote, timeout=remaining)
        (work / "remote-stdout.txt").write_text(completed.stdout)
        (work / "remote-stderr.txt").write_text(completed.stderr)
        subprocess.run(
            ["scp", "-i", str(key_path), "-P", str(port), "-r",
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", "UserKnownHostsFile=/dev/null",
             "-o", "IdentitiesOnly=yes",
             f"root@{host}:/workspace/rz1t/results/owt-dev-1e16",
             str(work / "owt-dev-1e16")],
            check=False, timeout=300,
        )
        result["status"] = "pair_finished"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, subprocess.CalledProcessError):
            (work / "remote-stdout.txt").write_text(exc.stdout or "")
            (work / "remote-stderr.txt").write_text(exc.stderr or "")
    finally:
        if pod and pod.get("id"):
            result["terminate"] = lp.terminate_pod(key, pod["id"])
        result["elapsed_seconds"] = time.time() - started
        (work / "launch.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-minutes", type=int, default=70)
    parser.add_argument("--spend-cap", type=float, default=12.0)
    parser.add_argument("--disk-gb", type=int, default=80)
    parser.add_argument("--n-docs", type=int, default=40_000)
    parser.add_argument("--name", default="rz1t-owt-dev-1e16")
    parser.add_argument("--work", default=str(ROOT / "results" / "owt-dev-launch"))
    args = parser.parse_args(argv)
    result = execute(args)
    printable = {k: result[k] for k in result if k != "quotes"}
    print(json.dumps(printable, indent=2, default=str))
    print(json.dumps({"quotes": result.get("quotes")}, indent=2, default=str))
    ok = result.get("decision") in {"dry_run", "execute"} and result.get("status") != "failed"
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
