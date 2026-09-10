"""Create a short-lived cheap community GPU pod, run the Shakespeare screen, terminate.

Does not download OWT, does not enable confirmation. Default is dry-run.

    python -m runpod.launch_screen
    python -m runpod.launch_screen --execute
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

GRAPHQL = "https://api.runpod.io/graphql"
REST = "https://rest.runpod.io/v1"
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".git", ".venv", ".pytest_cache", ".ruff_cache", "results", "data",
                "__pycache__", "rz1t.egg-info", ".cache"}
# Prefer 16–24 GB cards that can compile JAX; 8 GB 3070 is too tight.
CANDIDATES = (
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A4000",
    "NVIDIA GeForce RTX 3080 Ti",
    "NVIDIA GeForce RTX 4070 Ti",
    "NVIDIA GeForce RTX 3090 Ti",
)


def _headers(key, *, rest=False):
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 rz1t-profiler/0.1"}
    if rest:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def graphql(key, query, variables=None, timeout=60):
    body = {"query": query}
    if variables is not None:
        body["variables"] = variables
    req = urllib.request.Request(
        f"{GRAPHQL}?api_key={key}",
        data=json.dumps(body).encode(),
        headers=_headers(key),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"RunPod GraphQL HTTP {exc.code}: {exc.read().decode()[:2000]}") from exc
    if payload.get("errors"):
        raise RuntimeError(f"RunPod GraphQL error: {payload['errors']}")
    return payload["data"]


def rest(key, method, path, data=None, timeout=60):
    req = urllib.request.Request(
        f"{REST}{path}",
        data=None if data is None else json.dumps(data).encode(),
        headers=_headers(key, rest=True),
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read()
            if not raw:
                return {"status": response.status}
            return json.loads(raw.decode())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"RunPod REST {method} {path} HTTP {exc.code}: {exc.read().decode()[:2000]}") from exc


def quote_gpu(key, gpu_type, *, community):
    secure = "false" if community else "true"
    data = graphql(key, f"""
    query {{
      myself {{ id currentSpendPerHr clientBalance }}
      gpuTypes(input: {{id: "{gpu_type}"}}) {{
        id displayName memoryInGb communityPrice securePrice
        lowestPrice(input: {{gpuCount: 1, secureCloud: {secure}}}) {{
          uninterruptablePrice stockStatus availableGpuCounts
        }}
      }}
    }}
    """)
    gpu = data["gpuTypes"][0]
    price = gpu["lowestPrice"] or {}
    quoted = price.get("uninterruptablePrice")
    stock = price.get("stockStatus")
    return {
        "account": data["myself"],
        "gpu": gpu,
        "gpu_type": gpu_type,
        "community": community,
        "price_per_hour": quoted,
        "stock": stock,
        "available": stock not in (None, "None", "Unavailable"),
    }


def list_pods(key):
    payload = rest(key, "GET", "/pods")
    if isinstance(payload, list):
        return payload
    return payload.get("pods", [])


def create_pod(key, *, name, gpu_type, community, ssh_public, disk_gb):
    body = {
        "name": name,
        "imageName": IMAGE,
        "gpuTypeIds": [gpu_type],
        "gpuCount": 1,
        "cloudType": "COMMUNITY" if community else "SECURE",
        "containerDiskInGb": disk_gb,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "volumeMountPath": "/workspace",
        "env": {"PUBLIC_KEY": ssh_public.strip()},
        "supportPublicIp": True,
    }
    return rest(key, "POST", "/pods", body)


def get_pod(key, pod_id):
    try:
        for pod in list_pods(key):
            if pod.get("id") == pod_id:
                return pod
        return rest(key, "GET", f"/pods/{pod_id}")
    except (RuntimeError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def terminate_pod(key, pod_id):
    errors = []
    try:
        rest(key, "POST", f"/pods/{pod_id}/stop")
    except RuntimeError as exc:
        errors.append(str(exc))
    try:
        rest(key, "DELETE", f"/pods/{pod_id}")
        return {"terminated": pod_id, "stop_errors": errors}
    except RuntimeError as exc:
        errors.append(str(exc))
        try:
            graphql(key, f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}')
            return {"terminated": pod_id, "via": "graphql", "stop_errors": errors}
        except RuntimeError as gql_exc:
            return {"terminate_error": str(gql_exc), "pod_id": pod_id, "prior": errors}


def ssh_target(pod):
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if isinstance(port, dict) and int(port.get("privatePort") or 0) == 22 and port.get("ip") and port.get("publicPort"):
            return port["ip"], int(port["publicPort"])
    public_ip = pod.get("publicIp")
    mappings = pod.get("portMappings")
    if public_ip and isinstance(mappings, dict):
        ssh = mappings.get("22") or mappings.get("22/tcp") or mappings.get(22)
        if ssh:
            return public_ip, int(ssh)
    host = (pod.get("machine") or {}).get("podHostId") or pod.get("machineId")
    if host:
        return f"{host}.ssh.runpod.io", 22
    return None, None


def run_ssh(key_path, host, port, command, timeout, *, check=True):
    user_host = host if "@" in str(host) else f"root@{host}"
    completed = subprocess.run(
        ["ssh", "-i", str(key_path), "-p", str(port),
         "-o", "StrictHostKeyChecking=accept-new",
         "-o", "UserKnownHostsFile=/dev/null",
         "-o", "IdentitiesOnly=yes",
         "-o", "ConnectTimeout=15",
         user_host, command],
        check=False, capture_output=True, text=True, timeout=timeout,
    )
    if check and completed.returncode:
        raise subprocess.CalledProcessError(
            completed.returncode, completed.args, completed.stdout, completed.stderr)
    return completed


def pack_repo(destination):
    with tarfile.open(destination, "w:gz") as archive:
        for path in ROOT.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT)
            if any(part in EXCLUDE_DIRS for part in rel.parts):
                continue
            if rel.name.endswith(".pyc"):
                continue
            archive.add(path, arcname=str(Path("rz1t") / rel))
    return destination


def wait_ssh(key, pod_id, key_path, deadline):
    last = None
    while time.time() < deadline:
        try:
            last = get_pod(key, pod_id)
        except (RuntimeError, urllib.error.URLError, TimeoutError, OSError):
            last = last
            time.sleep(8)
            continue
        if last and str(last.get("desiredStatus", "")).upper() == "RUNNING":
            host, port = ssh_target(last)
            if host and port:
                try:
                    run_ssh(key_path, host, port, "uname -a && nvidia-smi -L", timeout=30)
                    return last, host, port
                except (subprocess.SubprocessError, OSError):
                    pass
        time.sleep(8)
    raise TimeoutError(f"pod {pod_id} SSH not ready; last={last}")


def choose_quote(key):
    quotes = []
    for gpu_type in CANDIDATES:
        quotes.append(quote_gpu(key, gpu_type, community=True))
    available = [q for q in quotes if q["available"] and q["price_per_hour"]]
    available.sort(key=lambda q: q["price_per_hour"])
    return quotes, available


def execute(args):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set")
    quotes, available = choose_quote(key)
    existing = list_pods(key)
    plan = {
        "quotes": [{"gpu_type": q["gpu_type"], "community": q["community"],
                    "price_per_hour": q["price_per_hour"], "stock": q["stock"],
                    "available": q["available"]} for q in quotes],
        "image": IMAGE,
        "max_minutes": args.max_minutes,
        "existing_pods": [{"id": p.get("id"), "name": p.get("name"),
                           "status": p.get("desiredStatus"), "costPerHr": p.get("costPerHr")}
                          for p in existing],
        "purpose": "shakespeare char architecture screen; terminate after",
    }
    if existing:
        plan["decision"] = "abort_existing_pods"
        return plan
    if not available:
        plan["decision"] = "abort_no_stock"
        return plan
    chosen = available[0]
    estimated = max(q["price_per_hour"] for q in available) * (args.max_minutes / 60)
    plan["candidates"] = [{k: q[k] for k in ("gpu_type", "community", "price_per_hour", "stock")} for q in available]
    plan["chosen"] = {k: chosen[k] for k in ("gpu_type", "community", "price_per_hour", "stock")}
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
             "-C", "rz1t-screen"],
            check=True, capture_output=True, text=True,
        )
    public = (work / "id_ed25519.pub").read_text().strip()
    archive = work / "rz1t-src.tar.gz"
    pack_repo(archive)
    started = time.time()
    deadline = started + args.max_minutes * 60
    pod = None
    result = dict(plan)
    result["decision"] = "execute"
    try:
        create_errors = []
        pod = None
        for candidate in available:
            try:
                pod = create_pod(key, name=args.name, gpu_type=candidate["gpu_type"],
                                 community=candidate["community"], ssh_public=public, disk_gb=args.disk_gb)
                chosen = candidate
                result["chosen"] = {k: chosen[k] for k in ("gpu_type", "community", "price_per_hour", "stock")}
                break
            except RuntimeError as exc:
                create_errors.append({"gpu_type": candidate["gpu_type"], "error": str(exc)})
        result["create_errors"] = create_errors
        if pod is None:
            raise RuntimeError(f"no community GPU created: {create_errors}")
        result["pod"] = pod
        (work / "pod.json").write_text(json.dumps(pod, indent=2) + "\n")
        pod_state, host, port = wait_ssh(key, pod["id"], key_path, min(deadline, time.time() + 600))
        result["ssh"] = {"host": host, "port": port}
        subprocess.run(
            ["scp", "-i", str(key_path), "-P", str(port),
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", "UserKnownHostsFile=/dev/null",
             "-o", "IdentitiesOnly=yes",
             str(archive), f"root@{host}:/tmp/rz1t-src.tar.gz"],
            check=True, timeout=120,
        )
        remaining = max(60, deadline - time.time() - 90)
        remote = f"""
set -euo pipefail
mkdir -p /workspace /workspace/rz1t/results/shakespeare-screen
tar -xzf /tmp/rz1t-src.tar.gz -C /workspace
cd /workspace/rz1t
nvidia-smi > /workspace/rz1t/results/shakespeare-screen/nvidia-smi-boot.txt || true
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
if bash runpod/pod_setup.sh --cuda; then
  export JAX_PLATFORMS=cuda
else
  echo 'CUDA extra failed; falling back to CPU JAX' | tee -a results/shakespeare-screen/setup.txt
  bash runpod/pod_setup.sh
  export JAX_PLATFORMS=cpu
fi
uv run python -m rz1t.screen \\
  --out results/shakespeare-screen \\
  --prepared data/shakespeare-char-128 \\
  --steps {args.steps} --batch-size {args.batch_size} --sequence {args.sequence} \\
  --n-embed {args.n_embed}
"""
        completed = run_ssh(key_path, host, port, remote, timeout=remaining, check=False)
        (work / "remote-stdout.txt").write_text(completed.stdout or "")
        (work / "remote-stderr.txt").write_text(completed.stderr or "")
        result["remote_returncode"] = completed.returncode
        subprocess.run(
            ["scp", "-i", str(key_path), "-P", str(port), "-r",
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", "UserKnownHostsFile=/dev/null",
             "-o", "IdentitiesOnly=yes",
             f"root@{host}:/workspace/rz1t/results/shakespeare-screen",
             str(work / "shakespeare-screen")],
            check=False, timeout=180,
        )
        if completed.returncode:
            raise RuntimeError(f"remote screen failed rc={completed.returncode}; see remote-stderr.txt")
        result["status"] = "screen_finished"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, subprocess.CalledProcessError):
            (work / "remote-stdout.txt").write_text(exc.stdout or "")
            (work / "remote-stderr.txt").write_text(exc.stderr or "")
    finally:
        if pod and pod.get("id"):
            result["terminate"] = terminate_pod(key, pod["id"])
        result["elapsed_seconds"] = time.time() - started
        (work / "launch.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-minutes", type=int, default=45)
    parser.add_argument("--spend-cap", type=float, default=3.0)
    parser.add_argument("--disk-gb", type=int, default=30)
    parser.add_argument("--name", default="rz1t-shakespeare-screen")
    parser.add_argument("--work", default=str(ROOT / "results" / "shakespeare-launch"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--n-embed", type=int, default=64)
    args = parser.parse_args(argv)
    result = execute(args)
    printable = {k: result[k] for k in result if k != "quotes"}
    print(json.dumps(printable, indent=2, default=str))
    print(json.dumps({"quotes": result.get("quotes")}, indent=2, default=str))
    return 0 if result.get("decision") in {"dry_run", "execute"} and result.get("status") != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
