"""Create a short-lived 1xH100 environment-profile pod, then terminate it.

Reads RUNPOD_API_KEY from the environment. Does not download OWT, does not
enable confirmation, and does not keep idle GPUs. Default is dry-run.

    python -m runpod.launch_profile
    python -m runpod.launch_profile --execute
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
GPU_TYPE = "NVIDIA H100 80GB HBM3"
IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_DIRS = {".git", ".venv", ".pytest_cache", ".ruff_cache", "results", "data",
                "__pycache__", "rz1t.egg-info", ".cache"}


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


def rest(key, method, path, data=None, timeout=60, attempts=4):
    last = None
    for attempt in range(attempts):
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
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"RunPod REST {method} {path} unreachable: {last}") from last


def quote_gpu(key, *, community):
    secure = "false" if community else "true"
    data = graphql(key, f"""
    query {{
      myself {{ id currentSpendPerHr clientBalance }}
      gpuTypes(input: {{id: "{GPU_TYPE}"}}) {{
        id displayName memoryInGb communityPrice securePrice
        lowestPrice(input: {{gpuCount: 1, secureCloud: {secure}}}) {{
          uninterruptablePrice stockStatus availableGpuCounts minVcpu minMemory
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
        "community": community,
        "price_per_hour": quoted,
        "stock": stock,
        "available": stock not in (None, "None"),
    }


def list_pods(key):
    payload = rest(key, "GET", "/pods")
    if isinstance(payload, list):
        return payload
    return payload.get("pods", [])


def create_pod(key, *, name, community, ssh_public, disk_gb):
    body = {
        "name": name,
        "imageName": IMAGE,
        "gpuTypeIds": [GPU_TYPE],
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
    rest_pod = None
    for pod in list_pods(key):
        if pod.get("id") == pod_id:
            rest_pod = pod
            break
    if rest_pod is None:
        try:
            rest_pod = rest(key, "GET", f"/pods/{pod_id}")
        except RuntimeError:
            rest_pod = None
    gql_pod = None
    try:
        data = graphql(key, """
        query {
          myself {
            pods {
              id name desiredStatus costPerHr gpuCount imageName
              runtime {
                uptimeInSeconds
                ports { ip isIpPublic privatePort publicPort type }
              }
              machine { podHostId gpuDisplayName dataCenterId }
            }
          }
        }
        """)
        for pod in data["myself"]["pods"]:
            if pod.get("id") == pod_id:
                gql_pod = pod
                break
    except RuntimeError:
        gql_pod = None
    if rest_pod is None and gql_pod is None:
        return None
    merged = dict(rest_pod or {})
    if gql_pod:
        merged.update({k: v for k, v in gql_pod.items() if v})
        if gql_pod.get("runtime"):
            merged["runtime"] = gql_pod["runtime"]
        if gql_pod.get("machine"):
            merged["machine"] = {**(merged.get("machine") or {}), **gql_pod["machine"]}
    return merged


def terminate_pod(key, pod_id, attempts=6):
    last = None
    for attempt in range(attempts):
        try:
            errors = []
            try:
                rest(key, "POST", f"/pods/{pod_id}/stop")
            except RuntimeError as exc:
                errors.append(str(exc))
            try:
                rest(key, "DELETE", f"/pods/{pod_id}")
                return {"terminated": pod_id, "stop_errors": errors, "attempts": attempt + 1}
            except RuntimeError as exc:
                errors.append(str(exc))
                graphql(key, f'mutation {{ podTerminate(input: {{podId: "{pod_id}"}}) }}')
                return {"terminated": pod_id, "via": "graphql", "stop_errors": errors, "attempts": attempt + 1}
        except Exception as exc:
            last = exc
            time.sleep(min(2 ** attempt, 20))
    return {"terminate_error": str(last), "pod_id": pod_id}


def ssh_candidates(pod):
    candidates = []
    runtime = pod.get("runtime") or {}
    for port in runtime.get("ports") or []:
        if isinstance(port, dict) and int(port.get("privatePort") or 0) == 22 and port.get("ip") and port.get("publicPort"):
            candidates.append((port["ip"], int(port["publicPort"])))
    public_ip = pod.get("publicIp")
    mappings = pod.get("portMappings")
    if public_ip and isinstance(mappings, dict):
        ssh = mappings.get("22") or mappings.get("22/tcp") or mappings.get(22)
        if ssh:
            candidates.append((public_ip, int(ssh)))
    if public_ip and isinstance(mappings, list):
        for port in mappings:
            if isinstance(port, dict) and int(port.get("privatePort") or 0) == 22 and port.get("publicPort"):
                candidates.append((public_ip, int(port["publicPort"])))
    host = (pod.get("machine") or {}).get("podHostId")
    pod_id = pod.get("id")
    if host:
        candidates.append((f"{host}.ssh.runpod.io", 22))
    if pod_id:
        candidates.append((f"{pod_id}@ssh.runpod.io", 22))
    unique = []
    seen = set()
    for host, port in candidates:
        if (host, port) not in seen and host and port:
            seen.add((host, port))
            unique.append((host, port))
    return unique


def ssh_target(pod):
    candidates = ssh_candidates(pod)
    if not candidates:
        return None, None
    return candidates[0]


def run_ssh(key_path, host, port, command, timeout):
    user_host = host if "@" in str(host) else f"root@{host}"
    return subprocess.run(
        ["ssh", "-i", str(key_path), "-p", str(port),
         "-o", "StrictHostKeyChecking=accept-new",
         "-o", "UserKnownHostsFile=/dev/null",
         "-o", "IdentitiesOnly=yes",
         "-o", "ConnectTimeout=15",
         "-o", "ServerAliveInterval=30",
         "-o", "ServerAliveCountMax=10",
         "-o", "TCPKeepAlive=yes",
         user_host, command],
        check=True, capture_output=True, text=True, timeout=timeout,
    )


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
        last = get_pod(key, pod_id)
        if last and str(last.get("desiredStatus", "")).upper() == "RUNNING":
            for host, port in ssh_candidates(last):
                try:
                    run_ssh(key_path, host, port, "uname -a && nvidia-smi -L", timeout=30)
                    return last, host, port
                except (subprocess.SubprocessError, OSError):
                    continue
        time.sleep(8)
    raise TimeoutError(f"pod {pod_id} SSH not ready; last={last}")


def execute(args):
    key = os.environ.get("RUNPOD_API_KEY")
    if not key:
        raise SystemExit("RUNPOD_API_KEY is not set")
    quotes = [quote_gpu(key, community=True), quote_gpu(key, community=False)]
    chosen = next((q for q in quotes if q["available"] and q["price_per_hour"]), None)
    if chosen is None:
        fallback = quotes[1]
        listed = fallback["gpu"].get("securePrice") or fallback["gpu"].get("communityPrice")
        if listed:
            chosen = dict(fallback)
            chosen.update(available=True, price_per_hour=listed, stock=chosen.get("stock") or "unknown")
    existing = list_pods(key)
    plan = {
        "quotes": quotes,
        "gpu_type": GPU_TYPE,
        "image": IMAGE,
        "max_minutes": args.max_minutes,
        "disk_gb": args.disk_gb,
        "existing_pods": [{"id": p.get("id"), "name": p.get("name"), "status": p.get("desiredStatus"),
                           "costPerHr": p.get("costPerHr")} for p in existing],
        "purpose": "1xH100 environment/cost profile; synthetic data; terminate after",
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
             "-C", "rz1t-h100-profile"],
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
        pod = create_pod(key, name=args.name, community=chosen["community"],
                         ssh_public=public, disk_gb=args.disk_gb)
        result["pod"] = pod
        (work / "pod.json").write_text(json.dumps(pod, indent=2) + "\n")
        gpu_name = str((pod.get("machine") or {}).get("gpuDisplayName") or pod.get("gpu") or "")
        if pod.get("gpuCount") not in (1, None) or (gpu_name and "H100" not in gpu_name.upper() and GPU_TYPE not in str(pod)):
            raise RuntimeError(f"refusing unexpected GPU allocation: {pod}")
        pod_state, host, port = wait_ssh(key, pod["id"], key_path, min(deadline, time.time() + 600))
        result["ssh"] = {"host": host, "port": port, "pod": pod_state}
        subprocess.run(
            ["scp", "-i", str(key_path), "-P", str(port),
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", "UserKnownHostsFile=/dev/null",
             "-o", "IdentitiesOnly=yes",
             str(archive), f"root@{host}:/tmp/rz1t-src.tar.gz"],
            check=True, timeout=120,
        )
        remaining = max(60, deadline - time.time() - 90)
        batches = " ".join(str(b) for b in args.batches)
        remote = f"""
set -euo pipefail
mkdir -p /workspace
tar -xzf /tmp/rz1t-src.tar.gz -C /workspace
cd /workspace/rz1t
mkdir -p /workspace/rz1t/results/h100-profile
nvidia-smi > /workspace/rz1t/results/h100-profile/nvidia-smi-boot.txt
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12
bash runpod/pod_setup.sh --cuda
# Separate processes so an OOM in one arm does not poison the other.
JAX_PLATFORMS=cuda uv run python -m rz1t.profile --out results/h100-profile \\
  --arm U12 --batches {batches} --warmup {args.warmup} --timed {args.timed}
JAX_PLATFORMS=cuda uv run python -m rz1t.profile --out results/h100-profile \\
  --arm R6x2 --batches {batches} --warmup {args.warmup} --timed {args.timed}
"""
        completed = run_ssh(key_path, host, port, remote, timeout=remaining)
        (work / "remote-stdout.txt").write_text(completed.stdout)
        (work / "remote-stderr.txt").write_text(completed.stderr)
        subprocess.run(
            ["scp", "-i", str(key_path), "-P", str(port), "-r",
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", "UserKnownHostsFile=/dev/null",
             "-o", "IdentitiesOnly=yes",
             f"root@{host}:/workspace/rz1t/results/h100-profile",
             str(work / "h100-profile")],
            check=False, timeout=120,
        )
        result["status"] = "profile_finished"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if pod and pod.get("id"):
            result["terminate"] = terminate_pod(key, pod["id"])
        result["elapsed_seconds"] = time.time() - started
        (work / "launch.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-minutes", type=int, default=40)
    parser.add_argument("--spend-cap", type=float, default=8.0)
    parser.add_argument("--disk-gb", type=int, default=40)
    parser.add_argument("--name", default="rz1t-h100-profile")
    parser.add_argument("--work", default=str(ROOT / "results" / "h100-launch"))
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed", type=int, default=8)
    args = parser.parse_args(argv)
    result = execute(args)
    print(json.dumps({k: result[k] for k in result if k != "quotes"}, indent=2, default=str))
    print(json.dumps({"quotes": result.get("quotes")}, indent=2, default=str))
    return 0 if result.get("decision") in {"dry_run", "execute"} and result.get("status") != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
