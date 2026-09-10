"""Measure GPU environment and U12/R6x2 training-step cost. Synthetic data only.

This is an engineering profile, not language-model evidence, not confirmation,
and not a hardware-energy result. Example:

    python -m rz1t.profile --out results/h100-profile
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from rz1t.checkpoint import atomic_json
from rz1t.config import ModelConfig
from rz1t.conventions import make_steps, parameter_counts, trainable_filter
from rz1t.flops import ACCOUNTING_VERSION, accounting
from rz1t.make_sweep import DEFAULT_MODEL, validate_primary_pair
from rz1t.model import create_model
from rz1t.train import environment_identity, source_identity


PROFILE_VERSION = "rz1t-h100-profile-v2"
PRIMARY_ARMS = {
    "U12": {"p": 0, "m": 12, "k": 1, "q": 0},
    "R6x2": {"p": 0, "m": 6, "k": 2, "q": 0},
}
OOM_MARKERS = ("out of memory", "resource_exhausted", "oom", "enomem", "cudaerror")


def _nvidia_smi():
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        full = subprocess.run(["nvidia-smi"], check=True, capture_output=True, text=True, timeout=30)
        return {"query": query.stdout.strip(), "report": full.stdout}
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}


def _device_memory():
    stats = []
    for device in jax.devices():
        row = {"id": str(device), "kind": device.device_kind}
        try:
            row["memory_stats"] = {str(k): int(v) if isinstance(v, (int, np.integer)) else v
                                   for k, v in device.memory_stats().items()}
        except Exception as exc:  # JAX CPU backends may lack memory_stats.
            row["memory_stats_error"] = str(exc)
        stats.append(row)
    return stats


def _batch(model_config, batch_size, seed=0):
    key = jax.random.PRNGKey(seed)
    tokens = jax.random.randint(key, (batch_size, model_config.sequence + 1), 0, model_config.vocab)
    return tokens[:, :-1], tokens[:, 1:]


def _time_steps(train_step, model, opt_state, x, y, steps):
    elapsed = []
    for _ in range(steps):
        tick = time.perf_counter()
        model, opt_state, value = train_step(model, opt_state, x, y)
        nll = float(value)
        elapsed.append(time.perf_counter() - tick)
        if not math_isfinite(nll):
            raise ValueError("nonfinite training loss during profile")
    return model, opt_state, elapsed, nll


def math_isfinite(value):
    return bool(np.isfinite(value))


def _is_oom(exc):
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in OOM_MARKERS)


def profile_arm(name, model_cfg, *, batches, warmup, timed, seed):
    model_config = ModelConfig.from_dict(model_cfg)
    model = create_model(model_config, jax.random.PRNGKey(seed))
    counts_by_batch = {batch: accounting(model_config, batch) for batch in batches}
    optimizer = optax.adamw(3e-4, b1=0.9, b2=0.95, weight_decay=0.1)
    opt_state = optimizer.init(eqx.filter(model, trainable_filter(model)))
    train_step, _ = make_steps(optimizer)
    result = {
        "arm": name,
        "model": model_config.to_dict(),
        "parameter_counts": parameter_counts(model),
        "accounting_version": ACCOUNTING_VERSION,
        "batches": {},
        "stopped_reason": None,
    }
    for batch in batches:
        x, y = _batch(model_config, batch, seed=seed + batch)
        try:
            compile_t0 = time.perf_counter()
            model, opt_state, value = train_step(model, opt_state, x, y)
            compile_s = time.perf_counter() - compile_t0
            nll = float(value)
            if not math_isfinite(nll):
                raise ValueError("nonfinite loss at compile step")
            reuse_t0 = time.perf_counter()
            model, opt_state, value = train_step(model, opt_state, x, y)
            reuse_s = time.perf_counter() - reuse_t0
            model, opt_state, warmup_times, nll = _time_steps(
                train_step, model, opt_state, x, y, warmup)
            model, opt_state, timed_times, nll = _time_steps(
                train_step, model, opt_state, x, y, timed)
        except Exception as exc:
            oom = _is_oom(exc)
            result["batches"][str(batch)] = {
                "status": "oom" if oom else "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=8),
                "device_memory": _device_memory(),
            }
            result["stopped_reason"] = f"batch {batch} {'oom' if oom else 'failed'}"
            break
        tokens = counts_by_batch[batch]["tokens_per_step"]
        mean_s = float(np.mean(timed_times))
        result["batches"][str(batch)] = {
            "status": "ok",
            "compile_seconds": compile_s,
            "cached_compile_seconds": reuse_s,
            "warmup_seconds": warmup_times,
            "timed_seconds": timed_times,
            "mean_step_seconds": mean_s,
            "tokens_per_second": tokens / mean_s if mean_s else None,
            "semantic_training_per_step": counts_by_batch[batch]["semantic_training_per_step"],
            "semantic_flops_per_second_proxy": (
                counts_by_batch[batch]["semantic_training_per_step"] / mean_s if mean_s else None),
            "final_train_nll": nll,
            "device_memory": _device_memory(),
        }
    return result


def run_profile(*, out, batches, warmup, timed, seed, model=None, arms=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = DEFAULT_MODEL | (model or {})
    u12 = base | PRIMARY_ARMS["U12"]
    r6 = base | PRIMARY_ARMS["R6x2"]
    validate_primary_pair(u12, r6, batch_size=batches[0])
    selected = list(arms or PRIMARY_ARMS)
    unknown = [name for name in selected if name not in PRIMARY_ARMS]
    if unknown:
        raise ValueError(f"unknown arms: {unknown}")
    existing = out / "profile.json"
    if existing.is_file() and set(selected) != set(PRIMARY_ARMS):
        record = json.loads(existing.read_text())
        record.setdefault("arms", {})
        record["requested"] = {
            "batches": batches, "warmup": warmup, "timed": timed, "seed": seed, "arms": selected,
        }
    else:
        record = {
            "profile_version": PROFILE_VERSION,
            "purpose": "environment/cost measurement; not H1 evidence",
            "started_unix": time.time(),
            "environment": environment_identity(),
            "source": source_identity(),
            "nvidia_smi": _nvidia_smi(),
            "jax_devices": [str(d) for d in jax.devices()],
            "device_memory_before": _device_memory(),
            "requested": {
                "batches": batches, "warmup": warmup, "timed": timed, "seed": seed, "arms": selected,
            },
            "os": {"platform": platform.platform(), "python": platform.python_version(),
                   "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "")},
            "arms": {},
        }
        atomic_json(out / "environment.json", record)
    configs = {"U12": u12, "R6x2": r6}
    for name in selected:
        record["arms"][name] = profile_arm(name, configs[name], batches=batches, warmup=warmup,
                                           timed=timed, seed=seed)
        atomic_json(out / "profile.json", record)
    record["finished_unix"] = time.time()
    record["elapsed_seconds"] = record["finished_unix"] - record.get("started_unix", record["finished_unix"])
    atomic_json(out / "profile.json", record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timed", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tiny", action="store_true",
                        help="tiny CPU fixture; not an H100 measurement")
    parser.add_argument("--arm", choices=sorted(PRIMARY_ARMS), action="append",
                        help="run only these arms; may be repeated. Default: both.")
    args = parser.parse_args(argv)
    if min(args.batches) < 1 or args.warmup < 1 or args.timed < 1:
        parser.error("batches/warmup/timed must be positive")
    model = None
    if args.tiny:
        model = dict(vocab=16, sequence=8, n_embed=8, aft_heads=2, aft_ksize=2,
                     linear_fan_in=4)
        if args.batches == [1, 2, 4, 8]:
            args.batches = [1, 2]
        if args.warmup == 3:
            args.warmup = 1
        if args.timed == 8:
            args.timed = 2
    record = run_profile(out=args.out, batches=args.batches, warmup=args.warmup,
                         timed=args.timed, seed=args.seed, model=model, arms=args.arm)
    print(json.dumps({"out": args.out, "elapsed_seconds": record["elapsed_seconds"],
                      "arms": {name: arm.get("stopped_reason") for name, arm in record["arms"].items()}},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
