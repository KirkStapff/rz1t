"""Generate reproducible DRAFT calibration/explore configs, never a science seal.

Usage: python -m rz1t.make_sweep --out manifests/calibration-v1 --lane calibration
Outputs JSON (valid YAML) configs and manifest.json. OWT data/source hashes,
LR selection, measured cost, power and preregistration are intentionally
unresolved. Do not interpret generated files as permission to train on a
confirmation holdout or spend money. Sealed production generation is blocked.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path

from rz1t.config import ModelConfig
from rz1t.flops import ACCOUNTING_VERSION, accounting, budget_to_steps

DEFAULT_BUDGETS = (10**16, 3*10**16, 10**17)
DEFAULT_MODEL = dict(vocab=50257, sequence=256, n_embed=384, p=0, m=6, k=2, q=0,
                     aft_heads=8, aft_ksize=4, linear_fan_in=4, tanh_linear=True,
                     tanh_mlp=True, dyt_alpha=0.5, remat=False, injection="none")


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _seed(namespace: str, base_seed: int, pair: int, stream: str) -> int:
    # Independent deterministic seed derivation, not adjacent generator draws.
    # Same pair's two arms share data/init seeds; architectures allocate keys
    # differently, so this does not imply equal initial functions or weights.
    digest = canonical_hash(["rz1t-seeds-v1", namespace, base_seed, pair, stream])
    return int(digest[:8], 16)


def validate_primary_pair(untied, recurrent, batch_size: int = 1) -> None:
    """Validate architecture and semantic-work match; never infer energy match."""
    u = asdict(untied) if is_dataclass(untied) else dict(untied)
    r = asdict(recurrent) if is_dataclass(recurrent) else dict(recurrent)
    for cfg, blocks, loops in ((u, 12, 1), (r, 6, 2)):
        if (cfg.get("p", 0), cfg.get("m"), cfg.get("k"), cfg.get("q", 0)) != (0, blocks, loops, 0):
            raise ValueError("primary requires exact U12 / R6x2 with p=q=0")
        if cfg.get("injection", "none") != "none" or cfg.get("linear_fan_in") != 4:
            raise ValueError("primary requires no injection and fan-in 4")
    # Compare every supplied config key except the two intended interventions.
    if {key: val for key, val in u.items() if key not in ("m", "k")} != {
            key: val for key, val in r.items() if key not in ("m", "k")}:
        raise ValueError("primary pair differs outside m,k")
    au, ar = accounting(u, batch_size), accounting(r, batch_size)
    if au["applied_depth"] != 12 or ar["applied_depth"] != 12:
        raise ValueError("primary applied depth must equal 12")
    if au["semantic_training_per_step"] != ar["semantic_training_per_step"]:
        raise ValueError("primary semantic step costs differ")


def generate_sweep(*, lane: str = "calibration", pairs: int = 4,
                   budgets=DEFAULT_BUDGETS, batch_size: int = 8,
                   base_seed: int = 2026, model: dict | None = None) -> dict:
    """Pure draft generator. All configs/run IDs are deterministic.

    Pair streams are shared across arms and budgets; distinct pair IDs use
    separate hash-derived streams. All evaluations across budgets for one pair
    remain dependent and may not be reinterpreted as independent seeds.
    Different lanes derive different streams. Collision detection is explicit.
    """
    if lane not in ("calibration", "explore"):
        raise ValueError("sealed/confirmation generation blocked: preregistration evidence and pilot gates unimplemented")
    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 2:
        raise ValueError("pairs must be an integer >= 2")
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    budgets = list(budgets)
    if len(budgets) != 3 or len(set(budgets)) != 3:
        raise ValueError("draft primary matrix requires three distinct budgets")
    cfg = ModelConfig.from_dict(DEFAULT_MODEL | (model or {})).to_dict()
    arms = {"U12": cfg | {"p": 0, "m": 12, "k": 1, "q": 0},
            "R6x2": cfg | {"p": 0, "m": 6, "k": 2, "q": 0}}
    validate_primary_pair(arms["U12"], arms["R6x2"], batch_size)
    seeds, used = {}, set()
    for pair in range(pairs):
        streams = {name: _seed(lane, base_seed, pair, name) for name in ("init", "data", "loop", "eval")}
        if len(set(streams.values())) != len(streams) or used.intersection(streams.values()):
            raise ValueError("seed collision; choose another base_seed")
        used.update(streams.values())
        seeds[f"{lane}-pair-{pair:03d}"] = streams
    runs, ids = [], set()
    for budget in sorted(budgets):
        for pair_id, streams in seeds.items():
            for arm, model_cfg in arms.items():
                counts = accounting(model_cfg, batch_size)
                schedule = budget_to_steps(budget, counts["semantic_training_per_step"])
                run_config = {
                    "lane": lane, "draft": True, "arm": arm, "pair_id": pair_id,
                    "seed": streams["init"], "seeds": streams,
                    "model": model_cfg,
                    "training": {"batch_size": batch_size, "steps": schedule["steps"],
                                 "semantic_budget": budget, "max_budget_overshoot": 0.01,
                                 "schedule_basis": "cumulative_semantic_budget"},
                    "optimizer": {"lr": 0.0003, "weight_decay": 0.1, "b1": 0.9,
                                  "b2": 0.95, "warmup": 100, "grad_clip": 1.0,
                                  "final_lr_ratio": 0.1},
                    "data": {"dataset_name": "openwebtext", "manifest": None,
                             "evaluation_split": lane if lane == "calibration" else "development"},
                    "precision": "float32", "accounting_version": ACCOUNTING_VERSION,
                }
                digest = canonical_hash(run_config)
                run_id = f"{lane}-{arm}-{digest[:16]}"
                if run_id in ids:
                    raise ValueError("duplicate run identity")
                ids.add(run_id)
                runs.append({"run_id": run_id, "arm": arm, "pair_id": pair_id,
                             "budget": budget, "config_sha256": digest,
                             "config": run_config, "budget_schedule": schedule,
                             "semantic_training_per_step": counts["semantic_training_per_step"]})
    return {
        "schema_version": 1, "status": "draft_not_preregistered", "lane": lane,
        "confirmation_authorized": False, "delta": 0.01, "alpha": 0.05,
        "budgets": sorted(budgets), "pairs": list(seeds), "seed_streams": seeds,
        "seed_mapping": "Same init/data/loop/eval streams across arms and budgets within pair; distinct hash-derived streams between pairs and lanes. Different model key allocation is NOT equal initialization.",
        "accounting_version": ACCOUNTING_VERSION,
        "required_before_confirmation": ["pinned data/source/environment hashes", "measured cost and reserve admission",
                                         "pilot-informed power and replication", "justified margin", "reviewed immutable preregistration"],
        "assumptions": ["float32; full BPTT; final-loop supervision; fixed k",
                        "equal applied depth is not equal measured energy", "semantic counting unvalidated against XLA"],
        "runs": runs,
    }


def write_sweep(manifest: dict, out: str | Path) -> Path:
    """Write a fresh directory only; never replace an existing experiment."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "configs").mkdir()
    for run in manifest["runs"]:
        (out / "configs" / f"{run['run_id']}.json").write_text(
            json.dumps(run["config"], indent=2, allow_nan=False) + "\n")
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (out / "manifest.sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lane", choices=("calibration", "explore", "sealed"), default="calibration")
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=2026)
    args = parser.parse_args(argv)
    try:
        manifest = generate_sweep(lane=args.lane, pairs=args.pairs,
                                  batch_size=args.batch_size, base_seed=args.base_seed)
        path = write_sweep(manifest, args.out)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(path)


if __name__ == "__main__":
    main()
