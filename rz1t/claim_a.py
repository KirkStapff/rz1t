"""Claim-A confirmation-holdout estimate. Not the original H1 matrix."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from rz1t.analyze import paired_interval

ARMS = ("U6", "R6x2", "U12")
SEEDS = (7, 8, 9, 10, 11, 12)
PROTOCOL = "claim-a-v1"
SLUGS = (("U6", "u6"), ("R6x2", "r6x2"), ("U12", "u12"))
RUNS = [(arm, seed) for seed in SEEDS for arm in ARMS]
FIRST = [(arm, seed) for seed in (7, 8, 9) for arm in ARMS]
REMAINING = [(arm, seed) for seed in (10, 11, 12) for arm in ARMS]
RECOVERY = [("U12", 12)]


def expected_run_id(seed, arm):
    return f"claim-a-v1-s{seed}-{arm}"


def analyze_claim_a(records, manifest):
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("manifest protocol mismatch")
    if manifest.get("evaluation_split") != "confirmation":
        raise ValueError("claim-a-v1 requires unused confirmation holdout")
    if manifest.get("seeds") != list(SEEDS) or manifest.get("arms") != list(ARMS):
        raise ValueError("seed/arm matrix mismatch")
    if manifest.get("h1_authorized") or manifest.get("confirmation_authorized"):
        raise ValueError("claim-a must not authorize the original H1 matrix")
    expected = {run["run_id"]: run for run in manifest["runs"]}
    if set(expected) != {expected_run_id(s, a) for s in SEEDS for a in ARMS}:
        raise ValueError("expected run identities drifted")
    observed = {}
    for row in records:
        run_id = row["run_id"]
        if run_id not in expected:
            raise ValueError(f"unexpected result {run_id}")
        if run_id in observed:
            raise ValueError("duplicate result; no checkpoint selection")
        exp = expected[run_id]
        if row.get("arm") != exp["arm"] or row.get("pair_id") != exp["pair_id"]:
            raise ValueError("arm/pair identity mismatch")
        if row.get("config_sha256") is not None and row.get("config_sha256") != exp["config_sha256"]:
            raise ValueError("config identity mismatch")
        if row.get("evaluation_split") not in (None, "confirmation"):
            raise ValueError("holdout mismatch")
        if row.get("status") == "completed":
            if row.get("checkpoint_selection") != "final_budget":
                raise ValueError("final-checkpoint selection required")
            nll = row.get("nll")
            if isinstance(nll, bool) or not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0:
                raise ValueError("completed result requires finite nonnegative nll")
        observed[run_id] = row
    missing = sorted(set(expected) - set(observed))
    failures = [{"run_id": rid, "status": observed[rid]["status"]}
                for rid in expected if rid in observed and observed[rid]["status"] != "completed"]
    complete = not missing and not failures
    table = []
    for seed in SEEDS:
        row = {"seed": seed}
        for arm in ARMS:
            rec = observed.get(expected_run_id(seed, arm))
            row[arm] = None if rec is None or rec["status"] != "completed" else rec["nll"]
        if all(row[arm] is not None for arm in ARMS):
            row["r_minus_u12"] = row["R6x2"] - row["U12"]
            row["r_minus_u6"] = row["R6x2"] - row["U6"]
            row["u6_minus_u12"] = row["U6"] - row["U12"]
            row["fraction_recovered"] = ((row["U6"] - row["R6x2"]) / (row["U6"] - row["U12"])
                                         if row["U6"] != row["U12"] else None)
        table.append(row)
    complete_rows = [row for row in table if "r_minus_u12" in row]
    contrasts = {}
    if len(complete_rows) >= 2:
        contrasts["r_minus_u12"] = paired_interval(
            [row["r_minus_u12"] for row in complete_rows], alpha=0.05, comparisons=2)
        contrasts["r_minus_u6"] = paired_interval(
            [row["r_minus_u6"] for row in complete_rows], alpha=0.05, comparisons=2)
    means = None
    if len(complete_rows) == len(SEEDS):
        means = {arm: statistics.mean(row[arm] for row in complete_rows) for arm in ARMS}
        means["r_vs_u12_ppl_increase_pct"] = 100 * math.expm1(means["R6x2"] - means["U12"])
        means["r_vs_u6_ppl_reduction_pct"] = 100 * (-math.expm1(means["R6x2"] - means["U6"]))
    return {
        "protocol": PROTOCOL,
        "evidence_status": "claim_a_confirmation_holdout_not_h1",
        "complete": complete,
        "missing_run_ids": missing,
        "failures": failures,
        "seeds": table,
        "contrasts": contrasts,
        "means": means,
        "n_complete_triples": len(complete_rows),
        "limitations": [
            "n=6 is a tighter estimate than C7, not a powered H1 non-inferiority test",
            "Same 40k-document subset; confirmation documents were unused for training and prior evaluation",
            "Do not pool C7 calibration numbers into these intervals",
            "Do not change delta after seeing results; do not relabel as original H1",
        ],
    }


def write_manifest(path):
    from rz1t.train import canonical_hash
    import yaml
    root = Path(__file__).resolve().parents[1]
    runs = []
    for seed in SEEDS:
        for arm, slug in SLUGS:
            cfg_path = root / "configs/claim-a" / f"claim-a-v1-{slug}-s{seed}.yaml"
            cfg = yaml.safe_load(cfg_path.read_text())
            runs.append({
                "run_id": cfg["run_id"], "arm": arm, "pair_id": cfg["pair_id"],
                "seed": seed, "config_path": str(cfg_path.relative_to(root)),
                "config_sha256": canonical_hash(cfg),
            })
    manifest = {
        "protocol": PROTOCOL, "evaluation_split": "confirmation",
        "seeds": list(SEEDS), "arms": list(ARMS),
        "steps": 15513, "tokens": 254164992,
        "confirmation_authorized": False, "h1_authorized": False,
        "runs": runs,
    }
    Path(path).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
