"""Paired non-inferiority analysis and deterministic metrics-to-figure CLI.

Input JSON list (or JSONL) contains ONE final-checkpoint result per run:
  {"run_id": "...", "status": "completed", "nll": 3.2,
   "checkpoint_selection": "final_budget", "config_sha256": "..."}
Failed/missing runs are retained in the expected manifest, never replaced.
Repeated checkpoints/tokens are not seed replicates. The CLI requires an
explicit expected manifest, preventing absent seeds from silently vanishing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def paired_interval(differences, *, alpha: float = 0.05, comparisons: int = 3) -> dict:
    """Two-sided paired Student-t CI with Bonferroni simultaneous coverage.

    A zero sample variance produces a zero-width interval (not infinite power
    evidence). At least two independent training pairs are required. Normality
    of seed-level gaps is an assumption, especially fragile at small n.
    """
    if not 0 < alpha < 1 or isinstance(comparisons, bool) or not isinstance(comparisons, int) or comparisons < 1:
        raise ValueError("invalid alpha or comparison count")
    values = [float(value) for value in differences]
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("at least two finite paired differences are required")
    from scipy.stats import t
    n = len(values)
    mean, sd = statistics.mean(values), statistics.stdev(values)
    critical = float(t.ppf(1-alpha/(2*comparisons), n-1))
    half = critical * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "lower": mean-half, "upper": mean+half,
            "critical_t": critical, "confidence": 1-alpha/comparisons,
            "differences": values}


def analyze_records(records: list[dict], manifest: dict, *, delta: float = 0.01,
                    alpha: float = 0.05) -> dict:
    """Analyze exactly manifest U12/R6x2 pairs; incomplete study => inconclusive.

    `decision` is a mathematical classification, not a scientific seal. Draft
    inputs produce `evidence_status=exploratory_not_preregistered`. An immutable
    preregistration audit is deliberately NOT inferred from a status string.
    Supplying extra runs, duplicate results, changed margins or non-final
    checkpoint records is an error, not an automatic best-checkpoint selection.
    """
    if not math.isfinite(delta) or delta <= 0 or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("invalid delta/alpha")
    if manifest.get("delta", delta) != delta or manifest.get("alpha", alpha) != alpha:
        raise ValueError("analysis margin/alpha differs from expected manifest")
    budgets = manifest.get("budgets", [])
    pairs = manifest.get("pairs", [])
    if len(budgets) != 3 or len(set(budgets)) != 3 or not all(math.isfinite(b) and b > 0 for b in budgets):
        raise ValueError("registered procedure requires exactly three distinct positive budgets")
    if len(pairs) < 2 or len(set(pairs)) != len(pairs):
        raise ValueError("manifest must declare at least two distinct expected pair IDs")
    expected, cells = {}, {}
    for run in manifest.get("runs", []):
        run_id = run["run_id"]
        cell = (run["budget"], run["pair_id"], run["arm"])
        if run_id in expected or cell in cells:
            raise ValueError("duplicate expected run identity or matrix cell")
        if cell[0] not in budgets or cell[1] not in pairs or cell[2] not in ("U12", "R6x2"):
            raise ValueError("unexpected budget, pair or arm in primary manifest")
        expected[run_id], cells[cell] = run, run_id
    if len(cells) != 6 * len(pairs):
        raise ValueError("expected manifest lacks required controls/seed pairs")
    observed = {}
    for row in records:
        run_id = row["run_id"]
        if run_id not in expected:
            raise ValueError(f"unexpected result {run_id}")
        if run_id in observed:
            raise ValueError(f"duplicate result {run_id}; do not select among checkpoints")
        exp = expected[run_id]
        for key in ("arm", "pair_id", "budget"):
            if key in row and row[key] != exp[key]:
                raise ValueError(f"result {key} mismatch for {run_id}")
        if "config_sha256" in exp and row.get("config_sha256") != exp["config_sha256"]:
            raise ValueError(f"config identity mismatch for {run_id}")
        if row.get("status") not in ("completed", "model_failure", "infrastructure_failure", "incomplete"):
            raise ValueError(f"unknown result status for {run_id}")
        if row["status"] == "completed":
            if row.get("checkpoint_selection") != "final_budget":
                raise ValueError("primary results must use final_budget checkpoint selection")
            nll = row.get("nll")
            if isinstance(nll, bool) or not isinstance(nll, (int, float)) or not math.isfinite(nll) or nll < 0:
                raise ValueError("completed result requires finite nonnegative nll; report divergence as model_failure")
        observed[run_id] = row
    missing = sorted(set(expected)-set(observed))
    failures = [{"run_id": run_id, "status": row["status"]} for run_id, row in observed.items()
                if row["status"] != "completed"]
    complete = not missing and not failures
    results = []
    for budget in sorted(budgets):
        differences, completed_pairs = [], []
        for pair in pairs:
            rows = [observed.get(cells[(budget, pair, arm)]) for arm in ("U12", "R6x2")]
            if all(row is not None and row["status"] == "completed" for row in rows):
                differences.append(rows[1]["nll"]-rows[0]["nll"])
                completed_pairs.append(pair)
        interval = paired_interval(differences, alpha=alpha, comparisons=3) if len(differences) >= 2 else None
        results.append({"budget": budget, "expected_pairs": len(pairs),
                        "completed_pairs": completed_pairs, "interval": interval,
                        "subset_only": len(completed_pairs) != len(pairs)})
    decision = "inconclusive"
    # Incomplete required replication cannot establish the registered all-budget claim.
    if complete:
        if all(result["interval"]["upper"] < delta for result in results):
            decision = "supported"
        elif any(result["interval"]["lower"] > delta for result in results):
            decision = "contradicted"
    return {
        "decision": decision, "complete": complete, "delta": delta, "alpha": alpha,
        "evidence_status": "exploratory_not_preregistered" if manifest.get("status") == "draft_not_preregistered"
                           else "preregistration_requires_external_audit",
        "method": "paired Student-t, two-sided Bonferroni over three budgets",
        "estimand": "mean recurrent-minus-untied NLL over training randomness on the fixed evaluation corpus",
        "missing_run_ids": missing, "failures": failures, "budgets": results,
        "limitations": ["Seed-level gap normality assumption; small n may give unreliable coverage",
                        "Fixed corpus; intervals do not quantify corpus sampling uncertainty",
                        "A manifest/status flag does not prove preregistration or holdout independence",
                        "Intervals from incomplete pairs are descriptive only; no success-only replacement"],
    }


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise ValueError("metrics JSON must be a list of final run records")
    return value


def plot_analysis(report: dict, output: str | Path) -> None:
    """Main gap figure only; never fabricate energy/storage curves from NLL."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4))
    points = [row for row in report["budgets"] if row["interval"] is not None]
    for row in points:
        ci = row["interval"]
        ax.errorbar(row["budget"], ci["mean"],
                    yerr=[[ci["mean"]-ci["lower"]], [ci["upper"]-ci["mean"]]],
                    fmt="o", color="gray" if row["subset_only"] else "C0", capsize=4)
    ax.axhline(report["delta"], color="C3", linestyle="--", label=f"margin = {report['delta']:g} nats")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Semantic training budget (estimated proxy units)")
    ax.set_ylabel("NLL(R6×2) − NLL(U12), nats/token")
    ax.set_title(f"{report['decision']} — {report['evidence_status'].replace('_', ' ')}", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(Path(args.manifest).read_text())
        report = analyze_records(load_records(args.metrics), manifest,
                                 delta=manifest.get("delta", 0.01), alpha=manifest.get("alpha", 0.05))
        output = Path(args.out)
        output.mkdir(parents=True, exist_ok=False)
        (output/"analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        if not args.no_plot:
            plot_analysis(report, output/"primary_gap.png")
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps({"decision": report["decision"], "complete": report["complete"],
                      "evidence_status": report["evidence_status"]}))


if __name__ == "__main__":
    main()
