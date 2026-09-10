"""Conditional whole-slab inference scheduling, NOT an optimal cache bound.

One autoregressive lockstep round delivers B tokens (one per active sequence).
Slabs replace the entire block store in fixed order. U12 at capacity six loads
1..6, then 7..12 every round. A resident six-block recurrent core loads once.
A generic six-slot cache can retain partial slabs and have different traffic:
this module deliberately does not optimize retention, ordering or batching.

Actual software weight/index bytes are payload proxies, not TSU encodings.
Final norm is separately resident and excluded from slab traffic/capacity.
Embedding/head, per-applied-depth histories, workspace and activation traffic
are excluded, NOT free. Cached decoding and hardware fidelity remain untested.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real

import equinox as eqx
import jax

from rz1t.conventions import trainable_filter
from rz1t.model import RecurrentZ1T


def _positive_int(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative(name, value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class BlockPayload:
    """One unique block's actual stored trainable and fixed-index bytes."""
    weight_bytes: int
    index_bytes: int

    def __post_init__(self):
        _positive_int("weight_bytes", self.weight_bytes)
        if type(self.index_bytes) is not int or self.index_bytes < 0:
            raise ValueError("index_bytes must be a nonnegative integer")

    @property
    def total_bytes(self):
        return self.weight_bytes + self.index_bytes


def _bytes(tree):
    return sum(int(x.size * x.dtype.itemsize) for x in jax.tree.leaves(tree) if eqx.is_array(x))


def slab_schedule(blocks, *, k: int, capacity_blocks: int = 6,
                  batch_size: int = 1, rounds: int = 100) -> dict:
    """Explicit replacement schedule for an untied stack or fully resident core.

    One load event programs ONE slab (up to capacity_blocks), not one block.
    Resident core: k applications per round, cold load once, steady zero loads.
    Untied multi-slab stack: every round loads every slab, including round one.
    No double-buffer capacity is assumed. Any modeled overlap needs independent
    justification. Norm startup/programming is outside these body counts.
    """
    for name, value in (("k", k), ("capacity_blocks", capacity_blocks),
                        ("batch_size", batch_size), ("rounds", rounds)):
        _positive_int(name, value)
    blocks = tuple(blocks)
    if not blocks or any(not isinstance(b, BlockPayload) for b in blocks):
        raise ValueError("blocks must be a nonempty sequence of BlockPayload")
    if k > 1 and len(blocks) > capacity_blocks:
        raise ValueError("recurrent unique core must fit entirely in capacity")
    slabs = [list(range(i + 1, min(i + capacity_blocks, len(blocks)) + 1))
             for i in range(0, len(blocks), capacity_blocks)]
    resident = len(slabs) == 1
    cold_weights = sum(b.weight_bytes for b in blocks)
    cold_indices = sum(b.index_bytes for b in blocks)

    def traffic(multiplier):
        weights, indices = cold_weights * multiplier, cold_indices * multiplier
        events = len(slabs) * multiplier
        return {"weight_bytes": weights, "index_bytes": indices,
                "body_bytes": weights + indices, "load_events": events,
                "blocks_loaded": len(blocks) * multiplier}

    def per_token(counts, tokens):
        return {name: value / tokens for name, value in counts.items()}

    cold = traffic(1)
    steady = traffic(0 if resident else 1)
    total = traffic(1 if resident else rounds)
    return {
        "status": "hypothetical_whole_slab_schedule_not_cache_optimum",
        "capacity_blocks": capacity_blocks, "unique_blocks": len(blocks),
        "applied_blocks": len(blocks) * k, "batch_size": batch_size,
        "rounds": rounds, "delivered_tokens": batch_size * rounds,
        "core_applications_per_round": k, "resident": resident,
        "slabs_one_based": slabs,
        "per_block_payloads": [asdict(b) for b in blocks],
        "initial_round": cold, "steady_round": steady, "finite_total": total,
        "initial_per_token": per_token(cold, batch_size),
        "steady_per_token": per_token(steady, batch_size),
        "finite_average_per_token": per_token(total, batch_size * rounds),
        "event_definition": "one complete slab programming event",
        "caveat": "Whole-slab replacement is not a six-slot cache traffic lower bound. "
                  "No head, norm, activation/history traffic, or hardware encoding included.",
    }


def residency_report(model, *, capacity_blocks=6, batch_size=1, rounds=100):
    """Extract byte counts from the optimizer partition of the actual model."""
    if not isinstance(model, RecurrentZ1T):
        raise ValueError("requires a RecurrentZ1T model")
    c = model.config
    if c.p or c.q or c.injection != "none":
        raise ValueError("supported schedule requires p=q=0 and injection=none")
    params = eqx.filter(model, trainable_filter(model))
    blocks = []
    for block, trainable in zip(model.core, params.core, strict=True):
        indices = eqx.filter(block, lambda x: eqx.is_array(x) and not eqx.is_inexact_array(x))
        blocks.append(BlockPayload(_bytes(trainable), _bytes(indices)))
    report = slab_schedule(blocks, k=c.k, capacity_blocks=capacity_blocks,
                           batch_size=batch_size, rounds=rounds)
    report["shared_resident_norm_bytes_excluded"] = _bytes(model.norm)
    report["embedding_classifier_bytes_excluded"] = _bytes((model.embedding, model.clf))
    report["payload_precision"] = "actual model array dtypes; no assumed quantization/TSU encoding"
    return report


@dataclass(frozen=True)
class ReloadScenario:
    """All coefficients must be supplied explicitly; none are measurements.

    common_compute_seconds/joules_per_round cover all non-reload work, including
    head, sampling, control and state transfers in the chosen scope. For a
    matched-depth R6x2/U12 comparison they may be held equal as an ASSUMPTION;
    use arm-specific values if work/utilization differs (especially U6).
    Compute joules are dynamic only: static watts are charged over total time.
    exposed_fraction multiplies transfer + programming TIME, including cold
    startup; it never discounts dynamic transfer/programming energy. Fraction
    zero presumes all loading is hidden behind other work, not free transfers.
    """
    effective_bandwidth_bytes_per_second: float
    programming_seconds_per_load_event: float
    exposed_fraction: float
    dynamic_joules_per_byte: float
    programming_joules_per_load_event: float
    common_compute_seconds_per_round: float
    common_compute_joules_per_round: float
    static_watts: float

    def __post_init__(self):
        for name, value in asdict(self).items():
            _nonnegative(name, value)
        if self.effective_bandwidth_bytes_per_second <= 0:
            raise ValueError("effective bandwidth must be positive")
        if self.common_compute_seconds_per_round <= 0:
            raise ValueError("common compute seconds must be positive")
        if self.exposed_fraction > 1:
            raise ValueError("exposed_fraction must lie in [0, 1]")


def conditional_cost(report: dict, scenario: ReloadScenario) -> dict:
    """Cost a schedule identically for every arm, in explicit SI units.

    T = rounds*C + exposed*(bytes/bandwidth + events*program_seconds)
    E = rounds*E_compute + bytes*J_per_byte + events*J_program + watts*T.
    Uses whole rounds, not per-request latency multiplied by batch size.
    """
    if not isinstance(scenario, ReloadScenario):
        raise ValueError("scenario must be ReloadScenario")
    for name in ("batch_size", "rounds"):
        _positive_int(name, report[name])

    def cost(counts, rounds):
        byte_count, events = counts["body_bytes"], counts["load_events"]
        for name, value in (("body_bytes", byte_count), ("load_events", events)):
            _nonnegative(name, value)
        transfer = byte_count / scenario.effective_bandwidth_bytes_per_second
        programming = events * scenario.programming_seconds_per_load_event
        exposed = scenario.exposed_fraction * (transfer + programming)
        compute = rounds * scenario.common_compute_seconds_per_round
        seconds = compute + exposed
        terms = {
            "common_compute": rounds * scenario.common_compute_joules_per_round,
            "transfer": byte_count * scenario.dynamic_joules_per_byte,
            "programming": events * scenario.programming_joules_per_load_event,
            "static": seconds * scenario.static_watts,
        }
        joules = sum(terms.values())
        tokens = rounds * report["batch_size"]
        values = {"unoverlapped_transfer_seconds": transfer,
                  "unoverlapped_programming_seconds": programming,
                  "exposed_reload_seconds": exposed, "compute_seconds": compute,
                  "total_seconds": seconds, "total_joules": joules,
                  "amortized_seconds_per_token": seconds / tokens,
                  "joules_per_token": joules / tokens,
                  "delivered_tokens_per_second": tokens / seconds}
        for name, value in values.items():
            _nonnegative(name, value)
        values["energy_terms_joules"] = terms
        return values

    return {"status": "conditional_not_measured_energy_or_decode_performance",
            "scenario": asdict(scenario),
            "initial_round": cost(report["initial_round"], 1),
            "steady_round": cost(report["steady_round"], 1),
            "finite_total": cost(report["finite_total"], report["rounds"])}


def main(argv=None):
    """CPU-only config -> JSON traffic reports; no hardware coefficients chosen."""
    import argparse
    import json
    from pathlib import Path
    import yaml
    from rz1t.config import ModelConfig
    from rz1t.model import create_model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--capacity-blocks", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=100)
    args = parser.parse_args(argv)
    jax.config.update("jax_platforms", "cpu")
    reports = []
    with jax.default_device(jax.devices("cpu")[0]):
        for path in args.configs:
            raw = yaml.safe_load(path.read_text())
            model = create_model(ModelConfig.from_dict(raw["model"]), jax.random.key(0))
            reports.append({"config": str(path), "arm": raw.get("arm"),
                            "report": residency_report(model, capacity_blocks=args.capacity_blocks,
                                                       batch_size=args.batch_size, rounds=args.rounds)})
    print(json.dumps(reports, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
