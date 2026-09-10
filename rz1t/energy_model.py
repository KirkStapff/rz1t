"""Conditional hardware sensitivity only; reference-energy audit is unresolved.

There are deliberately NO default Z1/FPGA energy constants and no absolute
nJ/token headline API. Callers must supply counts and a shared hypothetical
scenario for both arms. Every required operation/cache/control cost remains.
Software arithmetic counts cannot establish quantized sampling fidelity.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class WorkCounts:
    """Per delivered token, with traffic/sample counts supplied independently.

    `arithmetic_units` must use the same definition in both arms. Counts of
    semantic FLOPs are not pbit sample counts. Weight bytes distinguish loaded
    bytes from unique storage; an untied model may also be weight-resident.
    """
    arithmetic_units: float
    samples: float = 0
    weight_reload_bytes: float = 0
    state_transfer_bytes: float = 0
    control_operations: float = 0
    retained_state_byte_seconds: float = 0

    def __post_init__(self):
        _validate_nonnegative(asdict(self))


@dataclass(frozen=True)
class Scenario:
    """User-supplied hypothetical coefficients in joules and SI units.

    Static power is divided by *delivered throughput*, not multiplied by
    per-request latency without a concurrency denominator. No coefficients
    are supplied from the unaudited post. Supply every coefficient explicitly,
    including zeros/exclusions; run sensitivity over plausible ranges.
    """
    joules_per_arithmetic_unit: float
    joules_per_sample: float
    joules_per_reload_byte: float
    joules_per_state_byte: float
    joules_per_control_operation: float
    watts_per_retained_byte: float
    static_watts: float
    delivered_tokens_per_second: float

    def __post_init__(self):
        _validate_nonnegative(asdict(self))
        if self.delivered_tokens_per_second <= 0:
            raise ValueError("delivered throughput must be positive")


def _validate_nonnegative(values):
    for name, value in values.items():
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def _conditional_terms(work: WorkCounts, scenario: Scenario) -> dict[str, float]:
    return {
        "arithmetic": work.arithmetic_units * scenario.joules_per_arithmetic_unit,
        "sampling": work.samples * scenario.joules_per_sample,
        "weight_reload": work.weight_reload_bytes * scenario.joules_per_reload_byte,
        "state_transfer": work.state_transfer_bytes * scenario.joules_per_state_byte,
        "control": work.control_operations * scenario.joules_per_control_operation,
        "state_retention": work.retained_state_byte_seconds * scenario.watts_per_retained_byte,
        "static": scenario.static_watts / scenario.delivered_tokens_per_second,
    }


def compare_scenario(untied: WorkCounts, recurrent: WorkCounts, scenario: Scenario) -> dict:
    """Dimensionless ratio and term deltas under ONE scenario applied to both.

    This is sensitivity, not a measured energy result. Throughput is held equal
    in this scenario; unequal-throughput studies need independent measured or
    justified utilization inputs, not an automatic recurrent speedup.
    """
    u, r = _conditional_terms(untied, scenario), _conditional_terms(recurrent, scenario)
    total_u, total_r = sum(u.values()), sum(r.values())
    return {
        "status": "unvalidated_conditional_sensitivity",
        "reference_audit_passed": False,
        "ratio_recurrent_over_untied": total_r / total_u if total_u else None,
        "term_deltas_joules_per_token_conditional": {key: r[key]-u[key] for key in u},
        "scenario": asdict(scenario),
        "untied_counts": asdict(untied), "recurrent_counts": asdict(recurrent),
        "warning": "Not measured energy; requires sampling/precision fidelity and mapping validation",
    }


def reload_break_even(untied: WorkCounts, recurrent: WorkCounts,
                      scenario: Scenario) -> dict:
    """Solve delta_other + delta_reload_bytes * coefficient < 0.

    The coefficient is J/byte; all other scenario terms are fixed. Preserve
    negative thresholds and direction: some regions have no nonnegative
    coefficient where recurrence wins. No assumption that tying saves reloads.
    """
    u, r = _conditional_terms(untied, scenario), _conditional_terms(recurrent, scenario)
    other = sum(r[key]-u[key] for key in u if key != "weight_reload")
    slope = recurrent.weight_reload_bytes - untied.weight_reload_bytes
    if slope == 0:
        return {"status": "unvalidated_conditional_sensitivity", "threshold_joules_per_byte": None,
                "recurrent_wins_when": "all coefficients" if other < 0 else "no coefficients",
                "delta_other_joules_per_token": other, "delta_reload_bytes": slope}
    return {"status": "unvalidated_conditional_sensitivity", "threshold_joules_per_byte": -other/slope,
            "recurrent_wins_when": "coefficient > threshold" if slope < 0 else "coefficient < threshold",
            "coefficient_domain": "nonnegative only", "delta_other_joules_per_token": other,
            "delta_reload_bytes": slope}
