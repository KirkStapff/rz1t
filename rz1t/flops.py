"""Auditable *estimated semantic work*, not measured device FLOPs.

Version 1 counts a multiply and an add separately (2 per MAC), bias adds
separately, and each scalar nonlinearity/divide/comparison as one proxy unit.
The last convention is not a claim that exp/tanh costs one hardware FLOP.
Padded convolution taps are counted, cumsum is a serial mathematical sum,
and tensor gathers/reshapes/broadcasts are zero arithmetic (not zero traffic).
Backward is explicitly estimated as twice forward including loss. Optimizer,
compilation, evaluation and data handling are outside this semantic budget.
XLA/backward validation is outstanding; do not describe this as audited FLOPs.
"""
from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, ROUND_CEILING
import math
from typing import Any

ACCOUNTING_VERSION = "semantic-proxy-v1"


def _get(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _positive_int(value: Any, name: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def accounting(config: Any, batch_size: int = 1) -> dict[str, Any]:
    """Return JSON-safe per-step/per-token estimates for a ModelConfig or dict.

    Sequence is the number of supervised input positions (targets shifted by
    one); caller supplies batch size after any gradient accumulation. All
    applied blocks, not just unique blocks, incur forward/backward arithmetic.
    `executed_training_per_step_estimate` adds one forward block recomputation
    for remat; actual compiler work must be measured independently.
    """
    batch = _positive_int(batch_size, "batch_size")
    d, t, v, h, s = [_positive_int(_get(config, field), field) for field in
                     ("n_embed", "sequence", "vocab", "aft_heads", "aft_ksize")]
    if d % h:
        raise ValueError("n_embed must be divisible by aft_heads")
    p, m, k, q = [_positive_int(_get(config, field, default), field, zero=field in ("p", "q"))
                   for field, default in (("p", 0), ("m", 6), ("k", 2), ("q", 0))]
    injection = _get(config, "injection", "none")
    if injection not in ("none", "residual"):
        raise ValueError("accounting implements injection='none' or 'residual' only")
    if _get(config, "aft_kind", "conv") != "conv":
        raise ValueError("accounting implements AFT-conv only")
    c = _get(config, "linear_fan_in", 4)
    if c is not None:
        _positive_int(c, "linear_fan_in")
    for flag in ("tanh_linear", "tanh_mlp", "remat"):
        if not isinstance(_get(config, flag, False), bool):
            raise ValueError(f"{flag} must be boolean")
    depth = p + m * k + q
    tokens = batch * t

    # Whole-batch, one-applied-block counts, following z1t.components:293-355.
    block: dict[str, int] = {}
    for name, ni, no in (("qkv", d, 2*d+h), ("attn_out", d, d),
                          ("mlp_up", d, 4*d), ("mlp_down", 4*d, d)):
        fan = ni if c is None else min(c, ni)
        block[f"{name}_mac"] = tokens * 2 * fan * no
        block[f"{name}_bias"] = tokens * no
        block[f"{name}_tanh"] = tokens * no if _get(config, "tanh_linear", False) else 0
    block.update({
        "dyt_multiply_add": tokens * 6 * d,  # two norms, 2 multiplies + add
        "dyt_tanh": tokens * 2 * d,
        "key_tanh_exp": tokens * 2 * h,
        # Per-sequence kernel setup (compiler may hoist/CSE across examples).
        "kernel_exp_subtract": batch * 2 * h * s,
        "key_value_multiply": tokens * d,
        "convolution_mac": tokens * 2 * s * (d + h),
        "cumulative_pool_add": batch * (t - 1) * (d + h),
        "local_global_add": tokens * (d + h),
        "denominator_epsilon_add": tokens * h,
        "context_divide": tokens * d,
        "query_tanh_gate": tokens * 2 * d,
        "mlp_activation": tokens * 4 * d * (1 if _get(config, "tanh_mlp", False) else 5),
        "residual_add": tokens * 2 * d,
    })
    # SiLU convention: sigmoid = negate, exp, add, divide; then multiply.
    body = {name: count * depth for name, count in block.items()}
    body.update({"embedding_lookup_arithmetic": 0, "position_add": tokens*d,
                 "final_dyt_multiply_add": tokens*3*d, "final_dyt_tanh": tokens*d})
    extra_loops = max(k - 1, 0)
    body["injection_residual_add"] = (
        tokens * d * extra_loops if injection == "residual" else 0)
    head = {"classifier_mac": tokens*2*d*v, "classifier_bias": tokens*v}
    # Stable logsumexp NLL: max, subtract, exp, sum, log, target subtract,
    # then batch/token reduction and divide. Comparisons are proxy units.
    loss = {"logsumexp_max": tokens*(v-1), "logsumexp_subtract": tokens*v,
            "logsumexp_exp": tokens*v, "logsumexp_sum": tokens*(v-1),
            "logsumexp_log": tokens, "restore_max": tokens,
            "target_subtract": tokens, "mean_reduction": tokens}
    fb, fh, fl = sum(body.values()), sum(head.values()), sum(loss.values())
    forward = fb + fh + fl
    backward = 2 * forward
    remat = depth * sum(block.values()) if _get(config, "remat", False) else 0
    return {
        "version": ACCOUNTING_VERSION, "validated_against_xla": False,
        "units": "estimated semantic arithmetic/proxy units (not device FLOPs)",
        "backward_convention": "2 * (forward body + head + loss); optimizer excluded",
        "nonlinear_convention": "each scalar transcendental/divide/comparison = 1 proxy unit",
        "tokens_per_step": tokens, "applied_depth": depth, "unique_blocks": p+m+q,
        "forward_block_per_step": sum(block.values()),
        "forward_body_per_step": fb, "forward_head_per_step": fh, "forward_loss_per_step": fl,
        "forward_body_per_token": fb/tokens, "forward_head_per_token": fh/tokens,
        "forward_loss_per_token": fl/tokens,
        "backward_per_step_estimate": backward,
        "semantic_training_per_step": forward + backward,
        "rematerialization_per_step_estimate": remat,
        "executed_training_per_step_estimate": forward + backward + remat,
        "itemized_per_step": {"body": body, "head": head, "loss": loss},
        "limitations": ["No XLA validation yet", "Backward is a proxy, not symbolic autodiff counting",
                        "Cumsum implementation, padding and kernel hoisting may differ on device",
                        "No traffic, optimizer, evaluation, compilation or data-loading charge",
                        "Remat assumes one additional forward per applied block"],
    }


def budget_to_steps(target: float | int, per_step: float | int,
                    max_overshoot: float = 0.01) -> dict[str, Any]:
    """Ceil whole-step rule; reject budgets whose actual overshoot exceeds cap.

    Decimal string conversion avoids float ceiling errors for large budgets.
    A cap failure requires pre-seal budget/batch redesign, not silent rounding.
    """
    for value, name in ((target, "target"), (per_step, "per_step")):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(max_overshoot) or not 0 <= max_overshoot <= 0.01:
        raise ValueError("max_overshoot must be between zero and 0.01")
    budget, work = Decimal(str(target)), Decimal(str(per_step))
    steps = int((budget/work).to_integral_value(rounding=ROUND_CEILING))
    actual = steps * work
    overshoot = (actual-budget)/budget
    if overshoot > Decimal(str(max_overshoot)):
        raise ValueError(f"whole-step overshoot {float(overshoot):.3%} exceeds cap {max_overshoot:.3%}")
    return {"steps": steps, "target_budget": target, "actual_budget": float(actual),
            "overshoot_fraction": float(overshoot), "max_overshoot": max_overshoot}
