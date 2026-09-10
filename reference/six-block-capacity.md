# Six-block capacity: U6 / R6×2 / U12

## Implementation status

U6 development configs exist for seeds 0, 1, 2 at `configs/dev/owt-u6-1e16{,-s1,-s2}.yaml`. All three were trained on 2026-09-08 as C5, preserving the historical data/optimizer protocol and `lane: explore`. Mean NLL: U6 **6.045720**, R6×2 **6.036801**, U12 **6.025596**. Recurrence improves perplexity by approximately **0.888%** versus U6 at identical unique body capacity. See [experiment log](../EXPERIMENT_LOG.md#c5--u6-control-2026-09-08-completed).

Tests verify identical stored initialization for U6 and R6×2 (including indices, embedding and classifier), body parameters 102,589, and applied depth 6 versus 12. U12 has 204,409 body parameters. U12 uses a different upstream key split because unique depth differs: matching seed IDs does not imply identical embedding initialization across U12 and the six-block models.

At the 1e16 semantic-proxy budget U6 runs 5,204 steps / 85,262,336 tokens, versus 5,171 / 84,721,664 for R6×2 and U12. U6 sees 0.638% more tokens. Report this, and use a separately labeled matched-token sensitivity if needed. The accounting remains an unaudited backward-work proxy.

C5 passed the exact historical data identity gate (canonical manifest hash `53a5313d3fb3ae3a95e1a854d929524390e37ce05f32598e783e01014d3ce364`), byte-identical training-source verification, and matching recorded numerical-environment checks (host kernel differed). The prepared dataset and source JSONL are now archived locally in `results/u6-launch/data.tgz`; all final full-state checkpoints and metrics were retrieved with checksum verification. Preserve those artifacts and recheck identity before future comparisons.

## What the experiment answers

- U6 → R6×2: does spending extra applied computation improve quality at the same stored core capacity?
- R6×2 → U12: how much quality is sacrificed by sharing at matched applied depth?
- Neither comparison alone measures TSU speed, energy, silicon area, or inference-time extrapolation in k.

The current three-seed development gap of approximately 0.0112 nats means approximately 1.13% higher perplexity, not 1.13% higher NLL. Halving the body is not halving total GPU model storage: embedding and classifier are unchanged. If half of the *whole deployed model* could be removed at that quality cost, that would be a much stronger general GPU compression result.

## Explicit capacity-limited deployment schedule

`rz1t/residency.py` implements a **whole-slab, fixed-order replacement schedule**, not an optimal-cache lower bound. Assume a programmable fabric holding six complete block programs, with weights retained across passes/tokens. Final norm, head, AFT histories and workspace have separate capacity. A fixed-topology TSU may not permit arbitrary graph reprogramming; mapping compatibility and programming costs are unverified.

For autoregressive lockstep batch B, each round delivers B tokens:

| Arm | Applied blocks / round | Cold round block payload | Steady round block payload |
|---|---:|---:|---:|
| U6 | 6 | 6 blocks | 0 |
| R6×2 | 12 | 6 blocks | 0 |
| U12 | 12 | 12 blocks, two bank loads | 12 blocks, two bank loads |

U12 runs blocks 1–6, replaces the bank with 7–12, then returns to 1–6 for the next token round. R6×2 retains its bank and applies it twice. A partially retaining cache can reduce U12 traffic; these counts are **not a universal bound imposed by six-slot capacity**. Likewise, teacher-forced prefill can amortize loads across known sequence positions; that is not single-stream autoregressive decode. Batching amortizes bytes per delivered token but changes latency, memory and common compute costs. State histories remain distinct at each of the 12 applied positions for R6×2 and U12.

## Measured software payloads, not hardware encodings

Generated from actual model arrays in `reference/six-block-residency.json`:

- Per block: 67,880 weight bytes + 49,280 index bytes = **117,160 bytes**.
- Six-block bank: **702,960 bytes**.
- Separate shared final norm: **3,076 bytes**.
- U12 steady reload per round under this schedule: **1,405,920 bytes**.
- R6×2 steady body reload: **zero**, after its 702,960-byte cold load.

These use software fp32 weights and stored indices. Physical sparse wiring may eliminate index payloads or require different routing metadata; quantization, alignment, control registers and programming formats change these numbers. Do not label them TSU transfer measurements or pbit counts. The extractor uses the optimizer partition for weights and reports fixed indices separately.

## Conditional latency/energy model

Let W be the deployed six-block payload; B the active decode batch; N the number of rounds; beta effective transfer bandwidth in bytes/s; tau programming seconds per bank load; f the fraction of loading time exposed on the critical path.

For long resident executions under the slab schedule:

- U12 steady round loading time: `L = 2W/beta + 2*tau`.
- R6×2 steady round loading time: zero.
- If both have the same non-load round time C, `T_U = C + f*L`, `T_R = C`.
- Conditional steady throughput speedup: `T_U/T_R = 1 + f*L/C`.

For example, a **hypothetical** reload overhead equal to half of common compute time implies 1.5× throughput and 33% less time. It does not establish that such overhead exists on Z1. If f is near zero, the latency benefit is near zero even though transfer energy may still be saved.

More generally, recurrence wins latency when `C_R - C_U < f*L` (assuming its steady resident reload is zero). This permits utilization/control/state differences rather than silently equating execution time.

Let e be dynamic joules/byte, g programming joules/event, P static watts. For each arm independently:

```
T = N*C + f*(total_bytes/beta + total_events*tau)
E = N*E_compute_dynamic + total_bytes*e + total_events*g + P*T
energy_per_token = E/(N*B)
```

The implementation includes finite cold-start amortization: R6×2 loads W once, U12 loads 2W each round. It does not discard loading energy when latency is overlapped. Compute energy excludes static power to avoid double counting. All cost coefficients must be supplied explicitly; none are fitted to a claimed Z1 total.

Head/control/cache costs belong in the common or arm-specific compute terms; they are not zero. Overlap must be justified by a feasible buffering/scheduling design—not assumed for a six-block bank without spare storage.

## Paper presentation

Show quality, unique payload, applied work and conditional cost separately, rather than dividing perplexity by joules into a misleading scalar. The useful figure is a sensitivity plot over exposed reload time / common compute time, annotated with the measured quality gap. For energy use transfer/programming coefficients as axes or ranges, showing where benefits disappear. At unmatched quality this is a trade-off, not an iso-quality efficiency claim.

Ask Extropic specifically: what weight/topology state can remain resident, how many independent blocks fit, how reprogramming is performed, effective bandwidth and energy per program, and whether host/pooling/state traffic dominates. These answers determine whether the capacity threshold is relevant.

## Reproduce local analysis

From `repos/rz1t`:

```bash
JAX_PLATFORMS=cpu uv run pytest tests/test_u6_control.py tests/test_residency.py -q
JAX_PLATFORMS=cpu uv run python -m rz1t.residency \
  configs/dev/owt-u6-1e16.yaml configs/dev/owt-r6x2-1e16.yaml \
  configs/dev/owt-u12-1e16.yaml --capacity-blocks 6 --rounds 100 \
  > reference/six-block-residency.json
```

Focused validation: **87 tests passed** after integrating the final residency implementation. This is engineering validation and hypothetical scheduling analysis, not a new language-quality or hardware measurement.