# C6: instrumented duration diagnostic

## Outcome

Completed all three predeclared arms on the archived OWT subset. **Stronger observed recurrent quality–storage trade-off; not statistical confirmation.** One development seed (3), common 254,164,992 target-token presentations per arm, fixed longer cosine schedule. Protocol: [duration-v1](duration-protocol.md).

| Arm | Unique body parameters | Final dev NLL | Fixed train-probe NLL | Dev minus probe | Training seconds |
|---|---:|---:|---:|---:|---:|
| U6 | 102,589 | 5.754109 | 5.654471 | 0.099637 | 1,229.42 |
| R6×2 | 102,589 | 5.728967 | 5.639778 | 0.089188 | 1,998.39 |
| U12 | 204,409 | 5.722266 | 5.626189 | 0.096077 | 2,005.45 |

- R6×2 versus U12: **+0.006701 nats, +0.672% perplexity**, with **49.8% fewer unique body parameters**. It does not halve the whole model.
- R6×2 versus U6: **−0.025142 nats, 2.483% lower perplexity**, same unique body parameters, more applied computation. GPU training is about 1.63× U6; no inference timing claim.
- Recurrence recovers **78.96%** of the U6→U12 NLL gain at this endpoint. Ratio is descriptive within one seed and sensitive to the denominator.
- No H1 pass: one seed on a repeatedly inspected development corpus, not the independent confirmation procedure. No confidence interval from checkpoint or token pseudoreplication.

## What the curves say

16 held-out evaluations per arm: every 1,000 steps plus final 15,513. All use the identical 809,984 development targets, independent of training RNG. Fixed training probe uses 131,072 targets sampled with replacement via seed 1729; it is not the full training distribution or a paired corpus with development.

| Step | U6 dev NLL | R6×2 dev NLL | U12 dev NLL | R6×2 − U12 | U6 − R6×2 |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 6.422941 | 6.405458 | 6.403045 | 0.002413 | 0.017483 |
| 5,000 | 5.928659 | 5.914005 | 5.904990 | 0.009015 | 0.014654 |
| 10,000 | 5.795615 | 5.773618 | 5.766670 | 0.006948 | 0.021997 |
| 15,513 | 5.754109 | 5.728967 | 5.722266 | 0.006701 | 0.025142 |

All recorded development endpoints improve monotonically. From step 5,000 to final, R6×2's gain over U6 grows, and its deficit to U12 narrows. Late improvements are small but real in these recorded trajectories: from step 15,000 to final, development NLL falls ~0.0022–0.0024 across arms. There is no observed dev reversal or train-only improvement justifying an immediate dataset switch. A ~0.09–0.10-nat final train-probe/dev gap exists, but sampling/corpus differences contribute; it does not prove absence or presence of overfitting by itself.

**Important confound:** historical short-horizon endpoints used seeds 0–2 and a shorter cosine schedule (plus U6's slightly different token count). Do not attribute differences between C5 and C6 solely to duration. The within-C6 trajectories are a cleaner description; isolating horizon causally would need same-seed short/long schedule controls.

## Position bins at final checkpoint

Zero-based prediction positions; bins are descriptions of different scored tokens, not a controlled intervention on available context.

| Positions | U6 | R6×2 | U12 | U6 − R6×2 | R6×2 − U12 |
|---|---:|---:|---:|---:|---:|
| 0–63 | 5.783259 | 5.764446 | 5.751347 | 0.018814 | 0.013098 |
| 64–127 | 5.740110 | 5.713591 | 5.710401 | 0.026518 | 0.003191 |
| 128–255 | 5.746532 | 5.718915 | 5.713657 | 0.027618 | 0.005257 |

The recurrent advantage over U6 is larger later than in the first bin; the deficit to U12 is greatest in the first bin. This motivates, but does not establish a need for, a later controlled context-length study. Do not label it reasoning or long-context capability.

## Recommendation for the first paper

**Keep OWT and the three-arm design. Do not start a dataset hunt or claim H1.** The extra duration made a useful recurrent trade-off clearer in this seed. Next freeze a small independent estimation protocol at this token horizon, with a predeclared seed count and unused final evaluation corpus; include all development search history. A broader pinned OWT sample is an optional robustness experiment, not required to seek a larger effect before writing. Finish the source/data/accounting audit and conditional TSU residency figure in parallel. No further paid runs are authorized by this report.

## Execution and artifacts

- Secure Cloud NVIDIA H100 80GB SXM; allocation elapsed **6,031.69 seconds**.
- Data hash/identity exactly matched C5; no network OWT rebuild. Numerical environment matches historical packages/H100; host platform differs and is recorded.
- Released metrics: `artifacts/duration/`. Full-state checkpoints are not in git.

Reproduce from released metrics:

```bash
uv run python scripts/analyze_duration.py
```

Outputs: [curves](duration-review/curves.png) and `reference/duration-review/summary.json`. Historical records remain unchanged.