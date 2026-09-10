# C7: estimate-v1 independent assessment

## Outcome

Completed all nine predeclared runs. **Descriptive n=3 estimate on the unused calibration holdout; not H1 confirmation.** Protocol: [estimate-v1](estimate-protocol.md). Horizon matches C6: **15,513 steps / 254,164,992** presentations per run, batch 64, sequence 256, fp32, no injection, archived 40k-document OWT subset.

| Seed | U6 | R6×2 | U12 | R6×2 − U12 | R6×2 − U6 | Recovered fraction of U6→U12 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 5.932492 | 5.914567 | 5.897405 | +0.017162 | −0.017926 | 51.1% |
| 5 | 5.929431 | 5.906460 | 5.897729 | +0.008731 | −0.022971 | 72.5% |
| 6 | 5.929278 | 5.906944 | 5.896531 | +0.010412 | −0.022334 | 68.2% |
| **Mean** | **5.930400** | **5.909323** | **5.897222** | **+0.012101** | **−0.021077** | **63.9%** |

Unique body parameters: U6 = R6×2 = **102,589**; U12 = **204,409** (−49.8% for the recurrent arm).

- R6×2 versus U12: **+1.22% perplexity** on average (mean +0.0121 nats).
- R6×2 versus U6: **2.09% lower perplexity** on average at identical unique body capacity.

Paired Student-t intervals with Bonferroni over the two primary contrasts (97.5% per contrast; n=3, critical t = 6.205):

| Contrast | Mean | 97.5% interval |
|---|---:|---|
| NLL(R6×2) − NLL(U12) | +0.01210 | **[−0.00388, +0.02809]** |
| NLL(R6×2) − NLL(U6) | −0.02108 | **[−0.03092, −0.01123]** |

The sharing-cost interval **contains 0 and contains the old δ=0.01**. Do **not** treat this as an H1 pass or fail. n=3 is a lean descriptive estimate; the interval is wide because the critical t is large. The extra-compute contrast versus U6 does **not** contain 0: recurrence improved calibration NLL in every seed.

Final-checkpoint evaluation only. No favorable checkpoint selection. Confirmation remains blocked.

## What this is not

- Not the original three-budget H1 matrix.
- Not a claim that extra k at inference helps an already-trained model.
- Not TSU energy, silicon area, or reasoning.
- Not full OpenWebText. Same 40k-document subset; calibration documents were unused for training and prior evaluation.

## Artifact status

All **nine metrics bundles** are in `artifacts/estimate/` and checksum-verified. One historical full-state archive (U12-s4) was truncated during transfer and must not be used; seed-4 U12 **metrics** are intact. Full-state checkpoints are not in git.

Reproduce:

```bash
uv run python scripts/analyze_estimate.py
```

Outputs: `reference/estimate-review/summary.json`, [quality–storage figure](estimate-review/quality-storage.png).

## Execution

Nine runs on Secure Cloud NVIDIA H100 80GB SXM. One remainder wave continued after a launcher process died; metrics for all nine runs were recovered.

## Recommendation

Write the first paper as a **measured quality–storage trade-off** with this independent calibration estimate plus development history (C2–C6). Do not spend the $142 H1 matrix. Do not change δ after seeing the interval. Optional next science is a broader pinned OWT sample or the conditional TSU residency figure—not another architecture hunt.