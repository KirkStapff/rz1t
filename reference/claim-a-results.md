# C8: claim-a-v1 confirmation-holdout estimate

## Outcome

Completed all **18** predeclared runs. **Descriptive n=6 estimate on the unused confirmation split; not H1 confirmation.** Protocol: [claim-a-v1](claim-a-protocol.md). Horizon matches C6/C7: **15,513 steps / 254,164,992** presentations per run, batch 64, sequence 256, fp32, no injection, archived 40k-document OWT subset.

| Seed | U6 | R6×2 | U12 | R6×2 − U12 | R6×2 − U6 | Recovered fraction of U6→U12 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 5.883580 | 5.859483 | 5.850089 | +0.009394 | −0.024097 | 71.9% |
| 8 | 5.887879 | 5.861300 | 5.853415 | +0.007885 | −0.026580 | 77.1% |
| 9 | 5.880492 | 5.855362 | 5.851886 | +0.003476 | −0.025130 | 87.8% |
| 10 | 5.880047 | 5.855187 | 5.851729 | +0.003458 | −0.024860 | 87.8% |
| 11 | 5.889654 | 5.866393 | 5.850928 | +0.015464 | −0.023261 | 60.1% |
| 12 | 5.880896 | 5.862328 | 5.853363 | +0.008966 | −0.018567 | 67.4% |
| **Mean** | **5.883758** | **5.860009** | **5.851902** | **+0.008107** | **−0.023749** | **75.4%** |

Unique body parameters: U6 = R6×2 = **102,589**; U12 = **204,409** (−49.8% for the recurrent arm).

- R6×2 versus U12: **+0.81% perplexity** on average (mean +0.00811 nats).
- R6×2 versus U6: **2.35% lower perplexity** on average at identical unique body capacity.

Paired Student-t intervals with Bonferroni over the two primary contrasts (97.5% per contrast; n=6, critical t = 3.163):

| Contrast | Mean | 97.5% interval |
|---|---:|---|
| NLL(R6×2) − NLL(U12) | +0.00811 | **[+0.00234, +0.01387]** |
| NLL(R6×2) − NLL(U6) | −0.02375 | **[−0.02733, −0.02017]** |

The sharing-cost interval **excludes 0** and **contains the old δ=0.01**. Do **not** treat this as an H1 pass or fail. n=6 is a tighter estimate than C7, not a powered non-inferiority test. Extra compute versus U6 improved confirmation NLL in every seed; that interval also excludes 0.

Final-checkpoint evaluation only. No favorable checkpoint selection. Do not pool with C7. Original H1 matrix remains blocked.

## What this is not

- Not the original three-budget H1 matrix.
- Not a claim that extra k at inference helps an already-trained model.
- Not TSU energy, silicon area, or reasoning.
- Not full OpenWebText. Same 40k-document subset; confirmation documents were unused for training and prior evaluation.

## Artifact status

All **18 metrics** archives are in `artifacts/claim-a/` and checksum-verified. Seed-12 U12 was retrained after an infrastructure archive failure (same identity, not a new seed). Full-state checkpoints are not in git.

Reproduce:

```bash
uv run python scripts/analyze_claim_a.py
uv run python scripts/make_paper_figures.py
```

Outputs: `reference/claim-a-review/summary.json`, [quality–storage figure](claim-a-review/quality-storage.png), and `reference/paper-figures/`.

## Execution

All 18 runs used Secure Cloud NVIDIA H100 80GB SXM GPUs. One registered U12 seed-12 archive was lost to an idle terminate and retrained under the same identity.

## Recommendation

Claim A is now a complete n=6 confirmation-holdout estimate. Write the quality–storage paper from C8 plus labeled C2–C7 history. Do not change δ. Do not spend the $142 H1 matrix. Claim B (extra k) remains a separately frozen add-on.