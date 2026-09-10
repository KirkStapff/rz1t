# Estimate-v1 independent assessment

Prospectively frozen before launch. **Not H1 confirmation**, not a powered non-inferiority test, not a dataset hunt. All nine outcomes will be reported.

## Question

On the archived 40k-document OWT subset, at matched tokens and the C6 horizon, estimate:

1. Quality cost of sharing at matched applied depth: NLL(R6×2) − NLL(U12).
2. Quality gain of extra applied computation at matched unique body storage: NLL(R6×2) − NLL(U6).

Primary evaluation is the **unused calibration split** (829 documents, 3,114 windows, 797,184 scored targets). That split was created at preparation time and has not been used for training or previous evaluation. Confirmation documents remain unused and blocked.

## Frozen design

- Arms/order within each seed: U6, R6×2, U12.
- Fresh training seeds **4, 5, 6**. Same seed ID shares data stream across arms; U6/R6×2 share stored initialization. U12 does not.
- Same archived C5/C6 data (`results/u6-launch/data.tgz`); verify SHA256/bytes before upload; no refetch.
- Architecture: d=384, c=4, sequence 256, no injection, fixed k, fp32, full BPTT.
- **15,513 steps × 64 × 256 = 254,164,992** target-token presentations per run. Common longer cosine (peak 3e-4, warmup 100, to 0.1×). Fresh initialization.
- Lane `calibration`; `evaluation_split: calibration`. Periodic calibration evaluation every 1,000 steps plus final; fixed train probe seed 1729, 8 batches.
- Final-checkpoint evaluation only for the registered endpoints. No favorable checkpoint selection. No seed additions after seeing results.
- Uncertainty: paired Student-t intervals on the three complete triples, Bonferroni over the two primary contrasts (α=0.05). n=3 is descriptive. Incomplete triples or model failures make the study incomplete; do not replace failures until success.

## Resource limit

One H100, **360-minute allocation deadline, $25 admission cap including $0.50 reserve**. C6 training for three arms was ~87 minutes of trainer time plus setup; nine runs are about 3× that. No outcome-dependent early stopping. Confirmation remains blocked.

## Interpretation (predeclared)

Report means, seed-level values, and intervals. Do not convert a CI that happens to lie below the old δ=0.01 into an H1 pass. Do not change δ. Development history (C2–C6) stays separately labeled.

## Execution update

Completed 2026-09-08 as C7. Results: [estimate-results.md](estimate-results.md). Confirmation remains blocked.