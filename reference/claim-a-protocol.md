# Claim-A v1: confirmation-holdout estimate

Prospectively frozen before launch. **Not the original H1 matrix**, not extra-k / claim B, not a dataset hunt. All 18 outcomes will be reported.

## Question

Same two estimands as C7 / estimate-v1, now on the **unused confirmation split**:

1. Quality cost of sharing at matched applied depth: NLL(R6×2) − NLL(U12).
2. Quality gain of extra applied computation at matched unique body storage: NLL(R6×2) − NLL(U6).

Arms remain **U6 / R6×2 / U12** only. Architecture, horizon, and training recipe match C7. The holdout changes.

Confirmation documents: **788 documents, 3,166 windows, 811,456 scored targets** (`sequence=256`). Created at preparation time. Never used for training or previous evaluation (C2–C7 used development or calibration).

## Why this is “conclusive A” and not H1

C7’s sharing-cost interval is wide because **n=3** (Bonferroni t-crit ≈ 6.2). This protocol uses **n=6 independent training seeds** on a still-unused split. That tightens the same claim. It does **not**:

- unlock three semantic budgets;
- test non-inferiority at δ=0.01 as H1;
- mix calibration and confirmation numbers in one interval;
- add R6×3 / U18.

Do not convert a CI that happens to exclude 0 or lie below δ into an H1 pass. Do not change δ.

## Frozen design

- Arms/order within each seed: U6, R6×2, U12.
- Fresh training seeds **7, 8, 9, 10, 11, 12**. Same seed ID shares data stream across arms; U6/R6×2 share stored initialization. U12 does not.
- Same archived C5 data (`results/u6-launch/data.tgz`); verify SHA256/bytes before upload; no refetch.
- Architecture: d=384, c=4, sequence 256, no injection, fixed k, fp32, full BPTT.
- **15,513 steps × 64 × 256 = 254,164,992** presentations per run. Common cosine (peak 3e-4, warmup 100, to 0.1×). Fresh initialization.
- Lane `claim-a`; `evaluation_split: confirmation`. Trainer still rejects `lane: sealed` / `lane: confirmation` and the original H1 sweep generator.
- Periodic confirmation evaluation every 1,000 steps plus final; fixed train probe seed 1729, 8 batches.
- Final-checkpoint evaluation only. No favorable checkpoint selection. No seed additions after seeing results.
- Uncertainty: paired Student-t intervals on complete triples, Bonferroni over the two primary contrasts (α=0.05). Incomplete triples or model failures make the study incomplete; do not replace failures until success.
- C7 remains a **labeled calibration estimate**. Do not pool it with claim-A intervals.

## Resource limit

Registered matrix is **18 runs**. One H100 cannot host all 18 inside a safe idle-free window, so allocations are:

1. **First wave:** seeds 7–9 (9 runs), **360-minute deadline, $25 cap including $0.50 reserve**.
2. **Remainder:** seeds 10–12 (9 runs), same bounds, new work directory; do not overwrite first-wave archives.

C7 nine runs cost ~$20 all-in. Combined claim-A ceiling **$50**. No outcome-dependent early stopping. Remainder is a continuation of this frozen matrix, not a new protocol.

If a registered run finishes but its archive is lost (infrastructure failure), retrain **that same identity** only. Do not replace it with a different seed. A one-run U12-s12 recovery after a lost remaining-pod archive is that case, not a new protocol.

## Interpretation (predeclared)

Report means, seed-level values, and intervals. Development (C2–C6) and calibration (C7) stay separately labeled. Claim B (extra k) is not authorized by this protocol.

## Execution update

Completed 2026-09-09 as C8. Results: [claim-a-results.md](claim-a-results.md). Original H1 matrix remains blocked.