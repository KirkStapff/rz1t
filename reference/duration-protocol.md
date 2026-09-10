# Duration-v1 development diagnostic

Prospectively fixed before launch. Not H1 confirmation, not a powered statistical-significance test. All three outcomes will be reported.

- Arms/order: U6, R6×2, U12; one new preselected development seed ID **3** each. Existing seed derivation preserved; same data stream across arms, same stored initialization for U6/R6×2 (not U12).
- Archived C5 data: `results/u6-launch/data.tgz`, verify SHA256 and byte count before upload. Prepared dataset identity must exactly equal historical manifest/data identity; no refetch or dataset selection.
- Architecture: d=384, c=4, sequence 256, no injection, fixed k, fp32, full BPTT. Six/twelve applied depths as before.
- **15,513 steps × 64 × 256 = 254,164,992 target-token presentations per arm.** No semantic-budget stopping; per-arm semantic-proxy work reported separately. This is exactly 3× the old depth-12 token horizon, not new distinct data.
- Same peak LR 3e-4, warmup 100 steps, AdamW recipe, cosine over the full new horizon to 0.1×. Fresh initialization; do not resume old finished schedules. Early new checkpoints are not equal-protocol replications of historical final endpoints.
- Full development evaluation every 1,000 steps and at final step; 809,984 scored tokens. Fixed train probe: 8 batches of 64 windows (131,072 targets), independent fixed probe seed 1729 recreated every evaluation. Probe samples with replacement and may repeat windows; not exhaustive training loss.
- Position bins: zero-based prediction positions [0,64), [64,128), [128,256). Weight by actual token counts, not equal bin averages. Descriptive within-seed diagnostics, not independent replicates.
- Metrics recorded atomically with checkpoint every 1,000 steps; final full-state checkpoint and metrics bundle independently checksum-verified after each arm before advancing. Archived data already retained locally.
- Primary diagnostic endpoints: final development NLL, fixed-train-probe NLL, their separation, trajectory of U12−R6×2 and U6−R6×2, position bins. No favorable intermediate-checkpoint selection. No seed additions based on outcome.
- Interpretation: dev improvement with stable gaps supports persistence with duration; train-only improvement motivates broader data; increasing U12 advantage weakens approximation claim; increased R6×2 gain over U6 strengthens recurrent-compute utility. All valid outcomes.
- Source revision: new diagnostic evaluator and trainer hooks intentionally differ from historical training source. Freeze actual source hashes before allocation; require those on pod along with historical numerical environment (platform/kernel difference allowed and recorded).
- Resource limit: one H100, **150-minute allocation deadline, $10 admission cap including $0.50 reserve**; verify live quote/account/no existing pods before allocation. Historical training timings suggest ~1.5 hours of training for all three before extra eval/setup/archives; no guarantee of completion. No outcome-dependent early stopping. Infrastructure/model failure yields incomplete diagnostic, not replacement until favorable.

## Local checks before launch

Diagnostic/training focused suite: 22 tests passed, including deterministic fixed probes, token/bin aggregation, training RNG isolation, resume parity and periodic/final persistence. Further full-suite and launcher checks recorded separately in the experiment log. Numerical result fields remain empty until execution.