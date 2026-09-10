# rz1t experiment log

Executed measurements only. Not a preregistration. Not H1.

Append dated rows. Do not overwrite.

**Status:** claim-A (C8) completed as a confirmation-holdout quality–storage estimate. Original three-budget H1 remains unexecuted. Do not relabel C8 as H1.

---

## Runs

| ID | Date | What | GPU | Spend | Outcome |
|---|---|---|---|---:|---|
| A1 | 2026-09-07 | synthetic profile batch 1–8 | H100 SXM Secure `vubj90ubomzrbh` | ~$1.98 | U12/R6×2 ~82.9k tok/s at batch 8 |
| A2 | 2026-09-07 | aborted idle | H100 `55xis5r5ishb6p` | with A3 | SSH mapping bug; no profile |
| A3 | 2026-09-07 | synthetic profile batch 16–128 | H100 SXM Secure `07i6u6m7q1zmoq` | ~$0.63 with A2 | plateau batch 64 ~129–130k tok/s |
| C1 | 2026-09-07 | Shakespeare char screen, 5 arms, 1 seed | A4000 `cr40vquhvipja8` | ~$0.04 | U12 2.5123; R6×2 2.5196 |
| C2a | 2026-09-07 | OWT-subset 1e16 pair attempt | H100 `kbcif1lfocdbb7` | with C2 | failed (HF xet abort) |
| C2b | 2026-09-07 | OWT-subset 1e16 pair attempt | H100 `wxyp6y4ewdxhca` | with C2 | failed (SSH 255) |
| C2 | 2026-09-07 | OWT-subset 1e16 U12 vs R6×2, 1 seed | H100 SXM Secure `1r068531m3u52f` | ~$2.01 | U12 6.0267; R6×2 6.0396 |
| C3 | 2026-09-08 | OWT-subset 1e16 U12 vs R6×2, seeds 1–2 | H100 SXM Secure `2ptqzsdp7zc74x` | with C4 | gaps +0.0110, +0.0098 |
| C4 | 2026-09-08 | OWT-subset 1e16 U2 vs R2×6, seed 0 | same pod as C3 | ~$4.44 with C3 | U2 6.0736; R2×6 6.0539 |
| C7a | 2026-09-08 | estimate-v1 seeds 4–6 attempt | H100 SXM Secure `a2q8u7bk422dmh` | ~$6.39 | seed 4 complete; seeds 5–6 not started after truncated U12-s4 state copy |
| C7b | 2026-09-08 | estimate-v1 seeds 5–6 remainder | H100 SXM Secure `ho5gdmnccj543x` | ~$13.68 | six remaining runs complete; n=3 estimate finished |
| C7 | 2026-09-08 | estimate-v1 n=3 calibration | C7a+C7b | ~$20 | mean R6×2−U12 +0.0121 nats; R6×2−U6 −0.0211; not H1 |
| C8a | 2026-09-08/09 | claim-a-v1 seeds 7–9 | H100 SXM Secure `5h9bz7vvi5yho1` | ~$18.78 | 9/18 confirmation-split runs; remainder seeds 10–12 registered |
| C8b | 2026-09-09 | claim-a-v1 seeds 10–12 remainder | H100 SXM Secure `z5f0mdwt8bd8o6` | idle waste; launched $50.98 → $7.89 | 8/9 remainder runs archived; U12-s12 lost after idle terminate |
| C8c | 2026-09-09 | claim-a-v1 U12-s12 recovery | H100 SXM Secure `i7wfq4a4nci6fu` | ~$2.5 useful; launched $7.65 | same identity retrained; n=6 complete |
| C8 | 2026-09-09 | claim-a-v1 n=6 confirmation holdout | C8a+C8b+C8c | mixed; idle waste dominates C8b | mean R6×2−U12 +0.0081 nats; R6×2−U6 −0.0237; not H1 |

| A4 | 2026-09-09 | repeat synthetic profile batch 1–8 | H100 SXM Secure `nj0rrf4vn1yf5s` | ~$0.14 | reproduces A1 (~82.9k tok/s at B=8); no new science |

No GPU pods remaining after A4.

---

## A. H100 throughput (synthetic)

fp32, d=384, seq=256, vocab=50257, `injection=none`. Details: `reference/cost_table.md`.

| Arm | batch | tokens/s | peak mem |
|---|---:|---:|---:|
| U12 | 8 | 82 892 | 2.95 GiB |
| U12 | 64 | 129 348 | 12.83 GiB |
| U12 | 128 | 87 169 | 24.77 GiB |
| R6×2 | 8 | 82 907 | 4.63 GiB |
| R6×2 | 64 | 130 212 | 26.68 GiB |
| R6×2 | 128 | 100 807 | 51.99 GiB |

Working batch **64**. Compute-only at $3.49/h: 1e16 **$0.63**; confirmation n=8 **$142**.

---

## C1. Shakespeare char (toy)

d=64, seq=128, 2000 steps, batch 32, seed 0, development eval 26 752 tokens.

| Arm | p,m,k,q | injection | body | NLL |
|---|---|---|---:|---:|
| U12 | 0,12,1,0 | none | 34 377 | **2.5123** |
| R6×2 | 0,6,2,0 | none | 17 253 | 2.5196 |
| R6×2-residual | 0,6,2,0 | residual | 17 253 | 2.5226 |
| P1M2K4Q1-residual | 1,2,4,1 | residual | 11 545 | 2.5350 |
| R2×6-residual | 0,2,6,0 | residual | 5 837 | 2.5467 |

Not OWT. Not H1.

---

## C2. OWT subset, 1e16, development (one pair)

Not full OpenWebText. Not confirmation.

| | |
|---|---|
| Pod | `1r068531m3u52f`, Secure H100 SXM, AP-IN-1, $3.49/h, 1824 s |
| Data | first 40 000 streaming `Skylion007/openwebtext` docs; GPT-2 BPE; seq 256; eval = **development** |
| Manifest SHA256 | `53a5313d3fb3ae3a95e1a854d929524390e37ce05f32598e783e01014d3ce364` |
| Train | 5171 steps, batch 64, lr 3e-4, d=384, seed 0, budget 1e16, 84 721 664 tokens |
| Eval | 809 984 tokens |
| Artifacts | `results/owt-dev-launch-3/` |

| Arm | injection | body | NLL | train s |
|---|---|---:|---:|---:|
| U12 | none | 204 409 | **6.0267** | 710 |
| R6×2 | none | 102 589 | 6.0396 | 699 |

Gap **+0.0128** nats/token (R6×2 worse). Proposed δ is 0.01. One seed, subset corpus, development split.

**Read:** R6×2 is close, not under δ, not a CI. Do not seal. Do not launch the 48-run H1 matrix on this.

---

## C3. OWT subset, 1e16, two more U12/R6×2 pairs

Same recipe as C2: 40k streaming docs, GPT-2 BPE, seq 256, batch 64, lr 3e-4, development eval, `injection=none`, `lane: explore`. Manifest SHA256 `53a5313d3fb3ae3a95e1a854d929524390e37ce05f32598e783e01014d3ce364` (matches C2). Pod `2ptqzsdp7zc74x`, Secure H100 SXM, US-NE-1, $3.49/h. Artifacts: `results/owt-c34-launch/owt-c34/summary.json`.

| Seed | U12 NLL | R6×2 NLL | Gap |
|---:|---:|---:|---:|
| 0 (C2) | 6.0267 | 6.0396 | +0.0128 |
| 1 | 6.0255 | 6.0365 | +0.0110 |
| 2 | 6.0246 | 6.0344 | +0.0098 |

Mean gap **+0.0112** (n=3). Sample sd ≈ 0.0015. Informal 95% t interval ≈ **[0.0074, 0.0150]** — contains δ=0.01; lower bound > 0. This is **not** the H1 simultaneous procedure, not confirmation corpus, not a CI you can seal on.

All six C3/C4 runs completed. Full checkpoints were not archived (scp aborted after metrics). No confirmation lane.

**Read:** C2’s +0.013 was not a one-seed fluke. R6×2 is stably ~0.01 nats worse at 1e16 on this subset. Close to δ, not under it. Do not confirm H1.

---

## C4. OWT subset, 1e16, H2 probe (seed 0)

Same pod and data as C3. Unique-param control: U2 vs R2×6 (`m=2`, body 34 709 both). Applied depth 2 vs 12. Matched **semantic budget**, not matched tokens: U2 ran 5226 steps / 85.6M tokens because the shallower body is cheaper per step; R2×6 ran 5171 steps / 84.7M tokens like U12/R6×2.

| Arm | p,m,k,q | body | applied L | NLL | train s |
|---|---|---:|---:|---:|---:|
| U2 | 0,2,1,0 | 34 709 | 2 | 6.0736 | 282 |
| R2×6 | 0,2,6,0 | 34 709 | 12 | 6.0539 | 683 |
| R6×2 (C2) | 0,6,2,0 | 102 589 | 12 | 6.0396 | 699 |
| U12 (C2) | 0,12,1,0 | 204 409 | 12 | 6.0267 | 710 |

R2×6 beats U2 by **0.020** (extra `k` at fixed unique weights **helps**). R2×6 is still **+0.027** vs U12 and **+0.014** vs R6×2. Extreme sharing does not recover a 12-unique-block model.

**Read:** do **not** pivot the primary to small-`m` large-`k`. Extra time helps a 2-block core, but R6×2 remains closer to U12. H1-style tying is still the better candidate paper than “latent thinking via k=6.”

---

## Notes

- `injection: residual` unused in C2–C4.
- C2–C4 used `lane: explore`. Confirmation/sealed still rejected.
- Accounting: `semantic-proxy-v1`. Unvalidated vs XLA.
- Failed C2a/C2b pods were terminated.
- C3/C4 pod `2ptqzsdp7zc74x` stopped/deleted after metrics pull.

## Engineering addendum — U6 control and residency analysis

U6 configs for development seeds 0–2 are implemented, but **not trained**. Tests verify identical stored initialization/body count to R6×2, with applied depth 6 instead of 12. The 1e16 proxy schedule derives 5,204 steps / 85,262,336 tokens (0.638% more tokens than the depth-12 arms). Recover and verify the historical data/source identities before comparing new runs with C2–C4.

An explicit hypothetical six-block whole-bank replacement model now reports cold/steady traffic, batch amortization and conditional bandwidth/programming/energy costs. It is not an optimal-cache bound or a TSU measurement. See [analysis and commands](reference/six-block-capacity.md), `rz1t/residency.py`, and `reference/six-block-residency.json`.

Local validation: **175 tests passed** in the integrated CPU suite, including 87 focused U6/residency tests. No paid resources launched in this implementation step; no new NLL results or live account/pod status claimed.

Interpretation clarification (historical entries preserved): the first-paper plan now estimates quality–storage trade-offs rather than requiring a pass at δ=0.01. C4's R2×6 and R6×2 are different storage points; being worse in NLL does not by itself reject the smaller model. Mean C2/C3 gap +0.0112 nats corresponds to about **1.13% higher perplexity**, not a 1% increase in NLL.

## C5 — U6 control, 2026-09-08 (completed)

Supersedes the untrained-U6 status above; historical entries are preserved. Three U6 seeds completed on pod `1yznud8pku32i8`, Secure H100 SXM at $3.49/h. Allocation elapsed 1,882.46 s (31.37 min), estimated compute charge **$1.82**. Account balance snapshot $98.1325 → $96.3368 (approximately **$1.80**, potentially subject to billing lag). Pod stopped/deleted; live post-run query returned no pods, with account `currentSpendPerHr=0.024` (non-pod recurring charge not investigated here).

### Comparability and protocol

- Same 40k-document subset and development evaluation (809,984 tokens), not full OWT or confirmation.
- Rebuilt dataset passed `PreparedDataset` integrity/disjointness checks; full data identity exactly matches C2/C3, including canonical manifest hash `53a5313d3fb3ae3a95e1a854d929524390e37ce05f32598e783e01014d3ce364`.
- All recorded training-source hashes match historical baselines byte-for-byte. Python 3.12.14, numerical package versions, H100 device kind, precision settings and XLA flags match. Host kernel differs and is recorded in `provenance.json`.
- U6: d=384, seq=256, c=4, batch 64, float32, no injection; 5,204 steps / 85,262,336 tokens per seed at 1e16 semantic-proxy budget. U12/R6×2: 5,171 steps / 84,721,664 tokens. U6 receives 0.638% more training tokens; this is not matched-token evidence.
- U6/R6×2 have identical initial stored arrays per seed; matching U12 seed IDs does not imply identical embedding initialization.

### Results (historical baseline columns reused after verification)

| Seed | U6 NLL | R6×2 NLL | U12 NLL | R6×2 − U6 |
|---:|---:|---:|---:|---:|
| 0 | 6.048313 | 6.039560 | 6.026727 | −0.008754 |
| 1 | 6.045394 | 6.036488 | 6.025510 | −0.008906 |
| 2 | 6.043451 | 6.034355 | 6.024551 | −0.009096 |
| Mean | **6.045720** | **6.036801** | **6.025596** | **−0.008919** |

U6 training times: 412.65 / 411.78 / 417.37 seconds. These are recorded trainer timings, not TSU performance measurements or a controlled same-host GPU benchmark against historical runs.

**Interpretation:** recurrence improves the six-block model in all three development seeds: approximately **0.888% lower perplexity** than U6 with the same 102,589 unique body parameters. R6×2 remains approximately **1.127% higher perplexity** than U12 with 49.8% fewer unique body parameters. This closes the missing storage-matched control and supports a modest quality–storage trade-off story. It does not demonstrate reasoning, inference-only extra-k improvement, TSU energy savings, or the original H1 non-inferiority claim.

### Artifacts and validation

`results/u6-launch/`: `summary.json`, `provenance.json`, `launch.json`, `account-after.json`, source archive, setup/run logs, and per-seed metrics/state archives. Each `U6-s*-state.tgz` contains the final full model/optimizer/RNG checkpoint, its metadata and manifest, plus latest pointer. Remote checkpoint manifests were verified; all seven downloaded metrics/state/data bundles passed local SHA256 and byte-count verification before pod deletion. `data.tgz` preserves the exact source JSONL and prepared dataset. Historical C3/C4 missing checkpoints remain missing; this run does not repair that history.

New launcher: `python -m runpod.launch_u6` (dry-run by default); actual run used `--execute`, a 90-minute allocation deadline, $6 admission ceiling including $0.50 reserve, and failed-identity rejection before training. New provenance-gate and U6-control tests: **17 passed**; lints for the three new Python files were clean. This does not claim a fresh full-suite run.

## Offline curve review — no new training

Reviewed preserved C2–C5 metrics and the archived dataset; see [full review](../../docs/rz1t-curve-review.md). Generated `reference/curve-review/summary.json` and `training-curves.png` with `scripts/review_existing_curves.py` (executed successfully; lints clean).

- All inspected trajectories have **zero intermediate held-out evaluations** (`eval_every: 0`). Online minibatch loss cannot establish overfitting or held-out convergence.
- U12 seeds 0–1 survive through step 5,000 in checkpoint metadata. R6×2 seed-1 `metrics.json` is truncated; only **3,061 complete rows** can be recovered. The C3/C4 metrics tarball is also truncated. No late R6×2 slope is inferred. Historical final scalar outcomes remain separately available.
- U12 and U6 show similar late online-loss flattening for seeds 0–1 under cosine LR decay; there is no clear evidence that additional duration will preferentially favor either family.
- Dataset has **37,551,616 eligible training target positions**, versus ~84.72M presentations for depth-12 runs: about 2.256 replacement-sampled draws per window. Tripling duration on this subset principally increases repeated exposure, not corpus diversity.
- Recommendation: improve periodic/fixed-probe evaluation and run a predeclared three-arm matched-token duration diagnostic before changing dataset families. Broader pinned OWT is a subsequent robustness option, not a way to select a favorable result.

No paid resources launched, no new NLL endpoints, and no live account/pod status queried in this review.

## C6 — instrumented duration diagnostic, 2026-09-08 (completed)

Executed the [predeclared duration protocol](reference/duration-protocol.md). All three arms completed with new development seed ID 3 on the exact archived subset: **15,513 steps / 254,164,992 token presentations each**, batch 64, sequence 256, fp32, no injection, fresh initialization and a common longer cosine schedule. Matched tokens, not matched semantic-proxy compute. Full development evaluation every 1,000 steps plus final (16 evaluations), fixed 131,072-target train probe (seed 1729), and position bins. This is not confirmation and no favorable checkpoint was selected.

| Arm | Body parameters | Final development NLL | Fixed train probe NLL | Training seconds |
|---|---:|---:|---:|---:|
| U6 | 102,589 | 5.754109 | 5.654471 | 1,229.42 |
| R6×2 | 102,589 | 5.728967 | 5.639778 | 1,998.39 |
| U12 | 204,409 | 5.722266 | 5.626189 | 2,005.45 |

**Read:** R6×2 has **0.672% higher perplexity than U12** with 49.8% fewer unique body parameters, and **2.483% lower perplexity than U6** at identical unique capacity. It recovers **78.96%** of U6→U12's NLL gain in this seed. This strengthens the observed quality–storage trade-off, not statistical significance. All recorded dev trajectories keep improving; from step 5,000 to final the R6×2 advantage over U6 grows and its deficit to U12 narrows. No development reversal motivates an immediate dataset pivot. Position-bin benefits over U6 are larger later in windows; different scored targets mean this is not a controlled context-length effect.

Historical short endpoints differ in seed, schedule and U6 token matching. Do not attribute historical-to-C6 changes solely to duration, or claim H1 from a single endpoint below δ. Next recommendation: bounded independent estimation at a frozen horizon plus paper/audit work, not another dataset/architecture hunt.

Secure H100 SXM pod `jhexx4n74ced2z`, $3.49/h; elapsed 6,031.69 s (100.53 min), estimated allocation charge **$5.85**. Balance snapshots $96.27884 → $90.37917 (~$5.90, includes possible other charges/billing lag). Pod stopped/deleted; post-run query no pods, recurring non-pod spend $0.024/h unchanged. Six final metrics/state bundles locally checksum/byte-verified before deletion; archived data reused without refetch. Frozen revised evaluator/trainer source, exact data identity, and historical numerical-environment checks passed (host platform difference recorded).

Released metrics: `artifacts/duration/`; [full result](reference/duration-results.md); [curves](duration-review/curves.png). Reproduce with `uv run python scripts/analyze_duration.py`. Historical missing C3/C4 checkpoints remain missing.

## C7a — estimate-v1 independent assessment, 2026-09-08 (incomplete)

Frozen protocol: [estimate-v1](reference/estimate-protocol.md). Unused **calibration** holdout (797,184 scored tokens). Matched 15,513 steps / 254,164,992 presentations. Seeds 4–6 registered. Confirmation remains blocked.

Pod `a2q8u7bk422dmh` completed seed 4, then the launcher terminated after a truncated `U12-s4-state.tgz` download. Seeds 5–6 were never started. Metrics for all three seed-4 arms were checksum-verified locally. U6-s4 and R6×2-s4 full-state archives verified; U12-s4 state archive is truncated and must not be used. Account $90.195 → $83.805 (~$6.39). No remaining pods.

| Arm | Calibration NLL | Train probe | Training seconds |
|---|---:|---:|---:|
| U6-s4 | 5.932492 | 5.654308 | 1,312.86 |
| R6×2-s4 | 5.914567 | 5.642567 | 2,061.46 |
| U12-s4 | 5.897405 | 5.621915 | 1,981.88 |

One-seed calibration gaps: R6×2 − U12 **+0.01716** nats (**+1.73% perplexity**); R6×2 − U6 **−0.01793** nats (**1.78% lower perplexity**). This is **not** the registered n=3 estimate. Do not treat seed 4 as confirmation. Remainder launch for seeds 5–6 is a continuation of the same frozen matrix, not a new protocol.

## C7 — estimate-v1 independent assessment, 2026-09-08 (complete)

Supersedes the incomplete C7a status above; historical C7a text is preserved. Frozen protocol: [estimate-v1](reference/estimate-protocol.md). Unused **calibration** holdout (797,184 scored tokens). Matched 15,513 steps / 254,164,992 presentations. Seeds 4–6 all completed. Confirmation remains blocked.

| Seed | U6 | R6×2 | U12 | R6×2 − U12 | R6×2 − U6 |
|---:|---:|---:|---:|---:|---:|
| 4 | 5.932492 | 5.914567 | 5.897405 | +0.017162 | −0.017926 |
| 5 | 5.929431 | 5.906460 | 5.897729 | +0.008731 | −0.022971 |
| 6 | 5.929278 | 5.906944 | 5.896531 | +0.010412 | −0.022334 |
| **Mean** | **5.930400** | **5.909323** | **5.897222** | **+0.012101** | **−0.021077** |

Mean R6×2 is **+1.22% perplexity versus U12** with 49.8% fewer unique body parameters, and **2.09% lower perplexity versus U6** at identical unique body capacity. Bonferroni paired t intervals (n=3): sharing cost **[−0.00388, +0.02809]** (contains 0 and δ=0.01); extra-compute gain **[−0.03092, −0.01123]** (does not contain 0). **Not an H1 pass.** n=3 is descriptive.

All nine metrics bundles checksum-verified. U12-s4 full-state archive remains truncated and unused; other eight state archives verified. Remainder pod `ho5gdmnccj543x` stopped/deleted; live query returned no pods. Combined C7 spend ≈ **$20** (C7a $6.39 + remainder $83.805 → $70.124). Recurring non-pod spend $0.024/h unchanged.

Released metrics: `artifacts/estimate/`; [full result](reference/estimate-results.md); [figure](reference/estimate-review/quality-storage.png). Reproduce with `uv run python scripts/analyze_estimate.py`.

## C8a — claim-a-v1 first wave, 2026-09-08/09 (incomplete n=6)

Frozen protocol: [claim-a-v1](reference/claim-a-protocol.md). Unused **confirmation** holdout (788 documents, 3,166 windows, 810,496 scored tokens). Matched 15,513 steps / 254,164,992 presentations. Seeds **7–12** registered. This wave completed seeds **7–9** only. Original H1 matrix remains blocked.

Pod `5h9bz7vvi5yho1`, Secure H100 SXM, $3.49/h. Account **$70.046 → $51.267** (~$18.78 including idle after a local launcher timeout; remaining five runs continued over SSH on the same allocation). Pod stopped/deleted; live query returned no pods. Recurring non-pod spend $0.024/h unchanged.

| Seed | U6 | R6×2 | U12 | R6×2 − U12 | R6×2 − U6 |
|---:|---:|---:|---:|---:|---:|
| 7 | 5.883580 | 5.859483 | 5.850089 | +0.009394 | −0.024097 |
| 8 | 5.887879 | 5.861300 | 5.853415 | +0.007885 | −0.026580 |
| 9 | 5.880492 | 5.855362 | 5.851886 | +0.003476 | −0.025130 |
| **Mean (n=3, first wave only)** | **5.883984** | **5.858715** | **5.851797** | **+0.006918** | **−0.025269** |

All nine first-wave metrics and full-state archives checksum-verified locally. This is **not** the registered n=6 claim-A estimate. Do not pool with C7. Remainder launch for seeds 10–12 is a continuation of the same frozen matrix.

Released metrics: `artifacts/claim-a/` (first-wave seeds 7–9).

## C8 — claim-a-v1 independent assessment, 2026-09-09 (complete)

Supersedes the incomplete C8a status above; historical C8a text is preserved. Frozen protocol: [claim-a-v1](reference/claim-a-protocol.md). Unused **confirmation** holdout (788 documents, 3,166 windows, 810,496 scored tokens). Matched 15,513 steps / 254,164,992 presentations. Seeds 7–12 all completed. Original H1 matrix remains blocked.

| Seed | U6 | R6×2 | U12 | R6×2 − U12 | R6×2 − U6 |
|---:|---:|---:|---:|---:|---:|
| 7 | 5.883580 | 5.859483 | 5.850089 | +0.009394 | −0.024097 |
| 8 | 5.887879 | 5.861300 | 5.853415 | +0.007885 | −0.026580 |
| 9 | 5.880492 | 5.855362 | 5.851886 | +0.003476 | −0.025130 |
| 10 | 5.880047 | 5.855187 | 5.851729 | +0.003458 | −0.024860 |
| 11 | 5.889654 | 5.866393 | 5.850928 | +0.015464 | −0.023261 |
| 12 | 5.880896 | 5.862328 | 5.853363 | +0.008966 | −0.018567 |
| **Mean** | **5.883758** | **5.860009** | **5.851902** | **+0.008107** | **−0.023749** |

Mean R6×2 is **+0.81% perplexity versus U12** with 49.8% fewer unique body parameters, and **2.35% lower perplexity versus U6** at identical unique body capacity. Bonferroni paired t intervals (n=6): sharing cost **[+0.00234, +0.01387]** (excludes 0; contains δ=0.01); extra-compute gain **[−0.02733, −0.02017]** (does not contain 0). **Not an H1 pass.** Do not pool with C7.

All 18 metrics and full-state archives checksum-verified. U12-s12 is a same-identity recovery after remainder-pod idle terminate lost the first copy; not a replacement seed. Recovery pod `i7wfq4a4nci6fu` stopped/deleted; live query returned no pods. Recurring non-pod spend $0.024/h unchanged.

Released metrics: `artifacts/claim-a/`; [full result](reference/claim-a-results.md); [figure](reference/claim-a-review/quality-storage.png). Reproduce with `uv run python scripts/analyze_claim_a.py`.