# H100 environment / cost table

**Status:** two synthetic 1×H100 SXM environment profiles. Not H1 evidence, not a full-run forecast, not a sealed cost authorization.

**Accounting:** `semantic-proxy-v1` (itemized forward ops + 2×-forward backward estimate). Throughput below is **steady training-step wall time after compile**, on **synthetic random tokens**, with **no dataloader, evaluation, checkpoint I/O, or rematerialization**. Do not treat semantic-FLOP/s as measured device FLOPs.

## Session A — batch 1–8 (2026-09-07)

| Field | Value |
|---|---|
| Pod | `vubj90ubomzrbh` (`rz1t-h100-profile`) |
| GPU | NVIDIA H100 80GB HBM3, Secure Cloud, US-NE-1 |
| Image | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` |
| Quoted / billed rate | **$3.49 / GPU-hour** (Secure; Community H100 was out of stock) |
| Created | 2026-09-07 19:49:47 UTC |
| Stopped / deleted | 2026-09-07 20:23:59 UTC |
| Allocated uptime | **34.2 min ≈ 0.570 h** |
| Allocated compute charge | 0.570 × 3.49 ≈ **$1.99** |
| Software | Python 3.12.14, JAX/jaxlib 0.11.1, CUDA plugin 0.11.1, driver 580.126.09 |
| Data | synthetic `randint` tokens; width 384, seq 256, vocab 50257, fp32, `injection=none`, `remat=false` |

Idle GPU time before the profile (SSH/image wait after create) dominates this bill.

## Session B — batch 16–128 (2026-09-07)

| Field | Value |
|---|---|
| Pod | `07i6u6m7q1zmoq` (`rz1t-h100-batch`) |
| GPU | NVIDIA H100 80GB HBM3, Secure Cloud, AP-IN-1 |
| Image | same as session A |
| Rate | **$3.49 / GPU-hour** |
| Created | 2026-09-07 20:59:58 UTC |
| Stopped / deleted | after 246 s launcher elapsed; profile JAX work ~110 s |
| Prior aborted idle pod | SSH mapping bug; stopped/deleted with no profile |

No OOM at batch 128. Peak memory: U12 24.8 GiB / R6×2 52.0 GiB of ~80 GiB. Recurrence uses more activation memory at large batch; it is still not the 80 GiB limit.

## Steady-step measurements

Tokens/s = `batch × 256 / mean_step_seconds`. U12 batch 4 mean is polluted by one 86.6 ms outlier; median is also reported.

| Arm | batch | compile s | mean step s | tokens/s | semantic FLOP/s proxy | peak bytes_in_use |
|---|---:|---:|---:|---:|---:|---:|
| U12 | 1 | 9.49 | 0.01043 | 24 553 | 2.90e12 | 1.56 GiB |
| U12 | 2 | 7.36 | 0.01246 | 41 089 | 4.85e12 | 1.86 GiB |
| U12 | 4 | 7.78 | 0.02552 (median 0.01697) | 40 132 (median 60 338) | 4.74e12 | 2.22 GiB |
| U12 | 8 | 7.55 | 0.02471 | **82 892** | **9.79e12** | 2.95 GiB |
| U12 | 16 | 7.94 | 0.03860 | 106 101 | 1.25e13 | 4.28 GiB |
| U12 | 32 | 8.38 | 0.06349 | 129 036 | 1.52e13 | 7.12 GiB |
| U12 | 64 | 9.36 | 0.12667 | **129 348** | **1.53e13** | 12.83 GiB |
| U12 | 128 | 12.29 | 0.37591 | 87 169 | 1.03e13 | 24.77 GiB |
| R6×2 | 1 | 6.13 | 0.01064 | 24 058 | 2.84e12 | 2.95 GiB |
| R6×2 | 2 | 5.84 | 0.01245 | 41 136 | 4.86e12 | 2.95 GiB |
| R6×2 | 4 | 6.09 | 0.01673 | 61 217 | 7.23e12 | 2.95 GiB |
| R6×2 | 8 | 5.89 | 0.02470 | **82 907** | **9.79e12** | 4.63 GiB |
| R6×2 | 16 | 7.79 | 0.03866 | 105 955 | 1.25e13 | 7.48 GiB |
| R6×2 | 32 | 8.62 | 0.06419 | 127 612 | 1.51e13 | 13.71 GiB |
| R6×2 | 64 | 8.74 | 0.12583 | **130 212** | **1.54e13** | 26.68 GiB |
| R6×2 | 128 | 10.72 | 0.32506 | 100 807 | 1.19e13 | **51.99 GiB** |

At matched applied depth, U12 and R6×2 step times agree through batch 64. Recurrence did **not** show a large extra time cost until batch 128, where R6×2 is faster but uses ~2× the activation memory.

**Working training batch: 64.** Throughput plateaus 32→64 (~1.56× batch 8, not 3–4×) and **falls at 128**. Do not use 128 for cost forecasts.

Parameter counts: U12 body 204 409 vs R6×2 body 102 589; classifier ~19.35 M and embedding ~19.30 M dominate both.

## Implied compute-only cost

Using the slower primary arm at the chosen batch and $3.49/h. **Compute-only, no overhead.**

| Unit | Batch 8 (9.79e12) | Batch 64 (1.53e13, U12) |
|---|---:|---:|
| One 1e16 run | 0.284 h / $0.99 | 0.182 h / **$0.63** |
| One 3e16 run | 0.852 h / $2.97 | 0.546 h / **$1.90** |
| One 1e17 run | 2.839 h / $9.91 | 1.819 h / **$6.35** |
| One U12+R6×2 pair, all three budgets | 7.95 h / $27.74 | 5.09 h / **$17.78** |
| Calibration 4 pairs (24 runs) | 31.8 h / **$111** | 20.4 h / **$71** |
| Confirmation n=8 (48 runs, 2.24e18) | 63.6 h / **$222** | 40.7 h / **$142** |
| Confirmation n=12 | 95.4 h / $333 | 61.1 h / $213 |
| Confirmation n=16 | 127.1 h / $444 | 81.5 h / $284 |

Batch 64 saves about **36%** versus batch 8. It does **not** bring n=8 under the $75 hope from linear scaling. The sparse width-384 body saturates well below the plan’s 100 TFLOPS assumption (~15 TFLOPS of the semantic proxy at best).

Historical planning notes used a $600 lab ceiling. Those numbers are compute-only lower bounds, not a current authorization or account balance.

## What this does not measure

- Full-run tokens/s including first-step compile per process, evaluation, checkpoint write, data pipeline, and host idle.
- OWT I/O or GPT-2 tokenization.
- XLA cost analysis or backward-pass validation of `semantic-proxy-v1`.
- bf16/fp8, remat, gradient accumulation, or multi-GPU.
- Hardware nJ/token (energy audit still unresolved).
- A100 / 4090 SKU comparison.

## Next measurement (still M0/M1, not confirmation)

Optional: one short 1×H100 **bf16** profile at batch 64/128, both arms, then terminate. Or a 10-min A100/4090 SKU profile. Do not download OWT or enable sealed/confirmation lanes on that pod.

Until a full-run overhead measurement exists, treat **$142 / 48-run at batch 64** as the working compute-only lower bound on Secure H100 at $3.49/h.