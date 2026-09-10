# Sources and evidence ledger

## Verified local source

- Origin: https://github.com/extropic-ai/sparse-transformers
- Full source commit: `13051e90df9669be5b8f9f34fb097329fa82f674`.
- `upstream/` created using `git archive` of that commit, including its license. This is a vendored snapshot, not a git subtree with imported history.
- Package metadata lives at `upstream/research/z1t/pyproject.toml`; root package metadata does not exist upstream.
- Canonical `z1t.components` and `z1t.model` supply blocks, initialization, sparse indices and wrapping. The new model shares one stored core.
- Source PE applies stop_gradient but upstream optimizer filters all inexact arrays, allowing AdamW decay on PE. rz1t excludes PE explicitly. Parity tests compare equivalent frozen-PE partitions; this is not exact upstream trainer replication.

## Local validation

CPU Python 3.12 environment resolves through root `uv.lock`. Installed upstream and root package; ran model/trainer/data/accounting/analysis tests and synthetic training CLI, completion validation/resume and archival. See RESULTS.md for final test count. Run identities retain actual source hashes, package versions, device and precision.

## Measured H100 sessions (not a published hardware constant)

- 2026-09-07 Secure Cloud 1×H100 SXM.
- Session A: batch 1–8 ~82.9k tokens/s. Local profile JSON under gitignored `results/`.
- Session B: batch 16–128, plateau at batch 64 ~129–130k tokens/s.
- CUDA extra resolved on Linux H100: jax 0.11.1 + jax-cuda12-plugin 0.11.1. This does not validate XLA cost analysis.

## Inherited quotations, not verified hardware constants

The planning document quotes Z1 1.3e-14 J/sample, FPGA 0.2 pJ/matmul op, 3 pJ/scalar op, 1.5 W static, and a 294.52 nJ/token reference. These have **not** been independently audited here. Latency/static-power amortization, sample definition, classifier inclusion and concurrency remain unresolved. No external hardware reference or published frontier has been reproduced. `energy_model.py` accepts explicitly supplied scenarios without claiming validation.

## Assumptions and unresolved work

- Semantic-proxy-v1 uses itemized forward operations and a 2× forward backward estimate; no XLA/backward validation or measured executed FLOPs yet.
- Data splits use exact-content hashes and document-isolated windows. No near-duplicate decontamination. GPT-2 local asset hashes are caller-supplied pins (`19613966…` / `1ce16647…`). C2 used a 40 000-document **streaming subset** of `Skylion007/openwebtext` (unpinned revision), not full OWT. Confirmation corpus is unused and unlocked.
- Tiny Shakespeare source: Karpathy `char-rnn` `input.txt`, vendored at `reference/tinyshakespeare/input.txt` (1 115 394 chars) and packed to `documents.jsonl` (606 docs). Not a citation of a published LM result.
- No HF weights were used. No post archive or digitized figures are included.
- Cost ceilings in lab notes are historical planning limits, not current account state.