# Original H1 checklist — not sealed

This file records the **unexecuted** three-budget non-inferiority proposal.
It is **not** a preregistration of the paper result.

The published study is claim-A (`configs/claim-a/`, unused confirmation split,
n=6, one token budget). That study estimates a quality–storage trade-off; it
does **not** authorize or pass H1 at δ = 0.01. The trainer still rejects
`lane: sealed` / `lane: confirmation`.

## Original intended question (not the paper)

Candidate (not frozen): U12 versus R6×2 at applied depth 12, c=4, width 384 and
sequence 256, `injection=none`. Proposed non-inferiority margin 0.01 nats/token.
Three candidate semantic budgets 1e16, 3e16, 1e17 using an unaudited counting
proxy.

Executed measurements: `EXPERIMENT_LOG.md` and `reference/claim-a-results.md`.

## Before enabling confirmation

- [ ] Pin OWT/tokenizer provenance, disjoint document manifests and deduplication policy; lock confirmation corpus.
- [ ] Freeze target environment/precision and verify GPU parity, full-run memory, runtime, checkpoint/storage costs.
- [ ] Audit semantic operations/backward convention against hand counts and supported XLA analysis.
- [ ] Give both model families equal LR tuning opportunities on development data.
- [ ] Run independent calibration pairs; freeze conservative variance/power selection, joint-support assumptions and common seed count.
- [ ] Justify practical margin without confirmation inspection.
- [ ] Freeze budget schedules, initialization/data stream pairing, final-checkpoint selection, exclusions, failure rules and stopping policy.
- [ ] Implement reviewed manifest promotion linking final config/source/data hashes to expected run identities; do not reuse stale draft hashes.
- [ ] Implement a measured cost table and explicit remaining authorization/reserve audit.
- [ ] Independently review the confirmation access/promotion guard and analysis protocol.
- [ ] Freeze generated manifest, analysis code and protocol with checksums; create a git tag only with owner authorization.

Until all are satisfied, all development/calibration outputs remain exploratory. Post-result redesign requires fresh independent confirmation rather than relabeling a selected result. Missing or divergent seeds are never replaced until success.