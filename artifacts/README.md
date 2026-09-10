# Released metrics

Checksummed `*-metrics.tgz` archives only. Full-state checkpoints are local
under `results/` (gitignored) and are not required to regenerate the paper
figures.

| Directory | What | Runs |
|---|---|---|
| `claim-a/` | Confirmation-holdout paper result (C8) | 18 |
| `duration/` | One-seed development duration diagnostic (C6) | 3 |
| `estimate/` | Calibration-holdout estimate (C7), labeled history | 9 |

Hashes: [`MANIFEST.json`](MANIFEST.json).

```bash
uv run python scripts/analyze_claim_a.py
uv run python scripts/make_paper_figures.py
uv run python scripts/analyze_duration.py
```
