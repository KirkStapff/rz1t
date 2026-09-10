# Engineering checks

These are software tests, not the language-model result. The paper numbers are in
[`reference/claim-a-results.md`](reference/claim-a-results.md).

- `uv sync --frozen --python 3.12 --extra dev --extra prepare`
- `JAX_PLATFORMS=cpu uv run pytest -q`
- `uv run ruff check rz1t runpod tests --select E9,F63,F7,F82`
- Upstream CLI smoke: `python -m z1t.train --config upstream/configs/z1t_smoke.yaml`
- Recurrent CLI smoke: `python -m rz1t.train --config configs/dev/smoke.yaml --out results/smoke`

The parity optimizer freezes positional encoding for both models. That is an
intentional correction to the upstream all-inexact-leaf AdamW partition, not
exact upstream trainer equivalence.

Vendored upstream files match the pinned source bytes in
[`reference/upstream-files.json`](reference/upstream-files.json).

H100 synthetic throughput (fp32, d=384, seq=256): batch 64 plateau
~129–130k tokens/s. Details: [`reference/cost_table.md`](reference/cost_table.md).
