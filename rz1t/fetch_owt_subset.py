"""Write a local OpenWebText JSONL *subset*. Not full OWT, not confirmation data.

    uv run python -m rz1t.fetch_owt_subset --n-docs 80000 --out data/owt-subset.jsonl

Uses HuggingFace `Skylion007/openwebtext` streaming. Records source/revision and
the exact document count. Does not download tokenizer assets. Does not fetch
the full ~40 GB corpus.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASET = "Skylion007/openwebtext"


def fetch_subset(out: Path, n_docs: int, revision: str | None = None) -> dict:
    if n_docs < 1000:
        raise ValueError("n_docs must be at least 1000 for a usable development subset")
    if out.exists():
        raise ValueError(f"refusing to overwrite {out}")
    from datasets import load_dataset

    kwargs = {"split": "train", "streaming": True}
    if revision:
        kwargs["revision"] = revision
    stream = load_dataset(DATASET, **kwargs)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with out.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(stream):
            if written >= n_docs:
                break
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                skipped += 1
                continue
            record = {"id": f"owt-stream-{index:08d}", "text": text}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    if written < n_docs:
        raise ValueError(f"stream ended after {written} documents; wanted {n_docs}")
    meta = {
        "dataset": DATASET,
        "revision": revision,
        "n_docs": written,
        "skipped_empty": skipped,
        "selection": "first N non-empty streaming train documents",
        "out": str(out),
        "not_full_openwebtext": True,
        "not_confirmation_corpus": True,
    }
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-docs", type=int, default=80_000)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args(argv)
    meta = fetch_subset(Path(args.out), args.n_docs, args.revision)
    print(json.dumps(meta, indent=2), flush=True)
    # HuggingFace datasets/xet can SIGABRT during interpreter finalization.
    # The JSONL is already durable; skip atexit hooks.
    import os as _os
    _os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
