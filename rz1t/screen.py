"""Cheap architecture screen: not confirmation, not H1, not a seal.

Prepares local JSONL (never fetches corpora) and trains matched explore arms
on the development split. Default corpus is reference/tinyshakespeare/documents.jsonl.

    python -m rz1t.screen --out results/shakespeare-screen
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import yaml

from rz1t.config import ModelConfig
from rz1t.data import prepare_jsonl
from rz1t.train import train

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = ROOT / "reference" / "tinyshakespeare" / "documents.jsonl"

# Applied depth 12 except the literature-style prelude/core/coda (applied 10).
ARMS = (
    {"arm": "U12", "p": 0, "m": 12, "k": 1, "q": 0, "injection": "none"},
    {"arm": "R6x2", "p": 0, "m": 6, "k": 2, "q": 0, "injection": "none"},
    {"arm": "R6x2-residual", "p": 0, "m": 6, "k": 2, "q": 0, "injection": "residual"},
    {"arm": "R2x6-residual", "p": 0, "m": 2, "k": 6, "q": 0, "injection": "residual"},
    {"arm": "P1M2K4Q1-residual", "p": 1, "m": 2, "k": 4, "q": 1, "injection": "residual"},
)


def chunk_text(text: str, *, target=2048, minimum=129) -> list[str]:
    """Split on blank lines, then pack so each document has at least `minimum` chars."""
    if not text:
        raise ValueError("empty corpus")
    parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n") if p.strip()]
    if not parts:
        parts = [text.strip()]
    documents = []
    buf = ""
    for part in parts:
        candidate = f"{buf}\n\n{part}" if buf else part
        if len(candidate) < target:
            buf = candidate
            continue
        if buf and len(buf) >= minimum:
            documents.append(buf)
            buf = part
        else:
            documents.append(candidate)
            buf = ""
    if buf:
        if documents and len(buf) < minimum:
            documents[-1] = f"{documents[-1]}\n\n{buf}"
        else:
            documents.append(buf)
    documents = [d for d in documents if len(d) >= minimum]
    if not documents:
        raise ValueError("no document reached the minimum length")
    return documents


def write_jsonl(documents: list[str], path: Path, *, prefix="doc") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for index, text in enumerate(documents):
            stream.write(json.dumps({"id": f"{prefix}-{index:04d}", "text": text}, ensure_ascii=True) + "\n")
    return path


def arm_config(*, arm, vocab, sequence, n_embed, aft_heads, aft_ksize, linear_fan_in,
               steps, batch_size, eval_every, checkpoint_every, lr, warmup, seed,
               manifest, p, m, k, q, injection):
    ModelConfig(vocab=vocab, sequence=sequence, n_embed=n_embed, p=p, m=m, k=k, q=q,
                aft_heads=aft_heads, aft_ksize=aft_ksize, linear_fan_in=linear_fan_in,
                injection=injection)
    return {
        "lane": "explore",
        "seed": seed,
        "arm": arm,
        "model": {
            "vocab": vocab, "sequence": sequence, "n_embed": n_embed,
            "aft_heads": aft_heads, "aft_ksize": aft_ksize, "linear_fan_in": linear_fan_in,
            "p": p, "m": m, "k": k, "q": q, "injection": injection,
        },
        "data": {"kind": "prepared", "manifest": str(manifest), "evaluation_split": "development"},
        "training": {
            "steps": steps, "batch_size": batch_size,
            "eval_every": eval_every, "checkpoint_every": checkpoint_every,
        },
        "optimizer": {"lr": lr, "warmup": warmup},
    }


def run_screen(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    documents = Path(args.documents)
    if not documents.is_file():
        raise SystemExit(f"missing local documents JSONL: {documents}")
    prepared = Path(args.prepared)
    if prepared.exists():
        manifest = prepared / "manifest.json"
    else:
        manifest = prepare_jsonl(documents, prepared, sequence=args.sequence,
                                 tokenizer="char", seed=args.data_seed)
    meta = json.loads(Path(manifest).read_text())
    vocab = meta["tokenizer"]["vocab_size"]
    summary = {
        "purpose": "explore architecture screen; not confirmation; not H1",
        "documents": str(documents),
        "prepared": str(prepared),
        "manifest_sha256": meta.get("manifest_sha256"),
        "vocab": vocab,
        "sequence": args.sequence,
        "splits": {name: {k: v[k] for k in ("documents", "tokens", "windows")}
                   for name, v in meta["splits"].items()},
        "n_embed": args.n_embed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "started": time.time(),
        "runs": [],
    }
    (out / "screen.json").write_text(json.dumps(summary, indent=2) + "\n")
    for spec in ARMS:
        config = arm_config(
            arm=spec["arm"], vocab=vocab, sequence=args.sequence, n_embed=args.n_embed,
            aft_heads=args.aft_heads, aft_ksize=args.aft_ksize, linear_fan_in=args.linear_fan_in,
            steps=args.steps, batch_size=args.batch_size, eval_every=args.eval_every,
            checkpoint_every=args.checkpoint_every, lr=args.lr, warmup=args.warmup,
            seed=args.seed, manifest=manifest, **{k: spec[k] for k in ("p", "m", "k", "q", "injection")},
        )
        run_dir = out / spec["arm"]
        cfg_path = out / f"{spec['arm']}.yaml"
        cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))
        record = train(config, run_dir)
        summary["runs"].append(record)
        (out / "screen.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    summary["elapsed_seconds"] = time.time() - summary["started"]
    ranking = sorted(
        (r for r in summary["runs"] if r.get("status") == "completed" and r.get("nll") is not None),
        key=lambda r: r["nll"],
    )
    summary["ranking"] = [{"arm": r.get("arm"), "nll": r["nll"], "step": r["step"]} for r in ranking]
    (out / "screen.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS))
    parser.add_argument("--prepared", default=str(ROOT / "data" / "shakespeare-char-128"))
    parser.add_argument("--out", default=str(ROOT / "results" / "shakespeare-screen"))
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--n-embed", type=int, default=64)
    parser.add_argument("--aft-heads", type=int, default=4)
    parser.add_argument("--aft-ksize", type=int, default=4)
    parser.add_argument("--linear-fan-in", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    args = parser.parse_args(argv)
    run_screen(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
