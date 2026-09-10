#!/usr/bin/env bash
# Local, explicitly supplied OWT JSONL only. Never downloads a dataset/tokenizer.
# Usage: bash runpod/prepare_owt.sh /data/owt.jsonl /data/owt-prepared \
#   /assets/encoder.json /assets/vocab.bpe ENCODER_SHA256 VOCAB_SHA256 [sequence]
# Install pinned tiktoken==0.12.0 in your environment first. Obtain the original
# GPT-2 encoder.json/vocab.bpe independently, verify their authoritative source,
# and record their SHA256 values; caller pins do NOT prove official provenance.
# JSONL must have exactly {"id": "unique-document-id", "text": "..."} per row.
# Dedup is exact UTF-8 only, splits are hash-based, all contexts reset per window.
# Default fractions: .94 train, .02 development, .02 calibration, .02 confirmation.
# Outputs: manifest.json, per-split document manifests + uint32 mmap token bins.
# Reported splits differ intentionally from upstream validation conventions.
set -euo pipefail
if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo 'Usage: prepare_owt.sh LOCAL_JSONL OUT ENCODER_JSON VOCAB_BPE ENCODER_SHA256 VOCAB_SHA256 [SEQUENCE]' >&2
  exit 2
fi
PYTHON="${PYTHON:-python}"
exec "$PYTHON" -m rz1t.data prepare --input "$1" --out "$2" \
  --tokenizer gpt2 --encoder-json "$3" --vocab-bpe "$4" \
  --encoder-sha256 "$5" --vocab-sha256 "$6" --sequence "${7:-256}" --seed 0
