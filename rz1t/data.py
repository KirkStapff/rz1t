"""Offline document-level data preparation and immutable random-window datasets.

Prepare local JSONL {"id": str, "text": str} with ``python -m rz1t.data prepare
--input docs.jsonl --out data/dev --tokenizer char --sequence 256``.
Char vocabulary is fitted ONLY on training documents (0 is unknown). GPT-2
requires tiktoken==0.12.0 and both original local GPT-2 tokenizer assets plus
explicit SHA256 pins; nothing is downloaded. Exact-content deduplication is
not semantic/near-duplicate decontamination. Every window resets context/PE;
stride is sequence+1 and incomplete document suffixes are discarded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sqlite3
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

SPLITS = ('train', 'development', 'calibration', 'confirmation')
VERSION = 'rz1t-documents-v1'
WINDOW_POLICY = 'document-local-stride-sequence-plus-one-discard-tail-reset-context'
DEFAULT_FRACTIONS = (0.94, 0.02, 0.02, 0.02)
TIKTOKEN_VERSION = '0.12.0'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def _seed(value):
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError('seed must be a uint32 integer')
    return value


def _json_write(path, value):
    Path(path).write_bytes(_canonical(value) + b'\n')


def split_for_hash(content_hash, seed=0, fractions=DEFAULT_FRACTIONS):
    _seed(seed)
    if (len(fractions) != 4 or any(not math.isfinite(f) or f <= 0 for f in fractions)
            or not math.isclose(sum(fractions), 1., abs_tol=1e-12)):
        raise ValueError('four positive split fractions must sum to one')
    # Integer comparison avoids float-rounding an extreme hash up to one.
    number = int(_hash([VERSION, seed, content_hash]), 16)
    total = 0.
    for split, fraction in zip(SPLITS, fractions):
        total += fraction
        if number < int(total * 2**256):
            return split
    return SPLITS[-1]


def _gpt2_tokenizer(encoder_json, vocab_bpe, encoder_sha256, vocab_sha256):
    if not all((encoder_json, vocab_bpe, encoder_sha256, vocab_sha256)):
        raise ValueError('GPT-2 requires two local tokenizer assets and their SHA256 pins')
    assets = {}
    for name, path, expected in (('encoder_json', encoder_json, encoder_sha256),
                                 ('vocab_bpe', vocab_bpe, vocab_sha256)):
        if not Path(path).is_file() or file_hash(path) != expected:
            raise ValueError(f'local GPT-2 {name} checksum mismatch or missing file')
        assets[name] = expected
    if importlib.metadata.version('tiktoken') != TIKTOKEN_VERSION:
        raise ValueError(f'GPT-2 preparation requires tiktoken=={TIKTOKEN_VERSION}')
    import tiktoken
    from tiktoken.load import data_gym_to_mergeable_bpe_ranks
    # Local files only: do not call get_encoding(), which can fetch remote assets.
    ranks = data_gym_to_mergeable_bpe_ranks(
        str(Path(vocab_bpe).resolve()), str(Path(encoder_json).resolve()))
    if len(ranks) != 50256 or set(ranks.values()) != set(range(50256)):
        raise ValueError('GPT-2 assets must contain exactly 50256 mergeable token ranks')
    pattern = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    encoding = tiktoken.Encoding(name='gpt2-local-pinned', pat_str=pattern,
                                mergeable_ranks=ranks, special_tokens={'<|endoftext|>': 50256})
    provenance = {'kind': 'gpt2', 'implementation': 'tiktoken', 'version': TIKTOKEN_VERSION,
                  'assets_sha256': assets, 'mergeable_ranks_sha256':
                  _hash(sorted((key.hex(), value) for key, value in ranks.items())),
                  'pattern': pattern, 'special_token_policy': 'encode_ordinary; no EOS appended',
                  'vocab_size': 50257, 'asset_authority': 'caller-supplied pins; not independently authenticated'}
    return encoding.encode_ordinary, provenance


def prepare_jsonl(input_path, out, *, sequence, tokenizer='char', seed=0,
                  fractions=DEFAULT_FRACTIONS, encoder_json=None, vocab_bpe=None,
                  encoder_sha256=None, vocab_sha256=None):
    """Stream input through a disk SQLite dedup spool, then write mmap token bins.

    Publication is an atomic directory rename; existing outputs are never replaced.
    Sorting by content hash makes split/windows independent of input order. Each
    duplicate content retains the lexicographically smallest ID. Repeated IDs
    with different contents are rejected. No normalization of text is performed.
    """
    _positive(sequence, 'sequence')
    _seed(seed)
    split_for_hash('validation', seed, fractions)
    if tokenizer not in ('char', 'gpt2'):
        raise ValueError('unsupported tokenizer (choose char or gpt2)')
    if tokenizer == 'char' and any(value is not None for value in (
            encoder_json, vocab_bpe, encoder_sha256, vocab_sha256)):
        raise ValueError('GPT-2 asset options are unsupported for the char tokenizer')
    input_path, out = Path(input_path), Path(out)
    if out.exists():
        raise ValueError('output already exists; preparation never overwrites frozen data')
    source_hash = file_hash(input_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f'.{out.name}-', dir=out.parent))
    db = None
    try:
        db = sqlite3.connect(temp / 'spool.sqlite')
        db.execute('CREATE TABLE ids (id TEXT PRIMARY KEY, hash TEXT NOT NULL)')
        db.execute('CREATE TABLE docs (hash TEXT PRIMARY KEY, id TEXT, text TEXT, split TEXT)')
        n_input = 0
        with input_path.open(encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                if (not isinstance(item, dict) or set(item) != {'id', 'text'}
                        or not isinstance(item['id'], str) or not item['id']
                        or not isinstance(item['text'], str)):
                    raise ValueError('each JSONL row must contain exactly string id and text')
                digest = hashlib.sha256(item['text'].encode('utf-8')).hexdigest()
                previous = db.execute('SELECT hash FROM ids WHERE id=?', (item['id'],)).fetchone()
                if previous and previous[0] != digest:
                    raise ValueError('duplicate document id has conflicting content')
                db.execute('INSERT OR IGNORE INTO ids VALUES (?,?)', (item['id'], digest))
                db.execute('INSERT INTO docs VALUES (?,?,?,?) ON CONFLICT(hash) DO UPDATE SET id=min(id,excluded.id)',
                           (digest, item['id'], item['text'], split_for_hash(digest, seed, fractions)))
                n_input += 1
                if n_input % 10000 == 0:
                    db.commit()
        db.commit()
        if not n_input:
            raise ValueError('input has no documents')
        if file_hash(input_path) != source_hash:
            raise ValueError('input changed during preparation')
        if tokenizer == 'char':
            chars = set()
            for (text,) in db.execute("SELECT text FROM docs WHERE split='train'"):
                chars.update(text)
            alphabet = sorted(chars)
            if not alphabet:
                raise ValueError('no training characters; provide more documents or change split seed')
            vocabulary = {char: index + 1 for index, char in enumerate(alphabet)}
            encode = lambda text: [vocabulary.get(char, 0) for char in text]
            provenance = {'kind': 'char', 'version': 1, 'alphabet': alphabet,
                          'unknown_id': 0, 'fitted_on': 'train', 'normalization': 'none',
                          'vocab_size': len(alphabet) + 1, 'special_token_policy': 'none'}
        else:
            encode, provenance = _gpt2_tokenizer(encoder_json, vocab_bpe, encoder_sha256, vocab_sha256)
        split_meta = {}
        unique = 0
        for split in SPLITS:
            tokens_path, docs_path = temp / f'{split}.bin', temp / f'{split}.jsonl'
            offset = windows = count = dropped = 0
            with tokens_path.open('wb') as tokens, docs_path.open('wb') as documents:
                for digest, doc_id, text in db.execute(
                        'SELECT hash,id,text FROM docs WHERE split=? ORDER BY hash', (split,)):
                    ids = np.asarray(encode(text), dtype='<u4')
                    ids.tofile(tokens)
                    n_windows = len(ids) // (sequence + 1)
                    record = {'id': doc_id, 'content_sha256': digest, 'offset': offset,
                              'tokens': len(ids), 'windows': n_windows}
                    documents.write(_canonical(record) + b'\n')
                    offset += len(ids)
                    windows += n_windows
                    dropped += len(ids) % (sequence + 1)
                    count += 1
            split_meta[split] = {'tokens_file': tokens_path.name, 'tokens_sha256': file_hash(tokens_path),
                                 'documents_file': docs_path.name, 'documents_sha256': file_hash(docs_path),
                                 'tokens': offset, 'documents': count, 'windows': windows,
                                 'discarded_tail_tokens': dropped}
            unique += count
        manifest = {'version': VERSION, 'sequence': sequence, 'window_policy': WINDOW_POLICY,
                    'dtype': '<u4', 'token_weighting': 'all full-window targets equally',
                    'split_policy': {'algorithm': 'sha256-content-and-seed-v1', 'seed': seed,
                                     'fractions': list(fractions), 'dedup': 'exact UTF-8 content; no normalization'},
                    'source_sha256': source_hash, 'input_documents': n_input,
                    'unique_documents': unique, 'duplicates_removed': n_input - unique,
                    'tokenizer': provenance, 'splits': split_meta}
        manifest['manifest_sha256'] = _hash(manifest)
        _json_write(temp / 'manifest.json', manifest)
        db.close()
        db = None
        (temp / 'spool.sqlite').unlink()
        temp.rename(out)
        return out / 'manifest.json'
    finally:
        if db is not None:
            db.close()
        if temp.exists():
            shutil.rmtree(temp)


class WindowDataset:
    """Random training windows; deterministic exhaustive evaluation, no padding.

    A final partial *batch* is included; partial *windows* are never included.
    Dataset methods are host-side input plumbing (not intended to be jitted).
    """
    def _validate_request(self, batch_size, sequence, split):
        _positive(batch_size, 'batch_size')
        _positive(sequence, 'sequence')
        if sequence != self.sequence:
            raise ValueError('sequence must match prepared window length')
        if split not in SPLITS:
            raise ValueError(f'unknown split: {split}')
        self._check_unchanged()

    def _check_unchanged(self):
        pass

    def get_batch(self, key, batch_size, sequence, split='train'):
        self._validate_request(batch_size, sequence, split)
        size = self.window_counts[split]
        if not size:
            raise ValueError(f'{split} has no full windows')
        indices = np.asarray(jax.random.randint(key, (batch_size,), 0, size))
        rows = self._rows(split, indices)
        return jnp.asarray(rows[:, :-1], dtype=jnp.int32), jnp.asarray(rows[:, 1:], dtype=jnp.int32)

    def eval_batches(self, batch_size, sequence, split='development'):
        self._validate_request(batch_size, sequence, split)
        for start in range(0, self.window_counts[split], batch_size):
            self._check_unchanged()
            rows = self._rows(split, np.arange(start, min(start + batch_size, self.window_counts[split])))
            yield jnp.asarray(rows[:, :-1], dtype=jnp.int32), jnp.asarray(rows[:, 1:], dtype=jnp.int32)


class SyntheticDataset(WindowDataset):
    def __init__(self, *, vocab, sequence, seed=0, train_windows=16, eval_windows=4):
        _seed(seed)
        _positive(train_windows, 'train_windows')
        _positive(eval_windows, 'eval_windows')
        self.sequence = _positive(sequence, 'sequence')
        _positive(vocab, 'vocab')
        self.window_counts = {split: train_windows if split == 'train' else eval_windows for split in SPLITS}
        self.arrays = {}
        for index, split in enumerate(SPLITS):
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, index])))
            self.arrays[split] = rng.integers(0, vocab, (self.window_counts[split], sequence + 1), dtype=np.int32)
        self.identity = {'kind': 'synthetic', 'version': VERSION, 'seed': seed, 'sequence': sequence,
                         'vocab': vocab, 'window_counts': self.window_counts,
                         'purpose': 'offline infrastructure smoke; not language-model evidence',
                         'array_hashes': {s: hashlib.sha256(a.tobytes()).hexdigest() for s, a in self.arrays.items()}}

    def _rows(self, split, indices):
        return self.arrays[split][indices]


def _signature(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class PreparedDataset(WindowDataset):
    def __init__(self, manifest_path, *, vocab, sequence):
        path = Path(manifest_path).resolve()
        manifest = json.loads(path.read_text())
        claimed = manifest.pop('manifest_sha256', None)
        if claimed != _hash(manifest):
            raise ValueError('manifest checksum mismatch')
        if (manifest.get('version') != VERSION or manifest.get('window_policy') != WINDOW_POLICY
                or manifest.get('dtype') != '<u4' or set(manifest.get('splits', {})) != set(SPLITS)):
            raise ValueError('unsupported manifest schema/window policy')
        if sequence != manifest['sequence']:
            raise ValueError('sequence must match prepared window length')
        if vocab != manifest['tokenizer']['vocab_size']:
            raise ValueError('model vocab must exactly match tokenizer vocab_size')
        self.sequence = sequence
        self.identity = {'kind': 'prepared', 'version': VERSION, 'manifest_sha256': claimed,
                         'source_sha256': manifest['source_sha256'], 'tokenizer': manifest['tokenizer']}
        self.window_counts, self.arrays, self.offsets, self.cumulative = {}, {}, {}, {}
        self._signatures = {path: _signature(path)}
        seen_ids, seen_hashes = set(), set()
        used_paths = {path}
        for split in SPLITS:
            meta = manifest['splits'][split]
            paths = {}
            for kind in ('tokens', 'documents'):
                candidate = (path.parent / meta[f'{kind}_file']).resolve()
                if candidate.parent != path.parent or candidate in used_paths:
                    raise ValueError('manifest file escape or overlapping split files')
                used_paths.add(candidate)
                before = _signature(candidate)
                if file_hash(candidate) != meta[f'{kind}_sha256'] or _signature(candidate) != before:
                    raise ValueError(f'{split} {kind} checksum mismatch')
                self._signatures[candidate] = before
                paths[kind] = candidate
            offsets, counts = [], []
            offset = total = docs = discarded = 0
            with paths['documents'].open() as stream:
                for line in stream:
                    doc = json.loads(line)
                    digest = doc['content_sha256']
                    if doc['id'] in seen_ids or digest in seen_hashes:
                        raise ValueError('document id/content overlap across manifest')
                    seen_ids.add(doc['id'])
                    seen_hashes.add(digest)
                    policy = manifest['split_policy']
                    if split_for_hash(digest, policy['seed'], policy['fractions']) != split:
                        raise ValueError('document hash split mismatch')
                    if (type(doc['tokens']) is not int or doc['tokens'] < 0 or doc['offset'] != offset
                            or doc['windows'] != doc['tokens'] // (sequence + 1)):
                        raise ValueError('invalid document offsets/windows')
                    if doc['windows']:
                        offsets.append(offset)
                        counts.append(doc['windows'])
                    offset += doc['tokens']
                    total += doc['windows']
                    discarded += doc['tokens'] % (sequence + 1)
                    docs += 1
            if (offset != meta['tokens'] or total != meta['windows'] or docs != meta['documents']
                    or discarded != meta['discarded_tail_tokens']
                    or paths['tokens'].stat().st_size != offset * 4):
                raise ValueError('manifest count/size mismatch')
            array = np.memmap(paths['tokens'], dtype='<u4', mode='r') if offset else np.empty(0, dtype='<u4')
            for start in range(0, offset, 1048576):
                if np.any(array[start:start + 1048576] >= vocab):
                    raise ValueError('token outside tokenizer vocabulary')
            self.arrays[split] = array
            self.window_counts[split] = total
            self.offsets[split] = np.asarray(offsets, dtype=np.int64)
            self.cumulative[split] = np.cumsum(counts, dtype=np.int64)
        if len(seen_ids) != manifest['unique_documents']:
            raise ValueError('unique document count mismatch')
        self._check_unchanged()

    def _check_unchanged(self):
        # Full cryptographic verification occurs at load/resume; inode/ctime/mtime
        # guards detect subsequent ordinary mutation without rehashing OWT per step.
        for path, signature in self._signatures.items():
            if not path.exists() or _signature(path) != signature:
                raise ValueError('prepared dataset mutated after verification; reload required')

    def _rows(self, split, indices):
        cumulative = self.cumulative[split]
        documents = np.searchsorted(cumulative, indices, side='right')
        previous = np.where(documents == 0, 0, cumulative[np.maximum(documents - 1, 0)])
        starts = self.offsets[split][documents] + (indices - previous) * (self.sequence + 1)
        return self.arrays[split][starts[:, None] + np.arange(self.sequence + 1)]


def load_dataset(config_data, model_config):
    if not isinstance(config_data, dict):
        raise ValueError('data config must be a mapping')
    config = dict(config_data)
    # Compatibility with the trainer's initial smoke default only.
    if 'dataset_name' in config:
        if 'kind' in config or config.pop('dataset_name') != 'synthetic':
            raise ValueError('dataset_name is only supported as the legacy synthetic alias')
        config['kind'] = 'synthetic'
    kind = config.pop('kind', None)
    split = config.pop('evaluation_split', 'development')
    if split not in SPLITS[1:]:
        raise ValueError('evaluation_split must be development/calibration/confirmation')
    if kind == 'synthetic':
        if set(config) - {'seed', 'train_windows', 'eval_windows'}:
            raise ValueError('unknown/unimplemented synthetic data options')
        return SyntheticDataset(vocab=model_config.vocab, sequence=model_config.sequence, **config)
    if kind == 'prepared':
        if set(config) != {'manifest'}:
            raise ValueError('prepared data requires only manifest (plus evaluation_split)')
        return PreparedDataset(config['manifest'], vocab=model_config.vocab, sequence=model_config.sequence)
    raise ValueError('unsupported data kind (choose synthetic or prepared)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='prepare local JSONL; never downloads data/tokenizer')
    prep.add_argument('--input', required=True)
    prep.add_argument('--out', required=True)
    prep.add_argument('--sequence', type=int, required=True)
    prep.add_argument('--tokenizer', choices=('char', 'gpt2'), required=True)
    prep.add_argument('--seed', type=int, default=0)
    prep.add_argument('--fractions', type=float, nargs=4, default=DEFAULT_FRACTIONS,
                      metavar=('TRAIN', 'DEV', 'CALIBRATION', 'CONFIRMATION'))
    for name in ('encoder-json', 'vocab-bpe', 'encoder-sha256', 'vocab-sha256'):
        prep.add_argument('--' + name)
    args = vars(parser.parse_args(argv))
    args.pop('command')
    args['input_path'] = args.pop('input')
    try:
        result = prepare_jsonl(**args)
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError) as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
