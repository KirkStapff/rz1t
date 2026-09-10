"""Offline fixtures only: no dataset or tokenizer downloads."""
import hashlib
import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from rz1t.data import (SPLITS, PreparedDataset, _hash, file_hash, load_dataset,
                       main, prepare_jsonl, split_for_hash)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / 'documents.jsonl'
    rows = [{'id': f'doc-{i:04}', 'text': f'{i:04} abcdefghijklmnopqrstuvwxyz ' * 3} for i in range(120)]
    rows += [{'id': 'duplicate', 'text': rows[0]['text']}]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


def prepare(source, out):
    return prepare_jsonl(source, out, sequence=7, fractions=(.25, .25, .25, .25))


def dataset(path):
    meta = json.loads(path.read_text())
    return PreparedDataset(path, vocab=meta['tokenizer']['vocab_size'], sequence=7)


def test_split_dedup_document_windows_and_deterministic_eval(source, tmp_path):
    path = prepare(source, tmp_path / 'prepared')
    meta = json.loads(path.read_text())
    assert meta['duplicates_removed'] == 1
    ds = dataset(path)
    all_ids, all_hashes = set(), set()
    for split in SPLITS:
        docs = [json.loads(line) for line in (path.parent / f'{split}.jsonl').read_text().splitlines()]
        ids, hashes = {d['id'] for d in docs}, {d['content_sha256'] for d in docs}
        assert not (all_ids & ids or all_hashes & hashes)
        all_ids |= ids
        all_hashes |= hashes
        assert all(split_for_hash(d['content_sha256'], 0, (.25,) * 4) == split for d in docs)
        batches = list(ds.eval_batches(13, 7, split))
        xs = np.concatenate([np.asarray(x) for x, _ in batches])
        ys = np.concatenate([np.asarray(y) for _, y in batches])
        np.testing.assert_array_equal(xs[:, 1:], ys[:, :-1])
        assert len(xs) == meta['splits'][split]['windows']
        # Every token window lies wholly inside one document; tails discarded.
        expected = []
        for doc in docs:
            for start in range(doc['offset'], doc['offset'] + doc['windows'] * 8, 8):
                expected.append(ds.arrays[split][start:start + 8])
        np.testing.assert_array_equal(np.c_[xs, ys[:, -1]], np.asarray(expected))
        again = list(ds.eval_batches(13, 7, split))
        for (x, y), (xx, yy) in zip(batches, again):
            np.testing.assert_array_equal(x, xx)
            np.testing.assert_array_equal(y, yy)
    x, y = ds.get_batch(jax.random.PRNGKey(19), 5, 7)
    xx, yy = ds.get_batch(jax.random.PRNGKey(19), 5, 7)
    np.testing.assert_array_equal(x, xx)
    np.testing.assert_array_equal(y, yy)
    assert isinstance(ds.arrays['train'], np.memmap)


def test_order_independence_and_provenance(source, tmp_path):
    first = prepare(source, tmp_path / 'first')
    reverse = tmp_path / 'reverse.jsonl'
    reverse.write_text('\n'.join(reversed(source.read_text().splitlines())) + '\n')
    second = prepare(reverse, tmp_path / 'second')
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    assert a['splits'] == b['splits']
    assert a['tokenizer'] == b['tokenizer']
    assert a['source_sha256'] != b['source_sha256']
    assert a['tokenizer']['fitted_on'] == 'train'
    assert a['manifest_sha256'] == dataset(first).identity['manifest_sha256']
    with pytest.raises(ValueError, match='already exists'):
        prepare(source, first.parent)


def test_synthetic_deterministic_and_strict_configs():
    model = SimpleNamespace(sequence=7, vocab=11)
    config = {'kind': 'synthetic', 'seed': 0, 'train_windows': 16, 'eval_windows': 4}
    a, b = load_dataset(config, model), load_dataset(config, model)
    assert a.identity == b.identity
    assert a.identity != load_dataset({**config, 'seed': 1}, model).identity
    batches = list(a.eval_batches(3, 7))
    assert [x.shape for x, _ in batches] == [(3, 7), (1, 7)]
    assert load_dataset({'dataset_name': 'synthetic'}, model).identity == a.identity
    for bad in ({'kind': 'other'}, {'kind': 'synthetic', 'path': 'foo'},
                {'kind': 'synthetic', 'seed': -1}, {'kind': 'synthetic', 'train_windows': True},
                {'kind': 'prepared'}, {'kind': 'prepared', 'manifest': 'x', 'download': True}):
        with pytest.raises(ValueError):
            load_dataset(bad, model)
    with pytest.raises(ValueError, match='sequence'):
        a.get_batch(jax.random.PRNGKey(0), 2, 8)
    with pytest.raises(ValueError, match='unknown split'):
        a.get_batch(jax.random.PRNGKey(0), 2, 7, 'validation')


def test_corruption_and_runtime_mutation(source, tmp_path):
    manifest = prepare(source, tmp_path / 'prepared')
    ds = dataset(manifest)
    tokens = manifest.parent / 'train.bin'
    content = bytearray(tokens.read_bytes())
    content[0] ^= 1
    tokens.write_bytes(content)
    with pytest.raises(ValueError, match='mutated'):
        ds.get_batch(jax.random.PRNGKey(0), 1, 7)
    with pytest.raises(ValueError, match='checksum'):
        dataset(manifest)
    meta = json.loads(manifest.read_text())
    meta['sequence'] = 8
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='checksum'):
        dataset(manifest)


def test_overlap_rejected_even_with_recomputed_checksums(source, tmp_path):
    manifest = prepare(source, tmp_path / 'prepared')
    meta = json.loads(manifest.read_text())
    train_docs = manifest.parent / 'train.jsonl'
    dev_docs = manifest.parent / 'development.jsonl'
    train_doc = json.loads(train_docs.read_text().splitlines()[0])
    dev = [json.loads(line) for line in dev_docs.read_text().splitlines()]
    dev[0]['id'] = train_doc['id']
    dev_docs.write_text(''.join(json.dumps(d) + '\n' for d in dev))
    meta['splits']['development']['documents_sha256'] = file_hash(dev_docs)
    meta.pop('manifest_sha256')
    meta['manifest_sha256'] = _hash(meta)
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='overlap'):
        dataset(manifest)


def test_id_conflict_empty_and_local_gpt2_only(tmp_path, monkeypatch):
    import urllib.request
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **k: pytest.fail('network prohibited'))
    source = tmp_path / 'bad.jsonl'
    source.write_text('{"id":"same","text":"abc"}\n{"id":"same","text":"xyz"}\n')
    with pytest.raises(ValueError, match='conflicting'):
        prepare(source, tmp_path / 'bad')
    assert not (tmp_path / 'bad').exists()
    source.write_text('')
    with pytest.raises(ValueError, match='no documents'):
        prepare(source, tmp_path / 'empty')
    source.write_text('{"id":"one","text":"abc"}\n')
    with pytest.raises(ValueError, match='local tokenizer assets'):
        prepare_jsonl(source, tmp_path / 'gpt', sequence=2, tokenizer='gpt2')


def test_cli_and_prepared_model_compatibility(source, tmp_path):
    assert main(['prepare', '--input', str(source), '--out', str(tmp_path / 'cli'),
                 '--sequence', '7', '--tokenizer', 'char', '--fractions', '.25', '.25', '.25', '.25']) == 0
    path = tmp_path / 'cli' / 'manifest.json'
    meta = json.loads(path.read_text())
    config = {'kind': 'prepared', 'manifest': str(path)}
    model = SimpleNamespace(sequence=7, vocab=meta['tokenizer']['vocab_size'])
    ds = load_dataset(config, model)
    assert ds.window_counts['confirmation'] > 0
    with pytest.raises(ValueError, match='vocab'):
        load_dataset(config, SimpleNamespace(sequence=7, vocab=999))
    with pytest.raises(ValueError, match='sequence'):
        load_dataset(config, SimpleNamespace(sequence=8, vocab=model.vocab))


def test_char_vocab_fits_train_only_and_short_documents(tmp_path):
    # Construct exact split placement, then reserve a character for heldout text.
    rows = []
    for split in SPLITS:
        for i in range(10000):
            text = (('abc' if split == 'train' else '\u03a9') * 4) + str(i)
            digest = hashlib.sha256(text.encode()).hexdigest()
            if split_for_hash(digest, fractions=(.25,) * 4) == split:
                rows.append({'id': split, 'text': text})
                break
    source = tmp_path / 'chars.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    path = prepare(source, tmp_path / 'chars')
    meta = json.loads(path.read_text())
    assert '\u03a9' not in meta['tokenizer']['alphabet']
    ds = dataset(path)
    assert 0 in ds.arrays['development']
    assert ds.window_counts['development'] == 0
    assert list(ds.eval_batches(2, 7)) == []
    with pytest.raises(ValueError, match='no full windows'):
        ds.get_batch(jax.random.PRNGKey(0), 1, 7, 'development')
