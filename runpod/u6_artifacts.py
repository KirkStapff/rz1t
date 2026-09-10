"""Provenance gate and bounded artifact bundles for the U6 development control."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tarfile


def check_identity(expected, data, source, environment):
    if data != expected['data']:
        raise ValueError('historical data identity mismatch; do not train')
    if source != expected['source']:
        raise ValueError('historical training source mismatch; do not train')
    # Host kernel/platform may differ; all recorded numerical settings must match.
    for key, value in expected['environment'].items():
        if key != 'platform' and environment.get(key) != value:
            raise ValueError(f'historical numerical environment mismatch: {key}')


def gate(expected_path, out):
    from rz1t.data import PreparedDataset
    from rz1t.train import environment_identity, source_identity
    from rz1t.checkpoint import atomic_json
    expected = json.loads(Path(expected_path).read_text())
    dataset = PreparedDataset('data/owt-subset-256/manifest.json', vocab=50257, sequence=256)
    actual = dict(data=dataset.identity, source=source_identity(), environment=environment_identity())
    check_identity(expected, **actual)
    atomic_json(out, dict(status='passed', **actual))


def bundle(root, name):
    from rz1t.checkpoint import file_hash, verify_checkpoint, atomic_json
    root = Path(root)
    run = root / name
    final = json.loads((run / 'final.json').read_text())
    if final['status'] != 'completed':
        raise ValueError('incomplete U6 run')
    pointer = final['checkpoint']
    checkpoint = run / 'checkpoints' / pointer['directory']
    identity = json.loads((run / 'identity.json').read_text())
    verify_checkpoint(checkpoint, identity, pointer['manifest_sha256'])
    # Metrics are transferred separately, before any large state transfer.
    for kind, files in (
        ('metrics', sorted(p for p in run.iterdir() if p.is_file())),
        ('state', sorted(p for p in checkpoint.iterdir() if p.is_file()) + [run / 'checkpoints' / 'latest.json']),
    ):
        archive = root / f'{name}-{kind}.tgz'
        with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
            for path in files:
                tar.add(path, arcname=str(path.relative_to(root)))
        atomic_json(str(archive) + '.sha256.json', {'sha256': file_hash(archive), 'bytes': archive.stat().st_size})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    g = sub.add_parser('gate')
    g.add_argument('--expected', required=True)
    g.add_argument('--out', required=True)
    b = sub.add_parser('bundle')
    b.add_argument('--root', required=True)
    b.add_argument('--name', required=True)
    a = p.parse_args()
    if a.action == 'gate':
        gate(a.expected, a.out)
    else:
        bundle(a.root, a.name)


if __name__ == '__main__':
    main()
