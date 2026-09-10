"""Checksum-verified local archive, no pruning and no cloud provisioning.

Destination must be a separately managed durable filesystem. Copying onto the
pod's ephemeral disk is NOT backup. Quiesce training first; both run locks are
held while copying. Completed or interrupted verified checkpoints are accepted.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from rz1t.checkpoint import atomic_json, file_hash, validate_completion, verify_checkpoint
from runpod.queue import locked


def verify_archive(path):
    path = Path(path)
    if any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError('archive symlinks forbidden')
    manifest = json.loads((path / 'archive_manifest.json').read_text())
    if manifest.get('version') != 1:
        raise ValueError('unsupported archive version')
    actual = {str(p.relative_to(path)) for p in path.rglob('*') if p.is_file() and p.name != 'archive_manifest.json'}
    if actual != set(manifest['files']):
        raise ValueError('archive artifact set mismatch')
    for name, digest in manifest['files'].items():
        child = path / name
        if Path(name).is_absolute() or '..' in Path(name).parts or child.is_symlink():
            raise ValueError('invalid archive path')
        if file_hash(child) != digest:
            raise ValueError(f'archive checksum mismatch: {name}')
    identity = json.loads((path / 'identity.json').read_text())
    pointer = json.loads((path / 'checkpoints/latest.json').read_text())
    name = pointer['directory']
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('invalid checkpoint directory')
    verify_checkpoint(path / 'checkpoints' / name, identity, pointer['manifest_sha256'])
    if (path / 'completion.json').exists():
        validate_completion(path, identity)
    return manifest


def archive_run(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('source and archive must be disjoint')
    if destination.exists():
        raise ValueError('archive destination exists; verify it separately, never overwrite')
    with locked(source / '.queue.lock', nonblocking=True), locked(source / '.train.lock', nonblocking=True):
        if any(p.is_symlink() for p in source.rglob('*')):
            raise ValueError('archive source symlinks forbidden')
        for name in ('identity.json', 'config.json', 'metrics.json', 'checkpoints/latest.json'):
            if not (source / name).is_file():
                raise ValueError(f'missing archive source artifact: {name}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp = Path(tempfile.mkdtemp(prefix='.archive-', dir=destination.parent))
        try:
            for child in source.iterdir():
                if child.name.startswith('.'):
                    continue  # Do not copy advisory locks.
                if child.is_dir():
                    shutil.copytree(child, temp / child.name)
                elif child.is_file():
                    shutil.copy2(child, temp / child.name)
            files = {}
            for child in temp.rglob('*'):
                if child.is_file():
                    name = str(child.relative_to(temp))
                    digest = file_hash(child)
                    if digest != file_hash(source / name):
                        raise ValueError('source changed during archive')
                    with child.open('rb') as handle:
                        os.fsync(handle.fileno())
                    files[name] = digest
            atomic_json(temp / 'archive_manifest.json', {'version': 1, 'source': str(source), 'files': files})
            verify_archive(temp)
            # A destination-specific lock prevents two writers replacing one another.
            with locked(str(destination) + '.archive.lock', nonblocking=True):
                if destination.exists():
                    raise ValueError('archive destination appeared during copy')
                os.rename(temp, destination)
                fd = os.open(destination.parent, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            return verify_archive(destination)
        finally:
            if temp.exists():
                shutil.rmtree(temp)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source')
    parser.add_argument('--destination', required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            record = verify_archive(args.destination)
        elif args.source:
            record = archive_run(args.source, args.destination)
        else:
            raise ValueError('--source required for copying')
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(record, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
