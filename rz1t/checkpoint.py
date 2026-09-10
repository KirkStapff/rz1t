"""Atomic, checksummed Equinox state; no pickle and no silent corrupt fallback.

A checkpoint is an immutable generation directory. The last write is an atomic
latest.json pointer containing the manifest checksum. Readers verify all bytes
before deserialising against an independently constructed tree template.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import equinox as eqx

FORMAT_VERSION = 1


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _sync_dir(path.parent)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _child(root, name):
    if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
        raise ValueError("invalid checkpoint artifact path")
    path = Path(root) / name
    if path.is_symlink():
        raise ValueError("checkpoint symlinks are forbidden")
    return path


def verify_checkpoint(directory, identity=None, manifest_sha256=None):
    directory = Path(directory)
    manifest_path = _child(directory, "manifest.json")
    try:
        if manifest_sha256 is not None and file_hash(manifest_path) != manifest_sha256:
            raise ValueError("checkpoint manifest checksum mismatch")
        manifest = json.loads(manifest_path.read_text())
        if manifest["format_version"] != FORMAT_VERSION:
            raise ValueError("unsupported checkpoint format")
        if identity is not None and manifest["identity"] != identity:
            raise ValueError("checkpoint identity mismatch (config/data/source/environment)")
        if set(manifest["files"]) != {"state.eqx", "metadata.json"}:
            raise ValueError("checkpoint artifact set mismatch")
        for name, sha in manifest["files"].items():
            if file_hash(_child(directory, name)) != sha:
                raise ValueError(f"checkpoint checksum mismatch: {name}")
        return manifest
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint: {exc}") from exc


def save_checkpoint(root, state, metadata, identity):
    """Store full model (including fixed arrays), optimizer and all RNG streams.

    Caller must hold the run lock. Metadata includes Python-integer counters,
    work totals, schedule/data cursor and the durable metrics prefix. Keep only
    the latest two *verified* generations; corruption blocks pruning.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = f"step-{metadata['step']:012d}-{uuid.uuid4().hex}"
    temp = Path(tempfile.mkdtemp(prefix=".writing-", dir=root))
    try:
        eqx.tree_serialise_leaves(temp / "state.eqx", state)
        with (temp / "state.eqx").open("rb") as handle:
            os.fsync(handle.fileno())
        atomic_json(temp / "metadata.json", metadata)
        manifest = {"format_version": FORMAT_VERSION, "identity": identity,
                    "step": metadata["step"], "files": {
                        name: file_hash(temp / name) for name in ("state.eqx", "metadata.json")}}
        atomic_json(temp / "manifest.json", manifest)
        verify_checkpoint(temp, identity)
        os.replace(temp, root / name)
        _sync_dir(root)
        pointer = {"directory": name, "manifest_sha256": file_hash(root / name / "manifest.json")}
        atomic_json(root / "latest.json", pointer)
        # Verify before deleting anything; never hide a corrupt prior state.
        generations = []
        for directory in root.glob("step-*"):
            prior = verify_checkpoint(directory, identity)
            generations.append((prior["step"], directory.stat().st_mtime_ns, directory))
        for _, _, directory in sorted(generations, reverse=True)[2:]:
            shutil.rmtree(directory)
        _sync_dir(root)
        return pointer
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def load_checkpoint(root, template, identity):
    """Return (restored_tree, metadata, pointer); reject corruption/mismatch."""
    root = Path(root)
    try:
        pointer = json.loads((root / "latest.json").read_text())
        directory = _child(root, pointer["directory"])
        verify_checkpoint(directory, identity, pointer["manifest_sha256"])
        metadata = json.loads((directory / "metadata.json").read_text())
        state = eqx.tree_deserialise_leaves(directory / "state.eqx", template)
        return state, metadata, pointer
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid checkpoint: {exc}") from exc


def write_completion(root, identity, checkpoint, record):
    """Completion is committed last, after verified checkpoint and final record."""
    root = Path(root)
    verify_checkpoint(_child(root / "checkpoints", checkpoint["directory"]), identity,
                      checkpoint["manifest_sha256"])
    atomic_json(root / "final.json", record)
    atomic_json(root / "completion.json", {
        "format_version": FORMAT_VERSION, "identity": identity,
        "checkpoint": checkpoint, "files": {"final.json": file_hash(root / "final.json")},
    })


def validate_completion(root, identity=None):
    root = Path(root)
    try:
        completion = json.loads((root / "completion.json").read_text())
        if completion["format_version"] != FORMAT_VERSION:
            raise ValueError("unsupported completion format")
        if identity is not None and completion["identity"] != identity:
            raise ValueError("completion identity mismatch")
        if set(completion["files"]) != {"final.json"}:
            raise ValueError("completion artifact set mismatch")
        for name, sha in completion["files"].items():
            if file_hash(_child(root, name)) != sha:
                raise ValueError("completion artifact checksum mismatch")
        pointer = completion["checkpoint"]
        verify_checkpoint(_child(root / "checkpoints", pointer["directory"]),
                          completion["identity"], pointer["manifest_sha256"])
        record = json.loads((root / "final.json").read_text())
        if record["status"] != "completed":
            raise ValueError("completion does not reference a completed result")
        return record
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid completion: {exc}") from exc
