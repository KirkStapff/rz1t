"""Queue tests use mocked subprocesses; no training or provider resources."""
import json
from pathlib import Path
import subprocess

import pytest
from runpod import queue, archive
from rz1t.checkpoint import atomic_json, file_hash, write_completion


@pytest.fixture
def job(tmp_path):
    config = tmp_path / 'config.yaml'
    config.write_text('lane: dev\nseed: 1\n')
    manifest = tmp_path / 'jobs.json'
    atomic_json(manifest, {'jobs': [{'config': str(config), 'out': str(tmp_path / 'run'), 'estimated_seconds': 10}]})
    return queue.load_jobs(manifest)[0]


@pytest.fixture
def terms():
    return dict(cap=20., reserve=5., storage=1., price=360., gpus=1)


def complete(job, *, failure=False):
    out = Path(job['out'])
    identity = {'config_sha256': job['config_sha256']}
    directory = out / 'checkpoints' / 'step-1'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'state.eqx').write_bytes(b'fake state, checksums tested without deserialization')
    atomic_json(directory / 'metadata.json', {'step': 1, 'model_failure': failure})
    atomic_json(directory / 'manifest.json', {'format_version': 1, 'identity': identity, 'step': 1,
        'files': {p: file_hash(directory / p) for p in ('state.eqx', 'metadata.json')}})
    pointer = {'directory': 'step-1', 'manifest_sha256': file_hash(directory / 'manifest.json')}
    atomic_json(out / 'identity.json', identity)
    atomic_json(out / 'config.json', {'lane': 'dev', 'seed': 1})
    atomic_json(out / 'metrics.json', [{'step': 1}])
    atomic_json(out / 'checkpoints' / 'latest.json', pointer)
    write_completion(out, identity, pointer, {'status': 'completed', 'config_sha256': job['config_sha256']})
    if failure:
        (out / 'completion.json').unlink()
        (out / 'final.json').unlink()  # crash before terminal record: metadata still blocks retry


def test_dry_run_no_side_effects(job, terms, tmp_path, monkeypatch):
    monkeypatch.setattr(queue.subprocess, 'run', lambda *a, **kw: pytest.fail('dry-run executed'))
    assert queue.run_queue([job], tmp_path / 'ledger.json', **terms)[0]['action'] == 'would_run'
    assert not Path(job['out']).exists()
    assert not (tmp_path / 'ledger.json').exists()


def test_completion_and_observed_cost(job, terms, tmp_path, monkeypatch):
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] == 10 and kwargs['pass_fds']
        complete(job)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(queue.subprocess, 'run', child)
    clock = iter([100., 103.])
    monkeypatch.setattr(queue.time, 'monotonic', lambda: next(clock, 104.))
    ledger = tmp_path / 'ledger.json'
    assert queue.run_queue([job], ledger, execute=True, **terms)[0]['action'] == 'completed'
    assert json.loads(ledger.read_text())['spent'] == pytest.approx(.3)
    assert queue.run_queue([job], ledger, execute=True, **terms)[0]['action'] == 'validated_skip'
    assert len(calls) == 1
    (Path(job['out']) / 'final.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        queue.run_queue([job], ledger, execute=True, **terms)


def test_interruption_resume(job, terms, tmp_path, monkeypatch):
    def interrupt(command, **kwargs):
        complete(job)
        (Path(job['out']) / 'completion.json').unlink()
        raise subprocess.TimeoutExpired(command, kwargs['timeout'])
    monkeypatch.setattr(queue.subprocess, 'run', interrupt)
    ledger = tmp_path / 'ledger.json'
    assert queue.run_queue([job], ledger, execute=True, **terms)[0]['action'] == 'infrastructure_interrupted'
    def resume(command, **kwargs):
        assert '--resume' in command
        complete(job)
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(queue.subprocess, 'run', resume)
    assert queue.run_queue([job], ledger, execute=True, **terms)[0]['action'] == 'completed'


def test_killed_worker_reconciliation(job, terms, tmp_path, monkeypatch):
    def killed(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(queue.subprocess, 'run', killed)
    ledger = tmp_path / 'ledger.json'
    with pytest.raises(KeyboardInterrupt):
        queue.run_queue([job], ledger, execute=True, **terms)
    attempt = json.loads(ledger.read_text())['attempts'][0]
    assert attempt['status'] == 'active'
    with pytest.raises(ValueError, match='unreconciled'):
        queue.run_queue([job], ledger, execute=True, **terms)
    queue.reconcile(ledger, attempt['id'], 2.)
    assert json.loads(ledger.read_text())['spent'] == 2.
    with pytest.raises(ValueError, match='only an active'):
        queue.reconcile(ledger, attempt['id'], 2.)


def test_admission_duplicate_locks_and_policy(job, terms, tmp_path, monkeypatch):
    monkeypatch.setattr(queue.subprocess, 'run', lambda *a, **kw: pytest.fail('must not execute'))
    assert queue.run_queue([job], tmp_path / 'ledger.json', execute=True,
                           **(terms | {'cap': 6.5}))[0]['action'] == 'budget_blocked'
    assert queue.run_queue([job], tmp_path / 'ledger.json', **(terms | {'spent': 14.}))[0]['action'] == 'budget_blocked'
    with queue.locked(Path(job['out']) / '.queue.lock'):
        with pytest.raises(ValueError, match='another worker'):
            queue.run_queue([job], tmp_path / 'ledger.json', execute=True, **terms)


def test_failure_cost_and_no_success_only_retry(job, terms, tmp_path, monkeypatch):
    monkeypatch.setattr(queue.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess(a, 2))
    ledger = tmp_path / 'ledger.json'
    result = queue.run_queue([job], ledger, execute=True, **terms)
    assert result[0]['action'] == 'failed' and result[0]['actual_cost'] > 0
    with pytest.raises(ValueError, match='not an infrastructure'):
        queue.run_queue([job], ledger, execute=True, **terms)
    with pytest.raises(ValueError, match='terms mismatch'):
        queue.run_queue([job], ledger, execute=True, **(terms | {'cap': 100}))


def test_metadata_failure_and_resume_corruption(job, terms, tmp_path):
    complete(job, failure=True)
    with pytest.raises(ValueError, match='terminal model failure'):
        queue.run_queue([job], tmp_path / 'ledger.json', execute=True, **terms)
    (Path(job['out']) / 'checkpoints/step-1/state.eqx').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        queue.run_queue([job], tmp_path / 'ledger.json', execute=True, **terms)


def test_duplicate_manifest_and_nan(job, terms, tmp_path):
    manifest = tmp_path / 'duplicates.json'
    row = {k: job[k] for k in ('config', 'out', 'estimated_seconds')}
    atomic_json(manifest, {'jobs': [row, row]})
    with pytest.raises(ValueError, match='duplicate'):
        queue.load_jobs(manifest)
    with pytest.raises(ValueError, match='finite'):
        queue.run_queue([job], tmp_path / 'ledger.json', **(terms | {'cap': float('nan')}))


def test_archive_verified_no_pruning(tmp_path, job):
    complete(job)
    source, dest = Path(job['out']), tmp_path / 'backup'
    result = archive.archive_run(source, dest)
    assert archive.verify_archive(dest) == result
    assert (source / 'checkpoints/step-1/state.eqx').exists()
    with pytest.raises(ValueError, match='destination exists'):
        archive.archive_run(source, dest)
    (dest / 'metrics.json').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        archive.verify_archive(dest)


def test_archive_locks_and_symlinks(tmp_path, job):
    complete(job)
    source = Path(job['out'])
    with queue.locked(source / '.train.lock'):
        with pytest.raises(ValueError, match='another worker'):
            archive.archive_run(source, tmp_path / 'backup')
    (source / 'link').symlink_to('/tmp')
    with pytest.raises(ValueError, match='symlinks'):
        archive.archive_run(source, tmp_path / 'backup')
    assert not (tmp_path / 'backup').exists()


def test_archive_changed_copy_never_published(tmp_path, monkeypatch, job):
    complete(job)
    source = Path(job['out'])
    original = archive.shutil.copy2
    def changed(src, dst, *args, **kwargs):
        result = original(src, dst, *args, **kwargs)
        Path(dst).write_bytes(b'wrong')
        return result
    monkeypatch.setattr(archive.shutil, 'copy2', changed)
    with pytest.raises(ValueError, match='changed'):
        archive.archive_run(source, tmp_path / 'backup')
    assert not (tmp_path / 'backup').exists()
    assert not list(tmp_path.glob('.archive-*'))
