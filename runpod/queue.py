"""Local, serial trainer queue. Dry-run by default; never provisions/terminates pods.

Manifest: {"jobs": [{"config": "configs/dev/smoke.yaml", "out":
"results/smoke", "estimated_seconds": 300}]}. Paths are relative to the
current working directory. Use one shared --ledger across all workers charged
to a cap. Budget terms are immutable once the ledger exists. Each job has a
hard subprocess timeout equal to its reservation. A worker killed before
settlement leaves an active reservation that blocks all admissions until an
operator reconciles provider billing and the ledger (never automatically).

Cost covers allocated GPUs times job wall time (validation plus child execution),
not provider startup, time outside job attempts, idle, or teardown;
include those external charges in --spent and --storage before starting a new
ledger, or reconcile the existing ledger offline. This is admission control,
not a provider-enforced spending cap. No cloud lifecycle actions are taken.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import uuid

import yaml
from rz1t.checkpoint import atomic_json, canonical_hash, validate_completion, verify_checkpoint


@contextmanager
def locked(path, *, nonblocking=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0))
        except BlockingIOError as exc:
            raise ValueError(f'another worker holds {path}') from exc
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric')
    if value < 0 or (not zero and value == 0):
        raise ValueError(f'{name} out of range')
    return value


def load_jobs(path):
    manifest = yaml.safe_load(Path(path).read_text())
    if not isinstance(manifest, dict) or set(manifest) != {'jobs'} or not isinstance(manifest['jobs'], list):
        raise ValueError('manifest must contain only jobs list')
    jobs, seen = [], set()
    for row in manifest['jobs']:
        if not isinstance(row, dict) or set(row) != {'config', 'out', 'estimated_seconds'}:
            raise ValueError('each job requires config, out, estimated_seconds')
        job = dict(row)
        job['config'] = str(Path(job['config']).resolve(strict=True))
        job['out'] = str(Path(job['out']).resolve())
        positive(job['estimated_seconds'], 'estimated_seconds')
        if job['out'] in seen:
            raise ValueError('duplicate output run identity')
        seen.add(job['out'])
        config = yaml.safe_load(Path(job['config']).read_text())
        if not isinstance(config, dict):
            raise ValueError('config must be a mapping')
        job['config_sha256'] = canonical_hash(config)
        jobs.append(job)
    return jobs


def _read_ledger(path, terms):
    if path.exists():
        ledger = json.loads(path.read_text())
        if ledger.get('version') != 1 or ledger.get('terms') != terms:
            raise ValueError('ledger budget terms mismatch; reconcile offline, never reset to evade cap')
        positive(ledger['spent'], 'ledger spent', zero=True)
        if not isinstance(ledger['attempts'], list):
            raise ValueError('invalid ledger attempts')
        return ledger
    return {'version': 1, 'terms': terms, 'spent': terms['spent'], 'attempts': []}


def _completed(job):
    out = Path(job['out'])
    if not (out / 'completion.json').exists():
        return False
    identity = json.loads((out / 'identity.json').read_text())
    if identity.get('config_sha256') != job['config_sha256']:
        raise ValueError('completion config identity mismatch')
    record = validate_completion(out, identity)
    if record.get('config_sha256') != job['config_sha256']:
        raise ValueError('completion record config mismatch')
    return True


def _resume(job):
    out = Path(job['out'])
    latest = out / 'checkpoints' / 'latest.json'
    if not latest.exists():
        if (out / 'identity.json').exists():
            raise ValueError('partial run missing resume state; manual recovery required')
        return False
    identity = json.loads((out / 'identity.json').read_text())
    if identity.get('config_sha256') != job['config_sha256']:
        raise ValueError('resume config identity mismatch')
    pointer = json.loads(latest.read_text())
    name = pointer['directory']
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('invalid checkpoint path')
    directory = latest.parent / name
    if directory.is_symlink():
        raise ValueError('checkpoint symlink forbidden')
    verify_checkpoint(directory, identity, pointer['manifest_sha256'])
    if json.loads((directory / 'metadata.json').read_text()).get('model_failure'):
        raise ValueError('terminal model failure; do not replace this seed')
    return True


def reconcile(ledger_path, attempt_id, actual_cost):
    """Operator must first stop the allocation and verify its full bill."""
    positive(actual_cost, 'actual_cost', zero=True)
    ledger_path = Path(ledger_path)
    with locked(str(ledger_path) + '.lock'):
        state = json.loads(ledger_path.read_text())
        _read_ledger(ledger_path, state['terms'])
        attempt = next((a for a in state['attempts'] if a['id'] == attempt_id), None)
        if attempt is None or attempt['status'] != 'active':
            raise ValueError('only an active interrupted attempt may be reconciled')
        with locked(Path(attempt['out']) / '.queue.lock', nonblocking=True):
            state['spent'] += actual_cost
            attempt.update(status='infrastructure_interrupted', actual_cost=actual_cost,
                           reconciled_unix=time.time())
            atomic_json(ledger_path, state)


def run_queue(jobs, ledger_path, *, cap, reserve, price, storage, gpus, spent=0., execute=False):
    for name, value in [('cap', cap), ('price', price)]:
        positive(value, name)
    for name, value in [('reserve', reserve), ('storage', storage), ('spent', spent)]:
        positive(value, name, zero=True)
    if type(gpus) is not int or gpus < 1:
        raise ValueError('gpus must be a positive integer count of allocated GPUs charged to this worker')
    terms = dict(cap=cap, reserve=reserve, price=price, storage=storage, gpus=gpus, spent=spent)
    ledger_path = Path(ledger_path)
    rate = gpus * price / 3600
    results = []
    if not execute:
        # No locks, output directories, ledger, or subprocesses on dry-run.
        ledger = _read_ledger(ledger_path, terms)
        available = cap - reserve - storage - ledger['spent']
        blocked = any(a['status'] == 'active' for a in ledger['attempts'])
        for job in jobs:
            estimate = job['estimated_seconds'] * rate
            admitted = not blocked and estimate <= available
            results.append(dict(out=job['out'], action='would_run' if admitted else 'budget_blocked',
                                reserved_cost=estimate))
            if admitted:
                available -= estimate
        return results
    for job in jobs:
        out = Path(job['out'])
        with locked(out / '.queue.lock', nonblocking=True) as claim:
            allocated_start = time.monotonic()
            if canonical_hash(yaml.safe_load(Path(job['config']).read_text())) != job['config_sha256']:
                raise ValueError('config changed after manifest loading')
            # Never silently replace a terminal model failure.
            if (out / 'final.json').exists():
                record = json.loads((out / 'final.json').read_text())
                if record.get('status') == 'model_failure':
                    raise ValueError('terminal model failure; do not replace this seed')
            if _completed(job):
                results.append(dict(out=str(out), action='validated_skip'))
                continue
            resume = _resume(job)
            with locked(str(ledger_path) + '.lock'):
                ledger = _read_ledger(ledger_path, terms)
                # Active reservations are fail-closed, even if the worker lock
                # has disappeared: its final billing is unknown.
                if any(a['status'] == 'active' for a in ledger['attempts']):
                    raise ValueError('active/unreconciled reservation blocks admission')
                previous = [a for a in ledger['attempts'] if a['out'] == str(out)]
                if any(a['config_sha256'] != job['config_sha256'] for a in previous):
                    raise ValueError('run config identity changed')
                if previous and previous[-1]['status'] not in ('infrastructure_interrupted',):
                    raise ValueError('prior attempt is not an infrastructure interruption; reconcile explicitly')
                reservation = job['estimated_seconds'] * rate
                if ledger['spent'] + reserve + storage + reservation > cap:
                    results.append(dict(out=str(out), action='budget_blocked'))
                    break
                attempt_id = uuid.uuid4().hex
                ledger['attempts'].append(dict(id=attempt_id, out=str(out),
                    config_sha256=job['config_sha256'], status='active',
                    reserved_cost=reservation, started_unix=time.time()))
                atomic_json(ledger_path, ledger)
            command = [sys.executable, '-m', 'rz1t.train', '--config', job['config'], '--out', str(out)]
            if resume:
                command.append('--resume')
            started = allocated_start
            # Unexpected exceptions/worker death intentionally retain active
            # reservation. Timeout is safe to settle: run() kills and waits.
            with (out / 'queue.log').open('ab') as log:
                try:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=job['estimated_seconds'], check=False,
                                             pass_fds=(claim.fileno(),))
                    returncode = process.returncode
                    status = 'infrastructure_interrupted' if returncode < 0 or returncode in (124, 137, 143) else 'failed'
                except subprocess.TimeoutExpired:
                    returncode, status = 124, 'infrastructure_interrupted'
                except OSError:
                    returncode, status = 127, 'infrastructure_interrupted'
            validation_error = None
            if returncode == 0:
                try:
                    if not _completed(job):
                        raise ValueError('trainer exited without validated completion')
                    status = 'completed'
                except (ValueError, OSError, KeyError) as exc:
                    validation_error = str(exc)
            elapsed = time.monotonic() - started
            with locked(str(ledger_path) + '.lock'):
                ledger = _read_ledger(ledger_path, terms)
                attempt = next(a for a in ledger['attempts'] if a['id'] == attempt_id)
                attempt.update(status=status, observed_seconds=elapsed, actual_cost=elapsed * rate,
                               returncode=returncode, validation_error=validation_error)
                ledger['spent'] += elapsed * rate
                atomic_json(ledger_path, ledger)
            results.append(dict(out=str(out), action=status, actual_cost=elapsed * rate))
            if status != 'completed':
                break  # Resume on explicit next invocation, not success-only retries.
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--ledger', required=True)
    for name in ('cap', 'reserve', 'price', 'storage'):
        parser.add_argument('--' + name, type=float, required=True)
    parser.add_argument('--spent', type=float, default=0., help='pre-ledger external allocation charges')
    parser.add_argument('--gpus', type=int, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--reconcile-attempt')
    parser.add_argument('--actual-cost', type=float)
    parser.add_argument('--attest-allocation-stopped', action='store_true')
    args = vars(parser.parse_args(argv))
    try:
        attempt = args.pop('reconcile_attempt')
        actual = args.pop('actual_cost')
        attested = args.pop('attest_allocation_stopped')
        if attempt:
            if not args['execute'] or not attested or actual is None:
                raise ValueError('reconciliation requires --execute --actual-cost --attest-allocation-stopped')
            reconcile(args['ledger'], attempt, actual)
            print(json.dumps({'reconciled': attempt}))
            return 0
        jobs = load_jobs(args.pop('manifest'))
        result = run_queue(jobs, args.pop('ledger'), **args)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return int(any(row['action'] in ('failed', 'infrastructure_interrupted', 'budget_blocked') for row in result))


if __name__ == '__main__':
    raise SystemExit(main())
