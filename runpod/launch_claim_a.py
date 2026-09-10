"""Bounded claim-a-v1: 18 runs on unused confirmation split. Not original H1."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from runpod import launch_profile as lp
from rz1t.checkpoint import file_hash
from rz1t.claim_a import FIRST, RECOVERY, REMAINING, write_manifest
from rz1t.train import source_identity

ROOT = lp.ROOT


def execute(args):
    if (args.remaining or args.only) and Path(args.work).exists():
        raise RuntimeError('recovery/remainder work directory must be new; do not overwrite archives')
    key = os.environ['RUNPOD_API_KEY']
    quotes = [lp.quote_gpu(key, community=c) for c in (True, False)]
    chosen = next((q for q in quotes if q['available'] and q['price_per_hour']), None)
    if lp.list_pods(key):
        raise RuntimeError('existing pods; refusing new allocation')
    if not chosen:
        raise RuntimeError('no quoted stock')
    estimate = chosen['price_per_hour'] * args.max_minutes / 60
    reserve = .5
    buffer = 1.0 if args.only else 5.0
    if estimate + reserve > args.spend_cap or chosen['account']['clientBalance'] < args.spend_cap + buffer:
        raise RuntimeError('budget admission failed (includes $0.50 reserve)')
    if args.only:
        planned = RECOVERY
    elif args.remaining:
        planned = REMAINING
    else:
        planned = FIRST
    result = dict(chosen={k: chosen[k] for k in ('community', 'price_per_hour', 'stock')},
                  account_before=chosen['account'], compute_cap_estimate=estimate,
                  spend_cap=args.spend_cap, max_minutes=args.max_minutes,
                  protocol='claim-a-v1',
                  remaining_only=bool(args.remaining),
                  recovery_only=bool(args.only),
                  runs=[f'{arm}-s{seed}' for arm, seed in planned],
                  confirmation_authorized=False, h1_authorized=False, status='dry_run')
    print(json.dumps(result), flush=True)
    if not args.execute:
        return result
    work = Path(args.work).resolve()
    work.mkdir(parents=True, exist_ok=False)
    write_manifest(work / 'manifest.json')
    keypath = work / 'id_ed25519'
    subprocess.run(['ssh-keygen', '-t', 'ed25519', '-N', '', '-f', str(keypath)], check=True, capture_output=True)
    lp.pack_repo(work / 'source.tar.gz')
    result['source_archive_sha256'] = hashlib.sha256((work / 'source.tar.gz').read_bytes()).hexdigest()
    data_path = ROOT/'results/u6-launch/data.tgz'
    expected_hash = json.loads(Path(str(data_path)+'.sha256.json').read_text())
    if file_hash(data_path) != expected_hash['sha256'] or data_path.stat().st_size != expected_hash['bytes']:
        raise ValueError('archived data checksum failed before allocation')
    expected = json.loads((ROOT/'reference/u6-historical-identity.json').read_text())
    expected['source'] = source_identity()
    (work/'expected.json').write_text(json.dumps(expected, indent=2))
    result['data_archive_sha256'] = expected_hash['sha256']
    pod = None
    started = time.time()
    deadline = started + args.max_minutes * 60
    def remaining(reserve=0):
        seconds = deadline - time.time() - reserve
        if seconds <= 0:
            raise TimeoutError('allocation deadline reached')
        return seconds
    def record():
        (work / 'launch.json').write_text(json.dumps(result, indent=2) + '\n')
    try:
        pod = lp.create_pod(key, name='rz1t-claim-a-v1', community=chosen['community'],
                            ssh_public=keypath.with_suffix('.pub').read_text(), disk_gb=80)
        result['pod_id'] = pod['id']
        (work / 'pod.json').write_text(json.dumps(pod, indent=2))
        record()
        actual_rate = pod.get('costPerHr')
        if actual_rate is not None and float(actual_rate)*args.max_minutes/60 + .5 > args.spend_cap:
            raise RuntimeError('actual pod price exceeds admission cap')
        _, host, port = lp.wait_ssh(key, pod['id'], keypath, min(deadline-600, time.time()+600))
        result['ssh'] = dict(host=host, port=port)
        record()
        opts = ['-i', str(keypath), '-P', str(port), '-o', 'StrictHostKeyChecking=accept-new',
                '-o', 'UserKnownHostsFile=/dev/null', '-o', 'IdentitiesOnly=yes',
                '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3']
        def scp(src, dest):
            subprocess.run(['scp', *opts, src, str(dest)], check=True, timeout=min(900, remaining(90)), capture_output=True)
        def remote(label, command, reserve=600):
            print(label, flush=True)
            try:
                p = lp.run_ssh(keypath, host, port, 'bash -lc ' + shlex.quote(command), timeout=remaining(reserve))
            except subprocess.CalledProcessError as exc:
                (work / (label+'.stdout')).write_text(exc.stdout or '')
                (work / (label+'.stderr')).write_text(exc.stderr or '')
                raise
            (work / (label+'.stdout')).write_text(p.stdout)
            (work / (label+'.stderr')).write_text(p.stderr)
        prefix = 'set -euo pipefail\ncd /workspace/rz1t\nexport PATH="$HOME/.local/bin:$PATH"\n'
        def pull_bundle(name):
            base = f'root@{host}:/workspace/rz1t/results/claim-a/{name}'
            scp(base+'.sha256.json', work/(name+'.sha256.json'))
            scp(base, work/name)
            expected_local = json.loads((work/(name+'.sha256.json')).read_text())
            digest = hashlib.sha256()
            with (work/name).open('rb') as f:
                for chunk in iter(lambda: f.read(1024*1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected_local['sha256'] or (work/name).stat().st_size != expected_local['bytes']:
                raise ValueError('download checksum mismatch: '+name)
            result.setdefault('verified_bundles', []).append(name)
            record()
        scp(str(work/'source.tar.gz'), f'root@{host}:/tmp/rz1t-src.tar.gz')
        scp(str(data_path), f'root@{host}:/tmp/claim-a-data.tgz')
        scp(str(work/'expected.json'), f'root@{host}:/tmp/claim-a-expected.json')
        remote('setup', '''set -euo pipefail
mkdir -p /workspace
tar -xzf /tmp/rz1t-src.tar.gz -C /workspace
cd /workspace/rz1t
mkdir -p results/claim-a data
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.12.14
uv venv --python 3.12.14 .venv
bash runpod/pod_setup.sh --cuda
nvidia-smi > results/claim-a/nvidia-smi.txt
.venv/bin/python - <<'PY'
import tarfile
with tarfile.open('/tmp/claim-a-data.tgz') as t:
    t.extractall('.', filter='data')
PY
JAX_PLATFORMS=cuda .venv/bin/python -m runpod.duration_artifacts --expected /tmp/claim-a-expected.json --out results/claim-a/provenance.json
''')
        scp(f'root@{host}:/workspace/rz1t/results/claim-a/provenance.json', work/'provenance.json')
        slugs = {'U6': 'u6', 'R6x2': 'r6x2', 'U12': 'u12'}
        for arm, seed in planned:
            name = f'{arm}-s{seed}'
            cfg = f'configs/claim-a/claim-a-v1-{slugs[arm]}-s{seed}.yaml'
            remote(name, prefix + (
                f'JAX_PLATFORMS=cuda .venv/bin/python -m rz1t.train --config {cfg} '
                f'--out results/claim-a/{name}\n'
                f'JAX_PLATFORMS=cpu .venv/bin/python -m runpod.u6_artifacts bundle '
                f'--root results/claim-a --name {name}\n'))
            pull_bundle(name+'-metrics.tgz')
            pull_bundle(name+'-state.tgz')
        result['status'] = 'completed_archived'
    except Exception as exc:
        result['status'] = 'failed_or_partial'
        result['error'] = f'{type(exc).__name__}: {exc}'.replace(key, '[REDACTED]')
        print(result['error'], flush=True)
    finally:
        if pod:
            result['terminate'] = lp.terminate_pod(key, pod['id'])
        result['elapsed_seconds'] = time.time()-started
        record()
        result['remaining_pods'] = [p['id'] for p in lp.list_pods(key)]
        record()
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--remaining', action='store_true',
                   help='train only seeds 10–12; keep completed seeds 7–9 in the registered matrix')
    p.add_argument('--only', choices=['U12-s12'],
                   help='recover one registered run after a lost archive; not a new seed')
    p.add_argument('--max-minutes', type=int, default=360)
    p.add_argument('--spend-cap', type=float, default=25)
    p.add_argument('--work', default=str(ROOT/'results/claim-a-launch'))
    a = p.parse_args()
    if a.only and a.remaining:
        p.error('use either --remaining or --only, not both')
    if a.only:
        if not (40 <= a.max_minutes <= 90 and 0 < a.spend_cap <= 8):
            p.error('U12-s12 recovery: 40–90 minutes, at most $8')
        if a.work == str(ROOT/'results/claim-a-launch'):
            a.work = str(ROOT/'results/claim-a-u12-s12')
    elif a.remaining:
        if not (60 <= a.max_minutes <= 360 and 0 < a.spend_cap <= 25):
            p.error('bounded remainder launch: 60–360 minutes, at most $25')
        if a.work == str(ROOT/'results/claim-a-launch'):
            a.work = str(ROOT/'results/claim-a-remaining')
    elif not (60 <= a.max_minutes <= 360 and 0 < a.spend_cap <= 25):
        p.error('bounded claim-a first wave: 60–360 minutes, at most $25')
    r = execute(a)
    return 0 if r['status'] in ('dry_run', 'completed_archived') else 2


if __name__ == '__main__':
    raise SystemExit(main())
