"""Derive bounded duration launcher from the verified U6 transport implementation."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
s = (ROOT/'runpod/launch_u6.py').read_text()
s = s.replace('Bounded U6 seeds 0–2: exact historical identity gate, verified pulls, cleanup.', 'Bounded duration-v1 three-arm diagnostic; archived data and verified pulls.')
s = s.replace('from runpod.launch_owt_pair import ENCODER_SHA, VOCAB_SHA', 'from rz1t.checkpoint import file_hash\nfrom rz1t.train import source_identity')
s = s.replace("runs=['U6-s0', 'U6-s1', 'U6-s2']", "runs=['U6-s3', 'R6x2-s3', 'U12-s3']")
s = s.replace("    pod = None\n", """    data_path = ROOT/'results/u6-launch/data.tgz'
    expected_hash = json.loads(Path(str(data_path)+'.sha256.json').read_text())
    if file_hash(data_path) != expected_hash['sha256'] or data_path.stat().st_size != expected_hash['bytes']:
        raise ValueError('archived data checksum failed before allocation')
    expected = json.loads((ROOT/'reference/u6-historical-identity.json').read_text())
    expected['source'] = source_identity()  # intentional diagnostic-only source revision
    (work/'expected.json').write_text(json.dumps(expected, indent=2))
    result['data_archive_sha256'] = expected_hash['sha256']
    pod = None
""")
s = s.replace("name='rz1t-u6-control'", "name='rz1t-duration-v1'")
s = s.replace('results/u6/', 'results/duration/').replace('mkdir -p results/u6 data', 'mkdir -p results/duration data')
s = s.replace("        remote('setup', f'''", """        scp(str(data_path), f'root@{host}:/tmp/duration-data.tgz')
        scp(str(work/'expected.json'), f'root@{host}:/tmp/duration-expected.json')
        remote('setup', '''""")
s = s.replace('uv pip install datasets\n', '')
start = s.index('export HF_HUB_DISABLE_XET=1')
end = s.index("''')", start)
s = s[:start] + """.venv/bin/python - <<'PY'
from rz1t.checkpoint import file_hash
from pathlib import Path
import tarfile
with tarfile.open('/tmp/duration-data.tgz') as t:
    t.extractall('.', filter='data')
PY
JAX_PLATFORMS=cuda .venv/bin/python -m runpod.duration_artifacts --expected /tmp/duration-expected.json --out results/duration/provenance.json
""" + s[end:]
start = s.index('        for seed in range(3):')
end = s.index("        result['status'] = 'completed_archived'", start)
s = s[:start] + """        for arm, slug in [('U6', 'u6'), ('R6x2', 'r6x2'), ('U12', 'u12')]:
            name = f'{arm}-s3'
            remote(name, prefix + f'JAX_PLATFORMS=cuda .venv/bin/python -m rz1t.train --config configs/explore/duration-{slug}.yaml --out results/duration/{name}\\nJAX_PLATFORMS=cpu .venv/bin/python -m runpod.u6_artifacts bundle --root results/duration --name {name}\\n')
            pull_bundle(name+'-metrics.tgz')
            pull_bundle(name+'-state.tgz')
""" + s[end:]
s = s.replace('default=90', 'default=150').replace('default=6)', 'default=10)')
s = s.replace("ROOT/'results/u6-launch'", "ROOT/'results/duration-launch'")
s = s.replace('<= 90 and 0 < a.spend_cap <= 6', '<= 150 and 0 < a.spend_cap <= 10')
s = s.replace('bounded U6 launch: 15–90 minutes, at most $6', 'bounded duration launch: 15–150 minutes, at most $10')
(ROOT/'runpod/launch_duration.py').write_text(s)

if __name__ == '__main__':
    pass
