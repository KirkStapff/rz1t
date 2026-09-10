"""Duration-v1 provenance gate: archived data, frozen new source, historical numerics."""
import argparse
import json
from pathlib import Path
from rz1t.checkpoint import atomic_json
from rz1t.data import PreparedDataset
from rz1t.train import environment_identity, source_identity
from runpod.u6_artifacts import check_identity


def gate(expected_path, out):
    expected = json.loads(Path(expected_path).read_text())
    data = PreparedDataset('data/owt-subset-256/manifest.json', vocab=50257, sequence=256)
    actual = dict(data=data.identity, source=source_identity(), environment=environment_identity())
    check_identity(expected, **actual)
    atomic_json(out, dict(status='passed', protocol='duration-v1', **actual))

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    gate(a.expected, a.out)
