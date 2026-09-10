import copy
import json
from pathlib import Path
import pytest
from runpod.u6_artifacts import check_identity


def test_gate_accepts_only_host_platform_change():
    expected = json.loads((Path(__file__).parents[1] / 'reference/u6-historical-identity.json').read_text())
    actual = copy.deepcopy(expected)
    actual['environment']['platform'] = 'different host kernel'
    check_identity(expected, actual['data'], actual['source'], actual['environment'])


@pytest.mark.parametrize('section,key', [('data', 'manifest_sha256'), ('data', 'source_sha256'), ('source', 'rz1t/train.py'), ('environment', 'python'), ('environment', 'packages'), ('environment', 'devices')])
def test_gate_rejects_drift(section, key):
    expected = json.loads((Path(__file__).parents[1] / 'reference/u6-historical-identity.json').read_text())
    actual = copy.deepcopy(expected)
    actual[section][key] = 'changed'
    with pytest.raises(ValueError, match='mismatch'):
        check_identity(expected, actual['data'], actual['source'], actual['environment'])
