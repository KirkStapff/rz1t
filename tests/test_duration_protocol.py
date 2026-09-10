from pathlib import Path
import json
import yaml
import pytest
from rz1t.train import validate_config, _seeds
from runpod.u6_artifacts import check_identity

ROOT = Path(__file__).resolve().parents[1]

def test_duration_matching():
    configs = [yaml.safe_load((ROOT/f'configs/explore/duration-{a}.yaml').read_text()) for a in ('u6','r6x2','u12')]
    counts = []
    for cfg in configs:
        model, work, schedule, budget = validate_config(cfg)
        assert schedule['steps'] == 15513
        assert work['tokens_per_step'] * schedule['steps'] == 254164992
        assert cfg['training']['eval_every'] == 1000
        assert cfg['training']['diagnostic_probe_batches'] == 8
        assert model.injection == 'none'
        assert cfg['lane'] == 'explore' and cfg['seed'] == 3
        counts.append(work['semantic_training_per_step'])
    assert counts[0] < counts[1] == counts[2]
    assert len({json.dumps(_seeds(c), sort_keys=True) for c in configs}) == 1
    assert len({c['run_id'] for c in configs}) == 3
    assert configs[0]['optimizer'] == configs[1]['optimizer'] == configs[2]['optimizer']

def test_identity_rejects_changed_diagnostic_source():
    expected = dict(data={'manifest':'pinned'},source={'diagnostics.py':'fixed'},environment={'packages':{'jax':'pinned'}})
    check_identity(expected, **expected)
    with pytest.raises(ValueError, match='source'):
        check_identity(expected, expected['data'], {'diagnostics.py':'changed'}, expected['environment'])
