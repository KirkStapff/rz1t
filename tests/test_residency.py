from dataclasses import replace
import json

import equinox as eqx
import jax
import pytest
import yaml

from rz1t.config import ModelConfig
from rz1t.model import create_model
from rz1t.residency import (BlockPayload, ReloadScenario, conditional_cost,
                             main, residency_report, slab_schedule)


def blocks(n):
    return [BlockPayload(8, 4) for _ in range(n)]


def tiny(**changes):
    return ModelConfig(**(dict(vocab=11, sequence=3, n_embed=8,
                               aft_heads=2, aft_ksize=2, m=6, k=2) | changes))


def test_six_block_whole_slab_hand_schedule():
    u = slab_schedule(blocks(12), k=1, rounds=100)
    r = slab_schedule(blocks(6), k=2, rounds=100)
    u6 = slab_schedule(blocks(6), k=1, rounds=100)
    assert u['slabs_one_based'] == [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
    assert u['initial_round'] == u['steady_round'] == {
        'weight_bytes': 96, 'index_bytes': 48, 'body_bytes': 144,
        'load_events': 2, 'blocks_loaded': 12}
    assert u['finite_total']['body_bytes'] == 14400
    assert u['finite_total']['load_events'] == 200
    assert r['initial_round']['body_bytes'] == 72
    assert r['initial_round']['load_events'] == 1
    assert r['steady_round']['body_bytes'] == r['steady_round']['load_events'] == 0
    assert r['finite_total']['body_bytes'] == 72
    assert r['finite_average_per_token']['body_bytes'] == .72
    assert u6['finite_total'] == r['finite_total']
    assert u6['applied_blocks'] == 6 and r['applied_blocks'] == 12
    assert 'not_cache_optimum' in r['status']


def test_batching_cold_amortization_and_capacity():
    a = slab_schedule(blocks(6), k=2, batch_size=4, rounds=10)
    assert a['delivered_tokens'] == 40
    assert a['initial_per_token']['body_bytes'] == 18
    assert a['finite_average_per_token']['body_bytes'] == 1.8
    assert a['finite_total']['body_bytes'] == 72
    cold = slab_schedule(blocks(6), k=2, batch_size=4, rounds=1)
    assert cold['initial_per_token'] == cold['finite_average_per_token']
    u = slab_schedule(blocks(12), k=1, capacity_blocks=12)
    assert u['steady_round']['body_bytes'] == 0  # fair resident U12 scenario
    odd = slab_schedule(blocks(7), k=1, capacity_blocks=6, rounds=3)
    assert odd['slabs_one_based'][-1] == [7]
    assert odd['finite_total']['body_bytes'] == 7 * 12 * 3
    assert odd['finite_total']['load_events'] == 6


def test_payloads_from_actual_trainable_partition_and_indices():
    models = [create_model(tiny(m=m, k=k), jax.random.key(0))
              for m, k in ((6, 1), (6, 2), (12, 1))]
    reports = [residency_report(model) for model in models]
    for model, report in zip(models, reports, strict=True):
        expected_weights = sum(x.size * x.dtype.itemsize
                               for x in jax.tree.leaves(model.core)
                               if eqx.is_inexact_array(x))
        expected_indices = sum(x.size * x.dtype.itemsize
                               for x in jax.tree.leaves(model.core)
                               if eqx.is_array(x) and not eqx.is_inexact_array(x))
        assert report['initial_round']['weight_bytes'] == expected_weights
        assert report['initial_round']['index_bytes'] == expected_indices
        norm_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(model.norm)
                         if eqx.is_array(x))
        assert report['shared_resident_norm_bytes_excluded'] == norm_bytes > 0
        assert len(report['per_block_payloads']) == model.config.m
    assert reports[0]['per_block_payloads'] == reports[1]['per_block_payloads']
    assert reports[2]['initial_round']['body_bytes'] == 2 * reports[0]['initial_round']['body_bytes']


@pytest.mark.parametrize('changes', [dict(p=1), dict(q=1), dict(injection='residual'),
                                    dict(m=7, k=2)])
def test_unsupported_model(changes):
    with pytest.raises(ValueError):
        residency_report(create_model(tiny(**changes), jax.random.key(0)))


def test_wrong_model():
    with pytest.raises(ValueError):
        residency_report(object())


@pytest.mark.parametrize('name', ['capacity_blocks', 'batch_size', 'rounds', 'k'])
@pytest.mark.parametrize('bad', [0, -1, True, 1.5, float('nan'), float('inf')])
def test_schedule_invalid_counts(name, bad):
    with pytest.raises(ValueError):
        slab_schedule(blocks(6), **(dict(k=1) | {name: bad}))


def test_bad_payloads():
    for bad in ([], [12]):
        with pytest.raises(ValueError):
            slab_schedule(bad, k=1)
    for weights, indices in ((0, 1), (-1, 1), (True, 1), (1, -1), (1, float('inf'))):
        with pytest.raises(ValueError):
            BlockPayload(weights, indices)


def scenario():
    # Artificial unit arithmetic ONLY: these are not hardware values.
    return ReloadScenario(effective_bandwidth_bytes_per_second=12,
                          programming_seconds_per_load_event=2,
                          exposed_fraction=.5, dynamic_joules_per_byte=3,
                          programming_joules_per_load_event=5,
                          common_compute_seconds_per_round=10,
                          common_compute_joules_per_round=7, static_watts=2)


def test_energy_dimensions_static_and_batch_denominator():
    report = slab_schedule(blocks(12), k=1, batch_size=4, rounds=3)
    out = conditional_cost(report, scenario())
    first, total = out['initial_round'], out['finite_total']
    # Each round 144B / 12B/s + 2 events * 2s = 16s loading; half exposed.
    assert first['exposed_reload_seconds'] == 8
    assert first['total_seconds'] == 18
    assert first['energy_terms_joules'] == dict(common_compute=7, transfer=432,
                                              programming=10, static=36)
    assert first['total_joules'] == 485
    assert first['joules_per_token'] == 485/4
    assert first['amortized_seconds_per_token'] == 18/4
    assert total['total_seconds'] == 54
    assert total['total_joules'] == 1455
    assert total['delivered_tokens_per_second'] == 12/54
    r = conditional_cost(slab_schedule(blocks(6), k=2, batch_size=4, rounds=3), scenario())
    assert r['finite_total']['total_seconds'] == 34  # 30 compute + .5*(6+2)
    assert r['finite_total']['total_joules'] == 21 + 216 + 5 + 68
    assert r['steady_round']['total_seconds'] == 10
    assert r['steady_round']['total_joules'] == 27


def test_overlap_hides_time_not_dynamic_energy():
    report = slab_schedule(blocks(12), k=1)
    hidden = conditional_cost(report, replace(scenario(), exposed_fraction=0))['steady_round']
    full = conditional_cost(report, replace(scenario(), exposed_fraction=1))['steady_round']
    assert hidden['total_seconds'] == 10
    assert full['total_seconds'] == 26
    assert hidden['energy_terms_joules']['transfer'] == full['energy_terms_joules']['transfer']
    assert hidden['energy_terms_joules']['programming'] == full['energy_terms_joules']['programming']
    assert full['total_joules'] - hidden['total_joules'] == 16 * 2


@pytest.mark.parametrize('name', list(ReloadScenario.__dataclass_fields__))
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1, True, 'bad'])
def test_invalid_scenario_floats(name, bad):
    with pytest.raises(ValueError):
        replace(scenario(), **{name: bad})


def test_scenario_positive_and_fraction_bounds():
    for name in ('effective_bandwidth_bytes_per_second', 'common_compute_seconds_per_round'):
        with pytest.raises(ValueError):
            replace(scenario(), **{name: 0})
    with pytest.raises(ValueError):
        replace(scenario(), exposed_fraction=1.01)
    zero = replace(scenario(), static_watts=0, dynamic_joules_per_byte=0,
                   programming_joules_per_load_event=0, common_compute_joules_per_round=0)
    assert conditional_cost(slab_schedule(blocks(6), k=2), zero)['finite_total']['total_joules'] == 0


def test_cli_cpu_json_three_configs(tmp_path, capsys):
    paths = []
    for arm, m, k in [('U6', 6, 1), ('R6x2', 6, 2), ('U12', 12, 1)]:
        path = tmp_path / f'{arm}.yaml'
        path.write_text(yaml.safe_dump(dict(arm=arm, model=tiny(m=m, k=k).to_dict())))
        paths.append(str(path))
    main(paths + ['--capacity-blocks', '6', '--batch-size', '1', '--rounds', '100'])
    out = json.loads(capsys.readouterr().out)
    assert [entry['arm'] for entry in out] == ['U6', 'R6x2', 'U12']
    assert out[0]['report']['steady_round']['body_bytes'] == 0
    assert out[2]['report']['finite_total']['blocks_loaded'] == 1200
