from dataclasses import replace

import pytest

from rz1t.config import ModelConfig
from rz1t.flops import accounting, budget_to_steps
from rz1t.energy_model import WorkCounts, Scenario, compare_scenario, reload_break_even
from rz1t.make_sweep import generate_sweep, validate_primary_pair, write_sweep


def tiny(**kwargs):
    values = dict(vocab=11, sequence=3, n_embed=8, aft_heads=2,
                  aft_ksize=2, m=1, k=1)
    values.update(kwargs)
    return ModelConfig(**values)


def test_hand_counted_microcase():
    a = accounting(tiny(), batch_size=2)
    items = a['itemized_per_step']
    assert a['tokens_per_step'] == 6
    assert items['body']['qkv_mac'] == 6 * 2 * 4 * 18
    assert items['body']['convolution_mac'] == 6 * 2 * 2 * 10
    assert items['body']['cumulative_pool_add'] == 2 * 2 * 10
    assert items['body']['embedding_lookup_arithmetic'] == 0
    assert a['forward_head_per_step'] == 6 * (2 * 8 * 11 + 11)
    assert a['semantic_training_per_step'] == 3 * sum(sum(x.values()) for x in items.values())
    assert not a['validated_against_xla']


def test_tying_does_not_reduce_applied_work():
    u = replace(tiny(), m=12)
    r = replace(u, m=6, k=2)
    validate_primary_pair(u, r)
    au, ar = accounting(u), accounting(r)
    assert au['semantic_training_per_step'] == ar['semantic_training_per_step']
    assert au['unique_blocks'] == 12 and ar['unique_blocks'] == 6
    rr = accounting(replace(r, remat=True))
    assert rr['semantic_training_per_step'] == ar['semantic_training_per_step']
    assert rr['executed_training_per_step_estimate'] > ar['executed_training_per_step_estimate']
    assert accounting(replace(r, k=4))['forward_body_per_step'] > ar['forward_body_per_step']


def test_residual_injection_is_charged_only_after_first_loop():
    none = accounting(tiny(k=2, injection="none"))
    residual = accounting(tiny(k=2, injection="residual"))
    extra = residual['itemized_per_step']['body']['injection_residual_add']
    assert extra == 3 * 8 * 1
    assert residual['forward_body_per_step'] == none['forward_body_per_step'] + extra
    k1 = accounting(tiny(k=1, injection="residual"))
    assert k1['itemized_per_step']['body']['injection_residual_add'] == 0
    with pytest.raises(ValueError):
        accounting(tiny(injection="gather"))


def test_match_rejects_architecture_drift():
    u = replace(tiny(), m=12)
    with pytest.raises(ValueError):
        validate_primary_pair(u, replace(u, m=6, k=2, tanh_mlp=False))
    with pytest.raises(ValueError):
        validate_primary_pair(u, replace(u, m=3, k=4))


def test_budget_whole_step_cap_and_precision():
    assert budget_to_steps(1000, 100)['steps'] == 10
    assert budget_to_steps(999, 100)['steps'] == 10
    assert budget_to_steps(100, 101)['overshoot_fraction'] == 0.01
    assert budget_to_steps(10**20, 10**10)['steps'] == 10**10
    with pytest.raises(ValueError, match='overshoot'):
        budget_to_steps(101, 100)
    for bad in (0, -1, float('nan'), float('inf'), True):
        with pytest.raises(ValueError):
            budget_to_steps(bad, 10)
    with pytest.raises(ValueError):
        budget_to_steps(100, 10, max_overshoot=.02)


def test_draft_sweep_pairing_and_seal_block(tmp_path):
    manifest = generate_sweep(pairs=2)
    assert len(manifest['runs']) == 12
    assert not manifest['confirmation_authorized']
    assert manifest == generate_sweep(pairs=2)
    for budget in manifest['budgets']:
        for pair in manifest['pairs']:
            runs = [run for run in manifest['runs'] if run['budget'] == budget and run['pair_id'] == pair]
            assert len(runs) == 2
            assert runs[0]['config']['seeds'] == runs[1]['config']['seeds']
            assert runs[0]['budget_schedule'] == runs[1]['budget_schedule']
    streams = [value for seeds in manifest['seed_streams'].values() for value in seeds.values()]
    assert len(streams) == len(set(streams))
    assert generate_sweep(lane='explore', pairs=2)['seed_streams'] != manifest['seed_streams']
    path = write_sweep(manifest, tmp_path/'draft')
    assert path.exists()
    with pytest.raises(FileExistsError):
        write_sweep(manifest, tmp_path/'draft')
    with pytest.raises(ValueError, match='blocked'):
        generate_sweep(lane='sealed')
    with pytest.raises(ValueError, match='unknown'):
        generate_sweep(model={'auxiliary_loss': True})


def test_energy_is_conditional_and_dimensions_explicit():
    s = Scenario(1, 2, 3, 4, 5, 6, 10, 2)
    u = WorkCounts(100, weight_reload_bytes=20)
    r = WorkCounts(110, weight_reload_bytes=10)
    result = compare_scenario(u, r, s)
    assert result['ratio_recurrent_over_untied'] == pytest.approx(145/165)
    assert result['term_deltas_joules_per_token_conditional']['static'] == 0
    assert result['reference_audit_passed'] is False
    threshold = reload_break_even(u, r, s)
    assert threshold['threshold_joules_per_byte'] == 1
    assert threshold['recurrent_wins_when'] == 'coefficient > threshold'
    assert reload_break_even(u, u, s)['recurrent_wins_when'] == 'no coefficients'
    with pytest.raises(ValueError):
        Scenario(1, 2, 3, 4, 5, 6, 10, 0)
    with pytest.raises(ValueError):
        WorkCounts(-1)
