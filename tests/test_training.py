"""CPU microtests; fake in-memory data implements the public prepared-data API."""
from copy import deepcopy
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rz1t import train as trainer
from rz1t.checkpoint import (load_checkpoint, save_checkpoint, validate_completion,
                             verify_checkpoint)
from rz1t.config import ModelConfig
from rz1t.conventions import trainable_filter
from rz1t.flops import accounting
from rz1t.model import create_model


class TinyDataset:
    identity = {'kind': 'test_fixture', 'sha256': 'constant-fixture-v1'}

    def get_batch(self, key, batch_size, sequence, split='train'):
        assert split == 'train'
        tokens = jax.random.randint(key, (batch_size, sequence + 1), 0, 7)
        return tokens[:, :-1], tokens[:, 1:]

    def eval_batches(self, batch_size, sequence, split='development'):
        assert split in ('development', 'calibration')
        for offset in (0, 1):
            tokens = jnp.asarray((np.arange(sequence + 1) + offset) % 7)[None, :]
            yield tokens[:, :-1], tokens[:, 1:]


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr(trainer, 'load_dataset', lambda data, model: TinyDataset())
    return {'lane': 'dev', 'draft': False, 'seed': 42,
            'model': {'vocab': 7, 'sequence': 3, 'n_embed': 4, 'm': 1, 'k': 2,
                      'aft_heads': 1, 'aft_ksize': 2, 'linear_fan_in': 2},
            'training': {'steps': 4, 'batch_size': 1, 'checkpoint_every': 1, 'eval_every': 0},
            'optimizer': {'lr': 0.001, 'warmup': 0},
            'data': {'dataset_name': 'synthetic'}}


def restore(out, config):
    cfg, counts, schedule, budget = trainer.validate_config(config)
    seeds = trainer._seeds(config)
    model = create_model(cfg, jax.random.PRNGKey(seeds['init']))
    optimizer = trainer.make_optimizer(config, budget, counts)
    opt = optimizer.init(eqx.filter(model, trainable_filter(model)))
    rngs = {name: jax.random.PRNGKey(0) for name in ('model', 'data', 'loop', 'eval')}
    identity = json.loads((out / 'identity.json').read_text())
    return load_checkpoint(out / 'checkpoints', (model, opt, rngs), identity)


def assert_arrays_equal(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    assert len(la) == len(lb)
    for x, y in zip(la, lb):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_schedule_survives_int32_semantic_work():
    counts = {'semantic_training_per_step': 114_456_115_200}
    optimizer = trainer.make_optimizer({'optimizer': {'lr': 3e-4, 'warmup': 100}},
                                       2000 * counts['semantic_training_per_step'], counts)
    params = {'w': jnp.ones((2,), dtype=jnp.float32)}
    state = optimizer.init(params)
    updates, state = optimizer.update(params, state, params)
    assert jnp.all(jnp.isfinite(updates['w']))


def test_resume_full_state_and_completion(config, tmp_path):
    full, resumed = tmp_path / 'full', tmp_path / 'resumed'
    expected = trainer.train(config, full)
    stopped = trainer.train(config, resumed, max_steps=2)
    assert stopped['status'] == 'incomplete'
    assert not (resumed / 'completion.json').exists()
    actual = trainer.train(config, resumed, resume=True)
    state_a, meta_a, _ = restore(full, config)
    state_b, meta_b, _ = restore(resumed, config)
    assert_arrays_equal(state_a, state_b)
    for key in ('step', 'tokens', 'data_cursor', 'semantic_work', 'executed_work_estimate', 'metrics'):
        assert meta_a[key] == meta_b[key]
    assert actual['nll'] == expected['nll']
    assert actual['status'] == 'completed'
    assert actual['checkpoint_selection'] == 'final_budget'
    assert validate_completion(resumed)['nll'] == expected['nll']
    assert trainer.train(config, resumed, resume=True) == actual
    assert len(list((resumed / 'checkpoints').glob('step-*'))) == 2
    with pytest.raises(ValueError, match='completed'):
        trainer.train(config, resumed)
    (resumed / 'final.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        trainer.train(config, resumed, resume=True)


def test_eval_frequency_cannot_change_training_rng(config, tmp_path):
    frequent = deepcopy(config)
    frequent['training']['eval_every'] = 1
    trainer.train(config, tmp_path / 'none')
    trainer.train(frequent, tmp_path / 'every')
    (a, ma, _), (b, mb, _) = restore(tmp_path / 'none', config), restore(tmp_path / 'every', frequent)
    assert_arrays_equal(a, b)
    assert [x['train_nll'] for x in ma['metrics']] == [x['train_nll'] for x in mb['metrics']]
    assert ma['semantic_work'] == mb['semantic_work']
    initial = create_model(ModelConfig.from_dict(config['model']),
                           jax.random.PRNGKey(trainer._seeds(config)['init']))
    np.testing.assert_array_equal(a[0].pe.pe, initial.pe.pe)


def test_corrupt_state_and_identity_rejected(tmp_path):
    state = {'x': jnp.arange(3), 'key': jax.random.PRNGKey(2)}
    identity = {'config': 'a', 'data': 'b'}
    pointer = save_checkpoint(tmp_path, state, {'step': 1}, identity)
    assert_arrays_equal(load_checkpoint(tmp_path, state, identity)[0], state)
    with pytest.raises(ValueError, match='identity'):
        load_checkpoint(tmp_path, state, {'config': 'different'})
    directory = tmp_path / pointer['directory']
    (directory / 'state.eqx').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        load_checkpoint(tmp_path, state, identity)
    with pytest.raises(ValueError, match='checksum'):
        verify_checkpoint(directory, identity)


def test_budget_and_gate_validation(config, tmp_path):
    cfg = ModelConfig.from_dict(config['model'])
    per_step = accounting(cfg)['semantic_training_per_step']
    config['training']['semantic_budget'] = 4 * per_step
    result = trainer.train(config, tmp_path / 'budget', max_steps=1)
    assert result['semantic_work'] == per_step
    state, metadata, _ = restore(tmp_path / 'budget', config)
    assert metadata['target_budget'] == 4 * per_step
    for modification in ({'draft': True}, {'lane': 'sealed'}, {'lane': 'confirmation'}):
        with pytest.raises(ValueError):
            trainer.train(config | modification, tmp_path / 'blocked')
    with pytest.raises(ValueError, match='claim-a'):
        trainer.validate_config(config | {'data': {'evaluation_split': 'confirmation'}})
    with pytest.raises(ValueError, match='evaluation_split'):
        trainer.validate_config(config | {'data': {'evaluation_split': 'train'}})
    wrong = deepcopy(config)
    wrong['training']['steps'] = 3
    with pytest.raises(ValueError, match='steps disagree'):
        trainer.validate_config(wrong)
    wrong = deepcopy(config)
    wrong['training']['gradient_accumulation'] = 2
    with pytest.raises(ValueError, match='unknown/unimplemented'):
        trainer.validate_config(wrong)


def test_seed_mapping_and_resume_identity(config, tmp_path):
    config['seeds'] = {'init': 42, 'data': 12, 'eval': 13, 'loop': 14}
    trainer.train(config, tmp_path, max_steps=1)
    state, metadata, _ = restore(tmp_path, config)
    np.testing.assert_array_equal(state[2]['loop'], jax.random.PRNGKey(14))
    assert metadata['seed_mapping'] == config['seeds']
    changed = deepcopy(config)
    changed['training']['eval_every'] = 1
    with pytest.raises(ValueError, match='identity'):
        trainer.train(changed, tmp_path, resume=True)


def test_model_failure_is_not_success_replaced(config, tmp_path, monkeypatch):
    original = trainer.make_steps
    def bad_steps(optimizer):
        step, evaluate = original(optimizer)
        def bad(model, opt, x, y):
            model, opt, _ = step(model, opt, x, y)
            return model, opt, jnp.asarray(float('nan'))
        return bad, evaluate
    monkeypatch.setattr(trainer, 'make_steps', bad_steps)
    result = trainer.train(config, tmp_path)
    assert result['status'] == 'model_failure'
    assert result['step'] == 0
    assert not (tmp_path / 'completion.json').exists()
    with pytest.raises(ValueError, match='terminal'):
        trainer.train(config, tmp_path, resume=True)


def test_crash_then_restore_committed_cursor(config, tmp_path, monkeypatch):
    class CrashDataset(TinyDataset):
        calls = 0
        def get_batch(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError('simulated infrastructure crash')
            return super().get_batch(*args, **kwargs)

    full, crashed = tmp_path / 'full', tmp_path / 'crashed'
    trainer.train(config, full)
    monkeypatch.setattr(trainer, 'load_dataset', lambda data, model: CrashDataset())
    with pytest.raises(RuntimeError, match='infrastructure'):
        trainer.train(config, crashed)
    assert restore(crashed, config)[1]['data_cursor'] == 2
    monkeypatch.setattr(trainer, 'load_dataset', lambda data, model: TinyDataset())
    trainer.train(config, crashed, resume=True)
    assert_arrays_equal(restore(full, config)[0], restore(crashed, config)[0])
