"""Duration probes preserve scoring semantics, RNG isolation and exact resume."""
from copy import deepcopy
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from rz1t import train as trainer
from rz1t.config import ModelConfig
from rz1t.conventions import loss, make_steps
from rz1t.diagnostics import (evaluate_diagnostics, fixed_train_batches,
                              position_bins, summarize_batches, token_losses)
from rz1t.model import create_model
from test_training import TinyDataset, assert_arrays_equal, config, restore


@pytest.mark.parametrize('sequence,expected', [
    (3, [(0, 3)]), (64, [(0, 64)]), (65, [(0, 64), (64, 65)]),
    (128, [(0, 64), (64, 128)]), (256, [(0, 64), (64, 128), (128, 256)])])
def test_bins(sequence, expected):
    assert position_bins(sequence) == expected


def test_weighted_bins_match_scalar_with_partial_batches():
    cfg = ModelConfig(vocab=7, sequence=130, n_embed=4, m=1, k=1,
                      aft_heads=1, aft_ksize=2, linear_fan_in=2)
    model = create_model(cfg, jax.random.PRNGKey(5))
    class UnevenDataset(TinyDataset):
        def eval_batches(self, batch_size, sequence, split='development'):
            for n in (batch_size, 1):
                tokens = jnp.asarray(np.arange(n * (sequence + 1)).reshape(n, sequence + 1) % 7)
                yield tokens[:, :-1], tokens[:, 1:]
    dataset = UnevenDataset()
    result = evaluate_diagnostics(model, dataset, 2, 130, 'development', 2)
    _, eval_step = make_steps(optax.adam(0.001))
    scalar, tokens = trainer.evaluate(model, dataset, eval_step, 2, 130, 'development')
    dev = result['development']
    assert dev['tokens'] == tokens == 390
    assert result['train_probe']['tokens'] == 520
    assert [b['tokens'] for b in dev['position_bins']] == [192, 192, 6]
    assert sum(b['tokens'] * b['nll'] for b in dev['position_bins']) / tokens == pytest.approx(scalar, abs=1e-6)
    assert dev['nll'] == pytest.approx(scalar, abs=1e-6)
    x, y = next(dataset.eval_batches(2, 130))
    assert float(token_losses(model, x, y).mean()) == pytest.approx(float(loss(model, x, y)), abs=1e-6)
    assert float(token_losses(model, x[0], y[0]).mean()) == pytest.approx(float(loss(model, x[0], y[0])), abs=1e-6)


def test_probe_deterministic_and_seed_sensitive():
    dataset = TinyDataset()
    a = list(fixed_train_batches(dataset, 3, 4, 3, 1729))
    b = list(fixed_train_batches(dataset, 3, 4, 3, 1729))
    c = list(fixed_train_batches(dataset, 3, 4, 3, 1730))
    assert_arrays_equal(a, b)
    assert any(not np.array_equal(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(c)))


def test_diagnostics_do_not_change_training_and_resume(config, tmp_path):
    enabled = deepcopy(config)
    enabled['training'].update(diagnostic_probe_batches=2, diagnostic_probe_seed=1729, eval_every=3)
    trainer.train(config, tmp_path / 'disabled')
    final = trainer.train(enabled, tmp_path / 'enabled')
    trainer.train(enabled, tmp_path / 'resumed', max_steps=2)
    resumed = trainer.train(enabled, tmp_path / 'resumed', resume=True)
    a, ma, _ = restore(tmp_path / 'disabled', config)
    b, mb, _ = restore(tmp_path / 'enabled', enabled)
    c, mc, _ = restore(tmp_path / 'resumed', enabled)
    assert_arrays_equal(a, b)
    assert_arrays_equal(b, c)
    assert [r['train_nll'] for r in ma['metrics']] == [r['train_nll'] for r in mb['metrics']]
    assert mb['metrics'] == mc['metrics']
    assert final['diagnostics'] == resumed['diagnostics']
    assert final['diagnostics']['train_probe']['tokens'] == 6
    assert final['diagnostics']['development']['tokens'] == final['eval_tokens'] == 6
    assert final['nll'] == final['diagnostics']['development']['nll']
    assert [r['step'] for r in mb['metrics'] if 'diagnostics' in r] == [3, 4]
    saved = json.loads((tmp_path / 'enabled' / 'final.json').read_text())
    assert saved['diagnostics'] == final['diagnostics']
    assert 'rz1t/diagnostics.py' in trainer.source_identity()


@pytest.mark.parametrize('key,value', [('diagnostic_probe_batches', -1),
    ('diagnostic_probe_batches', True), ('diagnostic_probe_seed', -1),
    ('diagnostic_probe_seed', 1.5), ('diagnostic_probe_seed', 2**32)])
def test_config_rejects_invalid_probe(config, key, value):
    config['training'][key] = value
    with pytest.raises(ValueError):
        trainer.validate_config(config)


def test_nonfinite_and_empty_batches_fail_cleanly():
    x = y = jnp.zeros((1, 3), dtype=jnp.int32)
    summary = summarize_batches(None, [(x, y)], 3,
                                loss_step=lambda *args: jnp.full((1, 3), jnp.nan))
    assert not summary['finite'] and summary['nll'] is None
    json.dumps(summary, allow_nan=False)
    with pytest.raises(ValueError, match='no full windows'):
        summarize_batches(None, [], 3)
