from dataclasses import replace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from z1t.components import SparseLinear, TanhLinear
from z1t.model import create_model as upstream_create

from rz1t.config import ModelConfig
from rz1t.conventions import loss, make_steps, parameter_counts, trainable_filter
from rz1t.model import create_model
from rz1t.resources import resource_report


def tiny(**kwargs):
    values = dict(vocab=11, sequence=5, n_embed=8, m=1, k=1,
                  aft_heads=2, aft_ksize=2, linear_fan_in=4)
    values.update(kwargs)
    return ModelConfig(**values)


def aligned_arrays(model):
    """Canonical order despite different outer Equinox model field layouts."""
    blocks = (model.prelude + model.core + model.coda
              if hasattr(model, "core") else tuple(model.blocks))
    return jax.tree.leaves(eqx.filter(
        (model.embedding, model.pe, blocks, model.norm, model.clf), eqx.is_array))


def assert_arrays_close(left, right, *, atol=1e-6):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        np.testing.assert_allclose(a, b, atol=atol, rtol=1e-6)


def gradients(model, x, y):
    params, fixed = eqx.partition(model, trainable_filter(model))
    return eqx.filter_grad(lambda p: loss(eqx.combine(p, fixed), x, y))(params)


@pytest.mark.parametrize("wrapped,remat,p,q", [(True, False, 0, 0),
                                               (False, True, 1, 1)])
def test_k1_upstream_logits_gradients_ten_updates(wrapped, remat, p, q):
    config = tiny(tanh_linear=wrapped, tanh_mlp=wrapped, remat=remat, p=p, q=q)
    key = jax.random.PRNGKey(5)
    model = create_model(config, key)
    upstream = upstream_create(config.to_upstream(), key)
    x = jnp.array([[0, 1, 2, 3, 4], [3, 2, 1, 0, 4]])
    y = (x + 1) % config.vocab
    assert_arrays_close(aligned_arrays(model), aligned_arrays(upstream), atol=0)
    np.testing.assert_allclose(model(x[0]), upstream(x[0]), atol=1e-6, rtol=1e-6)
    assert isinstance(model.clf, eqx.nn.Linear)
    assert isinstance(model.core[0].mlp.proj1, TanhLinear) == wrapped
    assert_arrays_close(aligned_arrays(gradients(model, x, y)),
                        aligned_arrays(gradients(upstream, x, y)))
    original_pe = np.array(model.pe.pe)
    optim = optax.adamw(1e-3, weight_decay=0.1)
    state = optim.init(eqx.filter(model, trainable_filter(model)))
    ref_state = optim.init(eqx.filter(upstream, trainable_filter(upstream)))
    step, evaluate = make_steps(optim)
    for _ in range(10):
        model, state, value = step(model, state, x, y)
        upstream, ref_state, ref_value = step(upstream, ref_state, x, y)
        np.testing.assert_allclose(value, ref_value, atol=1e-6, rtol=1e-6)
    assert_arrays_close(aligned_arrays(model), aligned_arrays(upstream))
    np.testing.assert_array_equal(model.pe.pe, original_pe)
    np.testing.assert_allclose(evaluate(model, x, y), loss(model, x, y), atol=1e-6)


def unrolled(model, tokens):
    h = model.pe(jax.vmap(model.embedding)(tokens))
    for block in model.prelude:
        h = block(h)
    h0 = h
    for _ in range(model.k):
        z = h + h0 if model.config.injection == "residual" and _ > 0 else h
        for block in model.core:
            z = block(z)
        h = z
    for block in model.coda:
        h = block(h)
    return jax.vmap(model.clf)(model.norm(h))


@pytest.mark.parametrize("k,remat", [(2, False), (4, False), (2, True)])
def test_tying_matches_unrolled_full_bptt(k, remat):
    model = create_model(tiny(k=k, remat=remat, p=1, q=1), jax.random.PRNGKey(9))
    x = jnp.arange(5)
    np.testing.assert_allclose(model(x), unrolled(model, x), atol=1e-6, rtol=1e-6)
    actual = eqx.filter_grad(lambda m: jnp.sum(m(x) ** 2))(model)
    expected = eqx.filter_grad(lambda m: jnp.sum(unrolled(m, x) ** 2))(model)
    assert_arrays_close(aligned_arrays(actual), aligned_arrays(expected), atol=2e-5)
    shallow = create_model(replace(model.config, k=1), jax.random.PRNGKey(9))
    assert_arrays_close(aligned_arrays(model), aligned_arrays(shallow), atol=0)
    assert parameter_counts(model) == parameter_counts(shallow)
    assert len(model.core) == 1


def test_tied_gradient_is_sum_of_independent_use_gradients():
    model = create_model(tiny(k=2), jax.random.PRNGKey(0))
    x = jnp.arange(5)
    h = model.pe(jax.vmap(model.embedding)(x))
    block = model.core[0]

    def objective(a, b):
        return jnp.sum(model.norm(b(a(h))) ** 2)

    grad_a = eqx.filter_grad(lambda a: objective(a, block))(block)
    grad_b = eqx.filter_grad(lambda b: objective(block, b))(block)
    tied = eqx.filter_grad(lambda shared: objective(shared, shared))(block)
    leaves = lambda tree: jax.tree.leaves(eqx.filter(tree, eqx.is_inexact_array))
    assert_arrays_close(leaves(tied), [a + b for a, b in zip(leaves(grad_a), leaves(grad_b))])
    sparse = [node for node in jax.tree.leaves(block, is_leaf=lambda n: isinstance(n, SparseLinear))
              if isinstance(node, SparseLinear)]
    assert len(sparse) == 4
    assert all(node.k == model.config.linear_fan_in for node in sparse)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_causality(k):
    model = create_model(tiny(k=k), jax.random.PRNGKey(3))
    x = jnp.arange(5)
    changed = x.at[3:].set(jnp.array([9, 8]))
    np.testing.assert_allclose(model(x)[:3], model(changed)[:3], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("k", [1, 2, 4])
def test_residual_injection_skip_first(k):
    key = jax.random.PRNGKey(11)
    none = create_model(tiny(k=k, injection="none"), key)
    residual = create_model(tiny(k=k, injection="residual"), key)
    x = jnp.arange(5)
    if k == 1:
        np.testing.assert_allclose(none(x), residual(x), atol=0, rtol=0)
    else:
        assert not np.allclose(none(x), residual(x), atol=1e-5)
    np.testing.assert_allclose(residual(x), unrolled(residual, x), atol=1e-6, rtol=1e-6)
    changed = x.at[3:].set(jnp.array([9, 8]))
    np.testing.assert_allclose(residual(x)[:3], residual(changed)[:3], atol=1e-6, rtol=1e-6)
    actual = eqx.filter_grad(lambda m: jnp.sum(m(x) ** 2))(residual)
    expected = eqx.filter_grad(lambda m: jnp.sum(unrolled(m, x) ** 2))(residual)
    assert_arrays_close(aligned_arrays(actual), aligned_arrays(expected), atol=2e-5)
    report = resource_report(residual)
    if k == 1:
        assert report["retained_injection_h0_bytes"] == 0
    else:
        assert report["retained_injection_h0_bytes"] == 5 * 8 * 4


def test_k1_residual_still_matches_upstream():
    config = tiny(k=1, injection="residual")
    key = jax.random.PRNGKey(5)
    model = create_model(config, key)
    upstream = upstream_create(config.to_upstream(), key)
    x = jnp.arange(5)
    np.testing.assert_allclose(model(x), upstream(x), atol=1e-6, rtol=1e-6)


def test_partition_counts_and_resources():
    model = create_model(tiny(k=2), jax.random.PRNGKey(8))
    params, fixed = eqx.partition(model, trainable_filter(model))
    assert params.pe.pe is None
    np.testing.assert_array_equal(fixed.pe.pe, model.pe.pe)
    counts = parameter_counts(model)
    assert counts["embedding"] == 11 * 8
    assert counts["classifier"] == 11 * 8 + 11
    assert counts["pe"] == 5 * 8
    assert counts["indices"] == (2 * 8 + 2 + 8 + 4 * 8 + 8) * 4
    assert counts["body"] == sum(a.size for a in jax.tree.leaves(
        eqx.filter((model.core, model.norm), eqx.is_inexact_array)))
    report = resource_report(model)
    assert report["stored_bytes"]["body_weights"] == counts["body"] * 4
    assert report["hypothetical_cache_bytes"] == 2 * 2 * (8 + 2) * 4
    shallow_report = resource_report(create_model(replace(model.config, k=1), jax.random.PRNGKey(8)))
    assert report["stored_bytes"] == shallow_report["stored_bytes"]
    assert report["hypothetical_cache_bytes"] == 2 * shallow_report["hypothetical_cache_bytes"]


def test_shapes_fail_closed():
    model = create_model(tiny(), jax.random.PRNGKey(0))
    for x in (jnp.zeros((6,), dtype=int), jnp.zeros((0,), dtype=int), jnp.ones((2, 3)), jnp.ones(3)):
        with pytest.raises(ValueError):
            model(x)
    with pytest.raises(ValueError):
        loss(model, jnp.ones((2, 3), dtype=int), jnp.ones((3, 2), dtype=int))
