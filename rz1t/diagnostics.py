"""Deterministic, token-weighted probes for development duration diagnostics.

Positions are zero-based prediction positions, not independent replicates.
The train probe samples with replacement using its own fixed seed; it is not
an exhaustive training-set loss. No training RNG or mutable cursor is consumed.
"""
from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
import optax


@eqx.filter_jit
def token_losses(model, x, y):
    """Same aligned, unmasked next-token CE as conventions.loss, unreduced."""
    if x.shape != y.shape or x.ndim not in (1, 2) or x.size == 0:
        raise ValueError('x and y must have equal, nonempty (T,) or (B,T) shapes')
    logits = model(x) if x.ndim == 1 else jax.vmap(model)(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y)


def position_bins(sequence):
    """Clip [0,64), [64,128), [128,T) to T and omit empty bins."""
    if type(sequence) is not int or sequence < 1:
        raise ValueError('sequence must be a positive integer')
    return [(lo, min(hi, sequence)) for lo, hi in
            ((0, 64), (64, 128), (128, sequence)) if lo < sequence]


def summarize_batches(model, batches, sequence, loss_step=token_losses):
    bounds = position_bins(sequence)
    sums = np.zeros(len(bounds), dtype=np.float64)
    counts = np.zeros(len(bounds), dtype=np.int64)
    finite = True
    n_batches = 0
    for x, y in batches:
        values = np.asarray(loss_step(model, x, y), dtype=np.float64)
        if values.shape != tuple(y.shape) or values.ndim != 2 or values.shape[1] != sequence:
            raise ValueError('diagnostics require aligned (B,sequence) token losses')
        finite = finite and bool(np.isfinite(values).all())
        for i, (lo, hi) in enumerate(bounds):
            chunk = values[:, lo:hi]
            sums[i] += chunk.sum()
            counts[i] += chunk.size
        n_batches += 1
    tokens = int(counts.sum())
    if not tokens:
        raise ValueError('evaluation manifest has no full windows')
    return {'nll': float(sums.sum() / tokens) if finite else None,
            'tokens': tokens, 'batches': n_batches, 'finite': finite,
            'position_bins': [
                {'start': lo, 'end': hi, 'tokens': int(counts[i]),
                 'nll': float(sums[i] / counts[i]) if finite else None}
                for i, (lo, hi) in enumerate(bounds)]}


def fixed_train_batches(dataset, batch_size, sequence, batches, seed=1729):
    if type(batches) is not int or batches < 1:
        raise ValueError('probe batches must be a positive integer')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('probe seed must fit uint32')
    # Recreated on every call: identical windows across checkpoints and arms.
    key = jax.random.PRNGKey(seed)
    for _ in range(batches):
        key, batch_key = jax.random.split(key)
        yield dataset.get_batch(batch_key, batch_size, sequence, split='train')


def evaluate_diagnostics(model, dataset, batch_size, sequence, split,
                         probe_batches, probe_seed=1729):
    if split not in ('development', 'calibration', 'confirmation'):
        raise ValueError('diagnostics allow development/calibration/confirmation only')
    holdout = summarize_batches(
        model, dataset.eval_batches(batch_size, sequence, split=split), sequence)
    probe = summarize_batches(model, fixed_train_batches(
        dataset, batch_size, sequence, probe_batches, probe_seed), sequence)
    return {'version': 'duration-diagnostics-v1', 'evaluation_split': split,
            'holdout': holdout,
            'development': holdout,
            'train_probe': probe,
            'probe_seed': probe_seed, 'probe_batches': probe_batches,
            'probe_sampling': 'fixed random windows with replacement',
            'finite': holdout['finite'] and probe['finite']}
