"""Fixed-k, fp32 teacher-forced development trainer with exact-state resume.

No network fetch, cloud provisioning, confirmation access, or automatic seal.
Work counters use semantic-proxy-v1 estimates, not XLA-measured device FLOPs.
Example: python -m rz1t.train --config configs/dev/smoke.yaml --out results/smoke
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import yaml

from rz1t.checkpoint import (atomic_json, canonical_hash, file_hash, load_checkpoint,
                             save_checkpoint, validate_completion, write_completion)
from rz1t.config import ModelConfig
from rz1t.conventions import make_steps, parameter_counts, trainable_filter
from rz1t.diagnostics import evaluate_diagnostics
from rz1t.flops import ACCOUNTING_VERSION, accounting, budget_to_steps
from rz1t.model import create_model


def load_dataset(data_config, model_config):
    from rz1t.data import load_dataset as load
    return load(data_config, model_config)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _unknown(mapping, allowed, section):
    if not isinstance(mapping, dict):
        raise ValueError(f"{section} must be a mapping")
    extra = set(mapping) - set(allowed)
    if extra:
        raise ValueError(f"unknown/unimplemented {section} options: {sorted(extra)}")


def validate_config(config):
    _unknown(config, ('lane', 'draft', 'arm', 'pair_id', 'run_id', 'seed', 'seeds',
                     'model', 'training', 'optimizer', 'data', 'precision', 'accounting_version'), 'root')
    if type(config.get('draft', False)) is not bool or config.get('draft', False):
        raise ValueError('draft configs cannot train; resolve and regenerate identities first')
    if config.get('lane', 'dev') not in ('dev', 'explore', 'calibration', 'claim-a'):
        raise ValueError('sealed/H1-confirmation training blocked: scientific gates not implemented')
    if config.get('precision', 'float32') != 'float32':
        raise ValueError('only float32 precision is implemented')
    if config.get('accounting_version', ACCOUNTING_VERSION) != ACCOUNTING_VERSION:
        raise ValueError('accounting version mismatch')
    model = ModelConfig.from_dict(config['model'])
    training = config.get('training', {})
    _unknown(training, ('steps', 'batch_size', 'semantic_budget', 'max_budget_overshoot',
                        'schedule_basis', 'eval_every', 'checkpoint_every',
                        'diagnostic_probe_batches', 'diagnostic_probe_seed'), 'training')
    if training.get('schedule_basis', 'cumulative_semantic_budget') != 'cumulative_semantic_budget':
        raise ValueError('unsupported schedule_basis')
    batch = _integer(training.get('batch_size', 1), 'batch_size')
    counts = accounting(model, batch)
    step_work = counts['semantic_training_per_step']
    if 'semantic_budget' in training:
        schedule = budget_to_steps(training['semantic_budget'], step_work,
                                   max_overshoot=training.get('max_budget_overshoot', 0.01))
        steps = schedule['steps']
        if 'steps' in training and _integer(training['steps'], 'steps') != steps:
            raise ValueError('steps disagree with semantic budget')
        budget = training['semantic_budget']
    else:
        steps = _integer(training.get('steps', 10), 'steps')
        budget = steps * step_work
        schedule = budget_to_steps(budget, step_work)
    for name in ('eval_every', 'checkpoint_every'):
        _integer(training.get(name, 0 if name == 'eval_every' else 100), name,
                 minimum=0 if name == 'eval_every' else 1)
    _integer(training.get('diagnostic_probe_batches', 0), 'diagnostic_probe_batches', 0)
    probe_seed = _integer(training.get('diagnostic_probe_seed', 1729), 'diagnostic_probe_seed', 0)
    if probe_seed >= 2**32:
        raise ValueError('diagnostic_probe_seed must fit uint32')
    optimizer = config.get('optimizer', {})
    _unknown(optimizer, ('lr', 'weight_decay', 'b1', 'b2', 'warmup', 'grad_clip', 'final_lr_ratio'), 'optimizer')
    for key, default in (('lr', 3e-4), ('weight_decay', .1), ('b1', .9), ('b2', .95),
                         ('grad_clip', 1.), ('final_lr_ratio', .1)):
        value = optimizer.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
        if value < 0 or (key in ('lr', 'grad_clip') and value == 0):
            raise ValueError(f'invalid {key}')
        if key in ('b1', 'b2') and value >= 1:
            raise ValueError(f'{key} must be below one')
        if key == 'final_lr_ratio' and value > 1:
            raise ValueError('final_lr_ratio must be at most one')
    _integer(optimizer.get('warmup', 0), 'warmup', 0)
    if optimizer.get('warmup', 0) >= steps:
        raise ValueError('warmup must be shorter than the planned schedule')
    data = config.get('data', {'dataset_name': 'synthetic'})
    if not isinstance(data, dict):
        raise ValueError('data must be a mapping')
    split = data.get('evaluation_split', 'development')
    if split not in ('development', 'calibration', 'confirmation'):
        raise ValueError('evaluation_split must be development, calibration, or confirmation')
    lane = config.get('lane', 'dev')
    if split == 'calibration' and lane != 'calibration':
        raise ValueError('calibration evaluation requires calibration lane')
    if split == 'confirmation' and lane != 'claim-a':
        raise ValueError('confirmation-split evaluation requires claim-a lane; H1 sealed matrix remains blocked')
    if lane == 'claim-a' and split != 'confirmation':
        raise ValueError('claim-a lane must evaluate the unused confirmation split')
    return model, counts, schedule, budget


def _seeds(config):
    seed = _integer(config.get('seed', 0), 'seed', 0)
    supplied = config.get('seeds')
    if supplied is not None:
        if not isinstance(supplied, dict) or set(supplied) != {'init', 'data', 'loop', 'eval'}:
            raise ValueError('seeds must contain exactly init/data/loop/eval')
        seeds = dict(supplied)
        if 'seed' in config and seed != seeds['init']:
            raise ValueError('seed and seeds.init disagree')
    else:
        seeds = {name: int(canonical_hash(['rz1t-train-seeds-v1', seed, name])[:8], 16)
                 for name in ('init', 'data', 'loop', 'eval')}
    for name, value in seeds.items():
        if _integer(value, f'seeds.{name}', 0) >= 2**32:
            raise ValueError('PRNG seeds must fit uint32')
    return seeds


def environment_identity():
    packages = {}
    for name in ('jax', 'jaxlib', 'equinox', 'optax', 'numpy'):
        packages[name] = importlib.metadata.version(name)
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': packages, 'devices': [str(d.device_kind) for d in jax.devices()],
            'jax_enable_x64': bool(jax.config.jax_enable_x64),
            'jax_default_matmul_precision': str(jax.config.jax_default_matmul_precision),
            'XLA_FLAGS': os.environ.get('XLA_FLAGS', '')}


def source_identity():
    import z1t.components
    import z1t.model
    root = Path(__file__).parent
    files = {f'rz1t/{name}': file_hash(root / name) for name in
             ('train.py', 'checkpoint.py', 'model.py', 'config.py', 'conventions.py', 'flops.py',
              'diagnostics.py')}
    if (root / 'data.py').exists():
        files['rz1t/data.py'] = file_hash(root / 'data.py')
    for name, module in [('z1t/components.py', z1t.components), ('z1t/model.py', z1t.model)]:
        files[name] = file_hash(module.__file__)
    return files


def make_optimizer(config, target_budget, counts):
    options = config.get('optimizer', {})
    lr = options.get('lr', 3e-4)
    # Semantic work can exceed int32; keep the schedule in float32.
    step_work = float(counts['semantic_training_per_step'])
    warmup_work = float(options.get('warmup', 0)) * step_work
    target = float(target_budget)
    denom = max(target - warmup_work, 1.)
    ratio = options.get('final_lr_ratio', .1)
    def schedule(step):
        work = jnp.asarray(step, dtype=jnp.float32) * jnp.float32(step_work)
        fraction = jnp.clip((work - jnp.float32(warmup_work)) / jnp.float32(denom), 0., 1.)
        cosine = lr * (ratio + (1-ratio) * .5 * (1 + jnp.cos(jnp.pi * fraction)))
        if warmup_work:
            return jnp.where(work < warmup_work, lr * work / jnp.float32(warmup_work), cosine)
        return cosine
    return optax.chain(optax.clip_by_global_norm(options.get('grad_clip', 1.)),
                       optax.adamw(schedule, b1=options.get('b1', .9), b2=options.get('b2', .95),
                                   weight_decay=options.get('weight_decay', .1)))


def evaluate(model, dataset, eval_step, batch_size, sequence, split):
    weighted, tokens = 0., 0
    for x, y in dataset.eval_batches(batch_size, sequence, split=split):
        value = float(eval_step(model, x, y))
        weighted += value * int(y.size)
        tokens += int(y.size)
    if not tokens:
        raise ValueError('evaluation manifest has no full windows')
    return weighted / tokens, tokens


@eqx.filter_jit
def _finite_state(model, opt_state):
    checks = [jnp.all(jnp.isfinite(a)) for a in jax.tree.leaves((model, opt_state))
              if eqx.is_inexact_array(a)]
    return jnp.all(jnp.stack(checks))


@contextmanager
def _run_lock(out):
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.train.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('another trainer owns this run directory') from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def train(config, out, *, resume=False, max_steps=None):
    """Run to the *original* target; max_steps is an absolute debug stop step.

    Returns a final/incomplete record. Resume requires identical config, data,
    implementation bytes and environment. Evaluation cadence is in the config
    identity: compare independent runs, not modified resume identities. The
    sampler is random-window based; data cursor is the consumed batch count.
    """
    model_config, counts, schedule, budget = validate_config(config)
    seeds = _seeds(config)
    if max_steps is not None:
        _integer(max_steps, 'max_steps', 0)
    out = Path(out)
    with _run_lock(out):
        return _train_locked(config, model_config, counts, schedule, budget, seeds,
                             out, resume=resume, max_steps=max_steps)


def _train_locked(config, model_config, counts, schedule, budget, seeds, out, *, resume, max_steps):
    start = time.perf_counter()
    dataset = load_dataset(config.get('data', {'dataset_name': 'synthetic'}), model_config)
    identity = {'config_sha256': canonical_hash(config), 'data': dataset.identity,
                'source': source_identity(), 'environment': environment_identity()}
    digest = identity['config_sha256']
    default_id = (f"{config.get('lane', 'dev')}-{config['arm']}-{digest[:16]}"
                  if 'arm' in config else digest[:16])
    run_id = config.get('run_id', default_id)
    if (out / 'completion.json').exists():
        if not resume:
            raise ValueError('run already completed; use --resume to validate and skip')
        return validate_completion(out, identity)
    if not resume and ((out / 'identity.json').exists() or (out / 'checkpoints' / 'latest.json').exists()):
        raise ValueError('run already exists; use --resume (never overwrite an identity)')
    model = create_model(model_config, jax.random.PRNGKey(seeds['init']))
    optimizer = make_optimizer(config, budget, counts)
    opt_state = optimizer.init(eqx.filter(model, trainable_filter(model)))
    rngs = {name: jax.random.PRNGKey(seeds[seed_name]) for name, seed_name in
            [('model', 'init'), ('data', 'data'), ('loop', 'loop'), ('eval', 'eval')]}
    metadata = {'step': 0, 'data_cursor': 0, 'tokens': 0, 'semantic_work': 0,
                'executed_work_estimate': 0, 'metrics': [], 'elapsed_seconds': 0.,
                'training_seconds': 0., 'evaluation_seconds': 0.,
                'planned_steps': schedule['steps'], 'target_budget': budget,
                'accounting': counts, 'seed_mapping': seeds, 'model_failure': False}
    if resume:
        (model, opt_state, rngs), metadata, _ = load_checkpoint(
            out / 'checkpoints', (model, opt_state, rngs), identity)
        if metadata.get('model_failure'):
            raise ValueError('model-failure seed is terminal; do not retry until successful')
        if metadata['planned_steps'] != schedule['steps'] or metadata['target_budget'] != budget:
            raise ValueError('checkpoint schedule mismatch')
        if max_steps is not None and max_steps < metadata['step']:
            raise ValueError('max_steps precedes restored step')
    else:
        atomic_json(out / 'config.json', config)
        atomic_json(out / 'identity.json', identity)
    train_step, eval_step = make_steps(optimizer)
    training = config.get('training', {})
    batch = training.get('batch_size', 1)
    eval_every, ckpt_every = training.get('eval_every', 0), training.get('checkpoint_every', 100)
    split = config.get('data', {}).get('evaluation_split', 'development')
    probe_batches = training.get('diagnostic_probe_batches', 0)
    probe_seed = training.get('diagnostic_probe_seed', 1729)
    target = schedule['steps']
    stop = target if max_steps is None else min(target, max_steps)
    prior_elapsed = metadata['elapsed_seconds']

    def evaluate_row(row):
        tick = time.perf_counter()
        if probe_batches:
            diagnostics = evaluate_diagnostics(model, dataset, batch, model_config.sequence,
                                               split, probe_batches, probe_seed)
            row['diagnostics'] = diagnostics
            row['eval_nll'] = diagnostics['development']['nll']
            row['eval_tokens'] = diagnostics['development']['tokens']
            if not diagnostics['finite']:
                metadata['model_failure'] = True
        else:
            row['eval_nll'], row['eval_tokens'] = evaluate(
                model, dataset, eval_step, batch, model_config.sequence, split)
        metadata['evaluation_seconds'] += time.perf_counter() - tick
        if row['eval_nll'] is None or not math.isfinite(row['eval_nll']):
            row['eval_nll'] = None
            metadata['model_failure'] = True

    def checkpoint():
        metadata['elapsed_seconds'] = prior_elapsed + time.perf_counter() - start
        pointer = save_checkpoint(out / 'checkpoints', (model, opt_state, rngs), metadata, identity)
        # Metrics JSON is derived from the durable checkpoint prefix on resume.
        atomic_json(out / 'metrics.json', metadata['metrics'])
        return pointer

    if not resume:
        checkpoint()  # A first-step crash can resume the exact initial state.
    while metadata['step'] < stop:
        tick = time.perf_counter()
        next_data_key, batch_key = jax.random.split(rngs['data'])
        x, y = dataset.get_batch(batch_key, batch, model_config.sequence, split='train')
        new_model, new_opt, value = train_step(model, opt_state, x, y)
        nll = float(value)  # Synchronizes the step before timing/checkpointing.
        finite = math.isfinite(nll) and bool(_finite_state(new_model, new_opt))
        metadata['training_seconds'] += time.perf_counter() - tick
        if not finite:
            metadata['model_failure'] = True
            metadata['metrics'].append({'step': metadata['step'] + 1, 'event': 'model_failure',
                                         'reason': 'nonfinite loss/model/optimizer; prior finite state retained'})
            break
        model, opt_state = new_model, new_opt
        rngs['data'] = next_data_key
        metadata['step'] += 1
        metadata['data_cursor'] += 1
        metadata['tokens'] += counts['tokens_per_step']
        metadata['semantic_work'] += counts['semantic_training_per_step']
        metadata['executed_work_estimate'] += counts['executed_training_per_step_estimate']
        row = {'step': metadata['step'], 'train_nll': nll, 'tokens': metadata['tokens'],
               'semantic_work': metadata['semantic_work'], 'executed_work_estimate': metadata['executed_work_estimate']}
        if eval_every and metadata['step'] % eval_every == 0:
            evaluate_row(row)
        metadata['metrics'].append(row)
        if metadata['model_failure']:
            break
        if metadata['step'] % ckpt_every == 0:
            checkpoint()
    status = 'model_failure' if metadata['model_failure'] else ('completed' if metadata['step'] == target else 'incomplete')
    final_nll = None
    eval_tokens = 0
    final_diagnostics = None
    if status == 'completed':
        # Reuse a periodic final-step evaluation. When diagnostics are enabled,
        # keep the endpoint in the durable trajectory even off the cadence.
        last_row = metadata['metrics'][-1]
        final_row = last_row if probe_batches or 'eval_nll' in last_row else {}
        if 'eval_nll' not in final_row:
            evaluate_row(final_row)
        final_nll, eval_tokens = final_row['eval_nll'], final_row['eval_tokens']
        final_diagnostics = final_row.get('diagnostics')
        if metadata['model_failure'] or final_nll is None or not math.isfinite(final_nll):
            final_nll, status, metadata['model_failure'] = None, 'model_failure', True
    pointer = checkpoint()
    record = {'run_id': run_id, 'status': status, 'nll': final_nll,
              'checkpoint_selection': 'final_budget' if status == 'completed' else 'last_committed',
              'config_sha256': identity['config_sha256'], 'budget': budget,
              'step': metadata['step'], 'tokens': metadata['tokens'], 'eval_tokens': eval_tokens,
              'evaluation_split': split, 'semantic_work': metadata['semantic_work'],
              'executed_work_estimate': metadata['executed_work_estimate'],
              'work_units': counts['units'], 'accounting_version': ACCOUNTING_VERSION,
              'budget_overshoot_fraction': schedule['overshoot_fraction'] if status == 'completed' else None,
              'elapsed_seconds': metadata['elapsed_seconds'], 'training_seconds': metadata['training_seconds'],
              'evaluation_seconds': metadata['evaluation_seconds'], 'parameter_counts': parameter_counts(model),
              'checkpoint': pointer,
              'evidence_status': ('claim_a_confirmation_holdout_not_h1' if config.get('lane') == 'claim-a'
                                  else 'development_not_confirmatory')}
    if probe_batches:
        record['diagnostics'] = final_diagnostics
    for key in ('arm', 'pair_id'):
        if key in config:
            record[key] = config[key]
    if status == 'completed':
        write_completion(out, identity, pointer, record)
    else:
        atomic_json(out / 'final.json', record)
    # Direct input to analyze --metrics (one record per independent run).
    atomic_json(out / 'analysis_records.json', [record])
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-steps', type=int, help='absolute interruption debug step; does not change LR target')
    args = parser.parse_args(argv)
    try:
        config = yaml.safe_load(Path(args.config).read_text())
        record = train(config, args.out, resume=args.resume, max_steps=args.max_steps)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(json.dumps(record, sort_keys=True, allow_nan=False))
    return 2 if record['status'] == 'model_failure' else 0


if __name__ == '__main__':
    raise SystemExit(main())
