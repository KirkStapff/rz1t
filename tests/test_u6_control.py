"""U6 is a same-storage development control, not a confirmation arm.

These tests validate the local recipe, not historical dataset/source identity.
No data loading, training, or cloud calls are made.
"""
from copy import deepcopy
from pathlib import Path

import equinox as eqx
import jax
import numpy as np
import pytest
import yaml

from rz1t.conventions import parameter_counts
from rz1t.model import create_model
from rz1t.train import _seeds, validate_config


CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "dev"


def config(arm, seed):
    suffix = f"-s{seed}" if seed else ""
    path = CONFIGS / f"owt-{arm.lower()}-1e16{suffix}.yaml"
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_u6_recipe_changes_only_architecture_and_arm_identity(seed):
    u6 = config("U6", seed)
    for arm in ("R6x2", "U12"):
        baseline = config(arm, seed)
        expected = deepcopy(baseline)
        expected.update(arm="U6", run_id=f"owt-dev-1e16-s{seed}-U6")
        expected["model"].update(m=6, k=1)
        # Includes data, optimizer, precision, budget and logging fields.
        assert u6 == expected
        assert _seeds(u6) == _seeds(baseline)
    assert u6["lane"] == "explore"
    assert u6["data"]["evaluation_split"] == "development"
    assert u6["pair_id"] == f"owt-dev-1e16-s{seed}"
    assert "steps" not in u6["training"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_u6_same_full_size_stored_arrays_and_counts_as_r6x2(seed):
    u6_config, *_ = validate_config(config("U6", seed))
    r6_config, *_ = validate_config(config("R6x2", seed))
    key = jax.random.PRNGKey(_seeds(config("U6", seed))["init"])
    u6 = create_model(u6_config, key)
    r6 = create_model(r6_config, key)
    assert u6_config.unique_depth == r6_config.unique_depth == 6
    assert (u6_config.applied_depth, r6_config.applied_depth) == (6, 12)
    assert parameter_counts(u6) == parameter_counts(r6)
    assert parameter_counts(u6)["body"] == 102589
    # All stored arrays, including sparse indices, PE, embedding and classifier.
    # Static k/config metadata intentionally differs; initial functions do too.
    left = jax.tree.leaves(eqx.filter(u6, eqx.is_array))
    right = jax.tree.leaves(eqx.filter(r6, eqx.is_array))
    assert len(left) == len(right)
    for a, b in zip(left, right, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_u6_budget_derives_longer_schedule_not_copied_depth12_steps(seed):
    schedules = {}
    for arm in ("U6", "R6x2", "U12"):
        cfg = config(arm, seed)
        _, work, schedule, budget = validate_config(cfg)
        per_step = work["semantic_training_per_step"]
        expected = (budget + per_step - 1) // per_step
        assert schedule["steps"] == expected
        assert (expected - 1) * per_step < budget <= expected * per_step
        assert 0 <= schedule["overshoot_fraction"] <= 0.01
        assert work["tokens_per_step"] == 64 * 256
        schedules[arm] = (expected, per_step)
    assert schedules["U12"] == schedules["R6x2"]
    assert schedules["U6"][0] > schedules["R6x2"][0]
    assert schedules["U6"][1] < schedules["R6x2"][1]
    # Explicit regression guard: this protocol is matched budget, not tokens.
    bad = config("U6", seed)
    bad["training"]["steps"] = schedules["R6x2"][0]
    with pytest.raises(ValueError, match="steps disagree with semantic budget"):
        validate_config(bad)


def test_u6_run_identities_are_unique_across_three_arm_development_matrix():
    runs = [config(arm, seed) for seed in (0, 1, 2)
            for arm in ("U6", "R6x2", "U12")]
    assert len({run["run_id"] for run in runs}) == 9
    streams = [_seeds(config("U6", seed)) for seed in (0, 1, 2)]
    for name in ("init", "data", "loop", "eval"):
        assert len({stream[name] for stream in streams}) == 3
