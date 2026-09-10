from dataclasses import replace

import pytest

from rz1t.config import ModelConfig


def test_defaults_and_roundtrip():
    config = ModelConfig()
    assert config.applied_depth == 12
    assert config.unique_depth == 6
    assert ModelConfig.from_dict(config.to_dict()) == config
    upstream = config.to_upstream()
    assert upstream.n_layers == 6
    assert upstream.aft_kind == "conv"
    assert upstream.linear_fan_in == 4


@pytest.mark.parametrize("values", [
    {"k": 0}, {"k": True}, {"k": 2.5}, {"p": -1}, {"q": False},
    {"m": 0}, {"sequence": 0}, {"vocab": 1}, {"n_embed": 15},
    {"n_embed": 14, "aft_heads": 4}, {"linear_fan_in": 0},
    {"linear_fan_in": 500}, {"linear_fan_in": None}, {"remat": "false"},
    {"tanh_linear": 1}, {"dyt_alpha": float("nan")}, {"dyt_alpha": True},
    {"injection": "gather"}, {"dtype": "bfloat16"}, {"random_k": True},
    {"bptt": "truncated"}, {"quantization": 4}, {"aft_kind": "full"},
])
def test_invalid_configs_fail_closed(values):
    with pytest.raises(ValueError):
        ModelConfig.from_dict(values)


def test_prelude_coda_depth():
    config = replace(ModelConfig(), p=1, q=2, m=3, k=4)
    assert config.unique_depth == 6
    assert config.applied_depth == 15


def test_residual_injection_is_allowed():
    config = ModelConfig.from_dict({"injection": "residual", "m": 2, "k": 4})
    assert config.injection == "residual"
    assert config.applied_depth == 8
