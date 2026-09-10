"""Strict, immutable model configuration for the implemented fixed-k lane."""

from dataclasses import asdict, dataclass, fields
import math
from collections.abc import Mapping


@dataclass(frozen=True)
class ModelConfig:
    vocab: int = 50257
    sequence: int = 256
    n_embed: int = 384
    p: int = 0
    m: int = 6
    k: int = 2
    q: int = 0
    aft_heads: int = 8
    aft_ksize: int = 4
    linear_fan_in: int = 4
    tanh_linear: bool = True
    tanh_mlp: bool = True
    dyt_alpha: float = 0.5
    remat: bool = False
    injection: str = "none"
    dtype: str = "float32"

    def __post_init__(self):
        for name in ("vocab", "sequence", "n_embed", "m", "k", "aft_heads",
                     "aft_ksize", "linear_fan_in"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("p", "q"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("tanh_linear", "tanh_mlp", "remat"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.vocab < 2:
            raise ValueError("vocab must be at least 2")
        if self.n_embed % 2 or self.n_embed % self.aft_heads:
            raise ValueError("n_embed must be even and divisible by aft_heads")
        if self.linear_fan_in > self.n_embed:
            raise ValueError("linear_fan_in cannot exceed n_embed (no silent clamping)")
        if (isinstance(self.dyt_alpha, bool)
                or not isinstance(self.dyt_alpha, (float, int))
                or not math.isfinite(self.dyt_alpha)):
            raise ValueError("dyt_alpha must be a finite number")
        if self.injection not in ("none", "residual"):
            raise ValueError("injection must be 'none' or 'residual'")
        if self.dtype != "float32":
            raise ValueError("only dtype='float32' is implemented")

    @property
    def unique_depth(self) -> int:
        return self.p + self.m + self.q

    @property
    def applied_depth(self) -> int:
        return self.p + self.m * self.k + self.q

    @classmethod
    def from_dict(cls, values: Mapping):
        if not isinstance(values, Mapping):
            raise ValueError("model config must be a mapping")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown/unimplemented model options: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_upstream(self):
        """Initialize exactly one copy of each unique block using upstream keys."""
        from z1t.components import Config
        return Config(
            vocab=self.vocab, sequence=self.sequence, n_layers=self.unique_depth,
            n_embed=self.n_embed, aft_kind="conv", aft_heads=self.aft_heads,
            aft_ksize=self.aft_ksize, linear_fan_in=self.linear_fan_in,
            tanh_linear=self.tanh_linear, tanh_mlp=self.tanh_mlp,
            dyt_alpha=self.dyt_alpha, remat=self.remat,
        )
