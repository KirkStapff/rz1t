"""Weight-tied recurrent-depth Z1T using unmodified canonical upstream modules."""

import equinox as eqx
import jax
import jax.numpy as jnp
from z1t.model import create_model as create_upstream_model

from rz1t.config import ModelConfig


class RecurrentZ1T(eqx.Module):
    embedding: eqx.Module
    pe: eqx.Module
    prelude: tuple
    core: tuple
    coda: tuple
    norm: eqx.Module
    clf: eqx.Module
    config: ModelConfig = eqx.field(static=True)
    sequence: int = eqx.field(static=True)
    k: int = eqx.field(static=True)
    remat: bool = eqx.field(static=True)

    def __init__(self, config: ModelConfig, key):
        base = create_upstream_model(config.to_upstream(), key)
        # Upstream defaults can follow JAX's global x64 setting. The declared
        # lane always stores fp32 inexact arrays, including fixed PE.
        base = jax.tree.map(
            lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x,
            base,
        )
        self.embedding, self.pe = base.embedding, base.pe
        self.prelude = tuple(base.blocks[:config.p])
        self.core = tuple(base.blocks[config.p:config.p + config.m])
        self.coda = tuple(base.blocks[config.p + config.m:])
        self.norm, self.clf = base.norm, base.clf
        self.config = config
        self.sequence, self.k, self.remat = config.sequence, config.k, config.remat

    def _stack(self, blocks, h):
        def run(block, x):
            return block(x)
        apply = eqx.filter_checkpoint(run) if self.remat else run
        for block in blocks:
            h = apply(block, h)
        return h

    def hidden(self, tokens):
        """Teacher-forced final body state; context resets for every invocation."""
        if tokens.ndim != 1 or not 0 < tokens.shape[0] <= self.sequence:
            raise ValueError("tokens must be a nonempty vector no longer than sequence")
        if not jnp.issubdtype(tokens.dtype, jnp.integer):
            raise ValueError("tokens must have integer dtype")
        h = self.pe(jax.vmap(self.embedding)(tokens))
        h = self._stack(self.prelude, h)
        inject = self.config.injection
        # Skip-first residual: loop 0 is identical to injection=none, so k=1
        # residual matches upstream. Later loops receive h + post-prelude h0.
        if self.k == 1:
            h = self._stack(self.core, h)
        else:
            h0 = h

            def step(state, t):
                if inject == "residual":
                    z = jnp.where(t > 0, state + h0, state)
                else:
                    z = state
                return self._stack(self.core, z), None

            h, _ = jax.lax.scan(step, h, jnp.arange(self.k))
        return self._stack(self.coda, h)

    def __call__(self, tokens):
        return jax.vmap(self.clf)(self.norm(self.hidden(tokens)))


def create_model(config: ModelConfig, key) -> RecurrentZ1T:
    """No I/O; k>1 initialization equals an untied *unique-depth* model.

    The upstream split uses unique_depth+2 keys. Equal numeric seeds across
    different unique depths therefore do not imply equal initial functions or
    embedding initialization streams. Pairing must be specified in manifests.
    """
    if not isinstance(config, ModelConfig):
        raise TypeError("create_model requires ModelConfig")
    return RecurrentZ1T(config, key)
