"""Trainable partition and token-weighted, natural-log language-model loss."""

import equinox as eqx
import jax
import optax


def trainable_filter(model):
    """Boolean PyTree: all inexact leaves except fixed positional encodings.

    Use ``eqx.partition(model, trainable_filter(model))`` both to initialize
    optimizer state and to differentiate. Fixed sparse indices remain arrays in
    the other partition, so they survive serialization without being optimized.
    Works on the canonical upstream model as well as RecurrentZ1T.
    """
    mask = jax.tree.map(eqx.is_inexact_array, model)
    return eqx.tree_at(lambda x: x.pe.pe, mask, False)


def loss(model, x, y):
    """Mean NLL (nats/token); equally weighted nonpadding tokens, no label shift.

    x/y must already be aligned next-token windows. Supports a single (T,)
    sequence or a (B,T) batch. No masking, packing, or implicit context carry.
    """
    if x.shape != y.shape or x.ndim not in (1, 2) or x.size == 0:
        raise ValueError("x and y must have equal, nonempty (T,) or (B,T) shapes")
    logits = model(x) if x.ndim == 1 else jax.vmap(model)(x)
    return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()


def make_steps(optimizer):
    """Return jitted ``train_step(model,state,x,y)`` and ``eval_step(model,x,y)``.

    Initialize state with optimizer.init(eqx.filter(model,trainable_filter(model))).
    Evaluation is deterministic and has no access to training RNG streams.
    """
    @eqx.filter_jit
    def train_step(model, opt_state, x, y):
        params, fixed = eqx.partition(model, trainable_filter(model))

        def objective(trainable):
            return loss(eqx.combine(trainable, fixed), x, y)

        value, grads = eqx.filter_value_and_grad(objective)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = eqx.apply_updates(params, updates)
        return eqx.combine(params, fixed), opt_state, value

    @eqx.filter_jit
    def eval_step(model, x, y):
        return loss(model, x, y)

    return train_step, eval_step


def parameter_counts(model) -> dict[str, int]:
    """Disjoint stored categories; PE and indices are not trainable parameters."""
    def count(tree, predicate=eqx.is_inexact_array):
        return sum(int(x.size) for x in jax.tree.leaves(eqx.filter(tree, predicate)))

    trainable = eqx.filter(model, trainable_filter(model))
    embedding = count(trainable.embedding)
    classifier = count(trainable.clf)
    total = count(trainable)
    return {
        "body": total - embedding - classifier,
        "embedding": embedding,
        "classifier": classifier,
        "pe": count(model.pe),
        "indices": count(model, lambda x: eqx.is_array(x) and not eqx.is_inexact_array(x)),
        "total_trainable": total,
    }
