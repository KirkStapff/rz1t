"""Auditable software storage, with explicitly hypothetical streaming state.

No byte count here is a measured silicon/pbit/chip count or an energy estimate.
Streaming decode is not implemented by this module.
"""

import equinox as eqx
import jax

from rz1t.conventions import parameter_counts, trainable_filter


def _bytes(tree, predicate=eqx.is_array) -> int:
    return sum(int(x.size * x.dtype.itemsize)
               for x in jax.tree.leaves(eqx.filter(tree, predicate)))


def resource_report(model, *, batch_size: int = 1) -> dict:
    """Actual array storage plus a labeled sufficient-cache design estimate.

    Cache estimate per applied block: (s-1)*(d+h) past eKV/eK values plus
    d+h cumulative sums, at the model's declared fp32 dtype. It excludes PE
    position counters, allocator overhead, logits, temporary workspace and
    autodiff tapes. It is not a peak-memory measurement or a decode claim.
    Weight traffic scenarios exclude cache/activation traffic and the fixed PE
    table. They apply identically to untied and tied models.
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    params = eqx.filter(model, trainable_filter(model))
    embedding_bytes, classifier_bytes = _bytes(params.embedding), _bytes(params.clf)
    weights = _bytes(params)
    indices = _bytes(model, lambda x: eqx.is_array(x) and not eqx.is_inexact_array(x))
    c = model.config
    core_storage = _bytes((model.core,))
    noncore_storage = _bytes((model.prelude, model.coda, model.norm))
    body_storage = core_storage + noncore_storage
    # At fp32 the conceptual retained history is different at each effective
    # depth even though every loop uses the same core parameter tree.
    cache = batch_size * c.applied_depth * c.aft_ksize * (c.n_embed + c.aft_heads) * 4
    hidden_bytes = batch_size * c.sequence * c.n_embed * 4
    retained = hidden_bytes if c.injection == "residual" and c.k > 1 else 0
    return {
        "parameters": parameter_counts(model),
        "unique_depth": c.unique_depth,
        "applied_depth": c.applied_depth,
        "stored_bytes": {
            "body_weights": weights - embedding_bytes - classifier_bytes,
            "embedding_weights": embedding_bytes,
            "classifier_weights": classifier_bytes,
            "gather_indices": indices,
            "fixed_pe": _bytes(model.pe),
            "all_arrays": _bytes(model),
        },
        "hypothetical_cache_bytes": cache,
        "retained_injection_h0_bytes": retained,
        "one_hidden_sequence_bytes": hidden_bytes,
        "body_weight_and_index_traffic_bytes": {
            "reload_each_application_per_sequence_batch": noncore_storage + c.k * core_storage,
            "load_unique_once_per_sequence_batch": body_storage,
            "resident_steady_state": 0,
        },
        "caveat": "Cache/traffic are design estimates, not measured decode, energy or silicon.",
    }
