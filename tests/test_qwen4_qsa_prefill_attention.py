"""Fused QSA prefill attention matches the gathered reference arithmetic."""

import mlx.core as mx
import numpy as np
import pytest

from moespresso.runtime.qwen4.qsa import (
    qsa_attention_from_selected_rows,
    qsa_gather_selected_rows,
)
from moespresso.runtime.qwen4.qsa_prefill_attention import (
    fused_qsa_prefill_attention,
    fused_qsa_supported,
)


def _reference(queries, keys, values, selected):
    gathered_keys, gathered_values, valid = qsa_gather_selected_rows(keys, values, selected)
    mask = valid[:, :, None, :, None]
    return qsa_attention_from_selected_rows(
        queries,
        mx.where(mask, gathered_keys, 0),
        mx.where(mask, gathered_values, 0),
        valid,
        scale=256 ** -0.5,
    )


@pytest.mark.parametrize("queries_count, rows", [(9, 300), (33, 2600)])
def test_fused_prefill_attention_tracks_reference(queries_count, rows):
    rng = np.random.default_rng(queries_count)
    queries = mx.array(rng.standard_normal((1, queries_count, 24, 256)).astype(np.float32) * 0.5)
    keys = mx.array(rng.standard_normal((1, 2, rows, 256)).astype(np.float32) * 0.5)
    values = mx.array(rng.standard_normal((1, 2, rows, 256)).astype(np.float32))
    queries, keys, values = (x.astype(mx.bfloat16) for x in (queries, keys, values))
    selected = rng.integers(-1, rows + 20, (1, queries_count, 2051)).astype(np.int32)
    selected[0, 0, :] = -1
    selected[0, 1, 10:60] = selected[0, 1, 9]
    selected = mx.array(selected)

    expected = np.array(_reference(queries, keys, values, selected).astype(mx.float32))
    actual = np.array(
        fused_qsa_prefill_attention(queries, keys, values, selected, scale=256 ** -0.5).astype(
            mx.float32
        )
    )

    assert np.all(actual[0, 0] == 0)
    difference = np.abs(actual - expected)
    assert np.mean(difference == 0) > 0.99
    assert np.all(difference <= np.abs(expected) * 2 ** -6 + 1e-3)


def test_fused_prefill_attention_refuses_other_geometry():
    queries = mx.zeros((1, 2, 16, 256), dtype=mx.bfloat16)
    keys = mx.zeros((1, 2, 8, 256), dtype=mx.bfloat16)
    assert not fused_qsa_supported(queries, keys)
    with pytest.raises(ValueError, match="released geometry"):
        fused_qsa_prefill_attention(
            queries, keys, keys, mx.zeros((1, 2, 4), dtype=mx.int32), scale=0.0625
        )
