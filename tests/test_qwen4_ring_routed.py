"""Ring-slot routed decode matches the mlx-iqk two-dispatch stages bit for bit."""

import mlx.core as mx
import numpy as np
import pytest
from mlx_iqk.format import component_dtypes, component_shapes
from mlx_iqk.nn import IqkSwitchLinear
from mlx_iqk.routed import SUPPORTED_CODEC_TUPLES, down_reduce, gate_up_swiglu

from moespresso.runtime.qwen4.ring_routed import (
    ring_routed_decode,
    ring_routed_supported,
    table_routed_decode,
)

_EXPERTS = 20


def _projection(member, output, width, seed, streams=None):
    module = IqkSwitchLinear(member, _EXPERTS, output, width)
    if streams is None:
        rng = np.random.default_rng(seed)
        streams = {}
        for name, shape in component_shapes(member, _EXPERTS, output, width).items():
            dtype = component_dtypes(member)[name]
            if np.issubdtype(dtype, np.floating):
                value = (rng.standard_normal(shape) * 0.0001).astype(dtype)
            else:
                value = rng.integers(0, np.iinfo(dtype).max, shape, dtype=dtype)
            streams[name] = mx.array(value)
    module.load_streams(streams)
    mx.eval(*module._streams())
    return module


@pytest.fixture(scope="module", params=SUPPORTED_CODEC_TUPLES)
def projections(request):
    gate, up, down = request.param
    return (
        _projection(gate, 640, 2560, 101),
        _projection(up, 640, 2560, 103),
        _projection(down, 2560, 768, 107),
    )


def _route(seed):
    rng = np.random.default_rng(seed)
    source = (rng.permutation(_EXPERTS)[:10] * 7 + 3).astype(np.uint32)
    gate_slots = rng.permutation(_EXPERTS)[:10].astype(np.uint32)
    up_slots = rng.permutation(_EXPERTS)[:10].astype(np.uint32)
    down_slots = rng.permutation(_EXPERTS)[:10].astype(np.uint32)
    hidden = mx.array(rng.standard_normal((1, 1, 2560)).astype(np.float32)).astype(mx.bfloat16)
    weights = mx.array(rng.random(10).astype(np.float32))
    scores = (weights / mx.sum(weights)).astype(mx.bfloat16)
    return hidden, source, scores, gate_slots, up_slots, down_slots


def _incumbent(gate, up, down, hidden, source, scores, gate_slots, down_slots):
    order = np.argsort(source, kind="stable")
    activation = gate_up_swiglu(gate, up, hidden, mx.array(gate_slots[order]))
    return down_reduce(down, activation, mx.array(down_slots[order]),
                       scores[mx.array(order.astype(np.uint32))])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_shared_gate_up_slots_match_incumbent_bit_for_bit(projections, seed):
    gate, up, down = projections
    assert ring_routed_supported(gate, up, down)
    hidden, source, scores, gate_slots, _up, down_slots = _route(seed)
    expected = _incumbent(gate, up, down, hidden, source, scores, gate_slots, down_slots)
    actual = ring_routed_decode(
        gate, up, down, hidden, mx.array(source), scores,
        mx.array(gate_slots), mx.array(gate_slots), mx.array(down_slots),
    )
    assert np.array_equal(np.array(actual.view(mx.uint16)), np.array(expected.view(mx.uint16)))


@pytest.mark.parametrize("seed", [3, 4])
def test_independent_up_slots_read_their_own_rows(projections, seed):
    gate, up, down = projections
    hidden, source, scores, gate_slots, up_slots, down_slots = _route(seed)
    # Place each route's up row at its gate slot, so the shared-slot incumbent
    # computes the same product as independent gate and up addressing.
    placement = np.arange(_EXPERTS)
    placement[gate_slots] = up_slots
    moved = {
        name: mx.take(stream, mx.array(placement), axis=0)
        for name, stream in zip(up.stream_names(), up._streams(), strict=True)
    }
    relocated_up = _projection(up.member, 640, 2560, 0, streams=moved)
    expected = _incumbent(gate, relocated_up, down, hidden, source, scores, gate_slots, down_slots)
    actual = ring_routed_decode(
        gate, up, down, hidden, mx.array(source), scores,
        mx.array(gate_slots), mx.array(up_slots), mx.array(down_slots),
    )
    assert np.array_equal(np.array(actual.view(mx.uint16)), np.array(expected.view(mx.uint16)))


def test_unsupported_codec_pairs_are_refused():
    gate = _projection("iq2_k", 640, 2560, 5)
    up = _projection("iq3_k", 640, 2560, 6)
    down = _projection("iq2_k", 2560, 768, 7)
    assert not ring_routed_supported(gate, up, down)


@pytest.mark.parametrize("seed", [5, 6])
def test_table_addressing_matches_ring_slots_bit_for_bit(projections, seed):
    gate, up, down = projections
    hidden, source, scores, gate_slots, up_slots, down_slots = _route(seed)
    tables = []
    for slots in (gate_slots, up_slots, down_slots):
        table = np.full(512, 511, dtype=np.uint32)
        table[source] = slots
        tables.append(mx.array(table))
    expected = ring_routed_decode(
        gate, up, down, hidden, mx.array(source), scores,
        mx.array(gate_slots), mx.array(up_slots), mx.array(down_slots),
    )
    actual = table_routed_decode(gate, up, down, hidden, mx.array(source), scores, *tables)
    assert np.array_equal(np.array(actual.view(mx.uint16)), np.array(expected.view(mx.uint16)))
