"""Two-dispatch routed decode for host-published Qwen4 ring slots.

The mlx-iqk routed stages take one sorted slot array for gate and up and read
it before the kernel runs. On the pipelined ring path the host publishes three
independent slot buffers after the graph is built, so these variants read the
published buffers inside the kernels, through a device-computed source order.
The gated hidden row is a kernel input, which places every slot read after the
event wait. Arithmetic, accumulation order and output rounding are those of the
mlx-iqk stages; only the slot addressing differs.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

_MEMBERS = ("iq2_k", "iq3_k")


def ring_routed_supported(gate, up, down) -> bool:
    """Return whether the three projections match the supported kernel geometry."""
    try:
        from mlx_iqk import routed
    except ImportError:
        return False
    return bool(
        gate.member == up.member
        and gate.member in _MEMBERS
        and down.member in _MEMBERS
        and (gate.member, up.member, down.member) in routed.SUPPORTED_CODEC_TUPLES
    )


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"mlx-iqk routed source changed near {old[:48]!r}")
    return source.replace(old, new)


def _gate_up_source(member: str) -> str:
    from mlx_iqk import routed

    source = routed.gate_up_source(member, member)
    source = _replace_once(
        source,
        routed._projection_code_reads("u", "rid", "kg", routed.HIDDEN, member),
        routed._projection_code_reads("u", "urid", "kg", routed.HIDDEN, member),
    )
    source = _replace_once(
        source,
        routed._projection_scale("u", "rid", "kg", routed.HIDDEN, member),
        routed._projection_scale("u", "urid", "kg", routed.HIDDEN, member),
    )
    source = _replace_once(
        source,
        "uint expert = sel[slot];",
        "uint route = order[slot];\n"
        "    uint expert = sel[route];\n"
        "    uint uexpert = usel[route];",
    )
    return _replace_once(
        source,
        "ulong rid = (ulong)expert * ",
        f"ulong urid = (ulong)uexpert * {routed.INTERMEDIATE}ul + (ulong)(row0 + row);\n"
        "        ulong rid = (ulong)expert * ",
    )


def _down_source(member: str) -> str:
    from mlx_iqk import routed

    return _replace_once(
        routed.down_reduce_source(member),
        "uint expert = sel[slot];",
        "uint expert = sel[order[slot]];",
    )


@cache
def _gate_up_kernel(member: str):
    from mlx_iqk import routed

    return mx.fast.metal_kernel(
        name=f"moespresso_qwen4_ring_gate_up_swiglu_{member}",
        input_names=[*routed._gate_up_inputs(member, member), "usel", "order"],
        output_names=["out"],
        source=_gate_up_source(member),
    )


@cache
def _down_kernel(member: str):
    from mlx_iqk import routed

    return mx.fast.metal_kernel(
        name=f"moespresso_qwen4_ring_down_reduce_{member}",
        input_names=[*routed._down_inputs(member), "order"],
        output_names=["out"],
        source=_down_source(member),
    )


_ROUTE_ORDER = """
    threadgroup uint route_of[10];
    if (thread_position_in_threadgroup.x < 10u) {
        uint mine = ids[thread_position_in_threadgroup.x];
        uint rank = 0u;
        for (uint other = 0u; other < 10u; ++other) {
            rank += uint(ids[other] < mine);
        }
        route_of[rank] = thread_position_in_threadgroup.x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
"""


def _table_gate_up_source(member: str) -> str:
    source = _gate_up_source(member)
    source = _replace_once(
        source,
        "    uint route = order[slot];\n"
        "    uint expert = sel[route];\n"
        "    uint uexpert = usel[route];",
        _ROUTE_ORDER
        + "    uint route = route_of[slot];\n"
        "    uint expert = sel[ids[route]];\n"
        "    uint uexpert = usel[ids[route]];",
    )
    return source


def _table_down_source(member: str) -> str:
    from mlx_iqk import routed

    source = _replace_once(
        routed.down_reduce_source(member),
        "uint expert = sel[slot];",
        _ROUTE_ORDER + "    uint expert = sel[ids[route_of[slot]]];",
    )
    return _replace_once(
        source,
        "bfloat product = value * scores[position];",
        "bfloat product = value * scores[route_of[position]];",
    )


@cache
def _table_gate_up_kernel(member: str):
    from mlx_iqk import routed

    return mx.fast.metal_kernel(
        name=f"moespresso_qwen4_table_gate_up_swiglu_{member}",
        input_names=[*routed._gate_up_inputs(member, member), "usel", "ids"],
        output_names=["out"],
        source=_table_gate_up_source(member),
    )


@cache
def _table_down_kernel(member: str):
    from mlx_iqk import routed

    return mx.fast.metal_kernel(
        name=f"moespresso_qwen4_table_down_reduce_{member}",
        input_names=[*routed._down_inputs(member), "ids"],
        output_names=["out"],
        source=_table_down_source(member),
    )


def table_routed_decode(
    gate,
    up,
    down,
    hidden: mx.array,
    route_ids: mx.array,
    scores: mx.array,
    gate_table: mx.array,
    up_table: mx.array,
    down_table: mx.array,
) -> mx.array:
    """Return one reduced BF16 routed row, resolving slots from expert-id tables.

    The kernels order the ten distinct route ids and look up each projection's
    slot in its 512-entry table, so the route ordering, score permutation and
    slot gathers of ``ring_routed_decode`` need no separate dispatches. The
    arithmetic and reduction order are those of ``ring_routed_decode``.
    """
    from mlx_iqk import routed

    if int(route_ids.size) != routed.TOP_K or int(hidden.size) != routed.HIDDEN:
        raise ValueError("table routed decode requires one row and ten routes")
    ids = route_ids.reshape(-1).astype(mx.uint32)
    blocks = routed.TOP_K * (routed.INTERMEDIATE // routed.ROWS_PER_TG)
    activation = _table_gate_up_kernel(gate.member)(
        inputs=[
            hidden.reshape(1, routed.HIDDEN).astype(mx.float16),
            *gate._streams(),
            *up._streams(),
            routed.member_table(gate.member),
            gate_table,
            routed.sigmoid_fp16_table(),
            up_table,
            ids,
        ],
        grid=(blocks * routed._GATE_UP_THREADS, 1, 1),
        threadgroup=(routed._GATE_UP_THREADS, 1, 1),
        output_shapes=[(routed.TOP_K * routed.INTERMEDIATE,)],
        output_dtypes=[mx.float16],
    )[0]
    tiles = routed.HIDDEN // routed.ROWS_PER_TG
    return _table_down_kernel(down.member)(
        inputs=[
            activation.reshape(routed.TOP_K, routed.INTERMEDIATE),
            *down._streams(),
            routed.member_table(down.member),
            down_table,
            scores.reshape(-1),
            ids,
        ],
        grid=(tiles * routed._DOWN_THREADS, 1, 1),
        threadgroup=(routed._DOWN_THREADS, 1, 1),
        output_shapes=[(routed.HIDDEN,)],
        output_dtypes=[mx.bfloat16],
    )[0]


def ring_routed_decode(
    gate,
    up,
    down,
    hidden: mx.array,
    route_ids: mx.array,
    scores: mx.array,
    gate_slots: mx.array,
    up_slots: mx.array,
    down_slots: mx.array,
) -> mx.array:
    """Return one reduced BF16 routed row from route-ordered slot buffers.

    ``gate_slots``, ``up_slots`` and ``down_slots`` hold the slot of each route
    position. Routes are reduced in ascending source-expert order, matching
    ``mlx_iqk.routed.routed_moe``.
    """
    from mlx_iqk import routed

    if int(route_ids.size) != routed.TOP_K or int(hidden.size) != routed.HIDDEN:
        raise ValueError("ring routed decode requires one row and ten routes")
    order = mx.argsort(route_ids.reshape(-1)).astype(mx.uint32)
    sorted_scores = scores.reshape(-1)[order]
    blocks = routed.TOP_K * (routed.INTERMEDIATE // routed.ROWS_PER_TG)
    activation = _gate_up_kernel(gate.member)(
        inputs=[
            hidden.reshape(1, routed.HIDDEN).astype(mx.float16),
            *gate._streams(),
            *up._streams(),
            routed.member_table(gate.member),
            gate_slots.reshape(-1),
            routed.sigmoid_fp16_table(),
            up_slots.reshape(-1),
            order,
        ],
        grid=(blocks * routed._GATE_UP_THREADS, 1, 1),
        threadgroup=(routed._GATE_UP_THREADS, 1, 1),
        output_shapes=[(routed.TOP_K * routed.INTERMEDIATE,)],
        output_dtypes=[mx.float16],
    )[0]
    tiles = routed.HIDDEN // routed.ROWS_PER_TG
    return _down_kernel(down.member)(
        inputs=[
            activation.reshape(routed.TOP_K, routed.INTERMEDIATE),
            *down._streams(),
            routed.member_table(down.member),
            down_slots.reshape(-1),
            sorted_scores,
            order,
        ],
        grid=(tiles * routed._DOWN_THREADS, 1, 1),
        threadgroup=(routed._DOWN_THREADS, 1, 1),
        output_shapes=[(routed.HIDDEN,)],
        output_dtypes=[mx.bfloat16],
    )[0]
