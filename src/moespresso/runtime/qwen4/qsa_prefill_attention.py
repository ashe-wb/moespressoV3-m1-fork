"""Fused QSA prefill attention over a prepared BF16 row bank.

Prefill gathers every query's selected K/V rows into a new array before
attention: about 2,048 rows of two 256-wide heads per query, or roughly 4 MB
per query, which limits query batches to 64 and dominates QSA prefill time.
This kernel reads the selected rows from the prepared bank by index instead.

The arithmetic follows ``qsa_attention_from_selected_rows``: BF16 logits,
scaled by ``head_dim ** -0.5``, a precise FP32 softmax over the valid
selected rows, BF16 probabilities, and an FP32 accumulation of the value rows
rounded once to BF16. Selected rows use the same set semantics as
``qsa_normalize_selected_rows``. Only accumulation order differs.
"""

from __future__ import annotations

from functools import cache
import struct

import mlx.core as mx

_HEAD_DIM = 256
_GROUP = 12
_THREADS = 256

_SOURCE = f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint query = threadgroup_position_in_grid.x;
    const uint kv_head = threadgroup_position_in_grid.y;
    const uint rows = params[0];
    const uint width = params[1];
    const uint query_heads = params[2];
    const float scale = as_type<float>(params[3]);

    threadgroup float q_tile[{_GROUP}][{_HEAD_DIM}];
    threadgroup float p_tile[{_GROUP}][{_THREADS}];
    threadgroup int idx_tile[{_THREADS}];
    threadgroup float red_m[{_THREADS // 32}][{_GROUP}];
    threadgroup float red_s[{_THREADS // 32}][{_GROUP}];

    for (uint i = tid; i < {_GROUP * _HEAD_DIM}u; i += {_THREADS}u) {{
        uint h = i / {_HEAD_DIM}u;
        uint d = i - h * {_HEAD_DIM}u;
        ulong qoff = ((ulong)query * query_heads + kv_head * {_GROUP}u + h) * {_HEAD_DIM}ul + d;
        q_tile[h][d] = float(queries[qoff]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const device int* sel = selected + (ulong)query * width;
    const ulong bank = (ulong)kv_head * rows * {_HEAD_DIM}ul;
    const uint lane = tid & 31u;
    const uint sg = tid >> 5u;

    // Pass 1: each thread owns rows tid, tid + T, ...; online max/sum per head.
    float m[{_GROUP}];
    float s[{_GROUP}];
    for (uint h = 0; h < {_GROUP}u; ++h) {{ m[h] = -INFINITY; s[h] = 0.0f; }}
    for (uint j = tid; j < width; j += {_THREADS}u) {{
        int idx = sel[j];
        if (!(idx >= 0 && (uint)idx < rows && (j == 0u || sel[j - 1u] != idx))) continue;
        const device bfloat16_t* krow = keys + bank + (ulong)idx * {_HEAD_DIM}ul;
        float dot[{_GROUP}];
        for (uint h = 0; h < {_GROUP}u; ++h) dot[h] = 0.0f;
        for (uint d = 0; d < {_HEAD_DIM}u; d += 4u) {{
            float4 k4 = float4(float(krow[d]), float(krow[d + 1]), float(krow[d + 2]), float(krow[d + 3]));
            for (uint h = 0; h < {_GROUP}u; ++h) {{
                dot[h] += q_tile[h][d] * k4.x + q_tile[h][d + 1] * k4.y
                        + q_tile[h][d + 2] * k4.z + q_tile[h][d + 3] * k4.w;
            }}
        }}
        for (uint h = 0; h < {_GROUP}u; ++h) {{
            float logit = float(bfloat16_t(float(bfloat16_t(dot[h])) * scale));
            if (logit > m[h]) {{ s[h] = s[h] * metal::precise::exp(m[h] - logit) + 1.0f; m[h] = logit; }}
            else {{ s[h] += metal::precise::exp(logit - m[h]); }}
        }}
    }}
    for (uint h = 0; h < {_GROUP}u; ++h) {{
        float gm = simd_max(m[h]);
        float term = s[h] > 0.0f ? s[h] * metal::precise::exp(m[h] - gm) : 0.0f;
        float gs = simd_sum(term);
        if (lane == 0u) {{ red_m[sg][h] = gm; red_s[sg][h] = gs; }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float fm[{_GROUP}];
    float fs[{_GROUP}];
    for (uint h = 0; h < {_GROUP}u; ++h) {{
        float mx_ = -INFINITY;
        for (uint g = 0; g < {_THREADS // 32}u; ++g) mx_ = max(mx_, red_m[g][h]);
        float sum = 0.0f;
        for (uint g = 0; g < {_THREADS // 32}u; ++g)
            if (red_s[g][h] > 0.0f) sum += red_s[g][h] * metal::precise::exp(red_m[g][h] - mx_);
        fm[h] = mx_;
        fs[h] = sum;
    }}

    // Pass 2: blocks of T rows; row-parallel probabilities, then dim-parallel PV.
    float acc[{_GROUP}];
    for (uint h = 0; h < {_GROUP}u; ++h) acc[h] = 0.0f;
    for (uint base = 0; base < width; base += {_THREADS}u) {{
        uint j = base + tid;
        int idx = -1;
        if (j < width) {{
            int candidate = sel[j];
            if (candidate >= 0 && (uint)candidate < rows && (j == 0u || sel[j - 1u] != candidate)) idx = candidate;
        }}
        if (idx >= 0) {{
            const device bfloat16_t* krow = keys + bank + (ulong)idx * {_HEAD_DIM}ul;
            float dot[{_GROUP}];
            for (uint h = 0; h < {_GROUP}u; ++h) dot[h] = 0.0f;
            for (uint d = 0; d < {_HEAD_DIM}u; d += 4u) {{
                float4 k4 = float4(float(krow[d]), float(krow[d + 1]), float(krow[d + 2]), float(krow[d + 3]));
                for (uint h = 0; h < {_GROUP}u; ++h) {{
                    dot[h] += q_tile[h][d] * k4.x + q_tile[h][d + 1] * k4.y
                            + q_tile[h][d + 2] * k4.z + q_tile[h][d + 3] * k4.w;
                }}
            }}
            for (uint h = 0; h < {_GROUP}u; ++h) {{
                float logit = float(bfloat16_t(float(bfloat16_t(dot[h])) * scale));
                p_tile[h][tid] = float(bfloat16_t(metal::precise::exp(logit - fm[h]) / fs[h]));
            }}
        }} else {{
            for (uint h = 0; h < {_GROUP}u; ++h) p_tile[h][tid] = 0.0f;
        }}
        idx_tile[tid] = idx;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint block = min((uint){_THREADS}u, width - base);
        for (uint r = 0; r < block; ++r) {{
            int ridx = idx_tile[r];
            if (ridx < 0) continue;
            float v = float(values[bank + (ulong)ridx * {_HEAD_DIM}ul + tid]);
            for (uint h = 0; h < {_GROUP}u; ++h) acc[h] += p_tile[h][r] * v;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    for (uint h = 0; h < {_GROUP}u; ++h) {{
        ulong ooff = ((ulong)query * query_heads + kv_head * {_GROUP}u + h) * {_HEAD_DIM}ul + tid;
        out[ooff] = fs[h] > 0.0f ? bfloat16_t(acc[h]) : bfloat16_t(0.0f);
    }}
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="moespresso_qwen4_qsa_prefill_attention",
        input_names=["queries", "keys", "values", "selected", "params"],
        output_names=["out"],
        source=_SOURCE,
    )


def fused_qsa_supported(queries: mx.array, keys: mx.array) -> bool:
    """Return whether the fused kernel matches this attention geometry."""
    return bool(
        queries.ndim == 4
        and keys.ndim == 4
        and queries.shape[0] == 1
        and keys.shape[0] == 1
        and queries.shape[-1] == _HEAD_DIM
        and keys.shape[-1] == _HEAD_DIM
        and queries.shape[2] == keys.shape[1] * _GROUP
        and queries.dtype == mx.bfloat16
        and keys.dtype == mx.bfloat16
    )


def fused_qsa_prefill_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    selected_indices: mx.array,
    *,
    scale: float,
) -> mx.array:
    """Attend ``[1, Q, H, 256]`` queries over selected rows of a ``[1, G, N, 256]`` bank."""
    if not fused_qsa_supported(queries, keys) or values.shape != keys.shape:
        raise ValueError("fused QSA prefill attention requires the released geometry")
    _, query_count, query_heads, _ = queries.shape
    kv_heads, rows = int(keys.shape[1]), int(keys.shape[2])
    sentinel = mx.array(rows, dtype=mx.int32)
    selected = selected_indices.reshape(query_count, -1).astype(mx.int32)
    normalized = mx.sort(
        mx.where((selected >= 0) & (selected < rows), selected, sentinel), axis=-1
    )
    width = int(normalized.shape[-1])
    params = mx.array(
        [rows, width, query_heads, struct.unpack("<I", struct.pack("<f", float(scale)))[0]],
        dtype=mx.uint32,
    )
    return _kernel()(
        inputs=[queries.reshape(-1), keys.reshape(-1), values.reshape(-1), normalized, params],
        grid=(query_count * _THREADS, kv_heads, 1),
        threadgroup=(_THREADS, 1, 1),
        output_shapes=[(1, query_count, query_heads, _HEAD_DIM)],
        output_dtypes=[mx.bfloat16],
    )[0]
