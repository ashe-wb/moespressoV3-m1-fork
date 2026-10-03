"""Fused one-token QSA attention over a mutable KVarN K4/V4 selection.

The decode path previously gathered every selected row into dense BF16 K/V
arrays and attended with MLX matrix products. These kernels reconstruct each
selected row with the gather kernel's arithmetic, including the exact sink,
exact tail and pending rows, and consume it directly:

1. ``logits``: per KV head and row block, reconstruct the key row, reduce its
   dot product with the head group's twelve queries, round to BF16 and apply
   the exact power-of-two scale.
2. softmax: per query head, the masked FP32 maximum, exponential sum and
   BF16 probabilities in one threadgroup.
3. ``weighted``: per KV head and row block, reconstruct the value row and
   accumulate BF16 probabilities times values in FP32 partial sums.
4. ``reduce``: sum the partial blocks in block order and round to BF16.

The arithmetic types match the gathered path; the reduction orders of the dot
products and the softmax sum differ, so outputs can differ by one BF16 step.
"""

from __future__ import annotations

from functools import cache

import mlx.core as mx

_HEADS_PER_GROUP = 12
_LOGIT_ROWS = 4
_WEIGHTED_ROWS = 32

_ROW_PROLOGUE = r"""
    int logical = logical_ids[row];
    bool usable = valid_rows[row] && logical >= 0 && logical < frontier_value + pending_count;
"""

_RECONSTRUCT = r"""
    float element = 0.0f;
    if (usable) {
        if (logical < 128) {
            element = float(SINK[((uint)head * 128u + (uint)logical) * 256u + channel]);
        } else if (logical >= body_frontier) {
            if (logical < frontier_value) {
                int local = (logical - 128) % (int)TAIL_shape[2];
                element = float(TAIL[((uint)head * (uint)TAIL_shape[2] + (uint)local) * 256u + channel]);
            } else {
                int local = logical - frontier_value;
                element = float(PENDING[((uint)head * (uint)pending_count + (uint)local) * 256u + channel]);
            }
        } else {
            int body_local = logical - 128;
            int tile = body_local / 128;
            int token = body_local % 128;
            device const uchar *record =
                (device const uchar *)records + ((uint64_t)tile * 2u + (uint)head) * 35072u;
            uint code_column = channel >> 1u;
            uint shift = (channel & 1u) * 4u;
            DECODE
            for (uint stride = 1u; stride < 32u; stride <<= 1u) {
                float other = simd_shuffle_xor(value, stride);
                value = (channel & stride) ? other - value : value + other;
            }
            butterfly[channel] = value;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = 32u; stride <= 128u; stride <<= 1u) {
                float other = butterfly[channel ^ stride];
                value = (channel & stride) ? other - value : value + other;
                if (stride < 128u) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    butterfly[channel] = value;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            element = float(bfloat(value * 0.0625f));
        }
    }
"""

_KEY_DECODE = r"""
            uchar packed = record[(uint64_t)token * 128u + code_column];
            float code = float((packed >> shift) & 0x0fu);
            device const half *k_scale = (device const half *)(record + 16384u);
            device const half *k_zero = (device const half *)(record + 16896u);
            device const half *k_token_scale = (device const half *)(record + 17408u);
            float value = (code * float(k_scale[channel]) + float(k_zero[channel]))
                * float(k_token_scale[token]);
"""

_VALUE_DECODE = r"""
            uchar packed = record[17664u + (uint64_t)token * 128u + code_column];
            float code = float((packed >> shift) & 0x0fu);
            device const half *v_channel_scale = (device const half *)(record + 34048u);
            device const half *v_token_scale = (device const half *)(record + 34560u);
            device const half *v_zero = (device const half *)(record + 34816u);
            float value = (code * float(v_token_scale[token]) + float(v_zero[token]))
                * float(v_channel_scale[channel]);
"""


def _reconstruct(kind: str) -> str:
    if kind == "key":
        names = {"SINK": "sink_keys", "TAIL": "tail_keys", "PENDING": "pending_keys"}
        decode = _KEY_DECODE
    else:
        names = {"SINK": "sink_values", "TAIL": "tail_values", "PENDING": "pending_values"}
        decode = _VALUE_DECODE
    source = _RECONSTRUCT.replace("DECODE", decode.strip())
    for token, name in names.items():
        source = source.replace(f"{token}_shape", f"{name}_shape").replace(token, name)
    return source


_COMMON = r"""
    uint channel = thread_position_in_threadgroup.x;
    uint block = threadgroup_position_in_grid.x;
    uint head = threadgroup_position_in_grid.y;
    int width = (int)logical_ids_shape[0];
    int frontier_value = frontier[0];
    int body_frontier = 128 + record_count[0] * 128;
    int pending_count = (int)pending_keys_shape[2];
    threadgroup float butterfly[256];
"""

_LOGITS_SOURCE = (
    _COMMON
    + r"""
    uint lane = channel & 31u;
    uint group = channel >> 5u;
    float query[12];
    for (uint h = 0u; h < 12u; ++h) {
        query[h] = float(queries[((uint)head * 12u + h) * 256u + channel]);
    }
    threadgroup float partial[8][12];
    for (int index = 0; index < ROWS; ++index) {
        int row = (int)block * ROWS + index;
        if (row >= width) break;
"""
    + _ROW_PROLOGUE
    + r"""
        if (!usable) {
            if (channel < 12u) {
                logits[((uint)head * 12u + channel) * (uint)width + (uint)row] = bfloat(0.0f);
            }
            continue;
        }
"""
    + _reconstruct("key")
    + r"""
        for (uint h = 0u; h < 12u; ++h) {
            float product = simd_sum(query[h] * element);
            if (lane == 0u) partial[group][h] = product;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (channel < 12u) {
            float total = partial[0][channel];
            for (uint g = 1u; g < 8u; ++g) total += partial[g][channel];
            bfloat rounded = bfloat(total);
            logits[((uint)head * 12u + channel) * (uint)width + (uint)row] =
                bfloat(float(rounded) * scale[0]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
)

_WEIGHTED_SOURCE = (
    _COMMON
    + r"""
    float accumulator[12] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (int index = 0; index < ROWS; ++index) {
        int row = (int)block * ROWS + index;
        if (row >= width) break;
"""
    + _ROW_PROLOGUE
    + r"""
        if (!usable) continue;
"""
    + _reconstruct("value")
    + r"""
        for (uint h = 0u; h < 12u; ++h) {
            accumulator[h] += float(probabilities[((uint)head * 12u + h) * (uint)width + (uint)row]) * element;
        }
    }
    uint blocks = threadgroup_position_in_grid.x;
    for (uint h = 0u; h < 12u; ++h) {
        partials[(((uint64_t)blocks * 24u) + (uint)head * 12u + h) * 256u + channel] = accumulator[h];
    }
"""
)

_SOFTMAX_SOURCE = r"""
    uint head = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid & 31u;
    uint group = tid >> 5u;
    int width = (int)valid_rows_shape[0];
    threadgroup float shared[8];
    threadgroup float result;
    float maximum = -INFINITY;
    for (int i = (int)tid; i < width; i += 256) {
        if (valid_rows[i]) maximum = max(maximum, float(logits[(uint)head * (uint)width + (uint)i]));
    }
    maximum = simd_max(maximum);
    if (lane == 0u) shared[group] = maximum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float value = shared[0];
        for (uint g = 1u; g < 8u; ++g) value = max(value, shared[g]);
        result = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    maximum = result;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (int i = (int)tid; i < width; i += 256) {
        if (valid_rows[i]) total += metal::precise::exp(float(logits[(uint)head * (uint)width + (uint)i]) - maximum);
    }
    total = simd_sum(total);
    if (lane == 0u) shared[group] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float value = shared[0];
        for (uint g = 1u; g < 8u; ++g) value += shared[g];
        result = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = result;
    bool any_valid = maximum > -INFINITY;
    for (int i = (int)tid; i < width; i += 256) {
        float probability = 0.0f;
        if (any_valid && valid_rows[i]) {
            probability = metal::precise::exp(float(logits[(uint)head * (uint)width + (uint)i]) - maximum) / total;
        }
        probabilities[(uint)head * (uint)width + (uint)i] = bfloat(probability);
    }
"""


_REDUCE_SOURCE = r"""
    uint index = thread_position_in_grid.x;
    if (index >= 24u * 256u) return;
    int blocks = (int)partials_shape[0];
    float total = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        total += partials[(uint64_t)b * 24u * 256u + index];
    }
    output[index] = bfloat(total);
"""

_STORAGE_INPUTS = [
    "records",
    "sink_keys",
    "sink_values",
    "tail_keys",
    "tail_values",
    "pending_keys",
    "pending_values",
    "logical_ids",
    "valid_rows",
    "record_count",
    "frontier",
]


@cache
def _logits_kernel():
    return mx.fast.metal_kernel(
        name="moespresso_qwen4_qsa_decode_logits",
        input_names=[*_STORAGE_INPUTS, "queries", "scale"],
        output_names=["logits"],
        source=_LOGITS_SOURCE.replace("ROWS", str(_LOGIT_ROWS)),
    )


@cache
def _weighted_kernel():
    return mx.fast.metal_kernel(
        name="moespresso_qwen4_qsa_decode_weighted",
        input_names=[*_STORAGE_INPUTS, "probabilities"],
        output_names=["partials"],
        source=_WEIGHTED_SOURCE.replace("ROWS", str(_WEIGHTED_ROWS)),
    )


@cache
def _softmax_kernel():
    return mx.fast.metal_kernel(
        name="moespresso_qwen4_qsa_decode_softmax",
        input_names=["logits", "valid_rows"],
        output_names=["probabilities"],
        source=_SOFTMAX_SOURCE,
    )


@cache
def _reduce_kernel():
    return mx.fast.metal_kernel(
        name="moespresso_qwen4_qsa_decode_reduce",
        input_names=["partials"],
        output_names=["output"],
        source=_REDUCE_SOURCE,
    )


def qsa_kvarn_decode_attention(
    queries: mx.array,
    records: mx.array,
    exact_sink_keys: mx.array,
    exact_sink_values: mx.array,
    exact_tail_keys: mx.array,
    exact_tail_values: mx.array,
    pending_keys: mx.array,
    pending_values: mx.array,
    normalized: mx.array,
    valid: mx.array,
    *,
    frontier: int,
    record_count: int,
    scale: float,
) -> mx.array:
    """Return ``[1, 1, 24, 256]`` BF16 attention for one decode query.

    ``normalized`` and ``valid`` are the ascending selection and validity
    produced by ``qsa_normalize_selected_rows``. Storage arrays follow the
    mutable KVarN gather contract.
    """
    if queries.shape != (1, 1, 24, 256) or queries.dtype != mx.bfloat16:
        raise ValueError("fused QSA decode attention requires one [1, 1, 24, 256] BF16 query")
    if normalized.ndim != 3 or normalized.shape[:2] != (1, 1) or valid.shape != normalized.shape:
        raise ValueError("fused QSA decode attention requires one normalized selection")
    width = int(normalized.shape[-1])
    ids = mx.contiguous(normalized.reshape(-1))
    flags = mx.contiguous(valid.reshape(-1))
    storage = [
        mx.contiguous(records),
        mx.contiguous(exact_sink_keys),
        mx.contiguous(exact_sink_values),
        mx.contiguous(exact_tail_keys),
        mx.contiguous(exact_tail_values),
        mx.contiguous(pending_keys),
        mx.contiguous(pending_values),
        ids,
        flags,
        mx.array([record_count], dtype=mx.int32),
        mx.array([frontier], dtype=mx.int32),
    ]
    logit_blocks = -(-width // _LOGIT_ROWS)
    logits = _logits_kernel()(
        inputs=[*storage, mx.contiguous(queries.reshape(-1)), mx.array([scale], dtype=mx.float32)],
        output_shapes=[(24, width)],
        output_dtypes=[mx.bfloat16],
        grid=(logit_blocks * 256, 2, 1),
        threadgroup=(256, 1, 1),
    )[0]
    probabilities = _softmax_kernel()(
        inputs=[logits, flags],
        output_shapes=[(24, width)],
        output_dtypes=[mx.bfloat16],
        grid=(24 * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]
    weighted_blocks = -(-width // _WEIGHTED_ROWS)
    partials = _weighted_kernel()(
        inputs=[*storage, probabilities],
        output_shapes=[(weighted_blocks, 24, 256)],
        output_dtypes=[mx.float32],
        grid=(weighted_blocks * 256, 2, 1),
        threadgroup=(256, 1, 1),
    )[0]
    output = _reduce_kernel()(
        inputs=[partials],
        output_shapes=[(24 * 256,)],
        output_dtypes=[mx.bfloat16],
        grid=(24 * 256, 1, 1),
        threadgroup=(256, 1, 1),
    )[0]
    return output.reshape(1, 1, 24, 256)
