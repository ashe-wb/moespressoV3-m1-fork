"""Fused cache-prior selection with original FP32-softmax contribution weights.

Slot maps are immutable published snapshots; 512 denotes a missing projection.
The caller owns snapshot freshness and request lifetime.
"""


from functools import cache
import math

import mlx.core as mx

from moespresso.runtime.qwen4.cache_routing_config import (
    DEFAULT_CACHE_BONUS as DEFAULT_CACHE_BONUS,
    DEFAULT_PROTECTED_ROUTES,
    validate_cache_factor,
    validate_protected_routes,
)

_HEADER = r"""
// Order-preserving rank of a routing key: every NaN ranks highest, zeros of
// either sign are equal, and larger values rank higher. With the expert id as
// the low word, a larger (rank, id) pair is the better route: the larger key,
// or the larger id between equal keys or two NaNs.
inline uint route_rank(float value) {
    if (metal::isnan(value)) return 0xffffffffu;
    uint bits = as_type<uint>(value == 0.0f ? 0.0f : value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Largest (rank, id) pair across a simdgroup; exhausted lanes pass (0, 0).
inline uint2 route_simd_max(uint rank, uint id) {
    uint top = simd_max(rank);
    return uint2(top, simd_max(rank == top ? id : 0u));
}

// Top ten routes of 512 keys, best first. Each simdgroup takes
// its own ten with SIMD reductions only, then one simdgroup merges the four
// sorted lists.
inline void route_top10(threadgroup float* keys, threadgroup uint* chosen,
                        threadgroup uint* list_ranks, threadgroup uint* list_ids,
                        uint tid, uint sg, uint lane) {
    uint ranks[4], ids[4];
    for (uint j = 0; j < 4; ++j) {
        ids[j] = tid * 4 + j;
        ranks[j] = route_rank(keys[ids[j]]);
    }
    // Sort the thread's four pairs best first.
    for (uint a = 0; a < 3; ++a) {
        for (uint b = 0; b < 3 - a; ++b) {
            bool swap = ranks[b + 1] > ranks[b]
                || (ranks[b + 1] == ranks[b] && ids[b + 1] > ids[b]);
            if (swap) {
                uint r = ranks[b]; ranks[b] = ranks[b + 1]; ranks[b + 1] = r;
                uint i = ids[b]; ids[b] = ids[b + 1]; ids[b + 1] = i;
            }
        }
    }
    uint head = 0;
    for (uint route = 0; route < 10; ++route) {
        uint rank = head < 4 ? ranks[head] : 0u;
        uint id = head < 4 ? ids[head] : 0u;
        uint2 best = route_simd_max(rank, id);
        if (head < 4 && rank == best.x && id == best.y) ++head;
        if (lane == 0) {
            list_ranks[sg * 10 + route] = best.x;
            list_ids[sg * 10 + route] = best.y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        uint position = 0;
        for (uint route = 0; route < 10; ++route) {
            bool live = lane < 4 && position < 10;
            uint rank = live ? list_ranks[lane * 10 + position] : 0u;
            uint id = live ? list_ids[lane * 10 + position] : 0u;
            uint2 best = route_simd_max(rank, id);
            if (live && rank == best.x && id == best.y) ++position;
            if (lane == 0) chosen[route] = best.y;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void route_sort(threadgroup float* probabilities,
                       threadgroup uint* chosen, threadgroup uint* ranked,
                       uint tid) {
    if (tid < 10) {
        uint id = chosen[tid], rank = 0;
        float score = probabilities[id];
        for (uint j = 0; j < 10; ++j) {
            uint other_id = chosen[j];
            float other = probabilities[other_id];
            bool sn = metal::isnan(score), on = metal::isnan(other);
            bool before = (!sn && on) || (sn == on &&
                (other > score || ((sn || other == score) && other_id < id)));
            rank += uint(before);
        }
        ranked[rank] = id;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""

_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint sg = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint row = threadgroup_position_in_grid.x;
    threadgroup float probabilities[512], keys[512];
    threadgroup float maxima[32], sums[32];
    threadgroup uint list_ranks[40];
    threadgroup uint list_ids[40];
    threadgroup uint invalid[4];
    threadgroup uint chosen[10], original[10], ranked[10];
    threadgroup uint apply_bias;
    threadgroup float denominator;
    float values[4];

    if (sg == 0) { maxima[lane] = Limits<float>::min; sums[lane] = 0.0f; }
    for (uint j = 0; j < 4; ++j) values[j] = float(logits[row * 512 + tid * 4 + j]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_max = Limits<float>::finite_min;
    uint bad = 0;
    for (uint j = 0; j < 4; ++j) {
        local_max = local_max < values[j] ? values[j] : local_max;
        bad |= uint(!metal::isfinite(values[j]));
    }
    local_max = simd_max(local_max);
    bad = simd_max(bad);
    if (lane == 0) { maxima[sg] = local_max; invalid[sg] = bad; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float maximum = simd_max(maxima[lane]);
        if (lane == 0) maxima[0] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        values[j] = fast::exp(values[j] - maxima[0]);
        local_sum += values[j];
    }
    local_sum = simd_sum(local_sum);
    if (lane == 0) sums[sg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float total = simd_sum(sums[lane]);
        if (lane == 0) sums[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = 1.0f / sums[0];
    for (uint j = 0; j < 4; ++j) {
        uint id = tid * 4 + j;
        probabilities[id] = values[j] * inverse;
        keys[id] = probabilities[id];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    route_top10(keys, chosen, list_ranks, list_ids, tid, sg, lane);
    route_sort(probabilities, chosen, original, tid);

    if (tid == 0) {
        apply_bias = 0;
        if (cache_factor[0] > 1.0f && !(invalid[0] | invalid[1] | invalid[2] | invalid[3])) {
            for (uint j = protected_routes[0]; j < 10; ++j) {
                uint id = original[j];
                if (gate_slots[id] >= 512 || up_slots[id] >= 512 || down_slots[id] >= 512)
                    apply_bias = 1;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (apply_bias) {
        for (uint j = 0; j < 4; ++j) {
            uint id = tid * 4 + j;
            bool hot = gate_slots[id] < 512 && up_slots[id] < 512 && down_slots[id] < 512;
            bool protect = false;
            for (uint k = 0; k < protected_routes[0]; ++k) protect |= id == original[k];
            keys[id] = protect ? INFINITY
                : probabilities[id] * (hot ? cache_factor[0] : 1.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        route_top10(keys, chosen, list_ranks, list_ids, tid, sg, lane);
        route_sort(probabilities, chosen, ranked, tid);
    } else {
        if (tid < 10) ranked[tid] = original[tid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        float sum = 0.0f;
        uint replacements = 0;
        for (uint j = 0; j < 10; ++j) {
            sum = probabilities[ranked[j]] + sum;
            bool present = false;
            for (uint k = 0; k < 10; ++k) present |= ranked[j] == original[k];
            replacements += uint(!present);
        }
        denominator = sum;
        changed[row] = replacements;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 10) {
        output_ids[row * 10 + tid] = ranked[tid];
        output_scores[row * 10 + tid] = static_cast<bfloat16_t>(probabilities[ranked[tid]] / denominator);
    }
"""


@cache
def _kernel():
    return mx.fast.metal_kernel(
        name="moespresso_cache_prior_router",
        input_names=["logits", "gate_slots", "up_slots", "down_slots", "cache_factor", "protected_routes"],
        output_names=["output_ids", "output_scores", "changed"],
        header=_HEADER, source=_SOURCE,
    )


@cache
def _resident_only_kernel():
    old = ": probabilities[id] * (hot ? cache_factor[0] : 1.0f);"
    if _SOURCE.count(old) != 1:
        raise RuntimeError("cache-prior selection source changed")
    source = _SOURCE.replace(old, ": (hot ? probabilities[id] : -INFINITY);")
    source += "    if (tid < 10) original_ids[row * 10 + tid] = original[tid];\n"
    return mx.fast.metal_kernel(
        name="moespresso_resident_only_router",
        input_names=["logits", "gate_slots", "up_slots", "down_slots", "cache_factor", "protected_routes"],
        output_names=["output_ids", "output_scores", "changed", "original_ids"],
        header=_HEADER, source=source,
    )


def resident_only_route(logits, slot_maps):
    """Select the ten strongest experts resident in all three maps.

    Contribution weights are the original probabilities normalized over the
    selection, as in ``cache_prior_route``. Also returns the original top ten
    in descending probability. Requires at least ten experts resident in every
    map and finite logits; otherwise nonresident routes can remain selected.
    """
    if (logits.size != 512 or logits.dtype != mx.bfloat16 or len(slot_maps) != 3
            or any(x.shape != (512,) or x.dtype != mx.uint32 for x in slot_maps)):
        raise ValueError("requires one BF16 router row and three uint32 expert-slot maps")
    shape = (*logits.shape[:-1], 10)
    return _resident_only_kernel()(
        inputs=[mx.contiguous(logits), *slot_maps, _bonus(math.log(2.0)), _protected_routes(0)],
        output_shapes=[shape, shape, (1,), shape],
        output_dtypes=[mx.uint32, mx.bfloat16, mx.uint32, mx.uint32],
        grid=(128, 1, 1), threadgroup=(128, 1, 1),
    )


@cache
def _bonus(value):
    try:
        factor = validate_cache_factor(math.exp(value))
    except (OverflowError, ValueError) as exc:
        raise ValueError("cache bonus must produce a finite multiplier from 1 to 8") from exc
    return mx.array([factor], dtype=mx.float32)


@cache
def _protected_routes(value):
    return mx.array([value], dtype=mx.uint32)


def cache_prior_route(
    logits, slot_maps, *, bonus=DEFAULT_CACHE_BONUS, protected_routes=DEFAULT_PROTECTED_ROUTES,
):
    """Route independent rows against the same immutable three-map snapshot.

    Live integration uses one-row decode. Multirow inputs are supported only
    as independent stateless test rows; this does not model evolving residency
    during a prefill chunk. Ranking multiplies original FP32 probabilities by
    exp(bonus), preserving their underflow and tie behavior. The strongest
    protected_routes original routes remain selected. The multiplier must be
    between one and eight; the number of protected routes is bounded by configuration.
    """
    if type(bonus) not in (float, int) or bonus < 0:
        raise ValueError("cache bonus must be finite and nonnegative")
    factor = _bonus(bonus)
    validate_protected_routes(protected_routes)
    if (logits.ndim < 2 or logits.shape[-1] != 512 or logits.dtype != mx.bfloat16
            or logits.size == 0 or len(slot_maps) != 3
            or any(x.shape != (512,) or x.dtype != mx.uint32 for x in slot_maps)):
        raise ValueError("requires BF16 router rows and three uint32 expert-slot maps")
    rows = logits.size // 512
    shape = (*logits.shape[:-1], 10)
    return _kernel()(
        inputs=[mx.contiguous(logits), *slot_maps, factor, _protected_routes(protected_routes)],
        output_shapes=[shape, shape, (rows,)],
        output_dtypes=[mx.uint32, mx.bfloat16, mx.uint32],
        grid=(rows * 128, 1, 1), threadgroup=(128, 1, 1),
    )
