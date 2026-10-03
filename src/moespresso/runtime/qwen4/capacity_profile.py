"""Per-layer routed-expert capacity profiles for bounded Qwen4 pools.

The capacity planner gives every routed layer the same slot count. Miss rates
differ widely by layer: the first two layers keep original routing and miss on
most decode tokens, while several deeper layers rarely miss. A profile states
relative per-layer capacities measured for one package. At load, the planner's
uniform total is redistributed in those proportions, so the expert-pool memory
is unchanged.

On a 32 GB M1 Max at the 24 GB planner ceiling (209 slots per layer), a profile
derived from decode route traces measured 14.9 -> 16.0 tokens/s over four
prompts that were not used to derive it. Cache-Prior routing depends on
residency, so outputs can differ from uniform capacity in the same way they
differ across capacities.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

PROFILE_KIND = "moespresso_qwen4_capacity_profile"
PROFILE_VERSION = 1
PROFILE_ENV = "MOESPRESSO_QWEN4_CAPACITY_PROFILE"


class Qwen4CapacityProfileError(ValueError):
    """Raised when a capacity profile cannot be applied."""


def read_capacity_profile(
    path: str | Path,
    *,
    package_manifest_id: str | None,
    layers: int,
) -> tuple[int, ...]:
    """Return a profile's relative per-layer capacities after validating it."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Qwen4CapacityProfileError(f"cannot read capacity profile {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise Qwen4CapacityProfileError("capacity profile must be a JSON object")
    if data.get("kind") != PROFILE_KIND or data.get("version") != PROFILE_VERSION:
        raise Qwen4CapacityProfileError(
            f"capacity profile must declare kind {PROFILE_KIND!r} version {PROFILE_VERSION}"
        )
    declared = data.get("package_manifest_id")
    if package_manifest_id is None or declared != package_manifest_id:
        raise Qwen4CapacityProfileError(
            f"capacity profile was made for {declared!r}, not {package_manifest_id!r}"
        )
    values = data.get("layers")
    if (
        not isinstance(values, list)
        or len(values) != layers
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values)
    ):
        raise Qwen4CapacityProfileError(
            f"capacity profile layers must be {layers} positive integers"
        )
    return tuple(values)


def scale_capacity_profile(
    weights: tuple[int, ...],
    total_slots: int,
    *,
    min_capacity: int,
    max_capacity: int,
) -> dict[int, int]:
    """Distribute ``total_slots`` in profile proportions within per-layer bounds.

    The result never exceeds ``total_slots``. Every layer receives at least
    ``min_capacity`` and at most ``max_capacity`` slots.
    """
    layers = len(weights)
    if not 0 < min_capacity <= max_capacity:
        raise Qwen4CapacityProfileError("capacity bounds are invalid")
    if total_slots < layers * min_capacity:
        raise Qwen4CapacityProfileError("capacity total is below the per-layer minimum")
    total_slots = min(total_slots, layers * max_capacity)

    def clamped(scale: float) -> list[int]:
        return [
            min(max_capacity, max(min_capacity, math.floor(weight * scale)))
            for weight in weights
        ]

    low, high = 0.0, max_capacity / min(weights)
    for _ in range(64):
        middle = (low + high) / 2
        if sum(clamped(middle)) <= total_slots:
            low = middle
        else:
            high = middle
    capacities = clamped(low)
    remainder = total_slots - sum(capacities)
    order = sorted(
        range(layers),
        key=lambda layer: (-(weights[layer] * low - math.floor(weights[layer] * low)), layer),
    )
    while remainder > 0:
        progressed = False
        for layer in order:
            if remainder == 0:
                break
            if capacities[layer] < max_capacity:
                capacities[layer] += 1
                remainder -= 1
                progressed = True
        if not progressed:
            break
    return dict(enumerate(capacities))
