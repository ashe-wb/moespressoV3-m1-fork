"""Qwen4 per-layer capacity profiles redistribute the planner total safely."""

import json

import pytest

from moespresso.runtime.qwen4.capacity_profile import (
    PROFILE_KIND,
    Qwen4CapacityProfileError,
    read_capacity_profile,
    scale_capacity_profile,
)

_PACKAGE = "pkg:" + "a" * 64


def _write(tmp_path, **overrides):
    data = {
        "kind": PROFILE_KIND,
        "version": 1,
        "package_manifest_id": _PACKAGE,
        "layers": [512, 512] + [200] * 46,
    }
    data.update(overrides)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data))
    return path


def test_profile_reads_bound_layers(tmp_path):
    weights = read_capacity_profile(_write(tmp_path), package_manifest_id=_PACKAGE, layers=48)
    assert weights[:3] == (512, 512, 200)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"package_manifest_id": "pkg:" + "b" * 64}, "made for"),
        ({"kind": "other"}, "must declare kind"),
        ({"layers": [100] * 47}, "48 positive integers"),
        ({"layers": [0] + [100] * 47}, "48 positive integers"),
        ({"layers": [True] + [100] * 47}, "48 positive integers"),
    ],
)
def test_profile_validation_fails_closed(tmp_path, overrides, message):
    with pytest.raises(Qwen4CapacityProfileError, match=message):
        read_capacity_profile(
            _write(tmp_path, **overrides), package_manifest_id=_PACKAGE, layers=48
        )


def test_unreadable_profile_fails_closed(tmp_path):
    with pytest.raises(Qwen4CapacityProfileError, match="cannot read"):
        read_capacity_profile(tmp_path / "missing.json", package_manifest_id=_PACKAGE, layers=48)


@pytest.mark.parametrize("uniform", [96, 181, 209, 300])
def test_scaling_keeps_total_and_bounds(uniform):
    weights = (512, 512, 256, 88, 112) + (200,) * 43
    total = uniform * 48
    capacities = scale_capacity_profile(weights, total, min_capacity=20, max_capacity=512)
    assert sorted(capacities) == list(range(48))
    assert sum(capacities.values()) <= total
    assert total - sum(capacities.values()) <= 0 or all(
        capacity == 512 for capacity in capacities.values()
    )
    assert all(20 <= capacity <= 512 for capacity in capacities.values())
    assert capacities[0] >= capacities[5] >= capacities[3]


def test_scaling_matches_the_reference_total():
    weights = (512, 512, 256, 208) + (184,) * 44
    total = sum(weights)
    assert scale_capacity_profile(weights, total, min_capacity=20, max_capacity=512) == dict(
        enumerate(weights)
    )


def test_scaling_refuses_totals_below_the_minimum():
    with pytest.raises(Qwen4CapacityProfileError, match="below the per-layer minimum"):
        scale_capacity_profile((1,) * 48, 48 * 19, min_capacity=20, max_capacity=512)
