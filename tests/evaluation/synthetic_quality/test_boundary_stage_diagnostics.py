from __future__ import annotations

import numpy as np
import pytest

from pyosv.evaluation.synthetic_quality.boundary_stage_diagnostics import (
    stage_transition_correspondence_metrics,
)


def _metrics(
    source: np.ndarray,
    target: np.ndarray,
    *,
    radius: float = 0.0,
    margin: int = 1,
) -> dict[str, object]:
    return stage_transition_correspondence_metrics(
        source, target, match_radius=radius, edge_margin=margin
    )


def test_identity_is_fully_retained_with_no_introduced_voxels() -> None:
    mask = np.zeros((4, 5, 6), dtype=bool)
    mask[0, 2, 3] = True
    mask[2, 3, 4] = True

    result = _metrics(mask, mask)

    assert result["retained_source_count"] == 2
    assert result["retained_source_fraction"] == 1.0
    assert result["lost_source_fraction"] == 0.0
    assert result["introduced_target_count"] == 0
    assert result["introduced_target_fraction"] == 0.0
    for key in (
        "source_to_target_distance_median",
        "source_to_target_distance_p95",
        "target_to_source_distance_median",
        "target_to_source_distance_p95",
    ):
        assert result[key] == 0.0


def test_one_voxel_shift_depends_on_match_radius() -> None:
    source = np.zeros((3, 3, 4), dtype=bool)
    target = np.zeros_like(source)
    source[1, 1, 1] = True
    target[1, 1, 2] = True

    exact = _metrics(source, target, radius=0.0)
    nearby = _metrics(source, target, radius=1.0)

    assert exact["lost_source_count"] == 1
    assert exact["introduced_target_count"] == 1
    assert nearby["retained_source_count"] == 1
    assert nearby["introduced_target_count"] == 0
    assert nearby["source_to_target_distance_median"] == 1.0
    assert nearby["target_to_source_distance_p95"] == 1.0


def test_complete_disappearance() -> None:
    source = np.zeros((3, 3, 3), dtype=bool)
    source[1, 1, 1] = True
    result = _metrics(source, np.zeros_like(source), radius=2.0)

    assert result["retained_source_fraction"] == 0.0
    assert result["lost_source_fraction"] == 1.0
    assert result["introduced_target_fraction"] is None
    assert result["source_to_target_distance_median"] is None
    assert result["target_to_source_distance_p95"] is None


def test_complete_introduction() -> None:
    target = np.zeros((3, 3, 3), dtype=bool)
    target[1, 1, 1] = True
    result = _metrics(np.zeros_like(target), target, radius=2.0)

    assert result["retained_source_fraction"] is None
    assert result["introduced_target_fraction"] == 1.0
    assert result["target_to_source_distance_median"] is None
    assert result["source_to_target_distance_p95"] is None


def test_both_empty_has_only_zero_counts_and_none_metrics() -> None:
    empty = np.zeros((2, 3, 4), dtype=bool)

    result = _metrics(empty, empty)

    assert result["source_count"] == 0
    assert result["target_count"] == 0
    assert result["retained_source_count"] == 0
    assert result["introduced_target_count"] == 0
    assert result["retained_source_fraction"] is None
    assert result["introduced_target_fraction"] is None
    assert result["source_to_target_distance_median"] is None
    assert result["target_to_source_distance_p95"] is None


def test_partial_match_has_both_retained_and_introduced_voxels() -> None:
    source = np.zeros((3, 4, 5), dtype=bool)
    target = np.zeros_like(source)
    source[1, 2, 1:3] = True
    target[1, 2, 2:4] = True

    result = _metrics(source, target)

    assert result["retained_source_count"] == 1
    assert result["retained_source_fraction"] == 0.5
    assert result["introduced_target_count"] == 1
    assert result["introduced_target_fraction"] == 0.5


def test_region_matching_uses_full_opposite_mask_across_shell_boundary() -> None:
    source = np.zeros((5, 5, 5), dtype=bool)
    target = np.zeros_like(source)
    source[0, 2, 2] = True
    target[1, 2, 2] = True

    result = _metrics(source, target, radius=1.0, margin=1)
    regions = result["regions"]
    boundary = regions["boundary_shell"]
    interior = regions["interior"]

    assert boundary["source_count"] == 1
    assert boundary["retained_source_count"] == 1
    assert interior["target_count"] == 1
    assert interior["introduced_target_count"] == 0
    assert boundary["source_to_target_distance_median"] == 1.0
    assert interior["target_to_source_distance_median"] == 1.0


def test_non_cubic_shape_uses_euclidean_voxel_distance() -> None:
    source = np.zeros((2, 4, 7), dtype=bool)
    target = np.zeros_like(source)
    source[0, 1, 1] = True
    target[1, 3, 3] = True

    result = _metrics(source, target, radius=3.0)

    assert result["retained_source_count"] == 1
    assert result["matched_target_count"] == 1
    assert result["source_to_target_distance_median"] == 3.0


@pytest.mark.parametrize(
    ("source", "target", "radius", "margin"),
    [
        (np.zeros((2, 2)), np.zeros((2, 2)), 0.0, 0),
        (np.zeros((2, 2, 2)), np.zeros((2, 2, 3)), 0.0, 0),
        (np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), -1.0, 0),
        (np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.inf, 0),
        (np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.nan, 0),
        (np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), 0.0, -1),
        (np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), 0.0, 1.5),
        (np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), 0.0, True),
    ],
)
def test_invalid_inputs_raise_value_error(
    source: np.ndarray, target: np.ndarray, radius: float, margin: int
) -> None:
    with pytest.raises(ValueError):
        _metrics(source, target, radius=radius, margin=margin)


def test_inputs_are_not_modified() -> None:
    source = np.zeros((3, 3, 3), dtype=np.uint8)
    target = np.zeros((3, 3, 3), dtype=np.float32)
    source[0, 1, 1] = 2
    target[1, 1, 1] = 0.5
    source_before = source.copy()
    target_before = target.copy()

    _metrics(source, target, radius=1.0)

    np.testing.assert_array_equal(source, source_before)
    np.testing.assert_array_equal(target, target_before)
