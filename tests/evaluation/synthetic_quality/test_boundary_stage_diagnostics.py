from __future__ import annotations

import numpy as np
import pytest

from pyosv.evaluation.synthetic_quality.boundary_stage_diagnostics import (
    build_scanner_boundary_stage_diagnostics,
    stage_mask_profile,
    stage_transition_correspondence_metrics,
    transition_centroid_shift_metrics,
    volume_edge_distance_map,
)
from pyosv.evaluation.synthetic_quality.config import SyntheticTruthMetricConfig
from pyosv.evaluation.synthetic_quality.models import PipelineStageTrace3D
from pyosv.synthetic3d import make_single_vertical_plane_case


def test_scanner_builder_has_stable_stage_transition_and_volume_schema() -> None:
    case = make_single_vertical_plane_case((5, 5, 5))
    empty = np.zeros(case.shape, dtype=bool)
    trace = PipelineStageTrace3D(
        seed_candidate_mask=empty,
        seed_selected_mask=empty,
        fv_positive_mask=empty,
        fvt_positive_mask=empty,
        primary_skin_mask=empty,
        fallback_skin_mask=empty,
        final_skin_mask=empty,
        skinning_enabled=False,
        fallback_used=False,
    )
    scanner_ft = np.zeros(case.shape, dtype=np.float32)
    scanner_ft[2, 2, 2] = 1.0

    report, volumes = build_scanner_boundary_stage_diagnostics(
        case=case,
        scanner_volumes={"scanner_ft": scanner_ft, "scanner_fet": scanner_ft},
        stage_trace=trace,
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_diagnostics=None,
    )

    assert report["stage_order"][0:2] == ["scanner_ft_positive", "scanner_fet_positive"]
    assert report["stage_order"][-3:] == ["primary_skin", "fallback_skin", "final_skin"]
    assert report["transition_order"][-2:] == [
        "fvt_positive_to_fallback_skin",
        "primary_skin_to_fallback_skin",
    ]
    assert list(report["stages"]) == report["stage_order"]
    assert list(report["transitions"]) == report["transition_order"]
    assert len(volumes) == 10
    for volume in volumes.values():
        assert volume.shape == case.shape
        assert volume.dtype == np.float32
        assert set(np.unique(volume)).issubset({0.0, 1.0})


@pytest.mark.parametrize("fallback_used", (False, True))
def test_scanner_builder_reports_primary_fallback_and_final_masks(
    fallback_used: bool,
) -> None:
    case = make_single_vertical_plane_case((5, 5, 5))
    empty = np.zeros(case.shape, dtype=bool)
    primary = empty.copy()
    primary[2, 2, 2] = True
    fallback = empty.copy()
    if fallback_used:
        fallback[0, 0, 0] = True
    final = fallback if fallback_used else primary
    trace = PipelineStageTrace3D(
        seed_candidate_mask=empty,
        seed_selected_mask=empty,
        fv_positive_mask=empty,
        fvt_positive_mask=empty,
        primary_skin_mask=primary,
        fallback_skin_mask=fallback,
        final_skin_mask=final,
        skinning_enabled=True,
        fallback_used=fallback_used,
    )

    report, volumes = build_scanner_boundary_stage_diagnostics(
        case=case,
        scanner_volumes={"scanner_ft": empty, "scanner_fet": empty},
        stage_trace=trace,
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_diagnostics={
            "fallback_enabled": True,
            "fallback_reason": "test" if fallback_used else None,
        },
    )

    assert report["skinning"]["fallback_used"] is fallback_used
    assert report["stages"]["primary_skin"]["candidate_count"] == 1
    assert report["stages"]["fallback_skin"]["candidate_count"] == int(fallback_used)
    assert report["stages"]["final_skin"]["candidate_count"] == 1
    np.testing.assert_array_equal(volumes["scanner_boundary_stage_primary_skin"], primary)
    np.testing.assert_array_equal(volumes["scanner_boundary_stage_fallback_skin"], fallback)
    np.testing.assert_array_equal(volumes["scanner_boundary_stage_final_skin"], final)


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


def test_volume_edge_distance_map_handles_non_cubic_and_unit_axes() -> None:
    distances = volume_edge_distance_map((5, 6, 7))
    assert distances[0, 3, 3] == 0
    assert distances[1, 3, 3] == 1
    assert distances[2, 3, 3] == 2
    np.testing.assert_array_equal(volume_edge_distance_map((1, 3, 4)), 0)


def test_stage_profile_identity_and_empty_cases() -> None:
    truth = np.zeros((5, 5, 5), dtype=bool)
    truth[2, 2, 2] = True
    identity = stage_mask_profile(
        truth,
        truth_fault_mask=truth,
        truth_surface_mask=truth,
        match_radius=1.0,
        edge_margin=1,
    )["truth"]
    assert identity["truth_recall"] == 1.0
    assert identity["candidate_precision"] == 1.0
    assert identity["buffered_f1"] == 1.0
    assert identity["candidate_to_truth_distance_p95"] == 0.0

    empty_candidate = stage_mask_profile(
        np.zeros_like(truth),
        truth_fault_mask=truth,
        truth_surface_mask=truth,
        match_radius=1.0,
        edge_margin=1,
    )["truth"]
    assert empty_candidate["truth_recall"] == 0.0
    assert empty_candidate["candidate_precision"] is None
    assert empty_candidate["truth_to_candidate_distance_median"] is None

    empty_truth = stage_mask_profile(
        truth,
        truth_fault_mask=np.zeros_like(truth),
        truth_surface_mask=np.zeros_like(truth),
        match_radius=1.0,
        edge_margin=1,
    )["truth"]
    assert empty_truth["truth_recall"] is None
    assert empty_truth["candidate_precision"] == 0.0
    assert empty_truth["candidate_to_truth_distance_median"] is None


def test_stage_profile_regions_and_edge_bins_match_against_full_masks() -> None:
    truth = np.zeros((9, 9, 9), dtype=bool)
    candidate = np.zeros_like(truth)
    truth[0, 4, 4] = True
    truth[4, 4, 4] = True
    candidate[1, 4, 4] = True
    candidate[4, 4, 4] = True
    result = stage_mask_profile(
        candidate,
        truth_fault_mask=truth,
        truth_surface_mask=truth,
        match_radius=1.0,
        edge_margin=1,
        max_exact_edge_distance=3,
    )
    assert result["regions"]["boundary_shell"]["truth_count"] == 1
    assert result["regions"]["boundary_shell"]["truth_recall"] == 1.0
    assert result["regions"]["interior"]["candidate_count"] == 2
    assert list(result["edge_distance_profile"]) == ["0", "1", "2", "3", "4_plus"]
    assert result["edge_distance_profile"]["0"]["truth_recall"] == 1.0
    assert result["edge_distance_profile"]["1"]["candidate_precision"] == 1.0
    assert result["edge_distance_profile"]["4_plus"]["truth_count"] == 1


def test_stage_profile_components_use_18_neighborhood() -> None:
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    mask[0, 1, 1] = True  # edge-connected
    mask[1, 2, 1] = True  # edge-connected
    mask[3, 3, 3] = True  # corner-only from the preceding voxel
    result = stage_mask_profile(
        mask,
        truth_fault_mask=mask,
        truth_surface_mask=mask,
        match_radius=0.0,
        edge_margin=1,
    )["components"]
    assert result == {
        "connectivity": "edge",
        "component_count": 2,
        "largest_component_size": 3,
        "largest_component_fraction": 0.75,
    }


def test_centroid_shift_reorders_coordinates_and_projects_onto_normal() -> None:
    source = np.zeros((4, 4, 5), dtype=bool)
    target = np.zeros_like(source)
    source[1, 2, 1] = True
    target[1, 2, 2] = True
    strike = np.zeros(source.shape, dtype=np.float32)
    dip = np.zeros_like(strike)  # normal=(-1, 0, 0)
    result = transition_centroid_shift_metrics(
        source,
        target,
        truth_strike=strike,
        truth_dip=dip,
        truth_reference_mask=source,
    )
    assert result["source_centroid_x1_x2_x3"] == [1.0, 2.0, 1.0]
    assert result["shift_x1_x2_x3"] == [1.0, 0.0, 0.0]
    assert result["shift_magnitude"] == 1.0
    assert result["normal_shift"] == -1.0
    assert result["tangential_shift_magnitude"] == 0.0


def test_centroid_shift_handles_perpendicular_and_missing_inputs() -> None:
    source = np.zeros((3, 3, 3), dtype=bool)
    target = np.zeros_like(source)
    source[1, 1, 1] = True
    target[1, 2, 1] = True
    angles = np.zeros(source.shape)
    result = transition_centroid_shift_metrics(
        source,
        target,
        truth_strike=angles,
        truth_dip=angles,
        truth_reference_mask=source,
    )
    assert result["normal_shift"] == pytest.approx(0.0)
    assert result["tangential_shift_magnitude"] == pytest.approx(1.0)

    missing = transition_centroid_shift_metrics(
        np.zeros_like(source),
        target,
        truth_strike=angles,
        truth_dip=angles,
        truth_reference_mask=np.zeros_like(source),
    )
    assert missing["shift_x1_x2_x3"] is None
    assert missing["representative_truth_normal_x1_x2_x3"] is None
    assert missing["normal_shift"] is None


@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_centroid_shift_rejects_non_finite_truth_angles(bad_value: float) -> None:
    mask = np.zeros((2, 2, 2), dtype=bool)
    angles = np.zeros(mask.shape)
    angles[0, 0, 0] = bad_value
    with pytest.raises(ValueError):
        transition_centroid_shift_metrics(
            mask,
            mask,
            truth_strike=angles,
            truth_dip=np.zeros_like(angles),
            truth_reference_mask=mask,
        )
