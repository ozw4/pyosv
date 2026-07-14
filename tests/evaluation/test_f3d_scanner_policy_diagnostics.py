from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pytest

import pyosv.metrics as metrics
from pyosv.evaluation import f3d_scanner_policy_diagnostics as diagnostics
from pyosv.f3d_reference import crop_slices, interior_slices
from pyosv.metrics import sparse_ridge_distance_metrics, top_percentile_mask


def _outputs(
    fvt: np.ndarray,
    *,
    offset: float = 0.0,
) -> dict[str, np.ndarray]:
    shape = fvt.shape
    ramp = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    return {
        "ft_py.dat": ramp + np.float32(offset),
        "pt_py.dat": np.full(shape, 10.0 + offset, dtype=np.float32),
        "tt_py.dat": np.full(shape, 70.0 + offset, dtype=np.float32),
        "fet_py.dat": fvt.copy(),
        "fpt_py.dat": np.full(shape, 20.0 + offset, dtype=np.float32),
        "ftt_py.dat": np.full(shape, 60.0 + offset, dtype=np.float32),
        "fv_py.dat": fvt.copy(),
        "vp_py.dat": np.full(shape, 30.0 + offset, dtype=np.float32),
        "vt_py.dat": np.full(shape, 50.0 + offset, dtype=np.float32),
        "fvt_py.dat": fvt.copy(),
    }


def _ridge(
    shape: tuple[int, int, int],
    coordinates: list[tuple[int, int, int]],
    *,
    value: float = 1.0,
) -> np.ndarray:
    result = np.zeros(shape, dtype=np.float32)
    for coordinate in coordinates:
        result[coordinate] = np.float32(value)
    return result


def _report(
    *,
    reference: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    crop_start: tuple[int, int, int] = (10, 20, 30),
    interior_margin: int = 0,
    allowed_p95_delta: float = 0.0,
    max_points: int = 64,
    max_components: int = 8,
    **optional: np.ndarray,
) -> dict[str, object]:
    shape = reference.shape
    slices = tuple(
        slice(start, start + size) for start, size in zip(crop_start, shape, strict=True)
    )
    return diagnostics.build_public_fvt_distance_outlier_report(
        reference_fvt=reference,
        baseline_outputs=_outputs(baseline),
        candidate_outputs=_outputs(candidate, offset=1.0),
        crop_slices=slices,
        interior_margin=interior_margin,
        allowed_p95_delta=allowed_p95_delta,
        max_points=max_points,
        max_components=max_components,
        **optional,
    )


def test_outlier_distances_nearest_coordinate_and_p95_match_existing_metric() -> None:
    shape = (9, 9, 9)
    public_coordinate = (4, 4, 4)
    reference = _ridge(shape, [public_coordinate])
    baseline = _ridge(shape, [(4, 5, 4)])
    candidate_coordinates = [(1, 4, 4), (4, 1, 4), (4, 4, 6)]
    candidate = _ridge(shape, candidate_coordinates)

    report = _report(
        reference=reference,
        baseline=baseline,
        candidate=candidate,
        interior_margin=1,
        allowed_p95_delta=1.0,
    )

    assert report["status"] == "available"
    summary = report["summary"]
    assert isinstance(summary, Mapping)
    assert summary["baseline_candidate_to_public_p95"] == 1.0
    assert summary["candidate_candidate_to_public_p95"] == 3.0
    assert summary["allowed_candidate_p95"] == 2.0
    assert summary["outlier_count"] == 2

    local_interior = interior_slices(shape, margin=1)
    existing = sparse_ridge_distance_metrics(
        reference[local_interior],
        candidate[local_interior],
        percentile=99.0,
        positive_only=True,
    )
    assert summary["candidate_candidate_to_public_p95"] == existing["candidate_to_reference_p95"]

    points = report["points"]
    assert isinstance(points, list)
    assert [point["distance_to_public_fvt"] for point in points] == [3.0, 3.0]
    # Equal-distance points use interior-local i3/i2/i1 ascending order.
    assert [point["crop_local_coordinate"] for point in points] == [
        [1, 4, 4],
        [4, 1, 4],
    ]
    assert points[0]["nearest_public_fvt"] == {
        "interior_local_coordinate": [3, 3, 3],
        "crop_local_coordinate": [4, 4, 4],
        "global_coordinate": [14, 24, 34],
    }


def test_outlier_uses_positive_only_percentile_and_strict_exceedance() -> None:
    shape = (7, 7, 7)
    reference = _ridge(shape, [(3, 3, 3)])
    baseline = reference.copy()
    candidate = _ridge(shape, [(3, 3, 1), (3, 3, 0)])
    candidate[0, 0, 0] = -1000.0

    report = _report(
        reference=reference,
        baseline=baseline,
        candidate=candidate,
        allowed_p95_delta=2.0,
    )

    expected_mask = top_percentile_mask(candidate, 99.0, positive_only=True)
    assert np.count_nonzero(expected_mask) == 2
    assert not expected_mask[0, 0, 0]
    assert report["summary"]["candidate_sparse_ridge_count"] == 2
    # Distance two equals the allowed limit and is excluded; distance three is included.
    assert report["definition"]["allowed_candidate_p95"] == 2.0
    assert report["summary"]["outlier_count"] == 1
    assert report["points"][0]["crop_local_coordinate"] == [3, 3, 0]


def test_point_coordinate_transforms_crop_face_distance_and_values() -> None:
    shape = (7, 8, 9)
    reference = _ridge(shape, [(2, 2, 2)])
    baseline = reference.copy()
    candidate = _ridge(shape, [(3, 5, 6)])
    xs = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 100.0
    ep = xs + 200.0
    reference_fl = xs + 300.0
    reference_fv = xs + 400.0

    report = _report(
        reference=reference,
        baseline=baseline,
        candidate=candidate,
        crop_start=(100, 200, 300),
        interior_margin=1,
        xs=xs,
        ep=ep,
        reference_fl=reference_fl,
        reference_fv=reference_fv,
    )

    point = report["points"][0]
    assert point["interior_local_coordinate"] == [2, 4, 5]
    assert point["crop_local_coordinate"] == [3, 5, 6]
    assert point["global_coordinate"] == [103, 205, 306]
    assert point["nearest_public_fvt"]["interior_local_coordinate"] == [1, 1, 1]
    assert point["nearest_public_fvt"]["crop_local_coordinate"] == [2, 2, 2]
    assert point["nearest_public_fvt"]["global_coordinate"] == [102, 202, 302]
    assert point["crop_face_distance"] == {
        "minimum": 2,
        "per_axis_nearest_face": [3, 2, 2],
    }
    coordinate = (3, 5, 6)
    assert point["values"]["xs"] == float(xs[coordinate])
    assert point["values"]["ep"] == float(ep[coordinate])
    assert point["values"]["reference"] == {
        "fl": float(reference_fl[coordinate]),
        "fv": float(reference_fv[coordinate]),
        "fvt": 0.0,
    }
    assert set(point["values"]["shared"]) == {"ft", "pt", "tt"}
    assert "ft" not in point["values"]["baseline"]
    assert set(point["values"]["candidate"]) == {
        "fet",
        "fpt",
        "ftt",
        "fv",
        "vp",
        "vt",
        "fvt",
    }


def test_26_neighbor_components_sort_geometry_representative_and_truncation() -> None:
    shape = (10, 10, 10)
    reference = _ridge(shape, [(1, 1, 1)])
    baseline = reference.copy()
    # Three diagonal voxels are one 26-neighbor component.  The second component
    # has a larger maximum distance but fewer voxels, so voxel count sorts first.
    component_a = [(4, 4, 4), (5, 5, 5), (6, 6, 6)]
    component_b = [(1, 8, 8), (1, 8, 9)]
    candidate = _ridge(shape, component_a + component_b)

    report = _report(
        reference=reference,
        baseline=baseline,
        candidate=candidate,
        max_points=2,
        max_components=1,
    )

    summary = report["summary"]
    assert summary["outlier_count"] == 5
    assert summary["component_count"] == 2
    assert summary["stored_point_count"] == 2
    assert summary["stored_component_count"] == 1
    assert summary["points_truncated"] is True
    assert summary["components_truncated"] is True

    component = report["components"][0]
    assert component["component_id"] == 1
    assert component["voxel_count"] == 3
    assert component["crop_local_bounding_box"] == {
        "minimum": [4, 4, 4],
        "maximum": [6, 6, 6],
    }
    assert component["global_bounding_box"] == {
        "minimum": [14, 24, 34],
        "maximum": [16, 26, 36],
    }
    assert component["crop_local_centroid"] == [5.0, 5.0, 5.0]
    assert component["global_centroid"] == [15.0, 25.0, 35.0]
    assert component["representative_point"]["crop_local_coordinate"] == [6, 6, 6]
    assert component["truncated"] is True


def test_component_sort_uses_max_distance_then_minimum_global_coordinate() -> None:
    shape = (11, 11, 11)
    reference = _ridge(shape, [(5, 5, 5)])
    baseline = reference.copy()
    # All components contain one voxel. The distance-five component sorts first;
    # the two distance-four components then sort by minimum global coordinate.
    candidate = _ridge(shape, [(1, 5, 5), (5, 5, 0), (9, 5, 5)])

    report = _report(reference=reference, baseline=baseline, candidate=candidate)

    representatives = [
        component["representative_point"]["crop_local_coordinate"]
        for component in report["components"]
    ]
    assert representatives == [[5, 5, 0], [1, 5, 5], [9, 5, 5]]
    assert [component["component_id"] for component in report["components"]] == [1, 2, 3]


@pytest.mark.parametrize("empty_role", ["public", "baseline", "candidate"])
def test_empty_sparse_masks_are_unavailable_not_zero(empty_role: str) -> None:
    shape = (5, 5, 5)
    reference = _ridge(shape, [(2, 2, 2)])
    baseline = reference.copy()
    candidate = _ridge(shape, [(2, 2, 3)])
    if empty_role == "public":
        reference.fill(0.0)
    elif empty_role == "baseline":
        baseline.fill(0.0)
    else:
        candidate.fill(0.0)

    report = _report(reference=reference, baseline=baseline, candidate=candidate)

    assert report["status"] == "unavailable"
    assert empty_role in report["reason"]
    assert report["summary"]["outlier_count"] is None
    assert report["summary"]["baseline_candidate_to_public_p95"] is None
    assert report["points"] == []
    assert report["components"] == []


def test_zero_outliers_is_available_with_empty_details() -> None:
    shape = (5, 5, 5)
    ridge = _ridge(shape, [(2, 2, 2)])

    report = _report(reference=ridge, baseline=ridge, candidate=ridge)

    assert report["status"] == "available"
    assert report["summary"]["outlier_count"] == 0
    assert report["summary"]["component_count"] == 0
    assert report["points"] == []
    assert report["components"] == []


def test_shape_and_nonfinite_required_values_are_unavailable_and_controls_reject() -> None:
    shape = (5, 5, 5)
    ridge = _ridge(shape, [(2, 2, 2)])
    mismatched = _ridge((4, 5, 5), [(2, 2, 2)])
    slices = (slice(0, 5), slice(0, 5), slice(0, 5))

    mismatch_report = diagnostics.build_public_fvt_distance_outlier_report(
        reference_fvt=ridge,
        baseline_outputs=_outputs(ridge),
        candidate_outputs=_outputs(mismatched),
        crop_slices=slices,
        interior_margin=0,
    )
    assert mismatch_report["status"] == "unavailable"
    assert "shapes must match" in mismatch_report["reason"]

    nonfinite = ridge.copy()
    nonfinite[0, 0, 0] = np.nan
    nonfinite_report = _report(
        reference=ridge,
        baseline=ridge,
        candidate=nonfinite,
    )
    assert nonfinite_report["status"] == "unavailable"
    assert "finite" in nonfinite_report["reason"]

    for kwargs, match in (
        ({"max_points": 0}, "max_points"),
        ({"max_components": 0}, "max_components"),
        ({"ridge_percentile": np.inf}, "ridge_percentile"),
        ({"allowed_p95_delta": np.nan}, "allowed_p95_delta"),
    ):
        with pytest.raises(ValueError, match=match):
            diagnostics.build_public_fvt_distance_outlier_report(
                reference_fvt=ridge,
                baseline_outputs=_outputs(ridge),
                candidate_outputs=_outputs(ridge),
                crop_slices=slices,
                interior_margin=0,
                **kwargs,
            )


def test_outlier_report_is_strict_json_serializable_and_sanitizes_optional_scalar() -> None:
    shape = (5, 5, 5)
    reference = _ridge(shape, [(1, 1, 1)])
    baseline = reference.copy()
    candidate = _ridge(shape, [(3, 3, 3)])
    xs = np.zeros(shape, dtype=np.float32)
    xs[3, 3, 3] = np.inf

    report = _report(
        reference=reference,
        baseline=baseline,
        candidate=candidate,
        xs=xs,
    )

    assert report["points"][0]["values"]["xs"] is None
    serialized = json.dumps(report, allow_nan=False)
    assert json.loads(serialized)["status"] == "available"


def test_base_roi_mapping_and_extraction_use_global_bounds_near_volume_edge() -> None:
    full_shape = (10, 11, 12)
    center = (8, 9, 10)
    base_global = crop_slices(center, (4, 4, 4), full_shape=full_shape)
    context_global = crop_slices(center, (8, 8, 8), full_shape=full_shape)
    assert diagnostics.slices_to_json(base_global) == [
        {"axis": "i3", "start": 6, "stop": 10},
        {"axis": "i2", "start": 7, "stop": 11},
        {"axis": "i1", "start": 8, "stop": 12},
    ]
    assert diagnostics.map_base_roi_slices_within_context(base_global, context_global) == (
        slice(4, 8),
        slice(4, 8),
        slice(4, 8),
    )

    full = np.arange(np.prod(full_shape), dtype=np.float32).reshape(full_shape)
    context = full[context_global]
    extracted = diagnostics.extract_same_global_roi(
        context,
        base_global_slices=base_global,
        context_global_slices=context_global,
    )
    np.testing.assert_array_equal(extracted, full[base_global])

    outputs = {"ft_py.dat": context, "fvt_py.dat": context + 1.0}
    extracted_outputs = diagnostics.extract_same_global_roi_outputs(
        outputs,
        base_global_slices=base_global,
        context_global_slices=context_global,
    )
    np.testing.assert_array_equal(extracted_outputs["ft_py.dat"], full[base_global])
    np.testing.assert_array_equal(extracted_outputs["fvt_py.dat"], full[base_global] + 1.0)


def test_base_roi_mapping_rejects_context_that_does_not_contain_it() -> None:
    with pytest.raises(ValueError, match="not fully contained"):
        diagnostics.map_base_roi_slices_within_context(
            (slice(3, 7), slice(3, 7), slice(3, 7)),
            (slice(0, 6), slice(0, 8), slice(0, 8)),
        )


def test_identical_same_global_roi_stage_comparison_is_exact() -> None:
    shape = (5, 5, 5)
    fvt = np.zeros(shape, dtype=np.float32)
    fvt[:, 2, :] = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    outputs = _outputs(fvt)

    comparison = diagnostics.build_same_global_roi_stage_comparison(
        base_outputs=outputs,
        context_roi_outputs={name: value.copy() for name, value in outputs.items()},
    )

    assert comparison["status"] == "available"
    for stage in comparison["stages"].values():
        assert stage["shape_equal"] is True
        assert stage["finite"]["both"] is True
        assert stage["density_delta"] == 0.0
        assert stage["normalized_correlation"] == pytest.approx(1.0)
        assert stage["absolute_difference"] == {
            "mean": 0.0,
            "p95": 0.0,
            "maximum": 0.0,
        }
    ridge = comparison["stages"]["fvt"]["ridge_comparison"]
    assert ridge["buffered_ridge_overlap"]["buffered_f1"] == 1.0
    assert ridge["sparse_ridge_distance_metrics"]["candidate_to_reference_p95"] == 0.0
    assert ridge["ridge_mask_difference"]["base_only_ridge_count"] == 0
    assert ridge["ridge_mask_difference"]["context_only_ridge_count"] == 0


def test_same_global_roi_fvt_comparison_measures_one_voxel_shift() -> None:
    shape = (7, 7, 7)
    base_fvt = np.zeros(shape, dtype=np.float32)
    context_fvt = np.zeros(shape, dtype=np.float32)
    base_fvt[:, 3, :] = 1.0
    context_fvt[:, 4, :] = 1.0

    comparison = diagnostics.build_same_global_roi_stage_comparison(
        base_outputs=_outputs(base_fvt),
        context_roi_outputs=_outputs(context_fvt),
    )

    distance = comparison["stages"]["fvt"]["ridge_comparison"]["sparse_ridge_distance_metrics"]
    assert distance["candidate_to_reference_p95"] == 1.0
    assert distance["reference_to_candidate_p95"] == 1.0


def test_context_sparse_percentile_is_computed_on_extracted_base_roi() -> None:
    context_shape = (6, 6, 6)
    base_global = (slice(1, 5), slice(1, 5), slice(1, 5))
    context_global = (slice(0, 6), slice(0, 6), slice(0, 6))
    full_context_fvt = np.zeros(context_shape, dtype=np.float32)
    full_context_fvt[1, 1, 1] = 1.0
    full_context_fvt[4, 4, 4] = 2.0
    # This much larger value is outside the same-global base ROI.
    full_context_fvt[0, 0, 0] = 1000.0
    context_outputs = _outputs(full_context_fvt)
    context_roi_outputs = diagnostics.extract_same_global_roi_outputs(
        context_outputs,
        base_global_slices=base_global,
        context_global_slices=context_global,
    )
    base_fvt = context_roi_outputs["fvt_py.dat"].copy()

    comparison = diagnostics.build_same_global_roi_stage_comparison(
        base_outputs=_outputs(base_fvt),
        context_roi_outputs=context_roi_outputs,
    )

    sparse_counts = comparison["stages"]["fvt"]["ridge_comparison"]["ridge_mask_difference"]
    assert sparse_counts["base_sparse_ridge_count"] == 1
    assert sparse_counts["context_roi_sparse_ridge_count"] == 1


def test_context_outlier_persistence_maps_global_point_and_one_voxel_shift() -> None:
    shape = (7, 7, 7)
    public = _ridge(shape, [(3, 3, 1)])
    baseline = public.copy()
    base_candidate = _ridge(shape, [(3, 3, 4)])
    context_candidate = _ridge(shape, [(3, 3, 5)])
    base_global = (slice(10, 17), slice(20, 27), slice(30, 37))
    base_report = diagnostics.build_public_fvt_distance_outlier_report(
        reference_fvt=public,
        baseline_outputs=_outputs(baseline),
        candidate_outputs=_outputs(base_candidate, offset=1.0),
        crop_slices=base_global,
        interior_margin=0,
        allowed_p95_delta=0.0,
    )

    persistence = diagnostics.build_context_outlier_persistence_report(
        base_outlier_report=base_report,
        reference_fvt=public,
        base_baseline_outputs=_outputs(baseline),
        base_candidate_outputs=_outputs(base_candidate, offset=1.0),
        context_baseline_outputs=_outputs(baseline),
        context_candidate_outputs=_outputs(context_candidate, offset=2.0),
        base_global_slices=base_global,
        allowed_p95_delta=0.0,
    )

    assert persistence["status"] == "available"
    assert persistence["summary"]["base_outlier_count"] == 1
    assert persistence["summary"]["stored_base_outlier_count"] == 1
    assert persistence["summary"]["context_outlier_count"] == 1
    assert persistence["summary"]["base_outlier_points_with_context_candidate_within_radius"] == 1
    assert (
        persistence["summary"]["base_outlier_points_with_context_candidate_within_2_samples"] == 1
    )
    assert persistence["summary"]["persistence_fraction"] == 1.0
    point = persistence["points"][0]
    assert point["global_coordinate"] == [13, 23, 34]
    assert point["base_roi_local_coordinate"] == [3, 3, 4]
    assert point["base_candidate_sparse_mask_membership"] is True
    assert point["context_candidate_sparse_mask_membership"] is False
    assert point["nearest_context_candidate_sparse_ridge_distance"] == 1.0
    assert point["persists_within_radius"] is True
    assert point["persists_within_2_samples"] is True
    assert point["base_candidate_to_public_distance"] == 3.0
    # This is evaluated at the nearest context-candidate ridge coordinate
    # (i1=5), not at the original base outlier coordinate (i1=4).
    assert point["context_candidate_to_public_distance"] == 4.0
    assert point["stage_values"]["fet"] == {"base": 1.0, "context": 0.0}
    overlap = persistence["summary"]["base_context_outlier_component_overlap"]
    assert overlap["base_component_count"] == 1
    assert overlap["context_component_count"] == 1
    assert overlap["exact_overlap_voxel_count"] == 0
    json.dumps(persistence, allow_nan=False)


def test_persistence_sparse_percentile_ignores_values_outside_base_interior() -> None:
    shape = (7, 7, 7)
    public = _ridge(shape, [(3, 3, 2)])
    baseline = public.copy()
    base_candidate = _ridge(shape, [(3, 3, 4)])
    context_candidate = base_candidate.copy()
    # A full-crop percentile would select these dominant boundary samples and
    # drop the actual interior outlier. The outlier contract excludes them.
    base_candidate[0, 0, 0] = 1000.0
    context_candidate[0, 0, 0] = 2000.0
    base_global = (slice(10, 17), slice(20, 27), slice(30, 37))
    base_report = diagnostics.build_public_fvt_distance_outlier_report(
        reference_fvt=public,
        baseline_outputs=_outputs(baseline),
        candidate_outputs=_outputs(base_candidate),
        crop_slices=base_global,
        interior_margin=1,
        allowed_p95_delta=0.0,
    )
    assert base_report["summary"]["outlier_count"] == 1

    persistence = diagnostics.build_context_outlier_persistence_report(
        base_outlier_report=base_report,
        reference_fvt=public,
        base_baseline_outputs=_outputs(baseline),
        base_candidate_outputs=_outputs(base_candidate),
        context_baseline_outputs=_outputs(baseline),
        context_candidate_outputs=_outputs(context_candidate),
        base_global_slices=base_global,
        allowed_p95_delta=0.0,
    )

    point = persistence["points"][0]
    assert point["base_roi_local_coordinate"] == [3, 3, 4]
    assert point["interior_local_coordinate"] == [2, 2, 3]
    assert point["base_candidate_sparse_mask_membership"] is True
    assert point["context_candidate_sparse_mask_membership"] is True
    assert point["nearest_context_candidate_sparse_ridge_distance"] == 0.0
    assert point["persists_within_radius"] is True


def test_persistence_summary_uses_all_outliers_when_point_details_are_truncated() -> None:
    shape = (7, 7, 7)
    public = _ridge(shape, [(3, 3, 1)])
    baseline = public.copy()
    candidate_coordinates = [(1, 1, 5), (3, 3, 4), (5, 5, 5)]
    candidate = _ridge(shape, candidate_coordinates)
    base_global = (slice(10, 17), slice(20, 27), slice(30, 37))
    base_report = diagnostics.build_public_fvt_distance_outlier_report(
        reference_fvt=public,
        baseline_outputs=_outputs(baseline),
        candidate_outputs=_outputs(candidate),
        crop_slices=base_global,
        interior_margin=0,
        allowed_p95_delta=0.0,
        max_points=1,
    )
    assert base_report["summary"]["outlier_count"] == 3
    assert base_report["summary"]["stored_point_count"] == 1

    persistence = diagnostics.build_context_outlier_persistence_report(
        base_outlier_report=base_report,
        reference_fvt=public,
        base_baseline_outputs=_outputs(baseline),
        base_candidate_outputs=_outputs(candidate),
        context_baseline_outputs=_outputs(baseline),
        context_candidate_outputs=_outputs(candidate),
        base_global_slices=base_global,
        allowed_p95_delta=0.0,
    )

    assert len(persistence["points"]) == 1
    assert persistence["summary"]["base_outlier_count"] == 3
    assert persistence["summary"]["stored_base_outlier_count"] == 1
    assert persistence["summary"]["context_outlier_count"] == 3
    assert persistence["summary"]["base_outlier_points_with_context_candidate_within_radius"] == 3
    assert (
        persistence["summary"]["base_outlier_points_with_context_candidate_within_2_samples"] == 3
    )
    assert persistence["summary"]["persistence_fraction"] == 1.0


def test_context_candidate_to_public_distance_is_none_for_empty_context_ridge() -> None:
    shape = (7, 7, 7)
    public = _ridge(shape, [(3, 3, 1)])
    baseline = public.copy()
    candidate = _ridge(shape, [(3, 3, 4)])
    empty_context = np.zeros(shape, dtype=np.float32)
    base_global = (slice(10, 17), slice(20, 27), slice(30, 37))
    base_report = diagnostics.build_public_fvt_distance_outlier_report(
        reference_fvt=public,
        baseline_outputs=_outputs(baseline),
        candidate_outputs=_outputs(candidate),
        crop_slices=base_global,
        interior_margin=0,
        allowed_p95_delta=0.0,
    )

    persistence = diagnostics.build_context_outlier_persistence_report(
        base_outlier_report=base_report,
        reference_fvt=public,
        base_baseline_outputs=_outputs(baseline),
        base_candidate_outputs=_outputs(candidate),
        context_baseline_outputs=_outputs(baseline),
        context_candidate_outputs=_outputs(empty_context),
        base_global_slices=base_global,
        allowed_p95_delta=0.0,
    )

    point = persistence["points"][0]
    assert point["nearest_context_candidate_sparse_ridge_distance"] is None
    assert point["context_candidate_to_public_distance"] is None
    assert point["persists_within_2_samples"] is False
    assert persistence["summary"]["base_outlier_count"] == 1
    assert persistence["summary"]["stored_base_outlier_count"] == 1
    assert (
        persistence["summary"]["base_outlier_points_with_context_candidate_within_2_samples"] == 0
    )
    assert persistence["summary"]["persistence_fraction"] == 0.0


def test_shared_sparse_distance_field_returns_mask_distance_and_nearest_indices() -> None:
    values = np.zeros((5, 5, 5), dtype=np.float32)
    values[1, 2, 3] = 1.0
    original = values.copy()

    field = metrics._sparse_ridge_distance_field(
        values,
        percentile=99.0,
        positive_only=True,
        return_indices=True,
    )

    np.testing.assert_array_equal(values, original)
    assert field.count == 1
    assert field.mask[1, 2, 3]
    assert field.distance is not None
    assert field.nearest_indices is not None
    assert field.distance[4, 2, 3] == 3.0
    assert [field.nearest_indices[axis, 4, 2, 3] for axis in range(3)] == [1, 2, 3]

    empty = metrics._sparse_ridge_distance_field(np.zeros_like(values))
    assert empty.count == 0
    assert empty.distance is None
    assert empty.nearest_indices is None
