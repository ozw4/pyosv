"""Pure diagnostics for the F3 scanner-thinning policy comparison.

The functions in this module deliberately avoid plotting, file I/O, and F3
dataset discovery.  They reconstruct the existing sparse public-FVT distance
metric, describe its tail exceedances, and compare a base crop with the exact
same global ROI extracted from a larger context crop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import (
    generate_binary_structure,
    label,
)

from pyosv.f3d_reference import interior_slices
from pyosv.metrics import (
    _sparse_ridge_distance_field,
    _sparse_ridge_distance_summary,
    buffered_ridge_overlap,
    normalized_correlation,
    sparse_ridge_distance_metrics,
    top_percentile_mask,
)


DEFAULT_RIDGE_PERCENTILE = 99.0
DEFAULT_ALLOWED_P95_DELTA = 5.0
DEFAULT_MAX_POINTS = 64
DEFAULT_MAX_COMPONENTS = 8
DEFAULT_NONZERO_EPSILON = 1.0e-6
DEFAULT_RIDGE_BUFFER_RADIUS = 2.0
DEFAULT_PERSISTENCE_RADIUS = 2.0

_AXIS_NAMES = ("i3", "i2", "i1")
_BRANCH_VALUE_NAMES = ("fet", "fpt", "ftt", "fv", "vp", "vt", "fvt")
_SHARED_VALUE_NAMES = ("ft", "pt", "tt")
_COMPARISON_STAGES = (
    ("ft", "shared_raw_ft"),
    ("fet", "scanner_thinned_fet"),
    ("fv", "voted_fv"),
    ("fvt", "thinned_fvt"),
)


def build_public_fvt_distance_outlier_report(
    *,
    reference_fvt: np.ndarray,
    baseline_outputs: Mapping[str, np.ndarray],
    candidate_outputs: Mapping[str, np.ndarray],
    crop_slices: tuple[slice, slice, slice],
    interior_margin: int,
    ridge_percentile: float = DEFAULT_RIDGE_PERCENTILE,
    allowed_p95_delta: float = DEFAULT_ALLOWED_P95_DELTA,
    max_points: int = DEFAULT_MAX_POINTS,
    max_components: int = DEFAULT_MAX_COMPONENTS,
    xs: np.ndarray | None = None,
    ep: np.ndarray | None = None,
    reference_fl: np.ndarray | None = None,
    reference_fv: np.ndarray | None = None,
) -> dict[str, Any]:
    """Describe candidate ridges beyond the allowed public-FVT distance tail.

    The three sparse masks use exactly :func:`top_percentile_mask` with
    ``positive_only=True``.  Baseline and candidate p95 values are sampled from
    one distance transform of the public mask, matching
    :func:`sparse_ridge_distance_metrics` for candidate-to-reference distance.

    Invalid diagnostic controls raise ``ValueError``.  Data that cannot support
    the metric (shape mismatch, non-finite required values, or an empty sparse
    mask) yields a strict-JSON-safe ``status="unavailable"`` report instead of
    a misleading zero distance.
    """

    percentile = _validate_percentile(ridge_percentile)
    allowed_delta = _validate_finite_float(allowed_p95_delta, "allowed_p95_delta")
    point_limit = _validate_positive_int(max_points, "max_points")
    component_limit = _validate_positive_int(max_components, "max_components")
    if not isinstance(interior_margin, (int, np.integer)) or isinstance(
        interior_margin, (bool, np.bool_)
    ):
        raise ValueError("interior_margin must be a non-negative integer")
    margin = int(interior_margin)
    if margin < 0:
        raise ValueError("interior_margin must be a non-negative integer")

    definition = {
        "ridge_percentile": percentile,
        "positive_only": True,
        "interior_margin": margin,
        "allowed_p95_delta": allowed_delta,
        "allowed_candidate_p95": None,
        "distance_units": "samples",
        "outlier_operator": "candidate_distance_to_public_fvt > allowed_candidate_p95",
        "component_connectivity": 26,
    }

    try:
        reference = _required_array(reference_fvt, "reference_fvt")
        baseline_fvt = _required_output_array(baseline_outputs, "fvt")
        candidate_fvt = _required_output_array(candidate_outputs, "fvt")
    except ValueError as error:
        return _unavailable_outlier_report(definition, str(error))

    shape = reference.shape
    if baseline_fvt.shape != shape or candidate_fvt.shape != shape:
        return _unavailable_outlier_report(
            definition,
            "reference, baseline, and candidate fvt shapes must match",
        )
    try:
        starts, _, crop_shape = _slice_geometry(crop_slices, name="crop_slices")
    except ValueError as error:
        return _unavailable_outlier_report(definition, str(error))
    if crop_shape != shape:
        return _unavailable_outlier_report(
            definition,
            f"crop_slices shape {crop_shape} does not match fvt shape {shape}",
        )
    if not np.all(np.isfinite(reference)):
        return _unavailable_outlier_report(
            definition, "reference_fvt must contain only finite values"
        )
    if not np.all(np.isfinite(baseline_fvt)):
        return _unavailable_outlier_report(
            definition, "baseline fvt must contain only finite values"
        )
    if not np.all(np.isfinite(candidate_fvt)):
        return _unavailable_outlier_report(
            definition, "candidate fvt must contain only finite values"
        )

    optional_arrays = {
        "xs": xs,
        "ep": ep,
        "reference_fl": reference_fl,
        "reference_fv": reference_fv,
    }
    for name, values in optional_arrays.items():
        if values is not None and np.asarray(values).shape != shape:
            return _unavailable_outlier_report(
                definition,
                f"{name} shape {np.asarray(values).shape} does not match fvt shape {shape}",
            )
    for role, outputs in (("baseline", baseline_outputs), ("candidate", candidate_outputs)):
        mismatch = _first_present_output_shape_mismatch(outputs, shape)
        if mismatch is not None:
            return _unavailable_outlier_report(
                definition,
                f"{role} output {mismatch!r} does not match fvt shape {shape}",
            )

    try:
        local_interior = interior_slices(shape, margin=margin)
    except ValueError as error:
        return _unavailable_outlier_report(definition, str(error))
    interior_starts = tuple(int(value.start) for value in local_interior)
    reference_interior = reference[local_interior]
    baseline_interior = baseline_fvt[local_interior]
    candidate_interior = candidate_fvt[local_interior]

    public_field = _sparse_ridge_distance_field(
        reference_interior,
        percentile=percentile,
        positive_only=True,
        return_indices=True,
    )
    baseline_field = _sparse_ridge_distance_field(
        baseline_interior,
        percentile=percentile,
        positive_only=True,
    )
    candidate_field = _sparse_ridge_distance_field(
        candidate_interior,
        percentile=percentile,
        positive_only=True,
    )
    mask_counts = {
        "public_fvt": public_field.count,
        "baseline_fvt": baseline_field.count,
        "candidate_fvt": candidate_field.count,
    }
    empty = [name for name, count in mask_counts.items() if count == 0]
    if empty:
        return _unavailable_outlier_report(
            definition,
            "sparse mask is empty: " + ", ".join(empty),
            sparse_mask_counts=mask_counts,
        )

    # Both branches sample the exact shared field used by
    # sparse_ridge_distance_metrics. Nearest indices are retained only for the
    # deterministic point diagnostics.
    assert public_field.distance is not None
    assert public_field.nearest_indices is not None
    distance_to_public = public_field.distance
    nearest_indices = public_field.nearest_indices
    baseline_mask = baseline_field.mask
    candidate_mask = candidate_field.mask
    baseline_distances = distance_to_public[baseline_mask]
    candidate_distances = distance_to_public[candidate_mask]
    baseline_summary = _distance_summary(baseline_distances)
    candidate_summary = _distance_summary(candidate_distances)
    baseline_p95 = baseline_summary["p95"]
    candidate_p95 = candidate_summary["p95"]
    allowed_candidate_p95 = baseline_p95 + allowed_delta
    definition["allowed_candidate_p95"] = allowed_candidate_p95

    outlier_mask = candidate_mask & (distance_to_public > allowed_candidate_p95)
    outlier_count = int(np.count_nonzero(outlier_mask))
    labeled, raw_component_count = label(
        outlier_mask,
        structure=generate_binary_structure(rank=3, connectivity=3),
    )

    point_context = _PointContext(
        crop_shape=shape,
        crop_starts=starts,
        interior_starts=interior_starts,
        distance_to_public=distance_to_public,
        nearest_indices=nearest_indices,
        labeled=labeled,
        xs=None if xs is None else np.asarray(xs),
        ep=None if ep is None else np.asarray(ep),
        reference_fl=None if reference_fl is None else np.asarray(reference_fl),
        reference_fv=None if reference_fv is None else np.asarray(reference_fv),
        reference_fvt=reference,
        baseline_outputs=baseline_outputs,
        candidate_outputs=candidate_outputs,
    )

    component_work = _component_work_records(
        labeled=labeled,
        raw_component_count=raw_component_count,
        distance_to_public=distance_to_public,
        interior_starts=interior_starts,
        crop_starts=starts,
        crop_shape=shape,
    )
    component_work.sort(
        key=lambda item: (
            -item["voxel_count"],
            -item["distance_max"],
            item["minimum_global_coordinate"],
        )
    )
    raw_to_component_id = {
        int(item["raw_component_id"]): index for index, item in enumerate(component_work, start=1)
    }

    sorted_coordinates = sorted(
        (tuple(int(value) for value in coordinate) for coordinate in np.argwhere(outlier_mask)),
        key=lambda coordinate: (
            -float(distance_to_public[coordinate]),
            coordinate[0],
            coordinate[1],
            coordinate[2],
        ),
    )
    stored_coordinates = sorted_coordinates[:point_limit]
    stored_counts_by_component: dict[int, int] = {}
    points: list[dict[str, Any]] = []
    for rank, coordinate in enumerate(stored_coordinates, start=1):
        component_id = raw_to_component_id[int(labeled[coordinate])]
        stored_counts_by_component[component_id] = (
            stored_counts_by_component.get(component_id, 0) + 1
        )
        points.append(
            _point_record(
                coordinate,
                context=point_context,
                component_id=component_id,
                rank=rank,
            )
        )

    components: list[dict[str, Any]] = []
    for component_id, work in enumerate(component_work[:component_limit], start=1):
        representative_coordinate = work["representative_coordinate"]
        component = {
            "component_id": component_id,
            "voxel_count": work["voxel_count"],
            "crop_local_bounding_box": work["crop_local_bounding_box"],
            "global_bounding_box": work["global_bounding_box"],
            "crop_local_centroid": work["crop_local_centroid"],
            "global_centroid": work["global_centroid"],
            "distance_to_public_fvt": work["distance_to_public_fvt"],
            "minimum_crop_face_distance": work["minimum_crop_face_distance"],
            "representative_point": _point_record(
                representative_coordinate,
                context=point_context,
                component_id=component_id,
                rank=None,
            ),
            "stored_point_count": stored_counts_by_component.get(component_id, 0),
            "truncated": stored_counts_by_component.get(component_id, 0) < work["voxel_count"],
        }
        components.append(component)

    crop_face_minima = [
        _crop_face_distance(
            tuple(coordinate[axis] + interior_starts[axis] for axis in range(3)),
            shape,
        )["minimum"]
        for coordinate in sorted_coordinates
    ]
    summary = {
        "public_sparse_ridge_count": mask_counts["public_fvt"],
        "baseline_sparse_ridge_count": mask_counts["baseline_fvt"],
        "candidate_sparse_ridge_count": mask_counts["candidate_fvt"],
        "baseline_candidate_to_public_p95": baseline_p95,
        "candidate_candidate_to_public_p95": candidate_p95,
        "candidate_minus_baseline_p95": candidate_p95 - baseline_p95,
        "allowed_candidate_p95": allowed_candidate_p95,
        "outlier_count": outlier_count,
        "component_count": int(raw_component_count),
        "stored_point_count": len(points),
        "stored_component_count": len(components),
        "points_truncated": len(points) < outlier_count,
        "components_truncated": len(components) < int(raw_component_count),
        "minimum_crop_face_distance": min(crop_face_minima) if crop_face_minima else None,
        "maximum_crop_face_distance": max(crop_face_minima) if crop_face_minima else None,
    }
    return {
        "status": "available",
        "definition": definition,
        "distance_metrics": {
            "baseline_candidate_to_public_fvt": baseline_summary,
            "candidate_candidate_to_public_fvt": candidate_summary,
        },
        "summary": summary,
        "points": points,
        "components": components,
    }


def map_base_roi_slices_within_context(
    base_global_slices: tuple[slice, slice, slice],
    context_global_slices: tuple[slice, slice, slice],
) -> tuple[slice, slice, slice]:
    """Map an exact global base ROI into context-crop-local slices.

    Mapping is derived from global slice bounds, so it remains correct when a
    same-center context crop is shifted at a full-volume boundary.
    """

    base_starts, base_stops, _ = _slice_geometry(base_global_slices, name="base_global_slices")
    context_starts, context_stops, _ = _slice_geometry(
        context_global_slices, name="context_global_slices"
    )
    for axis, (base_start, base_stop, context_start, context_stop) in enumerate(
        zip(base_starts, base_stops, context_starts, context_stops, strict=True)
    ):
        if base_start < context_start or base_stop > context_stop:
            raise ValueError(
                f"base_global_slices[{axis}] is not fully contained in "
                f"context_global_slices[{axis}]"
            )
    return tuple(
        slice(base_start - context_start, base_stop - context_start)
        for base_start, base_stop, context_start in zip(
            base_starts, base_stops, context_starts, strict=True
        )
    )


def extract_same_global_roi(
    context_values: np.ndarray,
    *,
    base_global_slices: tuple[slice, slice, slice],
    context_global_slices: tuple[slice, slice, slice],
) -> np.ndarray:
    """Extract the base global ROI from one context-crop-local 3D array."""

    values = np.asarray(context_values)
    if values.ndim != 3:
        raise ValueError("context_values must be a 3D array")
    _, _, context_shape = _slice_geometry(context_global_slices, name="context_global_slices")
    if values.shape != context_shape:
        raise ValueError(
            f"context_values shape {values.shape} does not match context slice shape "
            f"{context_shape}"
        )
    roi_slices = map_base_roi_slices_within_context(
        base_global_slices,
        context_global_slices,
    )
    return values[roi_slices]


def extract_same_global_roi_outputs(
    context_outputs: Mapping[str, np.ndarray],
    *,
    base_global_slices: tuple[slice, slice, slice],
    context_global_slices: tuple[slice, slice, slice],
) -> dict[str, np.ndarray]:
    """Extract the exact base global ROI from every context pipeline output."""

    return {
        name: extract_same_global_roi(
            values,
            base_global_slices=base_global_slices,
            context_global_slices=context_global_slices,
        )
        for name, values in context_outputs.items()
    }


def slices_to_json(slices: tuple[slice, slice, slice]) -> list[dict[str, int | str]]:
    """Return strict-JSON slice bounds in repository axis order."""

    starts, stops, _ = _slice_geometry(slices, name="slices")
    return [
        {"axis": axis, "start": start, "stop": stop}
        for axis, start, stop in zip(_AXIS_NAMES, starts, stops, strict=True)
    ]


def build_same_global_roi_stage_comparison(
    *,
    base_outputs: Mapping[str, np.ndarray],
    context_roi_outputs: Mapping[str, np.ndarray],
    ridge_percentile: float = DEFAULT_RIDGE_PERCENTILE,
    ridge_buffer_radius: float = DEFAULT_RIDGE_BUFFER_RADIUS,
    nonzero_epsilon: float = DEFAULT_NONZERO_EPSILON,
) -> dict[str, Any]:
    """Compare fixed pipeline stages on one identical global ROI."""

    percentile = _validate_percentile(ridge_percentile)
    radius = _validate_nonnegative_float(ridge_buffer_radius, "ridge_buffer_radius")
    epsilon = _validate_nonnegative_float(nonzero_epsilon, "nonzero_epsilon")
    stages: dict[str, Any] = {}
    all_available = True
    for short_name, description in _COMPARISON_STAGES:
        base = _optional_output_array(base_outputs, short_name)
        context = _optional_output_array(context_roi_outputs, short_name)
        stage = _compare_stage_arrays(
            base,
            context,
            nonzero_epsilon=epsilon,
        )
        stage["stage"] = description
        if short_name == "fvt" and stage["status"] == "available":
            assert base is not None and context is not None
            stage["ridge_comparison"] = _compare_fvt_ridges(
                base,
                context,
                percentile=percentile,
                radius=radius,
            )
        stages[short_name] = stage
        all_available &= stage["status"] == "available"
    return {
        "status": "available" if all_available else "unavailable",
        "definition": {
            "same_global_roi": True,
            "nonzero_epsilon": epsilon,
            "ridge_percentile": percentile,
            "ridge_positive_only": True,
            "ridge_buffer_radius": radius,
        },
        "stages": stages,
    }


def build_context_outlier_persistence_report(
    *,
    base_outlier_report: Mapping[str, Any],
    reference_fvt: np.ndarray,
    base_baseline_outputs: Mapping[str, np.ndarray],
    base_candidate_outputs: Mapping[str, np.ndarray],
    context_baseline_outputs: Mapping[str, np.ndarray],
    context_candidate_outputs: Mapping[str, np.ndarray],
    base_global_slices: tuple[slice, slice, slice],
    ridge_percentile: float = DEFAULT_RIDGE_PERCENTILE,
    allowed_p95_delta: float = DEFAULT_ALLOWED_P95_DELTA,
    persistence_radius: float = DEFAULT_PERSISTENCE_RADIUS,
) -> dict[str, Any]:
    """Measure base outlier persistence in a context-derived ROI.

    ``context_*_outputs`` must already be sliced to the same global base ROI.
    Consequently, their sparse percentile is computed only over that ROI and is
    independent of values elsewhere in the larger context crop. Summary counts
    use all reconstructed base outliers; point details retain the base report's
    storage limit.
    """

    percentile = _validate_percentile(ridge_percentile)
    allowed_delta = _validate_finite_float(allowed_p95_delta, "allowed_p95_delta")
    radius = _validate_nonnegative_float(persistence_radius, "persistence_radius")
    base_starts, _, base_shape = _slice_geometry(base_global_slices, name="base_global_slices")
    reference = _required_array(reference_fvt, "reference_fvt")
    if reference.shape != base_shape:
        raise ValueError(
            f"reference_fvt shape {reference.shape} does not match base ROI shape {base_shape}"
        )

    required = {
        "base baseline fvt": _required_output_array(base_baseline_outputs, "fvt"),
        "base candidate fvt": _required_output_array(base_candidate_outputs, "fvt"),
        "context baseline fvt": _required_output_array(context_baseline_outputs, "fvt"),
        "context candidate fvt": _required_output_array(context_candidate_outputs, "fvt"),
    }
    mismatched = [name for name, values in required.items() if values.shape != base_shape]
    if mismatched:
        raise ValueError("same-global-ROI shape mismatch: " + ", ".join(mismatched))
    nonfinite = [name for name, values in required.items() if not np.all(np.isfinite(values))]
    if not np.all(np.isfinite(reference)):
        nonfinite.insert(0, "reference_fvt")
    if nonfinite:
        return {
            "status": "unavailable",
            "reason": "non-finite required array: " + ", ".join(nonfinite),
            "summary": {
                "base_outlier_count": _nested_int(base_outlier_report, "summary", "outlier_count"),
                "stored_base_outlier_count": len(_report_points(base_outlier_report)),
                "context_outlier_count": None,
                "base_outlier_points_with_context_candidate_within_radius": None,
                "base_outlier_points_with_context_candidate_within_2_samples": None,
                "persistence_fraction": None,
            },
            "points": [],
        }

    interior_margin = _report_interior_margin(base_outlier_report)
    local_interior = interior_slices(base_shape, margin=interior_margin)
    interior_starts = tuple(int(value.start) for value in local_interior)

    context_outlier_report = build_public_fvt_distance_outlier_report(
        reference_fvt=reference,
        baseline_outputs=context_baseline_outputs,
        candidate_outputs=context_candidate_outputs,
        crop_slices=base_global_slices,
        interior_margin=interior_margin,
        ridge_percentile=percentile,
        allowed_p95_delta=allowed_delta,
        max_points=max(1, len(_report_points(base_outlier_report))),
        max_components=max(
            1,
            _nested_int(base_outlier_report, "summary", "component_count") or 1,
        ),
    )

    base_candidate = required["base candidate fvt"]
    context_candidate = required["context candidate fvt"]
    # Match the original outlier metric exactly. Sparse thresholds and EDTs are
    # defined on the interior of the same-global base ROI, never on the full
    # base crop or on the larger context volume.
    base_candidate_field = _sparse_ridge_distance_field(
        base_candidate[local_interior],
        percentile=percentile,
        positive_only=True,
    )
    context_candidate_field = _sparse_ridge_distance_field(
        context_candidate[local_interior],
        percentile=percentile,
        positive_only=True,
        return_indices=True,
    )
    public_field = _sparse_ridge_distance_field(
        reference[local_interior],
        percentile=percentile,
        positive_only=True,
    )
    base_candidate_mask = base_candidate_field.mask
    context_candidate_mask = context_candidate_field.mask
    distance_to_public = public_field.distance
    distance_to_context_candidate = context_candidate_field.distance
    nearest_context_candidate_indices = context_candidate_field.nearest_indices

    persistence_points: list[dict[str, Any]] = []
    for point in _report_points(base_outlier_report):
        global_coordinate = _coordinate3(point.get("global_coordinate"), "global_coordinate")
        local_coordinate = tuple(global_coordinate[axis] - base_starts[axis] for axis in range(3))
        if any(
            coordinate < 0 or coordinate >= base_shape[axis]
            for axis, coordinate in enumerate(local_coordinate)
        ):
            raise ValueError(
                f"outlier global coordinate {global_coordinate} is outside base_global_slices"
            )
        interior_coordinate = tuple(
            local_coordinate[axis] - interior_starts[axis] for axis in range(3)
        )
        if any(
            coordinate < 0 or coordinate >= base_candidate_mask.shape[axis]
            for axis, coordinate in enumerate(interior_coordinate)
        ):
            raise ValueError(
                f"outlier global coordinate {global_coordinate} is outside the interior ROI"
            )
        nearest_distance = (
            None
            if distance_to_context_candidate is None
            else float(distance_to_context_candidate[interior_coordinate])
        )
        persists = nearest_distance is not None and nearest_distance <= radius
        persists_within_two = nearest_distance is not None and nearest_distance <= 2.0
        if nearest_context_candidate_indices is None or distance_to_public is None:
            public_distance = None
        else:
            nearest_context_coordinate = tuple(
                int(nearest_context_candidate_indices[axis][interior_coordinate])
                for axis in range(3)
            )
            public_distance = float(distance_to_public[nearest_context_coordinate])
        persistence_points.append(
            {
                "rank": _optional_int(point.get("rank")),
                "component_id": _optional_int(point.get("component_id")),
                "global_coordinate": list(global_coordinate),
                "base_roi_local_coordinate": list(local_coordinate),
                "interior_local_coordinate": list(interior_coordinate),
                "base_candidate_fvt_value": _array_scalar(base_candidate, local_coordinate),
                "context_candidate_fvt_value": _array_scalar(context_candidate, local_coordinate),
                "base_candidate_sparse_mask_membership": bool(
                    base_candidate_mask[interior_coordinate]
                ),
                "context_candidate_sparse_mask_membership": bool(
                    context_candidate_mask[interior_coordinate]
                ),
                "nearest_context_candidate_sparse_ridge_distance": nearest_distance,
                "persistence_radius": radius,
                "persists_within_radius": bool(persists),
                "persists_within_2_samples": bool(persists_within_two),
                "base_candidate_to_public_distance": _optional_finite_float(
                    point.get("distance_to_public_fvt")
                ),
                "context_candidate_to_public_distance": public_distance,
                "stage_values": {
                    name: {
                        "base": _optional_output_scalar(
                            base_candidate_outputs, name, local_coordinate
                        ),
                        "context": _optional_output_scalar(
                            context_candidate_outputs, name, local_coordinate
                        ),
                    }
                    for name in ("pt", "tt", "fet", "fv", "vp", "vt")
                },
            }
        )

    base_outlier_mask = _reconstruct_outlier_mask(
        reference,
        base_baseline_outputs,
        base_candidate_outputs,
        percentile=percentile,
        allowed_delta=allowed_delta,
        interior_margin=interior_margin,
    )
    context_outlier_mask = _reconstruct_outlier_mask(
        reference,
        context_baseline_outputs,
        context_candidate_outputs,
        percentile=percentile,
        allowed_delta=allowed_delta,
        interior_margin=interior_margin,
    )
    base_outlier_count = int(np.count_nonzero(base_outlier_mask))
    if distance_to_context_candidate is None:
        persistence_count = 0
        persistence_within_two_count = 0
    else:
        all_base_outlier_distances = distance_to_context_candidate[
            base_outlier_mask[local_interior]
        ]
        persistence_count = int(np.count_nonzero(all_base_outlier_distances <= radius))
        persistence_within_two_count = int(np.count_nonzero(all_base_outlier_distances <= 2.0))
    component_overlap = _component_overlap_summary(base_outlier_mask, context_outlier_mask)
    stored_count = len(persistence_points)
    context_outlier_count = _nested_int(context_outlier_report, "summary", "outlier_count")
    return {
        "status": (
            "available" if context_outlier_report.get("status") == "available" else "unavailable"
        ),
        "reason": (
            None
            if context_outlier_report.get("status") == "available"
            else context_outlier_report.get("reason")
        ),
        "definition": {
            "ridge_percentile": percentile,
            "positive_only": True,
            "same_global_base_roi": True,
            "persistence_radius": radius,
            "distance_units": "samples",
        },
        "summary": {
            "base_outlier_count": base_outlier_count,
            "stored_base_outlier_count": stored_count,
            "context_outlier_count": context_outlier_count,
            "base_outlier_points_with_context_candidate_within_radius": persistence_count,
            "base_outlier_points_with_context_candidate_within_2_samples": (
                persistence_within_two_count
            ),
            "persistence_fraction": (
                float(persistence_within_two_count / base_outlier_count)
                if base_outlier_count
                else None
            ),
            "base_context_outlier_component_overlap": component_overlap,
        },
        "points": persistence_points,
    }


class _PointContext:
    """Internal immutable-enough container for point record construction."""

    def __init__(
        self,
        *,
        crop_shape: tuple[int, int, int],
        crop_starts: tuple[int, int, int],
        interior_starts: tuple[int, int, int],
        distance_to_public: np.ndarray,
        nearest_indices: np.ndarray,
        labeled: np.ndarray,
        xs: np.ndarray | None,
        ep: np.ndarray | None,
        reference_fl: np.ndarray | None,
        reference_fv: np.ndarray | None,
        reference_fvt: np.ndarray,
        baseline_outputs: Mapping[str, np.ndarray],
        candidate_outputs: Mapping[str, np.ndarray],
    ) -> None:
        self.crop_shape = crop_shape
        self.crop_starts = crop_starts
        self.interior_starts = interior_starts
        self.distance_to_public = distance_to_public
        self.nearest_indices = nearest_indices
        self.labeled = labeled
        self.xs = xs
        self.ep = ep
        self.reference_fl = reference_fl
        self.reference_fv = reference_fv
        self.reference_fvt = reference_fvt
        self.baseline_outputs = baseline_outputs
        self.candidate_outputs = candidate_outputs


def _point_record(
    interior_coordinate: tuple[int, int, int],
    *,
    context: _PointContext,
    component_id: int,
    rank: int | None,
) -> dict[str, Any]:
    crop_coordinate = tuple(
        interior_coordinate[axis] + context.interior_starts[axis] for axis in range(3)
    )
    global_coordinate = tuple(
        crop_coordinate[axis] + context.crop_starts[axis] for axis in range(3)
    )
    nearest_interior = tuple(
        int(context.nearest_indices[axis][interior_coordinate]) for axis in range(3)
    )
    nearest_crop = tuple(
        nearest_interior[axis] + context.interior_starts[axis] for axis in range(3)
    )
    nearest_global = tuple(nearest_crop[axis] + context.crop_starts[axis] for axis in range(3))
    record: dict[str, Any] = {
        "distance_to_public_fvt": float(context.distance_to_public[interior_coordinate]),
        "interior_local_coordinate": list(interior_coordinate),
        "crop_local_coordinate": list(crop_coordinate),
        "global_coordinate": list(global_coordinate),
        "nearest_public_fvt": {
            "interior_local_coordinate": list(nearest_interior),
            "crop_local_coordinate": list(nearest_crop),
            "global_coordinate": list(nearest_global),
        },
        "crop_face_distance": _crop_face_distance(crop_coordinate, context.crop_shape),
        "component_id": int(component_id),
        "values": _point_values(context, crop_coordinate),
    }
    if rank is not None:
        record = {"rank": int(rank), **record}
    return record


def _point_values(
    context: _PointContext,
    crop_coordinate: tuple[int, int, int],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if context.xs is not None:
        values["xs"] = _array_scalar(context.xs, crop_coordinate)
    if context.ep is not None:
        values["ep"] = _array_scalar(context.ep, crop_coordinate)

    reference: dict[str, Any] = {"fvt": _array_scalar(context.reference_fvt, crop_coordinate)}
    if context.reference_fl is not None:
        reference["fl"] = _array_scalar(context.reference_fl, crop_coordinate)
    if context.reference_fv is not None:
        reference["fv"] = _array_scalar(context.reference_fv, crop_coordinate)
    values["reference"] = reference

    shared = {
        name: _optional_output_scalar(context.baseline_outputs, name, crop_coordinate)
        for name in _SHARED_VALUE_NAMES
        if _optional_output_array(context.baseline_outputs, name) is not None
    }
    if shared:
        values["shared"] = shared
    for role, outputs in (
        ("baseline", context.baseline_outputs),
        ("candidate", context.candidate_outputs),
    ):
        branch = {
            name: _optional_output_scalar(outputs, name, crop_coordinate)
            for name in _BRANCH_VALUE_NAMES
            if _optional_output_array(outputs, name) is not None
        }
        if branch:
            values[role] = branch
    return values


def _component_work_records(
    *,
    labeled: np.ndarray,
    raw_component_count: int,
    distance_to_public: np.ndarray,
    interior_starts: tuple[int, int, int],
    crop_starts: tuple[int, int, int],
    crop_shape: tuple[int, int, int],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_component_id in range(1, raw_component_count + 1):
        coordinates_array = np.argwhere(labeled == raw_component_id)
        coordinates = [tuple(int(value) for value in row) for row in coordinates_array]
        crop_coordinates = [
            tuple(coordinate[axis] + interior_starts[axis] for axis in range(3))
            for coordinate in coordinates
        ]
        global_coordinates = [
            tuple(coordinate[axis] + crop_starts[axis] for axis in range(3))
            for coordinate in crop_coordinates
        ]
        distances = np.asarray(
            [float(distance_to_public[coordinate]) for coordinate in coordinates],
            dtype=np.float64,
        )
        representative_coordinate = min(
            coordinates,
            key=lambda coordinate: (
                -float(distance_to_public[coordinate]),
                coordinate[0],
                coordinate[1],
                coordinate[2],
            ),
        )
        crop_array = np.asarray(crop_coordinates, dtype=np.int64)
        global_array = np.asarray(global_coordinates, dtype=np.int64)
        crop_face_minimum = min(
            _crop_face_distance(coordinate, crop_shape)["minimum"]
            for coordinate in crop_coordinates
        )
        records.append(
            {
                "raw_component_id": raw_component_id,
                "voxel_count": len(coordinates),
                "distance_max": float(np.max(distances)),
                "minimum_global_coordinate": min(global_coordinates),
                "representative_coordinate": representative_coordinate,
                "crop_local_bounding_box": {
                    "minimum": crop_array.min(axis=0).astype(int).tolist(),
                    "maximum": crop_array.max(axis=0).astype(int).tolist(),
                },
                "global_bounding_box": {
                    "minimum": global_array.min(axis=0).astype(int).tolist(),
                    "maximum": global_array.max(axis=0).astype(int).tolist(),
                },
                "crop_local_centroid": crop_array.mean(axis=0).astype(float).tolist(),
                "global_centroid": global_array.mean(axis=0).astype(float).tolist(),
                "distance_to_public_fvt": _distance_summary(distances),
                "minimum_crop_face_distance": int(crop_face_minimum),
            }
        )
    return records


def _compare_stage_arrays(
    base: np.ndarray | None,
    context: np.ndarray | None,
    *,
    nonzero_epsilon: float,
) -> dict[str, Any]:
    base_finite = None if base is None else bool(np.all(np.isfinite(base)))
    context_finite = None if context is None else bool(np.all(np.isfinite(context)))
    shape_equal = base is not None and context is not None and base.shape == context.shape
    finite_status = {
        "base": base_finite,
        "context_roi": context_finite,
        "both": bool(base_finite and context_finite),
    }
    result: dict[str, Any] = {
        "status": "unavailable",
        "reason": None,
        "shape_equal": bool(shape_equal),
        "shape_equality": bool(shape_equal),
        "base_shape": None if base is None else list(base.shape),
        "context_roi_shape": None if context is None else list(context.shape),
        "finite": finite_status,
        "finite_status": dict(finite_status),
        "base_nonzero_fraction": None,
        "context_roi_nonzero_fraction": None,
        "density_delta": None,
        "normalized_correlation": None,
        "absolute_difference": {"mean": None, "p95": None, "maximum": None},
        "absolute_difference_mean": None,
        "absolute_difference_p95": None,
        "absolute_difference_max": None,
    }
    if base is None or context is None:
        result["reason"] = "stage is missing from base or context outputs"
        return result
    if not shape_equal:
        result["reason"] = "base and context ROI shapes do not match"
        if base_finite and base.size:
            result["base_nonzero_fraction"] = _nonzero_fraction(base, nonzero_epsilon)
        if context_finite and context.size:
            result["context_roi_nonzero_fraction"] = _nonzero_fraction(context, nonzero_epsilon)
        if (
            result["base_nonzero_fraction"] is not None
            and result["context_roi_nonzero_fraction"] is not None
        ):
            result["density_delta"] = (
                result["context_roi_nonzero_fraction"] - result["base_nonzero_fraction"]
            )
        return result
    if not base_finite or not context_finite:
        result["reason"] = "base or context ROI contains non-finite values"
        return result
    if not base.size:
        result["reason"] = "stage arrays must not be empty"
        return result

    base_fraction = _nonzero_fraction(base, nonzero_epsilon)
    context_fraction = _nonzero_fraction(context, nonzero_epsilon)
    absolute_difference = np.abs(
        base.astype(np.float64, copy=False) - context.astype(np.float64, copy=False)
    )
    difference_mean = float(np.mean(absolute_difference))
    difference_p95 = float(np.percentile(absolute_difference, 95.0))
    difference_max = float(np.max(absolute_difference))
    result.update(
        {
            "status": "available",
            "base_nonzero_fraction": base_fraction,
            "context_roi_nonzero_fraction": context_fraction,
            "density_delta": context_fraction - base_fraction,
            "normalized_correlation": normalized_correlation(base, context),
            "absolute_difference": {
                "mean": difference_mean,
                "p95": difference_p95,
                "maximum": difference_max,
            },
            "absolute_difference_mean": difference_mean,
            "absolute_difference_p95": difference_p95,
            "absolute_difference_max": difference_max,
        }
    )
    return result


def _compare_fvt_ridges(
    base: np.ndarray,
    context: np.ndarray,
    *,
    percentile: float,
    radius: float,
) -> dict[str, Any]:
    base_mask = top_percentile_mask(base, percentile=percentile, positive_only=True)
    context_mask = top_percentile_mask(context, percentile=percentile, positive_only=True)
    base_only = base_mask & ~context_mask
    context_only = context_mask & ~base_mask
    base_count = int(np.count_nonzero(base_mask))
    context_count = int(np.count_nonzero(context_mask))
    base_only_count = int(np.count_nonzero(base_only))
    context_only_count = int(np.count_nonzero(context_only))
    return {
        "buffered_ridge_overlap": buffered_ridge_overlap(
            base,
            context,
            percentile=percentile,
            radius=radius,
            positive_only=True,
        ),
        "sparse_ridge_distance_metrics": sparse_ridge_distance_metrics(
            base,
            context,
            percentile=percentile,
            positive_only=True,
        ),
        "ridge_mask_difference": {
            "base_sparse_ridge_count": base_count,
            "context_roi_sparse_ridge_count": context_count,
            "base_only_ridge_count": base_only_count,
            "base_only_ridge_fraction": (
                float(base_only_count / base_count) if base_count else 0.0
            ),
            "context_only_ridge_count": context_only_count,
            "context_only_ridge_fraction": (
                float(context_only_count / context_count) if context_count else 0.0
            ),
        },
    }


def _reconstruct_outlier_mask(
    reference: np.ndarray,
    baseline_outputs: Mapping[str, np.ndarray],
    candidate_outputs: Mapping[str, np.ndarray],
    *,
    percentile: float,
    allowed_delta: float,
    interior_margin: int,
) -> np.ndarray:
    local_interior = interior_slices(reference.shape, margin=interior_margin)
    public_field = _sparse_ridge_distance_field(
        reference[local_interior],
        percentile=percentile,
        positive_only=True,
    )
    baseline_field = _sparse_ridge_distance_field(
        _required_output_array(baseline_outputs, "fvt")[local_interior],
        percentile=percentile,
        positive_only=True,
    )
    candidate_field = _sparse_ridge_distance_field(
        _required_output_array(candidate_outputs, "fvt")[local_interior],
        percentile=percentile,
        positive_only=True,
    )
    full = np.zeros(reference.shape, dtype=bool)
    if public_field.count == 0 or baseline_field.count == 0 or candidate_field.count == 0:
        return full
    assert public_field.distance is not None
    distance_to_public = public_field.distance
    baseline_distances = distance_to_public[baseline_field.mask]
    allowed = _sparse_ridge_distance_summary(baseline_distances)["p95"] + allowed_delta
    full[local_interior] = candidate_field.mask & (distance_to_public > allowed)
    return full


def _component_overlap_summary(
    base_outlier_mask: np.ndarray,
    context_outlier_mask: np.ndarray,
) -> dict[str, Any]:
    structure = generate_binary_structure(rank=3, connectivity=3)
    base_labels, base_count = label(base_outlier_mask, structure=structure)
    context_labels, context_count = label(context_outlier_mask, structure=structure)
    exact_overlap = base_outlier_mask & context_outlier_mask
    overlapping_base_ids = set(
        int(value) for value in np.unique(base_labels[exact_overlap]) if int(value) != 0
    )
    overlapping_context_ids = set(
        int(value) for value in np.unique(context_labels[exact_overlap]) if int(value) != 0
    )
    base_voxels = int(np.count_nonzero(base_outlier_mask))
    context_voxels = int(np.count_nonzero(context_outlier_mask))
    overlap_voxels = int(np.count_nonzero(exact_overlap))
    return {
        "base_component_count": int(base_count),
        "context_component_count": int(context_count),
        "base_components_with_exact_context_overlap": len(overlapping_base_ids),
        "context_components_with_exact_base_overlap": len(overlapping_context_ids),
        "exact_overlap_voxel_count": overlap_voxels,
        "base_outlier_overlap_fraction": (
            float(overlap_voxels / base_voxels) if base_voxels else None
        ),
        "context_outlier_overlap_fraction": (
            float(overlap_voxels / context_voxels) if context_voxels else None
        ),
    }


def _unavailable_outlier_report(
    definition: Mapping[str, Any],
    reason: str,
    *,
    sparse_mask_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "reason": str(reason),
        "definition": dict(definition),
        "summary": {
            "public_sparse_ridge_count": (
                None if sparse_mask_counts is None else sparse_mask_counts.get("public_fvt")
            ),
            "baseline_sparse_ridge_count": (
                None if sparse_mask_counts is None else sparse_mask_counts.get("baseline_fvt")
            ),
            "candidate_sparse_ridge_count": (
                None if sparse_mask_counts is None else sparse_mask_counts.get("candidate_fvt")
            ),
            "baseline_candidate_to_public_p95": None,
            "candidate_candidate_to_public_p95": None,
            "candidate_minus_baseline_p95": None,
            "allowed_candidate_p95": None,
            "outlier_count": None,
            "component_count": None,
            "stored_point_count": 0,
            "stored_component_count": 0,
            "points_truncated": False,
            "components_truncated": False,
            "minimum_crop_face_distance": None,
            "maximum_crop_face_distance": None,
        },
        "points": [],
        "components": [],
    }


def _distance_summary(distances: np.ndarray) -> dict[str, float]:
    values = np.asarray(distances, dtype=np.float64)
    shared = _sparse_ridge_distance_summary(values)
    return {
        "minimum": float(np.min(values)),
        **shared,
        "maximum": float(np.max(values)),
    }


def _crop_face_distance(
    crop_coordinate: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> dict[str, Any]:
    per_axis = [
        min(crop_coordinate[axis], shape[axis] - 1 - crop_coordinate[axis]) for axis in range(3)
    ]
    return {
        "minimum": int(min(per_axis)),
        "per_axis_nearest_face": [int(value) for value in per_axis],
    }


def _slice_geometry(
    slices: tuple[slice, slice, slice],
    *,
    name: str,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    if not isinstance(slices, tuple) or len(slices) != 3:
        raise ValueError(f"{name} must be a tuple of three slices")
    starts: list[int] = []
    stops: list[int] = []
    for axis, value in enumerate(slices):
        if not isinstance(value, slice):
            raise ValueError(f"{name}[{axis}] must be a slice")
        if value.step not in (None, 1):
            raise ValueError(f"{name}[{axis}] must have unit step")
        if value.start is None or value.stop is None:
            raise ValueError(f"{name}[{axis}] must have explicit start and stop")
        if not isinstance(value.start, (int, np.integer)) or not isinstance(
            value.stop, (int, np.integer)
        ):
            raise ValueError(f"{name}[{axis}] bounds must be integers")
        start = int(value.start)
        stop = int(value.stop)
        if start < 0 or stop <= start:
            raise ValueError(f"{name}[{axis}] must satisfy 0 <= start < stop")
        starts.append(start)
        stops.append(stop)
    shape = tuple(stop - start for start, stop in zip(starts, stops, strict=True))
    return tuple(starts), tuple(stops), shape  # type: ignore[return-value]


def _required_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 3 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 3D array")
    return array


def _output_key(outputs: Mapping[str, np.ndarray], short_name: str) -> str | None:
    for key in (f"{short_name}_py.dat", f"{short_name}.dat", short_name):
        if key in outputs:
            return key
    return None


def _required_output_array(
    outputs: Mapping[str, np.ndarray],
    short_name: str,
) -> np.ndarray:
    values = _optional_output_array(outputs, short_name)
    if values is None:
        raise ValueError(f"outputs are missing {short_name!r}")
    if values.ndim != 3 or values.size == 0:
        raise ValueError(f"output {short_name!r} must be a non-empty 3D array")
    return values


def _optional_output_array(
    outputs: Mapping[str, np.ndarray],
    short_name: str,
) -> np.ndarray | None:
    key = _output_key(outputs, short_name)
    return None if key is None else np.asarray(outputs[key])


def _first_present_output_shape_mismatch(
    outputs: Mapping[str, np.ndarray],
    shape: tuple[int, int, int],
) -> str | None:
    for short_name in (*_SHARED_VALUE_NAMES, *_BRANCH_VALUE_NAMES):
        key = _output_key(outputs, short_name)
        if key is not None and np.asarray(outputs[key]).shape != shape:
            return key
    return None


def _array_scalar(array: np.ndarray, coordinate: tuple[int, int, int]) -> float | None:
    return _optional_finite_float(np.asarray(array)[coordinate])


def _optional_output_scalar(
    outputs: Mapping[str, np.ndarray],
    short_name: str,
    coordinate: tuple[int, int, int],
) -> float | None:
    values = _optional_output_array(outputs, short_name)
    return None if values is None else _array_scalar(values, coordinate)


def _nonzero_fraction(values: np.ndarray, epsilon: float) -> float:
    return float(np.count_nonzero(np.abs(values) > epsilon) / values.size)


def _validate_percentile(value: float) -> float:
    percentile = _validate_finite_float(value, "ridge_percentile")
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError("ridge_percentile must be between 0 and 100")
    return percentile


def _validate_finite_float(value: float, name: str) -> float:
    if not np.isscalar(value):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_nonnegative_float(value: float, name: str) -> float:
    result = _validate_finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_positive_int(value: int, name: str) -> int:
    if (
        not isinstance(value, (int, np.integer))
        or isinstance(value, (bool, np.bool_))
        or int(value) < 1
    ):
        raise ValueError(f"{name} must be an integer >= 1")
    return int(value)


def _optional_finite_float(value: Any) -> float | None:
    if value is None or not np.isscalar(value):
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _optional_int(value: Any) -> int | None:
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    return None


def _coordinate3(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain three integer coordinates")
    coordinates: list[int] = []
    for coordinate in value:
        if not isinstance(coordinate, (int, np.integer)) or isinstance(
            coordinate, (bool, np.bool_)
        ):
            raise ValueError(f"{name} must contain three integer coordinates")
        coordinates.append(int(coordinate))
    return tuple(coordinates)  # type: ignore[return-value]


def _nested_int(report: Mapping[str, Any], *path: str) -> int | None:
    value: Any = report
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return _optional_int(value)


def _report_points(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    points = report.get("points")
    if not isinstance(points, list):
        return []
    return [point for point in points if isinstance(point, Mapping)]


def _report_interior_margin(report: Mapping[str, Any]) -> int:
    margin = _nested_int(report, "definition", "interior_margin")
    if margin is None or margin < 0:
        raise ValueError("base_outlier_report is missing a valid interior_margin")
    return margin
