"""Pure algebra validation for synthetic quality scalar reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import fsum, isclose, isfinite, sqrt
from numbers import Integral, Real
from typing import Any

DERIVED_SCALAR_REL_TOL = 1.0e-12
DERIVED_SCALAR_ABS_TOL = 1.0e-12

_DISTANCE_SUMMARY_NAMES = (
    "candidate_to_truth_mean",
    "candidate_to_truth_median",
    "candidate_to_truth_p90",
    "candidate_to_truth_p95",
    "truth_to_candidate_mean",
    "truth_to_candidate_median",
    "truth_to_candidate_p90",
    "truth_to_candidate_p95",
    "symmetric_chamfer_mean",
    "hausdorff_p95",
)
_ORIENTATION_SUMMARY_NAMES = (
    "strike_mean",
    "strike_median",
    "strike_p90",
    "strike_p95",
    "dip_mean",
    "dip_median",
    "dip_p90",
    "dip_p95",
)


def validate_quality_scalar_algebra(
    *,
    overlap: Mapping[str, Any],
    distance: Mapping[str, Any],
    orientation: Mapping[str, Any],
    shape: Sequence[int],
    context: str,
    edge: Mapping[str, Any] | None = None,
    orientation_duplicate_count: int = 0,
) -> None:
    """Validate one selection's quality reports without loading any volumes."""

    validate_overlap_algebra(overlap, f"{context}.buffered_overlap_radius2")
    validate_surface_distance_algebra(distance, shape, f"{context}.surface_distance")
    validate_orientation_algebra(orientation, f"{context}.orientation_error")
    if edge is not None:
        validate_edge_false_positive_algebra(edge, f"{context}.edge_false_positive")
    _validate_selection_counts(
        overlap=overlap,
        distance=distance,
        orientation=orientation,
        edge=edge,
        orientation_duplicate_count=orientation_duplicate_count,
        context=context,
    )


def validate_scanner_quality_scalar_algebra(
    report: Mapping[str, Any], shape: Sequence[int], context: str
) -> None:
    """Validate raw and thinned scanner quality without mixing their count families."""

    truth_count = report["ft_top_truth_count"]
    orientations = report["orientation_error"]
    validate_quality_scalar_algebra(
        overlap=truth_count["buffered_overlap_radius2"],
        distance=truth_count["surface_distance"],
        orientation=orientations["raw_scan_top_truth_count"],
        shape=shape,
        context=f"{context}.ft_top_truth_count",
    )
    validate_orientation_algebra(
        orientations["used_attributes_top_truth_count"],
        f"{context}.orientation_error.used_attributes_top_truth_count",
    )


def validate_downstream_quality_scalar_algebra(
    report: Mapping[str, Any], shape: Sequence[int], context: str
) -> None:
    """Validate every downstream selection in one quality report."""

    edges = report["edge_false_positive"]
    for name in (
        "fv_top_truth_count",
        "fvt_top_truth_count",
        "fv_positive_top_truth_count",
        "fvt_positive_top_truth_count",
    ):
        stage = report[name]
        validate_quality_scalar_algebra(
            overlap=stage["buffered_overlap_radius2"],
            distance=stage["surface_distance"],
            orientation=stage["orientation_error"],
            edge=edges[name],
            shape=shape,
            context=f"{context}.{name}",
        )
    skin = report["skin"]
    if skin is not None:
        validate_quality_scalar_algebra(
            overlap=skin["buffered_overlap_radius2"],
            distance=skin["surface_distance"],
            orientation=skin["orientation_error"],
            edge=edges["skin"],
            orientation_duplicate_count=skin["topology"]["duplicate_cell_count"],
            shape=shape,
            context=f"{context}.skin",
        )


def validate_overlap_algebra(report: Mapping[str, Any], context: str) -> None:
    """Validate exact and buffered overlap counts and derived ratios."""

    candidate_count = _count(report, "candidate_count", context)
    truth_count = _count(report, "truth_count", context)
    intersection_count = _count(report, "intersection_count", context)
    union_count = _count(report, "union_count", context)
    candidate_buffer_count = _count(report, "candidate_in_truth_buffer_count", context)
    truth_buffer_count = _count(report, "truth_in_candidate_buffer_count", context)

    if intersection_count > min(candidate_count, truth_count):
        raise ValueError(f"{context}.intersection_count exceeds a source count")
    if union_count != candidate_count + truth_count - intersection_count:
        raise ValueError(f"{context}.union_count is inconsistent with the candidate counts")
    if not intersection_count <= candidate_buffer_count <= candidate_count:
        raise ValueError(
            f"{context}.candidate_in_truth_buffer_count is inconsistent with overlap counts"
        )
    if not intersection_count <= truth_buffer_count <= truth_count:
        raise ValueError(
            f"{context}.truth_in_candidate_buffer_count is inconsistent with overlap counts"
        )

    precision = _precision(intersection_count, candidate_count)
    recall = _recall(intersection_count, truth_count)
    buffered_precision = _precision(candidate_buffer_count, candidate_count)
    buffered_recall = _recall(truth_buffer_count, truth_count)
    expected = {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "jaccard": 1.0 if union_count == 0 else intersection_count / union_count,
        "buffered_precision": buffered_precision,
        "buffered_recall": buffered_recall,
        "buffered_f1": _f1(buffered_precision, buffered_recall),
    }
    for name, value in expected.items():
        _require_close(report, name, float(value), context)
    buffered_precision_report = _number(report, "buffered_precision", context)
    precision_report = _number(report, "precision", context)
    buffered_recall_report = _number(report, "buffered_recall", context)
    recall_report = _number(report, "recall", context)
    if _meaningfully_below(buffered_precision_report, precision_report) or _meaningfully_below(
        buffered_recall_report, recall_report
    ):
        raise ValueError(f"{context} buffered precision/recall must not be below exact values")


def validate_surface_distance_algebra(
    report: Mapping[str, Any], shape: Sequence[int], context: str
) -> None:
    """Validate distance count conventions, ordering, and derived summaries."""

    candidate_count = _count(report, "candidate_count", context)
    truth_count = _count(report, "truth_count", context)
    values = {name: _number(report, name, context) for name in _DISTANCE_SUMMARY_NAMES}
    for prefix in ("candidate_to_truth", "truth_to_candidate"):
        if not (values[f"{prefix}_median"] <= values[f"{prefix}_p90"] <= values[f"{prefix}_p95"]):
            raise ValueError(f"{context}.{prefix} must satisfy median <= p90 <= p95")
    _require_close(
        report,
        "symmetric_chamfer_mean",
        0.5 * (values["candidate_to_truth_mean"] + values["truth_to_candidate_mean"]),
        context,
    )
    _require_close(
        report,
        "hausdorff_p95",
        max(values["candidate_to_truth_p95"], values["truth_to_candidate_p95"]),
        context,
    )

    if candidate_count == 0 and truth_count == 0:
        expected = 0.0
    elif candidate_count == 0 or truth_count == 0:
        dimensions = tuple(_positive_dimension(value, context) for value in shape)
        expected = sqrt(fsum((value - 1.0) ** 2 for value in dimensions))
    else:
        return
    for name in _DISTANCE_SUMMARY_NAMES:
        _require_close(report, name, expected, context)


def validate_orientation_algebra(report: Mapping[str, Any], context: str) -> None:
    """Validate orientation percentile ordering and the empty convention."""

    count = _count(report, "count", context)
    values = {name: _number(report, name, context) for name in _ORIENTATION_SUMMARY_NAMES}
    for prefix in ("strike", "dip"):
        if not (values[f"{prefix}_median"] <= values[f"{prefix}_p90"] <= values[f"{prefix}_p95"]):
            raise ValueError(f"{context}.{prefix} must satisfy median <= p90 <= p95")
    if count == 0:
        for name in _ORIENTATION_SUMMARY_NAMES:
            _require_close(report, name, 0.0, context)


def validate_edge_false_positive_algebra(report: Mapping[str, Any], context: str) -> None:
    """Validate edge count hierarchy and all derived fractions."""

    candidate_count = _count(report, "candidate_count", context)
    edge_candidate_count = _count(report, "edge_candidate_count", context)
    false_positive_count = _count(report, "edge_false_positive_count", context)
    if edge_candidate_count > candidate_count:
        raise ValueError(f"{context}.edge_candidate_count exceeds candidate_count")
    if false_positive_count > edge_candidate_count:
        raise ValueError(f"{context}.edge_false_positive_count exceeds edge_candidate_count")
    expected = {
        "edge_candidate_fraction": (
            edge_candidate_count / candidate_count if candidate_count else 0.0
        ),
        "edge_false_positive_fraction_of_candidates": (
            false_positive_count / candidate_count if candidate_count else 0.0
        ),
        "edge_false_positive_fraction_of_edge_candidates": (
            false_positive_count / edge_candidate_count if edge_candidate_count else 0.0
        ),
    }
    for name, value in expected.items():
        _require_close(report, name, float(value), context)


def _validate_selection_counts(
    *,
    overlap: Mapping[str, Any],
    distance: Mapping[str, Any],
    orientation: Mapping[str, Any],
    edge: Mapping[str, Any] | None,
    orientation_duplicate_count: int,
    context: str,
) -> None:
    candidate_count = _count(overlap, "candidate_count", context)
    duplicate_count = _nonnegative_integer(
        orientation_duplicate_count, f"{context}.orientation_duplicate_count"
    )
    if _count(distance, "candidate_count", context) != candidate_count:
        raise ValueError(f"{context} candidate counts must match for the same selection")
    if _count(orientation, "count", context) != candidate_count + duplicate_count:
        raise ValueError(f"{context} candidate counts must match for the same selection")
    if edge is not None and _count(edge, "candidate_count", context) != candidate_count:
        raise ValueError(f"{context} candidate counts must match for the same selection")


def _count(report: Mapping[str, Any], name: str, context: str) -> int:
    try:
        value = report[name]
    except KeyError as error:
        raise ValueError(f"{context}.{name} is required") from error
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{context}.{name} must be a non-negative integer")
    return int(value)


def _nonnegative_integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return int(value)


def _number(report: Mapping[str, Any], name: str, context: str) -> float:
    try:
        value = report[name]
    except KeyError as error:
        raise ValueError(f"{context}.{name} is required") from error
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context}.{name} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{context}.{name} must be a finite number")
    return result


def _require_close(report: Mapping[str, Any], name: str, expected: float, context: str) -> None:
    actual = _number(report, name, context)
    if not derived_scalars_close(actual, expected):
        raise ValueError(f"{context}.{name} is inconsistent with its source counts/summaries")


def derived_scalars_close(actual: float, expected: float) -> bool:
    """Return whether two derived scalars agree within the canonical tolerance."""

    return isclose(
        actual,
        expected,
        rel_tol=DERIVED_SCALAR_REL_TOL,
        abs_tol=DERIVED_SCALAR_ABS_TOL,
    )


def _meaningfully_below(actual: float, lower_bound: float) -> bool:
    return actual < lower_bound and not derived_scalars_close(actual, lower_bound)


def _positive_dimension(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{context} shape must contain positive integers")
    return int(value)


def _precision(intersection_count: int, candidate_count: int) -> float:
    return 1.0 if candidate_count == 0 else intersection_count / candidate_count


def _recall(intersection_count: int, truth_count: int) -> float:
    return 1.0 if truth_count == 0 else intersection_count / truth_count


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return 2.0 * precision * recall / denominator if denominator else 0.0


__all__ = [
    "DERIVED_SCALAR_ABS_TOL",
    "DERIVED_SCALAR_REL_TOL",
    "derived_scalars_close",
    "validate_edge_false_positive_algebra",
    "validate_downstream_quality_scalar_algebra",
    "validate_orientation_algebra",
    "validate_overlap_algebra",
    "validate_quality_scalar_algebra",
    "validate_scanner_quality_scalar_algebra",
    "validate_surface_distance_algebra",
]
