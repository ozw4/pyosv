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


def validate_selection_cardinality(
    *,
    selection: str,
    candidate_count: Any,
    truth_count: Any,
    context: str,
) -> None:
    """Validate candidate cardinality against the generating truth-surface support."""

    candidate = _nonnegative_integer(candidate_count, f"{context}.candidate_count")
    truth = _nonnegative_integer(truth_count, f"{context}.truth_count")
    if selection == "top_truth_count":
        if candidate != truth:
            raise ValueError(
                f"{context} top_truth_count candidate_count must equal surface_distance.truth_count"
            )
    elif selection == "positive_top_truth_count":
        if candidate > truth:
            raise ValueError(
                f"{context} positive_top_truth_count candidate_count must not exceed "
                "surface_distance.truth_count"
            )
    else:
        raise ValueError(f"{context}.selection has unsupported cardinality semantics")


def validate_scanner_quality_scalar_algebra(
    report: Mapping[str, Any], shape: Sequence[int], context: str
) -> None:
    """Validate raw and thinned scanner quality top-truth cardinality."""

    truth_count = report["ft_top_truth_count"]
    orientations = report["orientation_error"]
    validate_quality_scalar_algebra(
        overlap=truth_count["buffered_overlap_radius2"],
        distance=truth_count["surface_distance"],
        orientation=orientations["raw_scan_top_truth_count"],
        shape=shape,
        context=f"{context}.ft_top_truth_count",
    )
    validate_selection_cardinality(
        selection="top_truth_count",
        candidate_count=truth_count["buffered_overlap_radius2"]["candidate_count"],
        truth_count=truth_count["surface_distance"]["truth_count"],
        context=f"{context}.ft_top_truth_count",
    )
    thinned_context = f"{context}.orientation_error.used_attributes_top_truth_count"
    thinned_orientation = orientations["used_attributes_top_truth_count"]
    validate_orientation_algebra(thinned_orientation, thinned_context)
    validate_selection_cardinality(
        selection="top_truth_count",
        candidate_count=thinned_orientation["count"],
        truth_count=truth_count["surface_distance"]["truth_count"],
        context=thinned_context,
    )


def validate_downstream_quality_scalar_algebra(
    report: Mapping[str, Any], shape: Sequence[int], context: str
) -> None:
    """Validate every downstream selection in one quality report."""

    edges = report["edge_false_positive"]
    for name, selection in (
        ("fv_top_truth_count", "top_truth_count"),
        ("fvt_top_truth_count", "top_truth_count"),
        ("fv_positive_top_truth_count", "positive_top_truth_count"),
        ("fvt_positive_top_truth_count", "positive_top_truth_count"),
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
        validate_selection_cardinality(
            selection=selection,
            candidate_count=stage["buffered_overlap_radius2"]["candidate_count"],
            truth_count=stage["surface_distance"]["truth_count"],
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


def validate_skin_topology_algebra(
    report: Mapping[str, Any], context: str, *, require_empty: bool = False
) -> None:
    """Validate a skin topology summary using only its reported scalars."""

    skin_count = _count(report, "skin_count", context)
    cell_count = _count(report, "cell_count", context)
    unique_cell_count = _count(report, "unique_cell_count", context)
    duplicate_cell_count = _count(report, "duplicate_cell_count", context)
    largest_skin_size = _count(report, "largest_skin_size", context)
    small_skin_count = _count(report, "small_skin_count", context)
    small_skin_cell_count = _count(report, "small_skin_cell_count", context)

    if require_empty and skin_count != 0:
        raise ValueError(f"{context} must be empty when skinning is disabled")

    if unique_cell_count > cell_count:
        raise ValueError(f"{context}.unique_cell_count exceeds cell_count")
    if duplicate_cell_count != cell_count - unique_cell_count:
        raise ValueError(f"{context}.duplicate_cell_count is inconsistent with cell counts")
    if largest_skin_size > cell_count:
        raise ValueError(f"{context}.largest_skin_size exceeds cell_count")
    if small_skin_count > skin_count:
        raise ValueError(f"{context}.small_skin_count exceeds skin_count")
    if small_skin_cell_count > cell_count:
        raise ValueError(f"{context}.small_skin_cell_count exceeds cell_count")

    _require_close(
        report,
        "largest_skin_fraction",
        largest_skin_size / cell_count if cell_count else 0.0,
        context,
    )
    _require_close(
        report,
        "small_skin_cell_fraction",
        small_skin_cell_count / cell_count if cell_count else 0.0,
        context,
    )

    if skin_count == 0:
        zero_counts = {
            "cell_count": cell_count,
            "unique_cell_count": unique_cell_count,
            "duplicate_cell_count": duplicate_cell_count,
            "largest_skin_size": largest_skin_size,
            "small_skin_count": small_skin_count,
            "small_skin_cell_count": small_skin_cell_count,
        }
        if any(zero_counts.values()):
            raise ValueError(f"{context} empty skin topology must contain only zero counts")


def validate_component_topology_algebra(
    report: Mapping[str, Any],
    topology: Mapping[str, Any],
    context: str,
) -> None:
    """Validate component topology summaries and arrays without rebuilding incidence data."""

    truth_component_count = _count(report, "truth_component_count", context)
    covered_count = _count(report, "covered_truth_component_count", context)
    uncovered_count = _count(report, "uncovered_truth_component_count", context)
    skin_count = _count(report, "skin_count", context)
    skin_with_truth_count = _count(report, "skin_with_truth_count", context)
    skin_without_truth_count = _count(report, "skin_without_truth_count", context)
    over_merge_count = _count(report, "over_merge_skin_count", context)
    over_split_count = _count(report, "over_split_truth_component_count", context)

    truth_components = _mapping_array(report, "truth_components", context)
    skins = _mapping_array(report, "skins", context)
    if truth_component_count != len(truth_components):
        raise ValueError(f"{context}.truth_component_count does not match truth_components")
    if skin_count != len(skins):
        raise ValueError(f"{context}.skin_count does not match skins")
    if covered_count + uncovered_count != truth_component_count:
        raise ValueError(f"{context} covered and uncovered truth counts are inconsistent")
    if skin_with_truth_count + skin_without_truth_count != skin_count:
        raise ValueError(f"{context} with-truth and without-truth skin counts are inconsistent")
    if over_merge_count > skin_count:
        raise ValueError(f"{context}.over_merge_skin_count exceeds skin_count")
    if over_split_count > truth_component_count:
        raise ValueError(
            f"{context}.over_split_truth_component_count exceeds truth_component_count"
        )

    topology_skin_count = _count(topology, "skin_count", f"{context}.topology")
    if skin_count != topology_skin_count:
        raise ValueError(f"{context}.skin_count does not match skin topology")

    truth_ids: list[int] = []
    recalls: list[float] = []
    truth_touching_counts: list[int] = []
    actually_covered_count = 0
    for index, item in enumerate(truth_components):
        item_context = f"{context}.truth_components[{index}]"
        truth_id = _count(item, "truth_id", item_context)
        if truth_id <= 0:
            raise ValueError(f"{item_context}.truth_id must be positive")
        truth_ids.append(truth_id)
        truth_cell_count = _count(item, "truth_cell_count", item_context)
        if truth_cell_count <= 0:
            raise ValueError(f"{item_context}.truth_cell_count must be positive")
        covered_cell_count = _count(item, "covered_cell_count", item_context)
        if covered_cell_count > truth_cell_count:
            raise ValueError(f"{item_context}.covered_cell_count exceeds truth_cell_count")
        skin_count_touching = _count(item, "skin_count_touching", item_context)
        if skin_count_touching > skin_count:
            raise ValueError(f"{item_context}.skin_count_touching exceeds skin_count")
        dominant_skin_cell_count = _count(item, "dominant_skin_cell_count", item_context)
        if dominant_skin_cell_count > covered_cell_count:
            raise ValueError(f"{item_context}.dominant_skin_cell_count exceeds covered_cell_count")
        recall = covered_cell_count / truth_cell_count
        _require_close(item, "recall", recall, item_context)
        _require_close(
            item,
            "dominant_skin_fraction_of_truth",
            dominant_skin_cell_count / truth_cell_count,
            item_context,
        )
        dominant_skin_index = item.get("dominant_skin_index")
        if covered_cell_count == 0:
            if (
                skin_count_touching != 0
                or dominant_skin_index is not None
                or dominant_skin_cell_count != 0
            ):
                raise ValueError(f"{item_context} uncovered truth component is inconsistent")
        else:
            actually_covered_count += 1
            if skin_count_touching == 0:
                raise ValueError(f"{item_context}.skin_count_touching must be positive")
            if dominant_skin_cell_count == 0:
                raise ValueError(f"{item_context}.dominant_skin_cell_count must be positive")
            if not _valid_index(dominant_skin_index, skin_count):
                raise ValueError(f"{item_context}.dominant_skin_index is not a valid skin index")
        recalls.append(recall)
        truth_touching_counts.append(skin_count_touching)

    if truth_ids != sorted(set(truth_ids)):
        raise ValueError(f"{context}.truth_components truth_id values must be unique and sorted")
    if covered_count != actually_covered_count:
        raise ValueError(f"{context}.covered_truth_component_count does not match truth_components")

    truth_id_set = set(truth_ids)
    purities: list[float] = []
    skin_touching_counts: list[int] = []
    total_skin_cells = 0
    actually_with_truth_count = 0
    for index, item in enumerate(skins):
        item_context = f"{context}.skins[{index}]"
        if _count(item, "skin_index", item_context) != index:
            raise ValueError(f"{item_context}.skin_index must match its array index")
        cell_count = _count(item, "cell_count", item_context)
        truth_cell_count = _count(item, "truth_cell_count", item_context)
        background_cell_count = _count(item, "background_cell_count", item_context)
        if truth_cell_count + background_cell_count != cell_count:
            raise ValueError(f"{item_context} truth and background cell counts are inconsistent")
        touching_count = _count(item, "truth_component_count_touching", item_context)
        if touching_count > truth_component_count:
            raise ValueError(
                f"{item_context}.truth_component_count_touching exceeds truth_component_count"
            )
        dominant_truth_cell_count = _count(item, "dominant_truth_cell_count", item_context)
        if dominant_truth_cell_count > truth_cell_count:
            raise ValueError(f"{item_context}.dominant_truth_cell_count exceeds truth_cell_count")
        purity = dominant_truth_cell_count / cell_count if cell_count else 0.0
        _require_close(item, "purity", purity, item_context)
        dominant_truth_id = item.get("dominant_truth_id")
        if truth_cell_count == 0:
            if (
                touching_count != 0
                or dominant_truth_id is not None
                or dominant_truth_cell_count != 0
            ):
                raise ValueError(f"{item_context} background-only skin is inconsistent")
        else:
            actually_with_truth_count += 1
            if touching_count == 0:
                raise ValueError(f"{item_context}.truth_component_count_touching must be positive")
            if dominant_truth_cell_count == 0:
                raise ValueError(f"{item_context}.dominant_truth_cell_count must be positive")
            if (
                isinstance(dominant_truth_id, bool)
                or not isinstance(dominant_truth_id, Integral)
                or int(dominant_truth_id) not in truth_id_set
            ):
                raise ValueError(
                    f"{item_context}.dominant_truth_id is not a reported truth component"
                )
        total_skin_cells += cell_count
        purities.append(purity)
        skin_touching_counts.append(touching_count)

    if skin_with_truth_count != actually_with_truth_count:
        raise ValueError(f"{context}.skin_with_truth_count does not match skins")
    if total_skin_cells != _count(topology, "cell_count", f"{context}.topology"):
        raise ValueError(f"{context} per-skin cell count does not match skin topology")

    possible_over_merge_count = sum(count >= 2 for count in skin_touching_counts)
    if over_merge_count > possible_over_merge_count:
        raise ValueError(
            f"{context}.over_merge_skin_count exceeds skins touching multiple truth components"
        )
    possible_over_split_count = sum(count >= 2 for count in truth_touching_counts)
    if over_split_count > possible_over_split_count:
        raise ValueError(
            f"{context}.over_split_truth_component_count exceeds truth components touching "
            "multiple skins"
        )

    expected_summaries = {
        "max_truth_components_per_skin": max(skin_touching_counts, default=0),
        "max_skins_per_truth_component": max(truth_touching_counts, default=0),
        "mean_skin_purity": fsum(purities) / len(purities) if purities else 0.0,
        "min_skin_purity": min(purities, default=0.0),
        "mean_truth_component_recall": fsum(recalls) / len(recalls) if recalls else 0.0,
        "min_truth_component_recall": min(recalls, default=0.0),
    }
    for name, expected in expected_summaries.items():
        if name.startswith("max_"):
            if _count(report, name, context) != expected:
                raise ValueError(f"{context}.{name} does not match component arrays")
        else:
            _require_close(report, name, float(expected), context)


def validate_overlap_algebra(report: Mapping[str, Any], context: str) -> None:
    """Validate exact and buffered overlap counts and derived ratios."""

    candidate_count = _count(report, "candidate_count", context)
    truth_count = _count(report, "truth_count", context)
    intersection_count = _count(report, "intersection_count", context)
    union_count = _count(report, "union_count", context)
    candidate_buffer_count = _count(report, "candidate_in_truth_buffer_count", context)
    truth_buffer_count = _count(report, "truth_in_candidate_buffer_count", context)
    radius = _number(report, "radius", context)

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
    if candidate_count == 0 and (candidate_buffer_count != 0 or truth_buffer_count != 0):
        raise ValueError(f"{context} buffered overlap counts require a nonempty candidate mask")
    if truth_count == 0 and (candidate_buffer_count != 0 or truth_buffer_count != 0):
        raise ValueError(f"{context} buffered overlap counts require a nonempty truth mask")
    if radius == 0.0 and (
        candidate_buffer_count != intersection_count or truth_buffer_count != intersection_count
    ):
        raise ValueError(
            f"{context} radius-zero buffered overlap counts must equal intersection_count"
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


def _mapping_array(
    report: Mapping[str, Any], name: str, context: str
) -> tuple[Mapping[str, Any], ...]:
    try:
        value = report[name]
    except KeyError as error:
        raise ValueError(f"{context}.{name} is required") from error
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context}.{name} must be an array")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{context}.{name} must contain only objects")
    return tuple(value)


def _valid_index(value: Any, length: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, Integral) and 0 <= int(value) < length


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
    "validate_component_topology_algebra",
    "validate_edge_false_positive_algebra",
    "validate_downstream_quality_scalar_algebra",
    "validate_orientation_algebra",
    "validate_overlap_algebra",
    "validate_quality_scalar_algebra",
    "validate_scanner_quality_scalar_algebra",
    "validate_selection_cardinality",
    "validate_skin_topology_algebra",
    "validate_surface_distance_algebra",
]
