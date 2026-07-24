"""Pure algebra validation for synthetic quality scalar reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import fsum, isclose, isfinite, sqrt
from numbers import Integral, Real
from typing import Any

from ...synthetic_metrics import COMPONENT_QUALIFICATION_MIN_FRACTION

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

    validate_overlap_algebra(overlap, shape, f"{context}.buffered_overlap_radius2")
    validate_surface_distance_algebra(distance, shape, f"{context}.surface_distance")
    validate_orientation_algebra(orientation, f"{context}.orientation_error")
    if edge is not None:
        validate_edge_false_positive_algebra(
            edge,
            shape,
            f"{context}.edge_false_positive",
        )
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
    report: Mapping[str, Any],
    context: str,
    *,
    shape: Sequence[int],
    require_empty: bool = False,
) -> None:
    """Validate a skin topology summary using only its reported scalars."""

    skin_count = _count(report, "skin_count", context)
    cell_count = _count(report, "cell_count", context)
    unique_cell_count = _count(report, "unique_cell_count", context)
    duplicate_cell_count = _count(report, "duplicate_cell_count", context)
    largest_skin_size = _count(report, "largest_skin_size", context)
    small_skin_count = _count(report, "small_skin_count", context)
    small_skin_cell_count = _count(report, "small_skin_cell_count", context)
    voxel_count = volume_voxel_count(shape)

    if require_empty and skin_count != 0:
        raise ValueError(f"{context} must be empty when skinning is disabled")

    if unique_cell_count > voxel_count:
        raise ValueError(f"{context}.unique_cell_count exceeds volume voxel count")
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


def validate_skin_report_topology_algebra(
    topology: Mapping[str, Any],
    component_topology: Mapping[str, Any],
    context: str,
    *,
    small_skin_size: int,
    shape: Sequence[int],
) -> None:
    """Validate one enabled skin report against its effective fragmentation threshold."""

    validate_skin_topology_algebra(
        topology,
        f"{context}.topology",
        shape=shape,
    )
    validate_component_topology_algebra(
        component_topology,
        topology,
        f"{context}.component_topology",
    )

    threshold = _nonnegative_integer(small_skin_size, "effective small_skin_size")
    if _count(topology, "small_skin_size", f"{context}.topology") != threshold:
        raise ValueError(
            f"{context}.topology.small_skin_size does not match the effective configuration"
        )
    skins = _mapping_array(component_topology, "skins", f"{context}.component_topology")
    cell_counts = [
        _count(item, "cell_count", f"{context}.component_topology.skins[{index}]")
        for index, item in enumerate(skins)
    ]
    total_cell_count = sum(cell_counts)
    largest_skin_size = max(cell_counts, default=0)
    small_cell_counts = [count for count in cell_counts if count < threshold]
    small_skin_cell_count = sum(small_cell_counts)

    expected_counts = {
        "largest_skin_size": largest_skin_size,
        "small_skin_count": len(small_cell_counts),
        "small_skin_cell_count": small_skin_cell_count,
    }
    for name, expected in expected_counts.items():
        if _count(topology, name, f"{context}.topology") != expected:
            raise ValueError(f"{context}.topology.{name} does not match per-skin cell counts")
    _require_close(
        topology,
        "largest_skin_fraction",
        largest_skin_size / total_cell_count if total_cell_count else 0.0,
        f"{context}.topology",
    )
    _require_close(
        topology,
        "small_skin_cell_fraction",
        small_skin_cell_count / total_cell_count if total_cell_count else 0.0,
        f"{context}.topology",
    )


def validate_component_topology_algebra(
    report: Mapping[str, Any],
    topology: Mapping[str, Any],
    context: str,
) -> None:
    """Validate component topology summaries and arrays without rebuilding incidence data."""

    qualification_min_fraction = _number(report, "qualification_min_fraction", context)
    if not 0.0 <= qualification_min_fraction <= 1.0:
        raise ValueError(
            f"{context}.qualification_min_fraction must be in the closed unit interval"
        )
    if qualification_min_fraction != COMPONENT_QUALIFICATION_MIN_FRACTION:
        raise ValueError(
            f"{context}.qualification_min_fraction does not match the canonical contract"
        )

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
    truth_qualifying_counts: list[int] = []
    truth_incidence: dict[tuple[int, int], int] = {}
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
        skin_cell_counts = _mapping_array(item, "skin_cell_counts", item_context)
        incidence_skin_indices: list[int] = []
        incidence_counts: list[int] = []
        for incidence_index, incidence in enumerate(skin_cell_counts):
            incidence_context = f"{item_context}.skin_cell_counts[{incidence_index}]"
            skin_index = _count(incidence, "skin_index", incidence_context)
            if skin_index >= skin_count:
                raise ValueError(f"{incidence_context}.skin_index is not a valid skin index")
            incidence_count = _count(incidence, "covered_cell_count", incidence_context)
            if incidence_count <= 0:
                raise ValueError(f"{incidence_context}.covered_cell_count must be positive")
            if incidence_count > covered_cell_count:
                raise ValueError(
                    f"{incidence_context}.covered_cell_count exceeds component covered_cell_count"
                )
            incidence_skin_indices.append(skin_index)
            incidence_counts.append(incidence_count)
            truth_incidence[(truth_id, skin_index)] = incidence_count
        if incidence_skin_indices != sorted(set(incidence_skin_indices)):
            raise ValueError(f"{item_context}.skin_cell_counts skin_index values must be sorted")
        if covered_cell_count > sum(incidence_counts):
            raise ValueError(
                f"{item_context}.covered_cell_count exceeds summed per-skin covered counts"
            )
        skin_count_touching = _count(item, "skin_count_touching", item_context)
        if skin_count_touching != len(skin_cell_counts):
            raise ValueError(f"{item_context}.skin_count_touching does not match skin_cell_counts")
        dominant_skin_cell_count = _count(item, "dominant_skin_cell_count", item_context)
        expected_dominant_skin_index, expected_dominant_skin_count = _dominant_incidence(
            incidence_skin_indices, incidence_counts
        )
        dominant_skin_index = item.get("dominant_skin_index")
        if isinstance(dominant_skin_index, bool) or (
            dominant_skin_index is not None and not isinstance(dominant_skin_index, Integral)
        ):
            raise ValueError(f"{item_context}.dominant_skin_index must be an integer or null")
        if dominant_skin_index != expected_dominant_skin_index:
            raise ValueError(f"{item_context}.dominant_skin_index does not match skin_cell_counts")
        if dominant_skin_cell_count != expected_dominant_skin_count:
            raise ValueError(
                f"{item_context}.dominant_skin_cell_count does not match skin_cell_counts"
            )
        qualifying_skin_count = _count(item, "qualifying_skin_count", item_context)
        expected_qualifying_skin_count = sum(
            count / truth_cell_count >= qualification_min_fraction for count in incidence_counts
        )
        if qualifying_skin_count != expected_qualifying_skin_count:
            raise ValueError(
                f"{item_context}.qualifying_skin_count does not match skin_cell_counts"
            )
        recall = covered_cell_count / truth_cell_count
        _require_close(item, "recall", recall, item_context)
        _require_close(
            item,
            "dominant_skin_fraction_of_truth",
            dominant_skin_cell_count / truth_cell_count,
            item_context,
        )
        if covered_cell_count == 0:
            if skin_count_touching != 0:
                raise ValueError(f"{item_context} uncovered truth component is inconsistent")
        else:
            actually_covered_count += 1
        recalls.append(recall)
        truth_touching_counts.append(skin_count_touching)
        truth_qualifying_counts.append(qualifying_skin_count)

    if truth_ids != sorted(set(truth_ids)):
        raise ValueError(f"{context}.truth_components truth_id values must be unique and sorted")
    if covered_count != actually_covered_count:
        raise ValueError(f"{context}.covered_truth_component_count does not match truth_components")

    truth_id_set = set(truth_ids)
    purities: list[float] = []
    skin_touching_counts: list[int] = []
    skin_qualifying_counts: list[int] = []
    skin_incidence: dict[tuple[int, int], int] = {}
    total_skin_cells = 0
    total_truth_cells = 0
    total_background_cells = 0
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
        truth_component_cell_counts = _mapping_array(
            item, "truth_component_cell_counts", item_context
        )
        incidence_truth_ids: list[int] = []
        incidence_counts: list[int] = []
        for incidence_index, incidence in enumerate(truth_component_cell_counts):
            incidence_context = f"{item_context}.truth_component_cell_counts[{incidence_index}]"
            truth_id = _count(incidence, "truth_id", incidence_context)
            if truth_id not in truth_id_set:
                raise ValueError(f"{incidence_context}.truth_id is not a reported truth component")
            incidence_count = _count(incidence, "cell_count", incidence_context)
            if incidence_count <= 0:
                raise ValueError(f"{incidence_context}.cell_count must be positive")
            incidence_truth_ids.append(truth_id)
            incidence_counts.append(incidence_count)
            skin_incidence[(truth_id, index)] = incidence_count
        if incidence_truth_ids != sorted(set(incidence_truth_ids)):
            raise ValueError(
                f"{item_context}.truth_component_cell_counts truth_id values must be sorted"
            )
        if truth_cell_count != sum(incidence_counts):
            raise ValueError(
                f"{item_context}.truth_cell_count does not match truth_component_cell_counts"
            )
        touching_count = _count(item, "truth_component_count_touching", item_context)
        if touching_count != len(truth_component_cell_counts):
            raise ValueError(
                f"{item_context}.truth_component_count_touching does not match "
                "truth_component_cell_counts"
            )
        dominant_truth_cell_count = _count(item, "dominant_truth_cell_count", item_context)
        expected_dominant_truth_id, expected_dominant_truth_count = _dominant_incidence(
            incidence_truth_ids, incidence_counts
        )
        dominant_truth_id = item.get("dominant_truth_id")
        if isinstance(dominant_truth_id, bool) or (
            dominant_truth_id is not None and not isinstance(dominant_truth_id, Integral)
        ):
            raise ValueError(f"{item_context}.dominant_truth_id must be an integer or null")
        if dominant_truth_id != expected_dominant_truth_id:
            raise ValueError(
                f"{item_context}.dominant_truth_id does not match truth_component_cell_counts"
            )
        if dominant_truth_cell_count != expected_dominant_truth_count:
            raise ValueError(
                f"{item_context}.dominant_truth_cell_count does not match "
                "truth_component_cell_counts"
            )
        qualifying_truth_count = _count(item, "qualifying_truth_component_count", item_context)
        expected_qualifying_truth_count = (
            sum(count / cell_count >= qualification_min_fraction for count in incidence_counts)
            if cell_count
            else 0
        )
        if qualifying_truth_count != expected_qualifying_truth_count:
            raise ValueError(
                f"{item_context}.qualifying_truth_component_count does not match "
                "truth_component_cell_counts"
            )
        purity = dominant_truth_cell_count / cell_count if cell_count else 0.0
        _require_close(item, "purity", purity, item_context)
        if truth_cell_count == 0:
            if touching_count != 0:
                raise ValueError(f"{item_context} background-only skin is inconsistent")
        else:
            actually_with_truth_count += 1
        total_skin_cells += cell_count
        total_truth_cells += truth_cell_count
        total_background_cells += background_cell_count
        purities.append(purity)
        skin_touching_counts.append(touching_count)
        skin_qualifying_counts.append(qualifying_truth_count)

    if skin_with_truth_count != actually_with_truth_count:
        raise ValueError(f"{context}.skin_with_truth_count does not match skins")
    if total_skin_cells != _count(topology, "cell_count", f"{context}.topology"):
        raise ValueError(f"{context} per-skin cell count does not match skin topology")

    covered_truth_cells = sum(
        _count(item, "covered_cell_count", f"{context}.truth_components[{index}]")
        for index, item in enumerate(truth_components)
    )
    unique_cell_count = _count(topology, "unique_cell_count", f"{context}.topology")
    unique_background_cells = unique_cell_count - covered_truth_cells
    if covered_truth_cells > total_truth_cells or unique_background_cells < 0:
        raise ValueError(
            f"{context} unique covered truth cells are inconsistent with per-skin incidence"
        )
    if unique_background_cells > total_background_cells:
        raise ValueError(
            f"{context} unique background cells are inconsistent with per-skin incidence"
        )

    if set(skin_incidence) != set(truth_incidence):
        raise ValueError(f"{context} per-skin and per-truth incidence pair sets do not match")
    for pair, covered_cell_count in truth_incidence.items():
        if covered_cell_count > skin_incidence[pair]:
            raise ValueError(
                f"{context} per-truth unique incidence count exceeds per-skin cell count"
            )

    expected_over_merge_count = sum(count >= 2 for count in skin_qualifying_counts)
    if over_merge_count != expected_over_merge_count:
        raise ValueError(f"{context}.over_merge_skin_count does not match qualifying incidence")
    expected_over_split_count = sum(count >= 2 for count in truth_qualifying_counts)
    if over_split_count != expected_over_split_count:
        raise ValueError(
            f"{context}.over_split_truth_component_count does not match qualifying incidence"
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


def validate_component_topology_evidence(
    report: Mapping[str, Any],
    truth_evidence: Mapping[str, Any],
    overlap: Mapping[str, Any],
    context: str,
) -> None:
    """Bind persisted component topology to trial truth and exact skin overlap evidence."""

    fault_voxel_count = _count(truth_evidence, "fault_voxel_count", f"{context}.truth_evidence")
    intersection_count = _count(
        overlap,
        "intersection_count",
        f"{context}.buffered_overlap_radius2",
    )
    truth_components = _mapping_array(report, "truth_components", context)
    truth_component_count = _count(report, "truth_component_count", context)
    if truth_component_count > fault_voxel_count:
        raise ValueError(
            f"{context}.truth_component_count exceeds trial truth_evidence.fault_voxel_count"
        )
    truth_cell_count = sum(
        _count(item, "truth_cell_count", f"{context}.truth_components[{index}]")
        for index, item in enumerate(truth_components)
    )
    if truth_cell_count != fault_voxel_count:
        raise ValueError(
            f"{context} truth component cell count does not match "
            "trial truth_evidence.fault_voxel_count"
        )
    covered_cell_count = sum(
        _count(item, "covered_cell_count", f"{context}.truth_components[{index}]")
        for index, item in enumerate(truth_components)
    )
    if covered_cell_count != intersection_count:
        raise ValueError(
            f"{context} covered component cell count does not match "
            "quality.skin.buffered_overlap_radius2.intersection_count"
        )


def validate_overlap_algebra(report: Mapping[str, Any], shape: Sequence[int], context: str) -> None:
    """Validate exact and buffered overlap counts and derived ratios."""

    candidate_count = _count(report, "candidate_count", context)
    truth_count = _count(report, "truth_count", context)
    intersection_count = _count(report, "intersection_count", context)
    union_count = _count(report, "union_count", context)
    candidate_buffer_count = _count(report, "candidate_in_truth_buffer_count", context)
    truth_buffer_count = _count(report, "truth_in_candidate_buffer_count", context)
    radius = _number(report, "radius", context)
    voxel_count = volume_voxel_count(shape)
    maximum_distance = volume_diagonal(shape)

    for name, value in (
        ("candidate_count", candidate_count),
        ("truth_count", truth_count),
        ("intersection_count", intersection_count),
        ("union_count", union_count),
        ("candidate_in_truth_buffer_count", candidate_buffer_count),
        ("truth_in_candidate_buffer_count", truth_buffer_count),
    ):
        if value > voxel_count:
            raise ValueError(f"{context}.{name} exceeds volume voxel count")
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
    if candidate_count != 0 and truth_count != 0:
        if 0.0 <= radius < 1.0 and (
            candidate_buffer_count != intersection_count or truth_buffer_count != intersection_count
        ):
            if radius == 0.0:
                raise ValueError(
                    f"{context} radius-zero buffered overlap counts must equal intersection_count"
                )
            raise ValueError(
                f"{context} fractional-radius buffered overlap counts must equal intersection_count"
            )
        if radius >= maximum_distance and (
            candidate_buffer_count != candidate_count or truth_buffer_count != truth_count
        ):
            raise ValueError(
                f"{context} full-volume buffered overlap counts must equal their source counts"
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
    voxel_count = volume_voxel_count(shape)
    maximum_distance = volume_diagonal(shape)
    for name, value in (("candidate_count", candidate_count), ("truth_count", truth_count)):
        if value > voxel_count:
            raise ValueError(f"{context}.{name} exceeds volume voxel count")
    values = {name: _number(report, name, context) for name in _DISTANCE_SUMMARY_NAMES}
    for name, value in values.items():
        if value < 0.0:
            raise ValueError(f"{context}.{name} must be non-negative")
        if value > maximum_distance and not derived_scalars_close(value, maximum_distance):
            raise ValueError(f"{context}.{name} exceeds the volume diagonal")
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
        expected = maximum_distance
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


def validate_edge_false_positive_algebra(
    report: Mapping[str, Any], shape: Sequence[int], context: str
) -> None:
    """Validate edge count hierarchy and all derived fractions."""

    candidate_count = _count(report, "candidate_count", context)
    edge_candidate_count = _count(report, "edge_candidate_count", context)
    false_positive_count = _count(report, "edge_false_positive_count", context)
    voxel_count = volume_voxel_count(shape)
    for name, value in (
        ("candidate_count", candidate_count),
        ("edge_candidate_count", edge_candidate_count),
        ("edge_false_positive_count", false_positive_count),
    ):
        if value > voxel_count:
            raise ValueError(f"{context}.{name} exceeds volume voxel count")
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


def _dominant_incidence(ids: Sequence[int], counts: Sequence[int]) -> tuple[int | None, int]:
    if not ids:
        return None, 0
    dominant_id, dominant_count = min(
        zip(ids, counts, strict=True),
        key=lambda item: (-item[1], item[0]),
    )
    return dominant_id, dominant_count


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


def volume_diagonal(shape: Sequence[int]) -> float:
    """Return the largest voxel-index distance within a positive 3D shape."""

    if len(shape) != 3:
        raise ValueError("shape must contain exactly three positive integers")
    dimensions = tuple(_positive_dimension(value, "volume") for value in shape)
    return sqrt(fsum((value - 1.0) ** 2 for value in dimensions))


def volume_voxel_count(shape: Sequence[int]) -> int:
    """Return the exact voxel capacity of a positive 3D shape."""

    if len(shape) != 3:
        raise ValueError("shape must contain exactly three positive integers")
    n3, n2, n1 = (_positive_dimension(value, "volume") for value in shape)
    return n3 * n2 * n1


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
    "validate_component_topology_evidence",
    "validate_edge_false_positive_algebra",
    "validate_downstream_quality_scalar_algebra",
    "validate_orientation_algebra",
    "validate_overlap_algebra",
    "validate_quality_scalar_algebra",
    "validate_scanner_quality_scalar_algebra",
    "validate_selection_cardinality",
    "validate_skin_report_topology_algebra",
    "validate_skin_topology_algebra",
    "validate_surface_distance_algebra",
    "volume_diagonal",
    "volume_voxel_count",
]
