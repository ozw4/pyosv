"""Experimental boundary-skinning fallback orchestration."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.evaluation.synthetic_quality.config import SyntheticSkinningConfig
from pyosv.evaluation.synthetic_quality.variants import VariantSpec
from pyosv.experimental.boundary_thinning import dominant_fault_normal_array_axis
from pyosv.skinner import FaultSkinner, find_connected_component_skins
from pyosv.synthetic_metrics import skin_topology_metrics, surface_distance_metrics

from .skin_diagnostics import (
    primary_boundary_degraded_reasons,
    primary_skin_degraded_reasons,
)

SKIN_FALLBACK_FILTER_MAX_COMPONENTS = 3
SKIN_FALLBACK_FILTER_MIN_COMPONENT_SIZE_FLOOR = 8
SKIN_FALLBACK_FILTER_MIN_COMPONENT_FRACTION = 0.05
SKIN_FALLBACK_FILTER_MIN_COMPONENT_FRACTION_OF_LARGEST = 0.10
SKIN_FALLBACK_V5_MAX_SKIN_COUNT = 3
SKIN_FALLBACK_V5_MIN_COVERAGE_OF_FVT_POSITIVE = 0.75
SKIN_FALLBACK_V5_MAX_COVERAGE_OF_FVT_POSITIVE = 1.25
SKIN_FALLBACK_V5_MAX_SMALL_SKIN_CELL_FRACTION = 0.20
SKIN_FALLBACK_V5_MIN_LARGEST_SKIN_FRACTION = 0.50
SKIN_FALLBACK_V5_MAX_PRUNED_FRACTION = 0.60
NONZERO_EPSILON = quality_metrics.NONZERO_EPSILON
EDGE_MARGIN = quality_metrics.EDGE_FALSE_POSITIVE_MARGIN


def find_synthetic_skins(
    fv: np.ndarray,
    fvt: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    skinning_config: SyntheticSkinningConfig,
    diagnostics: dict[str, Any] | None = None,
) -> list[Any]:
    """Run the configured primary skinner; input arrays are not mutated."""

    kwargs: dict[str, Any] = {
        "method": skinning_config.method,
        "min_skin_size": skinning_config.min_skin_size,
    }
    if skinning_config.min_likelihood is not None:
        kwargs["min_likelihood"] = skinning_config.min_likelihood
    skinner = FaultSkinner(**kwargs)
    grow_volume = fvt if skinning_config.growth_source == "thinned" else fv
    return skinner.find_skins(
        grow_volume,
        vp,
        vt,
        min_likelihood=skinning_config.min_likelihood,
        ep=fvt,
        ft=grow_volume,
        pt=vp,
        tt=vt,
        d=skinning_config.d,
        ru=skinning_config.ru,
        rv=skinning_config.rv,
        rw=skinning_config.rw,
        max_steps=skinning_config.max_steps,
        du=skinning_config.du,
        max_delta_strike=skinning_config.max_delta_strike,
        reskin=skinning_config.reskin,
        accepted_occupancy_radius=skinning_config.accepted_occupancy_radius,
        diagnostics=diagnostics,
    )


def positive_mask_components(
    mask: np.ndarray, *, connectivity: str
) -> list[list[tuple[int, int, int]]]:
    """Return deterministically ordered connected components of a 3D mask."""

    unvisited = {tuple(int(value) for value in index) for index in np.argwhere(mask)}
    offsets = _connectivity_offsets(connectivity)
    components = []
    while unvisited:
        start = min(unvisited)
        queue = deque([start])
        unvisited.remove(start)
        component = []
        while queue:
            index = queue.popleft()
            component.append(index)
            for offset in offsets:
                neighbor = tuple(value + delta for value, delta in zip(index, offset))
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)
        component.sort()
        components.append(component)
    components.sort(key=lambda component: (-len(component), component[0]))
    return components


def _connectivity_offsets(connectivity: str) -> tuple[tuple[int, int, int], ...]:
    max_axis_steps = {"face": 1, "edge": 2, "corner": 3}[connectivity]
    return tuple(
        (d3, d2, d1)
        for d3 in (-1, 0, 1)
        for d2 in (-1, 0, 1)
        for d1 in (-1, 0, 1)
        if (d3 or d2 or d1) and abs(d3) + abs(d2) + abs(d1) <= max_axis_steps
    )


def filtered_fallback_min_component_size(candidate_cell_count: int) -> int:
    return max(
        SKIN_FALLBACK_FILTER_MIN_COMPONENT_SIZE_FLOOR,
        int(math.ceil(SKIN_FALLBACK_FILTER_MIN_COMPONENT_FRACTION * int(candidate_cell_count))),
    )


def filtered_fallback_components(
    components: Sequence[Sequence[tuple[int, int, int]]], *, candidate_cell_count: int
) -> list[list[tuple[int, int, int]]]:
    if not components or int(candidate_cell_count) <= 0:
        return []
    minimum = filtered_fallback_min_component_size(candidate_cell_count)
    largest_fraction_minimum = SKIN_FALLBACK_FILTER_MIN_COMPONENT_FRACTION_OF_LARGEST * len(
        components[0]
    )
    accepted = [
        list(component)
        for component in components
        if len(component) >= minimum and len(component) >= largest_fraction_minimum
    ]
    if not accepted:
        accepted = [list(components[0])]
    return accepted[:SKIN_FALLBACK_FILTER_MAX_COMPONENTS]


def mask_from_components(
    shape: tuple[int, ...], components: Sequence[Sequence[tuple[int, int, int]]]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for component in components:
        for index in component:
            mask[index] = True
    return mask


def fallback_component_diagnostics(
    fvt: np.ndarray,
    *,
    min_skin_size: int | None,
    small_component_size: int,
    connectivity: str,
    component_policy: str = "all",
) -> dict[str, int | float | str]:
    mask = quality_metrics.positive_candidate_mask(fvt)
    candidate_count = int(np.count_nonzero(mask))
    components = positive_mask_components(mask, connectivity=connectivity)
    sizes = [len(component) for component in components]
    if component_policy == "all":
        accepted = [
            component
            for component in components
            if min_skin_size is None or len(component) >= int(min_skin_size)
        ]
        filter_minimum, filter_fraction, filter_maximum = 0, 0.0, 0
    elif component_policy in {
        "degraded_primary_filtered",
        "degraded_primary_skeletonized",
        "degraded_primary_topology_guarded",
    }:
        accepted = filtered_fallback_components(components, candidate_cell_count=candidate_count)
        filter_minimum = filtered_fallback_min_component_size(candidate_count)
        filter_fraction = SKIN_FALLBACK_FILTER_MIN_COMPONENT_FRACTION_OF_LARGEST
        filter_maximum = SKIN_FALLBACK_FILTER_MAX_COMPONENTS
    else:
        raise ValueError(f"unknown fallback component policy: {component_policy}")
    accepted_sizes = [len(component) for component in accepted]
    largest = sizes[0] if sizes else 0
    top3 = int(sum(sizes[:3]))
    return {
        "skin_fallback_component_count": len(components),
        "skin_fallback_candidate_cell_count": candidate_count,
        "skin_fallback_largest_component_size": largest,
        "skin_fallback_largest_component_fraction": float(largest / candidate_count)
        if candidate_count
        else 0.0,
        "skin_fallback_top3_component_cell_count": top3,
        "skin_fallback_top3_component_fraction": float(top3 / candidate_count)
        if candidate_count
        else 0.0,
        "skin_fallback_small_component_count": sum(
            size < int(small_component_size) for size in sizes
        ),
        "skin_fallback_component_policy": component_policy,
        "skin_fallback_accepted_component_count": len(accepted_sizes),
        "skin_fallback_discarded_component_count": len(sizes) - len(accepted_sizes),
        "skin_fallback_accepted_component_cell_count": int(sum(accepted_sizes)),
        "skin_fallback_filter_min_component_size": filter_minimum,
        "skin_fallback_filter_min_component_fraction_of_largest": filter_fraction,
        "skin_fallback_filter_max_components": filter_maximum,
    }


def skeletonize_fallback_components(
    fvt: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    components: Sequence[Sequence[tuple[int, int, int]]],
    *,
    scanner_target_positive_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    fvt_array = np.asarray(fvt, dtype=np.float32)
    vp_array = np.asarray(vp, dtype=np.float32)
    vt_array = np.asarray(vt, dtype=np.float32)
    if fvt_array.shape != vp_array.shape or fvt_array.shape != vt_array.shape:
        raise ValueError("fvt, vp, and vt shapes must match")
    target = None
    if scanner_target_positive_mask is not None:
        target = np.asarray(scanner_target_positive_mask, dtype=bool)
        if target.shape != fvt_array.shape:
            raise ValueError("scanner_target_positive_mask shape must match fvt")
    retained: set[tuple[int, int, int]] = set()
    axis_counts = [0, 0, 0]
    raw_sizes, pruned_sizes = [], []
    for component in components:
        raw_sizes.append(len(component))
        groups: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
        for index in component:
            axis = dominant_fault_normal_array_axis(vp_array[index], vt_array[index])
            axis_counts[axis] += 1
            groups.setdefault(_line_key(index, axis), []).append(index)
        component_retained = set()
        for key, indices in groups.items():
            axis = key[0]
            run = []
            previous = None
            for index in sorted(indices, key=lambda item: item[axis]):
                if previous is not None and index[axis] != previous + 1:
                    component_retained.add(_select_run_sample(run, axis, fvt_array, target))
                    run = []
                run.append(index)
                previous = index[axis]
            if run:
                component_retained.add(_select_run_sample(run, axis, fvt_array, target))
        retained.update(component_retained)
        pruned_sizes.append(len(component_retained))
    mask = np.zeros(fvt_array.shape, dtype=bool)
    for index in retained:
        mask[index] = True
    raw_count, pruned_count = sum(raw_sizes), len(retained)
    return mask, {
        "skin_fallback_pruning_method": "fault_normal_line_collapse",
        "skin_fallback_raw_component_cell_count": raw_count,
        "skin_fallback_pruned_component_cell_count": pruned_count,
        "skin_fallback_pruned_fraction": float(pruned_count / raw_count) if raw_count else 0.0,
        "skin_fallback_largest_component_size_before_pruning": max(raw_sizes, default=0),
        "skin_fallback_largest_component_size_after_pruning": max(pruned_sizes, default=0),
        "skin_fallback_pruning_removed_cell_count": raw_count - pruned_count,
        "skin_fallback_skeletonization_axis_mode": _axis_mode(axis_counts),
    }


def _line_key(index: tuple[int, int, int], axis: int) -> tuple[int, int, int]:
    if axis == 0:
        return (axis, index[1], index[2])
    if axis == 1:
        return (axis, index[0], index[2])
    return (axis, index[0], index[1])


def _select_run_sample(
    run: Sequence[tuple[int, int, int]],
    axis: int,
    fvt: np.ndarray,
    target: np.ndarray | None,
) -> tuple[int, int, int]:
    if not run:
        raise ValueError("run must include at least one sample")
    maximum = max(float(fvt[index]) for index in run)
    tied = [index for index in run if float(fvt[index]) == maximum]
    if target is not None:
        target_maximum = max(int(target[index]) for index in tied)
        tied = [index for index in tied if int(target[index]) == target_maximum]
    center = 0.5 * (run[0][axis] + run[-1][axis])
    return min(tied, key=lambda index: (abs(index[axis] - center), index))


def _axis_mode(counts: Sequence[int]) -> str:
    if not counts or max(counts) == 0:
        return "none"
    labels = ("i3", "i2", "i1")
    maximum = max(counts)
    winners = [labels[index] for index, count in enumerate(counts) if count == maximum]
    return winners[0] if len(winners) == 1 else "mixed"


def skeletonized_fallback_boundary_trigger_sufficient(
    *,
    boundary_degraded_reasons: Sequence[str],
    scanner_target_positive_mask: np.ndarray | None,
) -> bool:
    return bool(scanner_target_positive_mask is not None) and any(
        reason in boundary_degraded_reasons
        for reason in (
            "fvt_far_from_scanner_target",
            "low_primary_coverage_with_edge_local_candidates",
        )
    )


def fallback_v5_guardrail_report(
    *, fallback_topology: Mapping[str, Any], fvt_positive_count: int, pruned_fraction: float
) -> dict[str, Any]:
    skin_count = int(fallback_topology["skin_count"])
    coverage = (
        float(int(fallback_topology["unique_cell_count"]) / int(fvt_positive_count))
        if int(fvt_positive_count) > 0
        else 0.0
    )
    small_fraction = float(fallback_topology["small_skin_cell_fraction"])
    largest_fraction = float(fallback_topology["largest_skin_fraction"])
    pruned_fraction = float(pruned_fraction)
    reasons = []
    if skin_count > SKIN_FALLBACK_V5_MAX_SKIN_COUNT:
        reasons.append("fallback_skin_count_exceeds_max")
    if not math.isfinite(coverage):
        reasons.append("coverage_of_fvt_positive_nonfinite")
    else:
        if coverage < SKIN_FALLBACK_V5_MIN_COVERAGE_OF_FVT_POSITIVE:
            reasons.append("coverage_of_fvt_positive_below_min")
        if coverage > SKIN_FALLBACK_V5_MAX_COVERAGE_OF_FVT_POSITIVE:
            reasons.append("coverage_of_fvt_positive_above_max")
    if (
        not math.isfinite(small_fraction)
        or small_fraction > SKIN_FALLBACK_V5_MAX_SMALL_SKIN_CELL_FRACTION
    ):
        reasons.append("small_skin_cell_fraction_exceeds_max")
    if (
        not math.isfinite(largest_fraction)
        or largest_fraction < SKIN_FALLBACK_V5_MIN_LARGEST_SKIN_FRACTION
    ):
        reasons.append("largest_skin_fraction_below_min")
    if not math.isfinite(pruned_fraction) or pruned_fraction > SKIN_FALLBACK_V5_MAX_PRUNED_FRACTION:
        reasons.append("pruned_fraction_exceeds_max")
    return {
        "enabled": True,
        "passed": not reasons,
        "reasons": reasons,
        "max_skin_count": SKIN_FALLBACK_V5_MAX_SKIN_COUNT,
        "fallback_skin_count": skin_count,
        "coverage_of_fvt_positive": coverage,
        "min_coverage_of_fvt_positive": SKIN_FALLBACK_V5_MIN_COVERAGE_OF_FVT_POSITIVE,
        "max_coverage_of_fvt_positive": SKIN_FALLBACK_V5_MAX_COVERAGE_OF_FVT_POSITIVE,
        "small_skin_cell_fraction": small_fraction,
        "max_small_skin_cell_fraction": SKIN_FALLBACK_V5_MAX_SMALL_SKIN_CELL_FRACTION,
        "largest_skin_fraction": largest_fraction,
        "min_largest_skin_fraction": SKIN_FALLBACK_V5_MIN_LARGEST_SKIN_FRACTION,
        "pruned_fraction": pruned_fraction,
        "max_pruned_fraction": SKIN_FALLBACK_V5_MAX_PRUNED_FRACTION,
    }


def _guardrail_defaults(enabled: bool) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "passed": False,
        "reasons": [],
        "max_skin_count": SKIN_FALLBACK_V5_MAX_SKIN_COUNT,
        "fallback_skin_count": 0,
        "coverage_of_fvt_positive": 0.0,
        "min_coverage_of_fvt_positive": SKIN_FALLBACK_V5_MIN_COVERAGE_OF_FVT_POSITIVE,
        "max_coverage_of_fvt_positive": SKIN_FALLBACK_V5_MAX_COVERAGE_OF_FVT_POSITIVE,
        "small_skin_cell_fraction": 0.0,
        "max_small_skin_cell_fraction": SKIN_FALLBACK_V5_MAX_SMALL_SKIN_CELL_FRACTION,
        "largest_skin_fraction": 0.0,
        "min_largest_skin_fraction": SKIN_FALLBACK_V5_MIN_LARGEST_SKIN_FRACTION,
        "pruned_fraction": 0.0,
        "max_pruned_fraction": SKIN_FALLBACK_V5_MAX_PRUNED_FRACTION,
    }


def fallback_degraded_reason_labels(reasons: Sequence[str]) -> list[str]:
    labels = {
        "empty_primary_skin": "empty_primary",
        "low_fvt_positive_coverage": "undercovered",
        "fragmented_primary_skins": "fragmented",
        "high_small_skin_cell_fraction": "small_skin_dominated",
    }
    return [labels.get(reason, reason) for reason in reasons]


def apply_boundary_skinner_fallback(
    skins: list[Any],
    fvt: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    skinning_config: SyntheticSkinningConfig,
    variant_spec: VariantSpec,
    diagnostics: dict[str, Any],
    scanner_target_positive_mask: np.ndarray | None = None,
) -> None:
    """Apply the resolved variant's fallback policy, replacing ``skins`` in place on success."""

    if not isinstance(variant_spec, VariantSpec):
        raise TypeError("variant_spec must be a VariantSpec")
    enabled = skinning_config.boundary_skinner_fallback
    policy = skinning_config.boundary_skinner_fallback_policy
    guarded = enabled and policy == "degraded_primary_topology_guarded"
    filtered_policies = {
        "degraded_primary_filtered",
        "degraded_primary_skeletonized",
        "degraded_primary_topology_guarded",
    }
    component_policy = policy if policy in filtered_policies else "all"
    component_diagnostics = fallback_component_diagnostics(
        fvt,
        min_skin_size=skinning_config.min_skin_size,
        small_component_size=skinning_config.small_skin_size,
        connectivity="edge",
        component_policy=component_policy,
    )
    positive_count = int(component_diagnostics["skin_fallback_candidate_cell_count"])
    primary_count = int(diagnostics.get("skin_primary_count", len(skins)))
    primary_cells = int(diagnostics.get("skin_primary_cell_count", sum(map(len, skins))))
    primary_unique = int(diagnostics.get("skin_primary_unique_cell_count", primary_cells))
    coverage_before = float(primary_unique / positive_count) if positive_count else 0.0
    degraded = primary_skin_degraded_reasons(
        fvt_positive_candidate_count=positive_count,
        skin_count=primary_count,
        cell_coverage_of_fvt_positive=coverage_before,
        largest_fraction=float(diagnostics.get("skin_primary_largest_fraction", 0.0)),
        small_skin_cell_fraction=float(diagnostics.get("skin_primary_small_cell_fraction", 0.0)),
    )
    positive_mask = quality_metrics.positive_candidate_mask(fvt)
    edge_fraction = quality_metrics.edge_candidate_fraction(positive_mask, edge_margin=EDGE_MARGIN)
    primary_edge_fraction = float(diagnostics.get("skin_primary_edge_shell_fraction", 0.0))
    target_edge_fraction = None
    target_distance_p95 = None
    if scanner_target_positive_mask is not None:
        target_mask = np.asarray(scanner_target_positive_mask, dtype=bool)
        if target_mask.shape != positive_mask.shape:
            raise ValueError("scanner_target_positive_mask shape must match fvt")
        target_edge_fraction = quality_metrics.edge_candidate_fraction(
            target_mask, edge_margin=EDGE_MARGIN
        )
        target_distance_p95 = surface_distance_metrics(positive_mask, target_mask)[
            "candidate_to_truth_p95"
        ]
    boundary_degraded = primary_boundary_degraded_reasons(
        generic_degraded=bool(degraded),
        fvt_positive_candidate_count=positive_count,
        cell_coverage_of_fvt_positive=coverage_before,
        fvt_positive_edge_shell_fraction=edge_fraction,
        primary_edge_shell_fraction=primary_edge_fraction,
        fvt_to_scanner_target_distance_p95=target_distance_p95,
    )
    diagnostics.update(
        {
            "fallback_enabled": enabled,
            "fallback_policy": policy if enabled else None,
            "fallback_used": False,
            "fallback_reason": None,
            "fallback_method": "connected_component_on_fvt" if enabled else None,
            "fallback_input": "fvt" if enabled else None,
            "fallback_skin_count": 0,
            "fallback_cell_count": 0,
            "fallback_triggered_by_degraded_primary": False,
            "fallback_degraded_reasons": [],
            "fallback_replaced_primary": False,
            "fallback_primary_skin_count": primary_count,
            "fallback_primary_cell_count": primary_cells,
            "fallback_candidate_count": positive_count,
            "fallback_coverage_before": coverage_before,
            "fallback_coverage_after": coverage_before,
            "skin_fvt_positive_edge_shell_fraction": edge_fraction,
            "skin_primary_edge_shell_fraction": primary_edge_fraction,
            "skin_scanner_target_positive_edge_shell_fraction": target_edge_fraction,
            "skin_fvt_to_scanner_target_distance_p95": target_distance_p95,
            "skin_primary_boundary_degraded_candidate": bool(boundary_degraded),
            "skin_primary_boundary_degraded_reasons": boundary_degraded,
            "skin_fallback_pruning_method": None,
            "skin_fallback_raw_component_cell_count": 0,
            "skin_fallback_pruned_component_cell_count": 0,
            "skin_fallback_pruned_fraction": 0.0,
            "skin_fallback_largest_component_size_before_pruning": 0,
            "skin_fallback_largest_component_size_after_pruning": 0,
            "skin_fallback_pruning_removed_cell_count": 0,
            "skin_fallback_skeletonization_axis_mode": None,
            "fallback_v5_guardrail": _guardrail_defaults(guarded),
            **component_diagnostics,
        }
    )
    if not enabled:
        return
    if positive_count == 0:
        diagnostics["fallback_reason"] = "empty_primary_skin_without_positive_fvt"
        return
    if guarded and primary_count == 0:
        diagnostics["fallback_degraded_reasons"] = degraded
        diagnostics["fallback_reason"] = "empty_primary_not_supported_by_topology_guarded"
        diagnostics["fallback_v5_guardrail"]["reasons"] = ["empty_primary_not_supported"]
        return
    if policy == "empty_primary":
        if skins:
            diagnostics["fallback_reason"] = "primary_skin_nonempty"
            return
        reason = "empty_primary_skin_with_positive_fvt"
    else:
        diagnostics["fallback_degraded_reasons"] = degraded
        if not degraded:
            diagnostics["fallback_reason"] = "primary_skin_healthy"
            return
        if not boundary_degraded:
            diagnostics["fallback_reason"] = "primary_boundary_degraded_not_detected"
            return
        if policy in {
            "degraded_primary_skeletonized",
            "degraded_primary_topology_guarded",
        } and not skeletonized_fallback_boundary_trigger_sufficient(
            boundary_degraded_reasons=boundary_degraded,
            scanner_target_positive_mask=scanner_target_positive_mask,
        ):
            diagnostics["fallback_reason"] = "primary_boundary_degraded_not_sufficient"
            return
        diagnostics["fallback_triggered_by_degraded_primary"] = True
        reason = "degraded_primary:" + ",".join(fallback_degraded_reason_labels(degraded))
    fallback_fvt = fvt
    fallback_minimum = skinning_config.min_skin_size
    if component_policy in filtered_policies:
        accepted = filtered_fallback_components(
            positive_mask_components(positive_mask, connectivity="edge"),
            candidate_cell_count=positive_count,
        )
        accepted_mask = mask_from_components(fvt.shape, accepted)
        if component_policy in {
            "degraded_primary_skeletonized",
            "degraded_primary_topology_guarded",
        }:
            accepted_mask, pruning = skeletonize_fallback_components(
                fvt,
                vp,
                vt,
                accepted,
                scanner_target_positive_mask=scanner_target_positive_mask,
            )
            diagnostics.update(pruning)
        fallback_fvt = np.where(accepted_mask, fvt, np.float32(0.0)).astype(np.float32)
        fallback_minimum = None
    fallback_skins = find_connected_component_skins(
        fallback_fvt,
        vp,
        vt,
        min_likelihood=NONZERO_EPSILON,
        min_skin_size=fallback_minimum,
        connectivity="edge",
    )
    if not fallback_skins:
        diagnostics["fallback_reason"] = "connected_component_fallback_empty"
        return
    topology = skin_topology_metrics(
        fallback_skins, fvt.shape, small_skin_size=skinning_config.small_skin_size
    )
    coverage_after = float(int(topology["unique_cell_count"]) / positive_count)
    if guarded:
        raw = int(diagnostics.get("skin_fallback_raw_component_cell_count", 0))
        guardrail = fallback_v5_guardrail_report(
            fallback_topology=topology,
            fvt_positive_count=positive_count,
            pruned_fraction=float(diagnostics.get("skin_fallback_pruning_removed_cell_count", 0))
            / raw
            if raw
            else 0.0,
        )
        diagnostics["fallback_v5_guardrail"] = guardrail
        if not guardrail["passed"]:
            diagnostics["fallback_reason"] = "fallback_v5_guardrail_failed"
            return
    replaced = bool(skins)
    skins[:] = fallback_skins
    diagnostics.update(
        {
            "fallback_used": True,
            "fallback_reason": reason,
            "fallback_skin_count": len(fallback_skins),
            "fallback_cell_count": sum(map(len, fallback_skins)),
            "fallback_replaced_primary": replaced,
            "fallback_coverage_after": coverage_after,
        }
    )
