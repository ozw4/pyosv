"""Voting, thinning, skinning, and metric orchestration for synthetic quality."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.evaluation.synthetic_quality.config import (
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
    _validate_nonnegative_finite_scalar,
)
from pyosv.evaluation.synthetic_quality.diagnostics import (
    _run_voter_thinning_diagnostic,
)
from pyosv.evaluation.synthetic_quality.models import (
    OrientationField3D,
    PipelineArtifacts,
    PipelineEvaluation,
    PipelineStageTrace3D,
    SkinningResult3D,
    ThinningResult3D,
    VotingResult3D,
)
from pyosv.evaluation.synthetic_quality.stage_cache import (
    AttributeStageKey,
    PipelineStageCache,
    PrimarySkinningStageKey,
    PrimarySkinningStageResult,
    SeedStageKey,
    SeedStageResult,
    ThinningStageKey,
    ThinningStageResult,
    VotingStageKey,
    VotingStageResult,
    diagnostic_items,
)
from pyosv.evaluation.synthetic_quality.variants import VariantSpec, effective_thin_mode
from pyosv.experimental.boundary_seed_selection import select_boundary_seed_retention_v1
from pyosv.experimental.boundary_skinning import (
    apply_boundary_skinner_fallback,
    find_synthetic_skins,
)
from pyosv.experimental.boundary_thinning import (
    FVT_RECENTER_MAX_SHIFT,
    apply_boundary_edge_thin_v1,
    fvt_recenter_target_distance_diagnostics,
    recenter_edge_fvt_to_target,
)
from pyosv.experimental.skin_diagnostics import add_primary_skin_diagnostics
from pyosv.synthetic3d import Synthetic3DCase
from pyosv.synthetic_metrics import (
    buffered_surface_overlap,
    edge_false_positive_ratio,
    masked_orientation_error,
    skin_mask_from_skins,
    skin_topology_metrics,
    skin_truth_metrics,
    surface_distance_metrics,
    top_positive_truth_count_mask,
    top_truth_count_mask,
)
from pyosv.voting3d import OptimalSurfaceVoter

EDGE_FALSE_POSITIVE_MARGIN = quality_metrics.EDGE_FALSE_POSITIVE_MARGIN
FORMAT_VERSION = 1
NONZERO_EPSILON = quality_metrics.NONZERO_EPSILON
THIN_HYBRID_ORIENTATION_GRADIENT_THRESHOLD = 8.0
THIN_HYBRID_V2_EDGE_MARGIN = 2
THIN_PLATEAU_TOLERANCE = 1.0e-6

# Case-local cache DAG (variant-specific work remains outside cached nodes):
# attributes -> seeds -> voting(fv/vp/vt) -> base thinning(fvt)
#   -> post-thinning -> primary skinning/min-size -> fallback -> metrics -> artifacts.
# Primary snapshots are cloned before diagnostics and fallback so mutable skins,
# cell links, and diagnostics never cross a variant boundary.


def _positive_candidate_mask(values: np.ndarray) -> np.ndarray:
    return np.asarray(values) > np.float32(NONZERO_EPSILON)


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    finite = np.isfinite(values)
    finite_values = values[finite].astype(np.float64, copy=False)
    minimum = float(np.min(finite_values)) if finite_values.size else float("nan")
    maximum = float(np.max(finite_values)) if finite_values.size else float("nan")
    mean = float(np.mean(finite_values)) if finite_values.size else float("nan")
    return {
        "shape": [int(size) for size in values.shape],
        "finite_count": int(np.count_nonzero(finite)),
        "finite_fraction": (float(np.count_nonzero(finite) / values.size) if values.size else 0.0),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "nonzero_fraction": (
            float(np.count_nonzero(np.abs(values) > NONZERO_EPSILON) / values.size)
            if values.size
            else 0.0
        ),
    }


def _skin_cell_json(cell: Any) -> dict[str, float | int]:
    return {
        "x1": float(cell.x1),
        "x2": float(cell.x2),
        "x3": float(cell.x3),
        "i1": int(cell.i1),
        "i2": int(cell.i2),
        "i3": int(cell.i3),
        "fl": float(cell.fl),
        "fp": float(cell.fp),
        "ft": float(cell.ft),
    }


def _skins_json_payload(skins: Sequence[Any]) -> dict[str, Any]:
    serialized = []
    for index, skin in enumerate(skins):
        cells = sorted(skin, key=lambda cell: (int(cell.i3), int(cell.i2), int(cell.i1)))
        serialized.append(
            {
                "skin_index": int(index),
                "cell_count": int(len(cells)),
                "cells": [_skin_cell_json(cell) for cell in cells],
            }
        )
    return {
        "format_version": FORMAT_VERSION,
        "skinning_enabled": True,
        "skin_count": int(len(serialized)),
        "skins": serialized,
    }


def _disabled_skins_json_payload() -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "skinning_enabled": False,
        "skin_count": 0,
        "skins": [],
    }


def _normalize_report_skin_metric_keys(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return quality_metrics.normalize_report_skin_metric_keys(metrics)


def run_voting_from_attributes(
    case: Synthetic3DCase,
    *,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
    voting_config: SyntheticVotingConfig,
    truth_metric_config: SyntheticTruthMetricConfig,
    skinning_config: SyntheticSkinningConfig,
    variant_spec: VariantSpec,
    include_thinning_diagnostic: bool = False,
    capture_stage_trace: bool = False,
    scanner_target_positive_mask: np.ndarray | None = None,
    fvt_recenter_target: np.ndarray | None = None,
    fvt_recenter_target_source: str | None = None,
    thinning_diagnostic_runner: Callable[..., Any] = _run_voter_thinning_diagnostic,
    recenter_distance_diagnostic_runner: Callable[
        ..., dict[str, float | None]
    ] = fvt_recenter_target_distance_diagnostics,
    stage_cache: PipelineStageCache | None = None,
    attribute_stage_key: AttributeStageKey | None = None,
) -> PipelineEvaluation:
    OrientationField3D(ft=ft, pt=pt, tt=tt)
    if stage_cache is not None:
        stage_cache.bind_case(case)
    voter = OptimalSurfaceVoter(
        ru=voting_config.ru,
        rv=voting_config.rv,
        rw=voting_config.rw,
    )
    voter.set_attribute_smoothing(voting_config.attribute_smoothing)
    voting_patch = variant_spec.voting
    surface_support_min_fraction = (
        voting_config.surface_support_min_fraction
        if voting_patch.support_min_fraction is None
        else voting_patch.support_min_fraction
    )
    surface_support_exponent = (
        voting_config.surface_support_exponent
        if voting_patch.support_exponent is None
        else voting_patch.support_exponent
    )
    voter.set_surface_support_policy(
        min_fraction=surface_support_min_fraction,
        exponent=surface_support_exponent,
    )
    if voting_patch.boundary_policy is not None:
        voter.set_surface_voting_boundary_policy(voting_patch.boundary_policy)
    if voting_patch.orientation_smoothing is not None:
        voter.set_surface_orientation_smoothing(voting_patch.orientation_smoothing)
    if voting_patch.final_normalization_smoothing is not None:
        voter.set_final_normalization_smoothing(voting_patch.final_normalization_smoothing)
    cache_enabled = stage_cache is not None and attribute_stage_key is not None
    if (
        variant_spec.seed_policy == "boundary_seed_retention_v1"
        and fvt_recenter_target is not None
        and fvt_recenter_target is not ft
    ):
        # A caller-provided target needs its own semantic token. The report
        # runners use the attribute likelihood itself; unknown combinations
        # conservatively bypass both dependent stages.
        cache_enabled = False
    primary_skinning_cache_enabled = cache_enabled and not (
        variant_spec.post_thinning_policy != "none"
        and fvt_recenter_target is not None
        and fvt_recenter_target is not ft
    )
    boundary_target_source = None
    boundary_edge_margin = None
    if variant_spec.seed_policy == "boundary_seed_retention_v1":
        boundary_target_source = (
            "ft_input" if fvt_recenter_target_source is None else fvt_recenter_target_source
        )
        boundary_edge_margin = EDGE_FALSE_POSITIVE_MARGIN
    seed_key = (
        SeedStageKey(
            attributes=attribute_stage_key,
            seed_policy=variant_spec.seed_policy,
            seed_distance=int(voting_config.seed_distance),
            seed_threshold=float(voting_config.seed_threshold),
            ru=int(voting_config.ru),
            rv=int(voting_config.rv),
            rw=int(voting_config.rw),
            boundary_target_source=boundary_target_source,
            boundary_edge_margin=boundary_edge_margin,
        )
        if attribute_stage_key is not None
        else None
    )
    cached_seed = (
        stage_cache.get_seed(seed_key)
        if cache_enabled and stage_cache is not None and seed_key is not None
        else None
    )
    if cached_seed is not None:
        seeds = cached_seed.seeds
        boundary_seed_retention_diagnostic = cached_seed.diagnostics()
    elif variant_spec.seed_policy == "boundary_seed_retention_v1":
        boundary_target = ft if fvt_recenter_target is None else fvt_recenter_target
        seed_result = select_boundary_seed_retention_v1(
            voting_config=voting_config,
            ft=ft,
            pt=pt,
            tt=tt,
            target=boundary_target,
            target_source=boundary_target_source,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        )
        seeds = tuple(seed_result.selected_seeds)
        boundary_seed_retention_diagnostic = dict(seed_result.diagnostics)
    else:
        seeds = tuple(
            voter.pick_seeds(
                d=voting_config.seed_distance,
                fm=voting_config.seed_threshold,
                ft=ft,
                pt=pt,
                tt=tt,
            )
        )
        boundary_seed_retention_diagnostic = None
    if cached_seed is None and cache_enabled and stage_cache is not None and seed_key is not None:
        stage_cache.put_seed(
            seed_key,
            SeedStageResult(
                seeds=tuple(seeds),
                diagnostic_items=(
                    None
                    if boundary_seed_retention_diagnostic is None
                    else diagnostic_items(boundary_seed_retention_diagnostic)
                ),
            ),
        )
    seed_candidate_mask: np.ndarray | None = None
    seed_selected_mask: np.ndarray | None = None
    if capture_stage_trace:
        seed_candidate_mask = np.asarray(ft) > np.float32(voting_config.seed_threshold)
        seed_selected_mask = np.zeros(seed_candidate_mask.shape, dtype=bool)
        for seed in seeds:
            seed_selected_mask[seed.i3, seed.i2, seed.i1] = True
    voting_key = (
        VotingStageKey(
            seed=seed_key,
            ru=int(voter.ru),
            rv=int(voter.rv),
            rw=int(voter.rw),
            bstrain1=int(voter.bstrain1),
            bstrain2=int(voter.bstrain2),
            attribute_smoothing=int(voter.attribute_smoothing),
            surface_smoothing1=float(voter.surface_smoothing1),
            surface_smoothing2=float(voter.surface_smoothing2),
            boundary_policy=voter.surface_voting_boundary_policy,
            support_min_fraction=float(voter.surface_support_min_fraction),
            support_exponent=float(voter.surface_support_exponent),
            orientation_smoothing=float(voter.surface_orientation_smoothing),
            orientation_backend=voter.surface_orientation_backend,
            final_normalization_smoothing=float(voter.final_normalization_smoothing),
        )
        if seed_key is not None
        else None
    )
    cached_voting = (
        stage_cache.get_voting(voting_key)
        if cache_enabled and stage_cache is not None and voting_key is not None
        else None
    )
    if cached_voting is None:
        fv, vp, vt = voter.apply_voting_from_seeds(
            seeds,
            ft=ft,
            pt=pt,
            tt=tt,
        )
        surface_voting_diagnostic_summary = voter.surface_voting_diagnostic_summary()
        if cache_enabled and stage_cache is not None and voting_key is not None:
            voting_result = VotingStageResult(
                fv=fv,
                vp=vp,
                vt=vt,
                diagnostic_items=diagnostic_items(surface_voting_diagnostic_summary),
            )
            stage_cache.put_voting(voting_key, voting_result)
    else:
        fv, vp, vt = cached_voting.fv, cached_voting.vp, cached_voting.vt
        surface_voting_diagnostic_summary = cached_voting.diagnostics()
    VotingResult3D(
        fv=fv,
        vp=vp,
        vt=vt,
        diagnostics=surface_voting_diagnostic_summary,
    )
    thin_mode = effective_thin_mode(variant_spec, voting_config)
    plateau_tie_breaker = ft if thin_mode in {"hybrid_v2", "normal_plateau"} else None
    thinning_key = (
        ThinningStageKey(
            voting=voting_key,
            thin_mode=thin_mode,
            reference_sigma=float(voting_config.reference_thin_sigma),
            hybrid_orientation_gradient_threshold=(THIN_HYBRID_ORIENTATION_GRADIENT_THRESHOLD),
            hybrid_v2_edge_margin=THIN_HYBRID_V2_EDGE_MARGIN,
            orientation_source="voting_vp_vt",
            tie_break_policy=("attribute_ft" if plateau_tie_breaker is not None else "voting_fv"),
            plateau_tolerance=THIN_PLATEAU_TOLERANCE,
        )
        if voting_key is not None
        else None
    )
    cached_thinning = (
        stage_cache.get_thinning(thinning_key)
        if cache_enabled and stage_cache is not None and thinning_key is not None
        else None
    )
    if cached_thinning is None:
        fvt = voter.thin(
            fv,
            vp,
            vt,
            mode=thin_mode,
            reference_sigma=voting_config.reference_thin_sigma,
            hybrid_orientation_gradient_threshold=(THIN_HYBRID_ORIENTATION_GRADIENT_THRESHOLD),
            hybrid_v2_edge_margin=THIN_HYBRID_V2_EDGE_MARGIN,
            plateau_tie_breaker=plateau_tie_breaker,
            plateau_tolerance=THIN_PLATEAU_TOLERANCE,
        )
        if cache_enabled and stage_cache is not None and thinning_key is not None:
            thinning_result = ThinningStageResult(fvt=fvt)
            stage_cache.put_thinning(thinning_key, thinning_result)
            fvt = thinning_result.fvt
    else:
        fvt = cached_thinning.fvt
    fvt_recenter_diagnostic: dict[str, Any] | None = None
    fvt_recenter_before_positive: np.ndarray | None = None
    boundary_edge_thin_diagnostic: dict[str, Any] | None = None
    if variant_spec.post_thinning_policy == "recenter_scanner_target":
        recenter_target = ft if fvt_recenter_target is None else fvt_recenter_target
        recenter_target_source = (
            "ft_input" if fvt_recenter_target_source is None else fvt_recenter_target_source
        )
        fvt_recenter_before_positive = _positive_candidate_mask(fvt)
        recenter_result = recenter_edge_fvt_to_target(
            fvt,
            vp,
            vt,
            target=recenter_target,
            target_source=recenter_target_source,
            max_shift=FVT_RECENTER_MAX_SHIFT,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        )
        fvt = recenter_result.output
        fvt_recenter_diagnostic = recenter_result.diagnostics
    if variant_spec.post_thinning_policy == "boundary_edge_thin_v1":
        boundary_target = ft if fvt_recenter_target is None else fvt_recenter_target
        boundary_target_source = (
            "ft_input" if fvt_recenter_target_source is None else fvt_recenter_target_source
        )
        boundary_thin_result = apply_boundary_edge_thin_v1(
            fvt,
            fv,
            vp,
            vt,
            voter=voter,
            target=boundary_target,
            target_source=boundary_target_source,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        )
        fvt = boundary_thin_result.output
        boundary_edge_thin_diagnostic = boundary_thin_result.diagnostics

    ThinningResult3D(
        fvt=fvt,
        diagnostics={
            "recenter": fvt_recenter_diagnostic,
            "boundary_edge_thin": boundary_edge_thin_diagnostic,
        },
    )

    truth_surface_half_width = _validate_nonnegative_finite_scalar(
        truth_metric_config.truth_surface_half_width,
        "truth_surface_half_width",
    )
    buffer_radius = _validate_nonnegative_finite_scalar(
        truth_metric_config.buffer_radius,
        "buffer_radius",
    )
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(truth_surface_half_width)
    truth_fault_mask = np.asarray(case.truth_fault_mask, dtype=bool)
    fv_top_truth_count = top_truth_count_mask(fv, truth_surface_mask)
    fvt_top_truth_count = top_truth_count_mask(fvt, truth_surface_mask)
    fv_positive_top_truth_count = top_positive_truth_count_mask(fv, truth_surface_mask)
    fvt_positive_top_truth_count = top_positive_truth_count_mask(fvt, truth_surface_mask)
    edge_false_positive_metrics = {
        "fv_top_truth_count": edge_false_positive_ratio(
            fv_top_truth_count,
            truth_surface_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        ),
        "fvt_top_truth_count": edge_false_positive_ratio(
            fvt_top_truth_count,
            truth_surface_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        ),
        "fv_positive_top_truth_count": edge_false_positive_ratio(
            fv_positive_top_truth_count,
            truth_surface_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        ),
        "fvt_positive_top_truth_count": edge_false_positive_ratio(
            fvt_positive_top_truth_count,
            truth_surface_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        ),
    }
    report = {
        "config": {
            "skinning": skinning_config.as_report_dict(),
        },
        "skinning": {"enabled": skinning_config.enabled},
        "pyosv": {
            "fv": _array_summary(fv),
            "fvt": _array_summary(fvt),
            "voting": {
                "surface_voting_boundary_policy": voter.surface_voting_boundary_policy,
                "surface_support_min_fraction": float(surface_support_min_fraction),
                "surface_support_exponent": float(surface_support_exponent),
                "diagnostic_summary": surface_voting_diagnostic_summary,
            },
        },
        "quality": {
            "fv_top_truth_count": {
                "buffered_overlap_radius2": buffered_surface_overlap(
                    fv_top_truth_count,
                    truth_fault_mask,
                    radius=buffer_radius,
                ),
                "surface_distance": surface_distance_metrics(
                    fv_top_truth_count,
                    truth_surface_mask,
                ),
                "orientation_error": masked_orientation_error(
                    vp,
                    vt,
                    case.truth_strike,
                    case.truth_dip,
                    fv_top_truth_count,
                ),
            },
            "fvt_top_truth_count": {
                "buffered_overlap_radius2": buffered_surface_overlap(
                    fvt_top_truth_count,
                    truth_fault_mask,
                    radius=buffer_radius,
                ),
                "surface_distance": surface_distance_metrics(
                    fvt_top_truth_count,
                    truth_surface_mask,
                ),
                "orientation_error": masked_orientation_error(
                    vp,
                    vt,
                    case.truth_strike,
                    case.truth_dip,
                    fvt_top_truth_count,
                ),
            },
            "fv_positive_top_truth_count": {
                "buffered_overlap_radius2": buffered_surface_overlap(
                    fv_positive_top_truth_count,
                    truth_fault_mask,
                    radius=buffer_radius,
                ),
                "surface_distance": surface_distance_metrics(
                    fv_positive_top_truth_count,
                    truth_surface_mask,
                ),
                "orientation_error": masked_orientation_error(
                    vp,
                    vt,
                    case.truth_strike,
                    case.truth_dip,
                    fv_positive_top_truth_count,
                ),
            },
            "fvt_positive_top_truth_count": {
                "buffered_overlap_radius2": buffered_surface_overlap(
                    fvt_positive_top_truth_count,
                    truth_fault_mask,
                    radius=buffer_radius,
                ),
                "surface_distance": surface_distance_metrics(
                    fvt_positive_top_truth_count,
                    truth_surface_mask,
                ),
                "orientation_error": masked_orientation_error(
                    vp,
                    vt,
                    case.truth_strike,
                    case.truth_dip,
                    fvt_positive_top_truth_count,
                ),
            },
            "edge_false_positive": edge_false_positive_metrics,
        },
    }
    if fvt_recenter_diagnostic is not None:
        fvt_recenter_diagnostic.update(
            recenter_distance_diagnostic_runner(
                before=fvt_recenter_before_positive,
                after=_positive_candidate_mask(fvt),
                target=_positive_candidate_mask(recenter_target),
            )
        )
        report["fvt_recenter"] = fvt_recenter_diagnostic
    if boundary_edge_thin_diagnostic is not None:
        report["boundary_edge_thin"] = boundary_edge_thin_diagnostic
    if boundary_seed_retention_diagnostic is not None:
        report["boundary_seed_retention"] = boundary_seed_retention_diagnostic
    diagnostic_volumes: dict[str, np.ndarray] = {}
    if include_thinning_diagnostic:
        thinning_diagnostic, diagnostic_volumes = thinning_diagnostic_runner(
            case=case,
            voter=voter,
            fv=fv,
            vp=vp,
            vt=vt,
            reference_sigma=voting_config.reference_thin_sigma,
            truth_metric_config=truth_metric_config,
            skinning_config=skinning_config,
        )
        report["thinning_diagnostic"] = thinning_diagnostic
    if skinning_config.enabled:
        post_thinning_target_source = None
        if variant_spec.post_thinning_policy != "none":
            post_thinning_target_source = (
                "ft_input" if fvt_recenter_target_source is None else fvt_recenter_target_source
            )
        primary_skinning_key = (
            PrimarySkinningStageKey(
                thinning=thinning_key,
                post_thinning_policy=variant_spec.post_thinning_policy,
                post_thinning_target_source=post_thinning_target_source,
                post_thinning_max_shift=(
                    FVT_RECENTER_MAX_SHIFT
                    if variant_spec.post_thinning_policy == "recenter_scanner_target"
                    else None
                ),
                post_thinning_edge_margin=(
                    EDGE_FALSE_POSITIVE_MARGIN
                    if variant_spec.post_thinning_policy != "none"
                    else None
                ),
                method=skinning_config.method,
                growth_source=skinning_config.growth_source,
                min_likelihood=skinning_config.min_likelihood,
                min_skin_size=skinning_config.min_skin_size,
                d=skinning_config.d,
                ru=skinning_config.ru,
                rv=skinning_config.rv,
                rw=skinning_config.rw,
                max_steps=skinning_config.max_steps,
                du=skinning_config.du,
                max_delta_strike=skinning_config.max_delta_strike,
                reskin=skinning_config.reskin,
                accepted_occupancy_radius=skinning_config.accepted_occupancy_radius,
                small_skin_size=skinning_config.small_skin_size,
            )
            if thinning_key is not None
            else None
        )
        primary_result = (
            stage_cache.get_primary_skinning(primary_skinning_key)
            if primary_skinning_cache_enabled
            and stage_cache is not None
            and primary_skinning_key is not None
            else None
        )
        if primary_result is None:
            primary_diagnostics: dict[str, Any] = {}
            primary_skins = find_synthetic_skins(
                fv,
                fvt,
                vp,
                vt,
                skinning_config=skinning_config,
                diagnostics=primary_diagnostics,
            )
            if (
                primary_skinning_cache_enabled
                and stage_cache is not None
                and primary_skinning_key is not None
            ):
                primary_result = PrimarySkinningStageResult.from_skins(
                    primary_skins, primary_diagnostics
                )
                stage_cache.put_primary_skinning(primary_skinning_key, primary_result)
        if primary_result is None:
            skins = primary_skins
            skin_diagnostics = primary_diagnostics
        else:
            skins, skin_diagnostics = primary_result.clone()
        primary_skin_mask = skin_mask_from_skins(skins, case.shape) if capture_stage_trace else None
        add_primary_skin_diagnostics(
            skin_diagnostics,
            skins,
            shape=case.shape,
            fvt_positive_candidate_count=int(np.count_nonzero(fvt_positive_top_truth_count)),
            small_skin_size=skinning_config.small_skin_size,
        )
        apply_boundary_skinner_fallback(
            skins,
            fvt,
            vp,
            vt,
            skinning_config=skinning_config,
            variant_spec=variant_spec,
            diagnostics=skin_diagnostics,
            scanner_target_positive_mask=scanner_target_positive_mask,
        )
        report["skinning"]["diagnostics"] = skin_diagnostics
        skin_metrics = skin_truth_metrics(
            skins,
            shape=case.shape,
            truth_fault_mask=truth_fault_mask,
            truth_surface_mask=truth_surface_mask,
            truth_strike=case.truth_strike,
            truth_dip=case.truth_dip,
            buffer_radius=buffer_radius,
            small_skin_size=skinning_config.small_skin_size,
            truth_fault_id=case.truth_fault_id,
        )
        skin_metrics = _normalize_report_skin_metric_keys(skin_metrics)
        report["pyosv"]["skins"] = skin_metrics["topology"]
        report["quality"]["skin"] = skin_metrics
        skin_mask = skin_mask_from_skins(skins, case.shape)
        report["quality"]["edge_false_positive"]["skin"] = edge_false_positive_ratio(
            skin_mask,
            truth_surface_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        )
        skins_output = _skins_json_payload(skins)
    else:
        skins = []
        skin_diagnostics = {}
        report["pyosv"]["skins"] = skin_topology_metrics(
            [],
            case.shape,
            small_skin_size=skinning_config.small_skin_size,
        )
        report["quality"]["skin"] = None
        skin_mask = np.zeros(case.shape, dtype=bool)
        skins_output = _disabled_skins_json_payload()

    SkinningResult3D(skins=skins, mask=skin_mask, diagnostics=skin_diagnostics)

    stage_trace = None
    if capture_stage_trace:
        empty_skin_mask = np.zeros(case.shape, dtype=bool)
        fallback_used = bool(skin_diagnostics.get("fallback_used", False))
        final_skin_mask = skin_mask if skinning_config.enabled else empty_skin_mask
        stage_trace = PipelineStageTrace3D(
            seed_candidate_mask=seed_candidate_mask,
            seed_selected_mask=seed_selected_mask,
            fv_positive_mask=_positive_candidate_mask(fv),
            fvt_positive_mask=_positive_candidate_mask(fvt),
            primary_skin_mask=(primary_skin_mask if skinning_config.enabled else empty_skin_mask),
            fallback_skin_mask=final_skin_mask if fallback_used else empty_skin_mask,
            final_skin_mask=final_skin_mask,
            skinning_enabled=skinning_config.enabled,
            fallback_used=fallback_used,
        )

    volumes = {
        "truth_fault_mask": case.truth_fault_mask.astype(np.float32),
        "truth_distance": case.truth_distance,
        "truth_strike": case.truth_strike,
        "truth_dip": case.truth_dip,
        "ft_oracle": case.ft_oracle,
        "pt_oracle": case.pt_oracle,
        "tt_oracle": case.tt_oracle,
        "fv_py": fv,
        "vp_py": vp,
        "vt_py": vt,
        "fvt_py": fvt,
        "skin_mask_py": skin_mask.astype(np.float32),
    }
    volumes.update(diagnostic_volumes)
    return PipelineEvaluation(
        report_payload=report,
        artifacts=PipelineArtifacts(
            volumes=volumes,
            skins_payload=skins_output,
            stage_trace=stage_trace,
        ),
    )
