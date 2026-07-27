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
    PipelineArtifacts,
    PipelineEvaluation,
    PipelineStageTrace3D,
    SkinningResult3D,
)
from pyosv.evaluation.synthetic_quality.stage_keys import (
    DEFAULT_PRIMARY_SKINNER_IDENTITY,
    build_thinning_scalar_evidence_key,
    build_voting_scalar_evidence_key,
)
from pyosv.evaluation.synthetic_quality.stage_cache import (
    AttributeStageKey,
    DownstreamScalarEvidence,
    DownstreamScalarEvidenceCache,
    PipelineStageCache,
    SCALAR_EVIDENCE_CONTRACT_VERSION,
)
from pyosv.evaluation.synthetic_quality.variants import VariantSpec
from pyosv.evaluation.workflow3d import VolumeVotingControls, execute_workflow3d
from pyosv.experimental.boundary_skinning import (
    apply_boundary_skinner_fallback,
    find_synthetic_skins,
)
from pyosv.experimental.boundary_thinning import (
    fvt_recenter_target_distance_diagnostics,
)
from pyosv.experimental.skin_diagnostics import primary_skin_degraded_reasons
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
from pyosv.voting3d import OptimalSurfaceVoter  # noqa: F401 - compatibility export

EDGE_FALSE_POSITIVE_MARGIN = quality_metrics.EDGE_FALSE_POSITIVE_MARGIN
FORMAT_VERSION = 1
NONZERO_EPSILON = quality_metrics.NONZERO_EPSILON

# Case-local cache DAG (variant-specific work remains outside cached nodes):
# attributes -> seeds -> voting(fv/vp/vt) -> base thinning(fvt)
#   -> post-thinning -> primary skinning/min-size -> fallback -> metrics -> artifacts.
# Scalar evidence is cached separately at voting and final post-thinning semantic keys.
# Primary snapshots are cloned before diagnostics and fallback so mutable skins,
# cell links, and diagnostics never cross a variant boundary.


def _positive_candidate_mask(values: np.ndarray) -> np.ndarray:
    return quality_metrics.positive_candidate_mask(values)


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
        "nonzero_fraction": quality_metrics.array_nonzero_fraction(values),
    }


def _build_downstream_scalar_evidence(
    values: np.ndarray,
    *,
    vp: np.ndarray,
    vt: np.ndarray,
    truth_fault_mask: np.ndarray,
    truth_surface_mask: np.ndarray,
    truth_strike: np.ndarray,
    truth_dip: np.ndarray,
    buffer_radius: float,
) -> DownstreamScalarEvidence:
    top = top_truth_count_mask(values, truth_surface_mask)
    positive_top = top_positive_truth_count_mask(values, truth_surface_mask)

    def quality(candidate_mask: np.ndarray) -> dict[str, Any]:
        return {
            "buffered_overlap_radius2": buffered_surface_overlap(
                candidate_mask,
                truth_fault_mask,
                radius=buffer_radius,
            ),
            "surface_distance": surface_distance_metrics(
                candidate_mask,
                truth_surface_mask,
            ),
            "orientation_error": masked_orientation_error(
                vp,
                vt,
                truth_strike,
                truth_dip,
                candidate_mask,
            ),
        }

    return DownstreamScalarEvidence(
        array_summary=_array_summary(values),
        top_truth_count=quality(top),
        positive_top_truth_count=quality(positive_top),
        edge_top_truth_count=edge_false_positive_ratio(
            top,
            truth_surface_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        ),
        edge_positive_top_truth_count=edge_false_positive_ratio(
            positive_top,
            truth_surface_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        ),
    )


def build_voting_scalar_evidence(
    fv: np.ndarray,
    *,
    vp: np.ndarray,
    vt: np.ndarray,
    truth_fault_mask: np.ndarray,
    truth_surface_mask: np.ndarray,
    truth_strike: np.ndarray,
    truth_dip: np.ndarray,
    buffer_radius: float,
) -> DownstreamScalarEvidence:
    """Build the canonical scalar report fragments for ``fv``."""

    return _build_downstream_scalar_evidence(
        fv,
        vp=vp,
        vt=vt,
        truth_fault_mask=truth_fault_mask,
        truth_surface_mask=truth_surface_mask,
        truth_strike=truth_strike,
        truth_dip=truth_dip,
        buffer_radius=buffer_radius,
    )


def build_thinning_scalar_evidence(
    fvt: np.ndarray,
    *,
    vp: np.ndarray,
    vt: np.ndarray,
    truth_fault_mask: np.ndarray,
    truth_surface_mask: np.ndarray,
    truth_strike: np.ndarray,
    truth_dip: np.ndarray,
    buffer_radius: float,
) -> DownstreamScalarEvidence:
    """Build the canonical scalar report fragments for final ``fvt``."""

    return _build_downstream_scalar_evidence(
        fvt,
        vp=vp,
        vt=vt,
        truth_fault_mask=truth_fault_mask,
        truth_surface_mask=truth_surface_mask,
        truth_strike=truth_strike,
        truth_dip=truth_dip,
        buffer_radius=buffer_radius,
    )


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


def _apply_truth_selected_primary_skin_diagnostics(
    diagnostics: dict[str, Any],
    *,
    fvt_positive_candidate_count: int,
) -> None:
    positive_count = int(fvt_positive_candidate_count)
    unique_cell_count = int(diagnostics["skin_primary_unique_cell_count"])
    largest_size = int(diagnostics["skin_primary_largest_size"])
    cell_coverage = float(unique_cell_count / positive_count) if positive_count else 0.0
    reasons = primary_skin_degraded_reasons(
        fvt_positive_candidate_count=positive_count,
        skin_count=int(diagnostics["skin_primary_count"]),
        cell_coverage_of_fvt_positive=cell_coverage,
        largest_fraction=float(diagnostics["skin_primary_largest_fraction"]),
        small_skin_cell_fraction=float(diagnostics["skin_primary_small_cell_fraction"]),
    )
    diagnostics.update(
        {
            "skin_primary_cell_coverage_of_fvt_positive": cell_coverage,
            "skin_primary_largest_coverage_of_fvt_positive": (
                float(largest_size / positive_count) if positive_count else 0.0
            ),
            "skin_primary_degraded_candidate": bool(reasons),
            "skin_primary_degraded_reasons": reasons,
        }
    )


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
    scalar_evidence_cache: DownstreamScalarEvidenceCache | None = None,
    attribute_stage_key: AttributeStageKey | None = None,
) -> PipelineEvaluation:
    if stage_cache is not None:
        stage_cache.bind_case(case)
    if scalar_evidence_cache is not None:
        scalar_evidence_cache.bind_case(case)

    workflow_result = execute_workflow3d(
        ft=ft,
        pt=pt,
        tt=tt,
        attribute_identity=attribute_stage_key,
        voting_settings=voting_config,
        voting_controls=VolumeVotingControls.resolve(voting_config, variant_spec),
        skinning_settings=skinning_config,
        variant_spec=variant_spec,
        stage_cache=stage_cache,
        scanner_target_positive_mask=scanner_target_positive_mask,
        fvt_recenter_target=fvt_recenter_target,
        fvt_recenter_target_source=fvt_recenter_target_source,
        recenter_distance_diagnostic_runner=recenter_distance_diagnostic_runner,
        primary_skinner=find_synthetic_skins,
        primary_skinner_identity=DEFAULT_PRIMARY_SKINNER_IDENTITY,
        boundary_fallback_runner=apply_boundary_skinner_fallback,
    )
    voting_controls = workflow_result.effective_settings.controls
    voter = workflow_result.voter
    fv = workflow_result.fv
    vp = workflow_result.vp
    vt = workflow_result.vt
    fvt = workflow_result.fvt
    skins = list(workflow_result.skins)
    skin_diagnostics = dict(workflow_result.diagnostics.skinning)
    surface_voting_diagnostic_summary = dict(workflow_result.diagnostics.voting)
    thinning_diagnostics = workflow_result.diagnostics.thinning
    fvt_recenter_diagnostic = thinning_diagnostics["recenter"]
    boundary_edge_thin_diagnostic = thinning_diagnostics["boundary_edge_thin"]
    boundary_seed_retention_diagnostic = (
        None if workflow_result.diagnostics.seed is None else dict(workflow_result.diagnostics.seed)
    )
    semantic_keys_safe = attribute_stage_key is not None
    if (
        variant_spec.seed_policy == "boundary_seed_retention_v1"
        and fvt_recenter_target is not None
        and fvt_recenter_target is not ft
    ):
        # The caller-provided target affects seed selection but has no semantic
        # fingerprint, so evidence derived from the resulting vote is unsafe to cache.
        semantic_keys_safe = False
    scalar_cache_enabled = scalar_evidence_cache is not None and semantic_keys_safe
    final_thinning_key_safe = semantic_keys_safe and not (
        variant_spec.post_thinning_policy != "none"
        and fvt_recenter_target is not None
        and fvt_recenter_target is not ft
    )
    voting_key = workflow_result.stage_keys.voting
    final_thinning_key = workflow_result.stage_keys.final_thinning
    seed_candidate_mask: np.ndarray | None = None
    seed_selected_mask: np.ndarray | None = None
    if capture_stage_trace:
        seed_candidate_mask = np.asarray(ft) > np.float32(voting_config.seed_threshold)
        seed_selected_mask = np.zeros(seed_candidate_mask.shape, dtype=bool)
        for i3, i2, i1 in workflow_result.seed_indices:
            seed_selected_mask[i3, i2, i1] = True

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
    voting_evidence_key = build_voting_scalar_evidence_key(
        case_id=case.case_id,
        case_token=id(case),
        shape=case.shape,
        voting_key=voting_key,
        truth_metric_config=truth_metric_config,
        contract_version=(
            scalar_evidence_cache.contract_version
            if scalar_evidence_cache is not None
            else SCALAR_EVIDENCE_CONTRACT_VERSION
        ),
    )

    def build_voting_evidence() -> DownstreamScalarEvidence:
        return build_voting_scalar_evidence(
            fv,
            vp=vp,
            vt=vt,
            truth_fault_mask=truth_fault_mask,
            truth_surface_mask=truth_surface_mask,
            truth_strike=case.truth_strike,
            truth_dip=case.truth_dip,
            buffer_radius=buffer_radius,
        )

    voting_evidence = (
        scalar_evidence_cache.get_or_build_voting(
            voting_evidence_key,
            build_voting_evidence,
        )
        if scalar_cache_enabled
        and scalar_evidence_cache is not None
        and voting_evidence_key is not None
        else build_voting_evidence()
    )

    thinning_evidence_key = build_thinning_scalar_evidence_key(
        case_id=case.case_id,
        case_token=id(case),
        shape=case.shape,
        final_thinning_key=final_thinning_key,
        truth_metric_config=truth_metric_config,
        contract_version=(
            scalar_evidence_cache.contract_version
            if scalar_evidence_cache is not None
            else SCALAR_EVIDENCE_CONTRACT_VERSION
        ),
    )

    def build_thinning_evidence() -> DownstreamScalarEvidence:
        return build_thinning_scalar_evidence(
            fvt,
            vp=vp,
            vt=vt,
            truth_fault_mask=truth_fault_mask,
            truth_surface_mask=truth_surface_mask,
            truth_strike=case.truth_strike,
            truth_dip=case.truth_dip,
            buffer_radius=buffer_radius,
        )

    thinning_evidence = (
        scalar_evidence_cache.get_or_build_thinning(
            thinning_evidence_key,
            build_thinning_evidence,
        )
        if scalar_cache_enabled
        and final_thinning_key_safe
        and scalar_evidence_cache is not None
        and thinning_evidence_key is not None
        else build_thinning_evidence()
    )
    if skinning_config.enabled:
        _apply_truth_selected_primary_skin_diagnostics(
            skin_diagnostics,
            fvt_positive_candidate_count=int(
                thinning_evidence.positive_top_truth_count["buffered_overlap_radius2"][
                    "candidate_count"
                ]
            ),
        )
    edge_false_positive_metrics = {
        "fv_top_truth_count": voting_evidence.edge_top_truth_count,
        "fvt_top_truth_count": thinning_evidence.edge_top_truth_count,
        "fv_positive_top_truth_count": voting_evidence.edge_positive_top_truth_count,
        "fvt_positive_top_truth_count": thinning_evidence.edge_positive_top_truth_count,
    }
    report = {
        "config": {
            "skinning": skinning_config.as_report_dict(),
        },
        "skinning": {"enabled": skinning_config.enabled},
        "pyosv": {
            "fv": voting_evidence.array_summary,
            "fvt": thinning_evidence.array_summary,
            "voting": {
                "surface_voting_boundary_policy": voter.surface_voting_boundary_policy,
                "surface_support_min_fraction": float(voting_controls.support_min_fraction),
                "surface_support_exponent": float(voting_controls.support_exponent),
                "diagnostic_summary": surface_voting_diagnostic_summary,
            },
        },
        "quality": {
            "fv_top_truth_count": voting_evidence.top_truth_count,
            "fvt_top_truth_count": thinning_evidence.top_truth_count,
            "fv_positive_top_truth_count": voting_evidence.positive_top_truth_count,
            "fvt_positive_top_truth_count": thinning_evidence.positive_top_truth_count,
            "edge_false_positive": edge_false_positive_metrics,
        },
    }
    if fvt_recenter_diagnostic is not None:
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
        primary_skin_mask = (
            workflow_result.skin.primary_mask.copy() if capture_stage_trace else None
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
