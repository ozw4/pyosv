"""Opt-in diagnostics for synthetic-quality scanner and thinning pipelines."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from pyosv.cells import FaultCell
from pyosv.synthetic3d import Synthetic3DCase
from pyosv.synthetic_metrics import (
    buffered_surface_overlap,
    edge_false_positive_ratio,
    masked_orientation_error,
    surface_distance_metrics,
    top_positive_truth_count_mask,
    top_truth_count_mask,
)
from pyosv.voting3d import OptimalSurfaceVoter

from . import quality_metrics
from .config import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
    _validate_nonnegative_finite_scalar,
)
from .variants import VariantSpec, effective_thin_mode


BoundarySeedSelector = Callable[
    ...,
    tuple[list[FaultCell], list[FaultCell], dict[str, Any]],
]


def _scanner_downstream_diagnostics(
    *,
    case: Synthetic3DCase,
    scanner_config: SyntheticScannerConfig,
    voting_config: SyntheticVotingConfig,
    variant_spec: VariantSpec,
    report: Mapping[str, Any],
    scanner_volumes: Mapping[str, np.ndarray],
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    fvt: np.ndarray,
    truth_metric_config: SyntheticTruthMetricConfig,
) -> dict[str, Any]:
    scanner_ft_positive = quality_metrics.positive_candidate_mask(scanner_volumes["scanner_ft"])
    scanner_fet_positive = quality_metrics.positive_candidate_mask(scanner_volumes["scanner_fet"])
    fv_positive = quality_metrics.positive_candidate_mask(fv)
    fvt_positive = quality_metrics.positive_candidate_mask(fvt)
    scanner_ft_positive_count = quality_metrics.candidate_count(scanner_ft_positive)
    scanner_fet_positive_count = quality_metrics.candidate_count(scanner_fet_positive)
    fv_positive_count = quality_metrics.candidate_count(fv_positive)
    fvt_positive_count = quality_metrics.candidate_count(fvt_positive)
    fvt_to_scanner_ft_distance = surface_distance_metrics(fvt_positive, scanner_ft_positive)
    fvt_to_fv_distance = surface_distance_metrics(fvt_positive, fv_positive)
    buffer_radius = _validate_nonnegative_finite_scalar(
        truth_metric_config.buffer_radius, "buffer_radius"
    )
    voter_thin_mode = effective_thin_mode(variant_spec, voting_config)
    plateau_source = "scanner_fet" if voter_thin_mode in {"hybrid_v2", "normal_plateau"} else None

    diagnostic = {
        "scanner_ft_positive_candidate_count": scanner_ft_positive_count,
        "scanner_fet_positive_candidate_count": scanner_fet_positive_count,
        "scanner_ft_to_fv_positive_candidate_count_ratio": quality_metrics.fraction_or_zero(
            fv_positive_count, scanner_ft_positive_count
        ),
        "scanner_ft_to_fvt_positive_candidate_count_ratio": quality_metrics.fraction_or_zero(
            fvt_positive_count, scanner_ft_positive_count
        ),
        "scanner_ft_to_fet_retention_fraction": quality_metrics.fraction_or_zero(
            scanner_fet_positive_count, scanner_ft_positive_count
        ),
        "fv_positive_candidate_count": fv_positive_count,
        "fvt_positive_candidate_count": fvt_positive_count,
        "fv_to_fvt_positive_candidate_count_ratio": quality_metrics.fraction_or_zero(
            fvt_positive_count, fv_positive_count
        ),
        "fvt_to_fv_positive_fraction": quality_metrics.fraction_or_zero(
            fvt_positive_count, fv_positive_count
        ),
        "scanner_ft_vs_fv_positive_buffered_overlap_radius2": quality_metrics.positive_pair_overlap(
            candidate_name="scanner_ft",
            reference_name="fv",
            candidate_mask=scanner_ft_positive,
            reference_mask=fv_positive,
            buffer_radius=buffer_radius,
        ),
        "scanner_ft_vs_fvt_positive_buffered_overlap_radius2": quality_metrics.positive_pair_overlap(
            candidate_name="scanner_ft",
            reference_name="fvt",
            candidate_mask=scanner_ft_positive,
            reference_mask=fvt_positive,
            buffer_radius=buffer_radius,
        ),
        "fv_vs_fvt_positive_buffered_overlap_radius2": quality_metrics.positive_pair_overlap(
            candidate_name="fv",
            reference_name="fvt",
            candidate_mask=fv_positive,
            reference_mask=fvt_positive,
            buffer_radius=buffer_radius,
        ),
        "fvt_candidate_to_scanner_ft_distance_p50": fvt_to_scanner_ft_distance[
            "candidate_to_truth_median"
        ],
        "fvt_candidate_to_scanner_ft_distance_p95": fvt_to_scanner_ft_distance[
            "candidate_to_truth_p95"
        ],
        "fvt_candidate_to_fv_distance_p50": fvt_to_fv_distance["candidate_to_truth_median"],
        "fvt_candidate_to_fv_distance_p95": fvt_to_fv_distance["candidate_to_truth_p95"],
        "scanner_ft_positive_edge_shell_fraction": quality_metrics.edge_candidate_fraction(
            scanner_ft_positive, edge_margin=quality_metrics.EDGE_FALSE_POSITIVE_MARGIN
        ),
        "scanner_fet_positive_edge_shell_fraction": quality_metrics.edge_candidate_fraction(
            scanner_fet_positive, edge_margin=quality_metrics.EDGE_FALSE_POSITIVE_MARGIN
        ),
        "fv_positive_edge_shell_fraction": quality_metrics.edge_candidate_fraction(
            fv_positive, edge_margin=quality_metrics.EDGE_FALSE_POSITIVE_MARGIN
        ),
        "fvt_positive_edge_shell_fraction": quality_metrics.edge_candidate_fraction(
            fvt_positive, edge_margin=quality_metrics.EDGE_FALSE_POSITIVE_MARGIN
        ),
        "fvt_positive_edge_candidate_fraction": quality_metrics.edge_candidate_fraction(
            fvt_positive, edge_margin=quality_metrics.EDGE_FALSE_POSITIVE_MARGIN
        ),
        "fvt_positive_edge_false_positive_fraction": report["quality"]["edge_false_positive"][
            "fvt_positive_top_truth_count"
        ]["edge_false_positive_fraction_of_candidates"],
        "voter_thin_mode": voter_thin_mode,
        "plateau_tie_breaker_source": plateau_source,
        "scanner_thin_mode": scanner_config.scanner_thin_mode,
    }

    voter = OptimalSurfaceVoter(ru=voting_config.ru, rv=voting_config.rv, rw=voting_config.rw)
    thinning_modes = {}
    for mode in ("reference", "hybrid", "hybrid_v2", "normal_plateau"):
        tie_breaker = (
            scanner_volumes["scanner_fet"] if mode in {"hybrid_v2", "normal_plateau"} else None
        )
        thinning_modes[mode] = _scanner_downstream_thinning_report(
            case=case,
            voter=voter,
            fv=fv,
            vp=vp,
            vt=vt,
            mode=mode,
            plateau_tie_breaker=tie_breaker,
            truth_metric_config=truth_metric_config,
            reference_sigma=voting_config.reference_thin_sigma,
        )
    diagnostic["thinning_modes"] = thinning_modes

    for key, tie_breaker in (
        ("hybrid_v2_tiebreaker_fet", scanner_volumes["scanner_fet"]),
        ("hybrid_v2_tiebreaker_fv", fv),
        ("hybrid_v2_tiebreaker_scanner_ft", scanner_volumes["scanner_ft"]),
    ):
        diagnostic[key] = _scanner_downstream_thinning_report(
            case=case,
            voter=voter,
            fv=fv,
            vp=vp,
            vt=vt,
            mode="hybrid_v2",
            plateau_tie_breaker=tie_breaker,
            truth_metric_config=truth_metric_config,
            reference_sigma=voting_config.reference_thin_sigma,
        )
    return diagnostic


def _scanner_stage_loss_diagnostics(
    *,
    case: Synthetic3DCase,
    voting_config: SyntheticVotingConfig,
    variant_spec: VariantSpec,
    scanner_volumes: Mapping[str, np.ndarray],
    fv: np.ndarray,
    fvt: np.ndarray,
    skin_mask: np.ndarray,
    truth_metric_config: SyntheticTruthMetricConfig,
    boundary_seed_selector: BoundarySeedSelector | None = None,
) -> dict[str, Any]:
    truth_surface_half_width = _validate_nonnegative_finite_scalar(
        truth_metric_config.truth_surface_half_width, "truth_surface_half_width"
    )
    buffer_radius = _validate_nonnegative_finite_scalar(
        truth_metric_config.buffer_radius, "buffer_radius"
    )
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(truth_surface_half_width)
    truth_fault_mask = np.asarray(case.truth_fault_mask, dtype=bool)
    seed_selected_mask = _seed_selection_diagnostic(
        shape=case.shape,
        voting_config=voting_config,
        ft=scanner_volumes["scanner_fet"],
        pt=scanner_volumes["scanner_fpt"],
        tt=scanner_volumes["scanner_ftt"],
    )
    boundary_seed_retention: dict[str, Any] | None = None
    if variant_spec.seed_policy == "boundary_seed_retention_v1":
        if boundary_seed_selector is None:
            raise ValueError("boundary_seed_selector is required for boundary seed retention")
        _, seeds, boundary_seed_retention = boundary_seed_selector(
            voting_config=voting_config,
            ft=scanner_volumes["scanner_fet"],
            pt=scanner_volumes["scanner_fpt"],
            tt=scanner_volumes["scanner_ftt"],
            target=scanner_volumes["scanner_fet"],
            target_source="scanner_fet",
            edge_margin=quality_metrics.EDGE_FALSE_POSITIVE_MARGIN,
        )
        seed_selected_mask = _seed_mask_from_seeds(case.shape, seeds)

    stage_masks = {
        "scanner_ft_positive": quality_metrics.positive_candidate_mask(
            scanner_volumes["scanner_ft"]
        ),
        "scanner_fet_positive": quality_metrics.positive_candidate_mask(
            scanner_volumes["scanner_fet"]
        ),
        "seed_candidate": np.asarray(scanner_volumes["scanner_fet"])
        > np.float32(voting_config.seed_threshold),
        "seed_selected": seed_selected_mask,
        "fv_positive": quality_metrics.positive_candidate_mask(fv),
        "fvt_positive": quality_metrics.positive_candidate_mask(fvt),
        "skin": np.asarray(skin_mask, dtype=bool),
    }
    stages = {
        name: quality_metrics.scanner_stage_metric(
            mask,
            truth_fault_mask=truth_fault_mask,
            truth_surface_mask=truth_surface_mask,
            buffer_radius=buffer_radius,
        )
        for name, mask in stage_masks.items()
    }
    if boundary_seed_retention is not None:
        stages["seed_selected"]["default_candidate_count"] = int(
            boundary_seed_retention["default_seed_count"]
        )
        stages["seed_selected"]["added_candidate_count"] = int(
            boundary_seed_retention["added_seed_count"]
        )

    transition_pairs = (
        ("scanner_ft_positive", "scanner_fet_positive"),
        ("scanner_fet_positive", "seed_candidate"),
        ("seed_candidate", "seed_selected"),
        ("seed_selected", "fv_positive"),
        ("fv_positive", "fvt_positive"),
        ("fvt_positive", "skin"),
        ("scanner_fet_positive", "seed_selected"),
    )
    transitions = {
        f"{source}_to_{target}": quality_metrics.scanner_stage_transition_metric(
            source_mask=stage_masks[source],
            target_mask=stage_masks[target],
            buffer_radius=buffer_radius,
        )
        for source, target in transition_pairs
    }
    return {"stages": stages, "transitions": transitions}


def _seed_selection_diagnostic(
    *,
    shape: tuple[int, int, int],
    voting_config: SyntheticVotingConfig,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
) -> np.ndarray:
    voter = OptimalSurfaceVoter(ru=voting_config.ru, rv=voting_config.rv, rw=voting_config.rw)
    seeds = voter.pick_seeds(
        d=voting_config.seed_distance,
        fm=voting_config.seed_threshold,
        ft=ft,
        pt=pt,
        tt=tt,
    )
    return _seed_mask_from_seeds(shape, seeds)


def _seed_mask_from_seeds(shape: tuple[int, int, int], seeds: Sequence[FaultCell]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    n3, n2, n1 = shape
    for seed in seeds:
        if 0 <= seed.i3 < n3 and 0 <= seed.i2 < n2 and 0 <= seed.i1 < n1:
            mask[seed.i3, seed.i2, seed.i1] = True
    return mask


def _scanner_downstream_thinning_report(
    *,
    case: Synthetic3DCase,
    voter: OptimalSurfaceVoter,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    mode: str,
    plateau_tie_breaker: np.ndarray | None,
    truth_metric_config: SyntheticTruthMetricConfig,
    reference_sigma: float,
) -> dict[str, Any]:
    fvt = voter.thin(
        fv,
        vp,
        vt,
        mode=mode,
        reference_sigma=reference_sigma,
        plateau_tie_breaker=plateau_tie_breaker,
    )
    truth_surface_half_width = _validate_nonnegative_finite_scalar(
        truth_metric_config.truth_surface_half_width, "truth_surface_half_width"
    )
    buffer_radius = _validate_nonnegative_finite_scalar(
        truth_metric_config.buffer_radius, "buffer_radius"
    )
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(truth_surface_half_width)
    truth_fault_mask = np.asarray(case.truth_fault_mask, dtype=bool)
    fvt_positive = top_positive_truth_count_mask(fvt, truth_surface_mask)
    quality = quality_metrics.top_truth_count_quality(
        fvt_positive,
        truth_fault_mask=truth_fault_mask,
        truth_surface_mask=truth_surface_mask,
        buffer_radius=buffer_radius,
    )
    return {
        "fvt_positive_candidate_count": int(quality["buffered_overlap_radius2"]["candidate_count"]),
        "fvt_positive_buffered_f1_r2": quality["buffered_overlap_radius2"]["buffered_f1"],
        "fvt_positive_distance_p95": quality["surface_distance"]["candidate_to_truth_p95"],
        "fvt_positive_edge_candidate_fraction": quality_metrics.edge_candidate_fraction(
            fvt > np.float32(quality_metrics.NONZERO_EPSILON),
            edge_margin=quality_metrics.EDGE_FALSE_POSITIVE_MARGIN,
        ),
        "fvt_positive_edge_false_positive_fraction": edge_false_positive_ratio(
            fvt_positive,
            truth_surface_mask,
            edge_margin=quality_metrics.EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        )["edge_false_positive_fraction_of_candidates"],
    }


def _run_voter_thinning_diagnostic(
    *,
    case: Synthetic3DCase,
    voter: OptimalSurfaceVoter,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    reference_sigma: float,
    truth_metric_config: SyntheticTruthMetricConfig,
    skinning_config: SyntheticSkinningConfig | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    del skinning_config
    truth_surface_half_width = _validate_nonnegative_finite_scalar(
        truth_metric_config.truth_surface_half_width, "truth_surface_half_width"
    )
    buffer_radius = _validate_nonnegative_finite_scalar(
        truth_metric_config.buffer_radius, "buffer_radius"
    )
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(truth_surface_half_width)
    truth_fault_mask = np.asarray(case.truth_fault_mask, dtype=bool)
    fvt_reference = voter.thin(fv, vp, vt, mode="reference", reference_sigma=reference_sigma)
    fvt_normal = voter.thin(fv, vp, vt, mode="normal", reference_sigma=reference_sigma)
    reference_report = _thinning_mode_diagnostic_report(
        fvt_reference,
        vp=vp,
        vt=vt,
        case=case,
        truth_fault_mask=truth_fault_mask,
        truth_surface_mask=truth_surface_mask,
        buffer_radius=buffer_radius,
    )
    normal_report = _thinning_mode_diagnostic_report(
        fvt_normal,
        vp=vp,
        vt=vt,
        case=case,
        truth_fault_mask=truth_fault_mask,
        truth_surface_mask=truth_surface_mask,
        buffer_radius=buffer_radius,
    )
    report = {
        "reference": reference_report,
        "normal": normal_report,
        "delta": {
            "normal_minus_reference": _thinning_quality_delta(
                normal_report["quality"]["fvt_top_truth_count"],
                reference_report["quality"]["fvt_top_truth_count"],
            )
        },
        "keep_mask": _thinning_keep_mask_comparison(
            fvt_reference > 0.0,
            fvt_normal > 0.0,
            truth_fault_mask=truth_fault_mask,
            buffer_radius=buffer_radius,
        ),
    }
    volumes = {
        "fvt_reference_thinning_diagnostic": fvt_reference,
        "fvt_normal_thinning_diagnostic": fvt_normal,
        "keep_reference_thinning_diagnostic": (fvt_reference > 0.0).astype(np.float32),
        "keep_normal_thinning_diagnostic": (fvt_normal > 0.0).astype(np.float32),
        "keep_both_thinning_diagnostic": ((fvt_reference > 0.0) & (fvt_normal > 0.0)).astype(
            np.float32
        ),
        "keep_reference_only_thinning_diagnostic": (
            (fvt_reference > 0.0) & ~(fvt_normal > 0.0)
        ).astype(np.float32),
        "keep_normal_only_thinning_diagnostic": (
            (fvt_normal > 0.0) & ~(fvt_reference > 0.0)
        ).astype(np.float32),
    }
    return report, volumes


def _thinning_mode_diagnostic_report(
    fvt: np.ndarray,
    *,
    vp: np.ndarray,
    vt: np.ndarray,
    case: Synthetic3DCase,
    truth_fault_mask: np.ndarray,
    truth_surface_mask: np.ndarray,
    buffer_radius: float,
) -> dict[str, Any]:
    fvt_top_truth_count = top_truth_count_mask(fvt, truth_surface_mask)
    quality = quality_metrics.top_truth_count_quality(
        fvt_top_truth_count,
        truth_fault_mask=truth_fault_mask,
        truth_surface_mask=truth_surface_mask,
        buffer_radius=buffer_radius,
    )
    quality["orientation_error"] = masked_orientation_error(
        vp, vt, case.truth_strike, case.truth_dip, fvt_top_truth_count
    )
    return {
        "pyosv": {"fvt": _array_summary(fvt)},
        "quality": {"fvt_top_truth_count": quality},
    }


def _thinning_quality_delta(
    normal_quality: Mapping[str, Any], reference_quality: Mapping[str, Any]
) -> dict[str, float]:
    return {
        "fvt_buffered_f1_r2": float(
            normal_quality["buffered_overlap_radius2"]["buffered_f1"]
            - reference_quality["buffered_overlap_radius2"]["buffered_f1"]
        ),
        "fvt_candidate_to_truth_p95": float(
            normal_quality["surface_distance"]["candidate_to_truth_p95"]
            - reference_quality["surface_distance"]["candidate_to_truth_p95"]
        ),
        "fvt_strike_median_error": float(
            normal_quality["orientation_error"]["strike_median"]
            - reference_quality["orientation_error"]["strike_median"]
        ),
        "fvt_dip_median_error": float(
            normal_quality["orientation_error"]["dip_median"]
            - reference_quality["orientation_error"]["dip_median"]
        ),
    }


def _thinning_keep_mask_comparison(
    keep_reference: np.ndarray,
    keep_normal: np.ndarray,
    *,
    truth_fault_mask: np.ndarray,
    buffer_radius: float,
) -> dict[str, Any]:
    reference = np.asarray(keep_reference, dtype=bool)
    normal = np.asarray(keep_normal, dtype=bool)
    if reference.shape != normal.shape:
        raise ValueError(f"keep mask shapes must match, got {reference.shape} and {normal.shape}")
    intersection = reference & normal
    union = reference | normal
    reference_only = reference & ~normal
    normal_only = normal & ~reference
    intersection_count = int(np.count_nonzero(intersection))
    union_count = int(np.count_nonzero(union))
    return {
        "reference_count": int(np.count_nonzero(reference)),
        "normal_count": int(np.count_nonzero(normal)),
        "intersection_count": intersection_count,
        "union_count": union_count,
        "reference_only_count": int(np.count_nonzero(reference_only)),
        "normal_only_count": int(np.count_nonzero(normal_only)),
        "jaccard": float(intersection_count / union_count) if union_count else 1.0,
        "reference_only_buffered_overlap_radius2": buffered_surface_overlap(
            reference_only, truth_fault_mask, radius=buffer_radius
        ),
        "normal_only_buffered_overlap_radius2": buffered_surface_overlap(
            normal_only, truth_fault_mask, radius=buffer_radius
        ),
    }


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
        "finite_fraction": float(np.count_nonzero(finite) / values.size) if values.size else 0.0,
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "nonzero_fraction": (
            float(np.count_nonzero(np.abs(values) > quality_metrics.NONZERO_EPSILON) / values.size)
            if values.size
            else 0.0
        ),
    }
