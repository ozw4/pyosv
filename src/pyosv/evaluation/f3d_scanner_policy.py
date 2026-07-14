"""Shared-scan F3 scanner-thinning policy comparison support.

This module keeps the F3 experiment narrow: both policy branches consume the
same reference-like scanner output and use the quality workflow downstream.
Only scanner thinning, and the edge-cleanup effect implied by that mode, may
differ between the branches.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyosv.f3d_reference import interior_slices
from pyosv.evaluation.promotion.scanner_policy import (
    QUALITY_WORKFLOW_SCANNER_THINNING_POLICY_PROFILE,
    REFERENCE_LIKE_NORMAL_SCANNER_POLICY_ID,
    REFERENCE_LIKE_REFERENCE_SCANNER_POLICY_ID,
    effective_remove_edge_effects,
)
from pyosv.metrics import (
    buffered_ridge_overlap,
    sparse_ridge_distance_metrics,
)

COMPARISON_PROFILE = QUALITY_WORKFLOW_SCANNER_THINNING_POLICY_PROFILE
BASELINE_POLICY_ID = REFERENCE_LIKE_REFERENCE_SCANNER_POLICY_ID
CANDIDATE_POLICY_ID = REFERENCE_LIKE_NORMAL_SCANNER_POLICY_ID

POLICY_ROLES = ("baseline", "candidate")
OUTPUT_NAMES = (
    "ft_py.dat",
    "pt_py.dat",
    "tt_py.dat",
    "fet_py.dat",
    "fpt_py.dat",
    "ftt_py.dat",
    "fv_py.dat",
    "vp_py.dat",
    "vt_py.dat",
    "fvt_py.dat",
)
FINITE_STAGE_NAMES = OUTPUT_NAMES
NONEMPTY_STAGE_NAMES = ("fet_py.dat", "fv_py.dat", "fvt_py.dat")

DEFAULT_DENSITY_RATIO_MIN = 0.5
DEFAULT_DENSITY_RATIO_MAX = 2.0
DEFAULT_EDGE_DENSITY_MAX_DELTA = 0.10
DEFAULT_SPARSE_DISTANCE_P95_MAX_DELTA = 5.0
DEFAULT_CROP_STABILITY_MAX_CV = 2.0
DEFAULT_NONZERO_EPSILON = 1.0e-6
DEFAULT_RIDGE_PERCENTILE = 99.0
DEFAULT_RIDGE_BUFFER_RADIUS = 2.0


@dataclass(frozen=True)
class ScannerPolicy:
    """One scanner-thinning branch in the fixed quality-workflow comparison."""

    role: str
    policy_id: str
    scanner_thin_mode: str
    requested_remove_edge_effects: bool = True

    @property
    def effective_remove_edge_effects(self) -> bool | None:
        """Return the edge-cleanup effect, or ``None`` when the mode ignores it."""

        return effective_remove_edge_effects(
            self.scanner_thin_mode,
            self.requested_remove_edge_effects,
        )


BASELINE_POLICY = ScannerPolicy(
    role="baseline",
    policy_id=BASELINE_POLICY_ID,
    scanner_thin_mode="reference",
)
CANDIDATE_POLICY = ScannerPolicy(
    role="candidate",
    policy_id=CANDIDATE_POLICY_ID,
    scanner_thin_mode="normal",
)
SCANNER_POLICIES = (BASELINE_POLICY, CANDIDATE_POLICY)


def policy_definition(role: str) -> ScannerPolicy:
    """Return the fixed policy definition for ``baseline`` or ``candidate``."""

    for policy in SCANNER_POLICIES:
        if policy.role == role:
            return policy
    raise ValueError("role must be one of: " + ", ".join(POLICY_ROLES))


def build_policy_config(
    policy: ScannerPolicy,
    *,
    reference_thin_sigma: float,
    ru: int,
    rv: int,
    rw: int,
    strain_max1: float,
    strain_max2: float,
    surface_smoothing1: float,
    surface_smoothing2: float,
    surface_orientation_smoothing: float | None,
    final_normalization_smoothing: float | None,
    d: int,
    fm: float,
) -> dict[str, Any]:
    """Build requested and effective configuration for one policy branch."""

    orientation_smoothing = (
        float(max(rv, rw))
        if surface_orientation_smoothing is None
        else float(surface_orientation_smoothing)
    )
    normalization_smoothing = (
        0.0 if final_normalization_smoothing is None else float(final_normalization_smoothing)
    )
    common = {
        "comparison_profile": COMPARISON_PROFILE,
        "workflow_mode": "quality",
        "scanner_backend": "reference-like",
        "scanner_thin_mode": policy.scanner_thin_mode,
        "requested_remove_edge_effects": bool(policy.requested_remove_edge_effects),
        "reference_thin_sigma": float(reference_thin_sigma),
        "voter_thin_mode": "hybrid_v2",
        "ru": int(ru),
        "rv": int(rv),
        "rw": int(rw),
        "strain_max1": float(strain_max1),
        "strain_max2": float(strain_max2),
        "surface_smoothing1": float(surface_smoothing1),
        "surface_smoothing2": float(surface_smoothing2),
        "surface_orientation_smoothing": orientation_smoothing,
        "final_normalization_smoothing": normalization_smoothing,
        "attribute_smoothing": 1,
        "surface_support_min_fraction": 0.0,
        "surface_support_exponent": 0.0,
        "surface_voting_boundary_policy": "reference",
        "hybrid_orientation_gradient_threshold": 8.0,
        "hybrid_v2_edge_margin": 2,
        "plateau_tolerance": 1.0e-6,
        "d": int(d),
        "fm": float(fm),
    }
    effective = dict(common)
    effective["effective_remove_edge_effects"] = policy.effective_remove_edge_effects
    return {
        "requested": common,
        "effective": effective,
    }


def build_shared_scanner_config(
    *,
    sigma1: float,
    sigma2: float,
    phi_min: float,
    phi_max: float,
    theta_min: float,
    theta_max: float,
) -> dict[str, Any]:
    """Build the common reference-like scanner configuration."""

    return {
        "backend": "reference-like",
        "method": "FaultOrientScanner3.scan",
        "sigma1": float(sigma1),
        "sigma2": float(sigma2),
        "phi_min": float(phi_min),
        "phi_max": float(phi_max),
        "theta_min": float(theta_min),
        "theta_max": float(theta_max),
        "execution_contract": "once_per_crop",
    }


def build_quality_downstream_config(
    *,
    reference_thin_sigma: float,
    ru: int,
    rv: int,
    rw: int,
    strain_max1: float,
    strain_max2: float,
    surface_smoothing1: float,
    surface_smoothing2: float,
    surface_orientation_smoothing: float | None,
    final_normalization_smoothing: float | None,
    d: int,
    fm: float,
) -> dict[str, Any]:
    """Return the downstream settings held fixed between policies."""

    config = build_policy_config(
        BASELINE_POLICY,
        reference_thin_sigma=reference_thin_sigma,
        ru=ru,
        rv=rv,
        rw=rw,
        strain_max1=strain_max1,
        strain_max2=strain_max2,
        surface_smoothing1=surface_smoothing1,
        surface_smoothing2=surface_smoothing2,
        surface_orientation_smoothing=surface_orientation_smoothing,
        final_normalization_smoothing=final_normalization_smoothing,
        d=d,
        fm=fm,
    )["effective"]
    excluded = {
        "comparison_profile",
        "scanner_backend",
        "scanner_thin_mode",
        "requested_remove_edge_effects",
        "effective_remove_edge_effects",
    }
    return {key: value for key, value in config.items() if key not in excluded}


def run_shared_scan_policy_pipeline(
    ep: np.ndarray,
    *,
    sigma1: float = 8.0,
    sigma2: float = 8.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    theta_min: float = 65.0,
    theta_max: float = 80.0,
    ru: int = 10,
    rv: int = 20,
    rw: int = 30,
    strain_max1: float = 0.25,
    strain_max2: float = 0.25,
    surface_smoothing1: float = 2.0,
    surface_smoothing2: float = 2.0,
    surface_orientation_smoothing: float | None = None,
    final_normalization_smoothing: float | None = None,
    d: int = 4,
    fm: float = 0.3,
    reference_thin_sigma: float = 1.0,
    scanner_factory: Callable[..., Any] | None = None,
    voter_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one raw scan and the two fixed scanner-thinning policy branches.

    A new voter is constructed for every branch. ``hybrid_v2`` receives that
    branch's scanner-thinned likelihood as its plateau tie-breaker.
    """

    if scanner_factory is None:
        from pyosv.orient3d import FaultOrientScanner3

        scanner_factory = FaultOrientScanner3
    if voter_factory is None:
        from pyosv.voting3d import OptimalSurfaceVoter

        voter_factory = OptimalSurfaceVoter

    scanner = scanner_factory(sigma1=sigma1, sigma2=sigma2)
    ft, pt, tt = scanner.scan(phi_min, phi_max, theta_min, theta_max, ep)

    policies: dict[str, dict[str, Any]] = {}
    for policy in SCANNER_POLICIES:
        fet, fpt, ftt = scanner.thin(
            ft,
            pt,
            tt,
            mode=policy.scanner_thin_mode,
            reference_sigma=reference_thin_sigma,
            remove_edge_effects=policy.requested_remove_edge_effects,
        )

        voter = voter_factory(ru=ru, rv=rv, rw=rw)
        voter.set_strain_max(strain_max1, strain_max2)
        voter.set_surface_smoothing(surface_smoothing1, surface_smoothing2)
        voter.set_surface_support_policy(min_fraction=0.0, exponent=0.0)
        voter.set_surface_voting_boundary_policy("reference")
        voter.set_surface_orientation_smoothing(
            max(rv, rw) if surface_orientation_smoothing is None else surface_orientation_smoothing
        )
        voter.set_final_normalization_smoothing(
            0.0 if final_normalization_smoothing is None else final_normalization_smoothing
        )
        fv, vp, vt = voter.apply_voting(d=d, fm=fm, ft=fet, pt=fpt, tt=ftt)
        fvt = voter.thin(
            fv,
            vp,
            vt,
            mode="hybrid_v2",
            reference_sigma=reference_thin_sigma,
            hybrid_orientation_gradient_threshold=8.0,
            hybrid_v2_edge_margin=2,
            plateau_tie_breaker=fet,
            plateau_tolerance=1.0e-6,
        )
        outputs = {
            "ft_py.dat": ft,
            "pt_py.dat": pt,
            "tt_py.dat": tt,
            "fet_py.dat": fet,
            "fpt_py.dat": fpt,
            "ftt_py.dat": ftt,
            "fv_py.dat": fv,
            "vp_py.dat": vp,
            "vt_py.dat": vt,
            "fvt_py.dat": fvt,
        }
        policies[policy.role] = {
            "policy_id": policy.policy_id,
            "config": build_policy_config(
                policy,
                reference_thin_sigma=reference_thin_sigma,
                ru=ru,
                rv=rv,
                rw=rw,
                strain_max1=strain_max1,
                strain_max2=strain_max2,
                surface_smoothing1=surface_smoothing1,
                surface_smoothing2=surface_smoothing2,
                surface_orientation_smoothing=surface_orientation_smoothing,
                final_normalization_smoothing=final_normalization_smoothing,
                d=d,
                fm=fm,
            ),
            "outputs": outputs,
        }

    return {
        "scanner_execution_count": 1,
        "shared_scanner": build_shared_scanner_config(
            sigma1=sigma1,
            sigma2=sigma2,
            phi_min=phi_min,
            phi_max=phi_max,
            theta_min=theta_min,
            theta_max=theta_max,
        ),
        "policies": policies,
    }


def build_stage_density_report(
    outputs: Mapping[str, np.ndarray],
    *,
    interior_margin: int,
    nonzero_epsilon: float = DEFAULT_NONZERO_EPSILON,
) -> dict[str, Any]:
    """Summarize scanner-thinned, voted, and final-ridge densities."""

    epsilon = _validate_nonnegative_float(nonzero_epsilon, "nonzero_epsilon")
    arrays = {
        "fet": _output_array(outputs, "fet_py.dat"),
        "fv": _output_array(outputs, "fv_py.dat"),
        "fvt": _output_array(outputs, "fvt_py.dat"),
    }
    shape = arrays["fet"].shape
    if any(values.shape != shape for values in arrays.values()):
        raise ValueError("fet, fv, and fvt output shapes must match")
    local_interior = interior_slices(shape, margin=interior_margin)

    stages: dict[str, Any] = {}
    for name, values in arrays.items():
        full_fraction = _nonzero_fraction(values, epsilon)
        interior_fraction = _nonzero_fraction(values[local_interior], epsilon)
        stages[name] = {
            "nonzero_count": int(np.count_nonzero(np.abs(values) > epsilon)),
            "nonzero_fraction": full_fraction,
            "interior_nonzero_fraction": interior_fraction,
            "edge_density_proxy": max(0.0, full_fraction - interior_fraction),
        }
    return stages


def build_direct_policy_comparison(
    baseline_outputs: Mapping[str, np.ndarray],
    candidate_outputs: Mapping[str, np.ndarray],
    *,
    interior_margin: int,
    ridge_percentile: float = DEFAULT_RIDGE_PERCENTILE,
    ridge_buffer_radius: float = DEFAULT_RIDGE_BUFFER_RADIUS,
    nonzero_epsilon: float = DEFAULT_NONZERO_EPSILON,
) -> dict[str, Any]:
    """Compare baseline and candidate final ridges without treating either as truth."""

    baseline = _output_array(baseline_outputs, "fvt_py.dat")
    candidate = _output_array(candidate_outputs, "fvt_py.dat")
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate fvt shapes must match")
    local_interior = interior_slices(baseline.shape, margin=interior_margin)
    baseline_interior = baseline[local_interior]
    candidate_interior = candidate[local_interior]

    baseline_density = _nonzero_fraction(baseline, nonzero_epsilon)
    candidate_density = _nonzero_fraction(candidate, nonzero_epsilon)
    density_ratio = _safe_density_ratio(candidate_density, baseline_density)

    epsilon = _validate_nonnegative_float(nonzero_epsilon, "nonzero_epsilon")
    baseline_mask = baseline > epsilon
    candidate_mask = candidate > epsilon
    candidate_only = candidate_mask & ~baseline_mask
    baseline_only = baseline_mask & ~candidate_mask
    shell = np.ones(baseline.shape, dtype=bool)
    shell[local_interior] = False

    baseline_count = int(np.count_nonzero(baseline_mask))
    candidate_count = int(np.count_nonzero(candidate_mask))
    candidate_only_count = int(np.count_nonzero(candidate_only))
    baseline_only_count = int(np.count_nonzero(baseline_only))
    shell_candidate_only_count = int(np.count_nonzero(candidate_only & shell))

    return {
        "fvt_density": {
            "baseline": baseline_density,
            "candidate": candidate_density,
            "candidate_over_baseline_ratio": density_ratio,
        },
        "buffered_ridge_overlap": {
            "interior": buffered_ridge_overlap(
                baseline_interior,
                candidate_interior,
                percentile=ridge_percentile,
                radius=ridge_buffer_radius,
            )
        },
        "sparse_ridge_distance_metrics": {
            "interior": sparse_ridge_distance_metrics(
                baseline_interior,
                candidate_interior,
                percentile=ridge_percentile,
            )
        },
        "ridge_mask_difference": {
            "mask_definition": "fvt > nonzero_epsilon",
            "nonzero_epsilon": epsilon,
            "baseline_count": baseline_count,
            "candidate_count": candidate_count,
            "candidate_only_count": candidate_only_count,
            "baseline_only_count": baseline_only_count,
            "candidate_only_fraction": _ratio(candidate_only_count, candidate_count),
            "candidate_only_fraction_denominator": "candidate_positive_fvt_count",
            "baseline_only_fraction": _ratio(baseline_only_count, baseline_count),
            "baseline_only_fraction_denominator": "baseline_positive_fvt_count",
            "edge_shell_candidate_only_count": shell_candidate_only_count,
            "edge_shell_candidate_only_fraction": _ratio(
                shell_candidate_only_count,
                candidate_only_count,
            ),
            "edge_shell_candidate_only_fraction_denominator": "candidate_only_count",
            "edge_shell_candidate_only_density": _ratio(
                shell_candidate_only_count,
                int(np.count_nonzero(shell)),
            ),
        },
    }


def aggregate_policy_crops(crops: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Build compact density, edge, distance, and finite summaries for a policy."""

    crop_list = list(crops)
    result: dict[str, Any] = {"crop_count": len(crop_list), "stage_density": {}}
    for stage in ("fet", "fv", "fvt"):
        values = _finite_values(
            _nested(crop, "stage_density", stage, "nonzero_fraction") for crop in crop_list
        )
        result["stage_density"][stage] = _distribution(values, include_cv=True)

    edge_values = _finite_values(
        _nested(crop, "stage_density", "fvt", "edge_density_proxy") for crop in crop_list
    )
    distance_values = _finite_values(
        _nested(
            crop,
            "sparse_ridge_distance_metrics",
            "interior",
            "fvt",
            "candidate_to_reference_p95",
        )
        for crop in crop_list
    )
    result["fvt_edge_density_proxy"] = _distribution(edge_values)
    result["public_fvt_sparse_distance_p95"] = _distribution(distance_values)
    result["finite_failure_count"] = sum(_crop_finite_failure_count(crop) for crop in crop_list)
    result["empty_stage_count"] = sum(_crop_empty_stage_count(crop) for crop in crop_list)
    result["existing_metrics"] = _aggregate_numeric_roots(
        crop_list,
        roots=(
            "pyosv",
            "pyosv_interior",
            "voting",
            "normalized_correlation",
            "top_percentile_overlap",
            "buffered_ridge_overlap",
            "sparse_ridge_distance_metrics",
        ),
    )
    return result


def aggregate_direct_comparisons(
    crop_comparisons: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate truthless direct baseline/candidate crop comparisons."""

    comparisons = list(crop_comparisons)
    paths = {
        "fvt_density_ratio": ("fvt_density", "candidate_over_baseline_ratio"),
        "buffered_precision": ("buffered_ridge_overlap", "interior", "buffered_precision"),
        "buffered_recall": ("buffered_ridge_overlap", "interior", "buffered_recall"),
        "candidate_to_baseline_distance_p95": (
            "sparse_ridge_distance_metrics",
            "interior",
            "candidate_to_reference_p95",
        ),
        "candidate_only_fraction": ("ridge_mask_difference", "candidate_only_fraction"),
        "baseline_only_fraction": ("ridge_mask_difference", "baseline_only_fraction"),
        "edge_shell_candidate_only_fraction": (
            "ridge_mask_difference",
            "edge_shell_candidate_only_fraction",
        ),
    }
    return {
        "crop_count": len(comparisons),
        **{
            name: _distribution(
                _finite_values(_nested(comparison, *path) for comparison in comparisons)
            )
            for name, path in paths.items()
        },
    }


def build_consensus(
    *,
    baseline_crops: Sequence[Mapping[str, Any]],
    candidate_crops: Sequence[Mapping[str, Any]],
    direct_comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build policy summaries and candidate-minus-baseline diagnostics."""

    baseline = aggregate_policy_crops(baseline_crops)
    candidate = aggregate_policy_crops(candidate_crops)
    direct = aggregate_direct_comparisons(direct_comparisons)

    baseline_density = _finite_float(_nested(baseline, "stage_density", "fvt", "mean"))
    candidate_density = _finite_float(_nested(candidate, "stage_density", "fvt", "mean"))
    baseline_edge = _finite_float(_nested(baseline, "fvt_edge_density_proxy", "mean"))
    candidate_edge = _finite_float(_nested(candidate, "fvt_edge_density_proxy", "mean"))
    baseline_distance = _finite_float(_nested(baseline, "public_fvt_sparse_distance_p95", "mean"))
    candidate_distance = _finite_float(_nested(candidate, "public_fvt_sparse_distance_p95", "mean"))
    baseline_cv = _finite_float(_nested(baseline, "stage_density", "fvt", "cv"))
    candidate_cv = _finite_float(_nested(candidate, "stage_density", "fvt", "cv"))

    return {
        "policies": {"baseline": baseline, "candidate": candidate},
        "candidate_minus_baseline": {
            "fvt_density_ratio": _safe_density_ratio(candidate_density, baseline_density),
            "fvt_edge_density_proxy_delta_mean": _difference(candidate_edge, baseline_edge),
            "public_fvt_sparse_distance_p95_delta_mean": _difference(
                candidate_distance,
                baseline_distance,
            ),
            "fvt_density_cv_delta": _difference(candidate_cv, baseline_cv),
            "direct_comparison": direct,
        },
    }


def validate_policy_comparison(
    *,
    baseline_crops: Sequence[Mapping[str, Any]],
    candidate_crops: Sequence[Mapping[str, Any]],
    direct_comparisons: Sequence[Mapping[str, Any]],
    baseline_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
    scanner_execution_count: int,
    expected_crop_count: int,
    density_ratio_min: float = DEFAULT_DENSITY_RATIO_MIN,
    density_ratio_max: float = DEFAULT_DENSITY_RATIO_MAX,
    edge_density_max_delta: float = DEFAULT_EDGE_DENSITY_MAX_DELTA,
    sparse_distance_p95_max_delta: float = DEFAULT_SPARSE_DISTANCE_P95_MAX_DELTA,
    crop_stability_max_cv: float = DEFAULT_CROP_STABILITY_MAX_CV,
) -> dict[str, Any]:
    """Apply conservative, truthless external-smoke checks to the comparison."""

    baseline_list = list(baseline_crops)
    candidate_list = list(candidate_crops)
    comparison_list = list(direct_comparisons)
    crop_counts_match = (
        len(baseline_list) == expected_crop_count
        and len(candidate_list) == expected_crop_count
        and len(comparison_list) == expected_crop_count
    )

    checks: dict[str, dict[str, Any]] = {}
    checks["shared_scan_contract"] = _check(
        scanner_execution_count == expected_crop_count and crop_counts_match,
        scanner_execution_count=scanner_execution_count,
        expected_scanner_execution_count=expected_crop_count,
        crop_counts={
            "baseline": len(baseline_list),
            "candidate": len(candidate_list),
            "direct": len(comparison_list),
        },
        reason="scanner execution count must equal the number of compared crops",
    )

    finite_failures = {
        "baseline": sum(_crop_finite_failure_count(crop) for crop in baseline_list),
        "candidate": sum(_crop_finite_failure_count(crop) for crop in candidate_list),
    }
    checks["finite_outputs"] = _check(
        sum(finite_failures.values()) == 0,
        failure_count=sum(finite_failures.values()),
        policy_failure_count=finite_failures,
        stages=list(FINITE_STAGE_NAMES),
        reason="one or more policy stages contain non-finite values or are missing",
    )

    empty_failures = {
        "baseline": _empty_stage_failures(baseline_list),
        "candidate": _empty_stage_failures(candidate_list),
    }
    checks["nonempty_stages"] = _check(
        not empty_failures["baseline"] and not empty_failures["candidate"],
        failures=empty_failures,
        stages=list(NONEMPTY_STAGE_NAMES),
        reason="fet, fv, and fvt must be nonempty for every policy and crop",
    )

    density_ratios = [
        _finite_float(_nested(comparison, "fvt_density", "candidate_over_baseline_ratio"))
        for comparison in comparison_list
    ]
    baseline_density_mean = _mean(
        _finite_values(
            _nested(crop, "stage_density", "fvt", "nonzero_fraction") for crop in baseline_list
        )
    )
    candidate_density_mean = _mean(
        _finite_values(
            _nested(crop, "stage_density", "fvt", "nonzero_fraction") for crop in candidate_list
        )
    )
    aggregate_density_ratio = _safe_density_ratio(candidate_density_mean, baseline_density_mean)
    density_available = (
        len(density_ratios) == expected_crop_count
        and all(value is not None for value in density_ratios)
        and aggregate_density_ratio is not None
    )
    finite_density_ratios = [value for value in density_ratios if value is not None]
    density_passed = density_available and all(
        density_ratio_min <= value <= density_ratio_max for value in finite_density_ratios
    )
    density_passed = bool(
        density_passed and density_ratio_min <= float(aggregate_density_ratio) <= density_ratio_max
    )
    checks["fvt_density_ratio"] = _check(
        density_passed,
        per_crop=density_ratios,
        minimum_crop_ratio=(min(finite_density_ratios) if finite_density_ratios else None),
        maximum_crop_ratio=(max(finite_density_ratios) if finite_density_ratios else None),
        aggregate=aggregate_density_ratio,
        minimum=float(density_ratio_min),
        maximum=float(density_ratio_max),
        reason="candidate/baseline fvt density ratio is unavailable or outside bounds",
    )

    edge_deltas = _paired_metric_deltas(
        baseline_list,
        candidate_list,
        ("stage_density", "fvt", "edge_density_proxy"),
    )
    edge_mean = _mean(edge_deltas)
    edge_max = max(edge_deltas) if edge_deltas else None
    checks["edge_density_proxy"] = _check(
        len(edge_deltas) == expected_crop_count
        and edge_mean is not None
        and edge_max is not None
        and edge_mean <= edge_density_max_delta
        and edge_max <= edge_density_max_delta,
        per_crop_delta=edge_deltas,
        aggregate_mean_delta=edge_mean,
        maximum_crop_delta=edge_max,
        maximum_allowed_delta=float(edge_density_max_delta),
        reason="candidate fvt edge-density proxy increased beyond the allowed delta",
    )

    distance_deltas = _paired_metric_deltas(
        baseline_list,
        candidate_list,
        (
            "sparse_ridge_distance_metrics",
            "interior",
            "fvt",
            "candidate_to_reference_p95",
        ),
    )
    distance_mean = _mean(distance_deltas)
    distance_max = max(distance_deltas) if distance_deltas else None
    checks["public_fvt_sparse_distance_p95"] = _check(
        len(distance_deltas) == expected_crop_count
        and distance_mean is not None
        and distance_max is not None
        and distance_mean <= sparse_distance_p95_max_delta
        and distance_max <= sparse_distance_p95_max_delta,
        per_crop_delta=distance_deltas,
        aggregate_mean_delta=distance_mean,
        maximum_crop_delta=distance_max,
        maximum_allowed_delta=float(sparse_distance_p95_max_delta),
        reason="candidate public-F3 fvt sparse distance p95 worsened beyond the smoke limit",
    )

    baseline_cv = _coefficient_of_variation(
        _finite_values(
            _nested(crop, "stage_density", "fvt", "nonzero_fraction") for crop in baseline_list
        )
    )
    candidate_cv = _coefficient_of_variation(
        _finite_values(
            _nested(crop, "stage_density", "fvt", "nonzero_fraction") for crop in candidate_list
        )
    )
    checks["crop_stability"] = _check(
        candidate_cv is not None and candidate_cv <= crop_stability_max_cv,
        baseline_fvt_density_cv=baseline_cv,
        candidate_fvt_density_cv=candidate_cv,
        candidate_minus_baseline_cv=_difference(candidate_cv, baseline_cv),
        maximum_candidate_cv=float(crop_stability_max_cv),
        reason="candidate fvt density crop-to-crop CV exceeded the smoke limit",
    )

    config_check = validate_configuration_contract(
        baseline_config=baseline_config,
        candidate_config=candidate_config,
    )
    checks["configuration_contract"] = config_check

    reasons = [
        str(check["reason"])
        for check in checks.values()
        if not check.get("passed", False) and check.get("reason")
    ]
    return {
        "role": "truthless_external_smoke",
        "passed": not reasons,
        "crop_count": int(expected_crop_count),
        "scanner_execution_count": int(scanner_execution_count),
        "checks": checks,
        "reasons": reasons,
    }


def validate_configuration_contract(
    *,
    baseline_config: Mapping[str, Any],
    candidate_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require requested/effective configs to differ only in allowed semantics."""

    baseline_requested = _as_mapping(baseline_config.get("requested"))
    candidate_requested = _as_mapping(candidate_config.get("requested"))
    baseline_effective = _as_mapping(baseline_config.get("effective"))
    candidate_effective = _as_mapping(candidate_config.get("effective"))
    requested_differences = recursive_difference_paths(
        baseline_requested,
        candidate_requested,
    )
    effective_differences = recursive_difference_paths(
        baseline_effective,
        candidate_effective,
    )
    expected_requested = ["scanner_thin_mode"]
    expected_effective = ["effective_remove_edge_effects", "scanner_thin_mode"]
    passed = (
        requested_differences == expected_requested
        and effective_differences == expected_effective
        and baseline_requested.get("scanner_thin_mode") == "reference"
        and candidate_requested.get("scanner_thin_mode") == "normal"
        and baseline_effective.get("effective_remove_edge_effects") is True
        and candidate_effective.get("effective_remove_edge_effects") is None
    )
    return _check(
        passed,
        requested_difference_paths=requested_differences,
        allowed_requested_difference_paths=expected_requested,
        effective_difference_paths=effective_differences,
        allowed_effective_difference_paths=expected_effective,
        reason="policy configs differ outside scanner thinning and its derived edge semantics",
    )


def recursive_difference_paths(
    baseline: Any,
    candidate: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Return sorted dotted paths whose values differ recursively."""

    if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
        differences: list[str] = []
        for key in sorted(set(baseline) | set(candidate), key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in baseline or key not in candidate:
                differences.append(path)
            else:
                differences.extend(
                    recursive_difference_paths(baseline[key], candidate[key], prefix=path)
                )
        return differences
    if isinstance(baseline, np.ndarray) or isinstance(candidate, np.ndarray):
        baseline_array = np.asarray(baseline)
        candidate_array = np.asarray(candidate)
        equal = baseline_array.dtype == candidate_array.dtype and np.array_equal(
            baseline_array, candidate_array, equal_nan=True
        )
    else:
        if type(baseline) is not type(candidate):
            equal = False
        else:
            try:
                equal = bool(baseline == candidate)
            except (TypeError, ValueError):
                equal = False
    return [] if equal else [prefix]


def json_safe(value: Any) -> Any:
    """Convert NumPy-backed report values to strict JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def report_to_json(report: Mapping[str, Any], *, pretty: bool = False) -> str:
    """Serialize a report as deterministic, strict JSON."""

    return (
        json.dumps(
            json_safe(report),
            allow_nan=False,
            indent=2 if pretty else None,
            sort_keys=True,
        )
        + "\n"
    )


def _output_array(outputs: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    if name not in outputs:
        raise KeyError(name)
    values = np.asarray(outputs[name])
    if values.ndim != 3:
        raise ValueError(f"{name} must be a 3D array")
    return values


def _nonzero_fraction(values: np.ndarray, epsilon: float) -> float:
    return float(np.count_nonzero(np.abs(values) > epsilon) / values.size) if values.size else 0.0


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _safe_density_ratio(candidate: Any, baseline: Any) -> float | None:
    candidate_value = _finite_float(candidate)
    baseline_value = _finite_float(baseline)
    if candidate_value is None or baseline_value is None:
        return None
    if baseline_value == 0.0:
        return 1.0 if candidate_value == 0.0 else None
    return float(candidate_value / baseline_value)


def _distribution(values: Sequence[float], *, include_cv: bool = False) -> dict[str, Any]:
    if not values:
        result: dict[str, Any] = {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
        if include_cv:
            result["cv"] = None
        return result
    array = np.asarray(values, dtype=np.float64)
    result = {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    if include_cv:
        result["cv"] = _coefficient_of_variation(values)
    return result


def _coefficient_of_variation(values: Sequence[float]) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    if mean == 0.0:
        return 0.0
    return float(np.std(array) / abs(mean))


def _aggregate_numeric_roots(
    crops: Sequence[Mapping[str, Any]],
    *,
    roots: Sequence[str],
) -> dict[str, Any]:
    values_by_path: dict[str, list[float]] = {}
    for crop in crops:
        for root in roots:
            if root not in crop:
                continue
            for path, value in _flatten_numeric(crop[root], prefix=root):
                number = _finite_float(value)
                if number is not None:
                    values_by_path.setdefault(path, []).append(number)
    metric_paths = sorted(values_by_path)
    return {
        "metric_paths": metric_paths,
        "per_metric_mean": {
            path: float(np.mean(np.asarray(values_by_path[path], dtype=np.float64)))
            for path in metric_paths
        },
        "per_metric_median": {
            path: float(np.median(np.asarray(values_by_path[path], dtype=np.float64)))
            for path in metric_paths
        },
        "per_metric_min": {path: min(values_by_path[path]) for path in metric_paths},
        "per_metric_max": {path: max(values_by_path[path]) for path in metric_paths},
    }


def _flatten_numeric(value: Any, *, prefix: str) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_numeric(item, prefix=path)
        return
    if isinstance(value, (list, tuple, np.ndarray)):
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float, np.generic)):
        yield prefix, value


def _finite_values(values: Iterable[Any]) -> list[float]:
    return [number for value in values if (number := _finite_float(value)) is not None]


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.generic)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else None


def _difference(candidate: Any, baseline: Any) -> float | None:
    candidate_value = _finite_float(candidate)
    baseline_value = _finite_float(baseline)
    if candidate_value is None or baseline_value is None:
        return None
    return float(candidate_value - baseline_value)


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _crop_finite_failure_count(crop: Mapping[str, Any]) -> int:
    checks = _as_mapping(_nested(crop, "finite_checks", "pyosv"))
    failures = 0
    for name in FINITE_STAGE_NAMES:
        report = _as_mapping(checks.get(name.removesuffix(".dat")))
        size = _finite_float(report.get("size"))
        finite_count = _finite_float(report.get("finite_count"))
        if size is None or finite_count is None or finite_count != size:
            failures += 1
    return failures


def _empty_stage_failures(crops: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    stage_keys = {"fet_py.dat": "fet", "fv_py.dat": "fv", "fvt_py.dat": "fvt"}
    for crop_position, crop in enumerate(crops, start=1):
        crop_index = int(crop.get("index", crop_position))
        for output_name, stage in stage_keys.items():
            count = _finite_float(_nested(crop, "stage_density", stage, "nonzero_count"))
            if count is None or count <= 0.0:
                failures.append({"crop_index": crop_index, "stage": output_name})
    return failures


def _crop_empty_stage_count(crop: Mapping[str, Any]) -> int:
    return len(_empty_stage_failures([crop]))


def _paired_metric_deltas(
    baseline_crops: Sequence[Mapping[str, Any]],
    candidate_crops: Sequence[Mapping[str, Any]],
    path: tuple[str, ...],
) -> list[float]:
    deltas: list[float] = []
    for baseline, candidate in zip(baseline_crops, candidate_crops, strict=False):
        delta = _difference(_nested(candidate, *path), _nested(baseline, *path))
        if delta is not None:
            deltas.append(delta)
    return deltas


def _check(passed: bool, *, reason: str, **details: Any) -> dict[str, Any]:
    result = {"passed": bool(passed), **details}
    if not passed:
        result["reason"] = reason
    return result


def _validate_nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.generic)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return number
