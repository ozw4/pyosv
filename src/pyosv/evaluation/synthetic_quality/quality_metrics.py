"""Pure metric composition for synthetic-quality reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from pyosv.synthetic_metrics import (
    buffered_surface_overlap,
    edge_false_positive_ratio,
    masked_orientation_error,
    surface_distance_metrics,
    top_truth_count_mask,
)

from .config import (
    SyntheticTruthMetricConfig,
    _validate_nonnegative_finite_scalar,
)
from .variants import BASELINE_VARIANT


EDGE_FALSE_POSITIVE_MARGIN = 2
NONZERO_EPSILON = 1.0e-6
VARIANT_COMPARISON_METRICS = (
    (
        "fvt_buffered_f1_r2_delta_vs_current",
        ("quality", "fvt_top_truth_count", "buffered_overlap_radius2", "buffered_f1"),
    ),
    (
        "fvt_candidate_to_truth_p95_delta_vs_current",
        ("quality", "fvt_top_truth_count", "surface_distance", "candidate_to_truth_p95"),
    ),
    (
        "fvt_strike_median_error_delta_vs_current",
        ("quality", "fvt_top_truth_count", "orientation_error", "strike_median"),
    ),
    (
        "fvt_dip_median_error_delta_vs_current",
        ("quality", "fvt_top_truth_count", "orientation_error", "dip_median"),
    ),
    (
        "fv_buffered_f1_r2_delta_vs_current",
        ("quality", "fv_top_truth_count", "buffered_overlap_radius2", "buffered_f1"),
    ),
    (
        "skin_buffered_f1_r2_delta_vs_current",
        ("quality", "skin", "buffered_overlap_radius2", "buffered_f1"),
    ),
    (
        "skin_candidate_to_truth_p95_delta_vs_current",
        ("quality", "skin", "surface_distance", "candidate_to_truth_p95"),
    ),
    (
        "skin_strike_median_error_delta_vs_current",
        ("quality", "skin", "orientation_error", "strike_median"),
    ),
    (
        "skin_dip_median_error_delta_vs_current",
        ("quality", "skin", "orientation_error", "dip_median"),
    ),
    (
        "skin_count_delta_vs_current",
        ("quality", "skin", "topology", "skin_count"),
    ),
)


def truth_report(case: Any, truth_metric_config: SyntheticTruthMetricConfig) -> dict[str, int]:
    truth_surface_half_width = _validate_nonnegative_finite_scalar(
        truth_metric_config.truth_surface_half_width,
        "truth_surface_half_width",
    )
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(truth_surface_half_width)
    truth_fault_mask = np.asarray(case.truth_fault_mask, dtype=bool)
    return {
        "fault_voxel_count": int(np.count_nonzero(truth_fault_mask)),
        "surface_voxel_count": int(np.count_nonzero(truth_surface_mask)),
    }


def scanner_truth_quality(
    case: Any,
    *,
    scanner_volumes: Mapping[str, np.ndarray],
    truth_metric_config: SyntheticTruthMetricConfig,
) -> dict[str, Any]:
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
    far_from_truth_mask = np.abs(case.truth_distance) >= np.float32(
        max(3.0, truth_surface_half_width + 2.0)
    )
    raw_ft_top_truth_count = top_truth_count_mask(scanner_volumes["scanner_ft"], truth_surface_mask)
    used_ft_top_truth_count = top_truth_count_mask(
        scanner_volumes["scanner_fet"], truth_surface_mask
    )
    return {
        "ft_top_truth_count": top_truth_count_quality(
            raw_ft_top_truth_count,
            truth_fault_mask=truth_fault_mask,
            truth_surface_mask=truth_surface_mask,
            buffer_radius=buffer_radius,
        ),
        "orientation_error": {
            "raw_scan_top_truth_count": masked_orientation_error(
                scanner_volumes["scanner_pt"],
                scanner_volumes["scanner_tt"],
                case.truth_strike,
                case.truth_dip,
                raw_ft_top_truth_count,
            ),
            "used_attributes_top_truth_count": masked_orientation_error(
                scanner_volumes["scanner_fpt"],
                scanner_volumes["scanner_ftt"],
                case.truth_strike,
                case.truth_dip,
                used_ft_top_truth_count,
            ),
        },
        "input_association": scanner_input_association(
            scanner_volumes["scanner_input"],
            truth_surface_mask=truth_surface_mask,
            far_from_truth_mask=far_from_truth_mask,
        ),
    }


def top_truth_count_quality(
    candidate_mask: np.ndarray,
    *,
    truth_fault_mask: np.ndarray,
    truth_surface_mask: np.ndarray,
    buffer_radius: float,
) -> dict[str, Any]:
    return {
        "buffered_overlap_radius2": buffered_surface_overlap(
            candidate_mask, truth_fault_mask, radius=buffer_radius
        ),
        "surface_distance": surface_distance_metrics(candidate_mask, truth_surface_mask),
    }


def positive_candidate_count(array: np.ndarray) -> int:
    return candidate_count(positive_candidate_mask(array))


def positive_candidate_mask(array: np.ndarray) -> np.ndarray:
    return np.asarray(array) > np.float32(NONZERO_EPSILON)


def candidate_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(mask, dtype=bool)))


def positive_pair_overlap(
    *,
    candidate_name: str,
    reference_name: str,
    candidate_mask: np.ndarray,
    reference_mask: np.ndarray,
    buffer_radius: float,
) -> dict[str, str | float | int]:
    return {
        "candidate_mask": candidate_name,
        "reference_mask": reference_name,
        **buffered_surface_overlap(candidate_mask, reference_mask, radius=buffer_radius),
    }


def fraction_or_zero(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def edge_candidate_fraction(candidate_mask: np.ndarray, *, edge_margin: int) -> float:
    candidates = np.asarray(candidate_mask, dtype=bool)
    count = int(np.count_nonzero(candidates))
    if count == 0:
        return 0.0
    return float(np.count_nonzero(candidates & edge_mask(candidates.shape, edge_margin)) / count)


def edge_mask(shape: tuple[int, ...], margin: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if margin <= 0 or mask.size == 0:
        return mask
    for axis, size in enumerate(shape):
        width = min(int(margin), int(size))
        lower = [slice(None)] * len(shape)
        upper = [slice(None)] * len(shape)
        lower[axis] = slice(0, width)
        upper[axis] = slice(size - width, size)
        mask[tuple(lower)] = True
        mask[tuple(upper)] = True
    return mask


def scanner_input_association(
    scanner_input: np.ndarray,
    *,
    truth_surface_mask: np.ndarray,
    far_from_truth_mask: np.ndarray,
) -> dict[str, float]:
    input_array = np.asarray(scanner_input, dtype=np.float64)
    truth_mean = masked_mean(input_array, truth_surface_mask)
    far_mean = masked_mean(input_array, far_from_truth_mask)
    return {
        "truth_surface_mean": truth_mean,
        "far_from_truth_mean": far_mean,
        "contrast": float(far_mean - truth_mean),
    }


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    sample_mask = np.asarray(mask, dtype=bool)
    if values.shape != sample_mask.shape:
        raise ValueError(f"array shapes must match, got {values.shape} and {sample_mask.shape}")
    if not np.any(sample_mask):
        return 0.0
    samples = values[sample_mask]
    if not np.all(np.isfinite(samples)):
        raise ValueError("masked values must contain only finite values")
    return float(np.mean(samples))


def scanner_stage_metric(
    candidate_mask: np.ndarray,
    *,
    truth_fault_mask: np.ndarray,
    truth_surface_mask: np.ndarray,
    buffer_radius: float,
) -> dict[str, int | float]:
    candidate = np.asarray(candidate_mask, dtype=bool)
    overlap = buffered_surface_overlap(candidate, truth_fault_mask, radius=buffer_radius)
    distance = surface_distance_metrics(candidate, truth_surface_mask)
    edge_false_positive = edge_false_positive_ratio(
        candidate,
        truth_surface_mask,
        edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        truth_buffer_radius=buffer_radius,
    )
    return {
        "candidate_count": int(overlap["candidate_count"]),
        "edge_shell_fraction": edge_candidate_fraction(
            candidate, edge_margin=EDGE_FALSE_POSITIVE_MARGIN
        ),
        "truth_buffered_f1_r2": float(overlap["buffered_f1"]),
        "candidate_to_truth_p95": float(distance["candidate_to_truth_p95"]),
        "truth_to_candidate_p95": float(distance["truth_to_candidate_p95"]),
        "edge_false_positive_fraction_of_candidates": float(
            edge_false_positive["edge_false_positive_fraction_of_candidates"]
        ),
    }


def scanner_stage_transition_metric(
    *, source_mask: np.ndarray, target_mask: np.ndarray, buffer_radius: float
) -> dict[str, int | float]:
    source = np.asarray(source_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    source_count = candidate_count(source)
    target_count = candidate_count(target)
    overlap = buffered_surface_overlap(target, source, radius=buffer_radius)
    distance = surface_distance_metrics(target, source)
    return {
        "source_count": source_count,
        "target_count": target_count,
        "target_to_source_count_ratio": fraction_or_zero(target_count, source_count),
        "buffered_f1_r2": float(overlap["buffered_f1"]),
        "target_to_source_distance_p95": float(distance["candidate_to_truth_p95"]),
    }


def variant_comparison(
    variant_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if BASELINE_VARIANT not in variant_reports:
        return {"baseline_variant": None, "variants": {}}
    baseline_values = {
        name: metric_value(variant_reports[BASELINE_VARIANT], path)
        for name, path in VARIANT_COMPARISON_METRICS
    }
    comparison = {
        variant: {
            name: delta_or_none(metric_value(report, path), baseline_values[name])
            for name, path in VARIANT_COMPARISON_METRICS
        }
        for variant, report in variant_reports.items()
    }
    return {"baseline_variant": BASELINE_VARIANT, "variants": comparison}


def metric_value(report: Mapping[str, Any], path: Sequence[str]) -> float | None:
    value: Any = report
    for key in path:
        if value is None or not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return None if value is None else float(value)


def delta_or_none(value: float | None, baseline_value: float | None) -> float | None:
    if value is None or baseline_value is None:
        return None
    return float(value - baseline_value)


def normalize_report_skin_metric_keys(metrics: Mapping[str, Any]) -> dict[str, Any]:
    report_metrics = dict(metrics)
    if "buffered_overlap_radius2" in report_metrics:
        return report_metrics
    buffered_keys = [
        key for key in report_metrics if str(key).startswith("buffered_overlap_radius")
    ]
    if len(buffered_keys) != 1:
        raise ValueError("skin metrics must include exactly one buffered overlap metric")
    report_metrics["buffered_overlap_radius2"] = report_metrics.pop(buffered_keys[0])
    return report_metrics
