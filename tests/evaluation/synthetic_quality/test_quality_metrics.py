from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyosv.evaluation.synthetic_quality.config import SyntheticTruthMetricConfig
from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.synthetic_metrics import top_positive_truth_count_mask, top_truth_count_mask


def test_positive_candidate_count_uses_report_epsilon() -> None:
    values = np.array(
        [0.0, quality_metrics.NONZERO_EPSILON, quality_metrics.NONZERO_EPSILON * 2.0],
        dtype=np.float32,
    )

    assert quality_metrics.positive_candidate_count(values) == 1
    np.testing.assert_array_equal(
        quality_metrics.positive_candidate_mask(values), [False, False, True]
    )


def test_array_nonzero_fraction_ignores_sub_epsilon_tails() -> None:
    values = np.array([0.0, 5.0e-8, 2.0e-6, -5.0e-8, -2.0e-6], dtype=np.float32)

    assert quality_metrics.array_nonzero_fraction(values) == 2 / 5


@pytest.mark.parametrize(
    ("values", "expected"),
    (
        (
            np.array(
                [-quality_metrics.NONZERO_EPSILON, quality_metrics.NONZERO_EPSILON],
                dtype=np.float32,
            ),
            0.0,
        ),
        (
            np.array(
                [
                    np.nextafter(np.float32(quality_metrics.NONZERO_EPSILON), np.float32(0.0)),
                    np.nextafter(np.float32(quality_metrics.NONZERO_EPSILON), np.float32(np.inf)),
                    np.nextafter(-np.float32(quality_metrics.NONZERO_EPSILON), np.float32(0.0)),
                    np.nextafter(-np.float32(quality_metrics.NONZERO_EPSILON), np.float32(-np.inf)),
                ],
                dtype=np.float32,
            ),
            0.5,
        ),
        (np.array([], dtype=np.float32), 0.0),
        (np.array([np.iinfo(np.int32).min, 0, 3], dtype=np.int32), 2 / 3),
    ),
)
def test_array_nonzero_fraction_epsilon_boundaries_and_dtypes(
    values: np.ndarray, expected: float
) -> None:
    original = values.copy()

    assert quality_metrics.array_nonzero_fraction(values) == expected
    np.testing.assert_array_equal(values, original)


@pytest.mark.parametrize(
    "values",
    (
        np.array(["not-numeric"]),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
    ),
)
def test_array_nonzero_fraction_rejects_invalid_values(values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="numeric|finite"):
        quality_metrics.array_nonzero_fraction(values)


def test_edge_candidate_fraction_and_zero_denominator() -> None:
    candidates = np.zeros((5, 5, 5), dtype=bool)
    candidates[0, 2, 2] = True
    candidates[2, 2, 2] = True

    assert quality_metrics.edge_candidate_fraction(candidates, edge_margin=1) == 0.5
    assert quality_metrics.edge_candidate_fraction(np.zeros_like(candidates), edge_margin=1) == 0.0
    assert quality_metrics.fraction_or_zero(4, 0) == 0.0


def test_top_truth_and_positive_truth_count_quality() -> None:
    truth = np.zeros((3, 3, 3), dtype=bool)
    truth[1, 1, 1] = True
    truth[1, 0, 1] = True
    likelihood = np.zeros(truth.shape, dtype=np.float32)
    likelihood[1, 1, 2] = 0.9
    top_truth = top_truth_count_mask(likelihood, truth)
    top_positive_truth = top_positive_truth_count_mask(likelihood, truth)

    top_quality = quality_metrics.top_truth_count_quality(
        top_truth,
        truth_fault_mask=truth,
        truth_surface_mask=truth,
        buffer_radius=0.0,
    )
    positive_quality = quality_metrics.top_truth_count_quality(
        top_positive_truth,
        truth_fault_mask=truth,
        truth_surface_mask=truth,
        buffer_radius=0.0,
    )

    assert top_quality["buffered_overlap_radius2"]["candidate_count"] == 2
    assert positive_quality["buffered_overlap_radius2"]["candidate_count"] == 1


def test_scanner_input_association() -> None:
    scanner_input = np.array([[[1.0, 2.0, 5.0]]], dtype=np.float32)
    truth = np.array([[[True, True, False]]])
    far = np.array([[[False, False, True]]])

    association = quality_metrics.scanner_input_association(
        scanner_input, truth_surface_mask=truth, far_from_truth_mask=far
    )

    assert association == {
        "truth_surface_mean": 1.5,
        "far_from_truth_mean": 5.0,
        "contrast": 3.5,
    }


def test_scanner_truth_quality_includes_orientation() -> None:
    shape = (3, 3, 3)
    truth_fault = np.zeros(shape, dtype=bool)
    truth_fault[1, 1, 1] = True
    distance = np.full(shape, 4.0, dtype=np.float32)
    distance[1, 1, 1] = 0.0
    truth_strike = np.full(shape, 30.0, dtype=np.float32)
    truth_dip = np.full(shape, 60.0, dtype=np.float32)
    ft = truth_fault.astype(np.float32)
    case = SimpleNamespace(
        truth_fault_mask=truth_fault,
        truth_distance=distance,
        truth_strike=truth_strike,
        truth_dip=truth_dip,
    )
    volumes = {
        "scanner_input": distance,
        "scanner_ft": ft,
        "scanner_fet": ft,
        "scanner_pt": truth_strike,
        "scanner_tt": truth_dip,
        "scanner_fpt": truth_strike,
        "scanner_ftt": truth_dip,
    }

    quality = quality_metrics.scanner_truth_quality(
        case,
        scanner_volumes=volumes,
        truth_metric_config=SyntheticTruthMetricConfig(),
    )

    assert quality["orientation_error"]["raw_scan_top_truth_count"]["strike_median"] == 0.0
    assert quality["orientation_error"]["used_attributes_top_truth_count"]["dip_median"] == 0.0


def test_nested_metric_delta_and_missing_value() -> None:
    report = {"quality": {"skin": {"score": 0.75}}}

    value = quality_metrics.metric_value(report, ("quality", "skin", "score"))

    assert value == 0.75
    assert quality_metrics.delta_or_none(value, 0.5) == 0.25
    assert quality_metrics.metric_value(report, ("quality", "missing")) is None
    assert quality_metrics.delta_or_none(None, 0.5) is None


def test_skin_metric_normalization_does_not_mutate_input() -> None:
    overlap = {"buffered_f1": 1.0}
    metrics = {"buffered_overlap_radius3": overlap, "topology": {"skin_count": 1}}

    normalized = quality_metrics.normalize_report_skin_metric_keys(metrics)

    assert metrics == {"buffered_overlap_radius3": overlap, "topology": {"skin_count": 1}}
    assert normalized["buffered_overlap_radius2"] is overlap
