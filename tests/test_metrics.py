import numpy as np
import pytest

import pyosv.metrics as metrics


def test_finite_value_report_counts_nonfinite_values() -> None:
    report = metrics.finite_value_report(np.array([[1.0, np.nan], [np.inf, -np.inf]]))

    assert report["shape"] == (2, 2)
    assert report["size"] == 4
    assert report["finite_count"] == 1
    assert report["nan_count"] == 1
    assert report["posinf_count"] == 1
    assert report["neginf_count"] == 1
    assert report["finite_fraction"] == 0.25
    assert report["finite_min"] == 1.0
    assert report["finite_max"] == 1.0
    assert report["finite_mean"] == 1.0


def test_normalized_correlation_perfect_positive() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = a * 2.0 + 5.0

    assert metrics.normalized_correlation(a, b) == pytest.approx(1.0)


def test_normalized_correlation_perfect_negative() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = -a

    assert metrics.normalized_correlation(a, b) == pytest.approx(-1.0)


def test_normalized_correlation_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        metrics.normalized_correlation(np.zeros((2, 2)), np.zeros((2, 3)))


def test_normalized_correlation_constant_arrays_return_zero() -> None:
    a = np.ones((2, 3), dtype=np.float32)
    b = np.arange(6, dtype=np.float32).reshape(2, 3)

    assert metrics.normalized_correlation(a, b) == 0.0
    assert metrics.normalized_correlation(a, a) == 0.0


def test_normalized_correlation_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        metrics.normalized_correlation(np.array([1.0, np.nan]), np.array([1.0, 2.0]))


def test_top_percentile_mask_selects_values_at_or_above_threshold() -> None:
    x = np.arange(10, dtype=np.float32)

    np.testing.assert_array_equal(
        metrics.top_percentile_mask(x, 80.0),
        np.array([False, False, False, False, False, False, False, False, True, True]),
    )


def test_top_percentile_overlap_reports_mask_agreement() -> None:
    a = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float32)
    b = np.array([[0.0, 3.0, 1.0, 2.0]], dtype=np.float32)

    overlap = metrics.top_percentile_overlap(a, b, percentile=50.0)

    assert overlap == {
        "percentile": 50.0,
        "a_count": 2.0,
        "b_count": 2.0,
        "overlap_count": 1.0,
        "union_count": 3.0,
        "a_fraction": 0.5,
        "b_fraction": 0.5,
        "overlap_fraction": 0.25,
        "overlap_over_a": 0.5,
        "overlap_over_b": 0.5,
        "jaccard": pytest.approx(1.0 / 3.0),
    }


def test_top_percentile_overlap_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        metrics.top_percentile_overlap(np.array([1.0, np.inf]), np.array([1.0, 2.0]))
