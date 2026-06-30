import json

import numpy as np
import pytest

from pyosv.metrics import (
    buffered_ridge_overlap,
    sparse_ridge_distance_metrics,
    top_percentile_mask,
)


def _line_2d(row: int, shape: tuple[int, int] = (9, 9)) -> np.ndarray:
    values = np.zeros(shape, dtype=np.float32)
    values[row, 1:-1] = 1.0
    return values


def _plane_3d(index: int, shape: tuple[int, int, int] = (5, 7, 7)) -> np.ndarray:
    values = np.zeros(shape, dtype=np.float32)
    values[:, index, 1:-1] = 1.0
    return values


def test_top_percentile_mask_positive_only_returns_empty_for_all_zero() -> None:
    mask = top_percentile_mask(np.zeros((4, 5), dtype=np.float32), percentile=99.0)

    assert mask.dtype == np.bool_
    assert mask.shape == (4, 5)
    assert not np.any(mask)


def test_identical_2d_line_ridges_have_perfect_overlap_and_zero_distance() -> None:
    reference = _line_2d(4)
    candidate = reference.copy()

    overlap = buffered_ridge_overlap(reference, candidate, percentile=99.0, radius=2.0)
    distances = sparse_ridge_distance_metrics(reference, candidate, percentile=99.0)

    assert overlap["reference_count"] == 7
    assert overlap["candidate_count"] == 7
    assert overlap["intersection_count"] == 7
    assert overlap["precision"] == 1.0
    assert overlap["recall"] == 1.0
    assert overlap["f1"] == 1.0
    assert overlap["jaccard"] == 1.0
    assert overlap["buffered_precision"] == 1.0
    assert overlap["buffered_recall"] == 1.0
    assert overlap["buffered_f1"] == 1.0
    assert distances["candidate_to_reference_mean"] == 0.0
    assert distances["candidate_to_reference_median"] == 0.0
    assert distances["candidate_to_reference_p90"] == 0.0
    assert distances["candidate_to_reference_p95"] == 0.0
    assert distances["reference_to_candidate_mean"] == 0.0
    assert distances["reference_to_candidate_median"] == 0.0
    assert distances["reference_to_candidate_p90"] == 0.0
    assert distances["reference_to_candidate_p95"] == 0.0
    json.dumps(overlap)
    json.dumps(distances)


def test_shifted_2d_line_ridges_use_buffer_and_report_known_distance() -> None:
    reference = _line_2d(3)
    candidate = _line_2d(5)

    exact = buffered_ridge_overlap(reference, candidate, percentile=99.0, radius=0.0)
    buffered = buffered_ridge_overlap(reference, candidate, percentile=99.0, radius=2.0)
    distances = sparse_ridge_distance_metrics(reference, candidate, percentile=99.0)

    assert exact["precision"] == 0.0
    assert exact["recall"] == 0.0
    assert exact["buffered_precision"] == 0.0
    assert exact["buffered_recall"] == 0.0
    assert buffered["precision"] == 0.0
    assert buffered["recall"] == 0.0
    assert buffered["buffered_precision"] == 1.0
    assert buffered["buffered_recall"] == 1.0
    assert distances["candidate_to_reference_median"] == 2.0
    assert distances["candidate_to_reference_p90"] == 2.0
    assert distances["reference_to_candidate_median"] == 2.0
    assert distances["reference_to_candidate_p95"] == 2.0


def test_shifted_3d_ridges_use_buffer_and_report_known_distance() -> None:
    reference = _plane_3d(2)
    candidate = _plane_3d(3)

    overlap = buffered_ridge_overlap(reference, candidate, percentile=99.0, radius=1.0)
    distances = sparse_ridge_distance_metrics(reference, candidate, percentile=99.0)

    assert overlap["reference_count"] == 25
    assert overlap["candidate_count"] == 25
    assert overlap["intersection_count"] == 0
    assert overlap["precision"] == 0.0
    assert overlap["recall"] == 0.0
    assert overlap["buffered_precision"] == 1.0
    assert overlap["buffered_recall"] == 1.0
    assert distances["candidate_to_reference_median"] == 1.0
    assert distances["reference_to_candidate_median"] == 1.0


def test_empty_ridge_metrics_are_stable() -> None:
    reference = np.zeros((4, 4), dtype=np.float32)
    candidate = np.zeros((4, 4), dtype=np.float32)

    overlap = buffered_ridge_overlap(reference, candidate)
    distances = sparse_ridge_distance_metrics(reference, candidate)

    assert overlap["reference_count"] == 0
    assert overlap["candidate_count"] == 0
    assert overlap["precision"] == 0.0
    assert overlap["recall"] == 0.0
    assert overlap["f1"] == 0.0
    assert overlap["jaccard"] == 0.0
    assert overlap["buffered_precision"] == 0.0
    assert overlap["buffered_recall"] == 0.0
    assert overlap["buffered_f1"] == 0.0
    assert distances["reference_count"] == 0
    assert distances["candidate_count"] == 0
    assert distances["candidate_to_reference_mean"] is None
    assert distances["reference_to_candidate_p95"] is None


def test_one_empty_ridge_distance_metrics_are_none() -> None:
    distances = sparse_ridge_distance_metrics(
        np.zeros((6, 6), dtype=np.float32), _line_2d(3, shape=(6, 6))
    )

    assert distances["reference_count"] == 0
    assert distances["candidate_count"] == 4
    assert distances["candidate_to_reference_mean"] is None
    assert distances["candidate_to_reference_median"] is None
    assert distances["reference_to_candidate_mean"] is None
    assert distances["reference_to_candidate_median"] is None


def test_ridge_metrics_reject_nonfinite_arrays() -> None:
    reference = _line_2d(3)
    candidate = _line_2d(3)
    candidate[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        top_percentile_mask(candidate, percentile=99.0)
    with pytest.raises(ValueError, match="finite"):
        buffered_ridge_overlap(reference, candidate)
    with pytest.raises(ValueError, match="finite"):
        sparse_ridge_distance_metrics(reference, candidate)


def test_ridge_metrics_reject_invalid_percentile_and_radius() -> None:
    reference = _line_2d(3)
    candidate = _line_2d(4)

    with pytest.raises(ValueError, match="percentile"):
        top_percentile_mask(reference, percentile=101.0)
    with pytest.raises(ValueError, match="percentile"):
        buffered_ridge_overlap(reference, candidate, percentile=-1.0)
    with pytest.raises(ValueError, match="percentile"):
        sparse_ridge_distance_metrics(reference, candidate, percentile=np.nan)
    with pytest.raises(ValueError, match="radius"):
        buffered_ridge_overlap(reference, candidate, radius=-1.0)
    with pytest.raises(ValueError, match="radius"):
        buffered_ridge_overlap(reference, candidate, radius=np.inf)
