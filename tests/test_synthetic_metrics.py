import numpy as np
import pytest

from pyosv.synthetic_metrics import (
    buffered_surface_overlap,
    masked_orientation_error,
    surface_distance_metrics,
    top_k_mask,
    top_truth_count_mask,
)


def test_top_k_mask_is_deterministic_and_selects_k_values() -> None:
    values = np.array([[5.0, 2.0, 5.0], [1.0, 4.0, 5.0]], dtype=np.float32)

    first = top_k_mask(values, 3)
    second = top_k_mask(values, 3)

    assert first.dtype == np.bool_
    assert first.shape == values.shape
    assert int(np.count_nonzero(first)) == 3
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(
        first,
        np.array([[True, False, True], [False, False, True]]),
    )


def test_top_k_mask_handles_empty_and_full_counts() -> None:
    values = np.arange(4, dtype=np.float32).reshape(2, 2)

    assert not np.any(top_k_mask(values, 0))
    assert np.all(top_k_mask(values, values.size))
    assert np.all(top_k_mask(values, values.size + 1))


def test_top_truth_count_mask_matches_truth_voxel_count() -> None:
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    truth = np.array([[False, True, False], [True, True, False]])

    mask = top_truth_count_mask(values, truth)

    assert int(np.count_nonzero(mask)) == int(np.count_nonzero(truth))


def test_buffered_surface_overlap_identical_masks_are_perfect() -> None:
    truth = np.zeros((3, 3, 3), dtype=bool)
    truth[1, 1, 1] = True

    overlap = buffered_surface_overlap(truth, truth, radius=1.0)

    assert overlap["precision"] == 1.0
    assert overlap["recall"] == 1.0
    assert overlap["f1"] == 1.0
    assert overlap["jaccard"] == 1.0
    assert overlap["buffered_precision"] == 1.0
    assert overlap["buffered_recall"] == 1.0
    assert overlap["buffered_f1"] == 1.0


def test_buffered_surface_overlap_one_voxel_shift_is_recovered_by_radius() -> None:
    truth = np.zeros((3, 3, 3), dtype=bool)
    candidate = np.zeros_like(truth)
    truth[1, 1, 1] = True
    candidate[1, 1, 2] = True

    overlap = buffered_surface_overlap(candidate, truth, radius=1.0)

    assert overlap["f1"] == 0.0
    assert overlap["jaccard"] == 0.0
    assert overlap["buffered_precision"] == 1.0
    assert overlap["buffered_recall"] == 1.0
    assert overlap["buffered_f1"] == 1.0


def test_buffered_surface_overlap_empty_mask_conventions() -> None:
    empty = np.zeros((2, 2), dtype=bool)
    truth = np.array([[True, False], [False, False]])

    both_empty = buffered_surface_overlap(empty, empty, radius=1.0)
    assert both_empty["precision"] == 1.0
    assert both_empty["recall"] == 1.0
    assert both_empty["f1"] == 1.0
    assert both_empty["buffered_f1"] == 1.0

    missing_candidate = buffered_surface_overlap(empty, truth, radius=1.0)
    assert missing_candidate["precision"] == 1.0
    assert missing_candidate["recall"] == 0.0
    assert missing_candidate["buffered_precision"] == 1.0
    assert missing_candidate["buffered_recall"] == 0.0

    false_positive = buffered_surface_overlap(truth, empty, radius=1.0)
    assert false_positive["precision"] == 0.0
    assert false_positive["recall"] == 1.0
    assert false_positive["buffered_precision"] == 0.0
    assert false_positive["buffered_recall"] == 1.0


def test_surface_distance_metrics_known_small_arrays() -> None:
    truth = np.zeros((3, 3), dtype=bool)
    candidate = np.zeros_like(truth)
    truth[1, 1] = True
    candidate[1, 2] = True

    distances = surface_distance_metrics(candidate, truth)

    assert distances["candidate_count"] == 1
    assert distances["truth_count"] == 1
    assert distances["candidate_to_truth_mean"] == pytest.approx(1.0)
    assert distances["candidate_to_truth_median"] == pytest.approx(1.0)
    assert distances["candidate_to_truth_p90"] == pytest.approx(1.0)
    assert distances["candidate_to_truth_p95"] == pytest.approx(1.0)
    assert distances["truth_to_candidate_mean"] == pytest.approx(1.0)
    assert distances["symmetric_chamfer_mean"] == pytest.approx(1.0)
    assert distances["hausdorff_p95"] == pytest.approx(1.0)


def test_surface_distance_metrics_empty_masks_use_finite_penalty() -> None:
    empty = np.zeros((3, 4), dtype=bool)
    truth = np.zeros_like(empty)
    truth[0, 0] = True

    both_empty = surface_distance_metrics(empty, empty)
    assert both_empty["symmetric_chamfer_mean"] == 0.0
    assert both_empty["hausdorff_p95"] == 0.0

    one_empty = surface_distance_metrics(empty, truth)
    penalty = np.sqrt((3 - 1) ** 2 + (4 - 1) ** 2)
    assert one_empty["candidate_to_truth_mean"] == pytest.approx(penalty)
    assert one_empty["truth_to_candidate_mean"] == pytest.approx(penalty)
    assert one_empty["symmetric_chamfer_mean"] == pytest.approx(penalty)
    assert one_empty["hausdorff_p95"] == pytest.approx(penalty)


def test_masked_orientation_error_wraps_strike_at_180_degrees() -> None:
    predicted_strike = np.array([179.0, 10.0], dtype=np.float32)
    truth_strike = np.array([1.0, 20.0], dtype=np.float32)
    predicted_dip = np.array([45.0, 70.0], dtype=np.float32)
    truth_dip = np.array([40.0, 60.0], dtype=np.float32)
    mask = np.array([True, False])

    errors = masked_orientation_error(
        predicted_strike,
        predicted_dip,
        truth_strike,
        truth_dip,
        mask,
    )

    assert errors["count"] == 1
    assert errors["strike_mean"] == pytest.approx(2.0)
    assert errors["strike_median"] == pytest.approx(2.0)
    assert errors["strike_p90"] == pytest.approx(2.0)
    assert errors["strike_p95"] == pytest.approx(2.0)
    assert errors["dip_mean"] == pytest.approx(5.0)


def test_masked_orientation_error_empty_mask_returns_zero_summaries() -> None:
    values = np.array([1.0], dtype=np.float32)

    errors = masked_orientation_error(values, values, values, values, np.array([False]))

    assert errors == {
        "count": 0,
        "strike_mean": 0.0,
        "strike_median": 0.0,
        "strike_p90": 0.0,
        "strike_p95": 0.0,
        "dip_mean": 0.0,
        "dip_median": 0.0,
        "dip_p90": 0.0,
        "dip_p95": 0.0,
    }


def test_synthetic_metrics_reject_shape_mismatch_and_invalid_radius() -> None:
    mask = np.zeros((2, 2), dtype=bool)
    mismatched = np.zeros((2, 3), dtype=bool)

    with pytest.raises(ValueError, match="shapes must match"):
        buffered_surface_overlap(mask, mismatched, radius=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        buffered_surface_overlap(mask, mask, radius=-1.0)
    with pytest.raises(ValueError, match="finite"):
        buffered_surface_overlap(mask, mask, radius=np.inf)
    with pytest.raises(ValueError, match="shapes must match"):
        surface_distance_metrics(mask, mismatched)
    with pytest.raises(ValueError, match="shapes must match"):
        top_truth_count_mask(np.zeros((2, 2), dtype=np.float32), mismatched)
    with pytest.raises(ValueError, match="shapes must match"):
        masked_orientation_error(mask, mask, mask, mask, mismatched)
