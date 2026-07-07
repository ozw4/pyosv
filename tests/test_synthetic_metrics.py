import numpy as np
import pytest

from pyosv.cells import FaultCell
from pyosv.skin import FaultSkin
from pyosv.synthetic_metrics import (
    buffered_surface_overlap,
    edge_false_positive_ratio,
    masked_orientation_error,
    skin_mask_from_skins,
    skin_orientation_error,
    skin_topology_metrics,
    skin_truth_metrics,
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


def test_edge_false_positive_ratio_counts_no_false_positives_inside_truth_buffer() -> None:
    truth = np.zeros((5, 5), dtype=bool)
    candidate = np.zeros_like(truth)
    truth[0, 1] = True
    candidate[0, 1] = True

    metrics = edge_false_positive_ratio(
        candidate,
        truth,
        edge_margin=0,
        truth_buffer_radius=0.0,
    )

    assert metrics == {
        "candidate_count": 1,
        "edge_candidate_count": 1,
        "edge_false_positive_count": 0,
        "edge_candidate_fraction": 1.0,
        "edge_false_positive_fraction_of_candidates": 0.0,
        "edge_false_positive_fraction_of_edge_candidates": 0.0,
        "edge_margin": 0,
        "truth_buffer_radius": 0.0,
    }


def test_edge_false_positive_ratio_counts_edge_candidates_outside_truth_buffer() -> None:
    truth = np.zeros((5, 5), dtype=bool)
    candidate = np.zeros_like(truth)
    truth[2, 2] = True
    candidate[0, 0] = True
    candidate[2, 2] = True

    metrics = edge_false_positive_ratio(
        candidate,
        truth,
        edge_margin=0,
        truth_buffer_radius=1.0,
    )

    assert metrics["candidate_count"] == 2
    assert metrics["edge_candidate_count"] == 1
    assert metrics["edge_false_positive_count"] == 1
    assert metrics["edge_candidate_fraction"] == pytest.approx(0.5)
    assert metrics["edge_false_positive_fraction_of_candidates"] == pytest.approx(0.5)
    assert metrics["edge_false_positive_fraction_of_edge_candidates"] == pytest.approx(1.0)
    assert metrics["edge_margin"] == 0
    assert metrics["truth_buffer_radius"] == 1.0


def test_edge_false_positive_ratio_empty_candidate_returns_zero_fractions() -> None:
    truth = np.zeros((3, 3), dtype=bool)
    candidate = np.zeros_like(truth)
    truth[1, 1] = True

    metrics = edge_false_positive_ratio(
        candidate,
        truth,
        edge_margin=1,
        truth_buffer_radius=1.0,
    )

    assert metrics["candidate_count"] == 0
    assert metrics["edge_candidate_count"] == 0
    assert metrics["edge_false_positive_count"] == 0
    assert metrics["edge_candidate_fraction"] == 0.0
    assert metrics["edge_false_positive_fraction_of_candidates"] == 0.0
    assert metrics["edge_false_positive_fraction_of_edge_candidates"] == 0.0


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
    with pytest.raises(ValueError, match="shapes must match"):
        edge_false_positive_ratio(mask, mismatched, edge_margin=1, truth_buffer_radius=1.0)
    with pytest.raises(ValueError, match="non-negative integer"):
        edge_false_positive_ratio(mask, mask, edge_margin=-1, truth_buffer_radius=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        edge_false_positive_ratio(mask, mask, edge_margin=1, truth_buffer_radius=-1.0)
    with pytest.raises(ValueError, match="finite"):
        edge_false_positive_ratio(mask, mask, edge_margin=1, truth_buffer_radius=np.nan)


def test_skin_mask_from_skins_uses_i3_i2_i1_indexing() -> None:
    skin = FaultSkin.from_cells(
        [
            FaultCell(1.0, 2.0, 0.0, 0.8, 10.0, 60.0),
            FaultCell(2.0, 2.0, 1.0, 0.7, 20.0, 50.0),
            FaultCell(3.0, 1.0, 2.0, 0.6, 30.0, 40.0),
        ]
    )

    mask = skin_mask_from_skins([skin], (3, 4, 5))

    assert mask.dtype == np.bool_
    assert mask.shape == (3, 4, 5)
    assert int(np.count_nonzero(mask)) == 3
    assert mask[0, 2, 1]
    assert mask[1, 2, 2]
    assert mask[2, 1, 3]


def test_empty_skins_return_empty_mask_and_zero_topology_counts() -> None:
    mask = skin_mask_from_skins([], (2, 3, 4))
    topology = skin_topology_metrics([], (2, 3, 4))

    assert not np.any(mask)
    assert topology == {
        "skin_count": 0,
        "cell_count": 0,
        "unique_cell_count": 0,
        "duplicate_cell_count": 0,
        "largest_skin_size": 0,
        "largest_skin_fraction": 0.0,
        "small_skin_size": 10,
        "small_skin_count": 0,
        "small_skin_cell_count": 0,
        "small_skin_cell_fraction": 0.0,
    }


def test_skin_topology_metrics_counts_duplicate_cells() -> None:
    first = FaultCell(1.0, 1.0, 1.0, 0.8, 10.0, 60.0)
    duplicate = FaultCell(1.0, 1.0, 1.0, 0.6, 20.0, 50.0)
    second = FaultCell(2.0, 1.0, 1.0, 0.7, 30.0, 40.0)
    skins = [
        FaultSkin.from_cells([first, duplicate]),
        FaultSkin.from_cells([second]),
    ]

    topology = skin_topology_metrics(skins, (3, 3, 4), small_skin_size=2)

    assert topology["skin_count"] == 2
    assert topology["cell_count"] == 3
    assert topology["unique_cell_count"] == 2
    assert topology["duplicate_cell_count"] == 1
    assert topology["largest_skin_size"] == 2
    assert topology["largest_skin_fraction"] == pytest.approx(2 / 3)
    assert topology["small_skin_count"] == 1
    assert topology["small_skin_cell_count"] == 1
    assert topology["small_skin_cell_fraction"] == pytest.approx(1 / 3)


def test_skin_mask_from_skins_rejects_out_of_bounds_cell() -> None:
    skin = FaultSkin.from_cells([FaultCell(4.0, 0.0, 0.0, 0.8, 10.0, 60.0)])

    with pytest.raises(ValueError, match="outside volume"):
        skin_mask_from_skins([skin], (2, 3, 4))


def test_skin_orientation_error_wraps_strike_at_180_degrees() -> None:
    truth_strike = np.zeros((2, 2, 2), dtype=np.float32)
    truth_dip = np.zeros_like(truth_strike)
    truth_strike[0, 0, 0] = 1.0
    truth_strike[0, 0, 1] = 20.0
    truth_dip[0, 0, 0] = 40.0
    truth_dip[0, 0, 1] = 60.0
    skin = FaultSkin.from_cells(
        [
            FaultCell(0.0, 0.0, 0.0, 0.8, 179.0, 45.0),
            FaultCell(1.0, 0.0, 0.0, 0.8, 10.0, 70.0),
        ]
    )

    errors = skin_orientation_error([skin], truth_strike, truth_dip)

    assert errors["count"] == 2
    assert errors["strike_mean"] == pytest.approx(6.0)
    assert errors["strike_median"] == pytest.approx(6.0)
    assert errors["strike_p90"] == pytest.approx(9.2)
    assert errors["dip_mean"] == pytest.approx(7.5)


def test_skin_truth_metrics_returns_expected_metric_families() -> None:
    shape = (3, 3, 4)
    truth_fault_mask = np.zeros(shape, dtype=bool)
    truth_surface_mask = np.zeros(shape, dtype=bool)
    truth_strike = np.zeros(shape, dtype=np.float32)
    truth_dip = np.zeros(shape, dtype=np.float32)
    truth_fault_mask[1, 1, 1] = True
    truth_surface_mask[1, 1, 1] = True
    skin = FaultSkin.from_cells([FaultCell(1.0, 1.0, 1.0, 0.8, 0.0, 0.0)])

    metrics = skin_truth_metrics(
        [skin],
        shape=shape,
        truth_fault_mask=truth_fault_mask,
        truth_surface_mask=truth_surface_mask,
        truth_strike=truth_strike,
        truth_dip=truth_dip,
        buffer_radius=2.0,
    )

    assert set(metrics) == {
        "topology",
        "buffered_overlap_radius2",
        "surface_distance",
        "orientation_error",
    }
    assert "buffered_f1" in metrics["buffered_overlap_radius2"]
    assert "candidate_to_truth_p95" in metrics["surface_distance"]
    assert "strike_median" in metrics["orientation_error"]


def test_skin_metrics_reject_invalid_shape_truth_arrays_and_orientation() -> None:
    skin = FaultSkin.from_cells([FaultCell(0.0, 0.0, 0.0, 0.8, 10.0, 60.0)])
    truth_strike = np.zeros((2, 2, 2), dtype=np.float32)
    truth_dip = np.zeros_like(truth_strike)

    with pytest.raises(ValueError, match="positive 3D tuple"):
        skin_mask_from_skins([], (2, 2))
    with pytest.raises(ValueError, match="truth arrays must be 3D"):
        skin_orientation_error([skin], np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="array shapes must match"):
        skin_orientation_error([skin], truth_strike, np.zeros((2, 2, 3), dtype=np.float32))

    nonfinite = FaultSkin.from_cells([FaultCell(0.0, 0.0, 0.0, 0.8, np.inf, 60.0)])
    with pytest.raises(ValueError, match="orientation must be finite"):
        skin_orientation_error([nonfinite], truth_strike, truth_dip)

    with pytest.raises(ValueError, match="truth_fault_mask shape"):
        skin_truth_metrics(
            [skin],
            shape=(2, 2, 2),
            truth_fault_mask=np.zeros((2, 2, 3), dtype=bool),
            truth_surface_mask=np.zeros((2, 2, 2), dtype=bool),
            truth_strike=truth_strike,
            truth_dip=truth_dip,
            buffer_radius=2.0,
        )
