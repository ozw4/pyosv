from __future__ import annotations

import weakref
from pathlib import Path

import numpy as np
import pytest

import pyosv.evaluation.f3d_mode_comparison.diagnostics as diagnostics_module
from pyosv.evaluation.f3d_mode_comparison.diagnostics import (
    F3_ORIENTATION_PAIRS,
    F3_REGION_SEMANTICS,
    build_region_partition,
    compute_orientation_diagnostics,
    compute_orientation_pair_diagnostic,
    compute_regional_reference_diagnostics,
    dip_absolute_difference,
    normal_vector_angular_difference,
    strike_circular_absolute_difference,
)


def _regional(candidate: np.ndarray, reference: np.ndarray, margin: int = 1):
    return compute_regional_reference_diagnostics(
        dataset_id="fixture",
        cell_label="RL-REF",
        scanner_backend="reference-like",
        workflow_mode="reference",
        stage="ft",
        candidate=candidate,
        reference=reference,
        margin=margin,
    )


def test_region_partition_is_disjoint_and_complete() -> None:
    partition = build_region_partition((5, 6, 7), 1)

    assert not np.any(partition.interior & partition.boundary_shell)
    assert np.all(partition.interior | partition.boundary_shell)
    assert sum(partition.counts.values()) == 5 * 6 * 7
    assert partition.semantics == F3_REGION_SEMANTICS

    zero = build_region_partition((2, 3, 4), 0)
    assert zero.counts == {"interior": 24, "boundary_shell": 0}
    with pytest.raises(ValueError, match="too large"):
        build_region_partition((4, 5, 6), 2)


def test_regional_basic_metrics_match_manual_mask_selection() -> None:
    candidate = np.arange(125, dtype=np.float32).reshape(5, 5, 5)
    reference = candidate + np.float32(2.0)
    rows = _regional(candidate, reference)
    partition = build_region_partition(candidate.shape, 1)

    for row in rows:
        mask = partition.mask_for(row.region)
        difference = candidate[mask].astype(np.float64) - reference[mask]
        assert row.metrics["voxel_count"] == np.count_nonzero(mask)
        assert row.metrics["mean_absolute_difference"] == pytest.approx(np.mean(np.abs(difference)))
        assert row.metrics["root_mean_square_difference"] == pytest.approx(
            np.sqrt(np.mean(difference * difference))
        )
        assert "sample_index" not in row.as_dict()
        assert "trial_index" not in row.as_dict()
        assert "replicate_index" not in row.as_dict()


def test_regional_basic_metrics_process_mask_selection_in_bounded_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk_sizes: list[int] = []
    original_add = diagnostics_module._RegionalBasicAccumulator.add

    def tracked_add(
        self: diagnostics_module._RegionalBasicAccumulator,
        candidate: np.ndarray,
        reference: np.ndarray,
    ) -> None:
        chunk_sizes.append(int(candidate.size))
        original_add(self, candidate, reference)

    monkeypatch.setattr(diagnostics_module, "_REGIONAL_BASIC_CHUNK_VOXELS", 7)
    monkeypatch.setattr(diagnostics_module._RegionalBasicAccumulator, "add", tracked_add)
    candidate = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    reference = candidate + np.float32(1.0)

    diagnostics_module._regional_basic_metrics(
        candidate,
        reference,
        np.ones(candidate.shape, dtype=bool),
    )

    assert len(chunk_sizes) > 1
    assert max(chunk_sizes) <= 7


def test_full_volume_validation_preserves_memmap_dtype_and_backing(tmp_path: Path) -> None:
    path = tmp_path / "volume.dat"
    writable = np.memmap(path, dtype=np.float32, mode="w+", shape=(2, 3, 4))
    writable[:] = 1.0
    writable.flush()
    del writable
    left = np.memmap(path, dtype=np.float32, mode="r", shape=(2, 3, 4))
    right = np.memmap(path, dtype=np.float32, mode="r", shape=(2, 3, 4))

    left_values, right_values = diagnostics_module._comparable_finite_3d(left, right)

    assert left_values.dtype == np.float32
    assert right_values.dtype == np.float32
    assert np.shares_memory(left_values, left)
    assert np.shares_memory(right_values, right)


def test_regional_distance_samples_global_distance_field() -> None:
    reference = np.zeros((5, 5, 5), dtype=np.float32)
    candidate = np.zeros_like(reference)
    reference[0, 2, 2] = 1.0
    candidate[1, 2, 2] = 1.0

    interior = next(row for row in _regional(candidate, reference) if row.region == "interior")

    # A crop-local transform would have no reference ridge in the interior crop.
    assert interior.metrics["positive_p99_distance_reference_count"] == 0
    assert interior.metrics["positive_p99_distance_candidate_count"] == 1
    assert interior.metrics["positive_p99_distance_candidate_to_reference_mean"] == 1.0


def test_regional_diagnostics_release_each_full_distance_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = diagnostics_module.distance_transform_edt
    previous: weakref.ReferenceType[np.ndarray] | None = None
    call_count = 0

    def tracked_distance_transform(values: np.ndarray) -> np.ndarray:
        nonlocal call_count, previous
        if previous is not None:
            assert previous() is None
        result = original(values)
        previous = weakref.ref(result)
        call_count += 1
        return result

    monkeypatch.setattr(
        diagnostics_module,
        "distance_transform_edt",
        tracked_distance_transform,
    )
    reference = np.zeros((5, 5, 5), dtype=np.float32)
    candidate = np.zeros_like(reference)
    reference[0, 2, 2] = 1.0
    candidate[1, 2, 2] = 1.0

    _regional(candidate, reference)

    assert call_count == 2


def test_orientation_difference_helpers_have_expected_geometry() -> None:
    strike = strike_circular_absolute_difference(np.array([359.0, 10.0]), np.array([1.0, 190.0]))
    dip = dip_absolute_difference(np.array([20.0, 30.0]), np.array([25.0, 10.0]))
    normal = normal_vector_angular_difference(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([90.0]),
    )

    assert strike.tolist() == pytest.approx([2.0, 180.0])
    assert dip.tolist() == pytest.approx([5.0, 20.0])
    assert normal.tolist() == pytest.approx([90.0])


def test_orientation_rows_use_common_support_and_keep_empty_pairs() -> None:
    likelihood = np.zeros((2, 2, 2), dtype=np.float32)
    row = compute_orientation_pair_diagnostic(
        dataset_id="fixture",
        stage="scanner",
        left_cell="RL-REF",
        right_cell="RL-QUAL",
        left_likelihood=likelihood,
        left_strike=likelihood,
        left_dip=likelihood,
        right_likelihood=likelihood,
        right_strike=likelihood,
        right_dip=likelihood,
    )

    assert row.support_count == 0
    assert row.strike_circular_absolute_difference == {
        "count": 0,
        "mean": None,
        "median": None,
        "p90": None,
        "p95": None,
    }

    fields = {cell: likelihood for cell in ("RL-REF", "RL-QUAL", "Q-REF", "Q-QUAL")}
    rows = compute_orientation_diagnostics(
        dataset_id="fixture",
        stage="voting",
        likelihoods=fields,
        strikes=fields,
        dips=fields,
    )
    assert tuple((row.left_cell, row.right_cell) for row in rows) == F3_ORIENTATION_PAIRS
    assert all(row.support_count == 0 for row in rows)


def test_diagnostics_module_does_not_import_matplotlib() -> None:
    import pyosv.evaluation.f3d_mode_comparison.diagnostics as diagnostics

    assert "matplotlib" not in diagnostics.__dict__
