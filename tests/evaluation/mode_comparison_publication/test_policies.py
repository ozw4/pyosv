from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pyosv.evaluation.mode_comparison_publication.figures import (
    _difference_scale,
    _exact_overlap_count,
    _overlay_slice_rgb,
    _signed_difference_slice,
    _shared_scale,
    _slice_selection,
)
from pyosv.evaluation.mode_comparison_publication.summary import _diagnostic_unit


def test_public_reference_slice_tie_uses_smallest_index() -> None:
    reference = np.zeros((3, 4, 5), dtype=np.float32)
    reference[0, 1, 1] = 1.0
    reference[1, 1, 1] = 1.0
    index, score = _slice_selection(
        "public_reference_peak",
        "i3",
        reference.shape,
        reference,
        threshold=0.5,
        difference=None,
    )
    assert (index, score) == (0, 1.0)


def test_ridge_overlay_uses_the_cell_specific_candidate_threshold() -> None:
    reference = np.zeros((1, 1, 1), dtype=np.float32)
    candidate = np.full((1, 1, 1), 0.6, dtype=np.float32)

    rgb = _overlay_slice_rgb(
        reference,
        candidate,
        axis="i3",
        index=0,
        reference_threshold=0.8,
        candidate_threshold=0.4,
        radius=2.0,
    )

    # With the obsolete shared 0.8 threshold, the 0.6 candidate would vanish.
    np.testing.assert_array_equal(rgb[0, 0], np.array((0.0, 0.25, 1.0), dtype=np.float32))


def test_ridge_overlay_exact_overlap_uses_the_cell_specific_candidate_threshold() -> None:
    reference = np.full((1, 1, 1), 0.9, dtype=np.float32)
    candidate = np.full((1, 1, 1), 0.6, dtype=np.float32)

    rgb = _overlay_slice_rgb(
        reference,
        candidate,
        axis="i3",
        index=0,
        reference_threshold=0.8,
        candidate_threshold=0.4,
        radius=2.0,
    )

    np.testing.assert_array_equal(rgb[0, 0], np.ones(3, dtype=np.float32))
    assert (
        _exact_overlap_count(
            reference,
            candidate,
            reference_threshold=0.8,
            candidate_threshold=0.4,
        )
        == 1
    )


def test_public_reference_peak_is_independent_of_candidate_thresholds() -> None:
    reference = np.zeros((3, 1, 2), dtype=np.float32)
    reference[1, 0, :] = 0.9
    reference[2, 0, 0] = 0.9

    # Candidate data and their thresholds intentionally do not enter this
    # policy: public_reference_peak is selected only from public reference.
    candidate_variants = (
        (np.zeros_like(reference), 0.4),
        (np.ones_like(reference), 0.99),
    )
    results = [
        _slice_selection(
            "public_reference_peak",
            "i3",
            reference.shape,
            reference,
            threshold=0.8,
            difference=None,
        )
        for _candidate, _candidate_threshold in candidate_variants
    ]

    assert results == [(1, 2.0), (1, 2.0)]


def test_difference_peak_tie_uses_smallest_index() -> None:
    difference = np.zeros((3, 4, 5), dtype=np.float32)
    difference[0, 1, 1] = 2.0
    difference[1, 1, 1] = 2.0
    index, score = _slice_selection(
        "end_to_end_difference_peak",
        "i3",
        difference.shape,
        difference,
        threshold=0.5,
        difference=difference,
    )
    assert (index, score) == (0, 2.0)


def test_all_zero_difference_has_finite_zero_centered_scale() -> None:
    observed, vmin, vmax = _difference_scale(np.zeros((2, 2), dtype=np.float32))
    assert (observed, vmin, vmax) == (0.0, -1.0e-6, 1.0e-6)


def test_shared_scale_uses_one_range_for_all_displayed_panels() -> None:
    rows = []
    for cell, low, high in (
        ("RL-REF", -1.0, 2.0),
        ("RL-QUAL", -2.0, 3.0),
        ("Q-REF", -3.0, 4.0),
        ("Q-QUAL", -4.0, 5.0),
    ):
        rows.extend(
            (
                SimpleNamespace(
                    stage="fv", cell_label=cell, selection="all", metric="candidate_min", value=low
                ),
                SimpleNamespace(
                    stage="fv", cell_label=cell, selection="all", metric="candidate_max", value=high
                ),
                SimpleNamespace(
                    stage="fv", cell_label=cell, selection="all", metric="reference_min", value=-0.5
                ),
                SimpleNamespace(
                    stage="fv", cell_label=cell, selection="all", metric="reference_max", value=1.5
                ),
            )
        )
    low, high = _shared_scale(SimpleNamespace(metric_rows=tuple(rows)), "fv", ("RL-REF", "Q-QUAL"))
    assert (low, high) == (-4.0, 5.0)


class _SliceOnlyArray:
    """Array wrapper that permits 2D slice reads but forbids full-volume math."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values
        self.slice_keys: list[object] = []

    def __getitem__(self, key: object) -> np.ndarray:
        self.slice_keys.append(key)
        return self.values[key]  # type: ignore[index]

    def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
        del args, kwargs
        raise AssertionError("full-volume __array__ conversion is forbidden")

    def __sub__(self, other: object) -> np.ndarray:
        del other
        raise AssertionError("full-volume subtraction is forbidden")


@pytest.mark.parametrize(("axis", "index"), (("i3", 1), ("i2", 2), ("i1", 3)))
def test_signed_difference_slice_never_materializes_full_volume(axis: str, index: int) -> None:
    left_values = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    right_values = np.full(left_values.shape, 2.0, dtype=np.float32)
    left = _SliceOnlyArray(left_values)
    right = _SliceOnlyArray(right_values)

    actual = _signed_difference_slice(left, right, axis, index)  # type: ignore[arg-type]
    expected = np.take(left_values, index, axis={"i3": 0, "i2": 1, "i1": 2}[axis]) - np.take(
        right_values, index, axis={"i3": 0, "i2": 1, "i1": 2}[axis]
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32
    assert len(left.slice_keys) == len(right.slice_keys) == 1


@pytest.mark.parametrize(
    ("metric", "expected"),
    (
        ("positive_p99_precision", "fraction"),
        ("positive_p99_recall", "fraction"),
        ("positive_p99_f1", "fraction"),
        ("positive_p99_jaccard", "fraction"),
        ("positive_p99_radius2_buffered_precision", "fraction"),
        ("positive_p99_radius2_buffered_recall", "fraction"),
        ("positive_p99_radius2_buffered_f1", "fraction"),
        ("positive_p99_distance_candidate_to_reference_p95", "voxel"),
        ("candidate_nonzero_fraction", "fraction"),
        ("nonzero_fraction_ratio", "ratio"),
        ("normalized_correlation", "correlation"),
        ("voxel_count", "count"),
        ("mean_absolute_difference", "value"),
        ("root_mean_square_difference", "value"),
    ),
)
def test_regional_diagnostic_unit_contract(metric: str, expected: str) -> None:
    assert _diagnostic_unit(metric) == expected
