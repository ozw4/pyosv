from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pyosv.evaluation.mode_comparison_publication.figures import (
    _difference_scale,
    _shared_scale,
    _slice_selection,
)


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
