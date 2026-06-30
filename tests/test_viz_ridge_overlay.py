from __future__ import annotations

import numpy as np
import pytest

from pyosv import viz


def _ridge_volume(row: int, shape: tuple[int, int, int] = (3, 7, 7)) -> np.ndarray:
    values = np.zeros(shape, dtype=np.float32)
    values[:, row, 1:-1] = 1.0
    return values


def test_ridge_mask_selects_positive_top_percentile_and_ignores_all_zero() -> None:
    values = np.zeros((4, 4), dtype=np.float32)
    values[1, 1] = 1.0
    values[2, 2] = 3.0

    mask = viz.ridge_mask(values, percentile=50.0)

    expected = np.zeros((4, 4), dtype=bool)
    expected[2, 2] = True
    assert mask.dtype == np.bool_
    np.testing.assert_array_equal(mask, expected)
    assert not np.any(viz.ridge_mask(np.zeros((4, 4), dtype=np.float32)))


def test_ridge_overlay_rgb_marks_exact_reference_and_candidate_distinctly() -> None:
    reference = np.zeros((4, 4), dtype=bool)
    candidate = np.zeros((4, 4), dtype=bool)
    reference[1, 1] = True
    reference[1, 2] = True
    candidate[1, 2] = True
    candidate[2, 2] = True

    rgb = viz._ridge_overlay_rgb(
        reference,
        candidate,
        reference_buffer=reference,
        candidate_buffer=candidate,
        has_buffer=False,
    )

    reference_only = rgb[1, 1]
    exact_overlap = rgb[1, 2]
    candidate_only = rgb[2, 2]
    assert not np.array_equal(reference_only, exact_overlap)
    assert not np.array_equal(candidate_only, exact_overlap)
    assert not np.array_equal(reference_only, candidate_only)


def test_ridge_overlay_rgb_marks_shifted_candidate_as_buffered_match() -> None:
    reference = np.zeros((5, 5), dtype=bool)
    candidate = np.zeros((5, 5), dtype=bool)
    reference[2, 2] = True
    candidate[3, 2] = True
    reference_buffer = reference.copy()
    reference_buffer[3, 2] = True
    candidate_buffer = candidate.copy()
    candidate_buffer[2, 2] = True

    rgb = viz._ridge_overlay_rgb(
        reference,
        candidate,
        reference_buffer=reference_buffer,
        candidate_buffer=candidate_buffer,
        has_buffer=True,
    )

    np.testing.assert_array_equal(rgb[2, 2], viz._RIDGE_BUFFERED_MATCH_RGB)
    np.testing.assert_array_equal(rgb[3, 2], viz._RIDGE_BUFFERED_MATCH_RGB)


def test_save_ridge_overlay_slice_writes_non_empty_png_for_identical_ridges(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    reference = _ridge_volume(3)

    output_path = viz.save_ridge_overlay_slice(
        tmp_path / "identical.png",
        reference=reference,
        candidate=reference.copy(),
        axis="i3",
        index=1,
        percentile=99.0,
    )

    assert output_path == tmp_path / "identical.png"
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_ridge_overlay_slice_writes_non_empty_png_for_shifted_buffered_ridges(
    tmp_path,
) -> None:
    pytest.importorskip("matplotlib")

    output_path = viz.save_ridge_overlay_slice(
        tmp_path / "shifted.png",
        reference=_ridge_volume(3),
        candidate=_ridge_volume(4),
        axis="i3",
        index=1,
        percentile=99.0,
        buffer_radius=1.0,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_buffered_ridge_overlay_slices_writes_deterministic_filenames(tmp_path) -> None:
    pytest.importorskip("matplotlib")

    written = viz.save_buffered_ridge_overlay_slices(
        tmp_path,
        reference=_ridge_volume(2),
        candidate=_ridge_volume(5),
        name="fvt",
        slice_indices={"i3": 1, "i2": 2, "i1": 3},
        percentile=99.0,
        buffer_radius=2.0,
    )

    assert written == {
        "i3": tmp_path / "fvt_ridge_overlay_i3_1.png",
        "i2": tmp_path / "fvt_ridge_overlay_i2_2.png",
        "i1": tmp_path / "fvt_ridge_overlay_i1_3.png",
    }
    for path in written.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_save_ridge_overlay_slice_handles_all_zero_inputs(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    values = np.zeros((3, 4, 5), dtype=np.float32)

    output_path = viz.save_ridge_overlay_slice(
        tmp_path / "empty.png",
        reference=values,
        candidate=values.copy(),
        axis="i2",
        index=2,
        percentile=99.0,
        buffer_radius=2.0,
    )

    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_ridge_overlay_slice_rejects_shape_mismatch(tmp_path) -> None:
    pytest.importorskip("matplotlib")

    with pytest.raises(ValueError, match="same shape"):
        viz.save_ridge_overlay_slice(
            tmp_path / "bad.png",
            reference=np.zeros((3, 4, 5), dtype=np.float32),
            candidate=np.zeros((3, 4, 6), dtype=np.float32),
            axis="i3",
            index=0,
        )
