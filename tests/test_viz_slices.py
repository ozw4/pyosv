from __future__ import annotations

import numpy as np
import pytest

from pyosv import viz


def test_slice_2d_respects_project_axis_convention() -> None:
    volume = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)

    np.testing.assert_array_equal(viz.slice_2d(volume, "i3", 2), volume[2, :, :])
    np.testing.assert_array_equal(viz.slice_2d(volume, 0, 2), volume[2, :, :])
    assert viz.slice_2d(volume, "i3", 2).shape == (5, 6)

    np.testing.assert_array_equal(viz.slice_2d(volume, "i2", 3), volume[:, 3, :])
    np.testing.assert_array_equal(viz.slice_2d(volume, 1, 3), volume[:, 3, :])
    assert viz.slice_2d(volume, "i2", 3).shape == (4, 6)

    np.testing.assert_array_equal(viz.slice_2d(volume, "i1", 4), volume[:, :, 4])
    np.testing.assert_array_equal(viz.slice_2d(volume, 2, 4), volume[:, :, 4])
    assert viz.slice_2d(volume, "i1", 4).shape == (4, 5)


def test_slice_2d_rejects_invalid_axis_and_index() -> None:
    volume = np.zeros((4, 5, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="axis"):
        viz.slice_2d(volume, "x", 0)

    with pytest.raises(ValueError, match="index"):
        viz.slice_2d(volume, "i3", 4)


def test_save_slice_panel_writes_non_empty_png(tmp_path) -> None:
    pytest.importorskip("matplotlib")

    panel_path = viz.save_slice_panel(
        tmp_path / "panel.png",
        [
            ("first", np.arange(12, dtype=np.float32).reshape(3, 4)),
            ("second", np.ones((3, 4), dtype=np.float32)),
        ],
        clip_percentiles=(0.0, 100.0),
    )

    assert panel_path == tmp_path / "panel.png"
    assert panel_path.is_file()
    assert panel_path.stat().st_size > 0


def test_save_volume_comparison_slices_writes_deterministic_filenames(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    reference = np.arange(4 * 5 * 6, dtype=np.float32).reshape(4, 5, 6)
    candidate = reference + 1.0

    written = viz.save_volume_comparison_slices(
        tmp_path,
        reference=reference,
        candidate=candidate,
        name="fault",
        slice_indices={"i3": 1, "i2": 2, "i1": 3},
        clip_percentiles=(0.0, 100.0),
    )

    assert written == {
        "i3": tmp_path / "fault_i3_1.png",
        "i2": tmp_path / "fault_i2_2.png",
        "i1": tmp_path / "fault_i1_3.png",
    }
    for path in written.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_save_volume_comparison_slices_rejects_shape_mismatch(tmp_path) -> None:
    pytest.importorskip("matplotlib")

    with pytest.raises(ValueError, match="same shape"):
        viz.save_volume_comparison_slices(
            tmp_path,
            reference=np.zeros((4, 5, 6), dtype=np.float32),
            candidate=np.zeros((4, 5, 7), dtype=np.float32),
            name="fault",
        )
