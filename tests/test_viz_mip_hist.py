from __future__ import annotations

import numpy as np
import pytest

from pyosv import viz


def test_maximum_intensity_projection_shapes_follow_axis_convention() -> None:
    volume = np.zeros((4, 5, 6), dtype=np.float32)
    volume[1, 2, 3] = 1.0

    i3_mip = viz.maximum_intensity_projection(volume, "i3")
    i2_mip = viz.maximum_intensity_projection(volume, "i2")
    i1_mip = viz.maximum_intensity_projection(volume, "i1")

    assert i3_mip.shape == (5, 6)
    assert i2_mip.shape == (4, 6)
    assert i1_mip.shape == (4, 5)
    assert i3_mip[2, 3] == pytest.approx(1.0)
    assert i2_mip[1, 3] == pytest.approx(1.0)
    assert i1_mip[1, 2] == pytest.approx(1.0)


def test_maximum_intensity_projection_rejects_non_3d_volume() -> None:
    with pytest.raises(ValueError, match="3D"):
        viz.maximum_intensity_projection(np.zeros((4, 5), dtype=np.float32), "i3")


def test_save_mip_comparison_writes_non_empty_png(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    reference = np.zeros((4, 5, 6), dtype=np.float32)
    candidate = reference.copy()
    reference[1, 2, 3] = 1.0
    candidate[2, 3, 4] = 0.75

    output_path = viz.save_mip_comparison(
        tmp_path / "mip.png",
        reference=reference,
        candidate=candidate,
        name="fvt",
        clip_percentiles=(0.0, 100.0),
    )

    assert output_path == tmp_path / "mip.png"
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_save_histogram_comparison_writes_non_empty_png_with_negative_values(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    reference = np.zeros((4, 5, 6), dtype=np.float32)
    candidate = reference.copy()
    candidate[1, 2, 3] = -0.01
    candidate[2, 3, 4] = 0.75

    output_path = viz.save_histogram_comparison(
        tmp_path / "hist.png",
        reference=reference,
        candidate=candidate,
        name="fvt",
        bins=20,
    )

    assert output_path == tmp_path / "hist.png"
    assert output_path.is_file()
    assert output_path.stat().st_size > 0


def test_volume_diagnostics_writes_deterministic_filenames_for_all_zero_arrays(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    values = np.zeros((4, 5, 6), dtype=np.float32)

    written = viz.save_volume_diagnostics(
        tmp_path,
        reference=values,
        candidate=values.copy(),
        name="fvt",
        clip_percentiles=(0.0, 100.0),
    )

    assert written == {
        "mip": tmp_path / "fvt_mip.png",
        "hist": tmp_path / "fvt_hist.png",
    }
    for path in written.values():
        assert path.is_file()
        assert path.stat().st_size > 0


def test_mip_and_histogram_comparisons_reject_shape_mismatch(tmp_path) -> None:
    pytest.importorskip("matplotlib")

    with pytest.raises(ValueError, match="same shape"):
        viz.save_mip_comparison(
            tmp_path / "mip.png",
            reference=np.zeros((4, 5, 6), dtype=np.float32),
            candidate=np.zeros((4, 5, 7), dtype=np.float32),
            name="fvt",
        )

    with pytest.raises(ValueError, match="same shape"):
        viz.save_histogram_comparison(
            tmp_path / "hist.png",
            reference=np.zeros((4, 5, 6), dtype=np.float32),
            candidate=np.zeros((4, 5, 7), dtype=np.float32),
            name="fvt",
        )
