from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyosv import viz


def _diagnostic_inputs(
    shape: tuple[int, int, int] = (4, 5, 6),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    amplitude = np.linspace(-4.0, 4.0, np.prod(shape), dtype=np.float32).reshape(shape)
    public = np.zeros(shape, dtype=bool)
    baseline = np.zeros(shape, dtype=bool)
    candidate = np.zeros(shape, dtype=bool)
    public[0, 2, 2] = True
    baseline[0, 2, 3] = True
    candidate[0, 2, 4] = True
    return amplitude, public, baseline, candidate


def test_outlier_orthogonal_overlay_uses_fixed_symmetric_clip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    amplitude, public, baseline, candidate = _diagnostic_inputs()
    imshow_limits: list[tuple[float, float]] = []
    original_imshow = Axes.imshow

    def recording_imshow(self: Axes, *args: Any, **kwargs: Any) -> Any:
        imshow_limits.append((float(kwargs["vmin"]), float(kwargs["vmax"])))
        return original_imshow(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", recording_imshow)
    output = tmp_path / "orthogonal_amplitude_overlay.png"
    written = viz.save_outlier_orthogonal_amplitude_overlay(
        output,
        amplitude=amplitude,
        public_fvt_mask=public,
        baseline_fvt_mask=baseline,
        candidate_fvt_mask=candidate,
        representative_coordinate=(0, 2, 4),
        nearest_public_coordinate=(0, 2, 2),
        crop_global_start=(100, 200, 300),
        amplitude_clip=2.5,
        window_radius=0,
        crop_index=1,
        component_id=3,
        distance_to_public=2.0,
    )

    assert written == output
    assert output.is_file()
    assert output.stat().st_size > 0
    assert imshow_limits == [(-2.5, 2.5)] * 15


def test_adjacent_overlay_skips_crop_edges_and_titles_actual_global_slices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    amplitude, public, baseline, candidate = _diagnostic_inputs()
    titles: list[str] = []
    original_set_title = Axes.set_title

    def recording_set_title(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        titles.append(label)
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_title", recording_set_title)
    output = tmp_path / "adjacent_i3.png"
    written = viz.save_outlier_adjacent_slice_overlay(
        output,
        amplitude=amplitude,
        public_fvt_mask=public,
        baseline_fvt_mask=baseline,
        candidate_fvt_mask=candidate,
        representative_coordinate=(0, 2, 4),
        nearest_public_coordinate=(0, 2, 2),
        crop_global_start=(100, 200, 300),
        amplitude_clip=3.0,
        window_radius=2,
        adjacent_slice_radius=2,
        axis="i3",
        crop_index=1,
        component_id=3,
    )

    assert written == output
    assert output.is_file()
    assert output.stat().st_size > 0
    assert "global i3=100 (representative)" in titles
    assert "global i3=101" in titles
    assert "global i3=102" in titles
    assert not any("global i3=99" in title for title in titles)


def test_context_orthogonal_comparison_writes_three_by_four_fixed_clip_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    amplitude, public, _, base = _diagnostic_inputs()
    context = np.zeros_like(base)
    context[0, 2, 5] = True
    imshow_limits: list[tuple[float, float]] = []
    original_imshow = Axes.imshow

    def recording_imshow(self: Axes, *args: Any, **kwargs: Any) -> Any:
        imshow_limits.append((float(kwargs["vmin"]), float(kwargs["vmax"])))
        return original_imshow(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "imshow", recording_imshow)
    output = tmp_path / "context_amplitude_overlay.png"
    written = viz.save_context_orthogonal_amplitude_comparison(
        output,
        amplitude=amplitude,
        base_candidate_fvt_mask=base,
        context_candidate_fvt_mask=context,
        representative_coordinate=(0, 2, 4),
        nearest_public_coordinate=tuple(int(value) for value in np.argwhere(public)[0]),
        crop_global_start=(100, 200, 300),
        amplitude_clip=1.75,
        window_radius=2,
        crop_index=1,
        component_id=3,
    )

    assert written == output
    assert output.is_file()
    assert output.stat().st_size > 0
    assert imshow_limits == [(-1.75, 1.75)] * 12


def test_outlier_amplitude_overlay_rejects_shape_and_clip_errors(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    amplitude, public, baseline, candidate = _diagnostic_inputs()
    common = {
        "amplitude": amplitude,
        "public_fvt_mask": public,
        "baseline_fvt_mask": baseline,
        "candidate_fvt_mask": candidate,
        "representative_coordinate": (0, 2, 4),
        "nearest_public_coordinate": (0, 2, 2),
        "crop_global_start": (100, 200, 300),
        "window_radius": 2,
        "crop_index": 1,
        "component_id": 3,
        "distance_to_public": 2.0,
    }

    with pytest.raises(ValueError, match="same shape"):
        viz.save_outlier_orthogonal_amplitude_overlay(
            tmp_path / "shape.png",
            amplitude_clip=2.0,
            **{**common, "candidate_fvt_mask": candidate[:, :, :-1]},
        )
    with pytest.raises(ValueError, match="amplitude_clip"):
        viz.save_outlier_orthogonal_amplitude_overlay(
            tmp_path / "clip.png",
            amplitude_clip=0.0,
            **common,
        )
