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


def test_orthogonal_overlay_routes_linewidths_without_halo_and_uses_original_markers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    amplitude, public, baseline, candidate = _diagnostic_inputs()
    contour_calls: list[tuple[str, float]] = []
    scatter_calls: list[dict[str, Any]] = []
    titles: list[str] = []
    original_contour = Axes.contour
    original_scatter = Axes.scatter
    original_set_title = Axes.set_title

    def recording_contour(self: Axes, *args: Any, **kwargs: Any) -> Any:
        contour_calls.append((str(kwargs["colors"][0]), float(kwargs["linewidths"])))
        return original_contour(self, *args, **kwargs)

    def recording_scatter(self: Axes, *args: Any, **kwargs: Any) -> Any:
        scatter_calls.append(dict(kwargs))
        return original_scatter(self, *args, **kwargs)

    def recording_set_title(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        titles.append(label)
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "contour", recording_contour)
    monkeypatch.setattr(Axes, "scatter", recording_scatter)
    monkeypatch.setattr(Axes, "set_title", recording_set_title)
    viz.save_outlier_orthogonal_amplitude_overlay(
        tmp_path / "orthogonal.png",
        amplitude=amplitude,
        public_fvt_mask=public,
        baseline_fvt_mask=baseline,
        candidate_fvt_mask=candidate,
        representative_coordinate=(0, 2, 4),
        nearest_public_coordinate=(0, 2, 2),
        crop_global_start=(100, 200, 300),
        amplitude_clip=2.5,
        window_radius=2,
        crop_index=1,
        component_id=3,
        distance_to_public=2.0,
    )

    assert {width for color, width in contour_calls if color == "#ff453a"} == {0.8}
    assert {width for color, width in contour_calls if color == "#00c7ff"} == {0.8}
    candidate_calls = [width for color, width in contour_calls if color == "#ffd60a"]
    assert candidate_calls == [2.0] * 6
    assert not any(color == "#111111" for color, _ in contour_calls)

    representative_calls = [call for call in scatter_calls if call.get("marker") == "*"]
    assert representative_calls
    assert all(call["s"] == 64 for call in representative_calls)
    assert all(call["c"] == "#ff2dff" for call in representative_calls)
    assert all(call["edgecolors"] == "#111111" for call in representative_calls)
    assert all(call["linewidths"] == 0.6 for call in representative_calls)
    nearest_calls = [call for call in scatter_calls if call.get("marker") == "x"]
    assert nearest_calls
    assert all(call["c"] == "#30d158" for call in nearest_calls)

    assert "seismic + public FVT top 1% ridge" in titles
    assert "seismic + baseline FVT top 1% ridge" in titles
    assert "seismic + candidate FVT top 5% ridge (display)" in titles
    labels = [handle.get_label() for handle in viz._outlier_legend_handles(include_all_ridges=True)]
    assert labels[:3] == [
        "public FVT top 1% ridge",
        "baseline FVT top 1% ridge",
        "candidate FVT top 5% ridge (display)",
    ]


def test_draw_mask_plane_validates_and_routes_linewidth_to_contour_and_fallback() -> None:
    class RecordingAxes:
        def __init__(self) -> None:
            self.contour_calls: list[dict[str, Any]] = []
            self.scatter_calls: list[dict[str, Any]] = []

        def contour(self, *args: Any, **kwargs: Any) -> None:
            self.contour_calls.append(kwargs)

        def scatter(self, *args: Any, **kwargs: Any) -> None:
            self.scatter_calls.append(kwargs)

    contour_axes = RecordingAxes()
    contour_mask = np.zeros((1, 2, 2), dtype=bool)
    contour_mask[0, 0, 0] = True
    common = {
        "axis_name": "i3",
        "fixed_index": 0,
        "global_start": (0, 0, 0),
        "color": "yellow",
    }
    viz._draw_mask_plane(
        contour_axes,
        contour_mask,
        row_slice=slice(0, 2),
        column_slice=slice(0, 2),
        **common,
    )
    assert contour_axes.contour_calls[0]["linewidths"] == 0.8

    fallback_axes = RecordingAxes()
    fallback_mask = np.ones((1, 1, 1), dtype=bool)
    viz._draw_mask_plane(
        fallback_axes,
        fallback_mask,
        row_slice=slice(0, 1),
        column_slice=slice(0, 1),
        linewidth=2.0,
        **common,
    )
    assert not fallback_axes.contour_calls
    assert fallback_axes.scatter_calls[0]["linewidths"] == 2.0

    for invalid in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="linewidth must be finite and > 0"):
            viz._draw_mask_plane(
                None,
                fallback_mask,
                row_slice=slice(0, 1),
                column_slice=slice(0, 1),
                linewidth=invalid,
                **common,
            )


def test_adjacent_overlay_routes_candidate_display_linewidth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    amplitude, public, baseline, candidate = _diagnostic_inputs()
    contour_calls: list[tuple[str, float]] = []
    original_contour = Axes.contour

    def recording_contour(self: Axes, *args: Any, **kwargs: Any) -> Any:
        contour_calls.append((str(kwargs["colors"][0]), float(kwargs["linewidths"])))
        return original_contour(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "contour", recording_contour)
    viz.save_outlier_adjacent_slice_overlay(
        tmp_path / "adjacent.png",
        amplitude=amplitude,
        public_fvt_mask=public,
        baseline_fvt_mask=baseline,
        candidate_fvt_mask=candidate,
        representative_coordinate=(0, 2, 4),
        nearest_public_coordinate=(0, 2, 2),
        crop_global_start=(100, 200, 300),
        amplitude_clip=2.5,
        window_radius=2,
        adjacent_slice_radius=1,
        axis="i3",
        crop_index=1,
        component_id=3,
    )

    assert {width for color, width in contour_calls if color == "#ff453a"} == {0.8}
    assert {width for color, width in contour_calls if color == "#00c7ff"} == {0.8}
    assert {width for color, width in contour_calls if color == "#ffd60a"} == {2.0}


def test_context_overlay_routes_all_candidate_masks_and_labels_as_top_five_percent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    amplitude, public, _, base = _diagnostic_inputs()
    context = np.zeros_like(base)
    context[0, 2, 4] = True
    context[0, 2, 5] = True
    contour_calls: list[tuple[str, float]] = []
    titles: list[str] = []
    original_contour = Axes.contour
    original_set_title = Axes.set_title

    def recording_contour(self: Axes, *args: Any, **kwargs: Any) -> Any:
        contour_calls.append((str(kwargs["colors"][0]), float(kwargs["linewidths"])))
        return original_contour(self, *args, **kwargs)

    def recording_set_title(self: Axes, label: str, *args: Any, **kwargs: Any) -> Any:
        titles.append(label)
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "contour", recording_contour)
    monkeypatch.setattr(Axes, "set_title", recording_set_title)
    viz.save_context_orthogonal_amplitude_comparison(
        tmp_path / "context.png",
        amplitude=amplitude,
        base_candidate_fvt_mask=base,
        context_candidate_fvt_mask=context,
        representative_coordinate=(0, 2, 4),
        nearest_public_coordinate=tuple(int(value) for value in np.argwhere(public)[0]),
        crop_global_start=(100, 200, 300),
        amplitude_clip=2.5,
        window_radius=2,
        crop_index=1,
        component_id=3,
    )

    assert contour_calls
    assert {width for _, width in contour_calls} == {2.0}
    assert "base candidate FVT top 5% ridge (display)" in titles
    assert "context candidate FVT top 5% ridge (display)" in titles
    labels = [handle.get_label() for handle in viz._context_legend_handles()]
    assert labels[:3] == [
        "base candidate FVT top 5% ridge (display)",
        "context candidate FVT top 5% ridge (display)",
        "exact top 5% mask overlap",
    ]


def test_candidate_detail_visualization_is_removed() -> None:
    assert not hasattr(viz, "save_candidate_detail_amplitude_overlay")
    assert not hasattr(viz, "_candidate_detail_threshold_masks")
