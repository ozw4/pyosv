from __future__ import annotations

import builtins

import numpy as np
import pytest

from pyosv import viz


def test_require_matplotlib_imports_when_available() -> None:
    pytest.importorskip("matplotlib")

    plt = viz.require_matplotlib()

    assert plt.__name__ == "matplotlib.pyplot"


def test_require_matplotlib_reports_extra_when_unavailable(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib":
            raise ImportError("matplotlib unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"pyosv\[viz\]"):
        viz.require_matplotlib()


def test_ensure_output_dir_creates_nested_directory(tmp_path) -> None:
    output_dir = viz.ensure_output_dir(tmp_path / "diagnostics" / "f3")

    assert output_dir.is_dir()
    assert output_dir == tmp_path / "diagnostics" / "f3"


def test_safe_percentile_threshold_handles_normal_zero_and_sparse_arrays() -> None:
    values = np.arange(10, dtype=np.float32)
    assert viz.safe_percentile_threshold(values, 50.0) == pytest.approx(4.5)
    assert viz.safe_percentile_threshold(np.zeros((3, 4), dtype=np.float32), 99.0) == 0.0

    sparse = np.zeros(100, dtype=np.float32)
    sparse[-1] = 10.0
    assert viz.safe_percentile_threshold(sparse, 100.0) == 10.0


def test_safe_percentile_threshold_handles_empty_and_nonfinite_arrays() -> None:
    assert viz.safe_percentile_threshold(np.array([], dtype=np.float32), 99.0) == 0.0
    assert viz.safe_percentile_threshold(np.array([np.nan, np.inf], dtype=np.float32), 99.0) == 0.0


@pytest.mark.parametrize("percentile", [-1.0, 101.0, np.nan])
def test_safe_percentile_threshold_rejects_invalid_percentiles(percentile: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        viz.safe_percentile_threshold(np.ones(4, dtype=np.float32), percentile)


def test_normalize_for_display_returns_finite_float_array_in_unit_range() -> None:
    values = np.array([0.0, 1.0, 2.0, np.nan, np.inf], dtype=np.float32)

    normalized = viz.normalize_for_display(values, clip_percentiles=(0.0, 100.0))

    assert normalized.dtype == np.float32
    assert normalized.shape == values.shape
    assert np.all(np.isfinite(normalized))
    assert float(np.min(normalized)) >= 0.0
    assert float(np.max(normalized)) <= 1.0


def test_normalize_for_display_handles_constant_and_empty_arrays() -> None:
    np.testing.assert_array_equal(
        viz.normalize_for_display(np.ones((2, 3), dtype=np.float32)),
        np.zeros((2, 3), dtype=np.float32),
    )
    np.testing.assert_array_equal(
        viz.normalize_for_display(np.array([], dtype=np.float32)),
        np.array([], dtype=np.float32),
    )


def test_select_center_slices_is_deterministic() -> None:
    assert viz.select_center_slices((64, 80, 100)) == {"i3": 32, "i2": 40, "i1": 50}


def test_select_center_slices_rejects_non_3d_shapes() -> None:
    with pytest.raises(ValueError, match="3D"):
        viz.select_center_slices((64, 80))
