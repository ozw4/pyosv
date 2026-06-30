from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from pyosv.f3d_reference import F3D_ENV_VAR


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
RUN_ENV_VAR = "PYOSV_RUN_F3D_LARGE_CROP_PIPELINE"
REQUIRED_FILES = ("ep.dat", "fv.dat", "fvt.dat")


def _import_validation_module(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop("run_3d_f3d_crop_validation", None)
    importlib.invalidate_caches()
    return importlib.import_module("run_3d_f3d_crop_validation")


def _gated_data_root() -> Path:
    if os.environ.get(RUN_ENV_VAR) != "1":
        pytest.skip(f"set {RUN_ENV_VAR}=1 to run the large F3 crop pipeline")

    root_text = os.environ.get(F3D_ENV_VAR)
    if root_text is None:
        pytest.skip(f"set {F3D_ENV_VAR} to the F3 reference data root")

    root = Path(root_text)
    if not root.is_dir():
        pytest.skip(f"{F3D_ENV_VAR} does not point to an existing directory: {root}")

    missing = [filename for filename in REQUIRED_FILES if not (root / filename).is_file()]
    if missing:
        pytest.skip(f"{F3D_ENV_VAR} is missing required files: {', '.join(missing)}")

    return root


def _assert_finite_metric_values(metrics: object) -> None:
    if metrics is None:
        return
    if isinstance(metrics, dict):
        for value in metrics.values():
            _assert_finite_metric_values(value)
        return

    assert np.isfinite(float(metrics))


@pytest.mark.f3d_reference
def test_f3d_large_crop_pipeline_schema_and_finite_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = _gated_data_root()
    module = _import_validation_module(monkeypatch)

    report = module.run_example(
        data_root_arg=data_root,
        crop_shape=(128, 128, 100),
        interior_margin=40,
        max_crops=1,
    )
    crop = report["crops"][0]

    assert report["config"]["crop_shape"] == [128, 128, 100]
    assert report["config"]["interior_margin"] == 40
    assert crop["crop_shape"] == [128, 128, 100]
    assert crop["crop_slices"]
    assert crop["interior_slices"]
    assert crop["interior_slices_in_crop"] == [
        {"axis": "i3", "start": 40, "stop": 88},
        {"axis": "i2", "start": 40, "stop": 88},
        {"axis": "i1", "start": 40, "stop": 60},
    ]

    _assert_finite_metric_values(crop["normalized_correlation"]["full_crop"])
    _assert_finite_metric_values(crop["normalized_correlation"]["interior"])
    _assert_finite_metric_values(crop["top_percentile_overlap"]["full_crop"])
    _assert_finite_metric_values(crop["top_percentile_overlap"]["interior"])
    _assert_finite_metric_values(crop["buffered_ridge_overlap"]["interior"]["fvt"])
    _assert_finite_metric_values(crop["sparse_ridge_distance_metrics"]["interior"]["fvt"])
