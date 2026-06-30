from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from pprint import pformat

import numpy as np
import pytest

from pyosv.f3d_reference import (
    F3D_ENV_VAR,
    crop_slices,
    interior_slices,
    pick_reference_centers,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
RUN_ENV_VAR = "PYOSV_RUN_F3D_CROP_PIPELINE"
REQUIRED_FILES = ("ep.dat", "fv.dat", "fvt.dat")


def _import_validation_module(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop("run_3d_f3d_crop_validation", None)
    importlib.invalidate_caches()
    return importlib.import_module("run_3d_f3d_crop_validation")


def _gated_data_root() -> Path:
    if os.environ.get(RUN_ENV_VAR) != "1":
        pytest.skip(f"set {RUN_ENV_VAR}=1 to run the F3 crop pipeline")

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


def _nonzero_count(array: np.ndarray) -> int:
    return int(np.count_nonzero(np.abs(array) > 1.0e-6))


def _assert_finite_metric_values(metrics: object) -> None:
    if isinstance(metrics, dict):
        for value in metrics.values():
            _assert_finite_metric_values(value)
        return

    assert np.isfinite(float(metrics))


@pytest.mark.f3d_reference
def test_f3d_reference_one_crop_pipeline(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = _gated_data_root()
    module = _import_validation_module(monkeypatch)
    crop_shape = (64, 64, 64)
    interior_margin = 16

    arrays = module.read_reference_arrays(data_root)
    centers = pick_reference_centers(
        arrays["fv.dat"],
        count=1,
        percentile=99.9,
        min_separation=48.0,
        crop_shape=crop_shape,
    )
    assert len(centers) == 1

    slices = crop_slices(centers[0], crop_shape, full_shape=arrays["ep.dat"].shape)
    ep_crop = module._crop(arrays["ep.dat"], slices)
    reference_fv = module._crop(arrays["fv.dat"], slices)
    reference_fvt = module._crop(arrays["fvt.dat"], slices)

    outputs = module.run_pipeline(
        ep_crop,
        sigma1=8.0,
        sigma2=8.0,
        phi_min=0.0,
        phi_max=360.0,
        theta_min=65.0,
        theta_max=80.0,
        ru=10,
        rv=20,
        rw=30,
        strain_max1=0.25,
        strain_max2=0.25,
        surface_smoothing1=2.0,
        surface_smoothing2=2.0,
        d=4,
        fm=0.3,
    )
    fv_py = outputs["fv_py.dat"]
    fvt_py = outputs["fvt_py.dat"]

    report = module.build_crop_report(
        crop_index=1,
        center=centers[0],
        slices=slices,
        crop_shape=crop_shape,
        outputs=outputs,
        reference_fv=reference_fv,
        reference_fvt=reference_fvt,
        interior_margin=interior_margin,
    )

    with capsys.disabled():
        print("\nF3 crop pipeline practical-equivalence report:\n" + pformat(report))

    assert fv_py.shape == crop_shape
    assert fvt_py.shape == crop_shape
    assert fv_py.dtype == np.float32
    assert fvt_py.dtype == np.float32
    assert np.isfinite(fv_py).all()
    assert np.isfinite(fvt_py).all()
    assert float(fv_py.max()) > 0.0
    assert float(fvt_py.max()) > 0.0
    assert _nonzero_count(fvt_py) < _nonzero_count(fv_py)

    slices_in_crop = interior_slices(crop_shape, margin=interior_margin)
    reference_interiors = {
        "fv": reference_fv[slices_in_crop],
        "fvt": reference_fvt[slices_in_crop],
    }
    for name, reference_interior in reference_interiors.items():
        if np.ptp(reference_interior) <= 0.0:
            continue

        assert np.isfinite(report["normalized_correlation"]["interior"][name])
        _assert_finite_metric_values(report["top_percentile_overlap"]["interior"][name])
