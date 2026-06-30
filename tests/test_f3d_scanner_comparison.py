from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from pprint import pformat

import numpy as np
import pytest

from pyosv.f3d_reference import F3D_ENV_VAR, crop_slices, pick_reference_centers


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
RUN_ENV_VAR = "PYOSV_RUN_F3D_SCANNER_COMPARISON"
REQUIRED_FILES = ("ep.dat", "fl.dat")


def _import_scanner_module(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop("report_3d_f3d_scanner", None)
    importlib.invalidate_caches()
    return importlib.import_module("report_3d_f3d_scanner")


def _synthetic_reference_arrays(shape: tuple[int, int, int] = (6, 6, 6)) -> dict[str, np.ndarray]:
    ep = np.zeros(shape, dtype=np.float32)
    fl = np.zeros(shape, dtype=np.float32)
    ep[3, 3, 3] = 1.0
    fl[3, 3, 3] = 1.0
    return {"ep.dat": ep, "fl.dat": fl}


def _synthetic_outputs(shape: tuple[int, int, int] = (6, 6, 6)) -> dict[str, np.ndarray]:
    ft = np.zeros(shape, dtype=np.float32)
    ft[3, 3, 3] = 1.0
    return {
        "ft_py.dat": ft,
        "pt_py.dat": np.full(shape, 10.0, dtype=np.float32),
        "tt_py.dat": np.full(shape, 70.0, dtype=np.float32),
    }


def _gated_data_root() -> Path:
    if os.environ.get(RUN_ENV_VAR) != "1":
        pytest.skip(f"set {RUN_ENV_VAR}=1 to run the F3 scanner comparison")

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
    if isinstance(metrics, dict):
        for value in metrics.values():
            _assert_finite_metric_values(value)
        return
    if isinstance(metrics, str):
        return

    assert np.isfinite(float(metrics))


def test_parser_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_scanner_module(monkeypatch)

    args = module.build_parser().parse_args([])

    assert args.data_root is None
    assert args.output_json is None
    assert args.output_dir is None
    assert args.save_volumes is False
    assert args.pretty is False
    assert args.crop_shape == (64, 64, 64)
    assert args.max_crops == 1
    assert args.percentile == 99.9
    assert args.min_separation == 48.0
    assert args.sigma1 == 8.0
    assert args.sigma2 == 8.0
    assert args.phi_min == 0.0
    assert args.phi_max == 360.0
    assert args.theta_min == 65.0
    assert args.theta_max == 80.0


def test_build_report_is_json_serializable_for_synthetic_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_scanner_module(monkeypatch)
    outputs = _synthetic_outputs()
    crop = module.build_crop_report(
        crop_index=1,
        center=(3, 3, 3),
        slices=(slice(0, 6), slice(0, 6), slice(0, 6)),
        outputs=outputs,
        reference_fl=outputs["ft_py.dat"].copy(),
    )
    report = module.build_report(
        data_root="/tmp/f3_reference",
        config=module.build_config(
            crop_shape=(6, 6, 6),
            max_crops=1,
            percentile=99.0,
            min_separation=1.0,
            sigma1=8.0,
            sigma2=8.0,
            phi_min=0.0,
            phi_max=360.0,
            theta_min=65.0,
            theta_max=80.0,
        ),
        crops=[crop],
    )

    loaded = json.loads(module.report_to_json(report, pretty=True))

    assert loaded["comparison"] == "scanner-only ft_py.dat versus public fl.dat"
    assert loaded["config"]["comparison"] == "scanner_only_ft_py_vs_fl_dat"
    assert loaded["crops"][0]["pyosv"]["ft_py"]["shape"] == [6, 6, 6]
    assert loaded["crops"][0]["reference"]["fl"]["max"] == 1.0
    assert loaded["crops"][0]["normalized_correlation"]["ft_py_vs_fl"] == pytest.approx(1.0)
    assert loaded["crops"][0]["top_percentile_overlap"]["ft_py_vs_fl"]["99"]["jaccard"] == (
        pytest.approx(1.0)
    )


def test_output_safety_rejects_json_and_dat_paths_inside_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_scanner_module(monkeypatch)
    data_root = tmp_path / "f3_reference"

    with pytest.raises(ValueError, match="inside the F3 data root"):
        module.run_example(
            data_root_arg=data_root,
            output_json=data_root / "scanner_report.json",
            crop_shape=(6, 6, 6),
        )

    with pytest.raises(ValueError, match="inside the F3 data root"):
        module.run_example(
            data_root_arg=data_root,
            output_dir=data_root / "outputs",
            save_volumes=True,
            crop_shape=(6, 6, 6),
        )


def test_run_example_writes_json_and_optional_volumes_without_f3_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_scanner_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_dir = tmp_path / "outputs"
    output_json = tmp_path / "reports" / "scanner.json"
    monkeypatch.setattr(module, "read_reference_arrays", lambda root: _synthetic_reference_arrays())
    monkeypatch.setattr(module, "run_scanner", lambda ep, **kwargs: _synthetic_outputs(ep.shape))

    report = module.run_example(
        data_root_arg=data_root,
        output_json=output_json,
        output_dir=output_dir,
        save_volumes=True,
        pretty=True,
        crop_shape=(6, 6, 6),
        max_crops=1,
        percentile=99.0,
        min_separation=1.0,
    )

    assert output_json.is_file()
    assert not (data_root / "scanner.json").exists()
    with output_json.open(encoding="utf-8") as file:
        loaded = json.load(file)
    assert loaded["format_version"] == 1
    assert loaded["data_root"] == str(data_root)
    assert loaded["crops"][0]["center"] == [3, 3, 3]
    assert report["config"]["reference"] == "fl.dat"

    crop_dir = output_dir / "crop_001"
    for name in module.VOLUME_NAMES:
        assert (crop_dir / name).is_file()


def test_save_volumes_requires_output_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_scanner_module(monkeypatch)

    with pytest.raises(ValueError, match="requires --output-dir"):
        module.run_example(
            data_root_arg="/tmp/f3_reference",
            save_volumes=True,
            crop_shape=(6, 6, 6),
        )


@pytest.mark.f3d_reference
def test_f3d_scanner_comparison_one_crop(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = _gated_data_root()
    module = _import_scanner_module(monkeypatch)
    crop_shape = (64, 64, 64)

    arrays = module.read_reference_arrays(data_root)
    centers = pick_reference_centers(
        arrays["fl.dat"],
        count=1,
        percentile=99.9,
        min_separation=48.0,
        crop_shape=crop_shape,
    )
    assert len(centers) == 1

    slices = crop_slices(centers[0], crop_shape, full_shape=arrays["ep.dat"].shape)
    ep_crop = module._crop(arrays["ep.dat"], slices)
    reference_fl = module._crop(arrays["fl.dat"], slices)
    outputs = module.run_scanner(
        ep_crop,
        sigma1=8.0,
        sigma2=8.0,
        phi_min=0.0,
        phi_max=360.0,
        theta_min=65.0,
        theta_max=80.0,
    )
    report = module.build_crop_report(
        crop_index=1,
        center=centers[0],
        slices=slices,
        outputs=outputs,
        reference_fl=reference_fl,
    )

    with capsys.disabled():
        print("\nF3 scanner-only ft_py vs fl.dat report:\n" + pformat(report))

    ft_py = outputs["ft_py.dat"]
    assert ft_py.shape == crop_shape
    assert ft_py.dtype == np.float32
    assert np.isfinite(ft_py).all()
    assert float(ft_py.max()) > 0.0
    assert np.isfinite(report["normalized_correlation"]["ft_py_vs_fl"])
    _assert_finite_metric_values(report["top_percentile_overlap"]["ft_py_vs_fl"])
    _assert_finite_metric_values(report["slice_correlation"]["ft_py_vs_fl_i3"])
