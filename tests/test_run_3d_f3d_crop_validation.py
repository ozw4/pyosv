from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pyosv.f3d_reference import F3D_ENV_VAR


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"


def _import_validation_module(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop("run_3d_f3d_crop_validation", None)
    importlib.invalidate_caches()
    return importlib.import_module("run_3d_f3d_crop_validation")


def _synthetic_reference_arrays(shape: tuple[int, int, int] = (6, 6, 6)) -> dict[str, np.ndarray]:
    ep = np.zeros(shape, dtype=np.float32)
    fv = np.zeros(shape, dtype=np.float32)
    fvt = np.zeros(shape, dtype=np.float32)
    fv[3, 3, 3] = 1.0
    fvt[3, 3, 3] = 1.0
    return {"ep.dat": ep, "fv.dat": fv, "fvt.dat": fvt}


def _synthetic_outputs(shape: tuple[int, int, int] = (6, 6, 6)) -> dict[str, np.ndarray]:
    base = np.zeros(shape, dtype=np.float32)
    base[3, 3, 3] = 1.0
    return {
        "ft_py.dat": base.copy(),
        "pt_py.dat": np.full(shape, 10.0, dtype=np.float32),
        "tt_py.dat": np.full(shape, 70.0, dtype=np.float32),
        "fet_py.dat": base.copy(),
        "fpt_py.dat": np.full(shape, 10.0, dtype=np.float32),
        "ftt_py.dat": np.full(shape, 70.0, dtype=np.float32),
        "fv_py.dat": base.copy(),
        "fvt_py.dat": base.copy(),
    }


def test_parser_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_validation_module(monkeypatch)

    args = module.build_parser().parse_args([])

    assert args.data_root is None
    assert args.output_dir is None
    assert args.save_volumes is False
    assert args.crop_shape is None
    assert args.center is None
    assert args.large_crop_preset is False
    assert args.max_crops == 1
    assert args.percentile == 99.9
    assert args.min_separation == 48.0
    assert args.sigma1 == 8.0
    assert args.sigma2 == 8.0
    assert args.phi_min == 0.0
    assert args.phi_max == 360.0
    assert args.theta_min == 65.0
    assert args.theta_max == 80.0
    assert args.ru == 10
    assert args.rv == 20
    assert args.rw == 30
    assert args.strain_max1 == 0.25
    assert args.strain_max2 == 0.25
    assert args.surface_smoothing1 == 2.0
    assert args.surface_smoothing2 == 2.0
    assert args.d == 4
    assert args.fm == 0.3
    assert args.interior_margin is None


def test_crop_shape_center_and_large_preset_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_validation_module(monkeypatch)

    args = module.build_parser().parse_args(
        ["--crop-shape", "128,128,100", "--center", "210,200,50", "--large-crop-preset"]
    )

    assert args.crop_shape == (128, 128, 100)
    assert args.center == (210, 200, 50)
    assert args.large_crop_preset is True


def test_large_crop_preset_resolves_default_shape_and_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_validation_module(monkeypatch)

    crop_shape, interior_margin = module.resolve_crop_config(
        crop_shape=None,
        interior_margin=None,
        large_crop_preset=True,
    )

    assert crop_shape == (128, 128, 100)
    assert interior_margin == 40


def test_interior_margin_rejects_impossible_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_validation_module(monkeypatch)

    with pytest.raises(ValueError, match="too large"):
        module.resolve_crop_config(
            crop_shape=(8, 8, 8),
            interior_margin=4,
            large_crop_preset=False,
        )


def test_metrics_helper_on_synthetic_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_validation_module(monkeypatch)
    outputs = _synthetic_outputs()
    reference = _synthetic_reference_arrays()

    report = module.build_crop_report(
        crop_index=1,
        center=(3, 3, 3),
        slices=(slice(0, 6), slice(0, 6), slice(0, 6)),
        crop_shape=(6, 6, 6),
        outputs=outputs,
        reference_fv=reference["fv.dat"],
        reference_fvt=reference["fvt.dat"],
        interior_margin=1,
    )

    assert report["crop_center"] == [3, 3, 3]
    assert report["crop_shape"] == [6, 6, 6]
    assert report["crop_slices"] == [
        {"axis": "i3", "start": 0, "stop": 6},
        {"axis": "i2", "start": 0, "stop": 6},
        {"axis": "i1", "start": 0, "stop": 6},
    ]
    assert report["interior_slices"] == [
        {"axis": "i3", "start": 1, "stop": 5},
        {"axis": "i2", "start": 1, "stop": 5},
        {"axis": "i1", "start": 1, "stop": 5},
    ]
    assert report["pyosv"]["fv"]["max"] == 1.0
    assert report["reference"]["fvt"]["nonzero_fraction"] == pytest.approx(1.0 / 216.0)
    assert report["normalized_correlation"]["full_crop"]["fv"] == pytest.approx(1.0)
    assert report["normalized_correlation"]["interior"]["fvt"] == pytest.approx(1.0)
    assert report["top_percentile_overlap"]["full_crop"]["fv"]["99"]["jaccard"] == pytest.approx(
        1.0
    )
    assert report["top_percentile_overlap"]["interior"]["fvt"]["99"]["jaccard"] == pytest.approx(
        1.0
    )
    assert report["buffered_ridge_overlap"]["interior"]["fvt"]["buffered_f1"] == pytest.approx(1.0)
    assert report["sparse_ridge_distance_metrics"]["interior"]["fvt"][
        "candidate_to_reference_mean"
    ] == pytest.approx(0.0)
    assert report["finite_checks"]["pyosv"]["fv_py"]["finite_fraction"] == 1.0


def test_run_example_writes_metrics_json_to_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_validation_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_dir = tmp_path / "outputs"
    monkeypatch.setenv(F3D_ENV_VAR, str(data_root))
    monkeypatch.setattr(module, "read_reference_arrays", lambda root: _synthetic_reference_arrays())
    monkeypatch.setattr(module, "run_pipeline", lambda ep, **kwargs: _synthetic_outputs(ep.shape))

    report = module.run_example(
        data_root_arg=None,
        output_dir=output_dir,
        crop_shape=(6, 6, 6),
        max_crops=1,
        percentile=99.0,
        min_separation=1.0,
        interior_margin=1,
        pretty=True,
    )

    metrics_path = output_dir / "metrics.json"
    assert metrics_path.is_file()
    assert not (data_root / "metrics.json").exists()
    with metrics_path.open(encoding="utf-8") as file:
        loaded = json.load(file)
    assert loaded["format_version"] == 2
    assert loaded["data_root"] == str(data_root)
    assert loaded["crops"][0]["crop_center"] == [3, 3, 3]
    assert report["config"]["crop_shape"] == [6, 6, 6]


def test_save_volumes_writes_crop_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_validation_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(module, "read_reference_arrays", lambda root: _synthetic_reference_arrays())
    monkeypatch.setattr(module, "run_pipeline", lambda ep, **kwargs: _synthetic_outputs(ep.shape))

    module.run_example(
        data_root_arg=data_root,
        output_dir=output_dir,
        save_volumes=True,
        crop_shape=(6, 6, 6),
        max_crops=1,
        percentile=99.0,
        min_separation=1.0,
        interior_margin=1,
    )

    crop_dir = output_dir / "crop_001"
    for name in module.VOLUME_NAMES:
        assert (crop_dir / name).is_file()


def test_run_example_rejects_output_inside_data_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_validation_module(monkeypatch)
    data_root = tmp_path / "f3_reference"

    with pytest.raises(ValueError, match="inside the F3 data root"):
        module.run_example(
            data_root_arg=data_root,
            output_dir=data_root / "outputs",
            crop_shape=(6, 6, 6),
            interior_margin=1,
        )


def test_save_volumes_requires_output_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_validation_module(monkeypatch)

    with pytest.raises(ValueError, match="requires --output-dir"):
        module.run_example(
            data_root_arg="/tmp/f3_reference",
            save_volumes=True,
            crop_shape=(6, 6, 6),
            interior_margin=1,
        )


def test_import_does_not_run_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_validation_module(monkeypatch)

    assert callable(module.build_parser)
    assert callable(module.main)
    assert callable(module.run_example)
