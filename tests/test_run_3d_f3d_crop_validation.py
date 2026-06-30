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
    fl = np.zeros(shape, dtype=np.float32)
    fv = np.zeros(shape, dtype=np.float32)
    fvt = np.zeros(shape, dtype=np.float32)
    fl[3, 3, 3] = 1.0
    fv[3, 3, 3] = 1.0
    fvt[3, 3, 3] = 1.0
    return {"ep.dat": ep, "fl.dat": fl, "fv.dat": fv, "fvt.dat": fvt}


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
    assert args.save_figures is False
    assert args.figure_percentile == 99.0
    assert args.ridge_buffer_radius == 2.0
    assert args.figure_slices == "center"
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
    assert args.scanner_thin_mode == "normal"
    assert args.voter_thin_mode == "normal"
    assert args.reference_thin_sigma == 1.0


def test_parser_accepts_and_rejects_thinning_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_validation_module(monkeypatch)

    args = module.build_parser().parse_args(
        [
            "--scanner-thin-mode",
            "reference",
            "--voter-thin-mode",
            "reference",
            "--reference-thin-sigma",
            "1.5",
        ]
    )

    assert args.scanner_thin_mode == "reference"
    assert args.voter_thin_mode == "reference"
    assert args.reference_thin_sigma == 1.5
    with pytest.raises(SystemExit):
        module.build_parser().parse_args(["--scanner-thin-mode", "bad"])


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
    assert loaded["config"]["scanner"]["thin_mode"] == "normal"
    assert loaded["config"]["voter"]["thin_mode"] == "normal"
    assert loaded["config"]["scanner"]["reference_thin_sigma"] == 1.0
    assert loaded["config"]["voter"]["reference_thin_sigma"] == 1.0
    assert not (output_dir / "crop_001" / "figures").exists()


def test_run_example_records_selected_thinning_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_validation_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_dir = tmp_path / "outputs"
    received_kwargs = {}
    monkeypatch.setattr(module, "read_reference_arrays", lambda root: _synthetic_reference_arrays())

    def fake_run_pipeline(ep: np.ndarray, **kwargs: object) -> dict[str, np.ndarray]:
        received_kwargs.update(kwargs)
        return _synthetic_outputs(ep.shape)

    monkeypatch.setattr(module, "run_pipeline", fake_run_pipeline)

    report = module.run_example(
        data_root_arg=data_root,
        output_dir=output_dir,
        crop_shape=(6, 6, 6),
        max_crops=1,
        percentile=99.0,
        min_separation=1.0,
        interior_margin=1,
        scanner_thin_mode="reference",
        voter_thin_mode="reference",
        reference_thin_sigma=1.25,
    )

    assert report["config"]["scanner"]["thin_mode"] == "reference"
    assert report["config"]["voter"]["thin_mode"] == "reference"
    assert report["config"]["scanner"]["reference_thin_sigma"] == 1.25
    assert report["config"]["voter"]["reference_thin_sigma"] == 1.25
    assert received_kwargs["scanner_thin_mode"] == "reference"
    assert received_kwargs["voter_thin_mode"] == "reference"
    assert received_kwargs["reference_thin_sigma"] == 1.25


def test_small_pipeline_accepts_reference_thinning(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_validation_module(monkeypatch)

    outputs = module.run_pipeline(
        np.zeros((4, 4, 4), dtype=np.float32),
        sigma1=1.0,
        sigma2=1.0,
        phi_min=0.0,
        phi_max=0.0,
        theta_min=70.0,
        theta_max=70.0,
        ru=1,
        rv=1,
        rw=1,
        strain_max1=0.25,
        strain_max2=0.25,
        surface_smoothing1=1.0,
        surface_smoothing2=1.0,
        d=1,
        fm=0.3,
        scanner_thin_mode="reference",
        voter_thin_mode="reference",
        reference_thin_sigma=1.0,
    )

    assert outputs["fet_py.dat"].shape == (4, 4, 4)
    assert outputs["fvt_py.dat"].shape == (4, 4, 4)


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


def test_save_figures_requires_output_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_validation_module(monkeypatch)

    with pytest.raises(ValueError, match="requires --output-dir"):
        module.run_example(
            data_root_arg="/tmp/f3_reference",
            save_figures=True,
            crop_shape=(6, 6, 6),
            interior_margin=1,
        )


def test_save_figures_writes_expected_pngs_and_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    module = _import_validation_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_dir = tmp_path / "outputs"
    monkeypatch.setattr(module, "read_reference_arrays", lambda root: _synthetic_reference_arrays())
    monkeypatch.setattr(module, "run_pipeline", lambda ep, **kwargs: _synthetic_outputs(ep.shape))

    report = module.run_example(
        data_root_arg=data_root,
        output_dir=output_dir,
        save_figures=True,
        crop_shape=(6, 6, 6),
        max_crops=1,
        percentile=99.0,
        min_separation=1.0,
        interior_margin=1,
        pretty=True,
    )

    figures_dir = output_dir / "crop_001" / "figures"
    expected_names = {
        "scanner_fl_vs_ftpy_i3_3.png",
        "scanner_fl_vs_ftpy_i2_3.png",
        "scanner_fl_vs_ftpy_i1_3.png",
        "fv_ref_vs_py_i3_3.png",
        "fv_ref_vs_py_i2_3.png",
        "fv_ref_vs_py_i1_3.png",
        "fvt_ref_vs_py_i3_3.png",
        "fvt_ref_vs_py_i2_3.png",
        "fvt_ref_vs_py_i1_3.png",
        "fvt_ridge_overlay_i3_3.png",
        "fvt_ridge_overlay_i2_3.png",
        "fvt_ridge_overlay_i1_3.png",
        "fv_mip.png",
        "fvt_mip.png",
        "fv_hist.png",
        "fvt_hist.png",
    }

    assert {path.name for path in figures_dir.iterdir()} == expected_names
    for name in expected_names:
        assert (figures_dir / name).stat().st_size > 0

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open(encoding="utf-8") as file:
        loaded = json.load(file)
    figures = loaded["crops"][0]["figures"]
    assert figures == report["crops"][0]["figures"]
    assert figures["directory"] == "crop_001/figures"
    assert figures["figure_slices"] == "center"
    assert figures["slice_indices"] == {"i1": 3, "i2": 3, "i3": 3}
    assert figures["files"]["scanner_fl_vs_ftpy"]["i3"] == (
        "crop_001/figures/scanner_fl_vs_ftpy_i3_3.png"
    )


def test_main_reports_viz_extra_when_matplotlib_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _import_validation_module(monkeypatch)
    from pyosv import viz

    def raise_missing_matplotlib() -> None:
        raise ImportError('matplotlib is required. Install it with `pip install "pyosv[viz]"`.')

    monkeypatch.setattr(viz, "require_matplotlib", raise_missing_matplotlib)

    exit_code = module.main(
        [
            "--data-root",
            str(tmp_path / "f3_reference"),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--save-figures",
            "--crop-shape",
            "6,6,6",
            "--interior-margin",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "pyosv[viz]" in captured.err


def test_import_does_not_run_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_validation_module(monkeypatch)

    assert callable(module.build_parser)
    assert callable(module.main)
    assert callable(module.run_example)
