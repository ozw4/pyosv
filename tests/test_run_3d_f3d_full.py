from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from pyosv.f3d_reference import F3D_ENV_VAR


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
RUN_ENV_VAR = "PYOSV_RUN_F3D_FULL_COMPARISON"
OUTPUT_ENV_VAR = "PYOSV_F3D_FULL_COMPARISON_OUTPUT_DIR"
REQUIRED_FILES = ("ep.dat", "fl.dat", "fv.dat", "fvt.dat")


def _import_full_module(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop("run_3d_f3d_full", None)
    importlib.invalidate_caches()
    return importlib.import_module("run_3d_f3d_full")


def _synthetic_outputs(shape: tuple[int, int, int] = (4, 4, 4)) -> dict[str, np.ndarray]:
    ft = np.zeros(shape, dtype=np.float32)
    fv = np.zeros(shape, dtype=np.float32)
    fvt = np.zeros(shape, dtype=np.float32)
    center = tuple(size // 2 for size in shape)
    ft[center] = 1.0
    fv[center] = 1.0
    fvt[center] = 1.0
    return {"ft_py.dat": ft, "fv_py.dat": fv, "fvt_py.dat": fvt}


def _full_synthetic_outputs(shape: tuple[int, int, int] = (4, 4, 4)) -> dict[str, np.ndarray]:
    outputs = _synthetic_outputs(shape)
    outputs.update(
        {
            "pt_py.dat": np.full(shape, 10.0, dtype=np.float32),
            "tt_py.dat": np.full(shape, 70.0, dtype=np.float32),
            "fet_py.dat": outputs["ft_py.dat"].copy(),
            "fpt_py.dat": np.full(shape, 10.0, dtype=np.float32),
            "ftt_py.dat": np.full(shape, 70.0, dtype=np.float32),
            "vp_py.dat": np.full(shape, 10.0, dtype=np.float32),
            "vt_py.dat": np.full(shape, 70.0, dtype=np.float32),
        }
    )
    return outputs


def _gated_data_root() -> Path:
    if os.environ.get(RUN_ENV_VAR) != "1":
        pytest.skip(f"set {RUN_ENV_VAR}=1 to run the full F3 comparison")

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


def test_parser_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _import_full_module(monkeypatch)

    args = module.build_parser().parse_args(["--output-dir", str(tmp_path)])

    assert args.output_dir == tmp_path
    assert args.data_root is None
    assert args.output_json is None
    assert args.pretty is False
    assert args.sigma1 == 8.0
    assert args.sigma2 == 8.0
    assert args.phi_min == 0.0
    assert args.phi_max == 360.0
    assert args.theta_min == 65.0
    assert args.theta_max == 80.0
    assert args.ru == 10
    assert args.rv == 20
    assert args.rw == 30
    assert args.d == 4
    assert args.fm == 0.3
    assert args.strain_max1 == 0.25
    assert args.strain_max2 == 0.25
    assert args.surface_smoothing1 == 2.0
    assert args.surface_smoothing2 == 2.0
    assert args.reuse_existing is False
    assert args.skip_save_intermediates is False
    assert args.save_volumes is True

    no_save_args = module.build_parser().parse_args(
        ["--output-dir", str(tmp_path), "--no-save-volumes"]
    )
    assert no_save_args.save_volumes is False


def test_output_dir_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_full_module(monkeypatch)

    with pytest.raises(SystemExit):
        module.build_parser().parse_args([])


def test_ensure_output_not_in_data_root_rejects_equal_and_nested_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_full_module(monkeypatch)
    data_root = tmp_path / "f3_reference"

    with pytest.raises(ValueError, match="inside the F3 data root"):
        module.ensure_output_not_in_data_root(data_root, data_root)

    with pytest.raises(ValueError, match="inside the F3 data root"):
        module.ensure_output_not_in_data_root(data_root / "outputs", data_root)

    assert module.ensure_output_not_in_data_root(tmp_path / "outputs", data_root) == (
        tmp_path / "outputs"
    ).resolve(strict=False)


def test_build_run_config_is_serializable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    module = _import_full_module(monkeypatch)
    config = module.build_run_config(
        data_root=tmp_path / "f3_reference",
        output_dir=tmp_path / "outputs",
        sigma1=8.0,
        sigma2=8.0,
        phi_min=0.0,
        phi_max=360.0,
        theta_min=65.0,
        theta_max=80.0,
        ru=10,
        rv=20,
        rw=30,
        d=4,
        fm=0.3,
        strain_max1=0.25,
        strain_max2=0.25,
        surface_smoothing1=2.0,
        surface_smoothing2=2.0,
        reuse_existing=True,
        skip_save_intermediates=True,
        save_volumes=True,
        output_json=tmp_path / "metrics.json",
    )

    loaded = json.loads(module.report_to_json(config))

    assert loaded["format_version"] == 1
    assert loaded["input"] == "ep.dat"
    assert loaded["reference"] == ["fl.dat", "fv.dat", "fvt.dat"]
    assert loaded["scanner"]["theta_min"] == 65.0
    assert loaded["voter"]["ru"] == 10
    assert loaded["reuse_existing"] is True
    assert loaded["skip_save_intermediates"] is True
    assert loaded["save_volumes"] is True
    assert loaded["outputs"]["report"] == ["ft_py.dat", "fv_py.dat", "fvt_py.dat"]
    assert loaded["outputs"]["final"] == ["fv_py.dat", "fvt_py.dat"]
    assert loaded["outputs"]["intermediate"] == []


def test_output_json_safety_rejects_path_under_data_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_full_module(monkeypatch)
    data_root = tmp_path / "f3_reference"

    with pytest.raises(ValueError, match="--output-json must not be inside"):
        module.run_example(
            data_root_arg=data_root,
            output_dir=tmp_path / "outputs",
            output_json=data_root / "metrics.json",
        )


def test_reuse_mode_reports_missing_intermediate_outputs_clearly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_full_module(monkeypatch)
    for name in module.REPORT_OUTPUT_NAMES:
        (tmp_path / name).write_bytes(b"data")

    with pytest.raises(FileNotFoundError, match="pt_py.dat"):
        module.run_or_reuse_pipeline(
            data_root=tmp_path / "f3_reference",
            output_dir=tmp_path,
            sigma1=8.0,
            sigma2=8.0,
            phi_min=0.0,
            phi_max=360.0,
            theta_min=65.0,
            theta_max=80.0,
            ru=10,
            rv=20,
            rw=30,
            d=4,
            fm=0.3,
            strain_max1=0.25,
            strain_max2=0.25,
            surface_smoothing1=2.0,
            surface_smoothing2=2.0,
            reuse_existing=True,
            skip_save_intermediates=False,
            save_volumes=False,
        )


def test_reuse_mode_reads_each_full_stage_output_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_full_module(monkeypatch)
    for name in module.OUTPUT_NAMES:
        (tmp_path / name).write_bytes(b"data")

    read_names: list[tuple[str, ...]] = []

    def fake_read_outputs(output_dir: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
        read_names.append(names)
        full_outputs = _full_synthetic_outputs()
        return {name: full_outputs[name] for name in names}

    monkeypatch.setattr(module, "read_outputs", fake_read_outputs)

    outputs, runtime = module.run_or_reuse_pipeline(
        data_root=tmp_path / "f3_reference",
        output_dir=tmp_path,
        sigma1=8.0,
        sigma2=8.0,
        phi_min=0.0,
        phi_max=360.0,
        theta_min=65.0,
        theta_max=80.0,
        ru=10,
        rv=20,
        rw=30,
        d=4,
        fm=0.3,
        strain_max1=0.25,
        strain_max2=0.25,
        surface_smoothing1=2.0,
        surface_smoothing2=2.0,
        reuse_existing=True,
        skip_save_intermediates=False,
        save_volumes=False,
    )

    assert read_names == [
        module.SCANNER_OUTPUT_NAMES,
        module.SCANNER_THIN_OUTPUT_NAMES,
        module.VOTING_OUTPUT_NAMES,
        ("fvt_py.dat",),
    ]
    assert tuple(outputs) == module.REPORT_OUTPUT_NAMES
    assert runtime["mode"] == "reuse_existing"
    assert runtime["reused_stages"] == ["scanner", "scanner_thin", "voting", "voter_thin"]
    assert runtime["computed_stages"] == []


def test_reuse_mode_rejects_completed_scanner_stage_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_full_module(monkeypatch)
    for name in module.SCANNER_OUTPUT_NAMES:
        (tmp_path / name).write_bytes(b"data")

    def fake_read_outputs(output_dir: Path, names: tuple[str, ...]) -> dict[str, np.ndarray]:
        raise AssertionError(f"reuse should fail before reading {names}")

    monkeypatch.setattr(module, "read_outputs", fake_read_outputs)
    monkeypatch.setattr(module, "read_f3d_file", lambda name, root: pytest.fail(name))

    with pytest.raises(FileNotFoundError, match="fet_py.dat"):
        module.run_or_reuse_pipeline(
            data_root=tmp_path / "f3_reference",
            output_dir=tmp_path,
            sigma1=8.0,
            sigma2=8.0,
            phi_min=0.0,
            phi_max=360.0,
            theta_min=65.0,
            theta_max=80.0,
            ru=10,
            rv=20,
            rw=30,
            d=4,
            fm=0.3,
            strain_max1=0.25,
            strain_max2=0.25,
            surface_smoothing1=2.0,
            surface_smoothing2=2.0,
            reuse_existing=True,
            skip_save_intermediates=False,
            save_volumes=False,
        )


def test_should_reuse_outputs_requires_flag_and_all_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_full_module(monkeypatch)
    (tmp_path / "ft_py.dat").write_bytes(b"data")

    assert module.should_reuse_outputs(tmp_path, ("ft_py.dat",), reuse_existing=False) is False
    assert module.should_reuse_outputs(tmp_path, ("ft_py.dat",), reuse_existing=True) is True
    assert (
        module.should_reuse_outputs(
            tmp_path,
            ("ft_py.dat", "pt_py.dat"),
            reuse_existing=True,
        )
        is False
    )


def test_write_outputs_skip_intermediates_keeps_only_report_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_full_module(monkeypatch)
    outputs = {
        "ft_py.dat": np.full((2, 2, 2), 2.0, dtype=np.float32),
        "fv_py.dat": np.zeros((2, 2, 2), dtype=np.float32),
        "vp_py.dat": np.ones((2, 2, 2), dtype=np.float32),
        "fvt_py.dat": np.ones((2, 2, 2), dtype=np.float32),
    }

    module.write_outputs(
        tmp_path,
        outputs,
        ("ft_py.dat", "fv_py.dat", "vp_py.dat", "fvt_py.dat"),
        skip_intermediates=True,
    )

    assert (tmp_path / "ft_py.dat").is_file()
    assert (tmp_path / "fv_py.dat").is_file()
    assert not (tmp_path / "vp_py.dat").exists()
    assert (tmp_path / "fvt_py.dat").is_file()


def test_run_example_writes_config_before_heavy_processing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_full_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_dir = tmp_path / "outputs"
    monkeypatch.setenv(F3D_ENV_VAR, str(data_root))

    def fail_pipeline(**kwargs: object) -> dict[str, np.ndarray]:
        raise RuntimeError("pipeline should fail after config is written")

    monkeypatch.setattr(module, "run_or_reuse_pipeline", fail_pipeline)

    with pytest.raises(RuntimeError, match="pipeline should fail"):
        module.run_example(data_root_arg=None, output_dir=output_dir)

    config_path = output_dir / "run_config.json"
    assert config_path.is_file()
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)
    assert config["data_root"] == str(data_root)
    assert config["output_dir"] == str(output_dir.resolve(strict=False))


def test_run_example_writes_final_outputs_and_metrics_without_f3_data(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_full_module(monkeypatch)
    data_root = tmp_path / "f3_reference"
    output_dir = tmp_path / "outputs"
    outputs = _synthetic_outputs()

    monkeypatch.setattr(
        module, "run_or_reuse_pipeline", lambda **kwargs: (outputs, {"mode": "test"})
    )
    monkeypatch.setattr(
        module,
        "read_f3d_file",
        lambda name, root: {
            "fl.dat": outputs["ft_py.dat"],
            "fv.dat": outputs["fv_py.dat"],
            "fvt.dat": outputs["fvt_py.dat"],
        }[name],
    )

    report = module.run_example(
        data_root_arg=data_root,
        output_dir=output_dir,
        skip_save_intermediates=True,
    )

    metrics_path = output_dir / "metrics.json"
    assert metrics_path.is_file()
    assert not (data_root / "metrics.json").exists()
    assert report["scanner"]["ft_py_vs_fl"]["normalized_correlation"] == pytest.approx(1.0)
    assert report["voting"]["fv_py_vs_fv"]["normalized_correlation"] == pytest.approx(1.0)
    assert report["thinning"]["fvt_py_vs_fvt"]["top_percentile_overlap"]["99"][
        "jaccard"
    ] == pytest.approx(1.0)


def test_build_metrics_report_on_small_synthetic_arrays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _import_full_module(monkeypatch)
    outputs = _synthetic_outputs()
    config = module.build_run_config(
        data_root=tmp_path / "f3_reference",
        output_dir=tmp_path / "outputs",
        output_json=tmp_path / "outputs" / "metrics.json",
        sigma1=8.0,
        sigma2=8.0,
        phi_min=0.0,
        phi_max=360.0,
        theta_min=65.0,
        theta_max=80.0,
        ru=10,
        rv=20,
        rw=30,
        d=4,
        fm=0.3,
        strain_max1=0.25,
        strain_max2=0.25,
        surface_smoothing1=2.0,
        surface_smoothing2=2.0,
        reuse_existing=True,
        skip_save_intermediates=False,
        save_volumes=False,
    )

    report = module.build_metrics_report(
        data_root=tmp_path / "f3_reference",
        config=config,
        pyosv_ft=outputs["ft_py.dat"],
        pyosv_fv=outputs["fv_py.dat"],
        pyosv_fvt=outputs["fvt_py.dat"],
        reference_fl=outputs["ft_py.dat"].copy(),
        reference_fv=outputs["fv_py.dat"].copy(),
        reference_fvt=outputs["fvt_py.dat"].copy(),
        runtime={"mode": "test"},
    )

    loaded = json.loads(module.report_to_json(report, pretty=True))
    assert loaded["data"]["shape"] == list(module.F3D_SHAPE)
    assert loaded["scanner"]["parameters"]["sigma1"] == 8.0
    assert loaded["scanner"]["ft_py_vs_fl"]["normalized_correlation"] == pytest.approx(1.0)
    assert loaded["voting"]["fv_py_vs_fv"]["nonzero_fraction_ratio"] == pytest.approx(1.0)
    assert loaded["thinning"]["fvt_py_vs_fvt"]["buffered_ridge_overlap"]["buffered_f1"] == (
        pytest.approx(1.0)
    )


@pytest.mark.f3d_reference
def test_gated_real_data_reuse_report_if_outputs_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data_root = _gated_data_root()
    module = _import_full_module(monkeypatch)
    output_dir_text = os.environ.get(OUTPUT_ENV_VAR)
    if output_dir_text is None:
        pytest.skip(f"set {OUTPUT_ENV_VAR} to an output directory with full F3 pyosv outputs")
    output_dir = Path(output_dir_text)

    if not all((output_dir / name).is_file() for name in module.OUTPUT_NAMES):
        pytest.skip("full F3 pyosv output set is not present for reuse-only report assembly")

    report = module.run_example(
        data_root_arg=data_root,
        output_dir=output_dir,
        output_json=tmp_path / "metrics.json",
        reuse_existing=True,
    )

    assert report["runtime"]["mode"] == "reuse_existing"
    assert "ft_py_vs_fl" in report["scanner"]
