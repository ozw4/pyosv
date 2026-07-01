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
RUN_ENV_VAR = "PYOSV_RUN_F3D_THINNING_ABLATION"
REQUIRED_FILES = ("ep.dat", "fv.dat", "fvt.dat")


def _import_ablation_module(monkeypatch: pytest.MonkeyPatch) -> object:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop("report_3d_f3d_thinning_ablation", None)
    importlib.invalidate_caches()
    return importlib.import_module("report_3d_f3d_thinning_ablation")


def _synthetic_reference_arrays(shape: tuple[int, int, int] = (8, 8, 8)) -> dict[str, np.ndarray]:
    ep = np.zeros(shape, dtype=np.float32)
    fv = np.zeros(shape, dtype=np.float32)
    fvt = np.zeros(shape, dtype=np.float32)
    ep[2, 2, 2] = 1.0
    fv[2, 2, 2] = 3.0
    fv[5, 5, 5] = 2.0
    fvt[2, 2, 2] = 3.0
    fvt[5, 5, 5] = 2.0
    return {"ep.dat": ep, "fv.dat": fv, "fvt.dat": fvt}


def _case_outputs(module: object, shape: tuple[int, int, int]) -> dict[str, dict[str, np.ndarray]]:
    base = np.zeros(shape, dtype=np.float32)
    center = tuple(size // 2 for size in shape)
    base[center] = 1.0
    outputs = {}
    for index, case in enumerate(module.CASE_DEFINITIONS, start=1):
        fv = base.copy()
        fvt = base.copy()
        fv[center] = np.float32(index)
        fvt[center] = np.float32(index)
        outputs[case["name"]] = {
            "ft_py.dat": base.copy(),
            "pt_py.dat": np.full(shape, 10.0, dtype=np.float32),
            "tt_py.dat": np.full(shape, 70.0, dtype=np.float32),
            "fet_py.dat": base.copy(),
            "fpt_py.dat": np.full(shape, 10.0, dtype=np.float32),
            "ftt_py.dat": np.full(shape, 70.0, dtype=np.float32),
            "fv_py.dat": fv,
            "fvt_py.dat": fvt,
        }
    return outputs


def _gated_data_root() -> Path:
    if os.environ.get(RUN_ENV_VAR) != "1":
        pytest.skip(f"set {RUN_ENV_VAR}=1 to run the F3 thinning ablation pipeline")

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


def test_parser_accepts_expected_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_ablation_module(monkeypatch)

    defaults = module.build_parser().parse_args([])
    assert defaults.data_root is None
    assert defaults.output_json is None
    assert defaults.save_figures is False
    assert defaults.write_markdown_index is False
    assert defaults.count == 3
    assert defaults.crop_shape == (64, 64, 64)
    assert defaults.interior_margin == 16
    assert defaults.scanner_backends == ("current",)
    assert defaults.center is None

    args = module.build_parser().parse_args(
        [
            "--output-json",
            "outputs/3d/f3d/thinning_ablation_001/metrics.json",
            "--count",
            "3",
            "--crop-shape",
            "64,64,64",
            "--interior-margin",
            "16",
            "--scanner-backends",
            "current,reference-like",
            "--pretty",
            "--save-figures",
            "--write-markdown-index",
            "--center",
            "2,3,4",
        ]
    )
    assert args.output_json == Path("outputs/3d/f3d/thinning_ablation_001/metrics.json")
    assert args.count == 3
    assert args.crop_shape == (64, 64, 64)
    assert args.interior_margin == 16
    assert args.scanner_backends == ("current", "reference-like")
    assert args.pretty is True
    assert args.save_figures is True
    assert args.write_markdown_index is True
    assert args.center == [(2, 3, 4)]

    singular = module.build_parser().parse_args(["--scanner-backend", "reference-like"])
    assert singular.scanner_backends == ("reference-like",)


def test_output_path_safety_rejects_data_root_and_reference_osv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_ablation_module(monkeypatch)
    data_root = tmp_path / "f3_reference"

    with pytest.raises(ValueError, match="F3 data root"):
        module.run_example(
            data_root_arg=data_root,
            output_json=data_root / "outputs" / "metrics.json",
            crop_shape=(6, 6, 6),
            interior_margin=1,
        )

    with pytest.raises(ValueError, match="reference_osv"):
        module.run_example(
            data_root_arg=data_root,
            output_json=REPO_ROOT / "reference_osv" / "outputs" / "metrics.json",
            crop_shape=(6, 6, 6),
            interior_margin=1,
        )


def test_run_example_writes_four_case_json_without_f3_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_ablation_module(monkeypatch)
    data_root = tmp_path / "missing_f3_reference"
    output_json = tmp_path / "outputs" / "metrics.json"
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )
    monkeypatch.setattr(
        module,
        "run_ablation_pipeline",
        lambda ep, **kwargs: _case_outputs(module, ep.shape),
    )

    report = module.run_example(
        data_root_arg=data_root,
        output_json=output_json,
        pretty=True,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
    )

    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    expected_case_names = [case["name"] for case in module.CASE_DEFINITIONS]
    assert report == loaded
    assert loaded["format_version"] == 1
    assert loaded["config"]["comparison"] == "f3d_thinning_ablation"
    assert loaded["config"]["scanner_backends"] == ["current"]
    assert [case["name"] for case in loaded["config"]["cases"]] == expected_case_names
    assert set(loaded["crops"][0]["cases"]) == set(expected_case_names)
    assert set(loaded["crops"][0]["backends"]) == {"current"}
    assert set(loaded["crops"][0]["backends"]["current"]["cases"]) == set(expected_case_names)
    assert set(loaded["aggregate"]["cases"]) == set(expected_case_names)
    assert set(loaded["aggregate"]["backends"]) == {"current"}
    assert (
        loaded["aggregate"]["cases"]["case_01_current_current"]["per_metric_mean"][
            "normalized_correlation.interior.fvt"
        ]
        is not None
    )
    assert loaded["aggregate"]["cases"]["case_01_current_current"]["per_metric_mean"][
        "pyosv.fv.nonzero_fraction"
    ] == pytest.approx(1.0 / 216.0)
    assert loaded["aggregate"]["cases"]["case_01_current_current"]["per_metric_median"][
        "pyosv.fvt.mean"
    ] == pytest.approx(1.0 / 216.0)


def test_run_example_writes_backend_separated_json_without_f3_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_ablation_module(monkeypatch)
    output_json = tmp_path / "outputs" / "metrics.json"
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )

    def fake_pipeline(ep: np.ndarray, **kwargs: object) -> dict[str, dict[str, np.ndarray]]:
        outputs = _case_outputs(module, ep.shape)
        if kwargs["scanner_backend"] == "reference-like":
            for case_outputs in outputs.values():
                case_outputs["fv_py.dat"] = case_outputs["fv_py.dat"] * np.float32(10.0)
                case_outputs["fvt_py.dat"] = case_outputs["fvt_py.dat"] * np.float32(10.0)
        return outputs

    monkeypatch.setattr(module, "run_ablation_pipeline", fake_pipeline)

    report = module.run_example(
        data_root_arg=tmp_path / "f3_reference",
        output_json=output_json,
        pretty=True,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
        scanner_backends=("current", "reference-like"),
    )

    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    expected_case_names = {case["name"] for case in module.CASE_DEFINITIONS}
    assert report == loaded
    assert loaded["config"]["scanner_backends"] == ["current", "reference-like"]
    assert set(loaded["crops"][0]["backends"]) == {"current", "reference-like"}
    assert set(loaded["crops"][0]["backends"]["current"]["cases"]) == expected_case_names
    assert set(loaded["crops"][0]["backends"]["reference-like"]["cases"]) == expected_case_names
    assert "cases" not in loaded["crops"][0]
    assert set(loaded["aggregate"]["backends"]) == {"current", "reference-like"}
    assert set(loaded["aggregate"]["backends"]["current"]["cases"]) == expected_case_names
    assert set(loaded["aggregate"]["backends"]["reference-like"]["cases"]) == expected_case_names
    assert "cases" not in loaded["aggregate"]
    assert loaded["aggregate"]["backends"]["current"]["cases"]["case_01_current_current"][
        "per_metric_mean"
    ]["pyosv.fvt.mean"] == pytest.approx(1.0 / 216.0)
    assert loaded["aggregate"]["backends"]["reference-like"]["cases"]["case_01_current_current"][
        "per_metric_mean"
    ]["pyosv.fvt.mean"] == pytest.approx(10.0 / 216.0)


def test_case_names_and_thinning_modes_are_recorded_in_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_ablation_module(monkeypatch)
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )
    monkeypatch.setattr(
        module,
        "run_ablation_pipeline",
        lambda ep, **kwargs: _case_outputs(module, ep.shape),
    )

    report = module.run_example(
        data_root_arg=tmp_path / "f3_reference",
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
    )

    cases = {case["name"]: case for case in report["config"]["cases"]}
    assert cases["case_01_current_current"] == {
        "name": "case_01_current_current",
        "scanner_thin_mode": "normal",
        "voter_thin_mode": "normal",
    }
    assert cases["case_02_current_reference_voter"]["voter_thin_mode"] == "reference"
    assert cases["case_03_reference_scanner_current"]["scanner_thin_mode"] == "reference"
    assert cases["case_04_reference_reference"] == {
        "name": "case_04_reference_reference",
        "scanner_thin_mode": "reference",
        "voter_thin_mode": "reference",
    }


def test_visual_report_writes_markdown_and_minimum_png_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("matplotlib")
    module = _import_ablation_module(monkeypatch)
    output_json = tmp_path / "outputs" / "metrics.json"
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )
    monkeypatch.setattr(
        module,
        "run_ablation_pipeline",
        lambda ep, **kwargs: _case_outputs(module, ep.shape),
    )

    module.run_example(
        data_root_arg=tmp_path / "f3_reference",
        output_json=output_json,
        save_figures=True,
        write_markdown_index=True,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
    )

    figures_dir = output_json.parent / "crop_001" / "case_01_current_current" / "figures"
    markdown = (output_json.parent / "visual_report.md").read_text(encoding="utf-8")
    assert (figures_dir / "fvt_ref_vs_py_i3_3.png").is_file()
    assert (figures_dir / "fvt_ridge_overlay_i3_3.png").is_file()
    assert (figures_dir / "fvt_mip.png").is_file()
    assert "case_01_current_current" in markdown
    assert "buffered F1" in markdown
    assert "crop_001/case_01_current_current/figures/fvt_mip.png" in markdown


def test_visual_report_uses_backend_case_nesting_for_multiple_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_ablation_module(monkeypatch)
    output_json = tmp_path / "outputs" / "metrics.json"
    monkeypatch.setattr(module.crop_validation, "require_figure_support", lambda: None)
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )
    monkeypatch.setattr(
        module,
        "run_ablation_pipeline",
        lambda ep, **kwargs: _case_outputs(module, ep.shape),
    )
    figure_dirs: list[Path] = []

    def fake_write_case_figures(output_dir: Path, **kwargs: object) -> dict[str, object]:
        figure_dirs.append(Path(output_dir))
        directory = Path(output_dir).relative_to(output_json.parent).as_posix()
        return {
            "directory": directory,
            "files": {
                "fvt": {"mip": f"{directory}/fvt_mip.png"},
                "fvt_ref_vs_py": {"i3": f"{directory}/fvt_ref_vs_py_i3_3.png"},
                "fvt_ridge_overlay": {"i3": f"{directory}/fvt_ridge_overlay_i3_3.png"},
            },
        }

    monkeypatch.setattr(module, "write_case_figures", fake_write_case_figures)

    module.run_example(
        data_root_arg=tmp_path / "f3_reference",
        output_json=output_json,
        save_figures=True,
        write_markdown_index=True,
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
        scanner_backends=("current", "reference-like"),
    )

    markdown = (output_json.parent / "visual_report.md").read_text(encoding="utf-8")
    assert output_json.parent / "crop_001" / "current" / "case_01_current_current" / "figures" in (
        figure_dirs
    )
    assert (
        output_json.parent / "crop_001" / "reference-like" / "case_01_current_current" / "figures"
        in figure_dirs
    )
    assert "| Backend | Case | fvt interior corr mean |" in markdown
    assert "`current` | `case_01_current_current`" in markdown
    assert "`reference-like` | `case_01_current_current`" in markdown
    assert "crop_001/current/case_01_current_current/figures/fvt_mip.png" in markdown
    assert "crop_001/reference-like/case_01_current_current/figures/fvt_mip.png" in markdown


def test_reference_like_scanner_backend_unavailable_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_ablation_module(monkeypatch)

    class CurrentOnlyScanner:
        def scan(self, *args: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            values = np.zeros((2, 2, 2), dtype=np.float32)
            return values, values, values

    with pytest.raises(ValueError, match="reference-like scanner backend is unavailable"):
        module._scan_backend(
            CurrentOnlyScanner(),
            backend="reference-like",
            phi_min=0.0,
            phi_max=1.0,
            theta_min=2.0,
            theta_max=3.0,
            ep=np.zeros((2, 2, 2), dtype=np.float32),
        )


def test_import_does_not_run_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_ablation_module(monkeypatch)

    assert callable(module.build_parser)
    assert callable(module.main)
    assert callable(module.run_example)


@pytest.mark.f3d_reference
def test_gated_real_data_thinning_ablation(monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = _gated_data_root()
    module = _import_ablation_module(monkeypatch)

    report = module.run_example(
        data_root_arg=data_root,
        count=1,
        crop_shape=(48, 48, 48),
        interior_margin=12,
        percentile=99.9,
        min_separation=24.0,
    )

    assert len(report["crops"]) == 1
    assert set(report["crops"][0]["cases"]) == {case["name"] for case in module.CASE_DEFINITIONS}
    assert report["aggregate"]["crop_count"] == 1
