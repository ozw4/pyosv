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
            "vp_py.dat": np.full(shape, 10.0, dtype=np.float32),
            "vt_py.dat": np.full(shape, 70.0, dtype=np.float32),
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
    assert defaults.scanner_backends == ("fast",)
    assert defaults.remove_scanner_edge_effects is True
    assert defaults.center is None
    assert defaults.final_normalization_smoothing is None
    assert "--final-normalization-smoothing" in module.build_parser().format_help()

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
            "fast,reference-like",
            "--pretty",
            "--save-figures",
            "--write-markdown-index",
            "--keep-scanner-edge-effects",
            "--center",
            "2,3,4",
            "--final-normalization-smoothing",
            "1.0",
        ]
    )
    assert args.output_json == Path("outputs/3d/f3d/thinning_ablation_001/metrics.json")
    assert args.count == 3
    assert args.crop_shape == (64, 64, 64)
    assert args.interior_margin == 16
    assert args.scanner_backends == ("fast", "reference-like")
    assert args.pretty is True
    assert args.save_figures is True
    assert args.write_markdown_index is True
    assert args.remove_scanner_edge_effects is False
    assert args.center == [(2, 3, 4)]
    assert args.final_normalization_smoothing == 1.0

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
    assert loaded["config"]["scanner_backends"] == ["fast"]
    assert [case["name"] for case in loaded["config"]["cases"]] == expected_case_names
    assert loaded["config"]["scanner"]["reference_remove_edge_effects"] is True
    assert loaded["config"]["voter"]["final_normalization_smoothing"] == 0.0
    assert loaded["config"]["voter"]["surface_voting_boundary_policy"] == (
        "reference-like-i2-i3-interior"
    )
    assert set(loaded["crops"][0]["cases"]) == set(expected_case_names)
    assert set(loaded["crops"][0]["backends"]) == {"fast"}
    assert set(loaded["crops"][0]["backends"]["fast"]["cases"]) == set(expected_case_names)
    assert set(loaded["aggregate"]["cases"]) == set(expected_case_names)
    assert set(loaded["aggregate"]["backends"]) == {"fast"}
    assert (
        loaded["aggregate"]["cases"]["case_01_normal_normal"]["per_metric_mean"][
            "normalized_correlation.interior.fvt"
        ]
        is not None
    )
    assert loaded["aggregate"]["cases"]["case_01_normal_normal"]["per_metric_mean"][
        "pyosv.fv.nonzero_fraction"
    ] == pytest.approx(1.0 / 216.0)
    assert loaded["aggregate"]["cases"]["case_01_normal_normal"]["per_metric_median"][
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
        scanner_backends=("fast", "reference-like"),
    )

    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    expected_case_names = {case["name"] for case in module.CASE_DEFINITIONS}
    assert report == loaded
    assert loaded["config"]["scanner_backends"] == ["fast", "reference-like"]
    assert set(loaded["crops"][0]["backends"]) == {"fast", "reference-like"}
    assert set(loaded["crops"][0]["backends"]["fast"]["cases"]) == expected_case_names
    assert set(loaded["crops"][0]["backends"]["reference-like"]["cases"]) == expected_case_names
    assert "cases" not in loaded["crops"][0]
    assert set(loaded["aggregate"]["backends"]) == {"fast", "reference-like"}
    assert set(loaded["aggregate"]["backends"]["fast"]["cases"]) == expected_case_names
    assert set(loaded["aggregate"]["backends"]["reference-like"]["cases"]) == expected_case_names
    assert "cases" not in loaded["aggregate"]
    assert loaded["aggregate"]["backends"]["fast"]["cases"]["case_01_normal_normal"][
        "per_metric_mean"
    ]["pyosv.fvt.mean"] == pytest.approx(1.0 / 216.0)
    assert loaded["aggregate"]["backends"]["reference-like"]["cases"]["case_01_normal_normal"][
        "per_metric_mean"
    ]["pyosv.fvt.mean"] == pytest.approx(10.0 / 216.0)


def test_case_names_and_thinning_modes_are_recorded_in_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_ablation_module(monkeypatch)
    received_kwargs = {}
    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: _synthetic_reference_arrays(),
    )

    def fake_pipeline(ep: np.ndarray, **kwargs: object) -> dict[str, dict[str, np.ndarray]]:
        received_kwargs.update(kwargs)
        return _case_outputs(module, ep.shape)

    monkeypatch.setattr(module, "run_ablation_pipeline", fake_pipeline)

    report = module.run_example(
        data_root_arg=tmp_path / "f3_reference",
        count=1,
        crop_shape=(6, 6, 6),
        interior_margin=1,
        centers=[(2, 2, 2)],
        final_normalization_smoothing=1.0,
    )

    assert report["config"]["voter"]["final_normalization_smoothing"] == 1.0
    assert received_kwargs["final_normalization_smoothing"] == 1.0
    cases = {case["name"]: case for case in report["config"]["cases"]}
    assert cases["case_01_normal_normal"] == {
        "name": "case_01_normal_normal",
        "scanner_thin_mode": "normal",
        "voter_thin_mode": "normal",
        "scanner_remove_edge_effects": None,
        "surface_voting_boundary_policy": "reference-like-i2-i3-interior",
    }
    assert cases["case_02_normal_reference_voter"]["voter_thin_mode"] == "reference"
    assert cases["case_03_reference_scanner_normal"]["scanner_thin_mode"] == "reference"
    assert cases["case_03_reference_scanner_normal"]["scanner_remove_edge_effects"] is True
    assert cases["case_04_reference_reference"] == {
        "name": "case_04_reference_reference",
        "scanner_thin_mode": "reference",
        "voter_thin_mode": "reference",
        "scanner_remove_edge_effects": True,
        "surface_voting_boundary_policy": "reference-like-i2-i3-interior",
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

    figures_dir = output_json.parent / "crop_001" / "case_01_normal_normal" / "figures"
    markdown = (output_json.parent / "visual_report.md").read_text(encoding="utf-8")
    assert (figures_dir / "fvt_ref_vs_py_i3_3.png").is_file()
    assert (figures_dir / "fvt_ridge_overlay_i3_3.png").is_file()
    assert (figures_dir / "fvt_mip.png").is_file()
    assert "case_01_normal_normal" in markdown
    assert "buffered F1" in markdown
    assert "scanner edge removal" in markdown
    assert "reference-like-i2-i3-interior" in markdown
    assert "crop_001/case_01_normal_normal/figures/fvt_mip.png" in markdown


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
        scanner_backends=("fast", "reference-like"),
    )

    markdown = (output_json.parent / "visual_report.md").read_text(encoding="utf-8")
    assert output_json.parent / "crop_001" / "fast" / "case_01_normal_normal" / "figures" in (
        figure_dirs
    )
    assert (
        output_json.parent / "crop_001" / "reference-like" / "case_01_normal_normal" / "figures"
        in figure_dirs
    )
    assert "| Backend | Case | fvt interior corr mean |" in markdown
    assert "`fast` | `case_01_normal_normal`" in markdown
    assert "`reference-like` | `case_01_normal_normal`" in markdown
    assert "crop_001/fast/case_01_normal_normal/figures/fvt_mip.png" in markdown
    assert "crop_001/reference-like/case_01_normal_normal/figures/fvt_mip.png" in markdown


def test_reference_like_scanner_backend_unavailable_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_ablation_module(monkeypatch)

    class FastOnlyScanner:
        def scan(self, *args: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            values = np.zeros((2, 2, 2), dtype=np.float32)
            return values, values, values

    with pytest.raises(ValueError, match="reference-like scanner backend is unavailable"):
        module._scan_backend(
            FastOnlyScanner(),
            backend="reference-like",
            phi_min=0.0,
            phi_max=1.0,
            theta_min=2.0,
            theta_max=3.0,
            ep=np.zeros((2, 2, 2), dtype=np.float32),
        )


def test_scan_backend_dispatch_uses_fast_and_explicit_reference_like(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_ablation_module(monkeypatch)
    calls: list[tuple[str, object]] = []

    class RecordingScanner:
        def scan_fast(
            self,
            phi_min: float,
            phi_max: float,
            theta_min: float,
            theta_max: float,
            ep: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            calls.append(("scan_fast", (phi_min, phi_max, theta_min, theta_max)))
            values = np.zeros_like(ep)
            return values, values, values

        def scan_reference_like(
            self,
            phi_min: float,
            phi_max: float,
            theta_min: float,
            theta_max: float,
            ep: np.ndarray,
            *,
            backend: str,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            calls.append(("scan_reference_like", backend))
            values = np.zeros_like(ep)
            return values, values, values

        def scan(self, *args: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            raise AssertionError("diagnostic backend dispatch must not call scan()")

    ep = np.zeros((2, 2, 2), dtype=np.float32)
    module._scan_backend(
        RecordingScanner(),
        backend="fast",
        phi_min=0.0,
        phi_max=1.0,
        theta_min=2.0,
        theta_max=3.0,
        ep=ep,
    )
    module._scan_backend(
        RecordingScanner(),
        backend="reference-like",
        phi_min=0.0,
        phi_max=1.0,
        theta_min=2.0,
        theta_max=3.0,
        ep=ep,
    )

    assert calls == [
        ("scan_fast", (0.0, 1.0, 2.0, 3.0)),
        ("scan_reference_like", "rotate_shear"),
    ]


def test_real_small_pipeline_outputs_build_crop_reports_and_serialize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_ablation_module(monkeypatch)
    shape = (9, 9, 9)
    _, i2, _ = np.indices(shape, dtype=np.float32)
    ep = np.exp(-0.5 * ((i2 - np.float32(4.0)) / np.float32(1.0)) ** 2).astype(
        np.float32,
        copy=False,
    )

    case_outputs = module.run_ablation_pipeline(
        ep,
        scanner_backend="fast",
        sigma1=1.0,
        sigma2=1.0,
        phi_min=0.0,
        phi_max=0.0,
        theta_min=90.0,
        theta_max=90.0,
        ru=1,
        rv=1,
        rw=1,
        strain_max1=0.5,
        strain_max2=0.5,
        surface_smoothing1=0.0,
        surface_smoothing2=0.0,
        d=1,
        fm=0.0,
        remove_scanner_edge_effects=True,
        final_normalization_smoothing=0.0,
    )

    expected_output_names = {
        "ft_py.dat",
        "pt_py.dat",
        "tt_py.dat",
        "fet_py.dat",
        "fpt_py.dat",
        "ftt_py.dat",
        "fv_py.dat",
        "vp_py.dat",
        "vt_py.dat",
        "fvt_py.dat",
    }
    slices = tuple(slice(0, size) for size in shape)
    reports = {}
    for case_name, outputs in case_outputs.items():
        assert set(outputs) == expected_output_names
        reports[case_name] = module.crop_validation.build_crop_report(
            crop_index=1,
            center=(4, 4, 4),
            slices=slices,
            crop_shape=shape,
            outputs=outputs,
            reference_fv=outputs["fv_py.dat"],
            reference_fvt=outputs["fvt_py.dat"],
            interior_margin=1,
        )

    json.dumps(reports, allow_nan=False)


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
