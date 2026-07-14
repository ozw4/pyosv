from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyosv.evaluation import f3d_scanner_policy as policy_evaluation


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
MODULE_NAME = "report_3d_f3d_scanner_thinning_policy"


def _import_report_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop(MODULE_NAME, None)
    importlib.invalidate_caches()
    return importlib.import_module(MODULE_NAME)


def _reference_arrays(shape: tuple[int, int, int] = (9, 9, 9)) -> dict[str, np.ndarray]:
    ep = np.zeros(shape, dtype=np.float32)
    fv = np.ones(shape, dtype=np.float32)
    fvt = np.ones(shape, dtype=np.float32)
    return {"ep.dat": ep, "fv.dat": fv, "fvt.dat": fvt}


def _policy_outputs(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    ridge = np.zeros(shape, dtype=np.float32)
    ridge[:, shape[1] // 2, :] = np.float32(1.0)
    return {
        "ft_py.dat": np.ones(shape, dtype=np.float32),
        "pt_py.dat": np.full(shape, 10.0, dtype=np.float32),
        "tt_py.dat": np.full(shape, 70.0, dtype=np.float32),
        "fet_py.dat": ridge.copy(),
        "fpt_py.dat": np.where(ridge > 0.0, np.float32(10.0), np.float32(0.0)),
        "ftt_py.dat": np.where(ridge > 0.0, np.float32(70.0), np.float32(0.0)),
        "fv_py.dat": ridge.copy(),
        "vp_py.dat": np.full(shape, 10.0, dtype=np.float32),
        "vt_py.dat": np.full(shape, 70.0, dtype=np.float32),
        "fvt_py.dat": ridge.copy(),
    }


def _outputs_from_fvt(fvt: np.ndarray, *, offset: float = 0.0) -> dict[str, np.ndarray]:
    ridge = np.asarray(fvt, dtype=np.float32)
    shape = ridge.shape
    return {
        "ft_py.dat": np.full(shape, 0.5 + offset, dtype=np.float32),
        "pt_py.dat": np.full(shape, 10.0 + offset, dtype=np.float32),
        "tt_py.dat": np.full(shape, 70.0 + offset, dtype=np.float32),
        "fet_py.dat": ridge.copy(),
        "fpt_py.dat": np.full(shape, 20.0 + offset, dtype=np.float32),
        "ftt_py.dat": np.full(shape, 60.0 + offset, dtype=np.float32),
        "fv_py.dat": ridge.copy(),
        "vp_py.dat": np.full(shape, 30.0 + offset, dtype=np.float32),
        "vt_py.dat": np.full(shape, 50.0 + offset, dtype=np.float32),
        "fvt_py.dat": ridge.copy(),
    }


def _shared_run_from_ridges(
    baseline_fvt: np.ndarray,
    candidate_fvt: np.ndarray,
) -> dict[str, Any]:
    return {
        "scanner_execution_count": 1,
        "policies": {
            "baseline": {
                "policy_id": policy_evaluation.BASELINE_POLICY_ID,
                "outputs": _outputs_from_fvt(baseline_fvt),
            },
            "candidate": {
                "policy_id": policy_evaluation.CANDIDATE_POLICY_ID,
                "outputs": _outputs_from_fvt(candidate_fvt, offset=1.0),
            },
        },
    }


def _fake_shared_run(ep: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    del kwargs
    return {
        "scanner_execution_count": 1,
        "policies": {
            "baseline": {
                "policy_id": policy_evaluation.BASELINE_POLICY_ID,
                "outputs": _policy_outputs(ep.shape),
            },
            "candidate": {
                "policy_id": policy_evaluation.CANDIDATE_POLICY_ID,
                "outputs": _policy_outputs(ep.shape),
            },
        },
    }


def test_import_does_not_run_the_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def unexpected_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        raise AssertionError("module import must not run the F3 pipeline")

    monkeypatch.setattr(
        policy_evaluation,
        "run_shared_scan_policy_pipeline",
        unexpected_run,
    )

    module = _import_report_module(monkeypatch)

    assert module.run_shared_scan_policy_pipeline is unexpected_run
    assert calls == []


def test_parser_exposes_formal_flags_and_fixed_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_report_module(monkeypatch)
    parser = module.build_parser()
    defaults = parser.parse_args([])

    assert defaults.data_root is None
    assert defaults.comparison_profile == "quality-workflow-scanner-thinning-v1"
    assert defaults.output_json is None
    assert defaults.save_volumes is False
    assert defaults.save_figures is False
    assert defaults.write_markdown_index is False
    assert defaults.fail_on_validation_failure is False
    assert defaults.outlier_diagnostics is False
    assert defaults.context_crop_shape is None
    assert defaults.context_crop_index is None
    assert defaults.outlier_max_points == 64
    assert defaults.outlier_max_components == 8
    assert defaults.outlier_window_radius == 24
    assert defaults.outlier_adjacent_slice_radius == 3
    assert defaults.amplitude_clip_percentile == 99.0
    assert defaults.pretty is False
    assert defaults.count == 3
    assert defaults.crop_shape == (64, 64, 64)
    assert defaults.interior_margin == 16
    assert defaults.center is None
    assert defaults.sigma1 == 8.0
    assert defaults.sigma2 == 8.0
    assert defaults.ru == 10
    assert defaults.rv == 20
    assert defaults.rw == 30
    assert defaults.surface_orientation_smoothing is None

    args = parser.parse_args(
        [
            "--data-root",
            "/data/f3",
            "--comparison-profile",
            module.COMPARISON_PROFILE,
            "--output-json",
            "outputs/3d/f3/scanner_policy/metrics.json",
            "--count",
            "3",
            "--crop-shape",
            "64,64,64",
            "--interior-margin",
            "16",
            "--center",
            "100,120,50",
            "--center",
            "180,220,50",
            "--save-volumes",
            "--save-figures",
            "--write-markdown-index",
            "--outlier-diagnostics",
            "--context-crop-shape",
            "9,9,9",
            "--context-crop-index",
            "1",
            "--context-crop-index",
            "2",
            "--outlier-max-points",
            "12",
            "--outlier-max-components",
            "4",
            "--outlier-window-radius",
            "8",
            "--outlier-adjacent-slice-radius",
            "2",
            "--amplitude-clip-percentile",
            "98.5",
            "--fail-on-validation-failure",
            "--pretty",
        ]
    )
    assert args.data_root == Path("/data/f3")
    assert args.comparison_profile == module.COMPARISON_PROFILE
    assert args.output_json == Path("outputs/3d/f3/scanner_policy/metrics.json")
    assert args.center == [(100, 120, 50), (180, 220, 50)]
    assert args.save_volumes is True
    assert args.save_figures is True
    assert args.write_markdown_index is True
    assert args.outlier_diagnostics is True
    assert args.context_crop_shape == (9, 9, 9)
    assert args.context_crop_index == [1, 2]
    assert args.outlier_max_points == 12
    assert args.outlier_max_components == 4
    assert args.outlier_window_radius == 8
    assert args.outlier_adjacent_slice_radius == 2
    assert args.amplitude_clip_percentile == 98.5
    assert args.fail_on_validation_failure is True
    assert args.pretty is True

    help_text = parser.format_help()
    assert "--comparison-profile" in help_text
    assert "--fail-on-validation-failure" in help_text
    assert "--scanner-thin-mode" not in help_text
    assert "--voter-thin-mode" not in help_text
    assert "--surface-support-min-fraction" not in help_text
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--comparison-profile", "scanner-thinning-policy-v1"])
    assert error.value.code == 2


def test_mock_run_builds_policy_report_volumes_markdown_and_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_report_module(monkeypatch)
    data_root = tmp_path / "f3_data"
    data_root.mkdir()
    output_json = tmp_path / "outputs" / "metrics.json"
    arrays = _reference_arrays()
    pipeline_calls: list[dict[str, Any]] = []
    volume_calls: list[tuple[Path, tuple[str, ...]]] = []
    figure_calls: list[Path] = []
    direct_figure_calls: list[Path] = []

    monkeypatch.setattr(
        module.crop_validation,
        "read_reference_arrays",
        lambda root: arrays,
    )
    monkeypatch.setattr(module, "read_f3d_file", lambda name, root: arrays["fv.dat"])
    monkeypatch.setattr(module.crop_validation, "require_figure_support", lambda: None)

    def fake_pipeline(ep: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        pipeline_calls.append({"shape": ep.shape, **kwargs})
        return _fake_shared_run(ep, **kwargs)

    monkeypatch.setattr(module, "run_shared_scan_policy_pipeline", fake_pipeline)

    real_write_volumes = module.crop_validation.write_crop_volumes

    def recording_write_volumes(
        output_dir: str | Path,
        outputs: Mapping[str, np.ndarray],
    ) -> list[Path]:
        directory = Path(output_dir)
        volume_calls.append((directory, tuple(outputs)))
        return real_write_volumes(directory, outputs)

    monkeypatch.setattr(module.crop_validation, "write_crop_volumes", recording_write_volumes)

    def fake_policy_figures(
        output_dir: str | Path,
        *,
        metrics_base_dir: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        directory = Path(output_dir)
        figure_calls.append(directory)
        relative = directory.relative_to(Path(metrics_base_dir)).as_posix()
        return {
            "directory": relative,
            "files": {
                "fvt_ref_vs_py": {"i3": f"{relative}/fvt_ref_vs_py_i3.png"},
            },
        }

    monkeypatch.setattr(module.crop_validation, "write_crop_figures", fake_policy_figures)

    def fake_direct_figures(
        output_dir: str | Path,
        *,
        metrics_base_dir: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        directory = Path(output_dir)
        direct_figure_calls.append(directory)
        relative = directory.relative_to(Path(metrics_base_dir)).as_posix()
        files = {
            key: {"i3": f"{relative}/{key}_i3.png"}
            for key in (
                "baseline_vs_candidate_fvt_slices",
                "baseline_candidate_ridge_overlay",
                "candidate_only_baseline_only_ridge_mask",
                "edge_shell_ridge_overlay",
            )
        }
        return {"directory": relative, "files": files}

    monkeypatch.setattr(module, "write_direct_comparison_figures", fake_direct_figures)

    report = module.run_example(
        data_root_arg=data_root,
        output_json=output_json,
        save_volumes=True,
        save_figures=True,
        write_markdown_index=True,
        pretty=True,
        count=3,
        crop_shape=(5, 5, 5),
        interior_margin=1,
        centers=[(3, 3, 3), (5, 5, 5)],
        sigma1=1.0,
        sigma2=1.0,
        phi_min=0.0,
        phi_max=0.0,
        theta_min=70.0,
        theta_max=70.0,
        ru=1,
        rv=1,
        rw=1,
        strain_max1=0.5,
        strain_max2=0.5,
        surface_smoothing1=0.0,
        surface_smoothing2=0.0,
        d=1,
        fm=0.0,
    )

    assert report["format_version"] == 1
    assert set(report) == {
        "format_version",
        "data_root",
        "config",
        "scanner_policies",
        "consensus",
        "policy_validation",
        "manual_review",
    }
    config = report["config"]
    assert config["comparison_profile"] == module.COMPARISON_PROFILE
    assert config["crop_selection"]["requested_count"] == 3
    assert config["crop_selection"]["selected_count"] == 2
    assert config["crop_selection"]["source"] == "explicit_centers"
    assert config["shared_scanner"]["backend"] == "reference-like"
    assert config["shared_scanner"]["execution_count"] == 2
    assert config["quality_downstream"]["workflow_mode"] == "quality"
    assert config["quality_downstream"]["voter_thin_mode"] == "hybrid_v2"
    assert config["quality_downstream"]["surface_support_min_fraction"] == 0.0
    assert config["quality_downstream"]["surface_support_exponent"] == 0.0
    assert config["quality_downstream"]["surface_voting_boundary_policy"] == "reference"
    assert config["quality_downstream"]["final_normalization_smoothing"] == 0.0

    policies = report["scanner_policies"]
    assert tuple(policies) == policy_evaluation.POLICY_ROLES
    assert policies["baseline"]["policy_id"] == policy_evaluation.BASELINE_POLICY_ID
    assert policies["candidate"]["policy_id"] == policy_evaluation.CANDIDATE_POLICY_ID
    assert policies["baseline"]["config"]["requested"]["scanner_thin_mode"] == "reference"
    assert policies["candidate"]["config"]["requested"]["scanner_thin_mode"] == "normal"
    assert policies["baseline"]["config"]["effective"]["effective_remove_edge_effects"] is True
    assert policies["candidate"]["config"]["effective"]["effective_remove_edge_effects"] is None
    for role in policy_evaluation.POLICY_ROLES:
        assert len(policies[role]["crops"]) == 2
        assert policies[role]["aggregate"]["crop_count"] == 2
        for crop in policies[role]["crops"]:
            assert set(crop["finite_checks"]["pyosv"]) == {
                name.removesuffix(".dat") for name in policy_evaluation.OUTPUT_NAMES
            }
            assert crop["stage_density"]["fet"]["nonzero_count"] > 0
            assert crop["stage_density"]["fv"]["nonzero_count"] > 0
            assert crop["stage_density"]["fvt"]["nonzero_count"] > 0

    consensus = report["consensus"]
    assert set(consensus["policies"]) == {"baseline", "candidate"}
    assert consensus["candidate_minus_baseline"]["fvt_density_ratio"] == 1.0
    assert len(consensus["candidate_minus_baseline"]["crops"]) == 2
    validation = report["policy_validation"]
    assert validation["role"] == "truthless_external_smoke"
    assert validation["passed"] is True
    assert validation["crop_count"] == 2
    assert validation["scanner_execution_count"] == 2
    assert validation["reasons"] == []
    assert all(check["passed"] for check in validation["checks"].values())
    assert report["manual_review"]["status"] == "pending"

    assert len(pipeline_calls) == 2
    assert all(call["shape"] == (5, 5, 5) for call in pipeline_calls)
    assert all(call["reference_thin_sigma"] == 1.0 for call in pipeline_calls)
    assert all(call["final_normalization_smoothing"] is None for call in pipeline_calls)
    assert len(volume_calls) == 4
    assert all(names == policy_evaluation.OUTPUT_NAMES for _, names in volume_calls)
    assert len(figure_calls) == 4
    assert len(direct_figure_calls) == 2
    for crop_index in (1, 2):
        for role in policy_evaluation.POLICY_ROLES:
            volume_dir = output_json.parent / f"crop_{crop_index:03d}" / role
            assert {path.name for path in volume_dir.glob("*.dat")} == set(
                policy_evaluation.OUTPUT_NAMES
            )

    report_text = output_json.read_text(encoding="utf-8")
    loaded = json.loads(report_text)
    assert loaded["policy_validation"]["passed"] is True
    json.dumps(report, allow_nan=False)
    assert "NaN" not in report_text
    assert "Infinity" not in report_text

    markdown_path = output_json.parent / "visual_report.md"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "# F3 Scanner-Thinning Policy Visual Report" in markdown
    assert policy_evaluation.BASELINE_POLICY_ID in markdown
    assert policy_evaluation.CANDIDATE_POLICY_ID in markdown
    assert "| Baseline | Candidate |" in markdown
    assert "crop_001/baseline/figures/fvt_ref_vs_py_i3.png" in markdown
    assert "crop_001/candidate/figures/fvt_ref_vs_py_i3.png" in markdown
    assert "candidate-only / baseline-only masks" in markdown
    assert "edge-shell ridge overlay" in markdown


def test_main_returns_two_for_requested_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _import_report_module(monkeypatch)
    report = {"policy_validation": {"passed": False, "reasons": ["failed smoke"]}}
    monkeypatch.setattr(module, "run_example", lambda **kwargs: report)

    assert module.main(["--fail-on-validation-failure"]) == 2
    assert json.loads(capsys.readouterr().out)["policy_validation"]["passed"] is False

    assert module.main([]) == 0


def test_main_returns_one_for_input_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _import_report_module(monkeypatch)

    def fail(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        raise ValueError("invalid F3 policy input")

    monkeypatch.setattr(module, "run_example", fail)

    assert module.main([]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: invalid F3 policy input" in captured.err


def test_output_safety_and_output_requirements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_report_module(monkeypatch)
    data_root = tmp_path / "f3_data"
    data_root.mkdir()
    forbidden_reference = tmp_path / "reference_osv"
    forbidden_issue_forge = tmp_path / "vendor" / "issue_forge"
    monkeypatch.setattr(module, "REFERENCE_OSV_DIR", forbidden_reference)
    monkeypatch.setattr(module, "ISSUE_FORGE_DIR", forbidden_issue_forge)

    with pytest.raises(ValueError, match="must not be inside F3 data root"):
        module.ensure_output_path_allowed(
            data_root / "metrics.json",
            data_root,
            option_name="--output-json",
        )
    with pytest.raises(ValueError, match="must not be inside reference_osv"):
        module.ensure_output_path_allowed(
            forbidden_reference / "metrics.json",
            data_root,
            option_name="--output-json",
        )
    with pytest.raises(ValueError, match="must not be inside vendor/issue_forge"):
        module.ensure_output_path_allowed(
            forbidden_issue_forge / "metrics.json",
            data_root,
            option_name="--output-json",
        )
    module.ensure_output_path_allowed(
        tmp_path / "outputs" / "metrics.json",
        data_root,
        option_name="--output-json",
    )

    for option, message in (
        ("save_volumes", "--save-volumes requires --output-json"),
        ("save_figures", "--save-figures requires --output-json"),
        ("write_markdown_index", "--write-markdown-index requires --output-json"),
    ):
        with pytest.raises(ValueError, match=message):
            module.run_example(data_root_arg=data_root, **{option: True})

    with pytest.raises(ValueError, match="comparison_profile must be"):
        module.run_example(
            data_root_arg=data_root,
            comparison_profile="scanner-thinning-policy-v1",
        )
    with pytest.raises(ValueError, match="count must be >= 1"):
        module.run_example(data_root_arg=data_root, count=0)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--context-crop-shape", "128,128,100"],
            "--context-crop-shape requires --outlier-diagnostics",
        ),
        (["--context-crop-index", "1"], "--context-crop-index requires --context-crop-shape"),
        (["--outlier-max-points", "0"], "outlier_max_points must be >= 1"),
        (["--outlier-max-components", "0"], "outlier_max_components must be >= 1"),
        (["--outlier-window-radius", "-1"], "outlier_window_radius must be >= 0"),
        (
            ["--outlier-adjacent-slice-radius", "-1"],
            "outlier_adjacent_slice_radius must be >= 0",
        ),
        (
            ["--amplitude-clip-percentile", "0"],
            "amplitude_clip_percentile must be finite and in (0, 100]",
        ),
        (
            [
                "--outlier-diagnostics",
                "--crop-shape",
                "64,64,64",
                "--context-crop-shape",
                "63,64,64",
            ],
            "context_crop_shape[0] must be >= base crop_shape[0]",
        ),
    ],
)
def test_diagnostic_cli_input_errors_return_one_without_traceback(
    argv: list[str],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _import_report_module(monkeypatch)
    assert module.main(argv) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert message in captured.err


def test_context_shape_and_indices_validate_against_selected_f3_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_report_module(monkeypatch)
    with pytest.raises(ValueError, match=r"context_crop_shape\[1].*F3 full_shape"):
        module.validate_context_selection(
            context_crop_shape=(9, 11, 9),
            full_shape=(10, 10, 10),
            context_crop_indices=None,
            crop_count=2,
        )
    with pytest.raises(ValueError, match="selected 1-origin crop range"):
        module.validate_context_selection(
            context_crop_shape=(9, 9, 9),
            full_shape=(10, 10, 10),
            context_crop_indices=[0, 3],
            crop_count=2,
        )
    assert module.validate_context_selection(
        context_crop_shape=(9, 9, 9),
        full_shape=(10, 10, 10),
        context_crop_indices=[2, 1, 2],
        crop_count=2,
    ) == [2, 1]


def test_outlier_diagnostics_are_opt_in_and_preserve_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_report_module(monkeypatch)
    data_root = tmp_path / "f3_data"
    data_root.mkdir()
    output_json = tmp_path / "outputs" / "metrics.json"
    full_shape = (15, 15, 15)
    center = (7, 7, 7)
    arrays = {
        "ep.dat": np.zeros(full_shape, dtype=np.float32),
        "fv.dat": np.ones(full_shape, dtype=np.float32),
        "fvt.dat": np.zeros(full_shape, dtype=np.float32),
    }
    arrays["fvt.dat"][center] = 1.0
    read_names: list[str] = []

    monkeypatch.setattr(module.crop_validation, "read_reference_arrays", lambda root: arrays)

    def fake_read(name: str, root: Path) -> np.ndarray:
        del root
        read_names.append(name)
        if name == "xs.dat":
            return np.linspace(-2.0, 2.0, np.prod(full_shape), dtype=np.float32).reshape(full_shape)
        if name == "fl.dat":
            return np.ones(full_shape, dtype=np.float32)
        raise AssertionError(name)

    monkeypatch.setattr(module, "read_f3d_file", fake_read)

    def fake_pipeline(ep: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        baseline = np.zeros(ep.shape, dtype=np.float32)
        candidate = np.zeros(ep.shape, dtype=np.float32)
        baseline[6, 6, 6] = 1.0
        candidate[1, 1, 1] = 1.0
        return _shared_run_from_ridges(baseline, candidate)

    monkeypatch.setattr(module, "run_shared_scan_policy_pipeline", fake_pipeline)

    base_report = module.run_example(
        data_root_arg=data_root,
        centers=[center],
        crop_shape=(13, 13, 13),
        interior_margin=1,
    )
    assert read_names == []
    assert "diagnostics" not in base_report["config"]
    assert "context_diagnostics" not in base_report
    base_direct = base_report["consensus"]["candidate_minus_baseline"]["crops"][0]
    assert "public_fvt_distance_outliers" not in base_direct

    monkeypatch.setattr(module.crop_validation, "require_figure_support", lambda: None)
    monkeypatch.setattr(
        module.crop_validation,
        "write_crop_figures",
        lambda output_dir, **kwargs: {"directory": str(output_dir), "files": {}},
    )
    monkeypatch.setattr(
        module,
        "write_direct_comparison_figures",
        lambda output_dir, **kwargs: {"directory": str(output_dir), "files": {}},
    )
    outlier_figure_calls: list[dict[str, Any]] = []

    def fake_outlier_figures(
        output_dir: str | Path,
        *,
        metrics_base_dir: str | Path,
        outlier_report: Mapping[str, Any],
        crop_index: int,
        **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        outlier_figure_calls.append(
            {
                "output_dir": Path(output_dir),
                "crop_index": crop_index,
                "coordinate": outlier_report["components"][0]["representative_point"][
                    "crop_local_coordinate"
                ],
                **kwargs,
            }
        )
        relative = Path(output_dir).relative_to(Path(metrics_base_dir)) / "component_001"
        return {
            1: {
                "orthogonal_amplitude_overlay": (
                    relative / "orthogonal_amplitude_overlay.png"
                ).as_posix(),
                "adjacent_i3": (relative / "adjacent_i3.png").as_posix(),
                "adjacent_i2": (relative / "adjacent_i2.png").as_posix(),
                "adjacent_i1": (relative / "adjacent_i1.png").as_posix(),
            }
        }

    monkeypatch.setattr(module, "write_outlier_diagnostic_figures", fake_outlier_figures)

    diagnostic_report = module.run_example(
        data_root_arg=data_root,
        output_json=output_json,
        save_figures=True,
        write_markdown_index=True,
        outlier_diagnostics=True,
        centers=[center],
        crop_shape=(13, 13, 13),
        interior_margin=1,
        pretty=True,
    )

    assert read_names == ["xs.dat", "fl.dat"]
    assert diagnostic_report["policy_validation"] == base_report["policy_validation"]
    assert diagnostic_report["policy_validation"]["passed"] is False
    assert len(diagnostic_report["policy_validation"]["checks"]) == 8
    assert diagnostic_report["manual_review"]["status"] == "pending"
    assert "context_diagnostics" not in diagnostic_report
    diagnostic = diagnostic_report["consensus"]["candidate_minus_baseline"]["crops"][0][
        "public_fvt_distance_outliers"
    ]
    assert diagnostic["status"] == "available"
    assert diagnostic["summary"]["outlier_count"] == 1
    assert diagnostic["summary"]["component_count"] == 1
    assert diagnostic["points"][0]["values"]["xs"] < 0.0
    assert outlier_figure_calls[0]["coordinate"] == [1, 1, 1]
    assert outlier_figure_calls[0]["crop_index"] == 1
    orthogonal = diagnostic["components"][0]["figures"]["orthogonal_amplitude_overlay"]
    assert orthogonal.endswith(
        "crop_001/policy_comparison/outlier_diagnostics/component_001/"
        "orthogonal_amplitude_overlay.png"
    )

    report_text = output_json.read_text(encoding="utf-8")
    json.loads(report_text, parse_constant=lambda value: pytest.fail(value))
    assert report_text.endswith("\n")
    markdown = (output_json.parent / "visual_report.md").read_text(encoding="utf-8")
    assert "## Public-FVT Distance Outlier Review" in markdown
    assert f"![crop 1 component 1 orthogonal amplitude review]({orthogonal})" in markdown
    assert "adjacent_i3.png" in markdown
    assert markdown.endswith("\n")


def test_outlier_free_crop_does_not_call_amplitude_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_report_module(monkeypatch)
    data_root = tmp_path / "f3_data"
    data_root.mkdir()
    shape = (9, 9, 9)
    arrays = _reference_arrays(shape)
    arrays["fvt.dat"].fill(0.0)
    arrays["fvt.dat"][4, 4, 4] = 1.0
    monkeypatch.setattr(module.crop_validation, "read_reference_arrays", lambda root: arrays)
    monkeypatch.setattr(module, "read_f3d_file", lambda name, root: np.ones(shape, np.float32))
    monkeypatch.setattr(module.crop_validation, "require_figure_support", lambda: None)
    monkeypatch.setattr(
        module.crop_validation,
        "write_crop_figures",
        lambda output_dir, **kwargs: {"directory": str(output_dir), "files": {}},
    )
    monkeypatch.setattr(
        module,
        "write_direct_comparison_figures",
        lambda output_dir, **kwargs: {"directory": str(output_dir), "files": {}},
    )

    def same_pipeline(ep: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        ridge = np.zeros(ep.shape, dtype=np.float32)
        ridge[2, 2, 2] = 1.0
        return _shared_run_from_ridges(ridge, ridge)

    monkeypatch.setattr(module, "run_shared_scan_policy_pipeline", same_pipeline)
    monkeypatch.setattr(
        module,
        "write_outlier_diagnostic_figures",
        lambda *args, **kwargs: pytest.fail("no outlier figure should be written"),
    )

    report = module.run_example(
        data_root_arg=data_root,
        output_json=tmp_path / "outputs" / "metrics.json",
        save_figures=True,
        outlier_diagnostics=True,
        centers=[(4, 4, 4)],
        crop_shape=(5, 5, 5),
        interior_margin=0,
    )
    diagnostic = report["consensus"]["candidate_minus_baseline"]["crops"][0][
        "public_fvt_distance_outliers"
    ]
    assert diagnostic["status"] == "available"
    assert diagnostic["summary"]["outlier_count"] == 0
    assert diagnostic["components"] == []


def test_context_pipeline_uses_separate_scan_count_and_exact_base_roi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_report_module(monkeypatch)
    data_root = tmp_path / "f3_data"
    data_root.mkdir()
    full_shape = (35, 35, 35)
    centers = [(8, 8, 8), (17, 17, 17), (26, 26, 26)]
    arrays = {
        "ep.dat": np.zeros(full_shape, dtype=np.float32),
        "fv.dat": np.ones(full_shape, dtype=np.float32),
        "fvt.dat": np.zeros(full_shape, dtype=np.float32),
    }
    for center in centers:
        arrays["fvt.dat"][center] = 1.0
    monkeypatch.setattr(module.crop_validation, "read_reference_arrays", lambda root: arrays)
    monkeypatch.setattr(module, "read_f3d_file", lambda name, root: np.ones(full_shape, np.float32))

    pipeline_shapes: list[tuple[int, int, int]] = []

    def fake_pipeline(ep: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        pipeline_shapes.append(ep.shape)
        baseline = np.zeros(ep.shape, dtype=np.float32)
        candidate = np.zeros(ep.shape, dtype=np.float32)
        center = tuple(size // 2 for size in ep.shape)
        baseline[center] = 1.0
        if ep.shape == (9, 9, 9) and len(pipeline_shapes) == 1:
            candidate[0, 0, 0] = 1.0
        elif ep.shape == (13, 13, 13):
            # Context local (2,2,2) maps to base local (0,0,0).
            candidate[2, 2, 2] = 1.0
        else:
            candidate[center] = 1.0
        return _shared_run_from_ridges(baseline, candidate)

    monkeypatch.setattr(module, "run_shared_scan_policy_pipeline", fake_pipeline)

    report = module.run_example(
        data_root_arg=data_root,
        outlier_diagnostics=True,
        context_crop_shape=(13, 13, 13),
        centers=centers,
        crop_shape=(9, 9, 9),
        interior_margin=0,
    )

    assert pipeline_shapes == [(9, 9, 9), (9, 9, 9), (9, 9, 9), (13, 13, 13)]
    assert report["policy_validation"]["scanner_execution_count"] == 3
    assert report["policy_validation"]["crop_count"] == 3
    assert report["policy_validation"]["passed"] is False
    assert len(report["policy_validation"]["checks"]) == 8
    context = report["context_diagnostics"]
    assert context["role"] == "diagnostic_context_ablation"
    assert context["selected_crop_indices"] == [1]
    assert context["context_scanner_execution_count"] == 1
    crop = context["crops"][0]
    assert [entry["start"] for entry in crop["base_roi_slices_within_context"]] == [2, 2, 2]
    assert [entry["stop"] for entry in crop["base_roi_slices_within_context"]] == [11, 11, 11]
    for role in policy_evaluation.POLICY_ROLES:
        stage_report = crop["policies"][role]
        assert stage_report["status"] == "available"
        assert set(stage_report["stages"]) == {"ft", "fet", "fv", "fvt"}
    candidate_fvt = crop["policies"]["candidate"]["stages"]["fvt"]
    assert candidate_fvt["shape_equal"] is True
    assert candidate_fvt["absolute_difference"]["maximum"] == 0.0
    persistence = crop["outlier_persistence"]
    assert persistence["summary"]["base_outlier_count"] == 1
    assert persistence["summary"]["context_outlier_count"] == 1
    assert persistence["summary"]["base_outlier_points_with_context_candidate_within_radius"] == 1
    assert persistence["summary"]["persistence_fraction"] == 1.0
    assert persistence["points"][0]["base_roi_local_coordinate"] == [0, 0, 0]
    assert persistence["points"][0]["nearest_context_candidate_sparse_ridge_distance"] == 0.0
    assert report["manual_review"]["status"] == "pending"


def test_context_orchestration_preserves_scanner_and_branch_voter_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyosv import orient3d, voting3d

    module = _import_report_module(monkeypatch)
    data_root = tmp_path / "f3_data"
    data_root.mkdir()
    full_shape = (25, 25, 25)
    centers = [(5, 5, 5), (12, 12, 12), (19, 19, 19)]
    arrays = {
        "ep.dat": np.zeros(full_shape, dtype=np.float32),
        "fv.dat": np.ones(full_shape, dtype=np.float32),
        "fvt.dat": np.ones(full_shape, dtype=np.float32),
    }
    monkeypatch.setattr(module.crop_validation, "read_reference_arrays", lambda root: arrays)
    monkeypatch.setattr(module, "read_f3d_file", lambda name, root: np.ones(full_shape, np.float32))

    scanners: list[Any] = []
    voters: list[Any] = []

    class ContractScanner:
        def __init__(self, *, sigma1: float, sigma2: float) -> None:
            self.sigmas = (sigma1, sigma2)
            self.scan_calls: list[np.ndarray] = []
            self.thin_calls: list[dict[str, Any]] = []
            self.thinned: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            self.ft: np.ndarray | None = None
            self.pt: np.ndarray | None = None
            self.tt: np.ndarray | None = None
            scanners.append(self)

        def scan(
            self,
            phi_min: float,
            phi_max: float,
            theta_min: float,
            theta_max: float,
            ep: np.ndarray,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            del phi_min, phi_max, theta_min, theta_max
            self.scan_calls.append(ep)
            self.ft = np.full(ep.shape, 0.5, dtype=np.float32)
            self.pt = np.full(ep.shape, 10.0, dtype=np.float32)
            self.tt = np.full(ep.shape, 70.0, dtype=np.float32)
            return self.ft, self.pt, self.tt

        def thin(
            self,
            ft: np.ndarray,
            pt: np.ndarray,
            tt: np.ndarray,
            **kwargs: Any,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            mode = str(kwargs["mode"])
            value = 1.0 if mode == "reference" else 2.0
            result = (
                np.full_like(ft, value),
                np.full_like(pt, 20.0 + value),
                np.full_like(tt, 60.0 + value),
            )
            self.thin_calls.append({"ft": ft, "pt": pt, "tt": tt, **kwargs})
            self.thinned[mode] = result
            return result

    class ContractVoter:
        def __init__(self, *, ru: int, rv: int, rw: int) -> None:
            self.radii = (ru, rv, rw)
            self.apply_calls: list[dict[str, Any]] = []
            self.thin_calls: list[dict[str, Any]] = []
            voters.append(self)

        def set_strain_max(self, value1: float, value2: float) -> None:
            del value1, value2

        def set_surface_smoothing(self, value1: float, value2: float) -> None:
            del value1, value2

        def set_surface_support_policy(self, **kwargs: float) -> None:
            del kwargs

        def set_surface_voting_boundary_policy(self, value: str) -> None:
            del value

        def set_surface_orientation_smoothing(self, value: float) -> None:
            del value

        def set_final_normalization_smoothing(self, value: float) -> None:
            del value

        def apply_voting(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            self.apply_calls.append(kwargs)
            likelihood = np.asarray(kwargs["ft"])
            strike = np.asarray(kwargs["pt"])
            dip = np.asarray(kwargs["tt"])
            return likelihood.copy(), strike.copy(), dip.copy()

        def thin(
            self,
            fv: np.ndarray,
            vp: np.ndarray,
            vt: np.ndarray,
            **kwargs: Any,
        ) -> np.ndarray:
            self.thin_calls.append({"fv": fv, "vp": vp, "vt": vt, **kwargs})
            return np.asarray(fv).copy()

    monkeypatch.setattr(orient3d, "FaultOrientScanner3", ContractScanner)
    monkeypatch.setattr(voting3d, "OptimalSurfaceVoter", ContractVoter)

    report = module.run_example(
        data_root_arg=data_root,
        outlier_diagnostics=True,
        context_crop_shape=(9, 9, 9),
        context_crop_indices=[1],
        centers=centers,
        crop_shape=(5, 5, 5),
        interior_margin=0,
        ru=1,
        rv=1,
        rw=1,
    )

    assert len(scanners) == 4
    assert sum(len(scanner.scan_calls) for scanner in scanners) == 4
    assert [scanner.scan_calls[0].shape for scanner in scanners] == [
        (5, 5, 5),
        (5, 5, 5),
        (5, 5, 5),
        (9, 9, 9),
    ]
    assert report["policy_validation"]["scanner_execution_count"] == 3
    assert report["context_diagnostics"]["context_scanner_execution_count"] == 1

    assert len(voters) == 8
    assert len({id(voter) for voter in voters}) == 8
    for crop_index, scanner in enumerate(scanners):
        baseline_voter, candidate_voter = voters[2 * crop_index : 2 * crop_index + 2]
        assert baseline_voter is not candidate_voter
        assert baseline_voter.apply_calls[0]["ft"] is scanner.thinned["reference"][0]
        assert candidate_voter.apply_calls[0]["ft"] is scanner.thinned["normal"][0]
        assert (
            baseline_voter.thin_calls[0]["plateau_tie_breaker"] is scanner.thinned["reference"][0]
        )
        assert candidate_voter.thin_calls[0]["plateau_tie_breaker"] is scanner.thinned["normal"][0]
        assert baseline_voter.thin_calls[0]["mode"] == "hybrid_v2"
        assert candidate_voter.thin_calls[0]["mode"] == "hybrid_v2"


def test_explicit_context_index_runs_only_requested_crop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_report_module(monkeypatch)
    data_root = tmp_path / "f3_data"
    data_root.mkdir()
    shape = (25, 25, 25)
    centers = [(7, 7, 7), (17, 17, 17)]
    arrays = {
        "ep.dat": np.zeros(shape, dtype=np.float32),
        "fv.dat": np.ones(shape, dtype=np.float32),
        "fvt.dat": np.zeros(shape, dtype=np.float32),
    }
    for center in centers:
        arrays["fvt.dat"][center] = 1.0
    monkeypatch.setattr(module.crop_validation, "read_reference_arrays", lambda root: arrays)
    monkeypatch.setattr(module, "read_f3d_file", lambda name, root: np.ones(shape, np.float32))
    calls: list[tuple[int, int, int]] = []

    def same_pipeline(ep: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        calls.append(ep.shape)
        ridge = np.zeros(ep.shape, dtype=np.float32)
        ridge[tuple(size // 2 for size in ep.shape)] = 1.0
        return _shared_run_from_ridges(ridge, ridge)

    monkeypatch.setattr(module, "run_shared_scan_policy_pipeline", same_pipeline)
    report = module.run_example(
        data_root_arg=data_root,
        outlier_diagnostics=True,
        context_crop_shape=(11, 11, 11),
        context_crop_indices=[2],
        centers=centers,
        crop_shape=(7, 7, 7),
        interior_margin=0,
    )
    assert calls == [(7, 7, 7), (7, 7, 7), (11, 11, 11)]
    assert report["policy_validation"]["scanner_execution_count"] == 2
    assert report["context_diagnostics"]["selected_crop_indices"] == [2]
    assert report["context_diagnostics"]["context_scanner_execution_count"] == 1


def test_context_figure_path_is_relative_and_linked_from_outlier_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_report_module(monkeypatch)
    data_root = tmp_path / "f3_data"
    data_root.mkdir()
    output_json = tmp_path / "outputs" / "metrics.json"
    shape = (15, 15, 15)
    arrays = {
        "ep.dat": np.zeros(shape, dtype=np.float32),
        "fv.dat": np.ones(shape, dtype=np.float32),
        "fvt.dat": np.zeros(shape, dtype=np.float32),
    }
    arrays["fvt.dat"][7, 7, 7] = 1.0
    monkeypatch.setattr(module.crop_validation, "read_reference_arrays", lambda root: arrays)
    monkeypatch.setattr(module, "read_f3d_file", lambda name, root: np.ones(shape, np.float32))
    monkeypatch.setattr(module.crop_validation, "require_figure_support", lambda: None)
    monkeypatch.setattr(
        module.crop_validation,
        "write_crop_figures",
        lambda output_dir, **kwargs: {"directory": str(output_dir), "files": {}},
    )
    monkeypatch.setattr(
        module,
        "write_direct_comparison_figures",
        lambda output_dir, **kwargs: {"directory": str(output_dir), "files": {}},
    )

    def fake_pipeline(ep: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        baseline = np.zeros(ep.shape, dtype=np.float32)
        candidate = np.zeros(ep.shape, dtype=np.float32)
        baseline[tuple(size // 2 for size in ep.shape)] = 1.0
        candidate[(1, 1, 1) if ep.shape == (13, 13, 13) else (2, 2, 2)] = 1.0
        return _shared_run_from_ridges(baseline, candidate)

    monkeypatch.setattr(module, "run_shared_scan_policy_pipeline", fake_pipeline)

    def fake_outlier_writer(
        output_dir: str | Path,
        *,
        metrics_base_dir: str | Path,
        **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        del kwargs
        relative = Path(output_dir).relative_to(Path(metrics_base_dir)) / "component_001"
        return {
            1: {
                "orthogonal_amplitude_overlay": (
                    relative / "orthogonal_amplitude_overlay.png"
                ).as_posix()
            }
        }

    context_calls: list[dict[str, Any]] = []

    def fake_context_writer(
        output_dir: str | Path,
        *,
        metrics_base_dir: str | Path,
        **kwargs: Any,
    ) -> dict[int, dict[str, Any]]:
        context_calls.append(kwargs)
        relative = Path(output_dir).relative_to(Path(metrics_base_dir)) / "component_001"
        return {
            1: {"context_comparison": (relative / "base_context_amplitude_overlay.png").as_posix()}
        }

    monkeypatch.setattr(module, "write_outlier_diagnostic_figures", fake_outlier_writer)
    monkeypatch.setattr(module, "write_context_diagnostic_figures", fake_context_writer)
    report = module.run_example(
        data_root_arg=data_root,
        output_json=output_json,
        save_figures=True,
        write_markdown_index=True,
        outlier_diagnostics=True,
        context_crop_shape=(15, 15, 15),
        context_crop_indices=[1],
        centers=[(7, 7, 7)],
        crop_shape=(13, 13, 13),
        interior_margin=1,
    )
    assert len(context_calls) == 1
    assert context_calls[0]["base_candidate_fvt"].shape == (13, 13, 13)
    assert context_calls[0]["context_candidate_fvt"].shape == (13, 13, 13)
    component = report["consensus"]["candidate_minus_baseline"]["crops"][0][
        "public_fvt_distance_outliers"
    ]["components"][0]
    context_path = component["figures"]["context_comparison"]
    assert context_path == (
        "crop_001/policy_comparison/context_diagnostics/component_001/"
        "base_context_amplitude_overlay.png"
    )
    markdown = (output_json.parent / "visual_report.md").read_text(encoding="utf-8")
    assert f"[same-global-ROI context comparison]({context_path})" in markdown


def test_outlier_markdown_finishes_crop_table_before_component_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_report_module(monkeypatch)
    available = {
        "status": "available",
        "summary": {
            "baseline_candidate_to_public_p95": 1.0,
            "candidate_candidate_to_public_p95": 7.0,
            "candidate_minus_baseline_p95": 6.0,
            "allowed_candidate_p95": 6.0,
            "outlier_count": 1,
            "component_count": 1,
            "minimum_crop_face_distance": 2,
            "maximum_crop_face_distance": 2,
        },
        "components": [
            {
                "component_id": 1,
                "voxel_count": 1,
                "minimum_crop_face_distance": 2,
                "distance_to_public_fvt": {"maximum": 7.0},
                "representative_point": {"global_coordinate": [3, 4, 5]},
            }
        ],
    }
    empty = {
        "status": "available",
        "summary": {
            "baseline_candidate_to_public_p95": 1.0,
            "candidate_candidate_to_public_p95": 1.0,
            "candidate_minus_baseline_p95": 0.0,
            "allowed_candidate_p95": 6.0,
            "outlier_count": 0,
            "component_count": 0,
            "minimum_crop_face_distance": None,
            "maximum_crop_face_distance": None,
        },
        "components": [],
    }
    markdown = module.visual_report_markdown(
        {
            "config": {"crop_selection": {}},
            "scanner_policies": {
                "baseline": {"crops": []},
                "candidate": {"crops": []},
            },
            "consensus": {
                "candidate_minus_baseline": {
                    "crops": [
                        {"index": 1, "public_fvt_distance_outliers": available},
                        {"index": 2, "public_fvt_distance_outliers": empty},
                    ]
                }
            },
            "policy_validation": {"checks": {}},
            "manual_review": {"status": "pending", "items": {}},
        }
    )
    assert markdown.index("| crop_002 |") < markdown.index("### crop_001 / component_001")
    assert markdown.endswith("\n")


def test_small_real_pipeline_builds_crop_reports_and_strict_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_report_module(monkeypatch)
    shape = (17, 17, 17)
    ep = np.random.default_rng(0).random(shape, dtype=np.float32)
    pipeline_kwargs = {
        "sigma1": 1.0,
        "sigma2": 1.0,
        "phi_min": 0.0,
        "phi_max": 0.0,
        "theta_min": 65.0,
        "theta_max": 80.0,
        "ru": 1,
        "rv": 1,
        "rw": 1,
        "strain_max1": 0.5,
        "strain_max2": 0.5,
        "surface_smoothing1": 0.0,
        "surface_smoothing2": 0.0,
        "surface_orientation_smoothing": 0.0,
        "final_normalization_smoothing": 0.0,
        "d": 1,
        "fm": 0.0,
        "reference_thin_sigma": 1.0,
    }
    run = policy_evaluation.run_shared_scan_policy_pipeline(ep, **pipeline_kwargs)
    assert run["scanner_execution_count"] == 1

    outputs_by_role = {
        role: run["policies"][role]["outputs"] for role in policy_evaluation.POLICY_ROLES
    }
    baseline_outputs = outputs_by_role["baseline"]
    slices = tuple(slice(0, size) for size in shape)
    crop_reports: dict[str, dict[str, Any]] = {}
    for role, outputs in outputs_by_role.items():
        assert tuple(outputs) == policy_evaluation.OUTPUT_NAMES
        for values in outputs.values():
            assert values.shape == shape
            assert values.dtype == np.float32
            assert np.all(np.isfinite(values))
        crop = module.crop_validation.build_crop_report(
            crop_index=1,
            center=(8, 8, 8),
            slices=slices,
            crop_shape=shape,
            outputs=outputs,
            reference_fv=baseline_outputs["fv_py.dat"],
            reference_fvt=baseline_outputs["fvt_py.dat"],
            interior_margin=2,
        )
        crop["stage_density"] = policy_evaluation.build_stage_density_report(
            outputs,
            interior_margin=2,
        )
        assert crop["stage_density"]["fet"]["nonzero_count"] > 0
        assert crop["stage_density"]["fv"]["nonzero_count"] > 0
        assert crop["stage_density"]["fvt"]["nonzero_count"] > 0
        crop_reports[role] = crop

    direct = policy_evaluation.build_direct_policy_comparison(
        outputs_by_role["baseline"],
        outputs_by_role["candidate"],
        interior_margin=2,
    )
    configs = {
        branch.role: policy_evaluation.build_policy_config(
            branch,
            reference_thin_sigma=1.0,
            ru=1,
            rv=1,
            rw=1,
            strain_max1=0.5,
            strain_max2=0.5,
            surface_smoothing1=0.0,
            surface_smoothing2=0.0,
            surface_orientation_smoothing=0.0,
            final_normalization_smoothing=0.0,
            d=1,
            fm=0.0,
        )
        for branch in policy_evaluation.SCANNER_POLICIES
    }
    validation = policy_evaluation.validate_policy_comparison(
        baseline_crops=[crop_reports["baseline"]],
        candidate_crops=[crop_reports["candidate"]],
        direct_comparisons=[direct],
        baseline_config=configs["baseline"],
        candidate_config=configs["candidate"],
        scanner_execution_count=1,
        expected_crop_count=1,
    )
    assert validation["passed"] is True

    outlier_report = module.policy_diagnostics.build_public_fvt_distance_outlier_report(
        reference_fvt=baseline_outputs["fvt_py.dat"],
        baseline_outputs=outputs_by_role["baseline"],
        candidate_outputs=outputs_by_role["candidate"],
        crop_slices=slices,
        interior_margin=2,
    )
    assert outlier_report["status"] == "available"
    context_comparison = module.policy_diagnostics.build_same_global_roi_stage_comparison(
        base_outputs=outputs_by_role["candidate"],
        context_roi_outputs=outputs_by_role["candidate"],
    )
    assert context_comparison["status"] == "available"
    persistence = module.policy_diagnostics.build_context_outlier_persistence_report(
        base_outlier_report=outlier_report,
        reference_fvt=baseline_outputs["fvt_py.dat"],
        base_baseline_outputs=outputs_by_role["baseline"],
        base_candidate_outputs=outputs_by_role["candidate"],
        context_baseline_outputs=outputs_by_role["baseline"],
        context_candidate_outputs=outputs_by_role["candidate"],
        base_global_slices=slices,
    )
    assert persistence["status"] == "available"

    serialized = module.report_to_json(
        {
            "format_version": 1,
            "scanner_policies": {
                "baseline": {"crops": [crop_reports["baseline"]]},
                "candidate": {"crops": [crop_reports["candidate"]]},
            },
            "direct_comparison": direct,
            "public_fvt_distance_outliers": outlier_report,
            "context_comparison": context_comparison,
            "outlier_persistence": persistence,
            "policy_validation": validation,
        },
        pretty=True,
    )
    loaded = json.loads(serialized)
    assert loaded["policy_validation"]["passed"] is True
    assert serialized.endswith("\n")
