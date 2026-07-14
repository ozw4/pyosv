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

    serialized = module.report_to_json(
        {
            "format_version": 1,
            "scanner_policies": {
                "baseline": {"crops": [crop_reports["baseline"]]},
                "candidate": {"crops": [crop_reports["candidate"]]},
            },
            "direct_comparison": direct,
            "policy_validation": validation,
        },
        pretty=True,
    )
    loaded = json.loads(serialized)
    assert loaded["policy_validation"]["passed"] is True
    assert serialized.endswith("\n")
