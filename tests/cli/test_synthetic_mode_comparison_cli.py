from __future__ import annotations

import json
from pathlib import Path

import pytest

from pyosv.cli import synthetic_mode_comparison
from pyosv.evaluation.synthetic_mode_comparison import (
    ARTIFACT_SCHEMA_VERSION,
    REQUIRED_BUNDLE_FILES,
    RUNTIME_CONTRACT_VERSION,
    SCALAR_EVIDENCE_CONTRACT_VERSION,
    SyntheticModeComparisonConfig,
    validate_completed_bundle,
)
from pyosv.evaluation.synthetic_mode_comparison import runner as comparison_runner
from pyosv.evaluation.synthetic_quality import SyntheticTruthMetricConfig
from pyosv.evaluation.synthetic_quality import runner as quality_runner


def test_parser_defaults(tmp_path: Path) -> None:
    args = synthetic_mode_comparison.build_parser().parse_args(
        ["--output-dir", str(tmp_path / "bundle")]
    )

    assert args.case_set == "minimal"
    assert args.case_ids is None
    assert args.shape == (33, 33, 33)
    assert args.trial_seeds == (20260707,)
    assert args.no_oracle_workflow_isolation is False
    assert args.skip_skinning is False
    assert args.pretty is False


def test_parser_accepts_separate_and_equals_forms(tmp_path: Path) -> None:
    parser = synthetic_mode_comparison.build_parser()
    separate = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path / "one"),
            "--case-ids",
            "curved_surface,single_vertical_plane",
            "--shape",
            "9,11,13",
            "--trial-seeds",
            "3,5",
            "--no-oracle-workflow-isolation",
            "--skip-skinning",
            "--pretty",
        ]
    )
    equals = parser.parse_args(
        [
            f"--output-dir={tmp_path / 'two'}",
            "--case-set=geometry",
            "--shape=9,11,13",
            "--trial-seeds=3,5",
        ]
    )

    assert separate.case_ids == ("curved_surface", "single_vertical_plane")
    assert separate.case_set is None
    assert separate.shape == equals.shape == (9, 11, 13)
    assert separate.trial_seeds == equals.trial_seeds == (3, 5)
    assert separate.no_oracle_workflow_isolation is True
    assert separate.skip_skinning is True
    assert separate.pretty is True
    assert equals.case_set == "geometry"


@pytest.mark.parametrize(
    "arguments",
    (
        ("--case-set", "minimal", "--case-ids", "single_vertical_plane"),
        ("--shape", "9,9"),
        ("--trial-seeds", ""),
        ("--trial-seeds", "1,,2"),
        ("--trial-seeds", "1,1"),
        ("--trial-seeds", "-1"),
        ("--trial-seeds", "true"),
        ("--trial-seeds", "1.5"),
        ("--case-ids", "missing"),
        ("--case-ids", "single_vertical_plane,single_vertical_plane"),
        ("--case-ids", "single_vertical_plane,"),
    ),
)
def test_parser_rejects_invalid_arguments(arguments: tuple[str, ...], tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        synthetic_mode_comparison.build_parser().parse_args(
            ["--output-dir", str(tmp_path / "bundle"), *arguments]
        )

    assert error.value.code == 2


def test_main_runs_writes_validates_and_prints_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "bundle"
    calls: list[tuple[str, object]] = []
    result = object()

    def fake_run(config: SyntheticModeComparisonConfig) -> object:
        calls.append(("run", config))
        return result

    def fake_write(
        actual_result: object,
        output_dir: Path,
        *,
        config: SyntheticModeComparisonConfig,
        pretty: bool,
    ) -> Path:
        calls.append(("write", (actual_result, output_dir, config, pretty)))
        output_dir.mkdir()
        return output_dir

    def fake_validate(path: Path) -> bool:
        calls.append(("validate", path))
        return True

    monkeypatch.setattr(synthetic_mode_comparison, "run_mode_comparison", fake_run)
    monkeypatch.setattr(synthetic_mode_comparison, "write_artifact_bundle", fake_write)
    monkeypatch.setattr(synthetic_mode_comparison, "validate_completed_bundle", fake_validate)

    code = synthetic_mode_comparison.main(
        [
            "--output-dir",
            str(output),
            "--case-ids",
            "curved_surface,single_vertical_plane",
            "--shape",
            "9,9,9",
            "--trial-seeds",
            "3,5",
            "--no-oracle-workflow-isolation",
            "--skip-skinning",
            "--pretty",
        ]
    )

    assert code == 0
    assert [name for name, _ in calls] == ["run", "write", "validate"]
    config = calls[0][1]
    assert isinstance(config, SyntheticModeComparisonConfig)
    assert config.case_set is None
    assert config.case_ids == ("curved_surface", "single_vertical_plane")
    assert config.shape == (9, 9, 9)
    assert config.trial_seeds == (3, 5)
    assert config.include_oracle_workflow_isolation is False
    assert config.skinning_config.enabled is False
    assert calls[1][1] == (result, output, config, True)
    captured = capsys.readouterr()
    assert captured.out == f"{output}\n"
    assert captured.err == ""


@pytest.mark.parametrize("failure_stage", ("run", "write", "validate"))
def test_main_reports_failures_without_a_final_bundle(
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "bundle"

    def fake_run(config: SyntheticModeComparisonConfig) -> object:
        if failure_stage == "run":
            raise RuntimeError("experiment failed")
        return object()

    def fake_write(
        result: object,
        output_dir: Path,
        *,
        config: SyntheticModeComparisonConfig,
        pretty: bool,
    ) -> Path:
        if failure_stage == "write":
            raise RuntimeError("serialization failed")
        output_dir.mkdir()
        return output_dir

    def fake_validate(path: Path) -> bool:
        if failure_stage == "validate":
            raise ValueError("completion failed")
        return True

    monkeypatch.setattr(synthetic_mode_comparison, "run_mode_comparison", fake_run)
    monkeypatch.setattr(synthetic_mode_comparison, "write_artifact_bundle", fake_write)
    monkeypatch.setattr(synthetic_mode_comparison, "validate_completed_bundle", fake_validate)

    assert synthetic_mode_comparison.main(["--output-dir", str(output)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "failed" in captured.err
    assert not output.exists()


def test_main_reports_cleanup_failure_and_removes_completion_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "bundle"

    monkeypatch.setattr(
        synthetic_mode_comparison,
        "run_mode_comparison",
        lambda config: object(),
    )

    def fake_write(result, output_dir, *, config, pretty):
        output_dir.mkdir()
        (output_dir / "completion.json").write_text("{}", encoding="utf-8")
        return output_dir

    monkeypatch.setattr(synthetic_mode_comparison, "write_artifact_bundle", fake_write)
    monkeypatch.setattr(
        synthetic_mode_comparison,
        "validate_completed_bundle",
        lambda path: (_ for _ in ()).throw(ValueError("completion failed")),
    )
    monkeypatch.setattr(
        synthetic_mode_comparison.shutil,
        "rmtree",
        lambda path: (_ for _ in ()).throw(OSError("cleanup blocked")),
    )

    assert synthetic_mode_comparison.main(["--output-dir", str(output)]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "completion failed" in captured.err
    assert "failed to remove invalid artifact bundle: cleanup blocked" in captured.err
    assert not (output / "completion.json").exists()


def test_real_small_skip_skinning_cli_writes_a_valid_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "bundle"

    assert (
        synthetic_mode_comparison.main(
            [
                "--output-dir",
                str(output),
                "--case-set",
                "minimal",
                "--shape",
                "9,9,9",
                "--skip-skinning",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.out == f"{output}\n"
    assert captured.err == ""
    assert {path.name for path in output.iterdir()} == set(REQUIRED_BUNDLE_FILES)
    assert validate_completed_bundle(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    reports = json.loads((output / "cell_reports.json").read_text(encoding="utf-8"))
    assert manifest["artifact_schema_version"] == ARTIFACT_SCHEMA_VERSION == 3
    assert manifest["scalar_evidence_contract_version"] == SCALAR_EVIDENCE_CONTRACT_VERSION == 5
    assert manifest["runtime_contract_version"] == RUNTIME_CONTRACT_VERSION == 4
    assert manifest["input_config"]["skinning_config"]["enabled"] is False
    assert {
        label
        for label, payload in reports[0]["cells"].items()
        if "scanner_metric_evidence" in payload
    } == {"RL-SCAN", "Q-SCAN", "RL-REF", "RL-QUAL", "Q-REF", "Q-QUAL"}


def test_empty_truth_surface_cli_failure_leaves_no_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "bundle"
    config_type = synthetic_mode_comparison.SyntheticModeComparisonConfig
    calls = {"case_generation": 0, "scanner_input": 0}
    original_case_factory = comparison_runner._build_trial_case
    original_scanner_input = quality_runner.make_scanner_input_from_case

    def counted_case_factory(*args, **kwargs):
        calls["case_generation"] += 1
        return original_case_factory(*args, **kwargs)

    def counted_scanner_input(*args, **kwargs):
        calls["scanner_input"] += 1
        return original_scanner_input(*args, **kwargs)

    def config_with_empty_truth_surface(**kwargs):
        return config_type(
            **kwargs,
            truth_metric_config=SyntheticTruthMetricConfig(
                truth_surface_half_width=0.0,
            ),
        )

    monkeypatch.setattr(
        synthetic_mode_comparison,
        "SyntheticModeComparisonConfig",
        config_with_empty_truth_surface,
    )
    monkeypatch.setattr(comparison_runner, "_build_trial_case", counted_case_factory)
    monkeypatch.setattr(quality_runner, "make_scanner_input_from_case", counted_scanner_input)

    code = synthetic_mode_comparison.main(
        [
            "--output-dir",
            str(output),
            "--case-ids",
            "single_dipping_plane",
            "--shape",
            "10,10,10",
            "--skip-skinning",
        ]
    )

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "empty truth-surface support" in captured.err
    assert calls == {"case_generation": 1, "scanner_input": 0}
    assert not output.exists()
    assert not (output / "completion.json").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))
    assert not tuple(tmp_path.iterdir())


def test_invalid_truth_metric_cli_failure_leaves_no_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "bundle"
    config_type = synthetic_mode_comparison.SyntheticModeComparisonConfig

    def config_with_invalid_truth_metric(**kwargs):
        return config_type(
            **kwargs,
            truth_metric_config=SyntheticTruthMetricConfig(buffer_radius=-0.1),
        )

    def unexpected_run(config):
        raise AssertionError("invalid config must fail before experiment execution")

    monkeypatch.setattr(
        synthetic_mode_comparison,
        "SyntheticModeComparisonConfig",
        config_with_invalid_truth_metric,
    )
    monkeypatch.setattr(synthetic_mode_comparison, "run_mode_comparison", unexpected_run)

    code = synthetic_mode_comparison.main(["--output-dir", str(output)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "buffer_radius must be non-negative" in captured.err
    assert not output.exists()
    assert not (output / "completion.json").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))
    assert not tuple(tmp_path.iterdir())


def test_main_rejects_an_existing_output_before_running(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "bundle"
    output.mkdir()
    called = False

    def fake_run(config: SyntheticModeComparisonConfig) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(synthetic_mode_comparison, "run_mode_comparison", fake_run)

    assert synthetic_mode_comparison.main(["--output-dir", str(output)]) == 1
    assert called is False
    assert "already exists" in capsys.readouterr().err
