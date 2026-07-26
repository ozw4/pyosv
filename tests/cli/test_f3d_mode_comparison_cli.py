from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyosv.cli import f3d_mode_comparison
from pyosv.evaluation.f3d_mode_comparison import F3ModeComparisonConfig


def test_parser_defaults_and_global_overrides(tmp_path: Path) -> None:
    parser = f3d_mode_comparison.build_parser()
    defaults = parser.parse_args(["--output-dir", str(tmp_path / "run")])

    assert defaults.data_root is None
    assert defaults.output_dir == tmp_path / "run"
    assert defaults.resume is False
    assert defaults.validate_only is False
    assert defaults.deep_validate is False
    assert defaults.pretty is False
    assert defaults.no_skinning is False
    assert defaults.boundary_margin == 16

    overrides = parser.parse_args(
        [
            "--data-root",
            str(tmp_path / "data"),
            "--output-dir",
            str(tmp_path / "run"),
            "--resume",
            "--deep-validate",
            "--pretty",
            "--no-skinning",
            "--boundary-margin",
            "7",
        ]
    )
    assert overrides.resume is True
    assert overrides.deep_validate is True
    assert overrides.pretty is True
    assert overrides.no_skinning is True
    assert overrides.boundary_margin == 7


def test_parser_rejects_invalid_combinations_and_margin(tmp_path: Path) -> None:
    parser = f3d_mode_comparison.build_parser()
    with pytest.raises(SystemExit) as conflict:
        parser.parse_args(
            [
                "--output-dir",
                str(tmp_path / "run"),
                "--resume",
                "--validate-only",
            ]
        )
    assert conflict.value.code == 2

    with pytest.raises(SystemExit) as margin:
        parser.parse_args(["--output-dir", str(tmp_path / "run"), "--boundary-margin", "-1"])
    assert margin.value.code == 2


def test_main_passes_complete_run_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    data.mkdir()
    output.mkdir()
    received: dict[str, object] = {}

    def fake_run(**kwargs: object) -> Path:
        received.update(kwargs)
        return output

    monkeypatch.setattr(f3d_mode_comparison, "run_experiment", fake_run)

    code = f3d_mode_comparison.main(
        [
            "--data-root",
            str(data),
            "--output-dir",
            str(output),
            "--resume",
            "--deep-validate",
            "--pretty",
            "--no-skinning",
            "--boundary-margin",
            "9",
        ]
    )

    assert code == 0
    assert capsys.readouterr().out == f"{output}\n"
    assert received["data_root"] == data
    assert received["output_dir"] == output
    assert received["resume"] is True
    assert received["deep"] is True
    assert received["pretty"] is True
    config = received["config"]
    assert isinstance(config, F3ModeComparisonConfig)
    assert config.skinning_enabled is False
    assert config.boundary_diagnostic_margin == 9


def test_validate_only_uses_existing_bundle_without_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "bundle"
    data.mkdir()
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps({"provenance": {"data_root": str(data)}}),
        encoding="utf-8",
    )
    seen: list[tuple[Path, bool]] = []
    monkeypatch.delenv("PYOSV_F3D_DATA_ROOT", raising=False)

    def fake_validate(path: Path, deep: bool = False) -> bool:
        seen.append((path, deep))
        return True

    monkeypatch.setattr(
        f3d_mode_comparison,
        "validate_completed_f3d_bundle",
        fake_validate,
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_experiment",
        lambda **kwargs: pytest.fail("validate-only must not compute"),
    )

    code = f3d_mode_comparison.main(
        ["--output-dir", str(output), "--validate-only", "--deep-validate"]
    )

    assert code == 0
    assert seen == [(output, True)]
    assert capsys.readouterr().out == f"{output}\n"


def test_validate_only_uses_manifest_data_root_for_nesting_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = tmp_path / "data"
    output = data / "bundle"
    output.mkdir(parents=True)
    (output / "run_manifest.json").write_text(
        json.dumps({"provenance": {"data_root": str(data)}}),
        encoding="utf-8",
    )
    monkeypatch.delenv("PYOSV_F3D_DATA_ROOT", raising=False)
    monkeypatch.setattr(
        f3d_mode_comparison,
        "validate_completed_f3d_bundle",
        lambda *args, **kwargs: pytest.fail("nested bundle must fail preflight"),
    )

    code = f3d_mode_comparison.main(["--output-dir", str(output), "--validate-only"])

    assert code == 1
    assert "inside the F3 data root" in capsys.readouterr().err


def test_main_rejects_existing_new_output_and_data_nested_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_experiment",
        lambda **kwargs: pytest.fail("preflight failure must not compute"),
    )

    assert f3d_mode_comparison.main(["--data-root", str(data), "--output-dir", str(existing)]) == 1
    assert "already exists" in capsys.readouterr().err

    assert (
        f3d_mode_comparison.main(
            [
                "--data-root",
                str(data),
                "--output-dir",
                str(data / "generated"),
            ]
        )
        == 1
    )
    assert "inside the F3 data root" in capsys.readouterr().err

    assert (
        f3d_mode_comparison.main(
            [
                "--data-root",
                str(data),
                "--output-dir",
                str(data / "bundle"),
                "--validate-only",
            ]
        )
        == 1
    )
    assert "inside the F3 data root" in capsys.readouterr().err


@pytest.mark.parametrize(
    "failure",
    [
        ValueError("dataset identity failed"),
        RuntimeError("compute failed"),
        OSError("artifact failed"),
        ValueError("validation failed"),
    ],
)
def test_main_reports_normal_failures_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure: Exception,
) -> None:
    data = tmp_path / "data"
    data.mkdir()

    def fail(**kwargs: object) -> Path:
        raise failure

    monkeypatch.setattr(f3d_mode_comparison, "run_experiment", fail)

    assert (
        f3d_mode_comparison.main(
            [
                "--data-root",
                str(data),
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"error: {failure}\n"
    assert "Traceback" not in captured.err


def test_complete_resume_validates_without_compute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    output.mkdir()
    completion = output / "completion.json"
    completion.write_text("{}\n", encoding="utf-8")
    source = SimpleNamespace(identity=object())

    class FakeSource:
        def __enter__(self) -> SimpleNamespace:
            return source

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        f3d_mode_comparison,
        "F3VolumeSource",
        lambda root: FakeSource(),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "build_f3d_mode_comparison_plan",
        lambda config: object(),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "prepare_run_workspace",
        lambda *args, **kwargs: SimpleNamespace(path=output),
    )
    seen: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        f3d_mode_comparison,
        "validate_completed_f3d_bundle",
        lambda path, deep=False: seen.append((path, deep)) or True,
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_scanner_stages",
        lambda *args, **kwargs: pytest.fail("complete resume must not compute"),
    )

    result = f3d_mode_comparison.run_experiment(
        config=F3ModeComparisonConfig(),
        data_root=data,
        output_dir=output,
        resume=True,
        deep=True,
    )

    assert result == output
    assert seen == [(output, True)]


def test_failed_post_run_validation_removes_only_new_root_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    output.mkdir()
    stage = output / "stages" / "scanner" / ("a" * 64)
    stage.mkdir(parents=True)
    stage_file = stage / "ft.dat"
    stage_file.write_bytes(b"valid-stage")
    source = SimpleNamespace(identity=object())

    class FakeSource:
        def __enter__(self) -> SimpleNamespace:
            return source

        def __exit__(self, *args: object) -> None:
            return None

    class FakeRSS:
        def process_peak(self) -> None:
            return None

    plan = SimpleNamespace(dataset_spec=SimpleNamespace(shape=(420, 400, 100)))
    workspace = SimpleNamespace(path=output)
    cell_result = SimpleNamespace(cells=("cells",), stage_runtime=("runtime",))
    monkeypatch.setattr(
        f3d_mode_comparison,
        "F3VolumeSource",
        lambda root: FakeSource(),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "build_f3d_mode_comparison_plan",
        lambda config: plan,
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "prepare_run_workspace",
        lambda *args, **kwargs: workspace,
    )
    monkeypatch.setattr(f3d_mode_comparison, "PeakRSSRecorder", FakeRSS)
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_scanner_stages",
        lambda *args, **kwargs: {"scanner": object()},
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_f3d_mode_comparison",
        lambda *args, **kwargs: cell_result,
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "extract_f3d_metrics",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "extract_f3d_diagnostics",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "extract_f3d_resources",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "F3ModeComparisonResult",
        SimpleNamespace(from_extractions=lambda **kwargs: object()),
    )

    def fake_finalize(*args: object, **kwargs: object) -> None:
        assert kwargs["deep"] is False
        (output / "completion.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(f3d_mode_comparison, "finalize_f3d_bundle", fake_finalize)

    def fail_validation(path: Path, deep: bool = False) -> bool:
        assert deep is True
        raise ValueError("deep validation failed")

    monkeypatch.setattr(
        f3d_mode_comparison,
        "validate_completed_f3d_bundle",
        fail_validation,
    )

    with pytest.raises(ValueError, match="deep validation failed"):
        f3d_mode_comparison.run_experiment(
            config=F3ModeComparisonConfig(),
            data_root=data,
            output_dir=output,
            resume=True,
            deep=True,
        )

    assert not (output / "completion.json").exists()
    assert stage_file.read_bytes() == b"valid-stage"


def test_help_describes_scope_reference_and_resume() -> None:
    help_text = " ".join(f3d_mode_comparison.build_parser().format_help().split())

    assert "full-volume F3 2x2" in help_text
    assert "four cells" in help_text
    assert "public-reference agreement, not geological accuracy" in help_text
    assert "matching run fingerprint" in help_text
    assert "validated without recomputation" in help_text
