from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import pyosv.evaluation.f3d_mode_comparison.result as result_module
from pyosv.cli import f3d_mode_comparison
from pyosv.evaluation.f3d_mode_comparison import (
    F3DatasetSpec,
    F3ModeComparisonConfig,
)
from tests.evaluation.f3d_mode_comparison.test_bundle_validation import (
    _complete_small_bundle,
)
from tests.evaluation.f3d_mode_comparison.test_publication_contract_v3_integration import (
    _fixed_runtime_identity,
)


def _complete_official_cli_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runtime_identity: dict[str, object],
) -> tuple[Path, Path]:
    shape = (3, 4, 5)
    files = (
        ("input", "ep.dat"),
        ("reference_fault_likelihood", "fl.dat"),
        ("reference_fault_votes", "fv.dat"),
        ("reference_thinned_fault_votes", "fvt.dat"),
    )
    spec = F3DatasetSpec(
        dataset_id="result-fixture",
        shape=shape,
        files=files,
        expected_bytes=3 * 4 * 5 * 4,
    )
    monkeypatch.setattr(result_module, "OFFICIAL_F3_DATASET_SPEC", spec)
    root = _complete_small_bundle(tmp_path, runtime_identity=runtime_identity)
    data_root = tmp_path / "data"
    monkeypatch.setattr(result_module, "F3_DATASET_ID", spec.dataset_id)
    return root, data_root


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


@pytest.mark.parametrize("deep", [False, True], ids=("shallow", "deep"))
def test_validate_only_uses_existing_bundle_without_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    deep: bool,
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

    arguments = ["--output-dir", str(output), "--validate-only"]
    if deep:
        arguments.append("--deep-validate")
    code = f3d_mode_comparison.main(arguments)

    assert code == 0
    assert seen == [(output, deep)]
    assert capsys.readouterr().out == f"{output}\n"


def test_complete_shallow_resume_rejects_recorded_publication_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid_recorded = deepcopy(_fixed_runtime_identity())
    invalid_recorded["python_hash_seed"] = "1"
    output, data = _complete_official_cli_bundle(
        tmp_path,
        monkeypatch,
        invalid_recorded,
    )
    calls: list[str] = []

    def forbidden(name: str):
        def fail(*args: object, **kwargs: object) -> object:
            del args, kwargs
            calls.append(name)
            pytest.fail(f"recorded-policy rejection reached {name}")

        return fail

    monkeypatch.setattr(
        f3d_mode_comparison,
        "F3VolumeSource",
        forbidden("dataset open"),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_scanner_stages",
        forbidden("scanner factory"),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_f3d_mode_comparison",
        forbidden("workflow compute"),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "extract_f3d_metrics",
        forbidden("metric compute"),
    )
    monkeypatch.setattr(
        result_module,
        "numerical_runtime_identity",
        lambda: pytest.fail("shallow resume must not inspect the current runtime"),
    )

    code = f3d_mode_comparison.main(
        [
            "--data-root",
            str(data),
            "--output-dir",
            str(output),
            "--resume",
        ]
    )

    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "PYTHONHASHSEED must equal 0" in captured.err
    assert not calls


@pytest.mark.parametrize(
    ("deep", "expected_code"),
    ((False, 0), (True, 1)),
    ids=("shallow-accepts-current-mismatch", "deep-rejects-current-mismatch"),
)
def test_validate_only_separates_recorded_and_current_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    deep: bool,
    expected_code: int,
) -> None:
    recorded = _fixed_runtime_identity()
    output, data = _complete_official_cli_bundle(tmp_path, monkeypatch, recorded)
    current = deepcopy(recorded)
    current["platform_machine"] = "other-publication-valid-machine"
    monkeypatch.setattr(result_module, "numerical_runtime_identity", lambda: current)
    monkeypatch.setattr(f3d_mode_comparison, "_recorded_data_root", lambda bundle: data)
    arguments = ["--output-dir", str(output), "--validate-only"]
    if deep:
        arguments.append("--deep-validate")

    code = f3d_mode_comparison.main(arguments)

    assert code == expected_code
    captured = capsys.readouterr()
    if deep:
        assert captured.out == ""
        assert "current runtime identity does not match run manifest" in captured.err
    else:
        assert captured.out == f"{output}\n"
        assert captured.err == ""


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


def test_validate_only_environment_cannot_override_manifest_nesting_preflight(
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
    unrelated_data = tmp_path / "unrelated-data"
    unrelated_data.mkdir()
    monkeypatch.setenv("PYOSV_F3D_DATA_ROOT", str(unrelated_data))
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
        "_recorded_runtime_identity",
        lambda root: {},
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


def test_complete_resume_validation_precedes_dataset_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "completion.json").write_text("{}\n", encoding="utf-8")

    def reject_recorded_runtime(*args: object, **kwargs: object) -> None:
        raise ValueError("recorded runtime rejected")

    monkeypatch.setattr(
        f3d_mode_comparison,
        "validate_completed_f3d_bundle",
        reject_recorded_runtime,
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "F3VolumeSource",
        lambda root: pytest.fail("complete resume validation must precede dataset open"),
    )

    with pytest.raises(ValueError, match="recorded runtime rejected"):
        f3d_mode_comparison.run_experiment(
            config=F3ModeComparisonConfig(),
            data_root=tmp_path / "data",
            output_dir=output,
            resume=True,
            deep=False,
        )


def test_publication_runtime_failure_precedes_dataset_open_and_stage_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def reject_runtime(_identity: object) -> None:
        raise ValueError("runtime rejected")

    monkeypatch.setattr(
        f3d_mode_comparison,
        "validate_publication_runtime_identity",
        reject_runtime,
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "F3VolumeSource",
        lambda root: pytest.fail("runtime rejection must precede dataset open"),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "build_f3d_mode_comparison_plan",
        lambda config: pytest.fail("runtime rejection must precede stage selection"),
    )

    with pytest.raises(ValueError, match="runtime rejected"):
        f3d_mode_comparison.run_experiment(
            config=F3ModeComparisonConfig(),
            data_root=tmp_path / "data",
            output_dir=tmp_path / "run",
            resume=False,
            deep=False,
        )


def test_requested_deep_validation_is_part_of_atomic_finalization(
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
        assert kwargs["deep"] is True
        assert kwargs["pretty"] is True
        raise ValueError("deep validation failed before completion")

    monkeypatch.setattr(f3d_mode_comparison, "finalize_f3d_bundle", fake_finalize)
    monkeypatch.setattr(
        f3d_mode_comparison,
        "validate_completed_f3d_bundle",
        lambda *args, **kwargs: pytest.fail("new runs validate inside finalization"),
    )

    with pytest.raises(ValueError, match="deep validation failed before completion"):
        f3d_mode_comparison.run_experiment(
            config=F3ModeComparisonConfig(),
            data_root=data,
            output_dir=output,
            resume=True,
            deep=True,
            pretty=True,
            _skip_current_publication_runtime_policy_for_testing=True,
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
