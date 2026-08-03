from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

import pyosv.evaluation.f3d_mode_comparison.reskin_policy_comparison as comparison_module
import pyosv.evaluation.f3d_mode_comparison.result as result_module
from pyosv.cli import f3d_mode_comparison
from pyosv.evaluation.f3d_mode_comparison import (
    F3DatasetSpec,
    F3ModeComparisonConfig,
    implementation_identity,
    validate_f3_reskin_policy_comparison,
    scanner_sampling_evidence,
)
from pyosv.evaluation.synthetic_quality import SyntheticSkinningConfig
from tests.evaluation.f3d_mode_comparison.test_bundle_validation import (
    _complete_small_bundle,
)
from tests.evaluation.f3d_mode_comparison.test_publication_contract_v3_integration import (
    _fixed_runtime_identity,
)
from tests.evaluation.f3d_mode_comparison.test_final_acceptance_integration import _official_fixture
from tests.evaluation.f3d_mode_comparison.test_integration import (
    _DeterministicScanner,
    _fixture_plan,
    _run_fixture,
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


def _file_snapshot(root: Path) -> dict[str, tuple[bytes, str, int, int]]:
    snapshot: dict[str, tuple[bytes, str, int, int]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        stat = path.stat()
        snapshot[path.relative_to(root).as_posix()] = (
            payload,
            hashlib.sha256(payload).hexdigest(),
            stat.st_size,
            stat.st_mtime_ns,
        )
    return snapshot


def _assert_source_snapshot_unchanged(
    before: dict[str, tuple[bytes, str, int, int]],
    after: dict[str, tuple[bytes, str, int, int]],
) -> None:
    comparison_prefix = "reskin_policy_comparison/"
    new_paths = set(after) - set(before)
    assert new_paths <= {path for path in after if path.startswith(comparison_prefix)}
    assert not (set(before) - set(after))
    for path, state in before.items():
        assert after[path] == state, f"source artifact changed: {path}"


def test_completed_bundle_comparison_only_cli_uses_real_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "bundle"
    spec = _official_fixture(data_root)
    runtime_identity = _fixed_runtime_identity()
    source_implementation = deepcopy(implementation_identity())
    source_implementation["software_versions"]["pyosv"] = "source-implementation-a"
    fixture_plan = _fixture_plan(spec)
    sampling_provider = _DeterministicScanner(0.0, 0.0, Counter())
    sampling_evidence = {
        backend: scanner_sampling_evidence(
            sampling_provider,
            fixture_plan.scanner_config_for(backend),
            backend,
            implementation_identity="cli-fixture-scanner-v1",
        )
        for backend in ("reference-like", "quality")
    }
    monkeypatch.setattr(
        result_module,
        "canonical_scanner_implementation_identity",
        lambda: "cli-fixture-scanner-v1",
    )
    monkeypatch.setattr(
        result_module,
        "canonical_scanner_sampling_evidence",
        lambda config, backend, **kwargs: sampling_evidence[backend],
    )
    source = _run_fixture(
        data_root,
        output_root,
        spec,
        Counter(),
        resume=False,
        monkeypatch=monkeypatch,
        workspace_implementation=source_implementation,
        workspace_runtime_identity=runtime_identity,
        scanner_implementation_identity="cli-fixture-scanner-v1",
        finalization_deep=False,
    )

    # Inject only the small official fixture's runtime/spec contract. All CLI dispatch,
    # source validation, result loading, comparison generation, and comparison
    # validation below use their production implementations.
    monkeypatch.setattr(result_module, "OFFICIAL_F3_DATASET_SPEC", spec)
    monkeypatch.setattr(
        result_module,
        "numerical_runtime_identity",
        lambda: deepcopy(runtime_identity),
    )
    monkeypatch.setattr(
        comparison_module,
        "numerical_runtime_identity",
        lambda: deepcopy(runtime_identity),
    )
    before = _file_snapshot(output_root)
    pair = [
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_root),
        "--compare-reskin-policies",
        "existing_cells_v1,reference_dense_v1",
    ]

    # State 1: a complete source bundle without a comparison.
    assert (
        f3d_mode_comparison.main(
            [
                "--data-root",
                str(data_root),
                "--output-dir",
                str(output_root),
                "--validate-only",
            ]
        )
        == 0
    )
    assert not (output_root / "reskin_policy_comparison").exists()
    assert _file_snapshot(output_root) == before

    # State 2: current code loads source identity A and creates only a shallow
    # comparison artifact; it must not attempt an exact source resume.
    assert f3d_mode_comparison.main([*pair, "--resume"]) == 0
    comparison_dir = output_root / "reskin_policy_comparison"
    completion_path = comparison_dir / "complete.json"
    shallow_completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert shallow_completion["validation_level"] == "shallow"
    shallow_snapshot = _file_snapshot(output_root)
    _assert_source_snapshot_unchanged(before, shallow_snapshot)
    assert set(shallow_snapshot) - set(before)

    # State 3: the same comparison-only dispatch promotes the existing
    # comparison to deep completion and then performs the explicit deep replay.
    assert f3d_mode_comparison.main([*pair, "--resume", "--deep-validate"]) == 0
    deep_completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert deep_completion["validation_level"] == "deep"
    final_snapshot = _file_snapshot(output_root)
    _assert_source_snapshot_unchanged(before, final_snapshot)
    assert validate_f3_reskin_policy_comparison(
        output_root,
        deep=True,
        require_deep=True,
    )

    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    report = json.loads(
        (comparison_dir / "reskin_policy_comparison.json").read_text(encoding="utf-8")
    )
    assert manifest["implementation_identity"] == source_implementation
    assert report["comparison_implementation_identity"] != manifest["implementation_identity"]
    assert (
        "evaluation/f3d_mode_comparison/skin_artifacts.py"
        in report["comparison_implementation_identity"]["algorithm_modules"]
    )
    assert source.run_fingerprint == manifest["run_fingerprint"]


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
    assert defaults.skinner_reskin_policy is None
    assert defaults.compare_reskin_policies is None
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
            "--skinner-reskin-policy",
            "reference_dense_v1",
            "--boundary-margin",
            "7",
        ]
    )
    assert overrides.resume is True
    assert overrides.deep_validate is True
    assert overrides.pretty is True
    assert overrides.no_skinning is True
    assert overrides.skinner_reskin_policy == "reference_dense_v1"
    assert overrides.boundary_margin == 7
    pair = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path / "pair"),
            "--compare-reskin-policies",
            "existing_cells_v1,reference_dense_v1",
        ]
    )
    assert pair.compare_reskin_policies == "existing_cells_v1,reference_dense_v1"
    validation_pair = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path / "pair"),
            "--validate-only",
            "--deep-validate",
            "--compare-reskin-policies",
            "existing_cells_v1,reference_dense_v1",
        ]
    )
    assert validation_pair.validate_only is True
    assert validation_pair.deep_validate is True
    assert validation_pair.compare_reskin_policies == "existing_cells_v1,reference_dense_v1"


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
    assert config.skinning_template.reskin_policy == "existing_cells_v1"
    assert config.boundary_diagnostic_margin == 9


def test_main_forwards_explicit_dense_reskin_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    data.mkdir()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_experiment",
        lambda **kwargs: captured.update(kwargs) or output,
    )

    assert (
        f3d_mode_comparison.main(
            [
                "--data-root",
                str(data),
                "--output-dir",
                str(output),
                "--skinner-reskin-policy",
                "reference_dense_v1",
            ]
        )
        == 0
    )

    config = captured["config"]
    assert isinstance(config, F3ModeComparisonConfig)
    assert config.skinning_template.reskin == SyntheticSkinningConfig().reskin
    assert config.skinning_template.reskin_policy == "reference_dense_v1"


def test_main_runs_fixed_reskin_pair_from_completed_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    data.mkdir()
    compared: list[Path] = []
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_experiment",
        lambda **kwargs: captured.update(kwargs) or output,
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "compare_reskin_policies_from_bundle",
        lambda bundle: compared.append(bundle),
    )

    assert (
        f3d_mode_comparison.main(
            [
                "--data-root",
                str(data),
                "--output-dir",
                str(output),
                "--compare-reskin-policies",
                "existing_cells_v1,reference_dense_v1",
            ]
        )
        == 0
    )
    config = captured["config"]
    assert isinstance(config, F3ModeComparisonConfig)
    assert config.skinning_template.reskin is True
    assert compared == [output]


def _write_completed_bundle_marker(output: Path, data: Path) -> None:
    output.mkdir()
    (output / "completion.json").write_text("{}\n", encoding="utf-8")
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "provenance": {"data_root": str(data)},
                "plan": {
                    "reference_workflow_settings": {
                        "skinning_config": {"reskin_policy": "existing_cells_v1"}
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_completed_resume_comparison_only_bypasses_source_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "bundle"
    data.mkdir()
    _write_completed_bundle_marker(output, data)
    comparison = output / "reskin_policy_comparison"
    comparison.mkdir()
    (comparison / "complete.json").write_text(
        json.dumps({"validation_level": "deep"}),
        encoding="utf-8",
    )
    source_paths = (output / "completion.json", output / "run_manifest.json")
    before = {
        path: (path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in source_paths
    }
    bundle_validations: list[tuple[Path, bool]] = []
    comparisons: list[tuple[Path, dict[str, bool]]] = []
    deep_validations: list[tuple[Path, dict[str, bool]]] = []

    monkeypatch.setattr(
        f3d_mode_comparison,
        "validate_completed_f3d_bundle",
        lambda path, deep=False: bundle_validations.append((path, deep)),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "compare_reskin_policies_from_bundle",
        lambda path, **kwargs: comparisons.append((path, kwargs)),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "validate_f3_reskin_policy_comparison",
        lambda path, **kwargs: deep_validations.append((path, kwargs)),
    )

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        pytest.fail("completed comparison-only resume entered source execution")

    for name in (
        "run_experiment",
        "prepare_run_workspace",
        "F3VolumeSource",
        "run_scanner_stages",
        "run_f3d_mode_comparison",
    ):
        monkeypatch.setattr(f3d_mode_comparison, name, forbidden)

    code = f3d_mode_comparison.main(
        [
            "--data-root",
            str(data),
            "--output-dir",
            str(output),
            "--resume",
            "--compare-reskin-policies",
            "existing_cells_v1,reference_dense_v1",
            "--deep-validate",
        ]
    )

    assert code == 0
    assert bundle_validations == [(output, True)]
    assert comparisons == [(output, {"resume": True, "deep": True})]
    assert deep_validations == [
        (output, {"deep": True, "require_deep": True}),
    ]
    assert capsys.readouterr().out == f"{output}\n"
    after = {
        path: (path.read_bytes(), path.stat().st_size, path.stat().st_mtime_ns)
        for path in source_paths
    }
    assert after == before


def test_completed_resume_comparison_only_rejects_invalid_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "bundle"
    data.mkdir()
    _write_completed_bundle_marker(output, data)
    source_paths = (output / "completion.json", output / "run_manifest.json")
    before = {path: path.read_bytes() for path in source_paths}

    def invalid_source(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ValueError("invalid source")

    monkeypatch.setattr(
        f3d_mode_comparison,
        "validate_completed_f3d_bundle",
        invalid_source,
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "compare_reskin_policies_from_bundle",
        lambda *args, **kwargs: pytest.fail("invalid source reached comparison"),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_experiment",
        lambda *args, **kwargs: pytest.fail("invalid source reached source run"),
    )

    code = f3d_mode_comparison.main(
        [
            "--data-root",
            str(data),
            "--output-dir",
            str(output),
            "--resume",
            "--compare-reskin-policies",
            "existing_cells_v1,reference_dense_v1",
        ]
    )

    assert code == 1
    assert {path: path.read_bytes() for path in source_paths} == before
    assert "invalid source" in capsys.readouterr().err


def test_incomplete_resume_with_comparison_uses_source_resume_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "workspace"
    data.mkdir()
    output.mkdir()
    called: dict[str, object] = {}
    monkeypatch.setattr(
        f3d_mode_comparison,
        "_resume_completed_bundle_comparison",
        lambda *args, **kwargs: pytest.fail("incomplete workspace used completed path"),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_experiment",
        lambda **kwargs: called.update(kwargs) or output,
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "compare_reskin_policies_from_bundle",
        lambda *args, **kwargs: None,
    )

    assert (
        f3d_mode_comparison.main(
            [
                "--data-root",
                str(data),
                "--output-dir",
                str(output),
                "--resume",
                "--compare-reskin-policies",
                "existing_cells_v1,reference_dense_v1",
            ]
        )
        == 0
    )
    assert called["resume"] is True


def test_completed_resume_without_comparison_uses_existing_source_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "bundle"
    data.mkdir()
    _write_completed_bundle_marker(output, data)
    called: dict[str, object] = {}
    monkeypatch.setattr(
        f3d_mode_comparison,
        "_resume_completed_bundle_comparison",
        lambda *args, **kwargs: pytest.fail("comparison-only path used without comparison"),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_experiment",
        lambda **kwargs: called.update(kwargs) or output,
    )

    assert (
        f3d_mode_comparison.main(
            [
                "--data-root",
                str(data),
                "--output-dir",
                str(output),
                "--resume",
            ]
        )
        == 0
    )
    assert called["resume"] is True


@pytest.mark.parametrize("resume", [False, True], ids=("fresh", "resume"))
def test_main_propagates_deep_validation_to_reskin_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    resume: bool,
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    data.mkdir()
    if resume:
        output.mkdir()
    compared: list[tuple[Path, dict[str, bool]]] = []
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_experiment",
        lambda **kwargs: output,
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "compare_reskin_policies_from_bundle",
        lambda bundle, **kwargs: compared.append((bundle, kwargs)),
    )
    arguments = [
        "--data-root",
        str(data),
        "--output-dir",
        str(output),
        "--compare-reskin-policies",
        "existing_cells_v1,reference_dense_v1",
        "--deep-validate",
    ]
    if resume:
        arguments.append("--resume")

    assert f3d_mode_comparison.main(arguments) == 0
    assert compared == [
        (
            output,
            {"deep": True, **({"resume": True} if resume else {})},
        )
    ]


def test_resume_recovers_recorded_dense_reskin_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    output = tmp_path / "run"
    data.mkdir()
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "plan": {
                    "reference_workflow_settings": {
                        "skinning_config": {
                            "reskin_policy": "reference_dense_v1",
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        f3d_mode_comparison,
        "run_experiment",
        lambda **kwargs: captured.update(kwargs) or output,
    )

    assert (
        f3d_mode_comparison.main(
            [
                "--data-root",
                str(data),
                "--output-dir",
                str(output),
                "--resume",
            ]
        )
        == 0
    )

    config = captured["config"]
    assert isinstance(config, F3ModeComparisonConfig)
    assert config.skinning_template.reskin_policy == "reference_dense_v1"


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


@pytest.mark.parametrize("deep", (False, True), ids=("shallow", "deep"))
def test_validate_only_validates_existing_reskin_comparison_without_generation(
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
        json.dumps({"provenance": {"data_root": str(data)}}), encoding="utf-8"
    )
    bundle_validations: list[tuple[Path, bool]] = []
    comparison_validations: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        f3d_mode_comparison,
        "validate_completed_f3d_bundle",
        lambda path, deep=False: bundle_validations.append((path, deep)),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "validate_f3_reskin_policy_comparison",
        lambda path, deep=False: comparison_validations.append((path, deep)),
    )
    monkeypatch.setattr(
        f3d_mode_comparison,
        "compare_reskin_policies_from_bundle",
        lambda *args, **kwargs: pytest.fail("validate-only must not generate a comparison"),
    )

    arguments = [
        "--output-dir",
        str(output),
        "--validate-only",
        "--compare-reskin-policies",
        "existing_cells_v1,reference_dense_v1",
    ]
    if deep:
        arguments.append("--deep-validate")
    assert f3d_mode_comparison.main(arguments) == 0
    assert bundle_validations == [(output, deep)]
    assert comparison_validations == [(output, deep)]
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
