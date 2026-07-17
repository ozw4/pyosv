from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from dataclasses import fields, replace
from pathlib import Path

import pytest

from pyosv.evaluation.synthetic_mode_comparison import (
    HASHED_BUNDLE_FILES,
    METRIC_SCHEMA_VERSION,
    REQUIRED_BUNDLE_FILES,
    AggregateRow,
    ContrastRow,
    MetricRow,
    RuntimeRow,
    SyntheticModeComparisonConfig,
    SyntheticModeComparisonResult,
    run_mode_comparison,
    validate_completed_bundle,
    write_artifact_bundle,
)
from pyosv.evaluation.synthetic_mode_comparison import artifacts
from pyosv.evaluation.synthetic_quality.stage_cache import PipelineStageCacheStats


def _fixture() -> tuple[SyntheticModeComparisonConfig, SyntheticModeComparisonResult]:
    config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=(9, 9, 9),
    )
    clock_values = iter(float(value) for value in range(8))

    class FakeEvaluation:
        pass

    def fake_runner(plan, trial, *, clock, runtime_recorder):
        evaluation = FakeEvaluation()
        evaluation.trial = trial
        evaluation.report_payload = {
            cell.label: {"cell_label": cell.label, "score": 1.0} for cell in plan.cells
        }
        evaluation.stage_cache_stats = PipelineStageCacheStats(*(0,) * 8)
        runtime_recorder.record(
            stage="case_generation",
            elapsed_seconds=0.5,
            shared_stage=True,
        )
        return evaluation

    base = run_mode_comparison(
        config,
        clock=lambda: next(clock_values),
        trial_runner=fake_runner,
        metric_extractor=lambda evaluation: (),
    )
    metric = MetricRow(
        schema_version=1,
        case_id="single_vertical_plane",
        trial_id="single_vertical_plane",
        seed=None,
        scope="scanner-only",
        cell_label="RL-SCAN",
        input_mode="scanner",
        scanner_backend="reference-like",
        scanner_refinement_factor=None,
        scanner_thin_mode="reference",
        workflow_mode=None,
        voter_thin_mode=None,
        skinner_method=None,
        variant="current_default",
        stage="scanner_raw",
        selection="all",
        metric="array_nonzero_fraction",
        value=1.25,
        unit="fraction",
        direction="higher",
        contrast_eligible=True,
    )
    contrast = ContrastRow(
        contrast_name="scanner_only_effect",
        case_id="single_vertical_plane",
        trial_id="single_vertical_plane",
        seed=None,
        stage="scanner_raw",
        selection="all",
        metric="array_nonzero_fraction",
        unit="fraction",
        direction="higher",
        component_cells=("Q-SCAN", "RL-SCAN"),
        raw_value=0.25,
        improvement_value=0.25,
    )
    metric_aggregate = AggregateRow(
        source="metric",
        case_id="single_vertical_plane",
        cell_label="RL-SCAN",
        contrast_name=None,
        stage="scanner_raw",
        selection="all",
        metric="array_nonzero_fraction",
        unit="fraction",
        direction="higher",
        n=1,
        mean=1.25,
        median=1.25,
        std=0.0,
        min=1.25,
        max=1.25,
        q25=1.25,
        q75=1.25,
    )
    contrast_aggregate = AggregateRow(
        source="contrast",
        case_id="single_vertical_plane",
        cell_label=None,
        contrast_name="scanner_only_effect",
        stage="scanner_raw",
        selection="all",
        metric="array_nonzero_fraction",
        unit="fraction",
        direction="higher",
        n=1,
        mean=0.25,
        median=0.25,
        std=0.0,
        min=0.25,
        max=0.25,
        q25=0.25,
        q75=0.25,
    )
    result = SyntheticModeComparisonResult(
        plan_metadata=base.plan_metadata,
        trial_metadata=base.trial_metadata,
        cell_reports=base.cell_reports,
        metric_rows=(metric,),
        contrast_rows=(contrast,),
        metric_aggregates=(metric_aggregate,),
        contrast_aggregates=(contrast_aggregate,),
        cache_stats=base.cache_stats,
        runtime_rows=(
            RuntimeRow(
                case_id=None,
                trial_id=None,
                seed=None,
                stage="experiment_total",
                cell_label=None,
                scanner_backend=None,
                elapsed_seconds=7.0,
                call_count=1,
                shared_stage=True,
            ),
        ),
    )
    return config, result


def _write_bundle(path: Path, *, pretty: bool = False) -> Path:
    config, result = _fixture()
    return write_artifact_bundle(result, path, config=config, pretty=pretty)


def _rehash(bundle: Path, filename: str) -> None:
    completion_path = bundle / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    payload = (bundle / filename).read_bytes()
    completion["files"][filename] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    completion_path.write_text(
        json.dumps(completion, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_writer_creates_complete_valid_bundle_with_stable_headers(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")

    assert {path.name for path in bundle.iterdir()} == set(REQUIRED_BUNDLE_FILES)
    assert validate_completed_bundle(bundle)
    expected_headers = {
        "metrics_long.csv": MetricRow,
        "metric_aggregates.csv": AggregateRow,
        "contrasts.csv": ContrastRow,
        "contrast_aggregates.csv": AggregateRow,
        "runtime.csv": RuntimeRow,
    }
    for filename, model in expected_headers.items():
        with (bundle / filename).open(encoding="utf-8", newline="") as stream:
            assert next(csv.reader(stream)) == [field.name for field in fields(model)]

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_config"]["case_set"] is None
    assert manifest["input_config"]["case_ids"] == ["single_vertical_plane"]
    assert manifest["resolved_plan"]["shape"] == [9, 9, 9]
    assert [cell["label"] for cell in manifest["canonical_cells"]] == [
        "RL-SCAN",
        "Q-SCAN",
        "ORACLE-REF",
        "ORACLE-QUAL",
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    ]
    assert manifest["trials"] == [
        {
            "order": 0,
            "case_id": "single_vertical_plane",
            "trial_id": "single_vertical_plane",
            "stochastic": False,
            "case_generation_seed": None,
            "scanner_input_seed": 20260706,
        }
    ]
    reports = json.loads((bundle / "cell_reports.json").read_text(encoding="utf-8"))
    assert reports[0]["trial_id"] == "single_vertical_plane"
    assert list(reports[0]["cells"]) == [cell["label"] for cell in manifest["canonical_cells"]]
    completion = json.loads((bundle / "completion.json").read_text(encoding="utf-8"))
    assert completion["required_files"] == list(REQUIRED_BUNDLE_FILES)
    for filename in HASHED_BUNDLE_FILES:
        payload = (bundle / filename).read_bytes()
        assert completion["files"][filename] == {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }


def test_default_manifest_records_only_minimal_case_set(tmp_path: Path) -> None:
    _, result = _fixture()
    bundle = write_artifact_bundle(
        result,
        tmp_path / "bundle",
        config=SyntheticModeComparisonConfig(shape=(9, 9, 9)),
    )

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_config"]["case_set"] == "minimal"
    assert manifest["input_config"]["case_ids"] is None


def test_same_result_and_pretty_setting_writes_identical_hashed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = {
        "status": "available",
        "method": "git_cli",
        "commit": "1" * 40,
        "dirty": False,
    }
    monkeypatch.setattr(artifacts, "_source_provenance", lambda: provenance)
    config, result = _fixture()

    first = write_artifact_bundle(result, tmp_path / "first", config=config, pretty=True)
    second = write_artifact_bundle(result, tmp_path / "second", config=config, pretty=True)

    assert all(
        (first / name).read_bytes() == (second / name).read_bytes() for name in HASHED_BUNDLE_FILES
    )


def test_pretty_only_changes_json_whitespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provenance = {
        "status": "available",
        "method": "git_cli",
        "commit": "1" * 40,
        "dirty": False,
    }
    monkeypatch.setattr(artifacts, "_source_provenance", lambda: provenance)
    config, result = _fixture()

    compact = write_artifact_bundle(result, tmp_path / "compact", config=config)
    pretty = write_artifact_bundle(
        result,
        tmp_path / "pretty",
        config=config,
        pretty=True,
    )

    for filename in (
        "metrics_long.csv",
        "metric_aggregates.csv",
        "contrasts.csv",
        "contrast_aggregates.csv",
        "runtime.csv",
    ):
        assert (compact / filename).read_bytes() == (pretty / filename).read_bytes()


def test_csv_rows_preserve_result_order(tmp_path: Path) -> None:
    config, result = _fixture()
    result = replace(
        result,
        metric_rows=(
            replace(result.metric_rows[0], selection="z-first"),
            replace(result.metric_rows[0], selection="a-second"),
        ),
        metric_aggregates=(
            replace(result.metric_aggregates[0], selection="z-first"),
            replace(result.metric_aggregates[0], selection="a-second"),
        ),
        contrast_rows=(
            replace(result.contrast_rows[0], selection="z-first"),
            replace(result.contrast_rows[0], selection="a-second"),
        ),
        contrast_aggregates=(
            replace(result.contrast_aggregates[0], selection="z-first"),
            replace(result.contrast_aggregates[0], selection="a-second"),
        ),
        runtime_rows=(
            replace(result.runtime_rows[0], stage="z-first"),
            replace(result.runtime_rows[0], stage="a-second"),
        ),
    )

    bundle = write_artifact_bundle(result, tmp_path / "bundle", config=config)
    ordered_fields = {
        "metrics_long.csv": ("selection", ["z-first", "a-second"]),
        "metric_aggregates.csv": ("selection", ["z-first", "a-second"]),
        "contrasts.csv": ("selection", ["z-first", "a-second"]),
        "contrast_aggregates.csv": ("selection", ["z-first", "a-second"]),
        "runtime.csv": ("stage", ["z-first", "a-second"]),
    }
    for filename, (field, expected) in ordered_fields.items():
        with (bundle / filename).open(encoding="utf-8", newline="") as stream:
            assert [row[field] for row in csv.DictReader(stream)] == expected


def test_existing_final_path_is_not_modified(tmp_path: Path) -> None:
    final = tmp_path / "bundle"
    final.mkdir()
    sentinel = final / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    config, result = _fixture()

    with pytest.raises(FileExistsError):
        write_artifact_bundle(result, final, config=config)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(final.iterdir()) == [sentinel]


def test_writer_rejects_result_from_a_different_config(tmp_path: Path) -> None:
    _, result = _fixture()
    different_config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=(11, 11, 11),
    )
    output = tmp_path / "bundle"

    with pytest.raises(ValueError, match="result plan metadata does not match config"):
        write_artifact_bundle(result, output, config=different_config)

    assert not output.exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_final_path_created_during_finalize_is_not_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "bundle"
    config, result = _fixture()
    original_rename_noreplace = artifacts._rename_noreplace

    def create_destination_then_rename(temporary: Path, destination: Path) -> None:
        destination.mkdir()
        original_rename_noreplace(temporary, destination)

    monkeypatch.setattr(artifacts, "_rename_noreplace", create_destination_then_rename)
    with pytest.raises(FileExistsError):
        write_artifact_bundle(result, final, config=config)

    assert final.is_dir()
    assert not list(final.iterdir())
    assert not list(tmp_path.glob(".bundle.tmp-*"))


@pytest.mark.parametrize("failed_write", (0, 3, 7))
def test_write_failures_remove_temporary_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_write: int
) -> None:
    config, result = _fixture()
    original = artifacts._write_bytes
    calls = 0

    def failing_write(path, payload):
        nonlocal calls
        if calls == failed_write:
            raise OSError("injected write failure")
        calls += 1
        original(path, payload)

    monkeypatch.setattr(artifacts, "_write_bytes", failing_write)
    with pytest.raises(OSError, match="injected write failure"):
        write_artifact_bundle(result, tmp_path / "bundle", config=config)

    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


@pytest.mark.parametrize(
    ("target", "message"),
    (
        ("_json_bytes", "serialization failed"),
        ("_file_metadata", "hashing failed"),
    ),
)
def test_serialization_and_hashing_failures_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    message: str,
) -> None:
    config, result = _fixture()
    monkeypatch.setattr(
        artifacts,
        target,
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError(message)),
    )

    with pytest.raises(OSError, match=message):
        write_artifact_bundle(result, tmp_path / "bundle", config=config)

    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_file_fsync_failure_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, result = _fixture()
    monkeypatch.setattr(
        artifacts.os,
        "fsync",
        lambda descriptor: (_ for _ in ()).throw(OSError("fsync failed")),
    )

    with pytest.raises(OSError, match="fsync failed"):
        write_artifact_bundle(result, tmp_path / "bundle", config=config)

    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_pre_finalize_failure_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, result = _fixture()
    monkeypatch.setattr(
        artifacts,
        "_fsync_directory",
        lambda path: (_ for _ in ()).throw(OSError("pre-finalize failed")),
    )

    with pytest.raises(OSError, match="pre-finalize failed"):
        write_artifact_bundle(result, tmp_path / "bundle", config=config)

    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_finalize_failure_cleans_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, result = _fixture()
    monkeypatch.setattr(
        artifacts,
        "_finalize_bundle",
        lambda temporary, final: (_ for _ in ()).throw(OSError("finalize failed")),
    )
    with pytest.raises(OSError, match="finalize failed"):
        write_artifact_bundle(result, tmp_path / "bundle", config=config)
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.tmp-*"))


def test_validator_rejects_missing_unexpected_and_changed_files(tmp_path: Path) -> None:
    missing_completion = _write_bundle(tmp_path / "missing-completion")
    (missing_completion / "completion.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        validate_completed_bundle(missing_completion)

    unexpected = _write_bundle(tmp_path / "unexpected")
    (unexpected / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        validate_completed_bundle(unexpected)

    changed = _write_bundle(tmp_path / "changed")
    with (changed / "manifest.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_completed_bundle(changed)

    same_size_change = _write_bundle(tmp_path / "same-size-change")
    manifest_path = same_size_change / "manifest.json"
    payload = bytearray(manifest_path.read_bytes())
    payload[0] = ord("[")
    manifest_path.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_completed_bundle(same_size_change)


def test_validator_rejects_malformed_and_nonfinite_content_after_valid_hash(
    tmp_path: Path,
) -> None:
    malformed = _write_bundle(tmp_path / "malformed")
    (malformed / "cell_reports.json").write_text("{", encoding="utf-8")
    _rehash(malformed, "cell_reports.json")
    with pytest.raises(ValueError, match="malformed JSON"):
        validate_completed_bundle(malformed)

    malformed_csv = _write_bundle(tmp_path / "malformed-csv")
    metrics_path = malformed_csv / "metrics_long.csv"
    metrics_path.write_text('"unterminated\n', encoding="utf-8", newline="\n")
    _rehash(malformed_csv, "metrics_long.csv")
    with pytest.raises(ValueError, match="malformed CSV"):
        validate_completed_bundle(malformed_csv)

    nonfinite_json = _write_bundle(tmp_path / "nonfinite-json")
    manifest_path = nonfinite_json / "manifest.json"
    manifest_text = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        manifest_text.replace("{", '{"bad":NaN,', 1),
        encoding="utf-8",
    )
    _rehash(nonfinite_json, "manifest.json")
    with pytest.raises(ValueError, match="non-finite JSON"):
        validate_completed_bundle(nonfinite_json)

    nonfinite_csv = _write_bundle(tmp_path / "nonfinite-csv")
    metrics_path = nonfinite_csv / "metrics_long.csv"
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    rows[1][[field.name for field in fields(MetricRow)].index("value")] = "Infinity"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    _rehash(nonfinite_csv, "metrics_long.csv")
    with pytest.raises(ValueError, match="non-finite number"):
        validate_completed_bundle(nonfinite_csv)


def test_validator_rejects_incompatible_metric_schema_after_valid_hash(
    tmp_path: Path,
) -> None:
    incompatible_manifest = _write_bundle(tmp_path / "incompatible-manifest")
    manifest_path = incompatible_manifest / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metric_schema_version"] = METRIC_SCHEMA_VERSION + 1
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _rehash(incompatible_manifest, "manifest.json")
    with pytest.raises(ValueError, match="unsupported metric schema version"):
        validate_completed_bundle(incompatible_manifest)

    incompatible_rows = _write_bundle(tmp_path / "incompatible-rows")
    metrics_path = incompatible_rows / "metrics_long.csv"
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    rows[1][[field.name for field in fields(MetricRow)].index("schema_version")] = str(
        METRIC_SCHEMA_VERSION + 1
    )
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)
    _rehash(incompatible_rows, "metrics_long.csv")
    with pytest.raises(ValueError, match="unsupported metric schema version"):
        validate_completed_bundle(incompatible_rows)


def test_git_failure_records_explicit_unavailable_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_git(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(subprocess, "run", fail_git)
    bundle = _write_bundle(tmp_path / "bundle")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["source_provenance"] == {
        "status": "not_available",
        "method": "git_cli",
        "commit": None,
        "dirty": None,
    }
    assert validate_completed_bundle(bundle)


def test_source_provenance_uses_package_location_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_file = Path(artifacts.__file__).resolve()
    source_path = source_file.parent
    source_root = source_path.parents[3]
    source_relative = source_file.relative_to(source_root).as_posix()
    calls: list[list[str]] = []

    def fake_git(command, **kwargs):
        calls.append(command)
        if command == [
            "git",
            "-C",
            str(source_path),
            "rev-parse",
            "--show-toplevel",
        ]:
            return subprocess.CompletedProcess(command, 0, f"{source_root}\n", "")
        if command == [
            "git",
            "-C",
            str(source_root),
            "ls-files",
            "--error-unmatch",
            "--",
            source_relative,
        ]:
            return subprocess.CompletedProcess(command, 0, f"{source_relative}\n", "")
        if command == ["git", "-C", str(source_root), "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(command, 0, f"{'1' * 40}\n", "")
        if command == ["git", "-C", str(source_root), "status", "--porcelain"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", fake_git)

    assert artifacts._source_provenance() == {
        "status": "available",
        "method": "git_cli",
        "commit": "1" * 40,
        "dirty": False,
    }
    assert calls[0][2] == str(source_path)


def test_source_provenance_rejects_untracked_package_in_enclosing_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "consumer"
    source_file = repository / ".venv/site-packages/pyosv/evaluation/artifacts.py"
    source_relative = source_file.relative_to(repository).as_posix()
    calls: list[list[str]] = []

    def fake_git(command, **kwargs):
        calls.append(command)
        if command == [
            "git",
            "-C",
            str(source_file.parent),
            "rev-parse",
            "--show-toplevel",
        ]:
            return subprocess.CompletedProcess(command, 0, f"{repository}\n", "")
        if command == [
            "git",
            "-C",
            str(repository),
            "ls-files",
            "--error-unmatch",
            "--",
            source_relative,
        ]:
            raise subprocess.CalledProcessError(1, command)
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(artifacts, "__file__", str(source_file))
    monkeypatch.setattr(subprocess, "run", fake_git)

    assert artifacts._source_provenance() == {
        "status": "not_available",
        "method": "git_cli",
        "commit": None,
        "dirty": None,
    }
    assert len(calls) == 2


def test_writer_and_validator_leave_no_open_handles(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path / "bundle")
    validate_completed_bundle(bundle)

    shutil.rmtree(bundle)

    assert not bundle.exists()
