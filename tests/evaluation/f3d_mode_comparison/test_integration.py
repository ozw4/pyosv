from __future__ import annotations

import csv
import json
import shutil
import weakref
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

import pyosv.evaluation.f3d_mode_comparison.artifacts as artifacts_module
import pyosv.evaluation.f3d_mode_comparison.builder as builder_module
import pyosv.evaluation.f3d_mode_comparison.data as data_module
import pyosv.evaluation.f3d_mode_comparison.result as result_module
import pyosv.evaluation.f3d_mode_comparison.runner as runner_module
from pyosv.evaluation.f3d_mode_comparison import (
    F3ArtifactError,
    F3DatasetSpec,
    F3_METRIC_ROW_FIELDS,
    F3ModeComparisonConfig,
    F3ModeComparisonPlan,
    F3ModeComparisonResult,
    F3ResultValidationError,
    F3ScannerConfig,
    F3VolumeSource,
    F3VotingControls,
    F3WorkspaceMismatchError,
    PeakRSSRecorder,
    artifact_file_metadata,
    canonical_json_bytes,
    extract_f3d_diagnostics,
    extract_f3d_metrics,
    extract_f3d_resources,
    finalize_f3d_bundle,
    implementation_identity,
    load_f3d_mode_comparison_result,
    numerical_runtime_identity,
    prepare_run_workspace,
    run_f3d_mode_comparison_cells,
    run_scanner_stages,
    scanner_sampling_evidence,
    validate_completed_f3d_bundle,
)

_SHAPE = (3, 4, 5)
_ROLES = (
    ("input", "ep.dat"),
    ("reference_fault_likelihood", "fl.dat"),
    ("reference_fault_votes", "fv.dat"),
    ("reference_thinned_fault_votes", "fvt.dat"),
)


class _DeterministicScanner:
    """Small numerical backend; this fixture is not an F3 publication result."""

    def __init__(self, sigma1: float, sigma2: float, calls: Counter[str]) -> None:
        del sigma1, sigma2
        self._calls = calls

    def scan(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, ...]:
        self._calls["reference-like scan"] += 1
        image = np.asarray(args[4], dtype=np.float32)
        return self._attributes(image, scale=np.float32(0.80))

    def scan_quality(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, ...]:
        self._calls["quality scan"] += 1
        image = np.asarray(args[4], dtype=np.float32)
        ft, pt, tt = self._attributes(image, scale=np.float32(0.90))
        confidence = np.full(image.shape, 0.75, dtype=np.float32)
        return ft, pt, tt, confidence

    def thin(
        self,
        ft: np.ndarray,
        pt: np.ndarray,
        tt: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, ...]:
        del kwargs
        self._calls["scanner thinning"] += 1
        return ft.copy(), pt.copy(), tt.copy()

    def reference_like_strike_sampling(self, *args: Any) -> np.ndarray:
        del args
        return np.asarray([20.0], dtype=np.float32)

    def reference_like_dip_sampling(self, *args: Any) -> np.ndarray:
        del args
        return np.asarray([70.0], dtype=np.float32)

    def refined_reference_like_strike_sampling(self, *args: Any, **kwargs: Any) -> np.ndarray:
        del args, kwargs
        return np.asarray([20.0, 21.0], dtype=np.float32)

    def refined_reference_like_dip_sampling(self, *args: Any, **kwargs: Any) -> np.ndarray:
        del args, kwargs
        return np.asarray([70.0, 71.0], dtype=np.float32)

    @staticmethod
    def _attributes(
        image: np.ndarray,
        *,
        scale: np.float32,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        maximum = np.max(image)
        normalized = image / maximum if maximum > 0.0 else image
        ft = np.asarray(normalized * scale, dtype=np.float32)
        pt = np.full(image.shape, 20.0, dtype=np.float32)
        tt = np.full(image.shape, 70.0, dtype=np.float32)
        return ft, pt, tt


def _fixture_spec(shape: tuple[int, int, int] = _SHAPE) -> F3DatasetSpec:
    return F3DatasetSpec(
        dataset_id="small-external-style-fixture",
        shape=shape,
        files=_ROLES,
        expected_bytes=int(np.prod(shape)) * np.dtype(">f4").itemsize,
    )


def _write_fixture(
    root: Path,
    shape: tuple[int, int, int] = _SHAPE,
) -> F3DatasetSpec:
    root.mkdir()
    spec = _fixture_spec(shape)
    base = np.linspace(0.05, 1.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
    for offset, (_, filename) in enumerate(_ROLES):
        values = np.clip(base + np.float32(offset * 0.01), 0.0, 1.0)
        values.astype(">f4").tofile(root / filename)
    return spec


def _fixture_plan(
    spec: F3DatasetSpec,
    config: F3ModeComparisonConfig | None = None,
) -> F3ModeComparisonPlan:
    """Build a supported non-public plan with an injected fixture dataset."""

    fixture_config = replace(
        config
        or F3ModeComparisonConfig(
            boundary_diagnostic_margin=0,
            skinning_enabled=True,
        ),
        shape=spec.shape,
    )
    return builder_module._build_f3d_mode_comparison_plan(fixture_config, spec)


def _run_fixture(
    data_root: Path,
    output_root: Path,
    spec: F3DatasetSpec,
    calls: Counter[str],
    *,
    resume: bool,
    monkeypatch: pytest.MonkeyPatch,
    plan_config: F3ModeComparisonConfig | None = None,
    workspace_implementation: Mapping[str, Any] | None = None,
    workspace_runtime_identity: Mapping[str, Any] | None = None,
    scanner_implementation_identity: str = "small-fixture-scanner-v1",
    workflow_implementation_identity: str = "small-fixture-workflow-v1",
    scanner_factory: Any = None,
    workflow_runner: Any = None,
) -> F3ModeComparisonResult:
    plan = _fixture_plan(spec, plan_config)
    with F3VolumeSource(data_root, spec=spec) as source:
        workspace = prepare_run_workspace(
            output_root,
            plan,
            source.identity,
            resume=resume,
            implementation=workspace_implementation,
            runtime_identity=workspace_runtime_identity,
        )
        if (workspace.path / "completion.json").exists():
            calls["complete result load"] += 1
            return load_f3d_mode_comparison_result(
                workspace.path,
                deep=True,
                _dataset_spec=spec,
            )

        rss = PeakRSSRecorder(lambda: 0, source="fixture", semantics="fixture")
        resolved_scanner_factory = scanner_factory or (
            lambda sigma1, sigma2: _DeterministicScanner(  # noqa: E731
                sigma1, sigma2, calls
            )
        )
        sampling_provider = _DeterministicScanner(0.0, 0.0, Counter())
        sampling_evidence_by_backend = {
            backend: scanner_sampling_evidence(
                sampling_provider,
                plan.scanner_config_for(backend),
                backend,
                implementation_identity=scanner_implementation_identity,
            )
            for backend in ("reference-like", "quality")
        }
        scanners = run_scanner_stages(
            workspace,
            source,
            plan,
            scanner_factory=resolved_scanner_factory,
            implementation_identity=scanner_implementation_identity,
            sampling_evidence_by_backend=sampling_evidence_by_backend,
            rss_recorder=rss,
        )

        def counted_workflow_runner(**kwargs: Any) -> Any:
            calls["workflow callback"] += 1
            if workflow_runner is not None:
                return workflow_runner(**kwargs)

            from pyosv.evaluation.workflow3d import execute_workflow3d

            return execute_workflow3d(**kwargs)

        cells = run_f3d_mode_comparison_cells(
            workspace,
            plan,
            scanners,
            workflow_runner=counted_workflow_runner,
            workflow_implementation_identity=workflow_implementation_identity,
            rss_recorder=rss,
        )
        calls["metric callback"] += 1
        metrics = extract_f3d_metrics(source, cells.cells, slab_depth=1)
        diagnostics = extract_f3d_diagnostics(source, cells.cells, plan)
        rss.process_peak()
        resources = extract_f3d_resources(
            cells.stage_runtime,
            shape=spec.shape,
            workspace=workspace,
            scanner_stages=scanners,
            rss_recorder=rss,
        )
        result = F3ModeComparisonResult.from_extractions(
            workspace=workspace,
            cells=cells.cells,
            metrics=metrics,
            diagnostics=diagnostics,
            resources=resources,
            _dataset_spec=spec,
        )
        return finalize_f3d_bundle(
            workspace,
            result,
            resume=resume,
            deep=True,
            _dataset_spec=spec,
        )


def _csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        return reader.fieldnames, list(reader)


def _rehash_report(output_root: Path, filename: str) -> None:
    completion_path = output_root / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["report_files"][filename] = artifact_file_metadata(
        output_root / "reports" / filename
    )
    completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")


def test_small_external_style_fixture_end_to_end_and_complete_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "fixture-data"
    output_root = tmp_path / "run"
    spec = _write_fixture(data_root)
    calls: Counter[str] = Counter()
    io_calls: Counter[str] = Counter()
    stream_sha256 = data_module._stream_sha256
    read_native_volume = F3VolumeSource.read_native_volume

    def counted_sha256(*args: Any, **kwargs: Any) -> str:
        io_calls["identity hash"] += 1
        return stream_sha256(*args, **kwargs)

    def counted_native_read(self: F3VolumeSource, role: str) -> np.ndarray:
        io_calls[f"native read:{role}"] += 1
        return read_native_volume(self, role)

    monkeypatch.setattr(data_module, "_stream_sha256", counted_sha256)
    monkeypatch.setattr(F3VolumeSource, "read_native_volume", counted_native_read)
    injected_runtime = numerical_runtime_identity()

    first = _run_fixture(
        data_root,
        output_root,
        spec,
        calls,
        resume=False,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=injected_runtime,
    )

    assert [cell.label for cell in first.cells] == [
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    ]
    assert calls["reference-like scan"] == 1
    assert calls["quality scan"] == 1
    assert calls["scanner thinning"] == 2
    assert calls["workflow callback"] == 4
    assert calls["metric callback"] == 1
    assert io_calls == Counter({"identity hash": 4, "native read:input": 1})
    assert len({cell.stages.scanner for cell in first.cells}) == 2
    scanner_reports = {
        cell.backend: json.loads(
            (output_root / "stages" / "scanner" / cell.stages.scanner / "report.json").read_text()
        )
        for cell in first.cells
    }
    assert scanner_reports["reference-like"]["sampling_count"] == {
        "strike": 1,
        "dip": 1,
        "orientations": 1,
    }
    assert scanner_reports["quality"]["sampling_count"] == {
        "strike": 2,
        "dip": 2,
        "orientations": 4,
    }
    assert all(
        report["sampling_evidence"] == report["resolved_stage_settings"]["sampling_evidence"]
        for report in scanner_reports.values()
    )
    assert len({cell.stages.voting for cell in first.cells}) == 2
    assert len({cell.stages.thinning for cell in first.cells}) == 4
    assert len({cell.stages.skinning for cell in first.cells}) == 4
    assert len(first.runtime_rows) == 14
    assert len(first.rss_snapshots) == 53
    assert len(first.storage_rows) == 13
    assert sum(path.is_file() for path in output_root.rglob("*")) == 82
    assert len(list((output_root / "reports").iterdir())) == 9

    for cell in first.cells:
        stages = cell.stages.as_dict()
        cell_payload = json.loads((output_root / "cells" / f"{cell.label}.json").read_text())
        assert cell_payload["stages"] == stages
        expected_parent: str | None = None
        for kind in ("scanner", "voting", "thinning", "skinning"):
            fingerprint = stages[kind]
            stage_manifest = json.loads(
                (output_root / "stages" / kind / fingerprint / "stage_manifest.json").read_text()
            )
            assert stage_manifest["fingerprint"] == fingerprint
            assert stage_manifest["parent_fingerprints"] == (
                [] if expected_parent is None else [expected_parent]
            )
            expected_parent = fingerprint

    manifest = json.loads((output_root / "run_manifest.json").read_text())
    assert manifest["run_fingerprint"] == first.run_fingerprint
    assert manifest["plan"]["dataset_spec"]["dataset_id"] == spec.dataset_id
    assert sorted(path.name for path in output_root.iterdir()) == [
        "cells",
        "completion.json",
        "reports",
        "run_manifest.json",
        "stages",
    ]
    assert validate_completed_f3d_bundle(output_root, _dataset_spec=spec)
    assert validate_completed_f3d_bundle(
        output_root,
        deep=True,
        _dataset_spec=spec,
    )

    cell_report = json.loads((output_root / "reports" / "cells.json").read_text())
    assert [row["label"] for row in cell_report["cells"]] == [
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    ]
    metric_header, metric_rows = _csv(output_root / "reports" / "metrics_long.csv")
    assert metric_header == list(F3_METRIC_ROW_FIELDS)
    assert len(metric_rows) == 932
    assert [
        (row["cell_label"], row["stage"], row["selection"], row["metric"]) for row in metric_rows
    ] == [
        (cell.label, definition.stage, definition.selection, definition.metric)
        for cell in first.cells
        for definition in result_module.METRIC_REGISTRY
    ]
    _, voxel_rows = _csv(output_root / "reports" / "voxel_contrast_summaries.csv")
    assert [(row["stage"], row["contrast_name"]) for row in voxel_rows] == [
        (stage, definition.name)
        for stage in result_module.F3_REFERENCE_STAGE_FILES
        for definition in result_module.CONTRAST_DEFINITIONS
    ]
    _, regional_rows = _csv(output_root / "reports" / "regional_metrics.csv")
    assert [(row["cell_label"], row["stage"], row["region"]) for row in regional_rows] == [
        (cell.label, stage, region)
        for stage in result_module.F3_REFERENCE_STAGE_FILES
        for cell in first.cells
        for region in result_module.F3_DIAGNOSTIC_REGIONS
    ]
    _, orientation_rows = _csv(output_root / "reports" / "orientation_diagnostics.csv")
    assert [(row["stage"], row["left_cell"], row["right_cell"]) for row in orientation_rows] == [
        (stage, left, right)
        for stage in ("scanner", "voting")
        for left, right in result_module.F3_ORIENTATION_PAIRS
    ]
    _, runtime_rows = _csv(output_root / "reports" / "runtime.csv")
    assert len(runtime_rows) == 14
    assert Counter(row["state"] for row in runtime_rows) == Counter({"computed": 12, "reused": 2})
    assert {row["stage_kind"] for row in runtime_rows} == {
        "scanner",
        "skinning",
        "voting",
        "thinning",
    }
    expected_runtime_order = [
        ("scanner", fingerprint, consumers[0])
        for fingerprint in dict.fromkeys(cell.stages.scanner for cell in first.cells)
        if (
            consumers := tuple(
                cell.label for cell in first.cells if cell.stages.scanner == fingerprint
            )
        )
    ]
    expected_runtime_order.extend(
        (kind, getattr(cell.stages, kind), cell.label)
        for cell in first.cells
        for kind in ("voting", "thinning", "skinning")
    )
    assert [
        (row["stage_kind"], row["fingerprint"], row["cell"]) for row in runtime_rows
    ] == expected_runtime_order

    before_resume = calls.copy()
    second = _run_fixture(
        data_root,
        output_root,
        spec,
        calls,
        resume=True,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=injected_runtime,
    )
    assert second == first
    assert calls - before_resume == Counter({"complete result load": 1})
    assert io_calls == Counter({"identity hash": 8, "native read:input": 1})

    forbidden = ("crop_center", "crop_shape", "tile_sample", "replicate_index")
    for path in output_root.rglob("*"):
        if path.is_file():
            payload = path.read_bytes()
            assert all(term.encode() not in payload for term in forbidden)


@pytest.mark.parametrize("missing_kind", ["voting", "thinning", "reports"])
def test_incomplete_resume_reuses_independent_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing_kind: str,
) -> None:
    data_root = tmp_path / "fixture-data"
    output_root = tmp_path / "run"
    spec = _write_fixture(data_root)
    first_calls: Counter[str] = Counter()
    injected_runtime = numerical_runtime_identity()
    first = _run_fixture(
        data_root,
        output_root,
        spec,
        first_calls,
        resume=False,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=injected_runtime,
    )
    untouched = output_root / "stages" / "thinning" / first.cells[-1].stages.thinning
    untouched_completion = (untouched / "complete.json").read_bytes()
    (output_root / "completion.json").unlink()
    if missing_kind in {"voting", "thinning"}:
        fingerprint = getattr(first.cells[0].stages, missing_kind)
        shutil.rmtree(output_root / "stages" / missing_kind / fingerprint)

    resume_calls: Counter[str] = Counter()
    resumed = _run_fixture(
        data_root,
        output_root,
        spec,
        resume_calls,
        resume=True,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=injected_runtime,
    )

    assert resumed.run_fingerprint == first.run_fingerprint
    assert resume_calls["reference-like scan"] == 0
    assert resume_calls["quality scan"] == 0
    assert resume_calls["scanner thinning"] == 0
    expected_workflows = {"voting": 2, "thinning": 1, "reports": 0}
    assert resume_calls["workflow callback"] == expected_workflows[missing_kind]
    assert resume_calls["metric callback"] == 1
    assert (untouched / "complete.json").read_bytes() == untouched_completion
    assert validate_completed_f3d_bundle(
        output_root,
        deep=True,
        _dataset_spec=spec,
    )


@pytest.mark.parametrize("filename", ["ep.dat", "fl.dat"])
def test_source_change_invalidates_resume_without_overwriting_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    data_root = tmp_path / "fixture-data"
    output_root = tmp_path / "run"
    spec = _write_fixture(data_root)
    first = _run_fixture(
        data_root,
        output_root,
        spec,
        Counter(),
        resume=False,
        monkeypatch=monkeypatch,
    )
    manifest_before = (output_root / "run_manifest.json").read_bytes()
    stage_completion = (
        output_root / "stages" / "thinning" / first.cells[-1].stages.thinning / "complete.json"
    )
    stage_before = stage_completion.read_bytes()
    changed = data_root / filename
    with changed.open("r+b") as stream:
        first_byte = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first_byte[0] ^ 0x01]))

    calls: Counter[str] = Counter()
    with pytest.raises(F3WorkspaceMismatchError, match="dataset_identity"):
        _run_fixture(
            data_root,
            output_root,
            spec,
            calls,
            resume=True,
            monkeypatch=monkeypatch,
        )

    assert not calls
    assert (output_root / "run_manifest.json").read_bytes() == manifest_before
    assert stage_completion.read_bytes() == stage_before


@pytest.mark.parametrize(
    "change",
    ["voting-control", "scanner-control", "implementation-version", "contract-version"],
)
def test_config_and_version_changes_reject_resume_before_artifact_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    data_root = tmp_path / "fixture-data"
    output_root = tmp_path / "run"
    spec = _write_fixture(data_root)
    first = _run_fixture(
        data_root,
        output_root,
        spec,
        Counter(),
        resume=False,
        monkeypatch=monkeypatch,
    )
    manifest_before = (output_root / "run_manifest.json").read_bytes()
    preserved = (
        output_root / "stages" / "thinning" / first.cells[-1].stages.thinning / "complete.json"
    )
    preserved_before = preserved.read_bytes()
    kwargs: dict[str, Any] = {}
    if change == "voting-control":
        kwargs["plan_config"] = F3ModeComparisonConfig(
            boundary_diagnostic_margin=0,
            voting_controls=F3VotingControls(seed_threshold=0.31),
        )
    elif change == "scanner-control":
        kwargs["plan_config"] = F3ModeComparisonConfig(
            boundary_diagnostic_margin=0,
            scanner_template=F3ScannerConfig(sigma1=7.5),
        )
    elif change == "implementation-version":
        changed_identity = implementation_identity()
        changed_identity["software_versions"]["pyosv"] += ".changed"
        kwargs["workspace_implementation"] = changed_identity
    else:
        monkeypatch.setattr(
            artifacts_module,
            "F3_STAGE_CONTRACT_VERSION",
            artifacts_module.F3_STAGE_CONTRACT_VERSION + 1,
        )

    calls: Counter[str] = Counter()
    with pytest.raises(F3WorkspaceMismatchError):
        _run_fixture(
            data_root,
            output_root,
            spec,
            calls,
            resume=True,
            monkeypatch=monkeypatch,
            **kwargs,
        )

    assert not calls
    assert (output_root / "run_manifest.json").read_bytes() == manifest_before
    assert preserved.read_bytes() == preserved_before


@pytest.mark.parametrize("changed_stage", ["scanner", "workflow"])
def test_stage_implementation_change_invalidates_only_dependent_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_stage: str,
) -> None:
    data_root = tmp_path / "fixture-data"
    output_root = tmp_path / "run"
    spec = _write_fixture(data_root)
    first = _run_fixture(
        data_root,
        output_root,
        spec,
        Counter(),
        resume=False,
        monkeypatch=monkeypatch,
    )
    old_stage_completions = {
        path: path.read_bytes() for path in output_root.glob("stages/*/*/complete.json")
    }
    (output_root / "completion.json").unlink()
    for path in (output_root / "cells").glob("*.json"):
        path.unlink()
    calls: Counter[str] = Counter()
    kwargs = (
        {"scanner_implementation_identity": "small-fixture-scanner-v2"}
        if changed_stage == "scanner"
        else {"workflow_implementation_identity": "small-fixture-workflow-v2"}
    )

    second = _run_fixture(
        data_root,
        output_root,
        spec,
        calls,
        resume=True,
        monkeypatch=monkeypatch,
        **kwargs,
    )

    if changed_stage == "scanner":
        assert calls["reference-like scan"] == 1
        assert calls["quality scan"] == 1
        assert all(
            left.stages.scanner != right.stages.scanner
            for left, right in zip(first.cells, second.cells, strict=True)
        )
    else:
        assert calls["reference-like scan"] == 0
        assert calls["quality scan"] == 0
        assert all(
            left.stages.scanner == right.stages.scanner
            for left, right in zip(first.cells, second.cells, strict=True)
        )
    assert calls["workflow callback"] == 4
    assert all(
        left.stages.voting != right.stages.voting
        and left.stages.thinning != right.stages.thinning
        and left.stages.skinning != right.stages.skinning
        for left, right in zip(first.cells, second.cells, strict=True)
    )
    assert all(path.read_bytes() == payload for path, payload in old_stage_completions.items())
    assert validate_completed_f3d_bundle(
        output_root,
        deep=True,
        _dataset_spec=spec,
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "stage-hash",
        "stage-completion",
        "unknown-extra",
        "cell-parent",
        "rehashed-report",
        "root-completion",
    ],
)
def test_corrupt_and_partial_fixture_artifacts_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    data_root = tmp_path / "fixture-data"
    output_root = tmp_path / "run"
    spec = _write_fixture(data_root)
    result = _run_fixture(
        data_root,
        output_root,
        spec,
        Counter(),
        resume=False,
        monkeypatch=monkeypatch,
    )
    stage = output_root / "stages" / "thinning" / result.cells[0].stages.thinning
    if corruption == "stage-hash":
        artifact = stage / "fvt.dat"
        payload = bytearray(artifact.read_bytes())
        payload[0] ^= 0x01
        artifact.write_bytes(payload)
    elif corruption == "stage-completion":
        (stage / "complete.json").unlink()
    elif corruption == "unknown-extra":
        (stage / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    elif corruption == "cell-parent":
        cell_path = output_root / "cells" / "RL-REF.json"
        cell = json.loads(cell_path.read_text(encoding="utf-8"))
        cell["stages"]["voting"] = result.cells[-1].stages.voting
        cell_path.write_bytes(canonical_json_bytes(cell) + b"\n")
    elif corruption == "rehashed-report":
        report = output_root / "reports" / "cells.json"
        payload = json.loads(report.read_text(encoding="utf-8"))
        payload["cells"][0]["stages"]["voting"] = result.cells[-1].stages.voting
        report.write_bytes(canonical_json_bytes(payload) + b"\n")
        _rehash_report(output_root, "cells.json")
    else:
        completion_path = output_root / "completion.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["run_fingerprint"] = "0" * 64
        completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")

    with pytest.raises((F3ArtifactError, F3ResultValidationError, ValueError)):
        validate_completed_f3d_bundle(
            output_root,
            deep=True,
            _dataset_spec=spec,
        )


def test_failure_preserves_stages_and_cleans_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "fixture-data"
    output_root = tmp_path / "run"
    spec = _write_fixture(data_root)
    original_write = result_module.atomic_write_artifact
    original_memmap = np.memmap
    opened_memmaps: list[np.memmap] = []

    class TrackedMemmap(original_memmap):  # type: ignore[misc, valid-type]
        def __new__(cls, *args: Any, **kwargs: Any) -> np.memmap:
            array = super().__new__(cls, *args, **kwargs)
            opened_memmaps.append(array)
            return array

    def fail_report(
        path: Path,
        payload: bytes,
        *,
        temporary_prefix: str,
    ) -> None:
        if Path(path).name == "metrics_long.csv":
            raise OSError("injected report failure")
        original_write(path, payload, temporary_prefix=temporary_prefix)

    monkeypatch.setattr(result_module, "atomic_write_artifact", fail_report)
    monkeypatch.setattr(np, "memmap", TrackedMemmap)
    with pytest.raises(OSError, match="injected report failure"):
        _run_fixture(
            data_root,
            output_root,
            spec,
            Counter(),
            resume=False,
            monkeypatch=monkeypatch,
        )

    assert (output_root / "run_manifest.json").is_file()
    completions = tuple(output_root.glob("stages/*/*/complete.json"))
    assert len(completions) == 12
    assert not (output_root / "completion.json").exists()
    assert opened_memmaps
    assert all(array._mmap.closed for array in opened_memmaps)
    assert not tuple(
        path
        for path in output_root.rglob("*")
        if ".tmp-" in path.name or path.name.startswith(".pyosv-stage-tmp-")
    )


def test_full_fixture_releases_stage_arrays_and_limits_deep_metric_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "fixture-data"
    output_root = tmp_path / "run"
    spec = _write_fixture(data_root)
    calls: Counter[str] = Counter()
    scanner_arrays: list[weakref.ReferenceType[np.ndarray]] = []
    workflow_arrays: list[weakref.ReferenceType[np.ndarray]] = []
    cached_workflow_arrays: list[weakref.ReferenceType[np.ndarray]] = []

    class LifecycleScanner(_DeterministicScanner):
        def scan(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, ...]:
            output = super().scan(*args, **kwargs)
            scanner_arrays.extend(weakref.ref(array) for array in output)
            return output

        def scan_quality(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, ...]:
            assert scanner_arrays
            assert all(reference() is None for reference in scanner_arrays)
            output = super().scan_quality(*args, **kwargs)
            scanner_arrays.extend(weakref.ref(array) for array in output)
            return output

        def thin(
            self,
            ft: np.ndarray,
            pt: np.ndarray,
            tt: np.ndarray,
            **kwargs: Any,
        ) -> tuple[np.ndarray, ...]:
            output = super().thin(ft, pt, tt, **kwargs)
            scanner_arrays.extend(weakref.ref(array) for array in output)
            return output

    original_persist = runner_module._persist_or_reuse_cell_stages
    original_write_reference = runner_module._write_or_reuse_cell_reference

    def tracked_persist(*args: Any, **kwargs: Any) -> Any:
        result = args[3]
        assert result is not None
        workflow_arrays.extend(
            weakref.ref(array)
            for array in (
                result.fv,
                result.vp,
                result.vt,
                result.fvt,
                result.skin.primary_mask,
            )
        )
        return original_persist(*args, **kwargs)

    def tracked_write_reference(*args: Any, **kwargs: Any) -> Any:
        assert workflow_arrays
        assert all(reference() is None for reference in workflow_arrays)
        assert cached_workflow_arrays
        assert all(reference() is None for reference in cached_workflow_arrays)
        return original_write_reference(*args, **kwargs)

    monkeypatch.setattr(runner_module, "_persist_or_reuse_cell_stages", tracked_persist)
    monkeypatch.setattr(runner_module, "_write_or_reuse_cell_reference", tracked_write_reference)
    cleared_caches = []
    original_clear = runner_module.PipelineStageCache.clear

    def tracked_clear(cache: Any) -> None:
        cached_workflow_arrays.extend(
            weakref.ref(array)
            for result in (
                *cache._voting.values(),
                *cache._thinning.values(),
                *cache._final_thinning.values(),
            )
            for array in (
                *((result.fv, result.vp, result.vt) if hasattr(result, "fv") else ()),
                *((result.fvt,) if hasattr(result, "fvt") else ()),
            )
        )
        original_clear(cache)
        assert not (
            cache._seeds
            or cache._voting
            or cache._thinning
            or cache._final_thinning
            or cache._primary_skinning
        )
        cleared_caches.append(cache)

    monkeypatch.setattr(runner_module.PipelineStageCache, "clear", tracked_clear)
    original_memmap = result_module.np.memmap
    original_close = result_module._close_memmap
    original_deep_metrics = result_module._deep_validate_reference_metrics
    active: set[int] = set()
    metric_pair_peak = 0
    metric_passes = 0
    tracking_metrics = False

    class TrackedMemmap(original_memmap):  # type: ignore[misc, valid-type]
        def __new__(cls, *args: Any, **kwargs: Any) -> np.memmap:
            nonlocal metric_pair_peak
            array = super().__new__(cls, *args, **kwargs)
            mode = kwargs.get("mode", args[2] if len(args) > 2 else "r+")
            if tracking_metrics and mode == "r":
                active.add(id(array))
                metric_pair_peak = max(metric_pair_peak, len(active))
            return array

    def tracked_close(array: np.memmap | None) -> None:
        original_close(array)
        if array is not None:
            active.discard(id(array))

    def tracked_deep_metrics(*args: Any, **kwargs: Any) -> None:
        nonlocal metric_passes, tracking_metrics
        metric_passes += 1
        tracking_metrics = True
        try:
            original_deep_metrics(*args, **kwargs)
        finally:
            tracking_metrics = False

    monkeypatch.setattr(result_module.np, "memmap", TrackedMemmap)
    monkeypatch.setattr(result_module, "_close_memmap", tracked_close)
    monkeypatch.setattr(result_module, "_deep_validate_reference_metrics", tracked_deep_metrics)
    result = _run_fixture(
        data_root,
        output_root,
        spec,
        calls,
        resume=False,
        monkeypatch=monkeypatch,
        scanner_factory=lambda sigma1, sigma2: LifecycleScanner(sigma1, sigma2, calls),
    )

    assert scanner_arrays
    assert all(reference() is None for reference in scanner_arrays)
    assert workflow_arrays
    assert all(reference() is None for reference in workflow_arrays)
    assert cached_workflow_arrays
    assert all(reference() is None for reference in cached_workflow_arrays)
    assert len(cleared_caches) == 4
    assert metric_passes == 1
    assert metric_pair_peak == 2
    assert not active
    result_module._reject_volume_bearing_values(result)
