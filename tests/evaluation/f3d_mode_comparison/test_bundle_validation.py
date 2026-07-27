from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

import pyosv.evaluation.f3d_mode_comparison.result as result_module
import pyosv.evaluation.f3d_mode_comparison.runner as runner_module
import pyosv.evaluation.f3d_mode_comparison.scanner as scanner_module
import pyosv.evaluation.workflow3d as workflow3d_module
from pyosv.evaluation.f3d_mode_comparison import (
    F3ModeComparisonConfig,
    F3ModeComparisonResult,
    F3ResultValidationError,
    F3RunWorkspace,
    RSSSnapshot,
    artifact_file_metadata,
    build_f3d_mode_comparison_plan,
    canonical_fingerprint,
    canonical_json_bytes,
    numerical_runtime_identity,
    extract_f3d_diagnostics,
    extract_f3d_metrics,
    extract_stage_resources,
    finalize_f3d_bundle,
    load_f3d_mode_comparison_result,
    run_f3d_mode_comparison_cells,
    scanner_stage_resource_rows,
    storage_report,
    validate_completed_f3d_bundle,
)
from pyosv.evaluation.f3d_mode_comparison.data import F3DatasetSpec, F3VolumeSource
from pyosv.evaluation.f3d_mode_comparison.artifacts import F3ArtifactError
from pyosv.evaluation.synthetic_quality.config import SyntheticSkinningConfig
from pyosv.evaluation.synthetic_quality.variants import (
    SkinningPatch,
    VariantSpec,
)
from pyosv.evaluation.workflow3d import execute_skinning_phase3d

from .test_runner import _LoadedScanner, _scanner_stage


@pytest.fixture(autouse=True)
def _small_official_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the canonical-only validator without allocating the full F3 volume."""

    shape = (3, 4, 5)
    spec = F3DatasetSpec(
        dataset_id="result-fixture",
        shape=shape,
        files=(
            ("input", "ep.dat"),
            ("reference_fault_likelihood", "fl.dat"),
            ("reference_fault_votes", "fv.dat"),
            ("reference_thinned_fault_votes", "fvt.dat"),
        ),
        expected_bytes=int(np.prod(shape)) * np.dtype(">f4").itemsize,
    )
    monkeypatch.setattr(result_module, "OFFICIAL_F3_DATASET_SPEC", spec)


def _complete_small_bundle(
    tmp_path: Path,
    *,
    skinning_enabled: bool = False,
    pretty: bool = False,
    config: F3ModeComparisonConfig | None = None,
    workflow_runner: Callable[..., Any] | None = None,
) -> Path:
    shape = (3, 4, 5)
    roles = (
        ("input", "ep.dat"),
        ("reference_fault_likelihood", "fl.dat"),
        ("reference_fault_votes", "fv.dat"),
        ("reference_thinned_fault_votes", "fvt.dat"),
    )
    data = tmp_path / "data"
    data.mkdir()
    spec = F3DatasetSpec(
        dataset_id="result-fixture",
        shape=shape,
        files=roles,
        expected_bytes=int(np.prod(shape)) * np.dtype(">f4").itemsize,
    )
    for offset, (_, filename) in enumerate(roles):
        values = np.linspace(0.0, 1.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
        (values + np.float32(offset * 0.01)).astype(">f4").tofile(data / filename)
    source = F3VolumeSource(data, spec=spec)

    root = tmp_path / "run"
    for kind in ("scanner", "voting", "thinning", "skinning"):
        (root / "stages" / kind).mkdir(parents=True)
    (root / "cells").mkdir()
    (root / "reports").mkdir()
    plan = build_f3d_mode_comparison_plan(
        config
        or F3ModeComparisonConfig(
            skinning_enabled=skinning_enabled,
            boundary_diagnostic_margin=0,
        )
    )
    plan_payload = plan.as_dict()
    plan_payload["dataset_spec"] = asdict(spec)
    computation = {
        "artifact_schema_version": 1,
        "stage_contract_version": 1,
        "fingerprint_contract_version": 3,
        "plan": plan_payload,
        "dataset_identity": source.identity.computation_identity,
        "implementation_identity": {"name": "test"},
        "runtime_identity": numerical_runtime_identity(),
    }
    fingerprint = canonical_fingerprint(computation)
    manifest = {
        **computation,
        "run_fingerprint": fingerprint,
        "provenance": {
            "dataset_files": [
                {
                    "role": item.role,
                    "filename": item.filename,
                    "resolved_path": str(item.resolved_path),
                }
                for item in source.identity.files
            ]
        },
    }
    (root / "run_manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    workspace = F3RunWorkspace(root, fingerprint, manifest, resumed=False)
    input_fingerprint = source.identity.file_for("input").sha256
    scanners = {
        backend: _scanner_stage(
            workspace,
            backend,
            shape=shape,
            input_fingerprint=input_fingerprint,
        )
        for backend in ("reference-like", "quality")
    }
    runner_kwargs: dict[str, Any] = {}
    if workflow_runner is not None:
        runner_kwargs["workflow_runner"] = workflow_runner
    cell_result = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: _LoadedScanner(stage, []),
        **runner_kwargs,
    )
    metrics = extract_f3d_metrics(source, cell_result.cells, slab_depth=1)
    diagnostics = extract_f3d_diagnostics(
        source,
        cell_result.cells,
        boundary_margin=0,
    )
    runtime = (
        *scanner_stage_resource_rows(scanners),
        *extract_stage_resources(cell_result.stage_runtime, shape=shape),
    )
    stage_keys = tuple(dict.fromkeys((row.stage_kind, row.fingerprint) for row in runtime))
    rss_snapshots = (
        *(
            RSSSnapshot(
                1,
                "stage_snapshot",
                f"{kind}:{stage_fingerprint}:fixture:{boundary}",
                0,
                "available",
                "fixture",
                "fixture",
            )
            for kind, stage_fingerprint in stage_keys
            for boundary in ("before", "after")
        ),
        RSSSnapshot(
            1,
            "process_peak",
            "fixture_end",
            0,
            "available",
            "fixture",
            "fixture",
        ),
    )
    result = F3ModeComparisonResult(
        fingerprint,
        spec.dataset_id,
        shape,
        spec.storage_dtype,
        cell_result.cells,
        metrics.metric_rows,
        metrics.metric_evidence,
        metrics.contrast_rows,
        metrics.voxelwise_contrasts,
        diagnostics.regional_rows,
        diagnostics.orientation_rows,
        tuple(runtime),
        tuple(rss_snapshots),
        storage_report(workspace),
    )
    finalize_f3d_bundle(workspace, result, pretty=pretty)
    source.close()
    return root


def _boundary_skin_config() -> F3ModeComparisonConfig:
    return F3ModeComparisonConfig(
        skinning_enabled=True,
        boundary_diagnostic_margin=0,
        skinning_template=SyntheticSkinningConfig(
            min_likelihood=0.5,
            min_skin_size=1,
            d=1,
            ru=2,
            rv=2,
            rw=2,
            max_steps=2,
            reskin=False,
            accepted_occupancy_radius=0,
            small_skin_size=1,
            boundary_skinner_fallback=True,
        ),
        skinner_method_explicit=True,
        skinner_min_likelihood_explicit=True,
        skinner_growth_source_explicit=True,
        skinner_accepted_occupancy_radius_explicit=True,
        skinner_boundary_fallback_explicit=True,
    )


def _controlled_boundary_skin_workflow(**kwargs: Any) -> Any:
    base = runner_module.execute_workflow3d(**kwargs)
    shape = base.fv.shape
    fvt = np.zeros(shape, dtype=np.float32)
    fvt[0, 0, 0] = np.float32(1.0)
    fvt[0, 0, 1] = np.float32(0.95)
    fvt[-1, -1, -1] = np.float32(0.9)
    fv = fvt.copy()
    vp = np.full(shape, np.float32(20.0), dtype=np.float32)
    vt = np.full(shape, np.float32(70.0), dtype=np.float32)
    skin = execute_skinning_phase3d(
        fv=fv,
        fvt=fvt,
        vp=vp,
        vt=vt,
        skinning_settings=kwargs["skinning_settings"],
        variant_spec=kwargs["variant_spec"],
        scanner_target_positive_mask=kwargs["scanner_target_positive_mask"],
        boundary_fallback_runner=kwargs["boundary_fallback_runner"],
    )
    return replace(
        base,
        fv=fv,
        fvt=fvt,
        vp=vp,
        vt=vt,
        skin=skin,
        diagnostics=replace(base.diagnostics, skinning=skin.diagnostics),
    )


def _complete_boundary_skin_bundle(tmp_path: Path) -> Path:
    return _complete_small_bundle(
        tmp_path,
        config=_boundary_skin_config(),
        workflow_runner=_controlled_boundary_skin_workflow,
    )


def _rehash_report(root: Path, filename: str) -> None:
    completion_path = root / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["report_files"][filename] = artifact_file_metadata(root / "reports" / filename)
    completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")


def _rehash_stage_artifact(root: Path, stage: Path, filename: str) -> None:
    manifest_path = stage / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][filename] = {
        **manifest["files"][filename],
        **artifact_file_metadata(stage / filename),
    }
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    stage_completion_path = stage / "complete.json"
    stage_completion = json.loads(stage_completion_path.read_text(encoding="utf-8"))
    stage_completion["files"][filename] = {
        **stage_completion["files"][filename],
        **artifact_file_metadata(stage / filename),
    }
    stage_completion["files"]["stage_manifest.json"] = artifact_file_metadata(manifest_path)
    stage_completion_path.write_bytes(canonical_json_bytes(stage_completion) + b"\n")

    completion_path = root / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    stage_key = f"{manifest['kind']}/{manifest['fingerprint']}"
    completion["stage_completions"][stage_key] = artifact_file_metadata(stage_completion_path)
    completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames is not None
        return reader.fieldnames, list(reader)


def _write_csv_rows(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_complete_bundle_strict_load_deep_validation_and_resume(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)

    assert validate_completed_f3d_bundle(root)
    assert validate_completed_f3d_bundle(root, deep=True)
    loaded = load_f3d_mode_comparison_result(root)
    resumed = finalize_f3d_bundle(root, resume=True)
    assert resumed == loaded


def test_completed_bundle_metric_artifacts_use_schema_version_2(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    reports = root / "reports"

    for filename in (
        "metrics_long.csv",
        "contrasts.csv",
        "voxel_contrast_summaries.csv",
    ):
        _, rows = _csv_rows(reports / filename)
        assert rows
        assert {row["schema_version"] for row in rows} == {"2"}
    evidence = json.loads((reports / "metric_evidence.json").read_text(encoding="utf-8"))
    assert evidence["metric_evidence"]
    assert {item["schema_version"] for item in evidence["metric_evidence"]} == {2}


@pytest.mark.parametrize(
    "filename",
    [
        "metrics_long.csv",
        "contrasts.csv",
        "voxel_contrast_summaries.csv",
        "metric_evidence.json",
    ],
)
def test_completed_bundle_rejects_legacy_or_mixed_metric_versions(
    tmp_path: Path,
    filename: str,
) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / filename
    if filename == "metric_evidence.json":
        payload = json.loads(report.read_text(encoding="utf-8"))
        payload["metric_evidence"][0]["schema_version"] = 1
        report.write_bytes(canonical_json_bytes(payload) + b"\n")
    else:
        fieldnames, rows = _csv_rows(report)
        rows[0]["schema_version"] = "1"
        _write_csv_rows(report, fieldnames, rows)
    _rehash_report(root, filename)

    with pytest.raises(ValueError, match="legacy strict-nonzero"):
        validate_completed_f3d_bundle(root)


def test_completed_bundle_rejects_missing_metric_evidence_version(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "metric_evidence.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    del payload["metric_evidence"][0]["schema_version"]
    report.write_bytes(canonical_json_bytes(payload) + b"\n")
    _rehash_report(root, "metric_evidence.json")

    with pytest.raises(F3ResultValidationError, match="metric evidence field set mismatch"):
        validate_completed_f3d_bundle(root)


def test_scanner_report_requires_exact_backend_summary_keys(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    stage = root / "stages" / "scanner" / loaded.cells[0].stages.scanner
    report_path = stage / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["raw"]["unexpected"] = report["raw"]["ft"]
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    _rehash_stage_artifact(root, stage, "report.json")

    with pytest.raises(F3ResultValidationError, match="raw summary key set"):
        validate_completed_f3d_bundle(root)


def test_deep_validation_recomputes_scanner_summary_and_sampling_count(
    tmp_path: Path,
) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    stages = {
        cell.backend: root / "stages" / "scanner" / cell.stages.scanner for cell in loaded.cells
    }

    report_path = stages["reference-like"] / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["raw"]["ft"].update({"min": 0.1, "mean": 0.1, "max": 0.1})
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    _rehash_stage_artifact(root, stages["reference-like"], "report.json")

    assert validate_completed_f3d_bundle(root)
    with pytest.raises(F3ResultValidationError, match="deep scanner summary mismatch"):
        validate_completed_f3d_bundle(root, deep=True)

    report["raw"]["ft"].update({"min": 0.0, "mean": 0.0, "max": 0.0})
    report["sampling_count"]["strike"] += 1
    report["sampling_count"]["orientations"] = (
        report["sampling_count"]["strike"] * report["sampling_count"]["dip"]
    )
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    _rehash_stage_artifact(root, stages["reference-like"], "report.json")

    assert validate_completed_f3d_bundle(root)
    with pytest.raises(F3ResultValidationError, match="deep scanner sampling_count mismatch"):
        validate_completed_f3d_bundle(root, deep=True)


def test_deep_scanner_validation_rejects_rehashed_trailing_dat_bytes(
    tmp_path: Path,
) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    stage = root / "stages" / "scanner" / loaded.cells[0].stages.scanner
    path = stage / "ft.dat"
    with path.open("ab") as stream:
        stream.write(b"\0\0\0\0")
    _rehash_stage_artifact(root, stage, path.name)

    with pytest.raises(
        F3ResultValidationError,
        match="deep scanner array validation failed: ft",
    ):
        result_module._deep_validate_scanner_stages(root, loaded, manifest["plan"])


def test_deep_scanner_validation_is_unique_and_sampling_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    original_scanner = result_module.FaultOrientScanner3
    original_summary = result_module.scanner_array_summary
    original_memmap = result_module.np.memmap
    constructions = 0
    summary_calls = 0
    mappings: list[Any] = []

    class SamplingOnlyScanner:
        def __init__(self, sigma1: float, sigma2: float) -> None:
            nonlocal constructions
            constructions += 1
            self._scanner = original_scanner(sigma1, sigma2)

        def __getattr__(self, name: str) -> Any:
            if name in {"scan", "scan_quality", "thin"}:
                raise AssertionError(f"deep scanner validation called {name}")
            return getattr(self._scanner, name)

    def tracked_summary(array: np.ndarray) -> dict[str, Any]:
        nonlocal summary_calls
        summary_calls += 1
        return original_summary(array)

    def tracked_memmap(*args: Any, **kwargs: Any) -> np.memmap:
        array = original_memmap(*args, **kwargs)
        mappings.append(array._mmap)
        return array

    monkeypatch.setattr(result_module, "FaultOrientScanner3", SamplingOnlyScanner)
    monkeypatch.setattr(result_module, "scanner_array_summary", tracked_summary)
    monkeypatch.setattr(result_module.np, "memmap", tracked_memmap)

    result_module._deep_validate_scanner_stages(root, loaded, manifest["plan"])
    assert constructions == 2
    assert summary_calls == 13
    assert len(mappings) == 13
    assert all(mapping.closed for mapping in mappings)
    shutil.rmtree(root / "stages" / "scanner")


def test_deep_scanner_validation_accepts_semantic_json_formatting(
    tmp_path: Path,
) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    stage = root / "stages" / "scanner" / loaded.cells[0].stages.scanner
    report_path = stage / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.write_bytes(canonical_json_bytes(report) + b" ")
    _rehash_stage_artifact(root, stage, "report.json")

    assert validate_completed_f3d_bundle(root, deep=True)


def test_rehashed_skin_schema_tamper_is_rejected_by_strict_validation(
    tmp_path: Path,
) -> None:
    root = _complete_small_bundle(tmp_path, skinning_enabled=True)
    loaded = load_f3d_mode_comparison_result(root)
    stage = root / "stages" / "skinning" / loaded.cells[0].stages.skinning
    skins_path = stage / "skins.json"
    payload = json.loads(skins_path.read_text(encoding="utf-8"))
    payload["unknown"] = True
    skins_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    _rehash_stage_artifact(root, stage, "skins.json")

    with pytest.raises(F3ResultValidationError, match="skin artifact schema"):
        validate_completed_f3d_bundle(root)


@pytest.mark.parametrize("filename", ("skin_mask.dat", "report.json"))
def test_rehashed_skin_cross_file_tamper_requires_deep_validation(
    tmp_path: Path,
    filename: str,
) -> None:
    root = _complete_small_bundle(tmp_path, skinning_enabled=True)
    loaded = load_f3d_mode_comparison_result(root)
    stage = root / "stages" / "skinning" / loaded.cells[0].stages.skinning
    artifact = stage / filename
    if filename == "skin_mask.dat":
        mask = np.fromfile(artifact, dtype=">f4").reshape(loaded.volume_shape)
        mask.flat[0] = 1.0
        mask.tofile(artifact)
    else:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        payload["topology"]["skin_count"] = 1
        artifact.write_bytes(canonical_json_bytes(payload) + b"\n")
    _rehash_stage_artifact(root, stage, filename)

    assert validate_completed_f3d_bundle(root)
    with pytest.raises(F3ResultValidationError, match="deep skin artifact mismatch"):
        validate_completed_f3d_bundle(root, deep=True)


def test_deep_skin_validation_is_unique_and_precedes_skin_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_small_bundle(tmp_path, skinning_enabled=True)
    loaded = load_f3d_mode_comparison_result(root)
    validated: list[str] = []
    original_validator = result_module.validate_skin_artifact_semantics
    original_metrics = result_module.compute_skin_metric_rows

    def validating(*args: object, **kwargs: object) -> object:
        fingerprint = Path(args[0]).name
        assert fingerprint not in validated
        result = original_validator(*args, **kwargs)
        validated.append(fingerprint)
        return result

    def computing(*args: object, **kwargs: object) -> object:
        assert kwargs["source_stage_fingerprint"] in validated
        return original_metrics(*args, **kwargs)

    monkeypatch.setattr(result_module, "validate_skin_artifact_semantics", validating)
    monkeypatch.setattr(result_module, "compute_skin_metric_rows", computing)

    assert validate_completed_f3d_bundle(root, deep=True)
    assert set(validated) == {
        cell.stages.skinning for cell in loaded.cells if cell.skinning_enabled
    }


def test_deep_validation_exactly_reexecutes_boundary_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_boundary_skin_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    fingerprints = {cell.stages.skinning for cell in loaded.cells if cell.skinning_enabled}
    fallback_calls = 0
    original_fallback = result_module.apply_boundary_skinner_fallback

    def tracked_fallback(*args: Any, **kwargs: Any) -> None:
        nonlocal fallback_calls
        fallback_calls += 1
        original_fallback(*args, **kwargs)

    for fingerprint in fingerprints:
        stage = root / "stages" / "skinning" / fingerprint
        report = json.loads((stage / "report.json").read_text(encoding="utf-8"))
        skins = json.loads((stage / "skins.json").read_text(encoding="utf-8"))
        assert report["diagnostics"]["fallback_used"] is True
        assert [skin["cell_count"] for skin in skins["skins"]] == [2, 1]

    monkeypatch.setattr(
        result_module,
        "apply_boundary_skinner_fallback",
        tracked_fallback,
    )

    assert validate_completed_f3d_bundle(root, deep=True)
    assert fallback_calls == len(fingerprints)


@pytest.mark.parametrize(
    ("stage_kind", "filename", "replacement"),
    (
        ("thinning", "fvt.dat", 0.75),
        ("voting", "vp.dat", 21.0),
        ("voting", "vt.dat", 71.0),
    ),
)
def test_deep_validation_rejects_rehashed_skin_parent_sample_tampering(
    tmp_path: Path,
    stage_kind: str,
    filename: str,
    replacement: float,
) -> None:
    root = _complete_boundary_skin_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    cell = loaded.cells[0]
    fingerprint = getattr(cell.stages, stage_kind)
    stage = root / "stages" / stage_kind / fingerprint
    artifact = stage / filename
    values = np.fromfile(artifact, dtype=">f4").reshape(loaded.volume_shape)
    values[0, 0, 0] = np.float32(replacement)
    values.tofile(artifact)
    _rehash_stage_artifact(root, stage, filename)

    assert validate_completed_f3d_bundle(root)
    with pytest.raises(
        F3ResultValidationError,
        match="cell (fl|fp|ft) does not match parent volume",
    ):
        validate_completed_f3d_bundle(root, deep=True)


def test_deep_validation_rejects_rehashed_subvoxel_geometry_tampering(
    tmp_path: Path,
) -> None:
    root = _complete_boundary_skin_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    stage = root / "stages" / "skinning" / loaded.cells[0].stages.skinning
    skins_path = stage / "skins.json"
    payload = json.loads(skins_path.read_text(encoding="utf-8"))
    payload["skins"][0]["cells"][0]["x1"] = 0.1
    skins_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    _rehash_stage_artifact(root, stage, "skins.json")

    assert validate_completed_f3d_bundle(root)
    with pytest.raises(
        F3ResultValidationError,
        match="does not exactly match skin-only recomputation",
    ):
        validate_completed_f3d_bundle(root, deep=True)


def test_deep_validation_rejects_rehashed_skin_order_tampering(
    tmp_path: Path,
) -> None:
    root = _complete_boundary_skin_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    stage = root / "stages" / "skinning" / loaded.cells[0].stages.skinning
    skins_path = stage / "skins.json"
    payload = json.loads(skins_path.read_text(encoding="utf-8"))
    payload["skins"].reverse()
    for skin_index, skin in enumerate(payload["skins"]):
        skin["skin_index"] = skin_index
    skins_path.write_bytes(canonical_json_bytes(payload) + b"\n")
    _rehash_stage_artifact(root, stage, "skins.json")

    assert validate_completed_f3d_bundle(root)
    with pytest.raises(
        F3ResultValidationError,
        match="does not exactly match skin-only recomputation",
    ):
        validate_completed_f3d_bundle(root, deep=True)


def test_shared_skin_stage_is_reexecuted_once_without_upstream_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_boundary_skin_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    cell = loaded.cells[0]
    fingerprint = cell.stages.skinning
    parsed = result_module.parse_skins_json(
        root / "stages" / "skinning" / fingerprint / "skins.json",
        loaded.volume_shape,
    )
    shared = replace(loaded, cells=(cell, cell))
    skinning_calls = 0
    original_skinning = result_module.execute_skinning_phase3d

    def tracked_skinning(**kwargs: Any) -> Any:
        nonlocal skinning_calls
        skinning_calls += 1
        return original_skinning(**kwargs)

    def unexpected_upstream(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("deep skin validation recomputed an upstream stage")

    monkeypatch.setattr(result_module, "execute_skinning_phase3d", tracked_skinning)
    monkeypatch.setattr(scanner_module, "run_scanner_stages", unexpected_upstream)
    monkeypatch.setattr(workflow3d_module, "execute_workflow3d", unexpected_upstream)
    monkeypatch.setattr(
        workflow3d_module.OptimalSurfaceVoter,
        "apply_voting_from_seeds",
        unexpected_upstream,
    )
    monkeypatch.setattr(
        workflow3d_module.OptimalSurfaceVoter,
        "thin",
        unexpected_upstream,
    )

    result_module._deep_validate_skin_artifacts(
        root,
        shared,
        {fingerprint: parsed},
    )

    assert skinning_calls == 1


def test_pretty_only_formats_root_completion(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path, pretty=True)

    assert (
        (root / "completion.json")
        .read_text(encoding="utf-8")
        .startswith('{\n  "artifact_schema_version"')
    )
    assert b"\n  " not in (root / "reports" / "cells.json").read_bytes()
    assert validate_completed_f3d_bundle(root)


def test_deep_validation_rejects_same_size_reference_file_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_small_bundle(tmp_path)
    reference = tmp_path / "data" / "fl.dat"
    reference.write_bytes(b"\0" * reference.stat().st_size)

    def unexpected_recomputation(*args: object, **kwargs: object) -> None:
        raise AssertionError("source digest must be checked before metric recomputation")

    monkeypatch.setattr(
        result_module,
        "compute_reference_metric_rows",
        unexpected_recomputation,
    )

    assert validate_completed_f3d_bundle(root)
    with pytest.raises(F3ResultValidationError, match="deep source file hash or size mismatch"):
        validate_completed_f3d_bundle(root, deep=True)


def test_report_hash_tamper_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "runtime.csv"
    report.write_bytes(report.read_bytes() + b"x")

    with pytest.raises(F3ResultValidationError, match="hash or size"):
        validate_completed_f3d_bundle(root)


def test_rehashed_runtime_coverage_tamper_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "runtime.csv"
    lines = report.read_bytes().splitlines(keepends=True)
    report.write_bytes(b"".join(lines[:-1]))
    _rehash_report(root, "runtime.csv")

    with pytest.raises(F3ResultValidationError, match="runtime stage coverage"):
        validate_completed_f3d_bundle(root)


def test_rehashed_shared_stage_cannot_be_reported_computed_twice(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "runtime.csv"
    fieldnames, rows = _csv_rows(report)
    voting_fingerprints = [row["fingerprint"] for row in rows if row["stage_kind"] == "voting"]
    shared = next(
        fingerprint
        for fingerprint in voting_fingerprints
        if voting_fingerprints.count(fingerprint) == 2
    )
    shared_rows = [
        row for row in rows if row["stage_kind"] == "voting" and row["fingerprint"] == shared
    ]
    for row in shared_rows:
        row["computed"] = "true"
        row["state"] = "computed"
        row["elapsed_semantics"] = "compute"
    _write_csv_rows(report, fieldnames, rows)
    _rehash_report(root, "runtime.csv")

    with pytest.raises(F3ResultValidationError, match="multiplicity is infeasible"):
        validate_completed_f3d_bundle(root)


@pytest.mark.parametrize(
    ("scope", "message"),
    [
        ("stage", "resource stage storage values mismatch"),
        ("workspace", "resource workspace storage values mismatch"),
    ],
)
def test_rehashed_storage_counts_must_match_workspace(
    tmp_path: Path,
    scope: str,
    message: str,
) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "resources.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    row = next(item for item in payload["storage"] if item["scope"] == scope)
    row["file_count"] += 1
    report.write_bytes(canonical_json_bytes(payload) + b"\n")
    _rehash_report(root, "resources.json")

    with pytest.raises(F3ResultValidationError, match=message):
        validate_completed_f3d_bundle(root)


@pytest.mark.parametrize("scope", ["process_peak", "stage_snapshot"])
def test_rehashed_rss_coverage_tamper_is_rejected(
    tmp_path: Path,
    scope: str,
) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "resources.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["rss"] = [row for row in payload["rss"] if row["scope"] != scope]
    report.write_bytes(canonical_json_bytes(payload) + b"\n")
    _rehash_report(root, "resources.json")

    with pytest.raises(F3ResultValidationError, match="resource RSS"):
        validate_completed_f3d_bundle(root)


@pytest.mark.parametrize(
    ("filename", "message"),
    [
        ("metrics_long.csv", "metric row order"),
        ("voxel_contrast_summaries.csv", "voxel contrast row order"),
        ("regional_metrics.csv", "regional diagnostic row order"),
        ("orientation_diagnostics.csv", "orientation diagnostic row order"),
        ("runtime.csv", "runtime row order"),
    ],
)
def test_rehashed_report_row_reordering_is_rejected(
    tmp_path: Path,
    filename: str,
    message: str,
) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / filename
    fieldnames, rows = _csv_rows(report)
    rows[0], rows[1] = rows[1], rows[0]
    _write_csv_rows(report, fieldnames, rows)
    _rehash_report(root, filename)

    with pytest.raises(F3ResultValidationError, match=message):
        validate_completed_f3d_bundle(root)


def test_rehashed_cell_resolved_config_tamper_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    for path in sorted((root / "cells").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["resolved_config"]["tampered"] = True
        path.write_bytes(canonical_json_bytes(payload) + b"\n")
    report = root / "reports" / "cells.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    for cell in payload["cells"]:
        cell["resolved_config"]["tampered"] = True
    report.write_bytes(canonical_json_bytes(payload) + b"\n")
    _rehash_report(root, "cells.json")

    with pytest.raises(F3ResultValidationError, match="resolved_config"):
        validate_completed_f3d_bundle(root)


@pytest.mark.parametrize("field", ("resolved_config", "resolved_stage_settings"))
def test_rehashed_scanner_report_resolved_controls_tamper_is_rejected(
    tmp_path: Path,
    field: str,
) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    fingerprint = loaded.cells[0].stages.scanner
    stage = root / "stages" / "scanner" / fingerprint
    report_path = stage / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report[field]["normalize"] = not report[field]["normalize"]
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    _rehash_stage_artifact(root, stage, "report.json")

    with pytest.raises(F3ResultValidationError, match=f"scanner report {field} mismatch"):
        validate_completed_f3d_bundle(root)


def test_rehashed_stage_report_crop_semantics_are_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    fingerprint = loaded.cells[0].stages.thinning
    stage = root / "stages" / "thinning" / fingerprint

    report_path = stage / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["crop_shape"] = list(loaded.volume_shape)
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    report_metadata = artifact_file_metadata(report_path)

    manifest_path = stage / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["report.json"] = report_metadata
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    stage_completion_path = stage / "complete.json"
    stage_completion = json.loads(stage_completion_path.read_text(encoding="utf-8"))
    stage_completion["files"]["report.json"] = report_metadata
    stage_completion["files"]["stage_manifest.json"] = artifact_file_metadata(manifest_path)
    stage_completion_path.write_bytes(canonical_json_bytes(stage_completion) + b"\n")

    root_completion_path = root / "completion.json"
    root_completion = json.loads(root_completion_path.read_text(encoding="utf-8"))
    root_completion["stage_completions"][f"thinning/{fingerprint}"] = artifact_file_metadata(
        stage_completion_path
    )
    root_completion_path.write_bytes(canonical_json_bytes(root_completion) + b"\n")

    with pytest.raises(F3ResultValidationError, match="crop/tile/center"):
        validate_completed_f3d_bundle(root)


def test_coherent_stage_settings_tamper_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    target = loaded.cells[0]
    old_fingerprint = target.stages.thinning
    old_stage = root / "stages" / "thinning" / old_fingerprint
    manifest_path = old_stage / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["resolved_settings"]["semantic_key"]["thinning"]["thin_mode"] = "hybrid_v2"
    computation = {name: manifest[name] for name in result_module._STAGE_COMPUTATION_FIELDS}
    new_fingerprint = canonical_fingerprint(computation)
    manifest["fingerprint"] = new_fingerprint

    report_path = old_stage / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["fingerprint"] = new_fingerprint
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    report_metadata = artifact_file_metadata(report_path)
    manifest["files"]["report.json"] = report_metadata
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    completion_path = old_stage / "complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["fingerprint"] = new_fingerprint
    completion["files"]["report.json"] = report_metadata
    completion["files"]["stage_manifest.json"] = artifact_file_metadata(manifest_path)
    completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")
    new_stage = old_stage.with_name(new_fingerprint)
    old_stage.rename(new_stage)

    cells = []
    for cell in loaded.cells:
        if cell.stages.thinning == old_fingerprint:
            cell = replace(
                cell,
                stages=replace(cell.stages, thinning=new_fingerprint),
            )
            cell.path.write_bytes(canonical_json_bytes(cell.as_dict()) + b"\n")
        cells.append(cell)
    tampered = replace(loaded, cells=tuple(cells))
    run_manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))

    with pytest.raises(F3ResultValidationError, match="resolved_settings"):
        result_module._validate_cells_and_stages(
            root,
            tampered,
            result_module._dataset_contract(run_manifest),
            run_manifest["plan"],
        )


def test_coherent_stage_refingerprint_cannot_omit_canonical_artifact(
    tmp_path: Path,
) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    cell = loaded.cells[0]
    old_stage = root / "stages" / "thinning" / cell.stages.thinning
    manifest_path = old_stage / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_schema"].pop("fvt.dat")
    manifest["files"].pop("fvt.dat")
    (old_stage / "fvt.dat").unlink()
    computation = {name: manifest[name] for name in result_module._STAGE_COMPUTATION_FIELDS}
    new_fingerprint = canonical_fingerprint(computation)
    manifest["fingerprint"] = new_fingerprint

    report_path = old_stage / "report.json"
    stage_report = json.loads(report_path.read_text(encoding="utf-8"))
    stage_report["fingerprint"] = new_fingerprint
    report_path.write_bytes(canonical_json_bytes(stage_report) + b"\n")
    manifest["files"]["report.json"] = artifact_file_metadata(report_path)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    completion_path = old_stage / "complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["fingerprint"] = new_fingerprint
    completion["files"].pop("fvt.dat")
    completion["files"]["report.json"] = artifact_file_metadata(report_path)
    completion["files"]["stage_manifest.json"] = artifact_file_metadata(manifest_path)
    completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")
    new_stage = old_stage.with_name(new_fingerprint)
    old_stage.rename(new_stage)

    with pytest.raises(F3ResultValidationError, match="canonical artifact schema mismatch"):
        result_module._validate_referenced_stage(
            root,
            "thinning",
            new_fingerprint,
            loaded.run_fingerprint,
            (cell.stages.voting,),
            {},
            loaded.volume_shape,
        )


def test_deep_validation_recomputes_voxel_contrast_scalars(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "voxel_contrast_summaries.csv"
    with report.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    assert rows[0]["contrast_name"] == "scanner_effect_ref"
    rows[0]["epsilon_nonzero_fraction"] = (
        "0.5" if rows[0]["epsilon_nonzero_fraction"] != "0.5" else "0.25"
    )
    with report.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _rehash_report(root, "voxel_contrast_summaries.csv")

    assert validate_completed_f3d_bundle(root)
    with pytest.raises(F3ResultValidationError, match="deep voxel contrast summary"):
        validate_completed_f3d_bundle(root, deep=True)


def test_deep_validation_recomputes_skin_metric_evidence(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path, skinning_enabled=True)
    loaded = load_f3d_mode_comparison_result(root)
    target = next(
        item
        for item in loaded.metric_evidence
        if item.cell_label == "RL-REF" and item.stage == "skin"
    )
    changed_counts = tuple(
        (name, value + 1 if name == "skin_count" else value) for name, value in target.counts
    )
    changed_evidence = tuple(
        replace(item, counts=changed_counts) if item is target else item
        for item in loaded.metric_evidence
    )
    changed_rows = tuple(
        replace(row, value=float(row.value) + 1.0)
        if row.cell_label == "RL-REF" and row.stage == "skin" and row.metric == "skin_count"
        else row
        for row in loaded.metric_rows
    )
    changed_contrasts = result_module.compute_contrast_rows(
        changed_rows,
        changed_evidence,
    )
    changed = replace(
        loaded,
        metric_rows=changed_rows,
        metric_evidence=changed_evidence,
        contrast_rows=changed_contrasts,
    )
    payloads = result_module._serialize_reports(changed)
    for filename in (
        "metrics_long.csv",
        "metric_evidence.json",
        "contrasts.csv",
    ):
        (root / "reports" / filename).write_bytes(payloads[filename])
        _rehash_report(root, filename)

    assert validate_completed_f3d_bundle(root)
    with pytest.raises(F3ResultValidationError, match="deep metric"):
        validate_completed_f3d_bundle(root, deep=True)


def test_metric_evidence_requires_canonical_nonzero_epsilon(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    target = next(
        item
        for item in loaded.metric_evidence
        if item.cell_label == "RL-REF" and item.stage == "ft" and item.selection == "all"
    )
    changed = replace(
        loaded,
        metric_evidence=tuple(
            replace(item, thresholds=(("nonzero_epsilon", 0.5),)) if item is target else item
            for item in loaded.metric_evidence
        ),
    )

    with pytest.raises(F3ResultValidationError, match="nonzero epsilon"):
        result_module.validate_f3d_mode_comparison_result(root, changed)


def test_nonofficial_dataset_cannot_be_validated_as_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_small_bundle(tmp_path)
    monkeypatch.setattr(
        result_module,
        "OFFICIAL_F3_DATASET_SPEC",
        replace(
            result_module.OFFICIAL_F3_DATASET_SPEC,
            dataset_id="f3d-official-v1",
        ),
    )

    with pytest.raises(F3ResultValidationError, match="official F3 dataset ID"):
        validate_completed_f3d_bundle(root)


def test_nonofficial_shape_cannot_be_validated_as_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_small_bundle(tmp_path)
    shape = (420, 400, 100)
    monkeypatch.setattr(
        result_module,
        "OFFICIAL_F3_DATASET_SPEC",
        replace(
            result_module.OFFICIAL_F3_DATASET_SPEC,
            shape=shape,
            expected_bytes=int(np.prod(shape)) * np.dtype(">f4").itemsize,
        ),
    )

    with pytest.raises(F3ResultValidationError, match="official F3 shape"):
        validate_completed_f3d_bundle(root)


@pytest.mark.parametrize(
    "field",
    [
        "crop_shape",
        "crop_center",
        "tile_shape",
        "tile_sample",
        "center",
        "replicate_index",
    ],
)
def test_crop_tile_and_center_semantics_are_rejected(field: str) -> None:
    with pytest.raises(F3ResultValidationError, match="crop/tile/center"):
        result_module._reject_crop_semantics({"plan": {field: [1, 2, 3]}})


@pytest.mark.parametrize("value", ["crop", "tiled-volume", "center_dimension"])
def test_crop_tile_and_center_semantics_in_values_are_rejected(value: str) -> None:
    with pytest.raises(F3ResultValidationError, match="crop/tile/center"):
        result_module._reject_crop_semantics({"plan": {"sample_mode": value}})


def test_semantic_tokens_do_not_reject_valid_percentile_or_recenter_fields() -> None:
    result_module._reject_crop_semantics(
        {
            "percentile": 99.0,
            "fvt_recenter_target": "scanner_fet",
            "regions_are_replicates": False,
        }
    )


def test_deep_validation_recomputes_orientation_diagnostics(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "orientation_diagnostics.csv"
    with report.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None
    rows[0]["support_count"] = "7"
    fabricated_summary = json.dumps(
        {"count": 7, "mean": 1.0, "median": 1.0, "p90": 1.0, "p95": 1.0},
        separators=(",", ":"),
        sort_keys=True,
    )
    for name in (
        "strike_circular_absolute_difference",
        "dip_absolute_difference",
        "normal_vector_angular_difference",
    ):
        rows[0][name] = fabricated_summary
    with report.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _rehash_report(root, "orientation_diagnostics.csv")

    assert validate_completed_f3d_bundle(root)
    with pytest.raises(F3ResultValidationError, match="deep orientation diagnostic"):
        validate_completed_f3d_bundle(root, deep=True)


def test_rehashed_orientation_source_fingerprint_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "orientation_diagnostics.csv"
    fieldnames, rows = _csv_rows(report)
    rows[0]["left_source_stage_fingerprint"] = "0" * 64
    _write_csv_rows(report, fieldnames, rows)
    _rehash_report(root, "orientation_diagnostics.csv")

    with pytest.raises(
        F3ResultValidationError,
        match="orientation source stage fingerprint",
    ):
        validate_completed_f3d_bundle(root)


def test_missing_completion_is_an_incomplete_workspace(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    (root / "completion.json").unlink()

    with pytest.raises(F3ResultValidationError, match="completion.json"):
        validate_completed_f3d_bundle(root)


def test_stage_byte_tamper_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    cells = json.loads((root / "reports" / "cells.json").read_text(encoding="utf-8"))
    fingerprint = cells["cells"][0]["stages"]["scanner"]
    artifact = root / "stages" / "scanner" / fingerprint / "ft.dat"
    payload = bytearray(artifact.read_bytes())
    payload[0] ^= 0x01
    artifact.write_bytes(payload)

    with pytest.raises(F3ArtifactError):
        validate_completed_f3d_bundle(root)


def test_rehashed_mixed_generation_cell_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "cells.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    by_label = {cell["label"]: cell for cell in payload["cells"]}
    replacement = by_label["Q-QUAL"]["stages"]["scanner"]
    by_label["RL-QUAL"]["stages"]["scanner"] = replacement
    report.write_bytes(canonical_json_bytes(payload) + b"\n")

    cell_path = root / "cells" / "RL-QUAL.json"
    cell_payload = json.loads(cell_path.read_text(encoding="utf-8"))
    cell_payload["stages"]["scanner"] = replacement
    cell_path.write_bytes(canonical_json_bytes(cell_payload) + b"\n")
    _rehash_report(root, "cells.json")

    with pytest.raises(
        F3ResultValidationError,
        match="scanner stage backend reuse mismatch",
    ):
        validate_completed_f3d_bundle(root)


def test_rehashed_metric_and_contrast_tamper_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    index = next(
        index
        for index, row in enumerate(loaded.metric_rows)
        if row.stage == "fvt" and row.contrast_eligible and row.value is not None
    )
    rows = list(loaded.metric_rows)
    rows[index] = replace(rows[index], value=float(rows[index].value) + 0.125)
    contrasts = result_module.compute_contrast_rows(tuple(rows), loaded.metric_evidence)
    metrics_path = root / "reports" / "metrics_long.csv"
    contrasts_path = root / "reports" / "contrasts.csv"
    metrics_path.write_bytes(result_module._csv_bytes(tuple(rows), result_module.MetricRow))
    contrasts_path.write_bytes(result_module._csv_bytes(contrasts, result_module.ContrastRow))
    _rehash_report(root, "metrics_long.csv")
    _rehash_report(root, "contrasts.csv")

    with pytest.raises(F3ResultValidationError, match="metric evidence scalar mismatch"):
        validate_completed_f3d_bundle(root)


def test_metric_axes_must_match_owning_cell(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    row = loaded.metric_rows[0]
    object.__setattr__(row, "scanner_backend", "quality")
    object.__setattr__(row, "workflow_mode", "quality")

    with pytest.raises(F3ResultValidationError, match="metric row axes"):
        result_module.validate_f3d_mode_comparison_result(root, loaded)


def test_rehashed_regional_count_tamper_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "regional_metrics.csv"
    fieldnames, rows = _csv_rows(report)
    metrics = json.loads(rows[0]["metrics"])
    metrics["voxel_count"] += 1
    rows[0]["metrics"] = json.dumps(metrics, separators=(",", ":"), sort_keys=True)
    _write_csv_rows(report, fieldnames, rows)
    _rehash_report(root, "regional_metrics.csv")

    with pytest.raises(F3ResultValidationError, match="regional counts"):
        validate_completed_f3d_bundle(root)


def test_coordinated_regional_partition_count_tamper_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "regional_metrics.csv"
    fieldnames, rows = _csv_rows(report)
    interior_row, boundary_row = rows[:2]
    assert interior_row["region"] == "interior"
    assert boundary_row["region"] == "boundary_shell"

    interior_metrics = json.loads(interior_row["metrics"])
    boundary_metrics = dict(interior_metrics)
    interior_metrics["voxel_count"] -= 1
    boundary_metrics["voxel_count"] = 1
    interior_row["metrics"] = json.dumps(
        interior_metrics,
        separators=(",", ":"),
        sort_keys=True,
    )
    boundary_row["metrics"] = json.dumps(
        boundary_metrics,
        separators=(",", ":"),
        sort_keys=True,
    )
    _write_csv_rows(report, fieldnames, rows)
    _rehash_report(root, "regional_metrics.csv")

    with pytest.raises(F3ResultValidationError, match="regional counts"):
        validate_completed_f3d_bundle(root)


def test_rehashed_regional_margin_tamper_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "regional_metrics.csv"
    fieldnames, rows = _csv_rows(report)
    for row in rows:
        row["boundary_margin"] = "1"
    _write_csv_rows(report, fieldnames, rows)
    _rehash_report(root, "regional_metrics.csv")

    with pytest.raises(F3ResultValidationError, match="run manifest"):
        validate_completed_f3d_bundle(root, deep=True)


def test_rehashed_regional_union_tamper_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "regional_metrics.csv"
    fieldnames, rows = _csv_rows(report)
    metrics = json.loads(rows[0]["metrics"])
    union_name = next(name for name in metrics if name.endswith("_union_count"))
    prefix = union_name.removesuffix("_union_count")
    metrics[union_name] += 1
    metrics[f"{prefix}_jaccard"] = metrics[f"{prefix}_intersection_count"] / metrics[union_name]
    rows[0]["metrics"] = json.dumps(metrics, separators=(",", ":"), sort_keys=True)
    _write_csv_rows(report, fieldnames, rows)
    _rehash_report(root, "regional_metrics.csv")

    with pytest.raises(F3ResultValidationError, match="regional overlap counts"):
        validate_completed_f3d_bundle(root)


def test_regional_axes_must_match_owning_cell(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)
    row = loaded.regional_rows[0]
    object.__setattr__(row, "scanner_backend", "quality")
    object.__setattr__(row, "workflow_mode", "quality")

    with pytest.raises(F3ResultValidationError, match="regional row axes"):
        result_module.validate_f3d_mode_comparison_result(root, loaded)


def test_rehashed_regional_source_fingerprint_tamper_is_rejected(
    tmp_path: Path,
) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "regional_metrics.csv"
    fieldnames, rows = _csv_rows(report)
    rows[0]["source_stage_fingerprint"] = "0" * 64
    _write_csv_rows(report, fieldnames, rows)
    _rehash_report(root, "regional_metrics.csv")

    with pytest.raises(
        F3ResultValidationError,
        match="regional source stage fingerprint",
    ):
        validate_completed_f3d_bundle(root)


def test_rehashed_contrast_only_tamper_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "contrasts.csv"
    fieldnames, rows = _csv_rows(report)
    rows[0]["raw_value"] = repr(float(rows[0]["raw_value"]) + 0.125)
    _write_csv_rows(report, fieldnames, rows)
    _rehash_report(root, "contrasts.csv")

    with pytest.raises(F3ResultValidationError, match="contrast rows"):
        validate_completed_f3d_bundle(root)


def test_run_manifest_mismatch_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["implementation_identity"]["name"] = "tampered"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(F3ResultValidationError, match="run manifest fingerprint"):
        validate_completed_f3d_bundle(root)


def test_shallow_validation_rejects_nonpublication_recorded_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_small_bundle(tmp_path)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["runtime_identity"]["thread_environment"]["OMP_NUM_THREADS"] = None
    computation = {name: manifest[name] for name in result_module._RUN_COMPUTATION_FIELDS}
    manifest["run_fingerprint"] = canonical_fingerprint(computation)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    monkeypatch.setattr(result_module, "F3_DATASET_ID", "result-fixture")

    with pytest.raises(
        F3ResultValidationError,
        match="run manifest publication runtime identity",
    ):
        validate_completed_f3d_bundle(root)


def test_previous_fingerprint_contract_is_rejected_even_when_rehashed(
    tmp_path: Path,
) -> None:
    root = _complete_small_bundle(tmp_path)
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fingerprint_contract_version"] = 2
    computation = {name: manifest[name] for name in result_module._RUN_COMPUTATION_FIELDS}
    manifest["run_fingerprint"] = canonical_fingerprint(computation)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")

    with pytest.raises(F3ResultValidationError, match="fingerprint contract"):
        validate_completed_f3d_bundle(root)


def test_rehashed_extra_cell_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "cells.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["cells"].append(payload["cells"][0])
    report.write_bytes(canonical_json_bytes(payload) + b"\n")
    _rehash_report(root, "cells.json")

    with pytest.raises(F3ResultValidationError, match="canonical coverage and order"):
        validate_completed_f3d_bundle(root)


def test_rehashed_unknown_cell_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    report = root / "reports" / "cells.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["cells"][0]["label"] = "UNKNOWN"
    report.write_bytes(canonical_json_bytes(payload) + b"\n")
    _rehash_report(root, "cells.json")

    with pytest.raises((F3ResultValidationError, ValueError)):
        validate_completed_f3d_bundle(root)


def test_disabled_skinning_fingerprint_tamper_is_rejected(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path, skinning_enabled=False)
    report = root / "reports" / "cells.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    cell = payload["cells"][0]
    cell["stages"]["skinning"] = "f" * 64
    report.write_bytes(canonical_json_bytes(payload) + b"\n")
    cell_path = root / "cells" / f"{cell['label']}.json"
    cell_path.write_bytes(canonical_json_bytes(cell) + b"\n")
    _rehash_report(root, "cells.json")

    with pytest.raises(F3ResultValidationError, match="skinning stage fingerprint mismatch"):
        validate_completed_f3d_bundle(root)


def test_disabled_skinning_does_not_parse_or_require_skin_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_small_bundle(tmp_path, skinning_enabled=False)
    assert not any((root / "stages" / "skinning").iterdir())

    def unexpected_parse(*args: object, **kwargs: object) -> object:
        raise AssertionError("disabled skinning must not parse skins.json")

    monkeypatch.setattr(result_module, "parse_skins_json", unexpected_parse)

    assert validate_completed_f3d_bundle(root)
    assert validate_completed_f3d_bundle(root, deep=True)


@pytest.mark.parametrize("failure", ["report_write", "deep_validation", "completion_write"])
def test_finalization_failure_never_leaves_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root = _complete_small_bundle(tmp_path)
    result = load_f3d_mode_comparison_result(root)
    (root / "completion.json").unlink()
    original_write = result_module.atomic_write_artifact

    if failure in {"report_write", "completion_write"}:
        target = "metrics_long.csv" if failure == "report_write" else "completion.json"

        def injected_write(path: Path, payload: bytes, *, temporary_prefix: str) -> None:
            if Path(path).name == target:
                raise OSError(f"injected {failure}")
            original_write(path, payload, temporary_prefix=temporary_prefix)

        monkeypatch.setattr(result_module, "atomic_write_artifact", injected_write)
    else:

        def injected_deep_validation(*args: object, **kwargs: object) -> None:
            raise RuntimeError("injected deep validation")

        monkeypatch.setattr(
            result_module,
            "_deep_validate_orientation_diagnostics",
            injected_deep_validation,
        )

    with pytest.raises((OSError, RuntimeError), match="injected"):
        finalize_f3d_bundle(root, result, deep=failure == "deep_validation")
    assert not (root / "completion.json").exists()
    assert not tuple(root.glob(".completion.json.tmp-*"))
    assert not tuple((root / "reports").glob(".*.tmp-*"))


def test_deep_finalization_runs_expensive_validation_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_small_bundle(tmp_path)
    result = load_f3d_mode_comparison_result(root)
    (root / "completion.json").unlink()
    scanner_calls = 0
    skin_calls = 0
    original_scanner = result_module._deep_validate_scanner_stages
    original_skin = result_module._deep_validate_skin_artifacts

    def tracked_scanner(*args: Any, **kwargs: Any) -> None:
        nonlocal scanner_calls
        scanner_calls += 1
        original_scanner(*args, **kwargs)

    def tracked_skin(*args: Any, **kwargs: Any) -> None:
        nonlocal skin_calls
        skin_calls += 1
        original_skin(*args, **kwargs)

    monkeypatch.setattr(result_module, "_deep_validate_scanner_stages", tracked_scanner)
    monkeypatch.setattr(result_module, "_deep_validate_skin_artifacts", tracked_skin)

    finalize_f3d_bundle(root, result, deep=True)

    assert scanner_calls == 1
    assert skin_calls == 1


def test_interrupted_completion_temporary_does_not_block_finalization(
    tmp_path: Path,
) -> None:
    root = _complete_small_bundle(tmp_path)
    result = load_f3d_mode_comparison_result(root)
    (root / "completion.json").unlink()
    temporary = root / ".completion.json.tmp-interrupted"
    temporary.write_text("interrupted\n", encoding="utf-8")
    result = replace(result, storage_rows=storage_report(root))

    finalized = finalize_f3d_bundle(root, result)

    assert finalized == result
    assert (root / "completion.json").is_file()
    assert not temporary.exists()


def test_valid_resume_does_not_serialize_or_write_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root)

    def unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError("resume attempted report computation")

    monkeypatch.setattr(result_module, "_serialize_reports", unexpected)
    monkeypatch.setattr(result_module, "atomic_write_artifact", unexpected)

    assert finalize_f3d_bundle(root, resume=True) == loaded


def test_deep_validation_releases_each_full_volume_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_small_bundle(tmp_path)
    original_memmap = result_module.np.memmap
    original_close = result_module._close_memmap
    original_orientation = result_module.compute_orientation_pair_diagnostic
    original_deep_orientation = result_module._deep_validate_orientation_diagnostics
    active: set[int] = set()
    peak = 0
    orientation_calls = 0
    tracking = False

    class TrackedMemmap(original_memmap):  # type: ignore[misc, valid-type]
        def __new__(cls, *args: object, **kwargs: object) -> np.memmap:
            nonlocal peak
            array = super().__new__(cls, *args, **kwargs)
            if tracking:
                active.add(id(array))
                peak = max(peak, len(active))
            return array

    def tracked_close(array: np.memmap | None) -> None:
        original_close(array)
        if array is not None:
            active.discard(id(array))

    def tracked_orientation(*args: object, **kwargs: object) -> object:
        nonlocal orientation_calls
        orientation_calls += 1
        assert len(active) == 6
        return original_orientation(*args, **kwargs)

    def tracked_deep_orientation(*args: object, **kwargs: object) -> None:
        nonlocal tracking
        tracking = True
        try:
            original_deep_orientation(*args, **kwargs)
        finally:
            tracking = False

    monkeypatch.setattr(result_module.np, "memmap", TrackedMemmap)
    monkeypatch.setattr(result_module, "_close_memmap", tracked_close)
    monkeypatch.setattr(
        result_module,
        "compute_orientation_pair_diagnostic",
        tracked_orientation,
    )
    monkeypatch.setattr(
        result_module,
        "_deep_validate_orientation_diagnostics",
        tracked_deep_orientation,
    )

    assert validate_completed_f3d_bundle(root, deep=True)
    assert orientation_calls == 2 * len(result_module.F3_ORIENTATION_PAIRS)
    assert peak == 6
    assert not active


def test_skin_recomputation_stages_one_read_only_parent_map_at_a_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (2, 3, 4)
    root = tmp_path / "run"
    sources = (
        ("voting", "voting-stage", "fv.dat"),
        ("thinning", "thinning-stage", "fvt.dat"),
        ("voting", "voting-stage", "vp.dat"),
        ("voting", "voting-stage", "vt.dat"),
    )
    scanner_source = ("scanner", "scanner-stage", "ft.dat")
    for offset, source in enumerate((*sources, scanner_source)):
        kind, fingerprint, filename = source
        stage = root / "stages" / kind / fingerprint
        stage.mkdir(parents=True, exist_ok=True)
        values = np.full(shape, np.float32(0.1 * (offset + 1)), dtype=np.float32)
        values.astype(">f4").tofile(stage / filename)

    variant = VariantSpec(
        "producing-variant",
        skinning=SkinningPatch(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary_filtered",
        ),
        experimental=False,
    )
    cell = SimpleNamespace(
        stages=SimpleNamespace(voting="voting-stage", thinning="thinning-stage"),
        resolved_config={"variant": asdict(variant)},
    )
    parsed = result_module.ParsedSkinArtifacts(skins=())
    skinning_config = asdict(
        SyntheticSkinningConfig(
            enabled=True,
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary_filtered",
        )
    )
    original_open = result_module._open_parent_volume
    original_close = result_module._close_memmap
    active: dict[int, tuple[str, str, str]] = {}
    opened: list[np.memmap] = []
    peak = 0
    fallback_called = False

    def tracked_open(
        root_path: Path,
        source: tuple[str, str, str],
        volume_shape: tuple[int, int, int],
    ) -> np.memmap:
        nonlocal peak
        values = original_open(root_path, source, volume_shape)
        opened.append(values)
        active[id(values)] = source
        peak = max(peak, len(active))
        return values

    def tracked_close(values: np.memmap | None) -> None:
        original_close(values)
        if values is not None:
            active.pop(id(values), None)

    def fake_skinning_phase(**kwargs: object) -> SimpleNamespace:
        assert kwargs["variant_spec"] == variant
        assert not active
        assert all(values.mode == "r" and not values.flags.writeable for values in opened)
        parents = tuple(kwargs[name] for name in ("fv", "fvt", "vp", "vt"))
        assert all(isinstance(values, np.ndarray) for values in parents)
        assert all(not values.flags.writeable for values in parents)
        assert kwargs["scanner_target_positive_mask"] is None
        boundary_fallback_runner = kwargs["boundary_fallback_runner"]
        assert callable(boundary_fallback_runner)
        boundary_fallback_runner()
        return SimpleNamespace(
            diagnostics={"fallback_used": False},
            skins=(),
        )

    def fake_boundary_fallback(*args: object, **kwargs: object) -> None:
        nonlocal fallback_called
        fallback_called = True
        assert not args
        scanner_mask = kwargs["scanner_target_positive_mask"]
        assert isinstance(scanner_mask, np.ndarray)
        assert scanner_mask.dtype == np.dtype(bool)
        assert not scanner_mask.flags.writeable
        assert np.all(scanner_mask)

    monkeypatch.setattr(result_module, "_open_parent_volume", tracked_open)
    monkeypatch.setattr(result_module, "_close_memmap", tracked_close)
    monkeypatch.setattr(result_module, "execute_skinning_phase3d", fake_skinning_phase)
    monkeypatch.setattr(
        result_module,
        "apply_boundary_skinner_fallback",
        fake_boundary_fallback,
    )

    result_module._recompute_skin_artifacts(
        root,
        cell,
        parsed,
        shape,
        skinning_config,
        False,
        scanner_source,
    )

    assert len(opened) == 5
    assert peak == 1
    assert fallback_called
    assert not active
    assert all(values._mmap.closed for values in opened)


def test_loaded_result_holds_no_open_workspace_handles(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root, deep=True)

    shutil.rmtree(root)

    assert loaded.cells
    assert not root.exists()
