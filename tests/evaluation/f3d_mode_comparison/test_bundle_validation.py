from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import pyosv.evaluation.f3d_mode_comparison.result as result_module
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

from .test_runner import _LoadedScanner, _scanner_stage


@pytest.fixture(autouse=True)
def _small_official_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the canonical-only validator without allocating the full F3 volume."""

    monkeypatch.setattr(result_module, "F3_DATASET_ID", "result-fixture")
    monkeypatch.setattr(result_module, "F3D_SHAPE", (3, 4, 5))


def _complete_small_bundle(tmp_path: Path, *, skinning_enabled: bool = False) -> Path:
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
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=skinning_enabled))
    plan_payload = plan.as_dict()
    plan_payload["dataset_spec"] = asdict(spec)
    computation = {
        "artifact_schema_version": 1,
        "stage_contract_version": 1,
        "fingerprint_contract_version": 1,
        "plan": plan_payload,
        "dataset_identity": source.identity.computation_identity,
        "implementation_identity": {"name": "test"},
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
    cell_result = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: _LoadedScanner(stage, []),
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
    finalize_f3d_bundle(workspace, result)
    source.close()
    return root


def _rehash_report(root: Path, filename: str) -> None:
    completion_path = root / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["report_files"][filename] = artifact_file_metadata(root / "reports" / filename)
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


def test_nonofficial_dataset_cannot_be_validated_as_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_small_bundle(tmp_path)
    monkeypatch.setattr(result_module, "F3_DATASET_ID", "f3d-official-v1")
    monkeypatch.setattr(result_module, "F3D_SHAPE", (420, 400, 100))

    with pytest.raises(F3ResultValidationError, match="official F3 dataset ID"):
        validate_completed_f3d_bundle(root)


def test_nonofficial_shape_cannot_be_validated_as_canonical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _complete_small_bundle(tmp_path)
    monkeypatch.setattr(result_module, "F3D_SHAPE", (420, 400, 100))

    with pytest.raises(F3ResultValidationError, match="official F3 shape"):
        validate_completed_f3d_bundle(root)


@pytest.mark.parametrize("field", ["crop_shape", "tile_shape", "center"])
def test_crop_tile_and_center_semantics_are_rejected(field: str) -> None:
    with pytest.raises(F3ResultValidationError, match="crop/tile/center"):
        result_module._reject_crop_semantics({"plan": {field: [1, 2, 3]}})


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

    with pytest.raises(F3ResultValidationError):
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


def test_loaded_result_holds_no_open_workspace_handles(tmp_path: Path) -> None:
    root = _complete_small_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root, deep=True)

    shutil.rmtree(root)

    assert loaded.cells
    assert not root.exists()
