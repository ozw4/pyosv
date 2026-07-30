from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import pytest

from pyosv.cli import f3d_mode_comparison
from pyosv.evaluation.f3d_mode_comparison import (
    F3_FINGERPRINT_CONTRACT_VERSION,
    F3_METRIC_SCHEMA_VERSION,
    F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE,
    F3_RESKIN_POLICY_COMPARISON_DIR,
    F3_RESKIN_POLICY_COMPARISON_FILES,
    F3_RUNTIME_IDENTITY_SCHEMA_VERSION,
    F3_SCANNER_STAGE_CONTRACT_VERSION,
    F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
    OFFICIAL_F3_DATASET_SPEC,
    canonical_fingerprint,
    load_f3d_mode_comparison_result,
    numerical_runtime_identity,
    validate_completed_f3d_bundle,
    validate_f3_reskin_policy_comparison,
    validate_publication_runtime_identity,
)
from pyosv.evaluation.f3d_mode_comparison.scanner import sampling_count_from_evidence


def _required_environment() -> tuple[Path, Path]:
    if os.environ.get("PYOSV_RUN_F3D_MODE_COMPARISON") != "1":
        pytest.skip("set PYOSV_RUN_F3D_MODE_COMPARISON=1 for the official F3 run")
    data_value = os.environ.get("PYOSV_F3D_DATA_ROOT")
    output_value = os.environ.get("PYOSV_F3D_MODE_COMPARISON_OUTPUT_DIR")
    if not data_value or not output_value:
        pytest.skip("official F3 data-root and output-dir environment variables are required")
    data_root = Path(data_value)
    output_root = Path(output_value)
    missing = [
        filename
        for filename in OFFICIAL_F3_DATASET_SPEC.required_files
        if not (data_root / filename).is_file()
    ]
    if missing:
        pytest.skip(f"official F3 files are unavailable: {', '.join(missing)}")
    if output_root.exists() and (not output_root.is_dir() or output_root.is_symlink()):
        pytest.skip(f"output path is unavailable: {output_root}")
    if not output_root.exists() and not output_root.parent.is_dir():
        pytest.skip(f"output parent is unavailable: {output_root.parent}")
    return data_root, output_root


def test_official_f3_full_volume_mode_comparison() -> None:
    data_root, output_root = _required_environment()
    runtime_identity = validate_publication_runtime_identity(numerical_runtime_identity())
    assert runtime_identity["effective_acceleration_state"] == "numba_jit_enabled"
    assert runtime_identity["numba_jit"]["enabled"] is True
    deep = os.environ.get("PYOSV_F3D_MODE_COMPARISON_DEEP_VALIDATE") == "1"
    arguments = [
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_root),
        "--compare-reskin-policies",
        "existing_cells_v1,reference_dense_v1",
    ]
    if output_root.exists():
        arguments.append("--resume")
    if deep:
        arguments.append("--deep-validate")

    assert f3d_mode_comparison.main(arguments) == 0
    assert validate_completed_f3d_bundle(output_root)
    assert validate_completed_f3d_bundle(output_root, deep=deep)
    result = load_f3d_mode_comparison_result(output_root, deep=deep)
    assert result.dataset_id == OFFICIAL_F3_DATASET_SPEC.dataset_id
    assert result.volume_shape == OFFICIAL_F3_DATASET_SPEC.shape
    assert [cell.label for cell in result.cells] == [
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    ]
    assert len({cell.stages.scanner for cell in result.cells}) == 2
    assert len({cell.stages.voting for cell in result.cells}) == 2
    assert len({cell.stages.thinning for cell in result.cells}) == 4
    assert len({cell.stages.skinning for cell in result.cells}) == 4

    manifest = json.loads((output_root / "run_manifest.json").read_text())
    assert manifest["fingerprint_contract_version"] == F3_FINGERPRINT_CONTRACT_VERSION
    assert manifest["runtime_identity"] == runtime_identity
    assert (
        manifest["runtime_identity"]["runtime_identity_schema_version"]
        == F3_RUNTIME_IDENTITY_SCHEMA_VERSION
    )
    assert manifest["runtime_identity"]["effective_acceleration_state"] == "numba_jit_enabled"
    assert manifest["runtime_identity"]["numba_jit"]["enabled"] is True
    assert manifest["runtime_identity"]["numba_environment"]["NUMBA_DISABLE_JIT"] == "0"
    assert manifest["runtime_identity"]["numba_environment"]["NUMBA_NUM_THREADS"] == "1"
    assert manifest["runtime_identity"]["numpy_runtime_cpu"]["status"] == "available"
    assert manifest["runtime_identity"]["numpy_runtime_blas"]["status"] == "available"
    assert manifest["runtime_identity"]["scipy_build"]["status"] == "available"
    assert all(
        library["effective_thread_count"] == 1
        for library in manifest["runtime_identity"]["numpy_runtime_blas"]["libraries"]
    )
    identities = manifest["dataset_identity"]["files"]
    assert [item["role"] for item in identities] == list(OFFICIAL_F3_DATASET_SPEC.roles)
    assert all(len(item["sha256"]) == 64 for item in identities)
    completion = json.loads((output_root / "completion.json").read_text())
    assert len(completion["stage_completions"]) == 12
    assert all(
        len(metadata["sha256"]) == 64 for metadata in completion["stage_completions"].values()
    )

    comparison_paths = validate_f3_reskin_policy_comparison(
        output_root,
        deep=deep,
        require_deep=deep,
    )
    comparison_root = output_root / F3_RESKIN_POLICY_COMPARISON_DIR
    comparison_completion_path = comparison_root / F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE
    comparison_completion = json.loads(comparison_completion_path.read_text())
    comparison_report = json.loads(comparison_paths[0].read_text())
    runtime_sha256 = canonical_fingerprint(runtime_identity)
    assert comparison_completion["source_runtime_identity_sha256"] == runtime_sha256
    assert comparison_completion["comparison_runtime_identity_sha256"] == runtime_sha256
    assert comparison_report["source_runtime_identity_sha256"] == runtime_sha256
    assert comparison_report["comparison_runtime_identity_sha256"] == runtime_sha256
    if deep:
        assert comparison_completion["validation_level"] == "deep"
        assert comparison_report["validation_level"] == "deep"

    comparison_files = (
        *comparison_paths,
        comparison_completion_path,
    )
    before_resume = {
        path: (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in comparison_files
    }
    resume_arguments = list(arguments)
    if "--resume" not in resume_arguments:
        resume_arguments.append("--resume")
    assert f3d_mode_comparison.main(resume_arguments) == 0
    assert {
        path: (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in comparison_files
    } == before_resume
    assert {path.name for path in comparison_paths} == set(F3_RESKIN_POLICY_COMPARISON_FILES)

    scanner_fingerprints = {cell.stages.scanner for cell in result.cells}
    for fingerprint in scanner_fingerprints:
        report = json.loads(
            (output_root / "stages" / "scanner" / fingerprint / "report.json").read_text()
        )
        assert report["scanner_stage_contract_version"] == F3_SCANNER_STAGE_CONTRACT_VERSION
        evidence = report["sampling_evidence"]
        assert evidence == report["resolved_stage_settings"]["sampling_evidence"]
        assert report["sampling_count"] == sampling_count_from_evidence(evidence)
        assert (
            evidence["scanner_stage_implementation_identity"]
            == (report["resolved_stage_settings"]["scanner_stage_implementation_identity"])
        )
        assert set(evidence["sampling_source_implementation_identity"]) == {
            "strike",
            "dip",
        }
    for cell in result.cells:
        manifest_path = (
            output_root / "stages" / "skinning" / cell.stages.skinning / "stage_manifest.json"
        )
        stage_manifest = json.loads(manifest_path.read_text())
        assert (
            stage_manifest["resolved_settings"]["skin_artifact_semantic_contract_version"]
            == F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION
        )

    with (output_root / "reports" / "metrics_long.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        assert {int(row["schema_version"]) for row in csv.DictReader(stream)} == {
            F3_METRIC_SCHEMA_VERSION
        }
