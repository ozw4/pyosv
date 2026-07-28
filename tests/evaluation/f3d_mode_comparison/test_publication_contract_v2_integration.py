from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pyosv.evaluation.f3d_mode_comparison.result as result_module
import pyosv.evaluation.workflow3d as workflow3d_module
from pyosv.evaluation.f3d_mode_comparison import (
    F3DatasetSpec,
    F3_FINGERPRINT_CONTRACT_VERSION,
    F3_METRIC_SCHEMA_VERSION,
    F3_RUNTIME_IDENTITY_SCHEMA_VERSION,
    F3_SCANNER_STAGE_CONTRACT_VERSION,
    F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
    F3ResultValidationError,
    F3WorkspaceMismatchError,
    canonical_fingerprint,
    canonical_json_bytes,
    load_f3d_mode_comparison_result,
    validate_completed_f3d_bundle,
)

from .test_bundle_validation import (
    _complete_boundary_skin_bundle,
    _rehash_report,
    _rehash_stage_artifact,
    _write_csv_rows,
)
from .test_integration import _csv, _run_fixture, _write_fixture


def _fixed_runtime_identity() -> dict[str, Any]:
    return {
        "runtime_identity_schema_version": F3_RUNTIME_IDENTITY_SCHEMA_VERSION,
        "python_implementation": "CPython",
        "platform_system": "Linux",
        "platform_machine": "test-machine",
        "byte_order": "little",
        "requested_acceleration_mode": "auto",
        "pyosv_accel": "auto",
        "numba_available": True,
        "numba_version": "test-numba",
        "numba_jit": {
            "status": "enabled",
            "enabled": True,
        },
        "effective_acceleration_state": "numba_jit_enabled",
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "GOTO_NUM_THREADS": None,
            "BLIS_NUM_THREADS": None,
            "VECLIB_MAXIMUM_THREADS": None,
        },
        "python_hash_seed": "0",
        "numpy_disable_cpu_features": None,
        "numba_environment": {
            "NUMBA_DISABLE_JIT": "0",
            "NUMBA_NUM_THREADS": "1",
            "NUMBA_THREADING_LAYER": None,
            "NUMBA_CPU_NAME": None,
            "NUMBA_CPU_FEATURES": None,
        },
        "openblas_coretype": None,
        "numpy_build": {
            "status": "available",
            "sha256": hashlib.sha256(b"test-numpy-build").hexdigest(),
        },
        "numpy_runtime_cpu": {
            "status": "available",
            "features": ["AVX2", "SSE2"],
        },
        "numpy_runtime_blas": {
            "status": "available",
            "libraries": [
                {
                    "implementation": "openblas",
                    "version": "test-openblas",
                    "threading_layer": "pthreads",
                    "architecture": "test-architecture",
                    "effective_thread_count": 1,
                }
            ],
        },
        "scipy_build": {
            "status": "available",
            "sha256": hashlib.sha256(b"test-scipy-build").hexdigest(),
        },
    }


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _tree_paths(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*")}


def _flip_sha256(value: str) -> str:
    return value[:-1] + ("0" if value[-1] != "0" else "1")


def test_publication_contract_v2_small_fixture_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "fixture-data"
    output_root = tmp_path / "run"
    spec = _write_fixture(data_root)
    calls: Counter[str] = Counter()
    runtime_identity = _fixed_runtime_identity()

    first = _run_fixture(
        data_root,
        output_root,
        spec,
        calls,
        resume=False,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=runtime_identity,
    )

    assert [cell.label for cell in first.cells] == [
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    ]
    assert len({cell.stages.scanner for cell in first.cells}) == 2
    assert len({cell.stages.voting for cell in first.cells}) == 2
    assert len({cell.stages.thinning for cell in first.cells}) == 4
    assert len({cell.stages.skinning for cell in first.cells}) == 4

    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fingerprint_contract_version"] == F3_FINGERPRINT_CONTRACT_VERSION
    assert manifest["runtime_identity"] == runtime_identity
    assert (
        manifest["runtime_identity"]["runtime_identity_schema_version"]
        == F3_RUNTIME_IDENTITY_SCHEMA_VERSION
    )
    completion = json.loads((output_root / "completion.json").read_text(encoding="utf-8"))
    assert len(completion["stage_completions"]) == 12

    scanner_fingerprints = {cell.stages.scanner for cell in first.cells}
    for fingerprint in scanner_fingerprints:
        report = json.loads(
            (output_root / "stages" / "scanner" / fingerprint / "report.json").read_text(
                encoding="utf-8"
            )
        )
        assert report["scanner_stage_contract_version"] == F3_SCANNER_STAGE_CONTRACT_VERSION
    for fingerprint in {cell.stages.skinning for cell in first.cells}:
        stage_manifest = json.loads(
            (output_root / "stages" / "skinning" / fingerprint / "stage_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert (
            stage_manifest["resolved_settings"]["skin_artifact_semantic_contract_version"]
            == F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION
        )

    for filename in (
        "metrics_long.csv",
        "contrasts.csv",
        "voxel_contrast_summaries.csv",
    ):
        _, rows = _csv(output_root / "reports" / filename)
        assert {int(row["schema_version"]) for row in rows} == {F3_METRIC_SCHEMA_VERSION}
    evidence = json.loads(
        (output_root / "reports" / "metric_evidence.json").read_text(encoding="utf-8")
    )
    assert {item["schema_version"] for item in evidence["metric_evidence"]} == {
        F3_METRIC_SCHEMA_VERSION
    }

    assert validate_completed_f3d_bundle(output_root, _dataset_spec=spec)

    scanner_summaries: Counter[str] = Counter()
    skin_recomputations = 0
    reference_pairs: Counter[tuple[str, str]] = Counter()
    upstream_calls: Counter[str] = Counter()
    original_summary = result_module.scanner_array_summary
    original_skinning = result_module.execute_skinning_phase3d
    original_reference_metrics = result_module.compute_reference_metric_rows

    def tracked_summary(values: np.ndarray) -> dict[str, Any]:
        scanner_summaries[Path(values.filename).parent.name] += 1  # type: ignore[attr-defined]
        return original_summary(values)

    def tracked_skinning(**kwargs: Any) -> Any:
        nonlocal skin_recomputations
        skin_recomputations += 1
        return original_skinning(**kwargs)

    def tracked_reference_metrics(**kwargs: Any) -> Any:
        reference_pairs[(kwargs["cell_label"], kwargs["stage"])] += 1
        return original_reference_metrics(**kwargs)

    def unexpected_upstream_compute(name: str) -> Any:
        def unexpected(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            upstream_calls[name] += 1
            raise AssertionError("deep validation recomputed an upstream numerical stage")

        return unexpected

    monkeypatch.setattr(result_module, "scanner_array_summary", tracked_summary)
    monkeypatch.setattr(result_module, "execute_skinning_phase3d", tracked_skinning)
    monkeypatch.setattr(result_module, "compute_reference_metric_rows", tracked_reference_metrics)
    monkeypatch.setattr(
        result_module.FaultOrientScanner3,
        "scan",
        unexpected_upstream_compute("scanner"),
    )
    monkeypatch.setattr(
        result_module.FaultOrientScanner3,
        "scan_quality",
        unexpected_upstream_compute("scanner"),
    )
    monkeypatch.setattr(
        result_module.FaultOrientScanner3,
        "thin",
        unexpected_upstream_compute("scanner thinning"),
    )
    monkeypatch.setattr(
        workflow3d_module.OptimalSurfaceVoter,
        "apply_voting_from_seeds",
        unexpected_upstream_compute("voting"),
    )
    monkeypatch.setattr(
        workflow3d_module.OptimalSurfaceVoter,
        "thin",
        unexpected_upstream_compute("base thinning"),
    )

    assert validate_completed_f3d_bundle(output_root, deep=True, _dataset_spec=spec)
    assert set(scanner_summaries) == scanner_fingerprints
    assert sorted(scanner_summaries.values()) == [6, 7]
    assert skin_recomputations == 4
    assert reference_pairs == Counter(
        (cell.label, stage) for cell in first.cells for stage in ("ft", "fv", "fvt")
    )
    assert not upstream_calls

    before_resume = calls.copy()
    resumed = _run_fixture(
        data_root,
        output_root,
        spec,
        calls,
        resume=True,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=runtime_identity,
    )
    assert resumed == first
    assert calls - before_resume == Counter({"complete result load": 1})


def test_runtime_identity_changes_reject_resume_without_workspace_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "fixture-data"
    output_root = tmp_path / "run"
    spec = _write_fixture(data_root)
    runtime_identity = _fixed_runtime_identity()
    assert runtime_identity["numpy_runtime_cpu"]["status"] == "available"
    assert runtime_identity["scipy_build"]["status"] == "available"
    _run_fixture(
        data_root,
        output_root,
        spec,
        Counter(),
        resume=False,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=runtime_identity,
    )
    workspace_before = _tree_bytes(output_root)

    changed_identities = []
    cpu_changed = deepcopy(runtime_identity)
    cpu_changed["numpy_runtime_cpu"]["features"] = sorted(
        {*cpu_changed["numpy_runtime_cpu"]["features"], "PYOSV_TEST_FEATURE"}
    )
    changed_identities.append(cpu_changed)
    scipy_changed = deepcopy(runtime_identity)
    scipy_changed["scipy_build"]["sha256"] = _flip_sha256(scipy_changed["scipy_build"]["sha256"])
    changed_identities.append(scipy_changed)
    threads_changed = deepcopy(runtime_identity)
    threads_changed["thread_environment"]["OMP_NUM_THREADS"] = "2"
    changed_identities.append(threads_changed)

    for changed in changed_identities:
        calls: Counter[str] = Counter()
        with pytest.raises(F3WorkspaceMismatchError):
            _run_fixture(
                data_root,
                output_root,
                spec,
                calls,
                resume=True,
                monkeypatch=monkeypatch,
                workspace_runtime_identity=changed,
            )
        assert not calls
        assert _tree_bytes(output_root) == workspace_before


def test_rehashed_contract_tampering_is_rejected_without_validation_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_spec = F3DatasetSpec(
        dataset_id="result-fixture",
        shape=(3, 4, 5),
        files=(
            ("input", "ep.dat"),
            ("reference_fault_likelihood", "fl.dat"),
            ("reference_fault_votes", "fv.dat"),
            ("reference_thinned_fault_votes", "fvt.dat"),
        ),
        expected_bytes=3 * 4 * 5 * np.dtype(">f4").itemsize,
    )
    monkeypatch.setattr(result_module, "OFFICIAL_F3_DATASET_SPEC", fixture_spec)
    base_parent = tmp_path / "base"
    base_parent.mkdir()
    base = _complete_boundary_skin_bundle(base_parent)
    source_before = _tree_bytes(tmp_path / "base" / "data")

    for case in (
        "skin-subvoxel",
        "skin-attributes",
        "duplicate-json-key",
        "scanner-summary",
        "scanner-sampling",
        "metric-schema",
        "runtime-schema",
    ):
        root = tmp_path / case
        shutil.copytree(base, root)
        loaded = load_f3d_mode_comparison_result(root, _dataset_spec=fixture_spec)

        if case.startswith("skin-") or case == "duplicate-json-key":
            stage = root / "stages" / "skinning" / loaded.cells[0].stages.skinning
            path = stage / "skins.json"
            if case == "duplicate-json-key":
                text = path.read_text(encoding="utf-8")
                start = text.index('"x1":')
                end = text.index(",", start)
                entry = text[start:end]
                path.write_text(text[: end + 1] + entry + "," + text[end + 1 :])
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                cell = payload["skins"][0]["cells"][0]
                if case == "skin-subvoxel":
                    cell["x1"] = float(cell["x1"]) + 0.1
                else:
                    cell["fl"] = 0.0 if float(cell["fl"]) > 0.5 else 1.0
                    cell["fp"] = float(cell["fp"]) + 0.01
                    cell["ft"] = (
                        float(cell["ft"]) + 0.01
                        if float(cell["ft"]) < 89.0
                        else float(cell["ft"]) - 0.01
                    )
                path.write_bytes(canonical_json_bytes(payload) + b"\n")
            _rehash_stage_artifact(root, stage, "skins.json")
        elif case.startswith("scanner-"):
            stage = root / "stages" / "scanner" / loaded.cells[0].stages.scanner
            path = stage / "report.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if case == "scanner-summary":
                payload["raw"]["ft"]["mean"] += 0.01
            else:
                payload["sampling_count"]["strike"] += 1
                payload["sampling_count"]["orientations"] = (
                    payload["sampling_count"]["strike"] * payload["sampling_count"]["dip"]
                )
            path.write_bytes(canonical_json_bytes(payload) + b"\n")
            _rehash_stage_artifact(root, stage, "report.json")
        elif case == "metric-schema":
            path = root / "reports" / "metrics_long.csv"
            with path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                assert reader.fieldnames is not None
                fieldnames = reader.fieldnames
                rows = list(reader)
            rows[0]["schema_version"] = "1"
            _write_csv_rows(path, fieldnames, rows)
            _rehash_report(root, "metrics_long.csv")
        else:
            path = root / "run_manifest.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["runtime_identity"]["runtime_identity_schema_version"] = 2
            computation = {name: payload[name] for name in result_module._RUN_COMPUTATION_FIELDS}
            payload["run_fingerprint"] = canonical_fingerprint(computation)
            path.write_bytes(canonical_json_bytes(payload) + b"\n")
            completion_path = root / "completion.json"
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            completion["run_fingerprint"] = payload["run_fingerprint"]
            completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")

        before_validation = _tree_bytes(root)
        paths_before_validation = _tree_paths(root)
        expected_error = (
            "runtime identity schema version must equal 3" if case == "runtime-schema" else None
        )
        with pytest.raises((F3ResultValidationError, ValueError), match=expected_error):
            validate_completed_f3d_bundle(root, deep=True, _dataset_spec=fixture_spec)
        assert _tree_bytes(root) == before_validation
        assert _tree_paths(root) == paths_before_validation
        assert _tree_bytes(tmp_path / "base" / "data") == source_before
        assert not any(path.name.endswith((".tmp", ".partial")) for path in root.rglob("*"))
