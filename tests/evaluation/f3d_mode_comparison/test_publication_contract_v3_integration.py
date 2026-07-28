from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pyosv._skinner.growth as skinner_growth_module
import pyosv.evaluation.f3d_mode_comparison.result as result_module
import pyosv.evaluation.f3d_mode_comparison.runner as runner_module
import pyosv.evaluation.f3d_mode_comparison.scanner as scanner_module
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
    _complete_small_bundle,
    _controlled_reskinned_primary_workflow,
    _rehash_report,
    _rehash_stage_artifact,
    _reskinned_primary_config,
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


def test_publication_contract_v3_small_fixture_end_to_end(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "fixture-data"
    output_root = tmp_path / "run"
    spec = _write_fixture(data_root, shape=(13, 13, 13))
    calls: Counter[str] = Counter()
    runtime_identity = _fixed_runtime_identity()
    reskin_transitions: list[
        tuple[
            tuple[tuple[np.float32, ...], ...],
            tuple[tuple[np.float32, ...], ...],
        ]
    ] = []
    original_reskin = skinner_growth_module._reskin_reference

    def tracked_reskin(skin: Any, **kwargs: Any) -> Any:
        before = tuple(
            tuple(
                np.float32(value)
                for value in (cell.x1, cell.x2, cell.x3, cell.fl, cell.fp, cell.ft)
            )
            for cell in skin
        )
        result = original_reskin(skin, **kwargs)
        after = tuple(
            tuple(
                np.float32(value)
                for value in (cell.x1, cell.x2, cell.x3, cell.fl, cell.fp, cell.ft)
            )
            for cell in result
        )
        reskin_transitions.append((before, after))
        return result

    monkeypatch.setattr(skinner_growth_module, "_reskin_reference", tracked_reskin)

    first = _run_fixture(
        data_root,
        output_root,
        spec,
        calls,
        resume=False,
        monkeypatch=monkeypatch,
        plan_config=_reskinned_primary_config(),
        workspace_runtime_identity=runtime_identity,
        workflow_runner=_controlled_reskinned_primary_workflow,
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
    assert (
        F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
        F3_RUNTIME_IDENTITY_SCHEMA_VERSION,
        F3_FINGERPRINT_CONTRACT_VERSION,
        F3_SCANNER_STAGE_CONTRACT_VERSION,
        F3_METRIC_SCHEMA_VERSION,
    ) == (3, 3, 4, 5, 2)
    assert reskin_transitions
    assert any(
        len(before) == len(after)
        and Counter((*cell[:3], *cell[4:]) for cell in before)
        != Counter((*cell[:3], *cell[4:]) for cell in after)
        for before, after in reskin_transitions
    )

    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fingerprint_contract_version"] == 4
    assert manifest["runtime_identity"] == runtime_identity
    assert manifest["runtime_identity"]["runtime_identity_schema_version"] == 3
    completion = json.loads((output_root / "completion.json").read_text(encoding="utf-8"))
    assert len(completion["stage_completions"]) == 12

    scanner_fingerprints = {cell.stages.scanner for cell in first.cells}
    scanner_reports: dict[str, dict[str, Any]] = {}
    for fingerprint in scanner_fingerprints:
        report = json.loads(
            (output_root / "stages" / "scanner" / fingerprint / "report.json").read_text(
                encoding="utf-8"
            )
        )
        assert report["scanner_stage_contract_version"] == 5
        scanner_reports[report["backend"]] = report
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
    for report in scanner_reports.values():
        evidence = report["sampling_evidence"]
        assert evidence == report["resolved_stage_settings"]["sampling_evidence"]
        assert evidence["scanner_stage_implementation_identity"] == "small-fixture-scanner-v1"
        assert evidence["sampling_source_implementation_identity"]

    reskinned_cell_count = 0
    for fingerprint in {cell.stages.skinning for cell in first.cells}:
        stage_manifest = json.loads(
            (output_root / "stages" / "skinning" / fingerprint / "stage_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        assert stage_manifest["resolved_settings"]["skin_artifact_semantic_contract_version"] == 3
        stage = output_root / "stages" / "skinning" / fingerprint
        report = json.loads((stage / "report.json").read_text(encoding="utf-8"))
        payload = json.loads((stage / "skins.json").read_text(encoding="utf-8"))
        cells = [item for skin in payload["skins"] for item in skin["cells"]]
        assert report["final_cell_value_provenance"] == "primary_reskinned"
        assert any(skin["cell_count"] > 1 for skin in payload["skins"])
        cell = next(item for item in first.cells if item.stages.skinning == fingerprint)
        voting = output_root / "stages" / "voting" / cell.stages.voting
        shape = spec.shape
        vp = np.fromfile(voting / "vp.dat", dtype=">f4").reshape(shape)
        vt = np.fromfile(voting / "vt.dat", dtype=">f4").reshape(shape)
        assert any(
            np.float32(item["fp"]) != vp[item["i3"], item["i2"], item["i1"]]
            or np.float32(item["ft"]) != vt[item["i3"], item["i2"], item["i1"]]
            for item in cells
        )
        reskinned_cell_count += len(cells)
    assert reskinned_cell_count > 4

    for filename in (
        "metrics_long.csv",
        "contrasts.csv",
        "voxel_contrast_summaries.csv",
    ):
        _, rows = _csv(output_root / "reports" / filename)
        assert {int(row["schema_version"]) for row in rows} == {2}
    evidence = json.loads(
        (output_root / "reports" / "metric_evidence.json").read_text(encoding="utf-8")
    )
    assert {item["schema_version"] for item in evidence["metric_evidence"]} == {2}

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
        plan_config=_reskinned_primary_config(),
        workspace_runtime_identity=runtime_identity,
        workflow_runner=_controlled_reskinned_primary_workflow,
    )
    assert resumed == first
    assert calls - before_resume == Counter({"complete result load": 1})


def test_injected_skin_stage_failure_preserves_inputs_and_cleans_runtime_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "fixture-data"
    output_root = tmp_path / "run"
    spec = _write_fixture(data_root, shape=(13, 13, 13))
    runtime_identity = _fixed_runtime_identity()
    first = _run_fixture(
        data_root,
        output_root,
        spec,
        Counter(),
        resume=False,
        monkeypatch=monkeypatch,
        plan_config=_reskinned_primary_config(),
        workspace_runtime_identity=runtime_identity,
        workflow_runner=_controlled_reskinned_primary_workflow,
    )

    (output_root / "completion.json").unlink()
    missing_skinning = first.cells[0].stages.skinning
    shutil.rmtree(output_root / "stages" / "skinning" / missing_skinning)
    source_before = _tree_bytes(data_root)
    prior_stages_before = _tree_bytes(output_root / "stages")
    completion_existed_before = (output_root / "completion.json").exists()

    original_memmap = np.memmap
    opened_memmaps: list[np.memmap] = []

    class TrackedMemmap(original_memmap):  # type: ignore[misc, valid-type]
        def __new__(cls, *args: Any, **kwargs: Any) -> np.memmap:
            array = super().__new__(cls, *args, **kwargs)
            opened_memmaps.append(array)
            return array

    original_clear = runner_module.PipelineStageCache.clear
    cleared_caches: list[tuple[Any, bool]] = []

    def tracked_clear(cache: Any) -> None:
        had_partial_skin = bool(cache._primary_skinning)
        original_clear(cache)
        cleared_caches.append((cache, had_partial_skin))

    original_write_json = runner_module._write_json

    def fail_skin_payload(path: Path, payload: Any) -> None:
        if path.name == "skins.json":
            raise OSError("injected skin artifact failure")
        original_write_json(path, payload)

    monkeypatch.setattr(np, "memmap", TrackedMemmap)
    monkeypatch.setattr(runner_module.PipelineStageCache, "clear", tracked_clear)
    monkeypatch.setattr(runner_module, "_write_json", fail_skin_payload)

    with pytest.raises(OSError, match="injected skin artifact failure"):
        _run_fixture(
            data_root,
            output_root,
            spec,
            Counter(),
            resume=True,
            monkeypatch=monkeypatch,
            plan_config=_reskinned_primary_config(),
            workspace_runtime_identity=runtime_identity,
            workflow_runner=_controlled_reskinned_primary_workflow,
        )

    assert completion_existed_before is False
    assert not (output_root / "completion.json").exists()
    assert _tree_bytes(data_root) == source_before
    assert _tree_bytes(output_root / "stages") == prior_stages_before
    assert opened_memmaps
    assert all(array._mmap.closed for array in opened_memmaps)
    assert any(had_partial_skin for _, had_partial_skin in cleared_caches)
    assert all(
        not (
            cache._seeds
            or cache._voting
            or cache._thinning
            or cache._final_thinning
            or cache._primary_skinning
        )
        for cache, _ in cleared_caches
    )
    assert not any(
        ".tmp-" in path.name
        or path.name.startswith((".pyosv-stage-tmp-", ".cell-tmp-"))
        or path.name.endswith(".partial")
        for path in output_root.rglob("*")
    )


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
    first = _run_fixture(
        data_root,
        output_root,
        spec,
        Counter(),
        resume=False,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=runtime_identity,
    )
    (output_root / "completion.json").unlink()
    missing_skinning = first.cells[0].stages.skinning
    shutil.rmtree(output_root / "stages" / "skinning" / missing_skinning)
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

    calls = Counter()
    resumed = _run_fixture(
        data_root,
        output_root,
        spec,
        calls,
        resume=True,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=runtime_identity,
    )
    assert resumed.cells == first.cells
    assert calls["workflow callback"] == 1
    assert calls["reference-like scan"] == 0
    assert calls["quality scan"] == 0
    assert (output_root / "completion.json").is_file()


@pytest.mark.parametrize(
    ("mode", "disable_jit", "expected"),
    (
        (
            "auto",
            "0",
            {
                "available": True,
                "jit": {"status": "enabled", "enabled": True},
                "state": "numba_jit_enabled",
            },
        ),
        (
            "auto",
            "1",
            {
                "available": True,
                "jit": {"status": "disabled", "enabled": False},
                "state": "numba_jit_disabled",
                "publication_accepted": False,
            },
        ),
        (
            "off",
            "0",
            {
                "available": False,
                "jit": {"status": "not_applicable", "enabled": None},
                "state": "python_only",
                "publication_accepted": False,
            },
        ),
    ),
)
def test_fresh_process_runtime_identity_and_publication_preflight(
    tmp_path: Path,
    mode: str,
    disable_jit: str,
    expected: dict[str, Any],
) -> None:
    fake_root = tmp_path / "fake-modules"
    package = fake_root / "numba"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        textwrap.dedent(
            """
            import os
            from types import SimpleNamespace

            __version__ = "test-fresh-numba"
            config = SimpleNamespace(
                DISABLE_JIT=int(os.environ.get("NUMBA_DISABLE_JIT") or "0")
            )

            def njit(*args, **kwargs):
                if len(args) == 1 and callable(args[0]) and not kwargs:
                    return args[0]
                return lambda function: function
            """
        ),
        encoding="utf-8",
    )
    source_root = Path(__file__).resolve().parents[3] / "src"
    output_root = tmp_path / "must-not-start"
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join((str(fake_root), str(source_root))),
            "PYOSV_ACCEL": mode,
            "NUMBA_DISABLE_JIT": disable_jit,
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "NUMBA_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import json
                import sys
                from pathlib import Path

                from pyosv.cli.f3d_mode_comparison import run_experiment
                from pyosv.evaluation.f3d_mode_comparison import (
                    F3ModeComparisonConfig,
                    numerical_runtime_identity,
                    validate_publication_runtime_identity,
                )

                identity = numerical_runtime_identity()
                publication_error = None
                try:
                    validate_publication_runtime_identity(identity)
                except ValueError as error:
                    publication_error = str(error)
                run_error = None
                if identity["effective_acceleration_state"] != "numba_jit_enabled":
                    try:
                        run_experiment(
                            config=F3ModeComparisonConfig(),
                            data_root=Path(sys.argv[1]),
                            output_dir=Path(sys.argv[2]),
                            resume=False,
                            deep=False,
                        )
                    except Exception as error:
                        run_error = {
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                print(json.dumps({
                    "available": identity["numba_available"],
                    "jit": identity["numba_jit"],
                    "state": identity["effective_acceleration_state"],
                    "publication_accepted": publication_error is None,
                    "publication_error": publication_error,
                    "run_error": run_error,
                }))
                """
            ),
            str(tmp_path / "missing-data"),
            str(output_root),
        ],
        cwd=source_root.parent,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)

    assert {name: observed[name] for name in expected} == expected
    if expected["state"] != "numba_jit_enabled":
        assert observed["publication_error"].startswith("publication runtime contract violation:")
        assert observed["run_error"]["type"] == "ValueError"
        assert observed["run_error"]["message"].startswith(
            "publication runtime contract violation:"
        )
        assert not output_root.exists()
    else:
        assert observed["run_error"] is None


def test_canonical_scanner_sampling_tamper_reaches_deep_rederivation(
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
    root = _complete_boundary_skin_bundle(tmp_path)
    loaded = load_f3d_mode_comparison_result(root, _dataset_spec=fixture_spec)
    scanner_fingerprints = {cell.stages.scanner for cell in loaded.cells}
    fingerprint_by_backend = {cell.backend: cell.stages.scanner for cell in loaded.cells}
    sampling_rederivations: Counter[str] = Counter()
    original_sampling = result_module.scanner_sampling_evidence

    def tracked_sampling(
        scanner: Any,
        config: Any,
        backend: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        sampling_rederivations[fingerprint_by_backend[backend]] += 1
        return original_sampling(scanner, config, backend, **kwargs)

    monkeypatch.setattr(result_module, "scanner_sampling_evidence", tracked_sampling)
    monkeypatch.setattr(scanner_module, "scanner_sampling_evidence", tracked_sampling)
    assert validate_completed_f3d_bundle(root, deep=True, _dataset_spec=fixture_spec)
    assert sampling_rederivations == Counter(
        {fingerprint: 1 for fingerprint in scanner_fingerprints}
    )

    stage = root / "stages" / "scanner" / loaded.cells[0].stages.scanner
    report_path = stage / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    evidence = deepcopy(report["sampling_evidence"])
    evidence["strike"]["sha256"] = _flip_sha256(evidence["strike"]["sha256"])
    report["sampling_evidence"] = evidence
    report["resolved_stage_settings"]["sampling_evidence"] = deepcopy(evidence)
    assert report["sampling_evidence"] == report["resolved_stage_settings"]["sampling_evidence"]
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    _rehash_stage_artifact(root, stage, "report.json")

    sampling_rederivations.clear()
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))

    with pytest.raises(F3ResultValidationError, match="deep scanner sampling evidence mismatch"):
        result_module._deep_validate_scanner_stages(root, loaded, manifest["plan"])
    assert sampling_rederivations == Counter({loaded.cells[0].stages.scanner: 1})


def test_rehashed_reskinned_cell_tampering_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (13, 13, 13)
    fixture_spec = F3DatasetSpec(
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
    monkeypatch.setattr(result_module, "OFFICIAL_F3_DATASET_SPEC", fixture_spec)
    base_parent = tmp_path / "base"
    base_parent.mkdir()
    base = _complete_small_bundle(
        base_parent,
        shape=shape,
        config=_reskinned_primary_config(),
        workflow_runner=_controlled_reskinned_primary_workflow,
    )
    source_before = _tree_bytes(base_parent / "data")

    for case in ("skin-subvoxel", "skin-attributes", "coherent-skin-artifacts"):
        root = tmp_path / case
        shutil.copytree(base, root)
        loaded = load_f3d_mode_comparison_result(root, _dataset_spec=fixture_spec)
        stage = root / "stages" / "skinning" / loaded.cells[0].stages.skinning
        report = json.loads((stage / "report.json").read_text(encoding="utf-8"))
        assert report["final_cell_value_provenance"] == "primary_reskinned"

        path = stage / "skins.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        skin = payload["skins"][0]
        cell = skin["cells"][0]
        if case == "coherent-skin-artifacts":
            assert skin["cell_count"] > 1
            changed_cell = skin["cells"][-1]
            assert (changed_cell["i1"], changed_cell["i2"], changed_cell["i3"]) != (
                cell["i1"],
                cell["i2"],
                cell["i3"],
            )
            for field in ("x1", "x2", "x3", "i1", "i2", "i3"):
                changed_cell[field] = cell[field]
            skin["cells"].sort(key=lambda item: (item["i3"], item["i2"], item["i1"]))
        elif case == "skin-subvoxel":
            cell["x1"] = float(cell["x1"]) + 0.1
        else:
            cell["fl"] = 0.0 if float(cell["fl"]) > 0.5 else 1.0
            cell["fp"] = float(cell["fp"]) + 0.01
            cell["ft"] = (
                float(cell["ft"]) + 0.01 if float(cell["ft"]) < 89.0 else float(cell["ft"]) - 0.01
            )
        path.write_bytes(canonical_json_bytes(payload) + b"\n")
        _rehash_stage_artifact(root, stage, "skins.json")

        if case == "coherent-skin-artifacts":
            parsed = result_module.parse_skins_json(path, shape)
            mask = np.zeros(shape, dtype=np.float32)
            for i1, i2, i3 in parsed.unique_indices:
                mask[i3, i2, i1] = np.float32(1.0)
            mask_path = stage / "skin_mask.dat"
            mask.astype(">f4").tofile(mask_path)
            _rehash_stage_artifact(root, stage, "skin_mask.dat")

            report_path = stage / "report.json"
            small_skin_size = int(report["topology"]["small_skin_size"])
            topology = runner_module.skin_topology_metrics(
                parsed.skins,
                shape,
                small_skin_size=small_skin_size,
            )
            report["topology"] = topology
            report["diagnostics"]["accepted_skin_count"] = topology["skin_count"]
            report["diagnostics"]["accepted_cell_count"] = topology["cell_count"]
            report_path.write_bytes(canonical_json_bytes(report) + b"\n")
            _rehash_stage_artifact(root, stage, "report.json")
            result_module.validate_skin_artifact_semantics(
                stage,
                shape,
                small_skin_size=small_skin_size,
                parsed=parsed,
            )
            resources_path = root / "reports" / "resources.json"
            resources = json.loads(resources_path.read_text(encoding="utf-8"))
            resources["storage"] = [row.as_dict() for row in result_module.storage_report(root)]
            resources_path.write_bytes(canonical_json_bytes(resources) + b"\n")
            _rehash_report(root, "resources.json")

        before_validation = _tree_bytes(root)
        paths_before_validation = _tree_paths(root)
        expected_error = (
            "skins.json does not exactly match skin-only recomputation"
            if case == "coherent-skin-artifacts"
            else None
        )
        with pytest.raises(F3ResultValidationError, match=expected_error):
            validate_completed_f3d_bundle(root, deep=True, _dataset_spec=fixture_spec)
        assert _tree_bytes(root) == before_validation
        assert _tree_paths(root) == paths_before_validation
        assert _tree_bytes(base_parent / "data") == source_before
        assert not any(path.name.endswith((".tmp", ".partial")) for path in root.rglob("*"))


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
        "skin-provenance",
        "fallback-parent-fvt",
        "fallback-parent-vp",
        "fallback-parent-vt",
        "duplicate-json-key",
        "scanner-summary",
        "scanner-sampling",
        "scanner-sampling-digest",
        "scanner-sampling-implementation",
        "scanner-contract",
        "metric-schema",
        "runtime-effective-jit",
        "runtime-schema",
    ):
        root = tmp_path / case
        shutil.copytree(base, root)
        loaded = load_f3d_mode_comparison_result(root, _dataset_spec=fixture_spec)

        if case == "duplicate-json-key":
            stage = root / "stages" / "skinning" / loaded.cells[0].stages.skinning
            path = stage / "skins.json"
            text = path.read_text(encoding="utf-8")
            start = text.index('"x1":')
            end = text.index(",", start)
            entry = text[start:end]
            path.write_text(text[: end + 1] + entry + "," + text[end + 1 :])
            _rehash_stage_artifact(root, stage, "skins.json")
        elif case == "skin-provenance":
            stage = root / "stages" / "skinning" / loaded.cells[0].stages.skinning
            path = stage / "report.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["final_cell_value_provenance"] = "primary_nearest_sample"
            path.write_bytes(canonical_json_bytes(payload) + b"\n")
            _rehash_stage_artifact(root, stage, "report.json")
        elif case.startswith("fallback-parent-"):
            skinning_stage = root / "stages" / "skinning" / loaded.cells[0].stages.skinning
            skinning_report = json.loads(
                (skinning_stage / "report.json").read_text(encoding="utf-8")
            )
            assert skinning_report["diagnostics"]["fallback_used"] is True
            assert skinning_report["final_cell_value_provenance"] == "connected_component_fallback"
            stage_kind, filename, replacement = {
                "fallback-parent-fvt": ("thinning", "fvt.dat", 0.75),
                "fallback-parent-vp": ("voting", "vp.dat", 21.0),
                "fallback-parent-vt": ("voting", "vt.dat", 71.0),
            }[case]
            fingerprint = getattr(loaded.cells[0].stages, stage_kind)
            stage = root / "stages" / stage_kind / fingerprint
            path = stage / filename
            values = np.fromfile(path, dtype=">f4").reshape(loaded.volume_shape)
            values[0, 0, 0] = np.float32(replacement)
            values.tofile(path)
            _rehash_stage_artifact(root, stage, filename)
        elif case.startswith("scanner-"):
            stage = root / "stages" / "scanner" / loaded.cells[0].stages.scanner
            path = stage / "report.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            if case == "scanner-summary":
                payload["raw"]["ft"]["mean"] += 0.01
            elif case == "scanner-sampling":
                payload["sampling_count"]["strike"] += 1
                payload["sampling_count"]["orientations"] = (
                    payload["sampling_count"]["strike"] * payload["sampling_count"]["dip"]
                )
            elif case == "scanner-sampling-digest":
                payload["sampling_evidence"]["strike"]["sha256"] = _flip_sha256(
                    payload["sampling_evidence"]["strike"]["sha256"]
                )
            elif case == "scanner-sampling-implementation":
                payload["sampling_evidence"]["sampling_source_implementation_identity"]["strike"][
                    "module"
                ] = "tampered.scanner"
            else:
                payload["scanner_stage_contract_version"] = 4
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
            if case == "runtime-effective-jit":
                payload["runtime_identity"]["effective_acceleration_state"] = "numba_jit_disabled"
                payload["runtime_identity"]["numba_jit"] = {
                    "status": "disabled",
                    "enabled": False,
                }
                payload["runtime_identity"]["numba_environment"]["NUMBA_DISABLE_JIT"] = "1"
            else:
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
        expected_error = None
        if case == "runtime-schema":
            expected_error = "runtime identity schema version must equal 3"
        elif case.startswith("fallback-parent-"):
            expected_error = "cell (fl|fp|ft) does not match parent volume"
        with pytest.raises((F3ResultValidationError, ValueError), match=expected_error):
            validate_completed_f3d_bundle(root, deep=True, _dataset_spec=fixture_spec)
        assert _tree_bytes(root) == before_validation
        assert _tree_paths(root) == paths_before_validation
        assert _tree_bytes(tmp_path / "base" / "data") == source_before
        assert not any(path.name.endswith((".tmp", ".partial")) for path in root.rglob("*"))
