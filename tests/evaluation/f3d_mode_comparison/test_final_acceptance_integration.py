from __future__ import annotations

import gc
import json
import shutil
import weakref
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import pyosv.evaluation.f3d_mode_comparison.result as result_module
from pyosv.evaluation.f3d_mode_comparison import data as data_module
from pyosv.evaluation.f3d_mode_comparison import (
    F3_DATASET_ID,
    F3_FINGERPRINT_CONTRACT_VERSION,
    F3_METRIC_SCHEMA_VERSION,
    F3_RUNTIME_IDENTITY_SCHEMA_VERSION,
    F3_SCANNER_STAGE_CONTRACT_VERSION,
    F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
    F3ResultValidationError,
    scanner_sampling_evidence,
    validate_completed_f3d_bundle,
)

from .test_integration import (
    _DeterministicScanner,
    _fixture_plan,
    _run_fixture,
    _write_fixture,
)
from .test_publication_contract_v3_integration import (
    _fixed_runtime_identity,
    _tree_bytes,
    _tree_paths,
)


def _official_fixture(data_root: Path):
    return replace(_write_fixture(data_root), dataset_id=F3_DATASET_ID)


def _sampling_contract(spec: Any, calls: Counter[str]) -> dict[str, dict[str, Any]]:
    plan = _fixture_plan(spec)
    provider = _TrackedScanner(0.0, 0.0, calls)
    return {
        backend: scanner_sampling_evidence(
            provider,
            plan.scanner_config_for(backend),
            backend,
            implementation_identity="final-acceptance-scanner-v1",
        )
        for backend in ("reference-like", "quality")
    }


class _TrackedScanner(_DeterministicScanner):
    def __init__(self, sigma1: float, sigma2: float, calls: Counter[str]) -> None:
        calls["scanner factory"] += 1
        super().__init__(sigma1, sigma2, calls)

    def reference_like_strike_sampling(self, *args: Any):
        self._calls["instance sampling helper"] += 1
        return super().reference_like_strike_sampling(*args)

    def reference_like_dip_sampling(self, *args: Any):
        self._calls["instance sampling helper"] += 1
        return super().reference_like_dip_sampling(*args)

    def refined_reference_like_strike_sampling(self, *args: Any, **kwargs: Any):
        self._calls["instance sampling helper"] += 1
        return super().refined_reference_like_strike_sampling(*args, **kwargs)

    def refined_reference_like_dip_sampling(self, *args: Any, **kwargs: Any):
        self._calls["instance sampling helper"] += 1
        return super().refined_reference_like_dip_sampling(*args, **kwargs)


class _MismatchedScanner(_TrackedScanner):
    def refined_reference_like_strike_sampling(self, *args: Any, **kwargs: Any):
        values = super().refined_reference_like_strike_sampling(*args, **kwargs)
        values[-1] += 1.0
        return values


def _factory(calls: Counter[str]):
    return lambda sigma1, sigma2: _TrackedScanner(sigma1, sigma2, calls)


def _generate_official_bundle(
    data_root: Path,
    output_root: Path,
    spec: Any,
    runtime_identity: dict[str, Any],
    calls: Counter[str],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    monkeypatch.setattr(result_module, "numerical_runtime_identity", lambda: runtime_identity)
    evidence_calls: Counter[str] = Counter()
    evidence = _sampling_contract(spec, evidence_calls)
    assert evidence_calls == Counter(
        {
            "scanner factory": 1,
            "instance sampling helper": 4,
        }
    )
    monkeypatch.setattr(
        result_module,
        "canonical_scanner_implementation_identity",
        lambda: "final-acceptance-scanner-v1",
    )
    monkeypatch.setattr(
        result_module,
        "canonical_scanner_sampling_evidence",
        lambda config, backend, **kwargs: evidence[backend],
    )
    return _run_fixture(
        data_root,
        output_root,
        spec,
        calls,
        resume=False,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=runtime_identity,
        scanner_implementation_identity="final-acceptance-scanner-v1",
        scanner_factory=_factory(calls),
        sampling_evidence_by_backend=evidence,
        finalization_deep=False,
    )


def test_official_small_fixture_first_run_shallow_deep_and_complete_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "bundle"
    spec = _official_fixture(data_root)
    runtime_identity = _fixed_runtime_identity()
    calls: Counter[str] = Counter()

    first = _generate_official_bundle(
        data_root,
        output_root,
        spec,
        runtime_identity,
        calls,
        monkeypatch,
    )

    assert [cell.label for cell in first.cells] == [
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    ]
    assert calls["scanner factory"] == 2
    assert calls["instance sampling helper"] == 4
    assert calls["reference-like scan"] == 1
    assert calls["quality scan"] == 1
    assert calls["scanner thinning"] == 2
    manifest = json.loads((output_root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime_identity"] == runtime_identity
    assert (
        F3_RUNTIME_IDENTITY_SCHEMA_VERSION,
        F3_FINGERPRINT_CONTRACT_VERSION,
        F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
        F3_SCANNER_STAGE_CONTRACT_VERSION,
        F3_METRIC_SCHEMA_VERSION,
    ) == (3, 4, 3, 5, 2)

    monkeypatch.setattr(
        result_module,
        "numerical_runtime_identity",
        lambda: pytest.fail("shallow validation inspected the current runtime"),
    )
    assert validate_completed_f3d_bundle(output_root, _dataset_spec=spec)

    deep_calls: Counter[str] = Counter()
    original_summary = result_module.scanner_array_summary
    original_skinning = result_module.execute_skinning_phase3d
    original_metrics = result_module.compute_reference_metric_rows

    def tracked_summary(values: Any) -> Any:
        deep_calls["scanner DAT summary"] += 1
        return original_summary(values)

    def tracked_skinning(**kwargs: Any) -> Any:
        deep_calls["skin-only replay"] += 1
        return original_skinning(**kwargs)

    def tracked_metrics(**kwargs: Any) -> Any:
        deep_calls["reference metrics"] += 1
        return original_metrics(**kwargs)

    monkeypatch.setattr(result_module, "numerical_runtime_identity", lambda: runtime_identity)
    monkeypatch.setattr(result_module, "scanner_array_summary", tracked_summary)
    monkeypatch.setattr(result_module, "execute_skinning_phase3d", tracked_skinning)
    monkeypatch.setattr(result_module, "compute_reference_metric_rows", tracked_metrics)
    assert validate_completed_f3d_bundle(output_root, deep=True, _dataset_spec=spec)
    assert deep_calls["scanner DAT summary"] > 0
    assert deep_calls["skin-only replay"] == 4
    assert deep_calls["reference metrics"] == 12

    native_reads: Counter[str] = Counter()
    original_read_native = data_module.F3VolumeSource.read_native_volume

    def tracked_read_native(source: Any, role: str) -> Any:
        native_reads[role] += 1
        return original_read_native(source, role)

    monkeypatch.setattr(data_module.F3VolumeSource, "read_native_volume", tracked_read_native)
    before = calls.copy()
    resumed = _run_fixture(
        data_root,
        output_root,
        spec,
        calls,
        resume=True,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=runtime_identity,
        scanner_factory=lambda *_: pytest.fail("complete resume constructed a scanner"),
        sampling_evidence_by_backend={},
    )
    assert resumed == first
    assert calls - before == Counter({"complete result load": 1})
    assert not native_reads


@pytest.mark.parametrize(
    ("case", "mutate", "match"),
    (
        (
            "jit-disabled",
            lambda identity: (
                identity.__setitem__("effective_acceleration_state", "numba_jit_disabled"),
                identity.__setitem__("numba_jit", {"status": "disabled", "enabled": False}),
                identity["numba_environment"].__setitem__("NUMBA_DISABLE_JIT", "1"),
            ),
            "JIT must be enabled",
        ),
        (
            "hash-seed",
            lambda identity: identity.__setitem__("python_hash_seed", "1"),
            "PYTHONHASHSEED must equal 0",
        ),
        (
            "thread-count",
            lambda identity: identity["thread_environment"].__setitem__("OMP_NUM_THREADS", "2"),
            "OMP_NUM_THREADS must equal 1",
        ),
    ),
)
def test_invalid_recorded_policy_precedes_bundle_deserialization_and_stage_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    mutate: Any,
    match: str,
) -> None:
    data_root = tmp_path / f"{case}-data"
    output_root = tmp_path / f"{case}-bundle"
    spec = _official_fixture(data_root)
    invalid = deepcopy(_fixed_runtime_identity())
    mutate(invalid)

    # Generate the fixture normally with the invalid identity recorded from the
    # outset. The temporary dataset-ID override only lets this test construct
    # the deliberately invalid official-like acceptance input.
    official_id = result_module.F3_DATASET_ID
    monkeypatch.setattr(result_module, "F3_DATASET_ID", "fixture-generation")
    _run_fixture(
        data_root,
        output_root,
        spec,
        Counter(),
        resume=False,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=invalid,
        finalization_deep=False,
    )
    monkeypatch.setattr(result_module, "F3_DATASET_ID", official_id)

    before_bytes = _tree_bytes(output_root)
    before_paths = _tree_paths(output_root)
    calls: Counter[str] = Counter()

    def forbidden(name: str):
        def fail(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            calls[name] += 1
            pytest.fail(f"recorded policy rejection reached {name}")

        return fail

    monkeypatch.setattr(result_module, "_load_reports", forbidden("report deserialization"))
    monkeypatch.setattr(result_module, "_validate_cells_and_stages", forbidden("stage validation"))
    monkeypatch.setattr(result_module, "_deep_validate_scanner_stages", forbidden("DAT read"))
    monkeypatch.setattr(
        result_module,
        "numerical_runtime_identity",
        forbidden("current runtime identity"),
    )

    with pytest.raises(F3ResultValidationError, match=match):
        validate_completed_f3d_bundle(output_root, _dataset_spec=spec)
    assert not calls
    assert _tree_bytes(output_root) == before_bytes
    assert _tree_paths(output_root) == before_paths


def test_current_runtime_difference_is_shallow_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "bundle"
    spec = _official_fixture(data_root)
    recorded = _fixed_runtime_identity()
    _generate_official_bundle(
        data_root,
        output_root,
        spec,
        recorded,
        Counter(),
        monkeypatch,
    )
    current = deepcopy(recorded)
    current["platform_machine"] = "other-publication-valid-machine"
    monkeypatch.setattr(result_module, "numerical_runtime_identity", lambda: current)

    assert validate_completed_f3d_bundle(output_root, _dataset_spec=spec)
    monkeypatch.setattr(
        result_module,
        "_load_reports",
        lambda *_: pytest.fail("deep runtime mismatch must precede report loading"),
    )
    with pytest.raises(
        F3ResultValidationError,
        match="current runtime identity does not match run manifest",
    ):
        validate_completed_f3d_bundle(output_root, deep=True, _dataset_spec=spec)


def test_scanner_all_reuse_and_one_backend_resume_are_construction_minimal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "bundle"
    spec = _official_fixture(data_root)
    runtime_identity = _fixed_runtime_identity()
    initial_calls: Counter[str] = Counter()
    first = _generate_official_bundle(
        data_root,
        output_root,
        spec,
        runtime_identity,
        initial_calls,
        monkeypatch,
    )
    evidence = _sampling_contract(spec, Counter())
    native_reads: Counter[str] = Counter()
    original_read_native = data_module.F3VolumeSource.read_native_volume

    def tracked_read_native(source: Any, role: str) -> Any:
        native_reads[role] += 1
        return original_read_native(source, role)

    monkeypatch.setattr(data_module.F3VolumeSource, "read_native_volume", tracked_read_native)

    (output_root / "completion.json").unlink()
    all_reuse_calls: Counter[str] = Counter()
    _run_fixture(
        data_root,
        output_root,
        spec,
        all_reuse_calls,
        resume=True,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=runtime_identity,
        scanner_implementation_identity="final-acceptance-scanner-v1",
        scanner_factory=_factory(all_reuse_calls),
        sampling_evidence_by_backend=evidence,
        finalization_deep=False,
    )
    assert all_reuse_calls["scanner factory"] == 0
    assert all_reuse_calls["instance sampling helper"] == 0
    assert all_reuse_calls["reference-like scan"] == 0
    assert all_reuse_calls["quality scan"] == 0
    assert all_reuse_calls["scanner thinning"] == 0
    assert native_reads["input"] == 0

    (output_root / "completion.json").unlink()
    quality_fingerprint = next(
        cell.stages.scanner for cell in first.cells if cell.backend == "quality"
    )
    shutil.rmtree(output_root / "stages" / "scanner" / quality_fingerprint)
    before_source = _tree_bytes(data_root)
    partial_calls: Counter[str] = Counter()
    _run_fixture(
        data_root,
        output_root,
        spec,
        partial_calls,
        resume=True,
        monkeypatch=monkeypatch,
        workspace_runtime_identity=runtime_identity,
        scanner_implementation_identity="final-acceptance-scanner-v1",
        scanner_factory=_factory(partial_calls),
        sampling_evidence_by_backend=evidence,
        finalization_deep=False,
    )
    assert partial_calls["scanner factory"] == 1
    assert partial_calls["instance sampling helper"] == 2
    assert partial_calls["reference-like scan"] == 0
    assert partial_calls["quality scan"] == 1
    assert partial_calls["scanner thinning"] == 1
    assert native_reads["input"] == 1
    assert _tree_bytes(data_root) == before_source
    assert not any(
        ".tmp-" in path.name or path.name.endswith(".partial") for path in output_root.rglob("*")
    )


def test_custom_sampling_mismatch_is_rejected_before_scan_or_source_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "bundle"
    spec = _official_fixture(data_root)
    runtime_identity = _fixed_runtime_identity()
    first = _generate_official_bundle(
        data_root,
        output_root,
        spec,
        runtime_identity,
        Counter(),
        monkeypatch,
    )
    (output_root / "completion.json").unlink()
    quality_fingerprint = next(
        cell.stages.scanner for cell in first.cells if cell.backend == "quality"
    )
    shutil.rmtree(output_root / "stages" / "scanner" / quality_fingerprint)

    evidence = _sampling_contract(spec, Counter())
    plan = _fixture_plan(spec)
    mismatch_provider = _MismatchedScanner(0.0, 0.0, Counter())
    evidence["quality"] = scanner_sampling_evidence(
        mismatch_provider,
        plan.scanner_config_for("quality"),
        "quality",
        implementation_identity="final-acceptance-scanner-v1",
    )
    source_before = _tree_bytes(data_root)
    before_bytes = _tree_bytes(output_root)
    before_paths = _tree_paths(output_root)
    calls: Counter[str] = Counter()
    native_reads: Counter[str] = Counter()
    original_read_native = data_module.F3VolumeSource.read_native_volume

    def tracked_read_native(source: Any, role: str) -> Any:
        native_reads[role] += 1
        return original_read_native(source, role)

    monkeypatch.setattr(data_module.F3VolumeSource, "read_native_volume", tracked_read_native)
    with pytest.raises(ValueError, match="sampling evidence does not match"):
        _run_fixture(
            data_root,
            output_root,
            spec,
            calls,
            resume=True,
            monkeypatch=monkeypatch,
            workspace_runtime_identity=runtime_identity,
            scanner_implementation_identity="final-acceptance-scanner-v1",
            scanner_factory=_factory(calls),
            sampling_evidence_by_backend=evidence,
            finalization_deep=False,
        )

    assert calls["scanner factory"] == 1
    assert calls["instance sampling helper"] == 2
    assert calls["reference-like scan"] == 0
    assert calls["quality scan"] == 0
    assert calls["scanner thinning"] == 0
    assert native_reads["input"] == 0
    assert _tree_bytes(data_root) == source_before
    assert _tree_bytes(output_root) == before_bytes
    assert _tree_paths(output_root) == before_paths
    assert not any(
        ".tmp-" in path.name or path.name.endswith(".partial") for path in output_root.rglob("*")
    )


def test_scanner_failure_releases_memmap_and_instance_without_mutating_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = tmp_path / "data"
    output_root = tmp_path / "bundle"
    spec = _official_fixture(data_root)
    runtime_identity = _fixed_runtime_identity()
    first = _generate_official_bundle(
        data_root,
        output_root,
        spec,
        runtime_identity,
        Counter(),
        monkeypatch,
    )
    (output_root / "completion.json").unlink()
    quality_fingerprint = next(
        cell.stages.scanner for cell in first.cells if cell.backend == "quality"
    )
    shutil.rmtree(output_root / "stages" / "scanner" / quality_fingerprint)
    source_before = _tree_bytes(data_root)
    bundle_before = _tree_bytes(output_root)
    paths_before = _tree_paths(output_root)

    original_memmap = np.memmap
    opened_memmaps: list[np.memmap] = []
    scanner_references: list[weakref.ReferenceType[_TrackedScanner]] = []
    calls: Counter[str] = Counter()

    class TrackedMemmap(original_memmap):  # type: ignore[misc, valid-type]
        def __new__(cls, *args: Any, **kwargs: Any) -> np.memmap:
            array = super().__new__(cls, *args, **kwargs)
            opened_memmaps.append(array)
            return array

    class FailingScanner(_TrackedScanner):
        def scan_quality(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, ...]:
            super().scan_quality(*args, **kwargs)
            raise RuntimeError("injected scanner failure")

    def failing_factory(sigma1: float, sigma2: float) -> FailingScanner:
        scanner = FailingScanner(sigma1, sigma2, calls)
        scanner_references.append(weakref.ref(scanner))
        return scanner

    monkeypatch.setattr(np, "memmap", TrackedMemmap)
    with pytest.raises(RuntimeError, match="injected scanner failure"):
        _run_fixture(
            data_root,
            output_root,
            spec,
            calls,
            resume=True,
            monkeypatch=monkeypatch,
            workspace_runtime_identity=runtime_identity,
            scanner_implementation_identity="final-acceptance-scanner-v1",
            scanner_factory=failing_factory,
            sampling_evidence_by_backend=_sampling_contract(spec, Counter()),
            finalization_deep=False,
        )

    gc.collect()
    assert opened_memmaps
    assert all(array._mmap.closed for array in opened_memmaps)
    assert scanner_references
    assert all(reference() is None for reference in scanner_references)
    assert _tree_bytes(data_root) == source_before
    assert _tree_bytes(output_root) == bundle_before
    assert _tree_paths(output_root) == paths_before
    assert not any(
        ".tmp-" in path.name
        or path.name.startswith(".pyosv-stage-tmp-")
        or path.name.endswith(".partial")
        for path in output_root.rglob("*")
    )
