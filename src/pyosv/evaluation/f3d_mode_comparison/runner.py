"""Canonical cell execution for the F3 full-volume 2-by-2 comparison."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol

import numpy as np

from pyosv.evaluation.synthetic_quality.config import SyntheticVotingConfig
from pyosv.evaluation.synthetic_quality.stage_cache import (
    FinalThinningStageResult,
    PipelineStageCache,
    VotingStageResult,
    diagnostic_items,
)
from pyosv.evaluation.synthetic_quality.stage_keys import (
    build_final_thinning_stage_key,
    build_seed_stage_key,
    build_thinning_stage_key,
    build_voting_stage_key,
)
from pyosv.evaluation.synthetic_quality.variants import VariantSpec
from pyosv.evaluation.workflow3d import (
    PreparedAttributeIdentity,
    VolumeVotingControls,
    Workflow3DResult,
    execute_workflow3d,
)
from pyosv.experimental.boundary_skinning import apply_boundary_skinner_fallback
from pyosv.synthetic_metrics import skin_mask_from_skins, skin_topology_metrics

from .artifacts import (
    F3RunWorkspace,
    F3StageArtifact,
    F3StageCorruptionError,
    _callable_implementation_identity,
    _workspace_dataset_file_identity,
    canonical_fingerprint,
    canonical_json_bytes,
    stage_computation_identity,
    stage_fingerprint,
    validate_stage,
)
from .models import F3CellSpec, F3ModeComparisonPlan
from .scanner import (
    F3LoadedScannerStage,
    F3ScannerStageResult,
    load_scanner_stage,
    scanner_stage_artifacts,
    scanner_stage_resolved_settings,
)

F3_CELL_RUNNER_CONTRACT_VERSION = 1
F3_VOTING_STAGE_IMPLEMENTATION = "pyosv-f3-voting-stage-v1"
F3_THINNING_STAGE_IMPLEMENTATION = "pyosv-f3-thinning-stage-v1"
F3_SKINNING_STAGE_IMPLEMENTATION = "pyosv-f3-skinning-stage-v1"
F3_CELL_REFERENCE_SCHEMA_VERSION = 1

_DAT_DTYPE = np.dtype(">f4")
_STAGE_REPORT = F3StageArtifact("report.json")
_DEFAULT_VARIANT = VariantSpec("f3-canonical", experimental=False)

WorkflowRunner = Callable[..., Workflow3DResult]
RuntimeHook = Callable[["F3StageRuntime"], None]
StageState = Literal["computed", "reused"]


class _StageRSSRecorder(Protocol):
    def stage_before(
        self,
        stage_kind: str,
        fingerprint: str,
        *,
        phase: str | None = None,
    ) -> object: ...

    def stage_after(
        self,
        stage_kind: str,
        fingerprint: str,
        *,
        phase: str | None = None,
    ) -> object: ...


@dataclass(frozen=True, slots=True)
class F3CellStageFingerprints:
    """Content identities referenced by one canonical cell."""

    scanner: str
    voting: str
    thinning: str
    skinning: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class F3CellReference:
    """Scalar-only reference to one completed canonical cell."""

    label: str
    backend: str
    workflow: str
    resolved_config: Mapping[str, Any]
    stages: F3CellStageFingerprints
    skinning_enabled: bool
    path: Path
    reused: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_reference_schema_version": F3_CELL_REFERENCE_SCHEMA_VERSION,
            "label": self.label,
            "backend": self.backend,
            "workflow": self.workflow,
            "resolved_config": dict(self.resolved_config),
            "stages": self.stages.as_dict(),
            "skinning": {
                "enabled": self.skinning_enabled,
                "state": "enabled" if self.skinning_enabled else "disabled",
            },
        }


@dataclass(frozen=True, slots=True)
class F3StageRuntime:
    """One stage-use event emitted without affecting computation identity."""

    kind: str
    fingerprint: str
    state: StageState
    elapsed_seconds: float
    source_bytes: int
    output_bytes: int
    cell_owner: str
    shared_consumers: tuple[str, ...]
    cell: str


@dataclass(frozen=True, slots=True)
class F3CellRunResult:
    """Completed canonical cells and their stage-use events."""

    cells: tuple[F3CellReference, ...]
    stage_runtime: tuple[F3StageRuntime, ...]

    def cell_for(self, label: str) -> F3CellReference:
        for cell in self.cells:
            if cell.label == label:
                return cell
        raise KeyError(label)

    def as_dict(self) -> dict[str, Any]:
        return {
            "cells": [cell.as_dict() for cell in self.cells],
            "stage_runtime": [asdict(event) for event in self.stage_runtime],
        }


@dataclass(frozen=True, slots=True)
class _CellExecution:
    spec: F3CellSpec
    attribute: PreparedAttributeIdentity
    voting_controls: VolumeVotingControls
    resolved_config: Mapping[str, Any]
    stages: F3CellStageFingerprints
    voting_settings: Mapping[str, Any]
    thinning_settings: Mapping[str, Any]
    skinning_settings: Mapping[str, Any]


def voting_stage_artifacts(shape: tuple[int, int, int]) -> tuple[F3StageArtifact, ...]:
    """Return the fixed voting artifact contract."""

    return (
        F3StageArtifact("fv.dat", shape, ">f4"),
        F3StageArtifact("vp.dat", shape, ">f4"),
        F3StageArtifact("vt.dat", shape, ">f4"),
        _STAGE_REPORT,
    )


def thinning_stage_artifacts(shape: tuple[int, int, int]) -> tuple[F3StageArtifact, ...]:
    """Return the fixed final-thinning artifact contract."""

    return (F3StageArtifact("fvt.dat", shape, ">f4"), _STAGE_REPORT)


def skinning_stage_artifacts(
    shape: tuple[int, int, int],
    *,
    enabled: bool,
) -> tuple[F3StageArtifact, ...]:
    """Return the fixed final-skinning contract, or no artifacts when disabled."""

    if not enabled:
        return ()
    return (
        F3StageArtifact("skin_mask.dat", shape, ">f4"),
        F3StageArtifact("skins.json"),
        _STAGE_REPORT,
    )


def build_f3d_cell_stage_fingerprints(
    workspace: F3RunWorkspace,
    plan: F3ModeComparisonPlan,
    scanner_stages: Mapping[str, F3ScannerStageResult],
) -> Mapping[str, F3CellStageFingerprints]:
    """Build the deterministic cell-to-stage mapping without executing a stage."""

    _validate_runner_inputs(
        workspace,
        plan,
        scanner_stages,
        execute_workflow3d,
        load_scanner_stage,
        None,
        None,
        _DEFAULT_VARIANT,
    )
    workflow_implementation = _workflow_implementation_identity(execute_workflow3d, None)
    return MappingProxyType(
        {
            cell.label: _cell_execution(
                workspace,
                plan,
                scanner_stages[cell.scanner_backend],
                cell,
                workflow_implementation=workflow_implementation,
            ).stages
            for cell in plan.cells
        }
    )


def run_f3d_mode_comparison_cells(
    workspace: F3RunWorkspace,
    plan: F3ModeComparisonPlan,
    scanner_stages: Mapping[str, F3ScannerStageResult],
    *,
    cell_order: Sequence[str | F3CellSpec] | None = None,
    workflow_runner: WorkflowRunner = execute_workflow3d,
    scanner_loader: Callable[[F3ScannerStageResult], F3LoadedScannerStage] = (load_scanner_stage),
    runtime_hook: RuntimeHook | None = None,
    workflow_implementation_identity: Mapping[str, Any] | str | None = None,
    variant_spec: VariantSpec = _DEFAULT_VARIANT,
    rss_recorder: _StageRSSRecorder | None = None,
) -> F3CellRunResult:
    """Run or resume all four cells while sharing semantic array stages.

    Scanner artifacts are only opened read-only. Cell directories receive one
    JSON reference and never receive copies or links of full-volume artifacts.
    """

    _validate_runner_inputs(
        workspace,
        plan,
        scanner_stages,
        workflow_runner,
        scanner_loader,
        runtime_hook,
        rss_recorder,
        variant_spec,
    )
    order = _resolve_cell_order(plan, cell_order)
    workflow_implementation = _workflow_implementation_identity(
        workflow_runner,
        workflow_implementation_identity,
    )
    executions = {
        cell.label: _cell_execution(
            workspace,
            plan,
            scanner_stages[cell.scanner_backend],
            cell,
            workflow_implementation=workflow_implementation,
        )
        for cell in plan.cells
    }
    expected_consumers = _stage_consumers(plan, executions)
    loaded_backend: str | None = None
    loaded_scanner: F3LoadedScannerStage | None = None
    workflow_attributes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    runtime: list[F3StageRuntime] = []
    references: dict[str, F3CellReference] = {}
    recomputed_stages: set[tuple[str, str]] = set()
    active_cache: PipelineStageCache | None = None
    active_hydrated: tuple[np.memmap, ...] = ()

    try:
        for cell_index, cell in enumerate(order):
            if loaded_backend is not None and loaded_backend != cell.scanner_backend:
                workflow_attributes = None
                assert loaded_scanner is not None
                loaded_scanner.close()
                loaded_scanner = None
                loaded_backend = None
            execution = executions[cell.label]
            scanner = scanner_stages[cell.scanner_backend]
            validation_elapsed_by_kind: dict[str, float] = {}

            def validate_timed(
                kind: str,
                parent: str,
                settings: Mapping[str, Any],
                artifacts: Sequence[F3StageArtifact],
                fingerprint: str,
            ) -> bool:
                stage_exists = _stage_path_exists(workspace, kind, fingerprint)
                if rss_recorder is not None and stage_exists:
                    rss_recorder.stage_before(
                        kind,
                        fingerprint,
                        phase=f"load_validation:{cell.label}",
                    )
                started = time.perf_counter()
                try:
                    return _validate_existing_stage(
                        workspace,
                        kind,
                        parent,
                        settings,
                        artifacts,
                        fingerprint,
                    )
                finally:
                    validation_elapsed_by_kind[kind] = (
                        validation_elapsed_by_kind.get(kind, 0.0) + time.perf_counter() - started
                    )
                    if rss_recorder is not None and stage_exists:
                        rss_recorder.stage_after(
                            kind,
                            fingerprint,
                            phase=f"load_validation:{cell.label}",
                        )

            voting_exists = validate_timed(
                "voting",
                scanner.fingerprint,
                execution.voting_settings,
                voting_stage_artifacts(scanner.shape),
                execution.stages.voting,
            )
            thinning_exists = validate_timed(
                "thinning",
                execution.stages.voting,
                execution.thinning_settings,
                thinning_stage_artifacts(scanner.shape),
                execution.stages.thinning,
            )
            skinning_exists = False
            if plan.skinning_enabled:
                skinning_exists = validate_timed(
                    "skinning",
                    execution.stages.thinning,
                    execution.skinning_settings,
                    skinning_stage_artifacts(scanner.shape, enabled=True),
                    execution.stages.skinning,
                )
            compute_voting = not voting_exists
            compute_thinning = (
                not thinning_exists
                or (compute_voting or ("voting", execution.stages.voting) in recomputed_stages)
                and ("thinning", execution.stages.thinning) not in recomputed_stages
            )
            compute_skinning = bool(
                plan.skinning_enabled
                and (
                    not skinning_exists
                    or (
                        compute_thinning
                        or ("thinning", execution.stages.thinning) in recomputed_stages
                    )
                    and ("skinning", execution.stages.skinning) not in recomputed_stages
                )
            )
            complete_chain = not (compute_voting or compute_thinning or compute_skinning)
            result: Workflow3DResult | None = None
            elapsed_by_kind: dict[str, float] = {}

            if not complete_chain:
                if loaded_scanner is None:
                    loaded_scanner = scanner_loader(scanner)
                    loaded_backend = cell.scanner_backend
                    workflow_attributes = _native_workflow_attributes(loaded_scanner)
                assert workflow_attributes is not None
                loaded = loaded_scanner
                workflow_ft, workflow_pt, workflow_tt = workflow_attributes
                active_cache = PipelineStageCache()
                active_hydrated = _hydrate_cache(
                    active_cache,
                    execution,
                    workspace,
                    scanner.shape,
                    voting=not compute_voting,
                    thinning=not compute_thinning,
                )

                def stage_timer(
                    stage: str,
                    semantic_key: Any,
                    operation: Callable[[], Any],
                ) -> Any:
                    del semantic_key
                    kind = _runtime_kind(stage)
                    fingerprint = (
                        getattr(execution.stages, kind)
                        if kind in {"voting", "thinning", "skinning"}
                        else None
                    )
                    if rss_recorder is not None and fingerprint is not None:
                        rss_recorder.stage_before(
                            kind,
                            fingerprint,
                            phase=f"compute:{cell.label}",
                        )
                    started = time.perf_counter()
                    try:
                        return operation()
                    finally:
                        elapsed_by_kind[kind] = (
                            elapsed_by_kind.get(kind, 0.0) + time.perf_counter() - started
                        )
                        if rss_recorder is not None and fingerprint is not None:
                            rss_recorder.stage_after(
                                kind,
                                fingerprint,
                                phase=f"compute:{cell.label}",
                            )

                scanner_mask = (
                    np.asarray(loaded.ft) > np.float32(0.0)
                    if execution.resolved_config["skinning"]["boundary_skinner_fallback"]
                    else None
                )

                def timed_boundary_fallback(*args: Any, **kwargs: Any) -> None:
                    started = time.perf_counter()
                    try:
                        apply_boundary_skinner_fallback(*args, **kwargs)
                    finally:
                        elapsed_by_kind["skinning"] = (
                            elapsed_by_kind.get("skinning", 0.0) + time.perf_counter() - started
                        )

                try:
                    result = workflow_runner(
                        ft=workflow_ft,
                        pt=workflow_pt,
                        tt=workflow_tt,
                        attribute_identity=execution.attribute,
                        voting_settings=plan.workflow_settings_for(
                            cell.workflow_mode
                        ).voting_config,
                        voting_controls=execution.voting_controls,
                        skinning_settings=plan.workflow_settings_for(
                            cell.workflow_mode
                        ).skinning_config,
                        variant_spec=variant_spec,
                        stage_cache=active_cache,
                        stage_timer=stage_timer,
                        scanner_target_positive_mask=scanner_mask,
                        fvt_recenter_target=workflow_ft,
                        fvt_recenter_target_source="scanner_fet",
                        boundary_fallback_runner=timed_boundary_fallback,
                    )
                finally:
                    scanner_mask = None
                _validate_workflow_result(result, execution, scanner.shape)

            stage_states, persistence_elapsed = _persist_or_reuse_cell_stages(
                workspace,
                execution,
                scanner,
                result,
                voting_exists=voting_exists,
                thinning_exists=thinning_exists,
                skinning_exists=skinning_exists,
                compute_voting=compute_voting,
                compute_thinning=compute_thinning,
                compute_skinning=compute_skinning,
                rss_recorder=rss_recorder,
            )
            result = None
            if active_cache is not None:
                active_cache.clear()
                active_cache = None
            _close_memmaps(active_hydrated)
            active_hydrated = ()
            if not complete_chain:
                del loaded, workflow_ft, workflow_pt, workflow_tt

            for kind, elapsed in persistence_elapsed.items():
                elapsed_by_kind[kind] = elapsed_by_kind.get(kind, 0.0) + elapsed
            for kind, (state, source_bytes, output_bytes) in stage_states.items():
                elapsed = (
                    elapsed_by_kind.get(kind, 0.0)
                    if state == "computed"
                    else validation_elapsed_by_kind.get(kind, 0.0)
                )
                event = F3StageRuntime(
                    kind=kind,
                    fingerprint=getattr(execution.stages, kind),
                    state=state,
                    elapsed_seconds=float(elapsed),
                    source_bytes=source_bytes,
                    output_bytes=output_bytes,
                    cell_owner=expected_consumers[(kind, getattr(execution.stages, kind))][0],
                    shared_consumers=expected_consumers[(kind, getattr(execution.stages, kind))],
                    cell=cell.label,
                )
                runtime.append(event)
                if state == "computed":
                    recomputed_stages.add((kind, event.fingerprint))
                if runtime_hook is not None:
                    runtime_hook(event)

            reference = _write_or_reuse_cell_reference(
                workspace,
                execution,
                skinning_enabled=plan.skinning_enabled,
            )
            references[cell.label] = reference

            next_backend = (
                order[cell_index + 1].scanner_backend if cell_index + 1 < len(order) else None
            )
            if loaded_scanner is not None and next_backend != loaded_backend:
                workflow_attributes = None
                loaded_scanner.close()
                loaded_scanner = None
                loaded_backend = None
    finally:
        if active_cache is not None:
            active_cache.clear()
        _close_memmaps(active_hydrated)
        workflow_attributes = None
        if loaded_scanner is not None:
            loaded_scanner.close()

    return F3CellRunResult(
        cells=tuple(references[cell.label] for cell in plan.cells),
        stage_runtime=tuple(runtime),
    )


def load_f3d_mode_comparison_cells(
    workspace: F3RunWorkspace,
    plan: F3ModeComparisonPlan,
    scanner_stages: Mapping[str, F3ScannerStageResult],
) -> tuple[F3CellReference, ...]:
    """Load and validate a complete set of cell references and parent stages."""

    _validate_runner_inputs(
        workspace,
        plan,
        scanner_stages,
        execute_workflow3d,
        load_scanner_stage,
        None,
        None,
        _DEFAULT_VARIANT,
    )
    executions = {
        cell.label: _cell_execution(
            workspace,
            plan,
            scanner_stages[cell.scanner_backend],
            cell,
            workflow_implementation=_workflow_implementation_identity(execute_workflow3d, None),
        )
        for cell in plan.cells
    }
    references: list[F3CellReference] = []
    known = {execution.stages.scanner for execution in executions.values()} | {
        fingerprint
        for execution in executions.values()
        for fingerprint in (
            execution.stages.voting,
            execution.stages.thinning,
            execution.stages.skinning,
        )
    }
    for cell in plan.cells:
        execution = executions[cell.label]
        path = workspace.path / "cells" / f"{cell.label}.json"
        payload = _read_json(path)
        stages = payload.get("stages")
        if not isinstance(stages, dict) or any(value not in known for value in stages.values()):
            raise F3StageCorruptionError(
                f"cell reference contains an unknown stage fingerprint: {cell.label}"
            )
        expected = _cell_payload(execution, skinning_enabled=plan.skinning_enabled)
        if payload != expected:
            raise F3StageCorruptionError(f"cell reference mismatch: {cell.label}")
        _validate_cell_stage_chain(workspace, execution, plan.skinning_enabled)
        references.append(_reference_from_payload(path, payload, reused=True))
    return tuple(references)


# A shorter public spelling for callers that treat the four cells as one run.
run_f3d_mode_comparison = run_f3d_mode_comparison_cells
load_f3d_mode_comparison = load_f3d_mode_comparison_cells


def _cell_execution(
    workspace: F3RunWorkspace,
    plan: F3ModeComparisonPlan,
    scanner: F3ScannerStageResult,
    cell: F3CellSpec,
    *,
    workflow_implementation: Mapping[str, Any] | str,
) -> _CellExecution:
    workflow = plan.workflow_settings_for(cell.workflow_mode)
    controls = _volume_voting_controls(plan)
    attribute = PreparedAttributeIdentity(
        dataset_fingerprint=workspace.fingerprint,
        stage_fingerprint=scanner.fingerprint,
        shape=scanner.shape,
        backend=cell.scanner_backend,
        scanner_thin_mode=plan.scanner_config_for(cell.scanner_backend).scanner_thin_mode,
        edge_policy=bool(
            plan.scanner_config_for(cell.scanner_backend).effective_remove_edge_effects
        ),
    )
    attribute_key = attribute.stage_key
    seed_key = build_seed_stage_key(
        attribute_key=attribute_key,
        voting_config=workflow.voting_config,
        variant_spec=_DEFAULT_VARIANT,
        target_source="scanner_fet",
    )
    voting_key = build_voting_stage_key(
        seed_key=seed_key,
        voting_config=workflow.voting_config,
        variant_spec=_DEFAULT_VARIANT,
        voting_controls=controls,
    )
    thinning_key = build_thinning_stage_key(
        voting_key=voting_key,
        voting_config=workflow.voting_config,
        variant_spec=_DEFAULT_VARIANT,
    )
    final_thinning_key = build_final_thinning_stage_key(
        thinning_key=thinning_key,
        variant_spec=_DEFAULT_VARIANT,
        target_source="scanner_fet",
    )
    assert voting_key is not None
    assert final_thinning_key is not None

    resolved_config = {
        "workflow_mode": cell.workflow_mode,
        "voting": asdict(workflow.voting_config),
        "voting_controls": asdict(controls),
        "skinning": asdict(workflow.skinning_config),
        "variant": asdict(_DEFAULT_VARIANT),
    }
    voting_settings = {
        "cell_runner_contract_version": F3_CELL_RUNNER_CONTRACT_VERSION,
        "implementation_contract": F3_VOTING_STAGE_IMPLEMENTATION,
        "workflow_runner_identity": workflow_implementation,
        "attribute_identity": asdict(attribute),
        "semantic_key": asdict(voting_key),
    }
    thinning_settings = {
        "cell_runner_contract_version": F3_CELL_RUNNER_CONTRACT_VERSION,
        "implementation_contract": F3_THINNING_STAGE_IMPLEMENTATION,
        "workflow_runner_identity": workflow_implementation,
        "semantic_key": asdict(final_thinning_key),
    }
    skinning_settings = {
        "cell_runner_contract_version": F3_CELL_RUNNER_CONTRACT_VERSION,
        "implementation_contract": F3_SKINNING_STAGE_IMPLEMENTATION,
        "workflow_runner_identity": workflow_implementation,
        "enabled": workflow.skinning_config.enabled,
        "resolved_skinner_config": asdict(workflow.skinning_config),
        "growth_source": workflow.skinning_config.growth_source,
        "fallback_policy": {
            "enabled": workflow.skinning_config.boundary_skinner_fallback,
            "policy": workflow.skinning_config.boundary_skinner_fallback_policy,
        },
        "primary_skinner_identity": ("pyosv.experimental.boundary_skinning.find_synthetic_skins"),
    }
    voting_fingerprint = stage_fingerprint(
        "voting",
        run_fingerprint_value=workspace.fingerprint,
        parent_fingerprints=(scanner.fingerprint,),
        resolved_settings=voting_settings,
        artifacts=voting_stage_artifacts(scanner.shape),
    )
    thinning_fingerprint = stage_fingerprint(
        "thinning",
        run_fingerprint_value=workspace.fingerprint,
        parent_fingerprints=(voting_fingerprint,),
        resolved_settings=thinning_settings,
        artifacts=thinning_stage_artifacts(scanner.shape),
    )
    if workflow.skinning_config.enabled:
        skinning_fingerprint = stage_fingerprint(
            "skinning",
            run_fingerprint_value=workspace.fingerprint,
            parent_fingerprints=(thinning_fingerprint,),
            resolved_settings=skinning_settings,
            artifacts=skinning_stage_artifacts(scanner.shape, enabled=True),
        )
    else:
        skinning_fingerprint = canonical_fingerprint(
            {
                "run_fingerprint": workspace.fingerprint,
                "kind": "skinning",
                "parent_fingerprint": thinning_fingerprint,
                "resolved_settings": skinning_settings,
                "artifact_schema": {},
            }
        )
    return _CellExecution(
        spec=cell,
        attribute=attribute,
        voting_controls=controls,
        resolved_config=MappingProxyType(resolved_config),
        stages=F3CellStageFingerprints(
            scanner=scanner.fingerprint,
            voting=voting_fingerprint,
            thinning=thinning_fingerprint,
            skinning=skinning_fingerprint,
        ),
        voting_settings=MappingProxyType(voting_settings),
        thinning_settings=MappingProxyType(thinning_settings),
        skinning_settings=MappingProxyType(skinning_settings),
    )


def _volume_voting_controls(plan: F3ModeComparisonPlan) -> VolumeVotingControls:
    controls = plan.voting_controls
    return VolumeVotingControls(
        strain_max1=controls.strain_max1,
        strain_max2=controls.strain_max2,
        surface_smoothing1=controls.surface_smoothing1,
        surface_smoothing2=controls.surface_smoothing2,
        boundary_policy=controls.surface_voting_boundary_policy,
        support_min_fraction=controls.surface_support_min_fraction,
        support_exponent=controls.surface_support_exponent,
        orientation_smoothing=controls.surface_orientation_smoothing,
        final_normalization_smoothing=controls.final_normalization_smoothing,
    )


def _validate_existing_stage(
    workspace: F3RunWorkspace,
    kind: str,
    parent: str,
    settings: Mapping[str, Any],
    artifacts: Sequence[F3StageArtifact],
    fingerprint: str,
) -> bool:
    path = workspace.stage_path(kind, fingerprint)
    if not path.exists() and not path.is_symlink():
        return False
    computation = stage_computation_identity(
        kind,
        run_fingerprint_value=workspace.fingerprint,
        parent_fingerprints=(parent,),
        resolved_settings=settings,
        artifacts=artifacts,
    )
    validate_stage(path, computation, fingerprint)
    return True


def _hydrate_cache(
    cache: PipelineStageCache,
    execution: _CellExecution,
    workspace: F3RunWorkspace,
    shape: tuple[int, int, int],
    *,
    voting: bool,
    thinning: bool,
) -> tuple[np.memmap, ...]:
    opened: list[np.memmap] = []
    attribute_key = execution.attribute.stage_key
    voting_config = execution.resolved_config["voting"]
    # The typed keys are reconstructed from the same resolved objects used by execution.
    config = SyntheticVotingConfig(**voting_config)
    seed_key = build_seed_stage_key(
        attribute_key=attribute_key,
        voting_config=config,
        variant_spec=_DEFAULT_VARIANT,
        target_source="scanner_fet",
    )
    voting_key = build_voting_stage_key(
        seed_key=seed_key,
        voting_config=config,
        variant_spec=_DEFAULT_VARIANT,
        voting_controls=execution.voting_controls,
    )
    assert voting_key is not None
    if voting and cache.get_voting(voting_key) is None:
        path = workspace.stage_path("voting", execution.stages.voting)
        report = _read_json(path / "report.json")
        fv = _open_dat(path / "fv.dat", shape)
        vp = _open_dat(path / "vp.dat", shape)
        vt = _open_dat(path / "vt.dat", shape)
        opened.extend((fv, vp, vt))
        cache.put_voting(
            voting_key,
            VotingStageResult(
                fv=fv,
                vp=vp,
                vt=vt,
                diagnostic_items=diagnostic_items(report.get("diagnostics", {})),
            ),
        )
    thinning_key = build_thinning_stage_key(
        voting_key=voting_key,
        voting_config=config,
        variant_spec=_DEFAULT_VARIANT,
    )
    assert thinning_key is not None
    final_thinning_key = build_final_thinning_stage_key(
        thinning_key=thinning_key,
        variant_spec=_DEFAULT_VARIANT,
        target_source="scanner_fet",
    )
    assert final_thinning_key is not None
    if thinning and cache.get_final_thinning(final_thinning_key) is None:
        path = workspace.stage_path("thinning", execution.stages.thinning)
        report = _read_json(path / "report.json")
        fvt = _open_dat(path / "fvt.dat", shape)
        opened.append(fvt)
        cache.put_final_thinning(
            final_thinning_key,
            FinalThinningStageResult(
                fvt=fvt,
                diagnostic_items=diagnostic_items(report.get("diagnostics", {})),
            ),
        )
    return tuple(opened)


def _persist_or_reuse_cell_stages(
    workspace: F3RunWorkspace,
    execution: _CellExecution,
    scanner: F3ScannerStageResult,
    result: Workflow3DResult | None,
    *,
    voting_exists: bool,
    thinning_exists: bool,
    skinning_exists: bool,
    compute_voting: bool,
    compute_thinning: bool,
    compute_skinning: bool,
    rss_recorder: _StageRSSRecorder | None,
) -> tuple[dict[str, tuple[StageState, int, int]], dict[str, float]]:
    shape = scanner.shape
    source_volume_bytes = int(np.prod(shape)) * np.dtype(">f4").itemsize
    states: dict[str, tuple[StageState, int, int]] = {}
    elapsed_by_kind: dict[str, float] = {}

    def persist_timed(
        kind: str,
        fingerprint: str,
        operation: Callable[[], Any],
    ) -> Any:
        if rss_recorder is not None:
            rss_recorder.stage_before(kind, fingerprint, phase="artifact_write_validation")
        started = time.perf_counter()
        try:
            return operation()
        finally:
            elapsed_by_kind[kind] = elapsed_by_kind.get(kind, 0.0) + time.perf_counter() - started
            if rss_recorder is not None:
                rss_recorder.stage_after(kind, fingerprint, phase="artifact_write_validation")

    if result is not None:
        voting_report = {
            "fingerprint": execution.stages.voting,
            "scanner_stage_fingerprint": scanner.fingerprint,
            "shape": list(shape),
            "diagnostics": dict(result.diagnostics.voting),
            "seed_count": len(result.seed_indices),
            "resolved_stage_settings": dict(execution.voting_settings),
        }

        def write_voting(path: Path) -> None:
            _write_dat(path / "fv.dat", result.fv)
            _write_dat(path / "vp.dat", result.vp)
            _write_dat(path / "vt.dat", result.vt)
            _write_json(path / "report.json", voting_report)

        voting_stage = (
            persist_timed(
                "voting",
                execution.stages.voting,
                lambda: workspace.write_or_reuse_stage(
                    "voting",
                    parent_fingerprints=(scanner.fingerprint,),
                    resolved_settings=execution.voting_settings,
                    artifacts=voting_stage_artifacts(shape),
                    writer=write_voting,
                    fingerprint=execution.stages.voting,
                    force_recompute=voting_exists,
                ),
            )
            if compute_voting
            else None
        )
        voting_path = (
            voting_stage.path
            if voting_stage is not None
            else workspace.stage_path("voting", execution.stages.voting)
        )
        states["voting"] = (
            "computed" if compute_voting else "reused",
            3 * source_volume_bytes,
            _stage_output_bytes(voting_path),
        )

        thinning_report = {
            "fingerprint": execution.stages.thinning,
            "voting_stage_fingerprint": execution.stages.voting,
            "shape": list(shape),
            "diagnostics": dict(result.diagnostics.thinning),
            "resolved_stage_settings": dict(execution.thinning_settings),
        }

        def write_thinning(path: Path) -> None:
            _write_dat(path / "fvt.dat", result.fvt)
            _write_json(path / "report.json", thinning_report)

        thinning_stage = (
            persist_timed(
                "thinning",
                execution.stages.thinning,
                lambda: workspace.write_or_reuse_stage(
                    "thinning",
                    parent_fingerprints=(execution.stages.voting,),
                    resolved_settings=execution.thinning_settings,
                    artifacts=thinning_stage_artifacts(shape),
                    writer=write_thinning,
                    fingerprint=execution.stages.thinning,
                    force_recompute=thinning_exists,
                ),
            )
            if compute_thinning
            else None
        )
        thinning_path = (
            thinning_stage.path
            if thinning_stage is not None
            else workspace.stage_path("thinning", execution.stages.thinning)
        )
        states["thinning"] = (
            "computed" if compute_thinning else "reused",
            3 * source_volume_bytes,
            _stage_output_bytes(thinning_path),
        )

        if execution.skinning_settings["enabled"]:
            skins = tuple(result.skins)
            skin_report = {
                "fingerprint": execution.stages.skinning,
                "thinning_stage_fingerprint": execution.stages.thinning,
                "shape": list(shape),
                "enabled": True,
                "diagnostics": dict(result.diagnostics.skinning),
                "topology": skin_topology_metrics(
                    skins,
                    shape,
                    small_skin_size=int(execution.resolved_config["skinning"]["small_skin_size"]),
                ),
                "resolved_stage_settings": dict(execution.skinning_settings),
            }

            def write_skinning(path: Path) -> None:
                mask = skin_mask_from_skins(skins, shape).astype(np.float32, copy=False)
                _write_dat(path / "skin_mask.dat", mask)
                _write_json(path / "skins.json", _skins_payload(skins))
                _write_json(path / "report.json", skin_report)

            skinning_stage = (
                persist_timed(
                    "skinning",
                    execution.stages.skinning,
                    lambda: workspace.write_or_reuse_stage(
                        "skinning",
                        parent_fingerprints=(execution.stages.thinning,),
                        resolved_settings=execution.skinning_settings,
                        artifacts=skinning_stage_artifacts(shape, enabled=True),
                        writer=write_skinning,
                        fingerprint=execution.stages.skinning,
                        force_recompute=skinning_exists,
                    ),
                )
                if compute_skinning
                else None
            )
            skinning_path = (
                skinning_stage.path
                if skinning_stage is not None
                else workspace.stage_path("skinning", execution.stages.skinning)
            )
            states["skinning"] = (
                "computed" if compute_skinning else "reused",
                4 * source_volume_bytes,
                _stage_output_bytes(skinning_path),
            )
    else:
        voting_path = workspace.stage_path("voting", execution.stages.voting)
        thinning_path = workspace.stage_path("thinning", execution.stages.thinning)
        states["voting"] = (
            "reused",
            3 * source_volume_bytes,
            _stage_output_bytes(voting_path),
        )
        states["thinning"] = (
            "reused",
            3 * source_volume_bytes,
            _stage_output_bytes(thinning_path),
        )
        if execution.skinning_settings["enabled"]:
            states["skinning"] = (
                "reused",
                4 * source_volume_bytes,
                _stage_output_bytes(workspace.stage_path("skinning", execution.stages.skinning)),
            )

    return states, elapsed_by_kind


def _validate_workflow_result(
    result: Workflow3DResult,
    execution: _CellExecution,
    shape: tuple[int, int, int],
) -> None:
    if not isinstance(result, Workflow3DResult):
        raise TypeError("workflow_runner must return Workflow3DResult")
    for name in ("fv", "vp", "vt", "fvt"):
        array = getattr(result, name)
        if array.shape != shape:
            raise ValueError(f"workflow {name} shape must be {shape}")
        if array.dtype != np.dtype("float32"):
            raise ValueError(f"workflow {name} dtype must be float32")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"workflow {name} must contain only finite values")
    if result.skin.enabled != execution.skinning_settings["enabled"]:
        raise ValueError("workflow skinning enabled state does not match resolved config")
    if result.stage_keys.attribute != execution.attribute.stage_key:
        raise ValueError("workflow attribute stage key does not match the cell identity")
    if (
        result.stage_keys.voting is None
        or asdict(result.stage_keys.voting) != execution.voting_settings["semantic_key"]
    ):
        raise ValueError("workflow voting stage key does not match the cell identity")
    if (
        result.stage_keys.final_thinning is None
        or asdict(result.stage_keys.final_thinning) != execution.thinning_settings["semantic_key"]
    ):
        raise ValueError("workflow thinning stage key does not match the cell identity")


def _write_or_reuse_cell_reference(
    workspace: F3RunWorkspace,
    execution: _CellExecution,
    *,
    skinning_enabled: bool,
) -> F3CellReference:
    path = workspace.path / "cells" / f"{execution.spec.label}.json"
    payload = _cell_payload(execution, skinning_enabled=skinning_enabled)
    if path.exists() or path.is_symlink():
        if _read_json(path) != payload:
            raise F3StageCorruptionError(f"cell reference mismatch: {execution.spec.label}")
        return _reference_from_payload(path, payload, reused=True)
    parent = path.parent
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".cell-tmp-", dir=parent)
        os.close(descriptor)
        temporary = Path(name)
        temporary.write_bytes(canonical_json_bytes(payload) + b"\n")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return _reference_from_payload(path, payload, reused=False)


def _cell_payload(
    execution: _CellExecution,
    *,
    skinning_enabled: bool,
) -> dict[str, Any]:
    payload = {
        "cell_reference_schema_version": F3_CELL_REFERENCE_SCHEMA_VERSION,
        "label": execution.spec.label,
        "backend": execution.spec.scanner_backend,
        "workflow": execution.spec.workflow_mode,
        "resolved_config": dict(execution.resolved_config),
        "stages": execution.stages.as_dict(),
        "skinning": {
            "enabled": skinning_enabled,
            "state": "enabled" if skinning_enabled else "disabled",
        },
    }
    # Normalize tuples and NumPy-compatible scalar values exactly as they are
    # represented on disk so an exact resume comparison is stable.
    normalized = json.loads(canonical_json_bytes(payload))
    assert isinstance(normalized, dict)
    return normalized


def _reference_from_payload(
    path: Path,
    payload: Mapping[str, Any],
    *,
    reused: bool,
) -> F3CellReference:
    stages = payload["stages"]
    return F3CellReference(
        label=payload["label"],
        backend=payload["backend"],
        workflow=payload["workflow"],
        resolved_config=MappingProxyType(dict(payload["resolved_config"])),
        stages=F3CellStageFingerprints(**stages),
        skinning_enabled=bool(payload["skinning"]["enabled"]),
        path=path,
        reused=reused,
    )


def _validate_cell_stage_chain(
    workspace: F3RunWorkspace,
    execution: _CellExecution,
    skinning_enabled: bool,
) -> None:
    scanner_path = workspace.stage_path("scanner", execution.stages.scanner)
    if not scanner_path.is_dir():
        raise F3StageCorruptionError("cell reference scanner stage is missing")
    if not _validate_existing_stage(
        workspace,
        "voting",
        execution.stages.scanner,
        execution.voting_settings,
        voting_stage_artifacts(execution.attribute.shape),
        execution.stages.voting,
    ):
        raise F3StageCorruptionError("cell reference voting stage is missing")
    if not _validate_existing_stage(
        workspace,
        "thinning",
        execution.stages.voting,
        execution.thinning_settings,
        thinning_stage_artifacts(execution.attribute.shape),
        execution.stages.thinning,
    ):
        raise F3StageCorruptionError("cell reference thinning stage is missing")
    if skinning_enabled and not _validate_existing_stage(
        workspace,
        "skinning",
        execution.stages.thinning,
        execution.skinning_settings,
        skinning_stage_artifacts(execution.attribute.shape, enabled=True),
        execution.stages.skinning,
    ):
        raise F3StageCorruptionError("cell reference skinning stage is missing")


def _stage_consumers(
    plan: F3ModeComparisonPlan,
    executions: Mapping[str, _CellExecution],
) -> dict[tuple[str, str], tuple[str, ...]]:
    result: dict[tuple[str, str], list[str]] = {}
    for cell in plan.cells:
        stages = executions[cell.label].stages
        for kind in ("voting", "thinning", "skinning"):
            result.setdefault((kind, getattr(stages, kind)), []).append(cell.label)
    return {key: tuple(value) for key, value in result.items()}


def _resolve_cell_order(
    plan: F3ModeComparisonPlan,
    requested: Sequence[str | F3CellSpec] | None,
) -> tuple[F3CellSpec, ...]:
    if requested is None:
        return plan.cells
    by_label = {cell.label: cell for cell in plan.cells}
    labels = tuple(item.label if isinstance(item, F3CellSpec) else item for item in requested)
    if len(labels) != len(plan.cells) or set(labels) != set(by_label):
        raise ValueError("cell_order must contain every canonical cell exactly once")
    return tuple(by_label[label] for label in labels)


def _workflow_implementation_identity(
    workflow_runner: WorkflowRunner,
    declared_identity: Mapping[str, Any] | str | None,
) -> Mapping[str, Any] | str:
    if declared_identity is not None:
        if isinstance(declared_identity, str):
            if not declared_identity:
                raise ValueError("workflow_implementation_identity must not be empty")
            return declared_identity
        if isinstance(declared_identity, Mapping):
            return dict(declared_identity)
        raise TypeError("workflow_implementation_identity must be a mapping, string, or None")
    return {
        "name": (
            "pyosv.evaluation.workflow3d.execute_workflow3d"
            if workflow_runner is execute_workflow3d
            else "injected-workflow-runner"
        ),
        "callable": _callable_implementation_identity(workflow_runner),
    }


def _validate_runner_inputs(
    workspace: F3RunWorkspace,
    plan: F3ModeComparisonPlan,
    scanner_stages: Mapping[str, F3ScannerStageResult],
    workflow_runner: WorkflowRunner,
    scanner_loader: Callable[..., Any],
    runtime_hook: RuntimeHook | None,
    rss_recorder: _StageRSSRecorder | None,
    variant_spec: VariantSpec,
) -> None:
    if not isinstance(workspace, F3RunWorkspace):
        raise TypeError("workspace must be an F3RunWorkspace")
    if not isinstance(plan, F3ModeComparisonPlan):
        raise TypeError("plan must be an F3ModeComparisonPlan")
    if set(scanner_stages) != {"reference-like", "quality"}:
        raise ValueError("scanner_stages must contain reference-like and quality")
    shapes: set[tuple[int, int, int]] = set()
    for backend, stage in scanner_stages.items():
        if not isinstance(stage, F3ScannerStageResult) or stage.backend != backend:
            raise ValueError("scanner_stages backend does not match its stage")
        _validate_scanner_stage(workspace, plan, stage)
        shapes.add(stage.shape)
    if len(shapes) != 1:
        raise ValueError("scanner stages must have the same shape")
    if not callable(workflow_runner) or not callable(scanner_loader):
        raise TypeError("workflow_runner and scanner_loader must be callable")
    if runtime_hook is not None and not callable(runtime_hook):
        raise TypeError("runtime_hook must be callable or None")
    if rss_recorder is not None and (
        not callable(getattr(rss_recorder, "stage_before", None))
        or not callable(getattr(rss_recorder, "stage_after", None))
    ):
        raise TypeError("rss_recorder must provide stage_before and stage_after")
    if variant_spec != _DEFAULT_VARIANT:
        raise ValueError("canonical F3 execution requires the fixed default variant")


def _validate_scanner_stage(
    workspace: F3RunWorkspace,
    plan: F3ModeComparisonPlan,
    stage: F3ScannerStageResult,
) -> None:
    expected_path = workspace.stage_path("scanner", stage.fingerprint)
    if stage.path.absolute() != expected_path.absolute():
        raise F3StageCorruptionError("scanner stage is not in the current workspace")
    workspace_input = _workspace_dataset_file_identity(workspace, "input")
    if workspace_input.get("sha256") != stage.input_fingerprint or workspace_input.get(
        "shape"
    ) != list(stage.shape):
        raise F3StageCorruptionError(
            "scanner stage input does not match the run manifest dataset identity"
        )

    manifest = _read_json(stage.path / "stage_manifest.json")
    settings = manifest.get("resolved_settings")
    if not isinstance(settings, dict):
        raise F3StageCorruptionError("scanner stage manifest has invalid resolved_settings")
    implementation_identity = settings.get("scanner_stage_implementation_identity")
    try:
        expected_settings = scanner_stage_resolved_settings(
            plan.scanner_config_for(stage.backend),
            stage.shape,
            implementation_identity=implementation_identity,
        )
        computation = stage_computation_identity(
            "scanner",
            run_fingerprint_value=workspace.fingerprint,
            input_fingerprints={"ep.dat": stage.input_fingerprint},
            resolved_settings=expected_settings,
            artifacts=scanner_stage_artifacts(stage.shape, stage.backend),
        )
    except (TypeError, ValueError) as error:
        raise F3StageCorruptionError(f"invalid scanner stage identity: {error}") from error
    if settings != expected_settings:
        raise F3StageCorruptionError("scanner stage manifest mismatch: resolved_settings")
    if canonical_fingerprint(computation) != stage.fingerprint:
        raise F3StageCorruptionError("scanner stage fingerprint mismatch")

    validate_stage(stage.path, computation, stage.fingerprint)
    report = _read_json(stage.path / "report.json")
    expected_report_identity = {
        "fingerprint": stage.fingerprint,
        "backend": stage.backend,
        "shape": list(stage.shape),
        "resolved_config": asdict(plan.scanner_config_for(stage.backend)),
        "resolved_stage_settings": expected_settings,
    }
    for name, value in expected_report_identity.items():
        if report.get(name) != value:
            raise F3StageCorruptionError(f"scanner report mismatch: {name}")
    input_identity = report.get("input_fingerprint")
    if (
        not isinstance(input_identity, dict)
        or input_identity.get("sha256") != stage.input_fingerprint
    ):
        raise F3StageCorruptionError("scanner report mismatch: input_fingerprint")
    if report != dict(stage.report):
        raise F3StageCorruptionError("scanner stage result report mismatch")


def _runtime_kind(stage: str) -> str:
    return {
        "seed_selection": "voting",
        "voting_volume": "voting",
        "base_thinning": "thinning",
        "primary_skinning": "skinning",
    }.get(stage, stage)


def _stage_path_exists(
    workspace: F3RunWorkspace,
    kind: str,
    fingerprint: str,
) -> bool:
    path = workspace.stage_path(kind, fingerprint)
    return path.exists() or path.is_symlink()


def _stage_output_bytes(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.iterdir()
        if item.name not in {"stage_manifest.json", "complete.json"}
    )


def _open_dat(path: Path, shape: tuple[int, int, int]) -> np.memmap:
    array = np.memmap(path, dtype=_DAT_DTYPE, mode="r", shape=shape, order="C")
    array.flags.writeable = False
    return array


def _native_workflow_attributes(
    scanner: F3LoadedScannerStage,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare scanner-thinned attributes for the native-float32 workflow API."""

    return tuple(
        _native_readonly_volume(name, values, scanner.shape)
        for name, values in (
            ("fet", scanner.fet),
            ("fpt", scanner.fpt),
            ("ftt", scanner.ftt),
        )
    )


def _native_readonly_volume(
    name: str,
    values: np.ndarray,
    shape: tuple[int, int, int],
) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    native = np.asarray(array, dtype=np.float32, order="C")
    readonly = native.view()
    readonly.flags.writeable = False
    return readonly


def _close_memmaps(arrays: Sequence[np.memmap]) -> None:
    for array in arrays:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()


def _write_dat(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as stream:
        for index in range(values.shape[0]):
            np.asarray(values[index : index + 1], dtype=_DAT_DTYPE, order="C").tofile(stream)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(payload) + b"\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, ValueError) as error:
        raise F3StageCorruptionError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise F3StageCorruptionError(f"JSON artifact must contain an object: {path}")
    return value


def _skins_payload(skins: Sequence[Any]) -> dict[str, Any]:
    serialized = []
    for skin_index, skin in enumerate(skins):
        cells = sorted(skin, key=lambda cell: (cell.i3, cell.i2, cell.i1))
        serialized.append(
            {
                "skin_index": skin_index,
                "cell_count": len(cells),
                "cells": [
                    {
                        "x1": float(cell.x1),
                        "x2": float(cell.x2),
                        "x3": float(cell.x3),
                        "i1": int(cell.i1),
                        "i2": int(cell.i2),
                        "i3": int(cell.i3),
                        "fl": float(cell.fl),
                        "fp": float(cell.fp),
                        "ft": float(cell.ft),
                    }
                    for cell in cells
                ],
            }
        )
    return {
        "format_version": 1,
        "skinning_enabled": True,
        "skin_count": len(serialized),
        "skins": serialized,
    }


__all__ = [
    "F3_CELL_REFERENCE_SCHEMA_VERSION",
    "F3_CELL_RUNNER_CONTRACT_VERSION",
    "F3_THINNING_STAGE_IMPLEMENTATION",
    "F3_VOTING_STAGE_IMPLEMENTATION",
    "F3_SKINNING_STAGE_IMPLEMENTATION",
    "F3CellReference",
    "F3CellRunResult",
    "F3CellStageFingerprints",
    "F3StageRuntime",
    "build_f3d_cell_stage_fingerprints",
    "load_f3d_mode_comparison",
    "load_f3d_mode_comparison_cells",
    "run_f3d_mode_comparison",
    "run_f3d_mode_comparison_cells",
    "skinning_stage_artifacts",
    "thinning_stage_artifacts",
    "voting_stage_artifacts",
]
