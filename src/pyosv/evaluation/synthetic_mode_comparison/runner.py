"""Shared-stage execution for one synthetic mode-comparison trial."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from numbers import Real
from time import perf_counter
from typing import Any, Protocol, TypeVar

import numpy as np
from pyosv.synthetic3d import Synthetic3DCase

from ..reporting.models import freeze_report_value
from ..synthetic_quality import quality_metrics
from ..synthetic_quality.cases import EXTENDED_CASES
from ..synthetic_quality.config import SyntheticScannerConfig, SyntheticTruthMetricConfig
from ..synthetic_quality.models import PipelineArtifacts
from ..synthetic_quality.profiles import ResolvedWorkflowSettings
from ..synthetic_quality.runner import (
    PreparedCaseInputs,
    PreparedScannerScalarEvidence,
    prepare_case_inputs,
    run_case_variant,
)
from ..synthetic_quality.stage_cache import PipelineStageCache, PipelineStageCacheStats
from .models import (
    END_TO_END_SCOPE,
    ORACLE_WORKFLOW_ISOLATION_SCOPE,
    SCANNER_ONLY_SCOPE,
    ModeCellSpec,
    SyntheticModeComparisonPlan,
)
from .trials import SyntheticTrialSpec

ScannerCellArtifacts = Mapping[str, np.ndarray]
SyntheticCellArtifacts = ScannerCellArtifacts | PipelineArtifacts
_T = TypeVar("_T")


class TrialRuntimeRecorder(Protocol):
    """Receiver for scalar stage timings produced by one trial."""

    def record(
        self,
        *,
        stage: str,
        elapsed_seconds: float,
        cell_label: str | None = None,
        scanner_backend: str | None = None,
        call_count: int = 1,
        shared_stage: bool,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class SyntheticCellEvaluation:
    """Report and case-local artifacts for one comparison cell."""

    cell: ModeCellSpec
    report_payload: Mapping[str, Any]
    artifacts: SyntheticCellArtifacts
    effective_scanner_config: SyntheticScannerConfig | None = None
    effective_workflow_settings: ResolvedWorkflowSettings | None = None
    variant: str = "current_default"


@dataclass(frozen=True, slots=True)
class SyntheticTrialEvaluation:
    """Complete, canonically ordered result for one synthetic trial."""

    trial: SyntheticTrialSpec
    cells: tuple[SyntheticCellEvaluation, ...]
    report_payload: Mapping[str, Mapping[str, Any]]
    stage_cache_stats: PipelineStageCacheStats
    truth_evidence: Mapping[str, int]
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "truth_evidence",
            freeze_report_value(_validate_trial_truth_evidence(self.truth_evidence)),
        )


def run_synthetic_trial(
    plan: SyntheticModeComparisonPlan,
    trial: SyntheticTrialSpec,
    *,
    clock: Any = perf_counter,
    runtime_recorder: TrialRuntimeRecorder | None = None,
) -> SyntheticTrialEvaluation:
    """Run every canonical cell for ``trial`` from one set of shared inputs."""

    _validate_trial(plan, trial)
    case = _timed_call(
        "case_generation",
        lambda: _build_trial_case(plan, trial),
        clock=clock,
        runtime_recorder=runtime_recorder,
        shared_stage=True,
    )
    if case.case_id != trial.case_id:
        raise ValueError(
            f"case factory returned {case.case_id!r}, expected trial case {trial.case_id!r}"
        )
    if case.shape != trial.shape:
        raise ValueError(
            f"case factory returned shape {case.shape}, expected trial shape {trial.shape}"
        )
    truth_evidence = _build_trial_truth_evidence(case, trial, plan.truth_metric_config)

    stage_cache = PipelineStageCache(case)
    evaluations: dict[str, SyntheticCellEvaluation] = {}
    try:
        prepared_inputs = prepare_case_inputs(
            case,
            scanner_config=plan.scanner_template,
            input_mode="both",
            scanner_backend_matrix=False,
            scanner_backends=("reference-like", "quality"),
            stage_timer=lambda stage, backend, operation: _timed_call(
                stage,
                operation,
                clock=clock,
                runtime_recorder=runtime_recorder,
                scanner_backend=backend,
                shared_stage=True,
            ),
        )
        scanner_evidence = _build_scanner_evidence(
            plan,
            prepared_inputs,
            clock=clock,
            runtime_recorder=runtime_recorder,
        )
        for cell in _execution_cells(plan):
            evaluation = _timed_call(
                "cell_execution",
                lambda cell=cell: _evaluate_cell(
                    plan,
                    cell,
                    prepared_inputs=prepared_inputs,
                    stage_cache=stage_cache,
                    prepared_scanner_scalar_evidence=(
                        scanner_evidence[cell.scanner_backend]
                        if cell.scanner_backend is not None
                        else None
                    ),
                ),
                clock=clock,
                runtime_recorder=runtime_recorder,
                cell_label=cell.label,
                scanner_backend=cell.scanner_backend,
                shared_stage=False,
            )
            evaluations[cell.label] = evaluation

        ordered_cells = tuple(evaluations[cell.label] for cell in plan.cells)
        report_payload = {
            evaluation.cell.label: evaluation.report_payload for evaluation in ordered_cells
        }
        return SyntheticTrialEvaluation(
            trial=trial,
            cells=ordered_cells,
            report_payload=report_payload,
            stage_cache_stats=stage_cache.stats,
            truth_evidence=truth_evidence,
            truth_metric_config=plan.truth_metric_config,
        )
    finally:
        stage_cache.clear()


def _evaluate_cell(
    plan: SyntheticModeComparisonPlan,
    cell: ModeCellSpec,
    *,
    prepared_inputs: PreparedCaseInputs,
    stage_cache: PipelineStageCache,
    prepared_scanner_scalar_evidence: PreparedScannerScalarEvidence | None,
) -> SyntheticCellEvaluation:
    if cell.scope == SCANNER_ONLY_SCOPE:
        return _evaluate_scanner_cell(
            cell,
            prepared_inputs=prepared_inputs,
            scanner_config=replace(plan.scanner_template, backend=cell.scanner_backend),
            prepared_scanner_scalar_evidence=prepared_scanner_scalar_evidence,
        )
    return _evaluate_downstream_cell(
        plan,
        cell,
        prepared_inputs=prepared_inputs,
        stage_cache=stage_cache,
        prepared_scanner_scalar_evidence=prepared_scanner_scalar_evidence,
    )


def _build_scanner_evidence(
    plan: SyntheticModeComparisonPlan,
    prepared_inputs: PreparedCaseInputs,
    *,
    clock: Any,
    runtime_recorder: TrialRuntimeRecorder | None,
) -> dict[str, PreparedScannerScalarEvidence]:
    scanner = prepared_inputs.scanner
    if scanner is None:
        raise RuntimeError("mode comparison requires prepared scanner attributes")
    from .metrics import build_scanner_metric_evidence

    case = prepared_inputs.case
    result = {}
    for backend, attributes in scanner.by_backend.items():
        result[backend] = _timed_call(
            "scanner_scalar_evidence",
            lambda backend=backend, attributes=attributes: _prepare_scanner_scalar_evidence(
                case=case,
                scanner_backend=backend,
                scanner_config=replace(plan.scanner_template, backend=backend),
                truth_metric_config=plan.truth_metric_config,
                scanner_report=attributes.report,
                scanner_volumes=attributes.volumes,
                metric_evidence_builder=build_scanner_metric_evidence,
            ),
            clock=clock,
            runtime_recorder=runtime_recorder,
            scanner_backend=backend,
            shared_stage=True,
        )
    return result


def _prepare_scanner_scalar_evidence(
    *,
    case: Synthetic3DCase,
    scanner_backend: str,
    scanner_config: SyntheticScannerConfig,
    truth_metric_config: SyntheticTruthMetricConfig,
    scanner_report: Mapping[str, Any],
    scanner_volumes: Mapping[str, np.ndarray],
    metric_evidence_builder: Callable[..., tuple[Mapping[str, Any], ...]],
) -> PreparedScannerScalarEvidence:
    metric_evidence = metric_evidence_builder(
        scanner_backend=scanner_backend,
        scanner_volumes=scanner_volumes,
        scanner_report=scanner_report,
        shape=case.shape,
        truth_fault=case.truth_fault_mask,
        truth_distance=case.truth_distance,
        truth_strike=case.truth_strike,
        truth_dip=case.truth_dip,
        truth_surface_half_width=truth_metric_config.truth_surface_half_width,
        buffer_radius=truth_metric_config.buffer_radius,
    )
    quality_by_stage = {
        entry["stage"]: entry["quality_report"]
        for entry in metric_evidence
        if "quality_report" in entry
    }
    try:
        raw_quality = quality_by_stage["scanner_raw"]
        thinned_quality = quality_by_stage["scanner_thinned"]
    except KeyError as error:
        raise ValueError("scanner metric evidence is missing canonical quality reports") from error
    half_width = truth_metric_config.truth_surface_half_width
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(half_width)
    far_from_truth_mask = np.abs(case.truth_distance) >= np.float32(max(3.0, half_width + 2.0))
    scanner_quality_report = {
        "ft_top_truth_count": {
            "buffered_overlap_radius2": raw_quality["buffered_overlap_radius2"],
            "surface_distance": raw_quality["surface_distance"],
        },
        "orientation_error": {
            "raw_scan_top_truth_count": raw_quality["orientation_error"],
            "used_attributes_top_truth_count": thinned_quality["orientation_error"],
        },
        "input_association": quality_metrics.scanner_input_association(
            scanner_volumes["scanner_input"],
            truth_surface_mask=truth_surface_mask,
            far_from_truth_mask=far_from_truth_mask,
        ),
    }
    return PreparedScannerScalarEvidence(
        case_id=case.case_id,
        case_token=id(case),
        shape=case.shape,
        scanner_backend=scanner_backend,
        scanner_config=scanner_config,
        truth_metric_config=truth_metric_config,
        scanner_report=scanner_report,
        scanner_quality_report=scanner_quality_report,
        scanner_metric_evidence=metric_evidence,
    )


def _timed_call(
    stage: str,
    operation: Callable[[], _T],
    *,
    clock: Any,
    runtime_recorder: TrialRuntimeRecorder | None,
    cell_label: str | None = None,
    scanner_backend: str | None = None,
    shared_stage: bool,
) -> _T:
    start = _clock_value(clock, stage)
    result = operation()
    end = _clock_value(clock, stage)
    elapsed = end - start
    if elapsed < 0.0:
        raise ValueError(f"clock moved backwards while timing {stage!r}")
    if runtime_recorder is not None:
        runtime_recorder.record(
            stage=stage,
            elapsed_seconds=elapsed,
            cell_label=cell_label,
            scanner_backend=scanner_backend,
            call_count=1,
            shared_stage=shared_stage,
        )
    return result


def _clock_value(clock: Any, stage: str) -> float:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"clock must return a finite number while timing {stage!r}")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"clock must return a finite number while timing {stage!r}")
    return normalized


def _validate_trial(plan: SyntheticModeComparisonPlan, trial: SyntheticTrialSpec) -> None:
    if not isinstance(plan, SyntheticModeComparisonPlan):
        raise ValueError("plan must be a SyntheticModeComparisonPlan")
    if not isinstance(trial, SyntheticTrialSpec):
        raise ValueError("trial must be a SyntheticTrialSpec")
    if trial not in plan.trials:
        raise ValueError("trial must belong to the mode-comparison plan")
    if trial.shape != plan.shape:
        raise ValueError(f"trial shape {trial.shape} does not match plan shape {plan.shape}")


def _build_trial_case(
    plan: SyntheticModeComparisonPlan, trial: SyntheticTrialSpec
) -> Synthetic3DCase:
    definitions = {definition.case_id: definition for definition in EXTENDED_CASES}
    return definitions[trial.case_id].build_case(trial.shape, seed=trial.seed)


def _build_trial_truth_evidence(
    case: Synthetic3DCase,
    trial: SyntheticTrialSpec,
    truth_metric_config: SyntheticTruthMetricConfig,
) -> Mapping[str, int]:
    evidence = quality_metrics.truth_report(case, truth_metric_config)
    evidence = _validate_trial_truth_evidence(evidence)
    if evidence["surface_voxel_count"] == 0:
        raise ValueError(
            "empty truth-surface support in mode-comparison configuration: "
            f"case_id={case.case_id!r}, trial_id={trial.trial_id!r}, "
            f"shape={case.shape}, "
            f"truth_surface_half_width={truth_metric_config.truth_surface_half_width}"
        )
    return freeze_report_value(evidence)


def _validate_trial_truth_evidence(evidence: Any) -> Mapping[str, int]:
    expected_fields = ("fault_voxel_count", "surface_voxel_count")
    if not isinstance(evidence, Mapping) or tuple(evidence) != expected_fields:
        raise ValueError("trial truth evidence fields do not match the canonical schema and order")
    for name in expected_fields:
        value = evidence[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"trial truth evidence {name} must be a non-negative integer")
    return evidence


def _execution_cells(plan: SyntheticModeComparisonPlan) -> tuple[ModeCellSpec, ...]:
    """Return cell execution order independently of result ordering."""

    return plan.cells


def _evaluate_scanner_cell(
    cell: ModeCellSpec,
    *,
    prepared_inputs: PreparedCaseInputs,
    scanner_config: SyntheticScannerConfig,
    prepared_scanner_scalar_evidence: PreparedScannerScalarEvidence | None,
) -> SyntheticCellEvaluation:
    scanner = prepared_inputs.scanner
    if scanner is None or cell.scanner_backend is None:
        raise RuntimeError("scanner-only comparison cell requires prepared scanner attributes")
    if prepared_scanner_scalar_evidence is None:
        raise RuntimeError("scanner-only comparison cell requires scanner scalar evidence")
    attributes = scanner.by_backend[cell.scanner_backend]
    scanner_volumes = attributes.volumes
    report = {
        "scanner": prepared_scanner_scalar_evidence.scanner_report,
        "scanner_quality": prepared_scanner_scalar_evidence.scanner_quality_report,
        "scanner_metric_evidence": (prepared_scanner_scalar_evidence.scanner_metric_evidence),
    }
    return SyntheticCellEvaluation(
        cell=cell,
        report_payload=report,
        artifacts=scanner_volumes,
        effective_scanner_config=scanner_config,
    )


def _evaluate_downstream_cell(
    plan: SyntheticModeComparisonPlan,
    cell: ModeCellSpec,
    *,
    prepared_inputs: PreparedCaseInputs,
    stage_cache: PipelineStageCache,
    prepared_scanner_scalar_evidence: PreparedScannerScalarEvidence | None,
) -> SyntheticCellEvaluation:
    if cell.workflow_mode == "reference":
        settings = plan.reference_workflow_settings
    elif cell.workflow_mode == "quality":
        settings = plan.quality_workflow_settings
    else:
        raise RuntimeError("downstream comparison cell requires a workflow mode")

    if cell.scope == ORACLE_WORKFLOW_ISOLATION_SCOPE:
        scanner_config = plan.scanner_template
    elif cell.scope == END_TO_END_SCOPE and cell.scanner_backend is not None:
        scanner_config = replace(plan.scanner_template, backend=cell.scanner_backend)
    else:
        raise RuntimeError(f"unsupported downstream comparison cell: {cell.label}")

    evaluation = run_case_variant(
        prepared_inputs.case,
        voting_config=settings.voting_config,
        scanner_config=scanner_config,
        truth_metric_config=plan.truth_metric_config,
        skinning_config=settings.skinning_config,
        variant=plan.comparison_variant,
        input_mode=cell.input_mode,
        scanner_backend_matrix=False,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=False,
        include_scanner_boundary_stage_diagnostics=False,
        prepared_inputs=prepared_inputs,
        prepared_scanner_scalar_evidence=prepared_scanner_scalar_evidence,
        stage_cache=stage_cache,
    )
    report_payload = evaluation.report_payload
    if cell.scope == END_TO_END_SCOPE:
        if prepared_scanner_scalar_evidence is None:
            raise RuntimeError("end-to-end scanner cell requires scanner metric evidence")
        report_payload = {
            **report_payload,
            "scanner_metric_evidence": (prepared_scanner_scalar_evidence.scanner_metric_evidence),
        }
    return SyntheticCellEvaluation(
        cell=cell,
        report_payload=report_payload,
        artifacts=evaluation.artifacts,
        effective_scanner_config=(scanner_config if cell.scope == END_TO_END_SCOPE else None),
        effective_workflow_settings=settings,
        variant=plan.comparison_variant,
    )
