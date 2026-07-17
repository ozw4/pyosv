"""Shared-stage execution for one synthetic mode-comparison trial."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from pyosv.synthetic3d import Synthetic3DCase

from ..synthetic_quality import quality_metrics
from ..synthetic_quality.cases import EXTENDED_CASES
from ..synthetic_quality.config import SyntheticScannerConfig, SyntheticTruthMetricConfig
from ..synthetic_quality.models import PipelineArtifacts
from ..synthetic_quality.profiles import ResolvedWorkflowSettings
from ..synthetic_quality.runner import PreparedCaseInputs, prepare_case_inputs, run_case_variant
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
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig()


def run_synthetic_trial(
    plan: SyntheticModeComparisonPlan,
    trial: SyntheticTrialSpec,
) -> SyntheticTrialEvaluation:
    """Run every canonical cell for ``trial`` from one set of shared inputs."""

    _validate_trial(plan, trial)
    case = _build_trial_case(plan, trial)
    if case.case_id != trial.case_id:
        raise ValueError(
            f"case factory returned {case.case_id!r}, expected trial case {trial.case_id!r}"
        )
    if case.shape != trial.shape:
        raise ValueError(
            f"case factory returned shape {case.shape}, expected trial shape {trial.shape}"
        )

    stage_cache = PipelineStageCache(case)
    evaluations: dict[str, SyntheticCellEvaluation] = {}
    try:
        prepared_inputs = prepare_case_inputs(
            case,
            scanner_config=plan.scanner_template,
            input_mode="both",
            scanner_backend_matrix=False,
            scanner_backends=("reference-like", "quality"),
        )
        for cell in _execution_cells(plan):
            if cell.scope == SCANNER_ONLY_SCOPE:
                evaluation = _evaluate_scanner_cell(
                    cell,
                    prepared_inputs=prepared_inputs,
                    scanner_config=replace(plan.scanner_template, backend=cell.scanner_backend),
                    truth_metric_config=plan.truth_metric_config,
                )
            else:
                evaluation = _evaluate_downstream_cell(
                    plan,
                    cell,
                    prepared_inputs=prepared_inputs,
                    stage_cache=stage_cache,
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
            truth_metric_config=plan.truth_metric_config,
        )
    finally:
        stage_cache.clear()


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


def _execution_cells(plan: SyntheticModeComparisonPlan) -> tuple[ModeCellSpec, ...]:
    """Return cell execution order independently of result ordering."""

    return plan.cells


def _evaluate_scanner_cell(
    cell: ModeCellSpec,
    *,
    prepared_inputs: PreparedCaseInputs,
    scanner_config: SyntheticScannerConfig,
    truth_metric_config: SyntheticTruthMetricConfig,
) -> SyntheticCellEvaluation:
    scanner = prepared_inputs.scanner
    if scanner is None or cell.scanner_backend is None:
        raise RuntimeError("scanner-only comparison cell requires prepared scanner attributes")
    attributes = scanner.by_backend[cell.scanner_backend]
    scanner_volumes = attributes.volumes
    report = {
        "scanner": dict(attributes.report),
        "scanner_quality": quality_metrics.scanner_truth_quality(
            prepared_inputs.case,
            scanner_volumes=scanner_volumes,
            truth_metric_config=truth_metric_config,
        ),
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
        stage_cache=stage_cache,
    )
    return SyntheticCellEvaluation(
        cell=cell,
        report_payload=evaluation.report_payload,
        artifacts=evaluation.artifacts,
        effective_scanner_config=(scanner_config if cell.scope == END_TO_END_SCOPE else None),
        effective_workflow_settings=settings,
        variant=plan.comparison_variant,
    )
