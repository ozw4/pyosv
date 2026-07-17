"""Scalar semantic validation for synthetic mode-comparison results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from numbers import Integral
from typing import Any

from .builder import build_mode_comparison_plan
from .config import SyntheticModeComparisonConfig
from .contrasts import (
    AggregateRow,
    ContrastRow,
    aggregate_contrast_rows,
    aggregate_metric_rows,
    compute_contrast_rows,
)
from .experiment import RuntimeRow, SyntheticModeComparisonResult
from .metrics import (
    METRIC_REGISTRY,
    METRIC_SCHEMA_VERSION,
    MetricDefinition,
    MetricRow,
    validate_metric_value,
)
from .models import SCANNER_ONLY_SCOPE, ModeCellSpec, SyntheticModeComparisonPlan

_CACHE_COUNTERS = (
    "seed_hits",
    "seed_misses",
    "voting_hits",
    "voting_misses",
    "thinning_hits",
    "thinning_misses",
    "primary_skinning_hits",
    "primary_skinning_misses",
)


def applicable_metric_definitions(
    plan: SyntheticModeComparisonPlan,
    cell: ModeCellSpec,
) -> tuple[MetricDefinition, ...]:
    """Return registry entries applicable to one resolved plan cell."""

    if not isinstance(plan, SyntheticModeComparisonPlan):
        raise ValueError("plan must be a SyntheticModeComparisonPlan")
    if not isinstance(cell, ModeCellSpec) or cell not in plan.cells:
        raise ValueError("cell must belong to the mode-comparison plan")
    if cell.scope == SCANNER_ONLY_SCOPE:
        stages = {"scanner_raw", "scanner_thinned"}
        if cell.scanner_backend == "quality":
            stages.add("scanner_confidence")
    else:
        settings = _workflow_settings(plan, cell)
        stages = {"fv", "fvt"}
        if settings.skinning_config.enabled:
            stages.add("skin")
    return tuple(definition for definition in METRIC_REGISTRY if definition.stage in stages)


def validate_mode_comparison_result(
    result: SyntheticModeComparisonResult,
    config: SyntheticModeComparisonConfig,
) -> None:
    """Validate that ``result`` is one complete canonical scalar experiment."""

    if not isinstance(result, SyntheticModeComparisonResult):
        raise ValueError("result must be a SyntheticModeComparisonResult")
    plan = build_mode_comparison_plan(config)
    _validate_plan_and_trial_metadata(result, plan)
    _validate_cell_reports(result.cell_reports, plan)
    _validate_cache_stats(result.cache_stats, plan)
    _validate_runtime_rows(result.runtime_rows, plan)
    _validate_metric_rows(result.metric_rows, plan)

    expected_contrasts = compute_contrast_rows(result.metric_rows)
    _require_typed_sequence(result.contrast_rows, ContrastRow, "contrast_rows")
    if result.contrast_rows != expected_contrasts:
        raise ValueError("contrast_rows do not match canonical contrasts recomputed from metrics")

    expected_metric_aggregates = aggregate_metric_rows(result.metric_rows)
    _require_typed_sequence(result.metric_aggregates, AggregateRow, "metric_aggregates")
    if result.metric_aggregates != expected_metric_aggregates:
        raise ValueError("metric_aggregates do not match canonical metric aggregation")

    expected_contrast_aggregates = aggregate_contrast_rows(expected_contrasts)
    _require_typed_sequence(result.contrast_aggregates, AggregateRow, "contrast_aggregates")
    if result.contrast_aggregates != expected_contrast_aggregates:
        raise ValueError("contrast_aggregates do not match canonical contrast aggregation")


def _validate_plan_and_trial_metadata(
    result: SyntheticModeComparisonResult,
    plan: SyntheticModeComparisonPlan,
) -> None:
    if _wire_value(result.plan_metadata) != _wire_value(asdict(plan)):
        raise ValueError("plan_metadata does not match the canonical plan")
    expected_trials = tuple(_wire_value(asdict(trial)) for trial in plan.trials)
    actual_trials = tuple(_wire_value(item) for item in result.trial_metadata)
    if actual_trials != expected_trials:
        raise ValueError("trial_metadata does not match canonical trials and order")


def _validate_cell_reports(
    reports: Sequence[Mapping[str, Any]],
    plan: SyntheticModeComparisonPlan,
) -> None:
    expected_trials = plan.trials
    if not isinstance(reports, tuple) or len(reports) != len(expected_trials):
        raise ValueError("cell_reports must contain exactly one report per canonical trial")
    expected_labels = tuple(cell.label for cell in plan.cells)
    for report, trial in zip(reports, expected_trials):
        if not isinstance(report, Mapping):
            raise ValueError("cell_reports must contain only mappings")
        if set(report) != {"case_id", "trial_id", "seed", "cells"}:
            raise ValueError("cell report fields do not match the canonical schema")
        if (report["case_id"], report["trial_id"], report["seed"]) != (
            trial.case_id,
            trial.trial_id,
            trial.seed,
        ):
            raise ValueError("cell report trial metadata does not match the canonical trial")
        cells = report["cells"]
        if not isinstance(cells, Mapping) or tuple(cells) != expected_labels:
            raise ValueError("cell report cells do not match the canonical cells and order")


def _validate_cache_stats(
    rows: Sequence[Mapping[str, Any]],
    plan: SyntheticModeComparisonPlan,
) -> None:
    if not isinstance(rows, tuple) or len(rows) != len(plan.trials):
        raise ValueError("cache_stats must contain exactly one row per canonical trial")
    expected_fields = ("case_id", "trial_id", "seed", *_CACHE_COUNTERS)
    for row, trial in zip(rows, plan.trials):
        if not isinstance(row, Mapping) or set(row) != set(expected_fields):
            raise ValueError("cache stat fields do not match the canonical schema")
        if (row["case_id"], row["trial_id"], row["seed"]) != (
            trial.case_id,
            trial.trial_id,
            trial.seed,
        ):
            raise ValueError("cache stat trial metadata does not match the canonical trial")
        for name in _CACHE_COUNTERS:
            value = row[name]
            if not isinstance(value, Integral) or isinstance(value, bool) or value < 0:
                raise ValueError(f"cache counter {name} must be a non-negative integer")


def _validate_runtime_rows(
    rows: Sequence[RuntimeRow],
    plan: SyntheticModeComparisonPlan,
) -> None:
    _require_typed_sequence(rows, RuntimeRow, "runtime_rows")
    actual = tuple(_runtime_identity(row) for row in rows)
    expected: list[tuple[Any, ...]] = []
    for trial in plan.trials:
        trial_metadata = (trial.case_id, trial.trial_id, trial.seed)
        expected.extend(
            (
                (*trial_metadata, "case_generation", None, None, 1, True),
                (*trial_metadata, "scanner_input_generation", None, None, 1, True),
                (
                    *trial_metadata,
                    "scanner_scan_thinning",
                    None,
                    "reference-like",
                    1,
                    True,
                ),
                (*trial_metadata, "scanner_scan_thinning", None, "quality", 1, True),
            )
        )
        expected.extend(
            (
                *trial_metadata,
                "cell_execution",
                cell.label,
                cell.scanner_backend,
                1,
                False,
            )
            for cell in plan.cells
        )
        expected.extend(
            (
                (*trial_metadata, "metric_extraction", None, None, 1, True),
                (*trial_metadata, "contrast_extraction", None, None, 1, True),
                (*trial_metadata, "trial_total", None, None, 1, True),
            )
        )
    expected.append((None, None, None, "experiment_total", None, None, 1, True))
    if actual != tuple(expected):
        raise ValueError("runtime_rows do not match canonical stage coverage and order")


def _runtime_identity(row: RuntimeRow) -> tuple[Any, ...]:
    return (
        row.case_id,
        row.trial_id,
        row.seed,
        row.stage,
        row.cell_label,
        row.scanner_backend,
        row.call_count,
        row.shared_stage,
    )


def _validate_metric_rows(
    rows: Sequence[MetricRow],
    plan: SyntheticModeComparisonPlan,
) -> None:
    _require_typed_sequence(rows, MetricRow, "metric_rows")
    expected: list[tuple[Any, ...]] = []
    expected_metadata: dict[tuple[Any, ...], dict[str, Any]] = {}
    registry_by_identity = {
        (definition.stage, definition.selection, definition.metric): definition
        for definition in METRIC_REGISTRY
    }
    for trial in plan.trials:
        for cell in plan.cells:
            metadata = _metric_metadata(plan, trial, cell)
            for definition in applicable_metric_definitions(plan, cell):
                identity = (
                    trial.case_id,
                    trial.trial_id,
                    trial.seed,
                    cell.label,
                    definition.stage,
                    definition.selection,
                    definition.metric,
                )
                expected.append(identity)
                expected_metadata[identity] = metadata
    actual = tuple(_metric_identity(row) for row in rows)
    if actual != tuple(expected):
        raise ValueError("metric_rows do not match canonical applicability, coverage, and order")

    metadata_fields = (
        "schema_version",
        "case_id",
        "trial_id",
        "seed",
        "scope",
        "cell_label",
        "input_mode",
        "scanner_backend",
        "scanner_refinement_factor",
        "scanner_thin_mode",
        "workflow_mode",
        "voter_thin_mode",
        "skinner_method",
        "variant",
    )
    for row in rows:
        identity = _metric_identity(row)
        actual_metadata = {name: getattr(row, name) for name in metadata_fields}
        if actual_metadata != expected_metadata[identity]:
            raise ValueError("metric row metadata does not match its resolved plan cell")
        definition = registry_by_identity[(row.stage, row.selection, row.metric)]
        if (row.unit, row.direction, row.contrast_eligible) != (
            definition.unit,
            definition.direction,
            definition.contrast_eligible,
        ):
            raise ValueError("metric row semantics do not match the metric registry")
        validate_metric_value(definition, row.value)


def _metric_identity(row: MetricRow) -> tuple[Any, ...]:
    return (
        row.case_id,
        row.trial_id,
        row.seed,
        row.cell_label,
        row.stage,
        row.selection,
        row.metric,
    )


def _metric_metadata(plan, trial, cell) -> dict[str, Any]:
    scanner_backend = None
    scanner_refinement_factor = None
    scanner_thin_mode = None
    workflow_mode = None
    voter_thin_mode = None
    skinner_method = None
    if cell.scope == SCANNER_ONLY_SCOPE or cell.input_mode == "scanner":
        scanner_backend = cell.scanner_backend
        scanner_thin_mode = plan.scanner_template.scanner_thin_mode
        if scanner_backend == "quality":
            scanner_refinement_factor = plan.scanner_template.refinement_factor
    if cell.scope != SCANNER_ONLY_SCOPE:
        settings = _workflow_settings(plan, cell)
        workflow_mode = settings.workflow_mode
        voter_thin_mode = settings.voting_config.voter_thin_mode
        if settings.skinning_config.enabled:
            skinner_method = settings.skinning_config.method
    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "case_id": trial.case_id,
        "trial_id": trial.trial_id,
        "seed": trial.seed,
        "scope": cell.scope,
        "cell_label": cell.label,
        "input_mode": cell.input_mode,
        "scanner_backend": scanner_backend,
        "scanner_refinement_factor": scanner_refinement_factor,
        "scanner_thin_mode": scanner_thin_mode,
        "workflow_mode": workflow_mode,
        "voter_thin_mode": voter_thin_mode,
        "skinner_method": skinner_method,
        "variant": plan.comparison_variant,
    }


def _workflow_settings(plan: SyntheticModeComparisonPlan, cell: ModeCellSpec):
    if cell.workflow_mode == "reference":
        return plan.reference_workflow_settings
    if cell.workflow_mode == "quality":
        return plan.quality_workflow_settings
    raise ValueError("downstream cell must have a canonical workflow mode")


def _require_typed_sequence(values, item_type, name: str) -> None:
    if not isinstance(values, tuple) or any(not isinstance(value, item_type) for value in values):
        raise ValueError(f"{name} must be a tuple of {item_type.__name__} values")


def _wire_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _wire_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_wire_value(item) for item in value]
    return value


__all__ = ["applicable_metric_definitions", "validate_mode_comparison_result"]
