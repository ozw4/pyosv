"""Scalar semantic validation for synthetic mode-comparison results."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, replace
from math import isfinite
from numbers import Integral, Real
from typing import Any

from ..reporting.models import thaw_report_value
from ..synthetic_quality import SyntheticSkinningConfig
from ..synthetic_quality.stage_keys import (
    PipelineStageKeys,
    build_oracle_attribute_stage_key,
    build_primary_skinning_stage_key,
    build_scanner_attribute_stage_key,
    build_seed_stage_key,
    build_thinning_stage_key,
    build_voting_stage_key,
)
from ..synthetic_quality.variants import effective_skinning_config, get_variant_spec
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
from .scalar_algebra import (
    validate_downstream_quality_scalar_algebra,
    validate_scanner_quality_scalar_algebra,
    validate_skin_report_topology_algebra,
    validate_skin_topology_algebra,
)
from .trials import SyntheticTrialSpec

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
_MISSING = object()
_OVERLAP_METRICS = {
    "candidate_count",
    "buffered_precision",
    "buffered_recall",
    "buffered_f1",
}
_DISTANCE_METRICS = {
    "candidate_to_truth_median",
    "candidate_to_truth_p95",
    "truth_to_candidate_median",
    "truth_to_candidate_p95",
    "hausdorff_p95",
}
_ORIENTATION_METRICS = {
    "strike_median",
    "strike_p95",
    "dip_median",
    "dip_p95",
}
_TOPOLOGY_METRICS = {
    "skin_count",
    "largest_skin_fraction",
    "small_skin_cell_fraction",
    "duplicate_cell_count",
}
_COMPONENT_METRICS = {
    "covered_truth_component_count",
    "uncovered_truth_component_count",
    "over_merge_skin_count",
    "over_split_truth_component_count",
    "mean_skin_purity",
    "min_skin_purity",
    "mean_truth_component_recall",
    "min_truth_component_recall",
}


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
    _validate_runtime_rows(result.runtime_rows, plan)
    _validate_cell_reports(result.cell_reports, plan)
    _validate_cache_stats(result.cache_stats, plan)
    _validate_metric_rows(result.metric_rows, plan)
    _validate_shared_stage_evidence(result, plan)

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

    _validate_reported_metric_values(result, plan)


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
        try:
            for cell, (label, payload) in zip(plan.cells, cells.items(), strict=True):
                context = f"cell_reports[{trial.trial_id}].cells.{label}"
                if "scanner_quality" in payload:
                    validate_scanner_quality_scalar_algebra(
                        payload["scanner_quality"], plan.shape, f"{context}.scanner_quality"
                    )
                if "quality" in payload:
                    validate_downstream_quality_scalar_algebra(
                        payload["quality"], plan.shape, f"{context}.quality"
                    )
                    settings = _workflow_settings(plan, cell)
                    skinning_config = effective_skinning_config(
                        get_variant_spec(plan.comparison_variant), settings.skinning_config
                    )
                    _validate_downstream_topology_algebra(payload, skinning_config, context)
                    if "pipelines" in payload:
                        pipeline = payload["pipelines"][cell.input_mode]
                        _validate_downstream_topology_algebra(
                            pipeline,
                            skinning_config,
                            f"{context}.pipelines.{cell.input_mode}",
                        )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid scalar evidence in cell_reports: {error}") from error

    # The artifact loader owns the fixed recursive cell-report schema. Reuse it here
    # so in-memory results and persisted bundles enforce exactly the same scalar
    # evidence contract without running any experiment stage.
    from .artifacts import _load_cell_reports

    wire_reports = [thaw_report_value(report) for report in reports]
    try:
        _load_cell_reports(wire_reports, plan)
    except ValueError as error:
        raise ValueError(f"invalid scalar evidence in cell_reports: {error}") from error


def _validate_downstream_topology_algebra(
    payload: Mapping[str, Any], skinning_config: SyntheticSkinningConfig, context: str
) -> None:
    enabled = skinning_config.enabled
    topology = payload["pyosv"]["skins"]
    validate_skin_topology_algebra(
        topology,
        f"{context}.pyosv.skins",
        require_empty=not enabled,
    )

    skin = payload["quality"]["skin"]
    if not enabled:
        if skin is not None:
            raise ValueError(f"{context}.quality.skin must be null when skinning is disabled")
        return

    if skin is None:
        raise ValueError(f"{context}.quality.skin must be present when skinning is enabled")
    quality_topology = skin["topology"]
    validate_skin_report_topology_algebra(
        quality_topology,
        skin["component_topology"],
        f"{context}.quality.skin",
        small_skin_size=skinning_config.small_skin_size,
    )
    if _wire_value(topology) != _wire_value(quality_topology):
        raise ValueError(f"{context}.pyosv.skins does not match quality.skin.topology")


def _validate_cache_stats(
    rows: Sequence[Mapping[str, Any]],
    plan: SyntheticModeComparisonPlan,
) -> None:
    if not isinstance(rows, tuple) or len(rows) != len(plan.trials):
        raise ValueError("cache_stats must contain exactly one row per canonical trial")
    expected_fields = ("case_id", "trial_id", "seed", *_CACHE_COUNTERS)
    expected_counters = _expected_cache_counters(plan)
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
            if value != expected_counters[name]:
                raise ValueError("cache stats do not match canonical shared-stage execution")


def _expected_cache_counters(plan: SyntheticModeComparisonPlan) -> dict[str, int]:
    counters = {name: 0 for name in _CACHE_COUNTERS}
    seen: dict[str, set[Any]] = {
        "seed": set(),
        "voting": set(),
        "thinning": set(),
        "primary_skinning": set(),
    }
    for cell in plan.cells:
        keys = _resolved_stage_keys_for_cell(plan, cell)
        for stage in seen:
            key = getattr(keys, stage)
            if key is None:
                continue
            outcome = "hits" if key in seen[stage] else "misses"
            counters[f"{stage}_{outcome}"] += 1
            seen[stage].add(key)
    return counters


def _resolved_stage_keys_for_cell(
    plan: SyntheticModeComparisonPlan,
    cell: ModeCellSpec,
    trial: SyntheticTrialSpec | None = None,
) -> PipelineStageKeys:
    """Derive the cache keys looked up by one canonical cell, without execution."""

    if cell.scope == SCANNER_ONLY_SCOPE:
        return PipelineStageKeys(None, None, None, None, None)

    trial = plan.trials[0] if trial is None else trial
    if cell.input_mode == "oracle":
        attribute_key = build_oracle_attribute_stage_key(
            case_id=trial.case_id,
            shape=trial.shape,
        )
        target_source = "oracle_ft"
    elif cell.input_mode == "scanner" and cell.scanner_backend is not None:
        scanner_config = replace(plan.scanner_template, backend=cell.scanner_backend)
        attribute_key = build_scanner_attribute_stage_key(
            case_id=trial.case_id,
            shape=trial.shape,
            scanner_config=scanner_config,
        )
        target_source = "scanner_fet"
    else:
        raise ValueError("downstream cell must have a canonical attribute source")

    settings = _workflow_settings(plan, cell)
    variant_spec = get_variant_spec(plan.comparison_variant)
    skinning_config = effective_skinning_config(variant_spec, settings.skinning_config)
    seed_key = build_seed_stage_key(
        attribute_key=attribute_key,
        voting_config=settings.voting_config,
        variant_spec=variant_spec,
        target_source=target_source,
    )
    voting_key = build_voting_stage_key(
        seed_key=seed_key,
        voting_config=settings.voting_config,
        variant_spec=variant_spec,
    )
    thinning_key = build_thinning_stage_key(
        voting_key=voting_key,
        voting_config=settings.voting_config,
        variant_spec=variant_spec,
    )
    primary_skinning_key = build_primary_skinning_stage_key(
        thinning_key=thinning_key,
        skinning_config=skinning_config,
        variant_spec=variant_spec,
        target_source=target_source,
    )
    return PipelineStageKeys(
        attribute=attribute_key,
        seed=seed_key,
        voting=voting_key,
        thinning=thinning_key,
        primary_skinning=primary_skinning_key,
    )


def _validate_shared_stage_evidence(
    result: SyntheticModeComparisonResult,
    plan: SyntheticModeComparisonPlan,
) -> None:
    """Validate exact scalar evidence reuse for semantically shared trial stages."""

    rows_by_cell_stage: dict[tuple[str, str, int | None, str, str], list[MetricRow]] = {}
    for row in result.metric_rows:
        key = (row.case_id, row.trial_id, row.seed, row.cell_label, row.stage)
        rows_by_cell_stage.setdefault(key, []).append(row)

    for report, trial in zip(result.cell_reports, plan.trials):
        cells = report["cells"]
        scanner_cells = tuple(cell for cell in plan.cells if cell.scanner_backend is not None)
        _require_shared_evidence(
            "scanner input",
            (
                (
                    cell.label,
                    {
                        "summary": cells[cell.label]["scanner"]["input"],
                        "config": cells[cell.label]["scanner"]["config"]["input"],
                    },
                )
                for cell in scanner_cells
            ),
        )

        attribute_groups: dict[Any, list[ModeCellSpec]] = {}
        voting_groups: dict[Any, list[ModeCellSpec]] = {}
        thinning_groups: dict[Any, list[ModeCellSpec]] = {}
        for cell in plan.cells:
            keys = _resolved_stage_keys_for_cell(plan, cell, trial)
            if cell.scanner_backend is not None:
                scanner_config = replace(plan.scanner_template, backend=cell.scanner_backend)
                attribute_key = build_scanner_attribute_stage_key(
                    case_id=trial.case_id,
                    shape=trial.shape,
                    scanner_config=scanner_config,
                )
                attribute_groups.setdefault(attribute_key, []).append(cell)
            if keys.voting is not None:
                voting_groups.setdefault(keys.voting, []).append(cell)
            if keys.thinning is not None:
                thinning_groups.setdefault(keys.thinning, []).append(cell)

        for group in attribute_groups.values():
            _require_shared_evidence(
                "attribute stage",
                (
                    (
                        cell.label,
                        {
                            "scanner": cells[cell.label]["scanner"],
                            "scanner_quality": cells[cell.label]["scanner_quality"],
                            "scanner_metric_evidence": cells[cell.label]["scanner_metric_evidence"],
                        },
                    )
                    for cell in group
                ),
            )
        for group in voting_groups.values():
            _require_shared_evidence(
                "voting stage",
                (
                    (
                        cell.label,
                        _downstream_stage_evidence(
                            cells[cell.label],
                            stage="fv",
                            metric_rows=rows_by_cell_stage[
                                (
                                    trial.case_id,
                                    trial.trial_id,
                                    trial.seed,
                                    cell.label,
                                    "fv",
                                )
                            ],
                        ),
                    )
                    for cell in group
                ),
            )
        for group in thinning_groups.values():
            _require_shared_evidence(
                "thinning stage",
                (
                    (
                        cell.label,
                        _downstream_stage_evidence(
                            cells[cell.label],
                            stage="fvt",
                            metric_rows=rows_by_cell_stage[
                                (
                                    trial.case_id,
                                    trial.trial_id,
                                    trial.seed,
                                    cell.label,
                                    "fvt",
                                )
                            ],
                        ),
                    )
                    for cell in group
                ),
            )


def _downstream_stage_evidence(
    payload: Mapping[str, Any],
    *,
    stage: str,
    metric_rows: Sequence[MetricRow],
) -> dict[str, Any]:
    selections = (f"{stage}_top_truth_count", f"{stage}_positive_top_truth_count")
    evidence = {
        "array": payload["pyosv"][stage],
        "quality": {name: payload["quality"][name] for name in selections},
        "edge_false_positive": {
            name: payload["quality"]["edge_false_positive"][name] for name in selections
        },
        "metric_rows": tuple(
            (
                row.stage,
                row.selection,
                row.metric,
                row.value,
                row.unit,
                row.direction,
                row.contrast_eligible,
            )
            for row in metric_rows
        ),
    }
    if stage == "fv":
        evidence["voting"] = payload["pyosv"]["voting"]
    return evidence


def _require_shared_evidence(
    stage: str,
    members: Iterable[tuple[str, Any]],
) -> None:
    normalized = tuple((label, _wire_value(evidence)) for label, evidence in members)
    if len(normalized) < 2:
        return
    reference_label, reference = normalized[0]
    for label, evidence in normalized[1:]:
        if evidence != reference:
            raise ValueError(
                f"cell_reports shared {stage} evidence does not match between "
                f"{reference_label} and {label}"
            )


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


def _validate_reported_metric_values(
    result: SyntheticModeComparisonResult,
    plan: SyntheticModeComparisonPlan,
) -> None:
    reports = {
        (report["case_id"], report["trial_id"], report["seed"]): report["cells"]
        for report in result.cell_reports
    }
    for row in result.metric_rows:
        cells = reports[(row.case_id, row.trial_id, row.seed)]
        payload = cells[row.cell_label]
        reported = _reported_metric_value(payload, row)
        if reported is _MISSING:
            continue
        if isinstance(reported, bool) or not isinstance(reported, Real):
            raise ValueError("cell report metric evidence must be numeric")
        normalized = float(reported)
        if not isfinite(normalized) or normalized != row.value:
            raise ValueError(
                "metric_rows do not match scalar evidence in cell_reports: "
                f"{row.cell_label}/{row.stage}/{row.selection}/{row.metric} "
                f"has {row.value}, report has {normalized}"
            )


def _reported_metric_value(payload: Mapping[str, Any], row: MetricRow) -> Any:
    if row.stage.startswith("scanner_") or row.stage == "scanner_confidence":
        return _reported_scanner_evidence_value(payload, row)

    if row.metric == "array_nonzero_fraction":
        report_name = {
            "fv": "fv",
            "fvt": "fvt",
        }[row.stage]
        return _report_path(payload, "pyosv", report_name, "nonzero_fraction")
    return _reported_downstream_quality_value(payload, row)


def _reported_scanner_evidence_value(payload: Mapping[str, Any], row: MetricRow) -> Any:
    evidence = payload.get("scanner_metric_evidence", _MISSING)
    if evidence is _MISSING or not isinstance(evidence, (tuple, list)):
        raise ValueError("applicable scanner metric is missing scanner_metric_evidence")
    identity = (row.stage, row.selection, row.metric)
    matches = []
    for entry in evidence:
        if not isinstance(entry, Mapping):
            raise ValueError("scanner_metric_evidence entries must be mappings")
        if tuple(entry.get(name) for name in ("stage", "selection", "metric")) == identity:
            matches.append(entry)
    if len(matches) != 1 or "value" not in matches[0]:
        raise ValueError(
            "applicable scanner metric evidence lookup failed for " + "/".join(identity)
        )
    return matches[0]["value"]


def _reported_downstream_quality_value(payload: Mapping[str, Any], row: MetricRow) -> Any:
    quality_key = "skin" if row.stage == "skin" else f"{row.stage}_{row.selection}"
    if row.metric in _OVERLAP_METRICS:
        return _report_path(
            payload,
            "quality",
            quality_key,
            "buffered_overlap_radius2",
            row.metric,
        )
    if row.metric in _DISTANCE_METRICS:
        return _report_path(
            payload,
            "quality",
            quality_key,
            "surface_distance",
            row.metric,
        )
    if row.metric in _ORIENTATION_METRICS:
        return _report_path(
            payload,
            "quality",
            quality_key,
            "orientation_error",
            row.metric,
        )
    if row.metric == "edge_false_positive_fraction_of_candidates":
        return _report_path(
            payload,
            "quality",
            "edge_false_positive",
            quality_key,
            row.metric,
        )
    if row.metric in _TOPOLOGY_METRICS:
        return _report_path(payload, "quality", "skin", "topology", row.metric)
    if row.metric in _COMPONENT_METRICS:
        return _report_path(payload, "quality", "skin", "component_topology", row.metric)
    return _MISSING


def _report_path(value: Any, *path: str) -> Any:
    current = value
    for name in path:
        if not isinstance(current, Mapping) or name not in current:
            raise ValueError(f"cell report is missing metric evidence at {'.'.join(path)}")
        current = current[name]
    return current


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
    if isinstance(value, bool):
        return (bool, value)
    if isinstance(value, Integral):
        return (int, int(value))
    if isinstance(value, Real):
        return (float, float(value))
    return (type(value), value)


__all__ = ["applicable_metric_definitions", "validate_mode_comparison_result"]
