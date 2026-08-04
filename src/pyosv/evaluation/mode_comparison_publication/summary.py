"""Build normalized publication tables from validated source rows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

import numpy as np

from .config import (
    CANONICAL_CELL_ORDER,
    CONTRAST_NAMES,
    F3_ORIENTATION_SUMMARY_HEADER,
    F3_REGIONAL_SUMMARY_HEADER,
    F3_SEMANTICS,
    PUBLICATION_CONTRASTS_HEADER,
    PUBLICATION_METRICS_HEADER,
    PUBLICATION_SUMMARY_HEADER,
    RUNTIME_SUMMARY_HEADER,
    SYNTHETIC_STAGE_ORDER,
    SYNTHETIC_SCANNER_CELL_ORDER,
    SYNTHETIC_SEMANTICS,
)
from .models import F3SourceBundle, SyntheticSourceBundle
from .registry import PUBLICATION_METRIC_REGISTRY, PublicationMetric


def _source_identity(entry: PublicationMetric) -> tuple[str, str, str]:
    return entry.stage, entry.selection, entry.metric


def _statistic_rows(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("publication summary values must be non-empty and finite")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "q25": float(np.quantile(array, 0.25, method="linear")),
        "q75": float(np.quantile(array, 0.75, method="linear")),
    }


def _synthetic_axes(row: Any) -> tuple[str | None, str | None]:
    return row.scanner_backend, row.workflow_mode


def _f3_axes(row: Any) -> tuple[str, str]:
    return row.scanner_backend, row.workflow_mode


def _selected_synthetic_rows(
    source: SyntheticSourceBundle, entry: PublicationMetric
) -> tuple[Any, ...]:
    allowed_cells = (
        SYNTHETIC_SCANNER_CELL_ORDER if entry.stage == "scanner_raw" else CANONICAL_CELL_ORDER
    )
    rows = tuple(
        row
        for row in source.metric_rows
        if row.cell_label in allowed_cells
        and (row.stage, row.selection, row.metric) == _source_identity(entry)
    )
    if not rows:
        if entry.required or source.skinning_enabled:
            raise ValueError(
                f"required synthetic publication metric is missing: {entry.identity!r}"
            )
        return ()
    by_trial_cell = {(row.case_id, row.trial_id, row.seed, row.cell_label) for row in rows}
    expected_cells = set(allowed_cells)
    for case_id, trial_id, seed, _cell in sorted(
        by_trial_cell,
        key=lambda item: (
            item[0],
            item[1],
            -1 if item[2] is None else item[2],
            item[3],
        ),
    ):
        present = {
            row.cell_label
            for row in rows
            if (row.case_id, row.trial_id, row.seed) == (case_id, trial_id, seed)
        }
        if present != expected_cells:
            raise ValueError(
                f"synthetic metric {entry.identity!r} has incomplete canonical cell coverage "
                f"for trial {trial_id!r}"
            )
    case_order = {case: index for index, case in enumerate(source.case_order)}
    cell_order = {cell: index for index, cell in enumerate(allowed_cells)}
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                case_order.get(row.case_id, len(case_order)),
                row.trial_id,
                -1 if row.seed is None else row.seed,
                cell_order[row.cell_label],
            ),
        )
    )


def _selected_f3_rows(source: F3SourceBundle, entry: PublicationMetric) -> tuple[Any, ...]:
    rows = tuple(
        row
        for row in source.result.metric_rows
        if row.cell_label in CANONICAL_CELL_ORDER
        and (row.stage, row.selection, row.metric) == _source_identity(entry)
    )
    if not rows:
        if entry.required:
            raise ValueError(f"required F3 publication metric is missing: {entry.identity!r}")
        return ()
    if {row.cell_label for row in rows} != set(CANONICAL_CELL_ORDER):
        raise ValueError(f"F3 publication metric has incomplete cell coverage: {entry.identity!r}")
    return tuple(sorted(rows, key=lambda row: CANONICAL_CELL_ORDER.index(row.cell_label)))


def _metric_table(
    synthetic: SyntheticSourceBundle,
    f3: F3SourceBundle,
) -> tuple[tuple[Mapping[str, Any], ...], dict[tuple[str, str, str, str], tuple[Any, ...]]]:
    output: list[Mapping[str, Any]] = []
    selected: dict[tuple[str, str, str, str], tuple[Any, ...]] = {}
    for entry in PUBLICATION_METRIC_REGISTRY:
        source_rows = (
            _selected_synthetic_rows(synthetic, entry)
            if entry.dataset == "synthetic"
            else _selected_f3_rows(f3, entry)
        )
        selected[entry.identity] = source_rows
        for row in source_rows:
            if entry.dataset == "synthetic":
                case_or_region = row.case_id
                trial_id = row.trial_id
                seed = row.seed
                scanner_backend, workflow_mode = _synthetic_axes(row)
                value = row.value
                source_artifact = "metrics_long.csv"
            else:
                case_or_region = "full"
                trial_id = None
                seed = None
                scanner_backend, workflow_mode = _f3_axes(row)
                value = row.value
                source_artifact = "reports/metrics_long.csv"
            output.append(
                {
                    "dataset": entry.dataset,
                    "evaluation_semantics": entry.evaluation_semantics,
                    "case_or_region": case_or_region,
                    "trial_id": trial_id,
                    "seed": seed,
                    "cell_label": row.cell_label,
                    "scanner_backend": scanner_backend,
                    "workflow_mode": workflow_mode,
                    "stage": row.stage,
                    "selection": row.selection,
                    "metric": row.metric,
                    "value": value,
                    "unit": entry.unit,
                    "direction": entry.direction,
                    "source_artifact": source_artifact,
                }
            )
    return tuple(output), selected


def _contrast_table(
    synthetic: SyntheticSourceBundle,
    f3: F3SourceBundle,
    selected: Mapping[tuple[str, str, str, str], tuple[Any, ...]],
) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    for entry in PUBLICATION_METRIC_REGISTRY:
        source_rows = selected[entry.identity]
        if not source_rows:
            continue
        if entry.dataset == "synthetic":
            source_contrasts = synthetic.contrast_rows
            for source_row in source_contrasts:
                if (
                    source_row.contrast_name not in CONTRAST_NAMES
                    or (source_row.stage, source_row.selection, source_row.metric)
                    != _source_identity(entry)
                    or source_row.case_id not in {row.case_id for row in source_rows}
                ):
                    continue
                output.append(
                    {
                        "dataset": "synthetic",
                        "evaluation_semantics": SYNTHETIC_SEMANTICS,
                        "case_or_region": source_row.case_id,
                        "trial_id": source_row.trial_id,
                        "seed": source_row.seed,
                        "contrast_name": source_row.contrast_name,
                        "stage": source_row.stage,
                        "selection": source_row.selection,
                        "metric": source_row.metric,
                        "raw_value": source_row.raw_value,
                        "improvement_value": source_row.improvement_value,
                        "unit": source_row.unit,
                        "direction": source_row.direction,
                        "component_cells": source_row.component_cells,
                        "source_artifact": "contrasts.csv",
                    }
                )
        else:
            source_contrasts = f3.result.contrast_rows
            for source_row in source_contrasts:
                if source_row.contrast_name not in CONTRAST_NAMES or (
                    source_row.stage,
                    source_row.selection,
                    source_row.metric,
                ) != _source_identity(entry):
                    continue
                output.append(
                    {
                        "dataset": "f3",
                        "evaluation_semantics": F3_SEMANTICS,
                        "case_or_region": "full",
                        "trial_id": None,
                        "seed": None,
                        "contrast_name": source_row.contrast_name,
                        "stage": source_row.stage,
                        "selection": source_row.selection,
                        "metric": source_row.metric,
                        "raw_value": source_row.raw_value,
                        "improvement_value": source_row.improvement_value,
                        "unit": source_row.unit,
                        "direction": source_row.direction,
                        "component_cells": source_row.component_cells,
                        "source_artifact": "reports/contrasts.csv",
                    }
                )
    contrast_order = {name: index for index, name in enumerate(CONTRAST_NAMES)}
    metric_order = {
        entry.identity: index for index, entry in enumerate(PUBLICATION_METRIC_REGISTRY)
    }
    return tuple(
        sorted(
            output,
            key=lambda row: (
                0 if row["dataset"] == "synthetic" else 1,
                row["case_or_region"],
                row["trial_id"] or "",
                -1 if row["seed"] is None else row["seed"],
                contrast_order[row["contrast_name"]],
                metric_order[
                    (
                        row["dataset"],
                        row["stage"],
                        row["selection"],
                        row["metric"],
                    )
                ],
            ),
        )
    )


def _summary_table(
    metric_rows: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], ...]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    metadata: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in metric_rows:
        value = row["value"]
        if value is None or value == "":
            # Nullable sparse-distance evidence remains absent from descriptive statistics.
            continue
        key = (
            row["dataset"],
            row["evaluation_semantics"],
            row["case_or_region"],
            row["stage"],
            row["selection"],
            row["metric"],
            row["cell_label"],
            row["unit"],
            row["direction"],
        )
        groups[key].append(float(value))
        metadata[key] = row

    output: list[Mapping[str, Any]] = []
    dataset_order = {"synthetic": 0, "f3": 1}
    case_order: dict[str, int] = {}
    for row in metric_rows:
        case_order.setdefault(row["case_or_region"], len(case_order))
    stage_order = {
        stage: index
        for index, stage in enumerate(SYNTHETIC_STAGE_ORDER + ("ft", "fv", "fvt", "skin"))
    }
    cell_order = {
        cell: index
        for index, cell in enumerate(SYNTHETIC_SCANNER_CELL_ORDER + CANONICAL_CELL_ORDER)
    }
    for key in sorted(
        groups,
        key=lambda value: (
            dataset_order.get(value[0], 99),
            case_order.get(value[2], 99),
            stage_order.get(value[3], 99),
            value[4],
            value[5],
            cell_order.get(value[6], 99),
        ),
    ):
        stats = _statistic_rows(groups[key])
        meta = metadata[key]
        output.append(
            {
                "dataset": key[0],
                "evaluation_semantics": key[1],
                "case_or_region": key[2],
                "stage": key[3],
                "selection": key[4],
                "metric": key[5],
                "cell_label": key[6],
                **stats,
                "unit": meta["unit"],
                "direction": meta["direction"],
            }
        )
    return tuple(output)


def _diagnostic_unit(metric: str) -> str:
    if metric.endswith("fraction") or "ratio" in metric or metric in {"precision", "recall"}:
        return "fraction"
    if metric.endswith("count") or metric in {"voxel_count", "candidate_count", "reference_count"}:
        return "count"
    if "distance" in metric:
        return "voxel"
    return "value"


def _regional_table(f3: F3SourceBundle) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    for row in f3.result.regional_rows:
        for metric, value in row.metrics.items():
            output.append(
                {
                    "dataset": "f3",
                    "evaluation_semantics": F3_SEMANTICS,
                    "case_or_region": row.region,
                    "stage": row.stage,
                    "cell_label": row.cell_label,
                    "scanner_backend": row.scanner_backend,
                    "workflow_mode": row.workflow_mode,
                    "region": row.region,
                    "metric": metric,
                    "display_label": metric.replace("_", " "),
                    "value": value,
                    "unit": _diagnostic_unit(metric),
                    "source_artifact": "reports/regional_metrics.csv",
                }
            )
    return tuple(output)


def _orientation_table(f3: F3SourceBundle) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    for row in f3.result.orientation_rows:
        summaries = (
            ("strike_circular_absolute_difference", row.strike_circular_absolute_difference),
            ("dip_absolute_difference", row.dip_absolute_difference),
            ("normal_vector_angular_difference", row.normal_vector_angular_difference),
        )
        for prefix, summary in summaries:
            for statistic, value in summary.items():
                output.append(
                    {
                        "dataset": "f3",
                        "evaluation_semantics": F3_SEMANTICS,
                        "case_or_region": "full",
                        "stage": row.stage,
                        "left_cell": row.left_cell,
                        "right_cell": row.right_cell,
                        "support_contract": row.support_contract,
                        "support_count": row.support_count,
                        "metric": f"{prefix}.{statistic}",
                        "display_label": f"{prefix.replace('_', ' ')} {statistic}",
                        "value": value,
                        "unit": "count" if statistic == "count" else "degree",
                        "source_artifact": "reports/orientation_diagnostics.csv",
                    }
                )
    return tuple(output)


def _runtime_table(
    synthetic: SyntheticSourceBundle,
    f3: F3SourceBundle,
) -> tuple[Mapping[str, Any], ...]:
    output: list[Mapping[str, Any]] = []
    for row in synthetic.runtime_rows:
        output.append(
            {
                "dataset": "synthetic",
                "evaluation_semantics": SYNTHETIC_SEMANTICS,
                "case_or_region": row.case_id or "experiment",
                "trial_id": row.trial_id,
                "seed": row.seed,
                "stage": row.stage,
                "fingerprint": "",
                "cell_label": row.cell_label,
                "cell_consumers": (),
                "state": "shared" if row.shared_stage else "cell-owned",
                "elapsed_seconds": row.elapsed_seconds,
                "elapsed_semantics": "within_experiment_attribution",
                "shared_stage": row.shared_stage,
                "attribution": "shared-stage" if row.shared_stage else "cell-owned-stage",
                "source_artifact": "runtime.csv",
            }
        )
    for row in f3.result.runtime_rows:
        consumers = tuple(row.cell_consumers)
        output.append(
            {
                "dataset": "f3",
                "evaluation_semantics": F3_SEMANTICS,
                "case_or_region": "full",
                "trial_id": None,
                "seed": None,
                "stage": row.stage_kind,
                "fingerprint": row.fingerprint,
                "cell_label": row.cell,
                "cell_consumers": consumers,
                "state": row.state,
                "elapsed_seconds": row.elapsed_seconds,
                "elapsed_semantics": row.elapsed_semantics,
                "shared_stage": len(consumers) > 1,
                "attribution": "shared-stage" if len(consumers) > 1 else "cell-owned-stage",
                "source_artifact": "reports/runtime.csv",
            }
        )
    return tuple(output)


def build_tables(
    synthetic: SyntheticSourceBundle,
    f3: F3SourceBundle,
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    """Build all machine-readable publication tables without rerunning a source."""

    metrics, selected = _metric_table(synthetic, f3)
    return {
        "publication_metrics.csv": metrics,
        "publication_contrasts.csv": _contrast_table(synthetic, f3, selected),
        "publication_summary.csv": _summary_table(metrics),
        "f3_regional_summary.csv": _regional_table(f3),
        "f3_orientation_summary.csv": _orientation_table(f3),
        "runtime_summary.csv": _runtime_table(synthetic, f3),
    }


TABLE_HEADERS: Mapping[str, tuple[str, ...]] = {
    "publication_metrics.csv": PUBLICATION_METRICS_HEADER,
    "publication_contrasts.csv": PUBLICATION_CONTRASTS_HEADER,
    "publication_summary.csv": PUBLICATION_SUMMARY_HEADER,
    "f3_regional_summary.csv": F3_REGIONAL_SUMMARY_HEADER,
    "f3_orientation_summary.csv": F3_ORIENTATION_SUMMARY_HEADER,
    "runtime_summary.csv": RUNTIME_SUMMARY_HEADER,
}

__all__ = ["TABLE_HEADERS", "build_tables"]
