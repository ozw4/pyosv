"""Validate-only import isolation.

This test intentionally does not use the generation fixture: that fixture has
to render figures and therefore legitimately needs matplotlib.  The clean
subprocess makes accidental matplotlib imports observable even when another
test in the parent process has already rendered a publication bundle.
"""

from __future__ import annotations

from collections import defaultdict
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from pyosv.evaluation import synthetic_mode_comparison
from pyosv.evaluation.f3d_mode_comparison import (
    F3_BUFFERED_PERCENTILE,
    F3_BUFFER_RADIUS,
    F3_METRIC_SCHEMA_VERSION,
    F3_REFERENCE_STAGE_FILES,
    MetricEvidence,
)
from pyosv.evaluation.mode_comparison_publication import artifacts
from pyosv.evaluation.mode_comparison_publication.config import (
    CANONICAL_CELL_ORDER,
    CANONICAL_STAGE_ORDER,
    F3_RIDGE_OVERLAY_SLOTS,
    F3_SEMANTICS,
    F3_SPATIAL_FIGURE_SLOTS,
    FIGURE_DATA_HEADER,
    FIGURE_DATA_IDENTITY_FIELDS,
    FIXED_SCALAR_FIGURE_IDS,
    ROOT_TABLE_FILES,
    SYNTHETIC_SCANNER_CELL_ORDER,
    SYNTHETIC_SEMANTICS,
    SYNTHETIC_SKIN_FIGURE_OMISSION_REASON,
)
from pyosv.evaluation.mode_comparison_publication.figures import (
    _data_row,
    _record,
    _runtime_panel_label,
)
from pyosv.evaluation.mode_comparison_publication.models import (
    F3SourceBundle,
    PublicationReport,
    SyntheticSourceBundle,
)
from pyosv.evaluation.mode_comparison_publication.registry import PUBLICATION_METRIC_REGISTRY
from pyosv.evaluation.mode_comparison_publication.semantic import (
    F3_SOURCE_IDENTITY_FIELDS,
    FIGURE_DATA_FIELD_TYPES,
    ROOT_TABLE_FIELD_TYPES,
    SYNTHETIC_SOURCE_IDENTITY_FIELDS,
    build_table_contract,
    canonical_digest,
)
from pyosv.evaluation.mode_comparison_publication.summary import TABLE_HEADERS
from pyosv.evaluation.mode_comparison_publication.validation import validate_publication_bundle
from pyosv.evaluation.publication_manifest import build_publication_manifest
from pyosv.evaluation.publication_manifest_io import (
    artifact_file_record,
    validate_publication_directory,
    write_publication_manifest,
)
from tests.evaluation.mode_comparison_publication.artifact_test_support import (
    write_csv_rows,
    write_png,
)


_CONDITION_AXES = {
    "RL-REF": ("reference-like", "reference"),
    "RL-QUAL": ("reference-like", "quality"),
    "Q-REF": ("quality", "reference"),
    "Q-QUAL": ("quality", "quality"),
}
_SCANNER_AXES = {
    "RL-SCAN": ("reference-like", None),
    "Q-SCAN": ("quality", None),
}
_F3_REFERENCE_SELECTION_THRESHOLDS = {"ft": 0.50, "fv": 0.50, "fvt": 0.50}
_F3_CANDIDATE_SELECTION_THRESHOLDS = {
    "ft": {"RL-REF": 0.40, "RL-QUAL": 0.45, "Q-REF": 0.50, "Q-QUAL": 0.55},
    "fv": {"RL-REF": 0.40, "RL-QUAL": 0.45, "Q-REF": 0.50, "Q-QUAL": 0.55},
    "fvt": {"RL-REF": 0.40, "RL-QUAL": 0.45, "Q-REF": 0.50, "Q-QUAL": 0.55},
}


class _MinimalDatasetSpec:
    """Only the source-manifest method exercised by the publication writer."""

    _FILENAMES = {
        "reference_fault_likelihood": "fl.dat",
        "reference_fault_votes": "fv.dat",
        "reference_thinned_fault_votes": "fvt.dat",
    }

    def filename_for(self, role: str) -> str:
        return self._FILENAMES[role]


def _f3_ridge_metric_evidence() -> tuple[MetricEvidence, ...]:
    return tuple(
        MetricEvidence(
            schema_version=F3_METRIC_SCHEMA_VERSION,
            dataset_id="minimal-f3",
            cell_label=cell,
            stage=stage,
            region="full",
            selection="positive_p99_radius2",
            reference_file=F3_REFERENCE_STAGE_FILES[stage],
            source_stage_fingerprint="a" * 64,
            reference_sha256="b" * 64,
            shape=(1, 1, 1),
            thresholds=(
                ("percentile", F3_BUFFERED_PERCENTILE),
                ("radius", F3_BUFFER_RADIUS),
                ("reference_threshold", _F3_REFERENCE_SELECTION_THRESHOLDS[stage]),
                (
                    "candidate_threshold",
                    _F3_CANDIDATE_SELECTION_THRESHOLDS[stage][cell],
                ),
            ),
        )
        for stage in CANONICAL_STAGE_ORDER
        for cell in CANONICAL_CELL_ORDER
    )


def _metric_tables() -> tuple[list[dict[str, Any]], list[SimpleNamespace], list[SimpleNamespace]]:
    """Build a one-trial/full-volume source-equivalent metric coverage set."""

    table_rows: list[dict[str, Any]] = []
    synthetic_rows: list[SimpleNamespace] = []
    f3_rows: list[SimpleNamespace] = []
    for entry in PUBLICATION_METRIC_REGISTRY:
        if entry.dataset == "synthetic":
            if entry.stage == "skin":
                continue
            cells = (
                SYNTHETIC_SCANNER_CELL_ORDER
                if entry.stage == "scanner_raw"
                else CANONICAL_CELL_ORDER
            )
            for cell in cells:
                scanner_backend, workflow_mode = (
                    _SCANNER_AXES[cell] if entry.stage == "scanner_raw" else _CONDITION_AXES[cell]
                )
                source = SimpleNamespace(
                    case_id="minimal_case",
                    trial_id="minimal_trial",
                    seed=7,
                    cell_label=cell,
                    stage=entry.stage,
                    selection=entry.selection,
                    metric=entry.metric,
                )
                synthetic_rows.append(source)
                table_rows.append(
                    {
                        "dataset": "synthetic",
                        "evaluation_semantics": SYNTHETIC_SEMANTICS,
                        "case_or_region": source.case_id,
                        "trial_id": source.trial_id,
                        "seed": source.seed,
                        "cell_label": cell,
                        "scanner_backend": scanner_backend,
                        "workflow_mode": workflow_mode,
                        "stage": entry.stage,
                        "selection": entry.selection,
                        "metric": entry.metric,
                        "value": 1.0,
                        "unit": entry.unit,
                        "direction": entry.direction,
                        "source_artifact": "metrics_long.csv",
                    }
                )
            continue

        if entry.stage == "skin":
            continue
        for cell in CANONICAL_CELL_ORDER:
            scanner_backend, workflow_mode = _CONDITION_AXES[cell]
            source = SimpleNamespace(
                cell_label=cell,
                stage=entry.stage,
                selection=entry.selection,
                metric=entry.metric,
            )
            f3_rows.append(source)
            table_rows.append(
                {
                    "dataset": "f3",
                    "evaluation_semantics": F3_SEMANTICS,
                    "case_or_region": "full",
                    "trial_id": None,
                    "seed": None,
                    "cell_label": cell,
                    "scanner_backend": scanner_backend,
                    "workflow_mode": workflow_mode,
                    "stage": entry.stage,
                    "selection": entry.selection,
                    "metric": entry.metric,
                    "value": None if entry.nullable else 1.0,
                    "unit": entry.unit,
                    "direction": entry.direction,
                    "source_artifact": "reports/metrics_long.csv",
                }
            )
    return table_rows, synthetic_rows, f3_rows


def _summary_table(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in metric_rows:
        if row["value"] is None:
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
        groups[key].append(float(row["value"]))
    stage_order = {
        stage: index
        for index, stage in enumerate(("scanner_raw", "fvt", "skin", "ft", "fv", "fvt"))
    }
    cell_order = {
        cell: index
        for index, cell in enumerate(SYNTHETIC_SCANNER_CELL_ORDER + CANONICAL_CELL_ORDER)
    }
    output: list[dict[str, Any]] = []
    for key in sorted(
        groups,
        key=lambda value: (
            0 if value[0] == "synthetic" else 1,
            value[2],
            stage_order[value[3]],
            value[4],
            value[5],
            cell_order[value[6]],
        ),
    ):
        values = groups[key]
        value = values[0]
        output.append(
            {
                "dataset": key[0],
                "evaluation_semantics": key[1],
                "case_or_region": key[2],
                "stage": key[3],
                "selection": key[4],
                "metric": key[5],
                "cell_label": key[6],
                "n": len(values),
                "mean": value,
                "median": value,
                "minimum": value,
                "maximum": value,
                "q25": value,
                "q75": value,
                "unit": key[7],
                "direction": key[8],
            }
        )
    return output


def _supporting_tables() -> tuple[
    list[dict[str, Any]],
    list[SimpleNamespace],
    list[dict[str, Any]],
    list[SimpleNamespace],
    list[dict[str, Any]],
    list[SimpleNamespace],
    list[SimpleNamespace],
]:
    regional_table: list[dict[str, Any]] = []
    regional_source: list[SimpleNamespace] = []
    for stage in CANONICAL_STAGE_ORDER:
        for cell in CANONICAL_CELL_ORDER:
            scanner_backend, workflow_mode = _CONDITION_AXES[cell]
            for region in ("interior", "boundary_shell"):
                metrics = {"candidate_nonzero_fraction": 1.0}
                regional_source.append(
                    SimpleNamespace(
                        stage=stage,
                        cell_label=cell,
                        region=region,
                        metrics=metrics,
                    )
                )
                regional_table.append(
                    {
                        "dataset": "f3",
                        "evaluation_semantics": F3_SEMANTICS,
                        "case_or_region": region,
                        "stage": stage,
                        "cell_label": cell,
                        "scanner_backend": scanner_backend,
                        "workflow_mode": workflow_mode,
                        "region": region,
                        "metric": "candidate_nonzero_fraction",
                        "display_label": "candidate nonzero fraction",
                        "value": 1.0,
                        "unit": "fraction",
                        "source_artifact": "reports/regional_metrics.csv",
                    }
                )

    orientation_source = [
        SimpleNamespace(
            stage="scanner",
            left_cell="RL-REF",
            right_cell="RL-QUAL",
            support_contract="minimal_positive_support",
            support_count=1,
            strike_circular_absolute_difference={"count": 1.0},
            dip_absolute_difference={"count": 1.0},
            normal_vector_angular_difference={"count": 1.0},
        )
    ]
    orientation_table = []
    for source in orientation_source:
        for prefix in (
            "strike_circular_absolute_difference",
            "dip_absolute_difference",
            "normal_vector_angular_difference",
        ):
            orientation_table.append(
                {
                    "dataset": "f3",
                    "evaluation_semantics": F3_SEMANTICS,
                    "case_or_region": "full",
                    "stage": source.stage,
                    "left_cell": source.left_cell,
                    "right_cell": source.right_cell,
                    "support_contract": source.support_contract,
                    "support_count": source.support_count,
                    "metric": f"{prefix}.count",
                    "display_label": f"{prefix.replace('_', ' ')} count",
                    "value": 1.0,
                    "unit": "count",
                    "source_artifact": "reports/orientation_diagnostics.csv",
                }
            )

    synthetic_runtime = [
        SimpleNamespace(
            case_id="minimal_case",
            trial_id="minimal_trial",
            seed=7,
            stage="case_generation",
            scanner_backend=None,
            call_count=1,
            cell_label=None,
            elapsed_seconds=1.0,
            shared_stage=True,
        )
    ]
    f3_runtime = [
        SimpleNamespace(
            stage_kind="scanner",
            fingerprint="c" * 64,
            cell="RL-REF",
            cell_consumers=("RL-REF",),
            state="computed",
            elapsed_seconds=1.0,
            elapsed_semantics="compute",
        )
    ]
    runtime_table = [
        {
            "dataset": "synthetic",
            "evaluation_semantics": SYNTHETIC_SEMANTICS,
            "case_or_region": "minimal_case",
            "trial_id": "minimal_trial",
            "seed": 7,
            "stage": "case_generation",
            "fingerprint": None,
            "scanner_backend": None,
            "call_count": 1,
            "cell_label": None,
            "cell_consumers": (),
            "state": "shared",
            "elapsed_seconds": 1.0,
            "elapsed_semantics": "within_experiment_attribution",
            "shared_stage": True,
            "attribution": "shared-stage",
            "source_artifact": "runtime.csv",
        },
        {
            "dataset": "f3",
            "evaluation_semantics": F3_SEMANTICS,
            "case_or_region": "full",
            "trial_id": None,
            "seed": None,
            "stage": "scanner",
            "fingerprint": "c" * 64,
            "scanner_backend": None,
            "call_count": 1,
            "cell_label": "RL-REF",
            "cell_consumers": ("RL-REF",),
            "state": "computed",
            "elapsed_seconds": 1.0,
            "elapsed_semantics": "compute",
            "shared_stage": False,
            "attribution": "cell-owned-stage",
            "source_artifact": "reports/runtime.csv",
        },
    ]
    return (
        regional_table,
        regional_source,
        orientation_table,
        orientation_source,
        runtime_table,
        synthetic_runtime,
        f3_runtime,
    )


def _contrast_tables() -> tuple[list[dict[str, Any]], list[SimpleNamespace]]:
    definition = next(
        item
        for item in synthetic_mode_comparison.CONTRAST_DEFINITIONS
        if item.name == "end_to_end_delta"
    )
    source = SimpleNamespace(
        contrast_name=definition.name,
        case_id="minimal_case",
        trial_id="minimal_trial",
        seed=7,
        stage="fvt",
        selection="positive_top_truth_count",
        metric="buffered_f1",
    )
    table = [
        {
            "dataset": "synthetic",
            "evaluation_semantics": SYNTHETIC_SEMANTICS,
            "case_or_region": source.case_id,
            "trial_id": source.trial_id,
            "seed": source.seed,
            "contrast_name": source.contrast_name,
            "stage": source.stage,
            "selection": source.selection,
            "metric": source.metric,
            "raw_value": 0.0,
            "improvement_value": 0.0,
            "unit": "fraction",
            "direction": "higher",
            "component_cells": definition.component_cells,
            "source_artifact": "contrasts.csv",
        }
    ]
    return table, [source]


def _figure_data_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return build_table_contract(
        FIGURE_DATA_HEADER,
        rows,
        FIGURE_DATA_IDENTITY_FIELDS,
        FIGURE_DATA_FIELD_TYPES,
    )


def _write_figure_record(
    root: Path,
    *,
    figure_id: str,
    dataset: str,
    category: str,
    figure_role: str,
    source_metric: str | None,
    source_stage: str | None,
    cell_labels: tuple[str, ...],
    panel_labels: tuple[str, ...],
    data_rows: list[dict[str, Any]],
    caption: str,
    contrast_name: str | None = None,
    axis: str | None = None,
    policy: str | None = None,
    selection_threshold: float | None = None,
    candidate_selection_thresholds: Mapping[str, float] | None = None,
    display_scale: Mapping[str, Any] | None = None,
    figure_role_override: str | None = None,
) -> dict[str, Any]:
    del figure_role_override
    path = root / "figures" / f"{figure_id}.png"
    write_png(path, width=1, height=1)
    data_path = root / "figure_data" / f"{figure_id}.csv"
    write_csv_rows(data_path, FIGURE_DATA_HEADER, data_rows)
    semantics = SYNTHETIC_SEMANTICS if dataset == "synthetic" else F3_SEMANTICS
    return _record(
        figure_id=figure_id,
        relative_path=f"figures/{figure_id}.png",
        dataset=dataset,
        category=category,
        figure_role=figure_role,
        evaluation_semantics=semantics,
        source_metric=source_metric,
        source_stage=source_stage,
        cell_labels=cell_labels,
        panel_labels=panel_labels,
        contrast_name=contrast_name,
        axis=axis,
        slice_index=0 if axis is not None else None,
        slice_selection_policy=policy,
        selection_percentile=F3_BUFFERED_PERCENTILE if axis is not None else None,
        buffer_radius=F3_BUFFER_RADIUS if axis is not None else None,
        selection_threshold=selection_threshold,
        candidate_selection_thresholds=candidate_selection_thresholds,
        display_scale=display_scale,
        figure_data_csv=f"figure_data/{figure_id}.csv",
        figure_data_contract=_figure_data_contract(data_rows),
        png_path=path,
        caption=caption,
    )


def _figure_row(
    figure_id: str,
    *,
    dataset: str,
    source_metric: str | None,
    source_stage: str | None,
    panel_label: str | None,
    cell_label: str | None = None,
    metric: str | None = None,
    value: float | None = 1.0,
    unit: str | None = None,
    direction: str | None = None,
    axis: str | None = None,
    policy: str | None = None,
    selection_threshold: float | None = None,
    candidate_selection_threshold: float | None = None,
    scale_policy: str | None = None,
    difference: bool = False,
    case_or_region: str | None = None,
    trial_id: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    semantics = SYNTHETIC_SEMANTICS if dataset == "synthetic" else F3_SEMANTICS
    return _data_row(
        figure_id,
        dataset=dataset,
        evaluation_semantics=semantics,
        source_metric=source_metric,
        source_stage=source_stage,
        case_or_region=case_or_region,
        trial_id=trial_id,
        seed=seed,
        cell_label=cell_label,
        panel_label=panel_label,
        metric=metric,
        value=value,
        unit=unit if unit is not None else ("fraction" if value is not None else None),
        direction=direction if direction is not None else ("higher" if value is not None else None),
        axis=axis,
        slice_index=0 if axis is not None else None,
        slice_selection_policy=policy,
        slice_score=1.0 if axis is not None else None,
        selection_threshold=selection_threshold,
        candidate_selection_threshold=candidate_selection_threshold,
        vmin=-1.0 if difference else (0.0 if axis is not None else None),
        vmax=1.0 if axis is not None else None,
        scale_policy=scale_policy,
        colormap="coolwarm" if difference else ("viridis" if axis is not None else None),
        difference_limit=1.0 if difference else None,
        difference_vmin=-1.0 if difference else None,
        difference_vmax=1.0 if difference else None,
    )


def _stdlib_figure_records(
    report: PublicationReport, root: Path
) -> tuple[tuple[dict[str, Any], ...], Mapping[str, str]]:
    """Create every fixed slot using 1x1 standard-library PNGs only."""

    (root / "figures").mkdir()
    (root / "figure_data").mkdir()
    records: list[dict[str, Any]] = []
    metric_rows = list(report.tables["publication_metrics.csv"])
    runtime_rows = list(report.tables["runtime_summary.csv"])

    def matching(**expected: str) -> list[dict[str, Any]]:
        return [
            row
            for row in metric_rows
            if all(row[name] == value for name, value in expected.items())
        ]

    def scalar_record(
        figure_id: str,
        *,
        dataset: str,
        category: str,
        role: str,
        source_metric: str | None,
        source_stage: str | None,
        cells: tuple[str, ...],
        panels: tuple[str, ...],
        data: list[dict[str, Any]],
        contrast_name: str | None = None,
        candidate_selection_thresholds: Mapping[str, float] | None = None,
        display_scale: Mapping[str, Any] | None = None,
    ) -> None:
        records.append(
            _write_figure_record(
                root,
                figure_id=figure_id,
                dataset=dataset,
                category=category,
                figure_role=role,
                source_metric=source_metric,
                source_stage=source_stage,
                cell_labels=cells,
                panel_labels=panels,
                data_rows=data,
                contrast_name=contrast_name,
                candidate_selection_thresholds=candidate_selection_thresholds,
                display_scale=display_scale,
                caption=f"minimal {figure_id} figure",
            )
        )

    fvt_metric_specs = {
        "synthetic_fvt_buffered_f1_by_case": ("buffered_f1", "fraction", "higher"),
        "synthetic_fvt_hausdorff_p95_by_case": ("hausdorff_p95", "voxel", "lower"),
    }
    for figure_id, (metric, unit, direction) in fvt_metric_specs.items():
        selected = matching(
            dataset="synthetic",
            stage="fvt",
            selection="positive_top_truth_count",
            metric=metric,
        )
        scalar_record(
            figure_id,
            dataset="synthetic",
            category="synthetic_scalar",
            role="main",
            source_metric=f"fvt/positive_top_truth_count/{metric}",
            source_stage="fvt",
            cells=CANONICAL_CELL_ORDER,
            panels=CANONICAL_CELL_ORDER,
            data=[
                _figure_row(
                    figure_id,
                    dataset="synthetic",
                    source_metric=f"fvt/positive_top_truth_count/{metric}",
                    source_stage="fvt",
                    panel_label=row["cell_label"],
                    cell_label=row["cell_label"],
                    metric=metric,
                    value=row["value"],
                    unit=unit,
                    direction=direction,
                    case_or_region="minimal_case",
                    trial_id="minimal_trial",
                    seed=7,
                )
                for row in selected
            ],
            display_scale={
                "scale_policy": "raw_metric_scale; IQR for stochastic cases",
                "vmin": None,
                "vmax": None,
                "colormap": "condition markers",
                "difference_scale": None,
            },
        )

    orientation_data = []
    for metric in ("strike_median", "dip_median"):
        for row in matching(
            dataset="synthetic",
            stage="scanner_raw",
            selection="top_truth_count",
            metric=metric,
        ):
            orientation_data.append(
                _figure_row(
                    "synthetic_scanner_orientation_error_by_case",
                    dataset="synthetic",
                    source_metric="scanner_raw/top_truth_count/strike_median,dip_median",
                    source_stage="scanner_raw",
                    panel_label=row["cell_label"],
                    cell_label=row["cell_label"],
                    metric=metric,
                    value=row["value"],
                    unit="degree",
                    direction="lower",
                    case_or_region="minimal_case",
                    trial_id="minimal_trial",
                    seed=7,
                )
            )
    scalar_record(
        "synthetic_scanner_orientation_error_by_case",
        dataset="synthetic",
        category="synthetic_scalar",
        role="supplementary",
        source_metric="scanner_raw/top_truth_count/strike_median,dip_median",
        source_stage="scanner_raw",
        cells=SYNTHETIC_SCANNER_CELL_ORDER,
        panels=SYNTHETIC_SCANNER_CELL_ORDER,
        data=orientation_data,
        display_scale={
            "scale_policy": "shared degree axis; IQR for stochastic cases",
            "vmin": None,
            "vmax": None,
            "colormap": "metric colors and scanner-backend markers",
            "difference_scale": None,
            "series_encoding": {
                "metric_color": {"strike_median": "#4c78a8", "dip_median": "#f58518"},
                "scanner_backend_marker": {"RL-SCAN": "o", "Q-SCAN": "s"},
            },
        },
    )

    heatmap_entries = [
        entry
        for entry in PUBLICATION_METRIC_REGISTRY
        if entry.dataset == "synthetic"
        and entry.stage in {"fvt", "skin"}
        and entry.direction != "neutral"
    ]
    scalar_record(
        "synthetic_end_to_end_improvement_heatmap",
        dataset="synthetic",
        category="synthetic_contrast",
        role="main",
        source_metric="selected directional publication metrics",
        source_stage="synthetic",
        cells=CANONICAL_CELL_ORDER,
        panels=tuple(entry.metric for entry in heatmap_entries),
        data=[
            _figure_row(
                "synthetic_end_to_end_improvement_heatmap",
                dataset="synthetic",
                source_metric=f"{entry.stage}/{entry.selection}/{entry.metric}",
                source_stage="synthetic",
                panel_label=entry.metric,
                metric=entry.metric,
                value=0.0,
                unit=entry.unit,
                direction=entry.direction,
                case_or_region="minimal_case",
            )
            for entry in heatmap_entries
        ],
        contrast_name="end_to_end_delta",
        display_scale={
            "vmin": -1.0,
            "vmax": 1.0,
            "scale_policy": "column-normalized",
            "colormap": "coolwarm",
            "difference_scale": "zero-centered divergent",
        },
    )

    omitted_skin = _record(
        figure_id="synthetic_skin_buffered_f1_by_case",
        relative_path="figures/synthetic_skin_buffered_f1_by_case.png",
        dataset="synthetic",
        category="synthetic_scalar",
        figure_role="main",
        evaluation_semantics=SYNTHETIC_SEMANTICS,
        source_metric="skin/skin_cells/buffered_f1",
        source_stage="skin",
        cell_labels=CANONICAL_CELL_ORDER,
        panel_labels=CANONICAL_CELL_ORDER,
        display_scale={
            "scale_policy": "raw_metric_scale; IQR for stochastic cases",
            "vmin": None,
            "vmax": None,
            "colormap": "condition markers",
            "difference_scale": None,
        },
        caption="minimal omitted skin figure",
        omitted=True,
        omission_reason=SYNTHETIC_SKIN_FIGURE_OMISSION_REASON,
    )
    records.append(omitted_skin)

    for dataset in ("synthetic", "f3"):
        rows = [row for row in runtime_rows if row["dataset"] == dataset]
        scalar_record(
            f"{dataset}_runtime_breakdown",
            dataset=dataset,
            category="runtime",
            role="main",
            source_metric="runtime stage name",
            source_stage="runtime",
            cells=tuple(row["cell_label"] for row in rows if row["cell_label"] is not None),
            panels=tuple(_runtime_panel_label(row) for row in rows),
            data=[
                _figure_row(
                    f"{dataset}_runtime_breakdown",
                    dataset=dataset,
                    source_metric=row["stage"],
                    source_stage="runtime",
                    panel_label=_runtime_panel_label(row),
                    cell_label=row["cell_label"],
                    metric="elapsed_seconds",
                    value=row["elapsed_seconds"],
                    unit="second",
                    case_or_region=row["case_or_region"],
                    trial_id=row["trial_id"],
                    seed=row["seed"],
                )
                for row in rows
            ],
            display_scale={
                "scale_policy": "stage-level within-experiment attribution",
                "vmin": 0.0,
                "vmax": None,
                "colormap": "shared versus cell-owned bars",
                "difference_scale": None,
            },
        )

    f3_specs = {
        "f3_normalized_correlation_by_stage": (("all", "normalized_correlation"),),
        "f3_buffered_f1_by_stage": (("positive_p99_radius2", "buffered_f1"),),
        "f3_nonzero_fraction_ratio_by_stage": (("all", "nonzero_fraction_ratio"),),
        "f3_sparse_distance_p95_by_stage": (
            ("positive_p99_distance", "candidate_to_reference_p95"),
            ("positive_p99_distance", "reference_to_candidate_p95"),
        ),
    }
    for figure_id, specs in f3_specs.items():
        data = []
        source_metrics = []
        for stage in CANONICAL_STAGE_ORDER:
            for selection, metric in specs:
                source_metric = f"{stage}/{selection}/{metric}"
                source_metrics.append(source_metric)
                for row in matching(dataset="f3", stage=stage, selection=selection, metric=metric):
                    data.append(
                        _figure_row(
                            figure_id,
                            dataset="f3",
                            source_metric=source_metric,
                            source_stage=stage,
                            panel_label=row["cell_label"],
                            cell_label=row["cell_label"],
                            metric=metric,
                            value=row["value"],
                            unit=row["unit"],
                            direction=row["direction"],
                            case_or_region="full",
                        )
                    )
        sparse = figure_id == "f3_sparse_distance_p95_by_stage"
        scalar_record(
            figure_id,
            dataset="f3",
            category="f3_scalar",
            role="main",
            source_metric=",".join(source_metrics),
            source_stage="ft,fv,fvt",
            cells=CANONICAL_CELL_ORDER,
            panels=CANONICAL_CELL_ORDER,
            data=data,
            display_scale={
                "scale_policy": "raw scalar metric; one full-volume evaluation unit",
                "vmin": None,
                "vmax": None,
                "colormap": "condition-colored bars",
                "difference_scale": None,
                "series_encoding": (
                    {
                        "cell_condition_color": {
                            cell: color
                            for cell, color in zip(
                                CANONICAL_CELL_ORDER,
                                ("#4c78a8", "#f58518", "#54a24b", "#e45756"),
                                strict=True,
                            )
                        },
                        "distance_direction_hatch": {
                            "candidate_to_reference_p95": "///",
                            "reference_to_candidate_p95": "\\\\",
                        },
                    }
                    if sparse
                    else None
                ),
            },
        )

    for stage, policy, axis in F3_SPATIAL_FIGURE_SLOTS:
        reference_threshold = _F3_REFERENCE_SELECTION_THRESHOLDS[stage]
        if stage == "ft":
            panels = (
                "PUBLIC-REF fl.dat",
                "reference-like scanner ft",
                "quality scanner ft",
                "quality - reference-like signed difference",
            )
            cells = ("RL-REF", "Q-REF")
            source_metric = "ft/all/output difference"
        else:
            panels = (
                "PUBLIC-REF",
                *CANONICAL_CELL_ORDER,
                "Q-QUAL - RL-REF signed difference",
            )
            cells = CANONICAL_CELL_ORDER
            source_metric = f"{stage}/all/output difference"
        figure_id = f"f3_{stage}_comparison_{policy}_{axis}_0"
        scalar_record(
            figure_id,
            dataset="f3",
            category="f3_spatial",
            role="main",
            source_metric=source_metric,
            source_stage=stage,
            cells=cells,
            panels=panels,
            data=[
                _figure_row(
                    figure_id,
                    dataset="f3",
                    source_metric=source_metric,
                    source_stage=stage,
                    panel_label=panel,
                    axis=axis,
                    policy=policy,
                    selection_threshold=reference_threshold,
                    scale_policy=(
                        "symmetric_zero_centered_difference"
                        if "signed difference" in panel
                        else "shared_stage_slice_scale"
                    ),
                    difference="signed difference" in panel,
                )
                for panel in panels
            ],
            contrast_name="end_to_end_delta" if stage in {"fv", "fvt"} else None,
            display_scale={
                "vmin": 0.0,
                "vmax": 1.0,
                "scale_policy": "validated_full_volume_min_max; shared across normal panels",
                "colormap": "viridis",
                "difference_limit": 1.0,
                "difference_vmin": -1.0,
                "difference_vmax": 1.0,
                "difference_scale": "symmetric around zero",
            },
        )
        records[-1]["axis"] = axis
        records[-1]["slice_index"] = 0
        records[-1]["slice_selection_policy"] = policy
        records[-1]["selection_percentile"] = F3_BUFFERED_PERCENTILE
        records[-1]["buffer_radius"] = F3_BUFFER_RADIUS
        records[-1]["selection_threshold"] = reference_threshold

    for policy, axis in F3_RIDGE_OVERLAY_SLOTS:
        figure_id = f"f3_fvt_ridge_overlay_{policy}_{axis}_0"
        reference_threshold = _F3_REFERENCE_SELECTION_THRESHOLDS["fvt"]
        candidate_thresholds = _F3_CANDIDATE_SELECTION_THRESHOLDS["fvt"]
        panels = tuple(f"PUBLIC-REF vs {cell}" for cell in CANONICAL_CELL_ORDER)
        scalar_record(
            figure_id,
            dataset="f3",
            category="f3_ridge_overlay",
            role="supplementary",
            source_metric="positive_p99_radius2/buffered_f1",
            source_stage="fvt",
            cells=CANONICAL_CELL_ORDER,
            panels=panels,
            data=[
                _figure_row(
                    figure_id,
                    dataset="f3",
                    source_metric="positive_p99_radius2/buffered_f1",
                    source_stage="fvt",
                    panel_label=panel,
                    cell_label=cell,
                    metric="exact_overlap_count",
                    value=1.0,
                    unit="count",
                    axis=axis,
                    policy=policy,
                    selection_threshold=reference_threshold,
                    candidate_selection_threshold=candidate_thresholds[cell],
                )
                for cell, panel in zip(CANONICAL_CELL_ORDER, panels, strict=True)
            ],
            display_scale={
                "vmin": 0.0,
                "vmax": 1.0,
                "scale_policy": "categorical ridge overlay",
                "colormap": "categorical",
                "difference_scale": None,
            },
            candidate_selection_thresholds=candidate_thresholds,
        )
        records[-1]["axis"] = axis
        records[-1]["slice_index"] = 0
        records[-1]["slice_selection_policy"] = policy
        records[-1]["selection_percentile"] = F3_BUFFERED_PERCENTILE
        records[-1]["buffer_radius"] = F3_BUFFER_RADIUS
        records[-1]["selection_threshold"] = reference_threshold

    assert {record["figure_id"] for record in records} == set(FIXED_SCALAR_FIGURE_IDS) | {
        f"f3_{stage}_comparison_{policy}_{axis}_0"
        for stage, policy, axis in F3_SPATIAL_FIGURE_SLOTS
    } | {f"f3_fvt_ridge_overlay_{policy}_{axis}_0" for policy, axis in F3_RIDGE_OVERLAY_SLOTS}
    return tuple(sorted(records, key=lambda record: record["figure_id"])), {
        "matplotlib": "stdlib-test-fixture",
        "backend": "none",
    }


def _minimal_report(root: Path) -> PublicationReport:
    metric_table, synthetic_metrics, f3_metrics = _metric_tables()
    contrast_table, synthetic_contrasts = _contrast_tables()
    (
        regional_table,
        regional_source,
        orientation_table,
        orientation_source,
        runtime_table,
        synthetic_runtime,
        f3_runtime,
    ) = _supporting_tables()
    tables: Mapping[str, tuple[Mapping[str, Any], ...]] = {
        "publication_metrics.csv": tuple(metric_table),
        "publication_contrasts.csv": tuple(contrast_table),
        "publication_summary.csv": tuple(_summary_table(metric_table)),
        "f3_regional_summary.csv": tuple(regional_table),
        "f3_orientation_summary.csv": tuple(orientation_table),
        "runtime_summary.csv": tuple(runtime_table),
    }
    assert set(tables) == set(ROOT_TABLE_FILES)
    assert all(tuple(TABLE_HEADERS[name]) == tuple(ROOT_TABLE_FIELD_TYPES[name]) for name in tables)

    synthetic_manifest = {
        "artifact_schema_version": 1,
        "metric_schema_version": 1,
        "scalar_evidence_contract_version": 1,
        "runtime_contract_version": 1,
        "source_provenance": {"fixture": "stdlib"},
        "trials": [
            {
                "case_id": "minimal_case",
                "trial_id": "minimal_trial",
                "case_generation_seed": 7,
            }
        ],
    }
    synthetic_completion = "a" * 64
    synthetic_manifest_sha = "b" * 64
    synthetic_identity = canonical_digest(
        {
            "artifact_schema_version": synthetic_manifest["artifact_schema_version"],
            "metric_schema_version": synthetic_manifest["metric_schema_version"],
            "scalar_evidence_contract_version": synthetic_manifest[
                "scalar_evidence_contract_version"
            ],
            "runtime_contract_version": synthetic_manifest["runtime_contract_version"],
            "manifest_sha256": synthetic_manifest_sha,
            "completion_sha256": synthetic_completion,
        }
    )
    synthetic = SyntheticSourceBundle(
        root / "synthetic-source",
        synthetic_manifest,
        synthetic_completion,
        synthetic_manifest_sha,
        synthetic_identity,
        tuple(synthetic_metrics),
        tuple(synthetic_contrasts),
        tuple(synthetic_runtime),
        False,
        ("minimal_case",),
    )

    dataset_identity = {"dataset_id": "minimal-f3", "files": []}
    f3_manifest = {
        "artifact_schema_version": 1,
        "stage_contract_version": 1,
        "fingerprint_contract_version": 1,
        "run_fingerprint": "c" * 64,
        "implementation_identity": {"fixture": "stdlib"},
        "provenance": {"source": "stdlib-fixture"},
    }
    f3_completion = "d" * 64
    f3_manifest_sha = "e" * 64
    f3_identity = canonical_digest(
        {
            "artifact_schema_version": f3_manifest["artifact_schema_version"],
            "result_schema_version": 1,
            "run_fingerprint": f3_manifest["run_fingerprint"],
            "dataset_identity": dataset_identity,
            "manifest_sha256": f3_manifest_sha,
            "completion_sha256": f3_completion,
        }
    )
    f3_result = SimpleNamespace(
        metric_rows=tuple(f3_metrics),
        contrast_rows=(),
        regional_rows=tuple(regional_source),
        orientation_rows=tuple(orientation_source),
        runtime_rows=tuple(f3_runtime),
        cells=tuple(
            SimpleNamespace(label=cell, skinning_enabled=False) for cell in CANONICAL_CELL_ORDER
        ),
        volume_shape=(1, 1, 1),
        storage_dtype=">f4",
        run_fingerprint=f3_manifest["run_fingerprint"],
    )
    f3 = F3SourceBundle(
        root / "f3-source",
        root / "f3-data-root",
        _MinimalDatasetSpec(),
        f3_manifest,
        f3_completion,
        f3_manifest_sha,
        f3_identity,
        f3_result,
        _f3_ridge_metric_evidence(),
        dataset_identity,
        1,
    )
    # Keep the fixed identity field lists in the fixture's source objects so
    # source identity revalidation exercises the same path-free field set.
    assert set(SYNTHETIC_SOURCE_IDENTITY_FIELDS) <= {
        "artifact_schema_version",
        "metric_schema_version",
        "scalar_evidence_contract_version",
        "runtime_contract_version",
        "manifest_sha256",
        "completion_sha256",
    }
    assert set(F3_SOURCE_IDENTITY_FIELDS) <= {
        "artifact_schema_version",
        "result_schema_version",
        "run_fingerprint",
        "dataset_identity",
        "manifest_sha256",
        "completion_sha256",
    }
    return PublicationReport(synthetic, f3, tables, {})


def _build_minimal_v3_bundle(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use the production writer with a stdlib-only fixed-slot renderer."""

    report = _minimal_report(root)
    monkeypatch.setattr(artifacts, "generate_figures", _stdlib_figure_records)
    return artifacts.write_publication_bundle(report, root / "publication")


def _build_minimal_v1_bundle(root: Path) -> Path:
    bundle = root / "publication-v1"
    bundle.mkdir()
    (bundle / "experiment.json").write_text("{}\n", encoding="utf-8")
    (bundle / "uv.lock").write_text("lock-version = 1\n", encoding="utf-8")
    experiment_record = artifact_file_record(
        bundle,
        "experiment.json",
        tier="primary",
        role="resolved_experiment",
    )
    lock_record = artifact_file_record(
        bundle,
        "uv.lock",
        tier="primary",
        role="environment_lock",
    )
    manifest = build_publication_manifest(
        created_at_utc="2026-08-12T00:00:00Z",
        code={"repository": "ozw4/pyosv", "git_commit": "a" * 40, "dirty": False},
        environment={
            "python": "3.10.14",
            "lock_file": "uv.lock",
            "lock_sha256": lock_record["sha256"],
            "controls": {
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "NUMBA_DISABLE_JIT": "0",
                "NUMBA_NUM_THREADS": "1",
                "PYOSV_ACCEL": "auto",
            },
        },
        datasets={
            "f3": {
                "dataset_id": "minimal-f3",
                "shape": [1, 1, 1],
                "dtype": ">f4",
                "files": [
                    {
                        "role": "input",
                        "filename": "input.dat",
                        "size": 4,
                        "sha256": "b" * 64,
                    }
                ],
            }
        },
        experiment={
            "config_file": "experiment.json",
            "config_sha256": experiment_record["sha256"],
            "source_runs": {
                "synthetic": {"completion_sha256": "c" * 64},
                "f3": {"completion_sha256": "d" * 64},
            },
        },
        semantics={
            "synthetic": "known_truth",
            "f3": "public_reference_agreement",
            "f3_public_reference_is_geological_truth": False,
            "f3_evaluation_units": 1,
        },
        artifacts=[experiment_record, lock_record],
    )
    write_publication_manifest(bundle, manifest)
    assert validate_publication_directory(bundle) == manifest
    return bundle


def test_validate_only_cli_does_not_import_matplotlib_or_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _build_minimal_v3_bundle(tmp_path, monkeypatch)
    assert validate_publication_bundle(bundle)
    source_root = Path(__file__).parents[3] / "src"
    script = """
import builtins
import sys

original_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise AssertionError('validate-only imported matplotlib')
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from pyosv.evaluation import f3d_mode_comparison, synthetic_mode_comparison
from pyosv.evaluation.mode_comparison_publication import loaders
from pyosv.evaluation import workflow3d
import pyosv.evaluation.mode_comparison_publication as publication

def forbidden(*args, **kwargs):
    raise AssertionError('validate-only accessed a source runner or data root')

# Importing source contracts is permitted; calling a runner/loader is not.
synthetic_mode_comparison.run_mode_comparison = forbidden
synthetic_mode_comparison.run_synthetic_trial = forbidden
f3d_mode_comparison.run_scanner_stages = forbidden
f3d_mode_comparison.run_f3d_mode_comparison = forbidden
workflow3d.execute_workflow3d = forbidden
loaders.load_synthetic_source = forbidden
loaders.load_f3_source = forbidden
publication.generate_publication_bundle = forbidden
from pyosv.cli.mode_comparison_publication import main

# No source arguments are present.  The valid stdlib-only fixture must succeed
# through validate-only without triggering either guard.
assert main([
    '--publication-contract', 'legacy',
    '--validate-only',
    '--output-dir', sys.argv[1],
]) == 0
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [sys.executable, "-c", script, str(bundle)],
        cwd=source_root.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_v1_validate_only_cli_imports_no_generation_or_runtime_stack(tmp_path: Path) -> None:
    bundle = _build_minimal_v1_bundle(tmp_path)
    source_root = Path(__file__).parents[3] / "src"
    script = """
import builtins
import os
import subprocess
import sys

original_import = builtins.__import__
forbidden_prefixes = (
    'matplotlib',
    'numba',
    'threadpoolctl',
    'pyosv.evaluation.mode_comparison_publication',
    'pyosv.evaluation.synthetic_mode_comparison',
    'pyosv.evaluation.f3d_mode_comparison',
    'pyosv.evaluation.workflow3d',
)

def guarded_import(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden_prefixes):
        raise AssertionError('v1 validate-only imported forbidden module: ' + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from pyosv.cli.mode_comparison_publication import main

def forbidden_git(*args, **kwargs):
    raise AssertionError('v1 validate-only executed an external command')

original_environment = os.environ
control_names = {
    'PYTHONHASHSEED',
    'OMP_NUM_THREADS',
    'OPENBLAS_NUM_THREADS',
    'MKL_NUM_THREADS',
    'NUMEXPR_NUM_THREADS',
    'NUMBA_DISABLE_JIT',
    'NUMBA_NUM_THREADS',
    'PYOSV_ACCEL',
}

class ForbiddenEnvironment:
    def get(self, key, *args, **kwargs):
        if key in control_names:
            raise AssertionError('v1 validate-only read publication controls')
        return original_environment.get(key, *args, **kwargs)

    def __getitem__(self, key):
        if key in control_names:
            raise AssertionError('v1 validate-only read publication controls')
        return original_environment[key]

subprocess.run = forbidden_git
os.environ = ForbiddenEnvironment()
assert main([
    '--validate-only',
    '--output-dir', sys.argv[1],
]) == 0
assert not any(
    name == prefix or name.startswith(prefix + '.')
    for name in sys.modules
    for prefix in forbidden_prefixes
)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)
    result = subprocess.run(
        [sys.executable, "-c", script, str(bundle)],
        cwd=source_root.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
