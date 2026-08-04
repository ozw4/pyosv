"""Publication figures with deterministic, contract-recorded display scales.

Matplotlib is intentionally imported only inside :func:`generate_figures`.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import binary_dilation

from pyosv.candidate_volume import NONZERO_EPSILON, positive_candidate_mask
from ..f3d_mode_comparison import (
    F3_BUFFER_RADIUS,
    F3_BUFFERED_PERCENTILE,
    F3RunWorkspace,
    F3VolumeSource,
    scanner_stage_artifacts,
    thinning_stage_artifacts,
    voting_stage_artifacts,
)
from ..f3d_mode_comparison.metrics import F3_REFERENCE_STAGE_ROLES

from .config import (
    CANONICAL_CELL_ORDER,
    CANONICAL_STAGE_ORDER,
    FIGURE_DATA_HEADER,
    F3_SEMANTICS,
    SYNTHETIC_SCANNER_CELL_ORDER,
    SYNTHETIC_SEMANTICS,
)
from .models import PublicationReport
from .registry import PUBLICATION_METRIC_REGISTRY


def _finite(value: Any) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("figure metadata values must be finite")
    return result


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(_finite(value))
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return str(value)


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(FIGURE_DATA_HEADER)
        for row in rows:
            writer.writerow(_csv_value(row.get(name)) for name in FIGURE_DATA_HEADER)


def _data_row(figure_id: str, **values: Any) -> dict[str, Any]:
    row = {name: None for name in FIGURE_DATA_HEADER}
    row["figure_id"] = figure_id
    row.update(values)
    return row


def _record(
    *,
    figure_id: str,
    relative_path: str,
    dataset: str,
    category: str,
    figure_role: str,
    evaluation_semantics: str,
    source_metric: str | None,
    source_stage: str | None,
    cell_labels: tuple[str, ...] = (),
    panel_labels: tuple[str, ...] = (),
    contrast_name: str | None = None,
    axis: str | None = None,
    slice_index: int | None = None,
    slice_selection_policy: str | None = None,
    selection_percentile: float | None = None,
    buffer_radius: float | None = None,
    selection_threshold: float | None = None,
    display_scale: Mapping[str, Any] | None = None,
    figure_data_csv: str | None = None,
    caption: str,
    omitted: bool = False,
    omission_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "relative_path": relative_path,
        "dataset": dataset,
        "category": category,
        "figure_role": figure_role,
        "evaluation_semantics": evaluation_semantics,
        "source_metric": source_metric,
        "source_stage": source_stage,
        "cell_labels": list(cell_labels),
        "panel_labels": list(panel_labels),
        "contrast_name": contrast_name,
        "axis": axis,
        "slice_index": slice_index,
        "slice_selection_policy": slice_selection_policy,
        "selection_percentile": selection_percentile,
        "buffer_radius": buffer_radius,
        "selection_threshold": selection_threshold,
        "display_scale": None if display_scale is None else dict(display_scale),
        "figure_data_csv": figure_data_csv,
        "caption": caption,
        "omitted": omitted,
        "omission_reason": omission_reason,
    }


def _matplotlib() -> tuple[Any, str, str]:
    from pyosv.viz import require_matplotlib

    plt = require_matplotlib()
    import matplotlib

    return plt, str(matplotlib.__version__), str(matplotlib.get_backend())


def _write_figure_data(root: Path, figure_id: str, rows: list[Mapping[str, Any]]) -> str:
    filename = f"{figure_id}.csv"
    path = root / "figure_data" / filename
    _write_csv(path, rows)
    return f"figure_data/{filename}"


def _save_scalar_plot(
    path: Path,
    *,
    title: str,
    ylabel: str,
    grouped_values: Mapping[str, Mapping[str, tuple[float, ...]]],
    group_order: tuple[str, ...],
    series_labels: tuple[str, ...] = ("value",),
    plt: Any,
) -> None:
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    markers = ("o", "s", "^", "D")
    fig, ax = plt.subplots(figsize=(max(8.0, 2.1 * len(group_order)), 4.8), constrained_layout=True)
    try:
        x_positions: list[float] = []
        x_labels: list[str] = []
        for group_index, group in enumerate(group_order):
            base = group_index * (len(CANONICAL_CELL_ORDER) + 1)
            for cell_index, cell in enumerate(CANONICAL_CELL_ORDER):
                x = float(base + cell_index)
                x_positions.append(x)
                x_labels.append(f"{group}\n{cell}")
                values = grouped_values.get(group, {}).get(cell, ())
                if not values:
                    continue
                median = float(np.median(values))
                if len(values) == 1:
                    low = high = median
                else:
                    low = float(np.quantile(values, 0.25, method="linear"))
                    high = float(np.quantile(values, 0.75, method="linear"))
                for series_index, label in enumerate(series_labels):
                    # A multi-series plot stores tuples under a synthetic key.
                    if len(series_labels) > 1:
                        continue
                    ax.errorbar(
                        x,
                        median,
                        yerr=[[median - low], [high - median]],
                        fmt=markers[cell_index],
                        color=colors[cell_index],
                        capsize=3,
                        label=cell if group_index == 0 else None,
                    )
            if group_index + 1 < len(group_order):
                ax.axvline(base + len(CANONICAL_CELL_ORDER) - 0.5, color="#bbbbbb", lw=0.7)
        ax.set_xticks(x_positions, x_labels, rotation=0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if x_positions:
            ax.legend(ncol=4, fontsize="small")
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)


def _summary_metric_groups(
    report: PublicationReport,
    *,
    dataset: str,
    stage: str,
    selection: str,
    metric: str,
) -> tuple[tuple[str, ...], dict[str, dict[str, tuple[float, ...]]], list[Mapping[str, Any]]]:
    rows = [
        row
        for row in report.tables["publication_metrics.csv"]
        if row["dataset"] == dataset
        and row["stage"] == stage
        and row["selection"] == selection
        and row["metric"] == metric
        and row["value"] is not None
    ]
    groups: dict[str, dict[str, list[float]]] = defaultdict(dict)
    for row in rows:
        groups.setdefault(row["case_or_region"], {}).setdefault(row["cell_label"], []).append(
            float(row["value"])
        )
    group_order = report.synthetic.case_order if dataset == "synthetic" else CANONICAL_STAGE_ORDER
    normalized = {
        group: {cell: tuple(values) for cell, values in cells.items()}
        for group, cells in groups.items()
    }
    return tuple(group_order), normalized, rows


def _synthetic_scalar_figure(
    report: PublicationReport,
    root: Path,
    *,
    figure_id: str,
    filename: str,
    stage: str,
    selection: str,
    metric: str,
    ylabel: str,
    title: str,
    caption: str,
    plt: Any,
) -> dict[str, Any]:
    groups, values, source_rows = _summary_metric_groups(
        report,
        dataset="synthetic",
        stage=stage,
        selection=selection,
        metric=metric,
    )
    data_rows = [
        _data_row(
            figure_id,
            dataset="synthetic",
            evaluation_semantics=SYNTHETIC_SEMANTICS,
            source_metric=f"{stage}/{selection}/{metric}",
            source_stage=stage,
            case_or_region=row["case_or_region"],
            trial_id=row["trial_id"],
            seed=row["seed"],
            cell_label=row["cell_label"],
            metric=metric,
            value=row["value"],
            unit=row["unit"],
            direction=row["direction"],
        )
        for row in source_rows
    ]
    data_path = _write_figure_data(root, figure_id, data_rows)
    path = root / "figures" / filename
    _save_scalar_plot(
        path,
        title=title,
        ylabel=ylabel,
        grouped_values=values,
        group_order=groups,
        plt=plt,
    )
    return _record(
        figure_id=figure_id,
        relative_path=f"figures/{filename}",
        dataset="synthetic",
        category="synthetic_scalar",
        figure_role="main",
        evaluation_semantics=SYNTHETIC_SEMANTICS,
        source_metric=f"{stage}/{selection}/{metric}",
        source_stage=stage,
        cell_labels=CANONICAL_CELL_ORDER,
        panel_labels=CANONICAL_CELL_ORDER,
        display_scale={
            "scale_policy": "raw_metric_scale; IQR for stochastic cases",
            "vmin": None,
            "vmax": None,
            "colormap": "condition markers",
            "difference_scale": None,
        },
        figure_data_csv=data_path,
        caption=caption,
    )


def _synthetic_orientation_figure(
    report: PublicationReport, root: Path, plt: Any
) -> dict[str, Any]:
    figure_id = "synthetic_scanner_orientation_error_by_case"
    metrics = ("strike_median", "dip_median")
    rows = [
        row
        for row in report.tables["publication_metrics.csv"]
        if row["dataset"] == "synthetic"
        and row["stage"] == "scanner_raw"
        and row["selection"] == "top_truth_count"
        and row["metric"] in metrics
    ]
    groups = report.synthetic.case_order
    fig, ax = plt.subplots(figsize=(max(8.0, 2.1 * len(groups)), 4.8), constrained_layout=True)
    data_rows = []
    try:
        for metric_index, metric in enumerate(metrics):
            color = ("#4c78a8", "#f58518")[metric_index]
            for group_index, group in enumerate(groups):
                for cell_index, cell in enumerate(SYNTHETIC_SCANNER_CELL_ORDER):
                    values = [
                        float(row["value"])
                        for row in rows
                        if row["metric"] == metric
                        and row["case_or_region"] == group
                        and row["cell_label"] == cell
                    ]
                    if not values:
                        continue
                    median = float(np.median(values))
                    low = median if len(values) == 1 else float(np.quantile(values, 0.25))
                    high = median if len(values) == 1 else float(np.quantile(values, 0.75))
                    x = group_index * 3 + cell_index + (metric_index - 0.5) * 0.13
                    ax.errorbar(
                        x,
                        median,
                        yerr=[[median - low], [high - median]],
                        fmt="o",
                        color=color,
                        capsize=3,
                        label=metric.replace("_", " ")
                        if group_index == 0 and cell_index == 0
                        else None,
                    )

        ticks = [index * 3 + 0.5 for index in range(len(groups))]
        ax.set_xticks(ticks, groups)
        ax.set_ylabel("orientation error (degree)")
        ax.set_title("Synthetic scanner orientation error by case")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        for row in rows:
            data_rows.append(
                _data_row(
                    figure_id,
                    dataset="synthetic",
                    evaluation_semantics=SYNTHETIC_SEMANTICS,
                    source_metric=f"scanner_raw/top_truth_count/{row['metric']}",
                    source_stage="scanner_raw",
                    case_or_region=row["case_or_region"],
                    trial_id=row["trial_id"],
                    seed=row["seed"],
                    cell_label=row["cell_label"],
                    metric=row["metric"],
                    value=row["value"],
                    unit=row["unit"],
                    direction=row["direction"],
                )
            )
        fig.savefig(root / "figures" / f"{figure_id}.png", dpi=150)
    finally:
        plt.close(fig)
    data_path = _write_figure_data(root, figure_id, data_rows)
    return _record(
        figure_id=figure_id,
        relative_path=f"figures/{figure_id}.png",
        dataset="synthetic",
        category="synthetic_scalar",
        figure_role="supplementary",
        evaluation_semantics=SYNTHETIC_SEMANTICS,
        source_metric="scanner_raw/top_truth_count/strike_median,dip_median",
        source_stage="scanner_raw",
        cell_labels=SYNTHETIC_SCANNER_CELL_ORDER,
        panel_labels=SYNTHETIC_SCANNER_CELL_ORDER,
        display_scale={
            "scale_policy": "shared degree axis; IQR for stochastic cases",
            "vmin": None,
            "vmax": None,
            "colormap": "condition markers",
            "difference_scale": None,
        },
        figure_data_csv=data_path,
        caption=(
            "Synthetic known-truth scanner strike and dip error from the two scanner-only "
            "cells; stochastic cases show median and IQR without a significance test."
        ),
    )


def _synthetic_heatmap(report: PublicationReport, root: Path, plt: Any) -> dict[str, Any]:
    figure_id = "synthetic_end_to_end_improvement_heatmap"
    entries = [
        entry
        for entry in PUBLICATION_METRIC_REGISTRY
        if (
            entry.dataset == "synthetic"
            and entry.stage in {"fvt", "skin"}
            and entry.direction != "neutral"
        )
    ]
    contrast_rows = report.tables["publication_contrasts.csv"]
    data_rows: list[Mapping[str, Any]] = []
    values = np.full((len(report.synthetic.case_order), len(entries)), np.nan, dtype=np.float64)
    for column, entry in enumerate(entries):
        by_case: dict[str, list[float]] = defaultdict(list)
        for row in contrast_rows:
            if (
                row["dataset"] == "synthetic"
                and row["contrast_name"] == "end_to_end_delta"
                and (row["stage"], row["selection"], row["metric"])
                == (entry.stage, entry.selection, entry.metric)
                and row["improvement_value"] is not None
            ):
                by_case[row["case_or_region"]].append(float(row["improvement_value"]))
        raw_by_case = {
            case: (float(np.median(item)) if item else None) for case, item in by_case.items()
        }
        maximum = max(
            (abs(value) for value in raw_by_case.values() if value is not None), default=0.0
        )
        for row_index, case in enumerate(report.synthetic.case_order):
            raw = raw_by_case.get(case)
            normalized = None if raw is None else (0.0 if maximum == 0.0 else raw / maximum)
            if normalized is not None:
                values[row_index, column] = normalized
            data_rows.append(
                _data_row(
                    figure_id,
                    dataset="synthetic",
                    evaluation_semantics=SYNTHETIC_SEMANTICS,
                    source_metric=f"{entry.stage}/{entry.selection}/{entry.metric}",
                    source_stage="synthetic",
                    case_or_region=case,
                    metric=entry.metric,
                    raw_improvement=raw,
                    normalized_value=normalized,
                    unit=entry.unit,
                    direction=entry.direction,
                )
            )
    data_path = _write_figure_data(root, figure_id, data_rows)
    fig, ax = plt.subplots(
        figsize=(max(9.0, 1.4 * len(entries)), max(3.5, 0.8 * len(report.synthetic.case_order))),
        constrained_layout=True,
    )
    try:
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad("#d0d0d0")
        image = ax.imshow(
            np.ma.masked_invalid(values), cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto"
        )
        ax.set_xticks(
            range(len(entries)), [entry.metric for entry in entries], rotation=60, ha="right"
        )
        ax.set_yticks(range(len(report.synthetic.case_order)), report.synthetic.case_order)
        ax.set_title("Synthetic end-to-end improvement (column-normalized)")
        ax.set_xlabel("directional publication metric")
        ax.set_ylabel("synthetic case")
        fig.colorbar(image, ax=ax, label="column-normalized improvement")
        fig.savefig(root / "figures" / f"{figure_id}.png", dpi=150)
    finally:
        plt.close(fig)
    return _record(
        figure_id=figure_id,
        relative_path=f"figures/{figure_id}.png",
        dataset="synthetic",
        category="synthetic_contrast",
        figure_role="main",
        evaluation_semantics=SYNTHETIC_SEMANTICS,
        source_metric="selected directional publication metrics",
        source_stage="synthetic",
        cell_labels=CANONICAL_CELL_ORDER,
        panel_labels=tuple(entry.metric for entry in entries),
        contrast_name="end_to_end_delta",
        display_scale={
            "vmin": -1.0,
            "vmax": 1.0,
            "scale_policy": "column-normalized",
            "colormap": "coolwarm",
            "difference_scale": "zero-centered divergent",
        },
        figure_data_csv=data_path,
        caption=(
            "End-to-end delta heatmap; column-normalized. Positive means direction-aware "
            "improvement. This is not a significance test."
        ),
    )


def _f3_metric_rows(
    report: PublicationReport, stage: str, selection: str, metric: str
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in report.tables["publication_metrics.csv"]
        if row["dataset"] == "f3"
        and row["stage"] == stage
        and row["selection"] == selection
        and row["metric"] == metric
    ]


def _f3_scalar_figure(
    report: PublicationReport,
    root: Path,
    *,
    figure_id: str,
    metric_specs: tuple[tuple[str, str, str], ...],
    ylabel: str,
    title: str,
    caption: str,
    plt: Any,
) -> dict[str, Any]:
    from matplotlib.patches import Patch

    metrics_by_stage: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for stage, selection, metric in metric_specs:
        if stage not in CANONICAL_STAGE_ORDER:
            raise ValueError(f"F3 scalar figure has an unknown stage: {stage!r}")
        metrics_by_stage[stage].append((selection, metric))
    if set(metrics_by_stage) != set(CANONICAL_STAGE_ORDER):
        raise ValueError("F3 scalar figure must cover ft, fv, and fvt")

    data_rows: list[Mapping[str, Any]] = []
    colors = ("#4c78a8", "#f58518", "#54a24b", "#e45756")
    stage_stride = len(CANONICAL_CELL_ORDER) + 1.0
    fig, ax = plt.subplots(figsize=(10.5, 5.2), constrained_layout=True)
    try:
        for stage_index, stage in enumerate(CANONICAL_STAGE_ORDER):
            stage_metrics = metrics_by_stage[stage]
            width = 0.78 / len(stage_metrics)
            base = stage_index * stage_stride
            for metric_index, (selection, metric) in enumerate(stage_metrics):
                rows = _f3_metric_rows(report, stage, selection, metric)
                lookup = {row["cell_label"]: row for row in rows}
                offset = (metric_index - (len(stage_metrics) - 1) / 2) * width
                for cell_index, cell in enumerate(CANONICAL_CELL_ORDER):
                    row = lookup.get(cell)
                    value = None if row is None else row["value"]
                    data_rows.append(
                        _data_row(
                            figure_id,
                            dataset="f3",
                            evaluation_semantics=F3_SEMANTICS,
                            source_metric=f"{stage}/{selection}/{metric}",
                            source_stage=stage,
                            case_or_region="full",
                            cell_label=cell,
                            metric=metric,
                            value=value,
                            unit=None if row is None else row["unit"],
                            direction=None if row is None else row["direction"],
                        )
                    )
                    x = base + cell_index + offset
                    height = np.nan if value is None else float(value)
                    ax.bar(
                        x,
                        height,
                        width=width * 0.92,
                        color=colors[cell_index],
                        label=cell if stage_index == 0 and metric_index == 0 else None,
                    )
            ax.axvline(base + len(CANONICAL_CELL_ORDER) - 0.5, color="#bbbbbb", lw=0.7)
        ax.set_xticks(
            [
                index * stage_stride + (len(CANONICAL_CELL_ORDER) - 1) / 2
                for index in range(len(CANONICAL_STAGE_ORDER))
            ],
            CANONICAL_STAGE_ORDER,
        )
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        condition_handles = [
            Patch(color=color, label=cell) for color, cell in zip(colors, CANONICAL_CELL_ORDER)
        ]
        metric_labels = []
        seen_metrics: set[str] = set()
        for _stage, _selection, metric in metric_specs:
            if metric not in seen_metrics:
                seen_metrics.add(metric)
                metric_labels.append(metric)
        metric_handles = [
            Patch(facecolor="white", edgecolor="black", label=metric) for metric in metric_labels
        ]
        ax.legend(handles=condition_handles + metric_handles, fontsize="small", ncol=2)
        fig.savefig(root / "figures" / f"{figure_id}.png", dpi=150)
    finally:
        plt.close(fig)
    data_path = _write_figure_data(root, figure_id, data_rows)
    return _record(
        figure_id=figure_id,
        relative_path=f"figures/{figure_id}.png",
        dataset="f3",
        category="f3_scalar",
        figure_role="main",
        evaluation_semantics=F3_SEMANTICS,
        source_metric=",".join(f"{a}/{b}/{c}" for a, b, c in metric_specs),
        source_stage="ft,fv,fvt",
        cell_labels=CANONICAL_CELL_ORDER,
        panel_labels=CANONICAL_CELL_ORDER,
        display_scale={
            "scale_policy": "raw scalar metric; one full-volume evaluation unit",
            "vmin": None,
            "vmax": None,
            "colormap": "condition bars",
            "difference_scale": None,
        },
        figure_data_csv=data_path,
        caption=caption,
    )


def _runtime_figure(
    report: PublicationReport, root: Path, dataset: str, plt: Any
) -> dict[str, Any]:
    figure_id = f"{dataset}_runtime_breakdown"
    rows = [row for row in report.tables["runtime_summary.csv"] if row["dataset"] == dataset]
    labels = [f"{row['stage']}:{row['cell_label'] or 'shared'}" for row in rows]
    values = [float(row["elapsed_seconds"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(rows)), 4.8), constrained_layout=True)
    try:
        colors = ["#4c78a8" if row["shared_stage"] else "#f58518" for row in rows]
        ax.bar(np.arange(len(rows)), values, color=colors)
        ax.set_xticks(np.arange(len(rows)), labels, rotation=70, ha="right")
        ax.set_ylabel("elapsed seconds")
        ax.set_title(f"{dataset.upper()} runtime breakdown")
        ax.grid(axis="y", alpha=0.25)
        fig.savefig(root / "figures" / f"{figure_id}.png", dpi=150)
    finally:
        plt.close(fig)
    data_rows = [
        _data_row(
            figure_id,
            dataset=dataset,
            evaluation_semantics=SYNTHETIC_SEMANTICS if dataset == "synthetic" else F3_SEMANTICS,
            source_stage="runtime",
            case_or_region=row["case_or_region"],
            trial_id=row["trial_id"],
            seed=row["seed"],
            cell_label=row["cell_label"],
            value=row["elapsed_seconds"],
            unit="second",
        )
        for row in rows
    ]
    data_path = _write_figure_data(root, figure_id, data_rows)
    semantics = SYNTHETIC_SEMANTICS if dataset == "synthetic" else F3_SEMANTICS
    return _record(
        figure_id=figure_id,
        relative_path=f"figures/{figure_id}.png",
        dataset=dataset,
        category="runtime",
        figure_role="main",
        evaluation_semantics=semantics,
        source_metric=None,
        source_stage="runtime",
        display_scale={
            "scale_policy": "stage-level within-experiment attribution",
            "vmin": 0.0,
            "vmax": None,
            "colormap": "shared versus cell-owned bars",
            "difference_scale": None,
        },
        figure_data_csv=data_path,
        caption=(
            "Runtime is a within-experiment attribution of shared and cell-owned stages; "
            "it is not an isolated-process benchmark."
        ),
    )


def _axis_number(axis: str) -> int:
    return {"i3": 0, "i2": 1, "i1": 2}[axis]


def _slice(array: np.ndarray, axis: str, index: int) -> np.ndarray:
    number = _axis_number(axis)
    return np.take(array, index, axis=number)


def _open_stage_volume(
    workspace: F3RunWorkspace,
    *,
    kind: str,
    fingerprint: str,
    artifact: Any,
    shape: tuple[int, int, int],
) -> np.memmap:
    artifact_shape = tuple(artifact.shape) if artifact.shape is not None else shape
    if artifact_shape != shape:
        raise ValueError(f"stage artifact shape does not match the F3 volume: {artifact.filename}")
    dtype = np.dtype(artifact.dtype or ">f4")
    path = workspace.stage_path(kind, fingerprint) / artifact.filename
    return np.memmap(path, dtype=dtype, mode="r", shape=artifact_shape, order="C")


def _contract_artifact(artifacts: tuple[Any, ...], name: str) -> Any:
    for artifact in artifacts:
        if artifact.filename == name:
            return artifact
    raise ValueError(f"artifact contract does not contain {name!r}")


def _metric_value(result: Any, stage: str, cell: str, metric: str) -> float:
    for row in result.metric_rows:
        if (
            row.stage == stage
            and row.cell_label == cell
            and row.selection == "all"
            and row.metric == metric
        ):
            if row.value is None:
                raise ValueError(f"F3 scale metric {stage}/{cell}/{metric} is nullable")
            return float(row.value)
    raise ValueError(f"F3 scale metric is missing: {stage}/{cell}/{metric}")


def _shared_scale(result: Any, stage: str, cells: tuple[str, ...]) -> tuple[float, float]:
    lows = [_metric_value(result, stage, cell, "candidate_min") for cell in cells]
    highs = [_metric_value(result, stage, cell, "candidate_max") for cell in cells]
    for cell in cells:
        lows.append(_metric_value(result, stage, cell, "reference_min"))
        highs.append(_metric_value(result, stage, cell, "reference_max"))
    low, high = min(lows), max(highs)
    if high <= low:
        margin = max(abs(low) * 0.05, 1.0e-6)
        low -= margin
        high += margin
    return _finite(low), _finite(high)


def _difference_scale(values: np.ndarray) -> tuple[float, float, float]:
    observed = float(np.max(np.abs(np.asarray(values, dtype=np.float64)))) if values.size else 0.0
    limit = max(observed, 1.0e-6)
    return observed, -limit, limit


def _threshold_for(source: Any, stage: str) -> float:
    for evidence in source.metric_evidence:
        if (
            evidence.cell_label == "RL-REF"
            and evidence.stage == stage
            and evidence.selection == "positive_p99_radius2"
        ):
            thresholds = dict(evidence.thresholds)
            if "reference_threshold" in thresholds:
                return float(thresholds["reference_threshold"])
    raise ValueError(f"validated F3 evidence has no positive-p99 threshold for {stage}")


def _slice_selection(
    policy: str,
    axis: str,
    shape: tuple[int, int, int],
    reference: np.ndarray,
    *,
    threshold: float,
    difference: np.ndarray | None,
) -> tuple[int, float]:
    number = _axis_number(axis)
    if policy == "center":
        index = shape[number] // 2
        score = float(np.count_nonzero(_slice(reference, axis, index) >= threshold))
        return index, score
    if policy == "public_reference_peak":
        best_index = 0
        best_score = -1
        for index in range(shape[number]):
            sample = np.asarray(_slice(reference, axis, index))
            mask = positive_candidate_mask(sample, epsilon=NONZERO_EPSILON)
            mask &= sample >= threshold
            score = int(np.count_nonzero(mask))
            if score > best_score:
                best_index, best_score = index, score
        return best_index, float(best_score)
    if policy == "end_to_end_difference_peak":
        if difference is None:
            raise ValueError("end_to_end_difference_peak requires a difference volume")
        best_index = 0
        best_score = -1.0
        for index in range(shape[number]):
            sample = np.asarray(_slice(difference, axis, index), dtype=np.float64)
            score = float(np.sum(np.abs(sample), dtype=np.float64))
            if score > best_score:
                best_index, best_score = index, score
        return best_index, best_score
    raise ValueError(f"unknown slice selection policy {policy!r}")


def _difference_peak_index(
    left: np.ndarray,
    right: np.ndarray,
    axis: str,
    shape: tuple[int, int, int],
) -> tuple[int, float]:
    """Find a difference peak by reading one 2D slice at a time."""

    number = _axis_number(axis)
    best_index = 0
    best_score = -1.0
    for index in range(shape[number]):
        difference = np.asarray(_slice(left, axis, index), dtype=np.float64) - np.asarray(
            _slice(right, axis, index), dtype=np.float64
        )
        score = float(np.sum(np.abs(difference), dtype=np.float64))
        if score > best_score:
            best_index, best_score = index, score
    return best_index, best_score


def _ball_structure(ndim: int, radius: float) -> np.ndarray:
    samples = int(math.ceil(radius))
    axes = np.ogrid[(slice(-samples, samples + 1),) * ndim]
    distance = np.zeros((2 * samples + 1,) * ndim, dtype=np.float64)
    for axis in axes:
        distance += axis.astype(np.float64) ** 2
    return distance <= radius * radius


def _ridge_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    mask = positive_candidate_mask(np.asarray(values), epsilon=NONZERO_EPSILON)
    return mask & (np.asarray(values) >= threshold)


def _overlay_rgb(
    reference_mask: np.ndarray, candidate_mask: np.ndarray, radius: float
) -> np.ndarray:
    reference_buffer = binary_dilation(reference_mask, structure=_ball_structure(3, radius))
    candidate_buffer = binary_dilation(candidate_mask, structure=_ball_structure(3, radius))
    exact = reference_mask & candidate_mask
    buffered = ((candidate_mask & reference_buffer) | (reference_mask & candidate_buffer)) & ~exact
    reference_only = reference_mask & ~candidate_mask & ~buffered
    candidate_only = candidate_mask & ~reference_mask & ~buffered
    rgb = np.zeros(reference_mask.shape + (3,), dtype=np.float32)
    rgb[reference_only] = (1.0, 0.0, 0.0)
    rgb[candidate_only] = (0.0, 0.25, 1.0)
    rgb[exact] = (1.0, 1.0, 1.0)
    rgb[buffered] = (0.0, 1.0, 1.0)
    return rgb


def _overlay_slice_rgb(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    axis: str,
    index: int,
    threshold: float,
    radius: float,
) -> np.ndarray:
    """Render one overlay slice using only a radius-sized 3D slab."""

    number = _axis_number(axis)
    radius_samples = int(math.ceil(radius))
    start = max(0, index - radius_samples)
    stop = min(reference.shape[number], index + radius_samples + 1)
    reference_slab = np.stack(
        [
            _ridge_mask(np.asarray(_slice(reference, axis, slab_index)), threshold)
            for slab_index in range(start, stop)
        ],
        axis=0,
    )
    candidate_slab = np.stack(
        [
            _ridge_mask(np.asarray(_slice(candidate, axis, slab_index)), threshold)
            for slab_index in range(start, stop)
        ],
        axis=0,
    )
    center = index - start
    reference_mask = reference_slab[center]
    candidate_mask = candidate_slab[center]
    reference_buffer = binary_dilation(reference_slab, structure=_ball_structure(3, radius))[center]
    candidate_buffer = binary_dilation(candidate_slab, structure=_ball_structure(3, radius))[center]
    exact = reference_mask & candidate_mask
    buffered = ((candidate_mask & reference_buffer) | (reference_mask & candidate_buffer)) & ~exact
    reference_only = reference_mask & ~candidate_mask & ~buffered
    candidate_only = candidate_mask & ~reference_mask & ~buffered
    rgb = np.zeros(reference_mask.shape + (3,), dtype=np.float32)
    rgb[reference_only] = (1.0, 0.0, 0.0)
    rgb[candidate_only] = (0.0, 0.25, 1.0)
    rgb[exact] = (1.0, 1.0, 1.0)
    rgb[buffered] = (0.0, 1.0, 1.0)
    return rgb


def _spatial_figure(
    root: Path,
    *,
    figure_id: str,
    dataset: str,
    stage: str,
    policy: str,
    axis: str,
    index: int,
    score: float,
    threshold: float,
    panels: tuple[tuple[str, np.ndarray, bool], ...],
    normal_scale: tuple[float, float],
    difference_scale: tuple[float, float, float],
    caption: str,
    source_metric: str,
    cell_labels: tuple[str, ...],
    plt: Any,
) -> dict[str, Any]:
    path = root / "figures" / f"{figure_id}.png"
    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.0 * len(panels), 4.2), squeeze=False, constrained_layout=True
    )
    data_rows = []
    try:
        normal_vmin, normal_vmax = normal_scale
        observed_difference, difference_vmin, difference_vmax = difference_scale
        for panel_index, (label, values, is_difference) in enumerate(panels):
            ax = axes[0, panel_index]
            display = np.asarray(values, dtype=np.float32)
            if is_difference:
                ax.imshow(
                    display,
                    cmap="coolwarm",
                    vmin=difference_vmin,
                    vmax=difference_vmax,
                    origin="upper",
                    aspect="auto",
                )
                cmap = "coolwarm"
            else:
                ax.imshow(
                    display,
                    cmap="viridis",
                    vmin=normal_vmin,
                    vmax=normal_vmax,
                    origin="upper",
                    aspect="auto",
                )
                cmap = "viridis"
            ax.set_title(label)
            ax.set_xticks([])
            ax.set_yticks([])
            data_rows.append(
                _data_row(
                    figure_id,
                    dataset=dataset,
                    evaluation_semantics=F3_SEMANTICS,
                    source_metric=source_metric,
                    source_stage=stage,
                    panel_label=label,
                    axis=axis,
                    slice_index=index,
                    slice_selection_policy=policy,
                    slice_score=score,
                    selection_threshold=threshold,
                    vmin=difference_vmin if is_difference else normal_vmin,
                    vmax=difference_vmax if is_difference else normal_vmax,
                    scale_policy="symmetric_zero_centered_difference"
                    if is_difference
                    else "shared_stage_slice_scale",
                    colormap=cmap,
                    difference_limit=observed_difference if is_difference else None,
                    difference_vmin=difference_vmin if is_difference else None,
                    difference_vmax=difference_vmax if is_difference else None,
                )
            )
        fig.suptitle(f"F3 public-reference agreement: {stage} {axis}={index} ({policy})")
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
    data_path = _write_figure_data(root, figure_id, data_rows)
    normal_vmin, normal_vmax = normal_scale
    observed_difference, difference_vmin, difference_vmax = difference_scale
    return _record(
        figure_id=figure_id,
        relative_path=f"figures/{figure_id}.png",
        dataset=dataset,
        category="f3_spatial",
        figure_role="main",
        evaluation_semantics=F3_SEMANTICS,
        source_metric=source_metric,
        source_stage=stage,
        cell_labels=cell_labels,
        panel_labels=tuple(panel[0] for panel in panels),
        contrast_name="end_to_end_delta" if stage in {"fv", "fvt"} else None,
        axis=axis,
        slice_index=index,
        slice_selection_policy=policy,
        selection_percentile=F3_BUFFERED_PERCENTILE,
        buffer_radius=F3_BUFFER_RADIUS,
        selection_threshold=threshold,
        display_scale={
            "vmin": normal_vmin,
            "vmax": normal_vmax,
            "scale_policy": "validated_full_volume_min_max; shared across normal panels",
            "colormap": "viridis",
            "difference_limit": observed_difference,
            "difference_vmin": difference_vmin,
            "difference_vmax": difference_vmax,
            "difference_scale": "symmetric around zero",
        },
        figure_data_csv=data_path,
        caption=caption,
    )


def _overlay_figure(
    root: Path,
    *,
    figure_id: str,
    axis: str,
    index: int,
    policy: str,
    score: float,
    threshold: float,
    reference: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    plt: Any,
) -> dict[str, Any]:
    from matplotlib.patches import Patch

    labels = tuple(CANONICAL_CELL_ORDER)
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2), squeeze=False, constrained_layout=True)
    data_rows = []
    try:
        for panel_index, cell in enumerate(labels):
            rgb = _overlay_slice_rgb(
                reference,
                candidates[cell],
                axis=axis,
                index=index,
                threshold=threshold,
                radius=F3_BUFFER_RADIUS,
            )
            ax = axes[0, panel_index]
            ax.imshow(rgb, origin="upper", aspect="auto", interpolation="nearest")
            ax.set_title(f"PUBLIC-REF vs {cell}")
            ax.set_xticks([])
            ax.set_yticks([])
            data_rows.extend(
                [
                    _data_row(
                        figure_id,
                        dataset="f3",
                        evaluation_semantics=F3_SEMANTICS,
                        source_metric="positive_p99_radius2/buffered_f1",
                        source_stage="fvt",
                        panel_label=f"PUBLIC-REF vs {cell}",
                        cell_label=cell,
                        axis=axis,
                        slice_index=index,
                        slice_selection_policy=policy,
                        slice_score=score,
                        selection_threshold=threshold,
                        value=int(
                            np.count_nonzero(
                                _ridge_mask(_slice(reference, axis, index), threshold)
                                & _ridge_mask(_slice(candidates[cell], axis, index), threshold)
                            )
                        ),
                        unit="count",
                        metric="exact_overlap_count",
                    ),
                ]
            )
        fig.suptitle(f"F3 fvt ridge overlay: {axis}={index} ({policy})")
        fig.legend(
            handles=(
                Patch(color="red", label="public-reference only"),
                Patch(color="#0040ff", label="candidate only"),
                Patch(color="white", label="exact overlap"),
                Patch(color="#00ffff", label="buffered match"),
            ),
            loc="lower center",
            ncol=4,
        )
        fig.savefig(root / "figures" / f"{figure_id}.png", dpi=150)
    finally:
        plt.close(fig)
    data_path = _write_figure_data(root, figure_id, data_rows)
    return _record(
        figure_id=figure_id,
        relative_path=f"figures/{figure_id}.png",
        dataset="f3",
        category="f3_ridge_overlay",
        figure_role="supplementary",
        evaluation_semantics=F3_SEMANTICS,
        source_metric="positive_p99_radius2/buffered_f1",
        source_stage="fvt",
        cell_labels=CANONICAL_CELL_ORDER,
        panel_labels=tuple(f"PUBLIC-REF vs {cell}" for cell in labels),
        axis=axis,
        slice_index=index,
        slice_selection_policy=policy,
        selection_percentile=F3_BUFFERED_PERCENTILE,
        buffer_radius=F3_BUFFER_RADIUS,
        selection_threshold=threshold,
        display_scale={
            "vmin": 0.0,
            "vmax": 1.0,
            "scale_policy": "categorical ridge overlay",
            "colormap": "categorical",
            "difference_scale": None,
        },
        figure_data_csv=data_path,
        caption=(
            "F3 public-reference ridge overlay: public-reference only, candidate only, "
            "exact overlap, and buffered match."
        ),
    )


def _generate_f3_spatial(report: PublicationReport, root: Path, plt: Any) -> list[dict[str, Any]]:
    f3 = report.f3
    root_path = f3.path
    manifest = f3.run_manifest
    workspace = F3RunWorkspace(root_path, f3.result.run_fingerprint, manifest, resumed=True)
    shape = tuple(f3.result.volume_shape)
    candidate_arrays: dict[str, dict[str, np.memmap]] = {
        stage: {} for stage in CANONICAL_STAGE_ORDER
    }
    opened: list[np.memmap] = []
    records: list[dict[str, Any]] = []
    with F3VolumeSource(f3.data_root, spec=f3.dataset_spec) as source:
        references: dict[str, np.memmap] = {}
        try:
            for stage in CANONICAL_STAGE_ORDER:
                references[stage] = source.open_memmap(F3_REFERENCE_STAGE_ROLES[stage])
                for cell in f3.result.cells:
                    if stage == "ft":
                        contract = scanner_stage_artifacts(shape, cell.backend)
                        kind = "scanner"
                        fingerprint = cell.stages.scanner
                        artifact = _contract_artifact(contract, "ft.dat")
                    elif stage == "fv":
                        contract = voting_stage_artifacts(shape)
                        kind = "voting"
                        fingerprint = cell.stages.voting
                        artifact = _contract_artifact(contract, "fv.dat")
                    else:
                        contract = thinning_stage_artifacts(shape)
                        kind = "thinning"
                        fingerprint = cell.stages.thinning
                        artifact = _contract_artifact(contract, "fvt.dat")
                    candidate_arrays[stage][cell.label] = _open_stage_volume(
                        workspace,
                        kind=kind,
                        fingerprint=fingerprint,
                        artifact=artifact,
                        shape=shape,
                    )
                    opened.append(candidate_arrays[stage][cell.label])

            policies_by_stage = {
                "ft": ("center", "public_reference_peak"),
                "fv": ("center", "public_reference_peak"),
                "fvt": ("center", "public_reference_peak", "end_to_end_difference_peak"),
            }
            for stage in CANONICAL_STAGE_ORDER:
                reference = references[stage]
                threshold = _threshold_for(f3, stage)
                for policy in policies_by_stage[stage]:
                    for axis in ("i3", "i2", "i1"):
                        if policy == "end_to_end_difference_peak":
                            index, score = _difference_peak_index(
                                candidate_arrays[stage]["Q-QUAL"],
                                candidate_arrays[stage]["RL-REF"],
                                axis,
                                shape,
                            )
                        else:
                            index, score = _slice_selection(
                                policy,
                                axis,
                                shape,
                                reference,
                                threshold=threshold,
                                difference=None,
                            )
                        if stage == "ft":
                            displayed_cells = ("RL-REF", "Q-REF")
                            panels = (
                                ("PUBLIC-REF fl.dat", _slice(reference, axis, index), False),
                                (
                                    "reference-like scanner ft",
                                    _slice(candidate_arrays[stage]["RL-REF"], axis, index),
                                    False,
                                ),
                                (
                                    "quality scanner ft",
                                    _slice(candidate_arrays[stage]["Q-REF"], axis, index),
                                    False,
                                ),
                                (
                                    "quality - reference-like signed difference",
                                    _slice(
                                        candidate_arrays[stage]["Q-REF"]
                                        - candidate_arrays[stage]["RL-REF"],
                                        axis,
                                        index,
                                    ),
                                    True,
                                ),
                            )
                            caption = "F3 public-reference agreement for ft before workflow; scanner outputs are shown once per backend."
                            source_metric = "ft/all/output difference"
                        else:
                            displayed_cells = CANONICAL_CELL_ORDER
                            panels = (
                                ("PUBLIC-REF", _slice(reference, axis, index), False),
                                *tuple(
                                    (
                                        cell,
                                        _slice(candidate_arrays[stage][cell], axis, index),
                                        False,
                                    )
                                    for cell in CANONICAL_CELL_ORDER
                                ),
                                (
                                    "Q-QUAL - RL-REF signed difference",
                                    _slice(
                                        candidate_arrays[stage]["Q-QUAL"]
                                        - candidate_arrays[stage]["RL-REF"],
                                        axis,
                                        index,
                                    ),
                                    True,
                                ),
                            )
                            caption = "F3 public-reference agreement; the signed difference is Q-QUAL - RL-REF and uses a separate zero-centered scale."
                            source_metric = f"{stage}/all/output difference"
                        normal_scale = _shared_scale(f3.result, stage, displayed_cells)
                        diff_values = panels[-1][1]
                        diff_scale = _difference_scale(diff_values)
                        figure_id = f"f3_{stage}_comparison_{policy}_{axis}_{index}"
                        records.append(
                            _spatial_figure(
                                root,
                                figure_id=figure_id,
                                dataset="f3",
                                stage=stage,
                                policy=policy,
                                axis=axis,
                                index=index,
                                score=score,
                                threshold=threshold,
                                panels=panels,
                                normal_scale=normal_scale,
                                difference_scale=diff_scale,
                                caption=caption,
                                source_metric=source_metric,
                                cell_labels=displayed_cells,
                                plt=plt,
                            )
                        )

                        if stage == "fvt" and policy in {
                            "public_reference_peak",
                            "end_to_end_difference_peak",
                        }:
                            records.append(
                                _overlay_figure(
                                    root,
                                    figure_id=f"f3_fvt_ridge_overlay_{policy}_{axis}_{index}",
                                    axis=axis,
                                    index=index,
                                    policy=policy,
                                    score=score,
                                    threshold=threshold,
                                    reference=reference,
                                    candidates={
                                        cell: candidate_arrays[stage][cell]
                                        for cell in CANONICAL_CELL_ORDER
                                    },
                                    plt=plt,
                                )
                            )
        finally:
            for array in opened:
                mapping = getattr(array, "_mmap", None)
                if mapping is not None and not mapping.closed:
                    mapping.close()
    return records


def _omitted_skin_figures(report: PublicationReport) -> list[dict[str, Any]]:
    if report.synthetic.skinning_enabled:
        return []
    output = []
    for figure_id in ("synthetic_skin_buffered_f1_by_case",):
        output.append(
            _record(
                figure_id=figure_id,
                relative_path=f"figures/{figure_id}.png",
                dataset="synthetic",
                category="synthetic_scalar",
                figure_role="main",
                evaluation_semantics=SYNTHETIC_SEMANTICS,
                source_metric="skin/skin_cells/buffered_f1",
                source_stage="skin",
                cell_labels=CANONICAL_CELL_ORDER,
                panel_labels=CANONICAL_CELL_ORDER,
                caption="Skin figure omitted because the validated synthetic source run disabled skinning.",
                omitted=True,
                omission_reason="source synthetic skinning is disabled",
            )
        )
    return output


def generate_figures(
    report: PublicationReport, root: str | Path
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, str]]:
    """Generate all fixed publication figures and their source CSVs."""

    output = Path(root)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    (output / "figure_data").mkdir(parents=True, exist_ok=True)
    plt, matplotlib_version, backend = _matplotlib()
    records: list[dict[str, Any]] = []
    records.append(_synthetic_heatmap(report, output, plt))
    records.append(
        _synthetic_scalar_figure(
            report,
            output,
            figure_id="synthetic_fvt_buffered_f1_by_case",
            filename="synthetic_fvt_buffered_f1_by_case.png",
            stage="fvt",
            selection="positive_top_truth_count",
            metric="buffered_f1",
            ylabel="buffered F1",
            title="Synthetic FVT buffered F1 by case",
            caption="Synthetic known-truth FVT buffered F1; deterministic cases are single points and stochastic cases show median and IQR without a significance test.",
            plt=plt,
        )
    )
    records.append(
        _synthetic_scalar_figure(
            report,
            output,
            figure_id="synthetic_fvt_hausdorff_p95_by_case",
            filename="synthetic_fvt_hausdorff_p95_by_case.png",
            stage="fvt",
            selection="positive_top_truth_count",
            metric="hausdorff_p95",
            ylabel="Hausdorff p95 (voxel)",
            title="Synthetic FVT Hausdorff p95 by case",
            caption="Synthetic known-truth FVT ridge Hausdorff p95; deterministic cases are single points and stochastic cases show median and IQR without a significance test.",
            plt=plt,
        )
    )
    records.append(_synthetic_orientation_figure(report, output, plt))
    if report.synthetic.skinning_enabled:
        records.append(
            _synthetic_scalar_figure(
                report,
                output,
                figure_id="synthetic_skin_buffered_f1_by_case",
                filename="synthetic_skin_buffered_f1_by_case.png",
                stage="skin",
                selection="skin_cells",
                metric="buffered_f1",
                ylabel="buffered F1",
                title="Synthetic skin buffered F1 by case",
                caption="Synthetic known-truth skin buffered F1; skinning rows are shown only when the validated source run enabled skinning.",
                plt=plt,
            )
        )
    else:
        records.extend(_omitted_skin_figures(report))

    records.extend(
        [
            _f3_scalar_figure(
                report,
                output,
                figure_id="f3_normalized_correlation_by_stage",
                metric_specs=tuple(
                    (stage, "all", "normalized_correlation") for stage in CANONICAL_STAGE_ORDER
                ),
                ylabel="F3 public-reference agreement: normalized correlation",
                title="F3 public-reference agreement by stage",
                caption="F3 public-reference agreement shown for one full-volume evaluation unit; it is a reference-agreement diagnostic, not a ground-truth assessment.",
                plt=plt,
            ),
            _f3_scalar_figure(
                report,
                output,
                figure_id="f3_buffered_f1_by_stage",
                metric_specs=tuple(
                    (stage, "positive_p99_radius2", "buffered_f1")
                    for stage in CANONICAL_STAGE_ORDER
                ),
                ylabel="F3 public-reference agreement: buffered F1",
                title="F3 ridge agreement by stage",
                caption="F3 public-reference agreement using the validated positive-p99 radius-2 ridge contract; one full-volume evaluation unit.",
                plt=plt,
            ),
            _f3_scalar_figure(
                report,
                output,
                figure_id="f3_sparse_distance_p95_by_stage",
                metric_specs=tuple(
                    (stage, "positive_p99_distance", metric)
                    for stage in CANONICAL_STAGE_ORDER
                    for metric in ("candidate_to_reference_p95", "reference_to_candidate_p95")
                ),
                ylabel="ridge distance p95 (voxel)",
                title="F3 ridge distance by stage",
                caption="F3 ridge displacement diagnostics against the public reference; nullable sparse-distance values remain missing and are not converted to zero.",
                plt=plt,
            ),
            _f3_scalar_figure(
                report,
                output,
                figure_id="f3_nonzero_fraction_ratio_by_stage",
                metric_specs=tuple(
                    (stage, "all", "nonzero_fraction_ratio") for stage in CANONICAL_STAGE_ORDER
                ),
                ylabel="density ratio",
                title="F3 density ratio by stage",
                caption="F3 density/stability diagnostic against the public reference; neutral ratios are descriptive and are not directional improvements.",
                plt=plt,
            ),
            _runtime_figure(report, output, "synthetic", plt),
            _runtime_figure(report, output, "f3", plt),
        ]
    )
    records.extend(_generate_f3_spatial(report, output, plt))
    # Stable order is part of the publication contract and independent of file-system order.
    records.sort(key=lambda item: item["figure_id"])
    return tuple(records), {"matplotlib": matplotlib_version, "backend": backend}


__all__ = ["generate_figures"]
