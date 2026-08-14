"""Amplitude-backed spatial figures for the F3 compact publication."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path, PurePath
from typing import Any

import numpy as np

from pyosv.candidate_volume import NONZERO_EPSILON, positive_candidate_mask

from ..f3d_mode_comparison.metrics import METRIC_REGISTRY
from .config import (
    AMPLITUDE_ALPHA_MAX,
    AMPLITUDE_PERCENTILE,
    ATTRIBUTE_COLORMAP,
    DIFFERENCE_COLORMAP,
    DIFFERENCE_PERCENTILE,
    DISPLAY_CELL,
    FIGURE_DATA_HEADER,
    PUBLIC_REFERENCE_LABEL,
    SLICE_AXIS,
    STAGE_ORDER,
)
from .models import CompactSourceContext, RidgeStageThresholds, StageSource

_CANDIDATE_TITLES = {
    "ft": "Q scanner output (Q-QUAL lineage)",
    "fv": "Q scanner voting output (Q-QUAL lineage)",
    "fvt": "Q-QUAL",
}
_SCALE_METRICS = ("reference_max", "candidate_max")
_METRIC_DEFINITIONS = {(item.stage, item.selection, item.metric): item for item in METRIC_REGISTRY}


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _basename(value: str, name: str) -> str:
    if not value or PurePath(value).name != value or Path(value).is_absolute():
        raise ValueError(f"{name} must be a basename")
    return value


def _close_memmap(volume: np.memmap | None) -> None:
    if volume is None:
        return
    mapping = getattr(volume, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def _read_i2_slice(
    path: Path,
    *,
    shape: tuple[int, int, int],
    storage_dtype: str,
    index: int,
    label: str,
) -> np.ndarray:
    volume: np.memmap | None = None
    try:
        volume = np.memmap(path, dtype=storage_dtype, mode="r", shape=shape, order="C")
        values = np.array(volume[:, index, :], dtype=np.float32, copy=True)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {label} selected slice: {path}") from error
    finally:
        _close_memmap(volume)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{label} selected slice must contain only finite values")
    return values


def _source_metric_max(context: CompactSourceContext, stage: str, metric: str) -> float:
    rows = tuple(context.f3.result.metric_rows)
    matches = tuple(
        row
        for row in rows
        if (
            getattr(row, "cell_label", None),
            getattr(row, "stage", None),
            getattr(row, "selection", None),
            getattr(row, "metric", None),
        )
        == (DISPLAY_CELL, stage, "all", metric)
    )
    if len(matches) != 1:
        raise ValueError(
            "F3 source must contain exactly one display-scale metric row for "
            f"{(DISPLAY_CELL, stage, 'all', metric)!r}; found {len(matches)}"
        )
    definition = _METRIC_DEFINITIONS.get((stage, "all", metric))
    if definition is None:
        raise ValueError(f"F3 metric registry has no definition for {(stage, 'all', metric)!r}")
    row = matches[0]
    if (getattr(row, "unit", None), getattr(row, "direction", None)) != (
        definition.unit,
        definition.direction,
    ):
        raise ValueError(
            f"F3 display-scale metric semantics do not match the registry for "
            f"{(stage, 'all', metric)!r}"
        )
    value = getattr(row, "value", None)
    if value is None:
        raise ValueError(f"F3 display-scale metric {(stage, 'all', metric)!r} is null")
    return _finite_number(value, f"F3 display-scale metric {stage}/{metric}")


def _percentile_limit(values: np.ndarray, percentile: float, name: str) -> float:
    observed = float(np.percentile(np.abs(values.astype(np.float64, copy=False)), percentile))
    if not math.isfinite(observed):
        raise ValueError(f"{name} percentile must be finite")
    return observed if observed > 0.0 else 1.0e-6


def _ridge_alpha(values: np.ndarray, threshold: float, vmax: float) -> np.ndarray:
    mask = positive_candidate_mask(values, epsilon=NONZERO_EPSILON)
    mask &= values >= threshold
    alpha = np.zeros(values.shape, dtype=np.float32)
    if vmax <= threshold:
        alpha[mask] = AMPLITUDE_ALPHA_MAX
    else:
        scaled = np.clip((values[mask] - threshold) / (vmax - threshold), 0.0, 1.0)
        alpha[mask] = AMPLITUDE_ALPHA_MAX * scaled
    return alpha


def _csv_value(value: object, field: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        raise ValueError(f"figure-data field {field!r} must not be bool")
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        return repr(_finite_number(value, f"figure-data field {field!r}"))
    return str(value)


def _write_figure_data(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(FIGURE_DATA_HEADER)
        for row in rows:
            if tuple(row) != FIGURE_DATA_HEADER:
                raise ValueError("figure-data row fields must match the fixed CSV header")
            writer.writerow(_csv_value(row[field], field) for field in FIGURE_DATA_HEADER)


def _figure_rows(
    context: CompactSourceContext,
    *,
    figure_id: str,
    source: StageSource,
    thresholds: RidgeStageThresholds,
    amplitude_limit: float,
    overlay_vmax: float,
    difference_limit: float,
) -> tuple[Mapping[str, object], ...]:
    common: dict[str, object] = {
        "figure_id": figure_id,
        "stage": source.stage,
        "axis": context.selected_slice.axis,
        "slice_index": context.selected_slice.index,
        "slice_selection_policy": context.selected_slice.policy,
        "amplitude_file": _basename(context.amplitude.filename, "amplitude filename"),
        "amplitude_sha256": context.amplitude.sha256,
        "amplitude_limit": amplitude_limit,
        "alpha_max": AMPLITUDE_ALPHA_MAX,
    }

    def row(**values: object) -> Mapping[str, object]:
        merged = {**common, **values}
        return {field: merged.get(field) for field in FIGURE_DATA_HEADER}

    return (
        row(
            panel_label=PUBLIC_REFERENCE_LABEL,
            source_label=PUBLIC_REFERENCE_LABEL,
            source_file=_basename(source.public_reference_filename, "public reference filename"),
            source_sha256=source.public_reference_sha256,
            source_stage_fingerprint=None,
            selection_threshold=thresholds.public_reference_threshold,
            overlay_vmin=0.0,
            overlay_vmax=overlay_vmax,
            colormap=ATTRIBUTE_COLORMAP,
            difference_limit=None,
        ),
        row(
            panel_label=DISPLAY_CELL,
            source_label=DISPLAY_CELL,
            source_file=_basename(source.candidate_filename, "Q-QUAL stage filename"),
            source_sha256=None,
            source_stage_fingerprint=source.candidate_fingerprint,
            selection_threshold=thresholds.q_qual_threshold,
            overlay_vmin=0.0,
            overlay_vmax=overlay_vmax,
            colormap=ATTRIBUTE_COLORMAP,
            difference_limit=None,
        ),
        row(
            panel_label="difference",
            source_label=f"{DISPLAY_CELL} - {PUBLIC_REFERENCE_LABEL}",
            source_file=None,
            source_sha256=None,
            source_stage_fingerprint=None,
            selection_threshold=None,
            overlay_vmin=-difference_limit,
            overlay_vmax=difference_limit,
            colormap=DIFFERENCE_COLORMAP,
            difference_limit=difference_limit,
        ),
    )


def _plot_stage(
    plt: Any,
    path: Path,
    *,
    stage: str,
    axis: str,
    index: int,
    amplitude: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    amplitude_limit: float,
    reference_threshold: float,
    candidate_threshold: float,
    overlay_vmax: float,
    difference: np.ndarray,
    difference_limit: float,
) -> None:
    reference_alpha = _ridge_alpha(reference, reference_threshold, overlay_vmax)
    candidate_alpha = _ridge_alpha(candidate, candidate_threshold, overlay_vmax)
    difference_alpha = AMPLITUDE_ALPHA_MAX * np.clip(
        np.abs(difference) / difference_limit, 0.0, 1.0
    )
    figure, axes = plt.subplots(1, 3, figsize=(16.0, 5.5), squeeze=False)
    panels = axes[0]
    try:
        for panel in panels:
            panel.imshow(
                amplitude,
                cmap="gray",
                vmin=-amplitude_limit,
                vmax=amplitude_limit,
                origin="upper",
                aspect="auto",
            )
            panel.set_xticks([])
            panel.set_yticks([])

        reference_image = panels[0].imshow(
            reference,
            cmap=ATTRIBUTE_COLORMAP,
            vmin=0.0,
            vmax=overlay_vmax,
            alpha=reference_alpha,
            origin="upper",
            aspect="auto",
        )
        panels[1].imshow(
            candidate,
            cmap=ATTRIBUTE_COLORMAP,
            vmin=0.0,
            vmax=overlay_vmax,
            alpha=candidate_alpha,
            origin="upper",
            aspect="auto",
        )
        difference_image = panels[2].imshow(
            difference,
            cmap=DIFFERENCE_COLORMAP,
            vmin=-difference_limit,
            vmax=difference_limit,
            alpha=difference_alpha,
            origin="upper",
            aspect="auto",
        )
        panels[0].set_title("Amplitude + PUBLIC-REF")
        panels[1].set_title(f"Amplitude + {_CANDIDATE_TITLES[stage]}")
        panels[2].set_title("Amplitude + Q-QUAL - PUBLIC-REF")
        figure.suptitle(f"F3 {stage}: PUBLIC-REF vs Q-QUAL at {axis}={index}", y=0.96)
        figure.subplots_adjust(left=0.025, right=0.985, bottom=0.22, top=0.83, wspace=0.08)
        attribute_axis = figure.add_axes((0.08, 0.075, 0.52, 0.035))
        difference_axis = figure.add_axes((0.70, 0.075, 0.25, 0.035))
        figure.colorbar(
            reference_image,
            cax=attribute_axis,
            orientation="horizontal",
            label="attribute value",
        )
        figure.colorbar(
            difference_image,
            cax=difference_axis,
            orientation="horizontal",
            label="Q-QUAL - PUBLIC-REF",
        )
        figure.savefig(path, dpi=150)
    finally:
        plt.close(figure)


def _caption(stage: str, axis: str, index: int) -> str:
    return (
        f"F3 {stage} comparison at {axis}={index}: amplitude-backed PUBLIC-REF, "
        f"{_CANDIDATE_TITLES[stage]}, and signed Q-QUAL minus PUBLIC-REF difference."
    )


def generate_figures(
    context: CompactSourceContext,
    root: str | Path,
) -> tuple[Mapping[str, object], ...]:
    """Generate the fixed three compact F3 figures and their figure-data CSV files."""

    if context.selected_slice.axis != SLICE_AXIS:
        raise ValueError(f"compact figures require slice axis {SLICE_AXIS!r}")
    if tuple(source.stage for source in context.stage_sources) != STAGE_ORDER:
        raise ValueError("compact stage sources must follow the fixed stage order")
    if tuple(item.stage for item in context.ridge_threshold_contract.stages) != STAGE_ORDER:
        raise ValueError("compact ridge thresholds must follow the fixed stage order")
    shape = tuple(context.f3.result.volume_shape)
    if len(shape) != 3 or any(
        isinstance(size, bool) or not isinstance(size, Integral) or int(size) <= 0 for size in shape
    ):
        raise ValueError("compact F3 shape must contain three positive integers")
    shape = tuple(int(size) for size in shape)
    index = context.selected_slice.index
    if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= int(index) < shape[1]:
        raise ValueError("compact i2 slice index is outside the F3 volume")
    index = int(index)
    storage_dtype = np.dtype(context.f3.result.storage_dtype).str
    if storage_dtype != context.amplitude.storage_dtype:
        raise ValueError("amplitude and stage storage dtypes must match")

    amplitude = _read_i2_slice(
        context.amplitude.resolved_path,
        shape=shape,
        storage_dtype=storage_dtype,
        index=index,
        label="amplitude",
    )
    amplitude_limit = _percentile_limit(amplitude, AMPLITUDE_PERCENTILE, "amplitude")

    prepared = []
    for source, thresholds in zip(
        context.stage_sources,
        context.ridge_threshold_contract.stages,
        strict=True,
    ):
        reference_threshold = _finite_number(
            thresholds.public_reference_threshold,
            f"public {source.stage} threshold",
        )
        candidate_threshold = _finite_number(
            thresholds.q_qual_threshold,
            f"Q-QUAL {source.stage} threshold",
        )
        reference = _read_i2_slice(
            source.public_reference_path,
            shape=shape,
            storage_dtype=storage_dtype,
            index=index,
            label=f"public {source.stage} reference",
        )
        candidate = _read_i2_slice(
            source.candidate_path,
            shape=shape,
            storage_dtype=storage_dtype,
            index=index,
            label=f"Q-QUAL {source.stage} candidate",
        )
        difference = candidate - reference
        difference_limit = _percentile_limit(
            difference,
            DIFFERENCE_PERCENTILE,
            f"{source.stage} difference",
        )
        maxima = tuple(
            _source_metric_max(context, source.stage, metric) for metric in _SCALE_METRICS
        )
        overlay_vmax = max(maxima)
        if overlay_vmax < 0.0:
            raise ValueError(f"{source.stage} overlay maximum must be non-negative")
        prepared.append(
            (
                source,
                thresholds,
                reference,
                candidate,
                difference,
                difference_limit,
                reference_threshold,
                candidate_threshold,
                overlay_vmax,
            )
        )

    output_root = Path(root)
    figures_root = output_root / "figures"
    data_root = output_root / "figure_data"
    figures_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    from pyosv.viz import require_matplotlib

    plt = require_matplotlib()
    records: list[Mapping[str, object]] = []
    for (
        source,
        thresholds,
        reference,
        candidate,
        difference,
        difference_limit,
        reference_threshold,
        candidate_threshold,
        overlay_vmax,
    ) in prepared:
        figure_id = f"f3_{source.stage}_public_ref_vs_q_qual_{SLICE_AXIS}_{index}"
        png_relative = f"figures/{figure_id}.png"
        csv_relative = f"figure_data/{figure_id}.csv"
        rows = _figure_rows(
            context,
            figure_id=figure_id,
            source=source,
            thresholds=thresholds,
            amplitude_limit=amplitude_limit,
            overlay_vmax=overlay_vmax,
            difference_limit=difference_limit,
        )
        _write_figure_data(output_root / csv_relative, rows)
        _plot_stage(
            plt,
            output_root / png_relative,
            stage=source.stage,
            axis=SLICE_AXIS,
            index=index,
            amplitude=amplitude,
            reference=reference,
            candidate=candidate,
            amplitude_limit=amplitude_limit,
            reference_threshold=reference_threshold,
            candidate_threshold=candidate_threshold,
            overlay_vmax=overlay_vmax,
            difference=difference,
            difference_limit=difference_limit,
        )
        records.append(
            {
                "figure_id": figure_id,
                "relative_path": png_relative,
                "figure_data_csv": csv_relative,
                "stage": source.stage,
                "caption": _caption(source.stage, SLICE_AXIS, index),
            }
        )
    return tuple(records)


__all__ = ["generate_figures"]
