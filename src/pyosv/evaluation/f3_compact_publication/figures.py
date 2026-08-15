"""Amplitude-backed spatial atlases for the F3 compact publication."""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path, PurePath
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation

from pyosv.candidate_volume import NONZERO_EPSILON, positive_candidate_mask

from ..f3d_mode_comparison.metrics import METRIC_REGISTRY
from .config import (
    AMPLITUDE_PERCENTILE,
    ATTRIBUTE_ALPHA_GAMMA,
    ATTRIBUTE_ALPHA_MAX,
    ATTRIBUTE_ALPHA_MIN,
    ATTRIBUTE_COLORMAP,
    ATTRIBUTE_DISPLAY_THRESHOLD_RATIO,
    ATTRIBUTE_HALO_ALPHA,
    ATTRIBUTE_HALO_ENABLED,
    ATTRIBUTE_HALO_RADIUS_PIXELS,
    ATTRIBUTE_HALO_STRUCTURE,
    DIFFERENCE_COLORMAP,
    DIFFERENCE_PERCENTILE,
    DISPLAY_CELL,
    FIGURE_DATA_HEADER,
    IMAGE_INTERPOLATION,
    PUBLIC_REFERENCE_LABEL,
    SECTION_GROUPS,
    SECTION_SELECTION_POLICY,
    SECTIONS_PER_AXIS,
    STAGE_ORDER,
)
from .models import (
    CompactSourceContext,
    RidgeStageThresholds,
    SelectedSection,
    StageSource,
)

_CANDIDATE_TITLES = {
    "ft": "Q scanner output (Q-QUAL lineage)",
    "fv": "Q scanner voting output (Q-QUAL lineage)",
    "fvt": "Q-QUAL",
}
_SCALE_METRICS = ("reference_max", "candidate_max")
_METRIC_DEFINITIONS = {(item.stage, item.selection, item.metric): item for item in METRIC_REGISTRY}
HALO_STRUCTURE = np.array(
    [
        [False, True, False],
        [True, True, True],
        [False, True, False],
    ],
    dtype=bool,
)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _display_threshold(source_threshold: float) -> float:
    return source_threshold * ATTRIBUTE_DISPLAY_THRESHOLD_RATIO


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


def _section(volume: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "i1":
        return volume[:, :, index]
    if axis == "i3":
        return volume[index, :, :].T
    raise ValueError(f"unsupported compact section axis: {axis!r}")


def _read_selected_sections(
    path: Path,
    *,
    shape: tuple[int, int, int],
    storage_dtype: str,
    selections: Sequence[SelectedSection],
    label: str,
) -> dict[str, tuple[np.ndarray, ...]]:
    volume: np.memmap | None = None
    grouped: dict[str, list[np.ndarray]] = {name: [] for name, _axis in SECTION_GROUPS}
    try:
        volume = np.memmap(path, dtype=storage_dtype, mode="r", shape=shape, order="C")
        for selected in selections:
            values = np.array(
                _section(volume, selected.axis, selected.index),
                dtype=np.float32,
                copy=True,
            )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{label} selected sections must contain only finite values")
            grouped[selected.section_group].append(values)
    except OSError as error:
        raise ValueError(f"cannot read {label} selected sections: {path}") from error
    finally:
        _close_memmap(volume)
    return {name: tuple(values) for name, values in grouped.items()}


def _source_metric_max(context: CompactSourceContext, stage: str, metric: str) -> float:
    matches = tuple(
        row
        for row in context.f3.result.metric_rows
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


def _percentile_limit(values: Sequence[np.ndarray], percentile: float, name: str) -> float:
    flattened = np.concatenate(
        [np.abs(value.astype(np.float64, copy=False)).ravel() for value in values]
    )
    observed = float(np.percentile(flattened, percentile))
    if not math.isfinite(observed):
        raise ValueError(f"{name} percentile must be finite")
    return observed if observed > 0.0 else 1.0e-6


def _display_mask(values: np.ndarray, display_threshold: float) -> np.ndarray:
    mask = positive_candidate_mask(values, epsilon=NONZERO_EPSILON)
    mask &= values >= display_threshold
    return mask


def _halo_mask(mask: np.ndarray) -> np.ndarray:
    if not ATTRIBUTE_HALO_ENABLED:
        return np.zeros(mask.shape, dtype=bool)
    halo = binary_dilation(
        mask,
        structure=HALO_STRUCTURE,
        iterations=ATTRIBUTE_HALO_RADIUS_PIXELS,
    )
    halo &= ~mask
    return halo


def _halo_alpha(mask: np.ndarray) -> np.ndarray:
    alpha = np.zeros(mask.shape, dtype=np.float32)
    alpha[mask] = ATTRIBUTE_HALO_ALPHA
    return alpha


def _ridge_alpha(
    values: np.ndarray,
    display_threshold: float,
    vmax: float,
) -> np.ndarray:
    mask = _display_mask(values, display_threshold)
    alpha = np.zeros(values.shape, dtype=np.float32)
    if vmax <= display_threshold:
        alpha[mask] = ATTRIBUTE_ALPHA_MAX
    else:
        scaled = np.clip(
            (values[mask] - display_threshold) / (vmax - display_threshold),
            0.0,
            1.0,
        )
        alpha[mask] = (
            ATTRIBUTE_ALPHA_MIN
            + (ATTRIBUTE_ALPHA_MAX - ATTRIBUTE_ALPHA_MIN) * scaled**ATTRIBUTE_ALPHA_GAMMA
        )
    return alpha


def _csv_value(value: object, field: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
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
    selections: Sequence[SelectedSection],
    amplitude_limit: float,
    overlay_vmax: float,
    difference_limit: float,
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for selected in selections:
        common: dict[str, object] = {
            "figure_id": figure_id,
            "stage": source.stage,
            "section_group": selected.section_group,
            "axis": selected.axis,
            "bin_index": selected.bin_index,
            "section_index": selected.index,
            "selection_policy": selected.policy,
            "ridge_count_score": selected.ridge_count_score,
            "amplitude_file": _basename(context.amplitude.filename, "amplitude filename"),
            "amplitude_sha256": context.amplitude.sha256,
            "amplitude_limit": amplitude_limit,
        }

        def row(**values: object) -> Mapping[str, object]:
            merged = {**common, **values}
            return {field: merged.get(field) for field in FIGURE_DATA_HEADER}

        rows.extend(
            (
                row(
                    panel_label=PUBLIC_REFERENCE_LABEL,
                    source_label=PUBLIC_REFERENCE_LABEL,
                    source_file=_basename(
                        source.public_reference_filename, "public reference filename"
                    ),
                    source_sha256=source.public_reference_sha256,
                    source_stage_fingerprint=None,
                    selection_threshold=thresholds.public_reference_threshold,
                    display_threshold=_display_threshold(thresholds.public_reference_threshold),
                    overlay_vmin=0.0,
                    overlay_vmax=overlay_vmax,
                    alpha_min=ATTRIBUTE_ALPHA_MIN,
                    alpha_max=ATTRIBUTE_ALPHA_MAX,
                    alpha_gamma=ATTRIBUTE_ALPHA_GAMMA,
                    colormap=ATTRIBUTE_COLORMAP,
                    interpolation=IMAGE_INTERPOLATION,
                    halo_enabled=ATTRIBUTE_HALO_ENABLED,
                    halo_radius_pixels=ATTRIBUTE_HALO_RADIUS_PIXELS,
                    halo_alpha=ATTRIBUTE_HALO_ALPHA,
                    halo_structure=ATTRIBUTE_HALO_STRUCTURE,
                    difference_limit=None,
                ),
                row(
                    panel_label=DISPLAY_CELL,
                    source_label=DISPLAY_CELL,
                    source_file=_basename(source.candidate_filename, "Q-QUAL stage filename"),
                    source_sha256=None,
                    source_stage_fingerprint=source.candidate_fingerprint,
                    selection_threshold=thresholds.q_qual_threshold,
                    display_threshold=_display_threshold(thresholds.q_qual_threshold),
                    overlay_vmin=0.0,
                    overlay_vmax=overlay_vmax,
                    alpha_min=ATTRIBUTE_ALPHA_MIN,
                    alpha_max=ATTRIBUTE_ALPHA_MAX,
                    alpha_gamma=ATTRIBUTE_ALPHA_GAMMA,
                    colormap=ATTRIBUTE_COLORMAP,
                    interpolation=IMAGE_INTERPOLATION,
                    halo_enabled=ATTRIBUTE_HALO_ENABLED,
                    halo_radius_pixels=ATTRIBUTE_HALO_RADIUS_PIXELS,
                    halo_alpha=ATTRIBUTE_HALO_ALPHA,
                    halo_structure=ATTRIBUTE_HALO_STRUCTURE,
                    difference_limit=None,
                ),
                row(
                    panel_label="difference",
                    source_label=f"{DISPLAY_CELL} - {PUBLIC_REFERENCE_LABEL}",
                    source_file=None,
                    source_sha256=None,
                    source_stage_fingerprint=None,
                    selection_threshold=None,
                    display_threshold=None,
                    overlay_vmin=-difference_limit,
                    overlay_vmax=difference_limit,
                    alpha_min=None,
                    alpha_max=ATTRIBUTE_ALPHA_MAX,
                    alpha_gamma=None,
                    colormap=DIFFERENCE_COLORMAP,
                    interpolation=IMAGE_INTERPOLATION,
                    halo_enabled=False,
                    halo_radius_pixels=None,
                    halo_alpha=None,
                    halo_structure=None,
                    difference_limit=difference_limit,
                ),
            )
        )
    return tuple(rows)


def _plot_atlas(
    plt: Any,
    path: Path,
    *,
    stage: str,
    section_group: str,
    axis: str,
    indices: Sequence[int],
    amplitudes: Sequence[np.ndarray],
    references: Sequence[np.ndarray],
    candidates: Sequence[np.ndarray],
    differences: Sequence[np.ndarray],
    amplitude_limit: float,
    reference_display_threshold: float,
    candidate_display_threshold: float,
    overlay_vmax: float,
    difference_limit: float,
) -> None:
    figure, axes = plt.subplots(SECTIONS_PER_AXIS, 3, figsize=(16.0, 15.0), squeeze=False)
    reference_image = None
    difference_image = None
    try:
        for row_index, (index, amplitude, reference, candidate, difference) in enumerate(
            zip(indices, amplitudes, references, candidates, differences, strict=True)
        ):
            panels = axes[row_index]
            for panel in panels:
                panel.imshow(
                    amplitude,
                    cmap="gray",
                    vmin=-amplitude_limit,
                    vmax=amplitude_limit,
                    interpolation=IMAGE_INTERPOLATION,
                    origin="upper",
                    aspect="auto",
                )
                panel.set_xticks([])
                panel.set_yticks([])
            reference_mask = _display_mask(reference, reference_display_threshold)
            reference_halo = _halo_mask(reference_mask)
            panels[0].imshow(
                np.where(reference_halo, reference, np.nan),
                cmap=ATTRIBUTE_COLORMAP,
                vmin=0.0,
                vmax=overlay_vmax,
                alpha=_halo_alpha(reference_halo),
                interpolation=IMAGE_INTERPOLATION,
                origin="upper",
                aspect="auto",
            )
            reference_image = panels[0].imshow(
                reference,
                cmap=ATTRIBUTE_COLORMAP,
                vmin=0.0,
                vmax=overlay_vmax,
                alpha=_ridge_alpha(reference, reference_display_threshold, overlay_vmax),
                interpolation=IMAGE_INTERPOLATION,
                origin="upper",
                aspect="auto",
            )
            candidate_mask = _display_mask(candidate, candidate_display_threshold)
            candidate_halo = _halo_mask(candidate_mask)
            panels[1].imshow(
                np.where(candidate_halo, candidate, np.nan),
                cmap=ATTRIBUTE_COLORMAP,
                vmin=0.0,
                vmax=overlay_vmax,
                alpha=_halo_alpha(candidate_halo),
                interpolation=IMAGE_INTERPOLATION,
                origin="upper",
                aspect="auto",
            )
            panels[1].imshow(
                candidate,
                cmap=ATTRIBUTE_COLORMAP,
                vmin=0.0,
                vmax=overlay_vmax,
                alpha=_ridge_alpha(candidate, candidate_display_threshold, overlay_vmax),
                interpolation=IMAGE_INTERPOLATION,
                origin="upper",
                aspect="auto",
            )
            difference_image = panels[2].imshow(
                difference,
                cmap=DIFFERENCE_COLORMAP,
                vmin=-difference_limit,
                vmax=difference_limit,
                alpha=ATTRIBUTE_ALPHA_MAX
                * np.clip(np.abs(difference) / difference_limit, 0.0, 1.0),
                interpolation=IMAGE_INTERPOLATION,
                origin="upper",
                aspect="auto",
            )
            panels[0].set_ylabel(f"{axis}={index}", rotation=0, labelpad=30, va="center")

        axes[0, 0].set_title("Amplitude + PUBLIC-REF")
        axes[0, 1].set_title(f"Amplitude + {_CANDIDATE_TITLES[stage]}")
        axes[0, 2].set_title("Amplitude + Q-QUAL - PUBLIC-REF")
        title = "time slices" if section_group == "time_slices" else "inline sections"
        figure.suptitle(f"F3 {stage}: PUBLIC-REF vs Q-QUAL — {title}", y=0.985)
        figure.supxlabel("crossline index (i2)", y=0.085)
        y_label = "inline index (i3)" if section_group == "time_slices" else "time sample (i1)"
        figure.supylabel(y_label, x=0.015)
        figure.subplots_adjust(
            left=0.07, right=0.985, bottom=0.115, top=0.95, wspace=0.06, hspace=0.10
        )
        attribute_axis = figure.add_axes((0.10, 0.04, 0.50, 0.018))
        difference_axis = figure.add_axes((0.70, 0.04, 0.25, 0.018))
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


def _caption(stage: str, section_group: str, axis: str, indices: Sequence[int]) -> str:
    label = "time-slice" if section_group == "time_slices" else "inline-section"
    joined = ", ".join(str(index) for index in indices)
    return (
        f"F3 {stage} {label} atlas at {axis} indices {joined}: amplitude-backed "
        f"PUBLIC-REF, {_CANDIDATE_TITLES[stage]}, and signed Q-QUAL minus PUBLIC-REF difference."
    )


def _selected_groups(
    context: CompactSourceContext,
    shape: tuple[int, int, int],
) -> dict[str, tuple[SelectedSection, ...]]:
    expected = tuple(
        (section_group, axis, bin_index)
        for section_group, axis in SECTION_GROUPS
        for bin_index in range(SECTIONS_PER_AXIS)
    )
    actual = tuple(
        (item.section_group, item.axis, item.bin_index) for item in context.selected_sections
    )
    if actual != expected or any(
        item.policy != SECTION_SELECTION_POLICY for item in context.selected_sections
    ):
        raise ValueError("compact selected sections must follow the fixed section contract")
    axis_positions = {"i1": 2, "i3": 0}
    for selected in context.selected_sections:
        if (
            isinstance(selected.index, bool)
            or not isinstance(selected.index, Integral)
            or not 0 <= int(selected.index) < shape[axis_positions[selected.axis]]
        ):
            raise ValueError(f"compact {selected.axis} section index is outside the F3 volume")
    return {
        name: tuple(item for item in context.selected_sections if item.section_group == name)
        for name, _axis in SECTION_GROUPS
    }


def generate_figures(
    context: CompactSourceContext,
    root: str | Path,
) -> tuple[Mapping[str, object], ...]:
    """Generate six fixed four-section F3 atlases and their figure-data CSV files."""

    if tuple(source.stage for source in context.stage_sources) != STAGE_ORDER:
        raise ValueError("compact stage sources must follow the fixed stage order")
    if tuple(item.stage for item in context.ridge_threshold_contract.stages) != STAGE_ORDER:
        raise ValueError("compact ridge thresholds must follow the fixed stage order")
    raw_shape = tuple(context.f3.result.volume_shape)
    if len(raw_shape) != 3 or any(
        isinstance(size, bool) or not isinstance(size, Integral) or int(size) <= 0
        for size in raw_shape
    ):
        raise ValueError("compact F3 shape must contain three positive integers")
    shape = tuple(int(size) for size in raw_shape)
    selections = _selected_groups(context, shape)
    storage_dtype = np.dtype(context.f3.result.storage_dtype).str
    if storage_dtype != context.amplitude.storage_dtype:
        raise ValueError("amplitude and stage storage dtypes must match")

    amplitude_sections = _read_selected_sections(
        context.amplitude.resolved_path,
        shape=shape,
        storage_dtype=storage_dtype,
        selections=context.selected_sections,
        label="amplitude",
    )
    amplitude_limits = {
        group: _percentile_limit(values, AMPLITUDE_PERCENTILE, f"{group} amplitude")
        for group, values in amplitude_sections.items()
    }

    prepared = []
    for source, thresholds in zip(
        context.stage_sources,
        context.ridge_threshold_contract.stages,
        strict=True,
    ):
        reference_source_threshold = _finite_number(
            thresholds.public_reference_threshold, f"public {source.stage} threshold"
        )
        candidate_source_threshold = _finite_number(
            thresholds.q_qual_threshold, f"Q-QUAL {source.stage} threshold"
        )
        reference_display_threshold = _display_threshold(reference_source_threshold)
        candidate_display_threshold = _display_threshold(candidate_source_threshold)
        references = _read_selected_sections(
            source.public_reference_path,
            shape=shape,
            storage_dtype=storage_dtype,
            selections=context.selected_sections,
            label=f"public {source.stage} reference",
        )
        candidates = _read_selected_sections(
            source.candidate_path,
            shape=shape,
            storage_dtype=storage_dtype,
            selections=context.selected_sections,
            label=f"Q-QUAL {source.stage} candidate",
        )
        maxima = tuple(
            _source_metric_max(context, source.stage, metric) for metric in _SCALE_METRICS
        )
        overlay_vmax = max(maxima)
        if overlay_vmax < 0.0:
            raise ValueError(f"{source.stage} overlay maximum must be non-negative")
        for section_group, axis in SECTION_GROUPS:
            differences = tuple(
                candidate - reference
                for reference, candidate in zip(
                    references[section_group], candidates[section_group], strict=True
                )
            )
            difference_limit = _percentile_limit(
                differences,
                DIFFERENCE_PERCENTILE,
                f"{source.stage}/{section_group} difference",
            )
            prepared.append(
                (
                    source,
                    thresholds,
                    section_group,
                    axis,
                    references[section_group],
                    candidates[section_group],
                    differences,
                    difference_limit,
                    reference_display_threshold,
                    candidate_display_threshold,
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
        section_group,
        axis,
        references,
        candidates,
        differences,
        difference_limit,
        reference_display_threshold,
        candidate_display_threshold,
        overlay_vmax,
    ) in prepared:
        group_selections = selections[section_group]
        indices = tuple(item.index for item in group_selections)
        figure_id = f"f3_{source.stage}_{section_group}"
        png_relative = f"figures/{figure_id}.png"
        csv_relative = f"figure_data/{figure_id}.csv"
        rows = _figure_rows(
            context,
            figure_id=figure_id,
            source=source,
            thresholds=thresholds,
            selections=group_selections,
            amplitude_limit=amplitude_limits[section_group],
            overlay_vmax=overlay_vmax,
            difference_limit=difference_limit,
        )
        _write_figure_data(output_root / csv_relative, rows)
        _plot_atlas(
            plt,
            output_root / png_relative,
            stage=source.stage,
            section_group=section_group,
            axis=axis,
            indices=indices,
            amplitudes=amplitude_sections[section_group],
            references=references,
            candidates=candidates,
            differences=differences,
            amplitude_limit=amplitude_limits[section_group],
            reference_display_threshold=reference_display_threshold,
            candidate_display_threshold=candidate_display_threshold,
            overlay_vmax=overlay_vmax,
            difference_limit=difference_limit,
        )
        records.append(
            {
                "figure_id": figure_id,
                "relative_path": png_relative,
                "figure_data_csv": csv_relative,
                "stage": source.stage,
                "section_group": section_group,
                "caption": _caption(source.stage, section_group, axis, indices),
            }
        )
    return tuple(records)


__all__ = ["generate_figures"]
