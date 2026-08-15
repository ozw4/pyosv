"""Deterministic primary summaries for the F3 compact publication."""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path, PurePath
from typing import Any

import numpy as np

from ..f3d_mode_comparison.artifacts import canonical_json_bytes
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
    EXPERIMENT_SCHEMA,
    IMAGE_INTERPOLATION,
    PUBLIC_REFERENCE_LABEL,
    SECTION_GROUPS,
    SECTION_SELECTION_POLICY,
    SECTIONS_PER_AXIS,
    STAGE_ORDER,
    SUMMARY_HEADER,
)
from .models import CompactSourceContext

_SUMMARY_METRICS = (
    ("normalized_correlation", "all", "normalized_correlation"),
    ("mean_absolute_difference", "all", "mean_absolute_difference"),
    ("nonzero_fraction_ratio", "all", "nonzero_fraction_ratio"),
    ("buffered_f1", "positive_p99_radius2", "buffered_f1"),
    (
        "candidate_to_reference_p95_voxel",
        "positive_p99_distance",
        "candidate_to_reference_p95",
    ),
    (
        "reference_to_candidate_p95_voxel",
        "positive_p99_distance",
        "reference_to_candidate_p95",
    ),
)
_DISTANCE_FIELDS = {
    "candidate_to_reference_p95_voxel",
    "reference_to_candidate_p95_voxel",
}
_STAGE_SEMANTICS = {
    "ft": "quality scanner output in Q-QUAL lineage",
    "fv": "quality scanner voting output in Q-QUAL lineage",
    "fvt": "Q-QUAL thinned voting output",
}
_EXPERIMENT_FIELDS = (
    "schema",
    "source",
    "dataset",
    "display",
    "stages",
    "sections",
    "ridge_thresholds",
    "visualization",
)
_REGISTRY = {
    (definition.stage, definition.selection, definition.metric): definition
    for definition in METRIC_REGISTRY
}


def _source_metric(
    rows: Sequence[Any],
    *,
    stage: str,
    selection: str,
    metric: str,
) -> float | None:
    matches = tuple(
        row
        for row in rows
        if (
            getattr(row, "cell_label", None),
            getattr(row, "stage", None),
            getattr(row, "selection", None),
            getattr(row, "metric", None),
        )
        == (DISPLAY_CELL, stage, selection, metric)
    )
    if len(matches) != 1:
        raise ValueError(
            "F3 source must contain exactly one metric row for "
            f"{(DISPLAY_CELL, stage, selection, metric)!r}; found {len(matches)}"
        )
    definition = _REGISTRY.get((stage, selection, metric))
    if definition is None:
        raise ValueError(f"F3 metric registry has no definition for {(stage, selection, metric)!r}")
    row = matches[0]
    if (getattr(row, "unit", None), getattr(row, "direction", None)) != (
        definition.unit,
        definition.direction,
    ):
        raise ValueError(
            f"F3 source metric semantics do not match the registry for "
            f"{(stage, selection, metric)!r}"
        )
    value = getattr(row, "value", None)
    if value is None:
        if not definition.nullable:
            raise ValueError(f"F3 source metric {(stage, selection, metric)!r} is not nullable")
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"F3 source metric {(stage, selection, metric)!r} must be numeric")
    return float(value)


def build_summary_rows(context: CompactSourceContext) -> tuple[Mapping[str, object], ...]:
    """Select the fixed Q-QUAL summary metrics from validated source evidence."""

    stages = {source.stage: source for source in context.stage_sources}
    if tuple(source.stage for source in context.stage_sources) != STAGE_ORDER:
        raise ValueError("compact stage sources must follow the fixed stage order")
    metric_rows = context.f3.result.metric_rows
    output: list[Mapping[str, object]] = []
    for stage in STAGE_ORDER:
        source = stages[stage]
        row: dict[str, object] = {
            "stage": stage,
            "public_reference_file": source.public_reference_filename,
            "q_qual_stage_fingerprint": source.candidate_fingerprint,
        }
        for field, selection, metric in _SUMMARY_METRICS:
            row[field] = _source_metric(
                metric_rows,
                stage=stage,
                selection=selection,
                metric=metric,
            )
        output.append(row)
    return tuple(output)


def _finite_csv_value(value: object, field: str) -> str:
    if value is None:
        if field not in _DISTANCE_FIELDS:
            raise ValueError(f"summary field {field!r} cannot be null")
        return ""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"summary field {field!r} must be numeric or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"summary field {field!r} must be finite")
    return repr(number)


def summary_csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    """Serialize fixed-header summary rows as deterministic UTF-8 CSV."""

    records = tuple(rows)
    for row in records:
        if not isinstance(row, Mapping) or set(row) != set(SUMMARY_HEADER):
            raise ValueError("summary row fields must match the fixed CSV header")
    if tuple(row.get("stage") for row in records) != STAGE_ORDER:
        raise ValueError("summary rows must follow the fixed stage order")
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(SUMMARY_HEADER)
    metric_fields = set(SUMMARY_HEADER[3:])
    for row in records:
        writer.writerow(
            _finite_csv_value(row[field], field) if field in metric_fields else str(row[field])
            for field in SUMMARY_HEADER
        )
    return stream.getvalue().encode("utf-8")


def _basename(filename: object, role: str) -> str:
    if (
        not isinstance(filename, str)
        or not filename
        or PurePath(filename).name != filename
        or Path(filename).is_absolute()
    ):
        raise ValueError(f"dataset role {role!r} must resolve to a basename")
    return filename


def _official_file_identities(context: CompactSourceContext) -> list[dict[str, object]]:
    identity = context.f3.dataset_identity
    spec = context.f3.dataset_spec
    if identity.get("dataset_id") != spec.dataset_id:
        raise ValueError("F3 dataset identity does not match its dataset contract")
    source_files = identity.get("files")
    if not isinstance(source_files, list):
        raise ValueError("F3 dataset identity files must be a list")
    files: list[dict[str, object]] = []
    for role in spec.roles:
        matches = tuple(
            item for item in source_files if isinstance(item, Mapping) and item.get("role") == role
        )
        if len(matches) != 1:
            raise ValueError(f"F3 dataset identity must contain exactly one {role!r} file")
        source = matches[0]
        if (
            source.get("shape") != list(spec.shape)
            or np.dtype(source.get("storage_dtype")).str != spec.storage_dtype
            or source.get("size") != spec.expected_bytes
        ):
            raise ValueError(f"F3 dataset file identity for {role!r} has an invalid layout")
        files.append(
            {
                "role": role,
                "filename": _basename(spec.filename_for(role), role),
                "shape": list(spec.shape),
                "storage_dtype": spec.storage_dtype,
                "size": source.get("size"),
                "sha256": source.get("sha256"),
            }
        )
    amplitude = context.amplitude
    files.append(
        {
            "role": amplitude.role,
            "filename": _basename(amplitude.filename, amplitude.role),
            "shape": list(amplitude.shape),
            "storage_dtype": amplitude.storage_dtype,
            "size": amplitude.size,
            "sha256": amplitude.sha256,
        }
    )
    files.sort(key=lambda item: str(item["role"]))
    return files


def build_experiment(context: CompactSourceContext) -> Mapping[str, object]:
    """Build the path-independent compact experiment snapshot."""

    if tuple(source.stage for source in context.stage_sources) != STAGE_ORDER:
        raise ValueError("compact stage sources must follow the fixed stage order")
    if tuple(item.stage for item in context.ridge_threshold_contract.stages) != STAGE_ORDER:
        raise ValueError("compact ridge thresholds must follow the fixed stage order")
    expected_sections = tuple(
        (section_group, axis, bin_index)
        for section_group, axis in SECTION_GROUPS
        for bin_index in range(SECTIONS_PER_AXIS)
    )
    actual_sections = tuple(
        (item.section_group, item.axis, item.bin_index) for item in context.selected_sections
    )
    if actual_sections != expected_sections or any(
        item.policy != SECTION_SELECTION_POLICY for item in context.selected_sections
    ):
        raise ValueError("compact selected sections must follow the fixed section contract")
    stage_sources = context.stage_sources
    ridge_thresholds = context.ridge_threshold_contract.stages
    spec = context.f3.dataset_spec
    return {
        "schema": EXPERIMENT_SCHEMA,
        "source": {"f3_completion_sha256": context.f3.completion_sha256},
        "dataset": {
            "dataset_id": spec.dataset_id,
            "shape": list(spec.shape),
            "storage_dtype": spec.storage_dtype,
            "files": _official_file_identities(context),
        },
        "display": {
            "public_reference_label": PUBLIC_REFERENCE_LABEL,
            "candidate_label": DISPLAY_CELL,
            "stage_order": list(STAGE_ORDER),
        },
        "stages": [
            {
                "stage": source.stage,
                "public_reference_role": source.public_reference_role,
                "public_reference_filename": source.public_reference_filename,
                "q_qual_stage_kind": source.candidate_source_kind,
                "q_qual_stage_fingerprint": source.candidate_fingerprint,
                "q_qual_stage_filename": source.candidate_filename,
                "q_qual_stage_semantics": _STAGE_SEMANTICS[source.stage],
            }
            for source in stage_sources
        ],
        "sections": {
            "selection_policy": SECTION_SELECTION_POLICY,
            "sections_per_axis": SECTIONS_PER_AXIS,
            "items": [
                {
                    "section_group": item.section_group,
                    "axis": item.axis,
                    "bin_index": item.bin_index,
                    "index": item.index,
                    "ridge_count_score": item.ridge_count_score,
                }
                for item in context.selected_sections
            ],
        },
        "ridge_thresholds": [
            {
                "stage": thresholds.stage,
                "public_reference_threshold": thresholds.public_reference_threshold,
                "q_qual_candidate_threshold": thresholds.q_qual_threshold,
            }
            for thresholds in ridge_thresholds
        ],
        "visualization": {
            "amplitude_role": context.amplitude.role,
            "amplitude_filename": context.amplitude.filename,
            "amplitude_percentile": AMPLITUDE_PERCENTILE,
            "attribute_colormap": ATTRIBUTE_COLORMAP,
            "attribute_display_threshold_ratio": ATTRIBUTE_DISPLAY_THRESHOLD_RATIO,
            "attribute_alpha_min": ATTRIBUTE_ALPHA_MIN,
            "attribute_alpha_max": ATTRIBUTE_ALPHA_MAX,
            "attribute_alpha_gamma": ATTRIBUTE_ALPHA_GAMMA,
            "attribute_halo_enabled": ATTRIBUTE_HALO_ENABLED,
            "attribute_halo_radius_pixels": ATTRIBUTE_HALO_RADIUS_PIXELS,
            "attribute_halo_alpha": ATTRIBUTE_HALO_ALPHA,
            "attribute_halo_structure": ATTRIBUTE_HALO_STRUCTURE,
            "image_interpolation": IMAGE_INTERPOLATION,
            "difference_colormap": DIFFERENCE_COLORMAP,
            "difference_percentile": DIFFERENCE_PERCENTILE,
        },
    }


def experiment_json_bytes(
    experiment: Mapping[str, object],
    *,
    pretty: bool = False,
) -> bytes:
    """Serialize a fixed-schema experiment as deterministic finite JSON."""

    if not isinstance(pretty, bool):
        raise ValueError("pretty must be bool")
    if not isinstance(experiment, Mapping) or tuple(experiment) != _EXPERIMENT_FIELDS:
        raise ValueError("experiment fields must match the fixed top-level schema")
    compact = canonical_json_bytes(experiment)
    if not pretty:
        return compact
    normalized = json.loads(compact)
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


__all__ = [
    "build_experiment",
    "build_summary_rows",
    "experiment_json_bytes",
    "summary_csv_bytes",
]
