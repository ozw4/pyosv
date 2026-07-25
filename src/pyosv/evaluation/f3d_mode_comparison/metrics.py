"""Full-volume reference-agreement metrics for the canonical F3 comparison."""

from __future__ import annotations

import json
import math
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Literal

import numpy as np

from pyosv.metrics import (
    buffered_ridge_overlap,
    sparse_ridge_distance_metrics,
    top_percentile_overlap,
)

from .artifacts import (
    F3StageCorruptionError,
    F3WorkspaceMismatchError,
    canonical_fingerprint,
    validate_stage,
)
from .data import F3VolumeSource
from .runner import F3CellReference

F3_METRIC_SCHEMA_VERSION = 1
F3_METRIC_ROW_FIELDS = (
    "schema_version",
    "dataset_id",
    "cell_label",
    "scanner_backend",
    "workflow_mode",
    "stage",
    "region",
    "selection",
    "reference_file",
    "metric",
    "value",
    "unit",
    "direction",
    "contrast_eligible",
)
F3_REFERENCE_STAGE_FILES = {
    "ft": "fl.dat",
    "fv": "fv.dat",
    "fvt": "fvt.dat",
}
F3_REFERENCE_STAGE_ROLES = {
    "ft": "reference_fault_likelihood",
    "fv": "reference_fault_votes",
    "fvt": "reference_thinned_fault_votes",
}
F3_PERCENTILES = (95.0, 99.0, 99.5)
F3_BUFFERED_PERCENTILE = 99.0
F3_BUFFER_RADIUS = 2.0

MetricDirection = Literal["higher", "lower", "neutral"]

_CELL_AXES = {
    "RL-REF": ("reference-like", "reference"),
    "RL-QUAL": ("reference-like", "quality"),
    "Q-REF": ("quality", "reference"),
    "Q-QUAL": ("quality", "quality"),
}
_CELL_ORDER = tuple(_CELL_AXES)
_SHA256_LENGTH = 64
_DAT_DTYPE = np.dtype(">f4")
_RUN_COMPUTATION_FIELDS = (
    "artifact_schema_version",
    "stage_contract_version",
    "fingerprint_contract_version",
    "plan",
    "dataset_identity",
    "implementation_identity",
)
_STAGE_COMPUTATION_FIELDS = (
    "artifact_schema_version",
    "stage_contract_version",
    "kind",
    "run_fingerprint",
    "parent_fingerprints",
    "input_fingerprints",
    "resolved_settings",
    "artifact_schema",
)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _shape(value: Any) -> tuple[int, int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(
            isinstance(size, bool) or not isinstance(size, Integral) or int(size) <= 0
            for size in value
        )
    ):
        raise ValueError("shape must contain exactly three positive integers")
    return tuple(int(size) for size in value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _direction(value: Any) -> MetricDirection:
    if value not in {"higher", "lower", "neutral"}:
        raise ValueError("direction must be 'higher', 'lower', or 'neutral'")
    return value


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validate_cell_axes(cell_label: str, scanner_backend: str, workflow_mode: str) -> None:
    try:
        expected = _CELL_AXES[cell_label]
    except KeyError as error:
        raise ValueError(f"unknown cell_label: {cell_label!r}") from error
    if (scanner_backend, workflow_mode) != expected:
        raise ValueError("cell_label is inconsistent with scanner_backend/workflow_mode")


def _unique_pairs(
    pairs: tuple[tuple[str, Any], ...],
    name: str,
    value_type: type,
) -> None:
    keys = []
    for key, value in pairs:
        _nonempty_string(key, f"{name} key")
        if value_type is int:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} values must be non-negative integers")
        else:
            _finite_number(value, f"{name} value")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} keys must be unique")


def _percentile_selection(percentile: float) -> str:
    return f"positive_p{str(percentile).replace('.', '_').removesuffix('_0')}"


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """One immutable entry in the F3 metric registry."""

    stage: str
    selection: str
    metric: str
    unit: str
    direction: MetricDirection
    contrast_eligible: bool = True
    nullable: bool = False

    def __post_init__(self) -> None:
        for name in ("stage", "selection", "metric", "unit"):
            _nonempty_string(getattr(self, name), name)
        _direction(self.direction)
        if not isinstance(self.contrast_eligible, bool):
            raise ValueError("contrast_eligible must be bool")
        if not isinstance(self.nullable, bool):
            raise ValueError("nullable must be bool")


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One canonical long-format F3 scalar metric."""

    schema_version: int
    dataset_id: str
    cell_label: str
    scanner_backend: str
    workflow_mode: str
    stage: str
    region: str
    selection: str
    reference_file: str | None
    metric: str
    value: float | None
    unit: str
    direction: MetricDirection
    contrast_eligible: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, Integral)
            or int(self.schema_version) != F3_METRIC_SCHEMA_VERSION
        ):
            raise ValueError(f"schema_version must be {F3_METRIC_SCHEMA_VERSION}")
        object.__setattr__(self, "schema_version", int(self.schema_version))
        for name in (
            "dataset_id",
            "cell_label",
            "scanner_backend",
            "workflow_mode",
            "stage",
            "region",
            "selection",
            "metric",
            "unit",
        ):
            _nonempty_string(getattr(self, name), name)
        _validate_cell_axes(self.cell_label, self.scanner_backend, self.workflow_mode)
        if self.region != "full":
            raise ValueError("region must be 'full'")
        expected_file = F3_REFERENCE_STAGE_FILES.get(self.stage)
        if expected_file is None:
            if self.stage != "skin":
                raise ValueError(f"unknown metric stage: {self.stage!r}")
            if self.reference_file is not None:
                raise ValueError("skin rows must not have a reference_file")
        elif self.reference_file != expected_file:
            raise ValueError(f"stage {self.stage!r} requires reference_file {expected_file!r}")
        if self.value is not None:
            object.__setattr__(self, "value", _finite_number(self.value, "value"))
        _direction(self.direction)
        if not isinstance(self.contrast_eligible, bool):
            raise ValueError("contrast_eligible must be bool")

    @property
    def identity(self) -> tuple[str, ...]:
        """Return the unique row identity, excluding the scalar value."""

        return (
            self.dataset_id,
            self.cell_label,
            self.stage,
            self.region,
            self.selection,
            self.reference_file or "",
            self.metric,
        )

    @property
    def csv_value(self) -> float | str:
        """Return the fixed CSV representation; nullable distances use an empty field."""

        return "" if self.value is None else self.value

    def as_dict(self, *, csv: bool = False) -> dict[str, Any]:
        """Return a mapping in the canonical field order."""

        output = asdict(self)
        if csv and self.value is None:
            output["value"] = ""
        return output


@dataclass(frozen=True, slots=True)
class MetricEvidence:
    """Recomputation evidence for one stage/selection metric block."""

    schema_version: int
    dataset_id: str
    cell_label: str
    stage: str
    region: str
    selection: str
    reference_file: str | None
    source_stage_fingerprint: str
    reference_sha256: str | None
    shape: tuple[int, int, int]
    thresholds: tuple[tuple[str, float], ...] = ()
    counts: tuple[tuple[str, int], ...] = ()
    accumulators: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != F3_METRIC_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {F3_METRIC_SCHEMA_VERSION}")
        for name in ("dataset_id", "cell_label", "stage", "region", "selection"):
            _nonempty_string(getattr(self, name), name)
        if self.cell_label not in _CELL_AXES:
            raise ValueError(f"unknown cell_label: {self.cell_label!r}")
        if self.region != "full":
            raise ValueError("region must be 'full'")
        expected_file = F3_REFERENCE_STAGE_FILES.get(self.stage)
        if expected_file is None:
            if self.stage != "skin" or self.reference_file is not None:
                raise ValueError("only skin evidence may omit reference_file")
            if self.reference_sha256 is not None:
                raise ValueError("skin evidence must not have reference_sha256")
        elif self.reference_file != expected_file:
            raise ValueError("reference_file does not match the stage mapping")
        _sha256(self.source_stage_fingerprint, "source_stage_fingerprint")
        if self.reference_sha256 is not None:
            _sha256(self.reference_sha256, "reference_sha256")
        _shape(self.shape)
        _unique_pairs(self.thresholds, "thresholds", float)
        _unique_pairs(self.counts, "counts", int)
        _unique_pairs(self.accumulators, "accumulators", float)

    @property
    def identity(self) -> tuple[str, ...]:
        return (
            self.dataset_id,
            self.cell_label,
            self.stage,
            self.region,
            self.selection,
            self.reference_file or "",
        )

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-safe evidence."""

        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "cell_label": self.cell_label,
            "stage": self.stage,
            "region": self.region,
            "selection": self.selection,
            "reference_file": self.reference_file,
            "source_stage_fingerprint": self.source_stage_fingerprint,
            "reference_sha256": self.reference_sha256,
            "shape": list(self.shape),
            "thresholds": dict(self.thresholds),
            "counts": dict(self.counts),
            "accumulators": dict(self.accumulators),
        }


@dataclass(frozen=True, slots=True)
class ContrastDefinition:
    """One fixed diagnostic linear contrast over the four F3 cells."""

    name: str
    coefficients: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.name, "name")
        if not self.coefficients:
            raise ValueError("coefficients must not be empty")
        cells = tuple(cell for cell, _ in self.coefficients)
        if len(cells) != len(set(cells)):
            raise ValueError("contrast cells must be unique")
        if set(cells) - set(_CELL_AXES):
            raise ValueError("contrast contains an unknown cell")
        normalized = []
        for cell, coefficient in self.coefficients:
            value = _finite_number(coefficient, "coefficient")
            if value == 0.0:
                raise ValueError("contrast coefficients must be nonzero")
            normalized.append((cell, value))
        object.__setattr__(self, "coefficients", tuple(normalized))

    @property
    def component_cells(self) -> tuple[str, ...]:
        return tuple(cell for cell, _ in self.coefficients)


CONTRAST_DEFINITIONS = (
    ContrastDefinition("scanner_effect_ref", (("Q-REF", 1.0), ("RL-REF", -1.0))),
    ContrastDefinition("scanner_effect_qual", (("Q-QUAL", 1.0), ("RL-QUAL", -1.0))),
    ContrastDefinition("workflow_effect_rl", (("RL-QUAL", 1.0), ("RL-REF", -1.0))),
    ContrastDefinition("workflow_effect_q", (("Q-QUAL", 1.0), ("Q-REF", -1.0))),
    ContrastDefinition("end_to_end_delta", (("Q-QUAL", 1.0), ("RL-REF", -1.0))),
    ContrastDefinition(
        "scanner_main_effect",
        (("Q-REF", 0.5), ("RL-REF", -0.5), ("Q-QUAL", 0.5), ("RL-QUAL", -0.5)),
    ),
    ContrastDefinition(
        "workflow_main_effect",
        (("RL-QUAL", 0.5), ("RL-REF", -0.5), ("Q-QUAL", 0.5), ("Q-REF", -0.5)),
    ),
    ContrastDefinition(
        "scanner_workflow_interaction",
        (("Q-QUAL", 1.0), ("Q-REF", -1.0), ("RL-QUAL", -1.0), ("RL-REF", 1.0)),
    ),
)


@dataclass(frozen=True, slots=True)
class ContrastRow:
    """One scalar diagnostic contrast over exactly paired metric rows."""

    schema_version: int
    dataset_id: str
    contrast_name: str
    stage: str
    region: str
    selection: str
    reference_file: str | None
    metric: str
    unit: str
    direction: MetricDirection
    component_cells: tuple[str, ...]
    raw_value: float
    improvement_value: float | None

    def __post_init__(self) -> None:
        if self.schema_version != F3_METRIC_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {F3_METRIC_SCHEMA_VERSION}")
        definition = _contrast_definition(self.contrast_name)
        if self.component_cells != definition.component_cells:
            raise ValueError("component_cells must match the contrast definition")
        for name in ("dataset_id", "stage", "region", "selection", "metric", "unit"):
            _nonempty_string(getattr(self, name), name)
        if self.region != "full":
            raise ValueError("region must be 'full'")
        expected_file = F3_REFERENCE_STAGE_FILES.get(self.stage)
        if expected_file is None:
            if self.stage != "skin" or self.reference_file is not None:
                raise ValueError("contrast stage/reference_file mapping is invalid")
        elif self.reference_file != expected_file:
            raise ValueError("contrast reference_file does not match its stage")
        raw = _finite_number(self.raw_value, "raw_value")
        object.__setattr__(self, "raw_value", raw)
        expected = _improvement(raw, self.direction)
        if expected is None:
            if self.improvement_value is not None:
                raise ValueError("neutral contrasts must not have improvement_value")
        elif self.improvement_value != expected:
            raise ValueError("improvement_value is inconsistent with direction")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VoxelwiseContrastSummary:
    """Streaming summary of one contrast volume; the volume itself is not retained."""

    schema_version: int
    dataset_id: str
    contrast_name: str
    stage: str
    region: str
    shape: tuple[int, int, int]
    registration_id: str
    component_cells: tuple[str, ...]
    component_stage_fingerprints: tuple[tuple[str, str], ...]
    mean: float
    std: float
    mean_absolute: float
    p95_absolute: float
    max_absolute: float
    epsilon: float
    epsilon_nonzero_fraction: float

    def __post_init__(self) -> None:
        if self.schema_version != F3_METRIC_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {F3_METRIC_SCHEMA_VERSION}")
        definition = _contrast_definition(self.contrast_name)
        if self.component_cells != definition.component_cells:
            raise ValueError("component_cells must match the contrast definition")
        for name in ("dataset_id", "stage", "region", "registration_id"):
            _nonempty_string(getattr(self, name), name)
        if self.region != "full":
            raise ValueError("region must be 'full'")
        _shape(self.shape)
        fingerprints = dict(self.component_stage_fingerprints)
        if tuple(fingerprints) != self.component_cells:
            raise ValueError("component_stage_fingerprints must follow component_cells")
        for fingerprint in fingerprints.values():
            _sha256(fingerprint, "component stage fingerprint")
        for name in ("mean", "std", "mean_absolute", "p95_absolute", "max_absolute"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name))
        epsilon = _finite_number(self.epsilon, "epsilon")
        if epsilon < 0.0:
            raise ValueError("epsilon must be non-negative")
        object.__setattr__(self, "epsilon", epsilon)
        fraction = _finite_number(self.epsilon_nonzero_fraction, "epsilon_nonzero_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("epsilon_nonzero_fraction must be in [0, 1]")
        object.__setattr__(self, "epsilon_nonzero_fraction", fraction)

    def as_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["shape"] = list(self.shape)
        output["component_stage_fingerprints"] = dict(self.component_stage_fingerprints)
        return output


@dataclass(frozen=True, slots=True)
class MetricExtraction:
    """Complete stable scalar and voxelwise evidence for one F3 evaluation unit."""

    metric_rows: tuple[MetricRow, ...]
    metric_evidence: tuple[MetricEvidence, ...]
    contrast_rows: tuple[ContrastRow, ...]
    voxelwise_contrasts: tuple[VoxelwiseContrastSummary, ...]


def _build_metric_registry() -> tuple[MetricDefinition, ...]:
    definitions: list[MetricDefinition] = []
    all_metrics = (
        ("voxel_count", "count", "neutral", False),
        ("candidate_finite_count", "count", "neutral", True),
        ("reference_finite_count", "count", "neutral", False),
        ("candidate_finite_fraction", "fraction", "neutral", True),
        ("reference_finite_fraction", "fraction", "neutral", False),
        ("candidate_min", "value", "neutral", True),
        ("candidate_max", "value", "neutral", True),
        ("candidate_mean", "value", "neutral", True),
        ("candidate_std", "value", "neutral", True),
        ("reference_min", "value", "neutral", False),
        ("reference_max", "value", "neutral", False),
        ("reference_mean", "value", "neutral", False),
        ("reference_std", "value", "neutral", False),
        ("candidate_nonzero_count", "count", "neutral", True),
        ("reference_nonzero_count", "count", "neutral", False),
        ("candidate_nonzero_fraction", "fraction", "neutral", True),
        ("reference_nonzero_fraction", "fraction", "neutral", False),
        ("nonzero_fraction_ratio", "ratio", "neutral", True),
        ("normalized_correlation", "correlation", "higher", True),
        ("mean_absolute_difference", "value", "lower", True),
        ("root_mean_square_difference", "value", "lower", True),
        ("absolute_difference_mean", "value", "lower", True),
        ("absolute_difference_median", "value", "lower", True),
        ("absolute_difference_p90", "value", "lower", True),
        ("absolute_difference_p95", "value", "lower", True),
        ("absolute_difference_p99", "value", "lower", True),
        ("absolute_difference_max", "value", "lower", True),
    )
    top_metrics = (
        ("reference_count", "count", "neutral", False),
        ("candidate_count", "count", "neutral", True),
        ("intersection_count", "count", "neutral", True),
        ("union_count", "count", "neutral", True),
        ("precision", "fraction", "higher", True),
        ("recall", "fraction", "higher", True),
        ("f1", "fraction", "higher", True),
        ("jaccard", "fraction", "higher", True),
    )
    buffered_metrics = (
        *top_metrics,
        ("candidate_in_reference_buffer_count", "count", "neutral", True),
        ("reference_in_candidate_buffer_count", "count", "neutral", True),
        ("buffered_precision", "fraction", "higher", True),
        ("buffered_recall", "fraction", "higher", True),
        ("buffered_f1", "fraction", "higher", True),
    )
    distance_metrics = (
        ("reference_count", "count", "neutral", False, False),
        ("candidate_count", "count", "neutral", True, False),
        ("candidate_to_reference_mean", "voxel", "lower", True, True),
        ("candidate_to_reference_median", "voxel", "lower", True, True),
        ("candidate_to_reference_p90", "voxel", "lower", True, True),
        ("candidate_to_reference_p95", "voxel", "lower", True, True),
        ("reference_to_candidate_mean", "voxel", "lower", True, True),
        ("reference_to_candidate_median", "voxel", "lower", True, True),
        ("reference_to_candidate_p90", "voxel", "lower", True, True),
        ("reference_to_candidate_p95", "voxel", "lower", True, True),
    )
    for stage in F3_REFERENCE_STAGE_FILES:
        definitions.extend(
            MetricDefinition(stage, "all", metric, unit, direction, eligible)
            for metric, unit, direction, eligible in all_metrics
        )
        for percentile in F3_PERCENTILES:
            selection = _percentile_selection(percentile)
            definitions.extend(
                MetricDefinition(stage, selection, metric, unit, direction, eligible)
                for metric, unit, direction, eligible in top_metrics
            )
        definitions.extend(
            MetricDefinition(stage, "positive_p99_radius2", metric, unit, direction, eligible)
            for metric, unit, direction, eligible in buffered_metrics
        )
        definitions.extend(
            MetricDefinition(
                stage,
                "positive_p99_distance",
                metric,
                unit,
                direction,
                eligible,
                nullable,
            )
            for metric, unit, direction, eligible, nullable in distance_metrics
        )
    for metric, unit in (
        ("skin_count", "count"),
        ("cell_count", "count"),
        ("unique_cell_count", "count"),
        ("duplicate_cell_count", "count"),
        ("largest_skin_fraction", "fraction"),
        ("small_skin_cell_fraction", "fraction"),
        ("accepted_skin_count", "count"),
        ("fallback_enabled", "flag"),
        ("fallback_used", "flag"),
        ("fallback_skin_count", "count"),
        ("fallback_cell_count", "count"),
    ):
        definitions.append(MetricDefinition("skin", "descriptive", metric, unit, "neutral"))
    identities = {(item.stage, item.selection, item.metric) for item in definitions}
    if len(identities) != len(definitions):
        raise RuntimeError("F3 metric registry contains duplicate identities")
    return tuple(definitions)


METRIC_REGISTRY = _build_metric_registry()
_DEFINITION_BY_IDENTITY = {
    (definition.stage, definition.selection, definition.metric): definition
    for definition in METRIC_REGISTRY
}
_METRIC_ORDER = {identity: index for index, identity in enumerate(_DEFINITION_BY_IDENTITY)}


def compute_reference_metric_rows(
    *,
    dataset_id: str,
    cell_label: str,
    scanner_backend: str,
    workflow_mode: str,
    stage: str,
    reference_file: str,
    candidate: np.ndarray,
    reference: np.ndarray,
    source_stage_fingerprint: str,
    reference_sha256: str,
    slab_depth: int = 8,
    temporary_directory: str | Path | None = None,
) -> tuple[tuple[MetricRow, ...], tuple[MetricEvidence, ...]]:
    """Compute one candidate/reference stage pair without partial-row failure."""

    _nonempty_string(dataset_id, "dataset_id")
    _validate_cell_axes(cell_label, scanner_backend, workflow_mode)
    expected_file = F3_REFERENCE_STAGE_FILES.get(stage)
    if expected_file is None:
        raise ValueError(f"unknown reference stage: {stage!r}")
    if reference_file != expected_file:
        raise ValueError(
            f"stage {stage!r} requires reference_file {expected_file!r}, got {reference_file!r}"
        )
    _sha256(source_stage_fingerprint, "source_stage_fingerprint")
    _sha256(reference_sha256, "reference_sha256")
    candidate_values, reference_values = _comparable_3d_arrays(candidate, reference)
    depth = _positive_int(slab_depth, "slab_depth")

    values: dict[tuple[str, str], float | int | None] = {}
    evidence: list[MetricEvidence] = []
    with tempfile.TemporaryDirectory(dir=temporary_directory) as temp_dir:
        basic, basic_counts, basic_accumulators = _all_voxel_metrics(
            candidate_values,
            reference_values,
            slab_depth=depth,
            temporary_directory=Path(temp_dir),
        )
        values.update((("all", metric), value) for metric, value in basic.items())
        evidence.append(
            _evidence(
                dataset_id,
                cell_label,
                stage,
                "all",
                reference_file,
                source_stage_fingerprint,
                reference_sha256,
                candidate_values.shape,
                counts=basic_counts,
                accumulators=basic_accumulators,
            )
        )

        for percentile in F3_PERCENTILES:
            selection = _percentile_selection(percentile)
            overlap = top_percentile_overlap(
                reference_values,
                candidate_values,
                percentile,
                positive_only=True,
            )
            exact = _exact_overlap_values(overlap)
            values.update(((selection, metric), value) for metric, value in exact.items())
            evidence.append(
                _evidence(
                    dataset_id,
                    cell_label,
                    stage,
                    selection,
                    reference_file,
                    source_stage_fingerprint,
                    reference_sha256,
                    candidate_values.shape,
                    thresholds=_selection_thresholds(
                        reference_values, candidate_values, percentile
                    ),
                    counts={name: int(exact[name]) for name in _COUNT_METRICS if name in exact},
                )
            )
            del overlap, exact

        buffered = buffered_ridge_overlap(
            reference_values,
            candidate_values,
            percentile=F3_BUFFERED_PERCENTILE,
            radius=F3_BUFFER_RADIUS,
            positive_only=True,
        )
        selection = "positive_p99_radius2"
        for definition in _definitions_for(stage, selection):
            values[(selection, definition.metric)] = buffered[definition.metric]
        evidence.append(
            _evidence(
                dataset_id,
                cell_label,
                stage,
                selection,
                reference_file,
                source_stage_fingerprint,
                reference_sha256,
                candidate_values.shape,
                thresholds={
                    **_selection_thresholds(
                        reference_values, candidate_values, F3_BUFFERED_PERCENTILE
                    ),
                    "radius": F3_BUFFER_RADIUS,
                },
                counts={name: int(buffered[name]) for name in _COUNT_METRICS if name in buffered},
            )
        )
        del buffered

        distance = sparse_ridge_distance_metrics(
            reference_values,
            candidate_values,
            percentile=F3_BUFFERED_PERCENTILE,
            positive_only=True,
        )
        selection = "positive_p99_distance"
        for definition in _definitions_for(stage, selection):
            values[(selection, definition.metric)] = distance[definition.metric]
        evidence.append(
            _evidence(
                dataset_id,
                cell_label,
                stage,
                selection,
                reference_file,
                source_stage_fingerprint,
                reference_sha256,
                candidate_values.shape,
                thresholds=_selection_thresholds(
                    reference_values, candidate_values, F3_BUFFERED_PERCENTILE
                ),
                counts={
                    "reference_count": int(distance["reference_count"]),
                    "candidate_count": int(distance["candidate_count"]),
                },
            )
        )
        del distance

    rows = []
    for definition in (item for item in METRIC_REGISTRY if item.stage == stage):
        value = values[(definition.selection, definition.metric)]
        if value is None and not definition.nullable:
            raise ValueError(f"metric {definition.metric!r} unexpectedly produced None")
        rows.append(
            MetricRow(
                schema_version=F3_METRIC_SCHEMA_VERSION,
                dataset_id=dataset_id,
                cell_label=cell_label,
                scanner_backend=scanner_backend,
                workflow_mode=workflow_mode,
                stage=stage,
                region="full",
                selection=definition.selection,
                reference_file=reference_file,
                metric=definition.metric,
                value=None if value is None else float(value),
                unit=definition.unit,
                direction=definition.direction,
                contrast_eligible=definition.contrast_eligible and value is not None,
            )
        )
    _unique_metric_rows(rows)
    return tuple(rows), tuple(evidence)


def compute_skin_metric_rows(
    *,
    dataset_id: str,
    cell_label: str,
    scanner_backend: str,
    workflow_mode: str,
    report: Mapping[str, Any],
    source_stage_fingerprint: str,
    shape: tuple[int, int, int],
) -> tuple[tuple[MetricRow, ...], tuple[MetricEvidence, ...]]:
    """Extract truthless descriptive skin rows from one validated stage report."""

    _validate_cell_axes(cell_label, scanner_backend, workflow_mode)
    _sha256(source_stage_fingerprint, "source_stage_fingerprint")
    volume_shape = _shape(shape)
    if report.get("fingerprint") != source_stage_fingerprint:
        raise ValueError("skin report source fingerprint mismatch")
    if tuple(report.get("shape", ())) != volume_shape:
        raise ValueError("skin report shape mismatch")
    if report.get("enabled") is not True:
        raise ValueError("skin report must describe an enabled stage")
    topology = _mapping(report.get("topology"), "topology")
    diagnostics = _mapping(report.get("diagnostics"), "diagnostics")
    source_values = {
        "skin_count": topology.get("skin_count"),
        "cell_count": topology.get("cell_count"),
        "unique_cell_count": topology.get("unique_cell_count"),
        "duplicate_cell_count": topology.get("duplicate_cell_count"),
        "largest_skin_fraction": topology.get("largest_skin_fraction"),
        "small_skin_cell_fraction": topology.get("small_skin_cell_fraction"),
        "accepted_skin_count": diagnostics.get("accepted_skin_count"),
        "fallback_enabled": int(bool(diagnostics.get("fallback_enabled"))),
        "fallback_used": int(bool(diagnostics.get("fallback_used"))),
        "fallback_skin_count": diagnostics.get("fallback_skin_count"),
        "fallback_cell_count": diagnostics.get("fallback_cell_count"),
    }
    rows = []
    for definition in _definitions_for("skin", "descriptive"):
        value = _finite_number(source_values[definition.metric], definition.metric)
        if definition.unit == "count" and (value < 0.0 or not value.is_integer()):
            raise ValueError(f"{definition.metric} must be a non-negative count")
        rows.append(
            MetricRow(
                F3_METRIC_SCHEMA_VERSION,
                dataset_id,
                cell_label,
                scanner_backend,
                workflow_mode,
                "skin",
                "full",
                "descriptive",
                None,
                definition.metric,
                value,
                definition.unit,
                definition.direction,
                definition.contrast_eligible,
            )
        )
    evidence = MetricEvidence(
        F3_METRIC_SCHEMA_VERSION,
        dataset_id,
        cell_label,
        "skin",
        "full",
        "descriptive",
        None,
        source_stage_fingerprint,
        None,
        volume_shape,
        counts=tuple(
            (name, int(value))
            for name, value in source_values.items()
            if name.endswith("count") and value is not None
        ),
    )
    return tuple(rows), (evidence,)


def compute_contrast_rows(
    rows: Sequence[MetricRow],
    evidence: Sequence[MetricEvidence] = (),
) -> tuple[ContrastRow, ...]:
    """Compute the canonical eight contrasts from exact metric identities."""

    metric_rows = tuple(rows)
    if any(not isinstance(row, MetricRow) for row in metric_rows):
        raise ValueError("rows must contain only MetricRow values")
    _unique_metric_rows(metric_rows)
    _validate_metric_definitions(metric_rows)
    _validate_shared_row_values(metric_rows)
    evidence_rows = tuple(evidence)
    if evidence_rows:
        _validate_evidence(metric_rows, evidence_rows)
        validate_shared_stage_metrics(metric_rows, evidence_rows)

    all_groups: dict[
        tuple[str, str, str, str, str, str],
        dict[str, MetricRow],
    ] = defaultdict(dict)
    groups: dict[tuple[str, str, str, str, str, str], dict[str, MetricRow]] = defaultdict(dict)
    for row in metric_rows:
        key = (
            row.dataset_id,
            row.stage,
            row.region,
            row.selection,
            row.reference_file or "",
            row.metric,
        )
        all_groups[key][row.cell_label] = row
        if row.contrast_eligible:
            groups[key][row.cell_label] = row

    output = []
    ordered_keys = sorted(
        groups,
        key=lambda key: (
            key[0],
            _METRIC_ORDER[(key[1], key[3], key[5])],
            key[2],
            key[4],
        ),
    )
    complete_keys = []
    for key in ordered_keys:
        by_cell = groups[key]
        if set(by_cell) != set(_CELL_AXES):
            definition = _DEFINITION_BY_IDENTITY[(key[1], key[3], key[5])]
            all_by_cell = all_groups[key]
            if (
                definition.nullable
                and set(all_by_cell) == set(_CELL_AXES)
                and any(row.value is None for row in all_by_cell.values())
            ):
                continue
            missing = set(_CELL_AXES) - set(by_cell)
            raise ValueError(
                "metric contrast is missing required cell(s): " + ",".join(sorted(missing))
            )
        components = tuple(by_cell[cell] for cell in _CELL_ORDER)
        if len({row.unit for row in components}) != 1:
            raise ValueError("metric contrast has mixed units")
        if len({row.direction for row in components}) != 1:
            raise ValueError("metric contrast has mixed directions")
        complete_keys.append(key)
    for definition in CONTRAST_DEFINITIONS:
        for key in complete_keys:
            by_cell = groups[key]
            raw = math.fsum(
                coefficient * by_cell[cell].value
                for cell, coefficient in definition.coefficients
                if by_cell[cell].value is not None
            )
            first = by_cell[definition.component_cells[0]]
            output.append(
                ContrastRow(
                    F3_METRIC_SCHEMA_VERSION,
                    first.dataset_id,
                    definition.name,
                    first.stage,
                    first.region,
                    first.selection,
                    first.reference_file,
                    first.metric,
                    first.unit,
                    first.direction,
                    definition.component_cells,
                    raw,
                    _improvement(raw, first.direction),
                )
            )
    return tuple(output)


def validate_shared_stage_metrics(
    rows: Sequence[MetricRow],
    evidence: Sequence[MetricEvidence],
) -> None:
    """Require workflow-paired ``ft`` and ``fv`` rows and fingerprints to be identical."""

    by_identity = {row.identity: row for row in rows}
    evidence_by_identity = {item.identity: item for item in evidence}
    for stage in ("ft", "fv"):
        for left, right in (("RL-REF", "RL-QUAL"), ("Q-REF", "Q-QUAL")):
            relevant = [row for row in rows if row.stage == stage and row.cell_label == left]
            for left_row in relevant:
                right_identity = (
                    left_row.dataset_id,
                    right,
                    left_row.stage,
                    left_row.region,
                    left_row.selection,
                    left_row.reference_file or "",
                    left_row.metric,
                )
                right_row = by_identity.get(right_identity)
                if right_row is None:
                    raise ValueError("shared stage workflow pair is incomplete")
                if left_row.value != right_row.value:
                    raise ValueError("shared stage workflow metric must be exactly equal")
                left_evidence_identity = left_row.identity[:-1]
                right_evidence_identity = right_identity[:-1]
                left_evidence = evidence_by_identity.get(left_evidence_identity)
                right_evidence = evidence_by_identity.get(right_evidence_identity)
                if left_evidence is None or right_evidence is None:
                    raise ValueError("shared stage evidence is incomplete")
                if (
                    left_evidence.source_stage_fingerprint
                    != right_evidence.source_stage_fingerprint
                ):
                    raise ValueError("shared stage workflow fingerprints must be identical")


def compute_voxelwise_contrast_summaries(
    *,
    dataset_id: str,
    stage: str,
    volumes: Mapping[str, np.ndarray],
    stage_fingerprints: Mapping[str, str],
    registration_id: str,
    epsilon: float = 1.0e-6,
    slab_depth: int = 8,
    temporary_directory: str | Path | None = None,
) -> tuple[VoxelwiseContrastSummary, ...]:
    """Stream exact summaries for all eight 2x2 contrast volumes."""

    _nonempty_string(dataset_id, "dataset_id")
    if stage not in F3_REFERENCE_STAGE_FILES:
        raise ValueError(f"unknown voxelwise contrast stage: {stage!r}")
    _nonempty_string(registration_id, "registration_id")
    epsilon_value = _finite_number(epsilon, "epsilon")
    if epsilon_value < 0.0:
        raise ValueError("epsilon must be non-negative")
    depth = _positive_int(slab_depth, "slab_depth")
    if tuple(volumes) != _CELL_ORDER:
        raise ValueError("volumes must follow canonical cell order")
    if tuple(stage_fingerprints) != _CELL_ORDER:
        raise ValueError("stage_fingerprints must follow canonical cell order")
    arrays = {cell: np.asarray(volumes[cell]) for cell in _CELL_ORDER}
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise ValueError("voxelwise contrast volume shapes must match")
    shape = _shape(next(iter(shapes)))
    for cell, array in arrays.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"voxelwise contrast input {cell} contains nonfinite values")
        _sha256(stage_fingerprints[cell], f"{cell} stage fingerprint")
    if stage in {"ft", "fv"}:
        for left, right in (("RL-REF", "RL-QUAL"), ("Q-REF", "Q-QUAL")):
            if stage_fingerprints[left] != stage_fingerprints[right]:
                raise ValueError("shared workflow stage fingerprints must be identical")
            for start in range(0, shape[0], depth):
                stop = min(shape[0], start + depth)
                if not np.array_equal(
                    arrays[left][start:stop],
                    arrays[right][start:stop],
                ):
                    raise ValueError("shared workflow stage volumes must be exactly equal")

    output = []
    with tempfile.TemporaryDirectory(dir=temporary_directory) as temp_dir:
        for definition in CONTRAST_DEFINITIONS:
            summary = _voxelwise_summary(
                definition,
                arrays,
                shape,
                epsilon_value,
                depth,
                Path(temp_dir),
            )
            output.append(
                VoxelwiseContrastSummary(
                    F3_METRIC_SCHEMA_VERSION,
                    dataset_id,
                    definition.name,
                    stage,
                    "full",
                    shape,
                    registration_id,
                    definition.component_cells,
                    tuple((cell, stage_fingerprints[cell]) for cell in definition.component_cells),
                    **summary,
                )
            )
    return tuple(output)


def extract_f3d_metric_rows(
    volume_source: F3VolumeSource,
    cells: Sequence[F3CellReference],
    *,
    slab_depth: int = 8,
    epsilon: float = 1.0e-6,
    temporary_directory: str | Path | None = None,
) -> MetricExtraction:
    """Extract complete reference, skin, scalar-contrast, and voxelwise evidence."""

    if not isinstance(volume_source, F3VolumeSource):
        raise TypeError("volume_source must be an F3VolumeSource")
    cell_rows = tuple(cells)
    if any(not isinstance(cell, F3CellReference) for cell in cell_rows):
        raise TypeError("cells must contain only F3CellReference values")
    if tuple(cell.label for cell in cell_rows) != _CELL_ORDER:
        raise ValueError("cells must follow canonical F3 cell order")
    dataset_id = volume_source.identity.dataset_id
    shape = volume_source.spec.shape
    workspace_path = _validated_extraction_workspace(volume_source, cell_rows)

    rows: list[MetricRow] = []
    evidence: list[MetricEvidence] = []
    for stage in F3_REFERENCE_STAGE_FILES:
        role = F3_REFERENCE_STAGE_ROLES[stage]
        reference_identity = volume_source.identity.file_for(role)
        reference = volume_source.open_memmap(role)
        try:
            for cell in cell_rows:
                fingerprint = _stage_fingerprint_for(cell, stage)
                candidate_path = _candidate_path(workspace_path, cell, stage)
                _validate_candidate_report(candidate_path.parent, cell, stage, shape)
                candidate = _open_dat(candidate_path, shape)
                try:
                    stage_rows, stage_evidence = compute_reference_metric_rows(
                        dataset_id=dataset_id,
                        cell_label=cell.label,
                        scanner_backend=cell.backend,
                        workflow_mode=cell.workflow,
                        stage=stage,
                        reference_file=F3_REFERENCE_STAGE_FILES[stage],
                        candidate=candidate,
                        reference=reference,
                        source_stage_fingerprint=fingerprint,
                        reference_sha256=reference_identity.sha256,
                        slab_depth=slab_depth,
                        temporary_directory=temporary_directory,
                    )
                    rows.extend(stage_rows)
                    evidence.extend(stage_evidence)
                finally:
                    _close_memmap(candidate)
        finally:
            volume_source._close_memmap(reference)

    for cell in cell_rows:
        if not cell.skinning_enabled:
            continue
        stage_path = workspace_path / "stages" / "skinning" / cell.stages.skinning
        report = _read_json(stage_path / "report.json")
        skin_rows, skin_evidence = compute_skin_metric_rows(
            dataset_id=dataset_id,
            cell_label=cell.label,
            scanner_backend=cell.backend,
            workflow_mode=cell.workflow,
            report=report,
            source_stage_fingerprint=cell.stages.skinning,
            shape=shape,
        )
        rows.extend(skin_rows)
        evidence.extend(skin_evidence)

    cell_order = {label: index for index, label in enumerate(_CELL_ORDER)}
    rows.sort(
        key=lambda row: (
            cell_order[row.cell_label],
            _METRIC_ORDER[(row.stage, row.selection, row.metric)],
        )
    )
    evidence_order = {
        (definition.stage, definition.selection): index
        for index, definition in reversed(tuple(enumerate(METRIC_REGISTRY)))
    }
    evidence.sort(
        key=lambda item: (
            cell_order[item.cell_label],
            evidence_order[(item.stage, item.selection)],
        )
    )
    metric_rows = tuple(rows)
    metric_evidence = tuple(evidence)
    _validate_evidence(metric_rows, metric_evidence)
    validate_shared_stage_metrics(metric_rows, metric_evidence)
    contrasts = compute_contrast_rows(metric_rows, metric_evidence)

    voxelwise = []
    for stage in F3_REFERENCE_STAGE_FILES:
        opened: list[np.memmap] = []
        try:
            volumes: dict[str, np.ndarray] = {}
            fingerprints: dict[str, str] = {}
            for cell in cell_rows:
                array = _open_dat(_candidate_path(workspace_path, cell, stage), shape)
                opened.append(array)
                volumes[cell.label] = array
                fingerprints[cell.label] = _stage_fingerprint_for(cell, stage)
            voxelwise.extend(
                compute_voxelwise_contrast_summaries(
                    dataset_id=dataset_id,
                    stage=stage,
                    volumes=volumes,
                    stage_fingerprints=fingerprints,
                    registration_id=f"{dataset_id}:{shape}",
                    epsilon=epsilon,
                    slab_depth=slab_depth,
                    temporary_directory=temporary_directory,
                )
            )
        finally:
            for array in opened:
                _close_memmap(array)
    return MetricExtraction(metric_rows, metric_evidence, contrasts, tuple(voxelwise))


def extract_f3d_metrics(
    volume_source: F3VolumeSource,
    cells: Sequence[F3CellReference],
    *,
    slab_depth: int = 8,
    epsilon: float = 1.0e-6,
    temporary_directory: str | Path | None = None,
) -> MetricExtraction:
    """Return :func:`extract_f3d_metric_rows` for callers naming the complete result."""

    return extract_f3d_metric_rows(
        volume_source,
        cells,
        slab_depth=slab_depth,
        epsilon=epsilon,
        temporary_directory=temporary_directory,
    )


_COUNT_METRICS = {
    "reference_count",
    "candidate_count",
    "intersection_count",
    "union_count",
    "candidate_in_reference_buffer_count",
    "reference_in_candidate_buffer_count",
}


def _all_voxel_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    slab_depth: int,
    temporary_directory: Path,
) -> tuple[dict[str, float | int], dict[str, int], dict[str, float]]:
    shape = candidate.shape
    size = int(candidate.size)
    absolute_path = temporary_directory / "absolute-difference.float64"
    absolute = np.memmap(absolute_path, dtype=np.float64, mode="w+", shape=(size,))
    candidate_moments = _Moments()
    reference_moments = _Moments()
    paired_moments = _PairedMoments()
    candidate_min = math.inf
    candidate_max = -math.inf
    reference_min = math.inf
    reference_max = -math.inf
    candidate_nonzero = 0
    reference_nonzero = 0
    absolute_sum_parts: list[float] = []
    squared_difference_parts: list[float] = []
    offset = 0
    try:
        for start in range(0, shape[0], slab_depth):
            stop = min(shape[0], start + slab_depth)
            candidate_slab = np.asarray(candidate[start:stop], dtype=np.float64)
            reference_slab = np.asarray(reference[start:stop], dtype=np.float64)
            if not np.all(np.isfinite(candidate_slab)):
                raise ValueError("candidate array must contain only finite values")
            if not np.all(np.isfinite(reference_slab)):
                raise ValueError("reference array must contain only finite values")
            candidate_flat = candidate_slab.ravel()
            reference_flat = reference_slab.ravel()
            difference = candidate_flat - reference_flat
            slab_absolute = np.abs(difference)
            count = slab_absolute.size
            absolute[offset : offset + count] = slab_absolute
            offset += count
            candidate_moments.add(candidate_flat)
            reference_moments.add(reference_flat)
            paired_moments.add(candidate_flat, reference_flat)
            candidate_min = min(candidate_min, float(np.min(candidate_flat)))
            candidate_max = max(candidate_max, float(np.max(candidate_flat)))
            reference_min = min(reference_min, float(np.min(reference_flat)))
            reference_max = max(reference_max, float(np.max(reference_flat)))
            candidate_nonzero += int(np.count_nonzero(candidate_flat))
            reference_nonzero += int(np.count_nonzero(reference_flat))
            absolute_sum_parts.append(float(np.sum(slab_absolute, dtype=np.float64)))
            squared_difference_parts.append(float(np.dot(difference, difference)))
        absolute.flush()
        quantiles = _exact_memmap_quantiles(absolute, (50.0, 90.0, 95.0, 99.0))
        absolute_sum = math.fsum(absolute_sum_parts)
        squared_difference_sum = math.fsum(squared_difference_parts)
        correlation = paired_moments.correlation
        metrics: dict[str, float | int] = {
            "voxel_count": size,
            "candidate_finite_count": size,
            "reference_finite_count": size,
            "candidate_finite_fraction": 1.0,
            "reference_finite_fraction": 1.0,
            "candidate_min": candidate_min,
            "candidate_max": candidate_max,
            "candidate_mean": candidate_moments.mean,
            "candidate_std": candidate_moments.std,
            "reference_min": reference_min,
            "reference_max": reference_max,
            "reference_mean": reference_moments.mean,
            "reference_std": reference_moments.std,
            "candidate_nonzero_count": candidate_nonzero,
            "reference_nonzero_count": reference_nonzero,
            "candidate_nonzero_fraction": candidate_nonzero / size,
            "reference_nonzero_fraction": reference_nonzero / size,
            "nonzero_fraction_ratio": (
                candidate_nonzero / reference_nonzero if reference_nonzero else 0.0
            ),
            "normalized_correlation": correlation,
            "mean_absolute_difference": absolute_sum / size,
            "root_mean_square_difference": math.sqrt(squared_difference_sum / size),
            "absolute_difference_mean": absolute_sum / size,
            "absolute_difference_median": quantiles[50.0],
            "absolute_difference_p90": quantiles[90.0],
            "absolute_difference_p95": quantiles[95.0],
            "absolute_difference_p99": quantiles[99.0],
            "absolute_difference_max": float(np.max(absolute)),
        }
        counts = {
            "voxel_count": size,
            "candidate_finite_count": size,
            "reference_finite_count": size,
            "candidate_nonzero_count": candidate_nonzero,
            "reference_nonzero_count": reference_nonzero,
        }
        accumulators = {
            "candidate_sum": candidate_moments.total,
            "candidate_sum_square": candidate_moments.total_square,
            "reference_sum": reference_moments.total,
            "reference_sum_square": reference_moments.total_square,
            "cross_product_sum": paired_moments.cross_product_sum,
            "absolute_difference_sum": absolute_sum,
            "squared_difference_sum": squared_difference_sum,
        }
        return metrics, counts, accumulators
    finally:
        mapping = getattr(absolute, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()


@dataclass(slots=True)
class _Moments:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    total_parts: list[float] | None = None
    square_parts: list[float] | None = None

    def __post_init__(self) -> None:
        self.total_parts = []
        self.square_parts = []

    def add(self, values: np.ndarray) -> None:
        count = int(values.size)
        mean = float(np.mean(values, dtype=np.float64))
        centered = values - mean
        m2 = float(np.dot(centered, centered))
        if self.count:
            delta = mean - self.mean
            combined = self.count + count
            self.m2 += m2 + delta * delta * self.count * count / combined
            self.mean += delta * count / combined
            self.count = combined
        else:
            self.count = count
            self.mean = mean
            self.m2 = m2
        assert self.total_parts is not None
        assert self.square_parts is not None
        self.total_parts.append(float(np.sum(values, dtype=np.float64)))
        self.square_parts.append(float(np.dot(values, values)))

    @property
    def std(self) -> float:
        return math.sqrt(max(0.0, self.m2 / self.count))

    @property
    def total(self) -> float:
        assert self.total_parts is not None
        return math.fsum(self.total_parts)

    @property
    def total_square(self) -> float:
        assert self.square_parts is not None
        return math.fsum(self.square_parts)


@dataclass(slots=True)
class _PairedMoments:
    count: int = 0
    mean_a: float = 0.0
    mean_b: float = 0.0
    m2_a: float = 0.0
    m2_b: float = 0.0
    covariance: float = 0.0
    cross_parts: list[float] | None = None

    def __post_init__(self) -> None:
        self.cross_parts = []

    def add(self, a: np.ndarray, b: np.ndarray) -> None:
        count = int(a.size)
        mean_a = float(np.mean(a, dtype=np.float64))
        mean_b = float(np.mean(b, dtype=np.float64))
        centered_a = a - mean_a
        centered_b = b - mean_b
        m2_a = float(np.dot(centered_a, centered_a))
        m2_b = float(np.dot(centered_b, centered_b))
        covariance = float(np.dot(centered_a, centered_b))
        if self.count:
            combined = self.count + count
            delta_a = mean_a - self.mean_a
            delta_b = mean_b - self.mean_b
            factor = self.count * count / combined
            self.m2_a += m2_a + delta_a * delta_a * factor
            self.m2_b += m2_b + delta_b * delta_b * factor
            self.covariance += covariance + delta_a * delta_b * factor
            self.mean_a += delta_a * count / combined
            self.mean_b += delta_b * count / combined
            self.count = combined
        else:
            self.count = count
            self.mean_a = mean_a
            self.mean_b = mean_b
            self.m2_a = m2_a
            self.m2_b = m2_b
            self.covariance = covariance
        assert self.cross_parts is not None
        self.cross_parts.append(float(np.dot(a, b)))

    @property
    def correlation(self) -> float:
        denominator = math.sqrt(max(0.0, self.m2_a) * max(0.0, self.m2_b))
        return 0.0 if denominator == 0.0 else self.covariance / denominator

    @property
    def cross_product_sum(self) -> float:
        assert self.cross_parts is not None
        return math.fsum(self.cross_parts)


def _exact_memmap_quantiles(values: np.memmap, percentiles: Sequence[float]) -> dict[float, float]:
    size = values.size
    positions = {percentile: (size - 1) * percentile / 100.0 for percentile in percentiles}
    indices = sorted(
        {int(math.floor(position)) for position in positions.values()}
        | {int(math.ceil(position)) for position in positions.values()}
    )
    values.partition(indices)
    output = {}
    for percentile, position in positions.items():
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        weight = position - lower
        output[percentile] = float(
            float(values[lower]) * (1.0 - weight) + float(values[upper]) * weight
        )
    return output


def _voxelwise_summary(
    definition: ContrastDefinition,
    volumes: Mapping[str, np.ndarray],
    shape: tuple[int, int, int],
    epsilon: float,
    slab_depth: int,
    temporary_directory: Path,
) -> dict[str, float]:
    size = int(np.prod(shape))
    path = temporary_directory / f"{definition.name}.absolute.float64"
    absolute = np.memmap(path, dtype=np.float64, mode="w+", shape=(size,))
    moments = _Moments()
    absolute_parts = []
    epsilon_count = 0
    offset = 0
    try:
        for start in range(0, shape[0], slab_depth):
            stop = min(shape[0], start + slab_depth)
            contrast: np.ndarray | None = None
            for cell, coefficient in definition.coefficients:
                component = np.asarray(volumes[cell][start:stop], dtype=np.float64)
                if contrast is None:
                    contrast = coefficient * component
                else:
                    contrast += coefficient * component
            assert contrast is not None
            flat = contrast.ravel()
            slab_absolute = np.abs(flat)
            count = flat.size
            absolute[offset : offset + count] = slab_absolute
            offset += count
            moments.add(flat)
            absolute_parts.append(float(np.sum(slab_absolute, dtype=np.float64)))
            epsilon_count += int(np.count_nonzero(slab_absolute > epsilon))
        absolute.flush()
        p95 = _exact_memmap_quantiles(absolute, (95.0,))[95.0]
        return {
            "mean": moments.mean,
            "std": moments.std,
            "mean_absolute": math.fsum(absolute_parts) / size,
            "p95_absolute": p95,
            "max_absolute": float(np.max(absolute)),
            "epsilon": epsilon,
            "epsilon_nonzero_fraction": epsilon_count / size,
        }
    finally:
        mapping = getattr(absolute, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()


def _exact_overlap_values(overlap: Mapping[str, float]) -> dict[str, float]:
    reference_count = overlap["a_count"]
    candidate_count = overlap["b_count"]
    intersection = overlap["overlap_count"]
    precision = overlap["overlap_over_b"]
    recall = overlap["overlap_over_a"]
    return {
        "reference_count": reference_count,
        "candidate_count": candidate_count,
        "intersection_count": intersection,
        "union_count": overlap["union_count"],
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall),
        "jaccard": overlap["jaccard"],
    }


def _selection_thresholds(
    reference: np.ndarray,
    candidate: np.ndarray,
    percentile: float,
) -> dict[str, float]:
    return {
        "percentile": percentile,
        "reference_threshold": _positive_percentile_threshold(reference, percentile),
        "candidate_threshold": _positive_percentile_threshold(candidate, percentile),
    }


def _positive_percentile_threshold(values: np.ndarray, percentile: float) -> float:
    positive = np.asarray(values)[np.asarray(values) > 0]
    if positive.size == 0:
        return 0.0
    return float(np.percentile(positive.astype(np.float64, copy=False), percentile))


def _evidence(
    dataset_id: str,
    cell_label: str,
    stage: str,
    selection: str,
    reference_file: str,
    source_fingerprint: str,
    reference_sha256: str,
    shape: tuple[int, ...],
    *,
    thresholds: Mapping[str, float] | None = None,
    counts: Mapping[str, int] | None = None,
    accumulators: Mapping[str, float] | None = None,
) -> MetricEvidence:
    return MetricEvidence(
        F3_METRIC_SCHEMA_VERSION,
        dataset_id,
        cell_label,
        stage,
        "full",
        selection,
        reference_file,
        source_fingerprint,
        reference_sha256,
        _shape(shape),
        tuple((name, float(value)) for name, value in (thresholds or {}).items()),
        tuple((name, int(value)) for name, value in (counts or {}).items()),
        tuple((name, float(value)) for name, value in (accumulators or {}).items()),
    )


def _validate_evidence(
    rows: Sequence[MetricRow],
    evidence: Sequence[MetricEvidence],
) -> None:
    identities = [item.identity for item in evidence]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate metric evidence identity")
    evidence_by_identity = {item.identity: item for item in evidence}
    for row in rows:
        if row.identity[:-1] not in evidence_by_identity:
            raise ValueError("metric row has no matching evidence")


def _validate_shared_row_values(rows: Sequence[MetricRow]) -> None:
    by_identity = {row.identity: row for row in rows}
    for row in rows:
        if row.stage not in {"ft", "fv"} or row.cell_label not in {"RL-REF", "Q-REF"}:
            continue
        paired_cell = "RL-QUAL" if row.cell_label == "RL-REF" else "Q-QUAL"
        paired_identity = (
            row.dataset_id,
            paired_cell,
            row.stage,
            row.region,
            row.selection,
            row.reference_file or "",
            row.metric,
        )
        paired = by_identity.get(paired_identity)
        if paired is not None and paired.value != row.value:
            raise ValueError("shared stage workflow metric must be exactly equal")


def _validate_metric_definitions(rows: Sequence[MetricRow]) -> None:
    for row in rows:
        try:
            definition = _DEFINITION_BY_IDENTITY[(row.stage, row.selection, row.metric)]
        except KeyError as error:
            raise ValueError("metric row is not in the F3 registry") from error
        if (row.unit, row.direction) != (definition.unit, definition.direction):
            raise ValueError("metric row semantics do not match the registry")
        if row.value is None and not definition.nullable:
            raise ValueError("only nullable ridge distances may be None")


def _unique_metric_rows(rows: Sequence[MetricRow]) -> None:
    identities = [row.identity for row in rows]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate metric row identity")


def _definitions_for(stage: str, selection: str) -> tuple[MetricDefinition, ...]:
    return tuple(
        definition
        for definition in METRIC_REGISTRY
        if definition.stage == stage and definition.selection == selection
    )


def _contrast_definition(name: str) -> ContrastDefinition:
    for definition in CONTRAST_DEFINITIONS:
        if definition.name == name:
            return definition
    raise ValueError(f"unknown contrast: {name!r}")


def _improvement(raw_value: float, direction: MetricDirection) -> float | None:
    _direction(direction)
    if direction == "higher":
        return raw_value
    if direction == "lower":
        return -raw_value
    return None


def _candidate_path(
    workspace_path: Path,
    cell: F3CellReference,
    stage: str,
) -> Path:
    kind, fingerprint, filename = {
        "ft": ("scanner", cell.stages.scanner, "ft.dat"),
        "fv": ("voting", cell.stages.voting, "fv.dat"),
        "fvt": ("thinning", cell.stages.thinning, "fvt.dat"),
    }[stage]
    return workspace_path / "stages" / kind / fingerprint / filename


def _validated_extraction_workspace(
    volume_source: F3VolumeSource,
    cells: Sequence[F3CellReference],
) -> Path:
    """Validate one complete cell chain and bind it to the supplied dataset."""

    workspace_paths = {cell.path.parent.parent for cell in cells}
    if len(workspace_paths) != 1:
        raise ValueError("cell references must belong to one workspace")
    workspace_path = next(iter(workspace_paths))
    run_manifest = _read_json(workspace_path / "run_manifest.json")
    if set(run_manifest) != {
        *_RUN_COMPUTATION_FIELDS,
        "run_fingerprint",
        "provenance",
    }:
        raise F3WorkspaceMismatchError("run manifest field set mismatch")
    computation = {name: run_manifest[name] for name in _RUN_COMPUTATION_FIELDS}
    run_fingerprint = _sha256(
        run_manifest["run_fingerprint"],
        "run manifest fingerprint",
    )
    if canonical_fingerprint(computation) != run_fingerprint:
        raise F3WorkspaceMismatchError("run manifest fingerprint mismatch")
    if run_manifest["dataset_identity"] != volume_source.identity.computation_identity:
        raise F3WorkspaceMismatchError(
            "volume_source dataset identity does not match the run workspace"
        )

    input_fingerprint = volume_source.identity.file_for("input").sha256
    validated: set[tuple[str, str, tuple[str, ...], tuple[tuple[str, str], ...]]] = set()
    for cell in cells:
        expected_cell_path = workspace_path / "cells" / f"{cell.label}.json"
        if cell.path.absolute() != expected_cell_path.absolute():
            raise F3StageCorruptionError(
                f"cell reference is not in the current workspace: {cell.label}"
            )
        if _read_json(expected_cell_path) != cell.as_dict():
            raise F3StageCorruptionError(f"cell reference mismatch: {cell.label}")

        chain = (
            ("scanner", cell.stages.scanner, (), {"ep.dat": input_fingerprint}),
            ("voting", cell.stages.voting, (cell.stages.scanner,), {}),
            ("thinning", cell.stages.thinning, (cell.stages.voting,), {}),
        )
        if cell.skinning_enabled:
            chain = (
                *chain,
                ("skinning", cell.stages.skinning, (cell.stages.thinning,), {}),
            )
        for kind, fingerprint, parents, inputs in chain:
            key = (kind, fingerprint, parents, tuple(sorted(inputs.items())))
            if key in validated:
                continue
            _validate_stored_stage(
                workspace_path,
                kind=kind,
                fingerprint=fingerprint,
                run_fingerprint=run_fingerprint,
                parent_fingerprints=parents,
                input_fingerprints=inputs,
            )
            validated.add(key)

        for stage in F3_REFERENCE_STAGE_FILES:
            _validate_candidate_report(
                _candidate_path(workspace_path, cell, stage).parent,
                cell,
                stage,
                volume_source.spec.shape,
            )
    return workspace_path


def _validate_stored_stage(
    workspace_path: Path,
    *,
    kind: str,
    fingerprint: str,
    run_fingerprint: str,
    parent_fingerprints: tuple[str, ...],
    input_fingerprints: Mapping[str, str],
) -> None:
    stage_path = workspace_path / "stages" / kind / fingerprint
    manifest = _read_json(stage_path / "stage_manifest.json")
    try:
        computation = {name: manifest[name] for name in _STAGE_COMPUTATION_FIELDS}
    except KeyError as error:
        raise F3StageCorruptionError(
            f"{kind} stage manifest is missing computation identity"
        ) from error
    expected = {
        "kind": kind,
        "run_fingerprint": run_fingerprint,
        "parent_fingerprints": list(parent_fingerprints),
        "input_fingerprints": dict(input_fingerprints),
    }
    if any(computation[name] != value for name, value in expected.items()):
        raise F3StageCorruptionError(f"{kind} stage computation identity mismatch")
    if canonical_fingerprint(computation) != fingerprint:
        raise F3StageCorruptionError(f"{kind} stage fingerprint mismatch")
    validate_stage(stage_path, computation, fingerprint)


def _stage_fingerprint_for(cell: F3CellReference, stage: str) -> str:
    return {
        "ft": cell.stages.scanner,
        "fv": cell.stages.voting,
        "fvt": cell.stages.thinning,
    }[stage]


def _validate_candidate_report(
    stage_path: Path,
    cell: F3CellReference,
    stage: str,
    shape: tuple[int, int, int],
) -> None:
    report = _read_json(stage_path / "report.json")
    expected_fingerprint = _stage_fingerprint_for(cell, stage)
    if report.get("fingerprint") != expected_fingerprint:
        raise ValueError("candidate source fingerprint mismatch")
    if tuple(report.get("shape", ())) != shape:
        raise ValueError("candidate source shape mismatch")
    if stage == "ft" and report.get("backend") != cell.backend:
        raise ValueError("scanner report backend mismatch")
    if stage == "fv" and report.get("scanner_stage_fingerprint") != cell.stages.scanner:
        raise ValueError("voting report source fingerprint mismatch")
    if stage == "fvt" and report.get("voting_stage_fingerprint") != cell.stages.voting:
        raise ValueError("thinning report source fingerprint mismatch")


def _open_dat(path: Path, shape: tuple[int, int, int]) -> np.memmap:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_bytes = int(np.prod(shape)) * _DAT_DTYPE.itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"DAT artifact size mismatch: {path}")
    array = np.memmap(path, dtype=_DAT_DTYPE, mode="r", shape=shape, order="C")
    array.flags.writeable = False
    return array


def _close_memmap(array: np.memmap) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _comparable_3d_arrays(
    candidate: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_values = np.asarray(candidate)
    reference_values = np.asarray(reference)
    if candidate_values.shape != reference_values.shape:
        raise ValueError(
            f"candidate/reference shape mismatch: "
            f"{candidate_values.shape} != {reference_values.shape}"
        )
    _shape(candidate_values.shape)
    if candidate_values.size == 0:
        raise ValueError("candidate/reference arrays must not be empty")
    return candidate_values, reference_values


__all__ = [
    "CONTRAST_DEFINITIONS",
    "F3_BUFFERED_PERCENTILE",
    "F3_BUFFER_RADIUS",
    "F3_METRIC_ROW_FIELDS",
    "F3_METRIC_SCHEMA_VERSION",
    "F3_PERCENTILES",
    "F3_REFERENCE_STAGE_FILES",
    "METRIC_REGISTRY",
    "ContrastDefinition",
    "ContrastRow",
    "MetricDefinition",
    "MetricEvidence",
    "MetricExtraction",
    "MetricRow",
    "VoxelwiseContrastSummary",
    "compute_contrast_rows",
    "compute_reference_metric_rows",
    "compute_skin_metric_rows",
    "compute_voxelwise_contrast_summaries",
    "extract_f3d_metric_rows",
    "extract_f3d_metrics",
    "validate_shared_stage_metrics",
]
