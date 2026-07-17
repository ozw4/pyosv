"""Canonical long-format metrics for synthetic mode-comparison trials."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Literal

import numpy as np

from pyosv.synthetic_metrics import (
    buffered_surface_overlap,
    edge_false_positive_ratio,
    masked_orientation_error,
    surface_distance_metrics,
    top_truth_count_mask,
)

from ..synthetic_quality.models import PipelineArtifacts
from ..synthetic_quality.quality_metrics import EDGE_FALSE_POSITIVE_MARGIN
from .models import SCANNER_ONLY_SCOPE
from .runner import SyntheticCellEvaluation, SyntheticTrialEvaluation

METRIC_SCHEMA_VERSION = 1
MetricDirection = Literal["higher", "lower", "neutral"]


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    """Registry entry defining one canonical stage/selection metric."""

    stage: str
    selection: str
    metric: str
    unit: str
    direction: MetricDirection
    contrast_eligible: bool = True

    def __post_init__(self) -> None:
        for name in ("stage", "selection", "metric", "unit"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.direction not in {"higher", "lower", "neutral"}:
            raise ValueError("direction must be 'higher', 'lower', or 'neutral'")
        if not isinstance(self.contrast_eligible, bool):
            raise ValueError("contrast_eligible must be bool")


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One finite canonical long-format metric observation."""

    schema_version: int
    case_id: str
    trial_id: str
    seed: int | None
    scope: str
    cell_label: str
    input_mode: str
    scanner_backend: str | None
    scanner_refinement_factor: int | None
    scanner_thin_mode: str | None
    workflow_mode: str | None
    voter_thin_mode: str | None
    skinner_method: str | None
    variant: str
    stage: str
    selection: str
    metric: str
    value: float
    unit: str
    direction: MetricDirection
    contrast_eligible: bool

    def __post_init__(self) -> None:
        if self.schema_version != METRIC_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {METRIC_SCHEMA_VERSION}")
        for name in (
            "case_id",
            "trial_id",
            "scope",
            "cell_label",
            "input_mode",
            "variant",
            "stage",
            "selection",
            "metric",
            "unit",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.value, bool) or not isinstance(self.value, Real):
            raise ValueError("value must be a finite number")
        value = float(self.value)
        if not np.isfinite(value):
            raise ValueError("value must be finite")
        object.__setattr__(self, "value", value)
        if self.direction not in {"higher", "lower", "neutral"}:
            raise ValueError("direction must be 'higher', 'lower', or 'neutral'")
        if not isinstance(self.contrast_eligible, bool):
            raise ValueError("contrast_eligible must be bool")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping in canonical field order."""

        return asdict(self)


_QUALITY_METRICS = (
    ("candidate_count", "count", "neutral"),
    ("buffered_precision", "fraction", "higher"),
    ("buffered_recall", "fraction", "higher"),
    ("buffered_f1", "fraction", "higher"),
    ("candidate_to_truth_median", "voxel", "lower"),
    ("candidate_to_truth_p95", "voxel", "lower"),
    ("truth_to_candidate_median", "voxel", "lower"),
    ("truth_to_candidate_p95", "voxel", "lower"),
    ("hausdorff_p95", "voxel", "lower"),
    ("strike_median", "degree", "lower"),
    ("strike_p95", "degree", "lower"),
    ("dip_median", "degree", "lower"),
    ("dip_p95", "degree", "lower"),
    ("edge_false_positive_fraction_of_candidates", "fraction", "lower"),
)


def _build_registry() -> tuple[MetricDefinition, ...]:
    definitions: list[MetricDefinition] = []
    for stage in ("scanner_raw", "scanner_thinned", "fv", "fvt"):
        definitions.append(
            MetricDefinition(stage, "all", "array_nonzero_fraction", "fraction", "neutral")
        )
        selections = (
            ("top_truth_count",)
            if stage.startswith("scanner_")
            else ("top_truth_count", "positive_top_truth_count")
        )
        for selection in selections:
            definitions.extend(
                MetricDefinition(stage, selection, metric, unit, direction)
                for metric, unit, direction in _QUALITY_METRICS
            )
    for selection in ("finite", "raw_top_truth_count"):
        definitions.extend(
            MetricDefinition(
                "scanner_confidence",
                selection,
                f"confidence_{summary}",
                "score",
                "neutral",
                False,
            )
            for summary in ("mean", "median", "p95")
        )
    definitions.extend(
        MetricDefinition("skin", "skin_cells", metric, unit, direction)
        for metric, unit, direction in (
            *_QUALITY_METRICS,
            ("skin_count", "count", "neutral"),
            ("largest_skin_fraction", "fraction", "neutral"),
            ("small_skin_cell_fraction", "fraction", "neutral"),
            ("duplicate_cell_count", "count", "neutral"),
            ("covered_truth_component_count", "count", "higher"),
            ("uncovered_truth_component_count", "count", "lower"),
            ("over_merge_skin_count", "count", "lower"),
            ("over_split_truth_component_count", "count", "lower"),
            ("mean_skin_purity", "fraction", "higher"),
            ("min_skin_purity", "fraction", "higher"),
            ("mean_truth_component_recall", "fraction", "higher"),
            ("min_truth_component_recall", "fraction", "higher"),
        )
    )
    identities = {(item.stage, item.selection, item.metric) for item in definitions}
    if len(identities) != len(definitions):
        raise RuntimeError("metric registry contains duplicate identities")
    return tuple(definitions)


METRIC_REGISTRY = _build_registry()


def extract_trial_metric_rows(evaluation: SyntheticTrialEvaluation) -> tuple[MetricRow, ...]:
    """Extract finite rows in plan-cell then registry order from one completed trial."""

    if not isinstance(evaluation, SyntheticTrialEvaluation):
        raise ValueError("evaluation must be a SyntheticTrialEvaluation")
    truth_surface_half_width = _finite_nonnegative(
        evaluation.truth_metric_config.truth_surface_half_width,
        "truth_surface_half_width",
    )
    buffer_radius = _finite_nonnegative(
        evaluation.truth_metric_config.buffer_radius,
        "buffer_radius",
    )
    truth_volumes = _truth_volumes(evaluation)
    truth_fault = _finite_array(
        _required(truth_volumes, "truth_fault_mask"),
        evaluation.trial.shape,
        "truth_fault_mask",
    )
    truth_distance = _finite_array(
        _required(truth_volumes, "truth_distance"),
        evaluation.trial.shape,
        "truth_distance",
    )
    truth_strike = _finite_array(
        _required(truth_volumes, "truth_strike"),
        evaluation.trial.shape,
        "truth_strike",
    )
    truth_dip = _finite_array(
        _required(truth_volumes, "truth_dip"),
        evaluation.trial.shape,
        "truth_dip",
    )
    truth_surface = np.abs(truth_distance) <= np.float32(truth_surface_half_width)

    rows: list[MetricRow] = []
    for cell_evaluation in evaluation.cells:
        values: dict[tuple[str, str, str], float | int] = {}
        if cell_evaluation.cell.scope == SCANNER_ONLY_SCOPE:
            _scanner_values(
                values,
                cell_evaluation,
                shape=evaluation.trial.shape,
                truth_fault=np.asarray(truth_fault, dtype=bool),
                truth_surface=truth_surface,
                truth_strike=truth_strike,
                truth_dip=truth_dip,
                buffer_radius=buffer_radius,
            )
        else:
            _downstream_values(values, cell_evaluation, shape=evaluation.trial.shape)

        metadata = _row_metadata(evaluation, cell_evaluation)
        for definition in METRIC_REGISTRY:
            key = (definition.stage, definition.selection, definition.metric)
            if key not in values:
                continue
            rows.append(
                MetricRow(
                    **metadata,
                    stage=definition.stage,
                    selection=definition.selection,
                    metric=definition.metric,
                    value=_finite_value(values[key], "/".join(key)),
                    unit=definition.unit,
                    direction=definition.direction,
                    contrast_eligible=definition.contrast_eligible,
                )
            )

    identities = [
        (row.case_id, row.trial_id, row.cell_label, row.stage, row.selection, row.metric)
        for row in rows
    ]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate metric row identity")
    return tuple(rows)


def extract_trial_metrics(evaluation: SyntheticTrialEvaluation) -> tuple[MetricRow, ...]:
    """Alias for :func:`extract_trial_metric_rows`."""

    return extract_trial_metric_rows(evaluation)


def _truth_volumes(evaluation: SyntheticTrialEvaluation) -> Mapping[str, Any]:
    for cell in evaluation.cells:
        if isinstance(cell.artifacts, PipelineArtifacts):
            return cell.artifacts.volumes
    raise ValueError("trial evaluation is missing downstream truth artifacts")


def _scanner_values(
    output: dict[tuple[str, str, str], float | int],
    evaluation: SyntheticCellEvaluation,
    *,
    shape: tuple[int, int, int],
    truth_fault: np.ndarray,
    truth_surface: np.ndarray,
    truth_strike: np.ndarray,
    truth_dip: np.ndarray,
    buffer_radius: float,
) -> None:
    if isinstance(evaluation.artifacts, PipelineArtifacts):
        raise ValueError("scanner-only cell must provide scanner artifacts")
    volumes = evaluation.artifacts
    for stage, names in (
        ("scanner_raw", ("scanner_ft", "scanner_pt", "scanner_tt")),
        ("scanner_thinned", ("scanner_fet", "scanner_fpt", "scanner_ftt")),
    ):
        values = _finite_array(_required(volumes, names[0]), shape, names[0])
        strike = _finite_array(_required(volumes, names[1]), shape, names[1])
        dip = _finite_array(_required(volumes, names[2]), shape, names[2])
        _put(output, stage, "all", "array_nonzero_fraction", np.count_nonzero(values) / values.size)
        candidate = top_truth_count_mask(values, truth_surface)
        _put_quality(
            output,
            stage,
            "top_truth_count",
            buffered_surface_overlap(candidate, truth_fault, radius=buffer_radius),
            surface_distance_metrics(candidate, truth_surface),
            masked_orientation_error(strike, dip, truth_strike, truth_dip, candidate),
            edge_false_positive_ratio(
                candidate,
                truth_surface,
                edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
                truth_buffer_radius=buffer_radius,
            ),
        )

    confidence = volumes.get("scanner_confidence")
    if confidence is not None:
        confidence_array = _finite_array(confidence, shape, "scanner_confidence")
        raw = _finite_array(_required(volumes, "scanner_ft"), shape, "scanner_ft")
        raw_support = top_truth_count_mask(raw, truth_surface)
        _put_summaries(output, "finite", confidence_array.ravel())
        _put_summaries(output, "raw_top_truth_count", confidence_array[raw_support])


def _downstream_values(
    output: dict[tuple[str, str, str], float | int],
    evaluation: SyntheticCellEvaluation,
    *,
    shape: tuple[int, int, int],
) -> None:
    if not isinstance(evaluation.artifacts, PipelineArtifacts):
        raise ValueError("downstream cell must provide pipeline artifacts")
    volumes = evaluation.artifacts.volumes
    for name in (
        "truth_fault_mask",
        "truth_distance",
        "truth_strike",
        "truth_dip",
        "fv_py",
        "vp_py",
        "vt_py",
        "fvt_py",
        "skin_mask_py",
    ):
        _finite_array(_required(volumes, name), shape, name)
    quality = _mapping(_required(evaluation.report_payload, "quality"), "quality")
    edge = _mapping(_required(quality, "edge_false_positive"), "quality.edge_false_positive")
    for stage, volume_name in (("fv", "fv_py"), ("fvt", "fvt_py")):
        array = np.asarray(volumes[volume_name])
        _put(output, stage, "all", "array_nonzero_fraction", np.count_nonzero(array) / array.size)
        for selection, report_key in (
            ("top_truth_count", f"{stage}_top_truth_count"),
            ("positive_top_truth_count", f"{stage}_positive_top_truth_count"),
        ):
            block = _mapping(_required(quality, report_key), f"quality.{report_key}")
            _put_quality(
                output,
                stage,
                selection,
                _mapping(_required(block, "buffered_overlap_radius2"), "buffered overlap"),
                _mapping(_required(block, "surface_distance"), "surface distance"),
                _mapping(_required(block, "orientation_error"), "orientation error"),
                _mapping(_required(edge, report_key), "edge false positive"),
            )

    settings = evaluation.effective_workflow_settings
    if settings is None:
        raise ValueError("downstream cell is missing effective workflow settings")
    report_skinning = _mapping(_required(evaluation.report_payload, "skinning"), "skinning")
    report_enabled = _required(report_skinning, "enabled")
    if not isinstance(report_enabled, bool) or report_enabled != settings.skinning_config.enabled:
        raise ValueError("skinning enabled state does not match effective configuration")
    if not report_enabled:
        return
    skin = _mapping(_required(quality, "skin"), "quality.skin")
    _put_quality(
        output,
        "skin",
        "skin_cells",
        _mapping(_required(skin, "buffered_overlap_radius2"), "skin buffered overlap"),
        _mapping(_required(skin, "surface_distance"), "skin surface distance"),
        _mapping(_required(skin, "orientation_error"), "skin orientation error"),
        _mapping(_required(edge, "skin"), "skin edge false positive"),
    )
    topology = _mapping(_required(skin, "topology"), "skin topology")
    component = _mapping(_required(skin, "component_topology"), "skin component topology")
    for metric in (
        "skin_count",
        "largest_skin_fraction",
        "small_skin_cell_fraction",
        "duplicate_cell_count",
    ):
        _put(output, "skin", "skin_cells", metric, _required(topology, metric))
    for metric in (
        "covered_truth_component_count",
        "uncovered_truth_component_count",
        "over_merge_skin_count",
        "over_split_truth_component_count",
        "mean_skin_purity",
        "min_skin_purity",
        "mean_truth_component_recall",
        "min_truth_component_recall",
    ):
        _put(output, "skin", "skin_cells", metric, _required(component, metric))


def _put_quality(
    output: dict[tuple[str, str, str], float | int],
    stage: str,
    selection: str,
    overlap: Mapping[str, Any],
    distance: Mapping[str, Any],
    orientation: Mapping[str, Any],
    edge: Mapping[str, Any],
) -> None:
    sources = {
        "candidate_count": overlap,
        "buffered_precision": overlap,
        "buffered_recall": overlap,
        "buffered_f1": overlap,
        "candidate_to_truth_median": distance,
        "candidate_to_truth_p95": distance,
        "truth_to_candidate_median": distance,
        "truth_to_candidate_p95": distance,
        "hausdorff_p95": distance,
        "strike_median": orientation,
        "strike_p95": orientation,
        "dip_median": orientation,
        "dip_p95": orientation,
        "edge_false_positive_fraction_of_candidates": edge,
    }
    for metric, source in sources.items():
        _put(output, stage, selection, metric, _required(source, metric))


def _put_summaries(
    output: dict[tuple[str, str, str], float | int], selection: str, samples: np.ndarray
) -> None:
    if samples.size == 0:
        raise ValueError(f"scanner confidence selection {selection} must not be empty")
    for summary, value in (
        ("mean", np.mean(samples)),
        ("median", np.median(samples)),
        ("p95", np.percentile(samples, 95.0)),
    ):
        _put(output, "scanner_confidence", selection, f"confidence_{summary}", value)


def _row_metadata(trial: SyntheticTrialEvaluation, cell: SyntheticCellEvaluation) -> dict[str, Any]:
    scanner = cell.effective_scanner_config
    workflow = cell.effective_workflow_settings
    if cell.cell.scope == SCANNER_ONLY_SCOPE and scanner is None:
        raise ValueError("scanner-only cell is missing effective scanner configuration")
    if cell.cell.scope != SCANNER_ONLY_SCOPE and workflow is None:
        raise ValueError("downstream cell is missing effective workflow settings")
    scanner_backend = scanner.backend if scanner is not None else None
    refinement = (
        scanner.refinement_factor if scanner is not None and scanner.backend == "quality" else None
    )
    skinning = workflow.skinning_config if workflow is not None else None
    return {
        "schema_version": METRIC_SCHEMA_VERSION,
        "case_id": trial.trial.case_id,
        "trial_id": trial.trial.trial_id,
        "seed": trial.trial.seed,
        "scope": cell.cell.scope,
        "cell_label": cell.cell.label,
        "input_mode": cell.cell.input_mode,
        "scanner_backend": scanner_backend,
        "scanner_refinement_factor": refinement,
        "scanner_thin_mode": scanner.scanner_thin_mode if scanner is not None else None,
        "workflow_mode": workflow.workflow_mode if workflow is not None else None,
        "voter_thin_mode": (
            workflow.voting_config.voter_thin_mode if workflow is not None else None
        ),
        "skinner_method": (skinning.method if skinning is not None and skinning.enabled else None),
        "variant": cell.variant,
    }


def _put(
    output: dict[tuple[str, str, str], float | int],
    stage: str,
    selection: str,
    metric: str,
    value: Any,
) -> None:
    key = (stage, selection, metric)
    if key in output:
        raise ValueError(f"duplicate generated metric: {'/'.join(key)}")
    output[key] = _finite_value(value, "/".join(key))


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    try:
        return mapping[key]
    except KeyError as error:
        raise ValueError(f"missing required metric artifact: {key}") from error


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _finite_array(value: Any, shape: tuple[int, int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    try:
        finite = np.isfinite(array)
    except TypeError as error:
        raise ValueError(f"{name} must be numeric") from error
    if not np.all(finite):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_value(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_nonnegative(value: Any, name: str) -> float:
    result = _finite_value(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


__all__ = [
    "METRIC_REGISTRY",
    "METRIC_SCHEMA_VERSION",
    "MetricDefinition",
    "MetricDirection",
    "MetricRow",
    "extract_trial_metric_rows",
    "extract_trial_metrics",
]
