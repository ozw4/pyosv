"""Curated, immutable metric selection for the public comparison report."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from ..f3d_mode_comparison.metrics import METRIC_REGISTRY as F3_METRIC_REGISTRY
from ..synthetic_mode_comparison.metrics import METRIC_REGISTRY as SYNTHETIC_METRIC_REGISTRY

from .config import F3_SEMANTICS, SYNTHETIC_SEMANTICS

MetricDataset = Literal["synthetic", "f3"]


@dataclass(frozen=True, slots=True)
class PublicationMetric:
    """One selected source metric and its publication display contract."""

    dataset: MetricDataset
    stage: str
    selection: str
    metric: str
    display_label: str
    unit: str
    direction: str
    figure_group: str
    required: bool
    evaluation_semantics: str
    nullable: bool = False

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return (self.dataset, self.stage, self.selection, self.metric)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _metric(
    dataset: MetricDataset,
    stage: str,
    selection: str,
    metric: str,
    label: str,
    unit: str,
    direction: str,
    group: str,
    *,
    required: bool = True,
    nullable: bool = False,
) -> PublicationMetric:
    return PublicationMetric(
        dataset,
        stage,
        selection,
        metric,
        label,
        unit,
        direction,
        group,
        required,
        SYNTHETIC_SEMANTICS if dataset == "synthetic" else F3_SEMANTICS,
        nullable,
    )


PUBLICATION_METRIC_REGISTRY = (
    _metric(
        "synthetic",
        "scanner_raw",
        "top_truth_count",
        "buffered_f1",
        "Scanner buffered F1",
        "fraction",
        "higher",
        "synthetic_scanner",
    ),
    _metric(
        "synthetic",
        "scanner_raw",
        "top_truth_count",
        "hausdorff_p95",
        "Scanner ridge Hausdorff p95",
        "voxel",
        "lower",
        "synthetic_scanner",
    ),
    _metric(
        "synthetic",
        "scanner_raw",
        "top_truth_count",
        "strike_median",
        "Scanner strike error median",
        "degree",
        "lower",
        "synthetic_orientation",
    ),
    _metric(
        "synthetic",
        "scanner_raw",
        "top_truth_count",
        "dip_median",
        "Scanner dip error median",
        "degree",
        "lower",
        "synthetic_orientation",
    ),
    _metric(
        "synthetic",
        "fvt",
        "positive_top_truth_count",
        "buffered_f1",
        "FVT buffered F1",
        "fraction",
        "higher",
        "synthetic_fvt",
    ),
    _metric(
        "synthetic",
        "fvt",
        "positive_top_truth_count",
        "hausdorff_p95",
        "FVT ridge Hausdorff p95",
        "voxel",
        "lower",
        "synthetic_fvt",
    ),
    _metric(
        "synthetic",
        "fvt",
        "positive_top_truth_count",
        "edge_false_positive_fraction_of_candidates",
        "FVT edge false-positive fraction",
        "fraction",
        "lower",
        "synthetic_fvt",
    ),
    _metric(
        "synthetic",
        "skin",
        "skin_cells",
        "buffered_f1",
        "Skin buffered F1",
        "fraction",
        "higher",
        "synthetic_skin",
        required=False,
    ),
    _metric(
        "synthetic",
        "skin",
        "skin_cells",
        "hausdorff_p95",
        "Skin ridge Hausdorff p95",
        "voxel",
        "lower",
        "synthetic_skin",
        required=False,
    ),
    _metric(
        "synthetic",
        "skin",
        "skin_cells",
        "over_merge_skin_count",
        "Over-merge skin count",
        "count",
        "lower",
        "synthetic_skin_descriptive",
        required=False,
    ),
    _metric(
        "synthetic",
        "skin",
        "skin_cells",
        "over_split_truth_component_count",
        "Over-split truth-component count",
        "count",
        "lower",
        "synthetic_skin_descriptive",
        required=False,
    ),
    _metric(
        "synthetic",
        "skin",
        "skin_cells",
        "mean_skin_purity",
        "Mean skin purity",
        "fraction",
        "higher",
        "synthetic_skin_descriptive",
        required=False,
    ),
    _metric(
        "synthetic",
        "skin",
        "skin_cells",
        "mean_truth_component_recall",
        "Mean truth-component recall",
        "fraction",
        "higher",
        "synthetic_skin_descriptive",
        required=False,
    ),
    _metric(
        "f3",
        "ft",
        "all",
        "normalized_correlation",
        "F3 public-reference agreement: normalized correlation",
        "correlation",
        "higher",
        "f3_scalar",
    ),
    _metric(
        "f3",
        "fv",
        "all",
        "normalized_correlation",
        "F3 public-reference agreement: normalized correlation",
        "correlation",
        "higher",
        "f3_scalar",
    ),
    _metric(
        "f3",
        "fvt",
        "all",
        "normalized_correlation",
        "F3 public-reference agreement: normalized correlation",
        "correlation",
        "higher",
        "f3_scalar",
    ),
    _metric(
        "f3",
        "ft",
        "all",
        "mean_absolute_difference",
        "Output difference: mean absolute difference",
        "value",
        "lower",
        "f3_scalar",
    ),
    _metric(
        "f3",
        "fv",
        "all",
        "mean_absolute_difference",
        "Output difference: mean absolute difference",
        "value",
        "lower",
        "f3_scalar",
    ),
    _metric(
        "f3",
        "fvt",
        "all",
        "mean_absolute_difference",
        "Output difference: mean absolute difference",
        "value",
        "lower",
        "f3_scalar",
    ),
    _metric(
        "f3",
        "ft",
        "all",
        "nonzero_fraction_ratio",
        "Density ratio: nonzero fraction ratio",
        "ratio",
        "neutral",
        "f3_scalar",
    ),
    _metric(
        "f3",
        "fv",
        "all",
        "nonzero_fraction_ratio",
        "Density ratio: nonzero fraction ratio",
        "ratio",
        "neutral",
        "f3_scalar",
    ),
    _metric(
        "f3",
        "fvt",
        "all",
        "nonzero_fraction_ratio",
        "Density ratio: nonzero fraction ratio",
        "ratio",
        "neutral",
        "f3_scalar",
    ),
    _metric(
        "f3",
        "ft",
        "positive_p99_radius2",
        "buffered_f1",
        "Ridge agreement: buffered F1",
        "fraction",
        "higher",
        "f3_scalar",
    ),
    _metric(
        "f3",
        "fv",
        "positive_p99_radius2",
        "buffered_f1",
        "Ridge agreement: buffered F1",
        "fraction",
        "higher",
        "f3_scalar",
    ),
    _metric(
        "f3",
        "fvt",
        "positive_p99_radius2",
        "buffered_f1",
        "Ridge agreement: buffered F1",
        "fraction",
        "higher",
        "f3_scalar",
    ),
    _metric(
        "f3",
        "ft",
        "positive_p99_distance",
        "candidate_to_reference_p95",
        "Ridge distance: candidate to reference p95",
        "voxel",
        "lower",
        "f3_scalar",
        nullable=True,
    ),
    _metric(
        "f3",
        "fv",
        "positive_p99_distance",
        "candidate_to_reference_p95",
        "Ridge distance: candidate to reference p95",
        "voxel",
        "lower",
        "f3_scalar",
        nullable=True,
    ),
    _metric(
        "f3",
        "fvt",
        "positive_p99_distance",
        "candidate_to_reference_p95",
        "Ridge distance: candidate to reference p95",
        "voxel",
        "lower",
        "f3_scalar",
        nullable=True,
    ),
    _metric(
        "f3",
        "ft",
        "positive_p99_distance",
        "reference_to_candidate_p95",
        "Ridge distance: reference to candidate p95",
        "voxel",
        "lower",
        "f3_scalar",
        nullable=True,
    ),
    _metric(
        "f3",
        "fv",
        "positive_p99_distance",
        "reference_to_candidate_p95",
        "Ridge distance: reference to candidate p95",
        "voxel",
        "lower",
        "f3_scalar",
        nullable=True,
    ),
    _metric(
        "f3",
        "fvt",
        "positive_p99_distance",
        "reference_to_candidate_p95",
        "Ridge distance: reference to candidate p95",
        "voxel",
        "lower",
        "f3_scalar",
        nullable=True,
    ),
)

# Skin rows are intentionally descriptive and neutral.  They are not folded into
# the F3 agreement score and remain absent when the source run disabled skinning.
for _stage_metric in (
    "skin_count",
    "cell_count",
    "unique_cell_count",
    "duplicate_cell_count",
    "largest_skin_fraction",
    "small_skin_cell_fraction",
    "fallback_used",
    "fallback_skin_count",
    "fallback_cell_count",
):
    _unit = (
        "fraction"
        if _stage_metric.endswith("fraction")
        else ("flag" if _stage_metric == "fallback_used" else "count")
    )
    PUBLICATION_METRIC_REGISTRY += (
        _metric(
            "f3",
            "skin",
            "descriptive",
            _stage_metric,
            f"F3 skin descriptive: {_stage_metric.replace('_', ' ')}",
            _unit,
            "neutral",
            "f3_skin_descriptive",
            required=False,
        ),
    )


def _validate_against_source_registries() -> None:
    synthetic = {
        (item.stage, item.selection, item.metric): item for item in SYNTHETIC_METRIC_REGISTRY
    }
    f3 = {(item.stage, item.selection, item.metric): item for item in F3_METRIC_REGISTRY}
    for entry in PUBLICATION_METRIC_REGISTRY:
        source = synthetic if entry.dataset == "synthetic" else f3
        try:
            definition = source[(entry.stage, entry.selection, entry.metric)]
        except KeyError as error:
            raise RuntimeError(
                "publication metric is not present in the domain source registry: "
                f"{entry.identity!r}"
            ) from error
        if (definition.unit, definition.direction) != (entry.unit, entry.direction):
            raise RuntimeError(f"publication metric metadata mismatch: {entry.identity!r}")
        if entry.nullable and not getattr(definition, "nullable", False):
            raise RuntimeError(
                f"publication metric is marked nullable incorrectly: {entry.identity!r}"
            )


_validate_against_source_registries()
PUBLICATION_METRIC_BY_IDENTITY = {entry.identity: entry for entry in PUBLICATION_METRIC_REGISTRY}

__all__ = [
    "PUBLICATION_METRIC_BY_IDENTITY",
    "PUBLICATION_METRIC_REGISTRY",
    "PublicationMetric",
]
