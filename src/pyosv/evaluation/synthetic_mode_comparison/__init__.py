"""Canonical planning models for synthetic scanner/workflow comparisons."""

from .builder import build_mode_comparison_plan
from .config import CANONICAL_COMPARISON_VARIANT, SyntheticModeComparisonConfig
from .contrasts import (
    CONTRAST_DEFINITIONS,
    AggregateRow,
    ContrastDefinition,
    ContrastRow,
    aggregate_contrast_rows,
    aggregate_metric_rows,
    compute_contrast_rows,
)
from .metrics import (
    METRIC_REGISTRY,
    METRIC_SCHEMA_VERSION,
    MetricDefinition,
    MetricDirection,
    MetricRow,
    extract_trial_metric_rows,
    extract_trial_metrics,
)
from .models import (
    CanonicalScannerBackend,
    CanonicalWorkflowMode,
    ModeCellScope,
    ModeCellSpec,
    ModeInputMode,
    SyntheticModeComparisonPlan,
)
from .runner import SyntheticCellEvaluation, SyntheticTrialEvaluation, run_synthetic_trial
from .trials import SyntheticTrialSpec, expand_synthetic_trials, validate_trial_seeds

__all__ = [
    "CANONICAL_COMPARISON_VARIANT",
    "CanonicalScannerBackend",
    "CanonicalWorkflowMode",
    "CONTRAST_DEFINITIONS",
    "AggregateRow",
    "ContrastDefinition",
    "ContrastRow",
    "METRIC_REGISTRY",
    "METRIC_SCHEMA_VERSION",
    "MetricDefinition",
    "MetricDirection",
    "MetricRow",
    "ModeCellScope",
    "ModeCellSpec",
    "ModeInputMode",
    "SyntheticModeComparisonConfig",
    "SyntheticModeComparisonPlan",
    "SyntheticCellEvaluation",
    "SyntheticTrialEvaluation",
    "SyntheticTrialSpec",
    "build_mode_comparison_plan",
    "aggregate_contrast_rows",
    "aggregate_metric_rows",
    "compute_contrast_rows",
    "expand_synthetic_trials",
    "extract_trial_metric_rows",
    "extract_trial_metrics",
    "run_synthetic_trial",
    "validate_trial_seeds",
]
