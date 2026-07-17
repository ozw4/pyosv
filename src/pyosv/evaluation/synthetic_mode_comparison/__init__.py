"""Canonical planning models for synthetic scanner/workflow comparisons."""

from .artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    COMPLETION_SCHEMA_VERSION,
    HASHED_BUNDLE_FILES,
    REQUIRED_BUNDLE_FILES,
    validate_completed_bundle,
    write_artifact_bundle,
)
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
from .experiment import RuntimeRow, SyntheticModeComparisonResult, run_mode_comparison
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
    "ARTIFACT_SCHEMA_VERSION",
    "CANONICAL_COMPARISON_VARIANT",
    "CanonicalScannerBackend",
    "CanonicalWorkflowMode",
    "CONTRAST_DEFINITIONS",
    "COMPLETION_SCHEMA_VERSION",
    "HASHED_BUNDLE_FILES",
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
    "RuntimeRow",
    "REQUIRED_BUNDLE_FILES",
    "SyntheticModeComparisonConfig",
    "SyntheticModeComparisonPlan",
    "SyntheticModeComparisonResult",
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
    "run_mode_comparison",
    "validate_completed_bundle",
    "validate_trial_seeds",
    "write_artifact_bundle",
]
