"""Canonical planning models for synthetic scanner/workflow comparisons."""

from .builder import build_mode_comparison_plan
from .config import CANONICAL_COMPARISON_VARIANT, SyntheticModeComparisonConfig
from .models import (
    CanonicalScannerBackend,
    CanonicalWorkflowMode,
    ModeCellScope,
    ModeCellSpec,
    ModeInputMode,
    SyntheticModeComparisonPlan,
)
from .trials import SyntheticTrialSpec, expand_synthetic_trials, validate_trial_seeds

__all__ = [
    "CANONICAL_COMPARISON_VARIANT",
    "CanonicalScannerBackend",
    "CanonicalWorkflowMode",
    "ModeCellScope",
    "ModeCellSpec",
    "ModeInputMode",
    "SyntheticModeComparisonConfig",
    "SyntheticModeComparisonPlan",
    "SyntheticTrialSpec",
    "build_mode_comparison_plan",
    "expand_synthetic_trials",
    "validate_trial_seeds",
]
