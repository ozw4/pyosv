"""Canonical planning contract for the F3 full-volume 2-by-2 comparison."""

from .builder import build_f3d_mode_comparison_plan
from .config import (
    F3ModeComparisonConfig,
    F3ScannerConfig,
    F3VotingControls,
)
from .models import (
    F3CellSpec,
    F3DatasetSpec,
    F3FixedControlEvidence,
    F3ModeComparisonPlan,
    F3ScannerBackend,
    F3WorkflowMode,
    canonical_f3_cells,
)

__all__ = [
    "F3CellSpec",
    "F3DatasetSpec",
    "F3FixedControlEvidence",
    "F3ModeComparisonConfig",
    "F3ModeComparisonPlan",
    "F3ScannerBackend",
    "F3ScannerConfig",
    "F3VotingControls",
    "F3WorkflowMode",
    "build_f3d_mode_comparison_plan",
    "canonical_f3_cells",
]
