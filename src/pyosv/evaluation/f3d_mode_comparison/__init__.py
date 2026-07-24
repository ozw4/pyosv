"""Canonical planning contract for the F3 full-volume 2-by-2 comparison."""

from .builder import build_f3d_mode_comparison_plan
from .config import (
    F3ModeComparisonConfig,
    F3ScannerConfig,
    F3VotingControls,
)
from .data import (
    F3_DATASET_ID,
    F3_FILE_ROLES,
    OFFICIAL_F3_DATASET_SPEC,
    F3DatasetIdentity,
    F3DatasetSpec,
    F3FileIdentity,
    F3VolumeSource,
    ensure_output_not_in_data_root,
)
from .models import (
    F3CellSpec,
    F3FixedControlEvidence,
    F3ModeComparisonPlan,
    F3ScannerBackend,
    F3WorkflowMode,
    canonical_f3_cells,
)

__all__ = [
    "F3CellSpec",
    "F3DatasetIdentity",
    "F3DatasetSpec",
    "F3FileIdentity",
    "F3FixedControlEvidence",
    "F3ModeComparisonConfig",
    "F3ModeComparisonPlan",
    "F3ScannerBackend",
    "F3ScannerConfig",
    "F3VotingControls",
    "F3WorkflowMode",
    "F3VolumeSource",
    "F3_DATASET_ID",
    "F3_FILE_ROLES",
    "OFFICIAL_F3_DATASET_SPEC",
    "build_f3d_mode_comparison_plan",
    "canonical_f3_cells",
    "ensure_output_not_in_data_root",
]
