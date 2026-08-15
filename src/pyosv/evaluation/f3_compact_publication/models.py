"""Immutable source identities for the F3 compact publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from ..f3d_mode_comparison.runner import F3CellReference
from ..mode_comparison_publication.models import F3SourceBundle


@dataclass(frozen=True, slots=True)
class AmplitudeIdentity:
    """Identity of the external amplitude volume used for visualization."""

    role: str
    filename: str
    resolved_path: Path
    shape: tuple[int, int, int]
    storage_dtype: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class StageSource:
    """Public-reference and Q-QUAL artifact handles for one processing stage."""

    stage: str
    public_reference_role: str
    public_reference_filename: str
    public_reference_path: Path
    public_reference_sha256: str
    candidate_source_kind: str
    candidate_fingerprint: str
    candidate_filename: str
    candidate_path: Path


@dataclass(frozen=True, slots=True)
class RidgeStageThresholds:
    """Source-recorded thresholds exposed for one compact comparison stage."""

    stage: str
    public_reference_threshold: float
    q_qual_threshold: float


@dataclass(frozen=True, slots=True)
class SourceRidgeThresholdContract:
    """Q-QUAL-only view of the validated source ridge threshold evidence."""

    selection: str
    percentile: float
    buffer_radius: float
    stages: tuple[RidgeStageThresholds, ...]


@dataclass(frozen=True, slots=True)
class SelectedSection:
    """One spatial section selected from an equal axis bin."""

    section_group: str
    axis: str
    bin_index: int
    index: int
    policy: str
    ridge_count_score: int


@dataclass(frozen=True, slots=True)
class CompactSourceContext:
    """Validated read-only inputs needed by compact publication derivation."""

    f3: F3SourceBundle
    amplitude: AmplitudeIdentity
    q_qual_cell: F3CellReference
    stage_sources: tuple[StageSource, ...]
    ridge_threshold_contract: SourceRidgeThresholdContract
    selected_sections: tuple[SelectedSection, ...]


__all__ = [
    "AmplitudeIdentity",
    "CompactSourceContext",
    "RidgeStageThresholds",
    "SelectedSection",
    "SourceRidgeThresholdContract",
    "StageSource",
]
