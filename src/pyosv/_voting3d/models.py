"""Internal data models for 3D optimal-surface voting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class _MaskedUVWBoxSamples:
    """Seed-local samples plus validity and full-box coordinate metadata."""

    costs: np.ndarray
    valid_lag_mask: np.ndarray
    w_offset: int
    v_offset: int
    full_tangential_shape: tuple[int, int]
    admissible_lag_count: int
    in_bounds_lag_count: int


@dataclass(frozen=True, slots=True)
class _TangentialRectangle:
    """Half-open full-box indices for a supported origin-containing rectangle."""

    w_start: int
    v_start: int
    w_stop: int
    v_stop: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.w_stop - self.w_start, self.v_stop - self.v_start)

    @property
    def size(self) -> int:
        nw, nv = self.shape
        return nw * nv


@dataclass(frozen=True, slots=True)
class _SurfaceVotingDiagnostic:
    """Compact immutable outcome for one seed's surface-voting attempt."""

    seed_index: tuple[int, int, int]
    policy: str
    full_tangential_column_count: int
    selected_tangential_column_count: int
    admissible_lag_count: int
    in_bounds_lag_count: int
    support_fraction: float
    surface_center_lag: float | None
    surface_projection_count: int
    selected_invalid_sample_count: int
    center_vote_write_count: int
    face_center_vote_count: int
    orientation_source: str | None
    skipped: bool
    skip_reason: str | None
