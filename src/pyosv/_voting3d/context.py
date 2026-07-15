"""Per-seed execution context for 3D surface-voting policies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyosv._voting3d.config import SurfaceVoterConfig
from pyosv._voting3d.models import _MaskedUVWBoxSamples, _ReferenceUVWBoxSamples
from pyosv.cells import FaultCell


@dataclass(frozen=True, slots=True)
class SurfaceVotingContext:
    """Inputs and facade-routed dependencies for processing one seed."""

    config: SurfaceVoterConfig
    cell: FaultCell
    ft: np.ndarray
    fe: np.ndarray
    vp: np.ndarray
    vt: np.ndarray
    vm: np.ndarray
    normal: np.ndarray
    dip: np.ndarray
    strike: np.ndarray
    sample_reference: Callable[..., np.ndarray]
    sample_reference_with_support: Callable[..., _ReferenceUVWBoxSamples]
    sample_masked: Callable[..., _MaskedUVWBoxSamples]
    find_surface: Callable[..., np.ndarray]
    find_surface_masked: Callable[..., tuple[np.ndarray | None, int]]
    score_reference: Callable[..., tuple[np.float32, int]]
    score_masked: Callable[..., tuple[np.float32, int, int]]
    accumulate_reference: Callable[..., Any]
    accumulate_masked: Callable[..., tuple[int, int, int]]
    surface_orientation: Callable[..., tuple[float, float]]
    count_reference_face_votes: Callable[..., int]
    select_supported_rectangle: Callable[..., Any]
    crop_masked_box: Callable[..., _MaskedUVWBoxSamples]
    surface_center_lag: Callable[[np.ndarray], float | None]
