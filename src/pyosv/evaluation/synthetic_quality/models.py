"""Typed stage results for the synthetic-quality pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _volume3(name: str, value: np.ndarray, shape: tuple[int, int, int] | None = None) -> None:
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array")
    if array.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True, slots=True)
class OrientationField3D:
    ft: np.ndarray
    pt: np.ndarray
    tt: np.ndarray
    confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        _volume3("ft", self.ft)
        shape = self.ft.shape
        _volume3("pt", self.pt, shape)
        _volume3("tt", self.tt, shape)
        if self.confidence is not None:
            _volume3("confidence", self.confidence, shape)


@dataclass(frozen=True, slots=True)
class VotingResult3D:
    fv: np.ndarray
    vp: np.ndarray
    vt: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        _volume3("fv", self.fv)
        _volume3("vp", self.vp, self.fv.shape)
        _volume3("vt", self.vt, self.fv.shape)


@dataclass(frozen=True, slots=True)
class ThinningResult3D:
    fvt: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        _volume3("fvt", self.fvt)


@dataclass(frozen=True, slots=True)
class SkinningResult3D:
    skins: Sequence[Any]
    mask: np.ndarray
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask)
        if mask.ndim != 3:
            raise ValueError("mask must be a 3D array")
        if mask.dtype != np.bool_:
            raise TypeError("mask must have dtype bool")


@dataclass(frozen=True, slots=True)
class PipelineStageTrace3D:
    """Boolean masks captured from one concrete pipeline execution."""

    seed_candidate_mask: np.ndarray
    seed_selected_mask: np.ndarray
    fv_positive_mask: np.ndarray
    fvt_positive_mask: np.ndarray
    primary_skin_mask: np.ndarray
    fallback_skin_mask: np.ndarray
    final_skin_mask: np.ndarray
    skinning_enabled: bool
    fallback_used: bool

    def __post_init__(self) -> None:
        mask_names = (
            "seed_candidate_mask",
            "seed_selected_mask",
            "fv_positive_mask",
            "fvt_positive_mask",
            "primary_skin_mask",
            "fallback_skin_mask",
            "final_skin_mask",
        )
        shape: tuple[int, ...] | None = None
        for name in mask_names:
            mask = np.asarray(getattr(self, name))
            if mask.ndim != 3:
                raise ValueError(f"{name} must be a 3D array")
            if mask.dtype != np.bool_:
                raise TypeError(f"{name} must have dtype bool")
            if shape is None:
                shape = mask.shape
            elif mask.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {mask.shape}")
            object.__setattr__(self, name, mask.copy())

        if not isinstance(self.skinning_enabled, (bool, np.bool_)):
            raise TypeError("skinning_enabled must be bool")
        if not isinstance(self.fallback_used, (bool, np.bool_)):
            raise TypeError("fallback_used must be bool")
        if not self.skinning_enabled:
            if self.fallback_used:
                raise ValueError("fallback_used must be false when skinning is disabled")
            if any(
                np.any(getattr(self, name))
                for name in ("primary_skin_mask", "fallback_skin_mask", "final_skin_mask")
            ):
                raise ValueError("skin masks must be empty when skinning is disabled")
        if self.fallback_used:
            if not np.array_equal(self.fallback_skin_mask, self.final_skin_mask):
                raise ValueError(
                    "fallback_skin_mask must equal final_skin_mask when fallback is used"
                )
        elif np.any(self.fallback_skin_mask):
            raise ValueError("fallback_skin_mask must be empty when fallback is not used")


@dataclass(frozen=True, slots=True)
class PipelineArtifacts:
    volumes: Mapping[str, Any]
    skins_payload: Mapping[str, Any]
    stage_trace: PipelineStageTrace3D | None = None


@dataclass(frozen=True, slots=True)
class PipelineEvaluation:
    report_payload: Mapping[str, Any]
    artifacts: PipelineArtifacts
