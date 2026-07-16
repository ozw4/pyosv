"""Reusable configuration for synthetic quality evaluation."""

from __future__ import annotations

from typing import Any

from .config import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from .models import (
    OrientationField3D,
    PipelineArtifacts,
    PipelineEvaluation,
    PipelineStageTrace3D,
    SkinningResult3D,
    ThinningResult3D,
    VotingResult3D,
)
from .profiles import ResolvedWorkflowSettings, resolve_workflow_settings


def __getattr__(name: str) -> Any:
    if name in {"build_report", "run_case"}:
        from . import application

        return getattr(application, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SyntheticScannerConfig",
    "SyntheticSkinningConfig",
    "SyntheticTruthMetricConfig",
    "SyntheticVotingConfig",
    "OrientationField3D",
    "PipelineArtifacts",
    "PipelineEvaluation",
    "PipelineStageTrace3D",
    "ResolvedWorkflowSettings",
    "SkinningResult3D",
    "ThinningResult3D",
    "VotingResult3D",
    "build_report",
    "resolve_workflow_settings",
    "run_case",
]
