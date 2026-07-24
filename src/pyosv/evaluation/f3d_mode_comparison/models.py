"""Immutable models for canonical F3 full-volume comparison plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

from pyosv.f3d_reference import F3D_DTYPE, F3D_EXPECTED_BYTES, F3D_SHAPE

from ..synthetic_quality import (
    ResolvedWorkflowSettings,
    SyntheticSkinningConfig,
    resolve_workflow_settings,
)
from .config import F3ScannerConfig, F3VotingControls

F3ScannerBackend = Literal["reference-like", "quality"]
F3WorkflowMode = Literal["reference", "quality"]


@dataclass(frozen=True, slots=True)
class F3DatasetSpec:
    """Official input-volume metadata for the comparison."""

    shape: tuple[int, int, int] = F3D_SHAPE
    input_file: str = "ep.dat"
    dtype: str = F3D_DTYPE
    expected_bytes: int = F3D_EXPECTED_BYTES

    def __post_init__(self) -> None:
        if self.shape != F3D_SHAPE:
            raise ValueError(f"shape must be the official F3 shape {F3D_SHAPE}")
        if self.input_file != "ep.dat":
            raise ValueError("input_file must be 'ep.dat'")
        if self.dtype != F3D_DTYPE:
            raise ValueError(f"dtype must be {F3D_DTYPE!r}")
        if self.expected_bytes != F3D_EXPECTED_BYTES:
            raise ValueError(f"expected_bytes must be {F3D_EXPECTED_BYTES}")


@dataclass(frozen=True, slots=True)
class F3CellSpec:
    """One cell in the fixed scanner-backend/workflow matrix."""

    label: str
    scanner_backend: F3ScannerBackend
    workflow_mode: F3WorkflowMode

    def __post_init__(self) -> None:
        if self.scanner_backend not in {"reference-like", "quality"}:
            raise ValueError(f"unknown scanner backend: {self.scanner_backend}")
        if self.workflow_mode not in {"reference", "quality"}:
            raise ValueError(f"unknown workflow mode: {self.workflow_mode}")
        backend_label = "RL" if self.scanner_backend == "reference-like" else "Q"
        workflow_label = "REF" if self.workflow_mode == "reference" else "QUAL"
        expected = f"{backend_label}-{workflow_label}"
        if self.label != expected:
            raise ValueError(
                f"cell label {self.label!r} is inconsistent with its axes; expected {expected!r}"
            )


def canonical_f3_cells() -> tuple[F3CellSpec, ...]:
    """Return the four publication cells in their fixed order."""

    return (
        F3CellSpec("RL-REF", "reference-like", "reference"),
        F3CellSpec("RL-QUAL", "reference-like", "quality"),
        F3CellSpec("Q-REF", "quality", "reference"),
        F3CellSpec("Q-QUAL", "quality", "quality"),
    )


@dataclass(frozen=True, slots=True)
class F3FixedControlEvidence:
    """Resolved evidence for controls fixed outside the two matrix axes."""

    scanner_thin_mode: str
    requested_remove_edge_effects: bool
    effective_remove_edge_effects: bool
    refinement_factor: int
    voting_controls: F3VotingControls
    full_volume_evaluation_units: int = 1


@dataclass(frozen=True, slots=True)
class F3ModeComparisonPlan:
    """Validated, execution-free canonical F3 comparison plan."""

    dataset_spec: F3DatasetSpec
    cells: tuple[F3CellSpec, ...]
    reference_like_scanner_config: F3ScannerConfig
    quality_scanner_config: F3ScannerConfig
    voting_controls: F3VotingControls
    skinning_template: SyntheticSkinningConfig
    skinning_enabled: bool
    boundary_diagnostic_margin: int
    skinner_method_explicit: bool
    skinner_min_likelihood_explicit: bool
    skinner_growth_source_explicit: bool
    skinner_accepted_occupancy_radius_explicit: bool
    skinner_boundary_fallback_explicit: bool
    reference_workflow_settings: ResolvedWorkflowSettings
    quality_workflow_settings: ResolvedWorkflowSettings
    fixed_control_evidence: F3FixedControlEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_spec, F3DatasetSpec):
            raise ValueError("dataset_spec must be an F3DatasetSpec")
        if self.cells != canonical_f3_cells():
            raise ValueError("cells must match the canonical F3 cells and order")
        if self.reference_like_scanner_config.backend != "reference-like":
            raise ValueError("reference_like_scanner_config must use backend 'reference-like'")
        if self.quality_scanner_config.backend != "quality":
            raise ValueError("quality_scanner_config must use backend 'quality'")
        if (
            replace(
                self.quality_scanner_config,
                backend="reference-like",
            )
            != self.reference_like_scanner_config
        ):
            raise ValueError("scanner configs may differ only by canonical scanner backend")
        _validate_canonical_scanner(self.reference_like_scanner_config)
        _validate_canonical_scanner(self.quality_scanner_config)
        if not isinstance(self.voting_controls, F3VotingControls):
            raise ValueError("voting_controls must be F3VotingControls")
        if not isinstance(self.skinning_template, SyntheticSkinningConfig):
            raise ValueError("skinning_template must be a SyntheticSkinningConfig")
        if not isinstance(self.skinning_enabled, bool):
            raise ValueError("skinning_enabled must be a bool")
        if (
            isinstance(self.boundary_diagnostic_margin, bool)
            or not isinstance(self.boundary_diagnostic_margin, int)
            or self.boundary_diagnostic_margin < 0
        ):
            raise ValueError("boundary_diagnostic_margin must be a non-negative integer")
        for name in (
            "skinner_method_explicit",
            "skinner_min_likelihood_explicit",
            "skinner_growth_source_explicit",
            "skinner_accepted_occupancy_radius_explicit",
            "skinner_boundary_fallback_explicit",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        expected_reference, expected_quality = _expected_workflows(self)
        if self.reference_workflow_settings != expected_reference:
            raise ValueError("reference_workflow_settings must match the plan controls")
        if self.quality_workflow_settings != expected_quality:
            raise ValueError("quality_workflow_settings must match the plan controls")
        if (
            self.reference_workflow_settings.include_thinning_diagnostic
            or self.quality_workflow_settings.include_thinning_diagnostic
        ):
            raise ValueError("diagnostic workflows are not canonical F3 cells")
        expected_evidence = F3FixedControlEvidence(
            scanner_thin_mode="reference",
            requested_remove_edge_effects=True,
            effective_remove_edge_effects=True,
            refinement_factor=2,
            voting_controls=self.voting_controls,
        )
        if self.fixed_control_evidence != expected_evidence:
            raise ValueError("fixed_control_evidence must match the canonical plan controls")

    def scanner_config_for(self, backend: F3ScannerBackend) -> F3ScannerConfig:
        """Return the resolved scanner config for a canonical backend."""

        if backend == "reference-like":
            return self.reference_like_scanner_config
        if backend == "quality":
            return self.quality_scanner_config
        raise ValueError(f"unknown scanner backend: {backend}")

    def workflow_settings_for(self, workflow_mode: F3WorkflowMode) -> ResolvedWorkflowSettings:
        """Return the resolved settings for a canonical workflow."""

        if workflow_mode == "reference":
            return self.reference_workflow_settings
        if workflow_mode == "quality":
            return self.quality_workflow_settings
        raise ValueError(f"unknown workflow mode: {workflow_mode}")

    def as_dict(self) -> dict[str, Any]:
        """Return deterministic manifest-ready plan data."""

        return asdict(self)


def _validate_canonical_scanner(scanner: F3ScannerConfig) -> None:
    if scanner.scanner_thin_mode != "reference":
        raise ValueError("scanner_thin_mode must be 'reference'")
    if not scanner.remove_edge_effects:
        raise ValueError("remove_edge_effects must be True")
    if not scanner.effective_remove_edge_effects:
        raise ValueError("effective_remove_edge_effects must be True")
    if scanner.refinement_factor != 2:
        raise ValueError("refinement_factor must be 2")


def _expected_workflows(
    plan: F3ModeComparisonPlan,
) -> tuple[ResolvedWorkflowSettings, ResolvedWorkflowSettings]:
    effective_skinning = replace(
        plan.skinning_template,
        enabled=plan.skinning_enabled,
    )
    common = {
        "skinning_config": effective_skinning,
        "skinner_method_explicit": plan.skinner_method_explicit,
        "skinner_min_likelihood_explicit": (plan.skinner_min_likelihood_explicit),
        "skinner_growth_source_explicit": (plan.skinner_growth_source_explicit),
        "skinner_accepted_occupancy_radius_explicit": (
            plan.skinner_accepted_occupancy_radius_explicit
        ),
        "skinner_boundary_fallback_explicit": (plan.skinner_boundary_fallback_explicit),
    }
    reference = resolve_workflow_settings(
        workflow_mode="reference",
        voting_config=plan.voting_controls.to_voting_config(voter_thin_mode="reference"),
        **common,
    )
    quality = resolve_workflow_settings(
        workflow_mode="quality",
        voting_config=plan.voting_controls.to_voting_config(voter_thin_mode="hybrid_v2"),
        **common,
    )
    return reference, quality
