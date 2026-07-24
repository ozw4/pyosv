"""Builder for the canonical F3 full-volume comparison plan."""

from __future__ import annotations

from dataclasses import replace

from pyosv.f3d_reference import F3D_SHAPE

from ..synthetic_quality import resolve_workflow_settings
from .config import F3ModeComparisonConfig
from .models import (
    F3DatasetSpec,
    F3FixedControlEvidence,
    F3ModeComparisonPlan,
    canonical_f3_cells,
)


def build_f3d_mode_comparison_plan(
    config: F3ModeComparisonConfig,
) -> F3ModeComparisonPlan:
    """Validate ``config`` and return the deterministic canonical plan."""

    if not isinstance(config, F3ModeComparisonConfig):
        raise ValueError("config must be an F3ModeComparisonConfig")
    if config.shape != F3D_SHAPE:
        raise ValueError(f"shape must be the official F3 shape {F3D_SHAPE}")
    if config.input_file != "ep.dat":
        raise ValueError("input_file must be 'ep.dat'")

    scanner = config.scanner_template
    if scanner.backend != "reference-like":
        raise ValueError("scanner_template.backend must be 'reference-like'")
    if scanner.scanner_thin_mode != "reference":
        raise ValueError("scanner_template.scanner_thin_mode must be 'reference'")
    if not scanner.remove_edge_effects:
        raise ValueError("scanner_template.remove_edge_effects must be True")
    if scanner.refinement_factor != 2:
        raise ValueError("scanner_template.refinement_factor must be 2")
    if scanner.reference_thin_sigma != config.voting_controls.reference_thin_sigma:
        raise ValueError("scanner and voting reference_thin_sigma must match")

    effective_skinning = replace(
        config.skinning_template,
        enabled=config.skinning_enabled,
    )
    workflow_common = {
        "skinning_config": effective_skinning,
        "skinner_method_explicit": config.skinner_method_explicit,
        "skinner_min_likelihood_explicit": (config.skinner_min_likelihood_explicit),
        "skinner_growth_source_explicit": (config.skinner_growth_source_explicit),
        "skinner_accepted_occupancy_radius_explicit": (
            config.skinner_accepted_occupancy_radius_explicit
        ),
        "skinner_boundary_fallback_explicit": (config.skinner_boundary_fallback_explicit),
    }
    reference_settings = resolve_workflow_settings(
        workflow_mode="reference",
        voting_config=config.voting_controls.to_voting_config(voter_thin_mode="reference"),
        **workflow_common,
    )
    quality_settings = resolve_workflow_settings(
        workflow_mode="quality",
        voting_config=config.voting_controls.to_voting_config(voter_thin_mode="hybrid_v2"),
        **workflow_common,
    )
    return F3ModeComparisonPlan(
        dataset_spec=F3DatasetSpec(
            shape=config.shape,
            input_file=config.input_file,
        ),
        cells=canonical_f3_cells(),
        reference_like_scanner_config=scanner,
        quality_scanner_config=replace(scanner, backend="quality"),
        voting_controls=config.voting_controls,
        skinning_template=config.skinning_template,
        skinning_enabled=config.skinning_enabled,
        boundary_diagnostic_margin=config.boundary_diagnostic_margin,
        skinner_method_explicit=config.skinner_method_explicit,
        skinner_min_likelihood_explicit=(config.skinner_min_likelihood_explicit),
        skinner_growth_source_explicit=(config.skinner_growth_source_explicit),
        skinner_accepted_occupancy_radius_explicit=(
            config.skinner_accepted_occupancy_radius_explicit
        ),
        skinner_boundary_fallback_explicit=(config.skinner_boundary_fallback_explicit),
        reference_workflow_settings=reference_settings,
        quality_workflow_settings=quality_settings,
        fixed_control_evidence=F3FixedControlEvidence(
            scanner_thin_mode=scanner.scanner_thin_mode,
            requested_remove_edge_effects=scanner.remove_edge_effects,
            effective_remove_edge_effects=(scanner.effective_remove_edge_effects),
            refinement_factor=scanner.refinement_factor,
            voting_controls=config.voting_controls,
        ),
    )
