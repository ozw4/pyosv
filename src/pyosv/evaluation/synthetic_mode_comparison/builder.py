"""Builder for canonical synthetic mode-comparison plans."""

from __future__ import annotations

from ..synthetic_quality import resolve_workflow_settings
from ..synthetic_quality.cases import (
    EXTENDED_CASES,
    SyntheticQualityCaseDefinition,
    validate_case_ids,
    validate_case_set,
)
from .config import SyntheticModeComparisonConfig
from .models import SyntheticModeComparisonPlan, canonical_mode_cells
from .trials import expand_synthetic_trials


def build_mode_comparison_plan(
    config: SyntheticModeComparisonConfig,
) -> SyntheticModeComparisonPlan:
    """Validate ``config`` and return the deterministic canonical plan."""

    if not isinstance(config, SyntheticModeComparisonConfig):
        raise ValueError("config must be a SyntheticModeComparisonConfig")
    _validate_scanner_template(config)
    case_definitions = _resolve_case_definitions(config)
    case_ids = tuple(definition.case_id for definition in case_definitions)
    workflow_kwargs = {
        "voting_config": config.voting_config,
        "skinning_config": config.skinning_config,
        "skinner_method_explicit": config.skinner_method_explicit,
        "skinner_min_likelihood_explicit": config.skinner_min_likelihood_explicit,
        "skinner_growth_source_explicit": config.skinner_growth_source_explicit,
        "skinner_accepted_occupancy_radius_explicit": (
            config.skinner_accepted_occupancy_radius_explicit
        ),
        "skinner_boundary_fallback_explicit": config.skinner_boundary_fallback_explicit,
    }
    reference_settings = resolve_workflow_settings(workflow_mode="reference", **workflow_kwargs)
    quality_settings = resolve_workflow_settings(workflow_mode="quality", **workflow_kwargs)
    return SyntheticModeComparisonPlan(
        case_ids=case_ids,
        trial_seeds=config.trial_seeds,
        trials=expand_synthetic_trials(case_definitions, config.trial_seeds, config.shape),
        shape=config.shape,
        scanner_template=config.scanner_template,
        voting_config=config.voting_config,
        skinning_config=config.skinning_config,
        truth_metric_config=config.truth_metric_config,
        include_oracle_workflow_isolation=config.include_oracle_workflow_isolation,
        comparison_variant=config.comparison_variant,
        skinner_method_explicit=config.skinner_method_explicit,
        skinner_min_likelihood_explicit=config.skinner_min_likelihood_explicit,
        skinner_growth_source_explicit=config.skinner_growth_source_explicit,
        skinner_accepted_occupancy_radius_explicit=(
            config.skinner_accepted_occupancy_radius_explicit
        ),
        skinner_boundary_fallback_explicit=config.skinner_boundary_fallback_explicit,
        reference_workflow_settings=reference_settings,
        quality_workflow_settings=quality_settings,
        cells=canonical_mode_cells(
            include_oracle_workflow_isolation=config.include_oracle_workflow_isolation
        ),
    )


def _resolve_case_definitions(
    config: SyntheticModeComparisonConfig,
) -> tuple[SyntheticQualityCaseDefinition, ...]:
    if config.case_ids is not None:
        case_ids = validate_case_ids(config.case_ids)
        definitions_by_id = {definition.case_id: definition for definition in EXTENDED_CASES}
        return tuple(definitions_by_id[case_id] for case_id in case_ids)
    return validate_case_set(config.case_set)


def _validate_scanner_template(config: SyntheticModeComparisonConfig) -> None:
    scanner = config.scanner_template
    if scanner.scanner_thin_mode != "reference":
        raise ValueError("scanner_template.scanner_thin_mode must be 'reference'")
    if not scanner.remove_edge_effects:
        raise ValueError("scanner_template.remove_edge_effects must be True")
    if scanner.refinement_factor != 2:
        raise ValueError("scanner_template.refinement_factor must be 2")
    if scanner.backend != "reference-like":
        raise ValueError("scanner_template.backend must be 'reference-like'")
