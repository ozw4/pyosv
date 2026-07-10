"""Workflow profile resolution for synthetic quality evaluation."""

from __future__ import annotations

from dataclasses import replace

from .config import SyntheticSkinningConfig

WORKFLOW_MODES = ("reference", "quality", "diagnostic")


def _validate_workflow_mode(value: str) -> str:
    if value not in WORKFLOW_MODES:
        raise ValueError("workflow_mode must be one of: " + ", ".join(WORKFLOW_MODES))
    return value


def _default_voter_thin_mode_for_workflow(workflow_mode: str) -> str:
    return "hybrid_v2" if workflow_mode == "quality" else "reference"


def _default_surface_support_policy_for_workflow(
    workflow_mode: str,
) -> tuple[float, float]:
    return 0.0, 0.0


def _default_skinner_method_for_workflow(workflow_mode: str) -> str:
    return "quality" if workflow_mode == "quality" else "reference"


def _default_skinner_min_likelihood_for_method(method: str) -> float | None:
    return None if method == "quality" else 0.5


def _effective_skinner_method(*, workflow_mode: str, skinner_method: str | None) -> str:
    if skinner_method is not None:
        return skinner_method
    return _default_skinner_method_for_workflow(workflow_mode)


def _effective_skinner_min_likelihood(
    *,
    skinner_method: str,
    min_likelihood: float | None,
) -> float | None:
    if min_likelihood is not None:
        return min_likelihood
    return _default_skinner_min_likelihood_for_method(skinner_method)


def _effective_skinning_config_for_workflow(
    *,
    workflow_mode: str,
    skinning_config: SyntheticSkinningConfig,
    skinner_method_explicit: bool = False,
    skinner_min_likelihood_explicit: bool = False,
    skinner_growth_source_explicit: bool = False,
    skinner_accepted_occupancy_radius_explicit: bool = False,
) -> SyntheticSkinningConfig:
    if workflow_mode != "quality" or skinning_config.method not in {"reference", "quality"}:
        return skinning_config
    if skinner_method_explicit and skinning_config.method != "quality":
        return skinning_config
    defaults = SyntheticSkinningConfig()
    min_likelihood = skinning_config.min_likelihood
    if not skinner_min_likelihood_explicit and min_likelihood == defaults.min_likelihood:
        min_likelihood = None
    accepted_occupancy_radius = skinning_config.accepted_occupancy_radius
    if (
        not skinner_accepted_occupancy_radius_explicit
        and accepted_occupancy_radius == defaults.accepted_occupancy_radius
    ):
        accepted_occupancy_radius = 1
    growth_source = skinning_config.growth_source
    if not skinner_growth_source_explicit and growth_source == defaults.growth_source:
        growth_source = "pre_thin"
    return replace(
        skinning_config,
        method="quality",
        min_likelihood=min_likelihood,
        accepted_occupancy_radius=accepted_occupancy_radius,
        growth_source=growth_source,
        boundary_skinner_fallback=True,
    )


def _effective_voter_thin_mode(*, workflow_mode: str, voter_thin_mode: str | None) -> str:
    if voter_thin_mode is not None:
        return voter_thin_mode
    return _default_voter_thin_mode_for_workflow(workflow_mode)


def _effective_surface_support_policy(
    *,
    workflow_mode: str,
    min_fraction: float | None,
    exponent: float | None,
) -> tuple[float, float]:
    default_min_fraction, default_exponent = _default_surface_support_policy_for_workflow(
        workflow_mode
    )
    if min_fraction is not None:
        default_min_fraction = min_fraction
    if exponent is not None:
        default_exponent = exponent
    return default_min_fraction, default_exponent


def _effective_include_thinning_diagnostic(
    *,
    workflow_mode: str,
    include_thinning_diagnostic: bool,
) -> bool:
    return include_thinning_diagnostic or workflow_mode == "diagnostic"
