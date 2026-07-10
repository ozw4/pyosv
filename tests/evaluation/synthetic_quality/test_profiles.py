from __future__ import annotations

import pytest

from pyosv.evaluation.synthetic_quality.config import SyntheticSkinningConfig
from pyosv.evaluation.synthetic_quality.profiles import (
    WORKFLOW_MODES,
    _effective_include_thinning_diagnostic,
    _effective_skinner_method,
    _effective_skinner_min_likelihood,
    _effective_skinning_config_for_workflow,
    _effective_surface_support_policy,
    _effective_voter_thin_mode,
    _validate_workflow_mode,
)


@pytest.mark.parametrize(
    ("workflow_mode", "voter_mode", "skinner_method", "include_diagnostic"),
    (
        ("reference", "reference", "reference", False),
        ("quality", "hybrid_v2", "quality", False),
        ("diagnostic", "reference", "reference", True),
    ),
)
def test_workflow_profile_defaults(
    workflow_mode: str,
    voter_mode: str,
    skinner_method: str,
    include_diagnostic: bool,
) -> None:
    assert _validate_workflow_mode(workflow_mode) == workflow_mode
    assert (
        _effective_voter_thin_mode(workflow_mode=workflow_mode, voter_thin_mode=None) == voter_mode
    )
    assert (
        _effective_skinner_method(workflow_mode=workflow_mode, skinner_method=None)
        == skinner_method
    )
    assert _effective_skinner_min_likelihood(
        skinner_method=skinner_method,
        min_likelihood=None,
    ) == (None if skinner_method == "quality" else 0.5)
    assert _effective_surface_support_policy(
        workflow_mode=workflow_mode,
        min_fraction=None,
        exponent=None,
    ) == (0.0, 0.0)
    assert (
        _effective_include_thinning_diagnostic(
            workflow_mode=workflow_mode,
            include_thinning_diagnostic=False,
        )
        is include_diagnostic
    )


def test_validate_workflow_mode_rejects_unknown_mode() -> None:
    assert WORKFLOW_MODES == ("reference", "quality", "diagnostic")
    with pytest.raises(ValueError, match="workflow_mode must be one of"):
        _validate_workflow_mode("missing")


def test_quality_workflow_resolves_skinning_defaults() -> None:
    effective = _effective_skinning_config_for_workflow(
        workflow_mode="quality",
        skinning_config=SyntheticSkinningConfig(),
    )

    assert effective.method == "quality"
    assert effective.min_likelihood is None
    assert effective.growth_source == "pre_thin"
    assert effective.accepted_occupancy_radius == 1
    assert effective.boundary_skinner_fallback is True


def test_explicit_values_take_priority_over_quality_defaults() -> None:
    config = SyntheticSkinningConfig(
        method="quality",
        min_likelihood=0.7,
        growth_source="thinned",
        accepted_occupancy_radius=None,
    )
    effective = _effective_skinning_config_for_workflow(
        workflow_mode="quality",
        skinning_config=config,
        skinner_method_explicit=True,
        skinner_min_likelihood_explicit=True,
        skinner_growth_source_explicit=True,
        skinner_accepted_occupancy_radius_explicit=True,
    )

    assert effective.method == "quality"
    assert effective.min_likelihood == 0.7
    assert effective.growth_source == "thinned"
    assert effective.accepted_occupancy_radius is None


def test_explicit_reference_skinner_is_not_overridden_by_quality_workflow() -> None:
    config = SyntheticSkinningConfig(method="reference", min_likelihood=0.6)

    assert (
        _effective_skinning_config_for_workflow(
            workflow_mode="quality",
            skinning_config=config,
            skinner_method_explicit=True,
        )
        is config
    )


def test_explicit_profile_overrides_take_priority() -> None:
    assert _effective_voter_thin_mode(workflow_mode="quality", voter_thin_mode="normal") == "normal"
    assert _effective_skinner_method(workflow_mode="quality", skinner_method="reference") == (
        "reference"
    )
    assert _effective_skinner_min_likelihood(
        skinner_method="quality", min_likelihood=0.25
    ) == pytest.approx(0.25)
    assert _effective_surface_support_policy(
        workflow_mode="quality", min_fraction=0.4, exponent=2.0
    ) == (0.4, 2.0)
    assert _effective_include_thinning_diagnostic(
        workflow_mode="reference", include_thinning_diagnostic=True
    )
