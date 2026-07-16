from __future__ import annotations

import pytest

from pyosv.evaluation.synthetic_quality import (
    ResolvedWorkflowSettings,
    resolve_workflow_settings,
)
from pyosv.evaluation.synthetic_quality.config import (
    SyntheticSkinningConfig,
    SyntheticVotingConfig,
)
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
    (
        "workflow_mode",
        "voter_mode",
        "skinner_method",
        "include_diagnostic",
        "expected_skinning",
    ),
    (
        (
            "reference",
            "reference",
            "reference",
            False,
            ("reference", 0.5, "thinned", None, False, "empty_primary"),
        ),
        (
            "quality",
            "hybrid_v2",
            "quality",
            False,
            ("quality", None, "pre_thin", 1, True, "empty_primary"),
        ),
        (
            "diagnostic",
            "reference",
            "reference",
            True,
            ("reference", 0.5, "thinned", None, False, "empty_primary"),
        ),
    ),
)
def test_workflow_profile_effective_defaults(
    workflow_mode: str,
    voter_mode: str,
    skinner_method: str,
    include_diagnostic: bool,
    expected_skinning: tuple[str, float | None, str, int | None, bool, str],
) -> None:
    # Workflow profiles resolve voter/skinner/diagnostic defaults only. Even though
    # both axes use "quality", a workflow mode is not a scanner backend preset and
    # cannot implicitly change the scanner backend or scanner_thin_mode.
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
    effective_skinning = _effective_skinning_config_for_workflow(
        workflow_mode=workflow_mode,
        skinning_config=SyntheticSkinningConfig(),
    )
    assert (
        effective_skinning.method,
        effective_skinning.min_likelihood,
        effective_skinning.growth_source,
        effective_skinning.accepted_occupancy_radius,
        effective_skinning.boundary_skinner_fallback,
        effective_skinning.boundary_skinner_fallback_policy,
    ) == expected_skinning


@pytest.mark.parametrize(
    ("workflow_mode", "voter_mode", "skinner_method", "include_diagnostic"),
    (
        ("reference", "reference", "reference", False),
        ("quality", "hybrid_v2", "quality", False),
        ("diagnostic", "reference", "reference", True),
    ),
)
def test_resolve_workflow_settings_effective_defaults(
    workflow_mode: str,
    voter_mode: str,
    skinner_method: str,
    include_diagnostic: bool,
) -> None:
    settings = resolve_workflow_settings(workflow_mode=workflow_mode)

    assert isinstance(settings, ResolvedWorkflowSettings)
    assert settings.workflow_mode == workflow_mode
    assert settings.voting_config.voter_thin_mode == voter_mode
    assert settings.voting_config.surface_support_min_fraction == 0.0
    assert settings.voting_config.surface_support_exponent == 0.0
    assert settings.skinning_config.method == skinner_method
    assert settings.include_thinning_diagnostic is include_diagnostic


def test_explicit_default_voting_config_is_not_a_workflow_default() -> None:
    explicit_config = SyntheticVotingConfig()

    default_settings = resolve_workflow_settings(workflow_mode="quality")
    explicit_settings = resolve_workflow_settings(
        workflow_mode="quality",
        voting_config=explicit_config,
    )

    assert default_settings.voting_config.voter_thin_mode == "hybrid_v2"
    assert explicit_settings.voting_config is explicit_config
    assert explicit_settings.voting_config.voter_thin_mode == "reference"


def test_resolve_workflow_settings_rejects_invalid_config_types() -> None:
    with pytest.raises(ValueError, match="voting_config must be"):
        resolve_workflow_settings(voting_config=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="skinning_config must be"):
        resolve_workflow_settings(skinning_config=object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("fallback", "explicit", "expected"),
    (
        (False, False, True),
        (True, True, True),
        (False, True, False),
    ),
)
def test_resolve_quality_boundary_fallback(
    fallback: bool,
    explicit: bool,
    expected: bool,
) -> None:
    settings = resolve_workflow_settings(
        workflow_mode="quality",
        skinning_config=SyntheticSkinningConfig(
            boundary_skinner_fallback=fallback,
            boundary_skinner_fallback_policy="degraded_primary",
        ),
        skinner_boundary_fallback_explicit=explicit,
    )

    assert settings.skinning_config.boundary_skinner_fallback is expected
    assert settings.skinning_config.boundary_skinner_fallback_policy == "degraded_primary"


@pytest.mark.parametrize("workflow_mode", ("reference", "diagnostic"))
def test_resolve_non_quality_workflow_preserves_skinning_config(
    workflow_mode: str,
) -> None:
    skinning_config = SyntheticSkinningConfig(
        method="connected_component",
        boundary_skinner_fallback=True,
        boundary_skinner_fallback_policy="degraded_primary",
    )

    settings = resolve_workflow_settings(
        workflow_mode=workflow_mode,
        skinning_config=skinning_config,
    )

    assert settings.skinning_config is skinning_config


def test_validate_workflow_mode_rejects_unknown_mode() -> None:
    assert WORKFLOW_MODES == ("reference", "quality", "diagnostic")
    with pytest.raises(ValueError, match="workflow_mode must be one of"):
        _validate_workflow_mode("missing")


def test_explicit_values_take_priority_over_quality_defaults() -> None:
    config = SyntheticSkinningConfig(
        method="quality",
        min_likelihood=0.7,
        growth_source="thinned",
        accepted_occupancy_radius=None,
        boundary_skinner_fallback=False,
        boundary_skinner_fallback_policy="degraded_primary",
    )
    effective = _effective_skinning_config_for_workflow(
        workflow_mode="quality",
        skinning_config=config,
        skinner_method_explicit=True,
        skinner_min_likelihood_explicit=True,
        skinner_growth_source_explicit=True,
        skinner_accepted_occupancy_radius_explicit=True,
        skinner_boundary_fallback_explicit=True,
    )

    assert effective.method == "quality"
    assert effective.min_likelihood == 0.7
    assert effective.growth_source == "thinned"
    assert effective.accepted_occupancy_radius is None
    assert effective.boundary_skinner_fallback is False
    assert effective.boundary_skinner_fallback_policy == "degraded_primary"


@pytest.mark.parametrize("fallback", (False, True))
def test_explicit_boundary_fallback_takes_priority_over_quality_default(
    fallback: bool,
) -> None:
    effective = _effective_skinning_config_for_workflow(
        workflow_mode="quality",
        skinning_config=SyntheticSkinningConfig(boundary_skinner_fallback=fallback),
        skinner_boundary_fallback_explicit=True,
    )

    assert effective.boundary_skinner_fallback is fallback


@pytest.mark.parametrize("workflow_mode", ("reference", "diagnostic"))
def test_non_quality_workflow_preserves_boundary_fallback(
    workflow_mode: str,
) -> None:
    config = SyntheticSkinningConfig(
        boundary_skinner_fallback=True,
        boundary_skinner_fallback_policy="degraded_primary",
    )

    assert (
        _effective_skinning_config_for_workflow(
            workflow_mode=workflow_mode,
            skinning_config=config,
        )
        is config
    )


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
