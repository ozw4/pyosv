from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

import pyosv.qqual3d.profile as profile_module
from pyosv.evaluation.f3d_mode_comparison import (
    F3ModeComparisonConfig,
    build_f3d_mode_comparison_plan,
)
from pyosv.evaluation.f3d_mode_comparison.runner import (
    _DEFAULT_VARIANT,
    _volume_voting_controls,
)
from pyosv.f3d_reference import F3D_SHAPE
from pyosv.qqual3d import QQual3DProfile, resolve_qqual3d_profile


def test_positive_shape_resolves_fixed_qqual_profile() -> None:
    profile = resolve_qqual3d_profile(shape=(3, 4, 5))

    assert isinstance(profile, QQual3DProfile)
    assert profile.shape == (3, 4, 5)
    assert profile.scanner_backend == "quality"
    assert profile.workflow_mode == "quality"
    assert profile.scanner_thinning_mode == "reference"
    assert profile.voting_config.voter_thin_mode == "hybrid_v2"
    assert profile.skinning_enabled


@pytest.mark.parametrize(
    "shape",
    [(), (1, 2), (1, 2, 3, 4), [1, 2, 3]],
)
def test_shape_must_be_three_dimensional(shape: object) -> None:
    with pytest.raises(ValueError, match="three dimensions"):
        resolve_qqual3d_profile(shape=shape)  # type: ignore[arg-type]


@pytest.mark.parametrize("shape", [(0, 2, 3), (1, -2, 3), (1, True, 3), (1, 2.5, 3)])
def test_shape_dimensions_must_be_positive_integers(shape: object) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        resolve_qqual3d_profile(shape=shape)  # type: ignore[arg-type]


def test_profile_matches_canonical_f3_qqual_cell() -> None:
    profile = resolve_qqual3d_profile(shape=F3D_SHAPE)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())
    cell = next(cell for cell in plan.cells if cell.label == "Q-QUAL")
    scanner = plan.scanner_config_for(cell.scanner_backend)
    workflow = plan.workflow_settings_for(cell.workflow_mode)

    assert profile.scanner_backend == cell.scanner_backend
    assert profile.workflow_mode == cell.workflow_mode
    assert profile.scanner_angular_range == (
        (scanner.phi_min, scanner.phi_max),
        (scanner.theta_min, scanner.theta_max),
    )
    assert profile.scanner_sigmas == (scanner.sigma1, scanner.sigma2)
    assert profile.scanner_refinement_factor == scanner.refinement_factor
    assert profile.scanner_thinning_mode == scanner.scanner_thin_mode
    assert profile.scanner_edge_cleanup is scanner.effective_remove_edge_effects
    assert profile.voting_config == workflow.voting_config
    assert profile.voting_controls == _volume_voting_controls(plan)
    assert profile.skinning_config == workflow.skinning_config
    assert profile.variant == _DEFAULT_VARIANT

    skinning = profile.skinning_config.as_report_dict()
    assert skinning["method"] == "quality"
    assert skinning["adaptive_min_likelihood"] is True
    assert skinning["growth_source"] == "pre_thin"
    assert skinning["accepted_occupancy_radius"] == 1
    assert skinning["boundary_skinner_fallback"] is True


def test_disabling_skinning_changes_only_skinning_enabled_state() -> None:
    enabled = resolve_qqual3d_profile(shape=(3, 4, 5))
    disabled = resolve_qqual3d_profile(shape=(3, 4, 5), skinning_enabled=False)

    assert not disabled.skinning_enabled
    assert disabled == replace(
        enabled,
        skinning_config=replace(enabled.skinning_config, enabled=False),
    )


def test_production_profile_does_not_import_mode_comparison() -> None:
    assert "f3d_mode_comparison" not in inspect.getsource(profile_module)
