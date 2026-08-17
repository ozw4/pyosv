from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np
import pytest

import pyosv.qqual3d as qqual3d
import pyosv.qqual3d.runner as runner_module
from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.evaluation.workflow3d import Workflow3DResult, execute_workflow3d
from pyosv.orient3d import FaultOrientScanner3
from pyosv.qqual3d import (
    QQual3DProfile,
    QQual3DResult,
    resolve_qqual3d_profile,
    run_qqual3d,
)
from pyosv.synthetic_metrics import skin_mask_from_skins


@pytest.fixture(scope="module")
def direct_chain() -> tuple[
    np.ndarray,
    np.ndarray,
    QQual3DResult,
    np.ndarray,
    Workflow3DResult,
]:
    shape = (11, 15, 17)
    _, i2, i1 = np.indices(shape, dtype=np.float32)
    ep = (np.exp(-0.5 * (i2 - 7.0) ** 2) * (1.0 + 0.05 * i1)).astype(np.float64)
    original = ep.copy()
    result = run_qqual3d(ep)

    profile = resolve_qqual3d_profile(shape=shape)
    scanner_input = np.array(ep, dtype=np.float32, copy=True, order="C")
    scanner = FaultOrientScanner3(profile.sigma1, profile.sigma2)
    ft_scan, pt_scan, tt_scan = scanner.scan_quality(
        profile.phi_min,
        profile.phi_max,
        profile.theta_min,
        profile.theta_max,
        scanner_input,
        refinement_factor=profile.scanner_refinement_factor,
    )
    ft, pt, tt = scanner.thin(
        ft_scan,
        pt_scan,
        tt_scan,
        mode=profile.scanner_thin_mode,
        reference_sigma=profile.voting_config.reference_thin_sigma,
        remove_edge_effects=profile.remove_edge_effects,
    )
    workflow = execute_workflow3d(
        ft=ft,
        pt=pt,
        tt=tt,
        attribute_identity=None,
        voting_settings=profile.voting_config,
        voting_controls=profile.voting_controls,
        skinning_settings=profile.skinning_config,
        variant_spec=profile.variant,
        scanner_target_positive_mask=quality_metrics.positive_candidate_mask(ft_scan),
        fvt_recenter_target=ft,
        fvt_recenter_target_source="scanner_fet",
    )
    return ep, original, result, ft_scan, workflow


def test_finite_3d_input_runs_without_mutating_input(
    direct_chain: tuple[
        np.ndarray,
        np.ndarray,
        QQual3DResult,
        np.ndarray,
        Workflow3DResult,
    ],
) -> None:
    ep, original, result, _, _ = direct_chain

    assert isinstance(result, QQual3DResult)
    np.testing.assert_array_equal(ep, original)


def test_result_arrays_are_owned_with_expected_shape_and_dtype(
    direct_chain: tuple[
        np.ndarray,
        np.ndarray,
        QQual3DResult,
        np.ndarray,
        Workflow3DResult,
    ],
) -> None:
    ep, _, result, _, _ = direct_chain

    for array in (result.ft, result.fv, result.fvt):
        assert array.shape == ep.shape
        assert array.dtype == np.dtype(np.float32)
        assert array.dtype.isnative
        assert array.flags.owndata
        assert not array.flags.writeable
        assert np.all(np.isfinite(array))
    assert result.skin_mask.shape == ep.shape
    assert result.skin_mask.dtype == np.dtype(bool)
    assert result.skin_mask.flags.owndata
    assert not result.skin_mask.flags.writeable
    assert isinstance(result.diagnostics, Mapping)


def test_run_matches_direct_existing_algorithm_chain(
    direct_chain: tuple[
        np.ndarray,
        np.ndarray,
        QQual3DResult,
        np.ndarray,
        Workflow3DResult,
    ],
) -> None:
    _, _, result, ft_scan, workflow = direct_chain

    np.testing.assert_array_equal(result.ft, ft_scan)
    np.testing.assert_array_equal(result.fv, workflow.fv)
    np.testing.assert_array_equal(result.fvt, workflow.fvt)
    np.testing.assert_array_equal(
        result.skin_mask,
        skin_mask_from_skins(workflow.skins, result.ft.shape),
    )
    assert result.profile == resolve_qqual3d_profile(shape=result.ft.shape)


def test_supplied_profile_shape_mismatch_fails_before_execution() -> None:
    profile = resolve_qqual3d_profile(shape=(3, 4, 5))

    with pytest.raises(ValueError, match="profile shape"):
        run_qqual3d(np.zeros((3, 4, 6), dtype=np.float32), profile=profile)


@pytest.mark.parametrize(
    "profile_update",
    [
        lambda profile: replace(profile, scanner_backend="reference-like"),
        lambda profile: replace(profile, workflow_mode="reference"),
        lambda profile: replace(
            profile,
            voting_config=replace(
                profile.voting_config,
                voter_thin_mode="reference",
            ),
        ),
        lambda profile: replace(
            profile,
            skinning_config=replace(profile.skinning_config, method="reference"),
        ),
    ],
    ids=[
        "scanner-backend",
        "workflow-mode",
        "voter-thin-mode",
        "skinner-method",
    ],
)
def test_modified_fixed_profile_is_rejected(
    profile_update: Callable[[QQual3DProfile], QQual3DProfile],
) -> None:
    shape = (3, 4, 5)
    profile = profile_update(resolve_qqual3d_profile(shape=shape))

    with pytest.raises(ValueError, match="fixed Q-QUAL contract"):
        run_qqual3d(np.zeros(shape, dtype=np.float32), profile=profile)


def test_skinning_disabled_profile_returns_no_skins() -> None:
    shape = (5, 7, 9)
    enabled_profile = resolve_qqual3d_profile(shape=shape)
    profile = replace(
        enabled_profile,
        skinning_config=replace(enabled_profile.skinning_config, enabled=False),
    )
    assert profile == resolve_qqual3d_profile(shape=shape, skinning_enabled=False)

    result = run_qqual3d(np.zeros(shape, dtype=np.float32), profile=profile)

    assert result.skins == ()
    assert not np.any(result.skin_mask)
    assert not result.profile.skinning_enabled


@pytest.mark.parametrize(
    ("ep", "message"),
    [
        (np.zeros((2, 3), dtype=np.float32), "3D"),
        (np.zeros((1, 0, 2), dtype=np.float32), "positive"),
        (np.full((1, 2, 3), np.nan, dtype=np.float32), "finite"),
    ],
)
def test_invalid_input_fails(ep: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_qqual3d(ep)


def test_public_all_contains_only_qqual_api() -> None:
    assert qqual3d.__all__ == [
        "QQual3DProfile",
        "QQual3DResult",
        "resolve_qqual3d_profile",
        "run_qqual3d",
    ]
    parameters = inspect.signature(run_qqual3d).parameters
    assert tuple(parameters) == ("ep", "profile")
    assert parameters["profile"].kind is inspect.Parameter.KEYWORD_ONLY


def test_production_runner_does_not_import_mode_comparison() -> None:
    assert "f3d_mode_comparison" not in inspect.getsource(runner_module)
