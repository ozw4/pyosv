"""In-memory execution of the fixed Q-QUAL 3D workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.evaluation.workflow3d import Workflow3DResult, execute_workflow3d
from pyosv.orient3d import FaultOrientScanner3
from pyosv.synthetic_metrics import skin_mask_from_skins

from .profile import QQual3DProfile, resolve_qqual3d_profile


@dataclass(frozen=True, slots=True)
class QQual3DResult:
    """Owned outputs from one fixed Q-QUAL 3D execution.

    ``ft`` is the raw quality-scanner likelihood. ``fv`` and ``fvt`` are the
    voted and voter-thinned likelihoods. ``skin_mask`` represents the returned
    final ``skins``.
    """

    ft: np.ndarray
    fv: np.ndarray
    fvt: np.ndarray
    skins: tuple[object, ...]
    skin_mask: np.ndarray
    diagnostics: Mapping[str, object]
    profile: QQual3DProfile


def _validated_input(ep: object) -> np.ndarray:
    try:
        array = np.asarray(ep)
    except (TypeError, ValueError) as error:
        raise ValueError("ep must be a NumPy-compatible array") from error
    if array.ndim != 3:
        raise ValueError("ep must be a 3D array")
    if any(size <= 0 for size in array.shape):
        raise ValueError("ep dimensions must be positive")
    try:
        result = np.array(array, dtype=np.float32, copy=True, order="C")
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("ep must be convertible to float32") from error
    if not np.all(np.isfinite(result)):
        raise ValueError("ep must contain only finite float32 values")
    return result


def _owned_float32(value: np.ndarray, name: str, shape: tuple[int, int, int]) -> np.ndarray:
    result = np.array(value, dtype=np.float32, copy=True, order="C")
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise RuntimeError(f"{name} must be a finite float32 volume with shape {shape}")
    result.flags.writeable = False
    return result


def _owned_skin_mask(value: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    result = np.array(value, dtype=bool, copy=True, order="C")
    if result.shape != shape:
        raise RuntimeError(f"skin_mask must have shape {shape}")
    result.flags.writeable = False
    return result


def _public_diagnostics(result: Workflow3DResult) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "seed": result.diagnostics.seed,
            "voting": result.diagnostics.voting,
            "thinning": result.diagnostics.thinning,
            "skinning": result.diagnostics.skinning,
        }
    )


def _validate_fixed_profile(
    profile: QQual3DProfile,
    shape: tuple[int, int, int],
) -> None:
    expected = resolve_qqual3d_profile(
        shape=shape,
        skinning_enabled=profile.skinning_enabled,
    )
    if profile != expected:
        raise ValueError(
            "profile must match the fixed Q-QUAL contract; only skinning_enabled may vary"
        )


def run_qqual3d(
    ep: np.ndarray,
    *,
    profile: QQual3DProfile | None = None,
) -> QQual3DResult:
    """Run the fixed Q-QUAL scanner and workflow on one in-memory 3D volume.

    A supplied profile must equal the profile resolved for ``ep.shape``. Only
    the canonical skinning-enabled and skinning-disabled forms are accepted.
    """

    scanner_input = _validated_input(ep)
    shape = tuple(scanner_input.shape)
    if profile is None:
        profile = resolve_qqual3d_profile(shape=shape)
    elif not isinstance(profile, QQual3DProfile):
        raise TypeError("profile must be a QQual3DProfile or None")
    elif profile.shape != shape:
        raise ValueError(f"profile shape {profile.shape} does not match ep shape {shape}")
    _validate_fixed_profile(profile, shape)

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
    scanner_target = (
        quality_metrics.positive_candidate_mask(ft_scan)
        if profile.skinning_enabled and profile.skinning_config.boundary_skinner_fallback
        else None
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
        scanner_target_positive_mask=scanner_target,
        fvt_recenter_target=ft,
        fvt_recenter_target_source="scanner_fet",
    )
    return QQual3DResult(
        ft=_owned_float32(ft_scan, "ft", shape),
        fv=_owned_float32(workflow.fv, "fv", shape),
        fvt=_owned_float32(workflow.fvt, "fvt", shape),
        skins=tuple(workflow.skins),
        skin_mask=_owned_skin_mask(skin_mask_from_skins(workflow.skins, shape), shape),
        diagnostics=_public_diagnostics(workflow),
        profile=profile,
    )
