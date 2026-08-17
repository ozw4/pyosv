"""Execution-free resolution of the fixed Q-QUAL 3D runtime profile."""

from __future__ import annotations

import numbers
from dataclasses import dataclass, replace

from pyosv.evaluation.synthetic_quality import (
    SyntheticSkinningConfig,
    SyntheticVotingConfig,
    resolve_workflow_settings,
)
from pyosv.evaluation.synthetic_quality.variants import VariantSpec
from pyosv.evaluation.workflow3d import VolumeVotingControls


@dataclass(frozen=True, slots=True)
class _CanonicalQQual3DSettings:
    scanner_backend: str
    phi_min: float
    phi_max: float
    theta_min: float
    theta_max: float
    sigma1: float
    sigma2: float
    scanner_refinement_factor: int
    orientation_backend: str
    interpolation_backend: str
    interpolation_order: int
    smoothing_sigma: float | None
    normalize: bool
    output_dtype: str
    scanner_thin_mode: str
    scanner_reference_thin_sigma: float
    remove_edge_effects: bool
    voting_config: SyntheticVotingConfig
    voting_controls: VolumeVotingControls
    workflow_mode: str
    variant: VariantSpec


_QQUAL3D_VOTING_CONFIG = SyntheticVotingConfig(
    ru=10,
    rv=20,
    rw=30,
    seed_distance=4,
    seed_threshold=0.3,
    attribute_smoothing=1,
    voter_thin_mode="hybrid_v2",
    reference_thin_sigma=1.0,
    surface_support_min_fraction=0.0,
    surface_support_exponent=0.0,
)
_QQUAL3D_VARIANT = VariantSpec("f3-canonical", experimental=False)
_CANONICAL_QQUAL3D_SETTINGS = _CanonicalQQual3DSettings(
    scanner_backend="quality",
    phi_min=0.0,
    phi_max=360.0,
    theta_min=65.0,
    theta_max=80.0,
    sigma1=8.0,
    sigma2=8.0,
    scanner_refinement_factor=2,
    orientation_backend="rotate_shear",
    interpolation_backend="scipy",
    interpolation_order=1,
    smoothing_sigma=None,
    normalize=True,
    output_dtype="float32",
    scanner_thin_mode="reference",
    scanner_reference_thin_sigma=_QQUAL3D_VOTING_CONFIG.reference_thin_sigma,
    remove_edge_effects=True,
    voting_config=_QQUAL3D_VOTING_CONFIG,
    voting_controls=VolumeVotingControls.resolve(
        _QQUAL3D_VOTING_CONFIG,
        _QQUAL3D_VARIANT,
    ),
    workflow_mode="quality",
    variant=_QQUAL3D_VARIANT,
)


@dataclass(frozen=True, slots=True)
class QQual3DProfile:
    """Resolved controls for the fixed, non-experimental Q-QUAL workflow."""

    shape: tuple[int, int, int]
    scanner_backend: str
    phi_min: float
    phi_max: float
    theta_min: float
    theta_max: float
    sigma1: float
    sigma2: float
    scanner_refinement_factor: int
    orientation_backend: str
    interpolation_backend: str
    interpolation_order: int
    smoothing_sigma: float | None
    normalize: bool
    output_dtype: str
    scanner_thin_mode: str
    scanner_reference_thin_sigma: float
    remove_edge_effects: bool
    voting_config: SyntheticVotingConfig
    voting_controls: VolumeVotingControls
    skinning_config: SyntheticSkinningConfig
    workflow_mode: str
    variant: VariantSpec

    @property
    def scanner_angular_range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return inclusive strike and dip scan ranges in degrees."""

        return (self.phi_min, self.phi_max), (self.theta_min, self.theta_max)

    @property
    def scanner_sigmas(self) -> tuple[float, float]:
        """Return scanner smoothing sigmas for axes 1 and 2."""

        return self.sigma1, self.sigma2

    @property
    def scanner_thinning_mode(self) -> str:
        """Return the scanner-stage thinning mode."""

        return self.scanner_thin_mode

    @property
    def scanner_edge_cleanup(self) -> bool:
        """Return whether scanner thinning removes edge effects."""

        return self.remove_edge_effects

    @property
    def skinning_enabled(self) -> bool:
        """Return whether the resolved workflow includes skinning."""

        return self.skinning_config.enabled

    @property
    def variant_identity(self) -> str:
        """Return the fixed runtime variant name."""

        return self.variant.name


def _validate_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError("shape must contain exactly three dimensions")
    if any(
        isinstance(size, bool) or not isinstance(size, numbers.Integral) or size <= 0
        for size in shape
    ):
        raise ValueError("shape dimensions must be positive integers")
    return tuple(int(size) for size in shape)


def resolve_qqual3d_profile(
    *,
    shape: tuple[int, int, int],
    skinning_enabled: bool = True,
) -> QQual3DProfile:
    """Resolve the fixed Q-QUAL runtime settings without processing arrays."""

    valid_shape = _validate_shape(shape)
    if not isinstance(skinning_enabled, bool):
        raise ValueError("skinning_enabled must be a bool")

    canonical = _CANONICAL_QQUAL3D_SETTINGS
    workflow = resolve_workflow_settings(
        workflow_mode=canonical.workflow_mode,
        voting_config=canonical.voting_config,
        skinning_config=replace(SyntheticSkinningConfig(), enabled=skinning_enabled),
    )
    return QQual3DProfile(
        shape=valid_shape,
        scanner_backend=canonical.scanner_backend,
        phi_min=canonical.phi_min,
        phi_max=canonical.phi_max,
        theta_min=canonical.theta_min,
        theta_max=canonical.theta_max,
        sigma1=canonical.sigma1,
        sigma2=canonical.sigma2,
        scanner_refinement_factor=canonical.scanner_refinement_factor,
        orientation_backend=canonical.orientation_backend,
        interpolation_backend=canonical.interpolation_backend,
        interpolation_order=canonical.interpolation_order,
        smoothing_sigma=canonical.smoothing_sigma,
        normalize=canonical.normalize,
        output_dtype=canonical.output_dtype,
        scanner_thin_mode=canonical.scanner_thin_mode,
        scanner_reference_thin_sigma=canonical.scanner_reference_thin_sigma,
        remove_edge_effects=canonical.remove_edge_effects,
        voting_config=workflow.voting_config,
        voting_controls=canonical.voting_controls,
        skinning_config=workflow.skinning_config,
        workflow_mode=workflow.workflow_mode,
        variant=canonical.variant,
    )
