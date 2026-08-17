"""Configuration models for the canonical F3 full-volume comparison."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass

from pyosv.qqual3d.profile import _CANONICAL_QQUAL3D_SETTINGS

from ..promotion.scanner_policy import effective_remove_edge_effects
from ..synthetic_quality import SyntheticSkinningConfig, SyntheticVotingConfig

F3_SCANNER_BACKENDS = ("reference-like", "quality")
F3_WORKFLOW_MODES = ("reference", "quality")
_QQUAL = _CANONICAL_QQUAL3D_SETTINGS


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_float(value: object, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
    return value


def _shape3(value: object, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly 3 dimensions")
    return tuple(_positive_int(size, f"{name}[{axis}]") for axis, size in enumerate(value))


@dataclass(frozen=True, slots=True)
class F3ScannerConfig:
    """Scanner controls resolved for one canonical scanner backend."""

    backend: str = "reference-like"
    phi_min: float = _QQUAL.phi_min
    phi_max: float = _QQUAL.phi_max
    theta_min: float = _QQUAL.theta_min
    theta_max: float = _QQUAL.theta_max
    sigma1: float = _QQUAL.sigma1
    sigma2: float = _QQUAL.sigma2
    refinement_factor: int = _QQUAL.scanner_refinement_factor
    orientation_backend: str = _QQUAL.orientation_backend
    interpolation_backend: str = _QQUAL.interpolation_backend
    interpolation_order: int = _QQUAL.interpolation_order
    smoothing_sigma: float | None = _QQUAL.smoothing_sigma
    normalize: bool = _QQUAL.normalize
    dtype: str = _QQUAL.output_dtype
    scanner_thin_mode: str = _QQUAL.scanner_thin_mode
    reference_thin_sigma: float = _QQUAL.scanner_reference_thin_sigma
    remove_edge_effects: bool = _QQUAL.remove_edge_effects

    def __post_init__(self) -> None:
        if self.backend not in F3_SCANNER_BACKENDS:
            raise ValueError("backend must be 'reference-like' or 'quality'")
        for name in ("phi_min", "phi_max", "theta_min", "theta_max"):
            object.__setattr__(self, name, _finite_float(getattr(self, name), name))
        if self.phi_max < self.phi_min:
            raise ValueError("phi_max must be greater than or equal to phi_min")
        if self.theta_max < self.theta_min:
            raise ValueError("theta_max must be greater than or equal to theta_min")
        object.__setattr__(self, "sigma1", _positive_float(self.sigma1, "sigma1"))
        object.__setattr__(self, "sigma2", _positive_float(self.sigma2, "sigma2"))
        factor = _positive_int(self.refinement_factor, "refinement_factor")
        if factor > 4:
            raise ValueError("refinement_factor must be between 1 and 4")
        object.__setattr__(self, "refinement_factor", factor)
        if self.orientation_backend not in {"rotate_shear", "directional"}:
            raise ValueError("orientation_backend must be 'rotate_shear' or 'directional'")
        if self.interpolation_backend not in {"scipy", "structured_linear"}:
            raise ValueError("interpolation_backend must be 'scipy' or 'structured_linear'")
        order = _nonnegative_int(self.interpolation_order, "interpolation_order")
        if order > 5:
            raise ValueError("interpolation_order must be between 0 and 5")
        if self.interpolation_backend == "structured_linear" and order != 1:
            raise ValueError(
                "interpolation_backend='structured_linear' requires interpolation_order=1"
            )
        if (
            self.interpolation_backend == "structured_linear"
            and self.orientation_backend != "rotate_shear"
        ):
            raise ValueError(
                "interpolation_backend='structured_linear' requires backend='rotate_shear'"
            )
        object.__setattr__(self, "interpolation_order", order)
        if self.smoothing_sigma is not None:
            object.__setattr__(
                self,
                "smoothing_sigma",
                _positive_float(self.smoothing_sigma, "smoothing_sigma"),
            )
        _bool(self.normalize, "normalize")
        if self.dtype != "float32":
            raise ValueError("dtype must be 'float32'")
        if self.scanner_thin_mode not in {"none", "reference", "normal"}:
            raise ValueError("scanner_thin_mode must be 'none', 'reference', or 'normal'")
        object.__setattr__(
            self,
            "reference_thin_sigma",
            _positive_float(self.reference_thin_sigma, "reference_thin_sigma"),
        )
        _bool(self.remove_edge_effects, "remove_edge_effects")

    @property
    def effective_remove_edge_effects(self) -> bool | None:
        """Return the edge policy effective for this scanner-thinning mode."""

        return effective_remove_edge_effects(
            self.scanner_thin_mode,
            self.remove_edge_effects,
        )


@dataclass(frozen=True, slots=True)
class F3VotingControls:
    """Controls held constant across all four F3 comparison cells."""

    ru: int = _QQUAL.voting_config.ru
    rv: int = _QQUAL.voting_config.rv
    rw: int = _QQUAL.voting_config.rw
    seed_distance: int = _QQUAL.voting_config.seed_distance
    seed_threshold: float = _QQUAL.voting_config.seed_threshold
    strain_max1: float = _QQUAL.voting_controls.strain_max1
    strain_max2: float = _QQUAL.voting_controls.strain_max2
    attribute_smoothing: int = _QQUAL.voting_config.attribute_smoothing
    surface_smoothing1: float = _QQUAL.voting_controls.surface_smoothing1
    surface_smoothing2: float = _QQUAL.voting_controls.surface_smoothing2
    surface_orientation_smoothing: float = _QQUAL.voting_controls.orientation_smoothing
    final_normalization_smoothing: float = _QQUAL.voting_controls.final_normalization_smoothing
    reference_thin_sigma: float = _QQUAL.voting_config.reference_thin_sigma
    surface_support_min_fraction: float = _QQUAL.voting_controls.support_min_fraction
    surface_support_exponent: float = _QQUAL.voting_controls.support_exponent
    surface_voting_boundary_policy: str = _QQUAL.voting_controls.boundary_policy

    def __post_init__(self) -> None:
        for name in ("ru", "rv", "rw", "seed_distance", "attribute_smoothing"):
            object.__setattr__(
                self,
                name,
                _nonnegative_int(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "seed_threshold",
            _nonnegative_float(self.seed_threshold, "seed_threshold"),
        )
        for name in ("strain_max1", "strain_max2"):
            value = _positive_float(getattr(self, name), name)
            if value > 1.0:
                raise ValueError(f"{name} must be at most 1")
            object.__setattr__(self, name, value)
        for name in (
            "surface_smoothing1",
            "surface_smoothing2",
            "surface_orientation_smoothing",
            "final_normalization_smoothing",
            "surface_support_exponent",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_float(getattr(self, name), name),
            )
        object.__setattr__(
            self,
            "reference_thin_sigma",
            _positive_float(self.reference_thin_sigma, "reference_thin_sigma"),
        )
        support_fraction = _nonnegative_float(
            self.surface_support_min_fraction,
            "surface_support_min_fraction",
        )
        if support_fraction > 1.0:
            raise ValueError("surface_support_min_fraction must be at most 1")
        object.__setattr__(
            self,
            "surface_support_min_fraction",
            support_fraction,
        )
        if self.surface_voting_boundary_policy not in {
            "reference",
            "masked_in_bounds",
        }:
            raise ValueError(
                "surface_voting_boundary_policy must be 'reference' or 'masked_in_bounds'"
            )

    def to_voting_config(self, *, voter_thin_mode: str) -> SyntheticVotingConfig:
        """Return the resolver input for one workflow-owned thinning mode."""

        if voter_thin_mode not in {"reference", "hybrid_v2"}:
            raise ValueError("voter_thin_mode must be 'reference' or 'hybrid_v2'")
        return SyntheticVotingConfig(
            ru=self.ru,
            rv=self.rv,
            rw=self.rw,
            seed_distance=self.seed_distance,
            seed_threshold=self.seed_threshold,
            attribute_smoothing=self.attribute_smoothing,
            voter_thin_mode=voter_thin_mode,
            reference_thin_sigma=self.reference_thin_sigma,
            surface_support_min_fraction=self.surface_support_min_fraction,
            surface_support_exponent=self.surface_support_exponent,
        )


@dataclass(frozen=True, slots=True)
class F3ModeComparisonConfig:
    """Inputs used to build the canonical full-volume F3 2-by-2 plan."""

    shape: tuple[int, int, int] = (420, 400, 100)
    input_file: str = "ep.dat"
    scanner_template: F3ScannerConfig = F3ScannerConfig()
    voting_controls: F3VotingControls = F3VotingControls()
    skinning_template: SyntheticSkinningConfig = SyntheticSkinningConfig()
    skinning_enabled: bool = True
    voter_thin_mode_override: str | None = None
    boundary_diagnostic_margin: int = 16
    skinner_method_explicit: bool = False
    skinner_min_likelihood_explicit: bool = False
    skinner_growth_source_explicit: bool = False
    skinner_accepted_occupancy_radius_explicit: bool = False
    skinner_boundary_fallback_explicit: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _shape3(self.shape, "shape"))
        if not isinstance(self.input_file, str) or not self.input_file:
            raise ValueError("input_file must be a non-empty string")
        if not isinstance(self.scanner_template, F3ScannerConfig):
            raise ValueError("scanner_template must be an F3ScannerConfig")
        if not isinstance(self.voting_controls, F3VotingControls):
            raise ValueError("voting_controls must be F3VotingControls")
        if not isinstance(self.skinning_template, SyntheticSkinningConfig):
            raise ValueError("skinning_template must be a SyntheticSkinningConfig")
        _bool(self.skinning_enabled, "skinning_enabled")
        if self.voter_thin_mode_override is not None and (
            not isinstance(self.voter_thin_mode_override, str)
            or self.voter_thin_mode_override not in {"reference", "hybrid_v2"}
        ):
            raise ValueError("voter_thin_mode_override must be None, 'reference', or 'hybrid_v2'")
        margin = _nonnegative_int(
            self.boundary_diagnostic_margin,
            "boundary_diagnostic_margin",
        )
        object.__setattr__(self, "boundary_diagnostic_margin", margin)
        if any(2 * margin >= size for size in self.shape):
            raise ValueError("boundary_diagnostic_margin is too large for shape")
        for name in (
            "skinner_method_explicit",
            "skinner_min_likelihood_explicit",
            "skinner_growth_source_explicit",
            "skinner_accepted_occupancy_radius_explicit",
            "skinner_boundary_fallback_explicit",
        ):
            _bool(getattr(self, name), name)
