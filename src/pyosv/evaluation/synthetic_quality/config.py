"""Configuration models for synthetic quality evaluation."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyosv.synthetic3d import SyntheticScannerInputConfig

SCANNER_BACKENDS = ("reference-like", "fast", "quality", "ensemble")
SKINNER_METHODS = ("reference", "quality", "connected_component")
SKINNER_GROWTH_SOURCES = ("thinned", "pre_thin")
BOUNDARY_SKINNER_FALLBACK_POLICIES = (
    "empty_primary",
    "degraded_primary",
    "degraded_primary_filtered",
    "degraded_primary_skeletonized",
    "degraded_primary_topology_guarded",
)
REFERENCE_SKINNER_SEED_MIN_EP = 0.8
QUALITY_SKINNER_SEED_MIN_EP = 0.5


def _validate_scanner_refinement_factor(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError("scanner_refinement_factor must be an integer from 1 to 4")
    factor = int(value)
    if factor < 1 or factor > 4:
        raise ValueError("scanner_refinement_factor must be an integer from 1 to 4")
    return factor


def _skinner_seed_min_ep_for_method(method: str) -> float | None:
    if method == "reference":
        return REFERENCE_SKINNER_SEED_MIN_EP
    if method == "quality":
        return QUALITY_SKINNER_SEED_MIN_EP
    return None


@dataclass(frozen=True, slots=True)
class SyntheticVotingConfig:
    """Configuration for synthetic oracle voting."""

    ru: int = 1
    rv: int = 2
    rw: int = 2
    seed_distance: int = 3
    seed_threshold: float = 0.5
    attribute_smoothing: int = 0
    voter_thin_mode: str = "reference"
    reference_thin_sigma: float = 1.0
    surface_support_min_fraction: float = 0.0
    surface_support_exponent: float = 0.0

    def as_report_dict(self) -> dict[str, int | float | str]:
        return {
            "ru": int(self.ru),
            "rv": int(self.rv),
            "rw": int(self.rw),
            "seed_distance": int(self.seed_distance),
            "seed_threshold": float(self.seed_threshold),
            "attribute_smoothing": int(self.attribute_smoothing),
            "voter_thin_mode": self.voter_thin_mode,
            "reference_thin_sigma": float(self.reference_thin_sigma),
            "surface_support_min_fraction": float(self.surface_support_min_fraction),
            "surface_support_exponent": float(self.surface_support_exponent),
        }


@dataclass(frozen=True, slots=True)
class SyntheticScannerConfig:
    """Configuration for scanner-inclusive synthetic report inputs."""

    backend: str = "reference-like"
    phi_min: float = 0.0
    phi_max: float = 180.0
    theta_min: float = 45.0
    theta_max: float = 90.0
    sigma1: float = 2.0
    sigma2: float = 2.0
    refinement_factor: int = 2
    scanner_thin_mode: str = "reference"
    remove_edge_effects: bool = True
    input_config: SyntheticScannerInputConfig = SyntheticScannerInputConfig()

    def __post_init__(self) -> None:
        if self.backend not in SCANNER_BACKENDS:
            raise ValueError(
                "scanner_backend must be 'reference-like', 'fast', 'quality', or 'ensemble'"
            )
        if self.scanner_thin_mode not in {"none", "reference", "normal"}:
            raise ValueError("scanner_thin_mode must be 'none', 'reference', or 'normal'")
        if not isinstance(self.remove_edge_effects, bool):
            raise ValueError("remove_edge_effects must be a bool")
        if not isinstance(self.input_config, SyntheticScannerInputConfig):
            raise ValueError("input_config must be a SyntheticScannerInputConfig")
        _validate_finite_scalar(self.phi_min, "scanner_phi_min")
        _validate_finite_scalar(self.phi_max, "scanner_phi_max")
        _validate_finite_scalar(self.theta_min, "scanner_theta_min")
        _validate_finite_scalar(self.theta_max, "scanner_theta_max")
        _validate_positive_finite_scalar(self.sigma1, "scanner_sigma1")
        _validate_positive_finite_scalar(self.sigma2, "scanner_sigma2")
        _validate_scanner_refinement_factor(self.refinement_factor)
        if self.phi_max < self.phi_min:
            raise ValueError("scanner_phi_max must be greater than or equal to scanner_phi_min")
        if self.theta_max < self.theta_min:
            raise ValueError("scanner_theta_max must be greater than or equal to scanner_theta_min")

    def as_report_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "phi_min": float(self.phi_min),
            "phi_max": float(self.phi_max),
            "theta_min": float(self.theta_min),
            "theta_max": float(self.theta_max),
            "sigma1": float(self.sigma1),
            "sigma2": float(self.sigma2),
            "refinement_factor": int(self.refinement_factor),
            "scanner_thin_mode": self.scanner_thin_mode,
            "remove_edge_effects": self.remove_edge_effects,
            "input": {
                "background": float(self.input_config.background),
                "fault_contrast": float(self.input_config.fault_contrast),
                "noise_sigma": float(self.input_config.noise_sigma),
                "seed": int(self.input_config.seed),
                "clip_min": float(self.input_config.clip_min),
                "clip_max": float(self.input_config.clip_max),
            },
        }


@dataclass(frozen=True, slots=True)
class SyntheticTruthMetricConfig:
    """Configuration for controlled truth metrics."""

    truth_surface_half_width: float = 0.5
    buffer_radius: float = 2.0

    def as_report_dict(self) -> dict[str, float]:
        return {
            "truth_surface_half_width": float(self.truth_surface_half_width),
            "buffer_radius": float(self.buffer_radius),
        }


def _validate_finite_scalar(value: float, name: str) -> float:
    if not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_positive_finite_scalar(value: float, name: str) -> float:
    result = _validate_finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_nonnegative_finite_scalar(value: float, name: str) -> float:
    result = _validate_finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _validate_optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _validate_nonnegative_int(value, name)


def _validate_skinner_radius(value: int, name: str) -> int:
    result = _validate_nonnegative_int(value, name)
    if result < 2:
        raise ValueError(f"{name} must be at least 2")
    return result


def _validate_optional_skinner_radius(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _validate_skinner_radius(value, name)


@dataclass(frozen=True, slots=True)
class SyntheticSkinningConfig:
    """Configuration for controlled synthetic skinning."""

    enabled: bool = True
    method: str = "reference"
    growth_source: str = "thinned"
    min_likelihood: float | None = 0.5
    min_skin_size: int | None = 1
    d: int = 1
    ru: int = 10
    rv: int | None = None
    rw: int | None = None
    max_steps: int = 10
    du: float = 5.0
    max_delta_strike: float = 30.0
    reskin: bool = True
    accepted_occupancy_radius: int | None = None
    small_skin_size: int = 10
    boundary_skinner_fallback: bool = False
    boundary_skinner_fallback_policy: str = "empty_primary"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if not isinstance(self.reskin, bool):
            raise ValueError("reskin must be a bool")
        if not isinstance(self.boundary_skinner_fallback, bool):
            raise ValueError("boundary_skinner_fallback must be a bool")
        if self.boundary_skinner_fallback_policy not in BOUNDARY_SKINNER_FALLBACK_POLICIES:
            raise ValueError(
                "boundary_skinner_fallback_policy must be one of: "
                + ", ".join(BOUNDARY_SKINNER_FALLBACK_POLICIES)
            )
        if self.method not in SKINNER_METHODS:
            raise ValueError("skinner_method must be one of: " + ", ".join(SKINNER_METHODS))
        if self.growth_source not in SKINNER_GROWTH_SOURCES:
            raise ValueError(
                "skinner_growth_source must be one of: " + ", ".join(SKINNER_GROWTH_SOURCES)
            )
        if self.min_likelihood is not None:
            _validate_nonnegative_finite_scalar(self.min_likelihood, "skinner_min_likelihood")
        _validate_optional_nonnegative_int(self.min_skin_size, "skinner_min_skin_size")
        _validate_nonnegative_int(self.d, "skinner_d")
        _validate_skinner_radius(self.ru, "skinner_ru")
        _validate_optional_skinner_radius(self.rv, "skinner_rv")
        _validate_optional_skinner_radius(self.rw, "skinner_rw")
        _validate_nonnegative_int(self.max_steps, "skinner_max_steps")
        _validate_nonnegative_finite_scalar(self.du, "skinner_du")
        _validate_nonnegative_finite_scalar(self.max_delta_strike, "skinner_max_delta_strike")
        _validate_optional_nonnegative_int(
            self.accepted_occupancy_radius,
            "skinner_accepted_occupancy_radius",
        )
        _validate_nonnegative_int(self.small_skin_size, "small_skin_size")

    def as_report_dict(self) -> dict[str, bool | int | float | str | None]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "growth_source": self.growth_source,
            "min_likelihood": (None if self.min_likelihood is None else float(self.min_likelihood)),
            "adaptive_min_likelihood": self.method == "quality" and self.min_likelihood is None,
            "seed_min_ep": _skinner_seed_min_ep_for_method(self.method),
            "seed_planarity_source": "fvt",
            "min_skin_size": (None if self.min_skin_size is None else int(self.min_skin_size)),
            "d": int(self.d),
            "ru": int(self.ru),
            "rv": None if self.rv is None else int(self.rv),
            "rw": None if self.rw is None else int(self.rw),
            "max_steps": int(self.max_steps),
            "du": float(self.du),
            "max_delta_strike": float(self.max_delta_strike),
            "reskin": self.reskin,
            "accepted_occupancy_radius": (
                None
                if self.accepted_occupancy_radius is None
                else int(self.accepted_occupancy_radius)
            ),
            "effective_accepted_occupancy_radius": (
                5 if self.accepted_occupancy_radius is None else int(self.accepted_occupancy_radius)
            ),
            "small_skin_size": int(self.small_skin_size),
            "boundary_skinner_fallback": self.boundary_skinner_fallback,
            "boundary_skinner_fallback_policy": self.boundary_skinner_fallback_policy,
        }
