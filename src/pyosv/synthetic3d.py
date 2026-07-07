"""Controlled synthetic 3D case contracts and coordinate helpers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from pyosv.geometry import fault_normal_vector_from_strike_and_dip, strike_and_dip_from_normal

__all__ = [
    "Synthetic3DCase",
    "SyntheticCurvedSurfaceSpec",
    "SyntheticPlaneSpec",
    "coordinate_grids3",
    "generate_curved_surface_case",
    "generate_single_plane_case",
    "make_boundary_plane_case",
    "make_crossing_planes_case",
    "make_curved_surface_case",
    "make_parallel_planes_case",
    "make_single_dipping_plane_case",
    "make_single_vertical_plane_case",
    "make_weak_noisy_plane_case",
    "validate_center3",
    "validate_shape3",
]


@dataclass(frozen=True, slots=True)
class SyntheticPlaneSpec:
    """Specification for a controlled planar 3D synthetic fault case."""

    case_id: str
    shape: tuple[int, int, int]
    center: tuple[float, float, float]
    strike: float
    dip: float
    likelihood_sigma: float = 1.25
    mask_half_width: float = 1.0

    def __post_init__(self) -> None:
        shape = validate_shape3(self.shape)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "center", validate_center3(self.center, shape))
        object.__setattr__(self, "strike", _validate_finite_real("strike", self.strike))
        object.__setattr__(self, "dip", _validate_finite_real("dip", self.dip))
        object.__setattr__(
            self,
            "likelihood_sigma",
            _validate_finite_real(
                "likelihood_sigma", self.likelihood_sigma, minimum=0.0, closed=False
            ),
        )
        object.__setattr__(
            self,
            "mask_half_width",
            _validate_finite_real("mask_half_width", self.mask_half_width, minimum=0.0),
        )


@dataclass(frozen=True, slots=True)
class SyntheticCurvedSurfaceSpec:
    """Specification for a controlled analytic curved 3D synthetic fault case."""

    case_id: str
    shape: tuple[int, int, int]
    center: tuple[float, float, float]
    slope2: float
    slope3: float
    curvature2: float
    curvature3: float
    likelihood_sigma: float = 1.25
    mask_half_width: float = 1.0

    def __post_init__(self) -> None:
        shape = validate_shape3(self.shape)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "center", validate_center3(self.center, shape))
        object.__setattr__(self, "slope2", _validate_finite_real("slope2", self.slope2))
        object.__setattr__(self, "slope3", _validate_finite_real("slope3", self.slope3))
        object.__setattr__(
            self,
            "curvature2",
            _validate_finite_real("curvature2", self.curvature2),
        )
        object.__setattr__(
            self,
            "curvature3",
            _validate_finite_real("curvature3", self.curvature3),
        )
        object.__setattr__(
            self,
            "likelihood_sigma",
            _validate_finite_real(
                "likelihood_sigma", self.likelihood_sigma, minimum=0.0, closed=False
            ),
        )
        object.__setattr__(
            self,
            "mask_half_width",
            _validate_finite_real("mask_half_width", self.mask_half_width, minimum=0.0),
        )


@dataclass(frozen=True, slots=True)
class Synthetic3DCase:
    """Generated controlled 3D synthetic case arrays."""

    case_id: str
    shape: tuple[int, int, int]
    truth_fault_mask: np.ndarray
    truth_fault_id: np.ndarray
    truth_distance: np.ndarray
    truth_strike: np.ndarray
    truth_dip: np.ndarray
    ft_oracle: np.ndarray
    pt_oracle: np.ndarray
    tt_oracle: np.ndarray

    def __post_init__(self) -> None:
        shape = validate_shape3(self.shape)
        object.__setattr__(self, "shape", shape)
        for name, dtype in (
            ("truth_fault_mask", bool),
            ("truth_fault_id", np.int32),
            ("truth_distance", np.float32),
            ("truth_strike", np.float32),
            ("truth_dip", np.float32),
            ("ft_oracle", np.float32),
            ("pt_oracle", np.float32),
            ("tt_oracle", np.float32),
        ):
            array = _validate_case_array(name, getattr(self, name), shape, dtype)
            object.__setattr__(self, name, array)


@dataclass(frozen=True, slots=True)
class _SyntheticFaultComponent:
    fault_id: int
    signed_distance: np.ndarray
    strike: np.ndarray
    dip: np.ndarray
    likelihood: np.ndarray
    mask_half_width: float

    def __post_init__(self) -> None:
        if not isinstance(self.fault_id, Integral) or isinstance(self.fault_id, bool):
            raise ValueError("fault_id must be a positive integer")
        fault_id = int(self.fault_id)
        if fault_id <= 0:
            raise ValueError("fault_id must be a positive integer")
        object.__setattr__(self, "fault_id", fault_id)

        signed_distance = _validate_component_array("signed_distance", self.signed_distance)
        shape = signed_distance.shape
        object.__setattr__(self, "signed_distance", signed_distance)
        object.__setattr__(self, "strike", _validate_component_array("strike", self.strike, shape))
        object.__setattr__(self, "dip", _validate_component_array("dip", self.dip, shape))
        likelihood = _validate_component_array("likelihood", self.likelihood, shape)
        object.__setattr__(self, "likelihood", np.clip(likelihood, 0.0, 1.0).astype(np.float32))
        object.__setattr__(
            self,
            "mask_half_width",
            _validate_finite_real("mask_half_width", self.mask_half_width, minimum=0.0),
        )

    @property
    def mask(self) -> np.ndarray:
        return np.abs(self.signed_distance) <= self.mask_half_width


def _compose_synthetic_components(
    *,
    case_id: str,
    shape: tuple[int, int, int],
    components: tuple[_SyntheticFaultComponent, ...],
) -> Synthetic3DCase:
    shape = validate_shape3(shape)
    if len(components) == 0:
        raise ValueError("components must contain at least one component")

    for component in components:
        if not isinstance(component, _SyntheticFaultComponent):
            raise ValueError("components must be _SyntheticFaultComponent instances")
        if component.signed_distance.shape != shape:
            raise ValueError("component arrays must have shape matching shape")

    ordered_components = sorted(components, key=lambda component: component.fault_id)
    first = ordered_components[0]
    truth_fault_mask = first.mask.copy()
    nearest_abs_distance = np.abs(first.signed_distance).copy()
    truth_fault_id = np.full(shape, first.fault_id, dtype=np.int32)
    truth_distance = first.signed_distance.copy()
    truth_strike = first.strike.copy()
    truth_dip = first.dip.copy()
    ft_oracle = first.likelihood.copy()

    for component in ordered_components[1:]:
        component_mask = component.mask
        truth_fault_mask |= component_mask
        component_abs_distance = np.abs(component.signed_distance)
        nearer = component_abs_distance < nearest_abs_distance
        nearest_abs_distance = np.where(nearer, component_abs_distance, nearest_abs_distance)
        truth_distance = np.where(nearer, component.signed_distance, truth_distance)
        truth_strike = np.where(nearer, component.strike, truth_strike)
        truth_dip = np.where(nearer, component.dip, truth_dip)
        truth_fault_id = np.where(nearer, component.fault_id, truth_fault_id)
        ft_oracle = np.maximum(ft_oracle, component.likelihood)

    truth_fault_id = np.where(truth_fault_mask, truth_fault_id, 0).astype(np.int32)
    return Synthetic3DCase(
        case_id=case_id,
        shape=shape,
        truth_fault_mask=truth_fault_mask,
        truth_fault_id=truth_fault_id,
        truth_distance=truth_distance,
        truth_strike=truth_strike,
        truth_dip=truth_dip,
        ft_oracle=ft_oracle,
        pt_oracle=truth_strike,
        tt_oracle=truth_dip,
    )


def validate_shape3(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Validate a 3D ``(n3, n2, n1)`` shape."""
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError("shape must be a 3D (n3, n2, n1) tuple")

    validated = []
    for axis, size in enumerate(shape):
        if not isinstance(size, Integral) or isinstance(size, bool):
            raise ValueError(f"shape[{axis}] must be a positive integer")
        size_int = int(size)
        if size_int <= 0:
            raise ValueError(f"shape[{axis}] must be a positive integer")
        validated.append(size_int)

    return tuple(validated)


def validate_center3(
    center: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Validate an OSV ``(x1, x2, x3)`` center inside a 3D volume."""
    n3, n2, n1 = validate_shape3(shape)
    if not isinstance(center, tuple) or len(center) != 3:
        raise ValueError("center must be an (x1, x2, x3) tuple")

    x1 = _validate_finite_real("center[0]", center[0])
    x2 = _validate_finite_real("center[1]", center[1])
    x3 = _validate_finite_real("center[2]", center[2])
    if not 0.0 <= x1 < n1:
        raise ValueError("center[0] must satisfy 0 <= x1 < n1")
    if not 0.0 <= x2 < n2:
        raise ValueError("center[1] must satisfy 0 <= x2 < n2")
    if not 0.0 <= x3 < n3:
        raise ValueError("center[2] must satisfy 0 <= x3 < n3")

    return (x1, x2, x3)


def coordinate_grids3(shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x1, x2, x3)`` float32 grids for a ``(n3, n2, n1)`` volume."""
    i3, i2, i1 = np.indices(validate_shape3(shape), dtype=np.float32)
    return i1, i2, i3


def generate_single_plane_case(spec: SyntheticPlaneSpec) -> Synthetic3DCase:
    """Generate a controlled single-plane 3D synthetic fault case."""
    if not isinstance(spec, SyntheticPlaneSpec):
        raise ValueError("spec must be a SyntheticPlaneSpec")

    component = _plane_component_from_spec(spec, fault_id=1)
    return _compose_synthetic_components(
        case_id=spec.case_id,
        shape=spec.shape,
        components=(component,),
    )


def _plane_component_from_spec(
    spec: SyntheticPlaneSpec,
    fault_id: int,
) -> _SyntheticFaultComponent:
    if not isinstance(spec, SyntheticPlaneSpec):
        raise ValueError("spec must be a SyntheticPlaneSpec")

    normal = fault_normal_vector_from_strike_and_dip(spec.strike, spec.dip).astype(
        np.float64,
        copy=False,
    )
    x1, x2, x3 = coordinate_grids3(spec.shape)
    x1c, x2c, x3c = spec.center
    truth_distance = (
        (x1.astype(np.float64) - x1c) * normal[0]
        + (x2.astype(np.float64) - x2c) * normal[1]
        + (x3.astype(np.float64) - x3c) * normal[2]
    )
    truth_strike = np.full(spec.shape, spec.strike, dtype=np.float32)
    truth_dip = np.full(spec.shape, spec.dip, dtype=np.float32)
    ft_oracle = np.exp(-0.5 * (truth_distance / spec.likelihood_sigma) ** 2)
    ft_oracle = np.clip(ft_oracle, 0.0, 1.0).astype(np.float32)

    return _SyntheticFaultComponent(
        fault_id=fault_id,
        signed_distance=truth_distance.astype(np.float32),
        strike=truth_strike,
        dip=truth_dip,
        likelihood=ft_oracle,
        mask_half_width=spec.mask_half_width,
    )


def generate_curved_surface_case(spec: SyntheticCurvedSurfaceSpec) -> Synthetic3DCase:
    """Generate a controlled analytic curved-surface 3D synthetic fault case."""
    if not isinstance(spec, SyntheticCurvedSurfaceSpec):
        raise ValueError("spec must be a SyntheticCurvedSurfaceSpec")

    x1, x2, x3 = coordinate_grids3(spec.shape)
    x1c, x2c, x3c = spec.center
    scale2, scale3 = _curved_surface_scales(spec.shape)
    dx2 = x2.astype(np.float64) - x2c
    dx3 = x3.astype(np.float64) - x3c
    x1_surface = (
        x1c
        + spec.slope2 * dx2
        + spec.slope3 * dx3
        + spec.curvature2 * (dx2**2 / scale2)
        + spec.curvature3 * (dx3**2 / scale3)
    )
    truth_distance = x1.astype(np.float64) - x1_surface

    dx1_dx2 = spec.slope2 + 2.0 * spec.curvature2 * dx2 / scale2
    dx1_dx3 = spec.slope3 + 2.0 * spec.curvature3 * dx3 / scale3
    normal1 = np.ones(spec.shape, dtype=np.float64)
    normal2 = -dx1_dx2
    normal3 = -dx1_dx3
    normal_norm = np.sqrt(normal1**2 + normal2**2 + normal3**2)
    truth_strike, truth_dip = strike_and_dip_from_normal(
        (normal1 / normal_norm).astype(np.float32),
        (normal2 / normal_norm).astype(np.float32),
        (normal3 / normal_norm).astype(np.float32),
    )

    truth_fault_mask = np.abs(truth_distance) <= spec.mask_half_width
    truth_fault_id = np.where(truth_fault_mask, 1, 0).astype(np.int32)
    ft_oracle = np.exp(-0.5 * (truth_distance / spec.likelihood_sigma) ** 2)
    ft_oracle = np.clip(ft_oracle, 0.0, 1.0).astype(np.float32)

    return Synthetic3DCase(
        case_id=spec.case_id,
        shape=spec.shape,
        truth_fault_mask=truth_fault_mask,
        truth_fault_id=truth_fault_id,
        truth_distance=truth_distance.astype(np.float32),
        truth_strike=truth_strike,
        truth_dip=truth_dip,
        ft_oracle=ft_oracle,
        pt_oracle=truth_strike,
        tt_oracle=truth_dip,
    )


def make_single_vertical_plane_case(
    shape: tuple[int, int, int] = (64, 64, 64),
) -> Synthetic3DCase:
    """Return the default controlled vertical single-plane synthetic case."""
    n3, n2, n1 = validate_shape3(shape)
    spec = SyntheticPlaneSpec(
        case_id="single_vertical_plane",
        shape=(n3, n2, n1),
        center=((n1 - 1) / 2.0, (n2 - 1) / 2.0, (n3 - 1) / 2.0),
        strike=0.0,
        dip=90.0,
        likelihood_sigma=1.25,
        mask_half_width=1.0,
    )
    return generate_single_plane_case(spec)


def make_single_dipping_plane_case(
    shape: tuple[int, int, int] = (64, 64, 64),
) -> Synthetic3DCase:
    """Return the default controlled dipping single-plane synthetic case."""
    n3, n2, n1 = validate_shape3(shape)
    spec = SyntheticPlaneSpec(
        case_id="single_dipping_plane",
        shape=(n3, n2, n1),
        center=((n1 - 1) / 2.0, (n2 - 1) / 2.0, (n3 - 1) / 2.0),
        strike=45.0,
        dip=65.0,
        likelihood_sigma=1.25,
        mask_half_width=1.0,
    )
    return generate_single_plane_case(spec)


def make_boundary_plane_case(
    shape: tuple[int, int, int] = (64, 64, 64),
) -> Synthetic3DCase:
    """Return a vertical plane case whose mask touches the ``i2=0`` boundary."""
    n3, n2, n1 = validate_shape3(shape)
    spec = SyntheticPlaneSpec(
        case_id="boundary_plane",
        shape=(n3, n2, n1),
        center=((n1 - 1) / 2.0, 1.0, (n3 - 1) / 2.0),
        strike=0.0,
        dip=90.0,
        likelihood_sigma=1.25,
        mask_half_width=1.0,
    )
    return generate_single_plane_case(spec)


def make_parallel_planes_case(
    shape: tuple[int, int, int] = (64, 64, 64),
) -> Synthetic3DCase:
    """Return a controlled case with two nearby vertical parallel planes."""
    n3, n2, n1 = validate_shape3(shape)
    x1c, x2c, x3c = (n1 - 1) / 2.0, (n2 - 1) / 2.0, (n3 - 1) / 2.0
    separation = min(max(6.0, 0.18 * n2), float(max(n2 - 1, 0)))
    common_kwargs = {
        "case_id": "parallel_planes",
        "shape": (n3, n2, n1),
        "strike": 0.0,
        "dip": 90.0,
        "likelihood_sigma": 1.25,
        "mask_half_width": 1.0,
    }
    component_a = _plane_component_from_spec(
        SyntheticPlaneSpec(center=(x1c, x2c - separation / 2.0, x3c), **common_kwargs),
        fault_id=1,
    )
    component_b = _plane_component_from_spec(
        SyntheticPlaneSpec(center=(x1c, x2c + separation / 2.0, x3c), **common_kwargs),
        fault_id=2,
    )
    return _compose_synthetic_components(
        case_id="parallel_planes",
        shape=(n3, n2, n1),
        components=(component_a, component_b),
    )


def make_crossing_planes_case(
    shape: tuple[int, int, int] = (64, 64, 64),
) -> Synthetic3DCase:
    """Return a controlled case with two intersecting planar faults."""
    n3, n2, n1 = validate_shape3(shape)
    center = ((n1 - 1) / 2.0, (n2 - 1) / 2.0, (n3 - 1) / 2.0)
    common_kwargs = {
        "case_id": "crossing_planes",
        "shape": (n3, n2, n1),
        "center": center,
        "likelihood_sigma": 1.25,
        "mask_half_width": 1.0,
    }
    component_a = _plane_component_from_spec(
        SyntheticPlaneSpec(strike=0.0, dip=90.0, **common_kwargs),
        fault_id=1,
    )
    component_b = _plane_component_from_spec(
        SyntheticPlaneSpec(strike=60.0, dip=70.0, **common_kwargs),
        fault_id=2,
    )
    return _compose_synthetic_components(
        case_id="crossing_planes",
        shape=(n3, n2, n1),
        components=(component_a, component_b),
    )


def make_weak_noisy_plane_case(
    shape: tuple[int, int, int] = (64, 64, 64),
) -> Synthetic3DCase:
    """Return a controlled single-plane case with weak deterministic noisy likelihood."""
    n3, n2, n1 = validate_shape3(shape)
    spec = SyntheticPlaneSpec(
        case_id="weak_noisy_plane",
        shape=(n3, n2, n1),
        center=((n1 - 1) / 2.0, (n2 - 1) / 2.0, (n3 - 1) / 2.0),
        strike=35.0,
        dip=70.0,
        likelihood_sigma=1.25,
        mask_half_width=1.0,
    )
    component = _plane_component_from_spec(spec, fault_id=1)
    rng = np.random.default_rng(20260707)
    base = np.exp(-0.5 * (component.signed_distance / spec.likelihood_sigma) ** 2)
    noise = rng.normal(0.0, 0.06, size=spec.shape)
    likelihood = np.clip(0.03 + 0.65 * base + noise, 0.0, 1.0).astype(np.float32)
    noisy_component = _SyntheticFaultComponent(
        fault_id=component.fault_id,
        signed_distance=component.signed_distance,
        strike=component.strike,
        dip=component.dip,
        likelihood=likelihood,
        mask_half_width=component.mask_half_width,
    )
    return _compose_synthetic_components(
        case_id=spec.case_id,
        shape=spec.shape,
        components=(noisy_component,),
    )


def make_curved_surface_case(
    shape: tuple[int, int, int] = (64, 64, 64),
) -> Synthetic3DCase:
    """Return the default controlled analytic curved-surface synthetic case."""
    n3, n2, n1 = validate_shape3(shape)
    spec = SyntheticCurvedSurfaceSpec(
        case_id="curved_surface",
        shape=(n3, n2, n1),
        center=((n1 - 1) / 2.0, (n2 - 1) / 2.0, (n3 - 1) / 2.0),
        slope2=0.18,
        slope3=-0.12,
        curvature2=0.35,
        curvature3=-0.25,
        likelihood_sigma=1.25,
        mask_half_width=1.0,
    )
    return generate_curved_surface_case(spec)


def _validate_finite_real(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    closed: bool = True,
) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite real number")

    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None:
        if closed:
            valid = result >= minimum
            comparator = ">="
        else:
            valid = result > minimum
            comparator = ">"
        if not valid:
            raise ValueError(f"{name} must be finite and {comparator} {minimum:g}")

    return result


def _curved_surface_scales(shape: tuple[int, int, int]) -> tuple[float, float]:
    n3, n2, _ = validate_shape3(shape)
    return float(max(n2 - 1, 1)), float(max(n3 - 1, 1))


def _validate_case_array(
    name: str,
    value: np.ndarray,
    shape: tuple[int, int, int],
    dtype: type | np.dtype,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to {np.dtype(dtype).name}") from exc

    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")

    if np.issubdtype(array.dtype, np.floating) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array


def _validate_component_array(
    name: str,
    value: np.ndarray,
    shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to float32") from exc

    if shape is None:
        validate_shape3(array.shape)
    elif array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")

    return array
