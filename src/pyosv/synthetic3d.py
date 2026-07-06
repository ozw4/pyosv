"""Controlled synthetic 3D case contracts and coordinate helpers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from pyosv.geometry import fault_normal_vector_from_strike_and_dip

__all__ = [
    "Synthetic3DCase",
    "SyntheticPlaneSpec",
    "coordinate_grids3",
    "generate_single_plane_case",
    "make_single_dipping_plane_case",
    "make_single_vertical_plane_case",
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
    truth_fault_mask = np.abs(truth_distance) <= spec.mask_half_width
    truth_fault_id = np.where(truth_fault_mask, 1, 0).astype(np.int32)
    truth_strike = np.full(spec.shape, spec.strike, dtype=np.float32)
    truth_dip = np.full(spec.shape, spec.dip, dtype=np.float32)
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

    return array
