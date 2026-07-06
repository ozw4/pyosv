"""Controlled synthetic 3D case contracts and coordinate helpers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

__all__ = [
    "Synthetic3DCase",
    "SyntheticPlaneSpec",
    "coordinate_grids3",
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
