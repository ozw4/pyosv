"""Typed, Atlas-independent contracts for fault-warping computations.

The classes in this module validate caller-owned NumPy arrays without casting,
copying, sorting, normalizing, mutating, or changing their writeability.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray


FAULT_WARPING_CONTRACT_VERSION = "pyosv.fault_warping.v1"


Float32Array = NDArray[np.float32]
Int64Array = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


def _require_ndarray(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    return value


def _require_array(
    value: object,
    name: str,
    *,
    dtype: np.dtype[object],
    ndim: int,
) -> np.ndarray:
    array = _require_ndarray(value, name)
    if array.dtype != dtype:
        raise TypeError(f"{name} must have exact dtype {dtype}")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    return array


def _require_float32_vector(value: object, name: str) -> Float32Array:
    return _require_array(value, name, dtype=np.dtype(np.float32), ndim=1)  # type: ignore[return-value]


def _require_int64_vector(value: object, name: str) -> Int64Array:
    return _require_array(value, name, dtype=np.dtype(np.int64), ndim=1)  # type: ignore[return-value]


def _require_bool_vector(value: object, name: str) -> BoolArray:
    return _require_array(value, name, dtype=np.dtype(np.bool_), ndim=1)  # type: ignore[return-value]


def _require_float32_volume(value: object, name: str) -> Float32Array:
    return _require_array(value, name, dtype=np.dtype(np.float32), ndim=3)  # type: ignore[return-value]


def _require_bool_volume(value: object, name: str) -> BoolArray:
    return _require_array(value, name, dtype=np.dtype(np.bool_), ndim=3)  # type: ignore[return-value]


def _require_finite(array: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")


def _require_finite_where(array: np.ndarray, mask: BoolArray, name: str) -> None:
    if np.any(mask) and not np.all(np.isfinite(array[mask])):
        raise ValueError(f"{name} must be finite where valid_mask is True")


def _require_same_length(arrays: tuple[np.ndarray, ...], names: tuple[str, ...]) -> int:
    length = len(arrays[0])
    for array, name in zip(arrays[1:], names[1:]):
        if len(array) != length:
            raise ValueError(f"{names[0]} and {name} must have the same length")
    return length


def _require_topology_indices(index: Int64Array, name: str, count: int) -> None:
    if np.any(index < -1) or np.any(index >= count):
        raise ValueError(f"{name} values must be -1 or valid surface row indices")

    rows = np.arange(count, dtype=np.int64)
    if np.any(index == rows):
        raise ValueError(f"{name} must not contain self-links")


def _require_reciprocal_direction(
    outgoing: Int64Array,
    incoming: Int64Array,
    outgoing_name: str,
    incoming_name: str,
) -> None:
    linked = outgoing != -1
    if not np.any(linked):
        return

    rows = np.arange(len(outgoing), dtype=np.int64)
    if np.any(incoming[outgoing[linked]] != rows[linked]):
        raise ValueError(f"{outgoing_name} and {incoming_name} links must be reciprocal")


def _require_reciprocal_pair(
    first: Int64Array,
    second: Int64Array,
    first_name: str,
    second_name: str,
) -> None:
    _require_reciprocal_direction(first, second, first_name, second_name)
    _require_reciprocal_direction(second, first, second_name, first_name)


def _require_real(value: object, name: str) -> Real:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return value


def _require_integer(value: object, name: str) -> int | np.integer[object]:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, not bool")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class FaultSurfaceGraph:
    """A row-aligned surface or surface-patch graph in local index coordinates.

    ``ca_index``/``cb_index`` and ``cl_index``/``cr_index`` are reciprocal
    row-index links. ``-1`` denotes no link. ``cell_support_weight`` is a
    generic numeric support value, not a DL probability or OSV confidence.
    """

    x1: Float32Array
    x2: Float32Array
    x3: Float32Array
    strike_deg: Float32Array
    dip_deg: Float32Array
    ca_index: Int64Array
    cb_index: Int64Array
    cl_index: Int64Array
    cr_index: Int64Array
    cell_support_weight: Float32Array | None = None

    def __post_init__(self) -> None:
        _validate_fault_surface_graph(self)


def _validate_fault_surface_graph(surface: FaultSurfaceGraph) -> None:
    float_arrays = (
        _require_float32_vector(surface.x1, "x1"),
        _require_float32_vector(surface.x2, "x2"),
        _require_float32_vector(surface.x3, "x3"),
        _require_float32_vector(surface.strike_deg, "strike_deg"),
        _require_float32_vector(surface.dip_deg, "dip_deg"),
    )
    topology_arrays = (
        _require_int64_vector(surface.ca_index, "ca_index"),
        _require_int64_vector(surface.cb_index, "cb_index"),
        _require_int64_vector(surface.cl_index, "cl_index"),
        _require_int64_vector(surface.cr_index, "cr_index"),
    )
    arrays = float_arrays + topology_arrays
    names = (
        "x1",
        "x2",
        "x3",
        "strike_deg",
        "dip_deg",
        "ca_index",
        "cb_index",
        "cl_index",
        "cr_index",
    )
    count = _require_same_length(arrays, names)
    if count == 0:
        raise ValueError("surface must contain at least one cell")

    for array, name in zip(float_arrays, names[:5]):
        _require_finite(array, name)

    if np.any(surface.strike_deg < 0.0) or np.any(surface.strike_deg >= 360.0):
        raise ValueError("strike_deg values must be in [0, 360)")
    if np.any(surface.dip_deg <= 0.0) or np.any(surface.dip_deg > 90.0):
        raise ValueError("dip_deg values must be in (0, 90]")

    for array, name in zip(topology_arrays, names[5:]):
        _require_topology_indices(array, name, count)
    _require_reciprocal_pair(surface.ca_index, surface.cb_index, "ca_index", "cb_index")
    _require_reciprocal_pair(surface.cl_index, surface.cr_index, "cl_index", "cr_index")

    support = surface.cell_support_weight
    if support is None:
        return
    support_array = _require_float32_vector(support, "cell_support_weight")
    if len(support_array) != count:
        raise ValueError("cell_support_weight must have the same length as the surface arrays")
    _require_finite(support_array, "cell_support_weight")
    if np.any(support_array < 0.0) or np.any(support_array > 1.0):
        raise ValueError("cell_support_weight values must be in [0, 1]")


@dataclass(frozen=True, slots=True, eq=False)
class ReflectorSlopeVolume:
    """Reflector slopes ``p2 = dx1 / dx2`` and ``p3 = dx1 / dx3``.

    Both arrays use index samples per index-grid unit. The shared volume-wide
    valid mask belongs to :class:`FaultWarpingInput`, which verifies that the
    slopes are finite at valid voxels.
    """

    p2: Float32Array
    p3: Float32Array

    def __post_init__(self) -> None:
        _validate_reflector_slope_volume(self)


def _validate_reflector_slope_volume(slopes: ReflectorSlopeVolume) -> None:
    p2 = _require_float32_volume(slopes.p2, "p2")
    p3 = _require_float32_volume(slopes.p3, "p3")
    if p2.shape != p3.shape:
        raise ValueError("p2 and p3 must have the same shape")


@dataclass(frozen=True, slots=True, eq=False)
class FaultWarpingInput:
    """Numerical inputs for a future apparent sample-axis shift estimator.

    ``amplitude`` and the required reflector slopes are finite at valid voxels;
    invalid-mask values are deliberately outside the numerical contract.
    """

    amplitude: Float32Array
    valid_mask: BoolArray
    surface: FaultSurfaceGraph
    reflector_slopes: ReflectorSlopeVolume

    def __post_init__(self) -> None:
        amplitude = _require_float32_volume(self.amplitude, "amplitude")
        valid_mask = _require_bool_volume(self.valid_mask, "valid_mask")
        if amplitude.shape != valid_mask.shape:
            raise ValueError("amplitude and valid_mask must have the same shape")
        if not isinstance(self.surface, FaultSurfaceGraph):
            raise TypeError("surface must be a FaultSurfaceGraph")
        if not isinstance(self.reflector_slopes, ReflectorSlopeVolume):
            raise TypeError("reflector_slopes must be a ReflectorSlopeVolume")

        _validate_fault_surface_graph(self.surface)
        _validate_reflector_slope_volume(self.reflector_slopes)
        p2 = self.reflector_slopes.p2
        p3 = self.reflector_slopes.p3
        if p2.shape != amplitude.shape or p3.shape != amplitude.shape:
            raise ValueError("amplitude, p2, and p3 must have the same shape")

        _require_finite_where(amplitude, valid_mask, "amplitude")
        _require_finite_where(p2, valid_mask, "p2")
        _require_finite_where(p3, valid_mask, "p3")
        _require_surface_within_volume(self.surface, amplitude.shape)


def _require_surface_within_volume(
    surface: FaultSurfaceGraph,
    volume_shape: tuple[int, int, int],
) -> None:
    n3, n2, n1 = volume_shape
    coordinates = (
        (surface.x1, n1, "x1"),
        (surface.x2, n2, "x2"),
        (surface.x3, n3, "x3"),
    )
    for values, size, name in coordinates:
        if np.any(values < 0.0) or np.any(values > float(size - 1)):
            raise ValueError(f"surface {name} coordinates must be within local volume bounds")


@dataclass(frozen=True, slots=True, eq=False)
class FaultWarpingConfig:
    """Explicit parameters for a future sample-axis apparent-shift search."""

    side_offset_grid: float
    window_radius_samples: int
    lag_min_samples: int
    lag_max_samples: int
    max_shift_strain: float
    minimum_valid_fraction: float
    similarity_metric: Literal["zncc"] = "zncc"
    subsample_refinement: bool = False

    def __post_init__(self) -> None:
        side_offset = _require_real(self.side_offset_grid, "side_offset_grid")
        window_radius = _require_integer(self.window_radius_samples, "window_radius_samples")
        lag_min = _require_integer(self.lag_min_samples, "lag_min_samples")
        lag_max = _require_integer(self.lag_max_samples, "lag_max_samples")
        strain = _require_real(self.max_shift_strain, "max_shift_strain")
        valid_fraction = _require_real(self.minimum_valid_fraction, "minimum_valid_fraction")

        if side_offset <= 0.0:
            raise ValueError("side_offset_grid must be greater than zero")
        if window_radius < 1:
            raise ValueError("window_radius_samples must be at least one")
        if lag_min > 0 or lag_max < 0:
            raise ValueError("lag range must include zero")
        if lag_min >= lag_max:
            raise ValueError("lag_min_samples must be less than lag_max_samples")
        if strain <= 0.0 or strain > 1.0:
            raise ValueError("max_shift_strain must be in (0, 1]")
        if valid_fraction <= 0.0 or valid_fraction > 1.0:
            raise ValueError("minimum_valid_fraction must be in (0, 1]")
        if not isinstance(self.similarity_metric, str) or self.similarity_metric != "zncc":
            raise ValueError("similarity_metric must be 'zncc'")
        if type(self.subsample_refinement) is not bool:
            raise TypeError("subsample_refinement must be an exact bool")


@dataclass(frozen=True, slots=True, eq=False)
class FaultWarpingResult:
    """Row-aligned apparent-shift estimates and diagnostics for a surface graph.

    ``shift_samples`` is an apparent sample-axis shift, not a true fault slip.
    Invalid rows have ``NaN`` for every floating diagnostic and ``False`` for
    ``boundary_hit``.
    """

    valid: BoolArray
    shift_samples: Float32Array
    correlation_before: Float32Array
    correlation_after: Float32Array
    cost_margin: Float32Array
    cycle_residual_samples: Float32Array
    valid_sample_fraction: Float32Array
    boundary_hit: BoolArray

    def __post_init__(self) -> None:
        valid = _require_bool_vector(self.valid, "valid")
        float_arrays = (
            _require_float32_vector(self.shift_samples, "shift_samples"),
            _require_float32_vector(self.correlation_before, "correlation_before"),
            _require_float32_vector(self.correlation_after, "correlation_after"),
            _require_float32_vector(self.cost_margin, "cost_margin"),
            _require_float32_vector(self.cycle_residual_samples, "cycle_residual_samples"),
            _require_float32_vector(self.valid_sample_fraction, "valid_sample_fraction"),
        )
        boundary_hit = _require_bool_vector(self.boundary_hit, "boundary_hit")
        arrays = (valid,) + float_arrays + (boundary_hit,)
        names = (
            "valid",
            "shift_samples",
            "correlation_before",
            "correlation_after",
            "cost_margin",
            "cycle_residual_samples",
            "valid_sample_fraction",
            "boundary_hit",
        )
        _require_same_length(arrays, names)

        if np.any(valid):
            for array, name in zip(float_arrays, names[1:-1]):
                if not np.all(np.isfinite(array[valid])):
                    raise ValueError(f"{name} must be finite for valid rows")

        invalid = ~valid
        if np.any(invalid):
            for array, name in zip(float_arrays, names[1:-1]):
                if not np.all(np.isnan(array[invalid])):
                    raise ValueError(f"{name} must be NaN for invalid rows")
            if np.any(boundary_hit[invalid]):
                raise ValueError("boundary_hit must be False for invalid rows")

        if np.any(valid):
            if np.any(self.correlation_before[valid] < -1.0) or np.any(
                self.correlation_before[valid] > 1.0
            ):
                raise ValueError("correlation_before must be in [-1, 1] for valid rows")
            if np.any(self.correlation_after[valid] < -1.0) or np.any(
                self.correlation_after[valid] > 1.0
            ):
                raise ValueError("correlation_after must be in [-1, 1] for valid rows")
            if np.any(self.cost_margin[valid] < 0.0):
                raise ValueError("cost_margin must be nonnegative for valid rows")
            if np.any(self.cycle_residual_samples[valid] < 0.0):
                raise ValueError("cycle_residual_samples must be nonnegative for valid rows")
            if np.any(self.valid_sample_fraction[valid] < 0.0) or np.any(
                self.valid_sample_fraction[valid] > 1.0
            ):
                raise ValueError("valid_sample_fraction must be in [0, 1] for valid rows")

    @property
    def correlation_gain(self) -> Float32Array:
        """Return ``correlation_after - correlation_before`` with invalid NaNs preserved."""
        return self.correlation_after - self.correlation_before
