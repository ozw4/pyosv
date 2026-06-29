"""3D optimal-surface voting entry points."""

from __future__ import annotations

import math
import numbers
import operator

import numpy as np

from pyosv.cells import FaultCell
from pyosv.dp import (
    shift_range,
    smooth_surface_2d,
    strain_to_bstrain,
    update_shift_ranges_3d,
)
from pyosv.filters import smooth3d
from pyosv.geometry import range360

__all__ = ["OptimalSurfaceVoter"]


class OptimalSurfaceVoter:
    """Configuration and state holder for 3D optimal-surface voting."""

    def __init__(self, ru: int, rv: int, rw: int) -> None:
        self.ru = _validate_nonnegative_int(ru, "ru")
        self.rv = _validate_nonnegative_int(rv, "rv")
        self.rw = _validate_nonnegative_int(rw, "rw")
        self.lmin, self.lmax, self.nl = shift_range(self.ru)
        self.bstrain1 = 4
        self.bstrain2 = 4
        self.attribute_smoothing = 1
        self.surface_smoothing1 = 2.0
        self.surface_smoothing2 = 2.0
        self.lmins: np.ndarray
        self.lmaxs: np.ndarray
        self._update_shift_ranges()

    def set_strain_max(self, strain_max1: float, strain_max2: float) -> None:
        """Set the maximum fault-surface strains in the first two dimensions."""

        bstrain1 = strain_to_bstrain(strain_max1)
        bstrain2 = strain_to_bstrain(strain_max2)
        self.bstrain1 = bstrain1
        self.bstrain2 = bstrain2

    def set_attribute_smoothing(self, attribute_smoothing: int) -> None:
        """Set the number of nonlinear smoothings for fault attributes."""

        self.attribute_smoothing = _validate_nonnegative_int(
            attribute_smoothing,
            "attribute_smoothing",
        )

    def set_surface_smoothing(
        self,
        surface_smoothing1: float,
        surface_smoothing2: float,
    ) -> None:
        """Set the smoothing extents used for extracted fault surfaces."""

        smoothing1 = _validate_nonnegative_float(
            surface_smoothing1,
            "surface_smoothing1",
        )
        smoothing2 = _validate_nonnegative_float(
            surface_smoothing2,
            "surface_smoothing2",
        )
        self.surface_smoothing1 = smoothing1
        self.surface_smoothing2 = smoothing2

    def _update_shift_ranges(self) -> None:
        self.lmins, self.lmaxs = update_shift_ranges_3d(self.ru, self.rv, self.rw)

    def pick_seeds(
        self,
        d: int,
        fm: float,
        ft: np.ndarray,
        pt: np.ndarray,
        tt: np.ndarray,
    ) -> list[FaultCell]:
        """Pick 3D seed cells above a fault-likelihood threshold."""

        distance = _validate_nonnegative_int(d, "d")
        ft_array, pt_array, tt_array = _validate_matching_finite_arrays3_many(
            (ft, pt, tt),
            ("ft", "pt", "tt"),
        )
        threshold = np.float32(fm)
        n3, n2, n1 = ft_array.shape

        candidates = [
            FaultCell(
                i1,
                i2,
                i3,
                ft_array[i3, i2, i1],
                pt_array[i3, i2, i1],
                tt_array[i3, i2, i1],
            )
            for i3 in range(n3)
            for i2 in range(n2)
            for i1 in range(n1)
            if ft_array[i3, i2, i1] > threshold
        ]
        candidates.sort(key=operator.attrgetter("fl"))

        mark = np.zeros((n3, n2, n1), dtype=np.bool_)
        seeds: list[FaultCell] = []
        for cell in reversed(candidates):
            i1 = cell.i1
            i2 = cell.i2
            i3 = cell.i3
            b1 = max(i1 - distance, 0)
            b2 = max(i2 - distance, 0)
            b3 = max(i3 - distance, 0)
            e1 = min(i1 + distance, n1 - 1)
            e2 = min(i2 + distance, n2 - 1)
            e3 = min(i3 + distance, n3 - 1)
            if mark[b3 : e3 + 1, b2 : e2 + 1, b1 : e1 + 1].any():
                continue

            seeds.append(cell)
            mark[i3, i2, i1] = True

        return seeds

    def get_seeds(
        self,
        c1: int,
        c2: int,
        c3: int,
        ft: np.ndarray,
        pt: np.ndarray,
        tt: np.ndarray,
    ) -> list[FaultCell]:
        """Return the seed at one image sample."""

        ft_array, pt_array, tt_array = _validate_matching_finite_arrays3_many(
            (ft, pt, tt),
            ("ft", "pt", "tt"),
        )
        i1 = _validate_int(c1, "c1")
        i2 = _validate_int(c2, "c2")
        i3 = _validate_int(c3, "c3")
        n3, n2, n1 = ft_array.shape
        if not 0 <= i1 < n1:
            raise ValueError("c1 must be inside the image bounds")
        if not 0 <= i2 < n2:
            raise ValueError("c2 must be inside the image bounds")
        if not 0 <= i3 < n3:
            raise ValueError("c3 must be inside the image bounds")

        return [
            FaultCell(
                i1,
                i2,
                i3,
                ft_array[i3, i2, i1],
                pt_array[i3, i2, i1],
                tt_array[i3, i2, i1],
            ),
        ]

    def update_vector_map(self, radius: int, vector: np.ndarray) -> np.ndarray:
        """Return displacement vectors for offsets ``[-radius, radius]``."""

        radius_int = _validate_nonnegative_int(radius, "radius")
        vector_array = _validate_vector3(vector, "vector")
        offsets = np.arange(-radius_int, radius_int + 1, dtype=np.float32)
        return vector_array[:, np.newaxis] * offsets[np.newaxis, :]

    def samples_in_uvw_box(
        self,
        c1: int,
        c2: int,
        c3: int,
        normal: np.ndarray,
        dip: np.ndarray,
        strike: np.ndarray,
        fx: np.ndarray,
    ) -> np.ndarray:
        """Sample ``1 - fx`` in the seed-centered local ``(w, v, u)`` box."""

        fx_array = _validate_array3(fx, "fx")
        n3, n2, n1 = fx_array.shape
        i1 = _validate_int(c1, "c1")
        i2 = _validate_int(c2, "c2")
        i3 = _validate_int(c3, "c3")
        if not 0 <= i1 < n1:
            raise ValueError("c1 must be inside the image bounds")
        if not 0 <= i2 < n2:
            raise ValueError("c2 must be inside the image bounds")
        if not 0 <= i3 < n3:
            raise ValueError("c3 must be inside the image bounds")

        normal_map = self.update_vector_map(self.ru, normal)
        dip_map = self.update_vector_map(self.rv, dip)
        strike_map = self.update_vector_map(self.rw, strike)
        x1 = (
            i1
            + strike_map[0, :, np.newaxis, np.newaxis]
            + dip_map[0, np.newaxis, :, np.newaxis]
            + normal_map[0, np.newaxis, np.newaxis, :]
        )
        x2 = (
            i2
            + strike_map[1, :, np.newaxis, np.newaxis]
            + dip_map[1, np.newaxis, :, np.newaxis]
            + normal_map[1, np.newaxis, np.newaxis, :]
        )
        x3 = (
            i3
            + strike_map[2, :, np.newaxis, np.newaxis]
            + dip_map[2, np.newaxis, :, np.newaxis]
            + normal_map[2, np.newaxis, np.newaxis, :]
        )
        j1 = np.floor(x1 + 0.5).astype(np.intp, copy=False)
        j2 = np.floor(x2 + 0.5).astype(np.intp, copy=False)
        j3 = np.floor(x3 + 0.5).astype(np.intp, copy=False)
        np.clip(j1, 0, n1 - 1, out=j1)
        np.clip(j2, 0, n2 - 1, out=j2)
        np.clip(j3, 0, n3 - 1, out=j3)

        sampled = (np.float32(1.0) - fx_array[j3, j2, j1]).astype(
            np.float32,
            copy=False,
        )
        costs = np.ones(
            (2 * self.rw + 1, 2 * self.rv + 1, 2 * self.ru + 1),
            dtype=np.float32,
        )
        for kw in range(costs.shape[0]):
            for kv in range(costs.shape[1]):
                ku_min = self.lmins[kw, kv] + self.ru
                ku_max = self.lmaxs[kw, kv] + self.ru
                costs[kw, kv, ku_min : ku_max + 1] = sampled[
                    kw,
                    kv,
                    ku_min : ku_max + 1,
                ]

        return costs


def _normalize_and_power_3d(
    x: np.ndarray,
    *,
    sigma: float = 1.0,
    power: int = 8,
) -> np.ndarray:
    x_array = _validate_finite_array3(x, "x").astype(np.float32, copy=True)
    sigma_float = _validate_nonnegative_float(sigma, "sigma")
    power_int = _validate_positive_int(power, "power")

    if x_array.size == 0:
        return x_array

    if sigma_float > 0.0:
        x_array = smooth3d(x_array, sigma_float).astype(np.float32, copy=False)

    _normalize_unit_range_in_place(x_array)
    enhanced = np.float32(1.0) - np.power(np.float32(1.0) - x_array, power_int)
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32, copy=False)


def _smooth_fault_likelihood_3d(
    ft: np.ndarray,
    *,
    sigma: float = 1.0,
) -> np.ndarray:
    ft_array = _validate_finite_array3(ft, "ft").astype(np.float32, copy=True)
    sigma_float = _validate_nonnegative_float(sigma, "sigma")

    if ft_array.size == 0:
        return ft_array

    if sigma_float > 0.0:
        ft_array = smooth3d(ft_array, sigma_float).astype(np.float32, copy=False)

    _normalize_unit_range_in_place(ft_array)
    return ft_array


def _surface_strike_and_dip(
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    *,
    sigma: float | None = None,
) -> tuple[float, float]:
    normal_array = _validate_finite_vector3(normal, "normal")
    dip_array = _validate_finite_vector3(dip, "dip")
    strike_array = _validate_finite_vector3(strike, "strike")
    surface_array = _validate_finite_array2(surface, "surface").astype(
        np.float32,
        copy=True,
    )
    if surface_array.shape[0] < 3 or surface_array.shape[1] < 3:
        raise ValueError("surface must have at least three samples along w and v")

    if sigma is not None:
        sigma_float = _validate_nonnegative_float(sigma, "sigma")
        if sigma_float > 0.0:
            surface_array = smooth_surface_2d(
                surface_array,
                sigma1=sigma_float,
                sigma2=sigma_float,
            ).astype(np.float32, copy=False)

    iw = surface_array.shape[0] // 2
    iv = surface_array.shape[1] // 2
    local_normal = np.array(
        [
            1.0,
            -0.5 * (surface_array[iw, iv + 1] - surface_array[iw, iv - 1]),
            -0.5 * (surface_array[iw + 1, iv] - surface_array[iw - 1, iv]),
        ],
        dtype=np.float32,
    )
    local_normal /= np.linalg.norm(local_normal)

    global_normal = (
        normal_array * local_normal[0]
        + dip_array * local_normal[1]
        + strike_array * local_normal[2]
    ).astype(np.float32, copy=False)
    normal_norm = np.linalg.norm(global_normal)
    if normal_norm == 0.0:
        raise ValueError("surface basis vectors must produce a nonzero normal")
    global_normal /= normal_norm

    if global_normal[0] > 0.0:
        global_normal = -global_normal

    dip_angle = float(np.rad2deg(np.arccos(np.clip(-global_normal[0], -1.0, 1.0))))
    strike_angle = range360(
        np.rad2deg(np.arctan2(-global_normal[2], global_normal[1])),
    )
    return strike_angle, dip_angle


def _normalize_unit_range_in_place(x: np.ndarray) -> None:
    x -= np.min(x)
    max_value = np.max(x)
    if max_value > 0.0:
        x /= max_value
    np.clip(x, 0.0, 1.0, out=x)


def _validate_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")

    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _validate_nonnegative_int(value: int, name: str) -> int:
    try:
        value_int = _validate_int(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc

    if value_int < 0:
        raise ValueError(f"{name} must be a nonnegative integer")

    return value_int


def _validate_positive_int(value: int, name: str) -> int:
    try:
        value_int = _validate_int(value, name)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc

    if value_int <= 0:
        raise ValueError(f"{name} must be a positive integer")

    return value_int


def _validate_nonnegative_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite nonnegative number")

    value_float = float(value)

    if not math.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")

    return value_float


def _validate_matching_finite_arrays3_many(
    arrays: tuple[np.ndarray, ...],
    names: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    validated = _validate_matching_arrays3(arrays, names)
    finite_arrays: list[np.ndarray] = []
    for array, name in zip(validated, names):
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                finite_array = array.astype(np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric finite values") from exc

        if not np.isfinite(finite_array).all():
            raise ValueError(f"{name} must contain only finite values")

        finite_arrays.append(finite_array)

    return tuple(finite_arrays)


def _validate_matching_arrays3(
    arrays: tuple[np.ndarray, ...],
    names: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    if len(arrays) != len(names):
        raise ValueError("arrays and names must have the same length")
    if not arrays:
        raise ValueError("at least one array is required")

    validated = tuple(_validate_array3(array, name) for array, name in zip(arrays, names))
    shape = validated[0].shape
    first_name = names[0]
    for array, name in zip(validated[1:], names[1:]):
        if array.shape != shape:
            raise ValueError(f"{first_name} and {name} shapes must match")

    return validated


def _validate_finite_array3(array: np.ndarray, name: str) -> np.ndarray:
    array = _validate_array3(array, name)
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric finite values") from exc

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")

    return array


def _validate_finite_array2(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")

    try:
        with np.errstate(over="ignore", invalid="ignore"):
            array = array.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric finite values") from exc

    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")

    return array


def _validate_array3(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array")

    return array


def _validate_vector3(vector: np.ndarray, name: str) -> np.ndarray:
    vector_array = np.asarray(vector, dtype=np.float32)
    if vector_array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")

    return vector_array


def _validate_finite_vector3(vector: np.ndarray, name: str) -> np.ndarray:
    vector_array = _validate_vector3(vector, name)
    if not np.isfinite(vector_array).all():
        raise ValueError(f"{name} must contain only finite values")

    return vector_array
