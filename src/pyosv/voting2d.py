"""2D optimal-path voting entry points."""

from __future__ import annotations

import math
import numbers
import operator
from collections.abc import Sequence

import numpy as np

from pyosv.cells import FaultCell2
from pyosv.dp import find_path_2d, shift_range, strain_to_bstrain, update_shift_ranges
from pyosv.filters import smooth2d

__all__ = ["OptimalPathVoter"]


class OptimalPathVoter:
    """Configuration and state holder for 2D optimal-path voting."""

    def __init__(self, ru: int, rv: int) -> None:
        self.ru = _validate_nonnegative_int(ru, "ru")
        self.rv = _validate_nonnegative_int(rv, "rv")
        self.lmin, self.lmax, self.nl = shift_range(self.ru)
        self.bstrain1 = 4
        self.attribute_smoothing = 1
        self.path_smoothing1 = 2.0
        self.lmins: np.ndarray
        self.lmaxs: np.ndarray
        self._update_shift_ranges()

    def set_strain_max(self, strain_max1: float) -> None:
        """Set the maximum fault-curve strain in the first dimension."""

        self.bstrain1 = strain_to_bstrain(strain_max1)

    def set_attribute_smoothing(self, attribute_smoothing: int) -> None:
        """Set the number of nonlinear smoothings for fault attributes."""

        self.attribute_smoothing = _validate_nonnegative_int(
            attribute_smoothing,
            "attribute_smoothing",
        )

    def set_path_smoothing(self, path_smoothing1: float) -> None:
        """Set the smoothing extent used for extracted fault paths."""

        self.path_smoothing1 = _validate_nonnegative_float(
            path_smoothing1,
            "path_smoothing1",
        )

    def _update_shift_ranges(self) -> None:
        self.lmins, self.lmaxs = update_shift_ranges(self.ru, self.rv)

    def pick_seeds(
        self,
        d: int,
        fm: float,
        ft: np.ndarray,
        pt: np.ndarray,
    ) -> list[FaultCell2]:
        """Pick 2D seed cells above a fault-likelihood threshold."""

        distance = _validate_nonnegative_int(d, "d")
        ft_array, pt_array = _validate_matching_2d_arrays(ft, pt, "ft", "pt")
        threshold = np.float32(fm)
        n2, n1 = ft_array.shape

        candidates = [
            FaultCell2(i1, i2, ft_array[i2, i1], pt_array[i2, i1])
            for i2 in range(n2)
            for i1 in range(n1)
            if ft_array[i2, i1] > threshold
        ]
        candidates.sort(key=operator.attrgetter("fl"))

        mark = np.zeros((n2, n1), dtype=np.bool_)
        seeds: list[FaultCell2] = []
        for cell in reversed(candidates):
            i1 = cell.i1
            i2 = cell.i2
            b1 = max(i1 - distance, 0)
            b2 = max(i2 - distance, 0)
            e1 = min(i1 + distance, n1 - 1)
            e2 = min(i2 + distance, n2 - 1)
            if mark[b2 : e2 + 1, b1 : e1 + 1].any():
                continue

            seeds.append(cell)
            mark[i2, i1] = True

        return seeds

    def get_seeds(
        self,
        c1: int,
        c2: int,
        ft: np.ndarray,
        pt: np.ndarray,
    ) -> list[FaultCell2]:
        """Return the seed at one image sample."""

        ft_array, pt_array = _validate_matching_2d_arrays(ft, pt, "ft", "pt")
        i1 = _validate_int(c1, "c1")
        i2 = _validate_int(c2, "c2")
        n2, n1 = ft_array.shape
        if not 0 <= i1 < n1:
            raise ValueError("c1 must be inside the image bounds")
        if not 0 <= i2 < n2:
            raise ValueError("c2 must be inside the image bounds")

        return [FaultCell2(i1, i2, ft_array[i2, i1], pt_array[i2, i1])]

    def apply_voting(
        self,
        d: int,
        fm: float,
        ft: np.ndarray,
        pt: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run 2D path voting for all selected seeds."""

        ft_array, pt_array = _validate_matching_finite_arrays2(ft, pt, "ft", "pt")
        seeds = self.pick_seeds(d, fm, ft_array, pt_array)

        fe = np.zeros_like(ft_array, dtype=np.float32)
        fc = np.zeros_like(ft_array, dtype=np.float32)
        w1 = np.zeros_like(ft_array, dtype=np.float32)
        w2 = np.zeros_like(ft_array, dtype=np.float32)

        for seed in seeds:
            self._path_voting(seed, ft_array, fc, fe, w1, w2)

        fv = _normalize_and_power_2d(fe)
        return fv, w1, w2

    def update_vector_map(self, radius: int, vector: np.ndarray) -> np.ndarray:
        """Return displacement vectors for offsets ``[-radius, radius]``."""

        radius_int = _validate_nonnegative_int(radius, "radius")
        vector_array = _validate_vector2(vector, "vector")
        offsets = np.arange(-radius_int, radius_int + 1, dtype=np.float32)
        return vector_array[:, np.newaxis] * offsets[np.newaxis, :]

    def samples_in_uv_box(
        self,
        c1: int,
        c2: int,
        normal: np.ndarray,
        strike: np.ndarray,
        fx: np.ndarray,
    ) -> np.ndarray:
        """Sample ``1 - fx`` in the seed-centered local ``(v, u)`` box."""

        fx_array = _validate_array2(fx, "fx")
        n2, n1 = fx_array.shape
        i1 = _validate_int(c1, "c1")
        i2 = _validate_int(c2, "c2")
        if not 0 <= i1 < n1:
            raise ValueError("c1 must be inside the image bounds")
        if not 0 <= i2 < n2:
            raise ValueError("c2 must be inside the image bounds")

        normal_map = self.update_vector_map(self.ru, normal)
        strike_map = self.update_vector_map(self.rv, strike)
        x1 = i1 + strike_map[0, :, np.newaxis] + normal_map[0, np.newaxis, :]
        x2 = i2 + strike_map[1, :, np.newaxis] + normal_map[1, np.newaxis, :]
        j1 = np.floor(x1 + 0.5).astype(np.intp, copy=False)
        j2 = np.floor(x2 + 0.5).astype(np.intp, copy=False)
        np.clip(j1, 0, n1 - 1, out=j1)
        np.clip(j2, 0, n2 - 1, out=j2)

        sampled = (np.float32(1.0) - fx_array[j2, j1]).astype(np.float32, copy=False)
        costs = np.ones((2 * self.rv + 1, 2 * self.ru + 1), dtype=np.float32)
        for kv in range(costs.shape[0]):
            ku_min = self.lmins[kv] + self.ru
            ku_max = self.lmaxs[kv] + self.ru
            costs[kv, ku_min : ku_max + 1] = sampled[kv, ku_min : ku_max + 1]

        return costs

    def _path_voting(
        self,
        cell: FaultCell2,
        ft: np.ndarray,
        fc: np.ndarray,
        fe: np.ndarray,
        w1: np.ndarray,
        w2: np.ndarray,
    ) -> None:
        """Accumulate one seed cell's 2D optimal-path vote in-place."""

        ft_array, fc_array, fe_array, w1_array, w2_array = _validate_matching_arrays2(
            (ft, fc, fe, w1, w2),
            ("ft", "fc", "fe", "w1", "w2"),
        )
        n2, n1 = ft_array.shape
        c1 = cell.i1
        c2 = cell.i2
        if not 0 <= c1 < n1:
            raise ValueError("cell.i1 must be inside the image bounds")
        if not 0 <= c2 < n2:
            raise ValueError("cell.i2 must be inside the image bounds")

        normal = cell.fault_normal()
        strike = cell.fault_strike_vector()
        costs = self.samples_in_uv_box(c1, c2, normal, strike, ft_array)
        path = find_path_2d(
            costs,
            lmin=self.lmin,
            bstrain=self.bstrain1,
            attribute_smoothing=self.attribute_smoothing,
            path_smoothing=self.path_smoothing1,
        )

        valid_path_samples: list[tuple[int, int, float, float]] = []
        fa = np.float32(0.0)
        for kv, iu in enumerate(path):
            if kv == 0:
                p2i = np.float32(path[1] - iu) if path.size > 1 else np.float32(0.0)
            else:
                p2i = np.float32(iu - path[kv - 1])

            iv = kv - self.rv
            x1 = c1 + iu * normal[0] + iv * strike[0]
            x2 = c2 + iu * normal[1] + iv * strike[1]
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            if i1 <= 0 or i1 >= n1 - 1 or i2 <= 0 or i2 >= n2 - 1:
                continue

            fa += ft_array[i2, i1]
            psi = np.float32(1.0 / math.sqrt(float(p2i * p2i + np.float32(1.0))))
            valid_path_samples.append((i2, i1, np.float32(-psi), p2i * psi))

        if not valid_path_samples:
            return

        fa /= np.float32(len(valid_path_samples))
        for i2, i1, p1i, p2i in valid_path_samples:
            fe_array[i2, i1] += fa
            if fa > fc_array[i2, i1]:
                fc_array[i2, i1] = fa
                w1_array[i2, i1] = normal[0] * p1i + strike[0] * p2i
                w2_array[i2, i1] = normal[1] * p1i + strike[1] * p2i

    def seed_to_image(
        self,
        fmin: float,
        shape: tuple[int, int],
        cells: Sequence[FaultCell2],
        ep: np.ndarray,
    ) -> np.ndarray:
        """Rasterize seed cells whose image value is greater than ``fmin``."""

        n2, n1 = _validate_shape2(shape, "shape")
        ep_array = np.asarray(ep)
        if ep_array.shape != (n2, n1):
            raise ValueError("ep shape must match shape")

        fs = np.zeros((n2, n1), dtype=np.float32)
        for cell in cells:
            i1 = cell.i1
            i2 = cell.i2
            if ep_array[i2, i1] > fmin:
                fs[i2, i1] = ep_array[i2, i1]

        return fs

    def seed_to_points(self, cells: Sequence[FaultCell2]) -> np.ndarray:
        """Convert seed cells to a ``(2, nseed)`` point array."""

        xs = np.empty((2, len(cells)), dtype=np.float32)
        for ic, cell in enumerate(cells):
            xs[0, ic] = cell.i1
            xs[1, ic] = cell.i2

        return xs


def _normalize_and_power_2d(
    x: np.ndarray,
    *,
    sigma: float = 1.0,
    power: int = 4,
) -> np.ndarray:
    x_array = _validate_array2(x, "x").astype(np.float32, copy=True)
    sigma_float = _validate_nonnegative_float(sigma, "sigma")
    power_int = _validate_positive_int(power, "power")

    if x_array.size == 0:
        return x_array

    if sigma_float > 0.0:
        x_array = smooth2d(x_array, sigma_float).astype(np.float32, copy=False)

    x_array -= np.min(x_array)
    max_value = np.max(x_array)
    if max_value > 0.0:
        x_array /= max_value

    enhanced = np.float32(1.0) - np.power(np.float32(1.0) - x_array, power_int)
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32, copy=False)


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


def _validate_matching_2d_arrays(
    first: np.ndarray,
    second: np.ndarray,
    first_name: str,
    second_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    first_array = np.asarray(first)
    second_array = np.asarray(second)

    if first_array.ndim != 2:
        raise ValueError(f"{first_name} must be a 2D array")

    if second_array.ndim != 2:
        raise ValueError(f"{second_name} must be a 2D array")

    if first_array.shape != second_array.shape:
        raise ValueError(f"{first_name} and {second_name} shapes must match")

    return first_array, second_array


def _validate_matching_finite_arrays2(
    first: np.ndarray,
    second: np.ndarray,
    first_name: str,
    second_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    first_array, second_array = _validate_matching_2d_arrays(
        first,
        second,
        first_name,
        second_name,
    )
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            first_float32 = first_array.astype(np.float32, copy=False)
            second_float32 = second_array.astype(np.float32, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{first_name} and {second_name} must contain numeric finite values",
        ) from exc

    if not np.isfinite(first_float32).all():
        raise ValueError(f"{first_name} must contain only finite values")
    if not np.isfinite(second_float32).all():
        raise ValueError(f"{second_name} must contain only finite values")

    return first_float32, second_float32


def _validate_matching_arrays2(
    arrays: tuple[np.ndarray, ...],
    names: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    if len(arrays) != len(names):
        raise ValueError("arrays and names must have the same length")
    if not arrays:
        raise ValueError("at least one array is required")

    validated = tuple(_validate_array2(array, name) for array, name in zip(arrays, names))
    shape = validated[0].shape
    first_name = names[0]
    for array, name in zip(validated[1:], names[1:]):
        if array.shape != shape:
            raise ValueError(f"{first_name} and {name} shapes must match")

    return validated


def _validate_array2(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")

    return array


def _validate_vector2(vector: np.ndarray, name: str) -> np.ndarray:
    vector_array = np.asarray(vector, dtype=np.float32)
    if vector_array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")

    return vector_array


def _validate_shape2(shape: tuple[int, int], name: str) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"{name} must have two dimensions")

    n2 = _validate_nonnegative_int(shape[0], f"{name}[0]")
    n1 = _validate_nonnegative_int(shape[1], f"{name}[1]")

    return n2, n1
