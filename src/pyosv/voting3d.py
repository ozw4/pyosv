"""3D optimal-surface voting entry points."""

from __future__ import annotations

import math
import numbers
import operator

import numpy as np

from pyosv._accel import NUMBA_AVAILABLE, njit
from pyosv.cells import FaultCell
from pyosv.dp import (
    find_surface_3d,
    shift_range,
    smooth_surface_2d,
    strain_to_bstrain,
    update_shift_ranges_3d,
)
from pyosv.filters import smooth3d
from pyosv.geometry import range360
from pyosv.interp import sample3

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

    def apply_voting(
        self,
        d: int,
        fm: float,
        ft: np.ndarray,
        pt: np.ndarray,
        tt: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run 3D surface voting for all selected seeds."""

        ft_array, pt_array, tt_array = _validate_matching_finite_arrays3_many(
            (ft, pt, tt),
            ("ft", "pt", "tt"),
        )
        seeds = self.pick_seeds(d, fm, ft_array, pt_array, tt_array)
        fs = _smooth_fault_likelihood_3d(ft_array)

        fe = np.zeros_like(ft_array, dtype=np.float32)
        vp = np.zeros_like(ft_array, dtype=np.float32)
        vt = np.zeros_like(ft_array, dtype=np.float32)
        vm = np.zeros_like(ft_array, dtype=np.float32)

        for seed in seeds:
            self._surface_voting(seed, fs, fe, vp, vt, vm)

        fv = _normalize_and_power_3d(fe)
        return fv, vp, vt

    def thin(self, fv: np.ndarray, vp: np.ndarray, vt: np.ndarray) -> np.ndarray:
        """Keep 3D voting-score maxima along the local fault-normal field."""

        fv_array, vp_array, vt_array = _validate_matching_finite_arrays3_many(
            (fv, vp, vt),
            ("fv", "vp", "vt"),
        )
        n3, n2, n1 = fv_array.shape
        thinned = np.zeros((n3, n2, n1), dtype=np.float32)
        if fv_array.size == 0:
            return thinned

        fs = smooth3d(fv_array, 1.0).astype(np.float32, copy=False)
        i3, i2, i1 = np.indices((n3, n2, n1), dtype=np.float32)
        w1, w2, w3 = _fault_normal_components_from_strike_and_dip(vp_array, vt_array)

        fp = sample3(fs, i1 + w1, i2 + w2, i3 + w3, order=1, mode="nearest")
        fm = sample3(fs, i1 - w1, i2 - w2, i3 - w3, order=1, mode="nearest")
        keep = (fp < fs) & (fm < fs)
        thinned[keep] = fv_array[keep]
        return thinned

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

        if NUMBA_AVAILABLE:
            return _samples_in_uvw_box_numba(
                i1,
                i2,
                i3,
                self.ru,
                self.rv,
                self.rw,
                normal,
                dip,
                strike,
                fx_array,
                self.lmins,
                self.lmaxs,
            )
        return _samples_in_uvw_box_python(
            i1,
            i2,
            i3,
            self.ru,
            self.rv,
            self.rw,
            normal,
            dip,
            strike,
            fx_array,
            self.lmins,
            self.lmaxs,
        )

    def _surface_voting(
        self,
        cell: FaultCell,
        ft: np.ndarray,
        fe: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        vm: np.ndarray,
    ) -> None:
        """Accumulate one seed cell's 3D optimal-surface vote in-place."""

        ft_array, fe_array, vp_array, vt_array, vm_array = _validate_matching_arrays3(
            (ft, fe, vp, vt, vm),
            ("ft", "fe", "vp", "vt", "vm"),
        )
        n3, n2, n1 = ft_array.shape
        c1 = cell.i1
        c2 = cell.i2
        c3 = cell.i3
        if not 0 <= c1 < n1:
            raise ValueError("cell.i1 must be inside the image bounds")
        if not 0 <= c2 < n2:
            raise ValueError("cell.i2 must be inside the image bounds")
        if not 0 <= c3 < n3:
            raise ValueError("cell.i3 must be inside the image bounds")

        normal = cell.fault_normal()
        dip = cell.fault_dip_vector()
        strike = cell.fault_strike_vector()
        costs = self.samples_in_uvw_box(c1, c2, c3, normal, dip, strike, ft_array)
        surface = find_surface_3d(
            costs,
            lmin=self.lmin,
            bstrain1=self.bstrain1,
            bstrain2=self.bstrain2,
            attribute_smoothing=self.attribute_smoothing,
            surface_smoothing1=self.surface_smoothing1,
            surface_smoothing2=self.surface_smoothing2,
        )

        fa, valid_count = _surface_vote_average(
            c1,
            c2,
            c3,
            self.rv,
            self.rw,
            normal,
            dip,
            strike,
            surface,
            ft_array,
        )
        if valid_count == 0:
            return

        strike_angle, dip_angle = _surface_strike_and_dip(
            normal,
            dip,
            strike,
            surface,
            sigma=None,
        )
        vp_value = np.float32(strike_angle)
        vt_value = np.float32(dip_angle)
        align_i3 = abs(normal[2]) > abs(normal[1])

        _accumulate_surface_votes(
            c1,
            c2,
            c3,
            self.rv,
            self.rw,
            fa,
            vp_value,
            vt_value,
            align_i3,
            normal,
            dip,
            strike,
            surface,
            fe_array,
            vp_array,
            vt_array,
            vm_array,
        )


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


def _samples_in_uvw_box_python(
    c1: int,
    c2: int,
    c3: int,
    ru: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    fx: np.ndarray,
    lmins: np.ndarray,
    lmaxs: np.ndarray,
) -> np.ndarray:
    n3, n2, n1 = fx.shape
    costs = np.ones((2 * rw + 1, 2 * rv + 1, 2 * ru + 1), dtype=np.float32)
    for kw in range(costs.shape[0]):
        iw = kw - rw
        for kv in range(costs.shape[1]):
            iv = kv - rv
            ku_min = lmins[kw, kv] + ru
            ku_max = lmaxs[kw, kv] + ru
            for ku in range(ku_min, ku_max + 1):
                iu = ku - ru
                x1 = c1 + iw * strike[0] + iv * dip[0] + iu * normal[0]
                x2 = c2 + iw * strike[1] + iv * dip[1] + iu * normal[1]
                x3 = c3 + iw * strike[2] + iv * dip[2] + iu * normal[2]
                j1 = math.floor(float(x1) + 0.5)
                j2 = math.floor(float(x2) + 0.5)
                j3 = math.floor(float(x3) + 0.5)
                j1 = min(max(j1, 0), n1 - 1)
                j2 = min(max(j2, 0), n2 - 1)
                j3 = min(max(j3, 0), n3 - 1)
                costs[kw, kv, ku] = np.float32(1.0) - fx[j3, j2, j1]

    return costs


@njit(cache=True)
def _samples_in_uvw_box_numba(
    c1: int,
    c2: int,
    c3: int,
    ru: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    fx: np.ndarray,
    lmins: np.ndarray,
    lmaxs: np.ndarray,
) -> np.ndarray:
    n3, n2, n1 = fx.shape
    costs = np.ones((2 * rw + 1, 2 * rv + 1, 2 * ru + 1), dtype=np.float32)
    for kw in range(costs.shape[0]):
        iw = kw - rw
        for kv in range(costs.shape[1]):
            iv = kv - rv
            ku_min = lmins[kw, kv] + ru
            ku_max = lmaxs[kw, kv] + ru
            for ku in range(ku_min, ku_max + 1):
                iu = ku - ru
                x1 = c1 + iw * strike[0] + iv * dip[0] + iu * normal[0]
                x2 = c2 + iw * strike[1] + iv * dip[1] + iu * normal[1]
                x3 = c3 + iw * strike[2] + iv * dip[2] + iu * normal[2]
                j1 = math.floor(x1 + 0.5)
                j2 = math.floor(x2 + 0.5)
                j3 = math.floor(x3 + 0.5)
                j1 = min(max(j1, 0), n1 - 1)
                j2 = min(max(j2, 0), n2 - 1)
                j3 = min(max(j3, 0), n3 - 1)
                costs[kw, kv, ku] = np.float32(1.0) - fx[j3, j2, j1]

    return costs


def _surface_vote_average(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    ft: np.ndarray,
) -> tuple[np.float32, int]:
    if NUMBA_AVAILABLE:
        return _surface_vote_average_numba(c1, c2, c3, rv, rw, normal, dip, strike, surface, ft)
    return _surface_vote_average_python(c1, c2, c3, rv, rw, normal, dip, strike, surface, ft)


def _surface_vote_average_python(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    ft: np.ndarray,
) -> tuple[np.float32, int]:
    n3, n2, n1 = ft.shape
    fa = np.float32(0.0)
    valid_count = 0
    for kw in range(surface.shape[0]):
        iw = kw - rw
        dw1 = c1 + iw * strike[0]
        dw2 = c2 + iw * strike[1]
        dw3 = c3 + iw * strike[2]
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv - rv
            x1 = iu * normal[0] + iv * dip[0] + dw1
            x2 = iu * normal[1] + iv * dip[1] + dw2
            x3 = iu * normal[2] + iv * dip[2] + dw3
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
                continue

            fa += ft[i3, i2, i1]
            valid_count += 1

    if valid_count > 0:
        fa /= np.float32(valid_count)
    return fa, valid_count


@njit(cache=True)
def _surface_vote_average_numba(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    ft: np.ndarray,
) -> tuple[np.float32, int]:
    n3, n2, n1 = ft.shape
    fa = np.float32(0.0)
    valid_count = 0
    for kw in range(surface.shape[0]):
        iw = kw - rw
        dw1 = c1 + iw * strike[0]
        dw2 = c2 + iw * strike[1]
        dw3 = c3 + iw * strike[2]
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv - rv
            x1 = iu * normal[0] + iv * dip[0] + dw1
            x2 = iu * normal[1] + iv * dip[1] + dw2
            x3 = iu * normal[2] + iv * dip[2] + dw3
            i1 = math.floor(x1 + 0.5)
            i2 = math.floor(x2 + 0.5)
            i3 = math.floor(x3 + 0.5)
            if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
                continue

            fa += ft[i3, i2, i1]
            valid_count += 1

    if valid_count > 0:
        fa /= np.float32(valid_count)
    return fa, valid_count


def _accumulate_surface_votes(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    align_i3: bool,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> None:
    if NUMBA_AVAILABLE:
        _accumulate_surface_votes_numba(
            c1,
            c2,
            c3,
            rv,
            rw,
            fa,
            vp_value,
            vt_value,
            align_i3,
            normal,
            dip,
            strike,
            surface,
            fe,
            vp,
            vt,
            vm,
        )
        return
    _accumulate_surface_votes_python(
        c1,
        c2,
        c3,
        rv,
        rw,
        fa,
        vp_value,
        vt_value,
        align_i3,
        normal,
        dip,
        strike,
        surface,
        fe,
        vp,
        vt,
        vm,
    )


def _accumulate_surface_votes_python(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    align_i3: bool,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> None:
    n3, n2, n1 = fe.shape
    for kw in range(surface.shape[0]):
        iw = kw - rw
        dw1 = c1 + iw * strike[0]
        dw2 = c2 + iw * strike[1]
        dw3 = c3 + iw * strike[2]
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv - rv
            x1 = iu * normal[0] + iv * dip[0] + dw1
            x2 = iu * normal[1] + iv * dip[1] + dw2
            x3 = iu * normal[2] + iv * dip[2] + dw3
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
                continue

            _add_surface_vote(i3, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            if align_i3:
                _add_surface_vote(i3 - 1, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
                _add_surface_vote(i3 + 1, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            else:
                _add_surface_vote(i3, i2 - 1, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
                _add_surface_vote(i3, i2 + 1, i1, fa, vp_value, vt_value, fe, vp, vt, vm)


@njit(cache=True)
def _accumulate_surface_votes_numba(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    align_i3: bool,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> None:
    n3, n2, n1 = fe.shape
    for kw in range(surface.shape[0]):
        iw = kw - rw
        dw1 = c1 + iw * strike[0]
        dw2 = c2 + iw * strike[1]
        dw3 = c3 + iw * strike[2]
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv - rv
            x1 = iu * normal[0] + iv * dip[0] + dw1
            x2 = iu * normal[1] + iv * dip[1] + dw2
            x3 = iu * normal[2] + iv * dip[2] + dw3
            i1 = math.floor(x1 + 0.5)
            i2 = math.floor(x2 + 0.5)
            i3 = math.floor(x3 + 0.5)
            if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
                continue

            _add_surface_vote_numba(i3, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            if align_i3:
                _add_surface_vote_numba(
                    i3 - 1,
                    i2,
                    i1,
                    fa,
                    vp_value,
                    vt_value,
                    fe,
                    vp,
                    vt,
                    vm,
                )
                _add_surface_vote_numba(
                    i3 + 1,
                    i2,
                    i1,
                    fa,
                    vp_value,
                    vt_value,
                    fe,
                    vp,
                    vt,
                    vm,
                )
            else:
                _add_surface_vote_numba(
                    i3,
                    i2 - 1,
                    i1,
                    fa,
                    vp_value,
                    vt_value,
                    fe,
                    vp,
                    vt,
                    vm,
                )
                _add_surface_vote_numba(
                    i3,
                    i2 + 1,
                    i1,
                    fa,
                    vp_value,
                    vt_value,
                    fe,
                    vp,
                    vt,
                    vm,
                )


def _add_surface_vote(
    i3: int,
    i2: int,
    i1: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> None:
    n3, n2, n1 = fe.shape
    if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
        return
    fe[i3, i2, i1] += fa
    _update_orientation_if_stronger(i3, i2, i1, fa, vp_value, vt_value, vp, vt, vm)


@njit(cache=True)
def _add_surface_vote_numba(
    i3: int,
    i2: int,
    i1: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> None:
    n3, n2, n1 = fe.shape
    if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
        return
    fe[i3, i2, i1] += fa
    if fa > vm[i3, i2, i1]:
        vm[i3, i2, i1] = fa
        vp[i3, i2, i1] = vp_value
        vt[i3, i2, i1] = vt_value


def _update_orientation_if_stronger(
    i3: int,
    i2: int,
    i1: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> None:
    if fa > vm[i3, i2, i1]:
        vm[i3, i2, i1] = fa
        vp[i3, i2, i1] = vp_value
        vt[i3, i2, i1] = vt_value


def _fault_normal_components_from_strike_and_dip(
    phi: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.deg2rad(phi).astype(np.float32, copy=False)
    t = np.deg2rad(theta).astype(np.float32, copy=False)
    cp = np.cos(p)
    sp = np.sin(p)
    ct = np.cos(t)
    st = np.sin(t)
    w1 = -ct
    w2 = st * cp
    w3 = -st * sp
    return (
        w1.astype(np.float32, copy=False),
        w2.astype(np.float32, copy=False),
        w3.astype(np.float32, copy=False),
    )


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
