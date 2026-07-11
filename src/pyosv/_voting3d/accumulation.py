"""Surface-vote accumulation kernels and helpers."""

from __future__ import annotations

import math

import numpy as np

from pyosv._accel import njit


def _count_reference_face_center_votes(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    volume_shape: tuple[int, int, int],
) -> int:
    n3, n2, n1 = volume_shape
    face_count = 0
    for kw in range(surface.shape[0]):
        iw = kw - rw
        dw1 = c1 + iw * strike[0]
        dw2 = c2 + iw * strike[1]
        dw3 = c3 + iw * strike[2]
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv - rv
            i1 = math.floor(float(iu * normal[0] + iv * dip[0] + dw1) + 0.5)
            i2 = math.floor(float(iu * normal[1] + iv * dip[1] + dw2) + 0.5)
            i3 = math.floor(float(iu * normal[2] + iv * dip[2] + dw3) + 0.5)
            if not _is_valid_surface_vote_sample(i1, i2, i3, n1, n2, n3):
                continue
            if i1 == 0 or i1 == n1 - 1 or i2 == 0 or i2 == n2 - 1 or i3 == 0 or i3 == n3 - 1:
                face_count += 1
    return face_count


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
            if not _is_valid_surface_vote_sample(i1, i2, i3, n1, n2, n3):
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
            if not _is_valid_surface_vote_sample(i1, i2, i3, n1, n2, n3):
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


def _accumulate_surface_votes_masked_python(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    w_offset: int,
    v_offset: int,
    lmin: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    align_i3: bool,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    valid_lag_mask: np.ndarray,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> tuple[int, int, int]:
    n3, n2, n1 = fe.shape
    nu = valid_lag_mask.shape[2]
    center_count = 0
    face_count = 0
    invalid_count = 0
    for kw in range(surface.shape[0]):
        iw = kw + w_offset - rw
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            ku = math.floor(float(iu - lmin) + 0.5)
            if not 0 <= ku < nu or not valid_lag_mask[kw, kv, ku]:
                invalid_count += 1
                continue
            iv = kv + v_offset - rv
            x1 = np.float32(
                float(c1)
                + float(iw) * float(strike[0])
                + float(iv) * float(dip[0])
                + float(iu) * float(normal[0])
            )
            x2 = np.float32(
                float(c2)
                + float(iw) * float(strike[1])
                + float(iv) * float(dip[1])
                + float(iu) * float(normal[1])
            )
            x3 = np.float32(
                float(c3)
                + float(iw) * float(strike[2])
                + float(iv) * float(dip[2])
                + float(iu) * float(normal[2])
            )
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
                invalid_count += 1
                continue
            center_count += 1
            if i1 == 0 or i1 == n1 - 1 or i2 == 0 or i2 == n2 - 1 or i3 == 0 or i3 == n3 - 1:
                face_count += 1

    if invalid_count > 0:
        return 0, 0, invalid_count

    for kw in range(surface.shape[0]):
        iw = kw + w_offset - rw
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv + v_offset - rv
            x1 = np.float32(
                float(c1)
                + float(iw) * float(strike[0])
                + float(iv) * float(dip[0])
                + float(iu) * float(normal[0])
            )
            x2 = np.float32(
                float(c2)
                + float(iw) * float(strike[1])
                + float(iv) * float(dip[1])
                + float(iu) * float(normal[1])
            )
            x3 = np.float32(
                float(c3)
                + float(iw) * float(strike[2])
                + float(iv) * float(dip[2])
                + float(iu) * float(normal[2])
            )
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            _add_surface_vote(i3, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            if align_i3:
                _add_surface_vote(i3 - 1, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
                _add_surface_vote(i3 + 1, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            else:
                _add_surface_vote(i3, i2 - 1, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
                _add_surface_vote(i3, i2 + 1, i1, fa, vp_value, vt_value, fe, vp, vt, vm)

    return center_count, face_count, 0


@njit(cache=True)
def _accumulate_surface_votes_masked_numba(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    w_offset: int,
    v_offset: int,
    lmin: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    align_i3: bool,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    valid_lag_mask: np.ndarray,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> tuple[int, int, int]:
    n3, n2, n1 = fe.shape
    nu = valid_lag_mask.shape[2]
    center_count = 0
    face_count = 0
    invalid_count = 0
    for kw in range(surface.shape[0]):
        iw = kw + w_offset - rw
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            ku = math.floor(float(iu - lmin) + 0.5)
            if not 0 <= ku < nu or not valid_lag_mask[kw, kv, ku]:
                invalid_count += 1
                continue
            iv = kv + v_offset - rv
            x1 = np.float32(
                float(c1)
                + float(iw) * float(strike[0])
                + float(iv) * float(dip[0])
                + float(iu) * float(normal[0])
            )
            x2 = np.float32(
                float(c2)
                + float(iw) * float(strike[1])
                + float(iv) * float(dip[1])
                + float(iu) * float(normal[1])
            )
            x3 = np.float32(
                float(c3)
                + float(iw) * float(strike[2])
                + float(iv) * float(dip[2])
                + float(iu) * float(normal[2])
            )
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
                invalid_count += 1
                continue
            center_count += 1
            if i1 == 0 or i1 == n1 - 1 or i2 == 0 or i2 == n2 - 1 or i3 == 0 or i3 == n3 - 1:
                face_count += 1

    if invalid_count > 0:
        return 0, 0, invalid_count

    for kw in range(surface.shape[0]):
        iw = kw + w_offset - rw
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv + v_offset - rv
            x1 = np.float32(
                float(c1)
                + float(iw) * float(strike[0])
                + float(iv) * float(dip[0])
                + float(iu) * float(normal[0])
            )
            x2 = np.float32(
                float(c2)
                + float(iw) * float(strike[1])
                + float(iv) * float(dip[1])
                + float(iu) * float(normal[1])
            )
            x3 = np.float32(
                float(c3)
                + float(iw) * float(strike[2])
                + float(iv) * float(dip[2])
                + float(iu) * float(normal[2])
            )
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            _add_surface_vote_numba(i3, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            if align_i3:
                _add_surface_vote_numba(i3 - 1, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
                _add_surface_vote_numba(i3 + 1, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            else:
                _add_surface_vote_numba(i3, i2 - 1, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
                _add_surface_vote_numba(i3, i2 + 1, i1, fa, vp_value, vt_value, fe, vp, vt, vm)

    return center_count, face_count, 0


@njit(cache=True)
def _is_valid_surface_vote_sample(
    i1: int,
    i2: int,
    i3: int,
    n1: int,
    n2: int,
    n3: int,
) -> bool:
    return 0 <= i1 < n1 and 0 < i2 < n2 - 1 and 0 < i3 < n3 - 1


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
    *,
    use_numba: bool,
) -> None:
    if use_numba:
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


def _accumulate_surface_votes_masked(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    w_offset: int,
    v_offset: int,
    lmin: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    align_i3: bool,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    valid_lag_mask: np.ndarray,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
    *,
    use_numba: bool,
) -> tuple[int, int, int]:
    if use_numba:
        return _accumulate_surface_votes_masked_numba(
            c1,
            c2,
            c3,
            rv,
            rw,
            w_offset,
            v_offset,
            lmin,
            fa,
            vp_value,
            vt_value,
            align_i3,
            normal,
            dip,
            strike,
            surface,
            valid_lag_mask,
            fe,
            vp,
            vt,
            vm,
        )
    return _accumulate_surface_votes_masked_python(
        c1,
        c2,
        c3,
        rv,
        rw,
        w_offset,
        v_offset,
        lmin,
        fa,
        vp_value,
        vt_value,
        align_i3,
        normal,
        dip,
        strike,
        surface,
        valid_lag_mask,
        fe,
        vp,
        vt,
        vm,
    )
