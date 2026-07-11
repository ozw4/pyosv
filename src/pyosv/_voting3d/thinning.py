"""Internal 3D voting thinning helpers."""

from __future__ import annotations

import numpy as np

from pyosv.filters import smooth3d
from pyosv.interp import sample3
from pyosv.thinning3d import reference_like_3d_thin_values


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


def _thin_reference_like_3d(
    fv: np.ndarray,
    vp: np.ndarray,
    *,
    reference_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    return reference_like_3d_thin_values(
        fv,
        vp,
        sigma=reference_sigma,
        reinforce_vertical=True,
    )


def _thin_fault_normal_3d(
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
) -> np.ndarray:
    n3, n2, n1 = fv.shape
    thinned = np.zeros((n3, n2, n1), dtype=np.float32)
    if fv.size == 0:
        return thinned

    fs = smooth3d(fv, 1.0).astype(np.float32, copy=False)
    i3, i2, i1 = np.indices((n3, n2, n1), dtype=np.float32)
    w1, w2, w3 = _fault_normal_components_from_strike_and_dip(vp, vt)

    fp = sample3(fs, i1 + w1, i2 + w2, i3 + w3, order=1, mode="nearest")
    fm = sample3(fs, i1 - w1, i2 - w2, i3 - w3, order=1, mode="nearest")
    keep = (fp < fs) & (fm < fs)
    thinned[keep] = fv[keep]
    return thinned


def _thin_fault_normal_plateau_3d(
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    plateau_tie_breaker: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    n3, n2, n1 = fv.shape
    thinned = np.zeros((n3, n2, n1), dtype=np.float32)
    if fv.size == 0:
        return thinned

    fs = smooth3d(fv, 1.0).astype(np.float32, copy=False)
    i3, i2, i1 = np.indices((n3, n2, n1), dtype=np.float32)
    w1, w2, w3 = _fault_normal_components_from_strike_and_dip(vp, vt)

    fp = sample3(fs, i1 + w1, i2 + w2, i3 + w3, order=1, mode="nearest")
    fm = sample3(fs, i1 - w1, i2 - w2, i3 - w3, order=1, mode="nearest")
    eps = np.float32(1.0e-6)
    candidate = (fv > eps) & (fs >= fp - np.float32(tolerance)) & (fs >= fm - np.float32(tolerance))
    if not np.any(candidate):
        return thinned

    dominant_axis = np.argmax(
        np.stack((np.abs(w1), np.abs(w2), np.abs(w3)), axis=0),
        axis=0,
    )
    axis_to_array_axis = (2, 1, 0)
    for normal_axis, array_axis in enumerate(axis_to_array_axis):
        axis_candidates = candidate & (dominant_axis == normal_axis)
        _collapse_candidate_runs_along_axis(
            fv,
            plateau_tie_breaker,
            axis_candidates,
            array_axis,
            thinned,
        )

    return thinned


def _collapse_candidate_runs_along_axis(
    fv: np.ndarray,
    tie_breaker: np.ndarray,
    candidate: np.ndarray,
    axis: int,
    thinned: np.ndarray,
) -> None:
    moved_candidate = np.moveaxis(candidate, axis, -1)
    moved_fv = np.moveaxis(fv, axis, -1)
    moved_tie_breaker = np.moveaxis(tie_breaker, axis, -1)
    moved_thinned = np.moveaxis(thinned, axis, -1)

    line_length = moved_candidate.shape[-1]
    for line_index in np.ndindex(moved_candidate.shape[:-1]):
        line = moved_candidate[line_index]
        start: int | None = None
        for offset in range(line_length + 1):
            in_run = offset < line_length and bool(line[offset])
            if in_run and start is None:
                start = offset
            if (not in_run) and start is not None:
                _retain_plateau_run_sample(
                    moved_fv[line_index],
                    moved_tie_breaker[line_index],
                    moved_thinned[line_index],
                    start,
                    offset,
                )
                start = None


def _retain_plateau_run_sample(
    fv_line: np.ndarray,
    tie_breaker_line: np.ndarray,
    thinned_line: np.ndarray,
    start: int,
    stop: int,
) -> None:
    run_tie_breaker = tie_breaker_line[start:stop]
    if np.all(run_tie_breaker == run_tie_breaker[0]):
        keep_offset = (start + stop - 1) // 2
    else:
        keep_offset = start + int(np.argmax(run_tie_breaker))
    thinned_line[keep_offset] = fv_line[keep_offset]


def _edge_region_mask_3d(shape: tuple[int, int, int], margin: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.bool_)
    if margin == 0 or mask.size == 0:
        return mask

    n3, n2, n1 = shape
    m3 = min(margin, n3)
    m2 = min(margin, n2)
    m1 = min(margin, n1)
    mask[:m3, :, :] = True
    mask[n3 - m3 :, :, :] = True
    mask[:, :m2, :] = True
    mask[:, n2 - m2 :, :] = True
    mask[:, :, :m1] = True
    mask[:, :, n1 - m1 :] = True
    return mask


def _local_candidate_count_3d(candidate: np.ndarray) -> np.ndarray:
    candidate_array = np.asarray(candidate, dtype=np.uint8)
    padded = np.pad(candidate_array, 1, mode="constant", constant_values=0)
    counts = np.zeros(candidate_array.shape, dtype=np.uint8)
    for d3 in range(3):
        for d2 in range(3):
            for d1 in range(3):
                counts += padded[
                    d3 : d3 + candidate_array.shape[0],
                    d2 : d2 + candidate_array.shape[1],
                    d1 : d1 + candidate_array.shape[2],
                ]
    return counts


def _orientation_roughness_3d(
    vp: np.ndarray,
    vt: np.ndarray,
    support: np.ndarray | None = None,
) -> np.ndarray:
    if vp.size == 0:
        return np.zeros(vp.shape, dtype=np.float32)

    if support is None:
        support_array = np.ones(vp.shape, dtype=np.bool_)
    else:
        support_array = np.asarray(support, dtype=np.bool_)
        if support_array.shape != vp.shape:
            raise ValueError("support shape must match vp shape")

    roughness_squared = np.zeros(vp.shape, dtype=np.float32)
    for axis in range(3):
        if vp.shape[axis] < 2:
            continue

        strike_diff = _strike_difference_degrees(
            np.diff(vp, axis=axis).astype(np.float32, copy=False)
        )
        dip_diff = np.diff(vt, axis=axis).astype(np.float32, copy=False)
        diff_squared = strike_diff ** np.float32(2.0) + dip_diff ** np.float32(2.0)

        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        pair_support = support_array[tuple(lower)] & support_array[tuple(upper)]
        diff_squared = np.where(pair_support, diff_squared, np.float32(0.0)).astype(
            np.float32,
            copy=False,
        )
        np.maximum(
            roughness_squared[tuple(lower)], diff_squared, out=roughness_squared[tuple(lower)]
        )
        np.maximum(
            roughness_squared[tuple(upper)], diff_squared, out=roughness_squared[tuple(upper)]
        )

    return np.sqrt(roughness_squared).astype(np.float32, copy=False)


def _strike_difference_degrees(delta: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(delta).astype(np.float32, copy=False)
    wrapped = 0.5 * np.rad2deg(np.arctan2(np.sin(2.0 * radians), np.cos(2.0 * radians)))
    return np.abs(wrapped).astype(np.float32, copy=False)
