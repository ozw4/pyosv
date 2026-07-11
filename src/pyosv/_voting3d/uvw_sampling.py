"""UVW sampling validation and supported-region operations."""

from __future__ import annotations

import numpy as np

from pyosv._voting3d.models import _MaskedUVWBoxSamples, _TangentialRectangle
from pyosv._voting3d.validation import _validate_array3, _validate_int


def _validate_uvw_sampling_origin(
    c1: int,
    c2: int,
    c3: int,
    fx: np.ndarray,
) -> tuple[int, int, int, np.ndarray]:
    """Validate a UVW sampler volume and its seed coordinates."""

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
    return i1, i2, i3, fx_array


def _select_supported_origin_rectangle(
    supported_columns: np.ndarray,
    *,
    origin_w: int,
    origin_v: int,
) -> _TangentialRectangle | None:
    """Choose the deterministic largest all-supported rectangle around the origin."""

    supported = np.asarray(supported_columns, dtype=np.bool_)
    if supported.ndim != 2:
        raise ValueError("supported_columns must have shape (nw, nv)")
    nw, nv = supported.shape
    if not (0 <= origin_w < nw and 0 <= origin_v < nv):
        raise ValueError("origin must be inside supported_columns")
    if not supported[origin_w, origin_v]:
        return None

    invalid = (~supported).astype(np.int32, copy=False)
    prefix = np.pad(invalid, ((1, 0), (1, 0)), mode="constant")
    prefix = np.cumsum(np.cumsum(prefix, axis=0), axis=1)
    best_key: tuple[int, int, int, int, int, int] | None = None
    best_rectangle: _TangentialRectangle | None = None
    for w_start in range(origin_w + 1):
        for w_stop in range(origin_w + 1, nw + 1):
            for v_start in range(origin_v + 1):
                for v_stop in range(origin_v + 1, nv + 1):
                    invalid_count = (
                        prefix[w_stop, v_stop]
                        - prefix[w_start, v_stop]
                        - prefix[w_stop, v_start]
                        + prefix[w_start, v_start]
                    )
                    if invalid_count != 0:
                        continue
                    area = (w_stop - w_start) * (v_stop - v_start)
                    asymmetry = abs((origin_w - w_start) - (w_stop - 1 - origin_w)) + abs(
                        (origin_v - v_start) - (v_stop - 1 - origin_v)
                    )
                    key = (-area, asymmetry, w_start, v_start, w_stop, v_stop)
                    if best_key is None or key < best_key:
                        best_key = key
                        best_rectangle = _TangentialRectangle(
                            w_start=w_start,
                            v_start=v_start,
                            w_stop=w_stop,
                            v_stop=v_stop,
                        )
    return best_rectangle


def _crop_masked_uvw_box(
    samples: _MaskedUVWBoxSamples,
    rectangle: _TangentialRectangle,
) -> _MaskedUVWBoxSamples:
    nw, nv = samples.full_tangential_shape
    if not (
        0 <= rectangle.w_start < rectangle.w_stop <= nw
        and 0 <= rectangle.v_start < rectangle.v_stop <= nv
    ):
        raise ValueError("rectangle must be inside the full tangential box")
    selection = (
        slice(rectangle.w_start, rectangle.w_stop),
        slice(rectangle.v_start, rectangle.v_stop),
        slice(None),
    )
    return _MaskedUVWBoxSamples(
        costs=samples.costs[selection].copy(),
        valid_lag_mask=samples.valid_lag_mask[selection].copy(),
        w_offset=rectangle.w_start,
        v_offset=rectangle.v_start,
        full_tangential_shape=samples.full_tangential_shape,
        admissible_lag_count=samples.admissible_lag_count,
        in_bounds_lag_count=samples.in_bounds_lag_count,
    )


def _surface_center_lag(surface: np.ndarray) -> float | None:
    if surface.size == 0:
        return None
    return float(surface[surface.shape[0] // 2, surface.shape[1] // 2])
