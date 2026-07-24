"""Local/world coordinate transforms and nearest-volume sampling."""

from __future__ import annotations

import numpy as np

from pyosv._skinner.models import _LocalTransformMap
from pyosv._skinner.validation import (
    _validate_matching_finite_arrays3_many,
    _validate_nonnegative_int,
)
from pyosv.cells import _java_round


def _update_transform_map(
    ru: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
) -> _LocalTransformMap:
    """Build local transform maps with u=normal, v=dip, and w=strike axes."""

    ru_int = _validate_nonnegative_int(ru, "ru")
    rv_int = _validate_nonnegative_int(rv, "rv")
    rw_int = _validate_nonnegative_int(rw, "rw")
    normal_array = _validate_finite_vector3(normal, "normal")
    dip_array = _validate_finite_vector3(dip, "dip")
    strike_array = _validate_finite_vector3(strike, "strike")
    return _LocalTransformMap(
        us=_axis_transform_map(ru_int, normal_array),
        vs=_axis_transform_map(rv_int, dip_array),
        ws=_axis_transform_map(rw_int, strike_array),
    )


def _local_index_to_world(
    iu: int,
    iv: int,
    iw: int,
    origin: tuple[float, float, float],
    transform_map: _LocalTransformMap,
) -> tuple[np.float32, np.float32, np.float32]:
    """Map local array indices to world coordinates for a seed origin."""

    u_index = _validate_transform_index(iu, transform_map.us, "iu")
    v_index = _validate_transform_index(iv, transform_map.vs, "iv")
    w_index = _validate_transform_index(iw, transform_map.ws, "iw")
    o1, o2, o3 = _validate_origin3(origin)
    x1 = np.float32(
        o1
        + float(transform_map.us[0, u_index])
        + float(transform_map.vs[0, v_index])
        + float(transform_map.ws[0, w_index]),
    )
    x2 = np.float32(
        o2
        + float(transform_map.us[1, u_index])
        + float(transform_map.vs[1, v_index])
        + float(transform_map.ws[1, w_index]),
    )
    x3 = np.float32(
        o3
        + float(transform_map.us[2, u_index])
        + float(transform_map.vs[2, v_index])
        + float(transform_map.ws[2, w_index]),
    )
    return x1, x2, x3


def _sample_volume_nearest_java_round(
    fv: np.ndarray,
    x1: float,
    x2: float,
    x3: float,
) -> np.float32:
    """Sample a 3D volume with Java-round nearest neighbor and zero outside."""

    fv_array = _validate_matching_finite_arrays3_many((fv,), ("fv",))[0]
    return _sample_validated_volume_nearest_java_round(fv_array, x1, x2, x3)


def _sample_validated_volume_nearest_java_round(
    fv: np.ndarray,
    x1: float,
    x2: float,
    x3: float,
) -> np.float32:
    n3, n2, n1 = fv.shape
    i1 = _java_round(x1)
    i2 = _java_round(x2)
    i3 = _java_round(x3)
    if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
        return np.float32(0.0)

    return np.float32(fv[i3, i2, i1])


def _axis_transform_map(radius: int, vector: np.ndarray) -> np.ndarray:
    axis_map = np.zeros((3, 2 * radius + 1), dtype=np.float32)
    center = radius
    for step in range(1, radius + 1):
        positive = center + step
        negative = center - step
        offset = np.float32(step) * vector
        axis_map[:, positive] = offset
        axis_map[:, negative] = -offset

    return axis_map


def _validate_finite_vector3(vector: np.ndarray, name: str) -> np.ndarray:
    vector_array = np.asarray(vector, dtype=np.float32)
    if vector_array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    if not np.isfinite(vector_array).all():
        raise ValueError(f"{name} must contain only finite values")

    return vector_array


def _validate_origin3(origin: tuple[float, float, float]) -> tuple[float, float, float]:
    origin_array = np.asarray(origin, dtype=np.float32)
    if origin_array.shape != (3,):
        raise ValueError("origin must have shape (3,)")
    if not np.isfinite(origin_array).all():
        raise ValueError("origin must contain only finite values")

    return (float(origin_array[0]), float(origin_array[1]), float(origin_array[2]))


def _validate_transform_index(index: int, transform_axis: np.ndarray, name: str) -> int:
    index_int = _validate_nonnegative_int(index, name)
    if index_int >= transform_axis.shape[1]:
        raise ValueError(f"{name} must be inside the local transform map")

    return index_int
