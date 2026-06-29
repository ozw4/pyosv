"""Fault-cell extraction from 3D voting outputs."""

from __future__ import annotations

from collections.abc import Iterable
import math
import numbers
import operator

import numpy as np

from pyosv.cells import FaultCell

__all__ = ["FaultSkinner"]


class FaultSkinner:
    """Configuration holder for extracting fault cells from voting volumes."""

    def __init__(
        self,
        min_likelihood: float = 0.0,
        min_skin_size: int | None = None,
        connectivity: str = "corner",
    ) -> None:
        self.min_likelihood = _validate_finite_float(min_likelihood, "min_likelihood")
        self.min_skin_size = _validate_optional_nonnegative_int(min_skin_size, "min_skin_size")
        self.connectivity = _validate_connectivity(connectivity)

    def cells_from_votes(
        self,
        fv: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        min_likelihood: float | None = None,
    ) -> list[FaultCell]:
        """Extract cells where ``fv >= min_likelihood``.

        Returned cells are sorted in global volume order: increasing ``i3``, then
        ``i2``, then ``i1``. Input volumes use the project-wide ``(n3, n2, n1)``
        shape convention.
        """

        fv_array, vp_array, vt_array = _validate_matching_finite_arrays3_many(
            (fv, vp, vt),
            ("fv", "vp", "vt"),
        )
        threshold = (
            self.min_likelihood
            if min_likelihood is None
            else _validate_finite_float(min_likelihood, "min_likelihood")
        )

        cells: list[FaultCell] = []
        for i3, i2, i1 in np.argwhere(fv_array >= threshold):
            cells.append(
                FaultCell(
                    operator.index(i1),
                    operator.index(i2),
                    operator.index(i3),
                    fv_array[i3, i2, i1],
                    vp_array[i3, i2, i1],
                    vt_array[i3, i2, i1],
                ),
            )

        return cells


def _validate_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number")

    value_float = float(value)
    if not math.isfinite(value_float):
        raise ValueError(f"{name} must be a finite number")

    return value_float


def _validate_optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer or None")

    try:
        value_int = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer or None") from exc

    if value_int < 0:
        raise ValueError(f"{name} must be a nonnegative integer or None")

    return value_int


def _validate_connectivity(connectivity: str) -> str:
    if not isinstance(connectivity, str) or connectivity not in {"face", "edge", "corner"}:
        raise ValueError("connectivity must be 'face', 'edge', or 'corner'")

    return connectivity


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
    arrays: Iterable[np.ndarray],
    names: Iterable[str],
) -> tuple[np.ndarray, ...]:
    arrays_tuple = tuple(arrays)
    names_tuple = tuple(names)
    if len(arrays_tuple) != len(names_tuple):
        raise ValueError("arrays and names must have the same length")
    if not arrays_tuple:
        raise ValueError("at least one array is required")

    validated = tuple(
        _validate_array3(array, name) for array, name in zip(arrays_tuple, names_tuple)
    )
    shape = validated[0].shape
    first_name = names_tuple[0]
    for array, name in zip(validated[1:], names_tuple[1:]):
        if array.shape != shape:
            raise ValueError(f"{first_name} and {name} shapes must match")

    return validated


def _validate_array3(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array")

    return array
