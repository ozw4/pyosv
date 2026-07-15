"""Dense accepted-skin occupancy for reference-like skinning."""

from __future__ import annotations

from collections.abc import Sequence
import operator

import numpy as np

from pyosv._skinner.grid import _index_key
from pyosv._skinner.validation import _validate_nonnegative_int


class _SkinOccupancyMask:
    """Dense volume occupancy with exact inclusive-box boundary semantics."""

    def __init__(self, shape: Sequence[int]) -> None:
        try:
            dimensions = tuple(shape)
        except TypeError as exc:
            raise ValueError("shape must contain three positive integers") from exc
        if len(dimensions) != 3:
            raise ValueError("shape must contain three positive integers")

        validated: list[int] = []
        for dimension in dimensions:
            if isinstance(dimension, bool):
                raise ValueError("shape must contain three positive integers")
            try:
                size = operator.index(dimension)
            except TypeError as exc:
                raise ValueError("shape must contain three positive integers") from exc
            if size <= 0:
                raise ValueError("shape must contain three positive integers")
            validated.append(size)

        # NumPy bool occupies one byte, so this is one byte per volume voxel.
        self._mask = np.zeros(tuple(validated), dtype=np.bool_, order="C")
        # Preserve sparse-grid semantics outside the volume without expanding
        # every out-of-volume voxel into a dictionary entry.
        self._overflow_boxes: list[tuple[int, int, int, int, int, int]] = []

    def mark_box(
        self,
        i1: int,
        i2: int,
        i3: int,
        r1: int,
        r2: int,
        r3: int,
    ) -> None:
        bounds = self._box_bounds(i1, i2, i3, r1, r2, r3)
        box = self._clipped_slices(bounds)
        if box is not None:
            self._mask[box] = True
        if self._extends_outside_volume(bounds):
            self._overflow_boxes.append(bounds)

    def any_in_box(
        self,
        i1: int,
        i2: int,
        i3: int,
        r1: int,
        r2: int,
        r3: int,
    ) -> bool:
        bounds = self._box_bounds(i1, i2, i3, r1, r2, r3)
        box = self._clipped_slices(bounds)
        if box is not None and bool(self._mask[box].any()):
            return True
        if not self._extends_outside_volume(bounds):
            return False
        return any(self._boxes_overlap(bounds, marked) for marked in self._overflow_boxes)

    @staticmethod
    def _box_bounds(
        i1: int,
        i2: int,
        i3: int,
        r1: int,
        r2: int,
        r3: int,
    ) -> tuple[int, int, int, int, int, int]:
        center1, center2, center3 = _index_key(i1, i2, i3)
        radius1 = _validate_nonnegative_int(r1, "r1")
        radius2 = _validate_nonnegative_int(r2, "r2")
        radius3 = _validate_nonnegative_int(r3, "r3")

        return (
            center1 - radius1,
            center1 + radius1 + 1,
            center2 - radius2,
            center2 + radius2 + 1,
            center3 - radius3,
            center3 + radius3 + 1,
        )

    def _clipped_slices(
        self,
        bounds: tuple[int, int, int, int, int, int],
    ) -> tuple[slice, slice, slice] | None:
        begin1, end1, begin2, end2, begin3, end3 = bounds
        n3, n2, n1 = self._mask.shape

        begin1 = max(begin1, 0)
        begin2 = max(begin2, 0)
        begin3 = max(begin3, 0)
        end1 = min(end1, n1)
        end2 = min(end2, n2)
        end3 = min(end3, n3)
        if begin1 >= end1 or begin2 >= end2 or begin3 >= end3:
            return None

        return (slice(begin3, end3), slice(begin2, end2), slice(begin1, end1))

    def _extends_outside_volume(
        self,
        bounds: tuple[int, int, int, int, int, int],
    ) -> bool:
        begin1, end1, begin2, end2, begin3, end3 = bounds
        n3, n2, n1 = self._mask.shape
        return begin1 < 0 or end1 > n1 or begin2 < 0 or end2 > n2 or begin3 < 0 or end3 > n3

    @staticmethod
    def _boxes_overlap(
        first: tuple[int, int, int, int, int, int],
        second: tuple[int, int, int, int, int, int],
    ) -> bool:
        return (
            max(first[0], second[0]) < min(first[1], second[1])
            and max(first[2], second[2]) < min(first[3], second[3])
            and max(first[4], second[4]) < min(first[5], second[5])
        )
