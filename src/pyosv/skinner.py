"""Fault-cell extraction from 3D voting outputs."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
import math
import numbers
import operator

import numpy as np

from pyosv.cells import FaultCell, _java_round
from pyosv.skin import FaultSkin

__all__ = ["ConnectedComponentSkinner", "FaultSkinner", "find_skins"]


@dataclass(slots=True, eq=False)
class _SkinCell:
    """Mutable internal fault cell for reference-like linking and growth."""

    x1: float
    x2: float
    x3: float
    fl: float
    fp: float
    ft: float
    i1: int = field(init=False)
    i2: int = field(init=False)
    i3: int = field(init=False)
    ca: _SkinCell | None = field(default=None, repr=False)
    cb: _SkinCell | None = field(default=None, repr=False)
    cl: _SkinCell | None = field(default=None, repr=False)
    cr: _SkinCell | None = field(default=None, repr=False)
    skin_id: int | None = None

    def __post_init__(self) -> None:
        self.x1 = float(self.x1)
        self.x2 = float(self.x2)
        self.x3 = float(self.x3)
        self.fl = float(self.fl)
        self.fp = float(self.fp)
        self.ft = float(self.ft)
        self.i1 = _java_round(self.x1)
        self.i2 = _java_round(self.x2)
        self.i3 = _java_round(self.x3)

    @property
    def index(self) -> tuple[int, int, int]:
        return (self.i1, self.i2, self.i3)

    def to_fault_cell(self) -> FaultCell:
        return FaultCell(self.x1, self.x2, self.x3, self.fl, self.fp, self.ft)

    def fault_normal(self) -> np.ndarray:
        return self.to_fault_cell().fault_normal()

    def fault_dip_vector(self) -> np.ndarray:
        return self.to_fault_cell().fault_dip_vector()

    def fault_strike_vector(self) -> np.ndarray:
        return self.to_fault_cell().fault_strike_vector()


class _SkinCellGrid:
    """Sparse grid keyed by rounded fault-cell indices."""

    def __init__(self) -> None:
        self._cells: dict[tuple[int, int, int], _SkinCell] = {}

    def set(self, cell: _SkinCell) -> None:
        self._validate_cell(cell)
        self._cells[cell.index] = cell

    def get(self, i1: int, i2: int, i3: int) -> _SkinCell | None:
        return self._cells.get(_index_key(i1, i2, i3))

    def set_cells_in_box(self, cell: _SkinCell, r1: int, r2: int, r3: int) -> None:
        self._validate_cell(cell)
        radius1 = _validate_nonnegative_int(r1, "r1")
        radius2 = _validate_nonnegative_int(r2, "r2")
        radius3 = _validate_nonnegative_int(r3, "r3")

        for i3 in range(cell.i3 - radius3, cell.i3 + radius3 + 1):
            for i2 in range(cell.i2 - radius2, cell.i2 + radius2 + 1):
                for i1 in range(cell.i1 - radius1, cell.i1 + radius1 + 1):
                    self._cells[(i1, i2, i3)] = cell

    def find_cells_in_box(
        self,
        i1: int,
        i2: int,
        i3: int,
        r1: int,
        r2: int,
        r3: int,
    ) -> list[_SkinCell]:
        center1, center2, center3 = _index_key(i1, i2, i3)
        radius1 = _validate_nonnegative_int(r1, "r1")
        radius2 = _validate_nonnegative_int(r2, "r2")
        radius3 = _validate_nonnegative_int(r3, "r3")

        found: list[_SkinCell] = []
        seen: set[int] = set()
        for j3 in range(center3 - radius3, center3 + radius3 + 1):
            for j2 in range(center2 - radius2, center2 + radius2 + 1):
                for j1 in range(center1 - radius1, center1 + radius1 + 1):
                    cell = self._cells.get((j1, j2, j3))
                    if cell is not None and id(cell) not in seen:
                        found.append(cell)
                        seen.add(id(cell))

        found.sort(key=lambda cell: cell.index)
        return found

    @staticmethod
    def _validate_cell(cell: _SkinCell) -> None:
        if not isinstance(cell, _SkinCell):
            msg = "_SkinCellGrid only stores _SkinCell instances"
            raise TypeError(msg)


def link_above_below(a: _SkinCell, b: _SkinCell) -> None:
    """Link two internal cells in the above/below direction."""

    _validate_skin_cell(a, "a")
    _validate_skin_cell(b, "b")
    a.cb = b
    b.ca = a


def link_left_right(left: _SkinCell, right: _SkinCell) -> None:
    """Link two internal cells in the left/right direction."""

    _validate_skin_cell(left, "left")
    _validate_skin_cell(right, "right")
    left.cr = right
    right.cl = left


class ConnectedComponentSkinner:
    """Fallback skinner that groups thresholded voxels by connectivity.

    Connectivity modes map to voxel adjacency over rounded ``FaultCell``
    indices: ``"face"`` is 6-connected, ``"edge"`` is 18-connected, and
    ``"corner"`` is 26-connected. This is an explicit fallback API and does
    not implement the reference Java ``FaultSkinner`` seed-growth, linking,
    smoothing, or reskinning algorithm.
    """

    def __init__(
        self,
        min_likelihood: float = 0.0,
        min_skin_size: int | None = None,
        connectivity: str = "corner",
    ) -> None:
        self.min_likelihood = _validate_nonnegative_finite_float(
            min_likelihood,
            "min_likelihood",
        )
        self.min_skin_size = _validate_optional_nonnegative_int(min_skin_size, "min_skin_size")
        self.connectivity = _validate_connectivity(connectivity)

    def cells_from_votes(
        self,
        fv: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        min_likelihood: float | None = None,
    ) -> list[FaultCell]:
        """Extract cells where positive ``fv`` values satisfy ``min_likelihood``.

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
            else _validate_nonnegative_finite_float(min_likelihood, "min_likelihood")
        )

        cells: list[FaultCell] = []
        mask = (fv_array > np.float32(0.0)) & (fv_array >= np.float32(threshold))
        for i3, i2, i1 in np.argwhere(mask):
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

    def find_skins(
        self,
        fv: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        min_likelihood: float | None = None,
    ) -> list[FaultSkin]:
        """Group thresholded fault cells into connected-component skins.

        Input volumes use the project-wide ``(n3, n2, n1)`` shape convention.
        Returned skins are sorted by descending size and then by the
        lexicographic first ``(i1, i2, i3)`` cell index in each component.
        """

        cells = self.cells_from_votes(fv, vp, vt, min_likelihood=min_likelihood)
        cells_by_index = {cell.index: cell for cell in cells}
        unvisited = set(cells_by_index)
        offsets = _connectivity_offsets(self.connectivity)

        skins: list[FaultSkin] = []
        while unvisited:
            start = min(unvisited)
            component_indices = _collect_component_indices(start, unvisited, offsets)
            if self.min_skin_size is None or len(component_indices) >= self.min_skin_size:
                component_cells = [cells_by_index[index] for index in component_indices]
                skins.append(FaultSkin.from_cells(component_cells))

        skins.sort(key=lambda skin: (-len(skin), skin.cells[0].index))
        return skins


class FaultSkinner:
    """Default fault skinner facade.

    The current implementation delegates to ``ConnectedComponentSkinner`` as a
    fallback. It preserves the existing public behavior while keeping an API
    boundary for a future reference-like grower based on seed selection,
    neighbor links, surface smoothing, and reskinning.
    """

    def __init__(
        self,
        min_likelihood: float = 0.0,
        min_skin_size: int | None = None,
        connectivity: str = "corner",
    ) -> None:
        self._fallback = ConnectedComponentSkinner(
            min_likelihood=min_likelihood,
            min_skin_size=min_skin_size,
            connectivity=connectivity,
        )

    @property
    def min_likelihood(self) -> float:
        return self._fallback.min_likelihood

    @min_likelihood.setter
    def min_likelihood(self, value: float) -> None:
        self._fallback.min_likelihood = _validate_nonnegative_finite_float(
            value,
            "min_likelihood",
        )

    @property
    def min_skin_size(self) -> int | None:
        return self._fallback.min_skin_size

    @min_skin_size.setter
    def min_skin_size(self, value: int | None) -> None:
        self._fallback.min_skin_size = _validate_optional_nonnegative_int(
            value,
            "min_skin_size",
        )

    @property
    def connectivity(self) -> str:
        return self._fallback.connectivity

    @connectivity.setter
    def connectivity(self, value: str) -> None:
        self._fallback.connectivity = _validate_connectivity(value)

    def cells_from_votes(
        self,
        fv: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        min_likelihood: float | None = None,
    ) -> list[FaultCell]:
        """Extract fallback cells from thresholded voting outputs."""

        return self._fallback.cells_from_votes(fv, vp, vt, min_likelihood=min_likelihood)

    def find_skins(
        self,
        fv: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        min_likelihood: float | None = None,
    ) -> list[FaultSkin]:
        """Find skins with the current connected-component fallback."""

        return self._fallback.find_skins(fv, vp, vt, min_likelihood=min_likelihood)


def find_skins(
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    min_likelihood: float | None = None,
) -> list[FaultSkin]:
    """Group thresholded 3D voting outputs with default ``FaultSkinner`` settings."""

    return FaultSkinner().find_skins(fv, vp, vt, min_likelihood=min_likelihood)


def _collect_component_indices(
    start: tuple[int, int, int],
    unvisited: set[tuple[int, int, int]],
    offsets: tuple[tuple[int, int, int], ...],
) -> list[tuple[int, int, int]]:
    queue: deque[tuple[int, int, int]] = deque([start])
    unvisited.remove(start)
    component: list[tuple[int, int, int]] = []

    while queue:
        index = queue.popleft()
        component.append(index)
        i1, i2, i3 = index
        for d1, d2, d3 in offsets:
            neighbor = (i1 + d1, i2 + d2, i3 + d3)
            if neighbor in unvisited:
                unvisited.remove(neighbor)
                queue.append(neighbor)

    component.sort()
    return component


def _connectivity_offsets(connectivity: str) -> tuple[tuple[int, int, int], ...]:
    max_axis_steps = {
        "face": 1,
        "edge": 2,
        "corner": 3,
    }[connectivity]

    offsets: list[tuple[int, int, int]] = []
    for d1 in (-1, 0, 1):
        for d2 in (-1, 0, 1):
            for d3 in (-1, 0, 1):
                if d1 == 0 and d2 == 0 and d3 == 0:
                    continue
                if abs(d1) + abs(d2) + abs(d3) <= max_axis_steps:
                    offsets.append((d1, d2, d3))

    return tuple(offsets)


def _validate_nonnegative_finite_float(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite nonnegative number")

    value_float = float(value)
    if not math.isfinite(value_float) or value_float < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")

    return value_float


def _validate_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer")

    try:
        value_int = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc

    if value_int < 0:
        raise ValueError(f"{name} must be a nonnegative integer")

    return value_int


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


def _validate_skin_cell(cell: _SkinCell, name: str) -> None:
    if not isinstance(cell, _SkinCell):
        raise TypeError(f"{name} must be a _SkinCell")


def _index_key(i1: int, i2: int, i3: int) -> tuple[int, int, int]:
    try:
        return (operator.index(i1), operator.index(i2), operator.index(i3))
    except TypeError as exc:
        raise ValueError("indices must be integers") from exc


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
