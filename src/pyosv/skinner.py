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


@dataclass(slots=True, frozen=True)
class _LocalTransformMap:
    """Local grow offsets where u=normal, v=dip, and w=strike."""

    us: np.ndarray
    vs: np.ndarray
    ws: np.ndarray


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

    def find_seeds(
        self,
        d: int,
        fm: float,
        ep: np.ndarray,
        ft: np.ndarray,
        pt: np.ndarray,
        tt: np.ndarray,
    ) -> list[FaultCell]:
        """Find reference-like seed cells from thinned likelihood volumes."""

        return [
            cell.to_fault_cell()
            for cell in _find_reference_seeds(
                d=d,
                fm=fm,
                ep=ep,
                ft=ft,
                pt=pt,
                tt=tt,
            )
        ]

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


def _find_reference_seeds(
    d: int,
    fm: float,
    ep: np.ndarray,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
) -> list[_SkinCell]:
    """Select reference-like starting cells for future skin growth."""

    distance = _validate_nonnegative_int(d, "d")
    threshold = _validate_nonnegative_finite_float(fm, "fm")
    ep_array, ft_array, pt_array, tt_array = _validate_matching_finite_arrays3_many(
        (ep, ft, pt, tt),
        ("ep", "ft", "pt", "tt"),
    )
    n3, n2, n1 = ft_array.shape

    candidates: list[tuple[float, int, int, int]] = []
    candidate_mask = (ep_array > np.float32(0.8)) & (ft_array > np.float32(threshold))
    for i3, i2, i1 in np.argwhere(candidate_mask):
        candidates.append(
            (
                float(ft_array[i3, i2, i1]),
                operator.index(i3),
                operator.index(i2),
                operator.index(i1),
            ),
        )

    candidates.sort(
        key=lambda candidate: (-candidate[0], candidate[1], candidate[2], candidate[3]),
    )

    mark = np.zeros((n3, n2, n1), dtype=np.bool_)
    seeds: list[_SkinCell] = []
    for _, i3, i2, i1 in candidates:
        b1 = max(i1 - distance, 0)
        b2 = max(i2 - distance, 0)
        b3 = max(i3 - distance, 0)
        e1 = min(i1 + distance, n1 - 1)
        e2 = min(i2 + distance, n2 - 1)
        e3 = min(i3 + distance, n3 - 1)
        if mark[b3 : e3 + 1, b2 : e2 + 1, b1 : e1 + 1].any():
            continue

        seeds.append(
            _SkinCell(
                i1,
                i2,
                i3,
                ft_array[i3, i2, i1],
                pt_array[i3, i2, i1],
                tt_array[i3, i2, i1],
            ),
        )
        mark[i3, i2, i1] = True

    return seeds


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
        + transform_map.us[0, u_index]
        + transform_map.vs[0, v_index]
        + transform_map.ws[0, w_index],
    )
    x2 = np.float32(
        o2
        + transform_map.us[1, u_index]
        + transform_map.vs[1, v_index]
        + transform_map.ws[1, w_index],
    )
    x3 = np.float32(
        o3
        + transform_map.us[2, u_index]
        + transform_map.vs[2, v_index]
        + transform_map.ws[2, w_index],
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


def _candidate_slice_above_below(
    fv: np.ndarray,
    transform_map: _LocalTransformMap,
    origin: tuple[float, float, float],
    ub: int,
    ue: int,
    vc: int,
    wc: int,
    direction: int,
    max_steps: int | None = None,
) -> np.ndarray:
    """Sample a candidate slice in the local v direction over the u range."""

    return _candidate_slice(
        fv=fv,
        transform_map=transform_map,
        origin=origin,
        ub=ub,
        ue=ue,
        vc=vc,
        wc=wc,
        direction=direction,
        axis="v",
        max_steps=max_steps,
    )


def _candidate_slice_left_right(
    fv: np.ndarray,
    transform_map: _LocalTransformMap,
    origin: tuple[float, float, float],
    ub: int,
    ue: int,
    vc: int,
    wc: int,
    direction: int,
    max_steps: int | None = None,
) -> np.ndarray:
    """Sample a candidate slice in the local w direction over the u range."""

    return _candidate_slice(
        fv=fv,
        transform_map=transform_map,
        origin=origin,
        ub=ub,
        ue=ue,
        vc=vc,
        wc=wc,
        direction=direction,
        axis="w",
        max_steps=max_steps,
    )


def _pick_candidate_us(ub: int, candidate_slice: np.ndarray) -> np.ndarray:
    """Pick the local u index with maximum likelihood for each slice row."""

    u_start = _validate_nonnegative_int(ub, "ub")
    slice_array = np.asarray(candidate_slice, dtype=np.float32)
    if slice_array.ndim != 2:
        raise ValueError("candidate_slice must be a 2D array")
    if slice_array.shape[1] == 0:
        raise ValueError("candidate_slice must contain at least one u sample")
    if not np.isfinite(slice_array).all():
        raise ValueError("candidate_slice must contain only finite values")

    return (u_start + np.argmax(slice_array, axis=1)).astype(np.int32, copy=False)


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


def _candidate_slice(
    fv: np.ndarray,
    transform_map: _LocalTransformMap,
    origin: tuple[float, float, float],
    ub: int,
    ue: int,
    vc: int,
    wc: int,
    direction: int,
    axis: str,
    max_steps: int | None,
) -> np.ndarray:
    fv_array = _validate_matching_finite_arrays3_many((fv,), ("fv",))[0]
    u_start = _validate_nonnegative_int(ub, "ub")
    u_stop = _validate_nonnegative_int(ue, "ue")
    if u_stop < u_start:
        raise ValueError("ue must be greater than or equal to ub")
    _validate_transform_index(u_start, transform_map.us, "ub")
    _validate_transform_index(u_stop, transform_map.us, "ue")
    v_center = _validate_transform_index(vc, transform_map.vs, "vc")
    w_center = _validate_transform_index(wc, transform_map.ws, "wc")
    step_sign = _validate_direction(direction)
    step_limit = None if max_steps is None else _validate_nonnegative_int(max_steps, "max_steps")

    if axis == "v":
        distance_to_edge = v_center if step_sign < 0 else transform_map.vs.shape[1] - 1 - v_center
    elif axis == "w":
        distance_to_edge = w_center if step_sign < 0 else transform_map.ws.shape[1] - 1 - w_center
    else:
        raise ValueError("axis must be 'v' or 'w'")

    row_count = distance_to_edge + 1
    if step_limit is not None:
        row_count = min(row_count, step_limit + 1)

    samples = np.zeros((row_count, u_stop - u_start + 1), dtype=np.float32)
    for row in range(row_count):
        iv = v_center + step_sign * row if axis == "v" else v_center
        iw = w_center + step_sign * row if axis == "w" else w_center
        for col, iu in enumerate(range(u_start, u_stop + 1)):
            x1, x2, x3 = _local_index_to_world(iu, iv, iw, origin, transform_map)
            samples[row, col] = _sample_validated_volume_nearest_java_round(
                fv_array,
                x1,
                x2,
                x3,
            )

    return samples


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


def _validate_direction(direction: int) -> int:
    try:
        direction_int = operator.index(direction)
    except TypeError as exc:
        raise ValueError("direction must be -1 or 1") from exc
    if direction_int not in {-1, 1}:
        raise ValueError("direction must be -1 or 1")

    return direction_int


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
