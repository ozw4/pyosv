"""Fault-cell extraction from 3D voting outputs."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
import heapq
import math
import numbers
import operator

import numpy as np

from pyosv.cells import FaultCell, _java_round
from pyosv.filters import smooth2d
from pyosv.geometry import strike_and_dip_from_local_surface_derivatives
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

    ``method="reference"`` uses reference-like seed selection and local
    geometry-aware growth. ``method="connected_component"`` explicitly selects
    the legacy connected-component fallback.
    """

    def __init__(
        self,
        min_likelihood: float = 0.0,
        min_skin_size: int | None = None,
        connectivity: str = "corner",
        method: str = "reference",
    ) -> None:
        self._fallback = ConnectedComponentSkinner(
            min_likelihood=min_likelihood,
            min_skin_size=min_skin_size,
            connectivity=connectivity,
        )
        self._method = _validate_skinner_method(method)

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, value: str) -> None:
        self._method = _validate_skinner_method(value)

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
        *,
        ep: np.ndarray | None = None,
        ft: np.ndarray | None = None,
        pt: np.ndarray | None = None,
        tt: np.ndarray | None = None,
        d: int = 1,
        ru: int = 150,
        rv: int | None = None,
        rw: int | None = None,
        max_steps: int = 10,
        du: float = 5.0,
        max_delta_strike: float = 30.0,
        reskin: bool = True,
    ) -> list[FaultSkin]:
        """Find skins with the configured backend."""

        should_reskin = _validate_bool(reskin, "reskin")
        if self.method == "connected_component":
            return self._fallback.find_skins(fv, vp, vt, min_likelihood=min_likelihood)

        threshold = (
            self.min_likelihood
            if min_likelihood is None
            else _validate_nonnegative_finite_float(min_likelihood, "min_likelihood")
        )
        return _find_reference_skins(
            fv=fv,
            vp=vp,
            vt=vt,
            ep=fv if ep is None else ep,
            ft=fv if ft is None else ft,
            pt=vp if pt is None else pt,
            tt=vt if tt is None else tt,
            d=d,
            fm=threshold,
            min_skin_size=self.min_skin_size,
            ru=ru,
            rv=rv,
            rw=rw,
            max_steps=max_steps,
            du=du,
            max_delta_strike=max_delta_strike,
            reskin=should_reskin,
        )

    def find_skin(
        self,
        seed: FaultCell,
        fv: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        *,
        min_likelihood: float | None = None,
        ru: int = 150,
        rv: int | None = None,
        rw: int | None = None,
        max_steps: int = 10,
        du: float = 5.0,
        max_delta_strike: float = 30.0,
        reskin: bool = True,
    ) -> FaultSkin:
        """Grow one reference-like skin from ``seed`` without changing defaults."""

        threshold = self.min_likelihood if min_likelihood is None else min_likelihood
        return _grow_reference_skin(
            seed,
            fv,
            vp,
            vt,
            fmin=threshold,
            ru=ru,
            rv=rv,
            rw=rw,
            max_steps=max_steps,
            du=du,
            max_delta_strike=max_delta_strike,
            reskin=reskin,
        )


def find_skins(
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    min_likelihood: float | None = None,
) -> list[FaultSkin]:
    """Group thresholded 3D voting outputs with the compatibility fallback."""

    return FaultSkinner(method="connected_component").find_skins(
        fv,
        vp,
        vt,
        min_likelihood=min_likelihood,
    )


def _find_reference_skins(
    *,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    ep: np.ndarray,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
    d: int,
    fm: float,
    min_skin_size: int | None,
    ru: int,
    rv: int | None,
    rw: int | None,
    max_steps: int,
    du: float,
    max_delta_strike: float,
    reskin: bool,
) -> list[FaultSkin]:
    should_reskin = _validate_bool(reskin, "reskin")
    threshold = _validate_nonnegative_finite_float(fm, "fm")
    fv_array, vp_array, vt_array = _validate_matching_finite_arrays3_many(
        (fv, vp, vt),
        ("fv", "vp", "vt"),
    )
    seeds = _find_reference_seeds(d=d, fm=threshold, ep=ep, ft=ft, pt=pt, tt=tt)
    skin_size = _validate_optional_nonnegative_int(min_skin_size, "min_skin_size")
    occupied = _SkinCellGrid()
    skins: list[FaultSkin] = []

    for seed in seeds:
        if occupied.find_cells_in_box(seed.i1, seed.i2, seed.i3, 2, 2, 2):
            continue

        skin = _grow_reference_skin(
            seed,
            fv_array,
            vp_array,
            vt_array,
            fmin=threshold,
            ru=ru,
            rv=rv,
            rw=rw,
            max_steps=max_steps,
            du=du,
            max_delta_strike=max_delta_strike,
            collision_grid=occupied,
            reskin=should_reskin,
        )
        if skin_size is not None and len(skin) < skin_size:
            continue

        skins.append(skin)
        _mark_occupied_skin(occupied, skin)

    return skins


def _mark_occupied_skin(occupied: _SkinCellGrid, skin: FaultSkin) -> None:
    for cell in skin:
        occupied.set(_SkinCell(cell.x1, cell.x2, cell.x3, cell.fl, cell.fp, cell.ft))


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


def _grow_reference_skin(
    seed: FaultCell | _SkinCell,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    fmin: float,
    ru: int = 150,
    rv: int | None = None,
    rw: int | None = None,
    max_steps: int = 10,
    du: float = 5.0,
    max_delta_strike: float = 30.0,
    collision_grid: _SkinCellGrid | None = None,
    reskin: bool = True,
) -> FaultSkin:
    """Grow one skin in a seed-local fault-coordinate grid."""

    should_reskin = _validate_bool(reskin, "reskin")
    seed_cell = _validate_seed_cell(seed)
    threshold = _validate_nonnegative_finite_float(fmin, "fmin")
    radius_u = _validate_nonnegative_int(ru, "ru")
    step_limit = _validate_nonnegative_int(max_steps, "max_steps")
    max_delta_u = _validate_nonnegative_finite_float(du, "du")
    max_delta_fp = _validate_nonnegative_finite_float(max_delta_strike, "max_delta_strike")
    fv_array, vp_array, vt_array = _validate_matching_finite_arrays3_many(
        (fv, vp, vt),
        ("fv", "vp", "vt"),
    )
    n3, n2, _ = fv_array.shape
    radius_v = max(n2, n3) if rv is None else _validate_nonnegative_int(rv, "rv")
    radius_w = max(n2, n3) if rw is None else _validate_nonnegative_int(rw, "rw")
    if radius_u < 2:
        raise ValueError("ru must be at least 2")
    if radius_v < 2:
        raise ValueError("rv must be at least 2")
    if radius_w < 2:
        raise ValueError("rw must be at least 2")

    origin = (seed_cell.x1, seed_cell.x2, seed_cell.x3)
    transform_map = _update_transform_map(
        radius_u,
        radius_v,
        radius_w,
        seed_cell.fault_normal(),
        seed_cell.fault_dip_vector(),
        seed_cell.fault_strike_vector(),
    )
    local_seed = _SkinCell(
        radius_u,
        radius_v,
        radius_w,
        seed_cell.fl,
        seed_cell.fp,
        seed_cell.ft,
    )
    local_cells: dict[tuple[int, int], _SkinCell] = {(radius_v, radius_w): local_seed}
    accepted: list[_SkinCell] = []
    accepted_world_indices: set[tuple[int, int, int]] = set()
    queue: list[tuple[float, int, _SkinCell]] = []
    sequence = 0
    heapq.heappush(queue, (-local_seed.fl, sequence, local_seed))

    while queue:
        _, _, cell = heapq.heappop(queue)
        if cell.skin_id is not None:
            continue

        world = _local_cell_to_world(cell, origin, transform_map)
        if not _is_world_interior(world, fv_array.shape):
            continue
        world_index = _world_index(world)
        if world_index in accepted_world_indices:
            continue
        if collision_grid is not None and collision_grid.find_cells_in_box(
            world_index[0],
            world_index[1],
            world_index[2],
            2,
            2,
            2,
        ):
            continue

        cell.skin_id = 0
        accepted.append(cell)
        accepted_world_indices.add(world_index)

        if not _is_local_cell_expandable(cell, transform_map):
            continue

        for axis, direction in (("v", -1), ("v", 1), ("w", -1), ("w", 1)):
            sequence = _grow_reference_direction(
                cell=cell,
                axis=axis,
                direction=direction,
                local_cells=local_cells,
                accepted_world_indices=accepted_world_indices,
                queue=queue,
                sequence=sequence,
                fv=fv_array,
                vp=vp_array,
                vt=vt_array,
                transform_map=transform_map,
                origin=origin,
                fmin=threshold,
                du=max_delta_u,
                max_delta_strike=max_delta_fp,
                max_steps=step_limit,
                collision_grid=collision_grid,
            )

    grown_skin = FaultSkin.from_cells(
        _local_cell_to_fault_cell(cell, origin, transform_map, fv_array, vp_array, vt_array)
        for cell in accepted
    )
    if not should_reskin:
        return grown_skin

    return _reskin_reference(grown_skin)


def _reskin_reference(skin: FaultSkin, *, smoothing_sigma: float = 1.0) -> FaultSkin:
    """Smooth and reorient a grown reference-like skin.

    This is an approximation of the reference weighted smoothing phase: cells
    are projected to a seed-local ``(v, w)`` grid, local ``u`` offsets are
    smoothed with likelihood weights, strike/dip are recomputed from surface
    derivatives, and neighbor links are rebuilt on the local grid.
    """

    if not isinstance(skin, FaultSkin):
        raise TypeError("skin must be a FaultSkin")

    sigma = _validate_nonnegative_finite_float(smoothing_sigma, "smoothing_sigma")
    cells = list(skin)
    if len(cells) <= 1:
        return FaultSkin.from_cells(cells)

    seed = _highest_likelihood_cell(cells)
    origin = np.array([seed.x1, seed.x2, seed.x3], dtype=np.float32)
    normal = seed.fault_normal()
    dip = seed.fault_dip_vector()
    strike = seed.fault_strike_vector()
    entries = _project_cells_to_local_surface(cells, origin, normal, dip, strike)
    if not entries:
        return FaultSkin()

    v_min = min(entry[0] for entry in entries)
    v_max = max(entry[0] for entry in entries)
    w_min = min(entry[1] for entry in entries)
    w_max = max(entry[1] for entry in entries)
    nv = v_max - v_min + 1
    nw = w_max - w_min + 1
    surface = np.zeros((nw, nv), dtype=np.float32)
    weights = np.zeros((nw, nv), dtype=np.float32)
    cells_by_key: dict[tuple[int, int], FaultCell] = {}
    order_by_key: dict[tuple[int, int], int] = {}

    for order, (iv, iw, iu, cell) in enumerate(entries):
        row = iw - w_min
        col = iv - v_min
        key = (iv, iw)
        if key in cells_by_key and cell.fl <= cells_by_key[key].fl:
            continue
        weight = np.float32(max(float(cell.fl), 0.0))
        surface[row, col] = np.float32(iu)
        weights[row, col] = weight if weight > 0.0 else np.float32(1.0)
        cells_by_key[key] = cell
        order_by_key.setdefault(key, order)

    smoothed_surface = _smooth_weighted_surface(surface, weights, sigma)
    local_cells: dict[tuple[int, int], _SkinCell] = {}
    public_cells: dict[tuple[int, int], FaultCell] = {}
    for key, cell in cells_by_key.items():
        iv, iw = key
        row = iw - w_min
        col = iv - v_min
        iu = float(smoothed_surface[row, col])
        fp, ft = _local_surface_strike_and_dip(
            normal,
            dip,
            strike,
            smoothed_surface,
            row,
            col,
        )
        world = origin + iu * normal + np.float32(iv) * dip + np.float32(iw) * strike
        public_cells[key] = FaultCell(world[0], world[1], world[2], cell.fl, fp, ft)
        local_cells[key] = _SkinCell(iu, iv, iw, cell.fl, fp, ft)

    _link_local_surface_cells(local_cells)
    _link_public_surface_cells(public_cells)
    ordered_keys = sorted(public_cells, key=lambda key: order_by_key[key])
    return FaultSkin.from_cells(public_cells[key] for key in ordered_keys)


def _highest_likelihood_cell(cells: list[FaultCell]) -> FaultCell:
    best_index = max(range(len(cells)), key=lambda index: (cells[index].fl, -index))
    return cells[best_index]


def _project_cells_to_local_surface(
    cells: list[FaultCell],
    origin: np.ndarray,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
) -> list[tuple[int, int, float, FaultCell]]:
    entries: list[tuple[int, int, float, FaultCell]] = []
    for cell in cells:
        offset = np.array([cell.x1, cell.x2, cell.x3], dtype=np.float32) - origin
        iu = float(np.dot(offset, normal))
        iv = _java_round(float(np.dot(offset, dip)))
        iw = _java_round(float(np.dot(offset, strike)))
        entries.append((iv, iw, iu, cell))

    return entries


def _smooth_weighted_surface(
    surface: np.ndarray,
    weights: np.ndarray,
    sigma: float,
) -> np.ndarray:
    if sigma == 0.0 or surface.size <= 1:
        return surface.copy()

    numerator = smooth2d(surface * weights, sigma)
    denominator = smooth2d(weights, sigma)
    smoothed = surface.copy()
    np.divide(
        numerator,
        denominator,
        out=smoothed,
        where=denominator > np.float32(1.0e-6),
    )
    return smoothed.astype(np.float32, copy=False)


def _local_surface_strike_and_dip(
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    row: int,
    col: int,
) -> tuple[float, float]:
    du_dv = _surface_derivative(surface, row, col, axis=1)
    du_dw = _surface_derivative(surface, row, col, axis=0)
    return strike_and_dip_from_local_surface_derivatives(
        normal,
        dip,
        strike,
        du_dv,
        du_dw,
    )


def _surface_derivative(surface: np.ndarray, row: int, col: int, *, axis: int) -> float:
    if axis == 1:
        if surface.shape[1] == 1:
            return 0.0
        if 0 < col < surface.shape[1] - 1:
            return float(0.5 * (surface[row, col + 1] - surface[row, col - 1]))
        if col == 0:
            return float(surface[row, col + 1] - surface[row, col])
        return float(surface[row, col] - surface[row, col - 1])

    if axis == 0:
        if surface.shape[0] == 1:
            return 0.0
        if 0 < row < surface.shape[0] - 1:
            return float(0.5 * (surface[row + 1, col] - surface[row - 1, col]))
        if row == 0:
            return float(surface[row + 1, col] - surface[row, col])
        return float(surface[row, col] - surface[row - 1, col])

    raise ValueError("axis must be 0 or 1")


def _link_local_surface_cells(local_cells: dict[tuple[int, int], _SkinCell]) -> None:
    for (iv, iw), cell in local_cells.items():
        below = local_cells.get((iv + 1, iw))
        right = local_cells.get((iv, iw + 1))
        if below is not None:
            link_above_below(cell, below)
        if right is not None:
            link_left_right(cell, right)


def _link_public_surface_cells(public_cells: dict[tuple[int, int], FaultCell]) -> None:
    for (iv, iw), cell in public_cells.items():
        below = public_cells.get((iv + 1, iw))
        right = public_cells.get((iv, iw + 1))
        if below is not None:
            _link_fault_cells_above_below(cell, below)
        if right is not None:
            _link_fault_cells_left_right(cell, right)


def _link_fault_cells_above_below(a: FaultCell, b: FaultCell) -> None:
    object.__setattr__(a, "cb", b)
    object.__setattr__(b, "ca", a)


def _link_fault_cells_left_right(left: FaultCell, right: FaultCell) -> None:
    object.__setattr__(left, "cr", right)
    object.__setattr__(right, "cl", left)


def _grow_reference_direction(
    *,
    cell: _SkinCell,
    axis: str,
    direction: int,
    local_cells: dict[tuple[int, int], _SkinCell],
    accepted_world_indices: set[tuple[int, int, int]],
    queue: list[tuple[float, int, _SkinCell]],
    sequence: int,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    transform_map: _LocalTransformMap,
    origin: tuple[float, float, float],
    fmin: float,
    du: float,
    max_delta_strike: float,
    max_steps: int,
    collision_grid: _SkinCellGrid | None,
) -> int:
    if not _link_slot_is_empty(cell, axis, direction):
        return sequence

    next_v = cell.i2 + direction if axis == "v" else cell.i2
    next_w = cell.i3 + direction if axis == "w" else cell.i3
    neighbor = local_cells.get((next_v, next_w))
    if neighbor is not None:
        if _candidate_matches_delta(cell, neighbor, origin, transform_map, du, max_delta_strike):
            _link_cells_for_direction(cell, neighbor, axis, direction)
        return sequence

    ub = max(cell.i1 - 5, 0)
    ue = min(cell.i1 + 5, transform_map.us.shape[1] - 1)
    if axis == "v":
        candidate_slice = _candidate_slice_above_below(
            fv,
            transform_map,
            origin,
            ub=ub,
            ue=ue,
            vc=cell.i2,
            wc=cell.i3,
            direction=direction,
            max_steps=max_steps,
        )
    else:
        candidate_slice = _candidate_slice_left_right(
            fv,
            transform_map,
            origin,
            ub=ub,
            ue=ue,
            vc=cell.i2,
            wc=cell.i3,
            direction=direction,
            max_steps=max_steps,
        )
    picked_us = _pick_candidate_us(ub, candidate_slice)

    previous = cell
    for row in range(1, len(picked_us)):
        iu = int(picked_us[row])
        likelihood = float(candidate_slice[row, iu - ub])
        if likelihood < fmin:
            break

        iv = cell.i2 + direction * row if axis == "v" else cell.i2
        iw = cell.i3 + direction * row if axis == "w" else cell.i3
        existing = local_cells.get((iv, iw))
        if existing is not None:
            if not _candidate_matches_delta(
                previous,
                existing,
                origin,
                transform_map,
                du,
                max_delta_strike,
            ):
                break
            _link_cells_for_direction(previous, existing, axis, direction)
            previous = existing
            continue

        candidate = _candidate_from_local_index(
            iu,
            iv,
            iw,
            likelihood,
            vp,
            vt,
            transform_map,
            origin,
        )
        world = _local_cell_to_world(candidate, origin, transform_map)
        if not _is_world_interior(world, fv.shape):
            break
        world_index = _world_index(world)
        if world_index in accepted_world_indices:
            break
        if collision_grid is not None and collision_grid.find_cells_in_box(
            world_index[0],
            world_index[1],
            world_index[2],
            2,
            2,
            2,
        ):
            break
        if not _candidate_matches_delta(
            previous,
            candidate,
            origin,
            transform_map,
            du,
            max_delta_strike,
        ):
            break

        local_cells[(iv, iw)] = candidate
        _link_cells_for_direction(previous, candidate, axis, direction)
        sequence += 1
        heapq.heappush(queue, (-candidate.fl, sequence, candidate))
        previous = candidate

    return sequence


def _candidate_from_local_index(
    iu: int,
    iv: int,
    iw: int,
    likelihood: float,
    vp: np.ndarray,
    vt: np.ndarray,
    transform_map: _LocalTransformMap,
    origin: tuple[float, float, float],
) -> _SkinCell:
    world = _local_index_to_world(iu, iv, iw, origin, transform_map)
    fp = _sample_validated_volume_nearest_java_round(vp, world[0], world[1], world[2])
    ft = _sample_validated_volume_nearest_java_round(vt, world[0], world[1], world[2])
    return _SkinCell(iu, iv, iw, likelihood, fp, ft)


def _candidate_matches_delta(
    previous: _SkinCell,
    candidate: _SkinCell,
    origin: tuple[float, float, float],
    transform_map: _LocalTransformMap,
    du: float,
    max_delta_strike: float,
) -> bool:
    if abs(candidate.x1 - previous.x1) > du:
        return False
    if _angle_delta_degrees(candidate.fp, previous.fp) > max_delta_strike:
        return False

    previous_world = _local_cell_to_world(previous, origin, transform_map)
    candidate_world = _local_cell_to_world(candidate, origin, transform_map)
    return abs(float(candidate_world[0]) - float(previous_world[0])) <= du


def _angle_delta_degrees(first: float, second: float) -> float:
    delta = abs(float(first) - float(second)) % 360.0
    return min(delta, 360.0 - delta)


def _link_slot_is_empty(cell: _SkinCell, axis: str, direction: int) -> bool:
    if axis == "v" and direction < 0:
        return cell.ca is None
    if axis == "v" and direction > 0:
        return cell.cb is None
    if axis == "w" and direction < 0:
        return cell.cl is None
    if axis == "w" and direction > 0:
        return cell.cr is None
    raise ValueError("axis must be 'v' or 'w' and direction must be -1 or 1")


def _link_cells_for_direction(
    cell: _SkinCell,
    neighbor: _SkinCell,
    axis: str,
    direction: int,
) -> None:
    if axis == "v" and direction < 0:
        link_above_below(neighbor, cell)
    elif axis == "v" and direction > 0:
        link_above_below(cell, neighbor)
    elif axis == "w" and direction < 0:
        link_left_right(neighbor, cell)
    elif axis == "w" and direction > 0:
        link_left_right(cell, neighbor)
    else:
        raise ValueError("axis must be 'v' or 'w' and direction must be -1 or 1")


def _local_cell_to_fault_cell(
    cell: _SkinCell,
    origin: tuple[float, float, float],
    transform_map: _LocalTransformMap,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
) -> FaultCell:
    x1, x2, x3 = _local_cell_to_world(cell, origin, transform_map)
    fl = _sample_validated_volume_nearest_java_round(fv, x1, x2, x3)
    fp = _sample_validated_volume_nearest_java_round(vp, x1, x2, x3)
    ft = _sample_validated_volume_nearest_java_round(vt, x1, x2, x3)
    return FaultCell(x1, x2, x3, fl, fp, ft)


def _local_cell_to_world(
    cell: _SkinCell,
    origin: tuple[float, float, float],
    transform_map: _LocalTransformMap,
) -> tuple[np.float32, np.float32, np.float32]:
    return _local_index_to_world(cell.i1, cell.i2, cell.i3, origin, transform_map)


def _is_local_cell_expandable(cell: _SkinCell, transform_map: _LocalTransformMap) -> bool:
    return (
        1 < cell.i1 < transform_map.us.shape[1] - 2
        and 1 < cell.i2 < transform_map.vs.shape[1] - 2
        and 1 < cell.i3 < transform_map.ws.shape[1] - 2
    )


def _is_world_interior(
    world: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> bool:
    n3, n2, n1 = shape
    x1, x2, x3 = (float(world[0]), float(world[1]), float(world[2]))
    return 1.0 < x1 < n1 - 2 and 1.0 < x2 < n2 - 2 and 1.0 < x3 < n3 - 2


def _world_index(world: tuple[float, float, float]) -> tuple[int, int, int]:
    return (_java_round(world[0]), _java_round(world[1]), _java_round(world[2]))


def _validate_seed_cell(seed: FaultCell | _SkinCell) -> _SkinCell:
    if isinstance(seed, _SkinCell):
        return seed
    if isinstance(seed, FaultCell):
        return _SkinCell(seed.x1, seed.x2, seed.x3, seed.fl, seed.fp, seed.ft)

    raise TypeError("seed must be a FaultCell")


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


def _validate_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")

    return value


def _index_key(i1: int, i2: int, i3: int) -> tuple[int, int, int]:
    try:
        return (operator.index(i1), operator.index(i2), operator.index(i3))
    except TypeError as exc:
        raise ValueError("indices must be integers") from exc


def _validate_connectivity(connectivity: str) -> str:
    if not isinstance(connectivity, str) or connectivity not in {"face", "edge", "corner"}:
        raise ValueError("connectivity must be 'face', 'edge', or 'corner'")

    return connectivity


def _validate_skinner_method(method: str) -> str:
    if not isinstance(method, str) or method not in {"reference", "connected_component"}:
        raise ValueError("method must be 'reference' or 'connected_component'")

    return method


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
