"""Reference-skin surface projection, smoothing, orientation, and linking."""

from __future__ import annotations

import numpy as np

from pyosv._skinner.models import (
    _SkinCell,
    link_above_below,
    link_left_right,
)
from pyosv._skinner.validation import _validate_nonnegative_finite_float
from pyosv.cells import FaultCell, _java_round
from pyosv.filters import smooth2d
from pyosv.geometry import strike_and_dip_from_local_surface_derivatives
from pyosv.skin import FaultSkin


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
