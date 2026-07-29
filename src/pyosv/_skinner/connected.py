"""Connected-component fault skinning."""

from collections import deque
import operator

import numpy as np

from pyosv._skinner.validation import (
    _validate_connectivity,
    _validate_matching_finite_arrays3_many,
    _validate_nonnegative_finite_float,
    _validate_optional_nonnegative_int,
)
from pyosv.cells import FAULT_CELL_GENERATION_CONNECTED_COMPONENT, FaultCell
from pyosv.skin import FaultSkin


class ConnectedComponentSkinner:
    """Fallback skinner that groups thresholded voxels by connectivity."""

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
        """Extract cells where positive ``fv`` values satisfy ``min_likelihood``."""

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
                    generation=FAULT_CELL_GENERATION_CONNECTED_COMPONENT,
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
        """Group thresholded fault cells into connected-component skins."""

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


def find_connected_component_skins(
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    min_likelihood: float | None = None,
    *,
    min_skin_size: int | None = None,
    connectivity: str = "corner",
) -> list[FaultSkin]:
    """Group thresholded 3D voting outputs with the explicit fallback."""

    return ConnectedComponentSkinner(
        min_skin_size=min_skin_size,
        connectivity=connectivity,
    ).find_skins(fv, vp, vt, min_likelihood=min_likelihood)


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
    max_axis_steps = {"face": 1, "edge": 2, "corner": 3}[connectivity]

    offsets: list[tuple[int, int, int]] = []
    for d1 in (-1, 0, 1):
        for d2 in (-1, 0, 1):
            for d3 in (-1, 0, 1):
                if d1 == 0 and d2 == 0 and d3 == 0:
                    continue
                if abs(d1) + abs(d2) + abs(d3) <= max_axis_steps:
                    offsets.append((d1, d2, d3))

    return tuple(offsets)
