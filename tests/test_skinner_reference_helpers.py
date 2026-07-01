import numpy as np
import pytest

from pyosv.cells import FaultCell
from pyosv.skinner import _SkinCell, _SkinCellGrid, link_above_below, link_left_right


def test_skin_cell_matches_public_fault_cell_rounding_and_vectors() -> None:
    skin_cell = _SkinCell(1.49, 2.50, -3.49, np.float32(0.8), 30.0, 60.0)
    fault_cell = FaultCell(1.49, 2.50, -3.49, np.float32(0.8), 30.0, 60.0)

    assert skin_cell.index == fault_cell.index == (1, 3, -3)
    assert skin_cell.i1 == fault_cell.i1
    assert skin_cell.i2 == fault_cell.i2
    assert skin_cell.i3 == fault_cell.i3
    np.testing.assert_allclose(skin_cell.fault_normal(), fault_cell.fault_normal())
    np.testing.assert_allclose(skin_cell.fault_dip_vector(), fault_cell.fault_dip_vector())
    np.testing.assert_allclose(skin_cell.fault_strike_vector(), fault_cell.fault_strike_vector())


def test_skin_cell_to_fault_cell_returns_public_immutable_cell() -> None:
    skin_cell = _SkinCell(1.2, 2.5, 3.7, 0.8, 30.0, 60.0)

    fault_cell = skin_cell.to_fault_cell()

    assert isinstance(fault_cell, FaultCell)
    assert fault_cell == FaultCell(1.2, 2.5, 3.7, 0.8, 30.0, 60.0)


def test_link_helpers_set_bidirectional_links() -> None:
    above = _SkinCell(1.0, 2.0, 3.0, 0.8, 30.0, 60.0)
    below = _SkinCell(1.0, 2.0, 4.0, 0.7, 30.0, 60.0)
    left = _SkinCell(0.0, 2.0, 3.0, 0.6, 30.0, 60.0)
    right = _SkinCell(2.0, 2.0, 3.0, 0.5, 30.0, 60.0)

    link_above_below(above, below)
    link_left_right(left, right)

    assert above.cb is below
    assert below.ca is above
    assert left.cr is right
    assert right.cl is left
    assert above.ca is None
    assert below.cb is None
    assert left.cl is None
    assert right.cr is None


def test_skin_cell_grid_set_and_get_use_rounded_indices() -> None:
    grid = _SkinCellGrid()
    cell = _SkinCell(1.2, 2.5, 3.7, 0.8, 30.0, 60.0)

    grid.set(cell)

    assert grid.get(1, 3, 4) is cell
    assert grid.get(1, 2, 4) is None


def test_skin_cell_grid_set_cells_in_box_marks_each_box_sample() -> None:
    grid = _SkinCellGrid()
    cell = _SkinCell(10.0, 20.0, 30.0, 0.8, 30.0, 60.0)

    grid.set_cells_in_box(cell, r1=1, r2=2, r3=0)

    assert grid.get(9, 18, 30) is cell
    assert grid.get(10, 20, 30) is cell
    assert grid.get(11, 22, 30) is cell
    assert grid.get(8, 20, 30) is None
    assert grid.get(10, 17, 30) is None
    assert grid.get(10, 20, 31) is None


def test_skin_cell_grid_find_cells_in_box_returns_unique_cells_sorted_by_index() -> None:
    grid = _SkinCellGrid()
    first = _SkinCell(5.0, 5.0, 5.0, 0.8, 30.0, 60.0)
    second = _SkinCell(4.0, 5.0, 5.0, 0.7, 30.0, 60.0)
    outside = _SkinCell(9.0, 5.0, 5.0, 0.6, 30.0, 60.0)
    grid.set_cells_in_box(first, r1=1, r2=0, r3=0)
    grid.set(second)
    grid.set(outside)

    found = grid.find_cells_in_box(5, 5, 5, r1=1, r2=0, r3=0)

    assert found == [second, first]


@pytest.mark.parametrize("radius_name", ["r1", "r2", "r3"])
def test_skin_cell_grid_rejects_invalid_box_radius(radius_name: str) -> None:
    grid = _SkinCellGrid()
    cell = _SkinCell(1.0, 2.0, 3.0, 0.8, 30.0, 60.0)
    radii = {"r1": 0, "r2": 0, "r3": 0}
    radii[radius_name] = -1

    with pytest.raises(ValueError, match=radius_name):
        grid.set_cells_in_box(cell, **radii)
