import numpy as np
import pytest

import pyosv._skinner.growth as growth_module
import pyosv._skinner.reference as reference_module
import pyosv._skinner.seeds as seeds_module
from pyosv._skinner.growth import _grow_reference_skin, _grow_reference_skin_validated
from pyosv._skinner.seeds import (
    _find_reference_seeds,
    _find_reference_seeds_validated,
)
from pyosv.cells import FaultCell
from pyosv.skinner import (
    _SkinCell,
    _SkinCellGrid,
    _SkinOccupancyMask,
    link_above_below,
    link_left_right,
)


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


def test_validated_seed_helper_matches_validating_wrapper() -> None:
    ep = np.ones((3, 3, 4), dtype=np.float32)
    ft = np.zeros_like(ep)
    pt = np.full_like(ep, 25.0)
    tt = np.full_like(ep, 70.0)
    ft[1, 1, 1] = 0.9
    ft[1, 1, 3] = 0.8

    wrapped = _find_reference_seeds(0, 0.5, ep, ft, pt, tt)
    validated = _find_reference_seeds_validated(0, 0.5, ep, ft, pt, tt)

    assert [(cell.index, cell.fl, cell.fp, cell.ft) for cell in validated] == [
        (cell.index, cell.fl, cell.fp, cell.ft) for cell in wrapped
    ]


def test_validated_growth_helper_matches_validating_wrapper() -> None:
    fv = np.zeros((9, 9, 9), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)
    fv[3:6, 4, 3:6] = 0.9
    seed = _SkinCell(4.0, 4.0, 4.0, 0.9, 0.0, 90.0)

    wrapped = _grow_reference_skin(
        seed,
        fv,
        vp,
        vt,
        fmin=0.5,
        ru=3,
        rv=4,
        rw=4,
        max_steps=2,
        reskin=False,
    )
    validated = _grow_reference_skin_validated(
        seed,
        fv,
        vp,
        vt,
        fmin=0.5,
        ru=3,
        rv=4,
        rw=4,
        max_steps=2,
        du=5.0,
        max_delta_strike=30.0,
        reskin=False,
    )

    assert validated.cells == wrapped.cells


@pytest.mark.parametrize("name", ["fv", "vp", "vt"])
def test_growth_wrapper_rejects_non_finite_volumes(name: str) -> None:
    volumes = {
        "fv": np.zeros((5, 5, 5), dtype=np.float32),
        "vp": np.zeros((5, 5, 5), dtype=np.float32),
        "vt": np.zeros((5, 5, 5), dtype=np.float32),
    }
    volumes[name][2, 2, 2] = np.nan
    seed = _SkinCell(2.0, 2.0, 2.0, 0.9, 0.0, 90.0)

    with pytest.raises(ValueError, match=f"{name} must contain only finite values"):
        _grow_reference_skin(
            seed,
            volumes["fv"],
            volumes["vp"],
            volumes["vt"],
            fmin=0.5,
            ru=2,
            rv=2,
            rw=2,
            reskin=False,
        )


def test_growth_wrapper_rejects_mismatched_volume_shapes() -> None:
    fv = np.zeros((5, 5, 5), dtype=np.float32)
    seed = _SkinCell(2.0, 2.0, 2.0, 0.9, 0.0, 90.0)

    with pytest.raises(ValueError, match="fv and vp shapes must match"):
        _grow_reference_skin(
            seed,
            fv,
            np.zeros((5, 5, 4), dtype=np.float32),
            np.zeros_like(fv),
            fmin=0.5,
            ru=2,
            rv=2,
            rw=2,
            reskin=False,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"fmin": np.nan}, "fmin"),
        ({"ru": 1}, "ru must be at least 2"),
        ({"rv": 1}, "rv must be at least 2"),
        ({"rw": 1}, "rw must be at least 2"),
        ({"max_steps": -1}, "max_steps"),
        ({"du": np.inf}, "du"),
        ({"max_delta_strike": np.nan}, "max_delta_strike"),
        ({"reskin": 1}, "reskin"),
    ],
)
def test_growth_wrapper_rejects_invalid_scalars(
    overrides: dict[str, object],
    match: str,
) -> None:
    volume = np.zeros((5, 5, 5), dtype=np.float32)
    seed = _SkinCell(2.0, 2.0, 2.0, 0.9, 0.0, 90.0)
    options: dict[str, object] = {
        "fmin": 0.5,
        "ru": 2,
        "rv": 2,
        "rw": 2,
        "max_steps": 1,
        "du": 5.0,
        "max_delta_strike": 30.0,
        "reskin": False,
    }
    options.update(overrides)

    with pytest.raises(ValueError, match=match):
        _grow_reference_skin(seed, volume, volume, volume, **options)  # type: ignore[arg-type]


def test_reference_orchestration_scans_all_volumes_for_finiteness_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fv = np.zeros((21, 21, 21), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)
    ep = np.ones_like(fv)
    fv[5, 5, 5] = 0.9
    fv[15, 15, 15] = 0.8
    validation_calls: list[tuple[str, ...]] = []
    validate = reference_module._validate_matching_finite_arrays3_many

    def validation_spy(
        arrays: tuple[np.ndarray, ...],
        names: tuple[str, ...],
    ) -> tuple[np.ndarray, ...]:
        validation_calls.append(names)
        return validate(arrays, names)

    for module in (reference_module, seeds_module, growth_module):
        monkeypatch.setattr(
            module,
            "_validate_matching_finite_arrays3_many",
            validation_spy,
        )
    diagnostics: dict[str, object] = {}

    reference_module._find_reference_skins(
        fv=fv,
        vp=vp,
        vt=vt,
        ep=ep,
        ft=fv,
        pt=vp,
        tt=vt,
        d=0,
        fm=0.5,
        min_skin_size=1,
        ru=3,
        rv=3,
        rw=3,
        max_steps=0,
        du=5.0,
        max_delta_strike=30.0,
        reskin=False,
        accepted_occupancy_radius=0,
        diagnostics=diagnostics,
    )

    assert diagnostics["grow_attempt_count"] == 2
    assert validation_calls == [("fv", "vp", "vt", "ep", "ft", "pt", "tt")]


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


def test_skin_occupancy_mask_has_dense_c_contiguous_bool_storage() -> None:
    occupied = _SkinOccupancyMask((2, 3, 4))

    assert occupied._mask.shape == (2, 3, 4)
    assert occupied._mask.dtype == np.bool_
    assert occupied._mask.flags.c_contiguous
    assert occupied._mask.nbytes == 2 * 3 * 4


@pytest.mark.parametrize(
    "shape",
    [(), (1, 2), (1, 2, 3, 4), (0, 2, 3), (1, -1, 3), (1, True, 3), (1, 2.5, 3)],
)
def test_skin_occupancy_mask_rejects_invalid_shape(shape: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        _SkinOccupancyMask(shape)  # type: ignore[arg-type]


def test_skin_occupancy_mask_matches_sparse_grid_truthiness() -> None:
    shape = (2, 3, 4)
    centers = [
        (i1, i2, i3) for i3 in range(shape[0]) for i2 in range(shape[1]) for i1 in range(shape[2])
    ]
    query_centers = centers + [
        (-1, 0, 0),
        (shape[2], 0, 0),
        (0, -1, 0),
        (0, shape[1], 0),
        (0, 0, -1),
        (0, 0, shape[0]),
        (-1, -1, 0),
        (shape[2], shape[1], 0),
        (-1, -1, -1),
        (shape[2], shape[1], shape[0]),
        (-max(shape) - 1, 0, 0),
        (shape[2] + max(shape), 0, 0),
    ]
    radii = (0, 1, max(shape) + 1)

    for marked_i1, marked_i2, marked_i3 in centers:
        cell = _SkinCell(marked_i1, marked_i2, marked_i3, 0.8, 30.0, 60.0)
        for marked_radius in radii:
            grid = _SkinCellGrid()
            grid.set_cells_in_box(cell, marked_radius, marked_radius, marked_radius)
            occupied = _SkinOccupancyMask(shape)
            occupied.mark_box(
                marked_i1,
                marked_i2,
                marked_i3,
                marked_radius,
                marked_radius,
                marked_radius,
            )

            for query_i1, query_i2, query_i3 in query_centers:
                for query_radius in radii:
                    expected = bool(
                        grid.find_cells_in_box(
                            query_i1,
                            query_i2,
                            query_i3,
                            query_radius,
                            query_radius,
                            query_radius,
                        )
                    )
                    assert (
                        occupied.any_in_box(
                            query_i1,
                            query_i2,
                            query_i3,
                            query_radius,
                            query_radius,
                            query_radius,
                        )
                        is expected
                    )


def test_skin_occupancy_mask_handles_marks_and_queries_at_volume_boundaries() -> None:
    occupied = _SkinOccupancyMask((3, 4, 5))

    occupied.mark_box(-1, 1, 1, 1, 0, 0)
    occupied.mark_box(5, 2, 1, 1, 0, 0)
    occupied.mark_box(2, -5, 1, 0, 1, 0)

    assert occupied.any_in_box(0, 1, 1, 0, 0, 0)
    assert occupied.any_in_box(4, 2, 1, 0, 0, 0)
    assert occupied.any_in_box(-1, 1, 1, 1, 0, 0)
    assert occupied.any_in_box(-1, 1, 1, 0, 0, 0)
    assert occupied.any_in_box(5, 2, 1, 0, 0, 0)
    assert occupied.any_in_box(2, -5, 1, 0, 1, 0)
    assert not occupied.any_in_box(100, 100, 100, 0, 0, 0)


def test_skin_occupancy_mask_unions_multiple_box_marks() -> None:
    occupied = _SkinOccupancyMask((3, 4, 5))

    occupied.mark_box(0, 0, 0, 1, 0, 0)
    occupied.mark_box(4, 3, 2, 0, 1, 0)

    assert occupied.any_in_box(1, 0, 0, 0, 0, 0)
    assert occupied.any_in_box(4, 2, 2, 0, 0, 0)
    assert occupied.any_in_box(4, 3, 2, 0, 0, 0)
    assert not occupied.any_in_box(2, 2, 1, 0, 0, 0)
