from dataclasses import replace

import numpy as np

from pyosv._skinner.models import _ReskinContext, _SkinCell
from pyosv.cells import FaultCell
from pyosv.skin import FaultSkin
from pyosv.skinner import (
    FaultSkinner,
    RESKIN_POLICY_EXISTING_CELLS_V1,
    RESKIN_POLICY_REFERENCE_DENSE_V1,
    _SkinOccupancyMask,
    _reskin_reference_dense_v1,
    _update_transform_map,
)


def _dense_fixture(
    *,
    valid_mask: np.ndarray | None = None,
) -> tuple[FaultSkin, _ReskinContext, np.ndarray, np.ndarray | None]:
    seed = FaultCell(6.0, 6.0, 6.0, 0.9, 0.0, 90.0)
    transform = _update_transform_map(
        3,
        4,
        4,
        seed.fault_normal(),
        seed.fault_dip_vector(),
        seed.fault_strike_vector(),
    )
    accepted = tuple(
        _SkinCell(3, iv, iw, 0.9, 0.0, 90.0)
        for iw in range(4, 7)
        for iv in range(4, 7)
        if (iv, iw) != (5, 5)
    )
    fv = np.full((13, 13, 13), 0.1, dtype=np.float32)
    skin = FaultSkin.from_cells(
        FaultCell(float(index), 0.0, 0.0, cell.fl, cell.fp, cell.ft)
        for index, cell in enumerate(accepted)
    )
    context = _ReskinContext(
        seed=_SkinCell(seed.x1, seed.x2, seed.x3, seed.fl, seed.fp, seed.ft),
        origin=(seed.x1, seed.x2, seed.x3),
        transform_map=transform,
        accepted_cells=accepted,
        fv=fv,
        volume_shape=fv.shape,
        collision_grid=None,
        du=5.0,
        valid_mask=valid_mask,
    )
    return skin, context, fv, valid_mask


def _signature(skin: FaultSkin) -> list[tuple[object, ...]]:
    def linked_index(cell: FaultCell | None) -> tuple[int, int, int] | None:
        return None if cell is None else cell.index

    return [
        (
            cell.x1,
            cell.x2,
            cell.x3,
            cell.fl,
            cell.fp,
            cell.ft,
            linked_index(cell.ca),
            linked_index(cell.cb),
            linked_index(cell.cl),
            linked_index(cell.cr),
        )
        for cell in skin
    ]


def test_reference_dense_v1_fills_missing_local_cell_and_rebuilds_links() -> None:
    skin, context, fv, _ = _dense_fixture()
    fv_before = fv.copy()
    accepted_before = tuple(
        (cell.x1, cell.x2, cell.x3, cell.fl, cell.ca, cell.cb, cell.cl, cell.cr)
        for cell in context.accepted_cells
    )

    dense = _reskin_reference_dense_v1(skin, context=context)

    assert len(skin) == 8
    assert len(dense) == 9
    by_index = {cell.index: cell for cell in dense}
    center = by_index[(7, 6, 7)]
    assert center.ca is not None and center.ca.cb is center
    assert center.cb is not None and center.cb.ca is center
    assert center.cl is not None and center.cl.cr is center
    assert center.cr is not None and center.cr.cl is center
    assert all(cell.fl == np.float32(0.1) for cell in dense)
    assert all(np.isfinite((cell.x1, cell.x2, cell.x3, cell.fp, cell.ft)).all() for cell in dense)
    np.testing.assert_array_equal(fv, fv_before)
    assert accepted_before == tuple(
        (cell.x1, cell.x2, cell.x3, cell.fl, cell.ca, cell.cb, cell.cl, cell.cr)
        for cell in context.accepted_cells
    )


def test_reference_dense_v1_is_deterministic_and_stays_in_observed_rectangle() -> None:
    skin, context, _, _ = _dense_fixture()

    first = _reskin_reference_dense_v1(skin, context=context)
    second = _reskin_reference_dense_v1(skin, context=context)

    assert _signature(first) == _signature(second)
    assert len(first) == 9
    assert {cell.index for cell in first} == {
        (i1, 6, i3) for i1 in range(6, 9) for i3 in range(6, 9)
    }


def test_reference_dense_v1_valid_mask_is_a_hard_barrier() -> None:
    valid_mask = np.ones((13, 13, 13), dtype=np.bool_)
    valid_mask[:, :, 7] = False
    skin, context, _, original_mask = _dense_fixture(valid_mask=valid_mask)
    mask_before = valid_mask.copy()

    dense = _reskin_reference_dense_v1(skin, context=context)

    assert {cell.i1 for cell in dense} == {6}
    np.testing.assert_array_equal(original_mask, mask_before)


def test_reference_dense_v1_strict_support_threshold_stops_regrowth() -> None:
    skin, context, _, _ = _dense_fixture()
    low_support_cells = tuple(
        _SkinCell(cell.x1, cell.x2, cell.x3, 0.2, cell.fp, cell.ft)
        for cell in context.accepted_cells
    )

    dense = _reskin_reference_dense_v1(
        skin,
        context=replace(context, accepted_cells=low_support_cells),
    )

    assert [cell.index for cell in dense] == [(6, 6, 6)]


def test_reference_dense_v1_prior_occupancy_is_a_hard_barrier() -> None:
    skin, context, _, _ = _dense_fixture()
    occupancy = _SkinOccupancyMask(context.volume_shape)
    occupancy.mark_box(7, 6, 7, 0, 0, 0)
    occupancy_before = occupancy._mask.copy()

    dense = _reskin_reference_dense_v1(
        skin,
        context=replace(context, collision_grid=occupancy),
    )

    assert [cell.index for cell in dense] == [(6, 6, 6)]
    np.testing.assert_array_equal(occupancy._mask, occupancy_before)


def test_reference_dense_v1_rejects_out_of_volume_world_coordinates() -> None:
    seed = FaultCell(0.0, 6.0, 6.0, 0.9, 0.0, 90.0)
    transform = _update_transform_map(
        3,
        4,
        4,
        seed.fault_normal(),
        seed.fault_dip_vector(),
        seed.fault_strike_vector(),
    )
    accepted = (
        _SkinCell(3, 4, 4, 0.9, 0.0, 90.0),
        _SkinCell(3, 2, 4, 0.9, 0.0, 90.0),
    )
    fv = np.ones((13, 13, 13), dtype=np.float32)
    context = _ReskinContext(
        seed=_SkinCell(seed.x1, seed.x2, seed.x3, seed.fl, seed.fp, seed.ft),
        origin=(seed.x1, seed.x2, seed.x3),
        transform_map=transform,
        accepted_cells=accepted,
        fv=fv,
        volume_shape=fv.shape,
        collision_grid=None,
        du=5.0,
        valid_mask=None,
    )
    skin = FaultSkin.from_cells((seed, FaultCell(1.0, 6.0, 6.0, 0.9, 0.0, 90.0)))

    dense = _reskin_reference_dense_v1(skin, context=context)

    assert [cell.index for cell in dense] == [(0, 6, 6)]


def test_reference_dense_v1_rounded_duplicate_prefers_observed_key() -> None:
    seed = FaultCell(6.3, 6.0, 6.0, 0.9, 0.0, 90.0)
    transform = _update_transform_map(
        3,
        4,
        4,
        seed.fault_normal(),
        seed.fault_dip_vector(),
        seed.fault_strike_vector(),
    )
    transform.vs[:] *= np.float32(0.2)
    accepted = (
        _SkinCell(3, 4, 4, 0.9, 0.0, 90.0),
        _SkinCell(3, 6, 4, 0.9, 0.0, 90.0),
    )
    fv = np.ones((13, 13, 13), dtype=np.float32)
    context = _ReskinContext(
        seed=_SkinCell(seed.x1, seed.x2, seed.x3, seed.fl, seed.fp, seed.ft),
        origin=(seed.x1, seed.x2, seed.x3),
        transform_map=transform,
        accepted_cells=accepted,
        fv=fv,
        volume_shape=fv.shape,
        collision_grid=None,
        du=5.0,
        valid_mask=None,
    )
    skin = FaultSkin.from_cells((seed, FaultCell(6.7, 6.0, 6.0, 0.9, 0.0, 90.0)))

    dense = _reskin_reference_dense_v1(skin, context=context)

    by_index = {cell.index: cell for cell in dense}
    assert set(by_index) == {(6, 6, 6), (7, 6, 6)}
    assert np.isclose(by_index[(7, 6, 6)].x1, 6.7)


def test_reference_dense_v1_public_policy_fills_one_and_two_cell_gaps() -> None:
    for size, missing, expected_sparse in (
        (3, {(7, 7)}, 8),
        (4, {(7, 7), (7, 8), (8, 7), (8, 8)}, 12),
    ):
        fv = np.zeros((15, 15, 15), dtype=np.float32)
        vp = np.zeros_like(fv)
        vt = np.full_like(fv, 90.0)
        for i3 in range(6, 6 + size):
            for i1 in range(6, 6 + size):
                if (i1, i3) not in missing:
                    fv[i3, 7, i1] = 0.9
        seed = FaultCell(6.0, 7.0, 6.0, 0.9, 0.0, 90.0)
        options = {
            "min_likelihood": 0.5,
            "ru": 3,
            "rv": 5,
            "rw": 5,
            "max_steps": 5,
        }

        sparse = FaultSkinner().find_skin(
            seed,
            fv,
            vp,
            vt,
            reskin_policy=RESKIN_POLICY_EXISTING_CELLS_V1,
            **options,
        )
        dense = FaultSkinner().find_skin(
            seed,
            fv,
            vp,
            vt,
            reskin_policy=RESKIN_POLICY_REFERENCE_DENSE_V1,
            **options,
        )

        assert len(sparse) == expected_sparse
        assert len(dense) == size**2
        assert all(
            {cell.index: cell for cell in dense}[index].fl == 0.0
            for i1, i3 in missing
            for index in [(i1, 7, i3)]
        )
