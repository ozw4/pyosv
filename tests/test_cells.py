from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pyosv.cells import FaultCell, FaultCell2
from pyosv.geometry import (
    fault_dip_vector_from_strike_and_dip,
    fault_normal_vector_from_strike_and_dip,
    fault_strike_vector_from_strike_and_dip,
)


def test_fault_cell2_basic_attributes_are_normalized() -> None:
    cell = FaultCell2(np.int32(3), np.int64(4), np.float32(0.75), np.float64(90.0))

    assert cell.i1 == 3
    assert cell.i2 == 4
    assert cell.fl == 0.75
    assert cell.fp == 90.0
    assert isinstance(cell.i1, int)
    assert isinstance(cell.i2, int)
    assert isinstance(cell.fl, float)
    assert isinstance(cell.fp, float)


def test_fault_cell2_index_is_i1_i2_tuple() -> None:
    cell = FaultCell2(7, 8, 0.5, 30.0)

    assert cell.index == (7, 8)


@pytest.mark.parametrize(
    ("fp", "normal", "strike"),
    [
        (0.0, [0.0, 1.0], [-1.0, 0.0]),
        (90.0, [1.0, 0.0], [0.0, 1.0]),
        (180.0, [0.0, -1.0], [1.0, 0.0]),
        (270.0, [-1.0, 0.0], [0.0, -1.0]),
    ],
)
def test_fault_cell2_cardinal_vectors(fp: float, normal: list[float], strike: list[float]) -> None:
    cell = FaultCell2(1, 2, 0.5, fp)

    actual_normal = cell.fault_normal()
    actual_strike = cell.fault_strike_vector()

    assert actual_normal.dtype == np.float32
    assert actual_strike.dtype == np.float32
    assert actual_normal.shape == (2,)
    assert actual_strike.shape == (2,)
    np.testing.assert_allclose(actual_normal, np.array(normal, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(actual_strike, np.array(strike, dtype=np.float32), atol=1e-6)


@pytest.mark.parametrize("fp", [0.0, 30.0, 90.0, 135.0, 180.0, 270.0])
def test_fault_cell2_vectors_are_unit_orthogonal(fp: float) -> None:
    cell = FaultCell2(1, 2, 0.5, fp)
    normal = cell.fault_normal()
    strike = cell.fault_strike_vector()

    np.testing.assert_allclose(np.dot(normal, strike), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(strike), 1.0, atol=1e-6)


def test_fault_cell2_is_frozen() -> None:
    cell = FaultCell2(1, 2, 0.5, 45.0)

    with pytest.raises(FrozenInstanceError):
        cell.i1 = 3


def test_fault_cell_index_uses_java_style_rounding() -> None:
    cell = FaultCell(1.2, 2.5, 3.7, 0.8, 30.0, 60.0)

    assert cell.i1 == 1
    assert cell.i2 == 3
    assert cell.i3 == 4
    assert cell.index == (1, 3, 4)


def test_fault_cell_basic_attributes_are_normalized() -> None:
    cell = FaultCell(
        np.float32(1.2),
        np.float64(2.5),
        3,
        np.float32(0.8),
        np.float64(30.0),
        np.int32(60),
    )

    assert cell.x1 == 1.2000000476837158
    assert cell.x2 == 2.5
    assert cell.x3 == 3.0
    assert cell.fl == 0.800000011920929
    assert cell.fp == 30.0
    assert cell.ft == 60.0
    assert isinstance(cell.x1, float)
    assert isinstance(cell.x2, float)
    assert isinstance(cell.x3, float)
    assert isinstance(cell.fl, float)
    assert isinstance(cell.fp, float)
    assert isinstance(cell.ft, float)


def test_fault_cell_vectors_match_geometry_helpers() -> None:
    cell = FaultCell(1.2, 2.5, 3.7, 0.8, 30.0, 60.0)

    normal = cell.fault_normal()
    dip = cell.fault_dip_vector()
    strike = cell.fault_strike_vector()

    assert normal.dtype == np.float32
    assert dip.dtype == np.float32
    assert strike.dtype == np.float32
    assert normal.shape == (3,)
    assert dip.shape == (3,)
    assert strike.shape == (3,)
    np.testing.assert_allclose(normal, fault_normal_vector_from_strike_and_dip(30.0, 60.0))
    np.testing.assert_allclose(dip, fault_dip_vector_from_strike_and_dip(30.0, 60.0))
    np.testing.assert_allclose(strike, fault_strike_vector_from_strike_and_dip(30.0, 60.0))


@pytest.mark.parametrize("fp", [0.0, 30.0, 90.0, 180.0, 270.0])
@pytest.mark.parametrize("ft", [30.0, 60.0, 90.0])
def test_fault_cell_vectors_are_unit_orthogonal(fp: float, ft: float) -> None:
    cell = FaultCell(1.0, 2.0, 3.0, 0.8, fp, ft)
    normal = cell.fault_normal()
    dip = cell.fault_dip_vector()
    strike = cell.fault_strike_vector()

    np.testing.assert_allclose(np.linalg.norm(normal), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(dip), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(strike), 1.0, atol=1e-6)

    np.testing.assert_allclose(np.dot(normal, dip), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.dot(normal, strike), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.dot(dip, strike), 0.0, atol=1e-6)


def test_fault_cell_repr_contains_only_stored_fields() -> None:
    cell = FaultCell(1.2, 2.5, 3.7, 0.8, 30.0, 60.0)

    representation = repr(cell)

    assert "FaultCell(" in representation
    assert "x1=1.2" in representation
    assert "ft=60.0" in representation
    assert "i1=" not in representation
    assert "index=" not in representation
    assert "array(" not in representation
