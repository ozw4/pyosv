from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pyosv.cells import FaultCell2


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
