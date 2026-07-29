from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pyosv.cells import (
    FAULT_CELL_GENERATION_CONNECTED_COMPONENT,
    FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED,
    FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED,
    FAULT_CELL_GENERATION_EXISTING_CELLS_RESKINNED,
    FAULT_CELL_GENERATION_GROWN,
    FaultCell,
)
from pyosv.skin import FaultSkin


def test_empty_fault_skin_is_iterable_and_returns_stable_arrays() -> None:
    skin = FaultSkin()

    assert len(skin) == 0
    assert list(skin) == []

    indices = skin.indices()
    likelihoods = skin.likelihoods()

    assert indices.shape == (0, 3)
    assert indices.dtype == np.int32
    assert likelihoods.shape == (0,)
    assert likelihoods.dtype == np.float32


def test_fault_skin_stores_and_iterates_fault_cells() -> None:
    first = FaultCell(1.2, 2.5, 3.7, 0.8, 30.0, 60.0)
    second = FaultCell(4.0, 5.1, 6.9, 0.4, 45.0, 70.0)
    skin = FaultSkin([first])

    skin.add(second)

    assert len(skin) == 2
    assert list(skin) == [first, second]


def test_fault_skin_append_and_helpers_are_deterministic() -> None:
    skin = FaultSkin()
    skin.append(FaultCell(1.2, 2.5, 3.7, 0.8, 30.0, 60.0))
    skin.append(FaultCell(4.0, 5.1, 6.9, 0.4, 45.0, 70.0))

    np.testing.assert_array_equal(
        skin.indices(),
        np.array(
            [
                [1, 3, 4],
                [4, 5, 7],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(skin.likelihoods(), np.array([0.8, 0.4], dtype=np.float32))


def test_fault_skin_from_cells_copies_iterable() -> None:
    cells = [FaultCell(1.0, 2.0, 3.0, 0.5, 30.0, 60.0)]
    skin = FaultSkin.from_cells(cells)

    cells.append(FaultCell(4.0, 5.0, 6.0, 0.7, 40.0, 70.0))

    assert len(skin) == 1


def test_fault_skin_rejects_non_fault_cells() -> None:
    skin = FaultSkin()

    with pytest.raises(TypeError, match="FaultCell"):
        skin.append(object())  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="FaultCell"):
        FaultSkin([object()])  # type: ignore[list-item]


def test_fault_skin_does_not_mutate_fault_cells() -> None:
    cell = FaultCell(1.2, 2.5, 3.7, 0.8, 30.0, 60.0)
    skin = FaultSkin([cell])

    _ = skin.indices()
    _ = skin.likelihoods()

    assert next(iter(skin)) is cell
    with pytest.raises(FrozenInstanceError):
        cell.fl = 0.1


def test_fault_cell_six_positional_constructor_defaults_to_grown_generation() -> None:
    cell = FaultCell(1.0, 2.0, 3.0, 0.8, 20.0, 60.0)

    assert cell.generation == FAULT_CELL_GENERATION_GROWN
    assert cell.reskin_support is None


def test_fault_cell_metadata_is_keyword_only() -> None:
    with pytest.raises(TypeError):
        FaultCell(  # type: ignore[call-arg]
            1.0,
            2.0,
            3.0,
            0.8,
            20.0,
            60.0,
            FAULT_CELL_GENERATION_GROWN,
        )


@pytest.mark.parametrize(
    ("generation", "support"),
    (
        (FAULT_CELL_GENERATION_GROWN, None),
        (FAULT_CELL_GENERATION_EXISTING_CELLS_RESKINNED, None),
        (FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED, 0.25),
        (FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED, 1.0),
        (FAULT_CELL_GENERATION_CONNECTED_COMPONENT, None),
    ),
)
def test_fault_cell_accepts_all_generation_contract_values(
    generation: str,
    support: float | None,
) -> None:
    cell = FaultCell(
        1,
        2,
        3,
        0.8,
        20,
        60,
        generation=generation,  # type: ignore[arg-type]
        reskin_support=support,
    )

    assert cell.generation == generation
    assert cell.reskin_support == support


@pytest.mark.parametrize("generation", ("", "unknown", 1, None))
def test_fault_cell_rejects_invalid_generation(generation: object) -> None:
    with pytest.raises(ValueError, match="generation"):
        FaultCell(1, 2, 3, 0.8, 20, 60, generation=generation)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "support",
    (None, np.nan, np.inf, -np.inf, True, -0.01, 1.01, "0.5"),
)
def test_fault_cell_rejects_invalid_dense_reskin_support(support: object) -> None:
    with pytest.raises(ValueError, match="reskin_support"):
        FaultCell(
            1,
            2,
            3,
            0.8,
            20,
            60,
            generation=FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED,
            reskin_support=support,  # type: ignore[arg-type]
        )


def test_fault_cell_metadata_does_not_change_geometric_equality() -> None:
    grown = FaultCell(1, 2, 3, 0.8, 20, 60)
    dense = FaultCell(
        1,
        2,
        3,
        0.8,
        20,
        60,
        generation=FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED,
        reskin_support=0.4,
    )

    assert grown == dense
