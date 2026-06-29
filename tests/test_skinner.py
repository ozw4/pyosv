import numpy as np
import pytest

from pyosv.cells import FaultCell
from pyosv.skinner import FaultSkinner


def test_constructor_stores_configuration_for_later_grouping() -> None:
    skinner = FaultSkinner(min_likelihood=np.float32(0.35), min_skin_size=np.int32(4))

    assert skinner.min_likelihood == pytest.approx(0.35)
    assert skinner.min_skin_size == 4
    assert skinner.connectivity == "corner"


@pytest.mark.parametrize("min_likelihood", [np.nan, np.inf, True, "0.5"])
def test_constructor_rejects_invalid_min_likelihood(min_likelihood: object) -> None:
    with pytest.raises(ValueError, match="min_likelihood"):
        FaultSkinner(min_likelihood=min_likelihood)  # type: ignore[arg-type]


@pytest.mark.parametrize("min_skin_size", [-1, 1.5, True, "1"])
def test_constructor_rejects_invalid_min_skin_size(min_skin_size: object) -> None:
    with pytest.raises(ValueError, match="min_skin_size"):
        FaultSkinner(min_skin_size=min_skin_size)  # type: ignore[arg-type]


def test_constructor_rejects_unknown_connectivity() -> None:
    with pytest.raises(ValueError, match="connectivity"):
        FaultSkinner(connectivity="diagonal")


def test_cells_from_votes_extracts_thresholded_fault_cells_in_volume_order() -> None:
    fv = np.zeros((2, 2, 3), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)
    fv[1, 0, 2] = 0.75
    fv[0, 1, 0] = 0.5
    fv[1, 1, 1] = 0.9
    vp[1, 0, 2] = 30.0
    vt[1, 0, 2] = 60.0
    vp[0, 1, 0] = 20.0
    vt[0, 1, 0] = 50.0
    vp[1, 1, 1] = 40.0
    vt[1, 1, 1] = 70.0

    cells = FaultSkinner().cells_from_votes(fv, vp, vt, min_likelihood=0.5)

    assert [cell.index for cell in cells] == [(0, 1, 0), (2, 0, 1), (1, 1, 1)]
    assert [cell.fl for cell in cells] == pytest.approx([0.5, 0.75, 0.9])
    assert [cell.fp for cell in cells] == pytest.approx([20.0, 30.0, 40.0])
    assert [cell.ft for cell in cells] == pytest.approx([50.0, 60.0, 70.0])
    assert all(isinstance(cell, FaultCell) for cell in cells)


def test_cells_from_votes_uses_configured_threshold_by_default() -> None:
    fv = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    vp = np.full_like(fv, 10.0)
    vt = np.full_like(fv, 20.0)

    cells = FaultSkinner(min_likelihood=0.5).cells_from_votes(fv, vp, vt)

    assert [cell.index for cell in cells] == [(1, 0, 0), (2, 0, 0)]


def test_cells_from_votes_accepts_call_threshold_override() -> None:
    fv = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    vp = np.full_like(fv, 10.0)
    vt = np.full_like(fv, 20.0)

    cells = FaultSkinner(min_likelihood=0.7).cells_from_votes(fv, vp, vt, min_likelihood=0.5)

    assert [cell.index for cell in cells] == [(1, 0, 0), (2, 0, 0)]


def test_cells_from_votes_returns_empty_list_without_candidates() -> None:
    fv = np.zeros((1, 2, 3), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)

    assert FaultSkinner().cells_from_votes(fv, vp, vt, min_likelihood=0.1) == []


@pytest.mark.parametrize(
    ("fv", "vp", "vt", "match"),
    [
        (
            np.zeros((1, 2, 3), dtype=np.float32),
            np.zeros((1, 2, 2), dtype=np.float32),
            np.zeros((1, 2, 3), dtype=np.float32),
            "shapes must match",
        ),
        (
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
            "3D array",
        ),
    ],
)
def test_cells_from_votes_rejects_invalid_shapes(
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        FaultSkinner().cells_from_votes(fv, vp, vt)


@pytest.mark.parametrize("name", ["fv", "vp", "vt"])
def test_cells_from_votes_rejects_non_finite_inputs(name: str) -> None:
    arrays = {
        "fv": np.zeros((1, 2, 3), dtype=np.float32),
        "vp": np.zeros((1, 2, 3), dtype=np.float32),
        "vt": np.zeros((1, 2, 3), dtype=np.float32),
    }
    arrays[name][0, 0, 0] = np.nan

    with pytest.raises(ValueError, match=f"{name} must contain only finite values"):
        FaultSkinner().cells_from_votes(arrays["fv"], arrays["vp"], arrays["vt"])


@pytest.mark.parametrize("min_likelihood", [np.nan, np.inf, True, "0.5"])
def test_cells_from_votes_rejects_invalid_threshold_override(min_likelihood: object) -> None:
    fv = np.zeros((1, 1, 1), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.zeros_like(fv)

    with pytest.raises(ValueError, match="min_likelihood"):
        FaultSkinner().cells_from_votes(
            fv,
            vp,
            vt,
            min_likelihood=min_likelihood,  # type: ignore[arg-type]
        )
