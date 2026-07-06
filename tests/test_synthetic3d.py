import numpy as np
import pytest

from pyosv.synthetic3d import (
    Synthetic3DCase,
    SyntheticPlaneSpec,
    coordinate_grids3,
    validate_center3,
    validate_shape3,
)


def test_coordinate_grids3_returns_x1_x2_x3_order() -> None:
    x1, x2, x3 = coordinate_grids3((2, 3, 4))

    assert x1.shape == (2, 3, 4)
    assert x2.shape == (2, 3, 4)
    assert x3.shape == (2, 3, 4)
    assert x1.dtype == np.float32
    assert x2.dtype == np.float32
    assert x3.dtype == np.float32
    np.testing.assert_array_equal(x1[0, 0, :], np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(x2[0, :, 0], np.array([0.0, 1.0, 2.0], dtype=np.float32))
    np.testing.assert_array_equal(x3[:, 0, 0], np.array([0.0, 1.0], dtype=np.float32))


@pytest.mark.parametrize(
    "shape",
    [
        [2, 3, 4],
        (2, 3),
        (2, 3, 0),
        (2, -3, 4),
        (2, True, 4),
        (2, 3.0, 4),
    ],
)
def test_validate_shape3_rejects_invalid_shapes(shape: object) -> None:
    with pytest.raises(ValueError):
        validate_shape3(shape)


@pytest.mark.parametrize(
    "center",
    [
        [1.0, 1.0, 1.0],
        (4.0, 1.0, 1.0),
        (1.0, 3.0, 1.0),
        (1.0, 1.0, 2.0),
        (-1.0, 1.0, 1.0),
        (1.0, np.inf, 1.0),
        (1.0, np.nan, 1.0),
        (1.0, False, 1.0),
    ],
)
def test_validate_center3_rejects_invalid_centers(center: object) -> None:
    with pytest.raises(ValueError):
        validate_center3(center, (2, 3, 4))


@pytest.mark.parametrize("likelihood_sigma", [0.0, -1.0, np.inf, np.nan, True])
def test_synthetic_plane_spec_rejects_invalid_likelihood_sigma(
    likelihood_sigma: object,
) -> None:
    with pytest.raises(ValueError):
        SyntheticPlaneSpec(
            case_id="invalid",
            shape=(2, 3, 4),
            center=(1.0, 1.0, 1.0),
            strike=30.0,
            dip=60.0,
            likelihood_sigma=likelihood_sigma,
        )


@pytest.mark.parametrize("mask_half_width", [-1.0, np.inf, np.nan, False])
def test_synthetic_plane_spec_rejects_invalid_mask_half_width(mask_half_width: object) -> None:
    with pytest.raises(ValueError):
        SyntheticPlaneSpec(
            case_id="invalid",
            shape=(2, 3, 4),
            center=(1.0, 1.0, 1.0),
            strike=30.0,
            dip=60.0,
            mask_half_width=mask_half_width,
        )


@pytest.mark.parametrize("field", ["shape", "center", "strike", "dip"])
def test_synthetic_plane_spec_rejects_invalid_geometry_fields(field: str) -> None:
    kwargs = {
        "case_id": "invalid",
        "shape": (2, 3, 4),
        "center": (1.0, 1.0, 1.0),
        "strike": 30.0,
        "dip": 60.0,
    }
    kwargs[field] = {
        "shape": (2, 3, 0),
        "center": (4.0, 1.0, 1.0),
        "strike": np.inf,
        "dip": np.nan,
    }[field]

    with pytest.raises(ValueError):
        SyntheticPlaneSpec(**kwargs)


def test_synthetic_plane_spec_accepts_valid_values() -> None:
    spec = SyntheticPlaneSpec(
        case_id="plane-a",
        shape=(2, 3, 4),
        center=(1, 1.5, np.float32(0.5)),
        strike=np.float32(30.0),
        dip=60,
        likelihood_sigma=1.25,
        mask_half_width=0.0,
    )

    assert spec.case_id == "plane-a"
    assert spec.shape == (2, 3, 4)
    assert spec.center == (1.0, 1.5, 0.5)
    assert spec.strike == 30.0
    assert spec.dip == 60.0
    assert spec.likelihood_sigma == 1.25
    assert spec.mask_half_width == 0.0


def test_synthetic3d_case_normalizes_array_dtypes() -> None:
    shape = (2, 3, 4)
    values = np.zeros(shape, dtype=np.float64)

    case = Synthetic3DCase(
        case_id="case-a",
        shape=shape,
        truth_fault_mask=values,
        truth_fault_id=values,
        truth_distance=values,
        truth_strike=values,
        truth_dip=values,
        ft_oracle=values,
        pt_oracle=values,
        tt_oracle=values,
    )

    assert case.truth_fault_mask.dtype == np.bool_
    assert case.truth_fault_id.dtype == np.int32
    for array in (
        case.truth_distance,
        case.truth_strike,
        case.truth_dip,
        case.ft_oracle,
        case.pt_oracle,
        case.tt_oracle,
    ):
        assert array.shape == shape
        assert array.dtype == np.float32


def test_synthetic3d_case_rejects_array_shape_mismatch() -> None:
    shape = (2, 3, 4)
    values = np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError):
        Synthetic3DCase(
            case_id="invalid",
            shape=shape,
            truth_fault_mask=np.zeros((2, 3, 3), dtype=bool),
            truth_fault_id=values,
            truth_distance=values,
            truth_strike=values,
            truth_dip=values,
            ft_oracle=values,
            pt_oracle=values,
            tt_oracle=values,
        )
