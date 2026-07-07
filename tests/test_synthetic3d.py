import numpy as np
import pytest

from pyosv.geometry import fault_normal_vector_from_strike_and_dip
from pyosv.synthetic3d import (
    Synthetic3DCase,
    SyntheticCurvedSurfaceSpec,
    SyntheticPlaneSpec,
    SyntheticScannerInputConfig,
    _SyntheticFaultComponent,
    _compose_synthetic_components,
    coordinate_grids3,
    generate_curved_surface_case,
    generate_single_plane_case,
    make_boundary_plane_case,
    make_crossing_planes_case,
    make_curved_surface_case,
    make_parallel_planes_case,
    make_scanner_input_from_case,
    make_single_dipping_plane_case,
    make_single_vertical_plane_case,
    make_weak_noisy_plane_case,
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


def test_synthetic_curved_surface_spec_accepts_valid_values() -> None:
    spec = SyntheticCurvedSurfaceSpec(
        case_id="curved-a",
        shape=(2, 3, 4),
        center=(1, 1.5, np.float32(0.5)),
        slope2=np.float32(0.2),
        slope3=-0.1,
        curvature2=0.3,
        curvature3=np.float32(-0.2),
        likelihood_sigma=1.25,
        mask_half_width=0.0,
    )

    assert spec.case_id == "curved-a"
    assert spec.shape == (2, 3, 4)
    assert spec.center == (1.0, 1.5, 0.5)
    assert spec.slope2 == pytest.approx(0.2)
    assert spec.slope3 == -0.1
    assert spec.curvature2 == 0.3
    assert spec.curvature3 == pytest.approx(-0.2)
    assert spec.likelihood_sigma == 1.25
    assert spec.mask_half_width == 0.0


@pytest.mark.parametrize("likelihood_sigma", [0.0, -1.0, np.inf, np.nan, True])
def test_synthetic_curved_surface_spec_rejects_invalid_likelihood_sigma(
    likelihood_sigma: object,
) -> None:
    with pytest.raises(ValueError):
        SyntheticCurvedSurfaceSpec(
            case_id="invalid",
            shape=(2, 3, 4),
            center=(1.0, 1.0, 1.0),
            slope2=0.2,
            slope3=-0.1,
            curvature2=0.3,
            curvature3=-0.2,
            likelihood_sigma=likelihood_sigma,
        )


@pytest.mark.parametrize("mask_half_width", [-1.0, np.inf, np.nan, False])
def test_synthetic_curved_surface_spec_rejects_invalid_mask_half_width(
    mask_half_width: object,
) -> None:
    with pytest.raises(ValueError):
        SyntheticCurvedSurfaceSpec(
            case_id="invalid",
            shape=(2, 3, 4),
            center=(1.0, 1.0, 1.0),
            slope2=0.2,
            slope3=-0.1,
            curvature2=0.3,
            curvature3=-0.2,
            mask_half_width=mask_half_width,
        )


@pytest.mark.parametrize(
    "field",
    ["shape", "center", "slope2", "slope3", "curvature2", "curvature3"],
)
def test_synthetic_curved_surface_spec_rejects_invalid_geometry_fields(field: str) -> None:
    kwargs = {
        "case_id": "invalid",
        "shape": (2, 3, 4),
        "center": (1.0, 1.0, 1.0),
        "slope2": 0.2,
        "slope3": -0.1,
        "curvature2": 0.3,
        "curvature3": -0.2,
    }
    kwargs[field] = {
        "shape": (2, 3, 0),
        "center": (4.0, 1.0, 1.0),
        "slope2": np.inf,
        "slope3": np.nan,
        "curvature2": True,
        "curvature3": np.inf,
    }[field]

    with pytest.raises(ValueError):
        SyntheticCurvedSurfaceSpec(**kwargs)


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


def test_synthetic_scanner_input_is_low_on_fault_and_high_away_from_fault() -> None:
    case = make_single_vertical_plane_case(shape=(17, 19, 21))
    scanner_input = make_scanner_input_from_case(case)

    truth_surface = case.truth_fault_mask
    far_from_fault = case.ft_oracle < 0.05

    assert np.any(truth_surface)
    assert np.any(far_from_fault)
    assert scanner_input[truth_surface].mean() < scanner_input[far_from_fault].mean()


def test_synthetic_scanner_input_is_float32_finite_and_shape_matching() -> None:
    case = make_single_dipping_plane_case(shape=(9, 11, 13))
    config = SyntheticScannerInputConfig(
        background=1.2,
        fault_contrast=1.0,
        clip_min=0.1,
        clip_max=0.9,
    )

    scanner_input = make_scanner_input_from_case(case, config)
    repeated = make_scanner_input_from_case(case, config)

    assert scanner_input.shape == case.shape
    assert scanner_input.dtype == np.float32
    assert np.all(np.isfinite(scanner_input))
    assert scanner_input.min() >= config.clip_min
    assert scanner_input.max() <= config.clip_max
    np.testing.assert_array_equal(scanner_input, repeated)


def test_synthetic_scanner_input_noise_is_deterministic_for_same_seed() -> None:
    case = make_curved_surface_case(shape=(9, 11, 13))
    config = SyntheticScannerInputConfig(noise_sigma=0.04, seed=1234)

    scanner_input = make_scanner_input_from_case(case, config)
    repeated = make_scanner_input_from_case(case, config)

    np.testing.assert_array_equal(scanner_input, repeated)


def test_synthetic_scanner_input_noise_changes_with_different_seed() -> None:
    case = make_curved_surface_case(shape=(9, 11, 13))
    config_a = SyntheticScannerInputConfig(noise_sigma=0.04, seed=1234)
    config_b = SyntheticScannerInputConfig(noise_sigma=0.04, seed=5678)

    scanner_input_a = make_scanner_input_from_case(case, config_a)
    scanner_input_b = make_scanner_input_from_case(case, config_b)

    assert not np.array_equal(scanner_input_a, scanner_input_b)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"background": np.nan},
        {"background": np.inf},
        {"background": True},
        {"fault_contrast": -0.1},
        {"fault_contrast": np.nan},
        {"fault_contrast": False},
        {"noise_sigma": -0.1},
        {"noise_sigma": np.inf},
        {"noise_sigma": True},
        {"seed": 1.5},
        {"seed": False},
        {"clip_min": np.nan},
        {"clip_max": np.inf},
        {"clip_min": 1.0, "clip_max": 1.0},
        {"clip_min": 2.0, "clip_max": 1.0},
    ],
)
def test_synthetic_scanner_input_config_rejects_invalid_values(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        SyntheticScannerInputConfig(**kwargs)


@pytest.mark.parametrize(
    "field",
    [
        "truth_distance",
        "truth_strike",
        "truth_dip",
        "ft_oracle",
        "pt_oracle",
        "tt_oracle",
    ],
)
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_synthetic3d_case_rejects_nonfinite_float_arrays(field: str, bad_value: float) -> None:
    shape = (2, 3, 4)
    values = np.zeros(shape, dtype=np.float32)
    kwargs = {
        "case_id": "invalid",
        "shape": shape,
        "truth_fault_mask": np.zeros(shape, dtype=bool),
        "truth_fault_id": np.zeros(shape, dtype=np.int32),
        "truth_distance": values,
        "truth_strike": values,
        "truth_dip": values,
        "ft_oracle": values,
        "pt_oracle": values,
        "tt_oracle": values,
    }
    invalid = values.copy()
    invalid[0, 0, 0] = bad_value
    kwargs[field] = invalid

    with pytest.raises(ValueError, match="finite"):
        Synthetic3DCase(**kwargs)


def test_compose_synthetic_components_builds_union_mask() -> None:
    shape = (1, 1, 5)
    component_a = _SyntheticFaultComponent(
        fault_id=1,
        signed_distance=np.array([[[-2.0, -0.5, 2.0, 2.0, 2.0]]], dtype=np.float32),
        strike=np.zeros(shape, dtype=np.float32),
        dip=np.zeros(shape, dtype=np.float32),
        likelihood=np.zeros(shape, dtype=np.float32),
        mask_half_width=1.0,
    )
    component_b = _SyntheticFaultComponent(
        fault_id=2,
        signed_distance=np.array([[[2.0, 2.0, 0.25, 0.75, 2.0]]], dtype=np.float32),
        strike=np.zeros(shape, dtype=np.float32),
        dip=np.zeros(shape, dtype=np.float32),
        likelihood=np.zeros(shape, dtype=np.float32),
        mask_half_width=1.0,
    )

    case = _compose_synthetic_components(
        case_id="multi",
        shape=shape,
        components=(component_a, component_b),
    )

    np.testing.assert_array_equal(
        case.truth_fault_mask,
        np.array([[[False, True, True, True, False]]], dtype=bool),
    )
    np.testing.assert_array_equal(
        case.truth_fault_id,
        np.array([[[0, 1, 2, 2, 0]]], dtype=np.int32),
    )


def test_compose_synthetic_components_uses_nearest_component_orientation() -> None:
    shape = (1, 1, 3)
    component_a = _SyntheticFaultComponent(
        fault_id=3,
        signed_distance=np.array([[[0.8, 0.2, 0.5]]], dtype=np.float32),
        strike=np.full(shape, 30.0, dtype=np.float32),
        dip=np.full(shape, 60.0, dtype=np.float32),
        likelihood=np.zeros(shape, dtype=np.float32),
        mask_half_width=1.0,
    )
    component_b = _SyntheticFaultComponent(
        fault_id=7,
        signed_distance=np.array([[[0.1, 0.6, 0.4]]], dtype=np.float32),
        strike=np.full(shape, 70.0, dtype=np.float32),
        dip=np.full(shape, 80.0, dtype=np.float32),
        likelihood=np.zeros(shape, dtype=np.float32),
        mask_half_width=1.0,
    )

    case = _compose_synthetic_components(
        case_id="multi",
        shape=shape,
        components=(component_a, component_b),
    )

    np.testing.assert_array_equal(case.truth_fault_id, np.array([[[7, 3, 7]]], dtype=np.int32))
    np.testing.assert_array_equal(case.truth_strike, np.array([[[70.0, 30.0, 70.0]]]))
    np.testing.assert_array_equal(case.truth_dip, np.array([[[80.0, 60.0, 80.0]]]))
    np.testing.assert_array_equal(case.pt_oracle, case.truth_strike)
    np.testing.assert_array_equal(case.tt_oracle, case.truth_dip)


def test_compose_synthetic_components_breaks_distance_tie_by_smaller_fault_id() -> None:
    shape = (1, 1, 1)
    component_high = _SyntheticFaultComponent(
        fault_id=9,
        signed_distance=np.array([[[0.5]]], dtype=np.float32),
        strike=np.full(shape, 90.0, dtype=np.float32),
        dip=np.full(shape, 95.0, dtype=np.float32),
        likelihood=np.zeros(shape, dtype=np.float32),
        mask_half_width=1.0,
    )
    component_low = _SyntheticFaultComponent(
        fault_id=2,
        signed_distance=np.array([[[-0.5]]], dtype=np.float32),
        strike=np.full(shape, 20.0, dtype=np.float32),
        dip=np.full(shape, 25.0, dtype=np.float32),
        likelihood=np.zeros(shape, dtype=np.float32),
        mask_half_width=1.0,
    )

    case = _compose_synthetic_components(
        case_id="tie",
        shape=shape,
        components=(component_high, component_low),
    )

    assert case.truth_fault_id[0, 0, 0] == 2
    assert case.truth_strike[0, 0, 0] == 20.0
    assert case.truth_dip[0, 0, 0] == 25.0


def test_compose_synthetic_components_uses_maximum_component_likelihood() -> None:
    shape = (1, 1, 2)
    component_a = _SyntheticFaultComponent(
        fault_id=1,
        signed_distance=np.full(shape, 0.25, dtype=np.float32),
        strike=np.zeros(shape, dtype=np.float32),
        dip=np.zeros(shape, dtype=np.float32),
        likelihood=np.array([[[0.2, 1.5]]], dtype=np.float32),
        mask_half_width=1.0,
    )
    component_b = _SyntheticFaultComponent(
        fault_id=2,
        signed_distance=np.full(shape, 0.5, dtype=np.float32),
        strike=np.zeros(shape, dtype=np.float32),
        dip=np.zeros(shape, dtype=np.float32),
        likelihood=np.array([[[0.7, -0.5]]], dtype=np.float32),
        mask_half_width=1.0,
    )

    case = _compose_synthetic_components(
        case_id="likelihood",
        shape=shape,
        components=(component_a, component_b),
    )

    np.testing.assert_array_equal(case.ft_oracle, np.array([[[0.7, 1.0]]], dtype=np.float32))


def test_make_single_vertical_plane_case_returns_expected_arrays() -> None:
    case = make_single_vertical_plane_case(shape=(5, 7, 9))

    assert isinstance(case, Synthetic3DCase)
    assert case.case_id == "single_vertical_plane"
    assert case.shape == (5, 7, 9)
    expected_dtypes = {
        "truth_fault_mask": np.bool_,
        "truth_fault_id": np.int32,
        "truth_distance": np.float32,
        "truth_strike": np.float32,
        "truth_dip": np.float32,
        "ft_oracle": np.float32,
        "pt_oracle": np.float32,
        "tt_oracle": np.float32,
    }
    for name, dtype in expected_dtypes.items():
        array = getattr(case, name)
        assert array.shape == case.shape
        assert array.dtype == dtype


def test_make_single_dipping_plane_case_returns_expected_arrays() -> None:
    case = make_single_dipping_plane_case(shape=(5, 7, 9))

    assert isinstance(case, Synthetic3DCase)
    assert case.case_id == "single_dipping_plane"
    assert case.shape == (5, 7, 9)
    expected_dtypes = {
        "truth_fault_mask": np.bool_,
        "truth_fault_id": np.int32,
        "truth_distance": np.float32,
        "truth_strike": np.float32,
        "truth_dip": np.float32,
        "ft_oracle": np.float32,
        "pt_oracle": np.float32,
        "tt_oracle": np.float32,
    }
    for name, dtype in expected_dtypes.items():
        array = getattr(case, name)
        assert array.shape == case.shape
        assert array.dtype == dtype


def test_make_boundary_plane_case_touches_i2_zero_face() -> None:
    case = make_boundary_plane_case(shape=(5, 7, 9))

    assert isinstance(case, Synthetic3DCase)
    assert case.case_id == "boundary_plane"
    assert case.shape == (5, 7, 9)
    _assert_synthetic3d_case_contract(case)
    np.testing.assert_array_equal(case.truth_fault_id, case.truth_fault_mask.astype(np.int32))
    np.testing.assert_array_equal(case.truth_strike, np.zeros(case.shape, dtype=np.float32))
    np.testing.assert_array_equal(case.truth_dip, np.full(case.shape, 90.0, dtype=np.float32))

    assert np.any(case.truth_fault_mask[:, 0, :])
    assert not np.any(case.truth_fault_mask[:, -1, :])
    masked_x2 = np.flatnonzero(case.truth_fault_mask.any(axis=(0, 2)))
    np.testing.assert_array_equal(masked_x2, np.array([0, 1, 2]))


def test_make_curved_surface_case_returns_expected_arrays() -> None:
    case = make_curved_surface_case(shape=(5, 7, 9))

    assert isinstance(case, Synthetic3DCase)
    assert case.case_id == "curved_surface"
    assert case.shape == (5, 7, 9)
    expected_dtypes = {
        "truth_fault_mask": np.bool_,
        "truth_fault_id": np.int32,
        "truth_distance": np.float32,
        "truth_strike": np.float32,
        "truth_dip": np.float32,
        "ft_oracle": np.float32,
        "pt_oracle": np.float32,
        "tt_oracle": np.float32,
    }
    for name, dtype in expected_dtypes.items():
        array = getattr(case, name)
        assert array.shape == case.shape
        assert array.dtype == dtype
        assert np.all(np.isfinite(array))

    assert np.all((case.truth_strike >= 0.0) & (case.truth_strike < 360.0))
    assert np.all((case.truth_dip >= 0.0) & (case.truth_dip <= 180.0))


def test_make_parallel_planes_case_returns_separated_faults() -> None:
    case = make_parallel_planes_case(shape=(17, 33, 21))

    assert isinstance(case, Synthetic3DCase)
    assert case.case_id == "parallel_planes"
    assert case.shape == (17, 33, 21)
    _assert_synthetic3d_case_contract(case)
    assert {0, 1, 2}.issubset(set(np.unique(case.truth_fault_id).tolist()))

    fault1_x2 = np.nonzero(case.truth_fault_id == 1)[1]
    fault2_x2 = np.nonzero(case.truth_fault_id == 2)[1]
    assert fault1_x2.size > 0
    assert fault2_x2.size > 0
    assert fault1_x2.max() < fault2_x2.min()

    background = ~case.truth_fault_mask
    assert case.ft_oracle[case.truth_fault_mask].mean() > case.ft_oracle[background].mean()


def test_make_crossing_planes_case_returns_two_orientations_and_is_deterministic() -> None:
    case = make_crossing_planes_case(shape=(21, 25, 27))
    repeated = make_crossing_planes_case(shape=(21, 25, 27))

    assert isinstance(case, Synthetic3DCase)
    assert case.case_id == "crossing_planes"
    assert case.shape == (21, 25, 27)
    _assert_synthetic3d_case_contract(case)
    assert {0, 1, 2}.issubset(set(np.unique(case.truth_fault_id).tolist()))

    mask_pairs = np.column_stack(
        (
            np.round(case.truth_strike[case.truth_fault_mask], decimals=3),
            np.round(case.truth_dip[case.truth_fault_mask], decimals=3),
        )
    )
    assert np.unique(mask_pairs, axis=0).shape[0] >= 2

    center_neighborhood = case.truth_strike[9:12, 11:14, 12:15]
    assert np.all(np.isfinite(center_neighborhood))
    assert np.all(np.isfinite(case.truth_dip[9:12, 11:14, 12:15]))
    assert np.all(np.isfinite(case.ft_oracle[9:12, 11:14, 12:15]))

    for name in (
        "truth_fault_mask",
        "truth_fault_id",
        "truth_distance",
        "truth_strike",
        "truth_dip",
        "ft_oracle",
        "pt_oracle",
        "tt_oracle",
    ):
        np.testing.assert_array_equal(getattr(case, name), getattr(repeated, name))


def test_make_weak_noisy_plane_case_returns_expected_arrays_and_orientation() -> None:
    case = make_weak_noisy_plane_case(shape=(21, 25, 27))

    assert isinstance(case, Synthetic3DCase)
    assert case.case_id == "weak_noisy_plane"
    assert case.shape == (21, 25, 27)
    _assert_synthetic3d_case_contract(case)
    np.testing.assert_array_equal(case.truth_fault_id, case.truth_fault_mask.astype(np.int32))
    np.testing.assert_array_equal(case.truth_strike, np.full(case.shape, 35.0, dtype=np.float32))
    np.testing.assert_array_equal(case.truth_dip, np.full(case.shape, 70.0, dtype=np.float32))
    np.testing.assert_array_equal(case.pt_oracle, case.truth_strike)
    np.testing.assert_array_equal(case.tt_oracle, case.truth_dip)


def test_make_weak_noisy_plane_case_is_deterministic() -> None:
    case = make_weak_noisy_plane_case(shape=(17, 19, 21))
    repeated = make_weak_noisy_plane_case(shape=(17, 19, 21))

    for name in (
        "truth_fault_mask",
        "truth_fault_id",
        "truth_distance",
        "truth_strike",
        "truth_dip",
        "ft_oracle",
        "pt_oracle",
        "tt_oracle",
    ):
        np.testing.assert_array_equal(getattr(case, name), getattr(repeated, name))


def test_make_weak_noisy_plane_case_has_degraded_likelihood() -> None:
    shape = (21, 25, 27)
    case = make_weak_noisy_plane_case(shape=shape)
    ideal = generate_single_plane_case(
        SyntheticPlaneSpec(
            case_id="ideal",
            shape=shape,
            center=(13.0, 12.0, 10.0),
            strike=35.0,
            dip=70.0,
            likelihood_sigma=1.25,
            mask_half_width=1.0,
        )
    )

    assert case.ft_oracle.max() > 0.5
    assert (
        case.ft_oracle[case.truth_fault_mask].mean() > case.ft_oracle[~case.truth_fault_mask].mean()
    )
    assert not np.array_equal(case.ft_oracle, ideal.ft_oracle)
    np.testing.assert_array_equal(case.truth_distance, ideal.truth_distance)


def test_single_plane_case_truth_mask_ids_and_oracle_likelihood_are_consistent() -> None:
    spec = SyntheticPlaneSpec(
        case_id="plane-a",
        shape=(5, 7, 9),
        center=(4.0, 3.0, 2.0),
        strike=30.0,
        dip=60.0,
        likelihood_sigma=1.25,
        mask_half_width=1.0,
    )
    case = generate_single_plane_case(spec)

    np.testing.assert_array_equal(case.truth_fault_id, case.truth_fault_mask.astype(np.int32))
    assert np.all(case.ft_oracle >= 0.0)
    assert np.all(case.ft_oracle <= 1.0)

    abs_distance = np.abs(case.truth_distance)
    near_index = np.unravel_index(np.argmin(abs_distance), abs_distance.shape)
    far_index = np.unravel_index(np.argmax(abs_distance), abs_distance.shape)
    assert case.ft_oracle[near_index] > case.ft_oracle[far_index]


def test_single_vertical_plane_case_mask_is_centered_near_x2() -> None:
    case = make_single_vertical_plane_case(shape=(5, 7, 9))
    masked_x2 = np.flatnonzero(case.truth_fault_mask.any(axis=(0, 2)))

    np.testing.assert_array_equal(masked_x2, np.array([2, 3, 4]))


def test_single_vertical_plane_case_has_constant_truth_orientation() -> None:
    case = make_single_vertical_plane_case(shape=(5, 7, 9))

    np.testing.assert_array_equal(case.truth_strike, np.zeros(case.shape, dtype=np.float32))
    np.testing.assert_array_equal(case.truth_dip, np.full(case.shape, 90.0, dtype=np.float32))
    np.testing.assert_array_equal(case.pt_oracle, case.truth_strike)
    np.testing.assert_array_equal(case.tt_oracle, case.truth_dip)


def test_single_dipping_plane_case_has_constant_truth_orientation() -> None:
    case = make_single_dipping_plane_case(shape=(5, 7, 9))

    np.testing.assert_array_equal(case.truth_strike, np.full(case.shape, 45.0, dtype=np.float32))
    np.testing.assert_array_equal(case.truth_dip, np.full(case.shape, 65.0, dtype=np.float32))
    np.testing.assert_array_equal(case.pt_oracle, case.truth_strike)
    np.testing.assert_array_equal(case.tt_oracle, case.truth_dip)


def test_single_dipping_plane_case_distance_follows_analytic_normal() -> None:
    case = make_single_dipping_plane_case(shape=(9, 9, 9))
    normal = fault_normal_vector_from_strike_and_dip(45.0, 65.0).astype(np.float64)
    center = np.array((4.0, 4.0, 4.0), dtype=np.float64)

    samples = [
        center,
        center + normal,
        center - normal,
        center + 2.0 * normal,
        center - 2.0 * normal,
    ]
    for sample in samples:
        x1, x2, x3 = sample
        nearest = (int(round(x3)), int(round(x2)), int(round(x1)))
        expected = float(np.dot(np.array((nearest[2], nearest[1], nearest[0])) - center, normal))
        assert case.truth_distance[nearest] == pytest.approx(expected, abs=1.0e-6)


def test_curved_surface_case_distance_follows_analytic_surface() -> None:
    spec = SyntheticCurvedSurfaceSpec(
        case_id="curved-a",
        shape=(7, 9, 11),
        center=(5.0, 4.0, 3.0),
        slope2=0.2,
        slope3=-0.1,
        curvature2=0.4,
        curvature3=-0.3,
        likelihood_sigma=1.25,
        mask_half_width=1.0,
    )
    case = generate_curved_surface_case(spec)
    scale2 = float(spec.shape[1] - 1)
    scale3 = float(spec.shape[0] - 1)

    for index in [(3, 4, 5), (3, 5, 6), (4, 2, 7), (1, 7, 3)]:
        x3, x2, x1 = index
        dx2 = x2 - spec.center[1]
        dx3 = x3 - spec.center[2]
        x1_surface = (
            spec.center[0]
            + spec.slope2 * dx2
            + spec.slope3 * dx3
            + spec.curvature2 * (dx2**2 / scale2)
            + spec.curvature3 * (dx3**2 / scale3)
        )
        expected = x1 - x1_surface
        assert case.truth_distance[index] == pytest.approx(expected, abs=1.0e-6)


def test_curved_surface_case_truth_orientation_varies_spatially() -> None:
    case = make_curved_surface_case(shape=(9, 11, 13))

    assert np.ptp(case.truth_strike) > 0.0
    assert np.ptp(case.truth_dip) > 0.0
    np.testing.assert_array_equal(case.pt_oracle, case.truth_strike)
    np.testing.assert_array_equal(case.tt_oracle, case.truth_dip)


def test_curved_surface_case_truth_mask_ids_and_oracle_likelihood_are_consistent() -> None:
    case = make_curved_surface_case(shape=(9, 11, 13))

    np.testing.assert_array_equal(case.truth_fault_id, case.truth_fault_mask.astype(np.int32))
    assert np.all(case.ft_oracle >= 0.0)
    assert np.all(case.ft_oracle <= 1.0)

    abs_distance = np.abs(case.truth_distance)
    near_index = np.unravel_index(np.argmin(abs_distance), abs_distance.shape)
    far_index = np.unravel_index(np.argmax(abs_distance), abs_distance.shape)
    assert case.ft_oracle[near_index] > 0.95
    assert case.ft_oracle[near_index] > case.ft_oracle[far_index]


def test_generate_single_plane_case_rejects_invalid_spec() -> None:
    with pytest.raises(ValueError):
        generate_single_plane_case(object())


def test_generate_curved_surface_case_rejects_invalid_spec() -> None:
    with pytest.raises(ValueError):
        generate_curved_surface_case(object())


def _assert_synthetic3d_case_contract(case: Synthetic3DCase) -> None:
    expected_dtypes = {
        "truth_fault_mask": np.bool_,
        "truth_fault_id": np.int32,
        "truth_distance": np.float32,
        "truth_strike": np.float32,
        "truth_dip": np.float32,
        "ft_oracle": np.float32,
        "pt_oracle": np.float32,
        "tt_oracle": np.float32,
    }
    for name, dtype in expected_dtypes.items():
        array = getattr(case, name)
        assert array.shape == case.shape
        assert array.dtype == dtype
        assert np.all(np.isfinite(array))
