from __future__ import annotations

from dataclasses import MISSING, FrozenInstanceError, fields

import numpy as np
import pytest

from pyosv.fault_warping import (
    FAULT_WARPING_CONTRACT_VERSION,
    FaultSurfaceGraph,
    FaultWarpingConfig,
    FaultWarpingInput,
    FaultWarpingResult,
    ReflectorSlopeVolume,
)


VOLUME_SHAPE = (4, 5, 6)  # (n3, n2, n1)
FLOAT_RESULT_FIELDS = (
    "shift_samples",
    "correlation_before",
    "correlation_after",
    "cost_margin",
    "cycle_residual_samples",
    "valid_sample_fraction",
)


def _surface_data(count: int = 4) -> dict[str, np.ndarray]:
    if count == 1:
        return {
            "x1": np.array([1.25], dtype=np.float32),
            "x2": np.array([2.5], dtype=np.float32),
            "x3": np.array([1.75], dtype=np.float32),
            "strike_deg": np.array([359.0], dtype=np.float32),
            "dip_deg": np.array([45.0], dtype=np.float32),
            "ca_index": np.array([-1], dtype=np.int64),
            "cb_index": np.array([-1], dtype=np.int64),
            "cl_index": np.array([-1], dtype=np.int64),
            "cr_index": np.array([-1], dtype=np.int64),
        }
    if count != 4:
        raise ValueError("test helper supports one or four cells")
    return {
        "x1": np.array([0.25, 5.0, 2.5, 4.75], dtype=np.float32),
        "x2": np.array([0.0, 4.0, 2.25, 3.0], dtype=np.float32),
        "x3": np.array([0.0, 3.0, 2.0, 1.5], dtype=np.float32),
        "strike_deg": np.array([0.0, 359.0, 90.0, 180.0], dtype=np.float32),
        "dip_deg": np.array([90.0, 80.0, 45.0, 10.0], dtype=np.float32),
        "ca_index": np.array([1, -1, 3, -1], dtype=np.int64),
        "cb_index": np.array([-1, 0, -1, 2], dtype=np.int64),
        "cl_index": np.array([2, 3, -1, -1], dtype=np.int64),
        "cr_index": np.array([-1, -1, 0, 1], dtype=np.int64),
    }


def _surface(
    count: int = 4,
    cell_support_weight: np.ndarray | None = None,
    **overrides: object,
) -> FaultSurfaceGraph:
    values: dict[str, object] = _surface_data(count)
    values.update(overrides)
    values["cell_support_weight"] = cell_support_weight
    return FaultSurfaceGraph(**values)  # type: ignore[arg-type]


def _slopes(shape: tuple[int, int, int] = VOLUME_SHAPE) -> ReflectorSlopeVolume:
    return ReflectorSlopeVolume(
        p2=np.full(shape, 0.25, dtype=np.float32),
        p3=np.full(shape, -0.5, dtype=np.float32),
    )


def _input(
    *,
    amplitude: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    surface: FaultSurfaceGraph | None = None,
    reflector_slopes: ReflectorSlopeVolume | None = None,
) -> FaultWarpingInput:
    return FaultWarpingInput(
        amplitude=np.ones(VOLUME_SHAPE, dtype=np.float32) if amplitude is None else amplitude,
        valid_mask=np.ones(VOLUME_SHAPE, dtype=np.bool_) if valid_mask is None else valid_mask,
        surface=_surface() if surface is None else surface,
        reflector_slopes=_slopes() if reflector_slopes is None else reflector_slopes,
    )


def _config_values() -> dict[str, object]:
    return {
        "side_offset_grid": 1.5,
        "window_radius_samples": 2,
        "lag_min_samples": -3,
        "lag_max_samples": 4,
        "max_shift_strain": 0.5,
        "minimum_valid_fraction": 0.75,
    }


def _config(**overrides: object) -> FaultWarpingConfig:
    values = _config_values()
    values.update(overrides)
    return FaultWarpingConfig(**values)  # type: ignore[arg-type]


def _result_values() -> dict[str, np.ndarray]:
    return {
        "valid": np.array([True, False, True, False], dtype=np.bool_),
        "shift_samples": np.array([-2.0, np.nan, 0.5, np.nan], dtype=np.float32),
        "correlation_before": np.array([-0.2, np.nan, 0.1, np.nan], dtype=np.float32),
        "correlation_after": np.array([0.3, np.nan, 0.4, np.nan], dtype=np.float32),
        "cost_margin": np.array([0.0, np.nan, 1.0, np.nan], dtype=np.float32),
        "cycle_residual_samples": np.array([0.0, np.nan, 0.25, np.nan], dtype=np.float32),
        "valid_sample_fraction": np.array([0.5, np.nan, 1.0, np.nan], dtype=np.float32),
        "boundary_hit": np.array([False, False, True, False], dtype=np.bool_),
    }


def _result(**overrides: object) -> FaultWarpingResult:
    values: dict[str, object] = _result_values()
    values.update(overrides)
    return FaultWarpingResult(**values)  # type: ignore[arg-type]


def test_public_contract_metadata_and_dataclass_shape() -> None:
    assert FAULT_WARPING_CONTRACT_VERSION == "pyosv.fault_warping.v1"
    expected_fields = {
        FaultSurfaceGraph: [
            "x1",
            "x2",
            "x3",
            "strike_deg",
            "dip_deg",
            "ca_index",
            "cb_index",
            "cl_index",
            "cr_index",
            "cell_support_weight",
        ],
        ReflectorSlopeVolume: ["p2", "p3"],
        FaultWarpingInput: ["amplitude", "valid_mask", "surface", "reflector_slopes"],
        FaultWarpingConfig: [
            "side_offset_grid",
            "window_radius_samples",
            "lag_min_samples",
            "lag_max_samples",
            "max_shift_strain",
            "minimum_valid_fraction",
            "similarity_metric",
            "subsample_refinement",
        ],
        FaultWarpingResult: [
            "valid",
            "shift_samples",
            "correlation_before",
            "correlation_after",
            "cost_margin",
            "cycle_residual_samples",
            "valid_sample_fraction",
            "boundary_hit",
        ],
    }
    for contract, expected in expected_fields.items():
        assert [field.name for field in fields(contract)] == expected
        assert contract.__dataclass_params__.frozen
        assert not contract.__dataclass_params__.eq
        assert hasattr(contract, "__slots__")

    config_fields = fields(FaultWarpingConfig)
    assert all(field.default is MISSING for field in config_fields[:6])
    assert config_fields[6].default == "zncc"
    assert config_fields[7].default is False


def test_surface_accepts_single_cell_fractional_coordinates_and_support() -> None:
    no_support = _surface(count=1)
    support = np.array([0.75], dtype=np.float32)
    with_support = _surface(count=1, cell_support_weight=support)

    assert no_support.cell_support_weight is None
    assert with_support.cell_support_weight is support
    assert float(no_support.x1[0]) == pytest.approx(1.25)
    with pytest.raises(FrozenInstanceError):
        no_support.x1 = np.array([0.0], dtype=np.float32)  # type: ignore[misc]


def test_surface_accepts_reciprocal_ca_cb_and_cl_cr_links() -> None:
    surface = _surface()

    np.testing.assert_array_equal(surface.ca_index, [1, -1, 3, -1])
    np.testing.assert_array_equal(surface.cb_index, [-1, 0, -1, 2])
    np.testing.assert_array_equal(surface.cl_index, [2, 3, -1, -1])
    np.testing.assert_array_equal(surface.cr_index, [-1, -1, 0, 1])


@pytest.mark.parametrize(
    ("field", "value", "exception"),
    [
        ("x1", [1.0], TypeError),
        ("x2", np.zeros((1, 1), dtype=np.float32), ValueError),
        ("x3", np.zeros(4, dtype=np.float64), TypeError),
        ("ca_index", np.zeros(4, dtype=np.int32), TypeError),
        ("cl_index", np.zeros((4, 1), dtype=np.int64), ValueError),
        ("x1", np.zeros(3, dtype=np.float32), ValueError),
        ("x1", np.array([], dtype=np.float32), ValueError),
    ],
)
def test_surface_rejects_wrong_array_types_dimensions_dtypes_and_lengths(
    field: str,
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        _surface(**{field: value})


@pytest.mark.parametrize("field", ["x1", "x2", "x3", "strike_deg", "dip_deg"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_surface_rejects_nonfinite_float_fields(field: str, nonfinite: float) -> None:
    values = _surface_data()
    values[field][0] = nonfinite
    with pytest.raises(ValueError):
        _surface(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strike_deg", -np.float32(0.01)),
        ("strike_deg", np.float32(360.0)),
        ("dip_deg", np.float32(0.0)),
        ("dip_deg", np.float32(90.01)),
    ],
)
def test_surface_rejects_angle_ranges(field: str, value: np.float32) -> None:
    values = _surface_data()
    values[field][0] = value
    with pytest.raises(ValueError):
        _surface(**values)


@pytest.mark.parametrize(
    ("field", "index", "value"),
    [
        ("ca_index", 0, 4),
        ("cb_index", 0, -2),
        ("cl_index", 0, 0),
        ("cr_index", 1, 1),
    ],
)
def test_surface_rejects_invalid_or_self_topology(
    field: str,
    index: int,
    value: int,
) -> None:
    values = _surface_data()
    values[field][index] = value
    with pytest.raises(ValueError):
        _surface(**values)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("cb_index", np.array([-1, -1, -1, 2], dtype=np.int64)),
        ("ca_index", np.array([-1, -1, 3, -1], dtype=np.int64)),
        ("cr_index", np.array([-1, -1, -1, 1], dtype=np.int64)),
        ("cl_index", np.array([-1, 3, -1, -1], dtype=np.int64)),
    ],
)
def test_surface_rejects_nonreciprocal_topology_in_both_directions(
    field: str,
    replacement: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        _surface(**{field: replacement})


@pytest.mark.parametrize(
    "weight", [np.array([-0.01], dtype=np.float32), np.array([1.01], dtype=np.float32)]
)
def test_surface_rejects_invalid_support_range(weight: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _surface(count=1, cell_support_weight=weight)


@pytest.mark.parametrize(
    ("weight", "exception"),
    [
        ([1.0], TypeError),
        (np.array([1.0], dtype=np.float64), TypeError),
        (np.array([[1.0]], dtype=np.float32), ValueError),
        (np.array([np.nan], dtype=np.float32), ValueError),
        (np.array([np.inf], dtype=np.float32), ValueError),
        (np.array([-np.inf], dtype=np.float32), ValueError),
        (np.array([0.5, 0.5], dtype=np.float32), ValueError),
    ],
)
def test_surface_rejects_invalid_support_type_shape_and_finiteness(
    weight: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        _surface(count=1, cell_support_weight=weight)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("p2", "p3", "exception"),
    [
        ([0.0], np.zeros(VOLUME_SHAPE, dtype=np.float32), TypeError),
        (np.zeros((5, 6), dtype=np.float32), np.zeros(VOLUME_SHAPE, dtype=np.float32), ValueError),
        (
            np.zeros(VOLUME_SHAPE, dtype=np.float64),
            np.zeros(VOLUME_SHAPE, dtype=np.float32),
            TypeError,
        ),
        (
            np.zeros(VOLUME_SHAPE, dtype=np.float32),
            np.zeros((4, 5, 5), dtype=np.float32),
            ValueError,
        ),
    ],
)
def test_reflector_slope_volume_rejects_invalid_arrays(
    p2: object,
    p3: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        ReflectorSlopeVolume(p2=p2, p3=p3)  # type: ignore[arg-type]


def test_input_accepts_valid_volumes_and_preserves_all_caller_arrays() -> None:
    data = _surface_data()
    amplitude = np.arange(np.prod(VOLUME_SHAPE), dtype=np.float32).reshape(VOLUME_SHAPE)
    valid_mask = np.ones(VOLUME_SHAPE, dtype=np.bool_)
    p2 = np.full(VOLUME_SHAPE, 0.25, dtype=np.float32)
    p3 = np.full(VOLUME_SHAPE, -0.5, dtype=np.float32)
    support = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    arrays = [*data.values(), amplitude, valid_mask, p2, p3, support]
    before = [array.copy() for array in arrays]
    for array in (data["x1"], data["ca_index"], amplitude, p2):
        array.flags.writeable = False
    flags = [array.flags.writeable for array in arrays]

    surface = _surface(cell_support_weight=support, **data)
    slopes = ReflectorSlopeVolume(p2=p2, p3=p3)
    inputs = _input(
        amplitude=amplitude,
        valid_mask=valid_mask,
        surface=surface,
        reflector_slopes=slopes,
    )

    assert inputs.amplitude is amplitude
    assert inputs.valid_mask is valid_mask
    assert inputs.surface is surface
    assert inputs.reflector_slopes is slopes
    assert surface.x1 is data["x1"]
    assert surface.ca_index is data["ca_index"]
    assert slopes.p2 is p2
    for original, snapshot, was_writeable in zip(arrays, before, flags):
        np.testing.assert_array_equal(original, snapshot)
        assert original.flags.writeable is was_writeable


def test_input_allows_nonfinite_values_only_where_mask_is_false() -> None:
    amplitude = np.ones(VOLUME_SHAPE, dtype=np.float32)
    p2 = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    p3 = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    valid_mask = np.ones(VOLUME_SHAPE, dtype=np.bool_)
    valid_mask[0, 0, 0] = False
    amplitude[0, 0, 0] = np.nan
    p2[0, 0, 0] = np.inf
    p3[0, 0, 0] = -np.inf

    inputs = _input(
        amplitude=amplitude,
        valid_mask=valid_mask,
        reflector_slopes=ReflectorSlopeVolume(p2=p2, p3=p3),
    )

    assert not inputs.valid_mask[0, 0, 0]


@pytest.mark.parametrize("field", ["amplitude", "p2", "p3"])
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_input_rejects_nonfinite_values_at_valid_voxels(field: str, nonfinite: float) -> None:
    amplitude = np.ones(VOLUME_SHAPE, dtype=np.float32)
    p2 = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    p3 = np.zeros(VOLUME_SHAPE, dtype=np.float32)
    target = {"amplitude": amplitude, "p2": p2, "p3": p3}[field]
    target[0, 0, 0] = nonfinite

    with pytest.raises(ValueError):
        _input(
            amplitude=amplitude,
            reflector_slopes=ReflectorSlopeVolume(p2=p2, p3=p3),
        )


@pytest.mark.parametrize(
    ("amplitude", "valid_mask", "surface", "slopes", "exception"),
    [
        ([1.0], np.ones(VOLUME_SHAPE, dtype=np.bool_), _surface(), _slopes(), TypeError),
        (
            np.ones(VOLUME_SHAPE, dtype=np.float64),
            np.ones(VOLUME_SHAPE, dtype=np.bool_),
            _surface(),
            _slopes(),
            TypeError,
        ),
        (
            np.ones(VOLUME_SHAPE, dtype=np.float32),
            np.ones(VOLUME_SHAPE, dtype=np.uint8),
            _surface(),
            _slopes(),
            TypeError,
        ),
        (
            np.ones(VOLUME_SHAPE, dtype=np.float32),
            np.ones((4, 5, 5), dtype=np.bool_),
            _surface(),
            _slopes(),
            ValueError,
        ),
        (
            np.ones(VOLUME_SHAPE, dtype=np.float32),
            np.ones(VOLUME_SHAPE, dtype=np.bool_),
            {"surface": 1},
            _slopes(),
            TypeError,
        ),
        (
            np.ones(VOLUME_SHAPE, dtype=np.float32),
            np.ones(VOLUME_SHAPE, dtype=np.bool_),
            _surface(),
            {"p2": 0},
            TypeError,
        ),
    ],
)
def test_input_rejects_invalid_volume_arrays_and_transport_types(
    amplitude: object,
    valid_mask: object,
    surface: object,
    slopes: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        FaultWarpingInput(
            amplitude=amplitude,  # type: ignore[arg-type]
            valid_mask=valid_mask,  # type: ignore[arg-type]
            surface=surface,  # type: ignore[arg-type]
            reflector_slopes=slopes,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("coordinate", "value"),
    [
        ("x1", -0.01),
        ("x1", 6.0),
        ("x2", -0.01),
        ("x2", 5.0),
        ("x3", -0.01),
        ("x3", 4.0),
    ],
)
def test_input_rejects_out_of_bounds_surface_coordinates(coordinate: str, value: float) -> None:
    values = _surface_data()
    values[coordinate][0] = value
    with pytest.raises(ValueError):
        _input(surface=_surface(**values))


def test_input_requires_explicit_reflector_slopes() -> None:
    with pytest.raises(TypeError):
        FaultWarpingInput(
            amplitude=np.ones(VOLUME_SHAPE, dtype=np.float32),
            valid_mask=np.ones(VOLUME_SHAPE, dtype=np.bool_),
            surface=_surface(),
        )  # type: ignore[call-arg]


def test_input_rejects_wrong_volume_ndim_and_slope_shape() -> None:
    with pytest.raises(ValueError):
        _input(amplitude=np.ones((5, 6), dtype=np.float32))
    with pytest.raises(ValueError):
        _input(reflector_slopes=_slopes((4, 5, 5)))


def test_config_accepts_explicit_valid_values_and_exact_bool() -> None:
    config = _config(subsample_refinement=True)

    assert config.side_offset_grid == 1.5
    assert config.similarity_metric == "zncc"
    assert config.subsample_refinement is True


@pytest.mark.parametrize("field", ["window_radius_samples", "lag_min_samples", "lag_max_samples"])
@pytest.mark.parametrize("boolean", [True, np.bool_(False)])
def test_config_rejects_bool_for_integer_parameters(field: str, boolean: object) -> None:
    with pytest.raises(TypeError):
        _config(**{field: boolean})


@pytest.mark.parametrize(
    "field", ["side_offset_grid", "max_shift_strain", "minimum_valid_fraction"]
)
def test_config_rejects_bool_for_float_parameters(field: str) -> None:
    with pytest.raises(TypeError):
        _config(**{field: True})


@pytest.mark.parametrize(
    "field", ["side_offset_grid", "max_shift_strain", "minimum_valid_fraction"]
)
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_config_rejects_nonfinite_float_parameters(field: str, nonfinite: float) -> None:
    with pytest.raises(ValueError):
        _config(**{field: nonfinite})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("side_offset_grid", 0.0),
        ("side_offset_grid", -1.0),
        ("window_radius_samples", 0),
        ("window_radius_samples", -1),
        ("max_shift_strain", 0.0),
        ("max_shift_strain", -0.1),
        ("max_shift_strain", 1.1),
        ("minimum_valid_fraction", 0.0),
        ("minimum_valid_fraction", -0.1),
        ("minimum_valid_fraction", 1.1),
    ],
)
def test_config_rejects_out_of_range_parameters(field: str, value: float | int) -> None:
    with pytest.raises(ValueError):
        _config(**{field: value})


@pytest.mark.parametrize(
    ("lag_min_samples", "lag_max_samples"),
    [(-3, -1), (1, 3), (0, 0), (2, 1)],
)
def test_config_requires_zero_in_a_nonempty_lag_range(
    lag_min_samples: int,
    lag_max_samples: int,
) -> None:
    with pytest.raises(ValueError):
        _config(lag_min_samples=lag_min_samples, lag_max_samples=lag_max_samples)


@pytest.mark.parametrize(
    ("overrides", "exception"),
    [
        ({"similarity_metric": "ssd"}, ValueError),
        ({"similarity_metric": 1}, ValueError),
        ({"subsample_refinement": 1}, TypeError),
        ({"subsample_refinement": np.bool_(True)}, TypeError),
    ],
)
def test_config_rejects_unknown_metric_and_non_exact_bool(
    overrides: dict[str, object],
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        _config(**overrides)


def test_result_accepts_valid_and_invalid_rows_and_computes_float32_gain() -> None:
    values = _result_values()
    result = FaultWarpingResult(**values)
    gain = result.correlation_gain

    assert result.valid is values["valid"]
    assert result.shift_samples is values["shift_samples"]
    assert gain.dtype == np.float32
    assert gain.shape == result.valid.shape
    np.testing.assert_allclose(gain[[0, 2]], [0.5, 0.3], rtol=0.0, atol=1e-6)
    assert np.isnan(gain[1])
    assert np.isnan(gain[3])


def test_result_preserves_caller_owned_arrays_and_writeability() -> None:
    values = _result_values()
    arrays = list(values.values())
    before = [array.copy() for array in arrays]
    values["shift_samples"].flags.writeable = False
    values["valid"].flags.writeable = False
    flags = [array.flags.writeable for array in arrays]

    result = FaultWarpingResult(**values)

    for name, original, snapshot, was_writeable in zip(values, arrays, before, flags):
        assert getattr(result, name) is original
        np.testing.assert_array_equal(original, snapshot)
        assert original.flags.writeable is was_writeable


@pytest.mark.parametrize(
    ("field", "replacement", "exception"),
    [
        ("valid", [True], TypeError),
        ("boundary_hit", np.zeros(4, dtype=np.uint8), TypeError),
        ("shift_samples", np.zeros((4, 1), dtype=np.float32), ValueError),
        ("cost_margin", np.zeros(4, dtype=np.float64), TypeError),
        ("correlation_after", np.zeros(3, dtype=np.float32), ValueError),
    ],
)
def test_result_rejects_wrong_array_types_dimensions_dtypes_and_lengths(
    field: str,
    replacement: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        _result(**{field: replacement})


@pytest.mark.parametrize("field", FLOAT_RESULT_FIELDS)
@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
def test_result_rejects_nonfinite_float_values_on_valid_rows(field: str, nonfinite: float) -> None:
    values = _result_values()
    values[field][0] = nonfinite
    with pytest.raises(ValueError):
        FaultWarpingResult(**values)


@pytest.mark.parametrize("field", FLOAT_RESULT_FIELDS)
def test_result_rejects_non_nan_float_values_on_invalid_rows(field: str) -> None:
    values = _result_values()
    values[field][1] = 0.0
    with pytest.raises(ValueError):
        FaultWarpingResult(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("correlation_before", -1.01),
        ("correlation_after", 1.01),
        ("cost_margin", -0.01),
        ("cycle_residual_samples", -0.01),
        ("valid_sample_fraction", -0.01),
        ("valid_sample_fraction", 1.01),
    ],
)
def test_result_rejects_invalid_diagnostic_ranges(field: str, value: float) -> None:
    values = _result_values()
    values[field][0] = value
    with pytest.raises(ValueError):
        FaultWarpingResult(**values)


def test_result_rejects_boundary_hit_on_invalid_row_and_is_frozen() -> None:
    values = _result_values()
    values["boundary_hit"][1] = True
    with pytest.raises(ValueError):
        FaultWarpingResult(**values)

    result = _result()
    with pytest.raises(FrozenInstanceError):
        result.valid = np.array([True], dtype=np.bool_)  # type: ignore[misc]
