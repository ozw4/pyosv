import numpy as np
import pytest
from scipy import ndimage

from pyosv.orient3d import FaultOrientScanner3
from pyosv.thinning3d import (
    reference_like_3d_nms_mask,
    reference_like_3d_thin_values,
    remove_reference_edge_effects_3d,
)
from pyosv.voting3d import OptimalSurfaceVoter


def test_reference_like_3d_nms_mask_validates_matching_3d_shapes() -> None:
    values = np.zeros((3, 4, 2), dtype=np.float32)
    strike = np.zeros((3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="3D array"):
        reference_like_3d_nms_mask(values, strike)

    with pytest.raises(ValueError, match="shapes must match"):
        reference_like_3d_nms_mask(values, np.zeros((3, 5, 2), dtype=np.float32))


@pytest.mark.parametrize(
    ("values", "strike", "message"),
    [
        (
            np.array([[[0.0, np.nan]]], dtype=np.float32),
            np.zeros((1, 1, 2), dtype=np.float32),
            "values",
        ),
        (
            np.zeros((1, 1, 2), dtype=np.float32),
            np.array([[[0.0, np.inf]]], dtype=np.float32),
            "strike",
        ),
        (
            np.array([[["bad"]]], dtype=object),
            np.zeros((1, 1, 1), dtype=np.float32),
            "numeric finite",
        ),
    ],
)
def test_reference_like_3d_nms_mask_rejects_non_finite_inputs(
    values: np.ndarray,
    strike: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reference_like_3d_nms_mask(values, strike)


def test_reference_like_3d_nms_mask_preserves_inputs_and_returns_bool_shape() -> None:
    values = np.zeros((5, 5, 2), dtype=np.float32)
    strike = np.full_like(values, 30.0)
    values[2, 2, 1] = 1.0
    values_before = values.copy()
    strike_before = strike.copy()

    mask = reference_like_3d_nms_mask(values, strike)

    assert mask.dtype == np.bool_
    assert mask.shape == values.shape
    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(strike, strike_before)


def test_reference_like_3d_nms_mask_horizontal_bin_keeps_i2_maximum() -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.zeros_like(values)
    values[2, 2, 0] = 3.0
    values[2, 1, 0] = 1.0
    values[2, 3, 0] = 2.0

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert mask[2, 2, 0]
    assert not mask[2, 1, 0]
    assert not mask[2, 3, 0]


def test_reference_like_3d_nms_mask_positive_diagonal_bin_keeps_diagonal_maximum() -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, 45.0)
    values[2, 2, 0] = 3.0
    values[1, 3, 0] = 1.0
    values[3, 1, 0] = 2.0

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert mask[2, 2, 0]
    assert not mask[1, 3, 0]
    assert not mask[3, 1, 0]


def test_reference_like_3d_nms_mask_45_degrees_uses_reference_diagonal() -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, 45.0)
    values[2, 2, 0] = 3.0
    values[3, 1, 0] = 2.0
    values[1, 3, 0] = 1.0
    values[3, 3, 0] = 4.0

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert mask[2, 2, 0]


def test_reference_like_3d_nms_mask_vertical_bin_keeps_i3_maximum() -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, 90.0)
    values[2, 2, 0] = 3.0
    values[1, 2, 0] = 1.0
    values[3, 2, 0] = 2.0

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert mask[2, 2, 0]
    assert not mask[1, 2, 0]
    assert not mask[3, 2, 0]


def test_reference_like_3d_nms_mask_negative_diagonal_bin_keeps_diagonal_maximum() -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, 135.0)
    values[2, 2, 0] = 3.0
    values[1, 1, 0] = 1.0
    values[3, 3, 0] = 2.0

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert mask[2, 2, 0]
    assert not mask[1, 1, 0]
    assert not mask[3, 3, 0]


def test_reference_like_3d_nms_mask_135_degrees_uses_reference_diagonal() -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, 135.0)
    values[2, 2, 0] = 3.0
    values[3, 3, 0] = 2.0
    values[1, 1, 0] = 1.0
    values[3, 1, 0] = 4.0

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert mask[2, 2, 0]


@pytest.mark.parametrize("strike_value", [0.0, 45.0, 90.0, 135.0])
def test_reference_like_3d_nms_mask_does_not_retain_boundary_samples(
    strike_value: float,
) -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, strike_value)
    values[0, 0, 0] = 10.0

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert not mask[0, 0, 0]


def test_reference_like_3d_nms_mask_constant_volume_has_no_strict_maxima() -> None:
    values = np.ones((5, 5, 2), dtype=np.float32)
    strike = np.full_like(values, 45.0)

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0)

    assert not mask.any()


def test_reference_like_3d_nms_mask_non_strict_allows_flat_interior() -> None:
    values = np.ones((3, 3, 1), dtype=np.float32)
    strike = np.full_like(values, 90.0)

    mask = reference_like_3d_nms_mask(values, strike, sigma=0.0, strict=False)

    assert mask[1, 1, 0]
    assert mask.sum() == 3


def test_reference_like_3d_thin_values_writes_smoothed_retained_values() -> None:
    values = np.zeros((7, 7, 1), dtype=np.float32)
    strike = np.zeros_like(values)
    values[3, 3, 0] = 10.0
    sigma = 1.0

    thinned, keep = reference_like_3d_thin_values(values, strike, sigma=sigma)
    expected = ndimage.gaussian_filter(
        values,
        sigma=(sigma, sigma, 0.0),
        mode="nearest",
    ).astype(np.float32, copy=False)

    assert keep[3, 3, 0]
    assert thinned[3, 3, 0] == pytest.approx(float(expected[3, 3, 0]))
    assert thinned[3, 3, 0] != np.float32(10.0)


@pytest.mark.parametrize("strike_value", [90.0, 270.0])
def test_reference_like_3d_thin_values_reinforces_folded_vertical_strike_neighbor(
    strike_value: float,
) -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, strike_value)
    values[2, 2, 0] = 3.0

    thinned, keep = reference_like_3d_thin_values(
        values,
        strike,
        sigma=0.0,
        reinforce_vertical=True,
    )

    assert keep[2, 2, 0]
    assert thinned[2, 2, 0] == np.float32(3.0)
    assert thinned[1, 2, 0] == np.float32(3.0)


@pytest.mark.parametrize("strike_value", [60.0, 120.0, 240.0, 300.0])
def test_reference_like_3d_thin_values_reinforcement_uses_strict_folded_boundaries(
    strike_value: float,
) -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, strike_value)
    values[2, 2, 0] = 3.0

    thinned, keep = reference_like_3d_thin_values(
        values,
        strike,
        sigma=0.0,
        reinforce_vertical=True,
    )

    assert keep[2, 2, 0]
    assert thinned[2, 2, 0] == np.float32(3.0)
    assert thinned[1, 2, 0] == np.float32(0.0)


@pytest.mark.parametrize("strike_value", [90.0, 270.0])
def test_reference_like_3d_thin_values_without_reinforcement_keeps_only_mask(
    strike_value: float,
) -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, strike_value)
    values[2, 2, 0] = 3.0

    thinned, keep = reference_like_3d_thin_values(
        values,
        strike,
        sigma=0.0,
        reinforce_vertical=False,
    )

    assert keep[2, 2, 0]
    assert thinned[2, 2, 0] == np.float32(3.0)
    assert thinned[1, 2, 0] == np.float32(0.0)


def test_remove_reference_edge_effects_3d_removes_boundary_only_i3_ridge() -> None:
    values = np.zeros((12, 12, 2), dtype=np.float32)
    strike = np.full_like(values, 90.0)
    dip = np.full_like(values, 90.0)
    values[1, :, :] = 2.0

    cleaned, cleaned_strike, cleaned_dip, keep = remove_reference_edge_effects_3d(
        values,
        strike,
        dip,
    )

    assert cleaned.dtype == np.float32
    assert cleaned_strike.dtype == np.float32
    assert cleaned_dip.dtype == np.float32
    assert keep.dtype == np.bool_
    assert not keep.any()
    assert not cleaned.any()
    assert not cleaned_strike.any()
    assert not cleaned_dip.any()


def test_remove_reference_edge_effects_3d_preserves_interior_ridge() -> None:
    values = np.zeros((12, 12, 1), dtype=np.float32)
    strike = np.full_like(values, 90.0)
    dip = np.full_like(values, 90.0)
    values[6, 6, 0] = 3.0

    cleaned, cleaned_strike, cleaned_dip, keep = remove_reference_edge_effects_3d(
        values,
        strike,
        dip,
    )

    assert keep[6, 6, 0]
    assert cleaned[6, 6, 0] == np.float32(3.0)
    assert cleaned_strike[6, 6, 0] == np.float32(90.0)
    assert cleaned_dip[6, 6, 0] == np.float32(90.0)
    assert np.count_nonzero(cleaned) == 1


@pytest.mark.parametrize("strike_value", [90.0, 270.0])
def test_remove_reference_edge_effects_3d_uses_folded_strike_for_i3_faces(
    strike_value: float,
) -> None:
    values = np.zeros((12, 12, 1), dtype=np.float32)
    strike = np.full_like(values, strike_value)
    dip = np.full_like(values, 90.0)
    values[1, 6, 0] = 3.0

    cleaned, _, _, keep = remove_reference_edge_effects_3d(values, strike, dip)

    assert not keep[1, 6, 0]
    assert cleaned[1, 6, 0] == np.float32(0.0)


def test_remove_reference_edge_effects_3d_keeps_non_parallel_i3_face_sample() -> None:
    values = np.zeros((12, 12, 1), dtype=np.float32)
    strike = np.zeros_like(values)
    dip = np.full_like(values, 90.0)
    values[1, 6, 0] = 3.0

    cleaned, _, _, keep = remove_reference_edge_effects_3d(values, strike, dip)

    assert keep[1, 6, 0]
    assert cleaned[1, 6, 0] == np.float32(3.0)


@pytest.mark.parametrize("strike_value", [0.0, 180.0])
def test_remove_reference_edge_effects_3d_removes_i2_face_parallel_samples(
    strike_value: float,
) -> None:
    values = np.zeros((12, 12, 1), dtype=np.float32)
    strike = np.full_like(values, strike_value)
    dip = np.full_like(values, 90.0)
    values[6, 1, 0] = 3.0

    cleaned, _, _, keep = remove_reference_edge_effects_3d(values, strike, dip)

    assert not keep[6, 1, 0]
    assert cleaned[6, 1, 0] == np.float32(0.0)


def test_remove_reference_edge_effects_3d_does_not_modify_inputs() -> None:
    values = np.zeros((12, 12, 1), dtype=np.float32)
    strike = np.full_like(values, 90.0)
    dip = np.full_like(values, 90.0)
    values[1, 6, 0] = 3.0
    values_before = values.copy()
    strike_before = strike.copy()
    dip_before = dip.copy()

    remove_reference_edge_effects_3d(values, strike, dip)

    np.testing.assert_array_equal(values, values_before)
    np.testing.assert_array_equal(strike, strike_before)
    np.testing.assert_array_equal(dip, dip_before)


def test_remove_reference_edge_effects_3d_empty_volume_is_safe_noop() -> None:
    values = np.zeros((0, 3, 2), dtype=np.float32)
    strike = np.zeros_like(values)
    dip = np.zeros_like(values)

    cleaned, cleaned_strike, cleaned_dip, keep = remove_reference_edge_effects_3d(
        values,
        strike,
        dip,
    )

    assert cleaned.shape == values.shape
    assert cleaned_strike.shape == values.shape
    assert cleaned_dip.shape == values.shape
    assert keep.shape == values.shape
    assert cleaned.dtype == np.float32
    assert keep.dtype == np.bool_


def test_remove_reference_edge_effects_3d_small_volume_does_not_crash() -> None:
    values = np.ones((3, 3, 1), dtype=np.float32)
    strike = np.full_like(values, 45.0)
    dip = np.zeros_like(values)

    cleaned, cleaned_strike, cleaned_dip, keep = remove_reference_edge_effects_3d(
        values,
        strike,
        dip,
    )

    np.testing.assert_array_equal(cleaned, values)
    np.testing.assert_array_equal(cleaned_strike, strike)
    np.testing.assert_array_equal(cleaned_dip, dip)
    assert keep.all()


def test_remove_reference_edge_effects_3d_validates_like_thinning_helper() -> None:
    values = np.zeros((3, 4, 2), dtype=np.float32)
    strike = np.zeros((3, 4), dtype=np.float32)
    dip = np.zeros_like(values)

    with pytest.raises(ValueError, match="3D array"):
        remove_reference_edge_effects_3d(values, strike, dip)

    with pytest.raises(ValueError, match="shapes must match"):
        remove_reference_edge_effects_3d(values, np.zeros((3, 5, 2), dtype=np.float32), dip)


def test_optimal_surface_voter_reference_thin_reinforces_folded_vertical_strike() -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, 270.0)
    dip = np.zeros_like(values)
    values[2, 2, 0] = 3.0

    thinned = OptimalSurfaceVoter(1, 1, 1).thin(
        values,
        strike,
        dip,
        mode="reference",
        reference_sigma=0.0,
    )

    assert thinned[2, 2, 0] == np.float32(3.0)
    assert thinned[1, 2, 0] == np.float32(3.0)


def test_fault_orient_scanner_reference_thin_does_not_reinforce_folded_vertical_strike() -> None:
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, 270.0)
    dip = np.zeros_like(values)
    values[2, 2, 0] = 3.0

    thinned, thinned_strike, thinned_dip = FaultOrientScanner3(1.0, 1.0).thin(
        values,
        strike,
        dip,
        mode="reference",
        reference_sigma=0.0,
    )

    assert thinned[2, 2, 0] == np.float32(3.0)
    assert thinned[1, 2, 0] == np.float32(0.0)
    assert thinned_strike[1, 2, 0] == np.float32(0.0)
    assert thinned_dip[1, 2, 0] == np.float32(0.0)


def test_fault_orient_scanner_reference_thin_edge_cleanup_is_explicitly_diagnostic() -> None:
    values = np.zeros((12, 12, 1), dtype=np.float32)
    strike = np.full_like(values, 90.0)
    dip = np.full_like(values, 90.0)
    values[1, 6, 0] = 3.0
    scanner = FaultOrientScanner3(1.0, 1.0)

    cleaned, _, _ = scanner.thin(
        values,
        strike,
        dip,
        mode="reference",
        reference_sigma=0.0,
    )
    retained, retained_strike, retained_dip = scanner.thin(
        values,
        strike,
        dip,
        mode="reference",
        reference_sigma=0.0,
        remove_edge_effects=False,
    )

    assert cleaned[1, 6, 0] == np.float32(0.0)
    assert retained[1, 6, 0] == np.float32(3.0)
    assert retained_strike[1, 6, 0] == np.float32(90.0)
    assert retained_dip[1, 6, 0] == np.float32(90.0)


def test_optimal_surface_voter_default_reference_thin_differs_from_normal_mode() -> None:
    values = np.zeros((5, 5, 5), dtype=np.float32)
    strike = np.full_like(values, 90.0)
    dip = np.zeros_like(values)
    values[2, 2, 2] = 3.0
    voter = OptimalSurfaceVoter(1, 1, 1)

    reference = voter.thin(values, strike, dip, reference_sigma=0.0)
    normal = voter.thin(values, strike, dip, mode="normal", reference_sigma=0.0)

    assert reference[2, 2, 2] == np.float32(3.0)
    assert reference[1, 2, 2] == np.float32(3.0)
    assert normal[2, 2, 2] == np.float32(3.0)
    assert normal[1, 2, 2] == np.float32(0.0)


@pytest.mark.parametrize(
    ("strike_value", "voter_neighbor", "scanner_neighbor"),
    [
        (60.0, 0.0, 0.0),
        (90.0, 3.0, 0.0),
        (120.0, 0.0, 0.0),
    ],
)
def test_reference_thin_wrappers_audit_strict_voter_reinforcement_only(
    strike_value: float,
    voter_neighbor: float,
    scanner_neighbor: float,
) -> None:
    # Audits OptimalSurfaceVoter.thin reference semantics separately from scanner thinning.
    values = np.zeros((5, 5, 1), dtype=np.float32)
    strike = np.full_like(values, strike_value)
    dip = np.zeros_like(values)
    values[2, 2, 0] = 3.0

    voter_thinned = OptimalSurfaceVoter(1, 1, 1).thin(
        values,
        strike,
        dip,
        mode="reference",
        reference_sigma=0.0,
    )
    scanner_thinned, scanner_strike, scanner_dip = FaultOrientScanner3(1.0, 1.0).thin(
        values,
        strike,
        dip,
        mode="reference",
        reference_sigma=0.0,
    )

    assert voter_thinned[2, 2, 0] == np.float32(3.0)
    assert scanner_thinned[2, 2, 0] == np.float32(3.0)
    assert voter_thinned[1, 2, 0] == np.float32(voter_neighbor)
    assert scanner_thinned[1, 2, 0] == np.float32(scanner_neighbor)
    assert scanner_strike[1, 2, 0] == np.float32(0.0)
    assert scanner_dip[1, 2, 0] == np.float32(0.0)
