import numpy as np
import pytest

from pyosv.voting2d import OptimalPathVoter


def test_constructor_initializes_range_and_default_configuration() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    assert voter.ru == 3
    assert voter.rv == 4
    assert voter.lmin == -3
    assert voter.lmax == 3
    assert voter.nl == 7
    assert voter.bstrain1 == 4
    assert voter.attribute_smoothing == 1
    assert voter.path_smoothing1 == 2.0
    np.testing.assert_array_equal(
        voter.lmins,
        np.array([-3, -3, 0, 0, 0, 0, 0, -3, -3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        voter.lmaxs,
        np.array([3, 3, 0, 0, 0, 0, 0, 3, 3], dtype=np.int32),
    )


def test_shift_range_arrays_match_strike_radius_shape() -> None:
    voter = OptimalPathVoter(ru=5, rv=6)

    assert voter.lmins.shape == (2 * voter.rv + 1,)
    assert voter.lmaxs.shape == (2 * voter.rv + 1,)


def test_set_strain_max_updates_bstrain_spacing() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)
    voter.lmins = np.full_like(voter.lmins, 99)
    voter.lmaxs = np.full_like(voter.lmaxs, 99)

    voter.set_strain_max(1.0)

    assert voter.bstrain1 == 1
    np.testing.assert_array_equal(
        voter.lmins,
        np.array([-3, -3, -2, -1, 0, -1, -2, -3, -3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        voter.lmaxs,
        np.array([3, 3, 2, 1, 0, 1, 2, 3, 3], dtype=np.int32),
    )


def test_set_strain_max_keeps_default_bstrain_spacing() -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    voter.set_strain_max(0.25)

    assert voter.bstrain1 == 4
    np.testing.assert_array_equal(
        voter.lmins,
        np.array([-3, -3, 0, 0, 0, 0, 0, -3, -3], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        voter.lmaxs,
        np.array([3, 3, 0, 0, 0, 0, 0, 3, 3], dtype=np.int32),
    )


@pytest.mark.parametrize("strain_max", [0.0, -0.25, 1.25, np.nan, np.inf])
def test_set_strain_max_rejects_invalid_values(strain_max: float) -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    with pytest.raises(ValueError, match="0 < strain_max <= 1"):
        voter.set_strain_max(strain_max)


@pytest.mark.parametrize("attribute_smoothing", [0, 1, np.int32(2)])
def test_set_attribute_smoothing_accepts_nonnegative_integers(
    attribute_smoothing: int,
) -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    voter.set_attribute_smoothing(attribute_smoothing)

    assert voter.attribute_smoothing == int(attribute_smoothing)


@pytest.mark.parametrize("attribute_smoothing", [-1, 1.5, True, "1"])
def test_set_attribute_smoothing_rejects_invalid_values(
    attribute_smoothing: object,
) -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    with pytest.raises(ValueError, match="attribute_smoothing"):
        voter.set_attribute_smoothing(attribute_smoothing)  # type: ignore[arg-type]


@pytest.mark.parametrize("path_smoothing1", [0.0, 1.25, np.float32(2.0)])
def test_set_path_smoothing_accepts_nonnegative_finite_numbers(
    path_smoothing1: float,
) -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    voter.set_path_smoothing(path_smoothing1)

    assert voter.path_smoothing1 == float(path_smoothing1)


@pytest.mark.parametrize("path_smoothing1", [-0.1, np.nan, np.inf, True, "1"])
def test_set_path_smoothing_rejects_invalid_values(path_smoothing1: object) -> None:
    voter = OptimalPathVoter(ru=3, rv=4)

    with pytest.raises(ValueError, match="path_smoothing1"):
        voter.set_path_smoothing(path_smoothing1)  # type: ignore[arg-type]
