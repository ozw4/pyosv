import numpy as np
import pytest

from pyosv.voting3d import OptimalSurfaceVoter


def test_constructor_initializes_range_and_default_configuration() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    assert voter.ru == 3
    assert voter.rv == 2
    assert voter.rw == 2
    assert voter.lmin == -3
    assert voter.lmax == 3
    assert voter.nl == 7
    assert voter.bstrain1 == 4
    assert voter.bstrain2 == 4
    assert voter.attribute_smoothing == 1
    assert voter.surface_smoothing1 == 2.0
    assert voter.surface_smoothing2 == 2.0
    np.testing.assert_array_equal(
        voter.lmins,
        np.array(
            [
                [-3, -2, 0, -2, -3],
                [-2, 0, 0, 0, -2],
                [0, 0, 0, 0, 0],
                [-2, 0, 0, 0, -2],
                [-3, -2, 0, -2, -3],
            ],
            dtype=np.int32,
        ),
    )
    np.testing.assert_array_equal(voter.lmaxs, -voter.lmins)


def test_shift_range_arrays_match_surface_radius_shape() -> None:
    voter = OptimalSurfaceVoter(ru=5, rv=6, rw=7)

    assert voter.lmins.shape == (2 * voter.rw + 1, 2 * voter.rv + 1)
    assert voter.lmaxs.shape == (2 * voter.rw + 1, 2 * voter.rv + 1)


@pytest.mark.parametrize(
    ("ru", "rv", "rw"),
    [
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
        (1.5, 0, 0),
        (0, True, 0),
        (0, 0, "1"),
    ],
)
def test_constructor_rejects_invalid_radii(ru: object, rv: object, rw: object) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        OptimalSurfaceVoter(ru, rv, rw)  # type: ignore[arg-type]


def test_set_strain_max_updates_only_bstrain_spacing() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)
    lmins_before = voter.lmins.copy()
    lmaxs_before = voter.lmaxs.copy()

    voter.set_strain_max(1.0, 0.5)

    assert voter.bstrain1 == 1
    assert voter.bstrain2 == 2
    np.testing.assert_array_equal(voter.lmins, lmins_before)
    np.testing.assert_array_equal(voter.lmaxs, lmaxs_before)


def test_set_strain_max_keeps_default_bstrain_spacing() -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    voter.set_strain_max(0.25, 0.25)

    assert voter.bstrain1 == 4
    assert voter.bstrain2 == 4


@pytest.mark.parametrize("strain_max", [0.0, -0.25, 1.25, np.nan, np.inf])
def test_set_strain_max_rejects_invalid_first_strain(strain_max: float) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="0 < strain_max <= 1"):
        voter.set_strain_max(strain_max, 0.25)


@pytest.mark.parametrize("strain_max", [0.0, -0.25, 1.25, np.nan, np.inf])
def test_set_strain_max_rejects_invalid_second_strain(strain_max: float) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="0 < strain_max <= 1"):
        voter.set_strain_max(0.25, strain_max)


@pytest.mark.parametrize("attribute_smoothing", [0, 1, np.int32(2)])
def test_set_attribute_smoothing_accepts_nonnegative_integers(
    attribute_smoothing: int,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    voter.set_attribute_smoothing(attribute_smoothing)

    assert voter.attribute_smoothing == int(attribute_smoothing)


@pytest.mark.parametrize("attribute_smoothing", [-1, 1.5, True, "1"])
def test_set_attribute_smoothing_rejects_invalid_values(
    attribute_smoothing: object,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="attribute_smoothing"):
        voter.set_attribute_smoothing(attribute_smoothing)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("surface_smoothing1", "surface_smoothing2"),
    [
        (0.0, 0.0),
        (1.25, 0.5),
        (np.float32(2.0), np.float32(3.0)),
    ],
)
def test_set_surface_smoothing_accepts_nonnegative_finite_numbers(
    surface_smoothing1: float,
    surface_smoothing2: float,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    voter.set_surface_smoothing(surface_smoothing1, surface_smoothing2)

    assert voter.surface_smoothing1 == float(surface_smoothing1)
    assert voter.surface_smoothing2 == float(surface_smoothing2)


@pytest.mark.parametrize("surface_smoothing", [-0.1, np.nan, np.inf, True, "1"])
def test_set_surface_smoothing_rejects_invalid_first_value(
    surface_smoothing: object,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="surface_smoothing1"):
        voter.set_surface_smoothing(surface_smoothing, 0.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("surface_smoothing", [-0.1, np.nan, np.inf, True, "1"])
def test_set_surface_smoothing_rejects_invalid_second_value(
    surface_smoothing: object,
) -> None:
    voter = OptimalSurfaceVoter(ru=3, rv=2, rw=2)

    with pytest.raises(ValueError, match="surface_smoothing2"):
        voter.set_surface_smoothing(0.0, surface_smoothing)  # type: ignore[arg-type]
