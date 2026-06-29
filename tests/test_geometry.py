import numpy as np

from pyosv.geometry import range180, range360


def test_range360_scalar_boundaries() -> None:
    assert range360(0.0) == 0.0
    assert range360(360.0) == 0.0
    assert range360(-1.0) == 359.0
    assert range360(720.0) == 0.0


def test_range180_scalar_boundaries() -> None:
    assert range180(180.0) == 180.0
    assert range180(-180.0) == -180.0
    assert range180(540.0) == 180.0
    assert range180(-540.0) == -180.0
    assert range180(181.0) == -179.0
    assert range180(-181.0) == 179.0


def test_scalar_inputs_return_python_float() -> None:
    assert isinstance(range360(np.float32(-1.0)), float)
    assert isinstance(range180(np.float32(181.0)), float)


def test_range360_array_input() -> None:
    angles = np.array([-720.0, -1.0, 0.0, 360.0, 721.0], dtype=np.float32)

    wrapped = range360(angles)

    assert isinstance(wrapped, np.ndarray)
    np.testing.assert_allclose(wrapped, np.array([0.0, 359.0, 0.0, 0.0, 1.0]))


def test_range180_array_input() -> None:
    angles = np.array([-540.0, -181.0, -180.0, 180.0, 181.0, 540.0], dtype=np.float32)

    wrapped = range180(angles)

    assert isinstance(wrapped, np.ndarray)
    np.testing.assert_allclose(wrapped, np.array([-180.0, 179.0, -180.0, 180.0, -179.0, 180.0]))


def test_large_finite_angles_remain_finite() -> None:
    angles = np.array([-1.0e6, 1.0e6, -123456.0, 123456.0], dtype=np.float32)

    wrapped360 = range360(angles)
    wrapped180 = range180(angles)

    assert np.isfinite(wrapped360).all()
    assert np.isfinite(wrapped180).all()
    assert ((0.0 <= wrapped360) & (wrapped360 < 360.0)).all()
    assert ((-180.0 <= wrapped180) & (wrapped180 <= 180.0)).all()
