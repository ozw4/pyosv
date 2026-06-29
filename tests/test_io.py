from pathlib import Path

import numpy as np
import pytest

from pyosv.io import read_dat, write_dat


def _write_raw(path: Path, array: np.ndarray, dtype: str) -> None:
    np.asarray(array, dtype=np.dtype(dtype)).tofile(path)


def test_read_big_endian_float32_known_array(tmp_path: Path) -> None:
    path = tmp_path / "big.dat"
    expected = np.array([[1.25, -2.5, 3.75], [4.5, 0.0, -6.25]], dtype=np.float32)
    _write_raw(path, expected, ">f4")

    actual = read_dat(path, (2, 3))

    np.testing.assert_allclose(actual, expected)


def test_read_little_endian_float32_known_array(tmp_path: Path) -> None:
    path = tmp_path / "little.dat"
    expected = np.array([[1.25, -2.5], [3.75, 4.5], [0.0, -6.25]], dtype=np.float32)
    _write_raw(path, expected, "<f4")

    actual = read_dat(path, (3, 2), endian="little")

    np.testing.assert_allclose(actual, expected)


def test_round_trip_write_read_2d(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "roundtrip_2d.dat"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4) - 5.0

    written_path = write_dat(path, expected)
    actual = read_dat(written_path, (3, 4))

    assert written_path == path
    np.testing.assert_allclose(actual, expected)


def test_round_trip_write_read_3d(tmp_path: Path) -> None:
    path = tmp_path / "roundtrip_3d.dat"
    expected = (np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 3.0) - 2.0

    write_dat(path, expected, endian="<")
    actual = read_dat(path, (2, 3, 4), endian="little")

    np.testing.assert_allclose(actual, expected)


def test_default_read_returns_finite_c_contiguous_float32(tmp_path: Path) -> None:
    path = tmp_path / "defaults.dat"
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    _write_raw(path, expected, ">f4")

    actual = read_dat(path, (2, 2))

    assert np.isfinite(actual).all()
    assert actual.flags.c_contiguous
    assert actual.dtype == np.float32


def test_shape_mismatch_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "mismatch.dat"
    _write_raw(path, np.arange(4, dtype=np.float32), ">f4")

    with pytest.raises(ValueError) as error:
        read_dat(path, (3,))

    message = str(error.value)
    assert str(path) in message
    assert "expected 12 bytes" in message
    assert "got 16 bytes" in message


@pytest.mark.parametrize("shape", [(), (0, 3), (-1, 3)])
def test_invalid_shape_raises_value_error(tmp_path: Path, shape: tuple[int, ...]) -> None:
    path = tmp_path / "shape.dat"
    _write_raw(path, np.arange(6, dtype=np.float32), ">f4")

    with pytest.raises(ValueError, match="shape"):
        read_dat(path, shape)


def test_invalid_endian_raises_value_error(tmp_path: Path) -> None:
    path = tmp_path / "endian.dat"
    _write_raw(path, np.arange(2, dtype=np.float32), ">f4")

    with pytest.raises(ValueError, match="endian"):
        read_dat(path, (2,), endian="middle")  # type: ignore[arg-type]
