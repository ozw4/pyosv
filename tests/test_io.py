from pathlib import Path

import numpy as np
import pytest

from pyosv.io import open_dat_memmap, read_dat, read_dat_region, write_dat


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


@pytest.mark.parametrize(
    ("shape", "endian"),
    [((3, 5), "big"), ((2, 3, 5), "little")],
)
def test_memmap_preserves_shape_endian_and_is_read_only(
    tmp_path: Path, shape: tuple[int, ...], endian: str
) -> None:
    path = tmp_path / f"memmap-{endian}.dat"
    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    storage_dtype = ">f4" if endian == "big" else "<f4"
    _write_raw(path, expected, storage_dtype)

    actual = open_dat_memmap(path, shape, endian=endian)  # type: ignore[arg-type]

    assert isinstance(actual, np.memmap)
    assert actual.shape == shape
    assert actual.dtype == np.dtype(storage_dtype)
    assert not actual.flags.writeable
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("shape", "region", "endian"),
    [
        ((3, 5), (slice(None), slice(None)), "big"),
        ((2, 3, 5), (slice(None), slice(None), slice(None)), "little"),
    ],
)
def test_full_region_exactly_matches_read_dat(
    tmp_path: Path,
    shape: tuple[int, ...],
    region: tuple[slice, ...],
    endian: str,
) -> None:
    path = tmp_path / f"full-{endian}.dat"
    expected = (np.arange(np.prod(shape), dtype=np.float32) - 7).reshape(shape)
    storage_dtype = ">f4" if endian == "big" else "<f4"
    _write_raw(path, expected, storage_dtype)

    actual = read_dat_region(
        path,
        shape,
        region,
        endian=endian,  # type: ignore[arg-type]
    )

    np.testing.assert_array_equal(actual, read_dat(path, shape, endian=endian))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("region", "expected_region"),
    [
        ((slice(1, 3), slice(2, 5), slice(1, 6)), np.s_[1:3, 2:5, 1:6]),
        ((slice(0, 1), slice(0, 1), slice(0, 1)), np.s_[0:1, 0:1, 0:1]),
        ((slice(-1, None), slice(-2, None), slice(-3, None)), np.s_[-1:, -2:, -3:]),
    ],
)
def test_read_dat_region_crops_non_cubic_volume(
    tmp_path: Path,
    region: tuple[slice, ...],
    expected_region: tuple[slice, ...],
) -> None:
    path = tmp_path / "crop.dat"
    expected = np.arange(4 * 6 * 8, dtype=np.float32).reshape(4, 6, 8)
    _write_raw(path, expected, ">f4")

    actual = read_dat_region(path, expected.shape, region)

    np.testing.assert_array_equal(actual, expected[expected_region])
    assert not isinstance(actual, np.memmap)
    assert actual.dtype == np.dtype(np.float32).newbyteorder("=")
    assert actual.flags.c_contiguous
    assert actual.flags.writeable
    actual[...] = -1
    np.testing.assert_array_equal(read_dat(path, expected.shape), expected)


@pytest.mark.parametrize(
    ("region", "message"),
    [
        ((slice(None),), "one slice per shape axis"),
        ((slice(None), slice(None), slice(None)), "one slice per shape axis"),
        ((slice(None), 1), "one slice per shape axis"),
        ((slice(None, None, 2), slice(None)), "step"),
        ((slice(2, 2), slice(None)), "non-empty"),
        ((slice(2, 1), slice(None)), "non-empty"),
        ((slice(-4, 1), slice(None)), "outside"),
        ((slice(None), slice(0, 6)), "outside"),
    ],
)
def test_read_dat_region_rejects_invalid_region(
    tmp_path: Path, region: tuple[slice, ...], message: str
) -> None:
    path = tmp_path / "region.dat"
    _write_raw(path, np.arange(15, dtype=np.float32).reshape(3, 5), ">f4")

    with pytest.raises(ValueError, match=message):
        read_dat_region(path, (3, 5), region)


@pytest.mark.parametrize("function", [open_dat_memmap, read_dat_region])
def test_new_read_apis_validate_shape_endian_and_exact_file_size(
    tmp_path: Path, function: object
) -> None:
    path = tmp_path / "invalid.dat"
    _write_raw(path, np.arange(4, dtype=np.float32), ">f4")
    args = ((slice(None),),) if function is read_dat_region else ()

    with pytest.raises(ValueError, match="shape"):
        function(path, (-1,), *args)  # type: ignore[operator]
    with pytest.raises(ValueError, match="endian"):
        function(path, (4,), *args, endian="middle")  # type: ignore[operator]
    with pytest.raises(ValueError, match="expected 20 bytes.*got 16 bytes"):
        function(path, (5,), *args)  # type: ignore[operator]


def test_read_dat_region_does_not_use_fromfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "no-full-read.dat"
    expected = np.arange(24, dtype=np.float32).reshape(4, 6)
    _write_raw(path, expected, ">f4")

    def fail_fromfile(*args: object, **kwargs: object) -> None:
        raise AssertionError("read_dat_region must not call np.fromfile")

    monkeypatch.setattr(np, "fromfile", fail_fromfile)

    actual = read_dat_region(path, expected.shape, np.s_[1:3, 2:5])

    np.testing.assert_array_equal(actual, expected[1:3, 2:5])
