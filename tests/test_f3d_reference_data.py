from pathlib import Path

import numpy as np
import pytest

from pyosv.f3d_reference import (
    F3D_ENV_VAR,
    F3D_FILENAMES,
    F3D_SHAPE,
    f3d_file_paths,
    f3d_file_specs,
    read_f3d_file,
    resolve_f3d_data_root,
)


def _f3d_data_root() -> Path:
    try:
        root = resolve_f3d_data_root()
    except ValueError:
        pytest.skip(f"{F3D_ENV_VAR} is not set")

    if not root.is_dir():
        pytest.skip(f"{F3D_ENV_VAR} does not point to an existing directory: {root}")

    return root


def _nonzero_count(data: np.ndarray) -> int:
    return int(np.count_nonzero(np.abs(data) > 1.0e-6))


def test_f3d_reference_files_exist_and_have_expected_size() -> None:
    root = _f3d_data_root()
    paths = f3d_file_paths(root)
    specs = f3d_file_specs()

    for filename in F3D_FILENAMES:
        path = paths[filename]
        assert path.is_file(), f"missing F3 reference file: {path}"
        assert path.stat().st_size == specs[filename].expected_bytes


@pytest.mark.parametrize("filename", F3D_FILENAMES)
def test_f3d_reference_file_reads_as_finite_volume(filename: str) -> None:
    root = _f3d_data_root()

    data = read_f3d_file(filename, root)
    data_min = float(data.min())
    data_max = float(data.max())
    data_mean = float(data.mean())
    nonzero = _nonzero_count(data)

    print(
        f"{filename}: shape={data.shape} min={data_min:.6g} "
        f"max={data_max:.6g} mean={data_mean:.6g} nonzero={nonzero}"
    )

    assert data.shape == F3D_SHAPE
    assert data.dtype == np.float32
    assert np.isfinite(data).all()

    if filename in {"ep.dat", "fl.dat", "fv.dat"}:
        assert data_min >= -1.0e-6
        assert data_max <= 1.0 + 1.0e-6

    if filename == "fvt.dat":
        assert data_min >= -1.0e-3
        assert data_max <= 1.01


def test_f3d_thinned_votes_are_sparser_than_votes() -> None:
    root = _f3d_data_root()

    fv_nonzero = _nonzero_count(read_f3d_file("fv.dat", root))
    fvt_nonzero = _nonzero_count(read_f3d_file("fvt.dat", root))

    print(f"fv.dat nonzero={fv_nonzero}; fvt.dat nonzero={fvt_nonzero}")

    assert fvt_nonzero < fv_nonzero
