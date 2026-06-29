from pathlib import Path

import numpy as np
import pytest

from pyosv.io import read_dat
from pyosv.reference import REFERENCE_DATASETS_2D, reference_root, resolve_reference_file

REQUIRED_REFERENCE_FILES = {
    "f3d2d": ("ft.dat", "pt.dat", "fv.dat", "fvt.dat"),
    "campos": ("ft.dat", "pt.dat", "fv.dat", "fvt.dat"),
}


def require_reference_root() -> Path:
    root = reference_root()
    if not root.exists():
        pytest.skip(f"reference_osv mount not available: {root}")
    return root


def _required_reference_cases() -> list[tuple[str, str]]:
    return [
        (dataset_name, file_name)
        for dataset_name, file_names in REQUIRED_REFERENCE_FILES.items()
        for file_name in file_names
    ]


@pytest.mark.parametrize(("dataset_name", "file_name"), _required_reference_cases())
def test_required_reference_dat_file_can_be_read(dataset_name: str, file_name: str) -> None:
    root = require_reference_root()
    dataset = REFERENCE_DATASETS_2D[dataset_name]
    path = resolve_reference_file(dataset, file_name, root=root)
    if not path.exists():
        pytest.skip(f"reference fixture not available: {path}")

    assert path.stat().st_size == dataset.sample_count * 4

    data = read_dat(path, dataset.shape, endian=dataset.endian)

    assert data.shape == dataset.shape
    assert data.dtype == np.float32
    assert np.isfinite(data).all()

    if file_name in {"fv.dat", "fvt.dat"}:
        assert data.min() >= -1e-6
        assert data.max() <= 1.0 + 1e-6

    if file_name in {"ft.dat", "pt.dat"}:
        assert np.any(data != 0.0)
