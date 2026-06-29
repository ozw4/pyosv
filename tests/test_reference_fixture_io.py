import numpy as np
import pytest

from pyosv.reference import REFERENCE_DATASETS_2D

from reference_fixtures import load_reference_2d_array, require_reference_2d_file

REQUIRED_REFERENCE_FILES = {
    "f3d2d": ("ft.dat", "pt.dat", "fv.dat", "fvt.dat"),
    "campos": ("ft.dat", "pt.dat", "fv.dat", "fvt.dat"),
}


def _required_reference_cases() -> list[tuple[str, str]]:
    return [
        (dataset_name, file_name)
        for dataset_name, file_names in REQUIRED_REFERENCE_FILES.items()
        for file_name in file_names
    ]


@pytest.mark.parametrize(("dataset_name", "file_name"), _required_reference_cases())
def test_required_reference_dat_file_can_be_read(dataset_name: str, file_name: str) -> None:
    dataset = REFERENCE_DATASETS_2D[dataset_name]
    path = require_reference_2d_file(dataset, file_name)

    assert path.stat().st_size == dataset.sample_count * 4

    data = load_reference_2d_array(dataset_name, file_name)

    assert data.shape == dataset.shape
    assert data.dtype == np.float32
    assert np.isfinite(data).all()

    if file_name in {"fv.dat", "fvt.dat"}:
        assert data.min() >= -1e-6
        assert data.max() <= 1.0 + 1e-6

    if file_name in {"ft.dat", "pt.dat"}:
        assert np.any(data != 0.0)
