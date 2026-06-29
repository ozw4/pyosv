from pathlib import Path

import pytest

from pyosv.reference import (
    REFERENCE_DATASETS_2D,
    reference_root,
    resolve_reference_file,
)


def test_f3d2d_metadata_shape() -> None:
    assert REFERENCE_DATASETS_2D["f3d2d"].shape == (440, 222)


def test_campos_metadata_shape() -> None:
    assert REFERENCE_DATASETS_2D["campos"].shape == (550, 300)


@pytest.mark.parametrize("dataset_name", ["f3d2d", "campos"])
def test_2d_sample_count_matches_n1_times_n2(dataset_name: str) -> None:
    dataset = REFERENCE_DATASETS_2D[dataset_name]

    assert dataset.n2 is not None
    assert dataset.sample_count == dataset.n1 * dataset.n2


@pytest.mark.parametrize("dataset_name", ["f3d2d", "campos"])
def test_2d_metadata_ndim(dataset_name: str) -> None:
    assert REFERENCE_DATASETS_2D[dataset_name].ndim == 2


def test_resolve_reference_file_constructs_path_without_requiring_file(tmp_path: Path) -> None:
    path = resolve_reference_file("f3d2d", "ft.dat", root=tmp_path)

    assert path == tmp_path / "data/2d/f3d2d" / "ft.dat"


def test_reference_root_uses_explicit_root(tmp_path: Path) -> None:
    assert reference_root(root=tmp_path) == tmp_path


def test_reference_root_uses_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYOSV_REFERENCE_OSV", "/tmp/reference-osv")

    assert reference_root() == Path("/tmp/reference-osv")


def test_unknown_dataset_name_raises_clear_key_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown reference dataset: missing"):
        resolve_reference_file("missing", "ft.dat", root=tmp_path)
