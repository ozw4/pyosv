from pathlib import Path

import pytest

from pyosv.f3d_reference import (
    F3D_DTYPE,
    F3D_ENV_VAR,
    F3D_EXPECTED_BYTES,
    F3D_FILENAMES,
    F3D_SHAPE,
    f3d_file_paths,
    f3d_file_specs,
    read_f3d_file,
    resolve_f3d_data_root,
)


def test_resolve_f3d_data_root_uses_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(F3D_ENV_VAR, "/tmp/f3d-reference")

    assert resolve_f3d_data_root() == Path("/tmp/f3d-reference")


def test_resolve_f3d_data_root_explicit_path_overrides_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(F3D_ENV_VAR, "/tmp/f3d-reference")

    assert resolve_f3d_data_root(tmp_path) == tmp_path


def test_resolve_f3d_data_root_missing_root_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(F3D_ENV_VAR, raising=False)

    with pytest.raises(ValueError, match=F3D_ENV_VAR):
        resolve_f3d_data_root()


def test_f3d_metadata_constants() -> None:
    assert F3D_SHAPE == (420, 400, 100)
    assert F3D_DTYPE == ">f4"
    assert F3D_EXPECTED_BYTES == 67_200_000
    assert F3D_FILENAMES == ("xs.dat", "ep.dat", "fl.dat", "fv.dat", "fvt.dat")


def test_f3d_file_specs_include_expected_files() -> None:
    specs = f3d_file_specs()

    assert tuple(specs) == F3D_FILENAMES
    for filename in F3D_FILENAMES:
        assert specs[filename].filename == filename
        assert specs[filename].expected_bytes == F3D_EXPECTED_BYTES
        assert specs[filename].meaning


def test_f3d_file_paths_returns_known_paths(tmp_path: Path) -> None:
    paths = f3d_file_paths(tmp_path)

    assert tuple(paths) == F3D_FILENAMES
    assert paths["xs.dat"] == tmp_path / "xs.dat"
    assert paths["ep.dat"] == tmp_path / "ep.dat"
    assert paths["fl.dat"] == tmp_path / "fl.dat"
    assert paths["fv.dat"] == tmp_path / "fv.dat"
    assert paths["fvt.dat"] == tmp_path / "fvt.dat"


def test_read_f3d_file_unknown_name_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown F3 3D reference file: missing.dat"):
        read_f3d_file("missing.dat", tmp_path)
