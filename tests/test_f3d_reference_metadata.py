from pathlib import Path

import numpy as np
import pytest

from pyosv.f3d_reference import (
    F3D_DTYPE,
    F3D_ENV_VAR,
    F3D_EXPECTED_BYTES,
    F3D_FILENAMES,
    F3D_SHAPE,
    crop_slices,
    f3d_file_paths,
    f3d_file_specs,
    interior_mask,
    interior_slices,
    parse_shape3,
    pick_reference_centers,
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


def test_parse_shape3_accepts_common_separators() -> None:
    assert parse_shape3("420,400,100") == (420, 400, 100)
    assert parse_shape3("420 x 400 x 100") == (420, 400, 100)
    assert parse_shape3("420 400 100") == (420, 400, 100)


@pytest.mark.parametrize("text", ["420,400", "420,0,100", "420,abc,100"])
def test_parse_shape3_rejects_invalid_shapes(text: str) -> None:
    with pytest.raises(ValueError):
        parse_shape3(text)


def test_pick_reference_centers_is_deterministic_and_enforces_separation() -> None:
    fv = np.zeros((5, 5, 5), dtype=np.float32)
    fv[3, 3, 4] = 0.95
    fv[3, 3, 3] = 0.95
    fv[1, 1, 2] = 0.90
    fv[1, 1, 1] = 0.90
    fv[4, 0, 0] = 0.80

    centers = pick_reference_centers(fv, count=3, percentile=0.0, min_separation=2.0)

    assert centers == [(3, 3, 3), (1, 1, 1), (4, 0, 0)]


def test_pick_reference_centers_validates_inputs() -> None:
    with pytest.raises(ValueError, match="3D"):
        pick_reference_centers(np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="percentile"):
        pick_reference_centers(np.zeros((4, 4, 4), dtype=np.float32), percentile=101.0)
    with pytest.raises(ValueError, match="finite"):
        pick_reference_centers(np.full((4, 4, 4), np.nan, dtype=np.float32))


def test_crop_slices_clamps_to_full_shape_bounds() -> None:
    slices = crop_slices((0, 1, 9), (4, 4, 4), full_shape=(10, 10, 10))

    assert slices == (slice(0, 4), slice(0, 4), slice(6, 10))


def test_crop_slices_validates_center_and_crop_size() -> None:
    with pytest.raises(ValueError, match="inside"):
        crop_slices((0, 0, 10), (4, 4, 4), full_shape=(10, 10, 10))
    with pytest.raises(ValueError, match="crop_shape"):
        crop_slices((0, 0, 0), (11, 4, 4), full_shape=(10, 10, 10))


def test_interior_mask_excludes_boundary_margin() -> None:
    mask = interior_mask((5, 6, 7), margin=1)

    assert mask.dtype == bool
    assert mask.shape == (5, 6, 7)
    assert mask.sum() == 3 * 4 * 5
    assert not mask[0].any()
    assert not mask[:, 0, :].any()
    assert not mask[:, :, 0].any()
    assert mask[1:-1, 1:-1, 1:-1].all()


def test_interior_slices_excludes_boundary_margin() -> None:
    assert interior_slices((5, 6, 7), margin=1) == (
        slice(1, 4),
        slice(1, 5),
        slice(1, 6),
    )


def test_interior_mask_margin_zero_returns_all_true() -> None:
    assert interior_mask((2, 3, 4), margin=0).all()
    assert interior_slices((2, 3, 4), margin=0) == (
        slice(0, 2),
        slice(0, 3),
        slice(0, 4),
    )


def test_interior_mask_rejects_empty_interior() -> None:
    with pytest.raises(ValueError, match="too large"):
        interior_mask((4, 5, 6), margin=2)
    with pytest.raises(ValueError, match="too large"):
        interior_slices((4, 5, 6), margin=2)
