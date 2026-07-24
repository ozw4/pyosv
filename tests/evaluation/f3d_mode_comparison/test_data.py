from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from pyosv.evaluation.f3d_mode_comparison import (
    F3DatasetSpec,
    F3VolumeSource,
    OFFICIAL_F3_DATASET_SPEC,
    ensure_output_not_in_data_root,
)
from pyosv.evaluation.f3d_mode_comparison import data as data_module
from pyosv.f3d_reference import (
    F3D_DTYPE,
    F3D_EXPECTED_BYTES,
    F3D_SHAPE,
)
from pyosv.io import write_dat


def _fixture_spec(
    shape: tuple[int, int, int] = (2, 3, 4),
) -> F3DatasetSpec:
    return F3DatasetSpec(
        dataset_id="test-fixture",
        shape=shape,
        files={"input": "ep.dat"},
        expected_bytes=int(np.prod(shape)) * np.dtype(">f4").itemsize,
    )


def _write_fixture(root: Path, spec: F3DatasetSpec) -> np.ndarray:
    values = np.arange(np.prod(spec.shape), dtype=np.float32).reshape(spec.shape)
    write_dat(root / "ep.dat", values, endian="big")
    return values


def test_read_only_storage_memmap_and_native_copy(tmp_path: Path) -> None:
    spec = _fixture_spec()
    expected = _write_fixture(tmp_path, spec)

    with F3VolumeSource(tmp_path, spec=spec) as source:
        storage = source.open_memmap("input")
        native = source.read_native_volume("input")

        assert storage.shape == spec.shape
        assert storage.dtype == np.dtype(">f4")
        assert storage.flags.c_contiguous
        assert not storage.flags.writeable
        assert native.dtype == np.dtype("=f4")
        assert native.flags.c_contiguous
        assert not native.flags.writeable
        np.testing.assert_array_equal(native, expected)
        with pytest.raises(ValueError):
            storage[0, 0, 0] = 10.0

    assert source.closed
    with pytest.raises(RuntimeError, match="closed"):
        source.open_memmap("input")


def test_checksum_is_streamed_and_matches_known_bytes(tmp_path: Path) -> None:
    sample_count = data_module.SHA256_BUFFER_SIZE // 4 + 1
    spec = _fixture_spec((1, 1, sample_count))
    _write_fixture(tmp_path, spec)
    expected = hashlib.sha256((tmp_path / "ep.dat").read_bytes()).hexdigest()

    source = F3VolumeSource(tmp_path, spec=spec)

    assert source.identity.file_for("input").sha256 == expected
    source.close()


def test_missing_wrong_size_nonregular_and_unknown_role_are_rejected(
    tmp_path: Path,
) -> None:
    spec = _fixture_spec()
    with pytest.raises(FileNotFoundError, match="ep.dat"):
        F3VolumeSource(tmp_path, spec=spec)

    (tmp_path / "ep.dat").write_bytes(b"short")
    with pytest.raises(ValueError, match="expected .* bytes"):
        F3VolumeSource(tmp_path, spec=spec)

    (tmp_path / "ep.dat").unlink()
    (tmp_path / "ep.dat").mkdir()
    with pytest.raises(ValueError, match="regular file"):
        F3VolumeSource(tmp_path, spec=spec)

    (tmp_path / "ep.dat").rmdir()
    _write_fixture(tmp_path, spec)
    with F3VolumeSource(tmp_path, spec=spec) as source:
        with pytest.raises(ValueError, match="unknown F3 file role"):
            source.open_memmap("missing")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is unavailable")
def test_fifo_is_rejected_without_opening_it(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "ep.dat")

    with pytest.raises(ValueError, match="regular file"):
        F3VolumeSource(tmp_path, spec=_fixture_spec())


def test_checksum_rejects_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _fixture_spec()
    _write_fixture(tmp_path, spec)
    stream_sha256 = data_module._stream_sha256

    def mutate_after_hash(
        source: object,
        *,
        buffer_size: int = data_module.SHA256_BUFFER_SIZE,
    ) -> str:
        result = stream_sha256(source, buffer_size=buffer_size)  # type: ignore[arg-type]
        path = tmp_path / "ep.dat"
        current = path.stat()
        os.utime(path, ns=(current.st_atime_ns, current.st_mtime_ns + 1))
        return result

    monkeypatch.setattr(data_module, "_stream_sha256", mutate_after_hash)

    with pytest.raises(ValueError, match="changed while computing checksum"):
        F3VolumeSource(tmp_path, spec=spec)


def test_content_identity_excludes_provenance_path_and_changes_with_bytes(
    tmp_path: Path,
) -> None:
    spec = _fixture_spec()
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    expected = _write_fixture(first_root, spec)
    write_dat(second_root / "ep.dat", expected, endian="big")

    first = F3VolumeSource(first_root, spec=spec)
    second = F3VolumeSource(second_root, spec=spec)
    assert first.identity == second.identity
    assert first.identity.sha256 == second.identity.sha256
    assert first.identity.data_root != second.identity.data_root
    assert (
        first.identity.file_for("input").resolved_path
        != second.identity.file_for("input").resolved_path
    )

    changed = expected.copy()
    changed.flat[0] += 1.0
    write_dat(second_root / "ep.dat", changed, endian="big")
    changed_source = F3VolumeSource(second_root, spec=spec)
    assert first.identity.sha256 != changed_source.identity.sha256

    first.close()
    second.close()
    changed_source.close()


def test_content_identity_excludes_provenance_filename(tmp_path: Path) -> None:
    first_spec = _fixture_spec()
    second_spec = F3DatasetSpec(
        dataset_id=first_spec.dataset_id,
        shape=first_spec.shape,
        files={"input": "renamed.dat"},
        expected_bytes=first_spec.expected_bytes,
    )
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    expected = _write_fixture(first_root, first_spec)
    write_dat(second_root / "renamed.dat", expected, endian="big")

    with (
        F3VolumeSource(first_root, spec=first_spec) as first,
        F3VolumeSource(second_root, spec=second_spec) as second,
    ):
        assert first.identity == second.identity
        assert first.identity.computation_identity == second.identity.computation_identity
        assert first.identity.sha256 == second.identity.sha256
        assert (
            first.identity.file_for("input").as_manifest_dict()["filename"]
            != second.identity.file_for("input").as_manifest_dict()["filename"]
        )


def test_context_releases_memmap_file_handle(tmp_path: Path) -> None:
    spec = _fixture_spec()
    _write_fixture(tmp_path, spec)
    path = tmp_path / "ep.dat"

    with F3VolumeSource(tmp_path, spec=spec) as source:
        source.open_memmap("input")

    renamed = tmp_path / "renamed.dat"
    path.rename(renamed)
    renamed.unlink()
    assert not renamed.exists()


def test_open_memmap_rejects_target_replaced_while_opening(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _fixture_spec()
    _write_fixture(tmp_path, spec)
    source = F3VolumeSource(tmp_path, spec=spec)
    source_path = tmp_path / "ep.dat"
    replacement_path = tmp_path / "replacement.dat"
    replacement = np.full(spec.shape, 99.0, dtype=np.float32)
    write_dat(replacement_path, replacement, endian="big")
    original_open = Path.open
    replaced = False

    def replace_before_open(path: Path, *args: object, **kwargs: object):
        nonlocal replaced
        if path == source_path and not replaced:
            replaced = True
            os.replace(replacement_path, source_path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", replace_before_open)

    with pytest.raises(ValueError, match="changed after identity validation"):
        source.open_memmap("input")
    source.close()


def test_output_workspace_must_be_outside_data_root(tmp_path: Path) -> None:
    data_root = tmp_path / "f3"

    with pytest.raises(ValueError, match="F3 data root"):
        ensure_output_not_in_data_root(data_root, data_root)
    with pytest.raises(ValueError, match="F3 data root"):
        ensure_output_not_in_data_root(data_root / "outputs", data_root)
    assert (
        ensure_output_not_in_data_root(tmp_path / "outputs", data_root)
        == (tmp_path / "outputs").resolve()
    )


def test_official_spec_matches_reference_metadata() -> None:
    spec = OFFICIAL_F3_DATASET_SPEC

    assert spec.dataset_id
    assert spec.shape == F3D_SHAPE
    assert spec.storage_dtype == F3D_DTYPE
    assert spec.expected_bytes == F3D_EXPECTED_BYTES
    assert spec.required_files == ("ep.dat", "fl.dat", "fv.dat", "fvt.dat")


@pytest.mark.parametrize(
    "kwargs",
    (
        {"storage_dtype": "<f4"},
        {"shape": (2, 3, 5)},
        {"files": {"input": "../ep.dat"}},
    ),
)
def test_custom_spec_rejects_dtype_layout_or_unsafe_filename(
    kwargs: dict[str, object],
) -> None:
    defaults: dict[str, object] = {
        "dataset_id": "bad-fixture",
        "shape": (2, 3, 4),
        "files": {"input": "ep.dat"},
        "expected_bytes": 96,
    }
    defaults.update(kwargs)

    with pytest.raises(ValueError):
        F3DatasetSpec(**defaults)  # type: ignore[arg-type]
