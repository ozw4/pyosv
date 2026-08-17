from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import pyosv.qqual3d.io as io_module
from pyosv.qqual3d import run_qqual3d
from pyosv.qqual3d.io import (
    load_qqual3d_input,
    require_new_output_directory,
    write_qqual3d_output_bundle,
)


def _write_input(path: Path, array: np.ndarray) -> bytes:
    payload = np.asarray(array, dtype=">f4").tobytes(order="C")
    path.write_bytes(payload)
    return payload


def test_load_big_endian_input_preserves_content_identity(tmp_path: Path) -> None:
    shape = (2, 3, 4)
    expected = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    path = tmp_path / "ep.dat"
    payload = _write_input(path, expected)
    before = path.stat()

    loaded = load_qqual3d_input(path, shape)

    np.testing.assert_array_equal(loaded.array, expected)
    assert loaded.array.dtype == np.dtype(np.float32)
    assert loaded.array.dtype.isnative
    assert loaded.array.flags.owndata
    assert loaded.filename == "ep.dat"
    assert loaded.size == len(payload)
    assert loaded.sha256 == hashlib.sha256(payload).hexdigest()
    after = path.stat()
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)


def test_load_rejects_wrong_size_and_symlink(tmp_path: Path) -> None:
    path = tmp_path / "ep.dat"
    path.write_bytes(b"too short")

    with pytest.raises(ValueError, match="byte size"):
        load_qqual3d_input(path, (2, 3, 4))

    target = tmp_path / "target.dat"
    _write_input(target, np.zeros((1, 2, 3), dtype=np.float32))
    link = tmp_path / "link.dat"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="non-symlink"):
        load_qqual3d_input(link, (1, 2, 3))


def test_output_bundle_has_big_endian_hashed_artifacts(tmp_path: Path) -> None:
    shape = (2, 3, 4)
    input_path = tmp_path / "ep.dat"
    _write_input(input_path, np.zeros(shape, dtype=np.float32))
    source = load_qqual3d_input(input_path, shape)
    result = run_qqual3d(source.array)
    output = tmp_path / "bundle"

    assert write_qqual3d_output_bundle(output, source=source, result=result) == output

    assert {path.name for path in output.iterdir()} == {
        "run.json",
        "ft.dat",
        "fv.dat",
        "fvt.dat",
        "skin_mask.dat",
        "skins.json",
    }
    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "pyosv.qqual3d.run/v1"
    assert manifest["input"] == {
        "filename": "ep.dat",
        "shape": list(shape),
        "storage_dtype": ">f4",
        "size": input_path.stat().st_size,
        "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    }
    assert "absolute" not in json.dumps(manifest["input"])
    scanner = manifest["profile"]["scanner"]
    assert scanner["backend"] == "quality"
    assert scanner["orientation_backend"] == "rotate_shear"
    assert scanner["interpolation_backend"] == "scipy"
    assert scanner["interpolation_order"] == 1
    assert scanner["smoothing_sigma"] is None
    assert scanner["normalize"] is True
    assert scanner["output_dtype"] == "float32"
    assert scanner["reference_thin_sigma"] == 1.0
    assert manifest["profile"]["workflow_mode"] == "quality"
    assert manifest["profile"]["voting"]["voter_thin_mode"] == "hybrid_v2"
    assert manifest["profile"]["skinning_enabled"] is True
    assert set(manifest["software"]) == {"python", "pyosv", "numpy", "scipy", "numba"}

    records = {record["filename"]: record for record in manifest["outputs"]}
    assert set(records) == {"ft.dat", "fv.dat", "fvt.dat", "skin_mask.dat", "skins.json"}
    for filename, record in records.items():
        payload = (output / filename).read_bytes()
        assert record["size"] == len(payload)
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    for filename in ("ft.dat", "fv.dat", "fvt.dat", "skin_mask.dat"):
        assert records[filename]["shape"] == list(shape)
        assert records[filename]["storage_dtype"] == ">f4"
        stored = np.fromfile(output / filename, dtype=">f4").reshape(shape)
        assert stored.dtype == np.dtype(">f4")
    skins = json.loads((output / "skins.json").read_text(encoding="utf-8"))
    assert skins == {
        "schema": "pyosv.qqual3d.skins/v1",
        "skinning_enabled": True,
        "skin_count": 0,
        "skins": [],
    }


def test_existing_output_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "bundle"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        require_new_output_directory(output)


def test_write_failure_removes_private_temporary_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (2, 3, 4)
    input_path = tmp_path / "ep.dat"
    _write_input(input_path, np.zeros(shape, dtype=np.float32))
    source = load_qqual3d_input(input_path, shape)
    result = run_qqual3d(source.array)
    output = tmp_path / "bundle"

    def fail_write(path: Path, value: np.ndarray) -> None:
        raise OSError("injected write failure")

    monkeypatch.setattr(io_module, "_write_dat", fail_write)
    with pytest.raises(OSError, match="injected"):
        write_qqual3d_output_bundle(output, source=source, result=result)

    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".bundle.tmp-*"))
