"""File contract for the minimal Q-QUAL 3D output bundle."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import scipy

import pyosv

from .profile import QQual3DProfile
from .runner import QQual3DResult

_STORAGE_DTYPE = ">f4"
_RUN_SCHEMA = "pyosv.qqual3d.run/v1"
_SKINS_SCHEMA = "pyosv.qqual3d.skins/v1"


@dataclass(frozen=True, slots=True)
class LoadedQQual3DInput:
    """One validated big-endian float32 input and its content identity."""

    array: np.ndarray
    filename: str
    size: int
    sha256: str


def _shape_tuple(shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) != 3 or any(isinstance(size, bool) or int(size) <= 0 for size in shape):
        raise ValueError("shape must contain three positive integers")
    return tuple(int(size) for size in shape)


def load_qqual3d_input(path: Path, shape: Sequence[int]) -> LoadedQQual3DInput:
    """Read a regular, non-symlink ``>f4`` C-order volume."""

    source = Path(path)
    volume_shape = _shape_tuple(shape)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"input must be a readable regular non-symlink file: {source}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"input must be a regular file: {source}")
        expected_size = (
            int(np.prod(volume_shape, dtype=np.int64)) * np.dtype(_STORAGE_DTYPE).itemsize
        )
        if metadata.st_size != expected_size:
            raise ValueError(
                f"input byte size {metadata.st_size} does not match shape {volume_shape} "
                f"({expected_size} bytes)"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        if len(payload) != expected_size:
            raise ValueError("input changed size while it was being read")
    finally:
        os.close(descriptor)

    array = np.frombuffer(payload, dtype=_STORAGE_DTYPE).reshape(volume_shape, order="C")
    native = np.array(array, dtype=np.float32, copy=True, order="C")
    if not np.all(np.isfinite(native)):
        raise ValueError("input must contain only finite float32 values")
    return LoadedQQual3DInput(
        array=native,
        filename=source.name,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def require_new_output_directory(path: Path) -> None:
    """Reject an output path that already names any filesystem entry."""

    destination = Path(path)
    if os.path.lexists(destination):
        raise FileExistsError(f"output directory already exists: {destination}")
    if not destination.parent.is_dir():
        raise ValueError(f"output directory parent does not exist: {destination.parent}")


def _json_bytes(value: object, *, pretty: bool) -> bytes:
    options: dict[str, Any] = {"allow_nan": False, "sort_keys": True}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(value, **options) + "\n").encode("utf-8")


def _write_dat(path: Path, value: np.ndarray) -> None:
    path.write_bytes(np.asarray(value, dtype=_STORAGE_DTYPE, order="C").tobytes(order="C"))


def _cell_payload(cell: object) -> dict[str, object]:
    support = getattr(cell, "reskin_support", None)
    return {
        "x1": float(getattr(cell, "x1")),
        "x2": float(getattr(cell, "x2")),
        "x3": float(getattr(cell, "x3")),
        "i1": int(getattr(cell, "i1")),
        "i2": int(getattr(cell, "i2")),
        "i3": int(getattr(cell, "i3")),
        "fl": float(getattr(cell, "fl")),
        "fp": float(getattr(cell, "fp")),
        "ft": float(getattr(cell, "ft")),
        "generation": getattr(cell, "generation", None),
        "reskin_support": None if support is None else float(support),
    }


def _skins_payload(skins: Sequence[Any]) -> dict[str, object]:
    serialized = []
    for skin_index, skin in enumerate(skins):
        cells = sorted(tuple(skin), key=lambda cell: (int(cell.i3), int(cell.i2), int(cell.i1)))
        serialized.append(
            {
                "skin_index": skin_index,
                "cell_count": len(cells),
                "cells": [_cell_payload(cell) for cell in cells],
            }
        )
    return {
        "schema": _SKINS_SCHEMA,
        "skinning_enabled": True,
        "skin_count": len(serialized),
        "skins": serialized,
    }


def _profile_payload(profile: QQual3DProfile) -> dict[str, object]:
    return {
        "scanner": {
            "backend": profile.scanner_backend,
            "angular_range_degrees": {
                "phi": [profile.phi_min, profile.phi_max],
                "theta": [profile.theta_min, profile.theta_max],
            },
            "sigmas": [profile.sigma1, profile.sigma2],
            "refinement_factor": profile.scanner_refinement_factor,
            "thinning_mode": profile.scanner_thin_mode,
            "edge_cleanup": profile.remove_edge_effects,
        },
        "voting": asdict(profile.voting_config),
        "voting_controls": asdict(profile.voting_controls),
        "skinning": asdict(profile.skinning_config),
        "workflow_mode": profile.workflow_mode,
        "variant": asdict(profile.variant),
        "skinning_enabled": profile.skinning_enabled,
    }


def _numba_identity() -> dict[str, object]:
    try:
        installed_version = version("numba")
    except PackageNotFoundError:
        return {"available": False, "version": None}
    return {"available": True, "version": installed_version}


def _software_payload() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "pyosv": pyosv.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": _numba_identity(),
    }


def _created_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _artifact_record(
    path: Path, *, role: str, shape: tuple[int, int, int] | None
) -> dict[str, object]:
    payload = path.read_bytes()
    record: dict[str, object] = {
        "role": role,
        "filename": path.name,
    }
    if shape is None:
        record["json_role"] = "fault_skins"
    else:
        record["shape"] = list(shape)
        record["storage_dtype"] = _STORAGE_DTYPE
    record["size"] = len(payload)
    record["sha256"] = hashlib.sha256(payload).hexdigest()
    return record


def write_qqual3d_output_bundle(
    output_dir: Path,
    *,
    source: LoadedQQual3DInput,
    result: QQual3DResult,
    pretty: bool = False,
) -> Path:
    """Atomically create the minimal hashed Q-QUAL output directory."""

    destination = Path(output_dir)
    require_new_output_directory(destination)
    shape = result.profile.shape
    if tuple(source.array.shape) != shape:
        raise ValueError("source and result shapes do not match")

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        dat_outputs = (
            ("ft.dat", "scanner_likelihood", result.ft),
            ("fv.dat", "voted_likelihood", result.fv),
            ("fvt.dat", "thinned_voted_likelihood", result.fvt),
        )
        output_records = []
        for filename, role, array in dat_outputs:
            artifact = temporary / filename
            _write_dat(artifact, array)
            output_records.append(_artifact_record(artifact, role=role, shape=shape))

        if result.profile.skinning_enabled:
            mask_path = temporary / "skin_mask.dat"
            _write_dat(mask_path, result.skin_mask)
            output_records.append(_artifact_record(mask_path, role="skin_mask", shape=shape))
            skins_path = temporary / "skins.json"
            skins_path.write_bytes(_json_bytes(_skins_payload(result.skins), pretty=False))
            output_records.append(_artifact_record(skins_path, role="fault_skins", shape=None))

        manifest = {
            "schema": _RUN_SCHEMA,
            "created_at_utc": _created_at_utc(),
            "input": {
                "filename": source.filename,
                "shape": list(shape),
                "storage_dtype": _STORAGE_DTYPE,
                "size": source.size,
                "sha256": source.sha256,
            },
            "profile": _profile_payload(result.profile),
            "software": _software_payload(),
            "outputs": output_records,
        }
        (temporary / "run.json").write_bytes(_json_bytes(manifest, pretty=pretty))
        require_new_output_directory(destination)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
