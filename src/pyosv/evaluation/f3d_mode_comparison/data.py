"""Immutable dataset identities and read-only access for the official F3 data."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import prod
from os import PathLike
from pathlib import Path
from typing import BinaryIO

import numpy as np

from pyosv.f3d_reference import F3D_DTYPE, F3D_EXPECTED_BYTES, F3D_SHAPE

F3_DATASET_ID = "f3d-official-v1"
F3_FILE_ROLES = (
    ("input", "ep.dat"),
    ("reference_fault_likelihood", "fl.dat"),
    ("reference_fault_votes", "fv.dat"),
    ("reference_thinned_fault_votes", "fvt.dat"),
)
SHA256_BUFFER_SIZE = 1024 * 1024


def _validated_shape(shape: object) -> tuple[int, int, int]:
    if not isinstance(shape, tuple) or len(shape) != 3:
        raise ValueError("shape must contain exactly 3 dimensions")
    if any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape):
        raise ValueError("shape must contain exactly 3 positive integers")
    return shape


def _validated_files(files: object) -> tuple[tuple[str, str], ...]:
    items = files.items() if isinstance(files, Mapping) else files
    try:
        result = tuple((role, filename) for role, filename in items)
    except (TypeError, ValueError) as error:
        raise ValueError("files must map semantic roles to filenames") from error
    if not result:
        raise ValueError("files must contain at least one required file")

    roles: set[str] = set()
    filenames: set[str] = set()
    for role, filename in result:
        if not isinstance(role, str) or not role:
            raise ValueError("file roles must be non-empty strings")
        if not isinstance(filename, str) or not filename:
            raise ValueError("filenames must be non-empty strings")
        if Path(filename).name != filename:
            raise ValueError("filenames must not contain directory components")
        if role in roles:
            raise ValueError(f"duplicate file role: {role}")
        if filename in filenames:
            raise ValueError(f"duplicate required filename: {filename}")
        roles.add(role)
        filenames.add(filename)
    return result


@dataclass(frozen=True, slots=True)
class F3DatasetSpec:
    """Storage contract for an F3-shaped collection of raw DAT volumes.

    Non-official instances are accepted only as a low-level fixture/testing
    facility. Canonical plan builders always use :data:`OFFICIAL_F3_DATASET_SPEC`.
    """

    dataset_id: str = F3_DATASET_ID
    shape: tuple[int, int, int] = F3D_SHAPE
    storage_dtype: str = F3D_DTYPE
    files: tuple[tuple[str, str], ...] = F3_FILE_ROLES
    expected_bytes: int = F3D_EXPECTED_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        shape = _validated_shape(self.shape)
        try:
            storage_dtype = np.dtype(self.storage_dtype)
        except TypeError as error:
            raise ValueError("storage_dtype must be big-endian float32") from error
        if storage_dtype.str != F3D_DTYPE:
            raise ValueError(f"storage_dtype must be {F3D_DTYPE!r}")
        files = _validated_files(self.files)
        if (
            isinstance(self.expected_bytes, bool)
            or not isinstance(self.expected_bytes, int)
            or self.expected_bytes <= 0
        ):
            raise ValueError("expected_bytes must be a positive integer")
        size_from_layout = prod(shape) * storage_dtype.itemsize
        if self.expected_bytes != size_from_layout:
            raise ValueError(
                "expected_bytes does not match shape and storage_dtype: "
                f"expected {size_from_layout}, got {self.expected_bytes}"
            )
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "storage_dtype", storage_dtype.str)
        object.__setattr__(self, "files", files)

    @property
    def roles(self) -> tuple[str, ...]:
        """Return semantic roles in their stable manifest order."""

        return tuple(role for role, _ in self.files)

    @property
    def required_files(self) -> tuple[str, ...]:
        """Return required filenames in their stable manifest order."""

        return tuple(filename for _, filename in self.files)

    @property
    def input_file(self) -> str:
        """Return the filename assigned to the canonical input role."""

        return self.filename_for("input")

    @property
    def dtype(self) -> str:
        """Return the storage dtype (the plan-model spelling)."""

        return self.storage_dtype

    def filename_for(self, role: str) -> str:
        """Resolve one semantic role to its required filename."""

        for known_role, filename in self.files:
            if role == known_role:
                return filename
        expected = ", ".join(self.roles)
        raise ValueError(f"unknown F3 file role: {role!r}; expected one of {expected}")


OFFICIAL_F3_DATASET_SPEC = F3DatasetSpec()


@dataclass(frozen=True, slots=True)
class F3FileIdentity:
    """Content identity and provenance for one validated source volume."""

    role: str
    filename: str = field(compare=False)
    resolved_path: Path = field(compare=False)
    size: int
    sha256: str
    shape: tuple[int, int, int]
    storage_dtype: str

    @property
    def computation_identity(self) -> dict[str, object]:
        """Return path-independent fields suitable for cache keys."""

        return {
            "role": self.role,
            "size": self.size,
            "sha256": self.sha256,
            "shape": list(self.shape),
            "storage_dtype": self.storage_dtype,
        }

    def as_manifest_dict(self) -> dict[str, object]:
        """Return identity plus resolved-path provenance for a manifest."""

        return {
            **self.computation_identity,
            "filename": self.filename,
            "resolved_path": str(self.resolved_path),
        }


@dataclass(frozen=True, slots=True)
class F3DatasetIdentity:
    """Validated content identity for all required files in one dataset."""

    dataset_id: str
    files: tuple[F3FileIdentity, ...]
    data_root: Path = field(compare=False)

    @property
    def computation_identity(self) -> dict[str, object]:
        """Return a deterministic, path-independent dataset identity."""

        return {
            "dataset_id": self.dataset_id,
            "files": [item.computation_identity for item in self.files],
        }

    @property
    def sha256(self) -> str:
        """Return a compact digest of the path-independent identity."""

        payload = json.dumps(
            self.computation_identity,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def as_manifest_dict(self) -> dict[str, object]:
        """Return identity and source paths for provenance serialization."""

        return {
            "dataset_id": self.dataset_id,
            "sha256": self.sha256,
            "data_root": str(self.data_root),
            "files": [item.as_manifest_dict() for item in self.files],
        }

    def file_for(self, role: str) -> F3FileIdentity:
        """Return the identity for ``role`` or reject an unknown role."""

        for item in self.files:
            if item.role == role:
                return item
        expected = ", ".join(item.role for item in self.files)
        raise ValueError(f"unknown F3 file role: {role!r}; expected one of {expected}")


@dataclass(frozen=True, slots=True)
class _FileState:
    size: int
    mtime_ns: int
    device: int | None
    inode: int | None


def _file_state(file_stat: os.stat_result) -> _FileState:
    return _FileState(
        size=file_stat.st_size,
        mtime_ns=file_stat.st_mtime_ns,
        device=getattr(file_stat, "st_dev", None),
        inode=getattr(file_stat, "st_ino", None),
    )


def _stream_sha256(source: BinaryIO, *, buffer_size: int = SHA256_BUFFER_SIZE) -> str:
    digest = hashlib.sha256()
    while chunk := source.read(buffer_size):
        digest.update(chunk)
    return digest.hexdigest()


def _identify_file(
    source_path: Path,
    *,
    role: str,
    filename: str,
    spec: F3DatasetSpec,
) -> tuple[F3FileIdentity, _FileState]:
    try:
        resolved_path = source_path.resolve(strict=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required F3 file is missing: {source_path}") from error

    try:
        before_path_stat = source_path.stat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"required F3 file is missing: {source_path}") from error
    if not stat.S_ISREG(before_path_stat.st_mode):
        raise ValueError(f"required F3 source is not a regular file: {source_path}")
    before_state = _file_state(before_path_stat)
    if before_state.size != spec.expected_bytes:
        raise ValueError(
            f"{source_path}: expected {spec.expected_bytes} bytes for shape {spec.shape}, "
            f"got {before_state.size} bytes"
        )

    with resolved_path.open("rb") as source:
        opened_stat = os.fstat(source.fileno())
        if not stat.S_ISREG(opened_stat.st_mode):
            raise ValueError(f"required F3 source is not a regular file: {source_path}")
        if _file_state(opened_stat) != before_state:
            raise ValueError(f"F3 source changed before checksum completed: {source_path}")
        sha256 = _stream_sha256(source)
        after_open_state = _file_state(os.fstat(source.fileno()))

    try:
        after_resolved_path = source_path.resolve(strict=True)
        after_path_state = _file_state(source_path.stat())
    except FileNotFoundError as error:
        raise ValueError(f"F3 source changed while computing checksum: {source_path}") from error
    if (
        after_resolved_path != resolved_path
        or after_open_state != before_state
        or after_path_state != before_state
    ):
        raise ValueError(f"F3 source changed while computing checksum: {source_path}")

    return (
        F3FileIdentity(
            role=role,
            filename=filename,
            resolved_path=resolved_path,
            size=before_state.size,
            sha256=sha256,
            shape=spec.shape,
            storage_dtype=spec.storage_dtype,
        ),
        before_state,
    )


def ensure_output_not_in_data_root(
    output_path: str | PathLike[str],
    data_root: str | PathLike[str],
    *,
    option_name: str = "--output-dir",
) -> Path:
    """Resolve an output/workspace path and keep it outside the F3 data root."""

    resolved_path = Path(output_path).resolve(strict=False)
    resolved_data_root = Path(data_root).resolve(strict=False)
    if resolved_path == resolved_data_root or resolved_path.is_relative_to(resolved_data_root):
        raise ValueError(f"{option_name} must not be inside the F3 data root: {resolved_path}")
    return resolved_path


class F3VolumeSource:
    """Process-local owner for validated read-only F3 full-volume access."""

    def __init__(
        self,
        data_root: str | PathLike[str],
        *,
        spec: F3DatasetSpec = OFFICIAL_F3_DATASET_SPEC,
    ) -> None:
        if not isinstance(spec, F3DatasetSpec):
            raise ValueError("spec must be an F3DatasetSpec")
        self._data_root = Path(data_root).resolve(strict=False)
        self._spec = spec
        self._closed = False
        self._memmaps: list[np.memmap] = []
        self._source_paths: dict[str, Path] = {}
        self._resolved_paths: dict[str, Path] = {}
        self._states: dict[str, _FileState] = {}

        identities = []
        for role, filename in spec.files:
            source_path = self._data_root / filename
            identity, state = _identify_file(
                source_path,
                role=role,
                filename=filename,
                spec=spec,
            )
            identities.append(identity)
            self._source_paths[role] = source_path
            self._resolved_paths[role] = identity.resolved_path
            self._states[role] = state

        for role in spec.roles:
            self._assert_source_unchanged(role)
        self._identity = F3DatasetIdentity(
            dataset_id=spec.dataset_id,
            files=tuple(identities),
            data_root=self._data_root,
        )

    @property
    def spec(self) -> F3DatasetSpec:
        """Return the immutable storage contract."""

        return self._spec

    @property
    def identity(self) -> F3DatasetIdentity:
        """Return the immutable, checksum-backed dataset identity."""

        return self._identity

    @property
    def closed(self) -> bool:
        """Whether this process-local volume owner has been closed."""

        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("F3VolumeSource is closed")

    def _assert_source_unchanged(self, role: str) -> None:
        source_path = self._source_paths[role]
        try:
            resolved_path = source_path.resolve(strict=True)
            state = _file_state(source_path.stat())
        except FileNotFoundError as error:
            raise ValueError(
                f"F3 source changed after identity validation: {source_path}"
            ) from error
        if state != self._states[role]:
            raise ValueError(f"F3 source changed after identity validation: {source_path}")
        if resolved_path != self._resolved_paths[role]:
            raise ValueError(f"F3 source changed after identity validation: {source_path}")

    def _open_verified_file(self, role: str) -> BinaryIO:
        self._require_open()
        self._spec.filename_for(role)
        source_path = self._source_paths[role]
        try:
            source = source_path.open("rb")
        except FileNotFoundError as error:
            raise ValueError(
                f"F3 source changed after identity validation: {source_path}"
            ) from error

        try:
            opened_stat = os.fstat(source.fileno())
            if not stat.S_ISREG(opened_stat.st_mode):
                raise ValueError(f"F3 source changed after identity validation: {source_path}")
            if _file_state(opened_stat) != self._states[role]:
                raise ValueError(f"F3 source changed after identity validation: {source_path}")
            self._assert_source_unchanged(role)
        except BaseException:
            source.close()
            raise
        return source

    def open_memmap(self, role: str) -> np.memmap:
        """Open and own a read-only full-volume map, preserving storage endian."""

        array: np.memmap | None = None
        with self._open_verified_file(role) as source:
            try:
                array = np.memmap(
                    source,
                    dtype=np.dtype(self._spec.storage_dtype),
                    mode="r",
                    shape=self._spec.shape,
                    order="C",
                )
                if _file_state(os.fstat(source.fileno())) != self._states[role]:
                    raise ValueError(
                        "F3 source changed while opening validated volume: "
                        f"{self._source_paths[role]}"
                    )
                self._assert_source_unchanged(role)
            except BaseException:
                if array is not None:
                    mapping = getattr(array, "_mmap", None)
                    if mapping is not None and not mapping.closed:
                        mapping.close()
                raise
        self._memmaps.append(array)
        return array

    def read_native_volume(self, role: str) -> np.ndarray:
        """Read one C-contiguous, read-only native float32 volume with one copy."""

        source = self.open_memmap(role)
        try:
            result = np.array(source, dtype=np.dtype("=f4"), order="C", copy=True)
        finally:
            self._close_memmap(source)
        result.flags.writeable = False
        return result

    def _close_memmap(self, array: np.memmap) -> None:
        self._memmaps = [item for item in self._memmaps if item is not array]
        mapping = getattr(array, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()

    def close(self) -> None:
        """Release all memory maps owned by this process-local source."""

        if self._closed:
            return
        for array in tuple(self._memmaps):
            self._close_memmap(array)
        self._closed = True

    def __enter__(self) -> F3VolumeSource:
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()
