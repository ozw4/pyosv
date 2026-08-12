"""Filesystem helpers for publication manifest v1 bundles."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import NoReturn, cast

from pyosv.evaluation.publication_manifest import (
    canonical_json_bytes,
    validate_publication_manifest,
)

_MANIFEST_NAME = "publication_manifest.json"
_WINDOWS_DRIVE_PATTERN = re.compile(r"[A-Za-z]:")
_HASH_CHUNK_SIZE = 1024 * 1024

__all__ = [
    "artifact_file_record",
    "validate_publication_directory",
    "write_publication_manifest",
]


def artifact_file_record(
    root: str | PathLike[str],
    relative_path: str,
    *,
    tier: str,
    role: str,
) -> dict[str, object]:
    """Build an artifact record from a regular file below a publication root."""
    root_path = _require_publication_root(root)
    normalized_path = _safe_relative_path(relative_path, "relative_path")
    if normalized_path == _MANIFEST_NAME:
        raise ValueError(f"{_MANIFEST_NAME} must not be an artifact")
    if type(tier) is not str or tier not in {"primary", "derived"}:
        raise ValueError("tier must be 'primary' or 'derived'")
    if type(role) is not str or not role:
        raise ValueError("role must be a non-empty string")

    size, sha256 = _hash_regular_file(root_path, normalized_path)
    if size <= 0:
        raise ValueError("artifact file must not be empty")
    return {
        "path": normalized_path,
        "tier": tier,
        "role": role,
        "size": size,
        "sha256": sha256,
    }


def validate_publication_directory(root: str | PathLike[str]) -> dict[str, object]:
    """Validate a manifest and every regular file in its publication directory."""
    root_path = _require_publication_root(root)
    manifest_path = root_path / _MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"{_MANIFEST_NAME} must be a regular non-symlink file")

    with manifest_path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=_reject_nonfinite_json)
    normalized = validate_publication_manifest(value)
    _validate_directory_contents(root_path, normalized, include_manifest=True)
    return normalized


def write_publication_manifest(
    root: str | PathLike[str],
    manifest: Mapping[str, object],
    *,
    pretty: bool = False,
) -> Path:
    """Atomically publish a validated manifest after all artifacts are complete."""
    if type(pretty) is not bool:
        raise ValueError("pretty must be a bool")
    root_path = _require_publication_root(root)
    destination = root_path / _MANIFEST_NAME
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    normalized = validate_publication_manifest(manifest)
    _validate_directory_contents(root_path, normalized, include_manifest=False)
    manifest_bytes = _manifest_json_bytes(normalized, pretty=pretty)

    descriptor, temporary_name = tempfile.mkstemp(
        dir=root_path,
        prefix=f".{_MANIFEST_NAME}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    published = False
    try:
        os.close(descriptor)
        _write_manifest_bytes(temporary, manifest_bytes)
        os.replace(temporary, destination)
        published = True
        _fsync_directory(root_path)
        validate_publication_directory(root_path)
    except BaseException:
        _unlink_if_present(temporary)
        if published:
            _unlink_if_present(destination)
        raise
    return destination


def _require_publication_root(root: str | PathLike[str]) -> Path:
    root_path = Path(root)
    if root_path.is_symlink() or not root_path.is_dir():
        raise ValueError("publication root must be an existing non-symlink directory")
    return root_path


def _safe_relative_path(value: object, context: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{context} must be a non-empty string")
    if value.startswith("/"):
        raise ValueError(f"{context} must be relative")
    if "\\" in value:
        raise ValueError(f"{context} must use POSIX separators")
    if _WINDOWS_DRIVE_PATTERN.match(value):
        raise ValueError(f"{context} must not be a Windows drive path")
    if any(component in {"", ".", ".."} for component in value.split("/")):
        raise ValueError(f"{context} contains an unsafe path component")
    return value


def _hash_regular_file(root: Path, relative_path: str) -> tuple[int, str]:
    path = root
    components = relative_path.split("/")
    for component in components[:-1]:
        path /= component
        if path.is_symlink() or not path.is_dir():
            raise ValueError(
                f"artifact parent must be a regular non-symlink directory: {relative_path}"
            )
    path /= components[-1]
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {relative_path}")

    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _validate_directory_contents(
    root: Path,
    manifest: Mapping[str, object],
    *,
    include_manifest: bool,
) -> None:
    artifacts = cast(list[dict[str, object]], manifest["artifacts"])
    artifact_paths: set[str] = set()
    artifacts_by_path: dict[str, dict[str, object]] = {}
    for artifact in artifacts:
        relative_path = cast(str, artifact["path"])
        artifact_paths.add(relative_path)
        artifacts_by_path[relative_path] = artifact
        size, sha256 = _hash_regular_file(root, relative_path)
        if size != artifact["size"]:
            raise ValueError(f"artifact size does not match manifest: {relative_path}")
        if sha256 != artifact["sha256"]:
            raise ValueError(f"artifact SHA-256 does not match manifest: {relative_path}")

    environment = cast(dict[str, object], manifest["environment"])
    experiment = cast(dict[str, object], manifest["experiment"])
    _require_linked_artifact(
        artifacts_by_path,
        cast(str, environment["lock_file"]),
        cast(str, environment["lock_sha256"]),
        "environment lock file",
    )
    _require_linked_artifact(
        artifacts_by_path,
        cast(str, experiment["config_file"]),
        cast(str, experiment["config_sha256"]),
        "experiment config file",
    )

    expected = artifact_paths | ({_MANIFEST_NAME} if include_manifest else set())
    actual = _regular_file_paths(root)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            f"publication directory has an invalid regular file set; "
            f"missing={missing}, unknown={unknown}"
        )


def _require_linked_artifact(
    artifacts_by_path: Mapping[str, Mapping[str, object]],
    relative_path: str,
    sha256: str,
    context: str,
) -> None:
    artifact = artifacts_by_path.get(relative_path)
    if artifact is None:
        raise ValueError(f"{context} must be listed as an artifact: {relative_path}")
    if artifact["sha256"] != sha256:
        raise ValueError(f"{context} SHA-256 does not match its artifact entry")


def _regular_file_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_names[:] = [
            name for name in directory_names if not (directory_path / name).is_symlink()
        ]
        for name in file_names:
            path = directory_path / name
            if path.is_file() and not path.is_symlink():
                paths.add(path.relative_to(root).as_posix())
    return paths


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _manifest_json_bytes(manifest: Mapping[str, object], *, pretty: bool) -> bytes:
    if not pretty:
        return canonical_json_bytes(manifest) + b"\n"
    return (
        json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_manifest_bytes(path: Path, value: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
