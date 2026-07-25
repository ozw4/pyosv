"""Content-addressed run workspaces for F3 full-volume comparisons."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

import pyosv

from .data import F3DatasetIdentity, ensure_output_not_in_data_root
from .models import F3ModeComparisonPlan

F3_ARTIFACT_SCHEMA_VERSION = 1
F3_STAGE_CONTRACT_VERSION = 1
F3_FINGERPRINT_CONTRACT_VERSION = 1

RUN_MANIFEST_FILE = "run_manifest.json"
STAGE_MANIFEST_FILE = "stage_manifest.json"
STAGE_COMPLETION_FILE = "complete.json"
STAGE_KINDS = ("scanner", "voting", "thinning", "skinning")
CELL_LABELS = ("RL-REF", "RL-QUAL", "Q-REF", "Q-QUAL")

_STAGE_TEMP_PREFIX = ".pyosv-stage-tmp-"
_MANIFEST_TEMP_PREFIX = ".run_manifest.json.tmp-"
_HASH_BUFFER_SIZE = 1024 * 1024
_SHA256_LENGTH = 64


class F3ArtifactError(ValueError):
    """Base class for invalid or corrupt F3 workspace artifacts."""


class F3WorkspaceMismatchError(F3ArtifactError):
    """Raised when an existing workspace belongs to another computation."""


class F3StageCorruptionError(F3ArtifactError):
    """Raised when a stage directory does not satisfy its completion contract."""


@dataclass(frozen=True, slots=True)
class F3StageArtifact:
    """Storage contract for one stage-local artifact."""

    filename: str
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    format: str | None = None

    def __post_init__(self) -> None:
        _validate_artifact_filename(self.filename)
        inferred_format = Path(self.filename).suffix.removeprefix(".").lower()
        artifact_format = inferred_format if self.format is None else self.format
        if artifact_format not in {"npy", "dat", "json"}:
            raise ValueError("artifact format must be 'npy', 'dat', or 'json'")
        if inferred_format != artifact_format:
            raise ValueError("artifact format must match the filename suffix")
        object.__setattr__(self, "format", artifact_format)

        if artifact_format == "json":
            if self.shape is not None or self.dtype is not None:
                raise ValueError("JSON artifacts must not specify shape or dtype")
            return

        if (
            not isinstance(self.shape, tuple)
            or not self.shape
            or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in self.shape
            )
        ):
            raise ValueError("array artifact shape must contain positive integers")
        default_dtype = ">f4" if artifact_format == "dat" else "float32"
        try:
            dtype = np.dtype(default_dtype if self.dtype is None else self.dtype)
        except TypeError as error:
            raise ValueError(f"invalid artifact dtype: {self.dtype!r}") from error
        if dtype.hasobject:
            raise ValueError("artifact dtype must not contain Python objects")
        if artifact_format == "dat" and dtype.str != ">f4":
            raise ValueError("DAT artifacts must use big-endian float32")
        object.__setattr__(self, "dtype", dtype.str)

    def as_dict(self) -> dict[str, object]:
        """Return the manifest representation."""

        result: dict[str, object] = {
            "filename": self.filename,
            "format": self.format,
        }
        if self.shape is not None:
            result["shape"] = list(self.shape)
        if self.dtype is not None:
            result["dtype"] = self.dtype
        return result


@dataclass(frozen=True, slots=True)
class F3StageResult:
    """The validated location and reuse status of one stage."""

    path: Path
    fingerprint: str
    reused: bool


@dataclass(frozen=True, slots=True)
class F3RunWorkspace:
    """A validated F3 run workspace with stage materialization helpers."""

    path: Path
    fingerprint: str
    manifest: Mapping[str, Any]
    resumed: bool

    def stage_path(self, kind: str, fingerprint: str) -> Path:
        """Return the fixed content-addressed path for a stage."""

        _validate_stage_kind(kind)
        _validate_sha256(fingerprint, "stage fingerprint")
        return self.path / "stages" / kind / fingerprint

    def write_or_reuse_stage(
        self,
        kind: str,
        *,
        parent_fingerprints: Sequence[str] = (),
        input_fingerprints: Mapping[str, str] | None = None,
        resolved_settings: Mapping[str, Any],
        artifacts: Sequence[F3StageArtifact],
        writer: Callable[[Path], None],
        fingerprint: str | None = None,
    ) -> F3StageResult:
        """Validate an existing exact stage or atomically materialize a new one."""

        return write_or_reuse_stage(
            self,
            kind,
            parent_fingerprints=parent_fingerprints,
            input_fingerprints=input_fingerprints,
            resolved_settings=resolved_settings,
            artifacts=artifacts,
            writer=writer,
            fingerprint=fingerprint,
        )


def _workspace_dataset_file_identity(
    workspace: F3RunWorkspace,
    role: str,
) -> Mapping[str, Any]:
    """Return one file identity recorded in a validated run manifest."""

    if not isinstance(workspace, F3RunWorkspace):
        raise TypeError("workspace must be an F3RunWorkspace")
    dataset_identity = workspace.manifest.get("dataset_identity")
    if not isinstance(dataset_identity, Mapping):
        raise F3WorkspaceMismatchError("run manifest has invalid dataset identity")
    files = dataset_identity.get("files")
    if not isinstance(files, list):
        raise F3WorkspaceMismatchError("run manifest has invalid dataset file identities")
    matches = [item for item in files if isinstance(item, Mapping) and item.get("role") == role]
    if len(matches) != 1:
        raise F3WorkspaceMismatchError(
            f"run manifest must contain exactly one dataset file for role {role!r}"
        )
    return matches[0]


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a finite JSON value deterministically as UTF-8."""

    normalized = _json_value(value, "value")
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_fingerprint(value: Any) -> str:
    """Return the SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def implementation_identity(
    *,
    software_versions: Mapping[str, str] | None = None,
    source_files: Mapping[str, str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    """Return version and algorithm-source identity used by run fingerprints.

    ``software_versions`` and ``source_files`` are injectable to make contract
    changes directly testable. Source identifiers, rather than source paths,
    are serialized into the computation identity.
    """

    versions = dict(software_versions) if software_versions is not None else _software_versions()
    expected_versions = {"pyosv", "python", "numpy", "scipy"}
    if set(versions) != expected_versions:
        raise ValueError("software_versions must contain exactly pyosv, python, numpy, and scipy")
    for name, version in versions.items():
        if not isinstance(version, str) or not version:
            raise ValueError(f"software version {name!r} must be a non-empty string")

    files = dict(source_files) if source_files is not None else _algorithm_source_files()
    if not files:
        raise ValueError("algorithm source identity must contain at least one file")
    modules: dict[str, dict[str, Any]] = {}
    for identifier, source in sorted(files.items()):
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("algorithm source identifiers must be non-empty strings")
        path = Path(source)
        _require_regular_nonsymlink(path, f"algorithm source {identifier}")
        metadata = _file_metadata(path)
        modules[identifier] = metadata
    return _validate_implementation_identity(
        {
            "software_versions": dict(sorted(versions.items())),
            "algorithm_modules": modules,
        }
    )


def run_computation_identity(
    plan: F3ModeComparisonPlan,
    dataset_identity: F3DatasetIdentity,
    *,
    implementation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete path- and time-independent run identity."""

    if not isinstance(plan, F3ModeComparisonPlan):
        raise ValueError("plan must be an F3ModeComparisonPlan")
    if not isinstance(dataset_identity, F3DatasetIdentity):
        raise ValueError("dataset_identity must be an F3DatasetIdentity")
    _validate_dataset_identity(dataset_identity, plan)
    identity = (
        implementation_identity()
        if implementation is None
        else _validate_implementation_identity(implementation)
    )
    return {
        "artifact_schema_version": F3_ARTIFACT_SCHEMA_VERSION,
        "stage_contract_version": F3_STAGE_CONTRACT_VERSION,
        "fingerprint_contract_version": F3_FINGERPRINT_CONTRACT_VERSION,
        "plan": plan.as_dict(),
        "dataset_identity": dataset_identity.computation_identity,
        "implementation_identity": identity,
    }


def run_fingerprint(
    plan: F3ModeComparisonPlan,
    dataset_identity: F3DatasetIdentity,
    *,
    implementation: Mapping[str, Any] | None = None,
) -> str:
    """Return the computation fingerprint for one resolved F3 run."""

    return canonical_fingerprint(
        run_computation_identity(
            plan,
            dataset_identity,
            implementation=implementation,
        )
    )


def prepare_run_workspace(
    output_dir: str | os.PathLike[str],
    plan: F3ModeComparisonPlan,
    dataset_identity: F3DatasetIdentity,
    *,
    resume: bool,
    implementation: Mapping[str, Any] | None = None,
    created_at: str | None = None,
    source_provenance: Mapping[str, Any] | None = None,
) -> F3RunWorkspace:
    """Create a fixed-layout workspace or validate it for exact resume."""

    if not isinstance(resume, bool):
        raise ValueError("resume must be a bool")
    if not isinstance(dataset_identity, F3DatasetIdentity):
        raise ValueError("dataset_identity must be an F3DatasetIdentity")
    requested_output_path = Path(output_dir).absolute()
    ensure_output_not_in_data_root(requested_output_path, dataset_identity.data_root)
    output_exists = requested_output_path.exists() or requested_output_path.is_symlink()
    computation = run_computation_identity(
        plan,
        dataset_identity,
        implementation=implementation,
    )
    fingerprint = canonical_fingerprint(computation)

    if output_exists:
        if not resume:
            raise FileExistsError(f"run workspace already exists: {requested_output_path}")
        _require_directory_nonsymlink(requested_output_path, "run workspace")
        output_path = requested_output_path.resolve(strict=True)
        manifest = _read_json_object(output_path / RUN_MANIFEST_FILE)
        _validate_run_manifest(manifest, computation, fingerprint)
        _validate_workspace_layout(output_path)
        _cleanup_stage_temporaries(output_path)
        return F3RunWorkspace(output_path, fingerprint, manifest, resumed=True)

    if resume:
        raise FileNotFoundError(f"run workspace does not exist: {requested_output_path}")

    output_path = requested_output_path.resolve(strict=False)
    manifest = {
        **computation,
        "run_fingerprint": fingerprint,
        "provenance": {
            "created_at": created_at or _utc_now(),
            "data_root": str(dataset_identity.data_root),
            "output_path": str(output_path),
            "dataset_files": [
                {
                    "role": item.role,
                    "filename": item.filename,
                    "resolved_path": str(item.resolved_path),
                }
                for item in dataset_identity.files
            ],
            "source": (
                dict(source_provenance) if source_provenance is not None else _source_provenance()
            ),
        },
    }
    # Validate all values before creating anything on disk.
    canonical_json_bytes(manifest)

    created = False
    try:
        output_path.mkdir()
        created = True
        for kind in STAGE_KINDS:
            (output_path / "stages" / kind).mkdir(parents=True)
        (output_path / "cells").mkdir()
        (output_path / "reports").mkdir()
        _atomic_write_json(output_path / RUN_MANIFEST_FILE, manifest)
        _fsync_directory(output_path)
    except BaseException as error:
        if created:
            _cleanup_path(output_path, error)
        raise
    return F3RunWorkspace(output_path, fingerprint, manifest, resumed=False)


def stage_computation_identity(
    kind: str,
    *,
    run_fingerprint_value: str,
    parent_fingerprints: Sequence[str] = (),
    input_fingerprints: Mapping[str, str] | None = None,
    resolved_settings: Mapping[str, Any],
    artifacts: Sequence[F3StageArtifact],
) -> dict[str, Any]:
    """Build the computation identity for a single content-addressed stage."""

    _validate_stage_kind(kind)
    _validate_sha256(run_fingerprint_value, "run fingerprint")
    parents = list(parent_fingerprints)
    for value in parents:
        _validate_sha256(value, "parent stage fingerprint")
    inputs = dict(input_fingerprints or {})
    for name, value in inputs.items():
        if not isinstance(name, str) or not name:
            raise ValueError("input fingerprint names must be non-empty strings")
        _validate_sha256(value, f"input fingerprint {name!r}")
    schema = _artifact_schema(artifacts)
    return {
        "artifact_schema_version": F3_ARTIFACT_SCHEMA_VERSION,
        "stage_contract_version": F3_STAGE_CONTRACT_VERSION,
        "kind": kind,
        "run_fingerprint": run_fingerprint_value,
        "parent_fingerprints": parents,
        "input_fingerprints": inputs,
        "resolved_settings": dict(resolved_settings),
        "artifact_schema": schema,
    }


def stage_fingerprint(
    kind: str,
    *,
    run_fingerprint_value: str,
    parent_fingerprints: Sequence[str] = (),
    input_fingerprints: Mapping[str, str] | None = None,
    resolved_settings: Mapping[str, Any],
    artifacts: Sequence[F3StageArtifact],
) -> str:
    """Return the content fingerprint for one stage contract."""

    return canonical_fingerprint(
        stage_computation_identity(
            kind,
            run_fingerprint_value=run_fingerprint_value,
            parent_fingerprints=parent_fingerprints,
            input_fingerprints=input_fingerprints,
            resolved_settings=resolved_settings,
            artifacts=artifacts,
        )
    )


def write_or_reuse_stage(
    workspace: F3RunWorkspace,
    kind: str,
    *,
    parent_fingerprints: Sequence[str] = (),
    input_fingerprints: Mapping[str, str] | None = None,
    resolved_settings: Mapping[str, Any],
    artifacts: Sequence[F3StageArtifact],
    writer: Callable[[Path], None],
    fingerprint: str | None = None,
) -> F3StageResult:
    """Reuse an exact complete stage or publish a newly written stage atomically."""

    if not isinstance(workspace, F3RunWorkspace):
        raise ValueError("workspace must be an F3RunWorkspace")
    if not callable(writer):
        raise ValueError("writer must be callable")
    computation = stage_computation_identity(
        kind,
        run_fingerprint_value=workspace.fingerprint,
        parent_fingerprints=parent_fingerprints,
        input_fingerprints=input_fingerprints,
        resolved_settings=resolved_settings,
        artifacts=artifacts,
    )
    expected_fingerprint = canonical_fingerprint(computation)
    if fingerprint is not None:
        _validate_sha256(fingerprint, "stage fingerprint")
        if fingerprint != expected_fingerprint:
            raise ValueError("provided stage fingerprint does not match stage computation")
    final_path = workspace.stage_path(kind, expected_fingerprint)
    if final_path.exists() or final_path.is_symlink():
        validate_stage(final_path, computation, expected_fingerprint)
        return F3StageResult(final_path, expected_fingerprint, reused=True)

    parent = final_path.parent
    _require_directory_nonsymlink(parent, f"{kind} stage parent")
    temporary_path = Path(tempfile.mkdtemp(prefix=_STAGE_TEMP_PREFIX, dir=parent))
    temporary_identity = _path_identity(temporary_path)
    try:
        writer(temporary_path)
        expected_names = set(computation["artifact_schema"])
        actual_names = {item.name for item in temporary_path.iterdir()}
        if actual_names != expected_names:
            raise F3StageCorruptionError(
                _file_set_message("stage writer", expected_names, actual_names)
            )
        artifact_files: dict[str, dict[str, Any]] = {}
        for artifact in artifacts:
            path = temporary_path / artifact.filename
            _validate_stage_artifact(path, artifact)
            _fsync_file(path)
            metadata = _file_metadata(path)
            if artifact.format == "dat":
                metadata.update(
                    {
                        "shape": list(artifact.shape or ()),
                        "dtype": artifact.dtype,
                    }
                )
            artifact_files[artifact.filename] = metadata

        stage_manifest = {
            **computation,
            "fingerprint": expected_fingerprint,
            "shape": _common_schema_value(artifacts, "shape"),
            "dtype": _common_schema_value(artifacts, "dtype"),
            "files": artifact_files,
        }
        _write_bytes(
            temporary_path / STAGE_MANIFEST_FILE,
            canonical_json_bytes(stage_manifest) + b"\n",
        )
        completed_files = {
            **artifact_files,
            STAGE_MANIFEST_FILE: _file_metadata(temporary_path / STAGE_MANIFEST_FILE),
        }
        completion = {
            "artifact_schema_version": F3_ARTIFACT_SCHEMA_VERSION,
            "stage_contract_version": F3_STAGE_CONTRACT_VERSION,
            "kind": kind,
            "fingerprint": expected_fingerprint,
            "files": completed_files,
        }
        # This marker is deliberately the final file created in the temporary tree.
        _write_bytes(
            temporary_path / STAGE_COMPLETION_FILE,
            canonical_json_bytes(completion) + b"\n",
        )
        _fsync_directory(temporary_path)
        _rename_noreplace(temporary_path, final_path)
        _fsync_directory(parent)
        validate_stage(final_path, computation, expected_fingerprint)
    except BaseException as error:
        _cleanup_path(temporary_path, error)
        _cleanup_path_if_identity(final_path, temporary_identity, error)
        raise

    return F3StageResult(final_path, expected_fingerprint, reused=False)


def validate_stage(
    stage_path: str | os.PathLike[str],
    expected_computation: Mapping[str, Any],
    expected_fingerprint: str,
) -> Path:
    """Validate schema, completion, file hashes, and arrays for one exact stage."""

    path = Path(stage_path)
    _validate_sha256(expected_fingerprint, "stage fingerprint")
    try:
        _require_directory_nonsymlink(path, "stage directory")
        completion = _read_json_object(path / STAGE_COMPLETION_FILE)
        manifest = _read_json_object(path / STAGE_MANIFEST_FILE)
        expected_manifest_fields = {
            **dict(expected_computation),
            "fingerprint": expected_fingerprint,
        }
        for name, value in expected_manifest_fields.items():
            if name not in manifest or not _exact_json_value(
                manifest[name], _json_value(value, name)
            ):
                raise F3StageCorruptionError(f"stage manifest mismatch: {name}")
        if manifest.get("shape") != _common_schema_mapping_value(
            expected_computation["artifact_schema"], "shape"
        ):
            raise F3StageCorruptionError("stage manifest mismatch: shape")
        if manifest.get("dtype") != _common_schema_mapping_value(
            expected_computation["artifact_schema"], "dtype"
        ):
            raise F3StageCorruptionError("stage manifest mismatch: dtype")

        for name, expected in (
            ("artifact_schema_version", F3_ARTIFACT_SCHEMA_VERSION),
            ("stage_contract_version", F3_STAGE_CONTRACT_VERSION),
            ("fingerprint", expected_fingerprint),
            ("kind", expected_computation["kind"]),
        ):
            if not _exact_json_value(completion.get(name), expected):
                raise F3StageCorruptionError(f"stage completion mismatch: {name}")

        schema = expected_computation["artifact_schema"]
        expected_hashed = {STAGE_MANIFEST_FILE, *schema}
        completion_files = completion.get("files")
        if not isinstance(completion_files, dict) or set(completion_files) != expected_hashed:
            raise F3StageCorruptionError("stage completion file list mismatch")
        expected_all = {*expected_hashed, STAGE_COMPLETION_FILE}
        actual_all = {item.name for item in path.iterdir()}
        if actual_all != expected_all:
            raise F3StageCorruptionError(
                _file_set_message("completed stage", expected_all, actual_all)
            )

        for filename in sorted(expected_hashed):
            artifact_path = path / filename
            _require_regular_nonsymlink(artifact_path, f"stage file {filename}")
            metadata = completion_files[filename]
            descriptor = schema.get(filename)
            if not _valid_stage_file_metadata(metadata, descriptor):
                raise F3StageCorruptionError(f"invalid file metadata: {filename}")
            actual_metadata = _file_metadata(artifact_path)
            if (
                actual_metadata["sha256"] != metadata["sha256"]
                or actual_metadata["size"] != metadata["size"]
            ):
                raise F3StageCorruptionError(f"stage file hash or size mismatch: {filename}")
        if manifest.get("files") != {name: completion_files[name] for name in sorted(schema)}:
            raise F3StageCorruptionError("stage manifest artifact file list mismatch")

        for filename, descriptor in schema.items():
            shape_value = descriptor.get("shape")
            artifact = F3StageArtifact(
                filename=filename,
                shape=(tuple(shape_value) if shape_value is not None else None),
                dtype=descriptor.get("dtype"),
                format=descriptor["format"],
            )
            _validate_stage_artifact(path / filename, artifact)
    except F3StageCorruptionError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise F3StageCorruptionError(f"invalid stage {path}: {error}") from error
    return path


def _artifact_schema(artifacts: Sequence[F3StageArtifact]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, F3StageArtifact):
            raise ValueError("artifacts must contain only F3StageArtifact values")
        if artifact.filename in result:
            raise ValueError(f"duplicate artifact filename: {artifact.filename}")
        result[artifact.filename] = artifact.as_dict()
    if not result:
        raise ValueError("stage must contain at least one artifact")
    return dict(sorted(result.items()))


def _validate_stage_artifact(path: Path, artifact: F3StageArtifact) -> None:
    if artifact.format == "npy":
        _validate_numpy_artifact(path, artifact)
    elif artifact.format == "dat":
        _validate_dat_artifact(path, artifact)
    elif artifact.format == "json":
        _read_json_object(path)
    else:  # pragma: no cover - F3StageArtifact validation makes this unreachable.
        raise F3StageCorruptionError(f"unknown artifact format: {artifact.format}")


def _validate_numpy_artifact(path: Path, artifact: F3StageArtifact) -> None:
    _require_regular_nonsymlink(path, f"stage artifact {artifact.filename}")
    array: np.ndarray | None = None
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not isinstance(array, np.ndarray):
            raise F3StageCorruptionError(
                f"stage artifact is not a NumPy array: {artifact.filename}"
            )
        if array.shape != artifact.shape:
            raise F3StageCorruptionError(
                f"stage artifact shape mismatch for {artifact.filename}: "
                f"expected {artifact.shape}, got {array.shape}"
            )
        if array.dtype.str != artifact.dtype:
            raise F3StageCorruptionError(
                f"stage artifact dtype mismatch for {artifact.filename}: "
                f"expected {artifact.dtype}, got {array.dtype.str}"
            )
    except F3StageCorruptionError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise F3StageCorruptionError(f"invalid NumPy stage artifact {artifact.filename}") from error
    finally:
        if isinstance(array, np.memmap):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()


def _validate_dat_artifact(path: Path, artifact: F3StageArtifact) -> None:
    _require_regular_nonsymlink(path, f"stage artifact {artifact.filename}")
    assert artifact.shape is not None
    assert artifact.dtype is not None
    expected_size = int(np.prod(artifact.shape)) * np.dtype(artifact.dtype).itemsize
    try:
        actual_size = path.stat().st_size
    except OSError as error:
        raise F3StageCorruptionError(f"invalid DAT stage artifact {artifact.filename}") from error
    if actual_size != expected_size:
        raise F3StageCorruptionError(
            f"stage artifact size mismatch for {artifact.filename}: "
            f"expected {expected_size}, got {actual_size}"
        )


def _validate_run_manifest(
    manifest: Mapping[str, Any],
    computation: Mapping[str, Any],
    fingerprint: str,
) -> None:
    expected = {**dict(computation), "run_fingerprint": fingerprint}
    for name, value in expected.items():
        if name not in manifest or not _exact_json_value(manifest[name], _json_value(value, name)):
            raise F3WorkspaceMismatchError(f"run manifest mismatch: {name}")
    if set(manifest) != {*expected, "provenance"}:
        raise F3WorkspaceMismatchError("run manifest field set mismatch")
    if not isinstance(manifest["provenance"], dict):
        raise F3WorkspaceMismatchError("run manifest provenance must be an object")


def _validate_dataset_identity(
    dataset_identity: F3DatasetIdentity,
    plan: F3ModeComparisonPlan,
) -> None:
    spec = plan.dataset_spec
    if dataset_identity.dataset_id != spec.dataset_id:
        raise ValueError("dataset identity does not match the plan dataset ID")
    if tuple(item.role for item in dataset_identity.files) != spec.roles:
        raise ValueError("dataset identity does not contain the plan's required file roles")
    for item, (role, filename) in zip(dataset_identity.files, spec.files, strict=True):
        if item.filename != filename:
            raise ValueError(f"dataset identity filename does not match the plan for role {role!r}")
        if item.shape != spec.shape:
            raise ValueError(f"dataset identity shape does not match the plan for role {role!r}")
        if item.storage_dtype != spec.storage_dtype:
            raise ValueError(f"dataset identity dtype does not match the plan for role {role!r}")
        if item.size != spec.expected_bytes:
            raise ValueError(f"dataset identity size does not match the plan for role {role!r}")
        try:
            _validate_sha256(item.sha256, f"dataset identity checksum for role {role!r}")
        except ValueError as error:
            raise ValueError(
                f"dataset identity checksum does not match the plan contract for role {role!r}"
            ) from error


def _validate_implementation_identity(
    implementation: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(implementation, Mapping):
        raise ValueError("implementation identity must be an object")
    if set(implementation) != {"software_versions", "algorithm_modules"}:
        raise ValueError(
            "implementation identity must contain exactly software_versions and algorithm_modules"
        )

    versions = implementation["software_versions"]
    expected_versions = {"pyosv", "python", "numpy", "scipy"}
    if not isinstance(versions, Mapping) or set(versions) != expected_versions:
        raise ValueError(
            "implementation software_versions must contain exactly pyosv, python, numpy, and scipy"
        )
    normalized_versions: dict[str, str] = {}
    for name, version in sorted(versions.items()):
        if not isinstance(version, str) or not version:
            raise ValueError(f"implementation software version {name!r} must be a non-empty string")
        normalized_versions[name] = version

    modules = implementation["algorithm_modules"]
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError("implementation algorithm_modules must be a non-empty object")
    normalized_modules: dict[str, dict[str, Any]] = {}
    for identifier, metadata in modules.items():
        if (
            not isinstance(identifier, str)
            or not identifier
            or "\\" in identifier
            or identifier.startswith("/")
            or any(part in {"", ".", ".."} for part in identifier.split("/"))
        ):
            raise ValueError(
                "implementation algorithm module identifiers must be "
                "non-empty relative logical paths"
            )
        if not isinstance(metadata, Mapping) or set(metadata) != {"sha256", "size"}:
            raise ValueError(
                f"implementation algorithm module {identifier!r} must contain "
                "exactly sha256 and size"
            )
        _validate_sha256(
            metadata["sha256"],
            f"implementation algorithm module {identifier!r} sha256",
        )
        size = metadata["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"implementation algorithm module {identifier!r} size "
                "must be a non-negative integer"
            )
        normalized_modules[identifier] = {
            "sha256": metadata["sha256"],
            "size": size,
        }

    return {
        "software_versions": normalized_versions,
        "algorithm_modules": dict(sorted(normalized_modules.items())),
    }


def _validate_workspace_layout(path: Path) -> None:
    _require_regular_nonsymlink(path / RUN_MANIFEST_FILE, RUN_MANIFEST_FILE)
    for directory in ("stages", "cells", "reports"):
        _require_directory_nonsymlink(path / directory, f"workspace {directory}")
    for kind in STAGE_KINDS:
        _require_directory_nonsymlink(path / "stages" / kind, f"{kind} stage parent")


def _cleanup_stage_temporaries(path: Path) -> None:
    for kind in STAGE_KINDS:
        parent = path / "stages" / kind
        for candidate in parent.iterdir():
            if not candidate.name.startswith(_STAGE_TEMP_PREFIX):
                continue
            suffix = candidate.name.removeprefix(_STAGE_TEMP_PREFIX)
            if not suffix or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in suffix.lower()
            ):
                continue
            info = candidate.lstat()
            if stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode):
                shutil.rmtree(candidate)


def _algorithm_source_files() -> dict[str, Path]:
    package_root = Path(pyosv.__file__).resolve().parent
    roots = (
        package_root / "_accel.py",
        package_root / "_seed_selection.py",
        package_root / "cells.py",
        package_root / "dp.py",
        package_root / "_dp",
        package_root / "filters.py",
        package_root / "geometry.py",
        package_root / "interp.py",
        package_root / "orient3d.py",
        package_root / "_orient3d",
        package_root / "voting3d.py",
        package_root / "_voting3d",
        package_root / "thinning3d.py",
        package_root / "skin.py",
        package_root / "skinner.py",
        package_root / "_skinner",
        package_root / "evaluation" / "workflow3d.py",
        package_root / "evaluation" / "f3d_mode_comparison",
        package_root / "evaluation" / "synthetic_quality",
        package_root / "experimental",
        package_root / "synthetic_metrics.py",
    )
    result: dict[str, Path] = {}
    for root in roots:
        paths = (root,) if root.is_file() else tuple(sorted(root.glob("*.py")))
        for path in paths:
            identifier = path.relative_to(package_root).as_posix()
            result[identifier] = path
    return result


def _software_versions() -> dict[str, str]:
    return {
        "pyosv": pyosv.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def _source_provenance() -> dict[str, Any]:
    unavailable = {
        "status": "not_available",
        "method": "git_cli",
        "commit": None,
        "dirty": None,
    }
    try:
        source_file = Path(__file__).resolve()
        root_text = subprocess.run(
            ["git", "-C", str(source_file.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if not root_text:
            return unavailable
        root = Path(root_text).resolve()
        source_relative = source_file.relative_to(root).as_posix()
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", source_relative],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()
        if tracked != [source_relative]:
            return unavailable
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status_text = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        _validate_sha256(commit, "Git commit")
    except Exception:
        return unavailable
    return {
        "status": "available",
        "method": "git_cli",
        "commit": commit,
        "dirty": bool(status_text),
    }


def _atomic_write_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=_MANIFEST_TEMP_PREFIX,
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    payload = canonical_json_bytes(value) + b"\n"
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            written = stream.write(payload)
            if written != len(payload):
                raise OSError(f"short artifact write for {path.name}")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        written = stream.write(payload)
        if written != len(payload):
            raise OSError(f"short artifact write for {path.name}")
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_file(path: Path) -> None:
    _require_regular_nonsymlink(path, f"artifact file {path.name}")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _file_metadata(path: Path) -> dict[str, Any]:
    _require_regular_nonsymlink(path, f"artifact file {path.name}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_BUFFER_SIZE):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def _rename_noreplace(source: Path, destination: Path) -> None:
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = library.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace rename is not supported",
                destination,
            ) from error
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), destination)
        return
    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        renamex_np = library.renamex_np
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(source), os.fsencode(destination), 4)
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), destination)
        return
    if os.name == "nt":
        os.rename(source, destination)
        return
    raise OSError(errno.ENOTSUP, "atomic no-replace rename is not supported", destination)


def _read_json_object(path: Path) -> dict[str, Any]:
    _require_regular_nonsymlink(path, f"JSON artifact {path.name}")
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise F3ArtifactError(f"malformed JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise F3ArtifactError(f"JSON artifact must contain an object: {path}")
    return value


def _json_value(value: Any, context: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{context} must contain only finite numbers")
        return value
    if isinstance(value, np.generic) or isinstance(value, np.ndarray):
        raise ValueError(f"{context} must not contain implicit NumPy values")
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context} object keys must be strings")
            output[key] = _json_value(item, f"{context}.{key}")
        return output
    if isinstance(value, (tuple, list)):
        return [_json_value(item, f"{context}[]") for item in value]
    raise ValueError(f"{context} is not JSON-safe: {type(value).__name__}")


def _exact_json_value(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_json_value(actual[name], value) for name, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _common_schema_value(artifacts: Sequence[F3StageArtifact], field: str) -> Any:
    values = [value for artifact in artifacts if (value := getattr(artifact, field)) is not None]
    if not values:
        return None
    normalized = [list(value) if isinstance(value, tuple) else value for value in values]
    return normalized[0] if all(value == normalized[0] for value in normalized) else None


def _common_schema_mapping_value(schema: Any, field: str) -> Any:
    if not isinstance(schema, Mapping) or not schema:
        raise F3StageCorruptionError("invalid expected artifact schema")
    values = [item[field] for item in schema.values() if item.get(field) is not None]
    if not values:
        return None
    return values[0] if all(value == values[0] for value in values) else None


def _valid_file_metadata(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {"sha256", "size"}:
        return False
    try:
        _validate_sha256(value["sha256"], "file hash")
    except ValueError:
        return False
    return (
        isinstance(value["size"], int)
        and not isinstance(value["size"], bool)
        and value["size"] >= 0
    )


def _valid_stage_file_metadata(value: Any, descriptor: Any) -> bool:
    if descriptor is None:
        return _valid_file_metadata(value)
    if not isinstance(descriptor, Mapping):
        return False
    expected_extra: dict[str, Any] = {}
    if descriptor.get("format") == "dat":
        expected_extra = {
            "shape": descriptor.get("shape"),
            "dtype": descriptor.get("dtype"),
        }
    if not isinstance(value, dict) or set(value) != {"sha256", "size", *expected_extra}:
        return False
    basic = {"sha256": value.get("sha256"), "size": value.get("size")}
    return _valid_file_metadata(basic) and all(
        _exact_json_value(value.get(name), expected) for name, expected in expected_extra.items()
    )


def _validate_artifact_filename(filename: str) -> None:
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in {STAGE_MANIFEST_FILE, STAGE_COMPLETION_FILE}
    ):
        raise ValueError(f"unsafe or reserved artifact filename: {filename!r}")


def _validate_stage_kind(kind: str) -> None:
    if kind not in STAGE_KINDS:
        raise ValueError(f"unknown stage kind: {kind!r}")


def _validate_sha256(value: Any, context: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 hex digest")


def _require_directory_nonsymlink(path: Path, context: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise F3ArtifactError(f"{context} is missing: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise F3ArtifactError(f"{context} is not a non-symlink directory: {path}")


def _require_regular_nonsymlink(path: Path, context: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise F3ArtifactError(f"{context} is missing: {path}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise F3ArtifactError(f"{context} is not a non-symlink regular file: {path}")


def _file_set_message(context: str, expected: set[str], actual: set[str]) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return f"{context} file set mismatch; missing={missing}, extra={extra}"


def _cleanup_path(path: Path, original_error: BaseException) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return
    try:
        shutil.rmtree(path)
    except BaseException as cleanup_error:
        add_note = getattr(original_error, "add_note", None)
        if add_note is not None:
            add_note(f"artifact cleanup also failed for {path}: {cleanup_error!r}")


def _path_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    return info.st_dev, info.st_ino


def _cleanup_path_if_identity(
    path: Path,
    expected_identity: tuple[int, int],
    original_error: BaseException,
) -> None:
    try:
        if _path_identity(path) != expected_identity:
            return
    except FileNotFoundError:
        return
    _cleanup_path(path, original_error)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
