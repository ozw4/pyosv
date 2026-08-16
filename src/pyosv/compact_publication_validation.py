"""Standalone validation for F3 compact publication directories."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import NoReturn, cast

F3_COMPACT_PUBLICATION_MANIFEST_SCHEMA = "pyosv.f3_compact_publication_manifest.v1"
PUBLICATION_MANIFEST_FILENAME = "publication_manifest.json"

_TOP_LEVEL_FIELDS = {
    "schema",
    "publication_id",
    "created_at_utc",
    "code",
    "environment",
    "source",
    "dataset",
    "experiment",
    "semantics",
    "artifacts",
}
_CONTROL_FIELDS = {
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_DISABLE_JIT",
    "NUMBA_NUM_THREADS",
    "PYOSV_ACCEL",
}
_ARTIFACT_TIERS = {"primary", "derived"}
_ARTIFACT_ROLES = {
    "environment_lock",
    "resolved_experiment",
    "summary_table",
    "figure_data",
    "figure",
    "report",
}
_DATASET_FILE_CONTRACT = {
    "input": "ep.dat",
    "reference_fault_likelihood": "fl.dat",
    "reference_fault_votes": "fv.dat",
    "reference_thinned_fault_votes": "fvt.dat",
    "seismic_amplitude": "xs.dat",
}
_STAGE_ORDER = ["ft", "fv", "fvt"]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_TIMESTAMP_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_WINDOWS_DRIVE_PATTERN = re.compile(r"[A-Za-z]:")
_HASH_CHUNK_SIZE = 1024 * 1024

__all__ = ["validate_compact_publication"]


def build_manifest(
    *,
    created_at_utc: str,
    code: Mapping[str, object],
    environment: Mapping[str, object],
    source: Mapping[str, object],
    dataset: Mapping[str, object],
    experiment: Mapping[str, object],
    semantics: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build, identify, and validate one compact publication manifest."""

    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, Sequence):
        raise ValueError("artifacts must be a sequence")
    sorted_artifacts = sorted(artifacts, key=_artifact_sort_key)
    without_id: dict[str, object] = {
        "schema": F3_COMPACT_PUBLICATION_MANIFEST_SCHEMA,
        "created_at_utc": created_at_utc,
        "code": code,
        "environment": environment,
        "source": source,
        "dataset": dataset,
        "experiment": experiment,
        "semantics": semantics,
        "artifacts": sorted_artifacts,
    }
    publication_id = compute_publication_id(without_id)
    manifest: dict[str, object] = {
        "schema": without_id["schema"],
        "publication_id": publication_id,
        "created_at_utc": without_id["created_at_utc"],
        "code": without_id["code"],
        "environment": without_id["environment"],
        "source": without_id["source"],
        "dataset": without_id["dataset"],
        "experiment": without_id["experiment"],
        "semantics": without_id["semantics"],
        "artifacts": without_id["artifacts"],
    }
    return validate_manifest(manifest)


def compute_publication_id(manifest_without_publication_id: Mapping[str, object]) -> str:
    """Hash provenance and primary records, excluding time and derived records."""

    source = _require_mapping(manifest_without_publication_id, "manifest")
    if "publication_id" in source:
        raise ValueError("manifest must not contain publication_id")
    candidate = dict(source)
    candidate["publication_id"] = "pending"
    normalized = validate_manifest(candidate, verify_publication_id=False)
    return _publication_id_from_normalized(normalized)


def validate_manifest(
    manifest: Mapping[str, object],
    *,
    verify_publication_id: bool = True,
) -> dict[str, object]:
    """Strictly validate and normalize a compact publication manifest."""

    if type(verify_publication_id) is not bool:
        raise ValueError("verify_publication_id must be a bool")
    source = _require_mapping(manifest, "manifest")
    _require_fields(source, _TOP_LEVEL_FIELDS, "manifest")

    schema = _require_string(source["schema"], "schema")
    if schema != F3_COMPACT_PUBLICATION_MANIFEST_SCHEMA:
        raise ValueError(f"schema must be {F3_COMPACT_PUBLICATION_MANIFEST_SCHEMA!r}")
    publication_id = _require_string(source["publication_id"], "publication_id")
    if verify_publication_id:
        _require_sha256(publication_id, "publication_id")
    elif not publication_id:
        raise ValueError("publication_id must not be empty")

    normalized: dict[str, object] = {
        "schema": schema,
        "publication_id": publication_id,
        "created_at_utc": _normalize_timestamp(source["created_at_utc"]),
        "code": _normalize_code(source["code"]),
        "environment": _normalize_environment(source["environment"]),
        "source": _normalize_source(source["source"]),
        "dataset": _normalize_dataset(source["dataset"]),
        "experiment": _normalize_experiment(source["experiment"]),
        "semantics": _normalize_semantics(source["semantics"]),
        "artifacts": _normalize_artifacts(source["artifacts"]),
    }
    if verify_publication_id:
        expected = _publication_id_from_normalized(normalized)
        if publication_id != expected:
            raise ValueError("publication_id does not match manifest content")
    return normalized


def write_manifest(
    root: str | PathLike[str],
    manifest: Mapping[str, object],
    *,
    pretty: bool = False,
) -> Path:
    """Atomically write a manifest after validating all recorded artifacts."""

    if type(pretty) is not bool:
        raise ValueError("pretty must be a bool")
    root_path = _require_root(root)
    destination = root_path / PUBLICATION_MANIFEST_FILENAME
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    normalized = validate_manifest(manifest)
    _validate_directory_contents(root_path, normalized, include_manifest=False)
    payload = _manifest_json_bytes(normalized, pretty=pretty)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=root_path,
        prefix=f".{PUBLICATION_MANIFEST_FILENAME}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    published = False
    try:
        os.close(descriptor)
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        published = True
        _fsync_directory(root_path)
        validate_compact_publication(root_path)
    except BaseException:
        _unlink_if_present(temporary)
        if published:
            _unlink_if_present(destination)
        raise
    return destination


def validate_compact_publication(root: str | PathLike[str]) -> Mapping[str, object]:
    """Validate a complete compact publication without consulting source data."""

    root_path = _require_root(root)
    manifest_path = root_path / PUBLICATION_MANIFEST_FILENAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"{PUBLICATION_MANIFEST_FILENAME} must be a regular non-symlink file")
    with manifest_path.open("r", encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=_reject_nonfinite_json)
    normalized = validate_manifest(value)
    _validate_directory_contents(root_path, normalized, include_manifest=True)
    return normalized


def _normalize_code(value: object) -> dict[str, object]:
    source = _require_mapping(value, "code")
    _require_fields(source, {"repository", "git_commit", "dirty"}, "code")
    repository = _require_nonempty_string(source["repository"], "code.repository")
    git_commit = _require_string(source["git_commit"], "code.git_commit")
    if _GIT_COMMIT_PATTERN.fullmatch(git_commit) is None:
        raise ValueError("code.git_commit must be 40-character lowercase hex")
    dirty = source["dirty"]
    if type(dirty) is not bool:
        raise ValueError("code.dirty must be a bool")
    return {"repository": repository, "git_commit": git_commit, "dirty": dirty}


def _normalize_environment(value: object) -> dict[str, object]:
    source = _require_mapping(value, "environment")
    _require_fields(source, {"python", "lock_file", "lock_sha256", "controls"}, "environment")
    controls = _require_mapping(source["controls"], "environment.controls")
    _require_fields(controls, _CONTROL_FIELDS, "environment.controls")
    lock_file = _normalize_safe_relative_path(source["lock_file"], "environment.lock_file")
    if "/" in lock_file:
        raise ValueError("environment.lock_file must be a basename")
    return {
        "python": _require_nonempty_string(source["python"], "environment.python"),
        "lock_file": lock_file,
        "lock_sha256": _require_sha256(source["lock_sha256"], "environment.lock_sha256"),
        "controls": {
            key: _require_string(controls[key], f"environment.controls.{key}")
            for key in sorted(_CONTROL_FIELDS)
        },
    }


def _normalize_source(value: object) -> dict[str, object]:
    source = _require_mapping(value, "source")
    _require_fields(source, {"f3_completion_sha256"}, "source")
    return {
        "f3_completion_sha256": _require_sha256(
            source["f3_completion_sha256"], "source.f3_completion_sha256"
        )
    }


def _normalize_dataset(value: object) -> dict[str, object]:
    source = _require_mapping(value, "dataset")
    _require_fields(source, {"dataset_id", "shape", "storage_dtype", "files"}, "dataset")
    shape = source["shape"]
    if type(shape) is not list or len(shape) != 3:
        raise ValueError("dataset.shape must be a three-element array")
    normalized_shape = [
        _require_positive_int(dimension, f"dataset.shape[{index}]")
        for index, dimension in enumerate(shape)
    ]
    if type(source["storage_dtype"]) is not str or source["storage_dtype"] != ">f4":
        raise ValueError("dataset.storage_dtype must be '>f4'")

    files = source["files"]
    if type(files) is not list or not files:
        raise ValueError("dataset.files must be a non-empty array")
    normalized_files: list[dict[str, object]] = []
    roles: set[str] = set()
    for index, item in enumerate(files):
        path = f"dataset.files[{index}]"
        record = _require_mapping(item, path)
        _require_fields(record, {"role", "filename", "size", "sha256"}, path)
        role = _require_nonempty_string(record["role"], f"{path}.role")
        if role in roles:
            raise ValueError("dataset file roles must be unique")
        roles.add(role)
        filename = _normalize_safe_relative_path(record["filename"], f"{path}.filename")
        if "/" in filename:
            raise ValueError(f"{path}.filename must be a basename")
        normalized_files.append(
            {
                "role": role,
                "filename": filename,
                "size": _require_positive_int(record["size"], f"{path}.size"),
                "sha256": _require_sha256(record["sha256"], f"{path}.sha256"),
            }
        )
    if roles != set(_DATASET_FILE_CONTRACT):
        missing = sorted(set(_DATASET_FILE_CONTRACT) - roles)
        unknown = sorted(roles - set(_DATASET_FILE_CONTRACT))
        raise ValueError(
            f"dataset file role coverage mismatch; missing={missing}, unknown={unknown}"
        )
    for record in normalized_files:
        role = cast(str, record["role"])
        if record["filename"] != _DATASET_FILE_CONTRACT[role]:
            raise ValueError(
                f"dataset file {role!r} must use filename {_DATASET_FILE_CONTRACT[role]!r}"
            )
    normalized_files.sort(key=lambda item: cast(str, item["role"]))
    return {
        "dataset_id": _require_nonempty_string(source["dataset_id"], "dataset.dataset_id"),
        "shape": normalized_shape,
        "storage_dtype": ">f4",
        "files": normalized_files,
    }


def _normalize_experiment(value: object) -> dict[str, object]:
    source = _require_mapping(value, "experiment")
    _require_fields(source, {"config_file", "config_sha256"}, "experiment")
    config_file = _normalize_safe_relative_path(source["config_file"], "experiment.config_file")
    return {
        "config_file": config_file,
        "config_sha256": _require_sha256(source["config_sha256"], "experiment.config_sha256"),
    }


def _normalize_semantics(value: object) -> dict[str, object]:
    source = _require_mapping(value, "semantics")
    fields = {
        "evaluation",
        "public_reference_is_geological_truth",
        "evaluation_units",
        "displayed_condition",
        "stage_order",
    }
    _require_fields(source, fields, "semantics")
    if source["evaluation"] != "f3_public_reference_agreement":
        raise ValueError("semantics.evaluation must be 'f3_public_reference_agreement'")
    if source["public_reference_is_geological_truth"] is not False:
        raise ValueError("semantics.public_reference_is_geological_truth must be false")
    if type(source["evaluation_units"]) is not int or source["evaluation_units"] != 1:
        raise ValueError("semantics.evaluation_units must be 1")
    if source["displayed_condition"] != "Q-QUAL":
        raise ValueError("semantics.displayed_condition must be 'Q-QUAL'")
    if type(source["stage_order"]) is not list or source["stage_order"] != _STAGE_ORDER:
        raise ValueError("semantics.stage_order must be ['ft', 'fv', 'fvt']")
    return {
        "evaluation": "f3_public_reference_agreement",
        "public_reference_is_geological_truth": False,
        "evaluation_units": 1,
        "displayed_condition": "Q-QUAL",
        "stage_order": list(_STAGE_ORDER),
    }


def _normalize_artifacts(value: object) -> list[dict[str, object]]:
    if type(value) is not list:
        raise ValueError("artifacts must be an array")
    normalized: list[dict[str, object]] = []
    paths: set[str] = set()
    for index, item in enumerate(value):
        path = f"artifacts[{index}]"
        record = _require_mapping(item, path)
        _require_fields(record, {"path", "tier", "role", "size", "sha256"}, path)
        artifact_path = _normalize_safe_relative_path(record["path"], f"{path}.path")
        if artifact_path == PUBLICATION_MANIFEST_FILENAME:
            raise ValueError(f"{PUBLICATION_MANIFEST_FILENAME} must not be an artifact")
        if artifact_path in paths:
            raise ValueError("artifact paths must be unique")
        paths.add(artifact_path)
        tier = _require_string(record["tier"], f"{path}.tier")
        if tier not in _ARTIFACT_TIERS:
            raise ValueError(f"{path}.tier must be 'primary' or 'derived'")
        role = _require_string(record["role"], f"{path}.role")
        if role not in _ARTIFACT_ROLES:
            raise ValueError(f"{path}.role is not allowed")
        normalized.append(
            {
                "path": artifact_path,
                "tier": tier,
                "role": role,
                "size": _require_positive_int(record["size"], f"{path}.size"),
                "sha256": _require_sha256(record["sha256"], f"{path}.sha256"),
            }
        )
    normalized.sort(key=lambda item: cast(str, item["path"]))
    return normalized


def _publication_id_from_normalized(manifest: Mapping[str, object]) -> str:
    artifacts = cast(list[dict[str, object]], manifest["artifacts"])
    identity = {
        "schema": manifest["schema"],
        "code": manifest["code"],
        "environment": manifest["environment"],
        "source": manifest["source"],
        "dataset": manifest["dataset"],
        "experiment": manifest["experiment"],
        "semantics": manifest["semantics"],
        "artifacts": [record for record in artifacts if record["tier"] == "primary"],
    }
    return hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()


def _validate_directory_contents(
    root: Path,
    manifest: Mapping[str, object],
    *,
    include_manifest: bool,
) -> None:
    artifacts = cast(list[dict[str, object]], manifest["artifacts"])
    by_path: dict[str, dict[str, object]] = {}
    for artifact in artifacts:
        relative_path = cast(str, artifact["path"])
        by_path[relative_path] = artifact
        size, sha256 = _hash_regular_file(root, relative_path)
        if size != artifact["size"]:
            raise ValueError(f"artifact size does not match manifest: {relative_path}")
        if sha256 != artifact["sha256"]:
            raise ValueError(f"artifact SHA-256 does not match manifest: {relative_path}")

    environment = cast(dict[str, object], manifest["environment"])
    experiment = cast(dict[str, object], manifest["experiment"])
    _require_linked_artifact(
        by_path,
        cast(str, environment["lock_file"]),
        cast(str, environment["lock_sha256"]),
        "environment_lock",
        "environment lock",
    )
    _require_linked_artifact(
        by_path,
        cast(str, experiment["config_file"]),
        cast(str, experiment["config_sha256"]),
        "resolved_experiment",
        "experiment config",
    )

    expected = set(by_path)
    if include_manifest:
        expected.add(PUBLICATION_MANIFEST_FILENAME)
    actual = _regular_file_paths(root)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            "publication directory has an invalid regular file set; "
            f"missing={missing}, unknown={unknown}"
        )


def _require_linked_artifact(
    artifacts: Mapping[str, Mapping[str, object]],
    relative_path: str,
    sha256: str,
    role: str,
    context: str,
) -> None:
    artifact = artifacts.get(relative_path)
    if artifact is None:
        raise ValueError(f"{context} must be listed as an artifact: {relative_path}")
    if artifact["role"] != role:
        raise ValueError(f"{context} artifact must have role {role!r}")
    if artifact["sha256"] != sha256:
        raise ValueError(f"{context} SHA-256 does not match its artifact entry")


def _require_root(root: str | PathLike[str]) -> Path:
    path = Path(root)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("publication root must be an existing non-symlink directory")
    return path


def _hash_regular_file(root: Path, relative_path: str) -> tuple[int, str]:
    path = root
    for component in relative_path.split("/")[:-1]:
        path /= component
        if path.is_symlink() or not path.is_dir():
            raise ValueError(
                f"artifact parent must be a regular non-symlink directory: {relative_path}"
            )
    path /= relative_path.split("/")[-1]
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {relative_path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_HASH_CHUNK_SIZE):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _regular_file_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                if entry.is_symlink():
                    raise ValueError(f"publication directory must not contain symlinks: {relative}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    paths.add(relative)
                else:
                    raise ValueError(
                        f"publication directory contains a non-regular entry: {relative}"
                    )
    return paths


def _canonical_json_bytes(value: object) -> bytes:
    _validate_json_value(value, "value")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _manifest_json_bytes(manifest: Mapping[str, object], *, pretty: bool) -> bytes:
    if not pretty:
        return _canonical_json_bytes(manifest) + b"\n"
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


def _validate_json_value(value: object, path: str) -> None:
    value_type = type(value)
    if value is None or value_type in {str, bool, int}:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{path} keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains a non-standard JSON value: {value_type.__name__}")


def _require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{path} keys must be strings")
    return value


def _require_fields(value: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{path} has invalid fields; missing={missing}, unknown={unknown}")


def _require_string(value: object, path: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{path} must be a string")
    return value


def _require_nonempty_string(value: object, path: str) -> str:
    result = _require_string(value, path)
    if not result:
        raise ValueError(f"{path} must not be empty")
    return result


def _require_positive_int(value: object, path: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _require_sha256(value: object, path: str) -> str:
    result = _require_string(value, path)
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{path} must be a lowercase SHA-256")
    return result


def _normalize_timestamp(value: object) -> str:
    timestamp = _require_string(value, "created_at_utc")
    if _TIMESTAMP_PATTERN.fullmatch(timestamp) is None:
        raise ValueError("created_at_utc must use YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("created_at_utc is not a valid UTC timestamp") from error
    return timestamp


def _normalize_safe_relative_path(value: object, path: str) -> str:
    result = _require_nonempty_string(value, path)
    if result.startswith("/"):
        raise ValueError(f"{path} must be relative")
    if "\\" in result:
        raise ValueError(f"{path} must use POSIX separators")
    if _WINDOWS_DRIVE_PATTERN.match(result):
        raise ValueError(f"{path} must not be a Windows drive path")
    if any(component in {"", ".", ".."} for component in result.split("/")):
        raise ValueError(f"{path} contains an unsafe path component")
    return result


def _artifact_sort_key(value: object) -> str:
    source = _require_mapping(value, "artifact")
    if "path" not in source:
        raise ValueError("artifact is missing path")
    return _require_string(source["path"], "artifact.path")


def _reject_nonfinite_json(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


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
