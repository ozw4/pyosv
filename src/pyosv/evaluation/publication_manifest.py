"""Pure data contract for publication manifest v1."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime

PUBLICATION_MANIFEST_SCHEMA = "pyosv.publication_manifest.v1"

_TOP_LEVEL_FIELDS = {
    "schema",
    "publication_id",
    "created_at_utc",
    "code",
    "environment",
    "datasets",
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
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_TIMESTAMP_PATTERN = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z\Z")
_WINDOWS_DRIVE_PATTERN = re.compile(r"[A-Za-z]:")

__all__ = [
    "PUBLICATION_MANIFEST_SCHEMA",
    "build_publication_manifest",
    "canonical_json_bytes",
    "compute_publication_id",
    "validate_publication_manifest",
]


def canonical_json_bytes(value: object) -> bytes:
    """Serialize standard JSON values to deterministic UTF-8 bytes."""
    _validate_json_value(value, "value")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def compute_publication_id(
    manifest_without_publication_id: Mapping[str, object],
) -> str:
    """Compute the identity from provenance and primary artifacts."""
    source = _require_mapping(manifest_without_publication_id, "manifest")
    if "publication_id" in source:
        raise ValueError("manifest must not contain publication_id")

    candidate = dict(source)
    candidate["publication_id"] = "pending"
    normalized = validate_publication_manifest(candidate, verify_publication_id=False)
    return _compute_publication_id_from_normalized(normalized)


def validate_publication_manifest(
    manifest: Mapping[str, object],
    *,
    verify_publication_id: bool = True,
) -> dict[str, object]:
    """Strictly validate and normalize a publication manifest."""
    if type(verify_publication_id) is not bool:
        raise ValueError("verify_publication_id must be a bool")

    source = _require_mapping(manifest, "manifest")
    _require_fields(source, _TOP_LEVEL_FIELDS, "manifest")

    schema = _require_string(source["schema"], "schema")
    if schema != PUBLICATION_MANIFEST_SCHEMA:
        raise ValueError(f"schema must be {PUBLICATION_MANIFEST_SCHEMA!r}")

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
        "datasets": _normalize_datasets(source["datasets"]),
        "experiment": _normalize_experiment(source["experiment"]),
        "semantics": _normalize_semantics(source["semantics"]),
        "artifacts": _normalize_artifacts(source["artifacts"]),
    }

    if verify_publication_id:
        expected = _compute_publication_id_from_normalized(normalized)
        if publication_id != expected:
            raise ValueError("publication_id does not match manifest content")

    return normalized


def build_publication_manifest(
    *,
    created_at_utc: str,
    code: Mapping[str, object],
    environment: Mapping[str, object],
    datasets: Mapping[str, object],
    experiment: Mapping[str, object],
    semantics: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build, identify, and validate a publication manifest."""
    if isinstance(artifacts, (str, bytes)) or not isinstance(artifacts, Sequence):
        raise ValueError("artifacts must be a sequence")
    sorted_artifacts = sorted(artifacts, key=_artifact_sort_key)

    without_publication_id: dict[str, object] = {
        "schema": PUBLICATION_MANIFEST_SCHEMA,
        "created_at_utc": created_at_utc,
        "code": code,
        "environment": environment,
        "datasets": datasets,
        "experiment": experiment,
        "semantics": semantics,
        "artifacts": sorted_artifacts,
    }
    publication_id = compute_publication_id(without_publication_id)
    manifest = {**without_publication_id, "publication_id": publication_id}
    return validate_publication_manifest(manifest)


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


def _normalize_code(value: object) -> dict[str, object]:
    source = _require_mapping(value, "code")
    _require_fields(source, {"repository", "git_commit", "dirty"}, "code")
    repository = _require_nonempty_string(source["repository"], "code.repository")
    git_commit = _require_string(source["git_commit"], "code.git_commit")
    if _GIT_COMMIT_PATTERN.fullmatch(git_commit) is None:
        raise ValueError("code.git_commit must be 40-character lowercase hex")
    if type(source["dirty"]) is not bool:
        raise ValueError("code.dirty must be a bool")
    return {"repository": repository, "git_commit": git_commit, "dirty": source["dirty"]}


def _normalize_environment(value: object) -> dict[str, object]:
    source = _require_mapping(value, "environment")
    _require_fields(source, {"python", "lock_file", "lock_sha256", "controls"}, "environment")
    controls = _require_mapping(source["controls"], "environment.controls")
    _require_fields(controls, _CONTROL_FIELDS, "environment.controls")
    normalized_controls = {
        key: _require_string(controls[key], f"environment.controls.{key}")
        for key in sorted(_CONTROL_FIELDS)
    }
    return {
        "python": _require_nonempty_string(source["python"], "environment.python"),
        "lock_file": _normalize_safe_relative_path(source["lock_file"], "environment.lock_file"),
        "lock_sha256": _require_sha256(source["lock_sha256"], "environment.lock_sha256"),
        "controls": normalized_controls,
    }


def _normalize_datasets(value: object) -> dict[str, object]:
    source = _require_mapping(value, "datasets")
    _require_fields(source, {"f3"}, "datasets")
    f3 = _require_mapping(source["f3"], "datasets.f3")
    _require_fields(f3, {"dataset_id", "shape", "dtype", "files"}, "datasets.f3")

    shape = f3["shape"]
    if type(shape) is not list or len(shape) != 3:
        raise ValueError("datasets.f3.shape must be a three-element array")
    normalized_shape = [
        _require_positive_int(dimension, f"datasets.f3.shape[{index}]")
        for index, dimension in enumerate(shape)
    ]
    if f3["dtype"] != ">f4" or type(f3["dtype"]) is not str:
        raise ValueError("datasets.f3.dtype must be '>f4'")

    files = f3["files"]
    if type(files) is not list or not files:
        raise ValueError("datasets.f3.files must be a non-empty array")
    normalized_files: list[dict[str, object]] = []
    roles: set[str] = set()
    for index, item in enumerate(files):
        item_path = f"datasets.f3.files[{index}]"
        file_source = _require_mapping(item, item_path)
        _require_fields(file_source, {"role", "filename", "size", "sha256"}, item_path)
        role = _require_nonempty_string(file_source["role"], f"{item_path}.role")
        if role in roles:
            raise ValueError("datasets.f3.files roles must be unique")
        roles.add(role)
        filename = _normalize_safe_relative_path(file_source["filename"], f"{item_path}.filename")
        if "/" in filename:
            raise ValueError(f"{item_path}.filename must be a basename")
        normalized_files.append(
            {
                "role": role,
                "filename": filename,
                "size": _require_positive_int(file_source["size"], f"{item_path}.size"),
                "sha256": _require_sha256(file_source["sha256"], f"{item_path}.sha256"),
            }
        )
    normalized_files.sort(key=lambda item: item["role"])

    return {
        "f3": {
            "dataset_id": _require_nonempty_string(f3["dataset_id"], "datasets.f3.dataset_id"),
            "shape": normalized_shape,
            "dtype": ">f4",
            "files": normalized_files,
        }
    }


def _normalize_experiment(value: object) -> dict[str, object]:
    source = _require_mapping(value, "experiment")
    _require_fields(source, {"config_file", "config_sha256", "source_runs"}, "experiment")
    source_runs = _require_mapping(source["source_runs"], "experiment.source_runs")
    _require_fields(source_runs, {"synthetic", "f3"}, "experiment.source_runs")

    normalized_runs: dict[str, object] = {}
    for run_name in ("synthetic", "f3"):
        run_path = f"experiment.source_runs.{run_name}"
        run = _require_mapping(source_runs[run_name], run_path)
        _require_fields(run, {"completion_sha256"}, run_path)
        normalized_runs[run_name] = {
            "completion_sha256": _require_sha256(
                run["completion_sha256"], f"{run_path}.completion_sha256"
            )
        }

    return {
        "config_file": _normalize_safe_relative_path(
            source["config_file"], "experiment.config_file"
        ),
        "config_sha256": _require_sha256(source["config_sha256"], "experiment.config_sha256"),
        "source_runs": normalized_runs,
    }


def _normalize_semantics(value: object) -> dict[str, object]:
    source = _require_mapping(value, "semantics")
    fields = {
        "synthetic",
        "f3",
        "f3_public_reference_is_geological_truth",
        "f3_evaluation_units",
    }
    _require_fields(source, fields, "semantics")
    if type(source["synthetic"]) is not str or source["synthetic"] != "known_truth":
        raise ValueError("semantics.synthetic must be 'known_truth'")
    if type(source["f3"]) is not str or source["f3"] != "public_reference_agreement":
        raise ValueError("semantics.f3 must be 'public_reference_agreement'")
    if source["f3_public_reference_is_geological_truth"] is not False:
        raise ValueError("semantics.f3_public_reference_is_geological_truth must be false")
    return {
        "synthetic": "known_truth",
        "f3": "public_reference_agreement",
        "f3_public_reference_is_geological_truth": False,
        "f3_evaluation_units": _require_positive_int(
            source["f3_evaluation_units"], "semantics.f3_evaluation_units"
        ),
    }


def _artifact_sort_key(value: object) -> str:
    source = _require_mapping(value, "artifact")
    if "path" not in source:
        raise ValueError("artifact is missing path")
    return _require_string(source["path"], "artifact.path")


def _normalize_artifacts(value: object) -> list[dict[str, object]]:
    if type(value) is not list:
        raise ValueError("artifacts must be an array")
    normalized: list[dict[str, object]] = []
    paths: list[str] = []
    for index, item in enumerate(value):
        item_path = f"artifacts[{index}]"
        source = _require_mapping(item, item_path)
        _require_fields(source, {"path", "tier", "role", "size", "sha256"}, item_path)
        artifact_path = _normalize_safe_relative_path(source["path"], f"{item_path}.path")
        if artifact_path == "publication_manifest.json":
            raise ValueError("publication_manifest.json must not be an artifact")
        tier = _require_string(source["tier"], f"{item_path}.tier")
        if tier not in {"primary", "derived"}:
            raise ValueError(f"{item_path}.tier must be 'primary' or 'derived'")
        paths.append(artifact_path)
        normalized.append(
            {
                "path": artifact_path,
                "tier": tier,
                "role": _require_nonempty_string(source["role"], f"{item_path}.role"),
                "size": _require_positive_int(source["size"], f"{item_path}.size"),
                "sha256": _require_sha256(source["sha256"], f"{item_path}.sha256"),
            }
        )

    if len(paths) != len(set(paths)):
        raise ValueError("artifact paths must be unique")
    if paths != sorted(paths):
        raise ValueError("artifacts must be sorted by path")
    return normalized


def _compute_publication_id_from_normalized(manifest: Mapping[str, object]) -> str:
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, list)
    identity = {
        "schema": manifest["schema"],
        "code": manifest["code"],
        "environment": manifest["environment"],
        "datasets": manifest["datasets"],
        "experiment": manifest["experiment"],
        "semantics": manifest["semantics"],
        "artifacts": [artifact for artifact in artifacts if artifact["tier"] == "primary"],
    }
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
