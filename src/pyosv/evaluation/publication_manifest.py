"""Minimal publication manifest contract for PyOSV results."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

PUBLICATION_MANIFEST_SCHEMA = "pyosv.publication_manifest.v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")

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


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value deterministically as UTF-8."""

    normalized = _json_value(value, "value")
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


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
    """Build and validate a canonical publication manifest."""

    artifact_items = [dict(item) for item in artifacts]
    ordered_artifacts = sorted(
        artifact_items,
        key=lambda item: _safe_path(item.get("path"), "artifact.path"),
    )
    draft: dict[str, object] = {
        "schema": PUBLICATION_MANIFEST_SCHEMA,
        "publication_id": "0" * 64,
        "created_at_utc": created_at_utc,
        "code": dict(code),
        "environment": dict(environment),
        "datasets": dict(datasets),
        "experiment": dict(experiment),
        "semantics": dict(semantics),
        "artifacts": ordered_artifacts,
    }
    normalized = validate_publication_manifest(draft, verify_publication_id=False)
    normalized["publication_id"] = compute_publication_id(normalized)
    return validate_publication_manifest(normalized)


def validate_publication_manifest(
    manifest: Mapping[str, object],
    *,
    verify_publication_id: bool = True,
) -> dict[str, object]:
    """Validate and normalize one publication manifest."""

    if not isinstance(manifest, Mapping):
        raise ValueError("publication manifest must be an object")
    _require_fields(manifest, _TOP_LEVEL_FIELDS, "publication manifest")

    schema = _string(manifest["schema"], "schema")
    if schema != PUBLICATION_MANIFEST_SCHEMA:
        raise ValueError(f"schema must equal {PUBLICATION_MANIFEST_SCHEMA!r}")
    publication_id = _sha256(manifest["publication_id"], "publication_id")
    created_at_utc = _timestamp(manifest["created_at_utc"])
    code = _code(manifest["code"])
    environment = _environment(manifest["environment"])
    datasets = _datasets(manifest["datasets"])
    experiment = _experiment(manifest["experiment"])
    semantics = _semantics(manifest["semantics"])
    artifacts = _artifacts(manifest["artifacts"])

    normalized: dict[str, object] = {
        "schema": schema,
        "publication_id": publication_id,
        "created_at_utc": created_at_utc,
        "code": code,
        "environment": environment,
        "datasets": datasets,
        "experiment": experiment,
        "semantics": semantics,
        "artifacts": artifacts,
    }
    canonical_json_bytes(normalized)
    if verify_publication_id:
        expected = compute_publication_id(normalized)
        if publication_id != expected:
            raise ValueError("publication_id does not match publication identity")
    return normalized


def compute_publication_id(manifest: Mapping[str, object]) -> str:
    """Return the SHA-256 identity of stable publication inputs and primary artifacts."""

    if not isinstance(manifest, Mapping):
        raise ValueError("publication manifest must be an object")
    required = _TOP_LEVEL_FIELDS - {"publication_id"}
    allowed = required if "publication_id" not in manifest else _TOP_LEVEL_FIELDS
    if set(manifest) != allowed:
        raise ValueError("publication identity input field set mismatch")
    primary_artifacts = [
        item for item in _artifacts(manifest["artifacts"]) if item["tier"] == "primary"
    ]
    schema = _string(manifest["schema"], "schema")
    if schema != PUBLICATION_MANIFEST_SCHEMA:
        raise ValueError(f"schema must equal {PUBLICATION_MANIFEST_SCHEMA!r}")
    payload = {
        "schema": schema,
        "code": _code(manifest["code"]),
        "environment": _environment(manifest["environment"]),
        "datasets": _datasets(manifest["datasets"]),
        "experiment": _experiment(manifest["experiment"]),
        "semantics": _semantics(manifest["semantics"]),
        "artifacts": primary_artifacts,
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _code(value: object) -> dict[str, object]:
    obj = _object(value, {"repository", "git_commit", "dirty"}, "code")
    return {
        "repository": _string(obj["repository"], "code.repository"),
        "git_commit": _git_sha(obj["git_commit"], "code.git_commit"),
        "dirty": _bool(obj["dirty"], "code.dirty"),
    }


def _environment(value: object) -> dict[str, object]:
    obj = _object(value, {"python", "lock_file", "lock_sha256", "controls"}, "environment")
    controls = _object(obj["controls"], _CONTROL_FIELDS, "environment.controls")
    return {
        "python": _string(obj["python"], "environment.python"),
        "lock_file": _safe_path(obj["lock_file"], "environment.lock_file"),
        "lock_sha256": _sha256(obj["lock_sha256"], "environment.lock_sha256"),
        "controls": {
            name: _string(controls[name], f"environment.controls.{name}")
            for name in sorted(_CONTROL_FIELDS)
        },
    }


def _datasets(value: object) -> dict[str, object]:
    obj = _object(value, {"f3"}, "datasets")
    f3 = _object(obj["f3"], {"dataset_id", "shape", "dtype", "files"}, "datasets.f3")
    dtype = _string(f3["dtype"], "datasets.f3.dtype")
    if dtype != ">f4":
        raise ValueError("datasets.f3.dtype must equal '>f4'")
    files_value = f3["files"]
    if not isinstance(files_value, list) or not files_value:
        raise ValueError("datasets.f3.files must be a non-empty array")
    files: list[dict[str, object]] = []
    roles: set[str] = set()
    for index, item in enumerate(files_value):
        context = f"datasets.f3.files[{index}]"
        file_obj = _object(item, {"role", "filename", "size", "sha256"}, context)
        role = _string(file_obj["role"], f"{context}.role")
        if role in roles:
            raise ValueError(f"duplicate F3 file role: {role}")
        roles.add(role)
        files.append(
            {
                "role": role,
                "filename": _basename(file_obj["filename"], f"{context}.filename"),
                "size": _positive_int(file_obj["size"], f"{context}.size"),
                "sha256": _sha256(file_obj["sha256"], f"{context}.sha256"),
            }
        )
    return {
        "f3": {
            "dataset_id": _string(f3["dataset_id"], "datasets.f3.dataset_id"),
            "shape": _shape(f3["shape"]),
            "dtype": dtype,
            "files": files,
        }
    }


def _experiment(value: object) -> dict[str, object]:
    obj = _object(value, {"config_file", "config_sha256", "source_runs"}, "experiment")
    runs = _object(obj["source_runs"], {"synthetic", "f3"}, "experiment.source_runs")
    normalized_runs: dict[str, object] = {}
    for name in ("synthetic", "f3"):
        run = _object(runs[name], {"completion_sha256"}, f"experiment.source_runs.{name}")
        normalized_runs[name] = {
            "completion_sha256": _sha256(
                run["completion_sha256"], f"experiment.source_runs.{name}.completion_sha256"
            )
        }
    return {
        "config_file": _safe_path(obj["config_file"], "experiment.config_file"),
        "config_sha256": _sha256(obj["config_sha256"], "experiment.config_sha256"),
        "source_runs": normalized_runs,
    }


def _semantics(value: object) -> dict[str, object]:
    fields = {
        "synthetic",
        "f3",
        "f3_public_reference_is_geological_truth",
        "f3_evaluation_units",
    }
    obj = _object(value, fields, "semantics")
    if obj["synthetic"] != "known_truth":
        raise ValueError("semantics.synthetic must equal 'known_truth'")
    if obj["f3"] != "public_reference_agreement":
        raise ValueError("semantics.f3 must equal 'public_reference_agreement'")
    if obj["f3_public_reference_is_geological_truth"] is not False:
        raise ValueError("F3 public reference must not be declared geological truth")
    return {
        "synthetic": "known_truth",
        "f3": "public_reference_agreement",
        "f3_public_reference_is_geological_truth": False,
        "f3_evaluation_units": _positive_int(
            obj["f3_evaluation_units"], "semantics.f3_evaluation_units"
        ),
    }


def _artifacts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError("artifacts must be an array")
    result: list[dict[str, object]] = []
    paths: set[str] = set()
    for index, item in enumerate(value):
        context = f"artifacts[{index}]"
        obj = _object(item, {"path", "tier", "role", "size", "sha256"}, context)
        path = _safe_path(obj["path"], f"{context}.path")
        if path == "publication_manifest.json":
            raise ValueError("publication_manifest.json must not list itself as an artifact")
        if path in paths:
            raise ValueError(f"duplicate artifact path: {path}")
        paths.add(path)
        tier = _string(obj["tier"], f"{context}.tier")
        if tier not in {"primary", "derived"}:
            raise ValueError(f"{context}.tier must be 'primary' or 'derived'")
        result.append(
            {
                "path": path,
                "tier": tier,
                "role": _string(obj["role"], f"{context}.role"),
                "size": _positive_int(obj["size"], f"{context}.size"),
                "sha256": _sha256(obj["sha256"], f"{context}.sha256"),
            }
        )
    if [item["path"] for item in result] != sorted(item["path"] for item in result):
        raise ValueError("artifacts must be sorted by path")
    return result


def _object(value: object, fields: set[str], context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    _require_fields(value, fields, context)
    return value


def _require_fields(value: Mapping[str, object], fields: set[str], context: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{context} field set mismatch")


def _string(value: object, context: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _bool(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be a bool")
    return value


def _positive_int(value: object, context: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


def _shape(value: object) -> list[int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError("datasets.f3.shape must contain exactly three dimensions")
    return [
        _positive_int(item, f"datasets.f3.shape[{index}]") for index, item in enumerate(value)
    ]


def _sha256(value: object, context: str) -> str:
    text = _string(value, context)
    if _SHA256_RE.fullmatch(text) is None:
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return text


def _git_sha(value: object, context: str) -> str:
    text = _string(value, context)
    if _GIT_SHA_RE.fullmatch(text) is None:
        raise ValueError(f"{context} must be a 40-character lowercase Git SHA")
    return text


def _timestamp(value: object) -> str:
    text = _string(value, "created_at_utc")
    if _TIMESTAMP_RE.fullmatch(text) is None:
        raise ValueError("created_at_utc must use YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError("created_at_utc is not a valid UTC timestamp") from error
    return text


def _safe_path(value: object, context: str) -> str:
    path = _string(value, context)
    if (
        path.startswith("/")
        or "\\" in path
        or _WINDOWS_DRIVE_RE.match(path) is not None
        or any(part in {"", ".", ".."} for part in path.split("/"))
    ):
        raise ValueError(f"{context} must be a safe POSIX relative path")
    return path


def _basename(value: object, context: str) -> str:
    filename = _string(value, context)
    if "/" in filename or "\\" in filename or filename in {".", ".."}:
        raise ValueError(f"{context} must be a basename")
    return filename


def _json_value(value: object, context: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError(f"{context} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{context} object keys must be strings")
            result[key] = _json_value(item, f"{context}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{context}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{context} contains a non-JSON value")
