"""Read-only loading and identity checks for completed source bundles."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..f3d_mode_comparison import (
    F3_FILE_ROLES,
    F3DatasetSpec,
    F3VolumeSource,
    load_f3d_mode_comparison_result,
    validate_completed_f3d_bundle,
)
from ..f3d_mode_comparison.artifacts import RUN_COMPLETION_FILE, RUN_MANIFEST_FILE
from ..f3d_mode_comparison.metrics import F3_REFERENCE_STAGE_ROLES, MetricEvidence
from ..synthetic_mode_comparison import validate_completed_bundle
from ..synthetic_mode_comparison.contrasts import ContrastRow
from ..synthetic_mode_comparison.experiment import RuntimeRow
from ..synthetic_mode_comparison.metrics import MetricRow

from .models import F3SourceBundle, SyntheticSourceBundle
from .semantic import (
    F3_SOURCE_IDENTITY_FIELDS,
    SYNTHETIC_SOURCE_IDENTITY_FIELDS,
    canonical_digest,
    source_identity_object,
)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value {value!r} in {path.name}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"cannot read finite JSON object {path}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(path: Path) -> tuple[dict[str, str], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            text = stream.read()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read source CSV {path}") from error
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        return tuple(dict(row) for row in reader)
    except csv.Error as error:
        raise ValueError(f"malformed source CSV {path}") from error


def _nullable_int(value: str, name: str) -> int | None:
    if value == "":
        return None
    try:
        return int(value, 10)
    except ValueError as error:
        raise ValueError(f"invalid integer field {name}") from error


def _int(value: str, name: str) -> int:
    parsed = _nullable_int(value, name)
    if parsed is None:
        raise ValueError(f"{name} must not be empty")
    return parsed


def _nullable_float(value: str, name: str) -> float | None:
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"invalid number field {name}") from error
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _float(value: str, name: str) -> float:
    parsed = _nullable_float(value, name)
    if parsed is None:
        raise ValueError(f"{name} must not be empty")
    return parsed


def _bool(value: str, name: str) -> bool:
    if value not in {"true", "false"}:
        raise ValueError(f"invalid boolean field {name}")
    return value == "true"


def _json_tuple(value: str, name: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON tuple field {name}") from error
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ValueError(f"invalid JSON tuple field {name}")
    return tuple(parsed)


def _require_header(
    records: tuple[dict[str, str], ...], expected: tuple[str, ...], path: Path
) -> None:
    # DictReader retains the header on the first read only, so read it directly here.
    with path.open("r", encoding="utf-8", newline="") as stream:
        header = next(csv.reader(stream), None)
    if header != list(expected):
        raise ValueError(f"source CSV header mismatch in {path}")


def _load_synthetic_metric_rows(path: Path) -> tuple[MetricRow, ...]:
    records = _records(path)
    expected = tuple(field.name for field in fields(MetricRow))
    _require_header(records, expected, path)
    output = []
    for row in records:
        output.append(
            MetricRow(
                schema_version=_int(row["schema_version"], "schema_version"),
                case_id=row["case_id"],
                trial_id=row["trial_id"],
                seed=_nullable_int(row["seed"], "seed"),
                scope=row["scope"],
                cell_label=row["cell_label"],
                input_mode=row["input_mode"],
                scanner_backend=row["scanner_backend"] or None,
                scanner_refinement_factor=_nullable_int(
                    row["scanner_refinement_factor"], "scanner_refinement_factor"
                ),
                scanner_thin_mode=row["scanner_thin_mode"] or None,
                workflow_mode=row["workflow_mode"] or None,
                voter_thin_mode=row["voter_thin_mode"] or None,
                skinner_method=row["skinner_method"] or None,
                variant=row["variant"],
                stage=row["stage"],
                selection=row["selection"],
                metric=row["metric"],
                value=_float(row["value"], "value"),
                unit=row["unit"],
                direction=row["direction"],
                contrast_eligible=_bool(row["contrast_eligible"], "contrast_eligible"),
            )
        )
    return tuple(output)


def _load_synthetic_contrast_rows(path: Path) -> tuple[ContrastRow, ...]:
    records = _records(path)
    expected = tuple(field.name for field in fields(ContrastRow))
    _require_header(records, expected, path)
    output = []
    for row in records:
        output.append(
            ContrastRow(
                contrast_name=row["contrast_name"],
                case_id=row["case_id"],
                trial_id=row["trial_id"],
                seed=_nullable_int(row["seed"], "seed"),
                stage=row["stage"],
                selection=row["selection"],
                metric=row["metric"],
                unit=row["unit"],
                direction=row["direction"],
                component_cells=_json_tuple(row["component_cells"], "component_cells"),
                raw_value=_float(row["raw_value"], "raw_value"),
                improvement_value=_nullable_float(row["improvement_value"], "improvement_value"),
            )
        )
    return tuple(output)


def _load_synthetic_runtime_rows(path: Path) -> tuple[RuntimeRow, ...]:
    records = _records(path)
    expected = tuple(field.name for field in fields(RuntimeRow))
    _require_header(records, expected, path)
    output = []
    for row in records:
        output.append(
            RuntimeRow(
                case_id=row["case_id"] or None,
                trial_id=row["trial_id"] or None,
                seed=_nullable_int(row["seed"], "seed"),
                stage=row["stage"],
                cell_label=row["cell_label"] or None,
                scanner_backend=row["scanner_backend"] or None,
                elapsed_seconds=_float(row["elapsed_seconds"], "elapsed_seconds"),
                call_count=_int(row["call_count"], "call_count"),
                shared_stage=_bool(row["shared_stage"], "shared_stage"),
            )
        )
    return tuple(output)


def load_synthetic_source(path: str | Path) -> SyntheticSourceBundle:
    """Validate and load only scalar evidence from a completed synthetic bundle."""

    bundle = Path(path)
    validate_completed_bundle(bundle)
    manifest_path = bundle / "manifest.json"
    completion_path = bundle / "completion.json"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("synthetic manifest must be an object")
    metric_rows = _load_synthetic_metric_rows(bundle / "metrics_long.csv")
    contrast_rows = _load_synthetic_contrast_rows(bundle / "contrasts.csv")
    runtime_rows = _load_synthetic_runtime_rows(bundle / "runtime.csv")
    input_config = manifest.get("input_config")
    skinning_config = (
        input_config.get("skinning_config") if isinstance(input_config, dict) else None
    )
    explicit_skinning = (
        skinning_config.get("enabled") if isinstance(skinning_config, dict) else None
    )
    skinning_enabled = (
        bool(explicit_skinning)
        if explicit_skinning is not None
        else any(row.stage == "skin" for row in metric_rows)
    )
    case_order_value = manifest.get("case_order")
    if not isinstance(case_order_value, list) or any(
        not isinstance(item, str) for item in case_order_value
    ):
        raise ValueError("synthetic manifest case_order is invalid")
    identity = {
        "artifact_schema_version": manifest.get("artifact_schema_version"),
        "scalar_evidence_contract_version": manifest.get("scalar_evidence_contract_version"),
        "runtime_contract_version": manifest.get("runtime_contract_version"),
        "metric_schema_version": manifest.get("metric_schema_version"),
        "manifest_sha256": _sha256(manifest_path),
        "completion_sha256": _sha256(completion_path),
    }
    return SyntheticSourceBundle(
        bundle.resolve(),
        manifest,
        identity["completion_sha256"],
        identity["manifest_sha256"],
        canonical_digest(source_identity_object(identity, SYNTHETIC_SOURCE_IDENTITY_FIELDS)),
        metric_rows,
        contrast_rows,
        runtime_rows,
        skinning_enabled,
        tuple(case_order_value),
    )


def _derive_f3_dataset_spec(manifest: Mapping[str, Any]) -> F3DatasetSpec:
    identity = manifest.get("dataset_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("F3 run manifest dataset_identity is missing")
    dataset_id = identity.get("dataset_id")
    files = identity.get("files")
    if not isinstance(dataset_id, str) or not isinstance(files, list) or not files:
        raise ValueError("F3 dataset identity is invalid")
    provenance = manifest.get("provenance")
    provenance_files = provenance.get("dataset_files") if isinstance(provenance, Mapping) else None
    filename_by_role = {role: filename for role, filename in F3_FILE_ROLES}
    if isinstance(provenance_files, list):
        for item in provenance_files:
            if isinstance(item, Mapping) and isinstance(item.get("role"), str):
                filename = item.get("filename")
                if isinstance(filename, str) and filename:
                    filename_by_role[item["role"]] = filename
    roles: list[tuple[str, str]] = []
    shape: tuple[int, int, int] | None = None
    dtype: str | None = None
    expected_bytes: int | None = None
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("F3 dataset file identity is invalid")
        role = item.get("role")
        item_shape = item.get("shape")
        item_dtype = item.get("storage_dtype")
        item_size = item.get("size")
        if (
            not isinstance(role, str)
            or not isinstance(item_shape, list)
            or len(item_shape) != 3
            or any(not isinstance(value, int) or value <= 0 for value in item_shape)
            or not isinstance(item_dtype, str)
            or not isinstance(item_size, int)
        ):
            raise ValueError("F3 dataset file layout identity is invalid")
        current_shape = tuple(item_shape)
        current_dtype = np.dtype(item_dtype).str
        if shape is None:
            shape, dtype, expected_bytes = current_shape, current_dtype, item_size
        elif (shape, dtype, expected_bytes) != (current_shape, current_dtype, item_size):
            raise ValueError("F3 dataset files have mixed layouts")
        try:
            filename = filename_by_role[role]
        except KeyError as error:
            raise ValueError(f"no F3 filename mapping for role {role!r}") from error
        roles.append((role, filename))
    if shape is None or dtype is None or expected_bytes is None:
        raise ValueError("F3 dataset identity has no file layout")
    return F3DatasetSpec(
        dataset_id=dataset_id,
        shape=shape,
        storage_dtype=dtype,
        files=tuple(roles),
        expected_bytes=expected_bytes,
    )


def _validate_f3_data_identity(
    manifest: Mapping[str, Any], data_root: Path, spec: F3DatasetSpec
) -> Mapping[str, Any]:
    expected_identity = manifest.get("dataset_identity")
    if not isinstance(expected_identity, Mapping):
        raise ValueError("F3 run manifest has no dataset identity")
    with F3VolumeSource(data_root, spec=spec) as source:
        actual_identity = source.identity.computation_identity
        if actual_identity != dict(expected_identity):
            raise ValueError(
                "F3 data root identity does not match the completed bundle: "
                "dataset ID, layout, required roles, size, or SHA-256 differs"
            )
        if source.identity.dataset_id != spec.dataset_id:
            raise ValueError("F3 data root dataset ID does not match the bundle")
        for stage, role in F3_REFERENCE_STAGE_ROLES.items():
            if source.identity.file_for(role).filename != spec.filename_for(role):
                raise ValueError(f"F3 public reference mapping for {stage} is inconsistent")
    return dict(actual_identity)


def load_f3_source(
    path: str | Path,
    data_root: str | Path,
) -> F3SourceBundle:
    """Validate an F3 result and checksum the external public-reference dataset."""

    bundle = Path(path)
    manifest_path = bundle / RUN_MANIFEST_FILE
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("F3 run manifest must be an object")
    spec = _derive_f3_dataset_spec(manifest)
    # The existing domain validator remains the source schema authority.  The
    # injected spec is only needed for small, non-official fixture datasets.
    validate_completed_f3d_bundle(bundle, _dataset_spec=spec)
    result = load_f3d_mode_comparison_result(bundle, _dataset_spec=spec)
    root = Path(data_root).resolve(strict=False)
    dataset_identity = _validate_f3_data_identity(manifest, root, spec)
    evidence = tuple(result.metric_evidence)
    if not all(isinstance(item, MetricEvidence) for item in evidence):
        raise ValueError("F3 metric evidence has an invalid type")
    completion_path = bundle / RUN_COMPLETION_FILE
    completion = _read_json(completion_path)
    if not isinstance(completion, dict) or not isinstance(
        completion.get("result_schema_version"), int
    ):
        raise ValueError("F3 completion result schema version is missing")
    identity = {
        "artifact_schema_version": manifest.get("artifact_schema_version"),
        "result_schema_version": completion["result_schema_version"],
        "run_fingerprint": manifest.get("run_fingerprint"),
        "dataset_identity": dataset_identity,
        "manifest_sha256": _sha256(manifest_path),
        "completion_sha256": _sha256(completion_path),
    }
    return F3SourceBundle(
        bundle.resolve(),
        root,
        spec,
        manifest,
        identity["completion_sha256"],
        identity["manifest_sha256"],
        canonical_digest(source_identity_object(identity, F3_SOURCE_IDENTITY_FIELDS)),
        result,
        evidence,
        dataset_identity,
        completion["result_schema_version"],
    )


__all__ = ["load_f3_source", "load_synthetic_source"]
