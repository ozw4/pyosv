"""Canonical semantic serialization for publication artifacts.

The publication writer and validate-only reader deliberately share these helpers.
File hashes in ``completion.json`` protect bytes; these helpers protect the
typed meaning and order of the CSV and JSON contracts.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from .config import PNG_MAX_DIMENSION

CSVType = Literal[
    "string",
    "nullable_string",
    "integer",
    "nullable_integer",
    "number",
    "nullable_number",
    "boolean",
    "json",
    "nullable_json",
]

# Kept beside the parser rather than duplicated by the figure writer and
# validate-only reader.  The mapping order is the figure-data CSV header
# order.  Nullable fields serialize an absent value as an empty CSV field,
# which remains distinct from numeric zero in the semantic digest.
FIGURE_DATA_FIELD_TYPES: dict[str, CSVType] = {
    "figure_id": "string",
    "dataset": "string",
    "evaluation_semantics": "string",
    "source_metric": "nullable_string",
    "source_stage": "nullable_string",
    "case_or_region": "nullable_string",
    "trial_id": "nullable_string",
    "seed": "nullable_integer",
    "cell_label": "nullable_string",
    "panel_label": "nullable_string",
    "metric": "nullable_string",
    "value": "nullable_number",
    "raw_improvement": "nullable_number",
    "normalized_value": "nullable_number",
    "unit": "nullable_string",
    "direction": "nullable_string",
    "axis": "nullable_string",
    "slice_index": "nullable_integer",
    "slice_selection_policy": "nullable_string",
    "slice_score": "nullable_number",
    "selection_threshold": "nullable_number",
    "vmin": "nullable_number",
    "vmax": "nullable_number",
    "scale_policy": "nullable_string",
    "colormap": "nullable_string",
    "difference_limit": "nullable_number",
    "difference_vmin": "nullable_number",
    "difference_vmax": "nullable_number",
}

# The root-table parsers are intentionally centralized here.  Writer-side
# table contracts and validate-only CSV reparsing therefore share the exact
# same nullability and scalar type rules.
ROOT_TABLE_FIELD_TYPES: dict[str, dict[str, CSVType]] = {
    "publication_metrics.csv": {
        "dataset": "string",
        "evaluation_semantics": "string",
        "case_or_region": "string",
        "trial_id": "nullable_string",
        "seed": "nullable_integer",
        "cell_label": "string",
        "scanner_backend": "string",
        "workflow_mode": "nullable_string",
        "stage": "string",
        "selection": "string",
        "metric": "string",
        "value": "nullable_number",
        "unit": "string",
        "direction": "string",
        "source_artifact": "string",
    },
    "publication_contrasts.csv": {
        "dataset": "string",
        "evaluation_semantics": "string",
        "case_or_region": "string",
        "trial_id": "nullable_string",
        "seed": "nullable_integer",
        "contrast_name": "string",
        "stage": "string",
        "selection": "string",
        "metric": "string",
        "raw_value": "number",
        "improvement_value": "nullable_number",
        "unit": "string",
        "direction": "string",
        "component_cells": "json",
        "source_artifact": "string",
    },
    "publication_summary.csv": {
        "dataset": "string",
        "evaluation_semantics": "string",
        "case_or_region": "string",
        "stage": "string",
        "selection": "string",
        "metric": "string",
        "cell_label": "string",
        "n": "integer",
        "mean": "number",
        "median": "number",
        "minimum": "number",
        "maximum": "number",
        "q25": "number",
        "q75": "number",
        "unit": "string",
        "direction": "string",
    },
    "f3_regional_summary.csv": {
        "dataset": "string",
        "evaluation_semantics": "string",
        "case_or_region": "string",
        "stage": "string",
        "cell_label": "string",
        "scanner_backend": "string",
        "workflow_mode": "string",
        "region": "string",
        "metric": "string",
        "display_label": "string",
        "value": "nullable_number",
        "unit": "string",
        "source_artifact": "string",
    },
    "f3_orientation_summary.csv": {
        "dataset": "string",
        "evaluation_semantics": "string",
        "case_or_region": "string",
        "stage": "string",
        "left_cell": "string",
        "right_cell": "string",
        "support_contract": "string",
        "support_count": "integer",
        "metric": "string",
        "display_label": "string",
        "value": "nullable_number",
        "unit": "string",
        "source_artifact": "string",
    },
    "runtime_summary.csv": {
        "dataset": "string",
        "evaluation_semantics": "string",
        "case_or_region": "string",
        "trial_id": "nullable_string",
        "seed": "nullable_integer",
        "stage": "string",
        "fingerprint": "nullable_string",
        "scanner_backend": "nullable_string",
        "call_count": "integer",
        "cell_label": "nullable_string",
        "cell_consumers": "json",
        "state": "string",
        "elapsed_seconds": "number",
        "elapsed_semantics": "string",
        "shared_stage": "boolean",
        "attribution": "string",
        "source_artifact": "string",
    },
}

ROOT_TABLE_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "publication_metrics.csv": (
        "dataset",
        "case_or_region",
        "trial_id",
        "seed",
        "cell_label",
        "stage",
        "selection",
        "metric",
    ),
    "publication_contrasts.csv": (
        "dataset",
        "case_or_region",
        "trial_id",
        "seed",
        "contrast_name",
        "stage",
        "selection",
        "metric",
    ),
    "publication_summary.csv": (
        "dataset",
        "case_or_region",
        "stage",
        "selection",
        "metric",
        "cell_label",
    ),
    "f3_regional_summary.csv": ("dataset", "stage", "cell_label", "region", "metric"),
    "f3_orientation_summary.csv": (
        "dataset",
        "stage",
        "left_cell",
        "right_cell",
        "support_contract",
        "metric",
    ),
    "runtime_summary.csv": (
        "dataset",
        "case_or_region",
        "trial_id",
        "seed",
        "stage",
        "fingerprint",
        "scanner_backend",
        "call_count",
        "cell_label",
        "cell_consumers",
        "state",
        "attribution",
    ),
}

SYNTHETIC_SOURCE_IDENTITY_FIELDS = (
    "artifact_schema_version",
    "metric_schema_version",
    "scalar_evidence_contract_version",
    "runtime_contract_version",
    "manifest_sha256",
    "completion_sha256",
)
F3_SOURCE_IDENTITY_FIELDS = (
    "artifact_schema_version",
    "result_schema_version",
    "run_fingerprint",
    "dataset_identity",
    "manifest_sha256",
    "completion_sha256",
)


def source_identity_object(source: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    """Build the fixed, path-free source identity object used by all callers."""

    if any(name not in source for name in fields):
        missing = next(name for name in fields if name not in source)
        raise ValueError(f"publication source identity is missing {missing}")
    return {name: finite_json_normalize(source[name]) for name in fields}


def finite_json_normalize(value: Any) -> Any:
    """Return a JSON-compatible finite value without losing scalar types.

    In particular, booleans are handled before integers so that ``true`` and
    ``1`` retain distinct canonical JSON encodings.  Paths are intentionally
    not normalized here; callers choose their explicit provenance/identity
    field sets instead.
    """

    if isinstance(value, np.generic):
        return finite_json_normalize(value.item())
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("publication JSON values must be finite")
        return float(value)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("publication JSON object keys must be strings")
            normalized[key] = finite_json_normalize(item)
        return normalized
    if isinstance(value, (tuple, list)):
        return [finite_json_normalize(item) for item in value]
    raise ValueError(f"publication JSON value is not supported: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize finite JSON using the one digest encoding used by publication."""

    return json.dumps(
        finite_json_normalize(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    """Return the canonical SHA-256 digest for a finite JSON value."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def png_metadata(path: Any) -> dict[str, int | str]:
    """Read bounded IHDR metadata without importing Pillow or Matplotlib."""

    try:
        with open(path, "rb") as stream:  # noqa: PTH123 - public path helper
            prefix = stream.read(8 + 4 + 4 + 13)
            stream.seek(0, 2)
            size = stream.tell()
    except OSError as error:
        raise ValueError(f"cannot read publication PNG: {path}") from error
    if len(prefix) < 8 + 4 + 4 + 13 or prefix[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("publication figure is not a PNG with a complete IHDR")
    length = struct.unpack(">I", prefix[8:12])[0]
    if length != 13 or prefix[12:16] != b"IHDR":
        raise ValueError("publication PNG first chunk must be IHDR with length 13")
    width, height = struct.unpack(">II", prefix[16:24])
    if not (0 < width <= PNG_MAX_DIMENSION and 0 < height <= PNG_MAX_DIMENSION):
        raise ValueError("publication PNG dimensions are outside the reasonable bound")
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as stream:  # noqa: PTH123 - public path helper
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot hash publication PNG: {path}") from error
    return {
        "pixel_width": width,
        "pixel_height": height,
        "png_size": size,
        "png_sha256": digest.hexdigest(),
    }


def _raw_value(value: Any, field_type: CSVType, context: str) -> Any:
    """Normalize a writer-side value according to one CSV field type."""

    if isinstance(value, np.generic):
        value = value.item()
    nullable = field_type.startswith("nullable_")
    base = field_type.removeprefix("nullable_")
    if value is None:
        if nullable:
            return None
        raise ValueError(f"publication CSV {context} must not be null")
    if nullable and base == "string" and value == "":
        # CSV has one canonical nullable-string encoding: the empty field.
        # Normalize writer-side legacy ``""`` values to the same null used by
        # reader-side parsing before computing a digest.
        return None
    if base == "string":
        if not isinstance(value, str):
            raise ValueError(f"publication CSV {context} must be a string")
        return value
    if base == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"publication CSV {context} must be an integer")
        return int(value)
    if base == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"publication CSV {context} must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"publication CSV {context} must be finite")
        return number
    if base == "boolean":
        if not isinstance(value, bool):
            raise ValueError(f"publication CSV {context} must be a boolean")
        return value
    if base == "json":
        return finite_json_normalize(value)
    raise RuntimeError(f"unknown publication CSV type {field_type!r}")


def normalize_typed_row(
    row: Mapping[str, Any],
    header: Sequence[str],
    field_types: Mapping[str, CSVType],
    *,
    context: str = "row",
) -> dict[str, Any]:
    """Normalize a writer-side typed CSV row in header order."""

    if tuple(field_types) != tuple(header):
        raise ValueError("publication CSV field types must match the header exactly")
    if set(row) - set(header):
        raise ValueError(f"publication CSV {context} has unexpected fields")
    return {
        name: _raw_value(row.get(name), field_types[name], f"{context}.{name}") for name in header
    }


def _csv_value(value: str, field_type: CSVType, context: str) -> Any:
    """Parse one CSV string using exactly the writer-side typed semantics."""

    nullable = field_type.startswith("nullable_")
    base = field_type.removeprefix("nullable_")
    if value == "" and nullable:
        return None
    if base == "string":
        if value == "" and not nullable:
            # Empty strings remain strings for required textual fields.  The
            # higher-level contract decides which identifiers may be empty.
            return ""
        return value
    if base == "integer":
        try:
            return int(value, 10)
        except ValueError as error:
            raise ValueError(f"invalid integer in publication CSV {context}") from error
    if base == "number":
        try:
            result = float(value)
        except ValueError as error:
            raise ValueError(f"invalid number in publication CSV {context}") from error
        if not math.isfinite(result):
            raise ValueError(f"non-finite number in publication CSV {context}")
        return result
    if base == "boolean":
        if value not in {"true", "false"}:
            raise ValueError(f"invalid boolean in publication CSV {context}")
        return value == "true"
    if base == "json":
        try:
            return finite_json_normalize(
                json.loads(
                    value,
                    parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
                )
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"invalid JSON in publication CSV {context}") from error
    raise RuntimeError(f"unknown publication CSV type {field_type!r}")


def normalize_typed_csv_row(
    row: Mapping[str, str],
    header: Sequence[str],
    field_types: Mapping[str, CSVType],
    *,
    context: str = "row",
) -> dict[str, Any]:
    """Parse a CSV row into its canonical typed semantic representation."""

    if tuple(field_types) != tuple(header):
        raise ValueError("publication CSV field types must match the header exactly")
    if set(row) != set(header):
        raise ValueError(f"publication CSV {context} field set mismatch")
    return {name: _csv_value(row[name], field_types[name], f"{context}.{name}") for name in header}


def ordered_row_identity_digest(
    rows: Sequence[Mapping[str, Any]], identity_fields: Sequence[str]
) -> str:
    """Digest ordered typed row identities, including null versus zero values."""

    if not identity_fields:
        raise ValueError("publication table identity fields must not be empty")
    values = []
    for row in rows:
        if any(name not in row for name in identity_fields):
            raise ValueError("publication table identity field is absent from a row")
        values.append({name: row[name] for name in identity_fields})
    return canonical_digest(values)


def ordered_semantic_rows_digest(header: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    """Digest every typed row in canonical header order and original row order."""

    return canonical_digest([{name: row[name] for name in header} for row in rows])


def build_table_contract(
    header: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    identity_fields: Sequence[str],
    field_types: Mapping[str, CSVType] | None = None,
) -> dict[str, Any]:
    """Build a manifest-ready table semantic contract from typed writer rows."""

    normalized_header = tuple(header)
    if len(normalized_header) != len(set(normalized_header)):
        raise ValueError("publication table header contains duplicate fields")
    if any(name not in normalized_header for name in identity_fields):
        raise ValueError("publication table identity field is absent from the header")
    normalized_rows = (
        tuple(
            normalize_typed_row(row, normalized_header, field_types, context=f"row[{index}]")
            for index, row in enumerate(rows)
        )
        if field_types is not None
        else tuple(
            {name: finite_json_normalize(row.get(name)) for name in normalized_header}
            for row in rows
        )
    )
    return {
        "header": list(normalized_header),
        "row_count": len(normalized_rows),
        "identity_fields": list(identity_fields),
        "ordered_identity_sha256": ordered_row_identity_digest(normalized_rows, identity_fields),
        "ordered_semantic_rows_sha256": ordered_semantic_rows_digest(
            normalized_header, normalized_rows
        ),
    }


def validate_table_contract(
    contract: Mapping[str, Any],
    header: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    field_types: Mapping[str, CSVType],
    *,
    context: str,
) -> tuple[dict[str, Any], ...]:
    """Recompute and compare a manifest table contract from parsed CSV rows."""

    expected_keys = {
        "header",
        "row_count",
        "identity_fields",
        "ordered_identity_sha256",
        "ordered_semantic_rows_sha256",
    }
    optional_keys = {"source_expected_identities", "source_expected_identity_sha256"}
    if set(contract) - expected_keys - optional_keys or not expected_keys <= set(contract):
        raise ValueError(f"publication table contract field set mismatch: {context}")
    if contract["header"] != list(header):
        raise ValueError(f"publication table contract header mismatch: {context}")
    identity_fields = contract["identity_fields"]
    if (
        not isinstance(identity_fields, list)
        or not identity_fields
        or any(not isinstance(name, str) or name not in header for name in identity_fields)
        or len(identity_fields) != len(set(identity_fields))
    ):
        raise ValueError(f"publication table contract identity fields are invalid: {context}")
    if isinstance(contract["row_count"], bool) or not isinstance(contract["row_count"], int):
        raise ValueError(f"publication table contract row count is invalid: {context}")
    if any(
        not isinstance(contract[name], str)
        for name in ("ordered_identity_sha256", "ordered_semantic_rows_sha256")
    ):
        raise ValueError(f"publication table contract digest is invalid: {context}")
    typed_rows = tuple(
        normalize_typed_csv_row(row, header, field_types, context=f"{context}[{index}]")
        for index, row in enumerate(rows)
    )
    actual = build_table_contract(header, typed_rows, identity_fields, field_types)
    if any(contract[name] != actual[name] for name in expected_keys):
        raise ValueError(f"publication table semantic contract mismatch: {context}")
    if "source_expected_identities" in contract:
        expected_identities = contract["source_expected_identities"]
        expected_digest = contract.get("source_expected_identity_sha256")
        if not isinstance(expected_identities, list) or not isinstance(expected_digest, str):
            raise ValueError(f"publication source coverage contract is invalid: {context}")
        if canonical_digest(expected_identities) != expected_digest:
            raise ValueError(f"publication source coverage digest mismatch: {context}")
    return typed_rows


def require_unique_row_identities(
    rows: Sequence[Mapping[str, Any]], identity_fields: Sequence[str], *, context: str
) -> None:
    """Reject duplicate typed row identities without using an artificial index."""

    seen: set[bytes] = set()
    for row in rows:
        identity = {name: row[name] for name in identity_fields}
        encoded = canonical_json_bytes(identity)
        if encoded in seen:
            raise ValueError(f"duplicate publication row identity: {context}")
        seen.add(encoded)


__all__ = [
    "CSVType",
    "F3_SOURCE_IDENTITY_FIELDS",
    "FIGURE_DATA_FIELD_TYPES",
    "ROOT_TABLE_FIELD_TYPES",
    "ROOT_TABLE_IDENTITY_FIELDS",
    "SYNTHETIC_SOURCE_IDENTITY_FIELDS",
    "build_table_contract",
    "canonical_digest",
    "canonical_json_bytes",
    "finite_json_normalize",
    "normalize_typed_csv_row",
    "normalize_typed_row",
    "ordered_row_identity_digest",
    "ordered_semantic_rows_digest",
    "png_metadata",
    "PNG_MAX_DIMENSION",
    "require_unique_row_identities",
    "source_identity_object",
    "validate_table_contract",
]
