"""Atomic artifact bundles for completed synthetic mode comparisons."""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import scipy

import pyosv

from .builder import build_mode_comparison_plan
from .config import SyntheticModeComparisonConfig
from .contrasts import AggregateRow, ContrastRow
from .experiment import RuntimeRow, SyntheticModeComparisonResult
from .metrics import METRIC_SCHEMA_VERSION, MetricRow

ARTIFACT_SCHEMA_VERSION = 1
COMPLETION_SCHEMA_VERSION = 1
METRIC_REGISTRY_ID = "pyosv.synthetic_mode_comparison.metrics"
METRIC_REGISTRY_DEFINITION_VERSION = 1
CONTRAST_DEFINITION_ID = "pyosv.synthetic_mode_comparison.contrasts"
CONTRAST_FORMULA_VERSION = 1

MANIFEST_FILE = "manifest.json"
CELL_REPORTS_FILE = "cell_reports.json"
METRICS_FILE = "metrics_long.csv"
METRIC_AGGREGATES_FILE = "metric_aggregates.csv"
CONTRASTS_FILE = "contrasts.csv"
CONTRAST_AGGREGATES_FILE = "contrast_aggregates.csv"
RUNTIME_FILE = "runtime.csv"
COMPLETION_FILE = "completion.json"

HASHED_BUNDLE_FILES = (
    MANIFEST_FILE,
    CELL_REPORTS_FILE,
    METRICS_FILE,
    METRIC_AGGREGATES_FILE,
    CONTRASTS_FILE,
    CONTRAST_AGGREGATES_FILE,
    RUNTIME_FILE,
)
REQUIRED_BUNDLE_FILES = (*HASHED_BUNDLE_FILES, COMPLETION_FILE)

_CSV_MODELS = {
    METRICS_FILE: MetricRow,
    METRIC_AGGREGATES_FILE: AggregateRow,
    CONTRASTS_FILE: ContrastRow,
    CONTRAST_AGGREGATES_FILE: AggregateRow,
    RUNTIME_FILE: RuntimeRow,
}
_CSV_INTEGER_FIELDS = {
    METRICS_FILE: {"schema_version", "seed", "scanner_refinement_factor"},
    METRIC_AGGREGATES_FILE: {"n"},
    CONTRASTS_FILE: {"seed"},
    CONTRAST_AGGREGATES_FILE: {"n"},
    RUNTIME_FILE: {"seed", "call_count"},
}
_CSV_FLOAT_FIELDS = {
    METRICS_FILE: {"value"},
    METRIC_AGGREGATES_FILE: {"mean", "median", "std", "min", "max", "q25", "q75"},
    CONTRASTS_FILE: {"raw_value", "improvement_value"},
    CONTRAST_AGGREGATES_FILE: {
        "mean",
        "median",
        "std",
        "min",
        "max",
        "q25",
        "q75",
    },
    RUNTIME_FILE: {"elapsed_seconds"},
}
_CSV_NULLABLE_NUMERIC_FIELDS = {
    METRICS_FILE: {"seed", "scanner_refinement_factor"},
    METRIC_AGGREGATES_FILE: set(),
    CONTRASTS_FILE: {"seed", "improvement_value"},
    CONTRAST_AGGREGATES_FILE: set(),
    RUNTIME_FILE: {"seed"},
}
_CSV_BOOLEAN_FIELDS = {
    METRICS_FILE: {"contrast_eligible"},
    METRIC_AGGREGATES_FILE: set(),
    CONTRASTS_FILE: set(),
    CONTRAST_AGGREGATES_FILE: set(),
    RUNTIME_FILE: {"shared_stage"},
}


def write_artifact_bundle(
    result: SyntheticModeComparisonResult,
    output_dir: str | PathLike[str],
    *,
    config: SyntheticModeComparisonConfig,
    pretty: bool = False,
) -> Path:
    """Atomically write one complete, self-validating result bundle."""

    if not isinstance(result, SyntheticModeComparisonResult):
        raise ValueError("result must be a SyntheticModeComparisonResult")
    if not isinstance(config, SyntheticModeComparisonConfig):
        raise ValueError("config must be a SyntheticModeComparisonConfig")
    if not isinstance(pretty, bool):
        raise ValueError("pretty must be bool")

    final_path = Path(output_dir)
    if os.path.lexists(final_path):
        raise FileExistsError(f"artifact output already exists: {final_path}")
    _validate_result_matches_config(result, config)

    source = _source_provenance()
    manifest = _build_manifest(result, config, source)
    cell_reports = _json_value(result.cell_reports)
    csv_rows: dict[str, Sequence[Any]] = {
        METRICS_FILE: result.metric_rows,
        METRIC_AGGREGATES_FILE: result.metric_aggregates,
        CONTRASTS_FILE: result.contrast_rows,
        CONTRAST_AGGREGATES_FILE: result.contrast_aggregates,
        RUNTIME_FILE: result.runtime_rows,
    }

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{final_path.name}.tmp-", dir=final_path.parent)
    )
    finalized = False
    try:
        file_bytes: dict[str, bytes] = {
            MANIFEST_FILE: _json_bytes(manifest, pretty=pretty),
            CELL_REPORTS_FILE: _json_bytes(cell_reports, pretty=pretty),
        }
        file_bytes.update(
            {
                filename: _csv_bytes(rows, _CSV_MODELS[filename])
                for filename, rows in csv_rows.items()
            }
        )

        metadata: dict[str, dict[str, Any]] = {}
        for filename in HASHED_BUNDLE_FILES:
            payload = file_bytes[filename]
            artifact_path = temporary_path / filename
            _write_bytes(artifact_path, payload)
            metadata[filename] = _file_metadata(artifact_path)

        completion = {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "status": "complete",
            "required_files": list(REQUIRED_BUNDLE_FILES),
            "files": metadata,
        }
        _write_bytes(
            temporary_path / COMPLETION_FILE,
            _json_bytes(completion, pretty=pretty),
        )
        _fsync_directory(temporary_path)
        _finalize_bundle(temporary_path, final_path)
        finalized = True
        _fsync_directory(final_path.parent)
    except BaseException as error:
        _cleanup_path(final_path if finalized else temporary_path, error)
        raise
    return final_path


def _validate_result_matches_config(
    result: SyntheticModeComparisonResult,
    config: SyntheticModeComparisonConfig,
) -> None:
    plan = build_mode_comparison_plan(config)
    expected_plan = _json_value(asdict(plan))
    actual_plan = _json_value(result.plan_metadata)
    if actual_plan != expected_plan:
        raise ValueError("result plan metadata does not match config")

    expected_trials = _json_value([asdict(trial) for trial in plan.trials])
    actual_trials = _json_value(result.trial_metadata)
    if actual_trials != expected_trials:
        raise ValueError("result trial metadata does not match config")


def validate_completed_bundle(path: str | PathLike[str]) -> bool:
    """Reject an incomplete, changed, non-finite, or malformed artifact bundle."""

    bundle = Path(path)
    if not bundle.is_dir() or bundle.is_symlink():
        raise ValueError("artifact bundle must be a directory")
    entries = {entry.name: entry for entry in bundle.iterdir()}
    expected = set(REQUIRED_BUNDLE_FILES)
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        unexpected = sorted(set(entries) - expected)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError("invalid artifact file set (" + "; ".join(details) + ")")
    if any(not entry.is_file() or entry.is_symlink() for entry in entries.values()):
        raise ValueError("artifact bundle entries must be regular files")

    completion = _read_json(entries[COMPLETION_FILE])
    if not isinstance(completion, dict):
        raise ValueError("completion.json must contain an object")
    if completion.get("schema_version") != COMPLETION_SCHEMA_VERSION:
        raise ValueError("unsupported completion schema version")
    if completion.get("status") != "complete":
        raise ValueError("completion status must be 'complete'")
    if completion.get("required_files") != list(REQUIRED_BUNDLE_FILES):
        raise ValueError("completion required_files do not match the bundle contract")
    metadata = completion.get("files")
    if not isinstance(metadata, dict) or set(metadata) != set(HASHED_BUNDLE_FILES):
        raise ValueError("completion file metadata do not match the bundle contract")

    for filename in HASHED_BUNDLE_FILES:
        record = metadata[filename]
        if not isinstance(record, dict) or set(record) != {"sha256", "size"}:
            raise ValueError(f"invalid completion metadata for {filename}")
        expected_size = record["size"]
        expected_hash = record["sha256"]
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise ValueError(f"invalid recorded size for {filename}")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"invalid recorded SHA-256 for {filename}")
        payload = entries[filename].read_bytes()
        if len(payload) != expected_size:
            raise ValueError(f"size mismatch for {filename}")
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {filename}")

    manifest = _read_json(entries[MANIFEST_FILE])
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain an object")
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported artifact schema version")
    reports = _read_json(entries[CELL_REPORTS_FILE])
    if not isinstance(reports, list):
        raise ValueError("cell_reports.json must contain an ordered reports array")
    for filename in _CSV_MODELS:
        _validate_csv(entries[filename], filename)
    return True


def _build_manifest(
    result: SyntheticModeComparisonResult,
    config: SyntheticModeComparisonConfig,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _json_value(result.plan_metadata)
    trials = _json_value(result.trial_metadata)
    if not isinstance(plan, dict) or not isinstance(trials, list):
        raise ValueError("result metadata must be JSON objects")
    cells = plan.get("cells")
    if not isinstance(cells, list):
        raise ValueError("result plan metadata must contain ordered cells")
    scanner = plan.get("scanner_template")
    if not isinstance(scanner, dict) or not isinstance(scanner.get("input_config"), dict):
        raise ValueError("result plan metadata must contain resolved scanner input settings")
    scanner_input_seed = scanner["input_config"].get("seed")

    case_stochastic = {
        trial.get("case_id"): trial.get("seed") is not None
        for trial in trials
        if isinstance(trial, dict)
    }
    trial_records = []
    for order, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ValueError("trial metadata entries must be objects")
        case_id = trial.get("case_id")
        trial_records.append(
            {
                "order": order,
                "case_id": case_id,
                "trial_id": trial.get("trial_id"),
                "stochastic": case_stochastic.get(case_id, False),
                "case_generation_seed": trial.get("seed"),
                "scanner_input_seed": scanner_input_seed,
            }
        )

    software_versions, software_version_status = _software_versions()
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "canonical_cells": [
            {
                "order": order,
                "label": cell.get("label"),
                "scope": cell.get("scope"),
            }
            for order, cell in enumerate(cells)
            if isinstance(cell, dict)
        ],
        "input_config": _json_value(asdict(config)),
        "resolved_plan": plan,
        "case_order": list(plan.get("case_ids", [])),
        "cases": [
            {
                "order": order,
                "case_id": case_id,
                "stochastic": case_stochastic.get(case_id, False),
            }
            for order, case_id in enumerate(plan.get("case_ids", []))
        ],
        "trials": trial_records,
        "shape": plan.get("shape"),
        "variant": plan.get("comparison_variant"),
        "oracle_workflow_isolation": plan.get("include_oracle_workflow_isolation"),
        "metric_registry": {
            "id": METRIC_REGISTRY_ID,
            "definition_version": METRIC_REGISTRY_DEFINITION_VERSION,
        },
        "contrast_definition": {
            "id": CONTRAST_DEFINITION_ID,
            "formula_version": CONTRAST_FORMULA_VERSION,
        },
        "cache_stats": _json_value(result.cache_stats),
        "software_versions": software_versions,
        "software_version_status": software_version_status,
        "source_provenance": _json_value(source),
    }


def _source_provenance() -> dict[str, Any]:
    unavailable = {
        "status": "not_available",
        "method": "git_cli",
        "commit": None,
        "dirty": None,
    }
    try:
        source_path = Path(__file__).resolve().parent
        root = subprocess.run(
            ["git", "-C", str(source_path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if not root:
            return unavailable
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", root, "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            return unavailable
    except Exception:
        return unavailable
    return {
        "status": "available",
        "method": "git_cli",
        "commit": commit,
        "dirty": bool(status),
    }


def _software_versions() -> tuple[dict[str, str | None], dict[str, str]]:
    getters = {
        "python": platform.python_version,
        "pyosv": lambda: pyosv.__version__,
        "numpy": lambda: np.__version__,
        "scipy": lambda: scipy.__version__,
    }
    versions: dict[str, str | None] = {}
    status: dict[str, str] = {}
    for name, getter in getters.items():
        try:
            value = getter()
            if not isinstance(value, str) or not value:
                raise ValueError("version must be a non-empty string")
        except Exception:
            versions[name] = None
            status[name] = "not_available"
        else:
            versions[name] = value
            status[name] = "available"
    return versions, status


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
    normalized = _json_value(value)
    options: dict[str, Any] = {"allow_nan": False, "ensure_ascii": False}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(normalized, **options) + "\n").encode("utf-8")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("artifact JSON values must be finite")
        return value
    if isinstance(value, np.generic) or isinstance(value, np.ndarray):
        raise ValueError("artifact JSON must not implicitly convert NumPy objects")
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact JSON object keys must be strings")
            output[key] = _json_value(item)
        return output
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise ValueError(f"artifact value is not JSON-safe: {type(value).__name__}")


def _csv_bytes(rows: Sequence[Any], model: type[Any]) -> bytes:
    field_names = tuple(field.name for field in fields(model))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(field_names)
    for row in rows:
        if not isinstance(row, model):
            raise ValueError(f"CSV rows must contain only {model.__name__} values")
        mapping = row.as_dict()
        writer.writerow(_csv_value(mapping[name]) for name in field_names)
    return stream.getvalue().encode("utf-8")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("artifact CSV values must be finite")
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list, Mapping)):
        return json.dumps(
            _json_value(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    raise ValueError(f"artifact CSV value has unsupported type {type(value).__name__}")


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        written = stream.write(payload)
        if written != len(payload):
            raise OSError(f"short artifact write for {path.name}")
        stream.flush()
        os.fsync(stream.fileno())


def _file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def _finalize_bundle(temporary_path: Path, final_path: Path) -> None:
    _rename_noreplace(temporary_path, final_path)


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
            -100,  # AT_FDCWD
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                os.strerror(error_number),
                destination,
            )
        return

    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        renamex_np = library.renamex_np
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        result = renamex_np(
            os.fsencode(source),
            os.fsencode(destination),
            4,  # RENAME_EXCL
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                os.strerror(error_number),
                destination,
            )
        return

    if os.name == "nt":
        os.rename(source, destination)
        return

    raise OSError(
        errno.ENOTSUP,
        "atomic no-replace rename is not supported",
        destination,
    )


def _cleanup_path(path: Path, original_error: BaseException) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except BaseException as cleanup_error:
        add_note = getattr(original_error, "add_note", None)
        if add_note is not None:
            add_note(f"artifact cleanup also failed for {path}: {cleanup_error!r}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> Any:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda constant: _raise_nonfinite_json(constant),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed JSON artifact: {path.name}") from error
    return _json_value(value)


def _raise_nonfinite_json(constant: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {constant}")


def _validate_csv(path: Path, filename: str) -> None:
    try:
        text = path.read_text(encoding="utf-8")
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"malformed CSV artifact: {filename}") from error
    header = [field.name for field in fields(_CSV_MODELS[filename])]
    if not rows or rows[0] != header:
        raise ValueError(f"invalid CSV header for {filename}")
    for row in rows[1:]:
        if len(row) != len(header):
            raise ValueError(f"malformed CSV row in {filename}")
        values = dict(zip(header, row, strict=True))
        for name in _CSV_INTEGER_FIELDS[filename]:
            value = values[name]
            if not value and name in _CSV_NULLABLE_NUMERIC_FIELDS[filename]:
                continue
            try:
                int(value, 10)
            except ValueError as error:
                raise ValueError(f"invalid integer in {filename}: {name}") from error
        for name in _CSV_FLOAT_FIELDS[filename]:
            value = values[name]
            if not value and name in _CSV_NULLABLE_NUMERIC_FIELDS[filename]:
                continue
            try:
                number = float(value)
            except ValueError as error:
                raise ValueError(f"invalid number in {filename}: {name}") from error
            if not np.isfinite(number):
                raise ValueError(f"non-finite number in {filename}: {name}")
        for name in _CSV_BOOLEAN_FIELDS[filename]:
            if values[name] not in {"true", "false"}:
                raise ValueError(f"invalid boolean in {filename}: {name}")
        if filename == CONTRASTS_FILE:
            try:
                component_cells = json.loads(values["component_cells"])
            except json.JSONDecodeError as error:
                raise ValueError("invalid component_cells in contrasts.csv") from error
            if not isinstance(component_cells, list) or any(
                not isinstance(cell, str) for cell in component_cells
            ):
                raise ValueError("invalid component_cells in contrasts.csv")


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "COMPLETION_SCHEMA_VERSION",
    "HASHED_BUNDLE_FILES",
    "REQUIRED_BUNDLE_FILES",
    "validate_completed_bundle",
    "write_artifact_bundle",
]
