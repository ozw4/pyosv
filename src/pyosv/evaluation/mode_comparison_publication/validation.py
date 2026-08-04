"""Read-only validation for completed publication bundles.

This module deliberately has no matplotlib import and never opens a source
experiment runner or a source data root.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..f3d_mode_comparison import CONTRAST_DEFINITIONS as F3_CONTRAST_DEFINITIONS
from ..f3d_mode_comparison import F3_BUFFERED_PERCENTILE, F3_BUFFER_RADIUS
from ..synthetic_mode_comparison import CONTRAST_DEFINITIONS as SYNTHETIC_CONTRAST_DEFINITIONS

from .config import (
    CANONICAL_CELL_ORDER,
    CANONICAL_STAGE_ORDER,
    CONTRAST_NAMES,
    FIGURE_DATA_HEADER,
    FIGURE_MANIFEST_FIELDS,
    F3_SEMANTICS,
    F3_ORIENTATION_SUMMARY_HEADER,
    F3_REGIONAL_SUMMARY_HEADER,
    FIGURE_SELECTION_POLICY,
    PUBLICATION_ARTIFACT_SCHEMA_VERSION,
    PUBLICATION_COMPLETION_SCHEMA_VERSION,
    PUBLICATION_CONTRASTS_HEADER,
    PUBLICATION_FIGURE_CONTRACT_VERSION,
    PUBLICATION_METRICS_HEADER,
    PUBLICATION_METRIC_SELECTION_VERSION,
    PUBLICATION_SUMMARY_HEADER,
    PUBLICATION_INTERPRETATION,
    REQUIRED_PUBLICATION_FILES,
    RUNTIME_SUMMARY_HEADER,
    SYNTHETIC_SCANNER_CELL_ORDER,
    SYNTHETIC_SEMANTICS,
)
from .registry import PUBLICATION_METRIC_BY_IDENTITY, PUBLICATION_METRIC_REGISTRY

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_AXES = {
    "RL-REF": ("reference-like", "reference"),
    "RL-QUAL": ("reference-like", "quality"),
    "Q-REF": ("quality", "reference"),
    "Q-QUAL": ("quality", "quality"),
}
_SCANNER_AXES = {
    "RL-SCAN": ("reference-like", ""),
    "Q-SCAN": ("quality", ""),
}


def _json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid publication JSON: {path}") from error


def _finite_json(value: Any, context: str = "JSON") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite number")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _finite_json(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, f"{context}[{index}]")


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _relative_files(root: Path) -> tuple[str, ...]:
    return tuple(
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink() and path.name != "completion.json"
    )


def _csv(path: Path, header: tuple[str, ...]) -> tuple[dict[str, str], ...]:
    try:
        payload = path.read_text(encoding="utf-8")
        reader = csv.reader(io.StringIO(payload, newline=""), strict=True)
        rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError(f"invalid publication CSV: {path}") from error
    if not rows or tuple(rows[0]) != header:
        raise ValueError(f"publication CSV header mismatch: {path.name}")
    output = []
    for values in rows[1:]:
        if len(values) != len(header):
            raise ValueError(f"publication CSV row width mismatch: {path.name}")
        output.append(dict(zip(header, values, strict=True)))
    return tuple(output)


def _optional_float(value: str, context: str) -> float | None:
    if value == "":
        return None
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"invalid number in {context}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite number in {context}")
    return result


def _float(value: str, context: str) -> float:
    result = _optional_float(value, context)
    if result is None:
        raise ValueError(f"empty number in {context}")
    return result


def _int(value: str, context: str) -> int:
    try:
        return int(value, 10)
    except ValueError as error:
        raise ValueError(f"invalid integer in {context}") from error


def _json_field(value: str, context: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON field in {context}") from error
    _finite_json(parsed, context)
    return parsed


def _validate_source_identity(manifest: Mapping[str, Any]) -> None:
    for name in ("synthetic_source", "f3_source"):
        source = manifest.get(name)
        if not isinstance(source, Mapping):
            raise ValueError(f"publication manifest {name} is missing")
        for field in ("identity_digest", "completion_sha256", "manifest_sha256"):
            value = source.get(field)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"publication manifest {name}.{field} is not a SHA-256 digest")
    f3 = manifest["f3_source"]
    if not isinstance(f3.get("run_fingerprint"), str) or not _SHA256.fullmatch(
        f3["run_fingerprint"]
    ):
        raise ValueError("publication F3 run fingerprint is invalid")


def _validate_metric_rows(rows: tuple[dict[str, str], ...], manifest: Mapping[str, Any]) -> None:
    by_identity: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dataset = row["dataset"]
        if dataset not in {"synthetic", "f3"}:
            raise ValueError("publication metric dataset is invalid")
        expected_semantics = SYNTHETIC_SEMANTICS if dataset == "synthetic" else F3_SEMANTICS
        if row["evaluation_semantics"] != expected_semantics:
            raise ValueError("publication metric evaluation semantics are invalid")
        if dataset == "f3":
            if row["case_or_region"] != "full" or row["trial_id"] or row["seed"]:
                raise ValueError("F3 metrics must describe one full-volume unit")
            if (
                "accuracy" in row["evaluation_semantics"].lower()
                or "ground-truth" in row["evaluation_semantics"].lower()
            ):
                raise ValueError("F3 metric semantics must not claim accuracy")
        else:
            if not row["case_or_region"] or not row["trial_id"]:
                raise ValueError("synthetic metric case/trial identity is missing")
        cell = row["cell_label"]
        if cell in _CONDITION_AXES:
            expected_backend, expected_workflow = _CONDITION_AXES[cell]
        elif dataset == "synthetic" and row["stage"] == "scanner_raw" and cell in _SCANNER_AXES:
            expected_backend, expected_workflow = _SCANNER_AXES[cell]
        else:
            raise ValueError("publication metrics contain an unsupported processing cell")
        if (row["scanner_backend"], row["workflow_mode"]) != (
            expected_backend,
            expected_workflow,
        ):
            raise ValueError("condition ID and scanner/workflow axes are inconsistent")
        identity = (dataset, row["stage"], row["selection"], row["metric"])
        entry = PUBLICATION_METRIC_BY_IDENTITY.get(identity)
        if entry is None:
            raise ValueError(f"publication metric is not in the curated registry: {identity!r}")
        if (row["unit"], row["direction"]) != (entry.unit, entry.direction):
            raise ValueError("publication metric unit or direction does not match the registry")
        value = _optional_float(row["value"], f"metric {identity}")
        if value is None and not entry.nullable:
            raise ValueError(f"non-nullable publication metric is empty: {identity!r}")
        if row["source_artifact"] not in {"metrics_long.csv", "reports/metrics_long.csv"}:
            raise ValueError("publication metric source artifact is invalid")
        by_identity[identity].append(row)

    source_skinning = any(
        identity[0] == "synthetic" and identity[1] == "skin" for identity in by_identity
    )
    f3_skinning = any(identity[0] == "f3" and identity[1] == "skin" for identity in by_identity)
    for entry in PUBLICATION_METRIC_REGISTRY:
        rows_for_entry = by_identity.get(entry.identity, [])
        if not entry.required and not rows_for_entry:
            if entry.stage == "skin" and (
                (entry.dataset == "synthetic" and source_skinning)
                or (entry.dataset == "f3" and f3_skinning)
            ):
                raise ValueError(f"enabled skinning metric coverage is missing: {entry.identity!r}")
            continue
        if entry.dataset == "synthetic" and entry.stage == "skin" and not source_skinning:
            if rows_for_entry:
                raise ValueError("synthetic skin rows exist without a skinning source")
            continue
        if not rows_for_entry:
            raise ValueError(f"required publication metric coverage is missing: {entry.identity!r}")
        if entry.dataset == "f3":
            if {row["cell_label"] for row in rows_for_entry} != set(CANONICAL_CELL_ORDER):
                raise ValueError(
                    f"F3 publication metric cell coverage is incomplete: {entry.identity!r}"
                )
        else:
            expected_cells = (
                set(SYNTHETIC_SCANNER_CELL_ORDER)
                if entry.stage == "scanner_raw"
                else set(CANONICAL_CELL_ORDER)
            )
            trials: dict[tuple[str, str, str], set[str]] = defaultdict(set)
            for row in rows_for_entry:
                trials[(row["case_or_region"], row["trial_id"], row["seed"])].add(row["cell_label"])
            if any(cells != expected_cells for cells in trials.values()):
                raise ValueError(
                    f"synthetic publication metric cell coverage is incomplete: {entry.identity!r}"
                )


def _validate_contrast_rows(rows: tuple[dict[str, str], ...]) -> None:
    definitions = {
        item.name: item for item in (*SYNTHETIC_CONTRAST_DEFINITIONS, *F3_CONTRAST_DEFINITIONS)
    }
    for row in rows:
        dataset = row["dataset"]
        expected = SYNTHETIC_SEMANTICS if dataset == "synthetic" else F3_SEMANTICS
        if row["evaluation_semantics"] != expected:
            raise ValueError("contrast evaluation semantics are invalid")
        if dataset == "f3" and (row["case_or_region"] != "full" or row["trial_id"] or row["seed"]):
            raise ValueError("F3 contrasts must describe one full-volume unit")
        if row["contrast_name"] not in CONTRAST_NAMES:
            raise ValueError("publication contrast is outside the fixed public contrast set")
        identity = (dataset, row["stage"], row["selection"], row["metric"])
        entry = PUBLICATION_METRIC_BY_IDENTITY.get(identity)
        if entry is None:
            raise ValueError(f"contrast metric is outside the curated registry: {identity!r}")
        if (row["unit"], row["direction"]) != (entry.unit, entry.direction):
            raise ValueError("contrast metric unit or direction does not match the registry")
        definition = definitions.get(row["contrast_name"])
        if definition is None:
            raise ValueError("unknown contrast definition")
        cells = _json_field(row["component_cells"], "component_cells")
        if tuple(cells) != definition.component_cells:
            raise ValueError("contrast component cells do not match the source definition")
        raw = _float(row["raw_value"], "raw_value")
        improvement = _optional_float(row["improvement_value"], "improvement_value")
        direction = row["direction"]
        expected_improvement = (
            raw if direction == "higher" else (-raw if direction == "lower" else None)
        )
        if expected_improvement is None:
            if improvement is not None:
                raise ValueError("neutral contrast must keep improvement_value empty")
        elif improvement != expected_improvement:
            raise ValueError("direction and improvement_value are inconsistent")
        if row["source_artifact"] not in {"contrasts.csv", "reports/contrasts.csv"}:
            raise ValueError("contrast source artifact is invalid")


def _validate_summary(
    summary_rows: tuple[dict[str, str], ...], metric_rows: tuple[dict[str, str], ...]
) -> None:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    metadata: dict[tuple[str, ...], dict[str, str]] = {}
    for row in metric_rows:
        value = _optional_float(row["value"], "publication metric value")
        if value is None:
            continue
        key = (
            row["dataset"],
            row["evaluation_semantics"],
            row["case_or_region"],
            row["stage"],
            row["selection"],
            row["metric"],
            row["cell_label"],
            row["unit"],
            row["direction"],
        )
        groups[key].append(value)
        metadata[key] = row
    actual: set[tuple[str, ...]] = set()
    for row in summary_rows:
        key = (
            row["dataset"],
            row["evaluation_semantics"],
            row["case_or_region"],
            row["stage"],
            row["selection"],
            row["metric"],
            row["cell_label"],
            row["unit"],
            row["direction"],
        )
        if key not in groups:
            raise ValueError("summary contains a group not present in publication metrics")
        actual.add(key)
        values = np.asarray(groups[key], dtype=np.float64)
        expected = {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
        }
        if _int(row["n"], "summary n") != expected["n"]:
            raise ValueError("summary n cannot be recomputed from publication metrics")
        for name in ("mean", "median", "minimum", "maximum", "q25", "q75"):
            if not math.isclose(
                _float(row[name], f"summary {name}"), expected[name], rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"summary {name} cannot be recomputed from publication metrics")
    if actual != set(groups):
        raise ValueError("publication summary does not cover all non-null metric groups")


def _validate_supporting_tables(root: Path) -> None:
    for filename, header in (
        ("f3_regional_summary.csv", F3_REGIONAL_SUMMARY_HEADER),
        ("f3_orientation_summary.csv", F3_ORIENTATION_SUMMARY_HEADER),
        ("runtime_summary.csv", RUNTIME_SUMMARY_HEADER),
    ):
        rows = _csv(root / filename, header)
        for row in rows:
            if row["dataset"] not in {"synthetic", "f3"}:
                raise ValueError(f"invalid dataset in {filename}")
            if row["dataset"] == "f3" and row["evaluation_semantics"] != F3_SEMANTICS:
                raise ValueError(f"invalid F3 semantics in {filename}")
            for name, value in row.items():
                if name in {"value", "elapsed_seconds", "support_count"} and value:
                    _optional_float(value, f"{filename}.{name}") if name == "value" else _float(
                        value, f"{filename}.{name}"
                    )


def _validate_figures(root: Path, manifest: Mapping[str, Any]) -> None:
    figure_manifest = _json(root / "figure_manifest.json")
    if not isinstance(figure_manifest, Mapping):
        raise ValueError("figure_manifest must be an object")
    if (
        figure_manifest.get("publication_figure_contract_version")
        != PUBLICATION_FIGURE_CONTRACT_VERSION
    ):
        raise ValueError("figure contract version mismatch")
    if tuple(figure_manifest.get("canonical_condition_order", ())) != CANONICAL_CELL_ORDER:
        raise ValueError("figure canonical condition order mismatch")
    if tuple(figure_manifest.get("canonical_stage_order", ())) != CANONICAL_STAGE_ORDER:
        raise ValueError("figure canonical stage order mismatch")
    shape = tuple(figure_manifest.get("volume_shape", ()))
    if len(shape) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape
    ):
        raise ValueError("figure volume shape is invalid")
    records = figure_manifest.get("figures")
    if not isinstance(records, list) or not records:
        raise ValueError("figure manifest has no figure records")
    figure_ids = [record.get("figure_id") for record in records if isinstance(record, Mapping)]
    if (
        len(figure_ids) != len(records)
        or any(not isinstance(value, str) or not value for value in figure_ids)
        or len(set(figure_ids)) != len(figure_ids)
        or figure_ids != sorted(figure_ids)
    ):
        raise ValueError("figure IDs must be unique and deterministically ordered")
    expected_records = []
    data_paths = []
    data_path_set: set[str] = set()
    figure_path_set: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping) or set(record) != set(FIGURE_MANIFEST_FIELDS):
            raise ValueError("figure record field set mismatch")
        _finite_json(record)
        relative = record["relative_path"]
        if (
            not isinstance(relative, str)
            or not relative.startswith("figures/")
            or ".." in Path(relative).parts
            or Path(relative).parent != Path("figures")
        ):
            raise ValueError("figure relative path is invalid")
        if relative in figure_path_set:
            raise ValueError("figure relative paths must be unique")
        figure_path_set.add(relative)
        dataset = record["dataset"]
        if dataset not in {"synthetic", "f3"}:
            raise ValueError("figure dataset is invalid")
        expected_semantics = SYNTHETIC_SEMANTICS if dataset == "synthetic" else F3_SEMANTICS
        if record["evaluation_semantics"] != expected_semantics:
            raise ValueError("figure evaluation semantics are invalid")
        if record["figure_role"] not in {"main", "supplementary"}:
            raise ValueError("figure role is invalid")
        if not isinstance(record["panel_labels"], list) or any(
            not isinstance(value, str) or not value for value in record["panel_labels"]
        ):
            raise ValueError("figure panel labels are invalid")
        figure_path = root / relative
        axis = record["axis"]
        index = record["slice_index"]
        policy = record["slice_selection_policy"]
        omitted = record["omitted"]
        if not isinstance(omitted, bool):
            raise ValueError("figure omitted must be boolean")
        if omitted:
            if figure_path.exists():
                raise ValueError("omitted figure has a PNG file")
            if not isinstance(record["omission_reason"], str) or not record["omission_reason"]:
                raise ValueError("omitted figure has no reason")
            if record["figure_data_csv"] is not None:
                raise ValueError("omitted figure must not have figure data")
        else:
            if not figure_path.is_file() or figure_path.is_symlink():
                raise ValueError("non-omitted figure PNG is missing")
            if (
                figure_path.stat().st_size == 0
                or figure_path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n"
            ):
                raise ValueError("figure is not a non-empty PNG")
            data_path = record["figure_data_csv"]
            if (
                not isinstance(data_path, str)
                or not data_path.startswith("figure_data/")
                or ".." in Path(data_path).parts
                or Path(data_path).parent != Path("figure_data")
            ):
                raise ValueError("figure data path is invalid")
            if data_path in data_path_set:
                raise ValueError("figure-data paths must be unique")
            data_path_set.add(data_path)
            if not (root / data_path).is_file():
                raise ValueError("figure data CSV is missing")
            data_rows = _csv(root / data_path, FIGURE_DATA_HEADER)
            if not data_rows:
                raise ValueError("figure data CSV is empty")
            allowed_source_stages = {
                item for item in str(record["source_stage"]).split(",") if item
            }
            if any(
                row["figure_id"] != record["figure_id"]
                or row["dataset"] != dataset
                or row["evaluation_semantics"] != expected_semantics
                or row["source_stage"] not in allowed_source_stages
                for row in data_rows
            ):
                raise ValueError("figure data row does not match figure manifest")
            for row in data_rows:
                for name in (
                    "value",
                    "raw_improvement",
                    "normalized_value",
                    "slice_score",
                    "selection_threshold",
                    "vmin",
                    "vmax",
                    "difference_limit",
                    "difference_vmin",
                    "difference_vmax",
                ):
                    _optional_float(row[name], f"{data_path}.{name}")
                if row["slice_index"]:
                    _int(row["slice_index"], f"{data_path}.slice_index")
                expected_axis = "" if axis is None else axis
                if row["axis"] != expected_axis:
                    raise ValueError("figure data axis does not match the manifest")
                expected_index = "" if index is None else str(index)
                if row["slice_index"] != expected_index:
                    raise ValueError("figure data slice index does not match the manifest")
                expected_policy = "" if policy is None else policy
                if row["slice_selection_policy"] != expected_policy:
                    raise ValueError("figure data slice policy does not match the manifest")
                expected_threshold = record["selection_threshold"]
                actual_threshold = _optional_float(
                    row["selection_threshold"], f"{data_path}.selection_threshold"
                )
                if (expected_threshold is None and actual_threshold is not None) or (
                    expected_threshold is not None and actual_threshold != float(expected_threshold)
                ):
                    raise ValueError("figure data selection threshold does not match the manifest")
                scale = record["display_scale"]
                if isinstance(scale, Mapping):
                    row_vmin = _optional_float(row["vmin"], f"{data_path}.vmin")
                    row_vmax = _optional_float(row["vmax"], f"{data_path}.vmax")
                    if row["scale_policy"] == "shared_stage_slice_scale":
                        if (row_vmin, row_vmax) != (
                            scale.get("vmin"),
                            scale.get("vmax"),
                        ):
                            raise ValueError("normal-panel scale is not shared")
                    if row["scale_policy"] == "symmetric_zero_centered_difference":
                        if (row_vmin, row_vmax) != (
                            scale.get("difference_vmin"),
                            scale.get("difference_vmax"),
                        ):
                            raise ValueError("difference-panel scale does not match the manifest")
            data_paths.append(data_path)
            expected_records.append(relative)
        cells = record["cell_labels"]
        allowed_figure_cells = set(CANONICAL_CELL_ORDER)
        if record["dataset"] == "synthetic" and record["source_stage"] == "scanner_raw":
            allowed_figure_cells.update(SYNTHETIC_SCANNER_CELL_ORDER)
        if not isinstance(cells, list) or any(cell not in allowed_figure_cells for cell in cells):
            raise ValueError("figure cell labels are invalid")
        if "PUBLIC-REF" in cells:
            raise ValueError("PUBLIC-REF must not be registered as a processing cell")
        if axis is None and index is not None:
            raise ValueError("figure slice index cannot exist without an axis")
        if axis is not None:
            if axis not in {"i3", "i2", "i1"} or not isinstance(index, int):
                raise ValueError("figure slice axis/index is invalid")
            dimension = {"i3": 0, "i2": 1, "i1": 2}[axis]
            if index < 0 or index >= shape[dimension]:
                raise ValueError("figure slice index is outside the volume shape")
        if policy is not None and policy not in {
            "center",
            "public_reference_peak",
            "end_to_end_difference_peak",
        }:
            raise ValueError("unknown slice selection policy")
        if axis is not None:
            if record["selection_threshold"] is None:
                raise ValueError("spatial figure has no selection threshold")
            if record["selection_percentile"] != F3_BUFFERED_PERCENTILE:
                raise ValueError("spatial figure percentile does not match the F3 contract")
            if record["buffer_radius"] != F3_BUFFER_RADIUS:
                raise ValueError("spatial figure buffer radius does not match the F3 contract")
        elif record["selection_threshold"] is not None:
            raise ValueError("non-spatial figure has a slice selection threshold")
        scale = record["display_scale"]
        if scale is not None:
            if not isinstance(scale, Mapping):
                raise ValueError("figure display scale must be an object")
            for name in ("vmin", "vmax"):
                if scale.get(name) is not None and not isinstance(scale.get(name), (int, float)):
                    raise ValueError("display scale endpoint is invalid")
                if scale.get(name) is not None and not math.isfinite(float(scale[name])):
                    raise ValueError("display scale endpoint is non-finite")
            if scale.get("vmin") is not None and scale.get("vmax") is not None:
                if not float(scale["vmin"]) < float(scale["vmax"]):
                    raise ValueError("display scale endpoints are not ordered")
            if scale.get("difference_vmin") is not None or scale.get("difference_vmax") is not None:
                if scale.get("difference_vmin") is None or scale.get("difference_vmax") is None:
                    raise ValueError("difference scale must provide both endpoints")
                low = float(scale.get("difference_vmin"))
                high = float(scale.get("difference_vmax"))
                if not low < 0.0 < high or not math.isclose(
                    abs(low), abs(high), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError("difference scale must be finite and zero-centered")
        for name in ("selection_percentile", "buffer_radius", "selection_threshold"):
            if record[name] is not None and not math.isfinite(float(record[name])):
                raise ValueError(f"figure {name} is non-finite")
        if not record["omitted"] and (
            not isinstance(record["caption"], str) or not record["caption"]
        ):
            raise ValueError("non-omitted figure caption is missing")
    actual_png = {
        path.relative_to(root).as_posix() for path in (root / "figures").iterdir() if path.is_file()
    }
    if any(not path.is_file() or path.is_symlink() for path in (root / "figures").iterdir()):
        raise ValueError("figures directory contains a non-file entry")
    if actual_png != set(expected_records):
        raise ValueError("figure PNG set does not match non-omitted manifest entries")
    actual_data = {
        path.relative_to(root).as_posix()
        for path in (root / "figure_data").iterdir()
        if path.is_file()
    }
    if any(not path.is_file() or path.is_symlink() for path in (root / "figure_data").iterdir()):
        raise ValueError("figure_data directory contains a non-file entry")
    if actual_data != set(data_paths):
        raise ValueError("figure-data CSV set does not match the figure manifest")


def validate_publication_bundle(path: str | Path) -> bool:
    """Validate a completed publication bundle without sources or matplotlib."""

    root = Path(path)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("publication bundle must be a regular directory")
    expected_top = set(REQUIRED_PUBLICATION_FILES)
    actual_top = {item.name for item in root.iterdir()}
    if actual_top != expected_top:
        raise ValueError("publication bundle top-level file/directory set mismatch")
    for directory in ("figures", "figure_data"):
        if not (root / directory).is_dir() or (root / directory).is_symlink():
            raise ValueError(f"publication {directory} directory is invalid")
    for filename in (
        "manifest.json",
        "figure_manifest.json",
        "report.md",
        "completion.json",
        "publication_metrics.csv",
        "publication_contrasts.csv",
        "publication_summary.csv",
        "f3_regional_summary.csv",
        "f3_orientation_summary.csv",
        "runtime_summary.csv",
    ):
        if not (root / filename).is_file() or (root / filename).is_symlink():
            raise ValueError(f"publication artifact is missing: {filename}")

    completion = _json(root / "completion.json")
    if not isinstance(completion, Mapping) or set(completion) != {
        "completion_schema_version",
        "status",
        "required_files",
        "files",
    }:
        raise ValueError("publication completion field set mismatch")
    if (
        completion["completion_schema_version"] != PUBLICATION_COMPLETION_SCHEMA_VERSION
        or completion["status"] != "complete"
    ):
        raise ValueError("publication completion schema/status mismatch")
    required_files = completion["required_files"]
    if not isinstance(required_files, list) or any(
        not isinstance(item, str)
        or item == "completion.json"
        or item.startswith("/")
        or ".." in Path(item).parts
        for item in required_files
    ):
        raise ValueError("publication completion required file paths are invalid")
    actual_files = set(_relative_files(root))
    if len(required_files) != len(actual_files) or set(required_files) != actual_files:
        raise ValueError("publication completion path set does not match the bundle")
    records = completion["files"]
    if (
        not isinstance(records, list)
        or len(records) != len(actual_files)
        or {item.get("path") for item in records if isinstance(item, Mapping)} != actual_files
    ):
        raise ValueError("publication completion file records do not match the bundle")
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {"path", "size", "sha256"}:
            raise ValueError("publication completion file record is invalid")
        relative = record["path"]
        target = root / relative
        if (
            not target.is_file()
            or target.stat().st_size != record["size"]
            or _digest(target) != record["sha256"]
        ):
            raise ValueError(f"publication completion hash/size mismatch: {relative}")

    manifest = _json(root / "manifest.json")
    if not isinstance(manifest, Mapping):
        raise ValueError("publication manifest must be an object")
    _finite_json(manifest)
    if manifest.get("publication_artifact_schema_version") != PUBLICATION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("publication artifact schema version mismatch")
    if manifest.get("publication_metric_selection_version") != PUBLICATION_METRIC_SELECTION_VERSION:
        raise ValueError("publication metric selection version mismatch")
    if manifest.get("publication_figure_contract_version") != PUBLICATION_FIGURE_CONTRACT_VERSION:
        raise ValueError("publication figure contract version mismatch")
    if manifest.get("interpretation") != PUBLICATION_INTERPRETATION:
        raise ValueError("publication interpretation is invalid")
    expected_policy = {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in FIGURE_SELECTION_POLICY.items()
    }
    if manifest.get("figure_selection_policy") != expected_policy:
        raise ValueError("publication figure-selection policy is invalid")
    if manifest.get("source_paths_are_provenance_only") is not True:
        raise ValueError("publication source paths are not marked as provenance-only")
    if manifest.get("canonical_condition_order") != list(CANONICAL_CELL_ORDER):
        raise ValueError("publication canonical cell order mismatch")
    if manifest.get("canonical_stage_order") != list(CANONICAL_STAGE_ORDER):
        raise ValueError("publication canonical stage order mismatch")
    _validate_source_identity(manifest)
    registry = manifest.get("curated_metric_registry")
    if registry != [entry.as_dict() for entry in PUBLICATION_METRIC_REGISTRY]:
        raise ValueError("publication metric registry does not match the fixed contract")

    metric_rows = _csv(root / "publication_metrics.csv", PUBLICATION_METRICS_HEADER)
    contrast_rows = _csv(root / "publication_contrasts.csv", PUBLICATION_CONTRASTS_HEADER)
    summary_rows = _csv(root / "publication_summary.csv", PUBLICATION_SUMMARY_HEADER)
    _validate_metric_rows(metric_rows, manifest)
    _validate_contrast_rows(contrast_rows)
    _validate_summary(summary_rows, metric_rows)
    _validate_supporting_tables(root)
    _validate_figures(root, manifest)
    if not (root / "report.md").read_text(encoding="utf-8").strip():
        raise ValueError("publication report is empty")
    return True


__all__ = ["validate_publication_bundle"]
