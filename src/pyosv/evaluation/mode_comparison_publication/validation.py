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
    FIGURE_DATA_IDENTITY_FIELDS,
    FIGURE_MANIFEST_FIELDS,
    F3_RIDGE_OVERLAY_SLOTS,
    F3_SPATIAL_FIGURE_SLOTS,
    F3_SEMANTICS,
    FIGURE_SELECTION_POLICY,
    FIXED_SCALAR_FIGURE_IDS,
    PUBLICATION_ARTIFACT_SCHEMA_VERSION,
    PUBLICATION_COMPLETION_SCHEMA_VERSION,
    PUBLICATION_FIGURE_CONTRACT_VERSION,
    PUBLICATION_METRIC_SELECTION_VERSION,
    PUBLICATION_TABLE_CONTRACT_VERSION,
    PUBLICATION_INTERPRETATION,
    REQUIRED_PUBLICATION_FILES,
    ROOT_TABLE_FILES,
    SYNTHETIC_SCANNER_CELL_ORDER,
    SYNTHETIC_SKIN_FIGURE_OMISSION_REASON,
    SYNTHETIC_SEMANTICS,
)
from .registry import PUBLICATION_METRIC_BY_IDENTITY, PUBLICATION_METRIC_REGISTRY
from .semantic import (
    F3_SOURCE_IDENTITY_FIELDS,
    FIGURE_DATA_FIELD_TYPES,
    ROOT_TABLE_FIELD_TYPES,
    ROOT_TABLE_IDENTITY_FIELDS,
    SYNTHETIC_SOURCE_IDENTITY_FIELDS,
    canonical_digest,
    canonical_json_bytes,
    finite_json_normalize,
    png_metadata,
    require_unique_row_identities,
    source_identity_object,
    validate_table_contract,
)
from .summary import TABLE_HEADERS

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_AXES = {
    "RL-REF": ("reference-like", "reference"),
    "RL-QUAL": ("reference-like", "quality"),
    "Q-REF": ("quality", "reference"),
    "Q-QUAL": ("quality", "quality"),
}
_SCANNER_AXES = {
    "RL-SCAN": ("reference-like", None),
    "Q-SCAN": ("quality", None),
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
    try:
        finite_json_normalize(value)
    except ValueError as error:
        raise ValueError(f"{context} contains an invalid or non-finite JSON value") from error


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


def _optional_float(value: Any, context: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid number in {context}") from error
    if not math.isfinite(result):
        raise ValueError(f"non-finite number in {context}")
    return result


def _float(value: Any, context: str) -> float:
    result = _optional_float(value, context)
    if result is None:
        raise ValueError(f"empty number in {context}")
    return result


def _int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"invalid integer in {context}")
    if isinstance(value, int):
        return value
    try:
        return int(value, 10)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid integer in {context}") from error


def _json_field(value: Any, context: str) -> Any:
    if not isinstance(value, str):
        try:
            return finite_json_normalize(value)
        except ValueError as error:
            raise ValueError(f"invalid JSON field in {context}") from error
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON field in {context}") from error
    _finite_json(parsed, context)
    return parsed


def _validate_source_identity(manifest: Mapping[str, Any]) -> None:
    synthetic = manifest.get("synthetic_source")
    f3 = manifest.get("f3_source")
    if not isinstance(synthetic, Mapping) or not isinstance(f3, Mapping):
        raise ValueError("publication manifest source identity is missing")
    for name, source, fields in (
        ("synthetic_source", synthetic, SYNTHETIC_SOURCE_IDENTITY_FIELDS),
        ("f3_source", f3, F3_SOURCE_IDENTITY_FIELDS),
    ):
        for field in ("identity_digest", "completion_sha256", "manifest_sha256"):
            value = source.get(field)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"publication manifest {name}.{field} is not a SHA-256 digest")
        try:
            actual = canonical_digest(source_identity_object(source, fields))
        except ValueError as error:
            raise ValueError(f"publication manifest {name} identity fields are invalid") from error
        if source["identity_digest"] != actual:
            raise ValueError(f"publication manifest {name} identity digest cannot be recomputed")
    if not isinstance(f3.get("run_fingerprint"), str) or not _SHA256.fullmatch(
        f3["run_fingerprint"]
    ):
        raise ValueError("publication F3 run fingerprint is invalid")
    dataset_identity = f3.get("dataset_identity")
    dataset_digest = f3.get("dataset_identity_digest")
    if not isinstance(dataset_identity, Mapping) or not isinstance(dataset_digest, str):
        raise ValueError("publication F3 dataset identity is missing")
    if (
        not _SHA256.fullmatch(dataset_digest)
        or canonical_digest(dataset_identity) != dataset_digest
    ):
        raise ValueError("publication F3 dataset identity digest cannot be recomputed")


def _validate_source_coverage(
    manifest: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    coverage = manifest.get("source_coverage")
    if not isinstance(coverage, Mapping) or set(coverage) != {"synthetic", "f3"}:
        raise ValueError("publication source coverage field set is invalid")
    synthetic = coverage["synthetic"]
    f3 = coverage["f3"]
    if not isinstance(synthetic, Mapping) or set(synthetic) != {
        "case_order",
        "trials",
        "skinning_enabled",
        "expected_scanner_only_cells",
        "expected_end_to_end_cells",
    }:
        raise ValueError("publication synthetic source coverage is invalid")
    case_order = synthetic["case_order"]
    trials = synthetic["trials"]
    if (
        not isinstance(case_order, list)
        or not case_order
        or any(not isinstance(case, str) or not case for case in case_order)
        or len(case_order) != len(set(case_order))
        or not isinstance(trials, list)
        or not trials
        or not isinstance(synthetic["skinning_enabled"], bool)
        or synthetic["expected_scanner_only_cells"] != list(SYNTHETIC_SCANNER_CELL_ORDER)
        or synthetic["expected_end_to_end_cells"] != list(CANONICAL_CELL_ORDER)
    ):
        raise ValueError("publication synthetic source coverage is invalid")
    trial_identities = []
    trials_by_case: dict[str, list[int | None]] = defaultdict(list)
    for record in trials:
        if not isinstance(record, Mapping) or set(record) != {"case_id", "trial_id", "seed"}:
            raise ValueError("publication synthetic trial coverage identity is invalid")
        case_id, trial_id, seed = record["case_id"], record["trial_id"], record["seed"]
        if (
            not isinstance(case_id, str)
            or case_id not in case_order
            or not isinstance(trial_id, str)
            or not trial_id
            or (seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)))
        ):
            raise ValueError("publication synthetic trial coverage identity is invalid")
        trial_identities.append((case_id, trial_id, seed))
        trials_by_case[case_id].append(seed)
    if len(trial_identities) != len(set(trial_identities)):
        raise ValueError("publication synthetic trial coverage has duplicate identities")
    if set(trials_by_case) != set(case_order):
        raise ValueError("publication synthetic trial coverage omits a declared case")
    if any(
        any(seed is None for seed in seeds) and len(seeds) != 1 for seeds in trials_by_case.values()
    ):
        raise ValueError("deterministic synthetic cases must have exactly one trial")

    if not isinstance(f3, Mapping) or set(f3) != {
        "evaluation_unit_count",
        "canonical_cell_order",
        "skinning_enabled_by_cell",
        "volume_shape",
        "dataset_id",
        "run_fingerprint",
    }:
        raise ValueError("publication F3 source coverage is invalid")
    skinning = f3["skinning_enabled_by_cell"]
    shape = f3["volume_shape"]
    if (
        isinstance(f3["evaluation_unit_count"], bool)
        or not isinstance(f3["evaluation_unit_count"], int)
        or f3["evaluation_unit_count"] != 1
        or f3["canonical_cell_order"] != list(CANONICAL_CELL_ORDER)
        or not isinstance(skinning, Mapping)
        or set(skinning) != set(CANONICAL_CELL_ORDER)
        or any(not isinstance(value, bool) for value in skinning.values())
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape
        )
        or not isinstance(f3["dataset_id"], str)
        or not f3["dataset_id"]
        or not isinstance(f3["run_fingerprint"], str)
        or not _SHA256.fullmatch(f3["run_fingerprint"])
    ):
        raise ValueError("publication F3 source coverage is invalid")
    f3_source = manifest["f3_source"]
    if f3["dataset_id"] != f3_source.get("dataset_id") or f3["run_fingerprint"] != f3_source.get(
        "run_fingerprint"
    ):
        raise ValueError("publication F3 source coverage does not match source identity")
    return synthetic, f3


def _identity_set(
    rows: tuple[dict[str, Any], ...], identity_fields: tuple[str, ...], *, context: str
) -> set[bytes]:
    values: set[bytes] = set()
    for row in rows:
        values.add(canonical_json_bytes({name: row[name] for name in identity_fields}))
    if len(values) != len(rows):
        raise ValueError(f"duplicate publication row identity: {context}")
    return values


def _validate_source_expected_identities(
    rows: tuple[dict[str, Any], ...],
    contract: Mapping[str, Any],
    identity_fields: tuple[str, ...],
    *,
    context: str,
) -> None:
    expected = contract.get("source_expected_identities")
    if not isinstance(expected, list):
        raise ValueError(f"publication source expected identities are missing: {context}")
    encoded_expected: set[bytes] = set()
    for identity in expected:
        if not isinstance(identity, Mapping) or set(identity) != set(identity_fields):
            raise ValueError(f"publication source expected identity is invalid: {context}")
        try:
            encoded_expected.add(
                canonical_json_bytes({name: identity[name] for name in identity_fields})
            )
        except ValueError as error:
            raise ValueError(
                f"publication source expected identity is invalid: {context}"
            ) from error
    if len(encoded_expected) != len(expected):
        raise ValueError(f"publication source expected identities are duplicated: {context}")
    if _identity_set(rows, identity_fields, context=context) != encoded_expected:
        raise ValueError(f"publication table source coverage mismatch: {context}")


def _validate_table_contracts(
    manifest: Mapping[str, Any], root: Path
) -> dict[str, tuple[dict[str, Any], ...]]:
    contracts = manifest.get("table_contracts")
    if not isinstance(contracts, Mapping) or set(contracts) != set(ROOT_TABLE_FILES):
        raise ValueError("publication table contract coverage is invalid")
    typed_rows: dict[str, tuple[dict[str, Any], ...]] = {}
    for filename in ROOT_TABLE_FILES:
        contract = contracts[filename]
        if not isinstance(contract, Mapping):
            raise ValueError(f"publication table contract is invalid: {filename}")
        if contract.get("identity_fields") != list(ROOT_TABLE_IDENTITY_FIELDS[filename]):
            raise ValueError(f"publication table identity contract is invalid: {filename}")
        rows = _csv(root / filename, TABLE_HEADERS[filename])
        parsed = validate_table_contract(
            contract,
            TABLE_HEADERS[filename],
            rows,
            ROOT_TABLE_FIELD_TYPES[filename],
            context=filename,
        )
        identity_fields = ROOT_TABLE_IDENTITY_FIELDS[filename]
        require_unique_row_identities(parsed, identity_fields, context=filename)
        if filename != "publication_summary.csv":
            _validate_source_expected_identities(
                parsed,
                contract,
                identity_fields,
                context=filename,
            )
        typed_rows[filename] = parsed
    return typed_rows


def _validate_metric_rows(
    rows: tuple[dict[str, Any], ...],
    synthetic_coverage: Mapping[str, Any],
    f3_coverage: Mapping[str, Any],
) -> None:
    by_identity: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset = row["dataset"]
        if dataset not in {"synthetic", "f3"}:
            raise ValueError("publication metric dataset is invalid")
        expected_semantics = SYNTHETIC_SEMANTICS if dataset == "synthetic" else F3_SEMANTICS
        if row["evaluation_semantics"] != expected_semantics:
            raise ValueError("publication metric evaluation semantics are invalid")
        if dataset == "f3":
            if (
                row["case_or_region"] != "full"
                or row["trial_id"] is not None
                or row["seed"] is not None
            ):
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

    source_skinning = synthetic_coverage["skinning_enabled"]
    f3_skinning_by_cell = f3_coverage["skinning_enabled_by_cell"]
    synthetic_trials = tuple(
        (item["case_id"], item["trial_id"], item["seed"]) for item in synthetic_coverage["trials"]
    )
    for entry in PUBLICATION_METRIC_REGISTRY:
        rows_for_entry = by_identity.get(entry.identity, [])
        if entry.dataset == "synthetic":
            if entry.stage == "skin" and not source_skinning:
                if rows_for_entry:
                    raise ValueError("synthetic skin rows exist without a skinning source")
                continue
            if not rows_for_entry:
                raise ValueError(
                    f"required publication metric coverage is missing: {entry.identity!r}"
                )
            expected_cells = (
                set(SYNTHETIC_SCANNER_CELL_ORDER)
                if entry.stage == "scanner_raw"
                else set(CANONICAL_CELL_ORDER)
            )
            observed: dict[tuple[str, str, int | None], set[str]] = defaultdict(set)
            for row in rows_for_entry:
                observed[(row["case_or_region"], row["trial_id"], row["seed"])].add(
                    row["cell_label"]
                )
            if set(observed) != set(synthetic_trials) or any(
                cells != expected_cells for cells in observed.values()
            ):
                raise ValueError(
                    f"synthetic publication metric coverage is incomplete: {entry.identity!r}"
                )
            continue

        enabled_cells = {cell for cell, enabled in f3_skinning_by_cell.items() if enabled}
        expected_cells = set(CANONICAL_CELL_ORDER)
        if entry.stage == "skin":
            if not enabled_cells:
                if rows_for_entry:
                    raise ValueError("F3 skin rows exist without a skinning source")
                continue
            expected_cells = enabled_cells
        if not rows_for_entry:
            raise ValueError(f"required publication metric coverage is missing: {entry.identity!r}")
        if (
            len(rows_for_entry) != len(expected_cells)
            or {row["cell_label"] for row in rows_for_entry} != expected_cells
        ):
            raise ValueError(
                f"F3 publication metric cell coverage is incomplete: {entry.identity!r}"
            )

    entry_order = {entry.identity: index for index, entry in enumerate(PUBLICATION_METRIC_REGISTRY)}
    case_order = {case: index for index, case in enumerate(synthetic_coverage["case_order"])}
    scanner_order = {cell: index for index, cell in enumerate(SYNTHETIC_SCANNER_CELL_ORDER)}
    cell_order = {cell: index for index, cell in enumerate(CANONICAL_CELL_ORDER)}

    def order(row: Mapping[str, Any]) -> tuple[Any, ...]:
        entry = (row["dataset"], row["stage"], row["selection"], row["metric"])
        if row["dataset"] == "synthetic":
            cells = scanner_order if row["stage"] == "scanner_raw" else cell_order
            return (
                entry_order[entry],
                case_order[row["case_or_region"]],
                row["trial_id"],
                -1 if row["seed"] is None else row["seed"],
                cells[row["cell_label"]],
            )
        return (entry_order[entry], 0, "", -1, cell_order[row["cell_label"]])

    if list(rows) != sorted(rows, key=order):
        raise ValueError("publication metrics are not in canonical row order")


def _validate_contrast_rows(
    rows: tuple[dict[str, Any], ...], metric_rows: tuple[dict[str, Any], ...]
) -> None:
    if not rows:
        raise ValueError("publication contrasts must not be an empty header-only table")
    definitions = {
        "synthetic": {item.name: item for item in SYNTHETIC_CONTRAST_DEFINITIONS},
        "f3": {item.name: item for item in F3_CONTRAST_DEFINITIONS},
    }
    metric_values: dict[tuple[Any, ...], float | None] = {}
    for row in metric_rows:
        metric_values[
            (
                row["dataset"],
                row["case_or_region"],
                row["trial_id"],
                row["seed"],
                row["stage"],
                row["selection"],
                row["metric"],
                row["cell_label"],
            )
        ] = _optional_float(row["value"], "publication metric value")
    for row in rows:
        dataset = row["dataset"]
        if dataset not in {"synthetic", "f3"}:
            raise ValueError("publication contrast dataset is invalid")
        expected = SYNTHETIC_SEMANTICS if dataset == "synthetic" else F3_SEMANTICS
        if row["evaluation_semantics"] != expected:
            raise ValueError("contrast evaluation semantics are invalid")
        if dataset == "f3" and (
            row["case_or_region"] != "full"
            or row["trial_id"] is not None
            or row["seed"] is not None
        ):
            raise ValueError("F3 contrasts must describe one full-volume unit")
        if dataset == "synthetic" and (not row["case_or_region"] or not row["trial_id"]):
            raise ValueError("synthetic contrast case/trial identity is missing")
        if row["contrast_name"] not in CONTRAST_NAMES:
            raise ValueError("publication contrast is outside the fixed public contrast set")
        identity = (dataset, row["stage"], row["selection"], row["metric"])
        entry = PUBLICATION_METRIC_BY_IDENTITY.get(identity)
        if entry is None:
            raise ValueError(f"contrast metric is outside the curated registry: {identity!r}")
        if (row["unit"], row["direction"]) != (entry.unit, entry.direction):
            raise ValueError("contrast metric unit or direction does not match the registry")
        definition = definitions[dataset].get(row["contrast_name"])
        if definition is None:
            raise ValueError("unknown contrast definition")
        cells = _json_field(row["component_cells"], "component_cells")
        if not isinstance(cells, list) or tuple(cells) != definition.component_cells:
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
        elif improvement is None or not math.isclose(
            improvement,
            expected_improvement,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("direction and improvement_value are inconsistent")
        if row["source_artifact"] not in {"contrasts.csv", "reports/contrasts.csv"}:
            raise ValueError("contrast source artifact is invalid")

        component_values = []
        for cell, coefficient in definition.coefficients:
            key = (
                dataset,
                row["case_or_region"],
                row["trial_id"],
                row["seed"],
                row["stage"],
                row["selection"],
                row["metric"],
                cell,
            )
            try:
                component = metric_values[key]
            except KeyError as error:
                raise ValueError("contrast component metric coverage is missing") from error
            if component is None:
                raise ValueError("contrast component metric must not be nullable/empty")
            component_values.append(component * float(coefficient))
        recomputed_raw = math.fsum(component_values)
        tolerance = (1.0e-7, 1.0e-9) if dataset == "synthetic" else (1.0e-9, 1.0e-12)
        if not math.isclose(raw, recomputed_raw, rel_tol=tolerance[0], abs_tol=tolerance[1]):
            raise ValueError("contrast raw_value cannot be recomputed from publication metrics")

    contrast_order = {name: index for index, name in enumerate(CONTRAST_NAMES)}
    metric_order = {
        entry.identity: index for index, entry in enumerate(PUBLICATION_METRIC_REGISTRY)
    }
    if list(rows) != sorted(
        rows,
        key=lambda row: (
            0 if row["dataset"] == "synthetic" else 1,
            row["case_or_region"],
            row["trial_id"] or "",
            -1 if row["seed"] is None else row["seed"],
            contrast_order[row["contrast_name"]],
            metric_order[(row["dataset"], row["stage"], row["selection"], row["metric"])],
        ),
    ):
        raise ValueError("publication contrasts are not in canonical row order")


def _validate_summary(
    summary_rows: tuple[dict[str, Any], ...],
    metric_rows: tuple[dict[str, Any], ...],
    synthetic_coverage: Mapping[str, Any],
) -> None:
    groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
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
            "q25": float(np.quantile(values, 0.25, method="linear")),
            "q75": float(np.quantile(values, 0.75, method="linear")),
        }
        if _int(row["n"], "summary n") != expected["n"]:
            raise ValueError("summary n cannot be recomputed from publication metrics")
        if row["dataset"] == "f3" and expected["n"] != 1:
            raise ValueError("F3 publication summary n must equal one full-volume unit")
        if row["dataset"] == "synthetic":
            expected_trials = sum(
                item["case_id"] == row["case_or_region"] for item in synthetic_coverage["trials"]
            )
            if expected["n"] != expected_trials:
                raise ValueError("synthetic publication summary n does not match source trials")
        for name in ("mean", "median", "minimum", "maximum", "q25", "q75"):
            if not math.isclose(
                _float(row[name], f"summary {name}"), expected[name], rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"summary {name} cannot be recomputed from publication metrics")
    if actual != set(groups):
        raise ValueError("publication summary does not cover all non-null metric groups")

    dataset_order = {"synthetic": 0, "f3": 1}
    case_order = {
        case: index for index, case in enumerate(synthetic_coverage["case_order"] + ["full"])
    }
    stage_order = {
        stage: index
        for index, stage in enumerate(("scanner_raw", "fvt", "skin", "ft", "fv", "fvt", "skin"))
    }
    cell_order = {
        cell: index
        for index, cell in enumerate(SYNTHETIC_SCANNER_CELL_ORDER + CANONICAL_CELL_ORDER)
    }
    if list(summary_rows) != sorted(
        summary_rows,
        key=lambda row: (
            dataset_order.get(row["dataset"], 99),
            case_order.get(row["case_or_region"], 99),
            stage_order.get(row["stage"], 99),
            row["selection"],
            row["metric"],
            cell_order.get(row["cell_label"], 99),
        ),
    ):
        raise ValueError("publication summary is not in canonical row order")


def _validate_expected_identity_order(
    rows: tuple[dict[str, Any], ...],
    contract: Mapping[str, Any],
    identity_fields: tuple[str, ...],
    *,
    context: str,
) -> None:
    expected = contract["source_expected_identities"]
    actual_encoded = [
        canonical_json_bytes({name: row[name] for name in identity_fields}) for row in rows
    ]
    expected_encoded = [
        canonical_json_bytes({name: identity[name] for name in identity_fields})
        for identity in expected
    ]
    if actual_encoded != expected_encoded:
        raise ValueError(
            f"publication supporting table is not in source canonical order: {context}"
        )


def _validate_supporting_tables(
    typed_rows: Mapping[str, tuple[dict[str, Any], ...]], manifest: Mapping[str, Any]
) -> None:
    contracts = manifest["table_contracts"]
    regional = typed_rows["f3_regional_summary.csv"]
    if not regional:
        raise ValueError("F3 regional summary must not be empty")
    if any(
        row["dataset"] != "f3" or row["evaluation_semantics"] != F3_SEMANTICS for row in regional
    ):
        raise ValueError("F3 regional summary semantics are invalid")
    if {row["cell_label"] for row in regional} != set(CANONICAL_CELL_ORDER):
        raise ValueError("F3 regional summary does not cover canonical cells")
    if not {"interior", "boundary_shell"} <= {row["region"] for row in regional}:
        raise ValueError("F3 regional summary lacks interior/boundary_shell coverage")
    for row in regional:
        if row["case_or_region"] != row["region"]:
            raise ValueError("F3 regional case/region identity is invalid")
        if row["source_artifact"] != "reports/regional_metrics.csv":
            raise ValueError("F3 regional source artifact is invalid")
        _optional_float(row["value"], "f3_regional_summary.csv.value")
    _validate_expected_identity_order(
        regional,
        contracts["f3_regional_summary.csv"],
        ROOT_TABLE_IDENTITY_FIELDS["f3_regional_summary.csv"],
        context="f3_regional_summary.csv",
    )

    orientation = typed_rows["f3_orientation_summary.csv"]
    if not orientation:
        raise ValueError("F3 orientation summary must not be empty")
    if any(
        row["dataset"] != "f3" or row["evaluation_semantics"] != F3_SEMANTICS for row in orientation
    ):
        raise ValueError("F3 orientation summary semantics are invalid")
    support_counts: dict[tuple[str, str, str, str, str], int] = {}
    count_groups: dict[tuple[str, str, str, str, str], int] = defaultdict(int)
    for row in orientation:
        if row["case_or_region"] != "full":
            raise ValueError("F3 orientation summary must describe the full volume")
        if row["source_artifact"] != "reports/orientation_diagnostics.csv":
            raise ValueError("F3 orientation source artifact is invalid")
        group = (
            row["dataset"],
            row["stage"],
            row["left_cell"],
            row["right_cell"],
            row["support_contract"],
        )
        if not row["support_contract"]:
            raise ValueError("F3 orientation support contract is missing")
        support = _int(row["support_count"], "f3_orientation_summary.csv.support_count")
        if support < 0:
            raise ValueError("F3 orientation support count is negative")
        previous = support_counts.setdefault(group, support)
        if previous != support:
            raise ValueError("F3 orientation support count is inconsistent within a group")
        value = _optional_float(row["value"], "f3_orientation_summary.csv.value")
        if row["metric"].endswith(".count"):
            count_groups[group] += 1
            if value is None or not math.isclose(value, support, abs_tol=0.0):
                raise ValueError("F3 orientation count metric does not match support_count")
        elif support == 0 and value is not None:
            raise ValueError("nullable orientation summary must remain empty at zero support")
        elif support > 0 and value is None:
            raise ValueError("non-empty orientation support requires a summary value")
    if set(count_groups) != set(support_counts) or any(
        count != 3 for count in count_groups.values()
    ):
        raise ValueError("F3 orientation count metric does not match support_count")
    _validate_expected_identity_order(
        orientation,
        contracts["f3_orientation_summary.csv"],
        ROOT_TABLE_IDENTITY_FIELDS["f3_orientation_summary.csv"],
        context="f3_orientation_summary.csv",
    )

    runtime = typed_rows["runtime_summary.csv"]
    if not runtime:
        raise ValueError("runtime summary must not be empty")
    for row in runtime:
        dataset = row["dataset"]
        expected_semantics = SYNTHETIC_SEMANTICS if dataset == "synthetic" else F3_SEMANTICS
        if dataset not in {"synthetic", "f3"} or row["evaluation_semantics"] != expected_semantics:
            raise ValueError("runtime summary semantics are invalid")
        if _float(row["elapsed_seconds"], "runtime_summary.csv.elapsed_seconds") < 0.0:
            raise ValueError("runtime elapsed_seconds must be non-negative")
        if _int(row["call_count"], "runtime_summary.csv.call_count") < 0:
            raise ValueError("runtime call_count must be non-negative")
        consumers = _json_field(row["cell_consumers"], "runtime_summary.csv.cell_consumers")
        if not isinstance(consumers, list) or any(
            not isinstance(cell, str) or cell not in CANONICAL_CELL_ORDER for cell in consumers
        ):
            raise ValueError("runtime cell_consumers are invalid")
        if row["cell_label"] == "PUBLIC-REF" or "PUBLIC-REF" in consumers:
            raise ValueError("PUBLIC-REF must not be registered as a runtime processing cell")
        shared = row["shared_stage"]
        if dataset == "synthetic":
            if consumers != [] or row["fingerprint"] is not None:
                raise ValueError("synthetic runtime ownership fields are invalid")
            if (
                row["source_artifact"] != "runtime.csv"
                or row["elapsed_semantics"] != "within_experiment_attribution"
            ):
                raise ValueError("synthetic runtime source semantics are invalid")
            expected_state = "shared" if shared else "cell-owned"
        else:
            if (
                row["case_or_region"] != "full"
                or row["trial_id"] is not None
                or row["seed"] is not None
            ):
                raise ValueError("F3 runtime must describe one full-volume unit")
            if row["scanner_backend"] is not None or row["call_count"] != 1:
                raise ValueError("F3 runtime source event fields are invalid")
            if (
                row["source_artifact"] != "reports/runtime.csv"
                or row["state"] not in {"computed", "reused"}
                or row["elapsed_semantics"]
                != ("compute" if row["state"] == "computed" else "load_validation")
            ):
                raise ValueError("F3 runtime source semantics are invalid")
            if not isinstance(row["fingerprint"], str) or not _SHA256.fullmatch(row["fingerprint"]):
                raise ValueError("F3 runtime fingerprint is invalid")
            if row["cell_label"] not in consumers:
                raise ValueError("F3 runtime owner is not a consumer")
            if shared != (len(consumers) > 1):
                raise ValueError("F3 shared runtime semantics are invalid")
            expected_state = row["state"]
        if row["state"] != expected_state:
            raise ValueError("runtime state/shared ownership semantics are invalid")
        expected_attribution = "shared-stage" if shared else "cell-owned-stage"
        if row["attribution"] != expected_attribution:
            raise ValueError("runtime attribution semantics are invalid")
    _validate_expected_identity_order(
        runtime,
        contracts["runtime_summary.csv"],
        ROOT_TABLE_IDENTITY_FIELDS["runtime_summary.csv"],
        context="runtime_summary.csv",
    )


def _runtime_panel_label(row: Mapping[str, Any]) -> str:
    owner = row["cell_label"] or "shared"
    pieces = [str(row["stage"]), str(owner), "shared" if row["shared_stage"] else "cell-owned"]
    for value in (row["scanner_backend"], row["fingerprint"]):
        if value:
            pieces.append(str(value))
    pieces.append(f"calls={row['call_count']}")
    if row["state"]:
        pieces.append(str(row["state"]))
    return ":".join(pieces)


def _validate_runtime_figure_data(
    record: Mapping[str, Any],
    data_rows: tuple[dict[str, Any], ...],
    runtime_rows: tuple[dict[str, Any], ...],
) -> None:
    source_rows = tuple(row for row in runtime_rows if row["dataset"] == record["dataset"])
    if len(data_rows) != len(source_rows):
        raise ValueError("runtime figure-data does not cover all source runtime events")
    for data, source in zip(data_rows, source_rows, strict=True):
        if (
            data["source_metric"] != source["stage"]
            or data["source_stage"] != "runtime"
            or data["case_or_region"] != source["case_or_region"]
            or data["trial_id"] != source["trial_id"]
            or data["seed"] != source["seed"]
            or data["cell_label"] != source["cell_label"]
            or data["panel_label"] != _runtime_panel_label(source)
            or data["metric"] != "elapsed_seconds"
            or data["unit"] != "second"
            or data["value"] != source["elapsed_seconds"]
        ):
            raise ValueError("runtime figure-data does not preserve source runtime identity")
    expected_labels = [_runtime_panel_label(row) for row in source_rows]
    expected_cells = list(
        dict.fromkeys(row["cell_label"] for row in source_rows if row["cell_label"] is not None)
    )
    if record["panel_labels"] != expected_labels or record["cell_labels"] != expected_cells:
        raise ValueError("runtime figure panels/cells do not match source runtime events")


def _validate_figures(
    root: Path,
    manifest: Mapping[str, Any],
    runtime_rows: tuple[dict[str, Any], ...],
) -> None:
    figure_manifest = _json(root / "figure_manifest.json")
    if not isinstance(figure_manifest, Mapping):
        raise ValueError("figure_manifest must be an object")
    figure_version = figure_manifest.get("publication_figure_contract_version")
    if figure_version == 1:
        raise ValueError(
            "publication figure contract version 1 predates the semantic figure-data contract; "
            "regenerate the publication bundle"
        )
    if figure_version != PUBLICATION_FIGURE_CONTRACT_VERSION:
        raise ValueError("figure contract version mismatch")
    if set(figure_manifest) != {
        "publication_figure_contract_version",
        "canonical_condition_order",
        "canonical_stage_order",
        "volume_shape",
        "fixed_scalar_figure_ids",
        "f3_spatial_figure_slots",
        "f3_ridge_overlay_slots",
        "figures",
    }:
        raise ValueError("figure_manifest field set mismatch")
    if tuple(figure_manifest.get("canonical_condition_order", ())) != CANONICAL_CELL_ORDER:
        raise ValueError("figure canonical condition order mismatch")
    if tuple(figure_manifest.get("canonical_stage_order", ())) != CANONICAL_STAGE_ORDER:
        raise ValueError("figure canonical stage order mismatch")
    if figure_manifest.get("fixed_scalar_figure_ids") != list(FIXED_SCALAR_FIGURE_IDS):
        raise ValueError("figure fixed scalar coverage contract mismatch")
    if figure_manifest.get("f3_spatial_figure_slots") != [
        list(slot) for slot in F3_SPATIAL_FIGURE_SLOTS
    ]:
        raise ValueError("figure spatial slot coverage contract mismatch")
    if figure_manifest.get("f3_ridge_overlay_slots") != [
        list(slot) for slot in F3_RIDGE_OVERLAY_SLOTS
    ]:
        raise ValueError("figure ridge-overlay slot coverage contract mismatch")
    shape = tuple(figure_manifest.get("volume_shape", ()))
    if len(shape) != 3 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape
    ):
        raise ValueError("figure volume shape is invalid")
    if list(shape) != manifest["source_coverage"]["f3"]["volume_shape"]:
        raise ValueError("figure volume shape does not match F3 source coverage")
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
    observed_spatial_slots: list[tuple[str, str, str]] = []
    observed_ridge_slots: list[tuple[str, str]] = []
    record_by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or set(record) != set(FIGURE_MANIFEST_FIELDS):
            raise ValueError("figure record field set mismatch")
        _finite_json(record)
        record_by_id[record["figure_id"]] = record
        relative = record["relative_path"]
        if (
            not isinstance(relative, str)
            or not relative.startswith("figures/")
            or ".." in Path(relative).parts
            or Path(relative).parent != Path("figures")
        ):
            raise ValueError("figure relative path is invalid")
        if relative != f"figures/{record['figure_id']}.png":
            raise ValueError("figure relative path does not match its deterministic figure ID")
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
        category = record["category"]
        if category not in {
            "synthetic_scalar",
            "synthetic_contrast",
            "f3_scalar",
            "runtime",
            "f3_spatial",
            "f3_ridge_overlay",
        }:
            raise ValueError("figure category is invalid")
        figure_id = record["figure_id"]
        if figure_id in FIXED_SCALAR_FIGURE_IDS:
            expected_fixed = {
                "synthetic_end_to_end_improvement_heatmap": (
                    "synthetic",
                    "synthetic_contrast",
                    "synthetic",
                ),
                "synthetic_fvt_buffered_f1_by_case": ("synthetic", "synthetic_scalar", "fvt"),
                "synthetic_fvt_hausdorff_p95_by_case": ("synthetic", "synthetic_scalar", "fvt"),
                "synthetic_scanner_orientation_error_by_case": (
                    "synthetic",
                    "synthetic_scalar",
                    "scanner_raw",
                ),
                "synthetic_skin_buffered_f1_by_case": ("synthetic", "synthetic_scalar", "skin"),
                "synthetic_runtime_breakdown": ("synthetic", "runtime", "runtime"),
                "f3_normalized_correlation_by_stage": ("f3", "f3_scalar", "ft,fv,fvt"),
                "f3_buffered_f1_by_stage": ("f3", "f3_scalar", "ft,fv,fvt"),
                "f3_sparse_distance_p95_by_stage": ("f3", "f3_scalar", "ft,fv,fvt"),
                "f3_nonzero_fraction_ratio_by_stage": ("f3", "f3_scalar", "ft,fv,fvt"),
                "f3_runtime_breakdown": ("f3", "runtime", "runtime"),
            }[figure_id]
            if (dataset, category, record["source_stage"]) != expected_fixed:
                raise ValueError("fixed scalar figure identity/category/stage is invalid")
        if not isinstance(record["panel_labels"], list) or any(
            not isinstance(value, str) or not value for value in record["panel_labels"]
        ):
            raise ValueError("figure panel labels are invalid")
        fixed_panels = {
            "synthetic_end_to_end_improvement_heatmap": [
                entry.metric
                for entry in PUBLICATION_METRIC_REGISTRY
                if entry.dataset == "synthetic"
                and entry.stage in {"fvt", "skin"}
                and entry.direction != "neutral"
            ],
            "synthetic_fvt_buffered_f1_by_case": list(CANONICAL_CELL_ORDER),
            "synthetic_fvt_hausdorff_p95_by_case": list(CANONICAL_CELL_ORDER),
            "synthetic_scanner_orientation_error_by_case": list(SYNTHETIC_SCANNER_CELL_ORDER),
            "synthetic_skin_buffered_f1_by_case": list(CANONICAL_CELL_ORDER),
            "f3_normalized_correlation_by_stage": list(CANONICAL_CELL_ORDER),
            "f3_buffered_f1_by_stage": list(CANONICAL_CELL_ORDER),
            "f3_sparse_distance_p95_by_stage": list(CANONICAL_CELL_ORDER),
            "f3_nonzero_fraction_ratio_by_stage": list(CANONICAL_CELL_ORDER),
        }
        if figure_id in fixed_panels and record["panel_labels"] != fixed_panels[figure_id]:
            raise ValueError("fixed scalar figure panel labels are invalid")
        fixed_cells = {
            "synthetic_end_to_end_improvement_heatmap": list(CANONICAL_CELL_ORDER),
            "synthetic_fvt_buffered_f1_by_case": list(CANONICAL_CELL_ORDER),
            "synthetic_fvt_hausdorff_p95_by_case": list(CANONICAL_CELL_ORDER),
            "synthetic_scanner_orientation_error_by_case": list(SYNTHETIC_SCANNER_CELL_ORDER),
            "synthetic_skin_buffered_f1_by_case": list(CANONICAL_CELL_ORDER),
            "f3_normalized_correlation_by_stage": list(CANONICAL_CELL_ORDER),
            "f3_buffered_f1_by_stage": list(CANONICAL_CELL_ORDER),
            "f3_sparse_distance_p95_by_stage": list(CANONICAL_CELL_ORDER),
            "f3_nonzero_fraction_ratio_by_stage": list(CANONICAL_CELL_ORDER),
        }
        if figure_id in fixed_cells and record["cell_labels"] != fixed_cells[figure_id]:
            raise ValueError("fixed scalar figure cell labels are invalid")
        figure_path = root / relative
        axis = record["axis"]
        index = record["slice_index"]
        policy = record["slice_selection_policy"]
        if category == "f3_spatial":
            if (
                dataset != "f3"
                or not isinstance(record["source_stage"], str)
                or (record["source_stage"], policy, axis) not in F3_SPATIAL_FIGURE_SLOTS
                or figure_id != f"f3_{record['source_stage']}_comparison_{policy}_{axis}_{index}"
            ):
                raise ValueError("spatial figure fixed slot identity is invalid")
            expected_panels = (
                [
                    "PUBLIC-REF fl.dat",
                    "reference-like scanner ft",
                    "quality scanner ft",
                    "quality - reference-like signed difference",
                ]
                if record["source_stage"] == "ft"
                else [
                    "PUBLIC-REF",
                    "RL-REF",
                    "RL-QUAL",
                    "Q-REF",
                    "Q-QUAL",
                    "Q-QUAL - RL-REF signed difference",
                ]
            )
            if record["panel_labels"] != expected_panels:
                raise ValueError("spatial figure panel labels are invalid")
            expected_cells = (
                ["RL-REF", "Q-REF"]
                if record["source_stage"] == "ft"
                else list(CANONICAL_CELL_ORDER)
            )
            if record["cell_labels"] != expected_cells:
                raise ValueError("spatial figure cell labels are invalid")
            observed_spatial_slots.append((record["source_stage"], policy, axis))
        if category == "f3_ridge_overlay":
            if (
                dataset != "f3"
                or record["source_stage"] != "fvt"
                or (policy, axis) not in F3_RIDGE_OVERLAY_SLOTS
                or figure_id != f"f3_fvt_ridge_overlay_{policy}_{axis}_{index}"
                or record["panel_labels"]
                != [f"PUBLIC-REF vs {cell}" for cell in CANONICAL_CELL_ORDER]
                or record["cell_labels"] != list(CANONICAL_CELL_ORDER)
            ):
                raise ValueError("ridge-overlay figure fixed slot identity is invalid")
            observed_ridge_slots.append((policy, axis))
        omitted = record["omitted"]
        if not isinstance(omitted, bool):
            raise ValueError("figure omitted must be boolean")
        if omitted:
            if figure_id != "synthetic_skin_buffered_f1_by_case":
                raise ValueError("only the fixed synthetic skin figure may be omitted")
            if figure_path.exists():
                raise ValueError("omitted figure has a PNG file")
            if not isinstance(record["omission_reason"], str) or not record["omission_reason"]:
                raise ValueError("omitted figure has no reason")
            if any(
                record[name] is not None
                for name in (
                    "figure_data_csv",
                    "figure_data_row_count",
                    "figure_data_identity_fields",
                    "figure_data_identity_sha256",
                    "figure_data_semantic_sha256",
                    "pixel_width",
                    "pixel_height",
                    "png_size",
                    "png_sha256",
                )
            ):
                raise ValueError("omitted figure must not have PNG or figure-data metadata")
        else:
            if not figure_path.is_file() or figure_path.is_symlink():
                raise ValueError("non-omitted figure PNG is missing")
            if record["omission_reason"] is not None:
                raise ValueError("non-omitted figure has an omission reason")
            if any(
                isinstance(record[name], bool) or not isinstance(record[name], int)
                for name in ("pixel_width", "pixel_height", "png_size")
            ) or not isinstance(record["png_sha256"], str):
                raise ValueError("non-omitted figure PNG metadata types are invalid")
            actual_png = png_metadata(figure_path)
            if any(record[name] != actual_png[name] for name in actual_png):
                raise ValueError("figure PNG IHDR metadata does not match the manifest")
            data_path = record["figure_data_csv"]
            if (
                not isinstance(data_path, str)
                or not data_path.startswith("figure_data/")
                or ".." in Path(data_path).parts
                or Path(data_path).parent != Path("figure_data")
            ):
                raise ValueError("figure data path is invalid")
            if data_path != f"figure_data/{figure_id}.csv":
                raise ValueError("figure-data path does not match its deterministic figure ID")
            if data_path in data_path_set:
                raise ValueError("figure-data paths must be unique")
            data_path_set.add(data_path)
            if not (root / data_path).is_file():
                raise ValueError("figure data CSV is missing")
            if (
                record["figure_data_identity_fields"] != list(FIGURE_DATA_IDENTITY_FIELDS)
                or isinstance(record["figure_data_row_count"], bool)
                or not isinstance(record["figure_data_row_count"], int)
                or not isinstance(record["figure_data_identity_sha256"], str)
                or not isinstance(record["figure_data_semantic_sha256"], str)
            ):
                raise ValueError("figure-data semantic contract is invalid")
            data_rows = validate_table_contract(
                {
                    "header": list(FIGURE_DATA_HEADER),
                    "row_count": record["figure_data_row_count"],
                    "identity_fields": record["figure_data_identity_fields"],
                    "ordered_identity_sha256": record["figure_data_identity_sha256"],
                    "ordered_semantic_rows_sha256": record["figure_data_semantic_sha256"],
                },
                FIGURE_DATA_HEADER,
                _csv(root / data_path, FIGURE_DATA_HEADER),
                FIGURE_DATA_FIELD_TYPES,
                context=data_path,
            )
            require_unique_row_identities(
                data_rows,
                FIGURE_DATA_IDENTITY_FIELDS,
                context=data_path,
            )
            if not data_rows:
                raise ValueError("figure data CSV is empty")
            if category == "runtime":
                _validate_runtime_figure_data(record, data_rows, runtime_rows)
            allowed_source_stages = {
                item for item in str(record["source_stage"]).split(",") if item
            }
            if any(
                row["figure_id"] != record["figure_id"]
                or row["dataset"] != dataset
                or row["evaluation_semantics"] != expected_semantics
                or row["source_stage"] not in allowed_source_stages
                or (
                    row["panel_label"] is not None
                    and row["panel_label"] not in record["panel_labels"]
                )
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
                expected_axis = axis
                if row["axis"] != expected_axis:
                    raise ValueError("figure data axis does not match the manifest")
                expected_index = index
                if row["slice_index"] != expected_index:
                    raise ValueError("figure data slice index does not match the manifest")
                expected_policy = policy
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
                        row_limit = _optional_float(
                            row["difference_limit"], f"{data_path}.difference_limit"
                        )
                        if row_limit != scale.get("difference_limit"):
                            raise ValueError("difference-panel limit does not match the manifest")
            data_paths.append(data_path)
            expected_records.append(relative)
        cells = record["cell_labels"]
        allowed_figure_cells = set(CANONICAL_CELL_ORDER)
        if record["dataset"] == "synthetic" and record["source_stage"] in {
            "scanner_raw",
            "runtime",
        }:
            allowed_figure_cells.update(SYNTHETIC_SCANNER_CELL_ORDER)
        if not isinstance(cells, list) or any(
            not isinstance(cell, str)
            or not cell
            or (category != "runtime" and cell not in allowed_figure_cells)
            or (category == "runtime" and dataset == "f3" and cell not in allowed_figure_cells)
            for cell in cells
        ):
            raise ValueError("figure cell labels are invalid")
        if "PUBLIC-REF" in cells:
            raise ValueError("PUBLIC-REF must not be registered as a processing cell")
        if axis is None and index is not None:
            raise ValueError("figure slice index cannot exist without an axis")
        if axis is not None:
            if (
                axis not in {"i3", "i2", "i1"}
                or isinstance(index, bool)
                or not isinstance(index, int)
            ):
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
        elif any(
            record[name] is not None
            for name in (
                "slice_selection_policy",
                "selection_percentile",
                "buffer_radius",
                "selection_threshold",
            )
        ):
            raise ValueError("non-spatial figure has spatial selection metadata")
        scale = record["display_scale"]
        if scale is not None:
            if not isinstance(scale, Mapping):
                raise ValueError("figure display scale must be an object")
            if category == "f3_spatial" and any(
                isinstance(scale.get(name), bool)
                or not isinstance(scale.get(name), (int, float))
                or not math.isfinite(float(scale[name]))
                for name in (
                    "vmin",
                    "vmax",
                    "difference_limit",
                    "difference_vmin",
                    "difference_vmax",
                )
            ):
                raise ValueError("spatial figure display scale is invalid")
            for name in ("vmin", "vmax"):
                if scale.get(name) is not None and (
                    isinstance(scale.get(name), bool)
                    or not isinstance(scale.get(name), (int, float))
                ):
                    raise ValueError("display scale endpoint is invalid")
                if scale.get(name) is not None and not math.isfinite(float(scale[name])):
                    raise ValueError("display scale endpoint is non-finite")
            if scale.get("vmin") is not None and scale.get("vmax") is not None:
                if not float(scale["vmin"]) < float(scale["vmax"]):
                    raise ValueError("display scale endpoints are not ordered")
            if category == "f3_spatial" and (
                scale.get("vmin") is None or scale.get("vmax") is None
            ):
                raise ValueError("spatial figure normal display scale is incomplete")
            if scale.get("difference_vmin") is not None or scale.get("difference_vmax") is not None:
                if scale.get("difference_vmin") is None or scale.get("difference_vmax") is None:
                    raise ValueError("difference scale must provide both endpoints")
                if any(
                    isinstance(scale.get(name), bool)
                    or not isinstance(scale.get(name), (int, float))
                    or not math.isfinite(float(scale[name]))
                    for name in ("difference_vmin", "difference_vmax")
                ):
                    raise ValueError("difference scale endpoints are invalid")
                low = float(scale.get("difference_vmin"))
                high = float(scale.get("difference_vmax"))
                if not low < 0.0 < high or not math.isclose(
                    abs(low), abs(high), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError("difference scale must be finite and zero-centered")
                if category == "f3_spatial":
                    observed = float(scale["difference_limit"])
                    if observed < 0.0 or not math.isclose(
                        abs(low), max(observed, 1.0e-6), rel_tol=1e-12, abs_tol=1e-12
                    ):
                        raise ValueError("difference scale limit is inconsistent")
            if record["figure_id"] == "synthetic_scanner_orientation_error_by_case":
                if scale.get("series_encoding") != {
                    "metric_color": {"strike_median": "#4c78a8", "dip_median": "#f58518"},
                    "scanner_backend_marker": {"RL-SCAN": "o", "Q-SCAN": "s"},
                }:
                    raise ValueError("synthetic orientation series encoding is invalid")
            if record["figure_id"] == "f3_sparse_distance_p95_by_stage":
                if scale.get("series_encoding") != {
                    "cell_condition_color": {
                        "RL-REF": "#4c78a8",
                        "RL-QUAL": "#f58518",
                        "Q-REF": "#54a24b",
                        "Q-QUAL": "#e45756",
                    },
                    "distance_direction_hatch": {
                        "candidate_to_reference_p95": "///",
                        "reference_to_candidate_p95": "\\\\",
                    },
                }:
                    raise ValueError("F3 sparse-distance series encoding is invalid")
        for name in ("selection_percentile", "buffer_radius", "selection_threshold"):
            if record[name] is not None and (
                isinstance(record[name], bool)
                or not isinstance(record[name], (int, float))
                or not math.isfinite(float(record[name]))
            ):
                raise ValueError(f"figure {name} is invalid or non-finite")
        if not record["omitted"] and (
            not isinstance(record["caption"], str) or not record["caption"]
        ):
            raise ValueError("non-omitted figure caption is missing")
    fixed_ids = set(FIXED_SCALAR_FIGURE_IDS)
    if (
        set(record_by_id)
        - fixed_ids
        - {
            record["figure_id"]
            for record in records
            if record["category"] in {"f3_spatial", "f3_ridge_overlay"}
        }
    ):
        raise ValueError("figure manifest contains an unsupported non-slot record")
    if fixed_ids - set(record_by_id):
        raise ValueError("figure manifest is missing a fixed scalar figure record")
    if len(observed_spatial_slots) != len(F3_SPATIAL_FIGURE_SLOTS) or set(
        observed_spatial_slots
    ) != set(F3_SPATIAL_FIGURE_SLOTS):
        raise ValueError("figure manifest spatial slot coverage is incomplete")
    if len(observed_ridge_slots) != len(F3_RIDGE_OVERLAY_SLOTS) or set(observed_ridge_slots) != set(
        F3_RIDGE_OVERLAY_SLOTS
    ):
        raise ValueError("figure manifest ridge-overlay slot coverage is incomplete")
    skin_record = record_by_id["synthetic_skin_buffered_f1_by_case"]
    skinning_enabled = manifest["source_coverage"]["synthetic"]["skinning_enabled"]
    if skinning_enabled:
        if skin_record["omitted"]:
            raise ValueError("enabled synthetic skinning must have a skin figure")
    elif (
        not skin_record["omitted"]
        or skin_record["omission_reason"] != SYNTHETIC_SKIN_FIGURE_OMISSION_REASON
    ):
        raise ValueError("disabled synthetic skinning requires the fixed omitted skin figure")
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
    artifact_version = manifest.get("publication_artifact_schema_version")
    if artifact_version == 1:
        raise ValueError(
            "publication artifact schema version 1 predates the semantic table contract; "
            "regenerate the publication bundle"
        )
    if artifact_version != PUBLICATION_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("publication artifact schema version mismatch")
    if manifest.get("publication_metric_selection_version") != PUBLICATION_METRIC_SELECTION_VERSION:
        raise ValueError("publication metric selection version mismatch")
    manifest_figure_version = manifest.get("publication_figure_contract_version")
    if manifest_figure_version == 1:
        raise ValueError(
            "publication figure contract version 1 predates the semantic figure-data contract; "
            "regenerate the publication bundle"
        )
    if manifest_figure_version != PUBLICATION_FIGURE_CONTRACT_VERSION:
        raise ValueError("publication figure contract version mismatch")
    if manifest.get("publication_table_contract_version") != PUBLICATION_TABLE_CONTRACT_VERSION:
        raise ValueError("publication table contract version mismatch")
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
    synthetic_coverage, f3_coverage = _validate_source_coverage(manifest)
    registry = manifest.get("curated_metric_registry")
    if registry != [entry.as_dict() for entry in PUBLICATION_METRIC_REGISTRY]:
        raise ValueError("publication metric registry does not match the fixed contract")

    typed_rows = _validate_table_contracts(manifest, root)
    metric_rows = typed_rows["publication_metrics.csv"]
    contrast_rows = typed_rows["publication_contrasts.csv"]
    summary_rows = typed_rows["publication_summary.csv"]
    _validate_metric_rows(metric_rows, synthetic_coverage, f3_coverage)
    _validate_contrast_rows(contrast_rows, metric_rows)
    _validate_summary(summary_rows, metric_rows, synthetic_coverage)
    _validate_supporting_tables(typed_rows, manifest)
    _validate_figures(root, manifest, typed_rows["runtime_summary.csv"])
    if not (root / "report.md").read_text(encoding="utf-8").strip():
        raise ValueError("publication report is empty")
    return True


__all__ = ["validate_publication_bundle"]
