"""Atomic publication-bundle writing and Markdown report serialization."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..f3d_mode_comparison.data import ensure_output_not_in_data_root

from .config import (
    CANONICAL_CELL_ORDER,
    CANONICAL_STAGE_ORDER,
    CONTRAST_NAMES,
    F3_RIDGE_OVERLAY_SLOTS,
    F3_SPATIAL_FIGURE_SLOTS,
    FIGURE_SELECTION_POLICY,
    FIXED_SCALAR_FIGURE_IDS,
    PUBLICATION_ARTIFACT_SCHEMA_VERSION,
    PUBLICATION_COMPLETION_SCHEMA_VERSION,
    PUBLICATION_FIGURE_CONTRACT_VERSION,
    PUBLICATION_INTERPRETATION,
    PUBLICATION_METRIC_SELECTION_VERSION,
    PUBLICATION_TABLE_CONTRACT_VERSION,
    ROOT_TABLE_FILES,
)
from .figures import build_f3_ridge_threshold_contract, generate_figures
from .models import PublicationReport
from .registry import PUBLICATION_METRIC_REGISTRY
from .summary import TABLE_HEADERS
from .semantic import (
    F3_SOURCE_IDENTITY_FIELDS,
    ROOT_TABLE_FIELD_TYPES,
    ROOT_TABLE_IDENTITY_FIELDS,
    SYNTHETIC_SOURCE_IDENTITY_FIELDS,
    build_table_contract,
    canonical_digest,
    finite_json_normalize,
    source_identity_object,
)
from .validation import validate_publication_bundle


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
    options: dict[str, Any] = {
        "ensure_ascii": False,
        "allow_nan": False,
        "sort_keys": True,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(finite_json_normalize(value), **options) + "\n").encode("utf-8")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError("publication CSV values must be finite")
        return repr(numeric)
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(
            finite_json_normalize(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    return str(value)


def _write_csv(path: Path, header: tuple[str, ...], rows: tuple[Mapping[str, Any], ...]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(_csv_value(row.get(name)) for name in header)


def _file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"path": path.name, "size": size, "sha256": digest.hexdigest()}


def _relative_files(root: Path) -> tuple[str, ...]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink() or path.name == "completion.json":
            continue
        files.append(path.relative_to(root).as_posix())
    return tuple(files)


def _identity_mapping(
    values: Mapping[str, Any], identity_fields: tuple[str, ...]
) -> dict[str, Any]:
    return {name: finite_json_normalize(values[name]) for name in identity_fields}


def _source_table_identities(report: PublicationReport, filename: str) -> list[dict[str, Any]]:
    """Extract expected identities directly from validated source evidence."""

    fields = ROOT_TABLE_IDENTITY_FIELDS[filename]
    if filename == "publication_metrics.csv":
        output = []
        for entry in PUBLICATION_METRIC_REGISTRY:
            if entry.dataset == "synthetic":
                allowed_cells = (
                    ("RL-SCAN", "Q-SCAN") if entry.stage == "scanner_raw" else CANONICAL_CELL_ORDER
                )
                source_rows = (
                    row
                    for row in report.synthetic.metric_rows
                    if row.cell_label in allowed_cells
                    and (row.stage, row.selection, row.metric)
                    == (entry.stage, entry.selection, entry.metric)
                )
                for row in source_rows:
                    output.append(
                        _identity_mapping(
                            {
                                "dataset": "synthetic",
                                "case_or_region": row.case_id,
                                "trial_id": row.trial_id,
                                "seed": row.seed,
                                "cell_label": row.cell_label,
                                "stage": row.stage,
                                "selection": row.selection,
                                "metric": row.metric,
                            },
                            fields,
                        )
                    )
            else:
                source_rows = (
                    row
                    for row in report.f3.result.metric_rows
                    if row.cell_label in CANONICAL_CELL_ORDER
                    and (row.stage, row.selection, row.metric)
                    == (entry.stage, entry.selection, entry.metric)
                )
                for row in source_rows:
                    output.append(
                        _identity_mapping(
                            {
                                "dataset": "f3",
                                "case_or_region": "full",
                                "trial_id": None,
                                "seed": None,
                                "cell_label": row.cell_label,
                                "stage": row.stage,
                                "selection": row.selection,
                                "metric": row.metric,
                            },
                            fields,
                        )
                    )
        return output

    if filename == "publication_contrasts.csv":
        selected = {
            (entry.dataset, entry.stage, entry.selection, entry.metric)
            for entry in PUBLICATION_METRIC_REGISTRY
        }
        output = []
        for row in report.synthetic.contrast_rows:
            if (
                row.contrast_name not in CONTRAST_NAMES
                or (
                    "synthetic",
                    row.stage,
                    row.selection,
                    row.metric,
                )
                not in selected
            ):
                continue
            output.append(
                _identity_mapping(
                    {
                        "dataset": "synthetic",
                        "case_or_region": row.case_id,
                        "trial_id": row.trial_id,
                        "seed": row.seed,
                        "contrast_name": row.contrast_name,
                        "stage": row.stage,
                        "selection": row.selection,
                        "metric": row.metric,
                    },
                    fields,
                )
            )
        for row in report.f3.result.contrast_rows:
            if (
                row.contrast_name not in CONTRAST_NAMES
                or (
                    "f3",
                    row.stage,
                    row.selection,
                    row.metric,
                )
                not in selected
            ):
                continue
            output.append(
                _identity_mapping(
                    {
                        "dataset": "f3",
                        "case_or_region": "full",
                        "trial_id": None,
                        "seed": None,
                        "contrast_name": row.contrast_name,
                        "stage": row.stage,
                        "selection": row.selection,
                        "metric": row.metric,
                    },
                    fields,
                )
            )
        return output

    if filename == "f3_regional_summary.csv":
        return [
            _identity_mapping(
                {
                    "dataset": "f3",
                    "stage": row.stage,
                    "cell_label": row.cell_label,
                    "region": row.region,
                    "metric": metric,
                },
                fields,
            )
            for row in report.f3.result.regional_rows
            for metric in row.metrics
        ]

    if filename == "f3_orientation_summary.csv":
        output = []
        for row in report.f3.result.orientation_rows:
            for prefix, summary in (
                ("strike_circular_absolute_difference", row.strike_circular_absolute_difference),
                ("dip_absolute_difference", row.dip_absolute_difference),
                ("normal_vector_angular_difference", row.normal_vector_angular_difference),
            ):
                for statistic in summary:
                    output.append(
                        _identity_mapping(
                            {
                                "dataset": "f3",
                                "stage": row.stage,
                                "left_cell": row.left_cell,
                                "right_cell": row.right_cell,
                                "support_contract": row.support_contract,
                                "metric": f"{prefix}.{statistic}",
                            },
                            fields,
                        )
                    )
        return output

    if filename == "runtime_summary.csv":
        output = []
        for row in report.synthetic.runtime_rows:
            output.append(
                _identity_mapping(
                    {
                        "dataset": "synthetic",
                        "case_or_region": row.case_id or "experiment",
                        "trial_id": row.trial_id,
                        "seed": row.seed,
                        "stage": row.stage,
                        "fingerprint": None,
                        "scanner_backend": row.scanner_backend,
                        "call_count": row.call_count,
                        "cell_label": row.cell_label,
                        "cell_consumers": (),
                        "state": "shared" if row.shared_stage else "cell-owned",
                        "attribution": ("shared-stage" if row.shared_stage else "cell-owned-stage"),
                    },
                    fields,
                )
            )
        for row in report.f3.result.runtime_rows:
            consumers = tuple(row.cell_consumers)
            output.append(
                _identity_mapping(
                    {
                        "dataset": "f3",
                        "case_or_region": "full",
                        "trial_id": None,
                        "seed": None,
                        "stage": row.stage_kind,
                        "fingerprint": row.fingerprint,
                        "scanner_backend": None,
                        "call_count": 1,
                        "cell_label": row.cell,
                        "cell_consumers": consumers,
                        "state": row.state,
                        "attribution": (
                            "shared-stage" if len(consumers) > 1 else "cell-owned-stage"
                        ),
                    },
                    fields,
                )
            )
        return output
    return []


def _table_contracts(report: PublicationReport) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for filename in ROOT_TABLE_FILES:
        contract = build_table_contract(
            TABLE_HEADERS[filename],
            tuple(report.tables[filename]),
            ROOT_TABLE_IDENTITY_FIELDS[filename],
            ROOT_TABLE_FIELD_TYPES[filename],
        )
        expected_identities = _source_table_identities(report, filename)
        if filename != "publication_summary.csv":
            contract["source_expected_identities"] = expected_identities
            contract["source_expected_identity_sha256"] = canonical_digest(expected_identities)
        contracts[filename] = contract
    return contracts


def _synthetic_source_coverage(report: PublicationReport) -> dict[str, Any]:
    trial_records = report.synthetic.manifest.get("trials")
    if not isinstance(trial_records, list):
        raise ValueError("validated synthetic source manifest has no trial metadata")
    trials = []
    for record in trial_records:
        if not isinstance(record, Mapping):
            raise ValueError("validated synthetic source trial metadata is invalid")
        case_id = record.get("case_id")
        trial_id = record.get("trial_id")
        seed = record.get("case_generation_seed")
        if (
            not isinstance(case_id, str)
            or not case_id
            or not isinstance(trial_id, str)
            or not trial_id
            or (seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)))
        ):
            raise ValueError("validated synthetic source trial identity is invalid")
        trials.append({"case_id": case_id, "trial_id": trial_id, "seed": seed})
    return {
        "case_order": list(report.synthetic.case_order),
        "trials": trials,
        "skinning_enabled": report.synthetic.skinning_enabled,
        "expected_scanner_only_cells": ["RL-SCAN", "Q-SCAN"],
        "expected_end_to_end_cells": list(CANONICAL_CELL_ORDER),
    }


def _f3_source_coverage(report: PublicationReport) -> dict[str, Any]:
    skinning_by_cell = {cell.label: cell.skinning_enabled for cell in report.f3.result.cells}
    if tuple(skinning_by_cell) != CANONICAL_CELL_ORDER:
        raise ValueError("validated F3 source cell order is not canonical")
    return {
        "evaluation_unit_count": 1,
        "canonical_cell_order": list(CANONICAL_CELL_ORDER),
        "skinning_enabled_by_cell": skinning_by_cell,
        "volume_shape": list(report.f3.result.volume_shape),
        "dataset_id": report.f3.dataset_identity.get("dataset_id"),
        "run_fingerprint": report.f3.result.run_fingerprint,
    }


def _source_coverage(report: PublicationReport) -> dict[str, Any]:
    return {
        "synthetic": _synthetic_source_coverage(report),
        "f3": _f3_source_coverage(report),
    }


def _synthetic_source_record(report: PublicationReport) -> dict[str, Any]:
    source_manifest = report.synthetic.manifest
    record: dict[str, Any] = {
        "path": str(report.synthetic.path),
        "completion_sha256": report.synthetic.completion_sha256,
        "manifest_sha256": report.synthetic.manifest_sha256,
        "artifact_schema_version": source_manifest.get("artifact_schema_version"),
        "metric_schema_version": source_manifest.get("metric_schema_version"),
        "scalar_evidence_contract_version": source_manifest.get("scalar_evidence_contract_version"),
        "runtime_contract_version": source_manifest.get("runtime_contract_version"),
        "source_implementation_identity": source_manifest.get("source_provenance"),
    }
    record["identity_digest"] = canonical_digest(
        source_identity_object(record, SYNTHETIC_SOURCE_IDENTITY_FIELDS)
    )
    if record["identity_digest"] != report.synthetic.identity_digest:
        raise ValueError("synthetic source identity does not match the validated loader")
    return record


def _f3_source_record(report: PublicationReport) -> dict[str, Any]:
    source_manifest = report.f3.run_manifest
    record: dict[str, Any] = {
        "path": str(report.f3.path),
        "completion_sha256": report.f3.completion_sha256,
        "manifest_sha256": report.f3.manifest_sha256,
        "run_fingerprint": source_manifest.get("run_fingerprint"),
        "dataset_id": report.f3.dataset_identity.get("dataset_id"),
        "dataset_identity_digest": canonical_digest(report.f3.dataset_identity),
        "dataset_identity": report.f3.dataset_identity,
        "artifact_schema_version": source_manifest.get("artifact_schema_version"),
        "stage_contract_version": source_manifest.get("stage_contract_version"),
        "fingerprint_contract_version": source_manifest.get("fingerprint_contract_version"),
        "result_schema_version": report.f3.result_schema_version,
        "public_reference_file_mapping": {
            stage: report.f3.dataset_spec.filename_for(role)
            for stage, role in _f3_reference_roles().items()
        },
        "source_implementation_identity": source_manifest.get("implementation_identity"),
        "source_commit_identity": (
            source_manifest.get("provenance", {}).get("source")
            if isinstance(source_manifest.get("provenance"), Mapping)
            else None
        ),
    }
    record["identity_digest"] = canonical_digest(
        source_identity_object(record, F3_SOURCE_IDENTITY_FIELDS)
    )
    if record["identity_digest"] != report.f3.identity_digest:
        raise ValueError("F3 source identity does not match the validated loader")
    return record


def _build_manifest(
    report: PublicationReport,
    *,
    matplotlib: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "publication_artifact_schema_version": PUBLICATION_ARTIFACT_SCHEMA_VERSION,
        "publication_metric_selection_version": PUBLICATION_METRIC_SELECTION_VERSION,
        "publication_figure_contract_version": PUBLICATION_FIGURE_CONTRACT_VERSION,
        "publication_table_contract_version": PUBLICATION_TABLE_CONTRACT_VERSION,
        "interpretation": PUBLICATION_INTERPRETATION,
        "synthetic_source": _synthetic_source_record(report),
        "f3_source": _f3_source_record(report),
        "source_coverage": _source_coverage(report),
        "table_contracts": _table_contracts(report),
        "canonical_condition_order": list(CANONICAL_CELL_ORDER),
        "canonical_stage_order": list(CANONICAL_STAGE_ORDER),
        "curated_metric_registry": [entry.as_dict() for entry in PUBLICATION_METRIC_REGISTRY],
        "figure_selection_policy": dict(FIGURE_SELECTION_POLICY),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib["matplotlib"],
        },
        "matplotlib": dict(matplotlib),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_paths_are_provenance_only": True,
    }


def _f3_reference_roles() -> dict[str, str]:
    return {
        "ft": "reference_fault_likelihood",
        "fv": "reference_fault_votes",
        "fvt": "reference_thinned_fault_votes",
    }


def _markdown_table(
    rows: tuple[Mapping[str, Any], ...], columns: tuple[str, ...], limit: int = 80
) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:limit]:
        values = []
        for column in columns:
            value = row.get(column)
            values.append("" if value is None else str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > limit:
        lines.append(f"| … | showing first {limit} of {len(rows)} rows |")
    return "\n".join(lines)


def _build_report_markdown(
    report: PublicationReport, records: tuple[Mapping[str, Any], ...]
) -> str:
    figure_lines = []
    for record in records:
        if record["omitted"]:
            figure_lines.append(f"- `{record['figure_id']}` omitted: {record['omission_reason']}")
        else:
            figure_lines.append(
                f"- [{record['figure_id']}]({record['relative_path']}) — {record['caption']}"
            )
    metrics = report.tables["publication_metrics.csv"]
    contrasts = report.tables["publication_contrasts.csv"]
    runtime = report.tables["runtime_summary.csv"]
    source_command = (
        "PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication "
        f"--synthetic-bundle {report.synthetic.path} "
        f"--f3-bundle {report.f3.path} "
        f"--f3-data-root {report.f3.data_root} "
        "--output-dir outputs/3d/mode_comparison_publication/publication_v2"
    )
    return "\n".join(
        (
            "# Mode Comparison Publication Report",
            "",
            "## Source bundles and provenance",
            "",
            "This document is a derived report generated from completed, validated source bundles. "
            "It does not run a scanner, voting, thinning, or skinning experiment.",
            "",
            f"- Synthetic source identity: `{report.synthetic.identity_digest}`; completion SHA-256: `{report.synthetic.completion_sha256}`.",
            f"- F3 source identity: `{report.f3.identity_digest}`; run fingerprint: `{report.f3.result.run_fingerprint}`.",
            f"- F3 dataset: `{report.f3.dataset_identity.get('dataset_id')}`, shape `{report.f3.result.volume_shape}`, storage dtype `{report.f3.result.storage_dtype}`.",
            "- The external F3 data root is checksummed against the source run manifest; public DAT files are read-only inputs and are not copied into this bundle.",
            "- Source identity digests are internal provenance-consistency checks, not cryptographic signatures or a tamper-proof commitment against a coherent rewrite.",
            "",
            "## Comparison conditions",
            "",
            "The fixed condition order is `RL-REF`, `RL-QUAL`, `Q-REF`, `Q-QUAL`. "
            "Scanner backend and workflow are independent axes. `PUBLIC-REF` is an F3 comparison target, not a processing cell.",
            "",
            "## Synthetic known-truth metrics",
            "",
            "Synthetic metrics are evaluated against generated known truth. Deterministic cases remain single observations; stochastic seeds are case-generation trials.",
            "",
            _markdown_table(
                tuple(row for row in metrics if row["dataset"] == "synthetic"),
                ("case_or_region", "cell_label", "stage", "metric", "value", "unit", "direction"),
            ),
            "",
            "## Synthetic scanner/workflow contrasts",
            "",
            "Contrasts are paired linear contrasts within the same case/trial, not significance tests. Interaction is a factorial contrast, not an additional quality score.",
            "",
            _markdown_table(
                tuple(row for row in contrasts if row["dataset"] == "synthetic"),
                (
                    "case_or_region",
                    "contrast_name",
                    "stage",
                    "metric",
                    "raw_value",
                    "improvement_value",
                ),
            ),
            "",
            "## F3 public-reference agreement",
            "",
            "F3 values describe agreement with the public reference outputs. They are not accuracy, correctness, or ground-truth F3 scores. The F3 comparison is one full-volume evaluation unit, not a collection of crop replicates.",
            "",
            _markdown_table(
                tuple(row for row in metrics if row["dataset"] == "f3"),
                ("stage", "cell_label", "metric", "value", "unit", "direction"),
            ),
            "",
            "## F3 full-volume spatial comparisons",
            "",
            "Spatial figures use deterministic center, public-reference-peak, and end-to-end-difference-peak slice policies. Normal panels share a validated full-volume scale; signed differences use a separate zero-centered scale. Ridge overlays use source positive-p99/radius-2 thresholds: public reference uses the stage reference threshold and each candidate uses its own cell threshold.",
            "",
            "## Runtime and resource interpretation",
            "",
            "Runtime is within-experiment attribution. Shared scanner, voting, and thinning stages are not apportioned to cells and are not isolated-process benchmarks.",
            "",
            _markdown_table(
                runtime,
                ("dataset", "stage", "cell_label", "state", "elapsed_seconds", "attribution"),
            ),
            "",
            "## Limitations",
            "",
            "No significance tests, automatic winner/default selection, promotion gate, or cross-domain aggregate score is produced. Synthetic spatial volume replay is outside the scalar-only source contract. Agreement with the public F3 reference is not a geological improvement claim.",
            "",
            "## Reproduction command",
            "",
            "```bash",
            source_command,
            "```",
            "",
            "## Artifact index",
            "",
            "The machine-readable tables are `publication_metrics.csv`, `publication_contrasts.csv`, `publication_summary.csv`, `f3_regional_summary.csv`, `f3_orientation_summary.csv`, and `runtime_summary.csv`. Figure provenance and scale metadata are in `figure_manifest.json`; each non-omitted PNG has a matching CSV under `figure_data/`.",
            "",
            *figure_lines,
            "",
        )
    )


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        if stream.write(payload) != len(payload):
            raise OSError(f"short publication write for {path}")
        stream.flush()
        os.fsync(stream.fileno())


def _rename_new(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"publication output already exists: {destination}")
    os.rename(source, destination)


def write_publication_bundle(
    report: PublicationReport,
    output_dir: str | os.PathLike[str],
    *,
    pretty: bool = False,
) -> Path:
    """Write one complete publication bundle using a private temporary directory."""

    if not isinstance(report, PublicationReport):
        raise TypeError("report must be a PublicationReport")
    if not isinstance(pretty, bool):
        raise TypeError("pretty must be bool")
    final_path = Path(output_dir)
    if os.path.lexists(final_path):
        raise FileExistsError(f"publication output already exists: {final_path}")
    resolved_final_path = final_path.resolve(strict=False)
    for source_path in (report.synthetic.path, report.f3.path):
        if resolved_final_path == source_path or resolved_final_path.is_relative_to(source_path):
            raise ValueError("publication output must not be inside a source bundle")
    ensure_output_not_in_data_root(resolved_final_path, report.f3.data_root)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{final_path.name}.tmp-", dir=final_path.parent)
    )
    try:
        for filename in ROOT_TABLE_FILES:
            _write_csv(
                temporary_path / filename,
                TABLE_HEADERS[filename],
                tuple(report.tables[filename]),
            )
        f3_ridge_threshold_contract = build_f3_ridge_threshold_contract(report.f3).as_dict()
        records, matplotlib = generate_figures(report, temporary_path)
        figure_manifest = {
            "publication_figure_contract_version": PUBLICATION_FIGURE_CONTRACT_VERSION,
            "canonical_condition_order": list(CANONICAL_CELL_ORDER),
            "canonical_stage_order": list(CANONICAL_STAGE_ORDER),
            "volume_shape": list(report.f3.result.volume_shape),
            "fixed_scalar_figure_ids": list(FIXED_SCALAR_FIGURE_IDS),
            "f3_spatial_figure_slots": [list(slot) for slot in F3_SPATIAL_FIGURE_SLOTS],
            "f3_ridge_overlay_slots": [list(slot) for slot in F3_RIDGE_OVERLAY_SLOTS],
            "f3_ridge_threshold_contract": f3_ridge_threshold_contract,
            "figures": list(records),
        }
        manifest = _build_manifest(report, matplotlib=matplotlib)
        _write_bytes(temporary_path / "manifest.json", _json_bytes(manifest, pretty=pretty))
        _write_bytes(
            temporary_path / "figure_manifest.json",
            _json_bytes(figure_manifest, pretty=pretty),
        )
        _write_bytes(
            temporary_path / "report.md",
            _build_report_markdown(report, records).encode("utf-8"),
        )

        required_files = _relative_files(temporary_path)
        metadata = []
        for relative in required_files:
            item = _file_metadata(temporary_path / relative)
            item["path"] = relative
            metadata.append(item)
        completion = {
            "completion_schema_version": PUBLICATION_COMPLETION_SCHEMA_VERSION,
            "status": "complete",
            "required_files": list(required_files),
            "files": metadata,
        }
        # Completion is deliberately the last published file in the temporary tree.
        _write_bytes(temporary_path / "completion.json", _json_bytes(completion, pretty=pretty))
        validate_publication_bundle(temporary_path)
        _rename_new(temporary_path, final_path)
    except BaseException:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)
        raise
    return final_path


__all__ = ["write_publication_bundle"]
