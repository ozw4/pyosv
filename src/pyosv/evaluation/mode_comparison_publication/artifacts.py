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
    FIGURE_SELECTION_POLICY,
    PUBLICATION_ARTIFACT_SCHEMA_VERSION,
    PUBLICATION_COMPLETION_SCHEMA_VERSION,
    PUBLICATION_FIGURE_CONTRACT_VERSION,
    PUBLICATION_INTERPRETATION,
    PUBLICATION_METRIC_SELECTION_VERSION,
    ROOT_TABLE_FILES,
)
from .figures import generate_figures
from .models import PublicationReport
from .registry import PUBLICATION_METRIC_REGISTRY
from .summary import TABLE_HEADERS
from .validation import validate_publication_bundle


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("publication JSON values must be finite")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise ValueError(f"publication JSON value is not supported: {type(value).__name__}")


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
    return (json.dumps(_json_safe(value), **options) + "\n").encode("utf-8")


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
            _json_safe(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")
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


def _build_manifest(
    report: PublicationReport,
    *,
    matplotlib: Mapping[str, str],
) -> dict[str, Any]:
    synthetic_manifest = report.synthetic.manifest
    f3_manifest = report.f3.run_manifest
    return {
        "publication_artifact_schema_version": PUBLICATION_ARTIFACT_SCHEMA_VERSION,
        "publication_metric_selection_version": PUBLICATION_METRIC_SELECTION_VERSION,
        "publication_figure_contract_version": PUBLICATION_FIGURE_CONTRACT_VERSION,
        "interpretation": PUBLICATION_INTERPRETATION,
        "synthetic_source": {
            "path": str(report.synthetic.path),
            "identity_digest": report.synthetic.identity_digest,
            "completion_sha256": report.synthetic.completion_sha256,
            "manifest_sha256": report.synthetic.manifest_sha256,
            "artifact_schema_version": synthetic_manifest.get("artifact_schema_version"),
            "metric_schema_version": synthetic_manifest.get("metric_schema_version"),
            "scalar_evidence_contract_version": synthetic_manifest.get(
                "scalar_evidence_contract_version"
            ),
            "runtime_contract_version": synthetic_manifest.get("runtime_contract_version"),
            "source_implementation_identity": synthetic_manifest.get("source_provenance"),
        },
        "f3_source": {
            "path": str(report.f3.path),
            "identity_digest": report.f3.identity_digest,
            "completion_sha256": report.f3.completion_sha256,
            "manifest_sha256": report.f3.manifest_sha256,
            "run_fingerprint": f3_manifest.get("run_fingerprint"),
            "dataset_id": report.f3.dataset_identity.get("dataset_id"),
            "dataset_identity_digest": _canonical_digest(report.f3.dataset_identity),
            "dataset_identity": report.f3.dataset_identity,
            "artifact_schema_version": f3_manifest.get("artifact_schema_version"),
            "stage_contract_version": f3_manifest.get("stage_contract_version"),
            "fingerprint_contract_version": f3_manifest.get("fingerprint_contract_version"),
            "result_schema_version": report.f3.result_schema_version,
            "public_reference_file_mapping": {
                stage: report.f3.dataset_spec.filename_for(role)
                for stage, role in _f3_reference_roles().items()
            },
            "source_implementation_identity": f3_manifest.get("implementation_identity"),
            "source_commit_identity": (
                f3_manifest.get("provenance", {}).get("source")
                if isinstance(f3_manifest.get("provenance"), Mapping)
                else None
            ),
        },
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


def _canonical_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


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
        "--output-dir outputs/3d/mode_comparison_publication/publication_v1"
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
            "Spatial figures use deterministic center, public-reference-peak, and end-to-end-difference-peak slice policies. Normal panels share a validated full-volume scale; signed differences use a separate zero-centered scale. Ridge overlays use the source positive-p99 and radius-2 contract.",
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
        records, matplotlib = generate_figures(report, temporary_path)
        figure_manifest = {
            "publication_figure_contract_version": PUBLICATION_FIGURE_CONTRACT_VERSION,
            "canonical_condition_order": list(CANONICAL_CELL_ORDER),
            "canonical_stage_order": list(CANONICAL_STAGE_ORDER),
            "volume_shape": list(report.f3.result.volume_shape),
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
