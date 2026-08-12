"""Generate publication-manifest-v1 bundles from validated source bundles."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, cast

from pyosv.evaluation.f3d_mode_comparison.data import ensure_output_not_in_data_root
from pyosv.evaluation.publication_experiment import publication_experiment_bytes
from pyosv.evaluation.publication_manifest import build_publication_manifest
from pyosv.evaluation.publication_manifest_io import (
    artifact_file_record,
    validate_publication_directory,
    write_publication_manifest,
)

from . import build_publication_report
from .config import ROOT_TABLE_FILES
from .figures import generate_figures
from .models import PublicationReport
from .summary import TABLE_HEADERS
from .v1_adapter import adapt_publication_sources

_MANIFEST_NAME = "publication_manifest.json"
_EXPERIMENT_NAME = "experiment.json"
_REPORT_NAME = "report.md"
_COPY_CHUNK_SIZE = 1024 * 1024
_TABLE_ROLES = {
    "publication_metrics.csv": "metric_table",
    "publication_contrasts.csv": "contrast_table",
    "publication_summary.csv": "summary_table",
    "f3_regional_summary.csv": "diagnostic_table",
    "f3_orientation_summary.csv": "diagnostic_table",
    "runtime_summary.csv": "runtime_table",
}
_RESERVED_ROOT_NAMES = {
    _MANIFEST_NAME,
    _EXPERIMENT_NAME,
    _REPORT_NAME,
    "figure_data",
    "figures",
    *ROOT_TABLE_FILES,
}
_SEMANTICS = {
    "synthetic": "known_truth",
    "f3": "public_reference_agreement",
    "f3_public_reference_is_geological_truth": False,
    "f3_evaluation_units": 1,
}

__all__ = ["generate_publication_bundle_v1"]


def generate_publication_bundle_v1(
    synthetic_bundle: str | PathLike[str],
    f3_bundle: str | PathLike[str],
    f3_data_root: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    environment_lock: str | PathLike[str],
    code: Mapping[str, object],
    environment_controls: Mapping[str, str],
    pretty: bool = False,
) -> Path:
    """Build and atomically publish a self-validating v1 publication bundle."""
    if type(pretty) is not bool:
        raise ValueError("pretty must be a bool")
    final_path = Path(output_dir)
    if os.path.lexists(final_path):
        raise FileExistsError(f"publication output already exists: {final_path}")

    lock_source, lock_name = _environment_lock_source(environment_lock)
    report = build_publication_report(synthetic_bundle, f3_bundle, f3_data_root)
    _assert_output_is_derived_only(report, final_path)

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{final_path.name}.tmp-", dir=final_path.parent)
    )
    try:
        artifacts: list[dict[str, object]] = []

        _copy_regular_file(lock_source, temporary_path / lock_name)
        lock_record = artifact_file_record(
            temporary_path,
            lock_name,
            tier="primary",
            role="environment_lock",
        )
        artifacts.append(lock_record)

        adapted = adapt_publication_sources(report)
        experiment = cast(Mapping[str, object], adapted["experiment"])
        _write_bytes(
            temporary_path / _EXPERIMENT_NAME,
            publication_experiment_bytes(experiment, pretty=pretty),
        )
        experiment_record = artifact_file_record(
            temporary_path,
            _EXPERIMENT_NAME,
            tier="primary",
            role="resolved_experiment",
        )
        artifacts.append(experiment_record)

        for filename in ROOT_TABLE_FILES:
            _write_csv(
                temporary_path / filename,
                TABLE_HEADERS[filename],
                tuple(report.tables[filename]),
            )
            artifacts.append(
                artifact_file_record(
                    temporary_path,
                    filename,
                    tier="primary",
                    role=_TABLE_ROLES[filename],
                )
            )

        figure_records = generate_figures(report, temporary_path)
        artifacts.extend(
            _records_below(
                temporary_path,
                "figure_data",
                suffix=".csv",
                tier="primary",
                role="figure_data",
            )
        )
        artifacts.extend(
            _records_below(
                temporary_path,
                "figures",
                suffix=".png",
                tier="derived",
                role="figure",
            )
        )

        _write_bytes(
            temporary_path / _REPORT_NAME,
            _render_report(report, experiment, figure_records).encode("utf-8"),
        )
        artifacts.append(
            artifact_file_record(
                temporary_path,
                _REPORT_NAME,
                tier="derived",
                role="report",
            )
        )

        manifest = build_publication_manifest(
            created_at_utc=_created_at_utc(),
            code=code,
            environment={
                "python": platform.python_version(),
                "lock_file": lock_record["path"],
                "lock_sha256": lock_record["sha256"],
                "controls": environment_controls,
            },
            datasets=cast(Mapping[str, object], adapted["datasets"]),
            experiment={
                "config_file": experiment_record["path"],
                "config_sha256": experiment_record["sha256"],
                "source_runs": adapted["source_runs"],
            },
            semantics=_SEMANTICS,
            artifacts=artifacts,
        )
        write_publication_manifest(temporary_path, manifest, pretty=pretty)
        validate_publication_directory(temporary_path)

        if os.path.lexists(final_path):
            raise FileExistsError(f"publication output already exists: {final_path}")
        os.rename(temporary_path, final_path)
    except BaseException:
        if os.path.lexists(temporary_path):
            shutil.rmtree(temporary_path)
        raise
    return final_path


def _environment_lock_source(value: str | PathLike[str]) -> tuple[Path, str]:
    path = Path(value)
    if path.is_symlink() or not path.is_file():
        raise ValueError("environment_lock must be an existing regular non-symlink file")
    name = path.name
    if not name or name in {".", ".."} or "\\" in name or name in _RESERVED_ROOT_NAMES:
        raise ValueError("environment lock basename conflicts with the publication layout")
    return path, name


def _assert_output_is_derived_only(report: PublicationReport, output_dir: Path) -> None:
    resolved = output_dir.resolve(strict=False)
    for source_path in (report.synthetic.path, report.f3.path):
        resolved_source = Path(source_path).resolve(strict=False)
        if resolved == resolved_source or resolved.is_relative_to(resolved_source):
            raise ValueError("publication output must not be inside a source bundle")
    ensure_output_not_in_data_root(resolved, report.f3.data_root)


def _copy_regular_file(source: Path, destination: Path) -> None:
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        while chunk := input_stream.read(_COPY_CHUNK_SIZE):
            output_stream.write(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_csv(
    path: Path,
    header: tuple[str, ...],
    rows: tuple[Mapping[str, Any], ...],
) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow(_csv_value(row.get(field)) for field in header)
        stream.flush()
        os.fsync(stream.fileno())


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("publication CSV values must be finite")
        return repr(value)
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if scalar is not value:
            return _csv_value(scalar)
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    return str(value)


def _records_below(
    root: Path,
    directory: str,
    *,
    suffix: str,
    tier: str,
    role: str,
) -> list[dict[str, object]]:
    base = root / directory
    paths = sorted(base.glob(f"*{suffix}")) if base.is_dir() else []
    if not paths:
        raise ValueError(f"figure generator produced no {directory}/*{suffix} files")
    return [
        artifact_file_record(
            root,
            path.relative_to(root).as_posix(),
            tier=tier,
            role=role,
        )
        for path in paths
    ]


def _render_report(
    report: PublicationReport,
    experiment: Mapping[str, object],
    figure_records: tuple[Mapping[str, Any], ...],
) -> str:
    f3 = cast(Mapping[str, object], experiment["f3"])
    condition_order = cast(list[str], experiment["condition_order"])
    figures = []
    for record in figure_records:
        figure_id = record.get("figure_id")
        if record.get("omitted"):
            figures.append(
                f"- `{figure_id}` omitted: {record.get('omission_reason', 'not available')}"
            )
        else:
            figures.append(
                f"- [{figure_id}]({record.get('relative_path')}) — {record.get('caption', '')}"
            )
    return "\n".join(
        (
            "# Mode Comparison Publication Report",
            "",
            "## Sources and experiment",
            "",
            "This report is derived from completed, validated source bundles. It does not run scanner, voting, thinning, or skinning stages.",
            f"- Synthetic completion SHA-256: `{report.synthetic.completion_sha256}`.",
            f"- F3 completion SHA-256: `{report.f3.completion_sha256}`.",
            f"- F3 dataset: `{f3['dataset_id']}`; shape `{tuple(cast(list[int], f3['shape']))}`; storage dtype `{f3['storage_dtype']}`.",
            f"- Condition order: `{', '.join(condition_order)}`.",
            "",
            "## Evaluation semantics",
            "",
            "Synthetic results are evaluated against generated known truth. F3 results describe public-reference agreement and must not be interpreted as geological truth.",
            "",
            "## Limitations",
            "",
            "No significance test, automatic winner, promotion decision, or cross-domain aggregate score is produced. Runtime rows are attribution diagnostics, not isolated-process benchmarks.",
            "",
            "## Artifacts and figures",
            "",
            "Machine-readable root CSV tables contain the selected metrics, contrasts, summaries, diagnostics, and runtime attribution. Figure source rows are under `figure_data/`.",
            "",
            *figures,
            "",
        )
    )


def _created_at_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
