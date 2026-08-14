"""Atomic generation of F3-only compact publication bundles."""

from __future__ import annotations

import os
import platform
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, cast

from pyosv.evaluation.publication_manifest_io import artifact_file_record

from .manifest import (
    build_manifest,
    validate_publication_directory,
    write_manifest,
)

_EXPERIMENT_NAME = "experiment.json"
_REPORT_NAME = "report.md"
_SUMMARY_NAME = "f3_q_qual_vs_public_ref_summary.csv"
_COPY_CHUNK_SIZE = 1024 * 1024
_STAGE_ORDER = ("ft", "fv", "fvt")
_RESERVED_ROOT_NAMES = {
    "publication_manifest.json",
    _EXPERIMENT_NAME,
    _REPORT_NAME,
    _SUMMARY_NAME,
    "figure_data",
    "figures",
}
_SEMANTICS = {
    "evaluation": "f3_public_reference_agreement",
    "public_reference_is_geological_truth": False,
    "evaluation_units": 1,
    "displayed_condition": "Q-QUAL",
    "stage_order": list(_STAGE_ORDER),
}

__all__ = [
    "generate_f3_compact_publication_bundle",
    "validate_f3_compact_publication_bundle",
]


def generate_f3_compact_publication_bundle(
    f3_bundle: str | PathLike[str],
    f3_data_root: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    environment_lock: str | PathLike[str],
    code: Mapping[str, object],
    environment_controls: Mapping[str, str],
    pretty: bool = False,
) -> Path:
    """Build and atomically publish one self-validating compact F3 bundle."""

    bundle_path = _path_argument(f3_bundle, "f3_bundle")
    data_path = _path_argument(f3_data_root, "f3_data_root")
    final_path = _path_argument(output_dir, "output_dir")
    lock_source, lock_name = _environment_lock_source(environment_lock)
    code_value = _mapping_argument(code, "code")
    controls = _controls_argument(environment_controls)
    if type(pretty) is not bool:
        raise ValueError("pretty must be a bool")
    if os.path.lexists(final_path):
        raise FileExistsError(f"compact publication output already exists: {final_path}")
    _require_derived_output(final_path, bundle_path, data_path)

    from .figures import generate_figures
    from .source import load_compact_source
    from .summary import (
        build_experiment,
        build_summary_rows,
        experiment_json_bytes,
        summary_csv_bytes,
    )

    context = load_compact_source(bundle_path, data_path)
    summary_rows = build_summary_rows(context)
    summary_payload = summary_csv_bytes(summary_rows)
    experiment = build_experiment(context)
    experiment_payload = experiment_json_bytes(experiment, pretty=False)

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

        _write_bytes(temporary_path / _EXPERIMENT_NAME, experiment_payload)
        experiment_record = artifact_file_record(
            temporary_path,
            _EXPERIMENT_NAME,
            tier="primary",
            role="resolved_experiment",
        )
        artifacts.append(experiment_record)

        _write_bytes(temporary_path / _SUMMARY_NAME, summary_payload)
        artifacts.append(
            artifact_file_record(
                temporary_path,
                _SUMMARY_NAME,
                tier="primary",
                role="summary_table",
            )
        )

        figure_records = _validate_figure_records(
            generate_figures(context, temporary_path),
            context.selected_slice.axis,
            context.selected_slice.index,
        )
        for record in figure_records:
            artifacts.append(
                artifact_file_record(
                    temporary_path,
                    cast(str, record["figure_data_csv"]),
                    tier="primary",
                    role="figure_data",
                )
            )
            artifacts.append(
                artifact_file_record(
                    temporary_path,
                    cast(str, record["relative_path"]),
                    tier="derived",
                    role="figure",
                )
            )

        report = _render_report(context, summary_rows, figure_records)
        _write_bytes(temporary_path / _REPORT_NAME, report.encode("utf-8"))
        artifacts.append(
            artifact_file_record(
                temporary_path,
                _REPORT_NAME,
                tier="derived",
                role="report",
            )
        )

        manifest = build_manifest(
            created_at_utc=_created_at_utc(),
            code=code_value,
            environment={
                "python": platform.python_version(),
                "lock_file": lock_record["path"],
                "lock_sha256": lock_record["sha256"],
                "controls": controls,
            },
            source={"f3_completion_sha256": context.f3.completion_sha256},
            dataset=_manifest_dataset(experiment),
            experiment={
                "config_file": experiment_record["path"],
                "config_sha256": experiment_record["sha256"],
            },
            semantics=_SEMANTICS,
            artifacts=artifacts,
        )
        write_manifest(temporary_path, manifest, pretty=pretty)
        validate_publication_directory(temporary_path)

        if os.path.lexists(final_path):
            raise FileExistsError(f"compact publication output already exists: {final_path}")
        os.rename(temporary_path, final_path)
    except BaseException:
        if os.path.lexists(temporary_path):
            shutil.rmtree(temporary_path)
        raise
    return final_path


def validate_f3_compact_publication_bundle(
    output_dir: str | PathLike[str],
) -> Mapping[str, object]:
    """Validate a compact bundle without consulting its scientific sources."""

    return validate_publication_directory(_path_argument(output_dir, "output_dir"))


def _path_argument(value: object, name: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{name} must be a path-like value")
    try:
        return Path(value)
    except (TypeError, ValueError, OSError) as error:
        raise ValueError(f"{name} must be a valid path-like value") from error


def _mapping_argument(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return dict(value)


def _controls_argument(value: object) -> dict[str, str]:
    controls = _mapping_argument(value, "environment_controls")
    if any(type(item) is not str for item in controls.values()):
        raise ValueError("environment_controls values must be strings")
    return cast(dict[str, str], controls)


def _environment_lock_source(value: object) -> tuple[Path, str]:
    path = _path_argument(value, "environment_lock")
    if path.is_symlink() or not path.is_file():
        raise ValueError("environment_lock must be an existing regular non-symlink file")
    name = path.name
    if not name or name in {".", ".."} or "\\" in name or name in _RESERVED_ROOT_NAMES:
        raise ValueError("environment lock basename conflicts with the compact bundle layout")
    return path, name


def _require_derived_output(output: Path, bundle: Path, data_root: Path) -> None:
    resolved_output = output.resolve(strict=False)
    for source, label in ((bundle, "F3 source bundle"), (data_root, "F3 data root")):
        resolved_source = source.resolve(strict=False)
        if resolved_output == resolved_source or resolved_output.is_relative_to(resolved_source):
            raise ValueError(f"compact publication output must not be inside the {label}")


def _copy_regular_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError("environment_lock must remain a regular non-symlink file")
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


def _validate_figure_records(
    records: Sequence[Mapping[str, object]],
    axis: str,
    index: int,
) -> tuple[Mapping[str, object], ...]:
    values = tuple(records)
    if any(not isinstance(record, Mapping) for record in values):
        raise ValueError("figure records must be mappings")
    if tuple(record.get("stage") for record in values) != _STAGE_ORDER:
        raise ValueError("figure records must follow the fixed ft, fv, fvt stage order")
    normalized: list[Mapping[str, object]] = []
    for stage, record in zip(_STAGE_ORDER, values, strict=True):
        figure_id = f"f3_{stage}_public_ref_vs_q_qual_{axis}_{index}"
        expected = {
            "figure_id": figure_id,
            "relative_path": f"figures/{figure_id}.png",
            "figure_data_csv": f"figure_data/{figure_id}.csv",
            "stage": stage,
        }
        if not isinstance(record, Mapping) or any(
            record.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(f"figure record does not match the fixed {stage} artifact contract")
        caption = record.get("caption")
        if type(caption) is not str or not caption:
            raise ValueError(f"figure record {stage!r} must have a caption")
        normalized.append(record)
    return tuple(normalized)


def _manifest_dataset(experiment: Mapping[str, object]) -> Mapping[str, object]:
    dataset = experiment.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("compact experiment dataset must be an object")
    files = dataset.get("files")
    if not isinstance(files, list):
        raise ValueError("compact experiment dataset files must be an array")
    return {
        "dataset_id": dataset.get("dataset_id"),
        "shape": dataset.get("shape"),
        "storage_dtype": dataset.get("storage_dtype"),
        "files": [
            {
                "role": item.get("role"),
                "filename": item.get("filename"),
                "size": item.get("size"),
                "sha256": item.get("sha256"),
            }
            for item in files
            if isinstance(item, Mapping)
        ],
    }


def _render_report(
    context: Any,
    summary_rows: Sequence[Mapping[str, object]],
    figure_records: Sequence[Mapping[str, object]],
) -> str:
    shape = tuple(context.f3.dataset_spec.shape)
    lines = [
        "# F3 PUBLIC-REF vs Q-QUAL Compact Publication",
        "",
        "## Source and experiment",
        "",
        "This publication is derived from a completed, validated F3 source bundle. It does not rerun scanner, voting, thinning, or skinning.",
        f"- Validated F3 completion SHA-256: `{context.f3.completion_sha256}`.",
        f"- Dataset: `{context.f3.dataset_spec.dataset_id}`; shape `{shape}`; storage dtype `{context.f3.dataset_spec.storage_dtype}`.",
        f"- Amplitude input: `xs.dat`; SHA-256 `{context.amplitude.sha256}`.",
        "- Displayed labels: `PUBLIC-REF`, `Q-QUAL`.",
        "",
        "## Stage comparison",
        "",
        "- `ft`: `fl.dat` compared with the quality scanner output in the Q-QUAL lineage.",
        "- `fv`: `fv.dat` compared with the quality scanner voting output in the Q-QUAL lineage.",
        "- `fvt`: `fvt.dat` compared with the Q-QUAL thinned voting output.",
        "",
        "The `ft` and `fv` candidates belong to the Q-QUAL lineage, but quality-workflow-specific processing has not acted at those stages; their differences must not be attributed to that workflow effect.",
        "",
        "## Selected slice",
        "",
        f"All stages use `{context.selected_slice.axis}={context.selected_slice.index}`, selected by `{context.selected_slice.policy}` with ridge-count score `{context.selected_slice.ridge_count_score}`.",
        "",
        "## Metrics",
        "",
        _summary_markdown(summary_rows),
        "",
        "## Figures",
        "",
    ]
    for record in figure_records:
        lines.append(f"- [{record['figure_id']}]({record['relative_path']}) — {record['caption']}")
    lines.extend(
        (
            "",
            "## Interpretation limits",
            "",
            "The F3 public reference is an agreement reference, not geological truth. These comparisons do not establish statistical significance, a preferred condition, or geological accuracy.",
            "",
        )
    )
    return "\n".join(lines)


def _summary_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    header = (
        "stage",
        "normalized correlation",
        "mean absolute difference",
        "nonzero fraction ratio",
        "buffered F1",
        "candidate→reference p95 voxel",
        "reference→candidate p95 voxel",
    )
    fields = (
        "stage",
        "normalized_correlation",
        "mean_absolute_difference",
        "nonzero_fraction_ratio",
        "buffered_f1",
        "candidate_to_reference_p95_voxel",
        "reference_to_candidate_p95_voxel",
    )
    table = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        table.append("| " + " | ".join(_markdown_cell(row.get(field)) for field in fields) + " |")
    return "\n".join(table)


def _markdown_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def _created_at_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
