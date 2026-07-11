"""Markdown v1 writer for controlled synthetic quality reports."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


def write_visual_report_markdown(
    report: Mapping[str, Any],
    output_dir: str | PathLike[str],
) -> Path:
    output_path = Path(output_dir) / "visual_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(visual_report_markdown(report), encoding="utf-8")
    return output_path


def visual_report_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Controlled Synthetic Quality Report", ""]
    input_mode = str(report.get("config", {}).get("input_mode", "oracle"))
    for case in report["cases"]:
        case_id = str(case["case_id"])
        lines.extend(
            [
                f"## {case_id}",
                "",
            ]
        )
        variants = case["variants"]
        for variant, variant_report in variants.items():
            pipelines = variant_report.get("pipelines", {})
            lines.extend([f"### {variant}", ""])
            if input_mode == "both" and isinstance(pipelines, Mapping):
                oracle_report = pipelines["oracle"]
                scanner_report = pipelines["scanner"]
                lines.extend(
                    _pipeline_comparison_table(
                        oracle_report=oracle_report,
                        scanner_report=scanner_report,
                    )
                )
                lines.extend(
                    _visual_pipeline_section(
                        "oracle",
                        oracle_report,
                        case_id=case_id,
                        variant=variant,
                        variant_count=len(variants),
                        path_pipeline="oracle",
                        include_scanner=False,
                    )
                )
                lines.extend(
                    _visual_pipeline_section(
                        "scanner",
                        scanner_report,
                        case_id=case_id,
                        variant=variant,
                        variant_count=len(variants),
                        path_pipeline="scanner",
                        include_scanner=True,
                    )
                )
            else:
                lines.extend(
                    _visual_pipeline_section(
                        None,
                        variant_report,
                        case_id=case_id,
                        variant=variant,
                        variant_count=len(variants),
                        path_pipeline=None,
                        include_scanner="scanner_quality" in variant_report,
                    )
                )
    return "\n".join(lines).rstrip() + "\n"


def _visual_pipeline_section(
    pipeline_label: str | None,
    pipeline_report: Mapping[str, Any],
    *,
    case_id: str,
    variant: str,
    variant_count: int,
    path_pipeline: str | None,
    include_scanner: bool,
) -> list[str]:
    quality = pipeline_report["quality"]["fvt_top_truth_count"]
    overlap = quality["buffered_overlap_radius2"]
    distance = quality["surface_distance"]
    orientation = quality["orientation_error"]
    overlay_path = _figure_path(
        case_id,
        variant=variant,
        variant_count=variant_count,
        pipeline=path_pipeline,
        filename="truth_vs_fvt_overlay_i3_center.png",
    )
    skin_overlay_path = _figure_path(
        case_id,
        variant=variant,
        variant_count=variant_count,
        pipeline=path_pipeline,
        filename="truth_vs_skin_overlay_i3_center.png",
    )
    lines: list[str] = []
    if pipeline_label is not None:
        lines.extend([f"#### {pipeline_label} pipeline", ""])
    lines.extend(
        [
            f"- buffered_f1_r2: {_format_markdown_metric(overlap['buffered_f1'])}",
            f"- distance_p95: {_format_markdown_metric(distance['candidate_to_truth_p95'])}",
            f"- strike_median_error: {_format_markdown_metric(orientation['strike_median'])}",
            f"- dip_median_error: {_format_markdown_metric(orientation['dip_median'])}",
        ]
    )
    if include_scanner:
        lines.extend(_scanner_markdown_metrics(pipeline_report))
    if "thinning_diagnostic" in pipeline_report:
        lines.extend(
            _thinning_diagnostic_markdown(
                pipeline_report["thinning_diagnostic"],
                case_id=case_id,
                variant=variant,
                variant_count=variant_count,
                pipeline=path_pipeline,
            )
        )
    lines.extend(["", f"![fvt overlay]({overlay_path.as_posix()})", ""])
    if include_scanner:
        scanner_scan_overlay_path = _figure_path(
            case_id,
            variant=variant,
            variant_count=variant_count,
            pipeline=path_pipeline,
            filename="truth_vs_ft_scan_overlay_i3_center.png",
        )
        scanner_used_overlay_path = _figure_path(
            case_id,
            variant=variant,
            variant_count=variant_count,
            pipeline=path_pipeline,
            filename="truth_vs_ft_used_overlay_i3_center.png",
        )
        lines.extend(
            [
                f"![scanner ft scan overlay]({scanner_scan_overlay_path.as_posix()})",
                "",
                f"![scanner ft used overlay]({scanner_used_overlay_path.as_posix()})",
                "",
            ]
        )
    if bool(pipeline_report["skinning"]["enabled"]):
        skin_quality = pipeline_report["quality"]["skin"]
        skin_topology = skin_quality["topology"]
        skin_overlap = skin_quality["buffered_overlap_radius2"]
        skin_distance = skin_quality["surface_distance"]
        skin_orientation = skin_quality["orientation_error"]
        lines.extend(
            [
                f"- skin_count: {_format_markdown_metric(skin_topology['skin_count'])}",
                f"- skin_cell_count: {_format_markdown_metric(skin_topology['cell_count'])}",
                f"- skin_buffered_f1_r2: {_format_markdown_metric(skin_overlap['buffered_f1'])}",
                "- skin_distance_p95: "
                f"{_format_markdown_metric(skin_distance['candidate_to_truth_p95'])}",
                "- skin_strike_median_error: "
                f"{_format_markdown_metric(skin_orientation['strike_median'])}",
                "- skin_dip_median_error: "
                f"{_format_markdown_metric(skin_orientation['dip_median'])}",
                "",
                f"![skin overlay]({skin_overlay_path.as_posix()})",
                "",
            ]
        )
    else:
        lines.extend(["- skinning disabled", ""])
    return lines


def _thinning_diagnostic_markdown(
    diagnostic: Mapping[str, Any],
    *,
    case_id: str,
    variant: str,
    variant_count: int,
    pipeline: str | None,
) -> list[str]:
    reference_quality = diagnostic["reference"]["quality"]["fvt_top_truth_count"]
    normal_quality = diagnostic["normal"]["quality"]["fvt_top_truth_count"]
    delta = diagnostic["delta"]["normal_minus_reference"]
    keep_mask = diagnostic["keep_mask"]
    links = [
        (
            "reference overlay",
            "truth_vs_fvt_reference_overlay_i3_center.png",
        ),
        (
            "normal overlay",
            "truth_vs_fvt_normal_overlay_i3_center.png",
        ),
        (
            "reference-only overlay",
            "truth_vs_keep_reference_only_overlay_i3_center.png",
        ),
        (
            "normal-only overlay",
            "truth_vs_keep_normal_only_overlay_i3_center.png",
        ),
        (
            "reference vs normal",
            "fvt_reference_vs_normal_i3_center.png",
        ),
    ]
    lines = [
        "",
        "##### thinning diagnostic",
        "",
        "- reference buffered F1: "
        f"{_format_markdown_metric(reference_quality['buffered_overlap_radius2']['buffered_f1'])}",
        "- normal buffered F1: "
        f"{_format_markdown_metric(normal_quality['buffered_overlap_radius2']['buffered_f1'])}",
        f"- normal-minus-reference delta: {_format_markdown_metric(delta['fvt_buffered_f1_r2'])}",
        f"- keep-mask Jaccard: {_format_markdown_metric(keep_mask['jaccard'])}",
    ]
    for label, filename in links:
        path = _thinning_diagnostic_figure_path(
            case_id,
            variant=variant,
            variant_count=variant_count,
            pipeline=pipeline,
            filename=filename,
        )
        lines.append(f"- [{label}]({path.as_posix()})")
    return lines


def _scanner_markdown_metrics(pipeline_report: Mapping[str, Any]) -> list[str]:
    scanner_quality = pipeline_report["scanner_quality"]
    input_association = scanner_quality["input_association"]
    ft_quality = scanner_quality["ft_top_truth_count"]
    ft_overlap = ft_quality["buffered_overlap_radius2"]
    ft_distance = ft_quality["surface_distance"]
    orientation = scanner_quality["orientation_error"]["raw_scan_top_truth_count"]
    return [
        f"- scanner input contrast: {_format_markdown_metric(input_association['contrast'])}",
        f"- scanner ft buffered_f1: {_format_markdown_metric(ft_overlap['buffered_f1'])}",
        "- scanner ft distance_p95: "
        f"{_format_markdown_metric(ft_distance['candidate_to_truth_p95'])}",
        f"- scanner strike median error: {_format_markdown_metric(orientation['strike_median'])}",
        f"- scanner dip median error: {_format_markdown_metric(orientation['dip_median'])}",
    ]


def _pipeline_comparison_table(
    *,
    oracle_report: Mapping[str, Any],
    scanner_report: Mapping[str, Any],
) -> list[str]:
    headers = (
        "pipeline",
        "scanner input contrast",
        "scanner ft buffered_f1",
        "scanner ft distance_p95",
        "scanner strike median error",
        "scanner dip median error",
        "fvt buffered_f1",
        "skin buffered_f1",
    )
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        _pipeline_comparison_row("oracle", oracle_report, include_scanner=False),
        _pipeline_comparison_row("scanner", scanner_report, include_scanner=True),
        "",
    ]


def _pipeline_comparison_row(
    pipeline: str,
    pipeline_report: Mapping[str, Any],
    *,
    include_scanner: bool,
) -> str:
    fvt_overlap = pipeline_report["quality"]["fvt_top_truth_count"]["buffered_overlap_radius2"][
        "buffered_f1"
    ]
    skin_quality = pipeline_report["quality"]["skin"]
    skin_overlap = (
        skin_quality["buffered_overlap_radius2"]["buffered_f1"]
        if skin_quality is not None
        else "skinning disabled"
    )
    scanner_values: tuple[object, ...]
    if include_scanner:
        scanner_quality = pipeline_report["scanner_quality"]
        scanner_ft_quality = scanner_quality["ft_top_truth_count"]
        scanner_orientation = scanner_quality["orientation_error"]["raw_scan_top_truth_count"]
        scanner_values = (
            scanner_quality["input_association"]["contrast"],
            scanner_ft_quality["buffered_overlap_radius2"]["buffered_f1"],
            scanner_ft_quality["surface_distance"]["candidate_to_truth_p95"],
            scanner_orientation["strike_median"],
            scanner_orientation["dip_median"],
        )
    else:
        scanner_values = ("n/a", "n/a", "n/a", "n/a", "n/a")
    values = (pipeline, *scanner_values, fvt_overlap, skin_overlap)
    return "| " + " | ".join(_format_markdown_metric(value) for value in values) + " |"


def _figure_path(
    case_id: str,
    *,
    variant: str,
    variant_count: int,
    pipeline: str | None,
    filename: str,
) -> PurePosixPath:
    parts = [case_id]
    if variant_count > 1:
        parts.append(variant)
    if pipeline is not None:
        parts.append(pipeline)
    parts.extend(("figures", filename))
    return PurePosixPath(*parts)


def _thinning_diagnostic_figure_path(
    case_id: str,
    *,
    variant: str,
    variant_count: int,
    pipeline: str | None,
    filename: str,
) -> PurePosixPath:
    parts = [case_id]
    if variant_count > 1:
        parts.append(variant)
    if pipeline is not None:
        parts.append(pipeline)
    parts.extend(("thinning_diagnostic", filename))
    return PurePosixPath(*parts)


def _format_markdown_metric(value: object) -> str:
    if isinstance(value, int | float | np.floating | np.integer):
        return f"{float(value):.6g}"
    return str(value)
