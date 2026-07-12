"""Deterministic summaries of scanner boundary-stage diagnostics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

__all__ = [
    "scanner_boundary_stage_summary_markdown",
    "select_scanner_boundary_stage_diagnostics",
    "summarize_scanner_boundary_stages",
]


def select_scanner_boundary_stage_diagnostics(
    report: Mapping[str, Any],
    *,
    case_id: str,
    variant: str,
) -> Mapping[str, Any]:
    """Select one diagnostic through the canonical scanner pipeline path."""

    if not isinstance(report, Mapping) or report.get("format_version") != 1:
        raise ValueError("report must use format_version=1")
    cases = report.get("cases")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise ValueError("report is missing cases")
    matches = [
        case for case in cases if isinstance(case, Mapping) and case.get("case_id") == case_id
    ]
    if not matches:
        raise ValueError(f"case {case_id!r} is missing")
    if len(matches) != 1:
        raise ValueError(f"case {case_id!r} occurs more than once")

    case = matches[0]
    pipelines = _required_mapping(case, "pipelines", f"case {case_id!r}")
    scanner = _required_mapping(pipelines, "scanner", f"case {case_id!r} pipelines")
    variants = _required_mapping(scanner, "variants", "scanner pipeline")
    variant_report = _required_mapping(variants, variant, "scanner pipeline variants")
    return _required_mapping(
        variant_report,
        "scanner_boundary_stage_diagnostics",
        f"scanner variant {variant!r}",
    )


def summarize_scanner_boundary_stages(
    diagnostic: Mapping[str, Any],
    *,
    retention_threshold: float,
) -> dict[str, Any]:
    """Rank observed boundary metrics without making a promotion decision."""

    threshold = _finite_number(retention_threshold, "retention_threshold")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("retention_threshold must be between 0 and 1")
    if not isinstance(diagnostic, Mapping):
        raise ValueError("diagnostic must be a mapping")

    transition_order = _ordered_names(diagnostic, "transition_order")
    transitions = _required_mapping(diagnostic, "transitions", "diagnostic")
    transition_rows: list[dict[str, Any]] = []
    penalty_rows: list[dict[str, Any]] = []
    introduced_rows: list[dict[str, Any]] = []
    applicable_transitions: dict[str, bool] = {}
    for transition_name in transition_order:
        transition = _required_mapping(transitions, transition_name, "diagnostic transitions")
        applicable = transition.get("applicable", True)
        if not isinstance(applicable, bool):
            raise ValueError(f"transition {transition_name!r} applicable must be bool")
        applicable_transitions[transition_name] = applicable
        if not applicable:
            continue
        regions = _required_mapping(transition, "regions", f"transition {transition_name!r}")
        boundary = _required_mapping(
            regions, "boundary_shell", f"transition {transition_name!r} regions"
        )
        interior = _required_mapping(regions, "interior", f"transition {transition_name!r} regions")
        boundary_retention = _optional_number(
            boundary.get("retained_source_fraction"),
            f"{transition_name} boundary retained_source_fraction",
        )
        interior_retention = _optional_number(
            interior.get("retained_source_fraction"),
            f"{transition_name} interior retained_source_fraction",
        )
        introduced = _optional_number(
            boundary.get("introduced_target_fraction"),
            f"{transition_name} boundary introduced_target_fraction",
        )
        if boundary_retention is not None:
            transition_rows.append({"transition": transition_name, "value": boundary_retention})
        if boundary_retention is not None and interior_retention is not None:
            penalty_rows.append(
                {
                    "transition": transition_name,
                    "boundary": boundary_retention,
                    "interior": interior_retention,
                    "boundary_minus_interior": boundary_retention - interior_retention,
                }
            )
        if introduced is not None:
            introduced_rows.append({"transition": transition_name, "value": introduced})

    stage_order = _ordered_names(diagnostic, "stage_order")
    stages = _required_mapping(diagnostic, "stages", "diagnostic")
    recalls: dict[str, float | None] = {}
    for stage_name in stage_order:
        stage = _required_mapping(stages, stage_name, "diagnostic stages")
        regions = _required_mapping(stage, "regions", f"stage {stage_name!r}")
        boundary = _required_mapping(regions, "boundary_shell", f"stage {stage_name!r} regions")
        recall_value: object
        if "truth_recall" in boundary:
            recall_value = boundary["truth_recall"]
        else:
            truth = _required_mapping(boundary, "truth", f"stage {stage_name!r} boundary_shell")
            recall_value = truth.get("truth_recall")
        recalls[stage_name] = _optional_number(recall_value, f"{stage_name} boundary truth_recall")

    recall_rows: list[dict[str, Any]] = []
    for transition_name in transition_order:
        if not applicable_transitions[transition_name]:
            continue
        stage_pair = _transition_stage_pair(transition_name, stage_order)
        if stage_pair is None:
            continue
        source_stage, target_stage = stage_pair
        source_recall = recalls[source_stage]
        target_recall = recalls[target_stage]
        if source_recall is None or target_recall is None:
            continue
        delta = target_recall - source_recall
        recall_rows.append(
            {
                "source_stage": source_stage,
                "target_stage": target_stage,
                "source_recall": source_recall,
                "target_recall": target_recall,
                "delta": delta,
                "is_drop": delta < 0.0,
            }
        )

    ranked = sorted(transition_rows, key=lambda row: row["value"])
    return {
        "retention_threshold": threshold,
        "first_boundary_retention_below_threshold": next(
            (row.copy() for row in transition_rows if row["value"] < threshold), None
        ),
        "lowest_boundary_retention": ranked[0].copy() if ranked else None,
        "largest_boundary_vs_interior_retention_penalty": (
            min(penalty_rows, key=lambda row: row["boundary_minus_interior"]).copy()
            if penalty_rows
            else None
        ),
        "largest_boundary_truth_recall_drop": (
            min(recall_rows, key=lambda row: row["delta"]).copy() if recall_rows else None
        ),
        "highest_boundary_introduced_fraction": (
            max(introduced_rows, key=lambda row: row["value"]).copy() if introduced_rows else None
        ),
        "ranked_transitions_by_boundary_retention": [row.copy() for row in ranked],
    }


def scanner_boundary_stage_summary_markdown(
    *,
    case_id: str,
    variant: str,
    summary: Mapping[str, Any],
) -> str:
    """Format a scanner boundary-stage summary as deterministic Markdown."""

    first = summary.get("first_boundary_retention_below_threshold")
    lowest = summary.get("lowest_boundary_retention")
    penalty = summary.get("largest_boundary_vs_interior_retention_penalty")
    recall = summary.get("largest_boundary_truth_recall_drop")
    introduced = summary.get("highest_boundary_introduced_fraction")
    lines = [
        "# Scanner Boundary Stage Summary",
        "",
        f"- case: `{_markdown_text(case_id)}`",
        f"- variant: `{_markdown_text(variant)}`",
        f"- screening retention threshold: {_metric(summary.get('retention_threshold'))}",
        f"- first transition below threshold: {_transition_metric(first)}",
        f"- lowest boundary retention: {_transition_metric(lowest)}",
        f"- largest boundary-vs-interior retention penalty: {_penalty_metric(penalty)}",
        f"- largest boundary truth recall drop: {_recall_metric(recall)}",
        f"- highest boundary introduced fraction: {_transition_metric(introduced)}",
        "",
        "## Boundary Retention Ranking",
        "",
        "| rank | transition | boundary retained source fraction |",
        "|---:|---|---:|",
    ]
    ranking = summary.get("ranked_transitions_by_boundary_retention", ())
    for rank, row in enumerate(ranking, start=1):
        lines.append(
            f"| {rank} | `{_markdown_text(row['transition'])}` | {_metric(row['value'])} |"
        )
    if not ranking:
        lines.append("| n/a | n/a | n/a |")
    return "\n".join(lines) + "\n"


def _required_mapping(parent: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} is missing {key!r}")
    return value


def _ordered_names(diagnostic: Mapping[str, Any], key: str) -> list[str]:
    value = diagnostic.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"diagnostic is missing {key!r}")
    if any(not isinstance(name, str) for name in value):
        raise ValueError(f"diagnostic {key!r} must contain strings")
    if len(set(value)) != len(value):
        raise ValueError(f"diagnostic {key!r} must not contain duplicates")
    return list(value)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _optional_number(value: object, name: str) -> float | None:
    return None if value is None else _finite_number(value, name)


def _transition_stage_pair(
    transition_name: str, stage_order: Sequence[str]
) -> tuple[str, str] | None:
    for source_stage in stage_order:
        prefix = f"{source_stage}_to_"
        if transition_name.startswith(prefix):
            target_stage = transition_name[len(prefix) :]
            if target_stage in stage_order:
                return source_stage, target_stage
    return None


def _metric(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def _transition_metric(row: object) -> str:
    if not isinstance(row, Mapping):
        return "n/a"
    return f"`{_markdown_text(row['transition'])}` ({_metric(row['value'])})"


def _penalty_metric(row: object) -> str:
    if not isinstance(row, Mapping):
        return "n/a"
    return (
        f"`{_markdown_text(row['transition'])}` "
        f"(boundary {_metric(row['boundary'])}, interior {_metric(row['interior'])}, "
        f"difference {_metric(row['boundary_minus_interior'])})"
    )


def _recall_metric(row: object) -> str:
    if not isinstance(row, Mapping):
        return "n/a"
    return (
        f"`{_markdown_text(row['source_stage'])}` -> `{_markdown_text(row['target_stage'])}` "
        f"({_metric(row['source_recall'])} -> {_metric(row['target_recall'])}, "
        f"delta {_metric(row['delta'])})"
    )


def _markdown_text(value: object) -> str:
    return str(value).replace("|", "\\|").replace("`", "\\`")
