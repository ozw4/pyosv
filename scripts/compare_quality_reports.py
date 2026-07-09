#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


MATCH_KEY_FIELDS = (
    "case_id",
    "pipeline",
    "input_mode",
    "workflow_mode",
    "scanner_backend",
    "shape_n3",
    "shape_n2",
    "shape_n1",
)

METRIC_COLUMNS = (
    "fv_buffered_f1_r2",
    "fv_distance_p95",
    "fv_edge_false_positive_fraction",
    "fvt_buffered_f1_r2",
    "fvt_distance_p95",
    "fvt_positive_candidate_count",
    "fvt_positive_buffered_f1_r2",
    "fvt_positive_distance_p95",
    "fvt_positive_edge_false_positive_fraction",
    "skin_count",
    "skin_cell_count",
    "skin_buffered_f1_r2",
    "skin_distance_p95",
    "skin_distance_candidate_to_truth_p95",
    "skin_strike_median_error",
    "skin_dip_median_error",
    "scanner_ft_buffered_f1_r2",
    "scanner_ft_distance_p95",
    "scanner_downstream_fvt_to_ft_distance_p95",
    "scanner_downstream_fvt_positive_edge_false_positive_fraction",
    "skin_fallback_used",
    "skin_fallback_replaced_primary",
)

HIGHER_IS_BETTER = {
    "fv_buffered_f1_r2",
    "fvt_buffered_f1_r2",
    "fvt_positive_buffered_f1_r2",
    "skin_buffered_f1_r2",
    "scanner_ft_buffered_f1_r2",
}

LOWER_IS_BETTER = {
    "fv_distance_p95",
    "fv_edge_false_positive_fraction",
    "fvt_distance_p95",
    "fvt_positive_distance_p95",
    "fvt_positive_edge_false_positive_fraction",
    "skin_distance_p95",
    "skin_distance_candidate_to_truth_p95",
    "skin_strike_median_error",
    "skin_dip_median_error",
    "scanner_ft_distance_p95",
    "scanner_downstream_fvt_to_ft_distance_p95",
    "scanner_downstream_fvt_positive_edge_false_positive_fraction",
}

MATERIAL_REGRESSION_THRESHOLDS = {
    "skin_buffered_f1_r2": ("lt", -0.02),
    "fvt_positive_buffered_f1_r2": ("lt", -0.02),
    "skin_distance_p95": ("gt", 2.0),
    "fvt_positive_distance_p95": ("gt", 2.0),
}


def _read_csv(path: Path, variant: str) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows: dict[tuple[str, ...], dict[str, str]] = {}
        for row in reader:
            if _text(row.get("variant")) != variant:
                continue
            key = _row_key(row)
            rows.setdefault(key, {field: _text(value) for field, value in row.items()})
        return rows


def _row_key(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_text(row.get(field)) for field in MATCH_KEY_FIELDS)


def _key_dict(key: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(MATCH_KEY_FIELDS, key, strict=True))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_value(row: dict[str, str] | None, column: str) -> float | bool | str | None:
    if row is None or column not in row:
        return None
    raw = _text(row.get(column))
    if raw == "":
        return None
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        value = float(raw)
    except ValueError:
        return raw
    if not math.isfinite(value):
        return None
    return value


def _numeric(value: float | bool | str | None) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _metric_delta(
    baseline: float | bool | str | None,
    candidate: float | bool | str | None,
) -> float | None:
    baseline_number = _numeric(baseline)
    candidate_number = _numeric(candidate)
    if baseline_number is None or candidate_number is None:
        return None
    return candidate_number - baseline_number


def _ratio(row: dict[str, str] | None) -> float | None:
    skin_cell_count = _numeric(_parse_value(row, "skin_cell_count"))
    fvt_count = _numeric(_parse_value(row, "fvt_positive_candidate_count"))
    if skin_cell_count is None or fvt_count is None or fvt_count == 0:
        return None
    return skin_cell_count / fvt_count


def _comparison(
    key: tuple[str, ...],
    baseline_row: dict[str, str],
    candidate_row: dict[str, str],
    baseline_variant: str,
    candidate_variant: str,
) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    regressions: list[str] = []
    improvements: list[str] = []
    for column in METRIC_COLUMNS:
        baseline = _parse_value(baseline_row, column)
        candidate = _parse_value(candidate_row, column)
        delta = _metric_delta(baseline, candidate)
        metrics[column] = {
            "baseline": baseline,
            "candidate": candidate,
            "delta": delta,
        }
        if delta is None or delta == 0:
            continue
        if column in HIGHER_IS_BETTER:
            (improvements if delta > 0 else regressions).append(column)
        elif column in LOWER_IS_BETTER:
            (improvements if delta < 0 else regressions).append(column)

    baseline_ratio = _ratio(baseline_row)
    candidate_ratio = _ratio(candidate_row)
    derived_delta = None
    if baseline_ratio is not None and candidate_ratio is not None:
        derived_delta = candidate_ratio - baseline_ratio

    return {
        "key": _key_dict(key),
        "baseline_variant": baseline_variant,
        "candidate_variant": candidate_variant,
        "metrics": metrics,
        "derived": {
            "skin_cell_to_fvt_positive_candidate_ratio": {
                "baseline": baseline_ratio,
                "candidate": candidate_ratio,
                "delta": derived_delta,
            }
        },
        "regressions": regressions,
        "improvements": improvements,
    }


def compare_reports(
    baseline_summary: Path,
    candidate_summary: Path,
    baseline_variant: str,
    candidate_variant: str,
    promotion_gate: str,
    strict_missing_rows: bool,
) -> dict[str, Any]:
    baseline_rows = _read_csv(baseline_summary, baseline_variant)
    candidate_rows = _read_csv(candidate_summary, candidate_variant)
    baseline_keys = set(baseline_rows)
    candidate_keys = set(candidate_rows)
    common_keys = sorted(baseline_keys & candidate_keys)
    missing_baseline_rows = [_key_dict(key) for key in sorted(candidate_keys - baseline_keys)]
    missing_candidate_rows = [_key_dict(key) for key in sorted(baseline_keys - candidate_keys)]

    comparisons = [
        _comparison(
            key,
            baseline_rows[key],
            candidate_rows[key],
            baseline_variant,
            candidate_variant,
        )
        for key in common_keys
    ]
    comparison_by_key = {
        tuple(comparison["key"][field] for field in MATCH_KEY_FIELDS): comparison
        for comparison in comparisons
    }
    report = {
        "format_version": 1,
        "config": {
            "baseline_summary": str(baseline_summary),
            "candidate_summary": str(candidate_summary),
            "baseline_variant": baseline_variant,
            "candidate_variant": candidate_variant,
            "promotion_gate": promotion_gate,
            "strict_missing_rows": strict_missing_rows,
        },
        "row_count": len(comparisons),
        "missing_baseline_rows": missing_baseline_rows,
        "missing_candidate_rows": missing_candidate_rows,
        "comparisons": comparisons,
        "promotion_gate": _promotion_gate(
            promotion_gate,
            candidate_rows,
            comparison_by_key,
        ),
    }
    return report


def _promotion_gate(
    name: str,
    candidate_rows: dict[tuple[str, ...], dict[str, str]],
    comparison_by_key: dict[tuple[str, ...], dict[str, Any]],
) -> dict[str, Any]:
    if name == "none":
        return {
            "name": "none",
            "passed": True,
            "boundary_plane": None,
            "non_boundary_regressions": [],
            "oracle_regressions": [],
            "reasons": [],
        }
    return _scanner_boundary_gate(candidate_rows, comparison_by_key)


def _scanner_boundary_gate(
    candidate_rows: dict[tuple[str, ...], dict[str, str]],
    comparison_by_key: dict[tuple[str, ...], dict[str, Any]],
) -> dict[str, Any]:
    reasons: list[str] = []
    boundary_keys = [
        key
        for key in sorted(candidate_rows)
        if _is_boundary_plane_scanner_49_quality(_key_dict(key))
    ]
    boundary_comparison = None
    if not boundary_keys:
        reasons.append("missing boundary_plane scanner quality 49^3 candidate row")
    else:
        boundary_comparison = comparison_by_key.get(boundary_keys[0])
        if boundary_comparison is None:
            reasons.append("missing matching baseline row for boundary_plane scanner quality 49^3")

    boundary_result = None
    if boundary_comparison is not None:
        boundary_result = _boundary_plane_result(boundary_comparison)
        reasons.extend(boundary_result["reasons"])

    non_boundary_regressions = _material_regressions(
        comparison_by_key.values(),
        pipeline="scanner",
        exclude_boundary=True,
    )
    oracle_regressions = _material_regressions(
        comparison_by_key.values(),
        pipeline="oracle",
        exclude_boundary=False,
    )
    for regression in non_boundary_regressions:
        reasons.append(f"non-boundary scanner regression: {regression['metric']}")
    for regression in oracle_regressions:
        reasons.append(f"oracle regression: {regression['metric']}")

    return {
        "name": "scanner-boundary",
        "passed": not reasons,
        "boundary_plane": boundary_result,
        "non_boundary_regressions": non_boundary_regressions,
        "oracle_regressions": oracle_regressions,
        "reasons": reasons,
    }


def _is_boundary_plane_scanner_49_quality(key: dict[str, str]) -> bool:
    return (
        key["case_id"] == "boundary_plane"
        and key["pipeline"] == "scanner"
        and key["workflow_mode"] == "quality"
        and _shape_is_49(key)
    )


def _shape_is_49(key: dict[str, str]) -> bool:
    return key["shape_n3"] == "49" and key["shape_n2"] == "49" and key["shape_n1"] == "49"


def _boundary_plane_result(comparison: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    metrics = comparison["metrics"]
    derived = comparison["derived"]["skin_cell_to_fvt_positive_candidate_ratio"]
    skin_f1 = metrics["skin_buffered_f1_r2"]["candidate"]
    skin_count = metrics["skin_count"]["candidate"]
    ratio = derived["candidate"]
    if _numeric(skin_f1) is None or _numeric(skin_f1) < 0.90:
        reasons.append("skin_buffered_f1_r2 below 0.90")
    if _numeric(skin_count) is None or _numeric(skin_count) > 3:
        reasons.append("skin_count above 3")
    if ratio is None or ratio < 0.75 or ratio > 1.25:
        reasons.append("skin_cell_to_fvt_positive_candidate_ratio outside [0.75, 1.25]")

    fvt_changed = _fvt_changed(metrics)
    if fvt_changed:
        fvt_f1 = metrics["fvt_positive_buffered_f1_r2"]["candidate"]
        fvt_distance = metrics["fvt_positive_distance_p95"]["candidate"]
        if _numeric(fvt_f1) is None or _numeric(fvt_f1) < 0.90:
            reasons.append("fvt_positive_buffered_f1_r2 below 0.90 after FVT change")
        if _numeric(fvt_distance) is None or _numeric(fvt_distance) > 2.0:
            reasons.append("fvt_positive_distance_p95 above 2.0 after FVT change")

    return {
        "key": comparison["key"],
        "passed": not reasons,
        "fvt_changed": fvt_changed,
        "metrics": {
            "skin_buffered_f1_r2": metrics["skin_buffered_f1_r2"],
            "skin_count": metrics["skin_count"],
            "skin_cell_to_fvt_positive_candidate_ratio": derived,
            "fvt_positive_buffered_f1_r2": metrics["fvt_positive_buffered_f1_r2"],
            "fvt_positive_distance_p95": metrics["fvt_positive_distance_p95"],
            "fvt_positive_candidate_count": metrics["fvt_positive_candidate_count"],
        },
        "reasons": reasons,
    }


def _fvt_changed(metrics: dict[str, dict[str, Any]]) -> bool:
    for column in ("fvt_positive_buffered_f1_r2", "fvt_positive_distance_p95"):
        delta = metrics[column]["delta"]
        if delta is not None and abs(delta) > 1e-9:
            return True
    count_delta = metrics["fvt_positive_candidate_count"]["delta"]
    return count_delta is not None and count_delta != 0


def _material_regressions(
    comparisons: Any,
    *,
    pipeline: str,
    exclude_boundary: bool,
) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    for comparison in comparisons:
        key = comparison["key"]
        if key["pipeline"] != pipeline or not _shape_is_49(key):
            continue
        if exclude_boundary and key["case_id"] == "boundary_plane":
            continue
        for metric, (direction, threshold) in MATERIAL_REGRESSION_THRESHOLDS.items():
            delta = comparison["metrics"][metric]["delta"]
            if delta is None:
                continue
            if direction == "lt" and delta >= threshold:
                continue
            if direction == "gt" and delta <= threshold:
                continue
            regressions.append(
                {
                    "key": key,
                    "metric": metric,
                    "baseline": comparison["metrics"][metric]["baseline"],
                    "candidate": comparison["metrics"][metric]["candidate"],
                    "delta": delta,
                    "threshold": threshold,
                }
            )
    return regressions


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    gate = report["promotion_gate"]
    lines = [
        "# Quality Delta",
        "",
        f"- baseline: `{report['config']['baseline_summary']}`",
        f"- candidate: `{report['config']['candidate_summary']}`",
        f"- baseline variant: `{report['config']['baseline_variant']}`",
        f"- candidate variant: `{report['config']['candidate_variant']}`",
        f"- row count: {report['row_count']}",
        f"- missing baseline rows: {len(report['missing_baseline_rows'])}",
        f"- missing candidate rows: {len(report['missing_candidate_rows'])}",
        f"- promotion gate: `{gate['name']}` {'pass' if gate['passed'] else 'fail'}",
        "",
    ]
    if gate["boundary_plane"] is not None:
        lines.extend(_boundary_markdown(gate["boundary_plane"]))
    regressions = gate["non_boundary_regressions"] + gate["oracle_regressions"]
    lines.extend(["## Material Regressions", ""])
    if not regressions:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| case_id | pipeline | metric | baseline | candidate | delta |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        for regression in regressions:
            key = regression["key"]
            lines.append(
                "| "
                f"{key['case_id']} | {key['pipeline']} | {regression['metric']} | "
                f"{_format_value(regression['baseline'])} | "
                f"{_format_value(regression['candidate'])} | "
                f"{_format_value(regression['delta'])} |"
            )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _boundary_markdown(boundary_plane: dict[str, Any]) -> list[str]:
    lines = [
        "## Boundary Plane Scanner 49^3",
        "",
        f"Gate result: {'pass' if boundary_plane['passed'] else 'fail'}",
        "",
        "| metric | baseline | candidate | delta |",
        "|---|---:|---:|---:|",
    ]
    for metric, values in boundary_plane["metrics"].items():
        lines.append(
            "| "
            f"{metric} | {_format_value(values['baseline'])} | "
            f"{_format_value(values['candidate'])} | {_format_value(values['delta'])} |"
        )
    lines.append("")
    return lines


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate synthetic quality summary.csv reports.",
    )
    parser.add_argument("baseline_summary", type=Path)
    parser.add_argument("candidate_summary", type=Path)
    parser.add_argument("--baseline-variant", default="current_default")
    parser.add_argument("--candidate-variant")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--promotion-gate", choices=("none", "scanner-boundary"), default="none")
    parser.add_argument("--fail-on-gate-failure", action="store_true")
    parser.add_argument("--strict-missing-rows", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidate_variant = args.candidate_variant or args.baseline_variant
    output_json = args.output_json or args.candidate_summary.with_name("quality_delta.json")
    report = compare_reports(
        args.baseline_summary,
        args.candidate_summary,
        args.baseline_variant,
        candidate_variant,
        args.promotion_gate,
        args.strict_missing_rows,
    )
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_markdown is not None:
        _write_markdown(report, args.output_markdown)

    missing_rows = report["missing_baseline_rows"] or report["missing_candidate_rows"]
    if args.strict_missing_rows and missing_rows:
        return 2
    if args.fail_on_gate_failure and not report["promotion_gate"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
