#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from compare_quality_reports import compare_reports


DEFAULT_CANDIDATES = (
    "boundary_edge_thin_v1",
    "boundary_seed_retention_v1",
    "quality_boundary_skinner_fallback_v5",
)

EXTENDED_CASES = (
    "single_vertical_plane",
    "single_dipping_plane",
    "curved_surface",
    "parallel_planes",
    "crossing_planes",
    "boundary_plane",
    "weak_noisy_plane",
)
TOPOLOGY_CASES = ("parallel_planes", "crossing_planes")
FALSE_FALLBACK_CASES = (
    "single_vertical_plane",
    "single_dipping_plane",
    "curved_surface",
    "weak_noisy_plane",
)
REQUIRED_SCANNER_BACKEND = "quality"
REQUIRED_SCANNER_REFINEMENT_FACTOR = "2"


def _parse_csv_list(text: str | None) -> tuple[str, ...]:
    if text is None:
        return ()
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("candidate list must not be empty")
    return values


def _candidate_variants(args: argparse.Namespace) -> tuple[str, ...]:
    if args.candidate_variant and args.candidate_variants:
        raise SystemExit("use either --candidate-variant or --candidate-variants, not both")
    if args.candidate_variant:
        return (args.candidate_variant,)
    if args.candidate_variants:
        return args.candidate_variants
    return DEFAULT_CANDIDATES


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    candidate_variants = _candidate_variants(args)
    candidate_reports = [
        _with_required_gate_coverage(
            compare_reports(
                args.baseline_summary,
                args.candidate_summary,
                args.baseline_variant,
                candidate_variant,
                args.promotion_gate,
                args.strict_missing_rows,
            )
        )
        for candidate_variant in candidate_variants
    ]
    candidates = {
        report["config"]["candidate_variant"]: report["promotion_gate"]
        for report in candidate_reports
    }
    promotable_candidates = [variant for variant, gate in candidates.items() if gate["passed"]]
    reasons = [
        f"{variant}: {reason}" for variant, gate in candidates.items() for reason in gate["reasons"]
    ]
    missing_rows = {
        report["config"]["candidate_variant"]: {
            "missing_baseline_rows": report["missing_baseline_rows"],
            "missing_candidate_rows": report["missing_candidate_rows"],
        }
        for report in candidate_reports
        if report["missing_baseline_rows"] or report["missing_candidate_rows"]
    }
    if args.strict_missing_rows:
        reasons.extend(f"{variant}: missing matched rows" for variant in sorted(missing_rows))
    return {
        "format_version": 1,
        "config": {
            "baseline_summary": str(args.baseline_summary),
            "candidate_summary": str(args.candidate_summary),
            "baseline_variant": args.baseline_variant,
            "candidate_variants": list(candidate_variants),
            "promotion_gate": args.promotion_gate,
            "strict_missing_rows": args.strict_missing_rows,
        },
        "promotion_gate": {
            "name": args.promotion_gate,
            "passed": not reasons,
            "promotable_candidates": promotable_candidates,
            "candidates": candidates,
            "missing_rows": missing_rows,
            "reasons": reasons,
        },
        "candidate_reports": candidate_reports,
    }


def _with_required_gate_coverage(report: dict[str, Any]) -> dict[str, Any]:
    if report["config"]["promotion_gate"] != "scanner-boundary":
        return report
    coverage = _required_gate_coverage(report)
    gate = dict(report["promotion_gate"])
    reasons = [*gate["reasons"], *coverage["reasons"]]
    gate["coverage"] = coverage
    gate["reasons"] = reasons
    gate["passed"] = not reasons
    report = dict(report)
    report["promotion_gate"] = gate
    return report


def _required_gate_coverage(report: dict[str, Any]) -> dict[str, Any]:
    comparisons = report["comparisons"]
    checks = [
        _coverage_check(
            "boundary_plane_scanner_quality_ref2_49",
            comparisons,
            pipeline="scanner",
            case_ids=("boundary_plane",),
        ),
        _coverage_check(
            "non_boundary_scanner_quality_ref2_49",
            comparisons,
            pipeline="scanner",
            case_ids=tuple(case_id for case_id in EXTENDED_CASES if case_id != "boundary_plane"),
        ),
        _coverage_check(
            "oracle_49",
            comparisons,
            pipeline="oracle",
            case_ids=EXTENDED_CASES,
        ),
        _coverage_check(
            "false_fallback_replacement_scanner_quality_ref2_49",
            comparisons,
            pipeline="scanner",
            case_ids=FALSE_FALLBACK_CASES,
        ),
        _coverage_check(
            "topology_scanner_quality_ref2_49",
            comparisons,
            pipeline="scanner",
            case_ids=TOPOLOGY_CASES,
        ),
    ]
    reasons = [
        f"missing required gate coverage: {check['name']} {case_id}"
        for check in checks
        for case_id in check["missing_case_ids"]
    ]
    return {
        "passed": not reasons,
        "checks": checks,
        "reasons": reasons,
    }


def _coverage_check(
    name: str,
    comparisons: list[dict[str, Any]],
    *,
    pipeline: str,
    case_ids: tuple[str, ...],
) -> dict[str, Any]:
    matched = {
        comparison["key"]["case_id"]
        for comparison in comparisons
        if _comparison_matches_required_coverage(comparison, pipeline=pipeline)
    }
    matched_case_ids = [case_id for case_id in case_ids if case_id in matched]
    missing_case_ids = [case_id for case_id in case_ids if case_id not in matched]
    return {
        "name": name,
        "pipeline": pipeline,
        "scanner_backend": REQUIRED_SCANNER_BACKEND if pipeline == "scanner" else None,
        "scanner_refinement_factor": (
            REQUIRED_SCANNER_REFINEMENT_FACTOR if pipeline == "scanner" else None
        ),
        "required_case_ids": list(case_ids),
        "matched_case_ids": matched_case_ids,
        "missing_case_ids": missing_case_ids,
        "passed": not missing_case_ids,
    }


def _comparison_matches_required_coverage(
    comparison: dict[str, Any],
    *,
    pipeline: str,
) -> bool:
    key = comparison["key"]
    if (
        key["pipeline"] != pipeline
        or key["workflow_mode"] != "quality"
        or key["shape_n3"] != "49"
        or key["shape_n2"] != "49"
        or key["shape_n1"] != "49"
    ):
        return False
    if pipeline != "scanner":
        return True
    return (
        key.get("scanner_backend") == REQUIRED_SCANNER_BACKEND
        and key.get("scanner_refinement_factor") == REQUIRED_SCANNER_REFINEMENT_FACTOR
    )


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    gate = report["promotion_gate"]
    lines = [
        "# Synthetic Quality Promotion Gate",
        "",
        f"- baseline: `{report['config']['baseline_summary']}`",
        f"- candidate: `{report['config']['candidate_summary']}`",
        f"- baseline variant: `{report['config']['baseline_variant']}`",
        f"- promotion gate: `{gate['name']}` {'pass' if gate['passed'] else 'fail'}",
        f"- promotable candidates: {_format_list(gate['promotable_candidates'])}",
        "",
        "| candidate | gate | boundary skin F1 | skin count | skin/FVT ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for variant, candidate_gate in gate["candidates"].items():
        boundary = candidate_gate["boundary_plane"]
        lines.append(
            "| "
            f"{variant} | {'pass' if candidate_gate['passed'] else 'fail'} | "
            f"{_boundary_metric(boundary, 'skin_buffered_f1_r2')} | "
            f"{_boundary_metric(boundary, 'skin_count')} | "
            f"{_boundary_metric(boundary, 'skin_cell_to_fvt_positive_candidate_ratio')} |"
        )
    lines.extend(["", "## Reasons", ""])
    if gate["reasons"]:
        lines.extend(f"- {reason}" for reason in gate["reasons"])
    else:
        lines.append("None.")
    lines.extend(["", "## Coverage", ""])
    for variant, candidate_gate in gate["candidates"].items():
        coverage = candidate_gate.get("coverage")
        if coverage is None:
            continue
        lines.append(f"### {variant}")
        lines.append("")
        for check in coverage["checks"]:
            status = "pass" if check["passed"] else "fail"
            missing = _format_list(check["missing_case_ids"])
            lines.append(f"- `{check['name']}`: {status}; missing: {missing}")
        lines.append("")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _boundary_metric(boundary: dict[str, Any] | None, metric: str) -> str:
    if boundary is None:
        return ""
    value = boundary["metrics"][metric]["candidate"]
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def _format_list(values: list[str]) -> str:
    if not values:
        return "none"
    return ", ".join(f"`{value}`" for value in values)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize scanner-boundary promotion gate results from synthetic "
            "quality summary.csv reports."
        ),
    )
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--baseline-variant", default="current_default")
    parser.add_argument("--candidate-variant")
    parser.add_argument("--candidate-variants", type=_parse_csv_list)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument(
        "--promotion-gate",
        choices=("scanner-boundary",),
        default="scanner-boundary",
    )
    parser.add_argument("--strict-missing-rows", action="store_true")
    parser.add_argument("--fail-on-gate-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output_json = args.output_json or args.candidate_summary.with_name("promotion_gate.json")
    report = build_report(args)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_markdown is not None:
        _write_markdown(report, args.output_markdown)
    if args.strict_missing_rows and report["promotion_gate"]["missing_rows"]:
        return 2
    if args.fail_on_gate_failure and not report["promotion_gate"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
