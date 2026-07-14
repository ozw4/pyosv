"""Comparison report construction for synthetic-quality summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .gates import add_required_coverage, evaluate_gate, validate_gate_comparison_profile
from .rows import MatchKey, MetricValue, SummaryRow, key_dict, numeric, read_summary_rows
from .scanner_policy import (
    SCANNER_POLICY_PROFILES,
    build_scanner_policy_contract,
    load_metrics_report,
    validate_summary_matches_metrics,
)
from .specifications import (
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    MATCH_KEY_FIELDS,
    METRIC_COLUMNS,
    SCANNER_BOUNDARY_REFERENCE_LIKE_GATE,
)


VARIANT_COMPARISON_PROFILE = "variant"
COMPARISON_PROFILES = (VARIANT_COMPARISON_PROFILE, *SCANNER_POLICY_PROFILES)


def metric_delta(baseline: MetricValue, candidate: MetricValue) -> float | None:
    baseline_number = numeric(baseline)
    candidate_number = numeric(candidate)
    if baseline_number is None or candidate_number is None:
        return None
    return candidate_number - baseline_number


def _ratio(row: SummaryRow) -> float | None:
    skin_cell_count = numeric(row.value("skin_cell_count"))
    fvt_count = numeric(row.value("fvt_positive_candidate_count"))
    if skin_cell_count is None or fvt_count in {None, 0}:
        return None
    return skin_cell_count / fvt_count


def compare_rows(
    key: MatchKey,
    baseline_row: SummaryRow,
    candidate_row: SummaryRow,
    baseline_variant: str,
    candidate_variant: str,
) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    regressions: list[str] = []
    improvements: list[str] = []
    for column in METRIC_COLUMNS:
        baseline = baseline_row.value(column)
        candidate = candidate_row.value(column)
        delta = metric_delta(baseline, candidate)
        metrics[column] = {"baseline": baseline, "candidate": candidate, "delta": delta}
        if delta is None or delta == 0:
            continue
        if column in HIGHER_IS_BETTER:
            (improvements if delta > 0 else regressions).append(column)
        elif column in LOWER_IS_BETTER:
            (improvements if delta < 0 else regressions).append(column)
    baseline_ratio = _ratio(baseline_row)
    candidate_ratio = _ratio(candidate_row)
    ratio_delta = None
    if baseline_ratio is not None and candidate_ratio is not None:
        ratio_delta = candidate_ratio - baseline_ratio
    return {
        "key": key_dict(key),
        "baseline_variant": baseline_variant,
        "candidate_variant": candidate_variant,
        "metrics": metrics,
        "derived": {
            "skin_cell_to_fvt_positive_candidate_ratio": {
                "baseline": baseline_ratio,
                "candidate": candidate_ratio,
                "delta": ratio_delta,
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
    promotion_gate: str = "none",
    strict_missing_rows: bool = False,
    *,
    comparison_profile: str = VARIANT_COMPARISON_PROFILE,
    baseline_metrics: Path | None = None,
    candidate_metrics: Path | None = None,
) -> dict[str, Any]:
    if comparison_profile not in COMPARISON_PROFILES:
        raise ValueError(f"unknown comparison profile: {comparison_profile}")
    validate_gate_comparison_profile(promotion_gate, comparison_profile)
    scanner_policy_contract = None
    if comparison_profile in SCANNER_POLICY_PROFILES:
        if baseline_metrics is None:
            raise ValueError(f"{comparison_profile} requires a baseline metrics path")
        if candidate_metrics is None:
            raise ValueError(f"{comparison_profile} requires a candidate metrics path")
        baseline_metrics_report = load_metrics_report(baseline_metrics, context="baseline")
        candidate_metrics_report = load_metrics_report(candidate_metrics, context="candidate")
        validate_summary_matches_metrics(
            baseline_summary,
            baseline_metrics_report,
            metrics_path=baseline_metrics,
            context="baseline",
        )
        validate_summary_matches_metrics(
            candidate_summary,
            candidate_metrics_report,
            metrics_path=candidate_metrics,
            context="candidate",
        )
        scanner_policy_contract = build_scanner_policy_contract(
            baseline_metrics_report,
            candidate_metrics_report,
            baseline_variant,
            candidate_variant,
            comparison_profile=comparison_profile,
        )

    baseline_rows = read_summary_rows(baseline_summary, baseline_variant)
    candidate_rows = read_summary_rows(candidate_summary, candidate_variant)
    baseline_keys, candidate_keys = set(baseline_rows), set(candidate_rows)
    common_keys = sorted(baseline_keys & candidate_keys)
    comparisons = [
        compare_rows(
            key, baseline_rows[key], candidate_rows[key], baseline_variant, candidate_variant
        )
        for key in common_keys
    ]
    comparison_by_key = {
        tuple(comparison["key"][field] for field in MATCH_KEY_FIELDS): comparison
        for comparison in comparisons
    }
    promotion_gate_result = evaluate_gate(promotion_gate, candidate_rows, comparison_by_key)
    config = {
        "baseline_summary": str(baseline_summary),
        "candidate_summary": str(candidate_summary),
        "baseline_variant": baseline_variant,
        "candidate_variant": candidate_variant,
        "promotion_gate": promotion_gate,
        "strict_missing_rows": strict_missing_rows,
    }
    report = {
        "format_version": 1,
        "config": config,
        "row_count": len(comparisons),
        "missing_baseline_rows": [key_dict(key) for key in sorted(candidate_keys - baseline_keys)],
        "missing_candidate_rows": [key_dict(key) for key in sorted(baseline_keys - candidate_keys)],
        "comparisons": comparisons,
        "promotion_gate": promotion_gate_result,
    }
    if promotion_gate == SCANNER_BOUNDARY_REFERENCE_LIKE_GATE.name:
        missing_reasons = []
        if report["missing_baseline_rows"]:
            missing_reasons.append(
                "scanner-boundary-reference-like requires zero missing baseline rows"
            )
        if report["missing_candidate_rows"]:
            missing_reasons.append(
                "scanner-boundary-reference-like requires zero missing candidate rows"
            )
        if missing_reasons:
            promotion_gate_result["reasons"] = [
                *promotion_gate_result["reasons"],
                *missing_reasons,
            ]
            promotion_gate_result["passed"] = False
    if scanner_policy_contract is None:
        return report

    config.update(
        {
            "comparison_profile": comparison_profile,
            "baseline_metrics": str(baseline_metrics),
            "candidate_metrics": str(candidate_metrics),
        }
    )
    contract_reasons = [
        f"scanner policy contract: {reason}" for reason in scanner_policy_contract["reasons"]
    ]
    promotion_gate_result["scanner_policy_contract"] = scanner_policy_contract
    promotion_gate_result["reasons"] = [
        *promotion_gate_result["reasons"],
        *contract_reasons,
    ]
    promotion_gate_result["passed"] = not promotion_gate_result["reasons"]
    report["scanner_policy_contract"] = scanner_policy_contract
    return add_required_coverage(report)
