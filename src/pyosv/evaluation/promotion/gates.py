"""Promotion-gate evaluation and aggregate report construction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .rows import MatchKey, SummaryRow, key_dict, numeric
from .specifications import (
    DEFAULT_CANDIDATES,
    MATERIAL_REGRESSION_THRESHOLDS,
    METRIC_COLUMNS,
    PROMOTION_GATES,
    TOPOLOGY_CASES,
    GateSpec,
)


def evaluate_gate(
    name: str,
    candidate_rows: dict[MatchKey, SummaryRow],
    comparison_by_key: dict[MatchKey, dict[str, Any]],
) -> dict[str, Any]:
    if name == "none":
        return {
            "name": "none",
            "passed": True,
            "boundary_plane": None,
            "non_boundary_regressions": [],
            "oracle_regressions": [],
            "false_fallback_replacements": [],
            "topology_regressions": [],
            "reasons": [],
        }
    spec = PROMOTION_GATES.get(name)
    if spec is None:
        raise ValueError(f"unknown promotion gate: {name}")
    return _scanner_boundary_gate(candidate_rows, comparison_by_key, spec)


def _scanner_boundary_gate(
    candidate_rows: dict[MatchKey, SummaryRow],
    comparison_by_key: dict[MatchKey, dict[str, Any]],
    spec: GateSpec,
) -> dict[str, Any]:
    reasons: list[str] = []
    boundary_keys = [
        key
        for key in sorted(candidate_rows)
        if key_dict(key)["case_id"] == "boundary_plane"
        and _is_scanner_gate_row(key_dict(key), spec)
    ]
    boundary_comparison = None
    scanner_description = _scanner_gate_description(spec)
    if not boundary_keys:
        reasons.append(f"missing boundary_plane {scanner_description} candidate row")
    else:
        boundary_comparison = comparison_by_key.get(boundary_keys[0])
        if boundary_comparison is None:
            reasons.append(
                f"missing matching baseline row for boundary_plane {scanner_description}"
            )
    boundary_result = None
    if boundary_comparison is not None:
        boundary_result = _boundary_plane_result(boundary_comparison, spec)
        reasons.extend(boundary_result["reasons"])

    comparisons = comparison_by_key.values()
    non_boundary = _material_regressions(
        comparisons, pipeline="scanner", exclude_boundary=True, spec=spec
    )
    oracle = _oracle_regressions(comparisons, spec)
    false_fallback = _false_fallback_replacements(comparisons, spec)
    topology = _topology_regressions(comparisons, spec)
    reasons.extend(f"non-boundary scanner regression: {item['metric']}" for item in non_boundary)
    reasons.extend(f"oracle regression: {item['metric']}" for item in oracle)
    reasons.extend(
        f"stable non-boundary false fallback replacement: {item['key']['case_id']}"
        for item in false_fallback
    )
    reasons.extend(
        f"parallel/crossing topology regression: {item['key']['case_id']} {item['metric']}"
        for item in topology
    )
    return {
        "name": spec.name,
        "passed": not reasons,
        "boundary_plane": boundary_result,
        "non_boundary_regressions": non_boundary,
        "oracle_regressions": oracle,
        "false_fallback_replacements": false_fallback,
        "topology_regressions": topology,
        "reasons": reasons,
    }


def _shape_matches(key: dict[str, str], spec: GateSpec) -> bool:
    return tuple(key[f"shape_n{axis}"] for axis in (3, 2, 1)) == spec.required_shape


def _scanner_gate_description(spec: GateSpec) -> str:
    if len(set(spec.required_shape)) == 1:
        shape = f"{spec.required_shape[0]}^3"
    else:
        shape = "x".join(spec.required_shape)
    return f"scanner {spec.scanner_backend} refinement-{spec.scanner_refinement_factor} {shape}"


def _is_scanner_gate_row(key: dict[str, str], spec: GateSpec) -> bool:
    return (
        key["pipeline"] == "scanner"
        and key["workflow_mode"] == "quality"
        and key["scanner_backend"] == spec.scanner_backend
        and key["scanner_refinement_factor"] == spec.scanner_refinement_factor
        and _shape_matches(key, spec)
    )


def _boundary_plane_result(comparison: dict[str, Any], spec: GateSpec) -> dict[str, Any]:
    reasons: list[str] = []
    metrics = comparison["metrics"]
    ratio = comparison["derived"]["skin_cell_to_fvt_positive_candidate_ratio"]
    skin_f1 = numeric(metrics["skin_buffered_f1_r2"]["candidate"])
    skin_count = numeric(metrics["skin_count"]["candidate"])
    candidate_ratio = ratio["candidate"]
    if skin_f1 is None or skin_f1 < spec.boundary_skin_f1_min:
        reasons.append("skin_buffered_f1_r2 below 0.90")
    if skin_count is None or skin_count > spec.boundary_skin_count_max:
        reasons.append("skin_count above 3")
    ratio_min, ratio_max = spec.boundary_ratio_range
    if candidate_ratio is None or not ratio_min <= candidate_ratio <= ratio_max:
        reasons.append("skin_cell_to_fvt_positive_candidate_ratio outside [0.75, 1.25]")
    fvt_changed = _fvt_changed(metrics)
    if fvt_changed or spec.always_check_boundary_fvt:
        fvt_f1 = numeric(metrics["fvt_positive_buffered_f1_r2"]["candidate"])
        fvt_distance = numeric(metrics["fvt_positive_distance_p95"]["candidate"])
        reason_suffix = "" if spec.always_check_boundary_fvt else " after FVT change"
        if fvt_f1 is None or fvt_f1 < spec.changed_fvt_f1_min:
            reasons.append(f"fvt_positive_buffered_f1_r2 below 0.90{reason_suffix}")
        if fvt_distance is None or fvt_distance > spec.changed_fvt_distance_max:
            reasons.append(f"fvt_positive_distance_p95 above 2.0{reason_suffix}")
    return {
        "key": comparison["key"],
        "passed": not reasons,
        "fvt_changed": fvt_changed,
        "metrics": {
            "skin_buffered_f1_r2": metrics["skin_buffered_f1_r2"],
            "skin_count": metrics["skin_count"],
            "skin_cell_to_fvt_positive_candidate_ratio": ratio,
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


def _oracle_regressions(
    comparisons: Iterable[dict[str, Any]], spec: GateSpec
) -> list[dict[str, Any]]:
    if not spec.require_unchanged_oracle_metrics:
        return _material_regressions(
            comparisons,
            pipeline="oracle",
            exclude_boundary=False,
            spec=spec,
        )

    regressions = []
    for comparison in comparisons:
        key = comparison["key"]
        if key["pipeline"] != "oracle" or not _shape_matches(key, spec):
            continue
        for metric in METRIC_COLUMNS:
            values = comparison["metrics"][metric]
            baseline = numeric(values["baseline"])
            candidate = numeric(values["candidate"])
            if baseline is None and candidate is None:
                continue
            if baseline is not None and candidate is not None and baseline == candidate:
                continue
            regressions.append(
                {
                    "key": key,
                    "metric": metric,
                    "baseline": values["baseline"],
                    "candidate": values["candidate"],
                    "delta": values["delta"],
                    "threshold": 0,
                }
            )
    return regressions


def _material_regressions(
    comparisons: Iterable[dict[str, Any]],
    *,
    pipeline: str,
    exclude_boundary: bool,
    spec: GateSpec,
) -> list[dict[str, Any]]:
    regressions = []
    for comparison in comparisons:
        key = comparison["key"]
        if pipeline == "scanner":
            if not _is_scanner_gate_row(key, spec):
                continue
        elif key["pipeline"] != pipeline or not _shape_matches(key, spec):
            continue
        if exclude_boundary and key["case_id"] == "boundary_plane":
            continue
        for metric, (direction, threshold) in MATERIAL_REGRESSION_THRESHOLDS.items():
            delta = comparison["metrics"][metric]["delta"]
            if delta is None or direction == "lt" and delta >= threshold:
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


def _has_material_improvement(metrics: dict[str, dict[str, Any]]) -> bool:
    return any(
        metrics[metric]["delta"] is not None and metrics[metric]["delta"] >= 0.02
        for metric in ("skin_buffered_f1_r2", "fvt_positive_buffered_f1_r2")
    ) or any(
        metrics[metric]["delta"] is not None and metrics[metric]["delta"] <= -2.0
        for metric in ("skin_distance_p95", "fvt_positive_distance_p95")
    )


def _false_fallback_replacements(
    comparisons: Iterable[dict[str, Any]], spec: GateSpec
) -> list[dict[str, Any]]:
    replacements = []
    for comparison in comparisons:
        key, metrics = comparison["key"], comparison["metrics"]
        if (
            not _is_scanner_gate_row(key, spec)
            or key["case_id"] in {"boundary_plane", *TOPOLOGY_CASES}
            or metrics["skin_fallback_replaced_primary"]["candidate"] is not True
            or metrics["skin_fallback_replaced_primary"]["baseline"] is True
            or (
                spec.allow_materially_improved_false_fallback and _has_material_improvement(metrics)
            )
        ):
            continue
        replacements.append(
            {
                "key": key,
                "baseline": metrics["skin_fallback_replaced_primary"]["baseline"],
                "candidate": metrics["skin_fallback_replaced_primary"]["candidate"],
            }
        )
    return replacements


def _topology_regressions(
    comparisons: Iterable[dict[str, Any]], spec: GateSpec
) -> list[dict[str, Any]]:
    regressions = []
    for comparison in comparisons:
        key = comparison["key"]
        if not _is_scanner_gate_row(key, spec) or key["case_id"] not in TOPOLOGY_CASES:
            continue
        for metric in ("skin_over_merge_count", "skin_over_split_count"):
            values = comparison["metrics"][metric]
            if values["delta"] is None or values["delta"] <= 0:
                continue
            regressions.append({"key": key, "metric": metric, **values, "threshold": 0})
    return regressions


def add_required_coverage(report: dict[str, Any]) -> dict[str, Any]:
    spec = PROMOTION_GATES.get(report["config"]["promotion_gate"])
    if spec is None:
        return report
    if "coverage" in report["promotion_gate"]:
        return report
    comparisons = report["comparisons"]
    checks = []
    for coverage_spec in spec.coverage:
        matched = {
            item["key"]["case_id"]
            for item in comparisons
            if _matches_coverage(item, coverage_spec.pipeline, spec)
        }
        missing = [case_id for case_id in coverage_spec.case_ids if case_id not in matched]
        checks.append(
            {
                "name": coverage_spec.name,
                "pipeline": coverage_spec.pipeline,
                "scanner_backend": (
                    spec.scanner_backend if coverage_spec.pipeline == "scanner" else None
                ),
                "scanner_refinement_factor": (
                    spec.scanner_refinement_factor if coverage_spec.pipeline == "scanner" else None
                ),
                "required_case_ids": list(coverage_spec.case_ids),
                "matched_case_ids": [x for x in coverage_spec.case_ids if x in matched],
                "missing_case_ids": missing,
                "passed": not missing,
            }
        )
    coverage_reasons = [
        f"missing required gate coverage: {check['name']} {case_id}"
        for check in checks
        for case_id in check["missing_case_ids"]
    ]
    coverage = {"passed": not coverage_reasons, "checks": checks, "reasons": coverage_reasons}
    gate = dict(report["promotion_gate"])
    gate["coverage"] = coverage
    gate["reasons"] = [*gate["reasons"], *coverage_reasons]
    gate["passed"] = not gate["reasons"]
    return {**report, "promotion_gate": gate}


def _matches_coverage(comparison: dict[str, Any], pipeline: str, spec: GateSpec) -> bool:
    key = comparison["key"]
    if (
        key["pipeline"] != pipeline
        or key["workflow_mode"] != "quality"
        or not _shape_matches(key, spec)
    ):
        return False
    return pipeline != "scanner" or _is_scanner_gate_row(key, spec)


def build_promotion_report(
    baseline_summary: Path,
    candidate_summary: Path,
    baseline_variant: str = "current_default",
    candidate_variants: tuple[str, ...] = DEFAULT_CANDIDATES,
    promotion_gate: str = "scanner-boundary",
    strict_missing_rows: bool = False,
    *,
    comparison_profile: str = "variant",
    baseline_metrics: Path | None = None,
    candidate_metrics: Path | None = None,
) -> dict[str, Any]:
    from .comparison import VARIANT_COMPARISON_PROFILE, compare_reports

    reports = []
    for variant in candidate_variants:
        comparison_args = (
            baseline_summary,
            candidate_summary,
            baseline_variant,
            variant,
            promotion_gate,
            strict_missing_rows,
        )
        if comparison_profile == VARIANT_COMPARISON_PROFILE:
            comparison_report = compare_reports(*comparison_args)
        else:
            comparison_report = compare_reports(
                *comparison_args,
                comparison_profile=comparison_profile,
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
            )
        reports.append(add_required_coverage(comparison_report))
    candidates = {
        report["config"]["candidate_variant"]: report["promotion_gate"] for report in reports
    }
    missing_rows = {
        report["config"]["candidate_variant"]: {
            "missing_baseline_rows": report["missing_baseline_rows"],
            "missing_candidate_rows": report["missing_candidate_rows"],
        }
        for report in reports
        if report["missing_baseline_rows"] or report["missing_candidate_rows"]
    }
    reasons = [
        f"{variant}: {reason}" for variant, gate in candidates.items() for reason in gate["reasons"]
    ]
    if strict_missing_rows:
        reasons.extend(f"{variant}: missing matched rows" for variant in sorted(missing_rows))
    config = {
        "baseline_summary": str(baseline_summary),
        "candidate_summary": str(candidate_summary),
        "baseline_variant": baseline_variant,
        "candidate_variants": list(candidate_variants),
        "promotion_gate": promotion_gate,
        "strict_missing_rows": strict_missing_rows,
    }
    report = {
        "format_version": 1,
        "config": config,
        "promotion_gate": {
            "name": promotion_gate,
            "passed": not reasons,
            "promotable_candidates": [x for x, gate in candidates.items() if gate["passed"]],
            "candidates": candidates,
            "missing_rows": missing_rows,
            "reasons": reasons,
        },
        "candidate_reports": reports,
    }
    if comparison_profile == VARIANT_COMPARISON_PROFILE:
        return report

    config.update(
        {
            "comparison_profile": comparison_profile,
            "baseline_metrics": str(baseline_metrics),
            "candidate_metrics": str(candidate_metrics),
        }
    )
    if len(reports) == 1:
        report["scanner_policy_contract"] = reports[0]["scanner_policy_contract"]
    return report
