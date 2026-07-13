"""Stable Markdown formatters for promotion reports."""

from __future__ import annotations

import json
from typing import Any


def _value(value: Any) -> str:
    if value is None:
        return ""
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def comparison_markdown(report: dict[str, Any]) -> str:
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
    contract = report.get("scanner_policy_contract")
    if contract is not None:
        lines.extend(_scanner_policy_contract_lines(contract))
    boundary = gate["boundary_plane"]
    if boundary is not None:
        lines.extend(
            [
                "## Boundary Plane Scanner 49^3",
                "",
                f"Gate result: {'pass' if boundary['passed'] else 'fail'}",
                "",
                "| metric | baseline | candidate | delta |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric, values in boundary["metrics"].items():
            lines.append(
                f"| {metric} | {_value(values['baseline'])} | "
                f"{_value(values['candidate'])} | {_value(values['delta'])} |"
            )
        lines.append("")
    regressions = (
        gate["non_boundary_regressions"] + gate["oracle_regressions"] + gate["topology_regressions"]
    )
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
        for item in regressions:
            key = item["key"]
            lines.append(
                f"| {key['case_id']} | {key['pipeline']} | {item['metric']} | "
                f"{_value(item['baseline'])} | {_value(item['candidate'])} | "
                f"{_value(item['delta'])} |"
            )
    lines.extend(["", "## False Fallback Replacements", ""])
    replacements = gate["false_fallback_replacements"]
    if not replacements:
        lines.append("None.")
    else:
        lines.extend(["| case_id | pipeline |", "|---|---|"])
        for item in replacements:
            lines.append(f"| {item['key']['case_id']} | {item['key']['pipeline']} |")
    lines.append("")
    return "\n".join(lines)


def _list(values: list[str]) -> str:
    return "none" if not values else ", ".join(f"`{value}`" for value in values)


def _boundary_metric(boundary: dict[str, Any] | None, name: str) -> str:
    if boundary is None:
        return ""
    return _value(boundary["metrics"][name]["candidate"])


def _contract_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _scanner_policy_contract_lines(contract: dict[str, Any]) -> list[str]:
    baseline = contract["baseline"]
    candidate = contract["candidate"]
    lines = [
        "## Scanner Policy Contract",
        "",
        f"Contract result: {'pass' if contract['passed'] else 'fail'}",
        "",
        f"- baseline policy ID: `{baseline['policy_id']}`",
        f"- candidate policy ID: `{candidate['policy_id']}`",
        f"- baseline scanner thin mode: `{baseline['scanner_thin_mode']}`",
        f"- candidate scanner thin mode: `{candidate['scanner_thin_mode']}`",
        "- baseline remove_edge_effects: "
        f"requested={_contract_value(baseline['requested_remove_edge_effects'])}, "
        f"effective={_contract_value(baseline['effective_remove_edge_effects'])}",
        "- candidate remove_edge_effects: "
        f"requested={_contract_value(candidate['requested_remove_edge_effects'])}, "
        f"effective={_contract_value(candidate['effective_remove_edge_effects'])}",
        "- allowed config difference paths: "
        + _list(sorted(contract["allowed_config_difference_paths"])),
        "",
        "### Allowed Config Differences",
        "",
    ]
    lines.extend(_config_difference_lines(contract["allowed_config_differences"]))
    lines.extend(["", "### Disallowed Config Differences", ""])
    lines.extend(_config_difference_lines(contract["disallowed_config_differences"]))
    lines.extend(["", "### Contract Failure Reasons", ""])
    lines.extend(
        (f"- {reason}" for reason in contract["reasons"]) if contract["reasons"] else ["None."]
    )
    lines.append("")
    return lines


def _config_difference_lines(differences: list[dict[str, Any]]) -> list[str]:
    if not differences:
        return ["None."]
    lines = [
        "| path | baseline | candidate |",
        "|---|---|---|",
    ]
    for difference in sorted(differences, key=lambda item: item["path"]):
        lines.append(
            f"| `{difference['path']}` | "
            f"`{_contract_value(difference['baseline'])}` | "
            f"`{_contract_value(difference['candidate'])}` |"
        )
    return lines


def promotion_markdown(report: dict[str, Any]) -> str:
    gate = report["promotion_gate"]
    lines = [
        "# Synthetic Quality Promotion Gate",
        "",
        f"- baseline: `{report['config']['baseline_summary']}`",
        f"- candidate: `{report['config']['candidate_summary']}`",
        f"- baseline variant: `{report['config']['baseline_variant']}`",
        f"- promotion gate: `{gate['name']}` {'pass' if gate['passed'] else 'fail'}",
        f"- promotable candidates: {_list(gate['promotable_candidates'])}",
        "",
        "| candidate | gate | boundary skin F1 | skin count | skin/FVT ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for variant, candidate_gate in gate["candidates"].items():
        boundary = candidate_gate["boundary_plane"]
        lines.append(
            f"| {variant} | {'pass' if candidate_gate['passed'] else 'fail'} | "
            f"{_boundary_metric(boundary, 'skin_buffered_f1_r2')} | "
            f"{_boundary_metric(boundary, 'skin_count')} | "
            f"{_boundary_metric(boundary, 'skin_cell_to_fvt_positive_candidate_ratio')} |"
        )
    contract = report.get("scanner_policy_contract")
    if contract is not None:
        lines.append("")
        lines.extend(_scanner_policy_contract_lines(contract))
    lines.extend(["", "## Reasons", ""])
    lines.extend((f"- {reason}" for reason in gate["reasons"]) if gate["reasons"] else ["None."])
    lines.extend(["", "## Coverage", ""])
    for variant, candidate_gate in gate["candidates"].items():
        coverage = candidate_gate.get("coverage")
        if coverage is None:
            continue
        lines.extend([f"### {variant}", ""])
        for check in coverage["checks"]:
            status = "pass" if check["passed"] else "fail"
            lines.append(
                f"- `{check['name']}`: {status}; missing: {_list(check['missing_case_ids'])}"
            )
        lines.append("")
    lines.append("")
    return "\n".join(lines)
