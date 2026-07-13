#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyosv.evaluation.promotion import build_promotion_report
from pyosv.evaluation.promotion.markdown import promotion_markdown
from pyosv.evaluation.promotion.specifications import DEFAULT_CANDIDATES

COMPARISON_PROFILES = (
    "variant",
    "scanner-thinning-policy-v1",
    "quality-workflow-scanner-thinning-v1",
)
SCANNER_POLICY_PROFILES = frozenset(COMPARISON_PROFILES[1:])
PROMOTION_GATES = ("scanner-boundary", "scanner-boundary-reference-like")


def _csv_list(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("candidate list must not be empty")
    return values


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize scanner-boundary promotion gate results from quality summaries."
    )
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--candidate-summary", required=True, type=Path)
    parser.add_argument("--baseline-variant", default="current_default")
    parser.add_argument("--candidate-variant")
    parser.add_argument("--candidate-variants", type=_csv_list)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--promotion-gate", choices=PROMOTION_GATES, default="scanner-boundary")
    parser.add_argument("--strict-missing-rows", action="store_true")
    parser.add_argument("--fail-on-gate-failure", action="store_true")
    parser.add_argument("--comparison-profile", choices=COMPARISON_PROFILES, default="variant")
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--candidate-metrics", type=Path)
    return parser


def _metrics_paths(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    if args.comparison_profile not in SCANNER_POLICY_PROFILES:
        return args.baseline_metrics, args.candidate_metrics
    return (
        args.baseline_metrics or args.baseline_summary.with_name("metrics.json"),
        args.candidate_metrics or args.candidate_summary.with_name("metrics.json"),
    )


def main() -> int:
    parser = _argument_parser()
    args = parser.parse_args()
    if args.candidate_variant and args.candidate_variants:
        parser.error("use either --candidate-variant or --candidate-variants, not both")
    variants = (
        (args.candidate_variant,)
        if args.candidate_variant
        else args.candidate_variants or DEFAULT_CANDIDATES
    )
    if args.comparison_profile in SCANNER_POLICY_PROFILES:
        if len(variants) != 1:
            parser.error(f"{args.comparison_profile} requires exactly one candidate variant")
        if args.baseline_variant != "current_default":
            parser.error(f"{args.comparison_profile} requires --baseline-variant current_default")
        if variants[0] != "current_default":
            parser.error(f"{args.comparison_profile} requires candidate variant current_default")
    baseline_metrics, candidate_metrics = _metrics_paths(args)
    try:
        positional_args = (
            args.baseline_summary,
            args.candidate_summary,
            args.baseline_variant,
            variants,
            args.promotion_gate,
            args.strict_missing_rows,
        )
        if args.comparison_profile in SCANNER_POLICY_PROFILES:
            report = build_promotion_report(
                *positional_args,
                comparison_profile=args.comparison_profile,
                baseline_metrics=baseline_metrics,
                candidate_metrics=candidate_metrics,
            )
        else:
            report = build_promotion_report(*positional_args)
    except ValueError as exc:
        parser.error(str(exc))
    output_json = args.output_json or args.candidate_summary.with_name("promotion_gate.json")
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_markdown is not None:
        args.output_markdown.write_text(promotion_markdown(report), encoding="utf-8")
    gate = report["promotion_gate"]
    if args.strict_missing_rows and gate["missing_rows"]:
        return 2
    return 2 if args.fail_on_gate_failure and not gate["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
