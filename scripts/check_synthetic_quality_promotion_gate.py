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


def _csv_list(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("candidate list must not be empty")
    return values


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--promotion-gate", choices=("scanner-boundary",), default="scanner-boundary"
    )
    parser.add_argument("--strict-missing-rows", action="store_true")
    parser.add_argument("--fail-on-gate-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.candidate_variant and args.candidate_variants:
        raise SystemExit("use either --candidate-variant or --candidate-variants, not both")
    variants = (
        (args.candidate_variant,)
        if args.candidate_variant
        else args.candidate_variants or DEFAULT_CANDIDATES
    )
    report = build_promotion_report(
        args.baseline_summary,
        args.candidate_summary,
        args.baseline_variant,
        variants,
        args.promotion_gate,
        args.strict_missing_rows,
    )
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
