#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyosv.evaluation.promotion import compare_reports
from pyosv.evaluation.promotion.markdown import comparison_markdown


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and candidate synthetic quality summary.csv reports."
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
    report = compare_reports(
        args.baseline_summary,
        args.candidate_summary,
        args.baseline_variant,
        args.candidate_variant or args.baseline_variant,
        args.promotion_gate,
        args.strict_missing_rows,
    )
    output_json = args.output_json or args.candidate_summary.with_name("quality_delta.json")
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_markdown is not None:
        args.output_markdown.write_text(comparison_markdown(report), encoding="utf-8")
    missing = report["missing_baseline_rows"] or report["missing_candidate_rows"]
    if args.strict_missing_rows and missing:
        return 2
    if args.fail_on_gate_failure and not report["promotion_gate"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
