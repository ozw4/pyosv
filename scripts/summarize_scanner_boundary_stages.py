#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyosv.evaluation.synthetic_quality.boundary_stage_summary import (
    scanner_boundary_stage_summary_markdown,
    select_scanner_boundary_stage_diagnostics,
    summarize_scanner_boundary_stages,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize scanner boundary-stage diagnostics from metrics.json."
    )
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--case-id", default="boundary_plane")
    parser.add_argument("--variant", default="current_default")
    parser.add_argument("--retention-threshold", type=float, default=0.80)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = json.loads(args.metrics.read_text(encoding="utf-8"))
        diagnostic = select_scanner_boundary_stage_diagnostics(
            report, case_id=args.case_id, variant=args.variant
        )
        summary = summarize_scanner_boundary_stages(
            diagnostic, retention_threshold=args.retention_threshold
        )
        markdown = scanner_boundary_stage_summary_markdown(
            case_id=args.case_id, variant=args.variant, summary=summary
        )
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if args.output_markdown is not None:
            args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
            args.output_markdown.write_text(markdown, encoding="utf-8")
        if args.output_json is None and args.output_markdown is None:
            sys.stdout.write(markdown)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
