"""CLI for the fixed derived mode-comparison publication bundle."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pyosv.evaluation.mode_comparison_publication import (
    generate_publication_bundle,
    validate_publication_bundle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a derived publication report from completed synthetic and F3 "
            "mode-comparison bundles."
        )
    )
    parser.add_argument("--synthetic-bundle", type=Path, help="Completed synthetic source bundle.")
    parser.add_argument("--f3-bundle", type=Path, help="Completed F3 source bundle.")
    parser.add_argument("--f3-data-root", type=Path, help="Checksummed external F3 data root.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New publication output directory."
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print publication JSON files."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate only an existing publication bundle; do not access sources or matplotlib.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.validate_only:
            if not validate_publication_bundle(args.output_dir):
                raise ValueError("publication bundle validation failed")
        else:
            missing = [
                name
                for name, value in (
                    ("--synthetic-bundle", args.synthetic_bundle),
                    ("--f3-bundle", args.f3_bundle),
                    ("--f3-data-root", args.f3_data_root),
                )
                if value is None
            ]
            if missing:
                raise ValueError("normal generation requires " + ", ".join(missing))
            generate_publication_bundle(
                args.synthetic_bundle,
                args.f3_bundle,
                args.f3_data_root,
                args.output_dir,
                pretty=args.pretty,
            )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
