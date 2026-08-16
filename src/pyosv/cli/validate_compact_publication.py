"""Command-line validation for a compact publication directory."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pyosv.compact_publication_validation import validate_compact_publication


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an extracted F3 compact publication directory."
    )
    parser.add_argument("root", type=Path, help="Extracted compact publication directory.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_compact_publication(args.root)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
