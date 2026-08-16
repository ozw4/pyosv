"""Command-line entry point for the fixed Q-QUAL 3D workflow."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pyosv.qqual3d import run_qqual3d
from pyosv.qqual3d.io import (
    load_qqual3d_input,
    require_new_output_directory,
    write_qqual3d_output_bundle,
)


def _parse_shape(value: str) -> tuple[int, int, int]:
    try:
        parts = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be n3,n2,n1") from error
    if len(parts) != 3 or any(size <= 0 for size in parts):
        raise argparse.ArgumentTypeError("shape must be three positive integers: n3,n2,n1")
    return parts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fixed Q-QUAL workflow on a 3D DAT volume."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Input C-order big-endian float32 DAT."
    )
    parser.add_argument(
        "--shape", type=_parse_shape, required=True, help="Input shape as n3,n2,n1."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print run.json.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        require_new_output_directory(args.output_dir)
        source = load_qqual3d_input(args.input, args.shape)
        result = run_qqual3d(source.array)
        write_qqual3d_output_bundle(
            args.output_dir,
            source=source,
            result=result,
            pretty=args.pretty,
        )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
