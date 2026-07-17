"""Command-line entry point for canonical synthetic mode comparisons."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Sequence
from os import PathLike
from pathlib import Path

from pyosv.evaluation.synthetic_mode_comparison import (
    SyntheticModeComparisonConfig,
    run_mode_comparison,
    validate_completed_bundle,
    validate_trial_seeds,
    write_artifact_bundle,
)
from pyosv.evaluation.synthetic_quality import SyntheticSkinningConfig
from pyosv.evaluation.synthetic_quality.cases import CASE_SETS, validate_case_ids

from .synthetic_quality import DEFAULT_SHAPE, parse_shape3

DEFAULT_TRIAL_SEEDS = (20260707,)


class _ArgumentParser(argparse.ArgumentParser):
    """Apply the case-set default after mutual-exclusion checks."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        if parsed.case_set is None and parsed.case_ids is None:
            parsed.case_set = "minimal"
        return parsed


def parse_case_ids(text: str) -> tuple[str, ...]:
    """Parse and validate an ordered comma-separated case-ID list."""

    parts = tuple(part.strip() for part in text.split(","))
    if any(not part for part in parts):
        raise argparse.ArgumentTypeError("case IDs must not contain empty elements")
    try:
        return validate_case_ids(parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_trial_seeds(text: str) -> tuple[int, ...]:
    """Parse a non-empty comma-separated list of unique non-negative seeds."""

    parts = tuple(part.strip() for part in text.split(","))
    if any(not part for part in parts):
        raise argparse.ArgumentTypeError("trial seeds must not contain empty elements")
    try:
        seeds = tuple(int(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "trial seeds must be comma-separated non-negative integers"
        ) from error
    try:
        return validate_trial_seeds(seeds)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    """Build the canonical synthetic mode-comparison argument parser."""

    parser = _ArgumentParser(
        description="Run a canonical 3D synthetic scanner/workflow mode comparison."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory where the atomic comparison bundle is written.",
    )
    cases = parser.add_mutually_exclusive_group()
    cases.add_argument(
        "--case-set",
        choices=tuple(CASE_SETS),
        default=None,
        help="Registered synthetic case set (default: minimal).",
    )
    cases.add_argument(
        "--case-ids",
        type=parse_case_ids,
        help="Ordered comma-separated registered case IDs.",
    )
    parser.add_argument(
        "--shape",
        type=parse_shape3,
        default=DEFAULT_SHAPE,
        help="Synthetic volume shape in n3,n2,n1 order.",
    )
    parser.add_argument(
        "--trial-seeds",
        type=parse_trial_seeds,
        default=DEFAULT_TRIAL_SEEDS,
        help="Comma-separated non-negative stochastic-case seeds.",
    )
    parser.add_argument(
        "--no-oracle-workflow-isolation",
        action="store_true",
        help="Omit the ORACLE-REF and ORACLE-QUAL isolation cells.",
    )
    parser.add_argument(
        "--skip-skinning",
        action="store_true",
        help="Disable skinning while retaining the canonical comparison plan.",
    )
    parser.add_argument("--pretty", action="store_true", help="Write indented JSON files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one comparison, write and validate its bundle, and report its path."""

    parser = build_parser()
    args = parser.parse_args(argv)
    written_path: Path | None = None
    try:
        if os.path.lexists(args.output_dir):
            raise FileExistsError(f"artifact output already exists: {args.output_dir}")
        config = SyntheticModeComparisonConfig(
            case_set=args.case_set,
            case_ids=args.case_ids,
            trial_seeds=args.trial_seeds,
            shape=args.shape,
            include_oracle_workflow_isolation=not args.no_oracle_workflow_isolation,
            skinning_config=SyntheticSkinningConfig(enabled=not args.skip_skinning),
        )
        result = run_mode_comparison(config)
        written_path = write_artifact_bundle(
            result,
            args.output_dir,
            config=config,
            pretty=args.pretty,
        )
        if not validate_completed_bundle(written_path):
            raise ValueError("artifact bundle validation failed")
    except Exception as error:
        cleanup_error: OSError | None = None
        if written_path is not None:
            try:
                _remove_failed_bundle(written_path)
            except OSError as caught:
                cleanup_error = caught
        print(f"error: {error}", file=sys.stderr)
        if cleanup_error is not None:
            print(
                f"error: failed to remove invalid artifact bundle: {cleanup_error}",
                file=sys.stderr,
            )
        return 1

    print(written_path)
    return 0


def _remove_failed_bundle(path: str | PathLike[str]) -> None:
    """Remove a just-written bundle when post-write validation fails."""

    bundle = Path(path)
    if bundle.is_symlink() or not bundle.is_dir():
        bundle.unlink(missing_ok=True)
        return

    completion_error: OSError | None = None
    try:
        (bundle / "completion.json").unlink(missing_ok=True)
    except OSError as error:
        completion_error = error
    try:
        shutil.rmtree(bundle)
    except OSError as error:
        if completion_error is not None:
            raise OSError(
                f"{error}; completion marker removal also failed: {completion_error}"
            ) from error
        raise


if __name__ == "__main__":
    raise SystemExit(main())
