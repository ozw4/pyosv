"""CLI for the fixed derived mode-comparison publication bundle."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_REPOSITORY = "ozw4/pyosv"
_ENVIRONMENT_CONTROL_NAMES = (
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_DISABLE_JIT",
    "NUMBA_NUM_THREADS",
    "PYOSV_ACCEL",
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
        "--publication-contract",
        choices=("v1", "legacy"),
        default="v1",
        help="Publication bundle contract (default: v1).",
    )
    parser.add_argument(
        "--environment-lock",
        type=Path,
        help="Environment lock file required for v1 generation.",
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


def _required_generation_sources(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    values = (
        ("--synthetic-bundle", args.synthetic_bundle),
        ("--f3-bundle", args.f3_bundle),
        ("--f3-data-root", args.f3_data_root),
    )
    missing = [name for name, value in values if value is None]
    if missing:
        raise ValueError("normal generation requires " + ", ".join(missing))
    synthetic_bundle = args.synthetic_bundle
    f3_bundle = args.f3_bundle
    f3_data_root = args.f3_data_root
    assert synthetic_bundle is not None
    assert f3_bundle is not None
    assert f3_data_root is not None
    return synthetic_bundle, f3_bundle, f3_data_root


def _collect_code_identity() -> dict[str, object]:
    import subprocess

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("unable to collect Git code identity") from error
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("git rev-parse HEAD did not return a lowercase 40-character commit")
    return {"repository": _REPOSITORY, "git_commit": commit, "dirty": bool(status)}


def _collect_environment_controls() -> dict[str, str]:
    import os

    missing = [name for name in _ENVIRONMENT_CONTROL_NAMES if not os.environ.get(name)]
    if missing:
        raise ValueError("v1 generation requires environment controls: " + ", ".join(missing))
    return {name: os.environ[name] for name in _ENVIRONMENT_CONTROL_NAMES}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.publication_contract == "v1":
            if args.validate_only:
                from pyosv.evaluation.publication_manifest_io import (
                    validate_publication_directory,
                )

                validate_publication_directory(args.output_dir)
            else:
                sources = _required_generation_sources(args)
                if args.environment_lock is None:
                    raise ValueError("v1 generation requires --environment-lock")
                code = _collect_code_identity()
                controls = _collect_environment_controls()
                from pyosv.evaluation.mode_comparison_publication.v1_bundle import (
                    generate_publication_bundle_v1,
                )

                generate_publication_bundle_v1(
                    *sources,
                    args.output_dir,
                    environment_lock=args.environment_lock,
                    code=code,
                    environment_controls=controls,
                    pretty=args.pretty,
                )
        else:
            if args.environment_lock is not None:
                raise ValueError("--environment-lock is only valid with --publication-contract v1")
            from pyosv.evaluation.mode_comparison_publication import (
                generate_legacy_publication_bundle,
                validate_legacy_publication_bundle,
            )

            if args.validate_only:
                if not validate_legacy_publication_bundle(args.output_dir):
                    raise ValueError("publication bundle validation failed")
            else:
                sources = _required_generation_sources(args)
                generate_legacy_publication_bundle(
                    *sources,
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
