"""CLI for the F3-only compact publication bundle."""

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
            "Generate or validate the fixed F3 PUBLIC-REF versus Q-QUAL compact publication bundle."
        )
    )
    parser.add_argument("--f3-bundle", type=Path, help="Completed F3 source bundle.")
    parser.add_argument("--f3-data-root", type=Path, help="Validated external F3 data root.")
    parser.add_argument(
        "--environment-lock",
        type=Path,
        help="Existing environment lock copied into the generated bundle.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New or existing compact bundle directory."
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print publication_manifest.json."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing bundle using only its recorded files.",
    )
    return parser


def _generation_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    values = (
        ("--f3-bundle", args.f3_bundle),
        ("--f3-data-root", args.f3_data_root),
        ("--environment-lock", args.environment_lock),
    )
    missing = [name for name, value in values if value is None]
    if missing:
        raise ValueError("normal generation requires " + ", ".join(missing))
    f3_bundle = args.f3_bundle
    f3_data_root = args.f3_data_root
    environment_lock = args.environment_lock
    assert f3_bundle is not None
    assert f3_data_root is not None
    assert environment_lock is not None
    return f3_bundle, f3_data_root, environment_lock


def _require_validate_only_arguments(args: argparse.Namespace) -> None:
    invalid = [
        name
        for name, value in (
            ("--f3-bundle", args.f3_bundle),
            ("--f3-data-root", args.f3_data_root),
            ("--environment-lock", args.environment_lock),
        )
        if value is not None
    ]
    if args.pretty:
        invalid.append("--pretty")
    if invalid:
        raise ValueError("--validate-only cannot be combined with " + ", ".join(invalid))


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
        raise ValueError("compact generation requires environment controls: " + ", ".join(missing))
    return {name: os.environ[name] for name in _ENVIRONMENT_CONTROL_NAMES}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.validate_only:
            _require_validate_only_arguments(args)
            from pyosv.evaluation.f3_compact_publication import (
                validate_f3_compact_publication_bundle,
            )

            validate_f3_compact_publication_bundle(args.output_dir)
        else:
            f3_bundle, f3_data_root, environment_lock = _generation_inputs(args)
            code = _collect_code_identity()
            controls = _collect_environment_controls()
            from pyosv.evaluation.f3_compact_publication import (
                generate_f3_compact_publication_bundle,
            )

            generate_f3_compact_publication_bundle(
                f3_bundle,
                f3_data_root,
                args.output_dir,
                environment_lock=environment_lock,
                code=code,
                environment_controls=controls,
                pretty=args.pretty,
            )
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
