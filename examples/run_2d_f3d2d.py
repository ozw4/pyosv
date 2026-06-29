"""Run the 2D voting workflow on the reference f3d2d dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_2d_reference import run_example as run_reference_example


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run pyosv 2D optimal-path voting on reference_osv/data/2d/f3d2d "
            "and write DAT outputs outside the reference tree."
        ),
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=None,
        help=(
            "Path to the reference_osv root. Defaults to PYOSV_REFERENCE_OSV, then ./reference_osv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for generated DAT outputs such as fv_py.dat and fvt_py.dat.",
    )
    parser.add_argument(
        "--ru",
        type=int,
        default=15,
        help="Voting half-width in the local normal direction.",
    )
    parser.add_argument(
        "--rv",
        type=int,
        default=30,
        help="Voting half-width in the local strike direction.",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=4,
        help="Seed exclusion distance in samples.",
    )
    parser.add_argument(
        "--fm",
        type=float,
        default=0.3,
        help="Minimum fault-likelihood value for seed picking.",
    )
    parser.add_argument(
        "--strain-max",
        type=float,
        default=0.25,
        help="Maximum fault-curve strain.",
    )
    parser.add_argument(
        "--path-smoothing",
        type=float,
        default=2.0,
        help="Smoothing extent used for extracted fault paths.",
    )
    parser.add_argument(
        "--no-thin",
        action="store_true",
        help="Skip the thinning step and only write fv_py.dat.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        output_paths = run_example(
            reference_root_arg=args.reference_root,
            output_dir=args.output_dir,
            ru=args.ru,
            rv=args.rv,
            d=args.d,
            fm=args.fm,
            strain_max=args.strain_max,
            path_smoothing=args.path_smoothing,
            thin=not args.no_thin,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        import sys

        print(f"error: {error}", file=sys.stderr)
        return 1

    for path in output_paths:
        print(path)
    return 0


def run_example(
    *,
    reference_root_arg: Path | None,
    output_dir: Path,
    ru: int = 15,
    rv: int = 30,
    d: int = 4,
    fm: float = 0.3,
    strain_max: float = 0.25,
    path_smoothing: float = 2.0,
    thin: bool = True,
) -> list[Path]:
    return run_reference_example(
        dataset_name="f3d2d",
        reference_root_arg=reference_root_arg,
        output_dir=output_dir,
        ru=ru,
        rv=rv,
        d=d,
        fm=fm,
        strain_max=strain_max,
        path_smoothing=path_smoothing,
        thin=thin,
    )


if __name__ == "__main__":
    raise SystemExit(main())
