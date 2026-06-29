"""Run the 2D voting workflow on the reference f3d2d dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
            thin=not args.no_thin,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for path in output_paths:
        print(path)
    return 0


def run_example(
    *,
    reference_root_arg: Path | None,
    output_dir: Path,
    thin: bool,
) -> list[Path]:
    from pyosv.io import read_dat, write_dat
    from pyosv.reference import REFERENCE_DATASETS_2D, reference_root, resolve_reference_file
    from pyosv.voting2d import OptimalPathVoter

    dataset = REFERENCE_DATASETS_2D["f3d2d"]
    root = reference_root(reference_root_arg)
    root_resolved = root.resolve(strict=False)
    output_dir_resolved = output_dir.resolve(strict=False)
    if output_dir_resolved == root_resolved or output_dir_resolved.is_relative_to(root_resolved):
        raise ValueError(f"--output-dir must not be inside reference root: {root}")

    ft_path = _require_file(resolve_reference_file(dataset, "ft.dat", root=root))
    pt_path = _require_file(resolve_reference_file(dataset, "pt.dat", root=root))

    ft = read_dat(ft_path, dataset.shape, endian=dataset.endian)
    pt = read_dat(pt_path, dataset.shape, endian=dataset.endian)
    _require_finite("ft.dat", ft)
    _require_finite("pt.dat", pt)

    voter = OptimalPathVoter(15, 30)
    voter.set_strain_max(0.25)
    voter.set_path_smoothing(2)

    fv, w1, w2 = voter.apply_voting(d=4, fm=0.3, ft=ft, pt=pt)
    _require_finite("fv_py.dat", fv)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not output_dir.is_dir():
        raise NotADirectoryError(f"--output-dir is not a directory: {output_dir}")

    output_paths = [
        write_dat(output_dir / "fv_py.dat", fv, endian=dataset.endian),
    ]

    if thin:
        fvt = voter.thin(fv, w1, w2)
        _require_finite("fvt_py.dat", fvt)
        output_paths.append(write_dat(output_dir / "fvt_py.dat", fvt, endian=dataset.endian))

    return output_paths


def _require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"reference file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"reference path is not a file: {path}")
    return path


def _require_finite(name: str, array: object) -> None:
    import numpy as np

    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")


if __name__ == "__main__":
    raise SystemExit(main())
