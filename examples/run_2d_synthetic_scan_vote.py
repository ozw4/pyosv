"""Run 2D orientation scanning and voting on a synthetic lineament."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run pyosv 2D orientation scanning and optimal-path voting on a "
            "synthetic Gaussian lineament."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for generated DAT outputs.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=25.0,
        help="Synthetic lineament orientation in degrees.",
    )
    parser.add_argument(
        "--theta-min",
        type=float,
        default=-75.0,
        help="Minimum scanner orientation in degrees.",
    )
    parser.add_argument(
        "--theta-max",
        type=float,
        default=75.0,
        help="Maximum scanner orientation in degrees.",
    )
    parser.add_argument(
        "--sigma1",
        type=float,
        default=2.0,
        help="Scanner smoothing parameter.",
    )
    parser.add_argument(
        "--ru",
        type=int,
        default=2,
        help="Voting half-width in the local normal direction.",
    )
    parser.add_argument(
        "--rv",
        type=int,
        default=5,
        help="Voting half-width in the local strike direction.",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=3,
        help="Seed exclusion distance in samples.",
    )
    parser.add_argument(
        "--fm",
        type=float,
        default=0.45,
        help="Minimum scanner likelihood for seed picking.",
    )
    parser.add_argument(
        "--no-thin",
        action="store_true",
        help="Skip voting thinning and omit fvt_py.dat.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_paths, summary = run_example(
        output_dir=args.output_dir,
        theta=args.theta,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        sigma1=args.sigma1,
        ru=args.ru,
        rv=args.rv,
        d=args.d,
        fm=args.fm,
        thin=not args.no_thin,
    )

    print(
        " ".join(
            [
                f"ft_max={summary['ft_max']:.6g}",
                f"fv_max={summary['fv_max']:.6g}",
                f"fv_nonzero={summary['fv_nonzero']}",
            ],
        ),
    )
    for path in output_paths:
        print(path)
    return 0


def run_example(
    *,
    output_dir: Path | None = None,
    theta: float = 25.0,
    theta_min: float = -75.0,
    theta_max: float = 75.0,
    sigma1: float = 2.0,
    ru: int = 2,
    rv: int = 5,
    d: int = 3,
    fm: float = 0.45,
    thin: bool = True,
) -> tuple[list[Path], dict[str, float | int]]:
    from pyosv.io import write_dat
    from pyosv.orient2d import FaultOrientScanner2
    from pyosv.voting2d import OptimalPathVoter

    image = _synthetic_lineament(theta)

    scanner = FaultOrientScanner2(sigma1)
    ft, pt = scanner.scan(theta_min, theta_max, image)

    voter = OptimalPathVoter(ru, rv)
    voter.set_attribute_smoothing(0)
    voter.set_path_smoothing(0.0)
    fv, w1, w2 = voter.apply_voting(d=d, fm=fm, ft=ft, pt=pt)
    fvt = voter.thin(fv, w1, w2) if thin else None

    summary: dict[str, float | int] = {
        "ft_max": float(np.max(ft)),
        "fv_max": float(np.max(fv)),
        "fv_nonzero": int(np.count_nonzero(fv)),
    }

    output_paths: list[Path] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths.extend(
            [
                write_dat(output_dir / "g_py.dat", image),
                write_dat(output_dir / "ft_py.dat", ft),
                write_dat(output_dir / "pt_py.dat", pt),
                write_dat(output_dir / "fv_py.dat", fv),
            ],
        )
        if fvt is not None:
            output_paths.append(write_dat(output_dir / "fvt_py.dat", fvt))

    return output_paths, summary


def _synthetic_lineament(theta_degrees: float) -> np.ndarray:
    n2, n1 = 48, 64
    x2, x1 = np.mgrid[:n2, :n1].astype(np.float32)
    x1 -= np.float32((n1 - 1) / 2.0)
    x2 -= np.float32((n2 - 1) / 2.0)

    theta_radians = np.deg2rad(theta_degrees)
    distance = x2 * np.cos(theta_radians) - x1 * np.sin(theta_radians)
    image = np.exp(-0.5 * (distance / np.float32(1.2)) ** 2)
    return image.astype(np.float32)


if __name__ == "__main__":
    raise SystemExit(main())
