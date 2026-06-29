"""Run 3D orientation scanning and voting on a synthetic planar fault."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run pyosv 3D orientation scanning and optimal-surface voting on a "
            "small synthetic Gaussian planar fault."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for generated DAT outputs.",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=0.0,
        help="Synthetic fault strike in degrees.",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=90.0,
        help="Synthetic fault dip in degrees.",
    )
    parser.add_argument(
        "--phi-min",
        type=float,
        default=0.0,
        help="Minimum scanner strike in degrees.",
    )
    parser.add_argument(
        "--phi-max",
        type=float,
        default=90.0,
        help="Maximum scanner strike in degrees.",
    )
    parser.add_argument(
        "--theta-min",
        type=float,
        default=45.0,
        help="Minimum scanner dip in degrees.",
    )
    parser.add_argument(
        "--theta-max",
        type=float,
        default=90.0,
        help="Maximum scanner dip in degrees.",
    )
    parser.add_argument(
        "--sigma1",
        type=float,
        default=2.0,
        help="Scanner dip smoothing/sampling control.",
    )
    parser.add_argument(
        "--sigma2",
        type=float,
        default=2.0,
        help="Scanner strike smoothing/sampling control.",
    )
    parser.add_argument(
        "--ru",
        type=int,
        default=1,
        help="Voting half-width in the local normal direction.",
    )
    parser.add_argument(
        "--rv",
        type=int,
        default=2,
        help="Voting half-width in the local dip direction.",
    )
    parser.add_argument(
        "--rw",
        type=int,
        default=2,
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
        default=0.5,
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
        phi=args.phi,
        theta=args.theta,
        phi_min=args.phi_min,
        phi_max=args.phi_max,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        sigma1=args.sigma1,
        sigma2=args.sigma2,
        ru=args.ru,
        rv=args.rv,
        rw=args.rw,
        d=args.d,
        fm=args.fm,
        thin=not args.no_thin,
    )

    print(
        " ".join(
            [
                f"ft_max={summary['ft_max']:.6g}",
                f"fv_max={summary['fv_max']:.6g}",
                f"fvt_max={summary['fvt_max']:.6g}",
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
    phi: float = 0.0,
    theta: float = 90.0,
    phi_min: float = 0.0,
    phi_max: float = 90.0,
    theta_min: float = 45.0,
    theta_max: float = 90.0,
    sigma1: float = 2.0,
    sigma2: float = 2.0,
    ru: int = 1,
    rv: int = 2,
    rw: int = 2,
    d: int = 3,
    fm: float = 0.5,
    thin: bool = True,
) -> tuple[list[Path], dict[str, float | int]]:
    from pyosv.io import write_dat
    from pyosv.orient3d import FaultOrientScanner3
    from pyosv.voting3d import OptimalSurfaceVoter

    image = _synthetic_planar_fault(phi, theta)

    scanner = FaultOrientScanner3(sigma1=sigma1, sigma2=sigma2)
    ft, pt, tt = scanner.scan(phi_min, phi_max, theta_min, theta_max, image)

    voter = OptimalSurfaceVoter(ru=ru, rv=rv, rw=rw)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    fv, vp, vt = voter.apply_voting(d=d, fm=fm, ft=ft, pt=pt, tt=tt)
    fvt = voter.thin(fv, vp, vt) if thin else None

    summary: dict[str, float | int] = {
        "ft_max": float(np.max(ft)),
        "fv_max": float(np.max(fv)),
        "fvt_max": float(np.max(fvt)) if fvt is not None else 0.0,
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
                write_dat(output_dir / "tt_py.dat", tt),
                write_dat(output_dir / "fv_py.dat", fv),
                write_dat(output_dir / "vp_py.dat", vp),
                write_dat(output_dir / "vt_py.dat", vt),
            ],
        )
        if fvt is not None:
            output_paths.append(write_dat(output_dir / "fvt_py.dat", fvt))

    return output_paths, summary


def _synthetic_planar_fault(phi_degrees: float, theta_degrees: float) -> np.ndarray:
    from pyosv.geometry import fault_normal_vector_from_strike_and_dip

    n3, n2, n1 = 17, 17, 17
    i3, i2, i1 = np.indices((n3, n2, n1), dtype=np.float32)
    center1 = np.float32(0.5 * (n1 - 1))
    center2 = np.float32(0.5 * (n2 - 1))
    center3 = np.float32(0.5 * (n3 - 1))
    w1, w2, w3 = fault_normal_vector_from_strike_and_dip(
        phi_degrees,
        theta_degrees,
    )
    distance = w1 * (i1 - center1) + w2 * (i2 - center2) + w3 * (i3 - center3)
    image = np.exp(-0.5 * (distance / np.float32(1.0)) ** 2)
    return image.astype(np.float32, copy=False)


if __name__ == "__main__":
    raise SystemExit(main())
