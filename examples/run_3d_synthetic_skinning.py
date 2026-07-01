"""Run 3D voting, thinning, and minimal skinning on a synthetic fault plane."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run pyosv 3D optimal-surface voting, thinning, and reference-like "
            "skinning on a small synthetic planar fault."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for generated DAT outputs and skin summary.",
    )
    parser.add_argument(
        "--min-likelihood",
        type=float,
        default=0.7,
        help="Minimum thinned vote likelihood for skin cell extraction.",
    )
    parser.add_argument(
        "--min-skin-size",
        type=int,
        default=20,
        help="Minimum skin size kept after reference-like growth.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_paths, summary = run_example(
        output_dir=args.output_dir,
        min_likelihood=args.min_likelihood,
        min_skin_size=args.min_skin_size,
    )

    print(
        " ".join(
            [
                f"fv_max={summary['fv_max']:.6g}",
                f"fvt_max={summary['fvt_max']:.6g}",
                f"skin_count={summary['skin_count']}",
                f"largest_skin_size={summary['largest_skin_size']}",
            ],
        ),
    )
    for path in output_paths:
        print(path)
    return 0


def run_example(
    *,
    output_dir: Path | None = None,
    min_likelihood: float = 0.7,
    min_skin_size: int = 20,
) -> tuple[list[Path], dict[str, float | int]]:
    from pyosv.io import write_dat
    from pyosv.skinner import FaultSkinner
    from pyosv.voting3d import OptimalSurfaceVoter

    ft, pt, tt = _synthetic_plane_attributes()

    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)
    fv, vp, vt = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)
    fvt = voter.thin(fv, vp, vt)

    skinner = FaultSkinner(
        method="reference",
        min_likelihood=min_likelihood,
        min_skin_size=min_skin_size,
        connectivity="corner",
    )
    skins = skinner.find_skins(fvt, vp, vt, ep=fvt, ft=fvt, pt=vp, tt=vt)
    largest_skin_size = max((len(skin) for skin in skins), default=0)

    summary: dict[str, float | int] = {
        "fv_max": float(np.max(fv)),
        "fvt_max": float(np.max(fvt)),
        "skin_count": len(skins),
        "largest_skin_size": largest_skin_size,
    }

    output_paths: list[Path] = []
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths.extend(
            [
                write_dat(output_dir / "ft_py.dat", ft),
                write_dat(output_dir / "pt_py.dat", pt),
                write_dat(output_dir / "tt_py.dat", tt),
                write_dat(output_dir / "fv_py.dat", fv),
                write_dat(output_dir / "vp_py.dat", vp),
                write_dat(output_dir / "vt_py.dat", vt),
                write_dat(output_dir / "fvt_py.dat", fvt),
            ],
        )
        skin_summary_path = output_dir / "skins.txt"
        skin_summary_path.write_text(_format_skin_summary(skins), encoding="utf-8")
        output_paths.append(skin_summary_path)

    return output_paths, summary


def _synthetic_plane_attributes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ft = np.zeros((11, 11, 11), dtype=np.float32)
    pt = np.zeros_like(ft)
    tt = np.full_like(ft, 90.0)
    ft[3:8, 5, 3:8] = 0.9
    return ft, pt, tt


def _format_skin_summary(skins: list[object]) -> str:
    lines = ["skin_index,size,first_i1,first_i2,first_i3"]
    for index, skin in enumerate(skins):
        first = skin.cells[0].index if len(skin) else (-1, -1, -1)
        lines.append(f"{index},{len(skin)},{first[0]},{first[1]},{first[2]}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
