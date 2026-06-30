"""Validate the practical 3D F3 crop scan/vote workflow against reference crops."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from pyosv.f3d_reference import (
    F3D_ENV_VAR,
    crop_slices,
    interior_mask,
    parse_shape3,
    pick_reference_centers,
    read_f3d_file,
    resolve_f3d_data_root,
)
from pyosv.metrics import finite_value_report, normalized_correlation, top_percentile_overlap

NONZERO_EPSILON = 1.0e-6
OVERLAP_PERCENTILES = (95.0, 99.0, 99.5)
VOLUME_NAMES = (
    "ft_py.dat",
    "pt_py.dat",
    "tt_py.dat",
    "fet_py.dat",
    "fpt_py.dat",
    "ftt_py.dat",
    "fv_py.dat",
    "fvt_py.dat",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pyosv 3D F3 crop scan/vote workflow and report practical "
            "metrics against reference fv.dat and fvt.dat crops."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Path to the F3 reference data root. Defaults to {F3D_ENV_VAR}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for metrics.json and, with --save-volumes, crop DAT outputs.",
    )
    parser.add_argument(
        "--save-volumes",
        action="store_true",
        help="Write crop-level pyosv DAT volumes under --output-dir.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write indented JSON.",
    )
    parser.add_argument(
        "--crop-shape",
        type=parse_shape3,
        default=(64, 64, 64),
        help="Crop shape in n3,n2,n1 order.",
    )
    parser.add_argument("--max-crops", type=int, default=1, help="Maximum number of crops.")
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.9,
        help="Reference fv percentile used to pick crop centers.",
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=48.0,
        help="Minimum center separation in samples.",
    )
    parser.add_argument("--sigma1", type=float, default=8.0, help="Scanner sigma1.")
    parser.add_argument("--sigma2", type=float, default=8.0, help="Scanner sigma2.")
    parser.add_argument("--phi-min", type=float, default=0.0, help="Minimum strike angle.")
    parser.add_argument("--phi-max", type=float, default=360.0, help="Maximum strike angle.")
    parser.add_argument("--theta-min", type=float, default=65.0, help="Minimum dip angle.")
    parser.add_argument("--theta-max", type=float, default=80.0, help="Maximum dip angle.")
    parser.add_argument("--ru", type=int, default=10, help="Voting normal half-width.")
    parser.add_argument("--rv", type=int, default=20, help="Voting dip half-width.")
    parser.add_argument("--rw", type=int, default=30, help="Voting strike half-width.")
    parser.add_argument(
        "--strain-max1",
        type=float,
        default=0.25,
        help="Maximum surface strain in the first voting dimension.",
    )
    parser.add_argument(
        "--strain-max2",
        type=float,
        default=0.25,
        help="Maximum surface strain in the second voting dimension.",
    )
    parser.add_argument(
        "--surface-smoothing1",
        type=float,
        default=2.0,
        help="Surface smoothing in the first voting dimension.",
    )
    parser.add_argument(
        "--surface-smoothing2",
        type=float,
        default=2.0,
        help="Surface smoothing in the second voting dimension.",
    )
    parser.add_argument("--d", type=int, default=4, help="Seed exclusion distance.")
    parser.add_argument("--fm", type=float, default=0.3, help="Minimum seed likelihood.")
    parser.add_argument(
        "--interior-margin",
        type=int,
        default=16,
        help="Boundary margin excluded from reference metrics.",
    )
    return parser


def run_example(
    *,
    data_root_arg: str | PathLike[str] | None,
    output_dir: str | PathLike[str] | None = None,
    save_volumes: bool = False,
    pretty: bool = False,
    crop_shape: tuple[int, int, int] = (64, 64, 64),
    max_crops: int = 1,
    percentile: float = 99.9,
    min_separation: float = 48.0,
    sigma1: float = 8.0,
    sigma2: float = 8.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    theta_min: float = 65.0,
    theta_max: float = 80.0,
    ru: int = 10,
    rv: int = 20,
    rw: int = 30,
    strain_max1: float = 0.25,
    strain_max2: float = 0.25,
    surface_smoothing1: float = 2.0,
    surface_smoothing2: float = 2.0,
    d: int = 4,
    fm: float = 0.3,
    interior_margin: int = 16,
) -> dict[str, Any]:
    data_root = resolve_f3d_data_root(data_root_arg)
    if output_dir is not None:
        ensure_output_not_in_data_root(output_dir, data_root)
    elif save_volumes:
        raise ValueError("--save-volumes requires --output-dir")

    arrays = read_reference_arrays(data_root)
    config = {
        "crop_shape": list(crop_shape),
        "max_crops": int(max_crops),
        "percentile": float(percentile),
        "min_separation": float(min_separation),
        "scanner": {
            "sigma1": float(sigma1),
            "sigma2": float(sigma2),
            "phi_min": float(phi_min),
            "phi_max": float(phi_max),
            "theta_min": float(theta_min),
            "theta_max": float(theta_max),
        },
        "voter": {
            "ru": int(ru),
            "rv": int(rv),
            "rw": int(rw),
            "strain_max1": float(strain_max1),
            "strain_max2": float(strain_max2),
            "surface_smoothing1": float(surface_smoothing1),
            "surface_smoothing2": float(surface_smoothing2),
            "d": int(d),
            "fm": float(fm),
        },
        "interior_margin": int(interior_margin),
        "overlap_percentiles": [float(p) for p in OVERLAP_PERCENTILES],
    }
    centers = pick_reference_centers(
        arrays["fv.dat"],
        count=max_crops,
        percentile=percentile,
        min_separation=min_separation,
    )

    crops = []
    for crop_index, center in enumerate(centers, start=1):
        slices = crop_slices(center, crop_shape, full_shape=arrays["ep.dat"].shape)
        ep_crop = _crop(arrays["ep.dat"], slices)
        reference_fv = _crop(arrays["fv.dat"], slices)
        reference_fvt = _crop(arrays["fvt.dat"], slices)
        outputs = run_pipeline(
            ep_crop,
            sigma1=sigma1,
            sigma2=sigma2,
            phi_min=phi_min,
            phi_max=phi_max,
            theta_min=theta_min,
            theta_max=theta_max,
            ru=ru,
            rv=rv,
            rw=rw,
            strain_max1=strain_max1,
            strain_max2=strain_max2,
            surface_smoothing1=surface_smoothing1,
            surface_smoothing2=surface_smoothing2,
            d=d,
            fm=fm,
        )

        if output_dir is not None and save_volumes:
            write_crop_volumes(Path(output_dir) / f"crop_{crop_index:03d}", outputs)

        crops.append(
            build_crop_report(
                crop_index=crop_index,
                center=center,
                slices=slices,
                crop_shape=ep_crop.shape,
                outputs=outputs,
                reference_fv=reference_fv,
                reference_fvt=reference_fvt,
                interior_margin=interior_margin,
            )
        )

    report = {
        "format_version": 1,
        "data_root": str(data_root),
        "config": config,
        "crops": crops,
    }

    if output_dir is not None:
        write_metrics_json(report, Path(output_dir) / "metrics.json", pretty=pretty)

    return report


def read_reference_arrays(data_root: str | PathLike[str]) -> dict[str, np.ndarray]:
    return {
        "ep.dat": read_f3d_file("ep.dat", data_root),
        "fv.dat": read_f3d_file("fv.dat", data_root),
        "fvt.dat": read_f3d_file("fvt.dat", data_root),
    }


def run_pipeline(
    ep: np.ndarray,
    *,
    sigma1: float,
    sigma2: float,
    phi_min: float,
    phi_max: float,
    theta_min: float,
    theta_max: float,
    ru: int,
    rv: int,
    rw: int,
    strain_max1: float,
    strain_max2: float,
    surface_smoothing1: float,
    surface_smoothing2: float,
    d: int,
    fm: float,
) -> dict[str, np.ndarray]:
    from pyosv.orient3d import FaultOrientScanner3
    from pyosv.voting3d import OptimalSurfaceVoter

    scanner = FaultOrientScanner3(sigma1=sigma1, sigma2=sigma2)
    ft, pt, tt = scanner.scan(phi_min, phi_max, theta_min, theta_max, ep)
    fet, fpt, ftt = scanner.thin(ft, pt, tt)

    voter = OptimalSurfaceVoter(ru=ru, rv=rv, rw=rw)
    voter.set_strain_max(strain_max1, strain_max2)
    voter.set_surface_smoothing(surface_smoothing1, surface_smoothing2)
    fv, vp, vt = voter.apply_voting(d=d, fm=fm, ft=fet, pt=fpt, tt=ftt)
    fvt = voter.thin(fv, vp, vt)

    return {
        "ft_py.dat": ft,
        "pt_py.dat": pt,
        "tt_py.dat": tt,
        "fet_py.dat": fet,
        "fpt_py.dat": fpt,
        "ftt_py.dat": ftt,
        "fv_py.dat": fv,
        "fvt_py.dat": fvt,
    }


def build_crop_report(
    *,
    crop_index: int,
    center: tuple[int, int, int],
    slices: tuple[slice, slice, slice],
    crop_shape: tuple[int, int, int],
    outputs: Mapping[str, np.ndarray],
    reference_fv: np.ndarray,
    reference_fvt: np.ndarray,
    interior_margin: int,
) -> dict[str, Any]:
    mask = interior_mask(crop_shape, margin=interior_margin)
    py_fv = np.asarray(outputs["fv_py.dat"])
    py_fvt = np.asarray(outputs["fvt_py.dat"])

    return {
        "index": int(crop_index),
        "center": [int(value) for value in center],
        "slices": [
            {"start": int(crop_slice.start), "stop": int(crop_slice.stop)} for crop_slice in slices
        ],
        "crop_shape": [int(size) for size in crop_shape],
        "pyosv": {
            "fv": summarize_array(py_fv),
            "fvt": summarize_array(py_fvt),
        },
        "reference": {
            "fv": summarize_array(reference_fv),
            "fvt": summarize_array(reference_fvt),
        },
        "normalized_correlation": {
            "fv": float(normalized_correlation(py_fv[mask], reference_fv[mask])),
            "fvt": float(normalized_correlation(py_fvt[mask], reference_fvt[mask])),
        },
        "top_percentile_overlap": {
            "fv": _overlaps(py_fv[mask], reference_fv[mask]),
            "fvt": _overlaps(py_fvt[mask], reference_fvt[mask]),
        },
        "finite_checks": {
            "pyosv": {
                name.removesuffix(".dat"): finite_report(values) for name, values in outputs.items()
            },
            "reference": {
                "fv": finite_report(reference_fv),
                "fvt": finite_report(reference_fvt),
            },
        },
    }


def finite_report(array: np.ndarray) -> dict[str, Any]:
    report = dict(finite_value_report(array))
    report["shape"] = [int(size) for size in report["shape"]]
    return report


def summarize_array(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array)
    finite = np.isfinite(values)
    finite_values = values[finite].astype(np.float64, copy=False)
    if finite_values.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "nonzero_fraction": 0.0,
        }

    return {
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values)),
        "nonzero_fraction": float(np.count_nonzero(np.abs(values) > NONZERO_EPSILON) / values.size),
    }


def write_metrics_json(
    report: Mapping[str, Any],
    output_json: str | PathLike[str],
    *,
    pretty: bool = False,
) -> Path:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_to_json(report, pretty=pretty), encoding="utf-8")
    return output_path


def write_crop_volumes(
    output_dir: str | PathLike[str], outputs: Mapping[str, np.ndarray]
) -> list[Path]:
    from pyosv.io import write_dat

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return [write_dat(directory / name, outputs[name]) for name in VOLUME_NAMES]


def report_to_json(report: Mapping[str, Any], *, pretty: bool = False) -> str:
    indent = 2 if pretty else None
    return json.dumps(_json_compatible(report), indent=indent, sort_keys=True) + "\n"


def ensure_output_not_in_data_root(
    output_dir: str | PathLike[str],
    data_root: str | PathLike[str],
) -> None:
    output_path = Path(output_dir).resolve(strict=False)
    data_root_path = Path(data_root).resolve(strict=False)
    try:
        output_path.relative_to(data_root_path)
    except ValueError:
        return
    raise ValueError(f"--output-dir must not be inside the F3 data root: {output_path}")


def _crop(array: np.ndarray, slices: tuple[slice, slice, slice]) -> np.ndarray:
    return np.ascontiguousarray(array[slices].astype(np.float32, copy=False))


def _overlaps(a: np.ndarray, b: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        percentile_key(p): top_percentile_overlap(a, b, percentile=p) for p in OVERLAP_PERCENTILES
    }


def percentile_key(percentile: float) -> str:
    return f"{percentile:g}"


def _json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = run_example(
            data_root_arg=args.data_root,
            output_dir=args.output_dir,
            save_volumes=args.save_volumes,
            pretty=args.pretty,
            crop_shape=args.crop_shape,
            max_crops=args.max_crops,
            percentile=args.percentile,
            min_separation=args.min_separation,
            sigma1=args.sigma1,
            sigma2=args.sigma2,
            phi_min=args.phi_min,
            phi_max=args.phi_max,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            ru=args.ru,
            rv=args.rv,
            rw=args.rw,
            strain_max1=args.strain_max1,
            strain_max2=args.strain_max2,
            surface_smoothing1=args.surface_smoothing1,
            surface_smoothing2=args.surface_smoothing2,
            d=args.d,
            fm=args.fm,
            interior_margin=args.interior_margin,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.output_dir is None:
        print(report_to_json(report, pretty=args.pretty), end="")
    else:
        print(Path(args.output_dir) / "metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
