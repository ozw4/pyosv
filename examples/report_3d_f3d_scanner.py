"""Compare scanner-only F3 3D output against public fl.dat crops."""

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
    parse_shape3,
    pick_reference_centers,
    read_f3d_file,
    resolve_f3d_data_root,
)
from pyosv.metrics import finite_value_report, normalized_correlation, top_percentile_overlap

NONZERO_EPSILON = 1.0e-6
OVERLAP_PERCENTILES = (95.0, 99.0, 99.5, 99.9)
VOLUME_NAMES = ("ft_py.dat", "pt_py.dat", "tt_py.dat")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pyosv 3D F3 scanner on ep.dat crops and compare ft_py "
            "against the public fl.dat fault-likelihood crop."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Path to the F3 reference data root. Defaults to {F3D_ENV_VAR}.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON output path. Parent directories are created as needed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for crop-level DAT outputs with --save-volumes.",
    )
    parser.add_argument(
        "--save-volumes",
        action="store_true",
        help="Write ft_py.dat, pt_py.dat, and tt_py.dat under --output-dir.",
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
        help="Reference fl percentile used to pick crop centers.",
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
    return parser


def run_example(
    *,
    data_root_arg: str | PathLike[str] | None,
    output_json: str | PathLike[str] | None = None,
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
) -> dict[str, Any]:
    data_root = resolve_f3d_data_root(data_root_arg)
    if output_json is not None:
        ensure_output_not_in_data_root(output_json, data_root, option_name="--output-json")
    if output_dir is not None:
        ensure_output_not_in_data_root(output_dir, data_root, option_name="--output-dir")
    elif save_volumes:
        raise ValueError("--save-volumes requires --output-dir")

    arrays = read_reference_arrays(data_root)
    config = build_config(
        crop_shape=crop_shape,
        max_crops=max_crops,
        percentile=percentile,
        min_separation=min_separation,
        sigma1=sigma1,
        sigma2=sigma2,
        phi_min=phi_min,
        phi_max=phi_max,
        theta_min=theta_min,
        theta_max=theta_max,
    )
    centers = pick_reference_centers(
        arrays["fl.dat"],
        count=max_crops,
        percentile=percentile,
        min_separation=min_separation,
        crop_shape=crop_shape,
    )

    crops = []
    for crop_index, center in enumerate(centers, start=1):
        slices = crop_slices(center, crop_shape, full_shape=arrays["ep.dat"].shape)
        ep_crop = _crop(arrays["ep.dat"], slices)
        reference_fl = _crop(arrays["fl.dat"], slices)
        outputs = run_scanner(
            ep_crop,
            sigma1=sigma1,
            sigma2=sigma2,
            phi_min=phi_min,
            phi_max=phi_max,
            theta_min=theta_min,
            theta_max=theta_max,
        )

        if output_dir is not None and save_volumes:
            write_crop_volumes(Path(output_dir) / f"crop_{crop_index:03d}", outputs)

        crops.append(
            build_crop_report(
                crop_index=crop_index,
                center=center,
                slices=slices,
                outputs=outputs,
                reference_fl=reference_fl,
            )
        )

    report = build_report(data_root=data_root, config=config, crops=crops)
    if output_json is not None:
        write_report_json(report, output_json, pretty=pretty)

    return report


def read_reference_arrays(data_root: str | PathLike[str]) -> dict[str, np.ndarray]:
    return {
        "ep.dat": read_f3d_file("ep.dat", data_root),
        "fl.dat": read_f3d_file("fl.dat", data_root),
    }


def build_config(
    *,
    crop_shape: tuple[int, int, int],
    max_crops: int,
    percentile: float,
    min_separation: float,
    sigma1: float,
    sigma2: float,
    phi_min: float,
    phi_max: float,
    theta_min: float,
    theta_max: float,
) -> dict[str, Any]:
    return {
        "input": "ep.dat",
        "reference": "fl.dat",
        "comparison": "scanner_only_ft_py_vs_fl_dat",
        "crop_selection": {
            "source": "fl.dat",
            "crop_shape": [int(size) for size in crop_shape],
            "max_crops": int(max_crops),
            "percentile": float(percentile),
            "min_separation": float(min_separation),
            "boundary_margin": "crop_shape",
        },
        "scanner": {
            "sigma1": float(sigma1),
            "sigma2": float(sigma2),
            "phi_min": float(phi_min),
            "phi_max": float(phi_max),
            "theta_min": float(theta_min),
            "theta_max": float(theta_max),
        },
        "overlap_percentiles": [float(p) for p in OVERLAP_PERCENTILES],
        "outputs": list(VOLUME_NAMES),
    }


def run_scanner(
    ep: np.ndarray,
    *,
    sigma1: float,
    sigma2: float,
    phi_min: float,
    phi_max: float,
    theta_min: float,
    theta_max: float,
) -> dict[str, np.ndarray]:
    from pyosv.orient3d import FaultOrientScanner3

    scanner = FaultOrientScanner3(sigma1=sigma1, sigma2=sigma2)
    ft, pt, tt = scanner.scan(phi_min, phi_max, theta_min, theta_max, ep)
    return dict(zip(VOLUME_NAMES, (ft, pt, tt), strict=True))


def build_report(
    *,
    data_root: str | PathLike[str] | None,
    config: Mapping[str, Any],
    crops: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "data_root": str(data_root) if data_root is not None else None,
        "comparison": "scanner-only ft_py.dat versus public fl.dat",
        "config": dict(config),
        "crops": list(crops),
    }


def build_crop_report(
    *,
    crop_index: int,
    center: tuple[int, int, int],
    slices: tuple[slice, slice, slice],
    outputs: Mapping[str, np.ndarray],
    reference_fl: np.ndarray,
) -> dict[str, Any]:
    ft_py = np.asarray(outputs["ft_py.dat"])
    pt_py = np.asarray(outputs["pt_py.dat"])
    tt_py = np.asarray(outputs["tt_py.dat"])
    fl_ref = np.asarray(reference_fl)

    return {
        "index": int(crop_index),
        "center": [int(value) for value in center],
        "slices": [
            {"start": int(crop_slice.start), "stop": int(crop_slice.stop)} for crop_slice in slices
        ],
        "crop_shape": [int(size) for size in ft_py.shape],
        "pyosv": {
            "ft_py": summarize_array(ft_py),
            "pt_py": summarize_array(pt_py),
            "tt_py": summarize_array(tt_py),
        },
        "reference": {
            "fl": summarize_array(fl_ref),
        },
        "normalized_correlation": {
            "ft_py_vs_fl": float(normalized_correlation(ft_py, fl_ref)),
        },
        "top_percentile_overlap": {
            "ft_py_vs_fl": _overlaps(ft_py, fl_ref),
        },
        "slice_correlation": {
            "ft_py_vs_fl_i3": slice_correlation_summary(ft_py, fl_ref),
        },
        "finite_checks": {
            "pyosv": {
                "ft_py": finite_report(ft_py),
                "pt_py": finite_report(pt_py),
                "tt_py": finite_report(tt_py),
            },
            "reference": {
                "fl": finite_report(fl_ref),
            },
        },
    }


def summarize_array(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    finite = np.isfinite(values)
    finite_values = values[finite].astype(np.float64, copy=False)
    summary: dict[str, Any] = {
        "shape": [int(size) for size in values.shape],
        "finite_count": int(np.count_nonzero(finite)),
        "min": None,
        "max": None,
        "mean": None,
        "nonzero_fraction": (
            float(np.count_nonzero(np.abs(values) > NONZERO_EPSILON) / values.size)
            if values.size
            else 0.0
        ),
    }
    if finite_values.size:
        summary.update(
            {
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
                "mean": float(np.mean(finite_values)),
            }
        )
    return summary


def finite_report(array: np.ndarray) -> dict[str, Any]:
    report = dict(finite_value_report(array))
    report["shape"] = [int(size) for size in report["shape"]]
    return report


def slice_correlation_summary(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    av = np.asarray(a)
    bv = np.asarray(b)
    if av.shape != bv.shape:
        raise ValueError(f"array shapes must match, got {av.shape} and {bv.shape}")
    if av.ndim != 3:
        raise ValueError("arrays must be 3D")

    correlations = np.array(
        [normalized_correlation(av[index], bv[index]) for index in range(av.shape[0])],
        dtype=np.float64,
    )
    return {
        "axis": "i3",
        "count": int(correlations.size),
        "finite_count": int(np.count_nonzero(np.isfinite(correlations))),
        "min": float(np.min(correlations)) if correlations.size else None,
        "max": float(np.max(correlations)) if correlations.size else None,
        "mean": float(np.mean(correlations)) if correlations.size else None,
    }


def write_report_json(
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
    output_dir: str | PathLike[str],
    outputs: Mapping[str, np.ndarray],
) -> list[Path]:
    from pyosv.io import write_dat

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return [write_dat(directory / name, outputs[name]) for name in VOLUME_NAMES]


def report_to_json(report: Mapping[str, Any], *, pretty: bool = False) -> str:
    indent = 2 if pretty else None
    return json.dumps(_json_compatible(report), indent=indent, sort_keys=True) + "\n"


def ensure_output_not_in_data_root(
    output_path: str | PathLike[str],
    data_root: str | PathLike[str],
    *,
    option_name: str,
) -> Path:
    resolved_output = Path(output_path).resolve(strict=False)
    resolved_data_root = Path(data_root).resolve(strict=False)
    try:
        resolved_output.relative_to(resolved_data_root)
    except ValueError:
        return resolved_output
    raise ValueError(f"{option_name} must not be inside the F3 data root: {resolved_output}")


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
            output_json=args.output_json,
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
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.output_json is None:
        print(report_to_json(report, pretty=args.pretty), end="")
    else:
        print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
