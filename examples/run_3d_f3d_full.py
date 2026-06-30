"""Run the manual full-volume 3D F3 scan/vote workflow."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from pyosv.f3d_reference import F3D_ENV_VAR, F3D_SHAPE, read_f3d_file, resolve_f3d_data_root
from pyosv.metrics import (
    buffered_ridge_overlap,
    finite_value_report,
    normalized_correlation,
    sparse_ridge_distance_metrics,
    top_percentile_overlap,
)

NONZERO_EPSILON = 1.0e-6
OVERLAP_PERCENTILES = (95.0, 99.0, 99.5)
RIDGE_PERCENTILE = 99.0
RIDGE_BUFFER_RADIUS = 2.0
REPORT_OUTPUT_NAMES = ("ft_py.dat", "fv_py.dat", "fvt_py.dat")
SCANNER_OUTPUT_NAMES = ("ft_py.dat", "pt_py.dat", "tt_py.dat")
SCANNER_THIN_OUTPUT_NAMES = ("fet_py.dat", "fpt_py.dat", "ftt_py.dat")
VOTING_OUTPUT_NAMES = ("fv_py.dat", "vp_py.dat", "vt_py.dat")
FINAL_OUTPUT_NAMES = ("fv_py.dat", "fvt_py.dat")
INTERMEDIATE_OUTPUT_NAMES = (
    "ft_py.dat",
    "pt_py.dat",
    "tt_py.dat",
    "fet_py.dat",
    "fpt_py.dat",
    "ftt_py.dat",
    "vp_py.dat",
    "vt_py.dat",
)
OUTPUT_NAMES = (
    "ft_py.dat",
    "pt_py.dat",
    "tt_py.dat",
    "fet_py.dat",
    "fpt_py.dat",
    "ftt_py.dat",
    "fv_py.dat",
    "vp_py.dat",
    "vt_py.dat",
    "fvt_py.dat",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the manual full-volume F3 3D scan/vote workflow and report "
            "metrics against reference fl.dat, fv.dat, and fvt.dat."
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
        required=True,
        help="Directory for run_config.json and reusable/generated DAT outputs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional metrics JSON path. Defaults to OUTPUT_DIR/metrics.json.",
    )
    parser.add_argument("--pretty", action="store_true", help="Write indented JSON.")
    parser.add_argument("--sigma1", type=float, default=8.0, help="Scanner sigma1.")
    parser.add_argument("--sigma2", type=float, default=8.0, help="Scanner sigma2.")
    parser.add_argument("--phi-min", type=float, default=0.0, help="Minimum strike angle.")
    parser.add_argument("--phi-max", type=float, default=360.0, help="Maximum strike angle.")
    parser.add_argument("--theta-min", type=float, default=65.0, help="Minimum dip angle.")
    parser.add_argument("--theta-max", type=float, default=80.0, help="Maximum dip angle.")
    parser.add_argument("--ru", type=int, default=10, help="Voting normal half-width.")
    parser.add_argument("--rv", type=int, default=20, help="Voting dip half-width.")
    parser.add_argument("--rw", type=int, default=30, help="Voting strike half-width.")
    parser.add_argument("--d", type=int, default=4, help="Seed exclusion distance.")
    parser.add_argument("--fm", type=float, default=0.3, help="Minimum seed likelihood.")
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
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse the full existing pyosv DAT output set in --output-dir.",
    )
    parser.add_argument(
        "--skip-save-intermediates",
        action="store_true",
        help="With --save-volumes, write only ft_py.dat, fv_py.dat, and fvt_py.dat.",
    )
    parser.add_argument(
        "--save-volumes",
        action="store_true",
        help="Write generated pyosv DAT outputs under --output-dir.",
    )
    return parser


def run_example(
    *,
    data_root_arg: str | PathLike[str] | None,
    output_dir: str | PathLike[str],
    output_json: str | PathLike[str] | None = None,
    pretty: bool = False,
    sigma1: float = 8.0,
    sigma2: float = 8.0,
    phi_min: float = 0.0,
    phi_max: float = 360.0,
    theta_min: float = 65.0,
    theta_max: float = 80.0,
    ru: int = 10,
    rv: int = 20,
    rw: int = 30,
    d: int = 4,
    fm: float = 0.3,
    strain_max1: float = 0.25,
    strain_max2: float = 0.25,
    surface_smoothing1: float = 2.0,
    surface_smoothing2: float = 2.0,
    reuse_existing: bool = False,
    skip_save_intermediates: bool = False,
    save_volumes: bool = False,
) -> dict[str, Any]:
    data_root = resolve_f3d_data_root(data_root_arg)
    output_path = ensure_output_not_in_data_root(output_dir, data_root)
    metrics_path = resolve_metrics_path(output_json, output_path)
    ensure_output_not_in_data_root(metrics_path, data_root, option_name="--output-json")
    output_path.mkdir(parents=True, exist_ok=True)

    config = build_run_config(
        data_root=data_root,
        output_dir=output_path,
        sigma1=sigma1,
        sigma2=sigma2,
        phi_min=phi_min,
        phi_max=phi_max,
        theta_min=theta_min,
        theta_max=theta_max,
        ru=ru,
        rv=rv,
        rw=rw,
        d=d,
        fm=fm,
        strain_max1=strain_max1,
        strain_max2=strain_max2,
        surface_smoothing1=surface_smoothing1,
        surface_smoothing2=surface_smoothing2,
        reuse_existing=reuse_existing,
        skip_save_intermediates=skip_save_intermediates,
        save_volumes=save_volumes,
        output_json=metrics_path,
    )
    write_json(output_path / "run_config.json", config, pretty=pretty)

    start_time = time.perf_counter()
    outputs, runtime = run_or_reuse_pipeline(
        data_root=data_root,
        output_dir=output_path,
        sigma1=sigma1,
        sigma2=sigma2,
        phi_min=phi_min,
        phi_max=phi_max,
        theta_min=theta_min,
        theta_max=theta_max,
        ru=ru,
        rv=rv,
        rw=rw,
        d=d,
        fm=fm,
        strain_max1=strain_max1,
        strain_max2=strain_max2,
        surface_smoothing1=surface_smoothing1,
        surface_smoothing2=surface_smoothing2,
        reuse_existing=reuse_existing,
        skip_save_intermediates=skip_save_intermediates,
        save_volumes=save_volumes,
    )
    runtime["total_elapsed_seconds"] = float(time.perf_counter() - start_time)

    reference_fl = read_f3d_file("fl.dat", data_root)
    reference_fv = read_f3d_file("fv.dat", data_root)
    reference_fvt = read_f3d_file("fvt.dat", data_root)
    metrics = build_metrics_report(
        data_root=data_root,
        config=config,
        pyosv_ft=outputs["ft_py.dat"],
        pyosv_fv=outputs["fv_py.dat"],
        pyosv_fvt=outputs["fvt_py.dat"],
        reference_fl=reference_fl,
        reference_fv=reference_fv,
        reference_fvt=reference_fvt,
        runtime=runtime,
    )
    write_json(metrics_path, metrics, pretty=pretty)
    return metrics


def build_run_config(
    *,
    data_root: str | PathLike[str],
    output_dir: str | PathLike[str],
    sigma1: float,
    sigma2: float,
    phi_min: float,
    phi_max: float,
    theta_min: float,
    theta_max: float,
    ru: int,
    rv: int,
    rw: int,
    d: int,
    fm: float,
    strain_max1: float,
    strain_max2: float,
    surface_smoothing1: float,
    surface_smoothing2: float,
    reuse_existing: bool,
    skip_save_intermediates: bool,
    save_volumes: bool,
    output_json: str | PathLike[str],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "data_root": str(Path(data_root)),
        "output_dir": str(Path(output_dir)),
        "output_json": str(Path(output_json)),
        "input": "ep.dat",
        "reference": ["fl.dat", "fv.dat", "fvt.dat"],
        "shape": [int(size) for size in F3D_SHAPE],
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
            "d": int(d),
            "fm": float(fm),
            "strain_max1": float(strain_max1),
            "strain_max2": float(strain_max2),
            "surface_smoothing1": float(surface_smoothing1),
            "surface_smoothing2": float(surface_smoothing2),
        },
        "reuse_existing": bool(reuse_existing),
        "skip_save_intermediates": bool(skip_save_intermediates),
        "save_volumes": bool(save_volumes),
        "outputs": {
            "report": list(REPORT_OUTPUT_NAMES),
            "final": list(FINAL_OUTPUT_NAMES),
            "intermediate": [] if skip_save_intermediates else list(INTERMEDIATE_OUTPUT_NAMES),
        },
    }


def run_or_reuse_pipeline(
    *,
    data_root: str | PathLike[str],
    output_dir: str | PathLike[str],
    sigma1: float,
    sigma2: float,
    phi_min: float,
    phi_max: float,
    theta_min: float,
    theta_max: float,
    ru: int,
    rv: int,
    rw: int,
    d: int,
    fm: float,
    strain_max1: float,
    strain_max2: float,
    surface_smoothing1: float,
    surface_smoothing2: float,
    reuse_existing: bool,
    skip_save_intermediates: bool,
    save_volumes: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    output_path = Path(output_dir)
    if reuse_existing:
        require_existing_outputs(output_path, OUTPUT_NAMES)
        return read_outputs(output_path, OUTPUT_NAMES), {
            "mode": "reuse_existing",
            "scanner_elapsed_seconds": 0.0,
            "scanner_thin_elapsed_seconds": 0.0,
            "voting_elapsed_seconds": 0.0,
            "voter_thin_elapsed_seconds": 0.0,
        }

    from pyosv.orient3d import FaultOrientScanner3
    from pyosv.voting3d import OptimalSurfaceVoter

    outputs: dict[str, np.ndarray] = {}
    runtime: dict[str, Any] = {"mode": "computed"}
    scanner = FaultOrientScanner3(sigma1=sigma1, sigma2=sigma2)
    voter = OptimalSurfaceVoter(ru=ru, rv=rv, rw=rw)
    voter.set_strain_max(strain_max1, strain_max2)
    voter.set_surface_smoothing(surface_smoothing1, surface_smoothing2)

    stage_start = time.perf_counter()
    ep = read_f3d_file("ep.dat", data_root)
    ft, pt, tt = scanner.scan(phi_min, phi_max, theta_min, theta_max, ep)
    outputs.update(dict(zip(SCANNER_OUTPUT_NAMES, (ft, pt, tt), strict=True)))
    runtime["scanner_elapsed_seconds"] = float(time.perf_counter() - stage_start)
    if save_volumes:
        write_outputs(
            output_path, outputs, SCANNER_OUTPUT_NAMES, skip_intermediates=skip_save_intermediates
        )

    stage_start = time.perf_counter()
    fet, fpt, ftt = scanner.thin(
        outputs["ft_py.dat"],
        outputs["pt_py.dat"],
        outputs["tt_py.dat"],
    )
    outputs.update(dict(zip(SCANNER_THIN_OUTPUT_NAMES, (fet, fpt, ftt), strict=True)))
    runtime["scanner_thin_elapsed_seconds"] = float(time.perf_counter() - stage_start)
    if save_volumes:
        write_outputs(
            output_path,
            outputs,
            SCANNER_THIN_OUTPUT_NAMES,
            skip_intermediates=skip_save_intermediates,
        )

    stage_start = time.perf_counter()
    fv, vp, vt = voter.apply_voting(
        d=d,
        fm=fm,
        ft=outputs["fet_py.dat"],
        pt=outputs["fpt_py.dat"],
        tt=outputs["ftt_py.dat"],
    )
    outputs.update(dict(zip(VOTING_OUTPUT_NAMES, (fv, vp, vt), strict=True)))
    runtime["voting_elapsed_seconds"] = float(time.perf_counter() - stage_start)
    if save_volumes:
        write_outputs(
            output_path, outputs, VOTING_OUTPUT_NAMES, skip_intermediates=skip_save_intermediates
        )

    stage_start = time.perf_counter()
    outputs["fvt_py.dat"] = voter.thin(
        outputs["fv_py.dat"],
        outputs["vp_py.dat"],
        outputs["vt_py.dat"],
    )
    runtime["voter_thin_elapsed_seconds"] = float(time.perf_counter() - stage_start)
    if save_volumes:
        write_outputs(
            output_path, outputs, ("fvt_py.dat",), skip_intermediates=skip_save_intermediates
        )

    return {name: outputs[name] for name in OUTPUT_NAMES}, runtime


def should_reuse_outputs(
    output_dir: str | PathLike[str],
    names: tuple[str, ...],
    reuse_existing: bool,
) -> bool:
    if not reuse_existing:
        return False
    directory = Path(output_dir)
    return all((directory / name).is_file() for name in names)


def read_outputs(
    output_dir: str | PathLike[str],
    names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    from pyosv.io import read_dat

    directory = Path(output_dir)
    return {name: read_dat(directory / name, F3D_SHAPE) for name in names}


def require_existing_outputs(output_dir: str | PathLike[str], names: tuple[str, ...]) -> None:
    directory = Path(output_dir)
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"--reuse-existing requires existing output files: {joined}")


def write_outputs(
    output_dir: str | PathLike[str],
    outputs: Mapping[str, np.ndarray],
    names: tuple[str, ...],
    *,
    skip_intermediates: bool,
) -> list[Path]:
    from pyosv.io import write_dat

    directory = Path(output_dir)
    paths = []
    for name in names:
        if skip_intermediates and name not in REPORT_OUTPUT_NAMES:
            continue
        paths.append(write_dat(directory / name, outputs[name]))
    return paths


def build_metrics_report(
    *,
    data_root: str | PathLike[str],
    config: Mapping[str, Any],
    pyosv_ft: np.ndarray,
    pyosv_fv: np.ndarray,
    pyosv_fvt: np.ndarray,
    reference_fl: np.ndarray,
    reference_fv: np.ndarray,
    reference_fvt: np.ndarray,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "config": config,
        "data": {
            "shape": [int(size) for size in F3D_SHAPE],
            "data_root": str(Path(data_root)),
            "files": data_file_report(data_root, ("ep.dat", "fl.dat", "fv.dat", "fvt.dat")),
        },
        "scanner": {
            "parameters": config["scanner"],
            "ft_py_vs_fl": comparison_metrics(pyosv_ft, reference_fl),
        },
        "voting": {
            "parameters": config["voter"],
            "fv_py_vs_fv": {
                **comparison_metrics(pyosv_fv, reference_fv),
                "nonzero_fraction_ratio": nonzero_fraction_ratio(pyosv_fv, reference_fv),
            },
        },
        "thinning": {
            "fvt_py_vs_fvt": {
                **comparison_metrics(pyosv_fvt, reference_fvt),
                "buffered_ridge_overlap": buffered_ridge_overlap(
                    reference_fvt,
                    pyosv_fvt,
                    percentile=RIDGE_PERCENTILE,
                    radius=RIDGE_BUFFER_RADIUS,
                ),
                "sparse_ridge_distance_metrics": sparse_ridge_distance_metrics(
                    reference_fvt,
                    pyosv_fvt,
                    percentile=RIDGE_PERCENTILE,
                ),
                "nonzero_fraction_ratio": nonzero_fraction_ratio(pyosv_fvt, reference_fvt),
            },
        },
        "runtime": dict(runtime),
        "pyosv": {
            "ft": summarize_array(pyosv_ft),
            "fv": summarize_array(pyosv_fv),
            "fvt": summarize_array(pyosv_fvt),
        },
        "reference": {
            "fl": summarize_array(reference_fl),
            "fv": summarize_array(reference_fv),
            "fvt": summarize_array(reference_fvt),
        },
        "finite_checks": {
            "pyosv": {
                "ft_py": finite_report(pyosv_ft),
                "fv_py": finite_report(pyosv_fv),
                "fvt_py": finite_report(pyosv_fvt),
            },
            "reference": {
                "fl": finite_report(reference_fl),
                "fv": finite_report(reference_fv),
                "fvt": finite_report(reference_fvt),
            },
        },
    }


def comparison_metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    return {
        "normalized_correlation": float(normalized_correlation(candidate, reference)),
        "top_percentile_overlap": _overlaps(candidate, reference),
    }


def nonzero_fraction_ratio(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate_fraction = summarize_array(candidate)["nonzero_fraction"]
    reference_fraction = summarize_array(reference)["nonzero_fraction"]
    if reference_fraction == 0.0:
        return 0.0
    return float(candidate_fraction / reference_fraction)


def data_file_report(
    data_root: str | PathLike[str], names: tuple[str, ...]
) -> dict[str, dict[str, Any]]:
    root = Path(data_root)
    report = {}
    for name in names:
        path = root / name
        report[name] = {
            "path": str(path),
            "exists": bool(path.is_file()),
            "size_bytes": int(path.stat().st_size) if path.is_file() else None,
        }
    return report


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


def write_json(
    path: str | PathLike[str], report: Mapping[str, Any], *, pretty: bool = False
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_to_json(report, pretty=pretty), encoding="utf-8")
    return output_path


def report_to_json(report: Mapping[str, Any], *, pretty: bool = False) -> str:
    indent = 2 if pretty else None
    return json.dumps(_json_compatible(report), indent=indent, sort_keys=True) + "\n"


def resolve_metrics_path(
    output_json: str | PathLike[str] | None,
    output_dir: str | PathLike[str],
) -> Path:
    if output_json is not None:
        return Path(output_json)
    return Path(output_dir) / "metrics.json"


def ensure_output_not_in_data_root(
    output_dir: str | PathLike[str],
    data_root: str | PathLike[str],
    *,
    option_name: str = "--output-dir",
) -> Path:
    output_path = Path(output_dir).resolve(strict=False)
    data_root_path = Path(data_root).resolve(strict=False)
    try:
        output_path.relative_to(data_root_path)
    except ValueError:
        return output_path
    raise ValueError(f"{option_name} must not be inside the F3 data root: {output_path}")


def _overlaps(a: np.ndarray, b: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        percentile_key(p): top_percentile_overlap(a, b, percentile=p) for p in OVERLAP_PERCENTILES
    }


def percentile_key(percentile: float) -> str:
    return f"{percentile:g}"


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run_example(
            data_root_arg=args.data_root,
            output_dir=args.output_dir,
            output_json=args.output_json,
            pretty=args.pretty,
            sigma1=args.sigma1,
            sigma2=args.sigma2,
            phi_min=args.phi_min,
            phi_max=args.phi_max,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
            ru=args.ru,
            rv=args.rv,
            rw=args.rw,
            d=args.d,
            fm=args.fm,
            strain_max1=args.strain_max1,
            strain_max2=args.strain_max2,
            surface_smoothing1=args.surface_smoothing1,
            surface_smoothing2=args.surface_smoothing2,
            reuse_existing=args.reuse_existing,
            skip_save_intermediates=args.skip_save_intermediates,
            save_volumes=args.save_volumes,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(resolve_metrics_path(args.output_json, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
