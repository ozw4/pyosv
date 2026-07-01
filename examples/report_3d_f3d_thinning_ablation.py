"""Run F3 crop scanner/voter thinning ablation diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

import run_3d_f3d_crop_validation as crop_validation
from pyosv.f3d_reference import (
    F3D_ENV_VAR,
    crop_slices,
    parse_shape3,
    pick_reference_centers,
    resolve_f3d_data_root,
)

DEFAULT_COUNT = 3
DEFAULT_CROP_SHAPE = (64, 64, 64)
DEFAULT_INTERIOR_MARGIN = 16
DEFAULT_PERCENTILE = 99.9
DEFAULT_MIN_SEPARATION = 48.0
REFERENCE_OSV_DIR = Path(__file__).resolve().parents[1] / "reference_osv"
DEFAULT_SCANNER_BACKENDS = ("current",)
SUPPORTED_SCANNER_BACKENDS = ("current", "reference-like")

CASE_DEFINITIONS: tuple[dict[str, str], ...] = (
    {
        "name": "case_01_current_current",
        "scanner_thin_mode": "normal",
        "voter_thin_mode": "normal",
    },
    {
        "name": "case_02_current_reference_voter",
        "scanner_thin_mode": "normal",
        "voter_thin_mode": "reference",
    },
    {
        "name": "case_03_reference_scanner_current",
        "scanner_thin_mode": "reference",
        "voter_thin_mode": "normal",
    },
    {
        "name": "case_04_reference_reference",
        "scanner_thin_mode": "reference",
        "voter_thin_mode": "reference",
    },
)
AGGREGATE_ROOTS = (
    "pyosv",
    "normalized_correlation",
    "top_percentile_overlap",
    "buffered_ridge_overlap",
    "sparse_ridge_distance_metrics",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run F3 crop thinning ablations that vary scanner thinning and "
            "final voter thinning independently."
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
        "--save-figures",
        action="store_true",
        help="Write per-crop, per-case PNG diagnostics under OUTPUT_JSON.parent.",
    )
    parser.add_argument(
        "--figure-percentile",
        type=float,
        default=99.0,
        help="Upper display clipping percentile and ridge percentile for PNG diagnostics.",
    )
    parser.add_argument(
        "--ridge-buffer-radius",
        type=float,
        default=2.0,
        help="Ridge overlay buffer radius for PNG diagnostics.",
    )
    parser.add_argument(
        "--write-markdown-index",
        action="store_true",
        help="Write visual_report.md next to the metrics JSON.",
    )
    parser.add_argument("--pretty", action="store_true", help="Write indented JSON.")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of deterministic crops to select when --center is omitted.",
    )
    parser.add_argument(
        "--crop-shape",
        type=parse_shape3,
        default=DEFAULT_CROP_SHAPE,
        help="Crop shape in n3,n2,n1 order.",
    )
    parser.add_argument(
        "--interior-margin",
        type=int,
        default=DEFAULT_INTERIOR_MARGIN,
        help="Boundary margin excluded from interior metrics.",
    )
    parser.add_argument(
        "--scanner-backends",
        "--scanner-backend",
        dest="scanner_backends",
        type=parse_scanner_backends,
        default=DEFAULT_SCANNER_BACKENDS,
        help="Comma-separated scanner backends: current,reference-like.",
    )
    parser.add_argument(
        "--center",
        action="append",
        type=crop_validation.parse_index3,
        default=None,
        help="Explicit crop center in i3,i2,i1 order. May be repeated.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help="Reference fv percentile used to pick crop centers.",
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=DEFAULT_MIN_SEPARATION,
        help="Minimum deterministic center separation in samples.",
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
    return parser


def run_example(
    *,
    data_root_arg: str | PathLike[str] | None,
    output_json: str | PathLike[str] | None = None,
    save_figures: bool = False,
    figure_percentile: float = 99.0,
    ridge_buffer_radius: float = 2.0,
    write_markdown_index: bool = False,
    pretty: bool = False,
    count: int = DEFAULT_COUNT,
    crop_shape: tuple[int, int, int] = DEFAULT_CROP_SHAPE,
    interior_margin: int = DEFAULT_INTERIOR_MARGIN,
    centers: Iterable[tuple[int, int, int]] | None = None,
    percentile: float = DEFAULT_PERCENTILE,
    min_separation: float = DEFAULT_MIN_SEPARATION,
    scanner_backends: Iterable[str] = DEFAULT_SCANNER_BACKENDS,
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
) -> dict[str, Any]:
    data_root = resolve_f3d_data_root(data_root_arg)
    if output_json is not None:
        ensure_output_path_allowed(output_json, data_root, option_name="--output-json")
    elif save_figures:
        raise ValueError("--save-figures requires --output-json")
    elif write_markdown_index:
        raise ValueError("--write-markdown-index requires --output-json")
    if save_figures:
        crop_validation.require_figure_support()
    if count < 0:
        raise ValueError("count must be >= 0")
    crop_shape, interior_margin = validate_crop_config(crop_shape, interior_margin)
    backend_names = validate_scanner_backends(scanner_backends)

    output_base_dir = Path(output_json).parent if output_json is not None else None
    arrays = crop_validation.read_reference_arrays(data_root)
    selected_centers = select_centers(
        arrays["fv.dat"],
        count=count,
        centers=centers,
        percentile=percentile,
        min_separation=min_separation,
        crop_shape=crop_shape,
    )
    config = build_config(
        crop_shape=crop_shape,
        interior_margin=interior_margin,
        count=count,
        centers=selected_centers,
        explicit_centers=centers is not None,
        percentile=percentile,
        min_separation=min_separation,
        scanner_backends=backend_names,
        save_figures=save_figures,
        figure_percentile=figure_percentile,
        ridge_buffer_radius=ridge_buffer_radius,
        write_markdown_index=write_markdown_index,
        visual_report_path=output_base_dir / "visual_report.md" if output_base_dir else None,
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

    crops = []
    for crop_index, center in enumerate(selected_centers, start=1):
        slices = crop_slices(center, crop_shape, full_shape=arrays["ep.dat"].shape)
        ep_crop = _crop(arrays["ep.dat"], slices)
        reference_fv = _crop(arrays["fv.dat"], slices)
        reference_fvt = _crop(arrays["fvt.dat"], slices)
        backend_reports: dict[str, Any] = {}
        for backend_name in backend_names:
            case_outputs = run_ablation_pipeline(
                ep_crop,
                scanner_backend=backend_name,
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

            case_reports: dict[str, Any] = {}
            for case in CASE_DEFINITIONS:
                case_name = case["name"]
                outputs = case_outputs[case_name]
                case_report = crop_validation.build_crop_report(
                    crop_index=crop_index,
                    center=center,
                    slices=slices,
                    crop_shape=ep_crop.shape,
                    outputs=outputs,
                    reference_fv=reference_fv,
                    reference_fvt=reference_fvt,
                    interior_margin=interior_margin,
                )
                case_report["case"] = dict(case)
                case_report["scanner_backend"] = backend_name
                if save_figures:
                    if output_base_dir is None:
                        raise ValueError("--save-figures requires --output-json")
                    case_report["figures"] = write_case_figures(
                        figure_output_dir(
                            output_base_dir,
                            crop_index=crop_index,
                            backend=backend_name,
                            case_name=case_name,
                            backend_count=len(backend_names),
                        ),
                        metrics_base_dir=output_base_dir,
                        reference_fvt=reference_fvt,
                        outputs=outputs,
                        figure_percentile=figure_percentile,
                        ridge_buffer_radius=ridge_buffer_radius,
                    )
                case_reports[case_name] = case_report
            backend_reports[backend_name] = {"cases": case_reports}

        crop_report = {
            "index": int(crop_index),
            "crop_center": [int(value) for value in center],
            "crop_slices": crop_validation.slices_to_json(slices),
            "crop_shape": [int(size) for size in ep_crop.shape],
            "interior_margin": int(interior_margin),
            "backends": backend_reports,
        }
        if backend_names == ("current",):
            crop_report["cases"] = backend_reports["current"]["cases"]
        crops.append(crop_report)

    aggregate = aggregate_backend_case_metrics(crops, scanner_backends=backend_names)
    if backend_names == ("current",):
        aggregate["cases"] = aggregate["backends"]["current"]["cases"]
    report = _json_compatible(
        {
            "format_version": 1,
            "data_root": str(data_root),
            "config": config,
            "crops": crops,
            "aggregate": aggregate,
        }
    )

    if output_json is not None:
        write_report_json(report, output_json, pretty=pretty)
        if write_markdown_index:
            write_visual_report_markdown(report, Path(output_json).parent / "visual_report.md")

    return report


def run_ablation_pipeline(
    ep: np.ndarray,
    *,
    scanner_backend: str = "current",
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
) -> dict[str, dict[str, np.ndarray]]:
    from pyosv.orient3d import FaultOrientScanner3
    from pyosv.voting3d import OptimalSurfaceVoter

    scanner = FaultOrientScanner3(sigma1=sigma1, sigma2=sigma2)
    ft, pt, tt = _scan_backend(
        scanner,
        backend=scanner_backend,
        phi_min=phi_min,
        phi_max=phi_max,
        theta_min=theta_min,
        theta_max=theta_max,
        ep=ep,
    )

    thinned_by_scanner_mode: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for mode in ("normal", "reference"):
        thinned_by_scanner_mode[mode] = scanner.thin(ft, pt, tt, mode=mode)

    voter = OptimalSurfaceVoter(ru=ru, rv=rv, rw=rw)
    voter.set_strain_max(strain_max1, strain_max2)
    voter.set_surface_smoothing(surface_smoothing1, surface_smoothing2)

    voting_by_scanner_mode: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for mode, thinned in thinned_by_scanner_mode.items():
        fet, fpt, ftt = thinned
        voting_by_scanner_mode[mode] = voter.apply_voting(d=d, fm=fm, ft=fet, pt=fpt, tt=ftt)

    outputs: dict[str, dict[str, np.ndarray]] = {}
    for case in CASE_DEFINITIONS:
        scanner_mode = case["scanner_thin_mode"]
        voter_mode = case["voter_thin_mode"]
        fet, fpt, ftt = thinned_by_scanner_mode[scanner_mode]
        fv, vp, vt = voting_by_scanner_mode[scanner_mode]
        fvt = voter.thin(fv, vp, vt, mode=voter_mode)
        outputs[case["name"]] = {
            "ft_py.dat": ft,
            "pt_py.dat": pt,
            "tt_py.dat": tt,
            "fet_py.dat": fet,
            "fpt_py.dat": fpt,
            "ftt_py.dat": ftt,
            "fv_py.dat": fv,
            "fvt_py.dat": fvt,
        }

    return outputs


def select_centers(
    fv: np.ndarray,
    *,
    count: int,
    centers: Iterable[tuple[int, int, int]] | None,
    percentile: float,
    min_separation: float,
    crop_shape: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    if centers is not None:
        return [tuple(int(index) for index in center) for center in centers]
    return pick_reference_centers(
        fv,
        count=count,
        percentile=percentile,
        min_separation=min_separation,
        crop_shape=crop_shape,
    )


def validate_crop_config(
    crop_shape: tuple[int, int, int],
    interior_margin: int,
) -> tuple[tuple[int, int, int], int]:
    if interior_margin < 0:
        raise ValueError("interior_margin must be >= 0")
    crop_validation.interior_slices(crop_shape, margin=interior_margin)
    return crop_shape, int(interior_margin)


def parse_scanner_backends(text: str) -> tuple[str, ...]:
    backends = tuple(part.strip() for part in text.split(",") if part.strip())
    try:
        return validate_scanner_backends(backends)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def validate_scanner_backends(backends: Iterable[str]) -> tuple[str, ...]:
    backend_tuple = tuple(backends)
    if not backend_tuple:
        raise ValueError("scanner_backends must contain at least one backend")
    unknown = [name for name in backend_tuple if name not in SUPPORTED_SCANNER_BACKENDS]
    if unknown:
        supported = ", ".join(SUPPORTED_SCANNER_BACKENDS)
        raise ValueError(f"unknown scanner backend: {unknown[0]} (supported: {supported})")
    return backend_tuple


def _scan_backend(
    scanner: Any,
    *,
    backend: str,
    phi_min: float,
    phi_max: float,
    theta_min: float,
    theta_max: float,
    ep: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if backend == "current":
        return scanner.scan(phi_min, phi_max, theta_min, theta_max, ep)
    if backend == "reference-like":
        scan_reference_like = getattr(scanner, "scan_reference_like", None)
        if not callable(scan_reference_like):
            raise ValueError("reference-like scanner backend is unavailable")
        return scan_reference_like(phi_min, phi_max, theta_min, theta_max, ep)
    raise ValueError(f"unknown scanner backend: {backend}")


def build_config(
    *,
    crop_shape: tuple[int, int, int],
    interior_margin: int,
    count: int,
    centers: list[tuple[int, int, int]],
    explicit_centers: bool,
    percentile: float,
    min_separation: float,
    scanner_backends: tuple[str, ...],
    save_figures: bool,
    figure_percentile: float,
    ridge_buffer_radius: float,
    write_markdown_index: bool,
    visual_report_path: Path | None,
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
) -> dict[str, Any]:
    return {
        "input": "ep.dat",
        "reference": ["fv.dat", "fvt.dat"],
        "comparison": "f3d_thinning_ablation",
        "scanner_backends": list(scanner_backends),
        "cases": [dict(case) for case in CASE_DEFINITIONS],
        "crop_selection": {
            "source": "explicit_centers" if explicit_centers else "fv.dat",
            "count": int(count),
            "selected_count": len(centers),
            "crop_shape": [int(size) for size in crop_shape],
            "centers": [[int(index) for index in center] for center in centers],
            "percentile": float(percentile),
            "min_separation": float(min_separation),
            "boundary_margin": "crop_shape" if not explicit_centers else None,
        },
        "interior_margin": int(interior_margin),
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
        "overlap_percentiles": [float(p) for p in crop_validation.OVERLAP_PERCENTILES],
        "ridge_metrics": {
            "percentile": float(crop_validation.RIDGE_PERCENTILE),
            "buffer_radius": float(crop_validation.RIDGE_BUFFER_RADIUS),
        },
        "aggregate_metric_roots": list(AGGREGATE_ROOTS),
        "visualization": {
            "save_figures": bool(save_figures),
            "figure_percentile": float(figure_percentile),
            "ridge_buffer_radius": float(ridge_buffer_radius),
            "figure_slices": "center",
            "write_markdown_index": bool(write_markdown_index),
            "markdown_index": (visual_report_path.name if visual_report_path is not None else None),
        },
    }


def aggregate_backend_case_metrics(
    crops: Iterable[Mapping[str, Any]],
    *,
    scanner_backends: Iterable[str],
) -> dict[str, Any]:
    crop_list = list(crops)
    aggregate = {
        "crop_count": len(crop_list),
        "backends": {},
    }
    for backend in scanner_backends:
        backend_aggregate = {"crop_count": len(crop_list), "cases": {}}
        for case in CASE_DEFINITIONS:
            case_name = case["name"]
            case_reports = []
            for crop in crop_list:
                crop_map = _as_mapping(crop)
                backend_report = _as_mapping(crop_map.get("backends", {})).get(backend)
                if backend_report is None and backend == "current":
                    case_report = _as_mapping(crop_map.get("cases", {})).get(case_name)
                else:
                    case_report = _as_mapping(_as_mapping(backend_report).get("cases", {})).get(
                        case_name
                    )
                if isinstance(case_report, Mapping):
                    case_reports.append(case_report)
            backend_aggregate["cases"][case_name] = aggregate_crop_metrics(case_reports)
        aggregate["backends"][backend] = backend_aggregate
    return aggregate


def aggregate_case_metrics(crops: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    aggregate = aggregate_backend_case_metrics(crops, scanner_backends=DEFAULT_SCANNER_BACKENDS)
    aggregate["cases"] = aggregate["backends"]["current"]["cases"]
    return aggregate


def aggregate_crop_metrics(crops: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values_by_path: dict[str, list[float | None]] = {}
    crop_list = list(crops)
    for crop in crop_list:
        for root in AGGREGATE_ROOTS:
            if root in crop:
                for path, value in _flatten_numeric(crop[root], prefix=root):
                    values_by_path.setdefault(path, []).append(value)

    metric_paths = sorted(values_by_path)
    summaries = {
        "crop_count": len(crop_list),
        "metric_paths": metric_paths,
        "per_metric_mean": {},
        "per_metric_median": {},
        "per_metric_min": {},
        "per_metric_max": {},
    }
    for path in metric_paths:
        numeric_values = [value for value in values_by_path[path] if value is not None]
        if not numeric_values:
            for key in ("per_metric_mean", "per_metric_median", "per_metric_min", "per_metric_max"):
                summaries[key][path] = None
            continue

        values = np.asarray(numeric_values, dtype=np.float64)
        summaries["per_metric_mean"][path] = float(np.mean(values))
        summaries["per_metric_median"][path] = float(np.median(values))
        summaries["per_metric_min"][path] = float(np.min(values))
        summaries["per_metric_max"][path] = float(np.max(values))

    return summaries


def write_case_figures(
    output_dir: str | PathLike[str],
    *,
    metrics_base_dir: str | PathLike[str],
    reference_fvt: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    figure_percentile: float,
    ridge_buffer_radius: float,
) -> dict[str, Any]:
    from pyosv import viz

    directory = Path(output_dir)
    base_dir = Path(metrics_base_dir)
    slice_indices = viz.select_center_slices(np.asarray(reference_fvt).shape)
    clip_percentiles = (1.0, float(figure_percentile))
    files = {
        "fvt_ref_vs_py": crop_validation.paths_for_metrics(
            viz.save_volume_comparison_slices(
                directory,
                reference=reference_fvt,
                candidate=outputs["fvt_py.dat"],
                name="fvt_ref_vs_py",
                slice_indices=slice_indices,
                clip_percentiles=clip_percentiles,
            ),
            base_dir,
        ),
        "fvt_ridge_overlay": crop_validation.paths_for_metrics(
            viz.save_buffered_ridge_overlay_slices(
                directory,
                reference=reference_fvt,
                candidate=outputs["fvt_py.dat"],
                name="fvt",
                slice_indices=slice_indices,
                percentile=figure_percentile,
                buffer_radius=ridge_buffer_radius,
            ),
            base_dir,
        ),
        "fvt": crop_validation.paths_for_metrics(
            viz.save_volume_diagnostics(
                directory,
                reference=reference_fvt,
                candidate=outputs["fvt_py.dat"],
                name="fvt",
                clip_percentiles=clip_percentiles,
            ),
            base_dir,
        ),
    }
    return {
        "directory": crop_validation.path_for_metrics(directory, base_dir),
        "figure_slices": "center",
        "slice_indices": {axis: int(index) for axis, index in slice_indices.items()},
        "figure_percentile": float(figure_percentile),
        "ridge_buffer_radius": float(ridge_buffer_radius),
        "files": files,
    }


def figure_output_dir(
    output_base_dir: str | PathLike[str],
    *,
    crop_index: int,
    backend: str,
    case_name: str,
    backend_count: int,
) -> Path:
    crop_dir = Path(output_base_dir) / f"crop_{crop_index:03d}"
    if backend_count > 1:
        crop_dir /= backend
    return crop_dir / case_name / "figures"


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


def write_visual_report_markdown(
    report: Mapping[str, Any],
    output_path: str | PathLike[str],
) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(visual_report_markdown(report), encoding="utf-8")
    return output_file


def visual_report_markdown(report: Mapping[str, Any]) -> str:
    config = _as_mapping(report.get("config", {}))
    aggregate = _as_mapping(report.get("aggregate", {}))
    aggregate_backends = _as_mapping(aggregate.get("backends", {}))
    if not aggregate_backends and "cases" in aggregate:
        aggregate_backends = {"current": {"cases": aggregate.get("cases", {})}}
    backend_names = list(config.get("scanner_backends", aggregate_backends.keys()))
    if not backend_names:
        backend_names = ["current"]
    crops = list(report.get("crops", []))
    lines = [
        "# F3 Thinning Ablation Visual Report",
        "",
        "## Scanner Backends",
        "",
    ]
    for backend in backend_names:
        lines.append(f"- `{backend}`")

    lines.extend(
        [
            "",
            "## Cases",
            "",
        ]
    )
    for case in config.get("cases", []):
        case_map = _as_mapping(case)
        lines.append(
            "- "
            f"`{case_map.get('name', '')}`: "
            f"scanner=`{case_map.get('scanner_thin_mode', '')}`, "
            f"voter=`{case_map.get('voter_thin_mode', '')}`"
        )

    lines.extend(
        [
            "",
            "## Aggregate Metrics",
            "",
            "| Backend | Case | fvt interior corr mean | fvt buffered F1 mean/median | "
            "candidate_to_reference_median mean | reference_to_candidate_median mean | "
            "candidate_count mean |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for backend in backend_names:
        backend_aggregate = _as_mapping(aggregate_backends.get(backend, {}))
        cases = _as_mapping(backend_aggregate.get("cases", {}))
        for case_name in sorted(cases):
            case_aggregate = _as_mapping(cases[case_name])
            means = _as_mapping(case_aggregate.get("per_metric_mean", {}))
            medians = _as_mapping(case_aggregate.get("per_metric_median", {}))
            f1_mean = means.get("buffered_ridge_overlap.interior.fvt.buffered_f1")
            f1_median = medians.get("buffered_ridge_overlap.interior.fvt.buffered_f1")
            lines.append(
                "| "
                f"`{backend}` | "
                f"`{case_name}` | "
                f"{_format_metric(means.get('normalized_correlation.interior.fvt'))} | "
                f"{_format_metric(f1_mean)} / {_format_metric(f1_median)} | "
                f"{_format_metric(means.get(_candidate_to_reference_path()))} | "
                f"{_format_metric(means.get(_reference_to_candidate_path()))} | "
                f"{_format_metric(means.get(_candidate_count_path()))} |"
            )

    lines.extend(
        [
            "",
            "## Crop Case Metrics",
            "",
            "| Crop | Center | Backend | Case | scanner thin | voter thin | interior fvt corr | "
            "buffered F1 | reference ridges | candidate ridges | Key figures |",
            "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for crop in crops:
        crop_map = _as_mapping(crop)
        crop_id = f"crop_{int(crop_map.get('index', 0)):03d}"
        crop_backends = _as_mapping(crop_map.get("backends", {}))
        if not crop_backends and "cases" in crop_map:
            crop_backends = {"current": {"cases": crop_map.get("cases", {})}}
        for backend in backend_names:
            backend_report = _as_mapping(crop_backends.get(backend, {}))
            cases = _as_mapping(backend_report.get("cases", {}))
            for case_name in sorted(cases):
                case_report = _as_mapping(cases[case_name])
                case_config = _as_mapping(case_report.get("case", {}))
                overlap = _as_mapping(
                    _nested(case_report, "buffered_ridge_overlap", "interior", "fvt")
                )
                links = ", ".join(
                    f"[{label}]({path})" for label, path in _important_figure_links(case_report)
                )
                lines.append(
                    "| "
                    f"{crop_id} | "
                    f"`{crop_map.get('crop_center', '')}` | "
                    f"`{backend}` | "
                    f"`{case_name}` | "
                    f"`{case_config.get('scanner_thin_mode', '')}` | "
                    f"`{case_config.get('voter_thin_mode', '')}` | "
                    f"{_format_metric(_nested(case_report, 'normalized_correlation', 'interior', 'fvt'))} | "
                    f"{_format_metric(overlap.get('buffered_f1'))} | "
                    f"{_format_metric(overlap.get('reference_count'))} | "
                    f"{_format_metric(overlap.get('candidate_count'))} | "
                    f"{links} |"
                )

    return "\n".join(lines) + "\n"


def _candidate_to_reference_path() -> str:
    return "sparse_ridge_distance_metrics.interior.fvt.candidate_to_reference_median"


def _reference_to_candidate_path() -> str:
    return "sparse_ridge_distance_metrics.interior.fvt.reference_to_candidate_median"


def _candidate_count_path() -> str:
    return "sparse_ridge_distance_metrics.interior.fvt.candidate_count"


def report_to_json(report: Mapping[str, Any], *, pretty: bool = False) -> str:
    indent = 2 if pretty else None
    return json.dumps(_json_compatible(report), indent=indent, sort_keys=True) + "\n"


def ensure_output_path_allowed(
    output_path: str | PathLike[str],
    data_root: str | PathLike[str],
    *,
    option_name: str,
) -> None:
    resolved_output = Path(output_path).resolve(strict=False)
    for forbidden_root, label in (
        (Path(data_root), "F3 data root"),
        (REFERENCE_OSV_DIR, "reference_osv"),
    ):
        resolved_forbidden = forbidden_root.resolve(strict=False)
        try:
            resolved_output.relative_to(resolved_forbidden)
        except ValueError:
            continue
        raise ValueError(f"{option_name} must not be inside {label}: {resolved_output}")


def _flatten_numeric(value: Any, *, prefix: str) -> Iterable[tuple[str, float | None]]:
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _flatten_numeric(value[key], prefix=f"{prefix}.{key}")
        return

    if value is None:
        yield prefix, None
        return

    if isinstance(value, bool):
        return

    if isinstance(value, int | float | np.generic):
        numeric_value = float(value)
        yield prefix, numeric_value if math.isfinite(numeric_value) else None


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _format_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int | float | np.generic):
        numeric_value = float(value)
        return f"{numeric_value:.4g}" if math.isfinite(numeric_value) else ""
    return str(value)


def _important_figure_links(case_report: Mapping[str, Any]) -> list[tuple[str, str]]:
    files = _as_mapping(_nested(case_report, "figures", "files"))
    candidates = (
        ("fvt i3", ("fvt_ref_vs_py", "i3")),
        ("ridge i3", ("fvt_ridge_overlay", "i3")),
        ("fvt MIP", ("fvt", "mip")),
    )
    links: list[tuple[str, str]] = []
    for label, path_keys in candidates:
        path = _nested(files, *path_keys)
        if isinstance(path, str):
            links.append((label, path))
    return links


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _crop(array: np.ndarray, slices: tuple[slice, slice, slice]) -> np.ndarray:
    return np.ascontiguousarray(array[slices].astype(np.float32, copy=False))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = run_example(
            data_root_arg=args.data_root,
            output_json=args.output_json,
            save_figures=args.save_figures,
            figure_percentile=args.figure_percentile,
            ridge_buffer_radius=args.ridge_buffer_radius,
            write_markdown_index=args.write_markdown_index,
            pretty=args.pretty,
            count=args.count,
            crop_shape=args.crop_shape,
            interior_margin=args.interior_margin,
            centers=args.center,
            percentile=args.percentile,
            min_separation=args.min_separation,
            scanner_backends=args.scanner_backends,
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
