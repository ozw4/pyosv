"""Report F3 crop seed-distribution diagnostics before final voting."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt

import run_3d_f3d_crop_validation as crop_validation
from pyosv.cells import FaultCell
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
DEFAULT_CENTER_PERCENTILE = 99.9
DEFAULT_MIN_SEPARATION = 48.0
DEFAULT_REFERENCE_PERCENTILE = 99.0
REFERENCE_OSV_DIR = Path(__file__).resolve().parents[1] / "reference_osv"
SCANNER_BACKENDS = ("current", "reference-like")
SCANNER_THIN_MODES = ("normal", "reference")
AGGREGATE_ROOTS = ("seed_diagnostics",)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run F3 crop diagnostics for seed coverage before final voting.",
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
        help="Write per-crop, per-case seed overlay PNG diagnostics.",
    )
    parser.add_argument(
        "--figure-percentile",
        type=float,
        default=99.0,
        help="Upper display clipping percentile for seed overlay figures.",
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
        "--center",
        action="append",
        type=crop_validation.parse_index3,
        default=None,
        help="Explicit crop center in i3,i2,i1 order. May be repeated.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_CENTER_PERCENTILE,
        help="Reference fv percentile used to pick crop centers.",
    )
    parser.add_argument(
        "--reference-percentile",
        type=float,
        default=DEFAULT_REFERENCE_PERCENTILE,
        help="Reference fv/fvt percentile used to build high masks.",
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=DEFAULT_MIN_SEPARATION,
        help="Minimum deterministic center separation in samples.",
    )
    parser.add_argument(
        "--scanner-backends",
        type=parse_scanner_backends,
        default=SCANNER_BACKENDS,
        help='Comma-separated scanner backends: "current", "reference-like".',
    )
    parser.add_argument(
        "--scanner-thin-modes",
        type=parse_scanner_thin_modes,
        default=SCANNER_THIN_MODES,
        help='Comma-separated scanner thinning modes: "normal", "reference".',
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
    parser.add_argument("--d", type=int, default=4, help="Seed exclusion distance.")
    parser.add_argument("--fm", type=float, default=0.3, help="Minimum seed likelihood.")
    parser.add_argument(
        "--reference-thin-sigma",
        type=float,
        default=1.0,
        help="Smoothing sigma used by reference-like thinning.",
    )
    return parser


def parse_scanner_backends(text: str) -> tuple[str, ...]:
    return _parse_csv_choices(text, choices=SCANNER_BACKENDS, option_name="scanner-backends")


def parse_scanner_thin_modes(text: str) -> tuple[str, ...]:
    return _parse_csv_choices(text, choices=SCANNER_THIN_MODES, option_name="scanner-thin-modes")


def run_example(
    *,
    data_root_arg: str | PathLike[str] | None,
    output_json: str | PathLike[str] | None = None,
    save_figures: bool = False,
    figure_percentile: float = 99.0,
    write_markdown_index: bool = False,
    pretty: bool = False,
    count: int = DEFAULT_COUNT,
    crop_shape: tuple[int, int, int] = DEFAULT_CROP_SHAPE,
    interior_margin: int = DEFAULT_INTERIOR_MARGIN,
    centers: Iterable[tuple[int, int, int]] | None = None,
    percentile: float = DEFAULT_CENTER_PERCENTILE,
    reference_percentile: float = DEFAULT_REFERENCE_PERCENTILE,
    min_separation: float = DEFAULT_MIN_SEPARATION,
    scanner_backends: Sequence[str] = SCANNER_BACKENDS,
    scanner_thin_modes: Sequence[str] = SCANNER_THIN_MODES,
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
    reference_thin_sigma: float = 1.0,
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
    selected_backends = validate_scanner_backends(scanner_backends)
    selected_thin_modes = validate_scanner_thin_modes(scanner_thin_modes)
    case_definitions = build_case_definitions(selected_backends, selected_thin_modes)

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
        reference_percentile=reference_percentile,
        min_separation=min_separation,
        scanner_backends=selected_backends,
        scanner_thin_modes=selected_thin_modes,
        case_definitions=case_definitions,
        save_figures=save_figures,
        figure_percentile=figure_percentile,
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
        d=d,
        fm=fm,
        reference_thin_sigma=reference_thin_sigma,
    )

    crops = []
    for crop_index, center in enumerate(selected_centers, start=1):
        slices = crop_slices(center, crop_shape, full_shape=arrays["ep.dat"].shape)
        ep_crop = _crop(arrays["ep.dat"], slices)
        reference_fv = _crop(arrays["fv.dat"], slices)
        reference_fvt = _crop(arrays["fvt.dat"], slices)
        case_outputs = run_seed_diagnostic_pipeline(
            ep_crop,
            scanner_backends=selected_backends,
            scanner_thin_modes=selected_thin_modes,
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
            reference_thin_sigma=reference_thin_sigma,
        )

        local_interior_slices = crop_validation.interior_slices(
            ep_crop.shape,
            margin=interior_margin,
        )
        global_interior_slices = tuple(
            slice(crop_slice.start + local_slice.start, crop_slice.start + local_slice.stop)
            for crop_slice, local_slice in zip(slices, local_interior_slices, strict=True)
        )
        case_reports: dict[str, Any] = {}
        for case in case_definitions:
            case_name = case["name"]
            outputs = case_outputs[case_name]
            seed_mask = seeds_to_mask(outputs["seeds"], ep_crop.shape)
            diagnostics = seed_diagnostics(
                outputs["seeds"],
                seed_mask=seed_mask,
                fet=outputs["fet_py.dat"],
                reference_fv=reference_fv,
                reference_fvt=reference_fvt,
                reference_percentile=reference_percentile,
            )
            case_report: dict[str, Any] = {
                "case": dict(case),
                "seed_diagnostics": diagnostics,
            }
            if save_figures:
                if output_base_dir is None:
                    raise ValueError("--save-figures requires --output-json")
                case_report["figures"] = write_case_figures(
                    output_base_dir / f"crop_{crop_index:03d}" / case_name / "figures",
                    metrics_base_dir=output_base_dir,
                    seeds=outputs["seeds"],
                    fet=outputs["fet_py.dat"],
                    reference_fv=reference_fv,
                    reference_fvt=reference_fvt,
                    reference_percentile=reference_percentile,
                    figure_percentile=figure_percentile,
                )
            case_reports[case_name] = case_report

        crops.append(
            {
                "index": int(crop_index),
                "crop_center": [int(value) for value in center],
                "crop_slices": crop_validation.slices_to_json(slices),
                "interior_margin": int(interior_margin),
                "interior_slices": crop_validation.slices_to_json(global_interior_slices),
                "interior_slices_in_crop": crop_validation.slices_to_json(local_interior_slices),
                "crop_shape": [int(size) for size in ep_crop.shape],
                "cases": case_reports,
            }
        )

    report = _json_compatible(
        {
            "format_version": 1,
            "data_root": str(data_root),
            "config": config,
            "crops": crops,
            "aggregate": aggregate_case_metrics(crops, case_definitions=case_definitions),
        }
    )

    if output_json is not None:
        write_report_json(report, output_json, pretty=pretty)
        if write_markdown_index:
            write_visual_report_markdown(report, Path(output_json).parent / "visual_report.md")

    return report


def run_seed_diagnostic_pipeline(
    ep: np.ndarray,
    *,
    scanner_backends: Sequence[str],
    scanner_thin_modes: Sequence[str],
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
    reference_thin_sigma: float,
) -> dict[str, dict[str, Any]]:
    from pyosv.orient3d import FaultOrientScanner3
    from pyosv.voting3d import OptimalSurfaceVoter

    scanner = FaultOrientScanner3(sigma1=sigma1, sigma2=sigma2)
    voter = OptimalSurfaceVoter(ru=ru, rv=rv, rw=rw)

    scanned_by_backend: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for backend in validate_scanner_backends(scanner_backends):
        scanned_by_backend[backend] = _scan_backend(
            scanner,
            backend=backend,
            phi_min=phi_min,
            phi_max=phi_max,
            theta_min=theta_min,
            theta_max=theta_max,
            ep=ep,
        )

    outputs: dict[str, dict[str, Any]] = {}
    for case in build_case_definitions(
        validate_scanner_backends(scanner_backends),
        validate_scanner_thin_modes(scanner_thin_modes),
    ):
        backend = case["scanner_backend"]
        thin_mode = case["scanner_thin_mode"]
        ft, pt, tt = scanned_by_backend[backend]
        fet, fpt, ftt = scanner.thin(
            ft,
            pt,
            tt,
            mode=thin_mode,
            reference_sigma=reference_thin_sigma,
        )
        seeds = voter.pick_seeds(d=d, fm=fm, ft=fet, pt=fpt, tt=ftt)
        outputs[case["name"]] = {
            "ft_py.dat": ft,
            "pt_py.dat": pt,
            "tt_py.dat": tt,
            "fet_py.dat": fet,
            "fpt_py.dat": fpt,
            "ftt_py.dat": ftt,
            "seeds": seeds,
        }

    return outputs


def seed_diagnostics(
    seeds: Sequence[FaultCell],
    *,
    seed_mask: np.ndarray,
    fet: np.ndarray,
    reference_fv: np.ndarray,
    reference_fvt: np.ndarray,
    reference_percentile: float,
) -> dict[str, Any]:
    fet_values = np.asarray(fet, dtype=np.float32)
    if fet_values.shape != seed_mask.shape:
        raise ValueError("fet and seed_mask shapes must match")
    reference_fv_values = np.asarray(reference_fv, dtype=np.float32)
    reference_fvt_values = np.asarray(reference_fvt, dtype=np.float32)
    if (
        reference_fv_values.shape != seed_mask.shape
        or reference_fvt_values.shape != seed_mask.shape
    ):
        raise ValueError("reference arrays and seed_mask shapes must match")

    high_fv = reference_high_mask(reference_fv_values, percentile=reference_percentile)
    high_fvt = reference_high_mask(reference_fvt_values, percentile=reference_percentile)
    likelihood = seed_likelihood_summary(seeds)
    seed_count = int(np.count_nonzero(seed_mask))
    size = int(seed_mask.size)

    return {
        "seed_count": seed_count,
        "seed_density": float(seed_count / size) if size else 0.0,
        "seed_likelihood_min": likelihood["min"],
        "seed_likelihood_max": likelihood["max"],
        "seed_likelihood_mean": likelihood["mean"],
        "seed_likelihood_percentiles": {
            "p50": likelihood["p50"],
            "p90": likelihood["p90"],
            "p95": likelihood["p95"],
            "p99": likelihood["p99"],
        },
        "fet_nonzero_fraction": float(np.count_nonzero(fet_values != 0.0) / fet_values.size)
        if fet_values.size
        else 0.0,
        "fet_mean": float(np.mean(fet_values.astype(np.float64, copy=False)))
        if fet_values.size
        else None,
        "fet_max": float(np.max(fet_values)) if fet_values.size else None,
        "reference_high_percentile": float(reference_percentile),
        "reference_high_counts": {
            "fv": int(np.count_nonzero(high_fv)),
            "fvt": int(np.count_nonzero(high_fvt)),
        },
        "distance": {
            "reference_high_fv_to_seed": mask_distance_summary(
                source_mask=high_fv,
                target_mask=seed_mask,
                source_name="reference",
                target_name="seed",
            ),
            "reference_high_fvt_to_seed": mask_distance_summary(
                source_mask=high_fvt,
                target_mask=seed_mask,
                source_name="reference",
                target_name="seed",
            ),
            "seed_to_reference_high_fv": mask_distance_summary(
                source_mask=seed_mask,
                target_mask=high_fv,
                source_name="seed",
                target_name="reference",
            ),
            "seed_to_reference_high_fvt": mask_distance_summary(
                source_mask=seed_mask,
                target_mask=high_fvt,
                source_name="seed",
                target_name="reference",
            ),
        },
    }


def seeds_to_mask(seeds: Sequence[FaultCell], shape: tuple[int, int, int]) -> np.ndarray:
    if len(shape) != 3:
        raise ValueError("shape must be a 3D (n3, n2, n1) tuple")
    mask = np.zeros(shape, dtype=bool)
    n3, n2, n1 = shape
    for seed in seeds:
        i1, i2, i3 = int(seed.i1), int(seed.i2), int(seed.i3)
        if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
            raise ValueError(f"seed index {(i1, i2, i3)} is outside shape {shape}")
        mask[i3, i2, i1] = True
    return mask


def reference_high_mask(reference: np.ndarray, *, percentile: float) -> np.ndarray:
    values = np.asarray(reference)
    if values.size == 0:
        raise ValueError("reference array must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("reference array must contain only finite values")
    if not np.isfinite(percentile) or percentile < 0.0 or percentile > 100.0:
        raise ValueError("percentile must be finite and between 0 and 100")
    threshold = float(np.percentile(values.astype(np.float64, copy=False), percentile))
    return values >= threshold


def mask_distance_summary(
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    source_name: str,
    target_name: str,
) -> dict[str, float | int | None]:
    source = np.asarray(source_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    if source.shape != target.shape:
        raise ValueError("source and target masks must have matching shapes")

    source_count = int(np.count_nonzero(source))
    target_count = int(np.count_nonzero(target))
    result: dict[str, float | int | None] = {
        f"{source_name}_count": source_count,
        f"{target_name}_count": target_count,
        "mean": None,
        "median": None,
        "p90": None,
        "p95": None,
    }
    if source_count == 0 or target_count == 0:
        return result

    distances = distance_transform_edt(~target)[source].astype(np.float64, copy=False)
    result.update(
        {
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "p90": float(np.percentile(distances, 90.0)),
            "p95": float(np.percentile(distances, 95.0)),
        }
    )
    return result


def seed_likelihood_summary(seeds: Sequence[FaultCell]) -> dict[str, float | None]:
    if not seeds:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
        }
    values = np.asarray([seed.fl for seed in seeds], dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50.0)),
        "p90": float(np.percentile(values, 90.0)),
        "p95": float(np.percentile(values, 95.0)),
        "p99": float(np.percentile(values, 99.0)),
    }


def build_case_definitions(
    scanner_backends: Sequence[str],
    scanner_thin_modes: Sequence[str],
) -> tuple[dict[str, str], ...]:
    backends = validate_scanner_backends(scanner_backends)
    thin_modes = validate_scanner_thin_modes(scanner_thin_modes)
    return tuple(
        {
            "name": f"{backend}_{thin_mode}",
            "scanner_backend": backend,
            "scanner_thin_mode": thin_mode,
        }
        for backend in backends
        for thin_mode in thin_modes
    )


def validate_scanner_backends(backends: Sequence[str]) -> tuple[str, ...]:
    return _validate_choices(backends, choices=SCANNER_BACKENDS, option_name="scanner_backends")


def validate_scanner_thin_modes(modes: Sequence[str]) -> tuple[str, ...]:
    return _validate_choices(modes, choices=SCANNER_THIN_MODES, option_name="scanner_thin_modes")


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


def build_config(
    *,
    crop_shape: tuple[int, int, int],
    interior_margin: int,
    count: int,
    centers: list[tuple[int, int, int]],
    explicit_centers: bool,
    percentile: float,
    reference_percentile: float,
    min_separation: float,
    scanner_backends: Sequence[str],
    scanner_thin_modes: Sequence[str],
    case_definitions: Sequence[Mapping[str, str]],
    save_figures: bool,
    figure_percentile: float,
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
    d: int,
    fm: float,
    reference_thin_sigma: float,
) -> dict[str, Any]:
    return {
        "input": "ep.dat",
        "reference": ["fv.dat", "fvt.dat"],
        "comparison": "f3d_seed_diagnostics",
        "cases": [dict(case) for case in case_definitions],
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
            "backends": list(scanner_backends),
            "thin_modes": list(scanner_thin_modes),
            "reference_thin_sigma": float(reference_thin_sigma),
        },
        "voter_seed_picker": {
            "ru": int(ru),
            "rv": int(rv),
            "rw": int(rw),
            "d": int(d),
            "fm": float(fm),
        },
        "reference_high_percentile": float(reference_percentile),
        "aggregate_metric_roots": list(AGGREGATE_ROOTS),
        "visualization": {
            "save_figures": bool(save_figures),
            "figure_percentile": float(figure_percentile),
            "figure_slices": "center",
            "write_markdown_index": bool(write_markdown_index),
            "markdown_index": (visual_report_path.name if visual_report_path is not None else None),
        },
    }


def aggregate_case_metrics(
    crops: Iterable[Mapping[str, Any]],
    *,
    case_definitions: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    crop_list = list(crops)
    aggregate = {
        "crop_count": len(crop_list),
        "cases": {},
    }
    for case in case_definitions:
        case_name = case["name"]
        reports = [
            _as_mapping(_as_mapping(crop).get("cases", {})).get(case_name) for crop in crop_list
        ]
        aggregate["cases"][case_name] = aggregate_crop_metrics(
            report for report in reports if isinstance(report, Mapping)
        )
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
    seeds: Sequence[FaultCell],
    fet: np.ndarray,
    reference_fv: np.ndarray,
    reference_fvt: np.ndarray,
    reference_percentile: float,
    figure_percentile: float,
) -> dict[str, Any]:
    from pyosv import viz

    directory = Path(output_dir)
    base_dir = Path(metrics_base_dir)
    slice_indices = viz.select_center_slices(np.asarray(fet).shape)
    high_fv = reference_high_mask(reference_fv, percentile=reference_percentile).astype(np.float32)
    high_fvt = reference_high_mask(reference_fvt, percentile=reference_percentile).astype(
        np.float32
    )
    files = {
        "fet_seed_overlay": paths_for_metrics(
            save_seed_overlay_slices(
                directory,
                volume=fet,
                seeds=seeds,
                name="fet_seed_overlay",
                slice_indices=slice_indices,
                clip_percentiles=(1.0, float(figure_percentile)),
            ),
            base_dir,
        ),
        "reference_fv_high_seed_overlay": paths_for_metrics(
            save_seed_overlay_slices(
                directory,
                volume=high_fv,
                seeds=seeds,
                name="reference_fv_high_seed_overlay",
                slice_indices=slice_indices,
                clip_percentiles=(0.0, 100.0),
            ),
            base_dir,
        ),
        "reference_fvt_high_seed_overlay": paths_for_metrics(
            save_seed_overlay_slices(
                directory,
                volume=high_fvt,
                seeds=seeds,
                name="reference_fvt_high_seed_overlay",
                slice_indices=slice_indices,
                clip_percentiles=(0.0, 100.0),
            ),
            base_dir,
        ),
    }
    return {
        "directory": path_for_metrics(directory, base_dir),
        "figure_slices": "center",
        "slice_indices": {axis: int(index) for axis, index in slice_indices.items()},
        "figure_percentile": float(figure_percentile),
        "reference_high_percentile": float(reference_percentile),
        "files": files,
    }


def save_seed_overlay_slices(
    output_dir: str | PathLike[str],
    *,
    volume: np.ndarray,
    seeds: Sequence[FaultCell],
    name: str,
    slice_indices: Mapping[str, int],
    clip_percentiles: tuple[float, float],
) -> dict[str, Path]:
    values = np.asarray(volume, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("volume must be a 3D (n3, n2, n1) array")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for axis in ("i3", "i2", "i1"):
        index = int(slice_indices[axis])
        path = directory / f"{name}_{axis}_{index}.png"
        written[axis] = save_seed_overlay_slice(
            path,
            volume=values,
            seeds=seeds,
            axis=axis,
            index=index,
            clip_percentiles=clip_percentiles,
            title=f"{name} {axis}={index}",
        )
    return written


def save_seed_overlay_slice(
    output_path: str | PathLike[str],
    *,
    volume: np.ndarray,
    seeds: Sequence[FaultCell],
    axis: str,
    index: int,
    clip_percentiles: tuple[float, float],
    title: str,
) -> Path:
    from pyosv import viz

    values = np.asarray(volume, dtype=np.float32)
    panel = viz.slice_2d(values, axis, index)
    display = viz.normalize_for_display(panel, clip_percentiles=clip_percentiles)
    x, y = _seed_slice_coordinates(seeds, axis=axis, index=index)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt = viz.require_matplotlib()
    fig, ax = plt.subplots(figsize=(4.0, 4.0), constrained_layout=True)
    try:
        ax.imshow(display, cmap="gray", vmin=0.0, vmax=1.0, origin="upper", aspect="auto")
        if x.size:
            ax.scatter(x, y, s=18, c="#ffcc00", edgecolors="#111111", linewidths=0.4)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(output_file, dpi=150)
    finally:
        plt.close(fig)
    return output_file


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
    crops = list(report.get("crops", []))
    lines = [
        "# F3 Seed Diagnostics Visual Report",
        "",
        "## Cases",
        "",
    ]
    for case in config.get("cases", []):
        case_map = _as_mapping(case)
        lines.append(
            "- "
            f"`{case_map.get('name', '')}`: "
            f"backend=`{case_map.get('scanner_backend', '')}`, "
            f"scanner_thin=`{case_map.get('scanner_thin_mode', '')}`"
        )

    lines.extend(
        [
            "",
            "## Crop Case Metrics",
            "",
            "| Crop | Center | Case | Seeds | Density | ref fv to seed p95 | "
            "seed to ref fvt p95 | Key figures |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for crop in crops:
        crop_map = _as_mapping(crop)
        crop_id = f"crop_{int(crop_map.get('index', 0)):03d}"
        cases = _as_mapping(crop_map.get("cases", {}))
        for case_name in sorted(cases):
            case_report = _as_mapping(cases[case_name])
            diagnostics = _as_mapping(case_report.get("seed_diagnostics", {}))
            links = ", ".join(
                f"[{label}]({path})" for label, path in _important_figure_links(case_report)
            )
            lines.append(
                "| "
                f"{crop_id} | "
                f"`{crop_map.get('crop_center', '')}` | "
                f"`{case_name}` | "
                f"{_format_metric(diagnostics.get('seed_count'))} | "
                f"{_format_metric(diagnostics.get('seed_density'))} | "
                f"{_format_metric(_nested(diagnostics, 'distance', 'reference_high_fv_to_seed', 'p95'))} | "
                f"{_format_metric(_nested(diagnostics, 'distance', 'seed_to_reference_high_fvt', 'p95'))} | "
                f"{links} |"
            )

    return "\n".join(lines) + "\n"


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


def paths_for_metrics(paths: Mapping[str, Path], base_dir: Path) -> dict[str, str]:
    return {key: path_for_metrics(path, base_dir) for key, path in paths.items()}


def path_for_metrics(path: str | PathLike[str], base_dir: str | PathLike[str]) -> str:
    output_path = Path(path)
    resolved_base = Path(base_dir).resolve(strict=False)
    try:
        return output_path.resolve(strict=False).relative_to(resolved_base).as_posix()
    except ValueError:
        return output_path.as_posix()


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


def _parse_csv_choices(text: str, *, choices: Sequence[str], option_name: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise TypeError(f"{option_name} must be a string")
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    return _validate_choices(values, choices=choices, option_name=option_name)


def _validate_choices(
    values: Sequence[str],
    *,
    choices: Sequence[str],
    option_name: str,
) -> tuple[str, ...]:
    if not values:
        raise ValueError(f"{option_name} must contain at least one value")
    invalid = [value for value in values if value not in choices]
    if invalid:
        joined = ", ".join(choices)
        raise ValueError(f"{option_name} must contain only {joined}; got {invalid[0]}")
    deduped = tuple(dict.fromkeys(values))
    return deduped


def _seed_slice_coordinates(
    seeds: Sequence[FaultCell],
    *,
    axis: str,
    index: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates: list[tuple[int, int]] = []
    for seed in seeds:
        i1, i2, i3 = int(seed.i1), int(seed.i2), int(seed.i3)
        if axis == "i3" and i3 == index:
            coordinates.append((i1, i2))
        elif axis == "i2" and i2 == index:
            coordinates.append((i1, i3))
        elif axis == "i1" and i1 == index:
            coordinates.append((i2, i3))
    if not coordinates:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    values = np.asarray(coordinates, dtype=np.float32)
    return values[:, 0], values[:, 1]


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


def _important_figure_links(case_report: Mapping[str, Any]) -> list[tuple[str, str]]:
    files = _as_mapping(_nested(case_report, "figures", "files"))
    candidates = (
        ("fet i3", ("fet_seed_overlay", "i3")),
        ("fv high i3", ("reference_fv_high_seed_overlay", "i3")),
        ("fvt high i3", ("reference_fvt_high_seed_overlay", "i3")),
    )
    links: list[tuple[str, str]] = []
    for label, path_keys in candidates:
        path = _nested(files, *path_keys)
        if isinstance(path, str):
            links.append((label, path))
    return links


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
            write_markdown_index=args.write_markdown_index,
            pretty=args.pretty,
            count=args.count,
            crop_shape=args.crop_shape,
            interior_margin=args.interior_margin,
            centers=args.center,
            percentile=args.percentile,
            reference_percentile=args.reference_percentile,
            min_separation=args.min_separation,
            scanner_backends=args.scanner_backends,
            scanner_thin_modes=args.scanner_thin_modes,
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
            reference_thin_sigma=args.reference_thin_sigma,
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
