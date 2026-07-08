"""Run deterministic multi-crop F3 scan/vote validation.

The JSON schema has three top-level sections:

``config``
    CLI/runtime settings, including crop selection mode and volume output policy.
``crops``
    Per-crop reports from ``run_3d_f3d_crop_validation.build_crop_report``.
``aggregate``
    Deterministic flattened metric summaries. Each ``per_metric_*`` mapping is
    keyed by a dotted path such as
    ``normalized_correlation.interior.fv`` or
    ``sparse_ridge_distance_metrics.interior.fvt.candidate_to_reference_median``.
    Empty sparse-mask distance values are reported as ``None``.
``consensus``
    Workflow-level truthless crop-stability summaries derived from the saved
    crop metrics. Compare reports also include quality-minus-reference
    consensus deltas.
"""

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
DEFAULT_CROP_SHAPE = (128, 128, 100)
DEFAULT_INTERIOR_MARGIN = 40
DEFAULT_PERCENTILE = 99.9
DEFAULT_MIN_SEPARATION = 48.0
AGGREGATE_ROOTS = (
    "normalized_correlation",
    "top_percentile_overlap",
    "buffered_ridge_overlap",
    "sparse_ridge_distance_metrics",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pyosv 3D F3 crop scan/vote workflow on multiple "
            "deterministic crops and report aggregate practical metrics."
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
    crop_validation.add_workflow_arguments(parser)
    parser.add_argument(
        "--compare-workflows",
        action="store_true",
        help="Run reference and quality workflows on the same selected crop centers.",
    )
    parser.add_argument(
        "--save-volumes",
        action="store_true",
        help="Write per-crop pyosv DAT volumes. Requires --volume-dir or --output-json.",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Write per-crop PNG diagnostics under OUTPUT_JSON.parent.",
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
    parser.add_argument(
        "--volume-dir",
        type=Path,
        default=None,
        help="Directory for crop DAT outputs. Defaults to OUTPUT_JSON.parent/volumes.",
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
    crop_validation.add_final_normalization_smoothing_argument(parser)
    crop_validation.add_surface_support_arguments(parser)
    parser.add_argument("--d", type=int, default=4, help="Seed exclusion distance.")
    parser.add_argument("--fm", type=float, default=0.3, help="Minimum seed likelihood.")
    crop_validation.add_thinning_arguments(parser)
    return parser


def run_example(
    *,
    data_root_arg: str | PathLike[str] | None,
    output_json: str | PathLike[str] | None = None,
    save_volumes: bool = False,
    save_figures: bool = False,
    figure_percentile: float = 99.0,
    ridge_buffer_radius: float = 2.0,
    write_markdown_index: bool = False,
    volume_dir: str | PathLike[str] | None = None,
    pretty: bool = False,
    count: int = DEFAULT_COUNT,
    crop_shape: tuple[int, int, int] = DEFAULT_CROP_SHAPE,
    interior_margin: int = DEFAULT_INTERIOR_MARGIN,
    centers: Iterable[tuple[int, int, int]] | None = None,
    percentile: float = DEFAULT_PERCENTILE,
    min_separation: float = DEFAULT_MIN_SEPARATION,
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
    final_normalization_smoothing: float | None = None,
    d: int = 4,
    fm: float = 0.3,
    scanner_thin_mode: str = "reference",
    voter_thin_mode: str | None = None,
    reference_thin_sigma: float = 1.0,
    remove_scanner_edge_effects: bool = True,
    workflow_mode: str = "reference",
    compare_workflows: bool = False,
    surface_support_min_fraction: float | None = None,
    surface_support_exponent: float | None = None,
) -> dict[str, Any]:
    data_root = resolve_f3d_data_root(data_root_arg)
    if output_json is not None:
        ensure_output_not_in_data_root(output_json, data_root, option_name="--output-json")
    elif save_figures:
        raise ValueError("--save-figures requires --output-json")
    elif write_markdown_index:
        raise ValueError("--write-markdown-index requires --output-json")

    output_base_dir = Path(output_json).parent if output_json is not None else None
    if save_figures:
        crop_validation.require_figure_support()

    resolved_volume_dir = resolve_volume_dir(
        output_json=output_json,
        volume_dir=volume_dir,
        save_volumes=save_volumes,
    )
    if resolved_volume_dir is not None:
        ensure_output_not_in_data_root(resolved_volume_dir, data_root, option_name="--volume-dir")

    if count < 0:
        raise ValueError("count must be >= 0")
    crop_shape, interior_margin = validate_crop_config(crop_shape, interior_margin)

    arrays = crop_validation.read_reference_arrays(data_root)
    selected_centers = select_centers(
        arrays["fv.dat"],
        count=count,
        centers=centers,
        percentile=percentile,
        min_separation=min_separation,
        crop_shape=crop_shape,
    )
    if save_figures and "fl.dat" not in arrays:
        arrays["fl.dat"] = crop_validation.read_f3d_file("fl.dat", data_root)

    workflow_names = ("reference", "quality") if compare_workflows else (workflow_mode,)
    workflow_reports: dict[str, Any] = {}
    for workflow_name in workflow_names:
        workflow_config, workflow_crops = run_workflow_crops(
            arrays=arrays,
            selected_centers=selected_centers,
            crop_shape=crop_shape,
            interior_margin=interior_margin,
            output_base_dir=output_base_dir,
            resolved_volume_dir=resolved_volume_dir,
            save_volumes=save_volumes,
            save_figures=save_figures,
            figure_percentile=figure_percentile,
            ridge_buffer_radius=ridge_buffer_radius,
            count=count,
            centers=selected_centers,
            explicit_centers=centers is not None,
            percentile=percentile,
            min_separation=min_separation,
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
            final_normalization_smoothing=final_normalization_smoothing,
            d=d,
            fm=fm,
            scanner_thin_mode=scanner_thin_mode,
            voter_thin_mode=voter_thin_mode,
            reference_thin_sigma=reference_thin_sigma,
            remove_scanner_edge_effects=remove_scanner_edge_effects,
            surface_support_min_fraction=surface_support_min_fraction,
            surface_support_exponent=surface_support_exponent,
            workflow_mode=workflow_name,
            compare_workflows=compare_workflows,
            write_markdown_index=write_markdown_index,
        )
        workflow_reports[workflow_name] = {
            "config": workflow_config,
            "crops": workflow_crops,
            "aggregate": aggregate_crop_metrics(workflow_crops),
        }

    consensus_workflows = {
        workflow_name: build_consensus_summary(workflow_report["crops"])
        for workflow_name, workflow_report in workflow_reports.items()
    }
    consensus: dict[str, Any] = {"workflows": consensus_workflows}
    if compare_workflows:
        consensus["workflow_comparison"] = {
            "quality_minus_reference": build_consensus_delta(
                consensus_workflows["reference"],
                consensus_workflows["quality"],
            )
        }

    if compare_workflows:
        config = dict(workflow_reports["reference"]["config"])
        config["compare_workflows"] = True
        config["workflow_modes"] = list(workflow_names)
        report_content: dict[str, Any] = {
            "format_version": 2,
            "data_root": str(data_root),
            "config": config,
            "workflows": workflow_reports,
            "consensus": consensus,
            "workflow_delta": {
                "quality_vs_reference": aggregate_delta(
                    workflow_reports["reference"]["aggregate"],
                    workflow_reports["quality"]["aggregate"],
                )
            },
        }
    else:
        single = workflow_reports[workflow_mode]
        report_content = {
            "format_version": 2,
            "data_root": str(data_root),
            "config": single["config"],
            "crops": single["crops"],
            "aggregate": single["aggregate"],
            "consensus": consensus,
        }

    report = _json_compatible(report_content)

    if output_json is not None:
        write_report_json(report, output_json, pretty=pretty)
        if write_markdown_index:
            write_visual_report_markdown(report, Path(output_json).parent / "visual_report.md")

    return report


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


def resolve_volume_dir(
    *,
    output_json: str | PathLike[str] | None,
    volume_dir: str | PathLike[str] | None,
    save_volumes: bool,
) -> Path | None:
    if not save_volumes:
        return None
    if volume_dir is not None:
        return Path(volume_dir)
    if output_json is not None:
        return Path(output_json).parent / "volumes"
    raise ValueError("--save-volumes requires --volume-dir or --output-json")


def run_workflow_crops(
    *,
    arrays: Mapping[str, np.ndarray],
    selected_centers: list[tuple[int, int, int]],
    crop_shape: tuple[int, int, int],
    interior_margin: int,
    output_base_dir: Path | None,
    resolved_volume_dir: Path | None,
    save_volumes: bool,
    save_figures: bool,
    figure_percentile: float,
    ridge_buffer_radius: float,
    count: int,
    centers: list[tuple[int, int, int]],
    explicit_centers: bool,
    percentile: float,
    min_separation: float,
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
    final_normalization_smoothing: float | None,
    d: int,
    fm: float,
    scanner_thin_mode: str,
    voter_thin_mode: str | None,
    reference_thin_sigma: float,
    remove_scanner_edge_effects: bool,
    surface_support_min_fraction: float | None,
    surface_support_exponent: float | None,
    workflow_mode: str,
    compare_workflows: bool,
    write_markdown_index: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    workflow_options = crop_validation.resolve_workflow_options(
        workflow_mode=workflow_mode,
        voter_thin_mode=voter_thin_mode,
        surface_support_min_fraction=surface_support_min_fraction,
        surface_support_exponent=surface_support_exponent,
    )
    effective_voter_thin_mode = str(workflow_options["voter_thin_mode"])
    effective_support_min_fraction = float(workflow_options["surface_support_min_fraction"])
    effective_support_exponent = float(workflow_options["surface_support_exponent"])

    workflow_volume_dir = resolved_volume_dir
    if compare_workflows and workflow_volume_dir is not None:
        workflow_volume_dir = workflow_volume_dir / workflow_mode

    config = build_config(
        crop_shape=crop_shape,
        interior_margin=interior_margin,
        count=count,
        centers=centers,
        explicit_centers=explicit_centers,
        percentile=percentile,
        min_separation=min_separation,
        save_volumes=save_volumes,
        volume_dir=workflow_volume_dir,
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
        final_normalization_smoothing=final_normalization_smoothing,
        d=d,
        fm=fm,
        scanner_thin_mode=scanner_thin_mode,
        voter_thin_mode=effective_voter_thin_mode,
        reference_thin_sigma=reference_thin_sigma,
        remove_scanner_edge_effects=remove_scanner_edge_effects,
        workflow_mode=workflow_mode,
        surface_support_min_fraction=effective_support_min_fraction,
        surface_support_exponent=effective_support_exponent,
    )

    crops = []
    for crop_index, center in enumerate(selected_centers, start=1):
        slices = crop_slices(center, crop_shape, full_shape=arrays["ep.dat"].shape)
        ep_crop = _crop(arrays["ep.dat"], slices)
        reference_fv = _crop(arrays["fv.dat"], slices)
        reference_fvt = _crop(arrays["fvt.dat"], slices)
        reference_fl = _crop(arrays["fl.dat"], slices) if save_figures else None
        outputs = crop_validation.run_pipeline(
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
            final_normalization_smoothing=final_normalization_smoothing,
            surface_support_min_fraction=effective_support_min_fraction,
            surface_support_exponent=effective_support_exponent,
            d=d,
            fm=fm,
            scanner_thin_mode=scanner_thin_mode,
            voter_thin_mode=effective_voter_thin_mode,
            reference_thin_sigma=reference_thin_sigma,
            remove_scanner_edge_effects=remove_scanner_edge_effects,
        )

        if workflow_volume_dir is not None:
            crop_validation.write_crop_volumes(
                workflow_volume_dir / f"crop_{crop_index:03d}",
                outputs,
            )

        crop_report = crop_validation.build_crop_report(
            crop_index=crop_index,
            center=center,
            slices=slices,
            crop_shape=ep_crop.shape,
            outputs=outputs,
            reference_fv=reference_fv,
            reference_fvt=reference_fvt,
            interior_margin=interior_margin,
        )
        if save_figures:
            if output_base_dir is None:
                raise ValueError("--save-figures requires --output-json")
            if reference_fl is None:
                raise ValueError("fl.dat is required when --save-figures is passed")
            figure_dir = output_base_dir / f"crop_{crop_index:03d}" / "figures"
            if compare_workflows:
                figure_dir = output_base_dir / "figures" / workflow_mode / f"crop_{crop_index:03d}"
            crop_report["figures"] = crop_validation.write_crop_figures(
                figure_dir,
                metrics_base_dir=output_base_dir,
                reference_fl=reference_fl,
                reference_fv=reference_fv,
                reference_fvt=reference_fvt,
                outputs=outputs,
                figure_percentile=figure_percentile,
                ridge_buffer_radius=ridge_buffer_radius,
                figure_slices="center",
            )
        crops.append(crop_report)

    return config, crops


def build_config(
    *,
    crop_shape: tuple[int, int, int],
    interior_margin: int,
    count: int,
    centers: list[tuple[int, int, int]],
    explicit_centers: bool,
    percentile: float,
    min_separation: float,
    save_volumes: bool,
    volume_dir: Path | None,
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
    final_normalization_smoothing: float | None,
    d: int,
    fm: float,
    scanner_thin_mode: str = "reference",
    voter_thin_mode: str = "reference",
    reference_thin_sigma: float = 1.0,
    remove_scanner_edge_effects: bool = True,
    workflow_mode: str = "reference",
    surface_support_min_fraction: float = 0.0,
    surface_support_exponent: float = 0.0,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "workflow_mode": workflow_mode,
        "input": "ep.dat",
        "reference": ["fv.dat", "fvt.dat"],
        "comparison": "scan_vote_thin_fv_fvt_multicrop",
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
            "thin_mode": scanner_thin_mode,
            "reference_thin_sigma": float(reference_thin_sigma),
            "remove_edge_effects": bool(remove_scanner_edge_effects),
        },
        "voter": {
            "ru": int(ru),
            "rv": int(rv),
            "rw": int(rw),
            "strain_max1": float(strain_max1),
            "strain_max2": float(strain_max2),
            "surface_smoothing1": float(surface_smoothing1),
            "surface_smoothing2": float(surface_smoothing2),
            "final_normalization_smoothing": float(
                0.0 if final_normalization_smoothing is None else final_normalization_smoothing
            ),
            "d": int(d),
            "fm": float(fm),
            "thin_mode": voter_thin_mode,
            "reference_thin_sigma": float(reference_thin_sigma),
            "surface_support_min_fraction": float(surface_support_min_fraction),
            "surface_support_exponent": float(surface_support_exponent),
            "surface_voting_boundary_policy": "reference-like-i2-i3-interior",
        },
        "overlap_percentiles": [float(p) for p in crop_validation.OVERLAP_PERCENTILES],
        "aggregate_metric_roots": list(AGGREGATE_ROOTS),
        "save_volumes": bool(save_volumes),
        "volume_dir": str(volume_dir) if volume_dir is not None else None,
    }
    if save_figures or write_markdown_index:
        config["visualization"] = {
            "save_figures": bool(save_figures),
            "figure_percentile": float(figure_percentile),
            "ridge_buffer_radius": float(ridge_buffer_radius),
            "figure_slices": "center",
            "write_markdown_index": bool(write_markdown_index),
            "markdown_index": (visual_report_path.name if visual_report_path is not None else None),
        }
    return config


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


def aggregate_delta(reference: Any, quality: Any) -> Any:
    if isinstance(reference, Mapping) and isinstance(quality, Mapping):
        keys = sorted(set(reference) | set(quality))
        return {str(key): aggregate_delta(reference.get(key), quality.get(key)) for key in keys}

    if reference is None or quality is None:
        return None
    if isinstance(reference, bool) or isinstance(quality, bool):
        return None
    if isinstance(reference, int | float | np.generic) and isinstance(
        quality, int | float | np.generic
    ):
        reference_value = float(reference)
        quality_value = float(quality)
        if math.isfinite(reference_value) and math.isfinite(quality_value):
            return quality_value - reference_value
    return None


def build_consensus_summary(crops: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    crop_list = list(crops)
    fvt_nonzero = _crop_metric_values(crop_list, ("pyosv", "fvt", "nonzero_fraction"))
    fv_nonzero = _crop_metric_values(crop_list, ("pyosv", "fv", "nonzero_fraction"))
    fvt_reference_correlation = _crop_metric_values(
        crop_list,
        ("normalized_correlation", "interior", "fvt"),
    )
    fvt_buffered_precision = _crop_metric_values(
        crop_list,
        ("buffered_ridge_overlap", "interior", "fvt", "buffered_precision"),
    )
    fvt_buffered_recall = _crop_metric_values(
        crop_list,
        ("buffered_ridge_overlap", "interior", "fvt", "buffered_recall"),
    )
    fvt_sparse_distance_p95 = _crop_metric_values(
        crop_list,
        ("sparse_ridge_distance_metrics", "interior", "fvt", "candidate_to_reference_p95"),
    )
    fvt_edge_density_proxy = []
    for crop in crop_list:
        edge_proxy = _edge_density_proxy(crop, "fvt")
        if edge_proxy is not None:
            fvt_edge_density_proxy.append(edge_proxy)

    return {
        "crop_count": len(crop_list),
        **_mean_std_cv("fvt_nonzero_fraction", fvt_nonzero),
        **_mean_std_cv("fv_nonzero_fraction", fv_nonzero),
        **_mean_std("fvt_reference_correlation", fvt_reference_correlation),
        "fvt_buffered_overlap_precision_mean": _mean_or_none(fvt_buffered_precision),
        "fvt_buffered_overlap_recall_mean": _mean_or_none(fvt_buffered_recall),
        "fvt_sparse_distance_p95_mean": _mean_or_none(fvt_sparse_distance_p95),
        **_mean_std("fvt_edge_density_proxy", fvt_edge_density_proxy),
        "finite_failure_count": sum(
            _finite_failure_count(crop.get("finite_checks", {})) for crop in crop_list
        ),
    }


def build_consensus_delta(
    reference: Mapping[str, Any], quality: Mapping[str, Any]
) -> dict[str, Any]:
    metric_map = {
        "fvt_nonzero_fraction_delta_mean": "fvt_nonzero_fraction_mean",
        "fvt_reference_correlation_delta_mean": "fvt_reference_correlation_mean",
        "fvt_edge_density_proxy_delta_mean": "fvt_edge_density_proxy_mean",
        "fvt_sparse_distance_p95_delta_mean": "fvt_sparse_distance_p95_mean",
    }
    delta: dict[str, Any] = {}
    for output_key, summary_key in metric_map.items():
        reference_value = _finite_float_or_none(reference.get(summary_key))
        quality_value = _finite_float_or_none(quality.get(summary_key))
        delta[output_key] = (
            quality_value - reference_value
            if reference_value is not None and quality_value is not None
            else None
        )
    delta["finite_failure_count_delta"] = _number_delta(
        reference.get("finite_failure_count"),
        quality.get("finite_failure_count"),
    )
    return delta


def _crop_metric_values(crops: list[Mapping[str, Any]], path: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for crop in crops:
        value = _finite_float_or_none(_nested(crop, *path))
        if value is not None:
            values.append(value)
    return values


def _edge_density_proxy(crop: Mapping[str, Any], name: str) -> float | None:
    full_fraction = _finite_float_or_none(_nested(crop, "pyosv", name, "nonzero_fraction"))
    interior_fraction = _finite_float_or_none(
        _nested(crop, "pyosv_interior", name, "nonzero_fraction")
    )
    if full_fraction is None or interior_fraction is None:
        return None
    return max(0.0, full_fraction - interior_fraction)


def _mean_std_cv(prefix: str, values: list[float]) -> dict[str, float | None]:
    result = _mean_std(prefix, values)
    mean_value = result[f"{prefix}_mean"]
    std_value = result[f"{prefix}_std"]
    if mean_value is None or std_value is None:
        result[f"{prefix}_cv"] = None
    elif mean_value == 0.0:
        result[f"{prefix}_cv"] = 0.0
    else:
        result[f"{prefix}_cv"] = float(std_value / abs(mean_value))
    return result


def _mean_std(prefix: str, values: list[float]) -> dict[str, float | None]:
    if not values:
        return {f"{prefix}_mean": None, f"{prefix}_std": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array)),
    }


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _finite_failure_count(value: Any) -> int:
    if isinstance(value, Mapping):
        if {"finite_count", "size"} <= set(value):
            finite_count = _finite_float_or_none(value.get("finite_count"))
            size = _finite_float_or_none(value.get("size"))
            return int(finite_count is not None and size is not None and finite_count < size)
        return sum(_finite_failure_count(item) for item in value.values())
    if isinstance(value, list | tuple):
        return sum(_finite_failure_count(item) for item in value)
    return 0


def _number_delta(reference: Any, quality: Any) -> float | int | None:
    reference_value = _finite_float_or_none(reference)
    quality_value = _finite_float_or_none(quality)
    if reference_value is None or quality_value is None:
        return None
    delta = quality_value - reference_value
    return int(delta) if float(delta).is_integer() else delta


def _finite_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | np.generic):
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            return numeric_value
    return None


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
    crop_selection = _as_mapping(config.get("crop_selection", {}))
    scanner = _as_mapping(config.get("scanner", {}))
    voter = _as_mapping(config.get("voter", {}))
    visualization = _as_mapping(config.get("visualization", {}))
    workflows = _as_mapping(report.get("workflows", {}))
    consensus = _as_mapping(report.get("consensus", {}))
    crops = list(report.get("crops", []))
    data_root = Path(str(report.get("data_root", "")))
    selected_count = crop_selection.get("selected_count", len(crops))
    if workflows and selected_count == 0:
        selected_count = max(
            (len(list(_as_mapping(workflow).get("crops", []))) for workflow in workflows.values()),
            default=0,
        )

    lines = [
        "# F3 Multi-Crop Visual Report",
        "",
        "## Run Configuration",
        "",
        f"- data_root: `{data_root}`",
        f"- data_root_basename: `{data_root.name}`",
        f"- comparison: `{config.get('comparison', '')}`",
        f"- crop_shape: `{crop_selection.get('crop_shape', '')}`",
        f"- interior_margin: `{config.get('interior_margin', '')}`",
        f"- crop_selection_source: `{crop_selection.get('source', '')}`",
        f"- selected_count: `{selected_count}`",
        f"- scanner_thin_mode: `{scanner.get('thin_mode', '')}`",
        f"- scanner_edge_effect_removal: `{scanner.get('remove_edge_effects', '')}`",
        f"- voter_thin_mode: `{voter.get('thin_mode', '')}`",
        f"- reference_thin_sigma: `{scanner.get('reference_thin_sigma', '')}`",
        f"- surface_voting_boundary_policy: `{voter.get('surface_voting_boundary_policy', '')}`",
        f"- scanner: `{scanner}`",
        f"- voter: `{voter}`",
    ]
    if visualization:
        lines.append(f"- visualization: `{visualization}`")

    if workflows:
        lines.extend(["", "## Workflows", ""])
        for workflow_name, workflow_report in _ordered_workflows(workflows):
            workflow_config = _as_mapping(_as_mapping(workflow_report).get("config", {}))
            workflow_scanner = _as_mapping(workflow_config.get("scanner", {}))
            workflow_voter = _as_mapping(workflow_config.get("voter", {}))
            workflow_skinner = _as_mapping(workflow_config.get("skinner", {}))
            lines.extend(
                [
                    f"### {workflow_name}",
                    "",
                    f"- workflow_mode: `{workflow_config.get('workflow_mode', workflow_name)}`",
                    f"- scanner_thin_mode: `{workflow_scanner.get('thin_mode', '')}`",
                    f"- scanner_edge_effect_removal: `{workflow_scanner.get('remove_edge_effects', '')}`",
                    f"- voter_thin_mode: `{workflow_voter.get('thin_mode', '')}`",
                    f"- surface_support_min_fraction: `{workflow_voter.get('surface_support_min_fraction', '')}`",
                    f"- surface_support_exponent: `{workflow_voter.get('surface_support_exponent', '')}`",
                    f"- surface_voting_boundary_policy: `{workflow_voter.get('surface_voting_boundary_policy', '')}`",
                    f"- voter: `{workflow_voter}`",
                ]
            )
            if workflow_skinner:
                lines.append(f"- skinner: `{workflow_skinner}`")
            lines.append("")

        _append_consensus_section(lines, consensus)

        for workflow_name, workflow_report in _ordered_workflows(workflows):
            workflow_crops = list(_as_mapping(workflow_report).get("crops", []))
            lines.extend([f"## {workflow_name} Crop Metrics", ""])
            _append_crop_metrics_table(lines, workflow_crops)

        any_figures = False
        for workflow_name, workflow_report in _ordered_workflows(workflows):
            workflow_crops = list(_as_mapping(workflow_report).get("crops", []))
            lines.extend(["", f"## {workflow_name} Figures", ""])
            any_figures = _append_figures(lines, workflow_crops) or any_figures
        if not any_figures:
            lines.append("No PNG figures were written for this run.")

        _append_workflow_delta(lines, _as_mapping(report.get("workflow_delta", {})))
        lines.extend(
            [
                "",
                "## Interpretation Checklist",
                "",
                "- scanner mismatch",
                "- voting mismatch",
                "- thinning/ridge shift",
                "- boundary artifact",
                "",
            ]
        )
        return "\n".join(lines)

    _append_consensus_section(lines, consensus)

    lines.extend(
        [
            "",
            "## Crop Metrics",
            "",
        ]
    )
    _append_crop_metrics_table(lines, crops)

    lines.extend(["", "## Figures", ""])
    if not _append_figures(lines, crops):
        lines.append("No PNG figures were written for this run.")

    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- scanner mismatch",
            "- voting mismatch",
            "- thinning/ridge shift",
            "- boundary artifact",
            "",
        ]
    )
    return "\n".join(lines)


def _ordered_workflows(workflows: Mapping[str, Any]) -> list[tuple[str, Any]]:
    ordered = [(name, workflows[name]) for name in ("reference", "quality") if name in workflows]
    ordered.extend(
        (str(name), workflow)
        for name, workflow in workflows.items()
        if name not in {"reference", "quality"}
    )
    return ordered


def _append_crop_metrics_table(lines: list[str], crops: list[Any]) -> None:
    lines.extend(
        [
            "| Crop | Center | Slices | normalized_correlation.interior.fv | "
            "normalized_correlation.interior.fvt | top_percentile_overlap.interior.fvt.99.jaccard | "
            "buffered_ridge_overlap.interior.fvt.buffered_f1 | "
            "sparse_ridge_distance_metrics.interior.fvt.candidate_to_reference_median |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for crop in crops:
        crop_map = _as_mapping(crop)
        crop_id = f"crop_{int(crop_map.get('index', 0)):03d}"
        lines.append(
            "| "
            f"{crop_id} | "
            f"`{crop_map.get('crop_center', '')}` | "
            f"`{_format_slices(crop_map.get('crop_slices', []))}` | "
            f"{_format_metric(_nested(crop_map, 'normalized_correlation', 'interior', 'fv'))} | "
            f"{_format_metric(_nested(crop_map, 'normalized_correlation', 'interior', 'fvt'))} | "
            f"{_format_metric(_nested(crop_map, 'top_percentile_overlap', 'interior', 'fvt', '99', 'jaccard'))} | "
            f"{_format_metric(_nested(crop_map, 'buffered_ridge_overlap', 'interior', 'fvt', 'buffered_f1'))} | "
            f"{_format_metric(_nested(crop_map, 'sparse_ridge_distance_metrics', 'interior', 'fvt', 'candidate_to_reference_median'))} |"
        )


def _append_figures(lines: list[str], crops: list[Any]) -> bool:
    any_figures = False
    for crop in crops:
        crop_map = _as_mapping(crop)
        crop_id = f"crop_{int(crop_map.get('index', 0)):03d}"
        figure_links = _important_figure_links(crop_map)
        if not figure_links:
            continue
        any_figures = True
        lines.extend([f"### {crop_id}", ""])
        for label, path in figure_links:
            lines.append(f"- [{label}]({path})")
        lines.append("")
    return any_figures


def _append_workflow_delta(lines: list[str], workflow_delta: Mapping[str, Any]) -> None:
    quality_vs_reference = _as_mapping(workflow_delta.get("quality_vs_reference", {}))
    per_metric_mean = _as_mapping(quality_vs_reference.get("per_metric_mean", {}))
    if not per_metric_mean:
        return

    metrics = [
        "normalized_correlation.interior.fv",
        "normalized_correlation.interior.fvt",
        "top_percentile_overlap.interior.fvt.99.jaccard",
        "buffered_ridge_overlap.interior.fvt.buffered_f1",
        "sparse_ridge_distance_metrics.interior.fvt.candidate_to_reference_median",
    ]
    lines.extend(
        [
            "",
            "## Workflow Delta",
            "",
            "quality_vs_reference per_metric_mean:",
            "",
            "| Metric | Delta |",
            "| --- | ---: |",
        ]
    )
    for metric in metrics:
        lines.append(f"| {metric} | {_format_metric(per_metric_mean.get(metric))} |")


def _append_consensus_section(lines: list[str], consensus: Mapping[str, Any]) -> None:
    workflows = _as_mapping(consensus.get("workflows", {}))
    if not workflows:
        return

    metrics = [
        "crop_count",
        "fvt_nonzero_fraction_mean",
        "fvt_nonzero_fraction_cv",
        "fv_nonzero_fraction_mean",
        "fv_nonzero_fraction_cv",
        "fvt_reference_correlation_mean",
        "fvt_buffered_overlap_precision_mean",
        "fvt_buffered_overlap_recall_mean",
        "fvt_sparse_distance_p95_mean",
        "fvt_edge_density_proxy_mean",
        "finite_failure_count",
    ]
    lines.extend(
        [
            "",
            "## Consensus",
            "",
            "Crop-to-crop stability summary computed from saved crop metrics.",
            "",
            "| Workflow | " + " | ".join(metrics) + " |",
            "| --- | " + " | ".join("---:" for _ in metrics) + " |",
        ]
    )
    for workflow_name, summary in _ordered_workflows(workflows):
        summary_map = _as_mapping(summary)
        lines.append(
            f"| {workflow_name} | "
            + " | ".join(_format_metric(summary_map.get(metric)) for metric in metrics)
            + " |"
        )

    comparison = _as_mapping(consensus.get("workflow_comparison", {}))
    quality_minus_reference = _as_mapping(comparison.get("quality_minus_reference", {}))
    if not quality_minus_reference:
        return

    delta_metrics = [
        "fvt_nonzero_fraction_delta_mean",
        "fvt_reference_correlation_delta_mean",
        "fvt_edge_density_proxy_delta_mean",
        "fvt_sparse_distance_p95_delta_mean",
        "finite_failure_count_delta",
    ]
    lines.extend(
        [
            "",
            "quality_minus_reference consensus delta:",
            "",
            "| Metric | Delta |",
            "| --- | ---: |",
        ]
    )
    for metric in delta_metrics:
        lines.append(f"| {metric} | {_format_metric(quality_minus_reference.get(metric))} |")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _format_slices(value: Any) -> str:
    if not isinstance(value, list):
        return str(value)

    formatted = []
    for item in value:
        if not isinstance(item, Mapping):
            return str(value)
        formatted.append(f"{item.get('axis')}:{item.get('start')}-{item.get('stop')}")
    return ", ".join(formatted)


def _format_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int | float | np.generic):
        numeric_value = float(value)
        return f"{numeric_value:.4g}" if math.isfinite(numeric_value) else ""
    return str(value)


def _important_figure_links(crop: Mapping[str, Any]) -> list[tuple[str, str]]:
    figures = _as_mapping(crop.get("figures", {}))
    files = _as_mapping(figures.get("files", {}))
    candidates = (
        ("scanner mismatch", ("scanner_fl_vs_ftpy", "i3")),
        ("voting mismatch", ("fv_ref_vs_py", "i3")),
        ("thinning/ridge shift", ("fvt_ridge_overlay", "i3")),
        ("fv MIP", ("fv", "mip")),
        ("fvt MIP", ("fvt", "mip")),
    )

    links: list[tuple[str, str]] = []
    for label, path_keys in candidates:
        path = _nested(files, *path_keys)
        if isinstance(path, str):
            links.append((label, path))
    return links


def report_to_json(report: Mapping[str, Any], *, pretty: bool = False) -> str:
    indent = 2 if pretty else None
    return json.dumps(_json_compatible(report), indent=indent, sort_keys=True) + "\n"


def ensure_output_not_in_data_root(
    output_path: str | PathLike[str],
    data_root: str | PathLike[str],
    *,
    option_name: str,
) -> None:
    resolved_output = Path(output_path).resolve(strict=False)
    resolved_data_root = Path(data_root).resolve(strict=False)
    try:
        resolved_output.relative_to(resolved_data_root)
    except ValueError:
        return
    raise ValueError(f"{option_name} must not be inside the F3 data root: {resolved_output}")


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
            save_volumes=args.save_volumes,
            save_figures=args.save_figures,
            figure_percentile=args.figure_percentile,
            ridge_buffer_radius=args.ridge_buffer_radius,
            write_markdown_index=args.write_markdown_index,
            volume_dir=args.volume_dir,
            pretty=args.pretty,
            count=args.count,
            crop_shape=args.crop_shape,
            interior_margin=args.interior_margin,
            centers=args.center,
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
            final_normalization_smoothing=args.final_normalization_smoothing,
            surface_support_min_fraction=args.surface_support_min_fraction,
            surface_support_exponent=args.surface_support_exponent,
            d=args.d,
            fm=args.fm,
            scanner_thin_mode=args.scanner_thin_mode,
            voter_thin_mode=(args.voter_thin_mode if args.voter_thin_mode_explicit else None),
            reference_thin_sigma=args.reference_thin_sigma,
            remove_scanner_edge_effects=args.remove_scanner_edge_effects,
            workflow_mode=args.workflow_mode,
            compare_workflows=args.compare_workflows,
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
