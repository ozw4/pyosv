"""Compare reference and normal scanner thinning on shared F3 raw scans."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable, Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

import run_3d_f3d_crop_validation as crop_validation
from pyosv.evaluation import f3d_scanner_policy as policy_evaluation
from pyosv.evaluation import f3d_scanner_policy_diagnostics as policy_diagnostics
from pyosv.f3d_reference import (
    F3D_ENV_VAR,
    crop_slices,
    parse_shape3,
    pick_reference_centers,
    read_f3d_file,
    resolve_f3d_data_root,
)
from pyosv.metrics import top_percentile_mask

DEFAULT_COUNT = 3
DEFAULT_CROP_SHAPE = (64, 64, 64)
DEFAULT_INTERIOR_MARGIN = 16
DEFAULT_PERCENTILE = 99.9
DEFAULT_MIN_SEPARATION = 48.0
DEFAULT_OUTLIER_MAX_POINTS = 64
DEFAULT_OUTLIER_MAX_COMPONENTS = 8
DEFAULT_OUTLIER_WINDOW_RADIUS = 24
DEFAULT_OUTLIER_ADJACENT_SLICE_RADIUS = 3
DEFAULT_AMPLITUDE_CLIP_PERCENTILE = 99.0
REFERENCE_OSV_DIR = Path(__file__).resolve().parents[1] / "reference_osv"
ISSUE_FORGE_DIR = Path(__file__).resolve().parents[1] / "vendor" / "issue_forge"

COMPARISON_PROFILE = policy_evaluation.COMPARISON_PROFILE
BASELINE_POLICY_ID = policy_evaluation.BASELINE_POLICY_ID
CANDIDATE_POLICY_ID = policy_evaluation.CANDIDATE_POLICY_ID
run_shared_scan_policy_pipeline = policy_evaluation.run_shared_scan_policy_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare reference and normal scanner thinning on one shared reference-like "
            "F3 scan while holding the quality workflow downstream fixed."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"Path to the F3 reference data root. Defaults to {F3D_ENV_VAR}.",
    )
    parser.add_argument(
        "--comparison-profile",
        choices=(COMPARISON_PROFILE,),
        default=COMPARISON_PROFILE,
        help="Fixed scanner-thinning comparison profile.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional metrics JSON path. Parent directories are created as needed.",
    )
    parser.add_argument(
        "--save-volumes",
        action="store_true",
        help="Write each policy's ten DAT volumes next to the metrics report.",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Write per-policy and direct-comparison PNG diagnostics.",
    )
    parser.add_argument(
        "--write-markdown-index",
        action="store_true",
        help="Write visual_report.md next to the metrics JSON.",
    )
    parser.add_argument(
        "--fail-on-validation-failure",
        action="store_true",
        help="Exit with status 2 when the truthless policy validation fails.",
    )
    parser.add_argument(
        "--outlier-diagnostics",
        action="store_true",
        help=(
            "Add public-FVT distance-outlier diagnostics; with --save-figures, "
            "also write seismic-amplitude review figures."
        ),
    )
    parser.add_argument(
        "--context-crop-shape",
        type=parse_shape3,
        default=None,
        help=(
            "Optional larger n3,n2,n1 crop used to recompute the same global base ROI; "
            "requires --outlier-diagnostics."
        ),
    )
    parser.add_argument(
        "--context-crop-index",
        action="append",
        type=int,
        default=None,
        help=(
            "1-origin base crop index for context recomputation; may be repeated. "
            "Defaults to crops with distance outliers."
        ),
    )
    parser.add_argument(
        "--outlier-max-points",
        type=int,
        default=DEFAULT_OUTLIER_MAX_POINTS,
        help="Maximum number of deterministically sorted outlier points stored per crop.",
    )
    parser.add_argument(
        "--outlier-max-components",
        type=int,
        default=DEFAULT_OUTLIER_MAX_COMPONENTS,
        help="Maximum number of connected-component details stored per crop.",
    )
    parser.add_argument(
        "--outlier-window-radius",
        type=int,
        default=DEFAULT_OUTLIER_WINDOW_RADIUS,
        help="Local half-window in samples for outlier amplitude figures.",
    )
    parser.add_argument(
        "--outlier-adjacent-slice-radius",
        type=int,
        default=DEFAULT_OUTLIER_ADJACENT_SLICE_RADIUS,
        help="Number of adjacent slices shown before and after an outlier slice.",
    )
    parser.add_argument(
        "--amplitude-clip-percentile",
        type=float,
        default=DEFAULT_AMPLITUDE_CLIP_PERCENTILE,
        help="Absolute seismic-amplitude percentile used for one symmetric crop-level clip.",
    )
    parser.add_argument("--pretty", action="store_true", help="Write indented JSON.")
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help="Number of deterministic crops when --center is omitted.",
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
        help="Boundary shell width excluded from interior metrics.",
    )
    parser.add_argument(
        "--center",
        action="append",
        type=crop_validation.parse_index3,
        default=None,
        help="Explicit center in i3,i2,i1 order; may be repeated.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help="Reference fv percentile used for deterministic crop selection.",
    )
    parser.add_argument(
        "--min-separation",
        type=float,
        default=DEFAULT_MIN_SEPARATION,
        help="Minimum deterministic crop-center separation in samples.",
    )
    parser.add_argument(
        "--figure-percentile",
        type=float,
        default=99.0,
        help="Display clipping and visual ridge percentile.",
    )
    parser.add_argument(
        "--ridge-buffer-radius",
        type=float,
        default=2.0,
        help="Visual ridge-overlay buffer radius.",
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
    parser.add_argument("--strain-max1", type=float, default=0.25)
    parser.add_argument("--strain-max2", type=float, default=0.25)
    parser.add_argument("--surface-smoothing1", type=float, default=2.0)
    parser.add_argument("--surface-smoothing2", type=float, default=2.0)
    parser.add_argument(
        "--surface-orientation-smoothing",
        type=float,
        default=None,
        help="Defaults to max(rv, rw), matching the quality workflow.",
    )
    parser.add_argument("--d", type=int, default=4, help="Seed exclusion distance.")
    parser.add_argument("--fm", type=float, default=0.3, help="Minimum seed likelihood.")
    return parser


def run_example(
    *,
    data_root_arg: str | PathLike[str] | None,
    comparison_profile: str = COMPARISON_PROFILE,
    output_json: str | PathLike[str] | None = None,
    save_volumes: bool = False,
    save_figures: bool = False,
    write_markdown_index: bool = False,
    outlier_diagnostics: bool = False,
    context_crop_shape: tuple[int, int, int] | None = None,
    context_crop_indices: Iterable[int] | None = None,
    outlier_max_points: int = DEFAULT_OUTLIER_MAX_POINTS,
    outlier_max_components: int = DEFAULT_OUTLIER_MAX_COMPONENTS,
    outlier_window_radius: int = DEFAULT_OUTLIER_WINDOW_RADIUS,
    outlier_adjacent_slice_radius: int = DEFAULT_OUTLIER_ADJACENT_SLICE_RADIUS,
    amplitude_clip_percentile: float = DEFAULT_AMPLITUDE_CLIP_PERCENTILE,
    pretty: bool = False,
    count: int = DEFAULT_COUNT,
    crop_shape: tuple[int, int, int] = DEFAULT_CROP_SHAPE,
    interior_margin: int = DEFAULT_INTERIOR_MARGIN,
    centers: Iterable[tuple[int, int, int]] | None = None,
    percentile: float = DEFAULT_PERCENTILE,
    min_separation: float = DEFAULT_MIN_SEPARATION,
    figure_percentile: float = 99.0,
    ridge_buffer_radius: float = 2.0,
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
    surface_orientation_smoothing: float | None = None,
    d: int = 4,
    fm: float = 0.3,
) -> dict[str, Any]:
    if comparison_profile != COMPARISON_PROFILE:
        raise ValueError(f"comparison_profile must be {COMPARISON_PROFILE!r}")
    if count < 1:
        raise ValueError("count must be >= 1")
    crop_shape, interior_margin = validate_crop_config(crop_shape, interior_margin)
    context_crop_indices = validate_diagnostic_options(
        outlier_diagnostics=outlier_diagnostics,
        context_crop_shape=context_crop_shape,
        context_crop_indices=context_crop_indices,
        base_crop_shape=crop_shape,
        outlier_max_points=outlier_max_points,
        outlier_max_components=outlier_max_components,
        outlier_window_radius=outlier_window_radius,
        outlier_adjacent_slice_radius=outlier_adjacent_slice_radius,
        amplitude_clip_percentile=amplitude_clip_percentile,
    )

    data_root = resolve_f3d_data_root(data_root_arg)
    if output_json is not None:
        ensure_output_path_allowed(output_json, data_root, option_name="--output-json")
    elif save_volumes:
        raise ValueError("--save-volumes requires --output-json")
    elif save_figures:
        raise ValueError("--save-figures requires --output-json")
    elif write_markdown_index:
        raise ValueError("--write-markdown-index requires --output-json")
    if save_figures:
        crop_validation.require_figure_support()

    output_base_dir = Path(output_json).parent if output_json is not None else None
    arrays = crop_validation.read_reference_arrays(data_root)
    full_shape = tuple(int(size) for size in arrays["ep.dat"].shape)
    selected_centers = select_centers(
        arrays["fv.dat"],
        count=count,
        centers=centers,
        percentile=percentile,
        min_separation=min_separation,
        crop_shape=crop_shape,
    )
    if centers is None and len(selected_centers) != count:
        raise ValueError(f"selected {len(selected_centers)} crop centers; requested {count}")
    if not selected_centers:
        raise ValueError("at least one crop center is required")
    context_crop_indices = validate_context_selection(
        context_crop_shape=context_crop_shape,
        full_shape=full_shape,
        context_crop_indices=context_crop_indices,
        crop_count=len(selected_centers),
    )
    if outlier_diagnostics:
        arrays["xs.dat"] = read_f3d_file("xs.dat", data_root)
        arrays["fl.dat"] = read_f3d_file("fl.dat", data_root)
    elif save_figures:
        arrays["fl.dat"] = read_f3d_file("fl.dat", data_root)

    policy_configs = {
        policy.role: policy_evaluation.build_policy_config(
            policy,
            reference_thin_sigma=1.0,
            ru=ru,
            rv=rv,
            rw=rw,
            strain_max1=strain_max1,
            strain_max2=strain_max2,
            surface_smoothing1=surface_smoothing1,
            surface_smoothing2=surface_smoothing2,
            surface_orientation_smoothing=surface_orientation_smoothing,
            final_normalization_smoothing=None,
            d=d,
            fm=fm,
        )
        for policy in policy_evaluation.SCANNER_POLICIES
    }
    policy_crops: dict[str, list[dict[str, Any]]] = {
        "baseline": [],
        "candidate": [],
    }
    direct_comparisons: list[dict[str, Any]] = []
    scanner_execution_count = 0
    pipeline_kwargs = {
        "sigma1": sigma1,
        "sigma2": sigma2,
        "phi_min": phi_min,
        "phi_max": phi_max,
        "theta_min": theta_min,
        "theta_max": theta_max,
        "ru": ru,
        "rv": rv,
        "rw": rw,
        "strain_max1": strain_max1,
        "strain_max2": strain_max2,
        "surface_smoothing1": surface_smoothing1,
        "surface_smoothing2": surface_smoothing2,
        "surface_orientation_smoothing": surface_orientation_smoothing,
        "final_normalization_smoothing": None,
        "d": d,
        "fm": fm,
        "reference_thin_sigma": 1.0,
    }
    base_diagnostic_states: list[dict[str, Any]] = []

    for crop_index, center in enumerate(selected_centers, start=1):
        slices = crop_slices(center, crop_shape, full_shape=arrays["ep.dat"].shape)
        ep_crop = _crop(arrays["ep.dat"], slices)
        reference_fv = _crop(arrays["fv.dat"], slices)
        reference_fvt = _crop(arrays["fvt.dat"], slices)
        reference_fl = (
            _crop(arrays["fl.dat"], slices) if save_figures or outlier_diagnostics else None
        )
        xs_crop = _crop(arrays["xs.dat"], slices) if outlier_diagnostics else None

        run = run_shared_scan_policy_pipeline(ep_crop, **pipeline_kwargs)
        scanner_execution_count += int(run["scanner_execution_count"])
        run_policies = _as_mapping(run.get("policies"))

        outputs_by_role: dict[str, Mapping[str, np.ndarray]] = {}
        for role in policy_evaluation.POLICY_ROLES:
            policy_run = _as_mapping(run_policies.get(role))
            outputs = _as_array_mapping(policy_run.get("outputs"))
            outputs_by_role[role] = outputs
            if output_base_dir is not None and save_volumes:
                crop_validation.write_crop_volumes(
                    output_base_dir / f"crop_{crop_index:03d}" / role,
                    outputs,
                )

            outputs_are_finite = _outputs_are_finite(outputs)
            if outputs_are_finite:
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
            else:
                crop_report = build_nonfinite_crop_report(
                    crop_index=crop_index,
                    center=center,
                    slices=slices,
                    crop_shape=ep_crop.shape,
                    outputs=outputs,
                    reference_fv=reference_fv,
                    reference_fvt=reference_fvt,
                    interior_margin=interior_margin,
                )
            crop_report["policy_id"] = (
                BASELINE_POLICY_ID if role == "baseline" else CANDIDATE_POLICY_ID
            )
            crop_report["stage_density"] = policy_evaluation.build_stage_density_report(
                outputs,
                interior_margin=interior_margin,
            )
            if save_figures and outputs_are_finite:
                if output_base_dir is None or reference_fl is None:
                    raise ValueError("--save-figures requires --output-json and fl.dat")
                crop_report["figures"] = crop_validation.write_crop_figures(
                    output_base_dir / f"crop_{crop_index:03d}" / role / "figures",
                    metrics_base_dir=output_base_dir,
                    reference_fl=reference_fl,
                    reference_fv=reference_fv,
                    reference_fvt=reference_fvt,
                    outputs=outputs,
                    figure_percentile=figure_percentile,
                    ridge_buffer_radius=ridge_buffer_radius,
                    figure_slices="center",
                )
            elif save_figures:
                crop_report["figures"] = {
                    "skipped": True,
                    "reason": "policy outputs contain non-finite values",
                }
            policy_crops[role].append(crop_report)

        direct_metadata = {
            "index": int(crop_index),
            "crop_center": [int(index) for index in center],
            "crop_slices": crop_validation.slices_to_json(slices),
        }
        direct_fvt_is_finite = all(
            np.all(np.isfinite(outputs_by_role[role]["fvt_py.dat"]))
            for role in policy_evaluation.POLICY_ROLES
        )
        if direct_fvt_is_finite:
            direct = {
                **direct_metadata,
                **policy_evaluation.build_direct_policy_comparison(
                    outputs_by_role["baseline"],
                    outputs_by_role["candidate"],
                    interior_margin=interior_margin,
                ),
            }
        else:
            direct = build_unavailable_direct_comparison(
                metadata=direct_metadata,
                baseline_crop=policy_crops["baseline"][-1],
                candidate_crop=policy_crops["candidate"][-1],
            )
        outlier_report: dict[str, Any] | None = None
        if outlier_diagnostics:
            outlier_report = policy_diagnostics.build_public_fvt_distance_outlier_report(
                reference_fvt=reference_fvt,
                baseline_outputs=outputs_by_role["baseline"],
                candidate_outputs=outputs_by_role["candidate"],
                crop_slices=slices,
                interior_margin=interior_margin,
                ridge_percentile=policy_evaluation.DEFAULT_RIDGE_PERCENTILE,
                allowed_p95_delta=policy_evaluation.DEFAULT_SPARSE_DISTANCE_P95_MAX_DELTA,
                max_points=outlier_max_points,
                max_components=outlier_max_components,
                xs=xs_crop,
                ep=ep_crop,
                reference_fl=reference_fl,
                reference_fv=reference_fv,
            )
            direct["public_fvt_distance_outliers"] = outlier_report
        if save_figures and direct_fvt_is_finite:
            if output_base_dir is None:
                raise ValueError("--save-figures requires --output-json")
            direct["figures"] = write_direct_comparison_figures(
                output_base_dir / f"crop_{crop_index:03d}" / "policy_comparison" / "figures",
                metrics_base_dir=output_base_dir,
                baseline_fvt=outputs_by_role["baseline"]["fvt_py.dat"],
                candidate_fvt=outputs_by_role["candidate"]["fvt_py.dat"],
                interior_margin=interior_margin,
                figure_percentile=figure_percentile,
                ridge_buffer_radius=ridge_buffer_radius,
            )
        elif save_figures:
            direct["figures"] = {
                "skipped": True,
                "reason": "policy fvt outputs contain non-finite values",
            }
        if (
            outlier_diagnostics
            and save_figures
            and outlier_report is not None
            and outlier_report.get("status") == "available"
            and int(_nested(outlier_report, "summary", "outlier_count") or 0) > 0
        ):
            if output_base_dir is None or xs_crop is None:
                raise ValueError("outlier amplitude figures require --output-json and xs.dat")
            component_figures = write_outlier_diagnostic_figures(
                output_base_dir
                / f"crop_{crop_index:03d}"
                / "policy_comparison"
                / "outlier_diagnostics",
                metrics_base_dir=output_base_dir,
                xs=xs_crop,
                reference_fvt=reference_fvt,
                baseline_fvt=outputs_by_role["baseline"]["fvt_py.dat"],
                candidate_fvt=outputs_by_role["candidate"]["fvt_py.dat"],
                crop_slices=slices,
                crop_index=crop_index,
                outlier_report=outlier_report,
                window_radius=outlier_window_radius,
                adjacent_slice_radius=outlier_adjacent_slice_radius,
                amplitude_clip_percentile=amplitude_clip_percentile,
            )
            _attach_component_figures(outlier_report, component_figures)
        if outlier_diagnostics:
            base_diagnostic_states.append(
                {
                    "index": int(crop_index),
                    "center": tuple(int(value) for value in center),
                    "crop_slices": slices,
                    "ep": ep_crop,
                    "xs": xs_crop,
                    "reference_fl": reference_fl,
                    "reference_fv": reference_fv,
                    "reference_fvt": reference_fvt,
                    "outputs_by_role": outputs_by_role,
                    "outlier_report": outlier_report,
                }
            )
        direct_comparisons.append(direct)

    consensus = policy_evaluation.build_consensus(
        baseline_crops=policy_crops["baseline"],
        candidate_crops=policy_crops["candidate"],
        direct_comparisons=direct_comparisons,
    )
    consensus["candidate_minus_baseline"]["crops"] = direct_comparisons
    validation = policy_evaluation.validate_policy_comparison(
        baseline_crops=policy_crops["baseline"],
        candidate_crops=policy_crops["candidate"],
        direct_comparisons=direct_comparisons,
        baseline_config=policy_configs["baseline"],
        candidate_config=policy_configs["candidate"],
        scanner_execution_count=scanner_execution_count,
        expected_crop_count=len(selected_centers),
    )
    context_report: dict[str, Any] | None = None
    if context_crop_shape is not None:
        selected_context_indices = (
            context_crop_indices
            if context_crop_indices is not None
            else [
                int(state["index"])
                for state in base_diagnostic_states
                if int(
                    _nested(
                        _as_mapping(state.get("outlier_report")),
                        "summary",
                        "outlier_count",
                    )
                    or 0
                )
                > 0
            ]
        )
        context_report = run_context_diagnostics(
            arrays=arrays,
            base_states=base_diagnostic_states,
            selected_crop_indices=selected_context_indices,
            context_crop_shape=context_crop_shape,
            pipeline_kwargs=pipeline_kwargs,
            save_figures=save_figures,
            output_base_dir=output_base_dir,
            outlier_window_radius=outlier_window_radius,
            amplitude_clip_percentile=amplitude_clip_percentile,
        )

    config = build_config(
        comparison_profile=comparison_profile,
        count=count,
        centers=selected_centers,
        explicit_centers=centers is not None,
        crop_shape=crop_shape,
        interior_margin=interior_margin,
        percentile=percentile,
        min_separation=min_separation,
        scanner_execution_count=scanner_execution_count,
        save_volumes=save_volumes,
        save_figures=save_figures,
        write_markdown_index=write_markdown_index,
        figure_percentile=figure_percentile,
        ridge_buffer_radius=ridge_buffer_radius,
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
        surface_orientation_smoothing=surface_orientation_smoothing,
        d=d,
        fm=fm,
    )
    if outlier_diagnostics:
        config["diagnostics"] = {
            "public_fvt_distance_outliers": {
                "enabled": True,
                "ridge_percentile": policy_evaluation.DEFAULT_RIDGE_PERCENTILE,
                "positive_only": True,
                "allowed_p95_delta": (policy_evaluation.DEFAULT_SPARSE_DISTANCE_P95_MAX_DELTA),
                "max_points": int(outlier_max_points),
                "max_components": int(outlier_max_components),
                "window_radius": int(outlier_window_radius),
                "adjacent_slice_radius": int(outlier_adjacent_slice_radius),
                "amplitude_clip_percentile": float(amplitude_clip_percentile),
            },
            "context_crop_shape": (
                None if context_crop_shape is None else [int(size) for size in context_crop_shape]
            ),
            "context_crop_indices": (
                None if context_report is None else list(context_report["selected_crop_indices"])
            ),
        }
    report = {
        "format_version": 1,
        "data_root": str(data_root),
        "config": config,
        "scanner_policies": {
            role: {
                "policy_id": BASELINE_POLICY_ID if role == "baseline" else CANDIDATE_POLICY_ID,
                "config": policy_configs[role],
                "crops": policy_crops[role],
                "aggregate": consensus["policies"][role],
            }
            for role in policy_evaluation.POLICY_ROLES
        },
        "consensus": consensus,
        "policy_validation": validation,
        "manual_review": build_pending_manual_review(),
    }
    if context_report is not None:
        report["context_diagnostics"] = context_report
    report = policy_evaluation.json_safe(report)
    if output_json is not None:
        write_report_json(report, output_json, pretty=pretty)
        if write_markdown_index:
            write_visual_report_markdown(report, Path(output_json).parent / "visual_report.md")
    return report


def build_config(
    *,
    comparison_profile: str,
    count: int,
    centers: list[tuple[int, int, int]],
    explicit_centers: bool,
    crop_shape: tuple[int, int, int],
    interior_margin: int,
    percentile: float,
    min_separation: float,
    scanner_execution_count: int,
    save_volumes: bool,
    save_figures: bool,
    write_markdown_index: bool,
    figure_percentile: float,
    ridge_buffer_radius: float,
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
    surface_orientation_smoothing: float | None,
    d: int,
    fm: float,
) -> dict[str, Any]:
    shared_scanner = policy_evaluation.build_shared_scanner_config(
        sigma1=sigma1,
        sigma2=sigma2,
        phi_min=phi_min,
        phi_max=phi_max,
        theta_min=theta_min,
        theta_max=theta_max,
    )
    shared_scanner["execution_count"] = int(scanner_execution_count)
    quality_downstream = policy_evaluation.build_quality_downstream_config(
        reference_thin_sigma=1.0,
        ru=ru,
        rv=rv,
        rw=rw,
        strain_max1=strain_max1,
        strain_max2=strain_max2,
        surface_smoothing1=surface_smoothing1,
        surface_smoothing2=surface_smoothing2,
        surface_orientation_smoothing=surface_orientation_smoothing,
        final_normalization_smoothing=None,
        d=d,
        fm=fm,
    )
    return {
        "comparison_profile": comparison_profile,
        "crop_selection": {
            "source": "explicit_centers" if explicit_centers else "fv.dat",
            "requested_count": int(count),
            "selected_count": len(centers),
            "centers": [[int(index) for index in center] for center in centers],
            "crop_shape": [int(size) for size in crop_shape],
            "interior_margin": int(interior_margin),
            "percentile": float(percentile),
            "min_separation": float(min_separation),
            "boundary_margin": None if explicit_centers else "crop_shape",
        },
        "shared_scanner": shared_scanner,
        "quality_downstream": quality_downstream,
        "edge_shell": {
            "definition": "crop minus interior",
            "interior_margin": int(interior_margin),
        },
        "metrics": {
            "nonzero_epsilon": policy_evaluation.DEFAULT_NONZERO_EPSILON,
            "ridge_percentile": policy_evaluation.DEFAULT_RIDGE_PERCENTILE,
            "ridge_buffer_radius": policy_evaluation.DEFAULT_RIDGE_BUFFER_RADIUS,
            "ridge_mask_difference": "positive fvt (> nonzero_epsilon)",
        },
        "validation_thresholds": {
            "fvt_density_ratio_min": policy_evaluation.DEFAULT_DENSITY_RATIO_MIN,
            "fvt_density_ratio_max": policy_evaluation.DEFAULT_DENSITY_RATIO_MAX,
            "edge_density_max_delta": policy_evaluation.DEFAULT_EDGE_DENSITY_MAX_DELTA,
            "public_fvt_sparse_distance_p95_max_delta": (
                policy_evaluation.DEFAULT_SPARSE_DISTANCE_P95_MAX_DELTA
            ),
            "candidate_fvt_density_cv_max": (policy_evaluation.DEFAULT_CROP_STABILITY_MAX_CV),
        },
        "outputs": {
            "save_volumes": bool(save_volumes),
            "save_figures": bool(save_figures),
            "write_markdown_index": bool(write_markdown_index),
            "figure_percentile": float(figure_percentile),
            "ridge_buffer_radius": float(ridge_buffer_radius),
        },
    }


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


def validate_diagnostic_options(
    *,
    outlier_diagnostics: bool,
    context_crop_shape: tuple[int, int, int] | None,
    context_crop_indices: Iterable[int] | None,
    base_crop_shape: tuple[int, int, int],
    outlier_max_points: int,
    outlier_max_components: int,
    outlier_window_radius: int,
    outlier_adjacent_slice_radius: int,
    amplitude_clip_percentile: float,
) -> list[int] | None:
    if outlier_max_points < 1:
        raise ValueError("outlier_max_points must be >= 1")
    if outlier_max_components < 1:
        raise ValueError("outlier_max_components must be >= 1")
    if outlier_window_radius < 0:
        raise ValueError("outlier_window_radius must be >= 0")
    if outlier_adjacent_slice_radius < 0:
        raise ValueError("outlier_adjacent_slice_radius must be >= 0")
    if not math.isfinite(amplitude_clip_percentile) or not (
        0.0 < amplitude_clip_percentile <= 100.0
    ):
        raise ValueError("amplitude_clip_percentile must be finite and in (0, 100]")

    indices = None
    if context_crop_indices is not None:
        indices = [int(index) for index in context_crop_indices]
    if context_crop_shape is not None and not outlier_diagnostics:
        raise ValueError("--context-crop-shape requires --outlier-diagnostics")
    if indices and context_crop_shape is None:
        raise ValueError("--context-crop-index requires --context-crop-shape")
    if context_crop_shape is not None:
        for axis, (context_size, base_size) in enumerate(
            zip(context_crop_shape, base_crop_shape, strict=True)
        ):
            if context_size < base_size:
                raise ValueError(f"context_crop_shape[{axis}] must be >= base crop_shape[{axis}]")
    return indices


def validate_context_selection(
    *,
    context_crop_shape: tuple[int, int, int] | None,
    full_shape: tuple[int, int, int],
    context_crop_indices: list[int] | None,
    crop_count: int,
) -> list[int] | None:
    if context_crop_shape is None:
        return None
    for axis, (context_size, full_size) in enumerate(
        zip(context_crop_shape, full_shape, strict=True)
    ):
        if context_size > full_size:
            raise ValueError(f"context_crop_shape[{axis}] must be <= F3 full_shape[{axis}]")
    if context_crop_indices is None:
        return None
    invalid = [index for index in context_crop_indices if index < 1 or index > crop_count]
    if invalid:
        raise ValueError(
            "context_crop_index must be within the selected 1-origin crop range; "
            f"invalid: {invalid}"
        )
    return list(dict.fromkeys(context_crop_indices))


def run_context_diagnostics(
    *,
    arrays: Mapping[str, np.ndarray],
    base_states: list[dict[str, Any]],
    selected_crop_indices: list[int],
    context_crop_shape: tuple[int, int, int],
    pipeline_kwargs: Mapping[str, Any],
    save_figures: bool,
    output_base_dir: Path | None,
    outlier_window_radius: int,
    amplitude_clip_percentile: float,
) -> dict[str, Any]:
    state_by_index = {int(state["index"]): state for state in base_states}
    missing = [index for index in selected_crop_indices if index not in state_by_index]
    if missing:
        raise ValueError(f"context diagnostic base crop state is unavailable: {missing}")

    full_shape = tuple(int(size) for size in np.asarray(arrays["ep.dat"]).shape)
    context_crops: list[dict[str, Any]] = []
    context_scanner_execution_count = 0
    for crop_index in selected_crop_indices:
        state = state_by_index[crop_index]
        center = tuple(int(value) for value in state["center"])
        base_global_slices = state["crop_slices"]
        context_global_slices = crop_slices(
            center,
            context_crop_shape,
            full_shape=full_shape,
        )
        base_roi_slices = policy_diagnostics.map_base_roi_slices_within_context(
            base_global_slices,
            context_global_slices,
        )
        ep_context = _crop(np.asarray(arrays["ep.dat"]), context_global_slices)
        context_run = run_shared_scan_policy_pipeline(ep_context, **dict(pipeline_kwargs))
        context_scanner_execution_count += int(context_run["scanner_execution_count"])
        context_policies = _as_mapping(context_run.get("policies"))

        context_roi_outputs: dict[str, dict[str, np.ndarray]] = {}
        policy_reports: dict[str, Any] = {}
        for role in policy_evaluation.POLICY_ROLES:
            role_run = _as_mapping(context_policies.get(role))
            context_outputs = _as_array_mapping(role_run.get("outputs"))
            roi_outputs = policy_diagnostics.extract_same_global_roi_outputs(
                context_outputs,
                base_global_slices=base_global_slices,
                context_global_slices=context_global_slices,
            )
            context_roi_outputs[role] = roi_outputs
            base_outputs = _as_array_mapping(_as_mapping(state["outputs_by_role"]).get(role))
            policy_reports[role] = policy_diagnostics.build_same_global_roi_stage_comparison(
                base_outputs=base_outputs,
                context_roi_outputs=roi_outputs,
                ridge_percentile=policy_evaluation.DEFAULT_RIDGE_PERCENTILE,
                ridge_buffer_radius=policy_evaluation.DEFAULT_RIDGE_BUFFER_RADIUS,
                nonzero_epsilon=policy_evaluation.DEFAULT_NONZERO_EPSILON,
            )

        base_outputs_by_role = _as_mapping(state["outputs_by_role"])
        base_outlier_report = _as_mapping(state["outlier_report"])
        persistence = policy_diagnostics.build_context_outlier_persistence_report(
            base_outlier_report=base_outlier_report,
            reference_fvt=np.asarray(state["reference_fvt"]),
            base_baseline_outputs=_as_array_mapping(base_outputs_by_role.get("baseline")),
            base_candidate_outputs=_as_array_mapping(base_outputs_by_role.get("candidate")),
            context_baseline_outputs=context_roi_outputs["baseline"],
            context_candidate_outputs=context_roi_outputs["candidate"],
            base_global_slices=base_global_slices,
            ridge_percentile=policy_evaluation.DEFAULT_RIDGE_PERCENTILE,
            allowed_p95_delta=policy_evaluation.DEFAULT_SPARSE_DISTANCE_P95_MAX_DELTA,
            persistence_radius=2.0,
        )
        crop_report: dict[str, Any] = {
            "index": int(crop_index),
            "center": [int(value) for value in center],
            "base_global_slices": policy_diagnostics.slices_to_json(base_global_slices),
            "context_global_slices": policy_diagnostics.slices_to_json(context_global_slices),
            "base_roi_slices_within_context": policy_diagnostics.slices_to_json(base_roi_slices),
            "policies": policy_reports,
            "outlier_persistence": persistence,
        }

        if save_figures and int(_nested(base_outlier_report, "summary", "outlier_count") or 0):
            if output_base_dir is None or state.get("xs") is None:
                raise ValueError("context amplitude figures require --output-json and xs.dat")
            context_component_figures = write_context_diagnostic_figures(
                output_base_dir
                / f"crop_{crop_index:03d}"
                / "policy_comparison"
                / "context_diagnostics",
                metrics_base_dir=output_base_dir,
                xs=np.asarray(state["xs"]),
                base_candidate_fvt=np.asarray(
                    _as_array_mapping(base_outputs_by_role["candidate"])["fvt_py.dat"]
                ),
                context_candidate_fvt=np.asarray(context_roi_outputs["candidate"]["fvt_py.dat"]),
                crop_slices=base_global_slices,
                crop_index=crop_index,
                outlier_report=base_outlier_report,
                window_radius=outlier_window_radius,
                amplitude_clip_percentile=amplitude_clip_percentile,
            )
            crop_report["figures"] = {
                "components": [
                    {"component_id": int(component_id), **files}
                    for component_id, files in context_component_figures.items()
                ]
            }
            _attach_context_component_figures(
                base_outlier_report,
                context_component_figures,
            )
        context_crops.append(crop_report)

    return {
        "status": "available",
        "role": "diagnostic_context_ablation",
        "requested_context_crop_shape": [int(size) for size in context_crop_shape],
        "context_scanner_execution_count": int(context_scanner_execution_count),
        "selected_crop_indices": [int(index) for index in selected_crop_indices],
        "crops": context_crops,
    }


def build_pending_manual_review() -> dict[str, Any]:
    return {
        "status": "pending",
        "required_before_default_promotion": True,
        "reviewer": None,
        "reviewed_at": None,
        "items": {
            "major_fault_continuity": None,
            "weak_or_small_fault_retention": None,
            "candidate_only_noise": None,
            "unnatural_parallel_ridges": None,
            "crop_face_artifacts": None,
            "local_strike_dip_continuity": None,
            "baseline_only_geologic_structure_loss": None,
        },
        "notes": [],
    }


def build_nonfinite_crop_report(
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
    """Build validation evidence when comparison metrics cannot consume NaN/Inf."""

    local_interior = crop_validation.interior_slices(crop_shape, margin=interior_margin)
    global_interior = tuple(
        slice(crop_slice.start + local_slice.start, crop_slice.start + local_slice.stop)
        for crop_slice, local_slice in zip(slices, local_interior, strict=True)
    )
    return {
        "index": int(crop_index),
        "crop_center": [int(value) for value in center],
        "crop_slices": crop_validation.slices_to_json(slices),
        "interior_margin": int(interior_margin),
        "interior_slices": crop_validation.slices_to_json(global_interior),
        "interior_slices_in_crop": crop_validation.slices_to_json(local_interior),
        "crop_shape": [int(size) for size in crop_shape],
        "metrics_unavailable_reason": "policy outputs contain non-finite values",
        "finite_checks": {
            "pyosv": {
                name.removesuffix(".dat"): crop_validation.finite_report(values)
                for name, values in outputs.items()
            },
            "reference": {
                "fv": crop_validation.finite_report(reference_fv),
                "fvt": crop_validation.finite_report(reference_fvt),
            },
        },
    }


def build_unavailable_direct_comparison(
    *,
    metadata: Mapping[str, Any],
    baseline_crop: Mapping[str, Any],
    candidate_crop: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_density = _number(_nested(baseline_crop, "stage_density", "fvt", "nonzero_fraction"))
    candidate_density = _number(_nested(candidate_crop, "stage_density", "fvt", "nonzero_fraction"))
    return {
        **metadata,
        "metrics_unavailable_reason": "policy fvt outputs contain non-finite values",
        "fvt_density": {
            "baseline": baseline_density,
            "candidate": candidate_density,
            "candidate_over_baseline_ratio": _density_ratio(
                candidate_density,
                baseline_density,
            ),
        },
        "buffered_ridge_overlap": {"interior": {}},
        "sparse_ridge_distance_metrics": {"interior": {}},
        "ridge_mask_difference": {
            "mask_definition": "unavailable for non-finite fvt output",
        },
    }


def write_direct_comparison_figures(
    output_dir: str | PathLike[str],
    *,
    metrics_base_dir: str | PathLike[str],
    baseline_fvt: np.ndarray,
    candidate_fvt: np.ndarray,
    interior_margin: int,
    figure_percentile: float,
    ridge_buffer_radius: float,
) -> dict[str, Any]:
    from pyosv import viz

    baseline = np.asarray(baseline_fvt, dtype=np.float32)
    candidate = np.asarray(candidate_fvt, dtype=np.float32)
    if baseline.shape != candidate.shape:
        raise ValueError("baseline and candidate fvt shapes must match")
    directory = Path(output_dir)
    base_dir = Path(metrics_base_dir)
    slice_indices = viz.select_center_slices(baseline.shape)
    clip_percentiles = (1.0, float(figure_percentile))
    baseline_mask = baseline > np.float32(policy_evaluation.DEFAULT_NONZERO_EPSILON)
    candidate_mask = candidate > np.float32(policy_evaluation.DEFAULT_NONZERO_EPSILON)
    shell = np.ones(baseline.shape, dtype=bool)
    shell[crop_validation.interior_slices(baseline.shape, margin=interior_margin)] = False

    files = {
        "baseline_vs_candidate_fvt_slices": crop_validation.paths_for_metrics(
            viz.save_volume_comparison_slices(
                directory,
                reference=baseline,
                candidate=candidate,
                name="baseline_vs_candidate_fvt",
                slice_indices=slice_indices,
                clip_percentiles=clip_percentiles,
            ),
            base_dir,
        ),
        "baseline_candidate_ridge_overlay": crop_validation.paths_for_metrics(
            viz.save_buffered_ridge_overlay_slices(
                directory,
                reference=baseline,
                candidate=candidate,
                name="baseline_candidate_fvt",
                slice_indices=slice_indices,
                percentile=figure_percentile,
                buffer_radius=ridge_buffer_radius,
            ),
            base_dir,
        ),
        "candidate_only_baseline_only_ridge_mask": crop_validation.paths_for_metrics(
            viz.save_buffered_ridge_overlay_slices(
                directory,
                reference=baseline_mask.astype(np.float32),
                candidate=candidate_mask.astype(np.float32),
                name="baseline_candidate_positive_masks",
                slice_indices=slice_indices,
                percentile=0.0,
                buffer_radius=0.0,
            ),
            base_dir,
        ),
        "edge_shell_ridge_overlay": crop_validation.paths_for_metrics(
            viz.save_buffered_ridge_overlay_slices(
                directory,
                reference=(baseline_mask & shell).astype(np.float32),
                candidate=(candidate_mask & shell).astype(np.float32),
                name="edge_shell_ridges",
                slice_indices=slice_indices,
                percentile=0.0,
                buffer_radius=0.0,
            ),
            base_dir,
        ),
    }
    return {
        "directory": crop_validation.path_for_metrics(directory, base_dir),
        "figure_slices": "center",
        "slice_indices": {axis: int(index) for axis, index in slice_indices.items()},
        "files": files,
    }


def write_outlier_diagnostic_figures(
    output_dir: str | PathLike[str],
    *,
    metrics_base_dir: str | PathLike[str],
    xs: np.ndarray,
    reference_fvt: np.ndarray,
    baseline_fvt: np.ndarray,
    candidate_fvt: np.ndarray,
    crop_slices: tuple[slice, slice, slice],
    crop_index: int,
    outlier_report: Mapping[str, Any],
    window_radius: int,
    adjacent_slice_radius: int,
    amplitude_clip_percentile: float,
) -> dict[int, dict[str, Any]]:
    from pyosv import viz

    amplitude = np.asarray(xs, dtype=np.float32)
    arrays = {
        "public": np.asarray(reference_fvt, dtype=np.float32),
        "baseline": np.asarray(baseline_fvt, dtype=np.float32),
        "candidate": np.asarray(candidate_fvt, dtype=np.float32),
    }
    if any(values.shape != amplitude.shape for values in arrays.values()):
        raise ValueError("xs and public/baseline/candidate fvt shapes must match")
    percentile = float(
        _nested(outlier_report, "definition", "ridge_percentile")
        or policy_evaluation.DEFAULT_RIDGE_PERCENTILE
    )
    interior_margin = int(_nested(outlier_report, "definition", "interior_margin") or 0)
    masks = {
        name: _full_sparse_ridge_mask(
            values,
            percentile=percentile,
            interior_margin=interior_margin,
        )
        for name, values in arrays.items()
    }
    clip = symmetric_amplitude_clip(amplitude, amplitude_clip_percentile)
    directory = Path(output_dir)
    base_dir = Path(metrics_base_dir)
    global_start = tuple(int(value.start) for value in crop_slices)
    written: dict[int, dict[str, Any]] = {}
    for component_value in outlier_report.get("components", []):
        component = _as_mapping(component_value)
        component_id = int(component.get("component_id", 0))
        if component_id < 1:
            continue
        representative = _as_mapping(component.get("representative_point"))
        coordinate = _coordinate3(
            representative.get("crop_local_coordinate"),
            "representative crop-local coordinate",
        )
        nearest = _coordinate3(
            _nested(representative, "nearest_public_fvt", "crop_local_coordinate"),
            "nearest-public crop-local coordinate",
        )
        distance = _number(representative.get("distance_to_public_fvt"))
        if distance is None:
            raise ValueError("representative outlier distance must be finite")
        component_dir = directory / f"component_{component_id:03d}"
        files: dict[str, Any] = {
            "amplitude_clip": {
                "percentile": float(amplitude_clip_percentile),
                "absolute_value": True,
                "symmetric": True,
                "clip": clip,
                "vmin": -clip,
                "vmax": clip,
            }
        }
        orthogonal = viz.save_outlier_orthogonal_amplitude_overlay(
            component_dir / "orthogonal_amplitude_overlay.png",
            amplitude=amplitude,
            public_fvt_mask=masks["public"],
            baseline_fvt_mask=masks["baseline"],
            candidate_fvt_mask=masks["candidate"],
            representative_coordinate=coordinate,
            nearest_public_coordinate=nearest,
            crop_global_start=global_start,
            amplitude_clip=clip,
            window_radius=window_radius,
            crop_index=int(crop_index),
            component_id=component_id,
            distance_to_public=distance,
        )
        files["orthogonal_amplitude_overlay"] = crop_validation.path_for_metrics(
            orthogonal,
            base_dir,
        )
        for axis in ("i3", "i2", "i1"):
            adjacent = viz.save_outlier_adjacent_slice_overlay(
                component_dir / f"adjacent_{axis}.png",
                amplitude=amplitude,
                public_fvt_mask=masks["public"],
                baseline_fvt_mask=masks["baseline"],
                candidate_fvt_mask=masks["candidate"],
                representative_coordinate=coordinate,
                nearest_public_coordinate=nearest,
                crop_global_start=global_start,
                amplitude_clip=clip,
                window_radius=window_radius,
                adjacent_slice_radius=adjacent_slice_radius,
                axis=axis,
                crop_index=int(crop_index),
                component_id=component_id,
            )
            files[f"adjacent_{axis}"] = crop_validation.path_for_metrics(
                adjacent,
                base_dir,
            )
        written[component_id] = files
    return written


def write_context_diagnostic_figures(
    output_dir: str | PathLike[str],
    *,
    metrics_base_dir: str | PathLike[str],
    xs: np.ndarray,
    base_candidate_fvt: np.ndarray,
    context_candidate_fvt: np.ndarray,
    crop_slices: tuple[slice, slice, slice],
    crop_index: int,
    outlier_report: Mapping[str, Any],
    window_radius: int,
    amplitude_clip_percentile: float,
) -> dict[int, dict[str, Any]]:
    from pyosv import viz

    amplitude = np.asarray(xs, dtype=np.float32)
    base_fvt = np.asarray(base_candidate_fvt, dtype=np.float32)
    context_fvt = np.asarray(context_candidate_fvt, dtype=np.float32)
    if base_fvt.shape != amplitude.shape or context_fvt.shape != amplitude.shape:
        raise ValueError("xs and base/context candidate fvt shapes must match")
    percentile = float(
        _nested(outlier_report, "definition", "ridge_percentile")
        or policy_evaluation.DEFAULT_RIDGE_PERCENTILE
    )
    interior_margin = int(_nested(outlier_report, "definition", "interior_margin") or 0)
    base_mask = _full_sparse_ridge_mask(
        base_fvt,
        percentile=percentile,
        interior_margin=interior_margin,
    )
    context_mask = _full_sparse_ridge_mask(
        context_fvt,
        percentile=percentile,
        interior_margin=interior_margin,
    )
    clip = symmetric_amplitude_clip(amplitude, amplitude_clip_percentile)
    directory = Path(output_dir)
    base_dir = Path(metrics_base_dir)
    global_start = tuple(int(value.start) for value in crop_slices)
    written: dict[int, dict[str, Any]] = {}
    for component_value in outlier_report.get("components", []):
        component = _as_mapping(component_value)
        component_id = int(component.get("component_id", 0))
        if component_id < 1:
            continue
        representative = _as_mapping(component.get("representative_point"))
        coordinate = _coordinate3(
            representative.get("crop_local_coordinate"),
            "representative crop-local coordinate",
        )
        nearest = _coordinate3(
            _nested(representative, "nearest_public_fvt", "crop_local_coordinate"),
            "nearest-public crop-local coordinate",
        )
        component_dir = directory / f"component_{component_id:03d}"
        path = viz.save_context_orthogonal_amplitude_comparison(
            component_dir / "base_context_amplitude_overlay.png",
            amplitude=amplitude,
            base_candidate_fvt_mask=base_mask,
            context_candidate_fvt_mask=context_mask,
            representative_coordinate=coordinate,
            nearest_public_coordinate=nearest,
            crop_global_start=global_start,
            amplitude_clip=clip,
            window_radius=window_radius,
            crop_index=int(crop_index),
            component_id=component_id,
        )
        written[component_id] = {
            "context_comparison": crop_validation.path_for_metrics(path, base_dir),
            "amplitude_clip": {
                "percentile": float(amplitude_clip_percentile),
                "absolute_value": True,
                "symmetric": True,
                "clip": clip,
                "vmin": -clip,
                "vmax": clip,
            },
        }
    return written


def symmetric_amplitude_clip(xs: np.ndarray, percentile: float) -> float:
    values = np.asarray(xs)
    finite = values[np.isfinite(values)]
    if finite.size:
        clip = float(np.percentile(np.abs(finite.astype(np.float64, copy=False)), percentile))
        if math.isfinite(clip) and clip > 0.0:
            return clip
    return 1.0


def _full_sparse_ridge_mask(
    values: np.ndarray,
    *,
    percentile: float,
    interior_margin: int,
) -> np.ndarray:
    array = np.asarray(values)
    local_interior = crop_validation.interior_slices(
        array.shape,
        margin=interior_margin,
    )
    mask = np.zeros(array.shape, dtype=bool)
    mask[local_interior] = top_percentile_mask(
        array[local_interior],
        percentile=percentile,
        positive_only=True,
    )
    return mask


def _attach_component_figures(
    outlier_report: Mapping[str, Any],
    figures: Mapping[int, Mapping[str, Any]],
) -> None:
    for component_value in outlier_report.get("components", []):
        if not isinstance(component_value, dict):
            continue
        component_id = int(component_value.get("component_id", 0))
        if component_id in figures:
            component_value["figures"] = dict(figures[component_id])


def _attach_context_component_figures(
    outlier_report: Mapping[str, Any],
    figures: Mapping[int, Mapping[str, Any]],
) -> None:
    for component_value in outlier_report.get("components", []):
        if not isinstance(component_value, dict):
            continue
        component_id = int(component_value.get("component_id", 0))
        context_files = figures.get(component_id)
        if context_files is None:
            continue
        existing = component_value.get("figures")
        if not isinstance(existing, dict):
            existing = {}
            component_value["figures"] = existing
        existing.update(context_files)


def visual_report_markdown(report: Mapping[str, Any]) -> str:
    config = _as_mapping(report.get("config"))
    crop_selection = _as_mapping(config.get("crop_selection"))
    policies = _as_mapping(report.get("scanner_policies"))
    baseline_crops = list(_as_mapping(policies.get("baseline")).get("crops", []))
    candidate_crops = list(_as_mapping(policies.get("candidate")).get("crops", []))
    validation = _as_mapping(report.get("policy_validation"))
    manual_review = _as_mapping(report.get("manual_review"))
    comparisons = list(
        _as_mapping(_as_mapping(report.get("consensus")).get("candidate_minus_baseline")).get(
            "crops", []
        )
    )
    lines = [
        "# F3 Scanner-Thinning Policy Visual Report",
        "",
        f"- comparison_profile: `{config.get('comparison_profile', '')}`",
        f"- baseline: `{_as_mapping(policies.get('baseline')).get('policy_id', '')}`",
        f"- candidate: `{_as_mapping(policies.get('candidate')).get('policy_id', '')}`",
        f"- crop_shape: `{crop_selection.get('crop_shape', '')}`",
        f"- centers: `{crop_selection.get('centers', '')}`",
        f"- scanner_execution_count: `{validation.get('scanner_execution_count', '')}`",
        f"- policy_validation.passed: `{validation.get('passed', '')}`",
        f"- manual_review.status: `{manual_review.get('status', '')}`",
        f"- manual_review.reviewer: `{manual_review.get('reviewer', '')}`",
        f"- manual_review.reviewed_at: `{manual_review.get('reviewed_at', '')}`",
        "",
        "F3 `fv.dat` and `fvt.dat` are public workflow outputs, not independent truth. "
        "These checks are a conservative truthless external smoke; default promotion still "
        "requires manual geological review.",
        "",
        "## Automated Validation",
        "",
        "| Check | Passed | Details |",
        "| --- | ---: | --- |",
    ]
    for name, check in _as_mapping(validation.get("checks")).items():
        check_map = _as_mapping(check)
        details = {key: value for key, value in check_map.items() if key not in {"passed"}}
        lines.append(f"| {name} | {check_map.get('passed', '')} | `{details}` |")

    lines.extend(
        [
            "",
            "## Per-Crop Policy Metrics",
            "",
            "| Crop | Baseline FVT density | Candidate FVT density | C/B ratio | "
            "Baseline edge proxy | Candidate edge proxy | Baseline public-FVT p95 | "
            "Candidate public-FVT p95 | Baseline buffered P/R | Candidate buffered P/R | "
            "Crop checks |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
        ]
    )
    for baseline, candidate, comparison in zip(
        baseline_crops,
        candidate_crops,
        comparisons,
        strict=False,
    ):
        baseline_map = _as_mapping(baseline)
        candidate_map = _as_mapping(candidate)
        comparison_map = _as_mapping(comparison)
        lines.append(
            "| "
            f"crop_{int(baseline_map.get('index', 0)):03d} | "
            f"{_format_metric(_nested(baseline_map, 'stage_density', 'fvt', 'nonzero_fraction'))} | "
            f"{_format_metric(_nested(candidate_map, 'stage_density', 'fvt', 'nonzero_fraction'))} | "
            f"{_format_metric(_nested(comparison_map, 'fvt_density', 'candidate_over_baseline_ratio'))} | "
            f"{_format_metric(_nested(baseline_map, 'stage_density', 'fvt', 'edge_density_proxy'))} | "
            f"{_format_metric(_nested(candidate_map, 'stage_density', 'fvt', 'edge_density_proxy'))} | "
            f"{_format_metric(_public_distance_p95(baseline_map))} | "
            f"{_format_metric(_public_distance_p95(candidate_map))} | "
            f"{_format_precision_recall(baseline_map)} | "
            f"{_format_precision_recall(candidate_map)} | "
            f"{_crop_validation_label(baseline_map, candidate_map, comparison_map)} |"
        )

    lines.extend(["", "## Side-by-Side Visual Diagnostics", ""])
    any_figures = False
    for baseline, candidate, comparison in zip(
        baseline_crops,
        candidate_crops,
        comparisons,
        strict=False,
    ):
        baseline_map = _as_mapping(baseline)
        candidate_map = _as_mapping(candidate)
        comparison_map = _as_mapping(comparison)
        crop_id = f"crop_{int(baseline_map.get('index', 0)):03d}"
        baseline_link = _nested(baseline_map, "figures", "files", "fvt_ref_vs_py", "i3")
        candidate_link = _nested(candidate_map, "figures", "files", "fvt_ref_vs_py", "i3")
        direct_files = _as_mapping(_nested(comparison_map, "figures", "files"))
        if isinstance(baseline_link, str) or isinstance(candidate_link, str) or direct_files:
            any_figures = True
            lines.extend(
                [
                    f"### {crop_id}",
                    "",
                    "| Baseline | Candidate |",
                    "| --- | --- |",
                    f"| {_image_or_blank(baseline_link, 'baseline FVT')} | "
                    f"{_image_or_blank(candidate_link, 'candidate FVT')} |",
                    "",
                ]
            )
            for label, key in (
                ("baseline vs candidate slices", "baseline_vs_candidate_fvt_slices"),
                ("baseline/candidate ridge overlay", "baseline_candidate_ridge_overlay"),
                ("candidate-only / baseline-only masks", "candidate_only_baseline_only_ridge_mask"),
                ("edge-shell ridge overlay", "edge_shell_ridge_overlay"),
            ):
                path = _nested(direct_files, key, "i3")
                if isinstance(path, str):
                    lines.append(f"- [{label}]({path})")
            lines.append("")
    if not any_figures:
        lines.append("No PNG figures were written for this run.")

    if any("public_fvt_distance_outliers" in comparison for comparison in comparisons):
        lines.extend(
            [
                "",
                "## Public-FVT Distance Outlier Review",
                "",
                "Public FVT is a comparison reference, not independent truth. The rows below "
                "preserve the automatic distance failure and provide amplitude/context "
                "evidence for human adjudication.",
                "",
                "| Crop | Status | Baseline p95 | Candidate p95 | Delta | Allowed candidate "
                "p95 | Outliers | Components | Crop-face distance min/max |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for comparison_value in comparisons:
            comparison = _as_mapping(comparison_value)
            diagnostics = _as_mapping(comparison.get("public_fvt_distance_outliers"))
            if not diagnostics:
                continue
            summary = _as_mapping(diagnostics.get("summary"))
            lines.append(
                "| "
                f"crop_{int(comparison.get('index', 0)):03d} | "
                f"{diagnostics.get('status', '')} | "
                f"{_format_metric(summary.get('baseline_candidate_to_public_p95'))} | "
                f"{_format_metric(summary.get('candidate_candidate_to_public_p95'))} | "
                f"{_format_metric(summary.get('candidate_minus_baseline_p95'))} | "
                f"{_format_metric(summary.get('allowed_candidate_p95'))} | "
                f"{summary.get('outlier_count', '')} | "
                f"{summary.get('component_count', '')} | "
                f"{_format_metric(summary.get('minimum_crop_face_distance'))}/"
                f"{_format_metric(summary.get('maximum_crop_face_distance'))} |"
            )
        lines.append("")
        for comparison_value in comparisons:
            comparison = _as_mapping(comparison_value)
            diagnostics = _as_mapping(comparison.get("public_fvt_distance_outliers"))
            if not diagnostics:
                continue
            if diagnostics.get("status") != "available":
                lines.append(
                    f"- crop_{int(comparison.get('index', 0)):03d}: "
                    f"unavailable: {diagnostics.get('reason', '')}"
                )
                continue
            for component_value in diagnostics.get("components", []):
                component = _as_mapping(component_value)
                representative = _as_mapping(component.get("representative_point"))
                figures = _as_mapping(component.get("figures"))
                component_id = int(component.get("component_id", 0))
                lines.extend(
                    [
                        "",
                        f"### crop_{int(comparison.get('index', 0)):03d} / "
                        f"component_{component_id:03d}",
                        "",
                        f"- voxel_count: `{component.get('voxel_count', '')}`",
                        f"- representative_global_coordinate: "
                        f"`{representative.get('global_coordinate', '')}`",
                        f"- max_distance_to_public_fvt: "
                        f"`{_format_metric(_nested(component, 'distance_to_public_fvt', 'maximum'))}`",
                        f"- minimum_crop_face_distance: "
                        f"`{component.get('minimum_crop_face_distance', '')}`",
                        "",
                    ]
                )
                orthogonal = figures.get("orthogonal_amplitude_overlay")
                if isinstance(orthogonal, str):
                    lines.extend(
                        [
                            _image_or_blank(
                                orthogonal,
                                f"crop {comparison.get('index', '')} component "
                                f"{component_id} orthogonal amplitude review",
                            ),
                            "",
                        ]
                    )
                adjacent_links = [
                    f"[{axis} adjacent slices]({figures[f'adjacent_{axis}']})"
                    for axis in ("i3", "i2", "i1")
                    if isinstance(figures.get(f"adjacent_{axis}"), str)
                ]
                if adjacent_links:
                    lines.append("- " + "; ".join(adjacent_links))
                context_link = figures.get("context_comparison")
                if isinstance(context_link, str):
                    lines.append(f"- [same-global-ROI context comparison]({context_link})")

    manual_items = _as_mapping(manual_review.get("items"))
    lines.extend(["", "## Manual Geological Review", ""])
    for item, result in manual_items.items():
        marker = "x" if result is True else " "
        suffix = " (failed)" if result is False else ""
        lines.append(f"- [{marker}] {item}{suffix}")
    lines.extend(
        [
            "",
            "Record reviewer, notes, and pass/fail results in the compact evidence manifest "
            "before changing any default.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report_json(
    report: Mapping[str, Any],
    output_json: str | PathLike[str],
    *,
    pretty: bool = False,
) -> Path:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        policy_evaluation.report_to_json(report, pretty=pretty), encoding="utf-8"
    )
    return output_path


def write_visual_report_markdown(
    report: Mapping[str, Any],
    output_path: str | PathLike[str],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(visual_report_markdown(report), encoding="utf-8")
    return path


def report_to_json(report: Mapping[str, Any], *, pretty: bool = False) -> str:
    return policy_evaluation.report_to_json(report, pretty=pretty)


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
        (ISSUE_FORGE_DIR, "vendor/issue_forge"),
    ):
        try:
            resolved_output.relative_to(forbidden_root.resolve(strict=False))
        except ValueError:
            continue
        raise ValueError(f"{option_name} must not be inside {label}: {resolved_output}")


def _crop_validation_label(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> str:
    ratio = _number(_nested(comparison, "fvt_density", "candidate_over_baseline_ratio"))
    edge_delta = _delta(
        _nested(candidate, "stage_density", "fvt", "edge_density_proxy"),
        _nested(baseline, "stage_density", "fvt", "edge_density_proxy"),
    )
    distance_delta = _delta(_public_distance_p95(candidate), _public_distance_p95(baseline))
    nonempty = all(
        (_number(_nested(crop, "stage_density", stage, "nonzero_count")) or 0.0) > 0.0
        for crop in (baseline, candidate)
        for stage in ("fet", "fv", "fvt")
    )
    passed = (
        ratio is not None
        and 0.5 <= ratio <= 2.0
        and edge_delta is not None
        and edge_delta <= 0.10
        and distance_delta is not None
        and distance_delta <= 5.0
        and nonempty
    )
    return "pass" if passed else "fail"


def _public_distance_p95(crop: Mapping[str, Any]) -> Any:
    return _nested(
        crop,
        "sparse_ridge_distance_metrics",
        "interior",
        "fvt",
        "candidate_to_reference_p95",
    )


def _format_precision_recall(crop: Mapping[str, Any]) -> str:
    precision = _format_metric(
        _nested(crop, "buffered_ridge_overlap", "interior", "fvt", "buffered_precision")
    )
    recall = _format_metric(
        _nested(crop, "buffered_ridge_overlap", "interior", "fvt", "buffered_recall")
    )
    return f"{precision}/{recall}"


def _image_or_blank(path: Any, label: str) -> str:
    return f"![{label}]({path})" if isinstance(path, str) else ""


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_array_mapping(value: Any) -> Mapping[str, np.ndarray]:
    if not isinstance(value, Mapping):
        raise ValueError("policy outputs must be a mapping")
    missing = [name for name in policy_evaluation.OUTPUT_NAMES if name not in value]
    if missing:
        raise ValueError("policy outputs are missing: " + ", ".join(missing))
    return value


def _outputs_are_finite(outputs: Mapping[str, np.ndarray]) -> bool:
    return all(
        np.all(np.isfinite(np.asarray(outputs[name]))) for name in policy_evaluation.OUTPUT_NAMES
    )


def _density_ratio(candidate: Any, baseline: Any) -> float | None:
    candidate_value = _number(candidate)
    baseline_value = _number(baseline)
    if candidate_value is None or baseline_value is None:
        return None
    if baseline_value == 0.0:
        return 1.0 if candidate_value == 0.0 else None
    return candidate_value / baseline_value


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.generic)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _coordinate3(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three coordinates")
    coordinates: list[int] = []
    for coordinate in value:
        if not isinstance(coordinate, (int, np.integer)) or isinstance(
            coordinate, (bool, np.bool_)
        ):
            raise ValueError(f"{name} must contain integer coordinates")
        coordinates.append(int(coordinate))
    return tuple(coordinates)  # type: ignore[return-value]


def _delta(candidate: Any, baseline: Any) -> float | None:
    candidate_value = _number(candidate)
    baseline_value = _number(baseline)
    if candidate_value is None or baseline_value is None:
        return None
    return candidate_value - baseline_value


def _format_metric(value: Any) -> str:
    number = _number(value)
    return f"{number:.4g}" if number is not None else ""


def _crop(array: np.ndarray, slices: tuple[slice, slice, slice]) -> np.ndarray:
    return np.ascontiguousarray(array[slices].astype(np.float32, copy=False))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = run_example(
            data_root_arg=args.data_root,
            comparison_profile=args.comparison_profile,
            output_json=args.output_json,
            save_volumes=args.save_volumes,
            save_figures=args.save_figures,
            write_markdown_index=args.write_markdown_index,
            outlier_diagnostics=args.outlier_diagnostics,
            context_crop_shape=args.context_crop_shape,
            context_crop_indices=args.context_crop_index,
            outlier_max_points=args.outlier_max_points,
            outlier_max_components=args.outlier_max_components,
            outlier_window_radius=args.outlier_window_radius,
            outlier_adjacent_slice_radius=args.outlier_adjacent_slice_radius,
            amplitude_clip_percentile=args.amplitude_clip_percentile,
            pretty=args.pretty,
            count=args.count,
            crop_shape=args.crop_shape,
            interior_margin=args.interior_margin,
            centers=args.center,
            percentile=args.percentile,
            min_separation=args.min_separation,
            figure_percentile=args.figure_percentile,
            ridge_buffer_radius=args.ridge_buffer_radius,
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
            surface_orientation_smoothing=args.surface_orientation_smoothing,
            d=args.d,
            fm=args.fm,
        )
    except (FileNotFoundError, KeyError, NotADirectoryError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.output_json is None:
        print(report_to_json(report, pretty=args.pretty), end="")
    else:
        print(args.output_json)
    if args.fail_on_validation_failure and not report["policy_validation"]["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
