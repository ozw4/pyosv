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
from pyosv.f3d_reference import (
    F3D_ENV_VAR,
    crop_slices,
    parse_shape3,
    pick_reference_centers,
    read_f3d_file,
    resolve_f3d_data_root,
)

DEFAULT_COUNT = 3
DEFAULT_CROP_SHAPE = (64, 64, 64)
DEFAULT_INTERIOR_MARGIN = 16
DEFAULT_PERCENTILE = 99.9
DEFAULT_MIN_SEPARATION = 48.0
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
    if save_figures:
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

    for crop_index, center in enumerate(selected_centers, start=1):
        slices = crop_slices(center, crop_shape, full_shape=arrays["ep.dat"].shape)
        ep_crop = _crop(arrays["ep.dat"], slices)
        reference_fv = _crop(arrays["fv.dat"], slices)
        reference_fvt = _crop(arrays["fvt.dat"], slices)
        reference_fl = _crop(arrays["fl.dat"], slices) if save_figures else None

        run = run_shared_scan_policy_pipeline(
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
            surface_orientation_smoothing=surface_orientation_smoothing,
            final_normalization_smoothing=None,
            d=d,
            fm=fm,
            reference_thin_sigma=1.0,
        )
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
