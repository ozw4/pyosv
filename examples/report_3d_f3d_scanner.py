"""Compare scanner-only F3 3D output against public fl.dat crops."""

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
DEFAULT_SCANNER_BACKENDS = ("current",)
SUPPORTED_SCANNER_BACKENDS = ("current", "reference-like")


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
        "--save-figures",
        action="store_true",
        help="Write per-crop PNG diagnostics under OUTPUT_JSON.parent.",
    )
    parser.add_argument(
        "--figure-percentile",
        type=float,
        default=99.0,
        help="Upper display clipping percentile for PNG diagnostics.",
    )
    parser.add_argument(
        "--write-markdown-index",
        action="store_true",
        help="Write visual_report.md next to the metrics JSON.",
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
    parser.add_argument(
        "--max-crops",
        "--count",
        dest="max_crops",
        type=int,
        default=1,
        help="Maximum number of deterministic crops.",
    )
    parser.add_argument(
        "--interior-margin",
        type=int,
        default=0,
        help="Boundary margin excluded from interior metrics.",
    )
    parser.add_argument(
        "--scanner-backends",
        type=parse_scanner_backends,
        default=list(DEFAULT_SCANNER_BACKENDS),
        help="Comma-separated scanner backends: current,reference-like.",
    )
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
    save_figures: bool = False,
    figure_percentile: float = 99.0,
    write_markdown_index: bool = False,
    pretty: bool = False,
    crop_shape: tuple[int, int, int] = (64, 64, 64),
    max_crops: int = 1,
    interior_margin: int = 0,
    scanner_backends: Iterable[str] = DEFAULT_SCANNER_BACKENDS,
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
    elif save_figures:
        raise ValueError("--save-figures requires --output-json")
    elif write_markdown_index:
        raise ValueError("--write-markdown-index requires --output-json")
    if output_dir is not None:
        ensure_output_not_in_data_root(output_dir, data_root, option_name="--output-dir")
    elif save_volumes:
        raise ValueError("--save-volumes requires --output-dir")
    if save_figures:
        require_figure_support()
    if max_crops < 0:
        raise ValueError("max_crops must be >= 0")

    backend_names = validate_scanner_backends(scanner_backends)
    crop_shape, interior_margin = validate_crop_config(crop_shape, interior_margin)
    output_base_dir = Path(output_json).parent if output_json is not None else None

    arrays = read_reference_arrays(data_root)
    config = build_config(
        crop_shape=crop_shape,
        max_crops=max_crops,
        interior_margin=interior_margin,
        scanner_backends=backend_names,
        percentile=percentile,
        min_separation=min_separation,
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
        backend_outputs = {
            backend_name: run_scanner(
                ep_crop,
                backend=backend_name,
                sigma1=sigma1,
                sigma2=sigma2,
                phi_min=phi_min,
                phi_max=phi_max,
                theta_min=theta_min,
                theta_max=theta_max,
            )
            for backend_name in backend_names
        }

        for backend_name, outputs in backend_outputs.items():
            if output_dir is not None and save_volumes:
                crop_volume_dir = Path(output_dir) / f"crop_{crop_index:03d}"
                if backend_names != ["current"]:
                    crop_volume_dir /= backend_name
                write_crop_volumes(crop_volume_dir, outputs)

        crop_report = build_backend_crop_report(
            crop_index=crop_index,
            center=center,
            slices=slices,
            backend_outputs=backend_outputs,
            reference_fl=reference_fl,
            interior_margin=interior_margin,
        )
        if save_figures:
            if output_base_dir is None:
                raise ValueError("--save-figures requires --output-json")
            crop_report["figures"] = {}
            for backend_name, outputs in backend_outputs.items():
                crop_report["figures"][backend_name] = write_backend_figures(
                    output_base_dir / f"crop_{crop_index:03d}" / backend_name / "figures",
                    metrics_base_dir=output_base_dir,
                    reference_fl=reference_fl,
                    outputs=outputs,
                    figure_percentile=figure_percentile,
                )
            if set(backend_names) >= {"current", "reference-like"}:
                crop_report["figures"]["backend_difference"] = write_backend_difference_figures(
                    output_base_dir / f"crop_{crop_index:03d}" / "backend_difference" / "figures",
                    metrics_base_dir=output_base_dir,
                    current_ft=backend_outputs["current"]["ft_py.dat"],
                    reference_like_ft=backend_outputs["reference-like"]["ft_py.dat"],
                    figure_percentile=figure_percentile,
                )
        crops.append(crop_report)

    report = build_report(data_root=data_root, config=config, crops=crops)
    if output_json is not None:
        write_report_json(report, output_json, pretty=pretty)
        if write_markdown_index:
            write_visual_report_markdown(report, Path(output_json).parent / "visual_report.md")

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
    interior_margin: int = 0,
    scanner_backends: Iterable[str] = DEFAULT_SCANNER_BACKENDS,
    save_figures: bool = False,
    figure_percentile: float = 99.0,
    write_markdown_index: bool = False,
    visual_report_path: Path | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "input": "ep.dat",
        "reference": "fl.dat",
        "comparison": "f3d_scanner_backend_comparison",
        "scanner_backends": list(scanner_backends),
        "crop_selection": {
            "source": "fl.dat",
            "crop_shape": [int(size) for size in crop_shape],
            "max_crops": int(max_crops),
            "percentile": float(percentile),
            "min_separation": float(min_separation),
            "boundary_margin": "crop_shape",
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
        "overlap_percentiles": [float(p) for p in OVERLAP_PERCENTILES],
        "outputs": list(VOLUME_NAMES),
    }
    if save_figures or write_markdown_index:
        config["visualization"] = {
            "save_figures": bool(save_figures),
            "figure_percentile": float(figure_percentile),
            "figure_slices": "center",
            "write_markdown_index": bool(write_markdown_index),
            "markdown_index": (visual_report_path.name if visual_report_path is not None else None),
        }
    return config


def run_scanner(
    ep: np.ndarray,
    *,
    backend: str = "current",
    sigma1: float,
    sigma2: float,
    phi_min: float,
    phi_max: float,
    theta_min: float,
    theta_max: float,
) -> dict[str, np.ndarray]:
    from pyosv.orient3d import FaultOrientScanner3

    scanner = FaultOrientScanner3(sigma1=sigma1, sigma2=sigma2)
    if backend == "current":
        ft, pt, tt = scanner.scan(phi_min, phi_max, theta_min, theta_max, ep)
    elif backend == "reference-like":
        scan_reference_like = getattr(scanner, "scan_reference_like", None)
        if scan_reference_like is None:
            raise NotImplementedError("FaultOrientScanner3.scan_reference_like is unavailable")
        ft, pt, tt = scan_reference_like(phi_min, phi_max, theta_min, theta_max, ep)
    else:
        raise ValueError(f"unknown scanner backend: {backend}")
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
        "aggregate": aggregate_backend_metrics(crops),
    }


def build_backend_crop_report(
    *,
    crop_index: int,
    center: tuple[int, int, int],
    slices: tuple[slice, slice, slice],
    backend_outputs: Mapping[str, Mapping[str, np.ndarray]],
    reference_fl: np.ndarray,
    interior_margin: int,
) -> dict[str, Any]:
    backends = {
        backend_name: build_backend_metrics(
            outputs=outputs,
            reference_fl=reference_fl,
            interior_margin=interior_margin,
        )
        for backend_name, outputs in backend_outputs.items()
    }
    first_outputs = next(iter(backend_outputs.values()))
    crop_shape = tuple(int(size) for size in np.asarray(reference_fl).shape)
    local_interior_slices = interior_slices(crop_shape, margin=interior_margin)
    global_interior_slices = tuple(
        slice(crop_slice.start + local_slice.start, crop_slice.start + local_slice.stop)
        for crop_slice, local_slice in zip(slices, local_interior_slices, strict=True)
    )
    report = build_crop_report(
        crop_index=crop_index,
        center=center,
        slices=slices,
        outputs=first_outputs,
        reference_fl=reference_fl,
    )
    report.update(
        {
            "crop_center": [int(value) for value in center],
            "crop_slices": slices_to_json(slices),
            "interior_margin": int(interior_margin),
            "interior_slices": slices_to_json(global_interior_slices),
            "interior_slices_in_crop": slices_to_json(local_interior_slices),
            "backends": backends,
        }
    )
    return report


def build_backend_metrics(
    *,
    outputs: Mapping[str, np.ndarray],
    reference_fl: np.ndarray,
    interior_margin: int,
) -> dict[str, Any]:
    ft_py = np.asarray(outputs["ft_py.dat"])
    pt_py = np.asarray(outputs["pt_py.dat"])
    tt_py = np.asarray(outputs["tt_py.dat"])
    fl_ref = np.asarray(reference_fl)
    local_interior_slices = interior_slices(ft_py.shape, margin=interior_margin)
    ft_interior = ft_py[local_interior_slices]
    fl_interior = fl_ref[local_interior_slices]

    return {
        "pyosv": {
            "ft_py": summarize_array(ft_py),
            "pt_py": summarize_array(pt_py),
            "tt_py": summarize_array(tt_py),
        },
        "reference": {
            "fl": summarize_array(fl_ref),
        },
        "normalized_correlation": {
            "full_crop": {
                "ft_vs_fl": float(normalized_correlation(ft_py, fl_ref)),
            },
            "interior": {
                "ft_vs_fl": float(normalized_correlation(ft_interior, fl_interior)),
            },
        },
        "top_percentile_overlap": {
            "full_crop": {
                "ft_vs_fl": _overlaps(ft_py, fl_ref),
            },
            "interior": {
                "ft_vs_fl": _overlaps(ft_interior, fl_interior),
            },
        },
        "slice_correlation": {
            "ft_py_vs_fl_i3": slice_correlation_summary(ft_py, fl_ref),
        },
        "finite_value_report": {
            "ft_py": finite_report(ft_py),
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


def parse_scanner_backends(text: str) -> list[str]:
    backends = [part.strip() for part in text.split(",") if part.strip()]
    try:
        return validate_scanner_backends(backends)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def validate_scanner_backends(backends: Iterable[str]) -> list[str]:
    backend_list = list(backends)
    if not backend_list:
        raise ValueError("scanner_backends must contain at least one backend")
    unknown = [name for name in backend_list if name not in SUPPORTED_SCANNER_BACKENDS]
    if unknown:
        supported = ", ".join(SUPPORTED_SCANNER_BACKENDS)
        raise ValueError(f"unknown scanner backend: {unknown[0]} (supported: {supported})")
    return backend_list


def validate_crop_config(
    crop_shape: tuple[int, int, int],
    interior_margin: int,
) -> tuple[tuple[int, int, int], int]:
    if interior_margin < 0:
        raise ValueError("interior_margin must be >= 0")
    interior_slices(crop_shape, margin=interior_margin)
    return crop_shape, int(interior_margin)


def interior_slices(shape: tuple[int, int, int], *, margin: int) -> tuple[slice, slice, slice]:
    slices = []
    for size in shape:
        start = int(margin)
        stop = int(size) - int(margin)
        if stop <= start:
            raise ValueError("interior_margin must leave a non-empty interior")
        slices.append(slice(start, stop))
    return tuple(slices)  # type: ignore[return-value]


def slices_to_json(slices: tuple[slice, slice, slice]) -> list[dict[str, int | str]]:
    return [
        {"axis": axis, "start": int(crop_slice.start), "stop": int(crop_slice.stop)}
        for axis, crop_slice in zip(("i3", "i2", "i1"), slices, strict=True)
    ]


def aggregate_backend_metrics(crops: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    crop_list = list(crops)
    backend_names = sorted(
        {
            str(backend_name)
            for crop in crop_list
            for backend_name in _as_mapping(crop.get("backends", {}))
        }
    )
    return {
        "crop_count": len(crop_list),
        "backends": {
            backend_name: aggregate_single_backend(crop_list, backend_name)
            for backend_name in backend_names
        },
    }


def aggregate_single_backend(
    crops: Iterable[Mapping[str, Any]],
    backend_name: str,
) -> dict[str, Any]:
    values_by_path: dict[str, list[float | None]] = {}
    crop_count = 0
    for crop in crops:
        backend = _as_mapping(_as_mapping(crop.get("backends", {})).get(backend_name, {}))
        if not backend:
            continue
        crop_count += 1
        for root in (
            "normalized_correlation",
            "top_percentile_overlap",
            "slice_correlation",
            "finite_value_report",
        ):
            if root in backend:
                for path, value in _flatten_numeric(backend[root], prefix=root):
                    values_by_path.setdefault(path, []).append(value)

    metric_paths = sorted(values_by_path)
    summaries: dict[str, Any] = {
        "crop_count": crop_count,
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


def require_figure_support() -> None:
    from pyosv import viz

    try:
        viz.require_matplotlib()
    except ImportError as error:
        raise ValueError(str(error)) from error


def write_backend_figures(
    output_dir: str | PathLike[str],
    *,
    metrics_base_dir: str | PathLike[str],
    reference_fl: np.ndarray,
    outputs: Mapping[str, np.ndarray],
    figure_percentile: float,
) -> dict[str, Any]:
    from pyosv import viz

    directory = Path(output_dir)
    base_dir = Path(metrics_base_dir)
    slice_indices = viz.select_center_slices(np.asarray(reference_fl).shape)
    clip_percentiles = (1.0, float(figure_percentile))
    written: dict[str, Any] = {
        "directory": path_for_metrics(directory, base_dir),
        "figure_slices": "center",
        "slice_indices": {axis: int(index) for axis, index in slice_indices.items()},
        "figure_percentile": float(figure_percentile),
        "files": {},
    }
    written["files"]["scanner_fl_vs_ftpy"] = paths_for_metrics(
        viz.save_volume_comparison_slices(
            directory,
            reference=reference_fl,
            candidate=outputs["ft_py.dat"],
            name="scanner_fl_vs_ftpy",
            slice_indices=slice_indices,
            clip_percentiles=clip_percentiles,
        ),
        base_dir,
    )
    written["files"]["ft"] = paths_for_metrics(
        viz.save_volume_diagnostics(
            directory,
            reference=reference_fl,
            candidate=outputs["ft_py.dat"],
            name="ft",
            clip_percentiles=clip_percentiles,
        ),
        base_dir,
    )
    return written


def write_backend_difference_figures(
    output_dir: str | PathLike[str],
    *,
    metrics_base_dir: str | PathLike[str],
    current_ft: np.ndarray,
    reference_like_ft: np.ndarray,
    figure_percentile: float,
) -> dict[str, Any]:
    from pyosv import viz

    directory = Path(output_dir)
    base_dir = Path(metrics_base_dir)
    slice_indices = viz.select_center_slices(np.asarray(current_ft).shape)
    clip_percentiles = (1.0, float(figure_percentile))
    written: dict[str, Any] = {
        "directory": path_for_metrics(directory, base_dir),
        "figure_slices": "center",
        "slice_indices": {axis: int(index) for axis, index in slice_indices.items()},
        "figure_percentile": float(figure_percentile),
        "files": {},
    }
    written["files"]["current_vs_reference_like_ft"] = paths_for_metrics(
        viz.save_volume_comparison_slices(
            directory,
            reference=current_ft,
            candidate=reference_like_ft,
            name="current_vs_reference_like_ft",
            slice_indices=slice_indices,
            clip_percentiles=clip_percentiles,
        ),
        base_dir,
    )
    return written


def paths_for_metrics(paths: Mapping[str, Path], base_dir: Path) -> dict[str, str]:
    return {key: path_for_metrics(path, base_dir) for key, path in paths.items()}


def path_for_metrics(path: str | PathLike[str], base_dir: str | PathLike[str]) -> str:
    output_path = Path(path)
    resolved_base = Path(base_dir).resolve(strict=False)
    try:
        return output_path.resolve(strict=False).relative_to(resolved_base).as_posix()
    except ValueError:
        return output_path.as_posix()


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
    crops = list(report.get("crops", []))
    lines = [
        "# F3 Scanner Backend Visual Report",
        "",
        "## Run Configuration",
        "",
        f"- comparison: `{config.get('comparison', '')}`",
        f"- scanner_backends: `{config.get('scanner_backends', [])}`",
        f"- crop_shape: `{crop_selection.get('crop_shape', '')}`",
        f"- interior_margin: `{config.get('interior_margin', '')}`",
        "",
        "## Crop Metrics",
        "",
        "| Crop | Backend | Center | normalized_correlation.interior.ft_vs_fl | "
        "top_percentile_overlap.interior.ft_vs_fl.99.jaccard |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for crop in crops:
        crop_map = _as_mapping(crop)
        crop_id = f"crop_{int(crop_map.get('index', 0)):03d}"
        for backend_name, backend in _as_mapping(crop_map.get("backends", {})).items():
            backend_map = _as_mapping(backend)
            lines.append(
                "| "
                f"{crop_id} | "
                f"`{backend_name}` | "
                f"`{crop_map.get('center', '')}` | "
                f"{_format_metric(_nested(backend_map, 'normalized_correlation', 'interior', 'ft_vs_fl'))} | "
                f"{_format_metric(_nested(backend_map, 'top_percentile_overlap', 'interior', 'ft_vs_fl', '99', 'jaccard'))} |"
            )

    lines.extend(["", "## Figures", ""])
    any_figures = False
    for crop in crops:
        crop_map = _as_mapping(crop)
        crop_id = f"crop_{int(crop_map.get('index', 0)):03d}"
        figures = _as_mapping(crop_map.get("figures", {}))
        for backend_name, figure_report in figures.items():
            links = _important_figure_links(_as_mapping(figure_report))
            if not links:
                continue
            any_figures = True
            lines.extend([f"### {crop_id} {backend_name}", ""])
            for label, path in links:
                lines.append(f"- [{label}]({path})")
            lines.append("")
    if not any_figures:
        lines.append("No PNG figures were written for this run.")
    lines.append("")
    return "\n".join(lines)


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


def _important_figure_links(figure_report: Mapping[str, Any]) -> list[tuple[str, str]]:
    files = _as_mapping(figure_report.get("files", {}))
    links: list[tuple[str, str]] = []
    scanner_slice = _nested(files, "scanner_fl_vs_ftpy", "i3")
    if isinstance(scanner_slice, str):
        links.append(("fl vs ft center slice", scanner_slice))
    ft_mip = _nested(files, "ft", "mip")
    if isinstance(ft_mip, str):
        links.append(("ft MIP", ft_mip))
    backend_diff = _nested(files, "current_vs_reference_like_ft", "i3")
    if isinstance(backend_diff, str):
        links.append(("current vs reference-like ft", backend_diff))
    return links


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
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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
        report = run_example(
            data_root_arg=args.data_root,
            output_json=args.output_json,
            output_dir=args.output_dir,
            save_volumes=args.save_volumes,
            save_figures=args.save_figures,
            figure_percentile=args.figure_percentile,
            write_markdown_index=args.write_markdown_index,
            pretty=args.pretty,
            crop_shape=args.crop_shape,
            max_crops=args.max_crops,
            interior_margin=args.interior_margin,
            scanner_backends=args.scanner_backends,
            percentile=args.percentile,
            min_separation=args.min_separation,
            sigma1=args.sigma1,
            sigma2=args.sigma2,
            phi_min=args.phi_min,
            phi_max=args.phi_max,
            theta_min=args.theta_min,
            theta_max=args.theta_max,
        )
    except (FileNotFoundError, NotADirectoryError, NotImplementedError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.output_json is None:
        print(report_to_json(report, pretty=args.pretty), end="")
    else:
        print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
