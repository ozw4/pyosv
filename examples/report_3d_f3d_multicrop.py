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
    d: int = 4,
    fm: float = 0.3,
    scanner_thin_mode: str = "normal",
    voter_thin_mode: str = "normal",
    reference_thin_sigma: float = 1.0,
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
    config = build_config(
        crop_shape=crop_shape,
        interior_margin=interior_margin,
        count=count,
        centers=selected_centers,
        explicit_centers=centers is not None,
        percentile=percentile,
        min_separation=min_separation,
        save_volumes=save_volumes,
        volume_dir=resolved_volume_dir,
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
        scanner_thin_mode=scanner_thin_mode,
        voter_thin_mode=voter_thin_mode,
        reference_thin_sigma=reference_thin_sigma,
    )
    if save_figures and "fl.dat" not in arrays:
        arrays["fl.dat"] = crop_validation.read_f3d_file("fl.dat", data_root)

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
            d=d,
            fm=fm,
            scanner_thin_mode=scanner_thin_mode,
            voter_thin_mode=voter_thin_mode,
            reference_thin_sigma=reference_thin_sigma,
        )

        if resolved_volume_dir is not None:
            crop_validation.write_crop_volumes(
                Path(resolved_volume_dir) / f"crop_{crop_index:03d}",
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
            crop_report["figures"] = crop_validation.write_crop_figures(
                output_base_dir / f"crop_{crop_index:03d}" / "figures",
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

    report = _json_compatible(
        {
            "format_version": 1,
            "data_root": str(data_root),
            "config": config,
            "crops": crops,
            "aggregate": aggregate_crop_metrics(crops),
        }
    )

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
    d: int,
    fm: float,
    scanner_thin_mode: str = "normal",
    voter_thin_mode: str = "normal",
    reference_thin_sigma: float = 1.0,
) -> dict[str, Any]:
    config: dict[str, Any] = {
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
            "thin_mode": voter_thin_mode,
            "reference_thin_sigma": float(reference_thin_sigma),
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
    crops = list(report.get("crops", []))
    data_root = Path(str(report.get("data_root", "")))

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
        f"- selected_count: `{crop_selection.get('selected_count', len(crops))}`",
        f"- scanner_thin_mode: `{scanner.get('thin_mode', '')}`",
        f"- voter_thin_mode: `{voter.get('thin_mode', '')}`",
        f"- reference_thin_sigma: `{scanner.get('reference_thin_sigma', '')}`",
        f"- scanner: `{scanner}`",
        f"- voter: `{voter}`",
    ]
    if visualization:
        lines.append(f"- visualization: `{visualization}`")

    lines.extend(
        [
            "",
            "## Crop Metrics",
            "",
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

    lines.extend(["", "## Figures", ""])
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
    if not any_figures:
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
            d=args.d,
            fm=args.fm,
            scanner_thin_mode=args.scanner_thin_mode,
            voter_thin_mode=args.voter_thin_mode,
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
