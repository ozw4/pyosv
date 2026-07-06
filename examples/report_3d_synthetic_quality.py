"""Report controlled 3D synthetic truth quality metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from pyosv.synthetic3d import (
    Synthetic3DCase,
    make_single_vertical_plane_case,
    validate_shape3,
)
from pyosv.synthetic_metrics import (
    buffered_surface_overlap,
    masked_orientation_error,
    surface_distance_metrics,
    top_truth_count_mask,
)
from pyosv.voting3d import OptimalSurfaceVoter

DEFAULT_SHAPE = (33, 33, 33)
FORMAT_VERSION = 1
VOLUME_NAMES = (
    "truth_fault_mask",
    "truth_distance",
    "truth_strike",
    "truth_dip",
    "ft_oracle",
    "pt_oracle",
    "tt_oracle",
    "fv_py",
    "vp_py",
    "vt_py",
    "fvt_py",
)
FIGURE_VOLUME_NAMES = ("ft_oracle", "fv_py", "fvt_py")
NONZERO_EPSILON = 1.0e-6


@dataclass(frozen=True, slots=True)
class SyntheticQualityCaseDefinition:
    """A controlled synthetic report case definition."""

    case_id: str
    factory: Callable[[tuple[int, int, int]], Synthetic3DCase]


@dataclass(frozen=True, slots=True)
class SyntheticVotingConfig:
    """Configuration for synthetic oracle voting."""

    ru: int = 1
    rv: int = 2
    rw: int = 2
    seed_distance: int = 3
    seed_threshold: float = 0.5
    attribute_smoothing: int = 0
    voter_thin_mode: str = "reference"
    reference_thin_sigma: float = 1.0

    def as_report_dict(self) -> dict[str, int | float | str]:
        return {
            "ru": int(self.ru),
            "rv": int(self.rv),
            "rw": int(self.rw),
            "seed_distance": int(self.seed_distance),
            "seed_threshold": float(self.seed_threshold),
            "attribute_smoothing": int(self.attribute_smoothing),
            "voter_thin_mode": self.voter_thin_mode,
            "reference_thin_sigma": float(self.reference_thin_sigma),
        }


@dataclass(frozen=True, slots=True)
class SyntheticTruthMetricConfig:
    """Configuration for controlled truth metrics."""

    truth_surface_half_width: float = 0.5
    buffer_radius: float = 2.0

    def as_report_dict(self) -> dict[str, float]:
        return {
            "truth_surface_half_width": float(self.truth_surface_half_width),
            "buffer_radius": float(self.buffer_radius),
        }


MINIMAL_CASES = (
    SyntheticQualityCaseDefinition(
        case_id="single_vertical_plane",
        factory=make_single_vertical_plane_case,
    ),
)
CASE_SETS = {"minimal": MINIMAL_CASES}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run controlled 3D synthetic oracle voting and write quality reports.",
    )
    parser.add_argument(
        "--case-set",
        choices=tuple(CASE_SETS),
        default="minimal",
        help="Synthetic case set to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where metrics.json and summary.csv are written.",
    )
    parser.add_argument(
        "--shape",
        type=parse_shape3,
        default=DEFAULT_SHAPE,
        help="Synthetic volume shape in n3,n2,n1 order.",
    )
    parser.add_argument("--pretty", action="store_true", help="Write indented JSON.")
    parser.add_argument(
        "--save-volumes",
        action="store_true",
        help="Write truth, oracle, and Python DAT volumes under each case directory.",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Write static PNG center-slice figures under each case directory.",
    )
    parser.add_argument(
        "--write-markdown-index",
        action="store_true",
        help="Write visual_report.md under OUTPUT_DIR.",
    )
    parser.add_argument("--ru", type=int, default=1, help="Voting shift radius in u.")
    parser.add_argument("--rv", type=int, default=2, help="Voting shift radius in v.")
    parser.add_argument("--rw", type=int, default=2, help="Voting shift radius in w.")
    parser.add_argument(
        "--seed-distance",
        type=int,
        default=3,
        help="Minimum seed spacing used by the voter.",
    )
    parser.add_argument(
        "--seed-threshold",
        type=float,
        default=0.5,
        help="Oracle ft threshold used for seed selection.",
    )
    parser.add_argument(
        "--attribute-smoothing",
        type=int,
        default=0,
        help="Number of voter attribute smoothing passes.",
    )
    parser.add_argument(
        "--voter-thin-mode",
        choices=("reference", "normal"),
        default="reference",
        help="Thinning mode passed to OptimalSurfaceVoter.thin().",
    )
    parser.add_argument(
        "--reference-thin-sigma",
        type=float,
        default=1.0,
        help="Smoothing sigma used by reference-like thinning.",
    )
    parser.add_argument(
        "--truth-surface-half-width",
        type=float,
        default=0.5,
        help="Half-width around the truth surface used for thin-surface metrics.",
    )
    parser.add_argument(
        "--buffer-radius",
        type=float,
        default=2.0,
        help="Distance radius used for buffered overlap metrics.",
    )
    return parser


def parse_shape3(text: str) -> tuple[int, int, int]:
    """Parse a 3D shape string as ``(n3, n2, n1)``."""
    try:
        parts = tuple(int(part) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be three comma-separated integers") from error
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be three comma-separated integers")
    try:
        return validate_shape3(parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def run_case(
    case_definition: SyntheticQualityCaseDefinition,
    *,
    shape: tuple[int, int, int],
    voting_config: SyntheticVotingConfig = SyntheticVotingConfig(),
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig(),
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case = case_definition.factory(shape)
    if case.case_id != case_definition.case_id:
        raise ValueError(
            f"case factory returned {case.case_id!r}, expected {case_definition.case_id!r}"
        )

    voter = OptimalSurfaceVoter(
        ru=voting_config.ru,
        rv=voting_config.rv,
        rw=voting_config.rw,
    )
    voter.set_attribute_smoothing(voting_config.attribute_smoothing)
    fv, vp, vt = voter.apply_voting(
        d=voting_config.seed_distance,
        fm=voting_config.seed_threshold,
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
    )
    fvt = voter.thin(
        fv,
        vp,
        vt,
        mode=voting_config.voter_thin_mode,
        reference_sigma=voting_config.reference_thin_sigma,
    )

    truth_surface_half_width = _validate_nonnegative_finite_scalar(
        truth_metric_config.truth_surface_half_width,
        "truth_surface_half_width",
    )
    buffer_radius = _validate_nonnegative_finite_scalar(
        truth_metric_config.buffer_radius,
        "buffer_radius",
    )
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(truth_surface_half_width)
    truth_fault_mask = np.asarray(case.truth_fault_mask, dtype=bool)
    fv_top_truth_count = top_truth_count_mask(fv, truth_surface_mask)
    fvt_top_truth_count = top_truth_count_mask(fvt, truth_surface_mask)
    report = {
        "case_id": case.case_id,
        "shape": [int(size) for size in case.shape],
        "pyosv": {
            "fv": _array_summary(fv),
            "fvt": _array_summary(fvt),
        },
        "truth": {
            "fault_voxel_count": int(np.count_nonzero(truth_fault_mask)),
            "surface_voxel_count": int(np.count_nonzero(truth_surface_mask)),
        },
        "quality": {
            "fv_top_truth_count": {
                "buffered_overlap_radius2": buffered_surface_overlap(
                    fv_top_truth_count,
                    truth_fault_mask,
                    radius=buffer_radius,
                ),
                "surface_distance": surface_distance_metrics(
                    fv_top_truth_count,
                    truth_surface_mask,
                ),
            },
            "fvt_top_truth_count": {
                "buffered_overlap_radius2": buffered_surface_overlap(
                    fvt_top_truth_count,
                    truth_fault_mask,
                    radius=buffer_radius,
                ),
                "surface_distance": surface_distance_metrics(
                    fvt_top_truth_count,
                    truth_surface_mask,
                ),
                "orientation_error": masked_orientation_error(
                    vp,
                    vt,
                    case.truth_strike,
                    case.truth_dip,
                    fvt_top_truth_count,
                ),
            },
        },
    }
    volumes = {
        "truth_fault_mask": case.truth_fault_mask.astype(np.float32),
        "truth_distance": case.truth_distance,
        "truth_strike": case.truth_strike,
        "truth_dip": case.truth_dip,
        "ft_oracle": case.ft_oracle,
        "pt_oracle": case.pt_oracle,
        "tt_oracle": case.tt_oracle,
        "fv_py": fv,
        "vp_py": vp,
        "vt_py": vt,
        "fvt_py": fvt,
    }
    return report, volumes


def build_report(
    *,
    case_set: str,
    shape: tuple[int, int, int],
    voting_config: SyntheticVotingConfig = SyntheticVotingConfig(),
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig(),
) -> dict[str, Any]:
    report, _ = _build_report_and_volumes(
        case_set=case_set,
        shape=shape,
        voting_config=voting_config,
        truth_metric_config=truth_metric_config,
    )
    return report


def _build_report_and_volumes(
    *,
    case_set: str,
    shape: tuple[int, int, int],
    voting_config: SyntheticVotingConfig = SyntheticVotingConfig(),
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig(),
) -> tuple[dict[str, Any], dict[str, Mapping[str, np.ndarray]]]:
    valid_shape = validate_shape3(shape)
    try:
        case_definitions = CASE_SETS[case_set]
    except KeyError as error:
        raise ValueError(f"unknown case_set: {case_set}") from error

    cases = []
    volume_outputs = {}
    for case_definition in case_definitions:
        case_report, case_volumes = run_case(
            case_definition,
            shape=valid_shape,
            voting_config=voting_config,
            truth_metric_config=truth_metric_config,
        )
        cases.append(case_report)
        volume_outputs[case_definition.case_id] = case_volumes

    report = {
        "format_version": FORMAT_VERSION,
        "config": {
            "case_set": case_set,
            "shape": [int(size) for size in valid_shape],
            "voting": voting_config.as_report_dict(),
            "truth_metrics": truth_metric_config.as_report_dict(),
        },
        "cases": cases,
    }
    return report, volume_outputs


def report_to_json(report: Mapping[str, Any], *, pretty: bool = False) -> str:
    indent = 2 if pretty else None
    return json.dumps(report, indent=indent, sort_keys=True) + "\n"


def write_metrics_json(
    report: Mapping[str, Any],
    output_dir: str | PathLike[str],
    *,
    pretty: bool = False,
) -> Path:
    output_path = Path(output_dir) / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_to_json(report, pretty=pretty), encoding="utf-8")
    return output_path


def write_summary_csv(report: Mapping[str, Any], output_dir: str | PathLike[str]) -> Path:
    output_path = Path(output_dir) / "summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "case_id",
                "shape_n3",
                "shape_n2",
                "shape_n1",
                "fv_max",
                "fv_mean",
                "fv_nonzero_fraction",
                "fv_buffered_f1_r2",
                "fv_distance_p95",
                "fvt_max",
                "fvt_mean",
                "fvt_nonzero_fraction",
                "fvt_buffered_f1_r2",
                "fvt_distance_p95",
                "fvt_strike_median_error",
                "fvt_dip_median_error",
            ),
        )
        writer.writeheader()
        for case in report["cases"]:
            n3, n2, n1 = case["shape"]
            pyosv = case["pyosv"]
            fv = pyosv["fv"]
            fvt = pyosv["fvt"]
            quality = case["quality"]
            fv_quality = quality["fv_top_truth_count"]
            fvt_quality = quality["fvt_top_truth_count"]
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "shape_n3": n3,
                    "shape_n2": n2,
                    "shape_n1": n1,
                    "fv_max": fv["max"],
                    "fv_mean": fv["mean"],
                    "fv_nonzero_fraction": fv["nonzero_fraction"],
                    "fv_buffered_f1_r2": fv_quality["buffered_overlap_radius2"]["buffered_f1"],
                    "fv_distance_p95": fv_quality["surface_distance"]["candidate_to_truth_p95"],
                    "fvt_max": fvt["max"],
                    "fvt_mean": fvt["mean"],
                    "fvt_nonzero_fraction": fvt["nonzero_fraction"],
                    "fvt_buffered_f1_r2": fvt_quality["buffered_overlap_radius2"]["buffered_f1"],
                    "fvt_distance_p95": fvt_quality["surface_distance"]["candidate_to_truth_p95"],
                    "fvt_strike_median_error": fvt_quality["orientation_error"]["strike_median"],
                    "fvt_dip_median_error": fvt_quality["orientation_error"]["dip_median"],
                }
            )
    return output_path


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    finite = np.isfinite(values)
    finite_values = values[finite].astype(np.float64, copy=False)
    if finite_values.size:
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        mean = float(np.mean(finite_values))
    else:
        minimum = float("nan")
        maximum = float("nan")
        mean = float("nan")

    return {
        "shape": [int(size) for size in values.shape],
        "finite_count": int(np.count_nonzero(finite)),
        "finite_fraction": (float(np.count_nonzero(finite) / values.size) if values.size else 0.0),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "nonzero_fraction": (
            float(np.count_nonzero(np.abs(values) > NONZERO_EPSILON) / values.size)
            if values.size
            else 0.0
        ),
    }


def write_case_volumes(
    volume_outputs: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: str | PathLike[str],
) -> list[Path]:
    from pyosv.io import write_dat

    written = []
    output_root = Path(output_dir)
    for case_id, volumes in volume_outputs.items():
        case_dir = _case_output_dir(output_root, case_id)
        for name in VOLUME_NAMES:
            written.append(write_dat(case_dir / f"{name}.dat", volumes[name]))
    return written


def write_case_figures(
    volume_outputs: Mapping[str, Mapping[str, np.ndarray]],
    output_dir: str | PathLike[str],
    *,
    buffer_radius: float = 2.0,
) -> list[Path]:
    from pyosv import viz

    written = []
    output_root = Path(output_dir)
    for case_id, volumes in volume_outputs.items():
        case_dir = _case_output_dir(output_root, case_id)
        figures_dir = case_dir / "figures"
        indices = viz.select_center_slices(np.asarray(volumes["fvt_py"]).shape)
        for axis in ("i3", "i2", "i1"):
            index = indices[axis]
            for name in FIGURE_VOLUME_NAMES:
                figure_path = figures_dir / f"{name}_{axis}_center.png"
                written.append(
                    viz.save_slice_panel(
                        figure_path,
                        [(name, viz.slice_2d(volumes[name], axis, index))],
                        title=f"{case_id} {name} {axis}=center",
                    )
                )
            written.append(
                viz.save_ridge_overlay_slice(
                    figures_dir / f"truth_vs_fvt_overlay_{axis}_center.png",
                    reference=volumes["truth_fault_mask"],
                    candidate=volumes["fvt_py"],
                    axis=axis,
                    index=index,
                    percentile=99.0,
                    buffer_radius=buffer_radius,
                    title=f"{case_id} truth vs fvt {axis}=center",
                )
            )
    return written


def write_visual_report_markdown(
    report: Mapping[str, Any],
    output_dir: str | PathLike[str],
) -> Path:
    output_path = Path(output_dir) / "visual_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(visual_report_markdown(report), encoding="utf-8")
    return output_path


def visual_report_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Controlled Synthetic Quality Report", ""]
    for case in report["cases"]:
        case_id = str(case["case_id"])
        quality = case["quality"]["fvt_top_truth_count"]
        overlap = quality["buffered_overlap_radius2"]
        distance = quality["surface_distance"]
        orientation = quality["orientation_error"]
        overlay_path = PurePosixPath(
            case_id,
            "figures",
            "truth_vs_fvt_overlay_i3_center.png",
        )
        lines.extend(
            [
                f"## {case_id}",
                "",
                f"- buffered_f1_r2: {_format_markdown_metric(overlap['buffered_f1'])}",
                f"- distance_p95: {_format_markdown_metric(distance['candidate_to_truth_p95'])}",
                f"- strike_median_error: {_format_markdown_metric(orientation['strike_median'])}",
                f"- dip_median_error: {_format_markdown_metric(orientation['dip_median'])}",
                "",
                f"![fvt overlay]({overlay_path.as_posix()})",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _format_markdown_metric(value: object) -> str:
    if isinstance(value, int | float | np.floating | np.integer):
        return f"{float(value):.6g}"
    return str(value)


def _case_output_dir(output_dir: Path, case_id: str) -> Path:
    relative_case_path = PurePosixPath(case_id)
    if (
        relative_case_path.is_absolute()
        or not relative_case_path.parts
        or any(part in {"", ".", ".."} for part in relative_case_path.parts)
    ):
        raise ValueError(f"case_id must be a relative path inside output_dir: {case_id!r}")
    return output_dir.joinpath(*relative_case_path.parts)


def _validate_nonnegative_finite_scalar(value: float, name: str) -> float:
    if not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def run_example(
    *,
    output_dir: str | PathLike[str],
    case_set: str = "minimal",
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    voting_config: SyntheticVotingConfig = SyntheticVotingConfig(),
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig(),
    pretty: bool = False,
    save_volumes: bool = False,
    save_figures: bool = False,
    write_markdown_index: bool = False,
) -> dict[str, Any]:
    report, volume_outputs = _build_report_and_volumes(
        case_set=case_set,
        shape=shape,
        voting_config=voting_config,
        truth_metric_config=truth_metric_config,
    )
    write_metrics_json(report, output_dir, pretty=pretty)
    write_summary_csv(report, output_dir)
    if save_volumes:
        write_case_volumes(volume_outputs, output_dir)
    if save_figures:
        write_case_figures(
            volume_outputs,
            output_dir,
            buffer_radius=truth_metric_config.buffer_radius,
        )
    if write_markdown_index:
        write_visual_report_markdown(report, output_dir)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run_example(
            case_set=args.case_set,
            output_dir=args.output_dir,
            shape=args.shape,
            voting_config=SyntheticVotingConfig(
                ru=args.ru,
                rv=args.rv,
                rw=args.rw,
                seed_distance=args.seed_distance,
                seed_threshold=args.seed_threshold,
                attribute_smoothing=args.attribute_smoothing,
                voter_thin_mode=args.voter_thin_mode,
                reference_thin_sigma=args.reference_thin_sigma,
            ),
            truth_metric_config=SyntheticTruthMetricConfig(
                truth_surface_half_width=args.truth_surface_half_width,
                buffer_radius=args.buffer_radius,
            ),
            pretty=args.pretty,
            save_volumes=args.save_volumes,
            save_figures=args.save_figures,
            write_markdown_index=args.write_markdown_index,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
