"""Report controlled 3D synthetic truth quality metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
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
VOLUME_NAMES = ("ft", "pt", "tt", "fv", "vp", "vt", "fvt")
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
        help="Write intermediate DAT volumes under OUTPUT_DIR/volumes.",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Accepted for the public CLI contract; figure output is added later.",
    )
    parser.add_argument(
        "--write-markdown-index",
        action="store_true",
        help="Accepted for the public CLI contract; markdown output is added later.",
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

    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(0.5)
    candidate_mask = top_truth_count_mask(fvt, truth_surface_mask)
    report = {
        "case_id": case.case_id,
        "shape": [int(size) for size in case.shape],
        "pyosv": {
            "fv": _array_summary(fv),
            "fvt": _array_summary(fvt),
        },
        "metrics": {
            "buffered_surface_overlap": buffered_surface_overlap(
                candidate_mask,
                case.truth_fault_mask,
                radius=2.0,
            ),
            "surface_distance": surface_distance_metrics(
                candidate_mask,
                case.truth_fault_mask,
            ),
            "orientation_error": masked_orientation_error(
                vp,
                vt,
                case.truth_strike,
                case.truth_dip,
                candidate_mask,
            ),
        },
    }
    volumes = {
        "ft": case.ft_oracle,
        "pt": case.pt_oracle,
        "tt": case.tt_oracle,
        "fv": fv,
        "vp": vp,
        "vt": vt,
        "fvt": fvt,
    }
    return report, volumes


def build_report(
    *,
    case_set: str,
    shape: tuple[int, int, int],
    voting_config: SyntheticVotingConfig = SyntheticVotingConfig(),
) -> dict[str, Any]:
    report, _ = _build_report_and_volumes(
        case_set=case_set,
        shape=shape,
        voting_config=voting_config,
    )
    return report


def _build_report_and_volumes(
    *,
    case_set: str,
    shape: tuple[int, int, int],
    voting_config: SyntheticVotingConfig = SyntheticVotingConfig(),
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
        )
        cases.append(case_report)
        volume_outputs[case_definition.case_id] = case_volumes

    report = {
        "format_version": FORMAT_VERSION,
        "config": {
            "case_set": case_set,
            "shape": [int(size) for size in valid_shape],
            "voting": voting_config.as_report_dict(),
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
                "fvt_max",
                "fvt_mean",
                "fvt_nonzero_fraction",
            ),
        )
        writer.writeheader()
        for case in report["cases"]:
            n3, n2, n1 = case["shape"]
            pyosv = case["pyosv"]
            fv = pyosv["fv"]
            fvt = pyosv["fvt"]
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "shape_n3": n3,
                    "shape_n2": n2,
                    "shape_n1": n1,
                    "fv_max": fv["max"],
                    "fv_mean": fv["mean"],
                    "fv_nonzero_fraction": fv["nonzero_fraction"],
                    "fvt_max": fvt["max"],
                    "fvt_mean": fvt["mean"],
                    "fvt_nonzero_fraction": fvt["nonzero_fraction"],
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
    volume_root = Path(output_dir) / "volumes"
    for case_id, volumes in volume_outputs.items():
        case_dir = volume_root / case_id
        for name in VOLUME_NAMES:
            written.append(write_dat(case_dir / f"{name}.dat", volumes[name]))
    return written


def run_example(
    *,
    output_dir: str | PathLike[str],
    case_set: str = "minimal",
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    voting_config: SyntheticVotingConfig = SyntheticVotingConfig(),
    pretty: bool = False,
    save_volumes: bool = False,
    save_figures: bool = False,
    write_markdown_index: bool = False,
) -> dict[str, Any]:
    del save_figures, write_markdown_index

    report, volume_outputs = _build_report_and_volumes(
        case_set=case_set,
        shape=shape,
        voting_config=voting_config,
    )
    write_metrics_json(report, output_dir, pretty=pretty)
    write_summary_csv(report, output_dir)
    if save_volumes:
        write_case_volumes(volume_outputs, output_dir)
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
