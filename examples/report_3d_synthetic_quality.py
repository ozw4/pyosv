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

from pyosv.synthetic3d import Synthetic3DCase, make_single_vertical_plane_case, validate_shape3
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


@dataclass(frozen=True, slots=True)
class SyntheticQualityCaseDefinition:
    """A controlled synthetic report case definition."""

    case_id: str
    factory: Callable[[tuple[int, int, int]], Synthetic3DCase]


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
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case = case_definition.factory(shape)
    if case.case_id != case_definition.case_id:
        raise ValueError(
            f"case factory returned {case.case_id!r}, expected {case_definition.case_id!r}"
        )

    voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
    voter.set_attribute_smoothing(0)
    fv, vp, vt = voter.apply_voting(
        d=3,
        fm=0.5,
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
    )
    fvt = voter.thin(fv, vp, vt)

    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(0.5)
    candidate_mask = top_truth_count_mask(fvt, truth_surface_mask)
    report = {
        "case_id": case.case_id,
        "shape": [int(size) for size in case.shape],
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
) -> dict[str, Any]:
    report, _ = _build_report_and_volumes(case_set=case_set, shape=shape)
    return report


def _build_report_and_volumes(
    *,
    case_set: str,
    shape: tuple[int, int, int],
) -> tuple[dict[str, Any], dict[str, Mapping[str, np.ndarray]]]:
    valid_shape = validate_shape3(shape)
    try:
        case_definitions = CASE_SETS[case_set]
    except KeyError as error:
        raise ValueError(f"unknown case_set: {case_set}") from error

    cases = []
    volume_outputs = {}
    for case_definition in case_definitions:
        case_report, case_volumes = run_case(case_definition, shape=valid_shape)
        cases.append(case_report)
        volume_outputs[case_definition.case_id] = case_volumes

    report = {
        "format_version": FORMAT_VERSION,
        "config": {
            "case_set": case_set,
            "shape": [int(size) for size in valid_shape],
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
        writer = csv.DictWriter(file, fieldnames=("case_id", "shape_n3", "shape_n2", "shape_n1"))
        writer.writeheader()
        for case in report["cases"]:
            n3, n2, n1 = case["shape"]
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "shape_n3": n3,
                    "shape_n2": n2,
                    "shape_n1": n1,
                }
            )
    return output_path


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
    pretty: bool = False,
    save_volumes: bool = False,
    save_figures: bool = False,
    write_markdown_index: bool = False,
) -> dict[str, Any]:
    del save_figures, write_markdown_index

    report, volume_outputs = _build_report_and_volumes(case_set=case_set, shape=shape)
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
