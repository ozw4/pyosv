"""Report summary statistics for the public F3 3D reference files."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from pyosv.f3d_reference import F3D_ENV_VAR, F3D_FILENAMES, read_f3d_file, resolve_f3d_data_root
from pyosv.metrics import normalized_correlation, top_percentile_overlap

SUMMARY_PERCENTILES = (50.0, 95.0, 99.0, 99.5, 99.9)
OVERLAP_PERCENTILES = (95.0, 99.0, 99.5, 99.9)
NONZERO_EPSILON = 1.0e-6


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize the public F3 3D reference files as a baseline JSON report.",
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
        "--pretty",
        action="store_true",
        help="Write indented JSON.",
    )
    return parser


def summarize_array(file_name: str, array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    finite = np.isfinite(values)
    finite_count = int(np.count_nonzero(finite))
    finite_values = values[finite].astype(np.float64, copy=False)

    summary: dict[str, Any] = {
        "file_name": file_name,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "finite_count": finite_count,
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "nonzero_fraction": (
            float(np.count_nonzero(np.abs(values) > NONZERO_EPSILON) / values.size)
            if values.size
            else 0.0
        ),
        "percentiles": {percentile_key(p): None for p in SUMMARY_PERCENTILES},
    }

    if finite_count:
        summary.update(
            {
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
                "mean": float(np.mean(finite_values)),
                "std": float(np.std(finite_values)),
                "percentiles": {
                    percentile_key(p): float(np.percentile(finite_values, p))
                    for p in SUMMARY_PERCENTILES
                },
            }
        )

    return summary


def build_report(
    arrays: Mapping[str, np.ndarray],
    *,
    data_root: str | PathLike[str] | None = None,
) -> dict[str, Any]:
    missing = [file_name for file_name in F3D_FILENAMES if file_name not in arrays]
    if missing:
        raise ValueError(f"missing F3 reference arrays: {', '.join(missing)}")

    fv = arrays["fv.dat"]
    fvt = arrays["fvt.dat"]
    return {
        "format_version": 1,
        "data_root": str(data_root) if data_root is not None else None,
        "files": [summarize_array(file_name, arrays[file_name]) for file_name in F3D_FILENAMES],
        "comparisons": {
            "fv_fvt": {
                "a_file": "fv.dat",
                "b_file": "fvt.dat",
                "normalized_correlation": float(normalized_correlation(fv, fvt)),
                "top_percentile_overlap": {
                    percentile_key(p): top_percentile_overlap(fv, fvt, percentile=p)
                    for p in OVERLAP_PERCENTILES
                },
            }
        },
    }


def read_reference_arrays(data_root: str | PathLike[str] | None = None) -> dict[str, np.ndarray]:
    return {file_name: read_f3d_file(file_name, data_root) for file_name in F3D_FILENAMES}


def report_to_json(report: Mapping[str, Any], *, pretty: bool = False) -> str:
    indent = 2 if pretty else None
    return json.dumps(report, indent=indent, sort_keys=True) + "\n"


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


def run_example(
    *,
    data_root_arg: str | PathLike[str] | None,
    output_json: str | PathLike[str] | None = None,
    pretty: bool = False,
) -> dict[str, Any]:
    data_root = resolve_f3d_data_root(data_root_arg)
    if output_json is not None:
        ensure_output_not_in_data_root(output_json, data_root)

    arrays = read_reference_arrays(data_root)
    report = build_report(arrays, data_root=data_root)

    if output_json is not None:
        write_report_json(report, output_json, pretty=pretty)

    return report


def ensure_output_not_in_data_root(
    output_json: str | PathLike[str],
    data_root: str | PathLike[str],
) -> None:
    output_path = Path(output_json).resolve(strict=False)
    data_root_path = Path(data_root).resolve(strict=False)
    try:
        output_path.relative_to(data_root_path)
    except ValueError:
        return
    raise ValueError(f"output JSON path must not be inside the F3 data root: {output_path}")


def percentile_key(percentile: float) -> str:
    return f"{percentile:g}"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        report = run_example(
            data_root_arg=args.data_root,
            output_json=args.output_json,
            pretty=args.pretty,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        import sys

        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.output_json is None:
        print(report_to_json(report, pretty=args.pretty), end="")
    else:
        print(args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
