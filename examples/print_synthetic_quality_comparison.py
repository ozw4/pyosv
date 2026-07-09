"""Print oracle-vs-scanner synthetic quality summary columns."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from pathlib import Path


OUTPUT_FIELDS = (
    "case_id",
    "variant",
    "oracle_fvt_positive_f1",
    "scanner_fvt_positive_f1",
    "delta_fvt",
    "oracle_skin_f1",
    "scanner_skin_f1",
    "delta_skin",
    "scanner_ft_f1",
    "scanner_downstream_fvt_to_ft_distance_p95",
    "fallback_used",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print concise oracle-vs-scanner quality comparisons from summary.csv."
    )
    parser.add_argument(
        "summary_csv",
        type=Path,
        help="Path to summary.csv written by report_3d_synthetic_quality.py.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=None,
        help="Variant to include; may be passed more than once. Defaults to all variants.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory for synthetic_quality_comparison.csv. "
            "The comparison table is always printed to stdout."
        ),
    )
    return parser


def comparison_rows(
    rows: Iterable[Mapping[str, str]],
    *,
    variants: set[str] | None = None,
) -> list[dict[str, str]]:
    rows_by_key = {
        (row.get("pipeline", ""), row.get("case_id", ""), row.get("variant", "")): row
        for row in rows
    }
    keys = sorted(
        (case_id, variant)
        for pipeline, case_id, variant in rows_by_key
        if pipeline in {"oracle", "scanner"} and case_id and variant
    )
    output = []
    for case_id, variant in dict.fromkeys(keys):
        if variants is not None and variant not in variants:
            continue
        oracle = rows_by_key.get(("oracle", case_id, variant), {})
        scanner = rows_by_key.get(("scanner", case_id, variant), {})
        oracle_fvt = _optional_float(oracle.get("fvt_positive_buffered_f1_r2"))
        scanner_fvt = _optional_float(scanner.get("fvt_positive_buffered_f1_r2"))
        oracle_skin = _optional_float(oracle.get("skin_buffered_f1_r2"))
        scanner_skin = _optional_float(scanner.get("skin_buffered_f1_r2"))
        output.append(
            {
                "case_id": case_id,
                "variant": variant,
                "oracle_fvt_positive_f1": _format_optional_float(oracle_fvt),
                "scanner_fvt_positive_f1": _format_optional_float(scanner_fvt),
                "delta_fvt": _format_optional_float(_delta(scanner_fvt, oracle_fvt)),
                "oracle_skin_f1": _format_optional_float(oracle_skin),
                "scanner_skin_f1": _format_optional_float(scanner_skin),
                "delta_skin": _format_optional_float(_delta(scanner_skin, oracle_skin)),
                "scanner_ft_f1": scanner.get("scanner_ft_buffered_f1_r2", ""),
                "scanner_downstream_fvt_to_ft_distance_p95": scanner.get(
                    "scanner_downstream_fvt_to_ft_distance_p95",
                    "",
                ),
                "fallback_used": scanner.get("skin_fallback_used", ""),
            }
        )
    return output


def _optional_float(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    return float(text)


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def run_example(
    *,
    summary_csv: str | PathLike[str],
    variants: set[str] | None = None,
) -> list[dict[str, str]]:
    with Path(summary_csv).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    return comparison_rows(rows, variants=variants)


def _write_rows(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_rows = run_example(
        summary_csv=args.summary_csv,
        variants=None if args.variant is None else set(args.variant),
    )
    writer = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(output_rows)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_rows(args.output_dir / "synthetic_quality_comparison.csv", output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
