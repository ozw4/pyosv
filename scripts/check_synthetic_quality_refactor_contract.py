#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPOSITORY_ROOT / "tests" / "fixtures" / "synthetic_quality_refactor"
METRICS_FIXTURE = FIXTURE_DIR / "17_quality_ref2_metrics.json"
SUMMARY_FIXTURE = FIXTURE_DIR / "17_quality_ref2_summary.csv"
ARTIFACT_FIXTURE = FIXTURE_DIR / "17_quality_ref2_artifact_sha256.json"
UPDATE_ENVIRONMENT_VARIABLE = "PYOSV_UPDATE_REFACTOR_CONTRACT"


def normalized_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_manifest(output_dir: Path) -> dict[str, dict[str, int | str]]:
    paths = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and (path.suffix == ".dat" or path.name == "skins.json")
    )
    return {
        path.relative_to(output_dir).as_posix(): {
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in paths
    }


def _json_differences(expected: Any, actual: Any, path: str = "$") -> list[str]:
    if type(expected) is not type(actual):
        return [
            f"JSON {path}: expected type {type(expected).__name__}, got {type(actual).__name__}"
        ]
    if isinstance(expected, dict):
        differences = []
        for key in sorted(expected.keys() | actual.keys()):
            child_path = f"{path}.{key}"
            if key not in actual:
                differences.append(f"JSON {child_path}: missing from actual output")
            elif key not in expected:
                differences.append(f"JSON {child_path}: unexpected in actual output")
            else:
                differences.extend(_json_differences(expected[key], actual[key], child_path))
        return differences
    if isinstance(expected, list):
        differences = []
        for index in range(max(len(expected), len(actual))):
            child_path = f"{path}[{index}]"
            if index >= len(actual):
                differences.append(f"JSON {child_path}: missing from actual output")
            elif index >= len(expected):
                differences.append(f"JSON {child_path}: unexpected in actual output")
            else:
                differences.extend(_json_differences(expected[index], actual[index], child_path))
        return differences
    if expected != actual:
        return [f"JSON {path}: expected {expected!r}, got {actual!r}"]
    return []


def _csv_differences(expected_bytes: bytes, actual_bytes: bytes) -> list[str]:
    if expected_bytes == actual_bytes:
        return []
    differences = [
        f"CSV bytes differ: expected {len(expected_bytes)} bytes, got {len(actual_bytes)} bytes"
    ]
    try:
        expected_rows = list(csv.reader(expected_bytes.decode("utf-8").splitlines()))
        actual_rows = list(csv.reader(actual_bytes.decode("utf-8").splitlines()))
    except UnicodeDecodeError as error:
        return [*differences, f"CSV cannot be decoded as UTF-8: {error}"]
    for row_index in range(max(len(expected_rows), len(actual_rows))):
        if row_index >= len(actual_rows):
            differences.append(f"CSV row {row_index + 1}: missing from actual output")
            continue
        if row_index >= len(expected_rows):
            differences.append(f"CSV row {row_index + 1}: unexpected in actual output")
            continue
        expected_row = expected_rows[row_index]
        actual_row = actual_rows[row_index]
        for column_index in range(max(len(expected_row), len(actual_row))):
            location = f"CSV row {row_index + 1}, column {column_index + 1}"
            if column_index >= len(actual_row):
                differences.append(f"{location}: missing from actual output")
            elif column_index >= len(expected_row):
                differences.append(f"{location}: unexpected value {actual_row[column_index]!r}")
            elif expected_row[column_index] != actual_row[column_index]:
                differences.append(
                    f"{location}: expected {expected_row[column_index]!r}, "
                    f"got {actual_row[column_index]!r}"
                )
    return differences


def _artifact_differences(
    expected: dict[str, dict[str, int | str]],
    actual: dict[str, dict[str, int | str]],
) -> list[str]:
    differences = []
    for path in sorted(expected.keys() | actual.keys()):
        if path not in actual:
            differences.append(f"artifact {path}: missing from actual output")
        elif path not in expected:
            differences.append(f"artifact {path}: unexpected in actual output")
        else:
            for field in ("size", "sha256"):
                if expected[path].get(field) != actual[path].get(field):
                    differences.append(
                        f"artifact {path} {field}: expected {expected[path].get(field)!r}, "
                        f"got {actual[path].get(field)!r}"
                    )
    return differences


def compare_output(output_dir: Path, fixture_dir: Path = FIXTURE_DIR) -> list[str]:
    differences = []
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.csv"
    if metrics_path.is_file():
        expected_metrics = normalized_json(fixture_dir / METRICS_FIXTURE.name)
        actual_metrics = normalized_json(metrics_path)
        differences.extend(_json_differences(expected_metrics, actual_metrics))
    else:
        differences.append("required output missing: metrics.json")
    if summary_path.is_file():
        differences.extend(
            _csv_differences(
                (fixture_dir / SUMMARY_FIXTURE.name).read_bytes(),
                summary_path.read_bytes(),
            )
        )
    else:
        differences.append("required output missing: summary.csv")
    expected_artifacts = normalized_json(fixture_dir / ARTIFACT_FIXTURE.name)
    differences.extend(_artifact_differences(expected_artifacts, artifact_manifest(output_dir)))
    return differences


def update_fixtures(output_dir: Path, fixture_dir: Path = FIXTURE_DIR) -> None:
    fixture_dir.mkdir(parents=True, exist_ok=True)
    metrics = normalized_json(output_dir / "metrics.json")
    (fixture_dir / METRICS_FIXTURE.name).write_text(
        json.dumps(metrics, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (fixture_dir / SUMMARY_FIXTURE.name).write_bytes((output_dir / "summary.csv").read_bytes())
    (fixture_dir / ARTIFACT_FIXTURE.name).write_text(
        json.dumps(artifact_manifest(output_dir), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def run_report(output_dir: Path) -> None:
    command = [
        sys.executable,
        "examples/report_3d_synthetic_quality.py",
        "--case-set",
        "extended",
        "--shape",
        "17,17,17",
        "--workflow-mode",
        "quality",
        "--variants",
        "current_default,boundary_aware_voter_v1",
        "--input-mode",
        "both",
        "--scanner-backend",
        "quality",
        "--scanner-refinement-factor",
        "2",
        "--scanner-downstream-diagnostics",
        "--output-dir",
        str(output_dir),
        "--pretty",
        "--save-volumes",
    ]
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "src")
    subprocess.run(command, cwd=REPOSITORY_ROOT, env=environment, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check the synthetic quality refactor contract.")
    parser.add_argument(
        "--existing-output",
        type=Path,
        help="Compare an existing report directory without running the report.",
    )
    parser.add_argument(
        "--update-fixtures",
        action="store_true",
        help=f"Update fixtures (also requires {UPDATE_ENVIRONMENT_VARIABLE}=1).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.update_fixtures and os.environ.get(UPDATE_ENVIRONMENT_VARIABLE) != "1":
        print(
            f"refusing to update fixtures: set {UPDATE_ENVIRONMENT_VARIABLE}=1 as well",
            file=sys.stderr,
        )
        return 2

    if args.existing_output is not None:
        output_dir = args.existing_output.resolve()
        if not output_dir.is_dir():
            print(f"existing output is not a directory: {output_dir}", file=sys.stderr)
            return 2
        if args.update_fixtures:
            update_fixtures(output_dir)
            print(f"updated refactor contract fixtures from {output_dir}")
            return 0
        differences = compare_output(output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="pyosv-refactor-contract-") as temporary:
            output_dir = Path(temporary)
            run_report(output_dir)
            if args.update_fixtures:
                update_fixtures(output_dir)
                print("updated refactor contract fixtures from a fresh report")
                return 0
            differences = compare_output(output_dir)

    if differences:
        print("synthetic quality refactor contract failed:", file=sys.stderr)
        for difference in differences:
            print(f"- {difference}", file=sys.stderr)
        return 1
    print("synthetic quality refactor contract passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
