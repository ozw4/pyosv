from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "report_3d_synthetic_quality.py"


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, str(SCRIPT.relative_to(REPO_ROOT)), *args],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


def test_report_3d_synthetic_quality_help_exits_successfully() -> None:
    result = _run_script("--help")

    assert result.returncode == 0
    assert "usage:" in result.stdout
    assert "--case-set" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--voter-thin-mode" in result.stdout
    assert "--truth-surface-half-width" in result.stdout
    assert "--buffer-radius" in result.stdout


def test_report_3d_synthetic_quality_minimal_case_writes_contract_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--pretty",
    )

    assert result.returncode == 0, result.stderr
    metrics_path = output_dir / "metrics.json"
    summary_path = output_dir / "summary.csv"
    assert metrics_path.is_file()
    assert summary_path.is_file()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["format_version"] == 1
    assert metrics["config"]["case_set"] == "minimal"
    assert metrics["config"]["shape"] == [17, 17, 17]
    case = metrics["cases"][0]
    assert case["case_id"] == "single_vertical_plane"
    assert case["shape"] == [17, 17, 17]
    assert case["truth"]["fault_voxel_count"] > case["truth"]["surface_voxel_count"] > 0
    quality = case["quality"]
    fvt_quality = quality["fvt_top_truth_count"]
    assert "buffered_f1" in fvt_quality["buffered_overlap_radius2"]
    assert "candidate_to_truth_p95" in fvt_quality["surface_distance"]
    assert "strike_median" in fvt_quality["orientation_error"]
    for name in ("fv", "fvt"):
        summary = case["pyosv"][name]
        assert summary["shape"] == [17, 17, 17]
        assert summary["finite_count"] == 17 * 17 * 17
        assert summary["finite_fraction"] == 1.0
        assert summary["max"] > 0.0
        assert 0.0 < summary["nonzero_fraction"] <= 1.0

    with summary_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["case_id"] == "single_vertical_plane"
    assert rows[0]["shape_n3"] == "17"
    assert rows[0]["shape_n2"] == "17"
    assert rows[0]["shape_n1"] == "17"
    assert float(rows[0]["fv_max"]) > 0.0
    assert float(rows[0]["fv_nonzero_fraction"]) > 0.0
    assert float(rows[0]["fv_buffered_f1_r2"]) > 0.0
    assert float(rows[0]["fv_distance_p95"]) >= 0.0
    assert float(rows[0]["fvt_max"]) > 0.0
    assert float(rows[0]["fvt_nonzero_fraction"]) > 0.0
    assert float(rows[0]["fvt_buffered_f1_r2"]) > 0.0
    assert float(rows[0]["fvt_distance_p95"]) >= 0.0
    assert float(rows[0]["fvt_strike_median_error"]) >= 0.0
    assert float(rows[0]["fvt_dip_median_error"]) >= 0.0


def test_report_3d_synthetic_quality_normal_thin_mode_passes(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--voter-thin-mode",
        "normal",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["voting"]["voter_thin_mode"] == "normal"
    assert metrics["cases"][0]["pyosv"]["fvt"]["max"] > 0.0


def test_report_3d_synthetic_quality_records_voting_options(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--ru",
        "2",
        "--rv",
        "3",
        "--rw",
        "4",
        "--seed-distance",
        "2",
        "--seed-threshold",
        "0.6",
        "--attribute-smoothing",
        "1",
        "--reference-thin-sigma",
        "1.5",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["voting"] == {
        "ru": 2,
        "rv": 3,
        "rw": 4,
        "seed_distance": 2,
        "seed_threshold": 0.6,
        "attribute_smoothing": 1,
        "voter_thin_mode": "reference",
        "reference_thin_sigma": 1.5,
    }


def test_report_3d_synthetic_quality_default_case_meets_smoke_thresholds(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    quality = metrics["cases"][0]["quality"]["fvt_top_truth_count"]

    assert quality["buffered_overlap_radius2"]["buffered_f1"] >= 0.80
    assert quality["surface_distance"]["candidate_to_truth_p95"] <= 3.0
    assert quality["orientation_error"]["strike_median"] <= 10.0
    assert quality["orientation_error"]["dip_median"] <= 10.0
