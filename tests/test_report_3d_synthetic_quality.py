from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "examples" / "report_3d_synthetic_quality.py"
GEOMETRY_CASE_IDS = ("single_vertical_plane", "single_dipping_plane", "curved_surface")
EXPECTED_VOLUME_FILES = (
    "truth_fault_mask.dat",
    "truth_distance.dat",
    "truth_strike.dat",
    "truth_dip.dat",
    "ft_oracle.dat",
    "pt_oracle.dat",
    "tt_oracle.dat",
    "fv_py.dat",
    "vp_py.dat",
    "vt_py.dat",
    "fvt_py.dat",
)
EXPECTED_I3_FIGURES = (
    "ft_oracle_i3_center.png",
    "fv_py_i3_center.png",
    "fvt_py_i3_center.png",
    "truth_vs_fvt_overlay_i3_center.png",
)


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
    assert "geometry" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--shape" in result.stdout
    assert "--variants" in result.stdout
    assert "--save-volumes" in result.stdout
    assert "--save-figures" in result.stdout
    assert "--write-markdown-index" in result.stdout
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
    assert set(case["variants"]) == {"current_default"}
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
    assert rows[0]["variant"] == "current_default"
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


def test_report_3d_synthetic_quality_geometry_case_set_writes_three_case_rows(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["format_version"] == 1
    assert metrics["config"]["case_set"] == "geometry"
    assert metrics["config"]["shape"] == [17, 17, 17]
    assert [case["case_id"] for case in metrics["cases"]] == list(GEOMETRY_CASE_IDS)
    for case in metrics["cases"]:
        assert case["shape"] == [17, 17, 17]
        assert set(case["variants"]) == {"current_default"}
        assert case["truth"]["surface_voxel_count"] > 0

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [(row["case_id"], row["variant"]) for row in rows] == [
        (case_id, "current_default") for case_id in GEOMETRY_CASE_IDS
    ]
    assert all(float(row["fvt_buffered_f1_r2"]) >= 0.0 for row in rows)


def test_report_3d_synthetic_quality_variants_write_metrics_and_summary_rows(
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
        "--variants",
        "current_default,voter_thin_normal",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    case = metrics["cases"][0]
    assert list(case["variants"]) == ["current_default", "voter_thin_normal"]
    assert case["variants"]["current_default"]["pyosv"]["fvt"]["max"] > 0.0
    assert case["variants"]["voter_thin_normal"]["pyosv"]["fvt"]["max"] > 0.0

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert [(row["case_id"], row["variant"]) for row in rows] == [
        ("single_vertical_plane", "current_default"),
        ("single_vertical_plane", "voter_thin_normal"),
    ]


def test_report_3d_synthetic_quality_diagnostic_variants_pass(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--variants",
        "no_surface_orientation_smoothing,final_norm_smoothing_1",
    )

    assert result.returncode == 0, result.stderr
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["config"]["variants"] == [
        "no_surface_orientation_smoothing",
        "final_norm_smoothing_1",
    ]
    assert set(metrics["cases"][0]["variants"]) == {
        "no_surface_orientation_smoothing",
        "final_norm_smoothing_1",
    }


def test_report_3d_synthetic_quality_unknown_variant_fails(tmp_path: Path) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--variants",
        "current_default,unknown_variant",
    )

    assert result.returncode != 0
    assert "unknown variant" in result.stderr


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


def test_report_3d_synthetic_quality_save_volumes_writes_expected_dat_files(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"
    shape = (17, 17, 17)

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        ",".join(str(size) for size in shape),
        "--output-dir",
        str(output_dir),
        "--save-volumes",
    )

    assert result.returncode == 0, result.stderr
    case_dir = output_dir / "single_vertical_plane"
    expected_size = shape[0] * shape[1] * shape[2] * 4
    for name in EXPECTED_VOLUME_FILES:
        path = case_dir / name
        assert path.is_file()
        assert path.stat().st_size == expected_size


def test_report_3d_synthetic_quality_geometry_save_volumes_splits_case_directories(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"
    shape = (17, 17, 17)

    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        ",".join(str(size) for size in shape),
        "--output-dir",
        str(output_dir),
        "--save-volumes",
    )

    assert result.returncode == 0, result.stderr
    expected_size = shape[0] * shape[1] * shape[2] * 4
    for case_id in GEOMETRY_CASE_IDS:
        case_dir = output_dir / case_id
        for name in EXPECTED_VOLUME_FILES:
            path = case_dir / name
            assert path.is_file()
            assert path.stat().st_size == expected_size


def test_report_3d_synthetic_quality_write_markdown_index_includes_case_and_metrics(
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
        "--write-markdown-index",
    )

    assert result.returncode == 0, result.stderr
    markdown = (output_dir / "visual_report.md").read_text(encoding="utf-8")
    assert "# Controlled Synthetic Quality Report" in markdown
    assert "## single_vertical_plane" in markdown
    assert "buffered_f1_r2" in markdown
    assert "distance_p95" in markdown
    assert "strike_median_error" in markdown
    assert "dip_median_error" in markdown
    assert "single_vertical_plane/figures/truth_vs_fvt_overlay_i3_center.png" in markdown


def test_report_3d_synthetic_quality_geometry_markdown_index_includes_each_case(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--write-markdown-index",
    )

    assert result.returncode == 0, result.stderr
    markdown = (output_dir / "visual_report.md").read_text(encoding="utf-8")
    for case_id in GEOMETRY_CASE_IDS:
        assert f"## {case_id}" in markdown
        assert f"{case_id}/figures/truth_vs_fvt_overlay_i3_center.png" in markdown


def test_report_3d_synthetic_quality_save_figures_writes_expected_pngs(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "minimal",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--save-figures",
    )

    assert result.returncode == 0, result.stderr
    figures_dir = output_dir / "single_vertical_plane" / "figures"
    for name in EXPECTED_I3_FIGURES:
        path = figures_dir / name
        assert path.is_file()
        assert path.stat().st_size > 0


def test_report_3d_synthetic_quality_geometry_save_figures_splits_case_directories(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    output_dir = tmp_path / "synthetic_quality"

    result = _run_script(
        "--case-set",
        "geometry",
        "--shape",
        "17,17,17",
        "--output-dir",
        str(output_dir),
        "--save-figures",
    )

    assert result.returncode == 0, result.stderr
    for case_id in GEOMETRY_CASE_IDS:
        figures_dir = output_dir / case_id / "figures"
        for name in EXPECTED_I3_FIGURES:
            path = figures_dir / name
            assert path.is_file()
            assert path.stat().st_size > 0


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
    variant_quality = metrics["cases"][0]["variants"]["current_default"]["quality"][
        "fvt_top_truth_count"
    ]

    assert quality["buffered_overlap_radius2"]["buffered_f1"] >= 0.80
    assert quality["surface_distance"]["candidate_to_truth_p95"] <= 3.0
    assert quality["orientation_error"]["strike_median"] <= 10.0
    assert quality["orientation_error"]["dip_median"] <= 10.0
    assert variant_quality["buffered_overlap_radius2"]["buffered_f1"] >= 0.80
    assert variant_quality["surface_distance"]["candidate_to_truth_p95"] <= 3.0
    assert variant_quality["orientation_error"]["strike_median"] <= 10.0
    assert variant_quality["orientation_error"]["dip_median"] <= 10.0
