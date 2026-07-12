from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from pyosv.evaluation.reporting.markdown_v1 import (
    _figure_path,
    _format_markdown_metric,
    visual_report_markdown,
    write_visual_report_markdown,
)


_SHA256_FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "reporting" / "artifact_sha256.json"
)


def _pipeline_report() -> dict[str, object]:
    return {
        "quality": {
            "fvt_top_truth_count": {
                "buffered_overlap_radius2": {"buffered_f1": 0.75},
                "surface_distance": {"candidate_to_truth_p95": 1.25},
                "orientation_error": {"strike_median": 2.5, "dip_median": 3.5},
            },
            "skin": None,
        },
        "skinning": {"enabled": False},
    }


def _stage_diagnostic() -> dict[str, object]:
    def stage(boundary: float | None, interior: float | None) -> dict[str, object]:
        return {
            "candidate_count": 7,
            "regions": {
                "boundary_shell": {"truth_recall": boundary},
                "interior": {"truth_recall": interior},
            },
            "components": {"component_count": 2, "largest_component_fraction": 0.75},
            "edge_distance_profile": {
                "0": {"truth_recall": None},
                "1": {"truth_recall": 0.5},
            },
        }

    def transition(boundary: float, interior: float) -> dict[str, object]:
        return {
            "regions": {
                "boundary_shell": {
                    "retained_source_fraction": boundary,
                    "introduced_target_fraction": 0.3,
                },
                "interior": {"retained_source_fraction": interior},
            },
            "target_to_source_distance_p95": 1.5,
            "normal_shift": None,
            "tangential_shift_magnitude": 2.5,
        }

    return {
        "stage_order": ["stage_second", "stage_first"],
        "transition_order": ["transition_second", "transition_first"],
        "stages": {"stage_first": stage(0.11, 0.12), "stage_second": stage(0.21, 0.22)},
        "transitions": {
            "transition_first": transition(0.31, 0.32),
            "transition_second": transition(0.41, 0.42),
        },
        "skinning": {
            "enabled": True,
            "fallback_enabled": True,
            "fallback_used": False,
            "fallback_reason": None,
        },
    }


def test_visual_report_markdown_contains_case_variant_metrics_and_figure_path() -> None:
    report = {
        "config": {"input_mode": "oracle"},
        "cases": [
            {
                "case_id": "geometry/plane",
                "variants": {"current_default": _pipeline_report()},
            }
        ],
    }
    markdown = visual_report_markdown(report)
    assert "## geometry/plane" in markdown
    assert "### current_default" in markdown
    assert "- buffered_f1_r2: 0.75" in markdown
    assert "- distance_p95: 1.25" in markdown
    assert "geometry/plane/figures/truth_vs_fvt_overlay_i3_center.png" in markdown


def test_write_visual_report_markdown_writes_exact_rendering(tmp_path: Path) -> None:
    report = {
        "config": {"input_mode": "oracle"},
        "cases": [{"case_id": "plane", "variants": {"current_default": _pipeline_report()}}],
    }
    path = write_visual_report_markdown(report, tmp_path)
    assert path == tmp_path / "visual_report.md"
    assert path.read_text(encoding="utf-8") == visual_report_markdown(report)
    expected = json.loads(_SHA256_FIXTURE.read_text(encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["visual_report.md"]
    assert path.read_text(encoding="utf-8") == (
        "# Controlled Synthetic Quality Report\n"
        "\n"
        "## plane\n"
        "\n"
        "### current_default\n"
        "\n"
        "- buffered_f1_r2: 0.75\n"
        "- distance_p95: 1.25\n"
        "- strike_median_error: 2.5\n"
        "- dip_median_error: 3.5\n"
        "\n"
        "![fvt overlay](plane/figures/truth_vs_fvt_overlay_i3_center.png)\n"
        "\n"
        "- skinning disabled\n"
    )


def test_figure_path_preserves_variant_and_pipeline_layout() -> None:
    path = _figure_path(
        "geometry/plane",
        variant="boundary_aware_voter_v1",
        variant_count=2,
        pipeline="scanner",
        filename="truth_vs_fvt_overlay_i3_center.png",
    )
    assert path.as_posix() == (
        "geometry/plane/boundary_aware_voter_v1/scanner/figures/truth_vs_fvt_overlay_i3_center.png"
    )


def test_visual_report_markdown_renders_ordered_stage_diagnostics() -> None:
    pipeline = _pipeline_report()
    pipeline["scanner_boundary_stage_diagnostics"] = _stage_diagnostic()
    report = {
        "config": {"input_mode": "oracle"},
        "cases": [{"case_id": "plane", "variants": {"current_default": pipeline}}],
    }
    markdown = visual_report_markdown(report)
    assert "##### scanner boundary stage diagnostics" in markdown
    assert markdown.index("| stage_second |") < markdown.index("| stage_first |")
    assert markdown.index("| transition_second |") < markdown.index("| transition_first |")
    assert "| stage_second | 7 | 0.21 | 0.22 | 2 | 0.75 | n/a | 0.5 |" in markdown
    assert "| transition_second | 0.41 | 0.42 | 0.3 | 1.5 | n/a | 2.5 |" in markdown
    assert "- fallback enabled: true" in markdown
    assert "- fallback used: false" in markdown
    assert "- fallback reason: n/a" in markdown


def test_visual_report_markdown_renders_nested_region_truth_recall() -> None:
    pipeline = _pipeline_report()
    diagnostic = deepcopy(_stage_diagnostic())
    for stage in diagnostic["stages"].values():
        for region in stage["regions"].values():
            region["truth"] = {"truth_recall": region.pop("truth_recall")}
    pipeline["scanner_boundary_stage_diagnostics"] = diagnostic
    report = {
        "config": {"input_mode": "oracle"},
        "cases": [{"case_id": "plane", "variants": {"current_default": pipeline}}],
    }

    markdown = visual_report_markdown(report)

    assert "| stage_second | 7 | 0.21 | 0.22 | 2 | 0.75 | n/a | 0.5 |" in markdown


def test_existing_markdown_metric_formatting_is_unchanged() -> None:
    assert _format_markdown_metric(None) == "None"
    assert _format_markdown_metric(True) == "1"
