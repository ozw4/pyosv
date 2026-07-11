from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pyosv.evaluation.reporting.markdown_v1 import (
    _figure_path,
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
