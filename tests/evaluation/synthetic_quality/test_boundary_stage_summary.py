from __future__ import annotations

import json

import pytest

from pyosv.evaluation.synthetic_quality.application import build_report
from pyosv.evaluation.synthetic_quality.boundary_stage_summary import (
    scanner_boundary_stage_summary_markdown,
    select_scanner_boundary_stage_diagnostics,
    summarize_scanner_boundary_stages,
)
from pyosv.evaluation.synthetic_quality.config import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
)
from pyosv.evaluation.reporting.json_v1 import write_metrics_json


def _diagnostic() -> dict[str, object]:
    def stage(recall: float | None) -> dict[str, object]:
        return {"regions": {"boundary_shell": {"truth_recall": recall}}}

    def transition(
        boundary: float | None, interior: float | None, introduced: float | None
    ) -> dict[str, object]:
        return {
            "regions": {
                "boundary_shell": {
                    "retained_source_fraction": boundary,
                    "introduced_target_fraction": introduced,
                },
                "interior": {"retained_source_fraction": interior},
            }
        }

    return {
        "stage_order": ["raw", "thin", "seed", "final"],
        "transition_order": ["raw_to_thin", "thin_to_seed", "seed_to_final", "unused"],
        "stages": {
            "raw": stage(0.9),
            "thin": stage(0.7),
            "seed": stage(0.7),
            "final": stage(None),
        },
        "transitions": {
            "raw_to_thin": transition(0.7, 0.9, 0.1),
            "thin_to_seed": transition(0.7, 0.8, 0.4),
            "seed_to_final": transition(0.5, 0.9, 0.2),
            "unused": transition(None, None, None),
        },
    }


def _report(diagnostic: dict[str, object]) -> dict[str, object]:
    scanner_variant = {"scanner_boundary_stage_diagnostics": diagnostic}
    return {
        "format_version": 1,
        "cases": [
            {
                "case_id": "boundary_plane",
                "scanner_boundary_stage_diagnostics": {"wrong": "top-level alias"},
                "pipelines": {
                    "oracle": {
                        "variants": {
                            "current_default": {
                                "scanner_boundary_stage_diagnostics": {"wrong": "oracle"}
                            }
                        }
                    },
                    "scanner": {"variants": {"current_default": scanner_variant}},
                },
            }
        ],
    }


def test_selects_only_canonical_scanner_pipeline() -> None:
    diagnostic = _diagnostic()

    assert (
        select_scanner_boundary_stage_diagnostics(
            _report(diagnostic), case_id="boundary_plane", variant="current_default"
        )
        is diagnostic
    )


@pytest.mark.parametrize("missing", ("case", "scanner", "variant", "diagnostic"))
def test_selector_rejects_missing_report_path(missing: str) -> None:
    report = _report(_diagnostic())
    case = report["cases"][0]
    if missing == "case":
        report["cases"] = []
    elif missing == "scanner":
        del case["pipelines"]["scanner"]
    elif missing == "variant":
        case["pipelines"]["scanner"]["variants"] = {}
    else:
        case["pipelines"]["scanner"]["variants"]["current_default"] = {}

    with pytest.raises(ValueError):
        select_scanner_boundary_stage_diagnostics(
            report, case_id="boundary_plane", variant="current_default"
        )


def test_selector_rejects_duplicate_case() -> None:
    report = _report(_diagnostic())
    report["cases"].append(report["cases"][0])

    with pytest.raises(ValueError, match="more than once"):
        select_scanner_boundary_stage_diagnostics(
            report, case_id="boundary_plane", variant="current_default"
        )


def test_summary_uses_explicit_order_ties_and_excludes_none() -> None:
    summary = summarize_scanner_boundary_stages(_diagnostic(), retention_threshold=0.8)

    assert summary["first_boundary_retention_below_threshold"] == {
        "transition": "raw_to_thin",
        "value": 0.7,
    }
    assert summary["lowest_boundary_retention"] == {
        "transition": "seed_to_final",
        "value": 0.5,
    }
    assert summary["largest_boundary_vs_interior_retention_penalty"] == {
        "transition": "seed_to_final",
        "boundary": 0.5,
        "interior": 0.9,
        "boundary_minus_interior": -0.4,
    }
    assert summary["largest_boundary_truth_recall_drop"]["source_stage"] == "raw"
    assert summary["largest_boundary_truth_recall_drop"]["delta"] == pytest.approx(-0.2)
    assert summary["highest_boundary_introduced_fraction"] == {
        "transition": "thin_to_seed",
        "value": 0.4,
    }
    assert [row["transition"] for row in summary["ranked_transitions_by_boundary_retention"]] == [
        "seed_to_final",
        "raw_to_thin",
        "thin_to_seed",
    ]
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("threshold", (-0.1, 1.1, float("nan"), float("inf"), True))
def test_summary_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        summarize_scanner_boundary_stages(_diagnostic(), retention_threshold=threshold)


def test_markdown_is_deterministic_and_uses_na_for_undefined() -> None:
    empty = _diagnostic()
    for transition in empty["transitions"].values():
        transition["regions"]["boundary_shell"]["retained_source_fraction"] = None
        transition["regions"]["boundary_shell"]["introduced_target_fraction"] = None
    summary = summarize_scanner_boundary_stages(empty, retention_threshold=0.8)

    markdown = scanner_boundary_stage_summary_markdown(
        case_id="boundary_plane", variant="current_default", summary=summary
    )

    assert markdown.endswith("\n")
    assert "- first transition below threshold: n/a" in markdown
    assert "| n/a | n/a | n/a |" in markdown
    assert markdown == scanner_boundary_stage_summary_markdown(
        case_id="boundary_plane", variant="current_default", summary=summary
    )


def test_small_serialized_scanner_report_summary_is_json_safe_and_ordered(tmp_path) -> None:
    report = build_report(
        case_set="minimal",
        shape=(9, 9, 9),
        variants=("current_default",),
        input_mode="scanner",
        scanner_config=SyntheticScannerConfig(
            backend="fast",
            phi_min=0.0,
            phi_max=0.0,
            theta_min=90.0,
            theta_max=90.0,
            sigma1=1.0,
            sigma2=1.0,
            scanner_thin_mode="none",
        ),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        include_scanner_boundary_stage_diagnostics=True,
    )
    metrics_path = write_metrics_json(report, tmp_path)
    serialized_report = json.loads(metrics_path.read_text(encoding="utf-8"))

    diagnostic = select_scanner_boundary_stage_diagnostics(
        serialized_report, case_id="single_vertical_plane", variant="current_default"
    )
    summary = summarize_scanner_boundary_stages(diagnostic, retention_threshold=0.8)

    json.dumps(summary, allow_nan=False)
    ranked_names = [
        row["transition"] for row in summary["ranked_transitions_by_boundary_retention"]
    ]
    assert all(name in diagnostic["transition_order"] for name in ranked_names)
