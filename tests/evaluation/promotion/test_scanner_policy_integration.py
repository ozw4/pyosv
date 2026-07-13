from __future__ import annotations

import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pyosv.evaluation.promotion.comparison import compare_reports
from pyosv.evaluation.promotion.gates import add_required_coverage, build_promotion_report
from pyosv.evaluation.promotion.markdown import comparison_markdown, promotion_markdown


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_quality_reports.py"
CASE_IDS = (
    "single_vertical_plane",
    "single_dipping_plane",
    "curved_surface",
    "parallel_planes",
    "crossing_planes",
    "boundary_plane",
    "weak_noisy_plane",
)


def _metrics(scanner_thin_mode: str) -> dict[str, object]:
    return {
        "format_version": 1,
        "config": {
            "case_set": "extended",
            "input_mode": "both",
            "workflow_mode": "quality",
            "shape": [49, 49, 49],
            "variants": ["current_default"],
            "variant_preset": "default",
            "voting": {"voter_thin_mode": "hybrid_v2", "seed_distance": 3},
            "truth_metrics": {"buffer_radius": 2.0},
            "skinning": {"method": "quality", "growth_source": "pre_thin"},
            "scanner_backend_matrix": False,
            "scanner_downstream_diagnostics": True,
            "scanner_boundary_stage_diagnostics": {"enabled": True},
            "scanner": {
                "backend": "quality",
                "phi_min": 0.0,
                "phi_max": 180.0,
                "theta_min": 45.0,
                "theta_max": 90.0,
                "sigma1": 2.0,
                "sigma2": 2.0,
                "refinement_factor": 2,
                "scanner_thin_mode": scanner_thin_mode,
                "remove_edge_effects": True,
                "input": {
                    "background": 1.0,
                    "fault_contrast": 0.85,
                    "noise_sigma": 0.0,
                    "seed": 20260706,
                    "clip_min": 0.0,
                    "clip_max": 1.0,
                },
            },
        },
    }


def _write_metrics(path: Path, report: dict[str, object]) -> None:
    path.write_text(json.dumps(report), encoding="utf-8")


def _summary_rows(scanner_thin_mode: str, variants: tuple[str, ...]) -> list[dict[str, object]]:
    rows = []
    for variant in variants:
        for pipeline in ("scanner", "oracle"):
            for case_id in CASE_IDS:
                rows.append(
                    {
                        "case_id": case_id,
                        "pipeline": pipeline,
                        "variant": variant,
                        "input_mode": "synthetic" if pipeline == "scanner" else "oracle",
                        "workflow_mode": "quality",
                        "scanner_backend": "quality" if pipeline == "scanner" else "",
                        "scanner_refinement_factor": 2 if pipeline == "scanner" else "",
                        "scanner_thin_mode": scanner_thin_mode,
                        "shape_n3": 49,
                        "shape_n2": 49,
                        "shape_n1": 49,
                        "skin_buffered_f1_r2": 0.95,
                        "skin_count": 2,
                        "skin_cell_count": 100,
                        "fvt_positive_candidate_count": 100,
                        "fvt_positive_buffered_f1_r2": 0.95,
                        "fvt_positive_distance_p95": 1.0,
                        "skin_distance_p95": 1.0,
                        "skin_fallback_replaced_primary": False,
                        "skin_over_merge_count": 0,
                        "skin_over_split_count": 0,
                    }
                )
    return rows


def _write_summary(path: Path, scanner_thin_mode: str, *variants: str) -> None:
    rows = _summary_rows(scanner_thin_mode, tuple(variants))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _policy_inputs(
    tmp_path: Path,
    *,
    baseline_mode: str = "reference",
    candidate_mode: str = "normal",
    variants: tuple[str, ...] = ("current_default", "other"),
) -> tuple[Path, Path, Path, Path, dict[str, object], dict[str, object]]:
    baseline_summary = tmp_path / "baseline.csv"
    candidate_summary = tmp_path / "candidate.csv"
    baseline_metrics = tmp_path / "baseline_metrics.json"
    candidate_metrics = tmp_path / "candidate_metrics.json"
    baseline_report = _metrics(baseline_mode)
    candidate_report = _metrics(candidate_mode)
    _write_summary(baseline_summary, baseline_mode, *variants)
    _write_summary(candidate_summary, candidate_mode, *variants)
    _write_metrics(baseline_metrics, baseline_report)
    _write_metrics(candidate_metrics, candidate_report)
    return (
        baseline_summary,
        candidate_summary,
        baseline_metrics,
        candidate_metrics,
        baseline_report,
        candidate_report,
    )


def _compare(
    paths: tuple[Path, Path, Path, Path, dict[str, object], dict[str, object]],
    *,
    baseline_variant: str = "current_default",
    candidate_variant: str = "current_default",
) -> dict[str, object]:
    baseline_summary, candidate_summary, baseline_metrics, candidate_metrics, _, _ = paths
    return compare_reports(
        baseline_summary,
        candidate_summary,
        baseline_variant,
        candidate_variant,
        promotion_gate="scanner-boundary",
        strict_missing_rows=True,
        comparison_profile="scanner-thinning-policy-v1",
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
    )


def test_valid_scanner_policy_comparison_matches_rows_and_passes_gate(tmp_path: Path) -> None:
    report = _compare(_policy_inputs(tmp_path))

    contract = report["scanner_policy_contract"]
    assert report["row_count"] == 14
    assert report["missing_baseline_rows"] == []
    assert report["missing_candidate_rows"] == []
    assert contract["passed"] is True
    assert contract["baseline"] == {
        "policy_id": "quality_scanner_reference_v1",
        "scanner_thin_mode": "reference",
        "requested_remove_edge_effects": True,
        "effective_remove_edge_effects": True,
    }
    assert contract["candidate"] == {
        "policy_id": "quality_scanner_thin_normal_v1",
        "scanner_thin_mode": "normal",
        "requested_remove_edge_effects": True,
        "effective_remove_edge_effects": None,
    }
    assert contract["observed_config_differences"] == [
        {
            "path": "config.scanner.scanner_thin_mode",
            "baseline": "reference",
            "candidate": "normal",
            "allowed": True,
        }
    ]
    assert report["promotion_gate"]["coverage"]["passed"] is True
    assert report["promotion_gate"]["passed"] is True


def test_scanner_policy_comparison_requires_both_metrics_paths(tmp_path: Path) -> None:
    paths = _policy_inputs(tmp_path)
    with pytest.raises(ValueError, match="requires a baseline metrics path"):
        compare_reports(
            paths[0],
            paths[1],
            "current_default",
            "current_default",
            comparison_profile="scanner-thinning-policy-v1",
        )
    with pytest.raises(ValueError, match="requires a candidate metrics path"):
        compare_reports(
            paths[0],
            paths[1],
            "current_default",
            "current_default",
            comparison_profile="scanner-thinning-policy-v1",
            baseline_metrics=paths[2],
        )


def test_variant_profile_does_not_read_optional_metrics_paths(tmp_path: Path) -> None:
    paths = _policy_inputs(tmp_path)
    report = compare_reports(
        paths[0],
        paths[1],
        "current_default",
        "current_default",
        baseline_metrics=tmp_path / "missing-baseline.json",
        candidate_metrics=tmp_path / "missing-candidate.json",
    )

    assert "comparison_profile" not in report["config"]
    assert "scanner_policy_contract" not in report


@pytest.mark.parametrize(
    ("mutation", "baseline_variant", "candidate_variant", "reason_fragment"),
    (
        ("candidate_reference", "current_default", "current_default", "scanner_thin_mode"),
        ("baseline_normal", "current_default", "current_default", "scanner_thin_mode"),
        ("candidate_backend", "current_default", "current_default", "scanner.backend"),
        ("candidate_refinement", "current_default", "current_default", "refinement_factor"),
        ("candidate_sigma", "current_default", "current_default", "scanner.sigma1"),
        ("candidate_seed", "current_default", "current_default", "scanner.input.seed"),
        ("candidate_voting", "current_default", "current_default", "voting.voter_thin_mode"),
        ("candidate_skinning", "current_default", "current_default", "skinning.method"),
        ("candidate_edges", "current_default", "current_default", "remove_edge_effects"),
        ("variant_mismatch", "current_default", "other", "selected variants differ"),
        ("non_default", "other", "other", "selected variant must be 'current_default'"),
    ),
)
def test_contract_failure_forces_numeric_gate_failure(
    tmp_path: Path,
    mutation: str,
    baseline_variant: str,
    candidate_variant: str,
    reason_fragment: str,
) -> None:
    paths = _policy_inputs(tmp_path)
    baseline_report = copy.deepcopy(paths[4])
    candidate_report = copy.deepcopy(paths[5])
    baseline_scanner = baseline_report["config"]["scanner"]
    candidate_config = candidate_report["config"]
    candidate_scanner = candidate_config["scanner"]
    if mutation == "candidate_reference":
        candidate_scanner["scanner_thin_mode"] = "reference"
    elif mutation == "baseline_normal":
        baseline_scanner["scanner_thin_mode"] = "normal"
    elif mutation == "candidate_backend":
        candidate_scanner["backend"] = "fast"
    elif mutation == "candidate_refinement":
        candidate_scanner["refinement_factor"] = 3
    elif mutation == "candidate_sigma":
        candidate_scanner["sigma1"] = 3.0
    elif mutation == "candidate_seed":
        candidate_scanner["input"]["seed"] = 7
    elif mutation == "candidate_voting":
        candidate_config["voting"]["voter_thin_mode"] = "reference"
    elif mutation == "candidate_skinning":
        candidate_config["skinning"]["method"] = "reference"
    elif mutation == "candidate_edges":
        candidate_scanner["remove_edge_effects"] = False
    _write_metrics(paths[2], baseline_report)
    _write_metrics(paths[3], candidate_report)

    report = _compare(
        paths,
        baseline_variant=baseline_variant,
        candidate_variant=candidate_variant,
    )

    assert report["scanner_policy_contract"]["passed"] is False
    assert report["promotion_gate"]["passed"] is False
    assert any(reason_fragment in reason for reason in report["scanner_policy_contract"]["reasons"])
    assert any(
        reason.startswith("scanner policy contract:")
        for reason in report["promotion_gate"]["reasons"]
    )


def test_contract_failure_makes_fail_on_gate_failure_exit_two(tmp_path: Path) -> None:
    paths = _policy_inputs(tmp_path, candidate_mode="reference")
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(paths[0]),
            str(paths[1]),
            "--comparison-profile",
            "scanner-thinning-policy-v1",
            "--baseline-metrics",
            str(paths[2]),
            "--candidate-metrics",
            str(paths[3]),
            "--promotion-gate",
            "scanner-boundary",
            "--fail-on-gate-failure",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_aggregate_contract_failure_makes_fail_on_gate_failure_exit_two(
    tmp_path: Path,
) -> None:
    paths = _policy_inputs(tmp_path, candidate_mode="reference")
    gate_script = REPO_ROOT / "scripts" / "check_synthetic_quality_promotion_gate.py"
    result = subprocess.run(
        [
            sys.executable,
            str(gate_script),
            "--baseline-summary",
            str(paths[0]),
            "--candidate-summary",
            str(paths[1]),
            "--candidate-variant",
            "current_default",
            "--comparison-profile",
            "scanner-thinning-policy-v1",
            "--baseline-metrics",
            str(paths[2]),
            "--candidate-metrics",
            str(paths[3]),
            "--fail-on-gate-failure",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_policy_contract_is_preserved_by_coverage_and_aggregate_report(tmp_path: Path) -> None:
    paths = _policy_inputs(tmp_path)
    comparison = add_required_coverage(_compare(paths))
    aggregate = build_promotion_report(
        paths[0],
        paths[1],
        candidate_variants=("current_default",),
        comparison_profile="scanner-thinning-policy-v1",
        baseline_metrics=paths[2],
        candidate_metrics=paths[3],
    )

    assert comparison["promotion_gate"]["coverage"]["passed"] is True
    assert comparison["promotion_gate"]["passed"] is True
    assert aggregate["scanner_policy_contract"]["passed"] is True
    assert aggregate["promotion_gate"]["passed"] is True


def test_new_profile_markdown_shows_policy_semantics_and_sorted_differences(
    tmp_path: Path,
) -> None:
    paths = _policy_inputs(tmp_path)
    valid_text = comparison_markdown(_compare(paths))
    assert "Contract result: pass" in valid_text
    assert "quality_scanner_reference_v1" in valid_text
    assert "quality_scanner_thin_normal_v1" in valid_text
    assert "requested=true, effective=true" in valid_text
    assert "requested=true, effective=null" in valid_text

    candidate_report = copy.deepcopy(paths[5])
    candidate_report["config"]["voting"]["voter_thin_mode"] = "reference"
    candidate_report["config"]["scanner"]["sigma1"] = 3.0
    _write_metrics(paths[3], candidate_report)
    comparison = _compare(paths)
    aggregate = build_promotion_report(
        paths[0],
        paths[1],
        candidate_variants=("current_default",),
        comparison_profile="scanner-thinning-policy-v1",
        baseline_metrics=paths[2],
        candidate_metrics=paths[3],
    )

    comparison_text = comparison_markdown(comparison)
    promotion_text = promotion_markdown(aggregate)
    for text in (comparison_text, promotion_text):
        assert "## Scanner Policy Contract" in text
        assert "Contract result: fail" in text
        assert "quality_scanner_reference_v1" in text
        assert "quality_scanner_thin_normal_v1" not in text
        assert "requested=true, effective=true" in text
        assert "requested=true, effective=null" in text
        assert text.index("config.scanner.sigma1") < text.index("config.voting.voter_thin_mode")
        assert text.endswith("\n")
