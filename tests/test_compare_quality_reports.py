from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from pyosv.evaluation.reporting import write_summary_csv

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_quality_reports.py"
GATE_SCRIPT = REPO_ROOT / "scripts" / "check_synthetic_quality_promotion_gate.py"
FIXTURE = Path("tests/fixtures/synthetic_quality_refactor/known_49_quality_summary.csv")
HASH_FIXTURE = REPO_ROOT / (
    "tests/fixtures/synthetic_quality_refactor/known_49_promotion_artifact_sha256.json"
)
METRICS_FIXTURE = REPO_ROOT / (
    "tests/fixtures/synthetic_quality_refactor/17_quality_ref2_metrics.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_hashes() -> dict[str, str]:
    return json.loads(HASH_FIXTURE.read_text(encoding="utf-8"))


def _policy_metrics(scanner_thin_mode: str) -> dict[str, object]:
    report = json.loads(METRICS_FIXTURE.read_text(encoding="utf-8"))
    report["config"]["shape"] = [49, 49, 49]
    for case in report["cases"]:
        case["shape"] = [49, 49, 49]
        for variant in case["pipelines"]["scanner"]["variants"].values():
            quality = variant["quality"]
            fvt_count = quality["fvt_positive_top_truth_count"]["buffered_overlap_radius2"][
                "candidate_count"
            ]
            quality["skin"]["buffered_overlap_radius2"]["buffered_f1"] = 0.95
            quality["skin"]["topology"]["cell_count"] = fvt_count
    _replace_scanner_mode(report, scanner_thin_mode)
    return report


def _replace_scanner_mode(value: object, scanner_thin_mode: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "scanner_thin_mode":
                value[key] = scanner_thin_mode
            else:
                _replace_scanner_mode(child, scanner_thin_mode)
    elif isinstance(value, list):
        for child in value:
            _replace_scanner_mode(child, scanner_thin_mode)


def _write_policy_inputs(tmp_path: Path) -> tuple[Path, Path]:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_report = _policy_metrics("reference")
    candidate_report = _policy_metrics("normal")
    baseline_summary = write_summary_csv(baseline_report, baseline_dir)
    candidate_summary = write_summary_csv(candidate_report, candidate_dir)
    (baseline_dir / "metrics.json").write_text(json.dumps(baseline_report), encoding="utf-8")
    (candidate_dir / "metrics.json").write_text(json.dumps(candidate_report), encoding="utf-8")
    return baseline_summary, candidate_summary


def _reference_like_policy_metrics(scanner_thin_mode: str) -> dict[str, object]:
    report = _policy_metrics(scanner_thin_mode)
    _replace_scanner_backend(report)
    report["config"]["variants"] = ["current_default"]
    for case in report["cases"]:
        case["variants"] = {"current_default": case["variants"]["current_default"]}
        for pipeline in case["pipelines"].values():
            pipeline["variants"] = {"current_default": pipeline["variants"]["current_default"]}
            comparisons = pipeline["variant_comparison"]["variants"]
            pipeline["variant_comparison"]["variants"] = {
                "current_default": comparisons.get("current_default", {})
            }
        scanner_quality = case["pipelines"]["scanner"]["variants"]["current_default"]["quality"]
        fvt_quality = scanner_quality["fvt_positive_top_truth_count"]
        fvt_quality["buffered_overlap_radius2"]["buffered_f1"] = 0.95
        fvt_quality["surface_distance"]["candidate_to_truth_p95"] = 1.0
    return report


def _replace_scanner_backend(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "backend" and child == "quality":
                value[key] = "reference-like"
            else:
                _replace_scanner_backend(child)
    elif isinstance(value, list):
        for child in value:
            _replace_scanner_backend(child)


def _write_reference_like_policy_inputs(tmp_path: Path) -> tuple[Path, Path]:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_report = _reference_like_policy_metrics("reference")
    candidate_report = _reference_like_policy_metrics("normal")
    baseline_summary = write_summary_csv(baseline_report, baseline_dir)
    candidate_summary = write_summary_csv(candidate_report, candidate_dir)
    (baseline_dir / "metrics.json").write_text(json.dumps(baseline_report), encoding="utf-8")
    (candidate_dir / "metrics.json").write_text(json.dumps(candidate_report), encoding="utf-8")
    return baseline_summary, candidate_summary


def _write_reference_like_one_row_inputs(tmp_path: Path) -> tuple[Path, Path]:
    full_summary = write_summary_csv(_reference_like_policy_metrics("normal"), tmp_path / "full")
    lines = full_summary.read_text(encoding="utf-8").splitlines()
    boundary_scanner_row = next(
        line for line in lines[1:] if line.startswith("boundary_plane,scanner,current_default,")
    )
    content = f"{lines[0]}\n{boundary_scanner_row}\n"
    baseline_summary = tmp_path / "one-row-baseline" / "summary.csv"
    candidate_summary = tmp_path / "one-row-candidate" / "summary.csv"
    baseline_summary.parent.mkdir()
    candidate_summary.parent.mkdir()
    baseline_summary.write_text(content, encoding="utf-8")
    candidate_summary.write_text(content, encoding="utf-8")
    return baseline_summary, candidate_summary


def _assert_required_profile_usage_error(
    result: subprocess.CompletedProcess[str],
    output_json: Path,
    output_markdown: Path,
) -> None:
    assert result.returncode == 2
    assert "requires comparison profile" in result.stderr
    assert "scanner-boundary-reference-like" in result.stderr
    assert "quality-workflow-scanner-thinning-v1" in result.stderr
    assert "variant" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_json.exists()
    assert not output_markdown.exists()


def test_compare_quality_reports_cli_preserves_json_and_markdown_contract(tmp_path: Path) -> None:
    output_json = tmp_path / "quality_delta.json"
    output_markdown = tmp_path / "quality_delta.md"
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(FIXTURE),
            str(FIXTURE),
            "--candidate-variant",
            "boundary_aware_voter_v1",
            "--promotion-gate",
            "scanner-boundary",
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    expected = _expected_hashes()
    assert _sha256(output_json) == expected["comparison.json"]
    assert _sha256(output_markdown) == expected["comparison.md"]


def test_promotion_gate_cli_preserves_json_markdown_and_failure_exit(tmp_path: Path) -> None:
    output_json = tmp_path / "promotion_gate.json"
    output_markdown = tmp_path / "promotion_gate.md"
    result = subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--baseline-summary",
            str(FIXTURE),
            "--candidate-summary",
            str(FIXTURE),
            "--candidate-variant",
            "boundary_aware_voter_v1",
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
            "--fail-on-gate-failure",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2, result.stderr
    expected = _expected_hashes()
    assert _sha256(output_json) == expected["promotion.json"]
    assert _sha256(output_markdown) == expected["promotion.md"]


def test_scanner_policy_compare_cli_infers_metrics_paths(tmp_path: Path) -> None:
    baseline_summary, candidate_summary = _write_policy_inputs(tmp_path)
    output_json = tmp_path / "quality_delta.json"
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(baseline_summary),
            str(candidate_summary),
            "--comparison-profile",
            "scanner-thinning-policy-v1",
            "--promotion-gate",
            "scanner-boundary",
            "--strict-missing-rows",
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["scanner_policy_contract"]["passed"] is True
    assert report["promotion_gate"]["passed"] is True


def test_scanner_policy_compare_cli_explicit_metrics_paths_take_precedence(
    tmp_path: Path,
) -> None:
    baseline_summary, candidate_summary = _write_policy_inputs(tmp_path)
    baseline_summary.with_name("metrics.json").write_text("not json", encoding="utf-8")
    candidate_summary.with_name("metrics.json").write_text("not json", encoding="utf-8")
    explicit_baseline = tmp_path / "baseline_metrics.json"
    explicit_candidate = tmp_path / "candidate_metrics.json"
    explicit_baseline.write_text(json.dumps(_policy_metrics("reference")), encoding="utf-8")
    explicit_candidate.write_text(json.dumps(_policy_metrics("normal")), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(baseline_summary),
            str(candidate_summary),
            "--comparison-profile",
            "scanner-thinning-policy-v1",
            "--baseline-metrics",
            str(explicit_baseline),
            "--candidate-metrics",
            str(explicit_candidate),
            "--output-json",
            str(tmp_path / "quality_delta.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_reference_like_policy_compare_cli_infers_metrics_paths(tmp_path: Path) -> None:
    baseline_summary, candidate_summary = _write_reference_like_policy_inputs(tmp_path)
    output_json = tmp_path / "quality_delta.json"
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(baseline_summary),
            str(candidate_summary),
            "--comparison-profile",
            "quality-workflow-scanner-thinning-v1",
            "--promotion-gate",
            "scanner-boundary-reference-like",
            "--strict-missing-rows",
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(output_json.read_text(encoding="utf-8"))
    contract = report["scanner_policy_contract"]
    assert report["row_count"] == 14
    assert report["missing_baseline_rows"] == []
    assert report["missing_candidate_rows"] == []
    assert contract["passed"] is True
    assert contract["baseline"] == {
        "policy_id": "quality_reference_like_scanner_thin_reference_v1",
        "scanner_thin_mode": "reference",
        "requested_remove_edge_effects": True,
        "effective_remove_edge_effects": True,
    }
    assert contract["candidate"] == {
        "policy_id": "quality_reference_like_scanner_thin_normal_v1",
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


@pytest.mark.parametrize(
    "profile_args",
    [(), ("--comparison-profile", "variant")],
    ids=("omitted-profile", "explicit-variant"),
)
def test_reference_like_gate_compare_cli_requires_quality_workflow_profile(
    tmp_path: Path,
    profile_args: tuple[str, ...],
) -> None:
    baseline_summary, candidate_summary = _write_reference_like_one_row_inputs(tmp_path)
    output_json = tmp_path / "must-not-exist.json"
    output_markdown = tmp_path / "must-not-exist.md"
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(baseline_summary),
            str(candidate_summary),
            "--baseline-variant",
            "current_default",
            "--candidate-variant",
            "current_default",
            "--promotion-gate",
            "scanner-boundary-reference-like",
            "--strict-missing-rows",
            "--fail-on-gate-failure",
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
            *profile_args,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    _assert_required_profile_usage_error(result, output_json, output_markdown)


def test_reference_like_policy_compare_cli_explicit_metrics_paths_take_precedence(
    tmp_path: Path,
) -> None:
    baseline_summary, candidate_summary = _write_reference_like_policy_inputs(tmp_path)
    baseline_summary.with_name("metrics.json").write_text("not json", encoding="utf-8")
    candidate_summary.with_name("metrics.json").write_text("not json", encoding="utf-8")
    explicit_baseline = tmp_path / "baseline_metrics.json"
    explicit_candidate = tmp_path / "candidate_metrics.json"
    explicit_baseline.write_text(
        json.dumps(_reference_like_policy_metrics("reference")), encoding="utf-8"
    )
    explicit_candidate.write_text(
        json.dumps(_reference_like_policy_metrics("normal")), encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(baseline_summary),
            str(candidate_summary),
            "--comparison-profile",
            "quality-workflow-scanner-thinning-v1",
            "--baseline-metrics",
            str(explicit_baseline),
            "--candidate-metrics",
            str(explicit_candidate),
            "--output-json",
            str(tmp_path / "quality_delta.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_reference_like_gate_rejects_missing_matched_rows(tmp_path: Path) -> None:
    baseline_summary, candidate_summary = _write_reference_like_policy_inputs(tmp_path)
    candidate_metrics = candidate_summary.with_name("metrics.json")
    candidate_report = json.loads(candidate_metrics.read_text(encoding="utf-8"))
    candidate_report["cases"].pop()
    write_summary_csv(candidate_report, candidate_summary.parent)
    candidate_metrics.write_text(json.dumps(candidate_report), encoding="utf-8")
    output_json = tmp_path / "quality_delta.json"

    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(baseline_summary),
            str(candidate_summary),
            "--comparison-profile",
            "quality-workflow-scanner-thinning-v1",
            "--promotion-gate",
            "scanner-boundary-reference-like",
            "--strict-missing-rows",
            "--fail-on-gate-failure",
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2, result.stderr
    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert len(report["missing_candidate_rows"]) == 2
    assert report["promotion_gate"]["passed"] is False
    assert (
        "scanner-boundary-reference-like requires zero missing candidate rows"
        in report["promotion_gate"]["reasons"]
    )


def test_scanner_policy_compare_cli_reports_metrics_errors_without_traceback(
    tmp_path: Path,
) -> None:
    baseline_summary, candidate_summary = _write_policy_inputs(tmp_path)
    baseline_summary.with_name("metrics.json").unlink()
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(baseline_summary),
            str(candidate_summary),
            "--comparison-profile",
            "scanner-thinning-policy-v1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "metrics" in result.stderr
    assert "Traceback" not in result.stderr


def test_scanner_policy_compare_cli_rejects_mismatched_evidence(tmp_path: Path) -> None:
    baseline_summary, candidate_summary = _write_policy_inputs(tmp_path)
    output_json = tmp_path / "must-not-exist.json"
    result = subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            str(candidate_summary),
            str(candidate_summary),
            "--comparison-profile",
            "scanner-thinning-policy-v1",
            "--baseline-metrics",
            str(baseline_summary.with_name("metrics.json")),
            "--candidate-metrics",
            str(candidate_summary.with_name("metrics.json")),
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "baseline" in result.stderr
    assert "summary" in result.stderr
    assert "metrics" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_json.exists()


def test_scanner_policy_aggregate_cli_accepts_one_current_default_candidate(
    tmp_path: Path,
) -> None:
    baseline_summary, candidate_summary = _write_policy_inputs(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--baseline-summary",
            str(baseline_summary),
            "--candidate-summary",
            str(candidate_summary),
            "--candidate-variant",
            "current_default",
            "--comparison-profile",
            "scanner-thinning-policy-v1",
            "--output-json",
            str(tmp_path / "promotion_gate.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_reference_like_policy_aggregate_cli_accepts_one_current_default_candidate(
    tmp_path: Path,
) -> None:
    baseline_summary, candidate_summary = _write_reference_like_policy_inputs(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--baseline-summary",
            str(baseline_summary),
            "--candidate-summary",
            str(candidate_summary),
            "--candidate-variant",
            "current_default",
            "--comparison-profile",
            "quality-workflow-scanner-thinning-v1",
            "--promotion-gate",
            "scanner-boundary-reference-like",
            "--output-json",
            str(tmp_path / "promotion_gate.json"),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "profile_args",
    [(), ("--comparison-profile", "variant")],
    ids=("omitted-profile", "explicit-variant"),
)
def test_reference_like_gate_aggregate_cli_requires_quality_workflow_profile(
    tmp_path: Path,
    profile_args: tuple[str, ...],
) -> None:
    baseline_summary, candidate_summary = _write_reference_like_one_row_inputs(tmp_path)
    output_json = tmp_path / "must-not-exist.json"
    output_markdown = tmp_path / "must-not-exist.md"
    result = subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--baseline-summary",
            str(baseline_summary),
            "--candidate-summary",
            str(candidate_summary),
            "--baseline-variant",
            "current_default",
            "--candidate-variant",
            "current_default",
            "--promotion-gate",
            "scanner-boundary-reference-like",
            "--strict-missing-rows",
            "--fail-on-gate-failure",
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
            *profile_args,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    _assert_required_profile_usage_error(result, output_json, output_markdown)


def test_reference_like_policy_aggregate_cli_rejects_multiple_candidates(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--baseline-summary",
            str(tmp_path / "baseline.csv"),
            "--candidate-summary",
            str(tmp_path / "candidate.csv"),
            "--candidate-variants",
            "current_default,current_default",
            "--comparison-profile",
            "quality-workflow-scanner-thinning-v1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "exactly one candidate variant" in result.stderr
    assert "Traceback" not in result.stderr


def test_reference_like_policy_aggregate_cli_rejects_non_default_variants(
    tmp_path: Path,
) -> None:
    for option, value, expected in (
        ("--baseline-variant", "boundary_aware_voter_v1", "--baseline-variant"),
        ("--candidate-variant", "boundary_aware_voter_v1", "candidate variant"),
    ):
        result = subprocess.run(
            [
                sys.executable,
                str(GATE_SCRIPT),
                "--baseline-summary",
                str(tmp_path / "baseline.csv"),
                "--candidate-summary",
                str(tmp_path / "candidate.csv"),
                "--candidate-variant",
                "current_default",
                option,
                value,
                "--comparison-profile",
                "quality-workflow-scanner-thinning-v1",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        assert result.returncode == 2
        assert expected in result.stderr
        assert "Traceback" not in result.stderr


def test_scanner_policy_aggregate_cli_rejects_mismatched_evidence(tmp_path: Path) -> None:
    baseline_summary, candidate_summary = _write_policy_inputs(tmp_path)
    output_json = tmp_path / "must-not-exist.json"
    result = subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--baseline-summary",
            str(candidate_summary),
            "--candidate-summary",
            str(candidate_summary),
            "--candidate-variant",
            "current_default",
            "--comparison-profile",
            "scanner-thinning-policy-v1",
            "--baseline-metrics",
            str(baseline_summary.with_name("metrics.json")),
            "--candidate-metrics",
            str(candidate_summary.with_name("metrics.json")),
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "baseline" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output_json.exists()


def test_scanner_policy_aggregate_cli_rejects_multiple_candidates(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--baseline-summary",
            str(tmp_path / "baseline.csv"),
            "--candidate-summary",
            str(tmp_path / "candidate.csv"),
            "--candidate-variants",
            "current_default,current_default",
            "--comparison-profile",
            "scanner-thinning-policy-v1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "exactly one candidate variant" in result.stderr
    assert "Traceback" not in result.stderr


def test_scanner_policy_aggregate_cli_rejects_non_default_variant(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--baseline-summary",
            str(tmp_path / "baseline.csv"),
            "--candidate-summary",
            str(tmp_path / "candidate.csv"),
            "--candidate-variant",
            "boundary_aware_voter_v1",
            "--comparison-profile",
            "scanner-thinning-policy-v1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "candidate variant current_default" in result.stderr
    assert "Traceback" not in result.stderr


def test_legacy_aggregate_cli_keeps_multiple_candidate_flow(tmp_path: Path) -> None:
    output_json = tmp_path / "promotion_gate.json"
    result = subprocess.run(
        [
            sys.executable,
            str(GATE_SCRIPT),
            "--baseline-summary",
            str(FIXTURE),
            "--candidate-summary",
            str(FIXTURE),
            "--candidate-variants",
            "current_default,boundary_aware_voter_v1",
            "--output-json",
            str(output_json),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(output_json.read_text(encoding="utf-8"))["config"]["candidate_variants"] == [
        "current_default",
        "boundary_aware_voter_v1",
    ]
