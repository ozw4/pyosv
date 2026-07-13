from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_quality_reports.py"
GATE_SCRIPT = REPO_ROOT / "scripts" / "check_synthetic_quality_promotion_gate.py"
FIXTURE = Path("tests/fixtures/synthetic_quality_refactor/known_49_quality_summary.csv")
HASH_FIXTURE = REPO_ROOT / (
    "tests/fixtures/synthetic_quality_refactor/known_49_promotion_artifact_sha256.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _expected_hashes() -> dict[str, str]:
    return json.loads(HASH_FIXTURE.read_text(encoding="utf-8"))


def _policy_metrics(scanner_thin_mode: str) -> dict[str, object]:
    return {
        "format_version": 1,
        "config": {
            "case_set": "extended",
            "input_mode": "both",
            "workflow_mode": "quality",
            "shape": [49, 49, 49],
            "variants": ["current_default"],
            "variant_preset": "default",
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
            "voting": {"voter_thin_mode": "hybrid_v2"},
            "skinning": {"method": "quality"},
            "truth_metrics": {"buffer_radius": 2.0},
            "scanner_downstream_diagnostics": True,
            "scanner_boundary_stage_diagnostics": True,
        },
    }


def _write_policy_inputs(tmp_path: Path) -> tuple[Path, Path]:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    baseline_dir.mkdir()
    candidate_dir.mkdir()
    baseline_summary = baseline_dir / "summary.csv"
    candidate_summary = candidate_dir / "summary.csv"
    shutil.copyfile(FIXTURE, baseline_summary)
    with FIXTURE.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        assert fieldnames is not None
        candidate_rows = list(reader)
    for row in candidate_rows:
        row["scanner_thin_mode"] = "normal"
    with candidate_summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(candidate_rows)
    (baseline_dir / "metrics.json").write_text(
        json.dumps(_policy_metrics("reference")), encoding="utf-8"
    )
    (candidate_dir / "metrics.json").write_text(
        json.dumps(_policy_metrics("normal")), encoding="utf-8"
    )
    return baseline_summary, candidate_summary


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
