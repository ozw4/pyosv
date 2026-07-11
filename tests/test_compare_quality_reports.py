from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pyosv.evaluation.promotion import build_promotion_report, compare_reports
from pyosv.evaluation.promotion.markdown import comparison_markdown, promotion_markdown


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "compare_quality_reports.py"
GATE_SCRIPT = REPO_ROOT / "scripts" / "check_synthetic_quality_promotion_gate.py"
FIXTURE = REPO_ROOT / "tests/fixtures/synthetic_quality_refactor/known_49_quality_summary.csv"


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
    expected = compare_reports(
        FIXTURE,
        FIXTURE,
        "current_default",
        "boundary_aware_voter_v1",
        "scanner-boundary",
    )
    assert json.loads(output_json.read_text(encoding="utf-8")) == expected
    assert output_markdown.read_text(encoding="utf-8") == comparison_markdown(expected)


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
    expected = build_promotion_report(
        FIXTURE,
        FIXTURE,
        candidate_variants=("boundary_aware_voter_v1",),
    )
    assert json.loads(output_json.read_text(encoding="utf-8")) == expected
    assert output_markdown.read_text(encoding="utf-8") == promotion_markdown(expected)
