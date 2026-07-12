from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from tests.evaluation.synthetic_quality.test_boundary_stage_summary import _diagnostic, _report

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "summarize_scanner_boundary_stages.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_script_writes_json_and_markdown(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    output_json = tmp_path / "nested" / "summary.json"
    output_markdown = tmp_path / "nested" / "summary.md"
    metrics.write_text(json.dumps(_report(_diagnostic())), encoding="utf-8")

    result = _run(
        str(metrics),
        "--output-json",
        str(output_json),
        "--output-markdown",
        str(output_markdown),
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert json.loads(output_json.read_text(encoding="utf-8"))["retention_threshold"] == 0.8
    assert output_json.read_text(encoding="utf-8").endswith("\n")
    assert output_markdown.read_text(encoding="utf-8").endswith("\n")


def test_script_prints_markdown_when_no_outputs_are_requested(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text(json.dumps(_report(_diagnostic())), encoding="utf-8")

    result = _run(str(metrics))

    assert result.returncode == 0
    assert result.stdout.startswith("# Scanner Boundary Stage Summary\n")


def test_script_reports_invalid_report_on_stderr(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"format_version": 2}', encoding="utf-8")

    result = _run(str(metrics))

    assert result.returncode != 0
    assert "format_version=1" in result.stderr
