from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from pyosv.cli import synthetic_quality
from pyosv.evaluation.synthetic_quality import build_report, run_case


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "report_3d_synthetic_quality.py"


def _run(*command: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "src"
    return subprocess.run(
        [sys.executable, *command],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_package_cli_help() -> None:
    result = _run("-m", "pyosv.cli.synthetic_quality", "--help")

    assert result.returncode == 0
    assert "--output-dir" in result.stdout
    assert "--scanner-boundary-stage-diagnostics" in result.stdout


def test_example_wrapper_help() -> None:
    result = _run(str(EXAMPLE.relative_to(REPO_ROOT)), "--help")

    assert result.returncode == 0
    assert "--output-dir" in result.stdout


def test_explicit_long_option_accepts_separate_and_equals_forms() -> None:
    has_option = synthetic_quality.main.__globals__["_argv_has_long_option"]

    assert has_option(("--skinner-min-likelihood", "0.2"), "--skinner-min-likelihood")
    assert has_option(("--skinner-min-likelihood=0.2",), "--skinner-min-likelihood")
    assert not has_option(("--other=0.2",), "--skinner-min-likelihood")


def test_domain_report_apis_are_exported_from_package() -> None:
    assert callable(build_report)
    assert callable(run_case)


def test_main_forwards_scanner_boundary_stage_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_example(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(synthetic_quality, "run_example", fake_run_example)

    result = synthetic_quality.main(
        [
            "--output-dir",
            str(tmp_path),
            "--scanner-boundary-stage-diagnostics",
        ]
    )

    assert result == 0
    assert captured["include_scanner_boundary_stage_diagnostics"] is True
