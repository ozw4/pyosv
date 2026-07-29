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


@pytest.mark.parametrize(
    ("option", "expected"),
    (
        (None, None),
        ("--skinner-boundary-fallback", True),
        ("--no-skinner-boundary-fallback", False),
    ),
)
def test_boundary_fallback_cli_is_tri_state(
    option: str | None,
    expected: bool | None,
    tmp_path: Path,
) -> None:
    argv = ["--output-dir", str(tmp_path)]
    if option is not None:
        argv.append(option)

    args = synthetic_quality.build_parser().parse_args(argv)

    assert args.skinner_boundary_fallback is expected


def test_boundary_fallback_cli_rejects_positive_and_negative_options(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        synthetic_quality.build_parser().parse_args(
            [
                "--output-dir",
                str(tmp_path),
                "--skinner-boundary-fallback",
                "--no-skinner-boundary-fallback",
            ]
        )


def test_reskin_policy_cli_uses_core_choices(tmp_path: Path) -> None:
    parser = synthetic_quality.build_parser()

    default = parser.parse_args(["--output-dir", str(tmp_path)])
    dense = parser.parse_args(
        [
            "--output-dir",
            str(tmp_path),
            "--skinner-reskin-policy",
            "reference_dense_v1",
        ]
    )

    assert default.skinner_reskin_policy == "existing_cells_v1"
    assert dense.skinner_reskin_policy == "reference_dense_v1"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--output-dir",
                str(tmp_path),
                "--skinner-reskin-policy",
                "unknown",
            ]
        )


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


@pytest.mark.parametrize(
    ("option", "expected_value", "expected_explicit"),
    (
        (None, False, False),
        ("--skinner-boundary-fallback", True, True),
        ("--no-skinner-boundary-fallback", False, True),
    ),
)
def test_main_forwards_boundary_fallback_value_and_explicit_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    option: str | None,
    expected_value: bool,
    expected_explicit: bool,
) -> None:
    captured: dict[str, object] = {}

    def fake_run_example(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(synthetic_quality, "run_example", fake_run_example)
    argv = ["--output-dir", str(tmp_path), "--workflow-mode", "quality"]
    if option is not None:
        argv.append(option)

    assert synthetic_quality.main(argv) == 0

    skinning_config = captured["skinning_config"]
    assert isinstance(skinning_config, synthetic_quality.SyntheticSkinningConfig)
    assert skinning_config.boundary_skinner_fallback is expected_value
    assert captured["skinner_boundary_fallback_explicit"] is expected_explicit
