from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"
EXAMPLE_SCRIPTS = tuple(sorted(EXAMPLES_DIR.glob("*.py")))
EXAMPLE_MODULES = tuple(script.stem for script in EXAMPLE_SCRIPTS)
REFERENCE_EXAMPLE_MODULES = ("run_2d_f3d2d", "run_2d_reference")
REPOSITORY_ROOT_OUTPUTS = ("fv_py.dat", "fvt_py.dat")
SYNTHETIC_OUTPUTS = (
    "g_py.dat",
    "ft_py.dat",
    "pt_py.dat",
    "tt_py.dat",
    "fv_py.dat",
    "vp_py.dat",
    "vt_py.dat",
    "fvt_py.dat",
)


def _run_example(script_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(script_path.relative_to(REPO_ROOT)), *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result


def _import_example_module(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> object:
    monkeypatch.syspath_prepend(str(EXAMPLES_DIR))
    sys.modules.pop(module_name, None)
    importlib.invalidate_caches()
    return importlib.import_module(module_name)


@pytest.mark.parametrize("script_path", EXAMPLE_SCRIPTS, ids=lambda path: path.name)
def test_example_help_exits_successfully(script_path: Path) -> None:
    result = _run_example(script_path, "--help")

    assert result.returncode == 0
    assert "usage:" in result.stdout
    expected_output_arg = (
        "--output-json" if script_path.name.startswith("report_") else "--output-dir"
    )
    assert expected_output_arg in result.stdout


def test_run_2d_f3d2d_help_mentions_dataset() -> None:
    result = _run_example(EXAMPLES_DIR / "run_2d_f3d2d.py", "--help")

    assert result.returncode == 0
    assert "f3d2d" in result.stdout
    assert "--output-dir" in result.stdout


def test_run_2d_reference_help_mentions_supported_datasets() -> None:
    result = _run_example(EXAMPLES_DIR / "run_2d_reference.py", "--help")

    assert result.returncode == 0
    assert "--dataset" in result.stdout
    assert "f3d2d" in result.stdout
    assert "campos" in result.stdout
    assert "--ru" in result.stdout
    assert "--path-smoothing" in result.stdout


def test_run_2d_reference_unknown_dataset_exits_clearly(tmp_path: Path) -> None:
    result = _run_example(
        EXAMPLES_DIR / "run_2d_reference.py",
        "--dataset",
        "missing",
        "--output-dir",
        str(tmp_path),
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
    assert "missing" in result.stderr


@pytest.mark.parametrize("module_name", EXAMPLE_MODULES)
def test_example_modules_import_without_running_workflow(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_example_module(module_name, monkeypatch)

    assert callable(module.build_parser)
    assert callable(module.main)
    assert callable(module.run_example)


@pytest.mark.parametrize("module_name", REFERENCE_EXAMPLE_MODULES)
def test_reference_example_output_dir_is_required(
    module_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_example_module(module_name, monkeypatch)
    parser = module.build_parser()
    output_dir_actions = [
        action for action in parser._actions if "--output-dir" in action.option_strings
    ]

    assert len(output_dir_actions) == 1
    assert output_dir_actions[0].required is True


def test_synthetic_scan_vote_example_output_dir_is_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _import_example_module("run_2d_synthetic_scan_vote", monkeypatch)
    parser = module.build_parser()
    output_dir_actions = [
        action for action in parser._actions if "--output-dir" in action.option_strings
    ]

    assert len(output_dir_actions) == 1
    assert output_dir_actions[0].required is False


@pytest.mark.parametrize(
    ("script_name", "args"),
    (
        ("run_2d_f3d2d.py", ()),
        ("run_2d_reference.py", ("--dataset", "f3d2d")),
    ),
)
def test_example_missing_output_dir_does_not_write_to_repository_root(
    script_name: str,
    args: tuple[str, ...],
) -> None:
    output_paths = [REPO_ROOT / name for name in REPOSITORY_ROOT_OUTPUTS]
    before = {path: _path_signature(path) for path in output_paths}

    result = _run_example(EXAMPLES_DIR / script_name, *args)

    after = {path: _path_signature(path) for path in output_paths}
    assert result.returncode != 0
    assert "--output-dir" in result.stderr
    assert after == before


def test_synthetic_scan_vote_without_output_dir_does_not_write_to_repository_root() -> None:
    output_paths = [REPO_ROOT / name for name in SYNTHETIC_OUTPUTS]
    before = {path: _path_signature(path) for path in output_paths}

    result = _run_example(EXAMPLES_DIR / "run_2d_synthetic_scan_vote.py")

    after = {path: _path_signature(path) for path in output_paths}
    assert result.returncode == 0
    assert "fv_nonzero=" in result.stdout
    assert after == before


def test_3d_synthetic_scan_vote_without_output_dir_does_not_write_to_repository_root() -> None:
    output_paths = [REPO_ROOT / name for name in SYNTHETIC_OUTPUTS]
    before = {path: _path_signature(path) for path in output_paths}

    result = _run_example(EXAMPLES_DIR / "run_3d_synthetic_scan_vote.py")

    after = {path: _path_signature(path) for path in output_paths}
    assert result.returncode == 0
    assert "fv_nonzero=" in result.stdout
    assert "fvt_max=" in result.stdout
    assert after == before


def _path_signature(path: Path) -> tuple[bool, int | None, int | None]:
    if not path.exists():
        return (False, None, None)
    stat = path.stat()
    return (True, stat.st_size, stat.st_mtime_ns)
