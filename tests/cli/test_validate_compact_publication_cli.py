from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

import pyosv.cli.validate_compact_publication as cli_module
from tests.test_compact_publication_validation import _CONTROLS, _prepare_compact_publication


def test_cli_validates_one_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _prepare_compact_publication(tmp_path / "publication")
    for name in _CONTROLS:
        monkeypatch.delenv(name, raising=False)

    def reject_subprocess(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("validation must not inspect Git or external environment controls")

    monkeypatch.setattr(subprocess, "run", reject_subprocess)

    assert cli_module.main([str(root)]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == str(root)
    assert captured.err == ""


def test_cli_returns_failure_for_invalid_directory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = _prepare_compact_publication(tmp_path / "publication")
    (root / "uv.lock").write_bytes(b"tampered\n")

    assert cli_module.main([str(root)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err


def test_cli_parser_exposes_only_one_directory_argument() -> None:
    parser = cli_module.build_parser()
    positionals = [action.dest for action in parser._actions if not action.option_strings]
    options = {option for action in parser._actions for option in action.option_strings if option}

    assert positionals == ["root"]
    assert options == {"-h", "--help"}


def test_example_is_a_thin_cli_main_wrapper() -> None:
    path = Path(__file__).parents[2] / "examples" / "validate_compact_publication.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert len(imports) == 1
    assert imports[0].module == "pyosv.cli.validate_compact_publication"
    assert [alias.name for alias in imports[0].names] == ["main"]
    assert any(isinstance(call.func, ast.Name) and call.func.id == "main" for call in calls)
