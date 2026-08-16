from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

import pyosv.cli.qqual3d as cli_module
import pyosv.qqual3d.io as io_module
from pyosv.qqual3d import run_qqual3d


EXPECTED_FILES = {
    "run.json",
    "ft.dat",
    "fv.dat",
    "fvt.dat",
    "skin_mask.dat",
    "skins.json",
}


def _write_input(path: Path, array: np.ndarray) -> bytes:
    payload = np.asarray(array, dtype=">f4").tobytes(order="C")
    path.write_bytes(payload)
    return payload


def _run_cli(input_path: Path, shape: tuple[int, int, int], output: Path, *extra: str) -> int:
    return cli_module.main(
        [
            "--input",
            str(input_path),
            "--shape",
            ",".join(str(size) for size in shape),
            "--output-dir",
            str(output),
            *extra,
        ]
    )


def test_cli_generates_exact_bundle_matching_library_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shape = (3, 4, 5)
    ep = np.zeros(shape, dtype=np.float32)
    input_path = tmp_path / "ep.dat"
    original_bytes = _write_input(input_path, ep)
    original_stat = input_path.stat()
    output = tmp_path / "output"

    assert _run_cli(input_path, shape, output) == 0
    assert capsys.readouterr().out.strip() == str(output)

    assert {path.name for path in output.iterdir()} == EXPECTED_FILES
    expected = run_qqual3d(ep)
    for filename, array in (
        ("ft.dat", expected.ft),
        ("fv.dat", expected.fv),
        ("fvt.dat", expected.fvt),
        ("skin_mask.dat", expected.skin_mask.astype(np.float32)),
    ):
        stored = np.fromfile(output / filename, dtype=">f4").reshape(shape)
        assert stored.dtype == np.dtype(">f4")
        np.testing.assert_array_equal(stored, array)

    manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
    for record in manifest["outputs"]:
        payload = (output / record["filename"]).read_bytes()
        assert record["size"] == len(payload)
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    current_stat = input_path.stat()
    assert input_path.read_bytes() == original_bytes
    assert current_stat.st_mtime_ns == original_stat.st_mtime_ns


def test_cli_rejects_existing_output_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shape = (1, 2, 3)
    input_path = tmp_path / "ep.dat"
    _write_input(input_path, np.zeros(shape, dtype=np.float32))
    output = tmp_path / "output"
    output.mkdir()

    def unexpected_run(ep: np.ndarray) -> object:
        raise AssertionError("Q-QUAL must not run for an existing output")

    monkeypatch.setattr(cli_module, "run_qqual3d", unexpected_run)
    assert _run_cli(input_path, shape, output) == 1
    assert "already exists" in capsys.readouterr().err


def test_cli_rejects_wrong_input_size_without_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "ep.dat"
    input_path.write_bytes(b"wrong")
    output = tmp_path / "output"

    assert _run_cli(input_path, (2, 3, 4), output) == 1
    assert "byte size" in capsys.readouterr().err
    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_cli_write_failure_leaves_no_final_or_temporary_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shape = (1, 2, 3)
    input_path = tmp_path / "ep.dat"
    _write_input(input_path, np.zeros(shape, dtype=np.float32))
    output = tmp_path / "output"

    def fail_write(path: Path, value: np.ndarray) -> None:
        raise OSError("injected failure")

    monkeypatch.setattr(io_module, "_write_dat", fail_write)
    assert _run_cli(input_path, shape, output) == 1
    assert "injected failure" in capsys.readouterr().err
    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".output.tmp-*"))


def test_pretty_changes_only_run_json_formatting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shape = (1, 2, 3)
    input_path = tmp_path / "ep.dat"
    _write_input(input_path, np.zeros(shape, dtype=np.float32))
    compact = tmp_path / "compact"
    pretty = tmp_path / "pretty"
    monkeypatch.setattr(io_module, "_created_at_utc", lambda: "2026-08-16T00:00:00Z")

    assert _run_cli(input_path, shape, compact) == 0
    assert _run_cli(input_path, shape, pretty, "--pretty") == 0
    capsys.readouterr()

    assert json.loads((compact / "run.json").read_text()) == json.loads(
        (pretty / "run.json").read_text()
    )
    assert (compact / "run.json").read_bytes() != (pretty / "run.json").read_bytes()
    for filename in EXPECTED_FILES - {"run.json"}:
        assert (compact / filename).read_bytes() == (pretty / filename).read_bytes()


def test_public_parser_exposes_no_mode_selectors() -> None:
    option_strings = {
        option for action in cli_module.build_parser()._actions for option in action.option_strings
    }
    assert option_strings == {"-h", "--help", "--input", "--shape", "--output-dir", "--pretty"}


def test_example_is_a_thin_cli_main_wrapper() -> None:
    path = Path(__file__).parents[2] / "examples" / "run_qqual3d.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert len(imports) == 1
    assert imports[0].module == "pyosv.cli.qqual3d"
    assert [alias.name for alias in imports[0].names] == ["main"]
    assert any(isinstance(call.func, ast.Name) and call.func.id == "main" for call in calls)
