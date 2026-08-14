from __future__ import annotations

import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from pyosv.cli import f3_compact_publication
from pyosv.evaluation import f3_compact_publication as compact_api
from pyosv.evaluation.f3_compact_publication.manifest import build_manifest, write_manifest
from pyosv.evaluation.publication_manifest_io import artifact_file_record

_CONTROLS = {name: "1" for name in f3_compact_publication._ENVIRONMENT_CONTROL_NAMES}
_CODE = {"repository": "ozw4/pyosv", "git_commit": "a" * 40, "dirty": False}


def _generation_arguments(tmp_path: Path, *, pretty: bool = False) -> list[str]:
    arguments = [
        "--f3-bundle",
        str(tmp_path / "source"),
        "--f3-data-root",
        str(tmp_path / "data"),
        "--environment-lock",
        str(tmp_path / "uv.lock"),
        "--output-dir",
        str(tmp_path / "publication"),
    ]
    if pretty:
        arguments.append("--pretty")
    return arguments


def _minimal_bundle(root: Path) -> Path:
    root.mkdir()
    (root / "experiment.json").write_text("{}\n", encoding="utf-8")
    (root / "uv.lock").write_text("lock-version = 1\n", encoding="utf-8")
    experiment = artifact_file_record(
        root,
        "experiment.json",
        tier="primary",
        role="resolved_experiment",
    )
    lock = artifact_file_record(root, "uv.lock", tier="primary", role="environment_lock")
    manifest = build_manifest(
        created_at_utc="2026-08-14T00:00:00Z",
        code=_CODE,
        environment={
            "python": "3.10.14",
            "lock_file": "uv.lock",
            "lock_sha256": lock["sha256"],
            "controls": _CONTROLS,
        },
        source={"f3_completion_sha256": "b" * 64},
        dataset={
            "dataset_id": "compact-cli-fixture",
            "shape": [1, 1, 1],
            "storage_dtype": ">f4",
            "files": [
                {"role": role, "filename": filename, "size": 4, "sha256": digest * 64}
                for role, filename, digest in (
                    ("input", "ep.dat", "1"),
                    ("reference_fault_likelihood", "fl.dat", "2"),
                    ("reference_fault_votes", "fv.dat", "3"),
                    ("reference_thinned_fault_votes", "fvt.dat", "4"),
                    ("seismic_amplitude", "xs.dat", "5"),
                )
            ],
        },
        experiment={
            "config_file": "experiment.json",
            "config_sha256": experiment["sha256"],
        },
        semantics={
            "evaluation": "f3_public_reference_agreement",
            "public_reference_is_geological_truth": False,
            "evaluation_units": 1,
            "displayed_condition": "Q-QUAL",
            "stage_order": ["ft", "fv", "fvt"],
        },
        artifacts=[experiment, lock],
    )
    write_manifest(root, manifest)
    return root


def test_parser_accepts_fixed_generation_arguments(tmp_path: Path) -> None:
    args = f3_compact_publication.build_parser().parse_args(
        _generation_arguments(tmp_path, pretty=True)
    )

    assert args.f3_bundle == tmp_path / "source"
    assert args.f3_data_root == tmp_path / "data"
    assert args.environment_lock == tmp_path / "uv.lock"
    assert args.output_dir == tmp_path / "publication"
    assert args.pretty is True
    assert args.validate_only is False


def test_missing_generation_arguments_fail_with_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = f3_compact_publication.main(["--output-dir", str(tmp_path / "publication")])

    captured = capsys.readouterr()
    assert result == 1
    assert captured.out == ""
    assert captured.err.startswith("error: normal generation requires ")


def test_generation_routes_identity_controls_and_pretty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(f3_compact_publication, "_collect_code_identity", lambda: _CODE)
    monkeypatch.setattr(f3_compact_publication, "_collect_environment_controls", lambda: _CONTROLS)
    monkeypatch.setattr(
        compact_api,
        "generate_f3_compact_publication_bundle",
        lambda *args, **kwargs: calls.append((args, kwargs)) or Path(args[2]),
    )

    result = f3_compact_publication.main(_generation_arguments(tmp_path, pretty=True))

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == f"{tmp_path / 'publication'}\n"
    assert captured.err == ""
    assert calls == [
        (
            (tmp_path / "source", tmp_path / "data", tmp_path / "publication"),
            {
                "environment_lock": tmp_path / "uv.lock",
                "code": _CODE,
                "environment_controls": _CONTROLS,
                "pretty": True,
            },
        )
    ]


def test_validate_only_succeeds_without_generation_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "publication"
    calls: list[Path] = []

    def forbidden() -> dict[str, object]:
        raise AssertionError("validate-only collected generation identity")

    monkeypatch.setattr(f3_compact_publication, "_collect_code_identity", forbidden)
    monkeypatch.setattr(f3_compact_publication, "_collect_environment_controls", forbidden)
    monkeypatch.setattr(
        compact_api,
        "validate_f3_compact_publication_bundle",
        lambda root: calls.append(Path(root)) or {},
    )

    result = f3_compact_publication.main(["--validate-only", "--output-dir", str(output)])

    captured = capsys.readouterr()
    assert result == 0
    assert calls == [output]
    assert captured.out == f"{output}\n"
    assert captured.err == ""


@pytest.mark.parametrize(
    "argument,value",
    [
        ("--f3-bundle", "source"),
        ("--f3-data-root", "data"),
        ("--environment-lock", "uv.lock"),
        ("--pretty", None),
    ],
)
def test_validate_only_rejects_generation_arguments(
    tmp_path: Path,
    argument: str,
    value: str | None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = ["--validate-only", "--output-dir", str(tmp_path / "publication"), argument]
    if value is not None:
        arguments.append(value)

    assert f3_compact_publication.main(arguments) == 1
    assert capsys.readouterr().err.startswith("error: --validate-only cannot be combined with ")


def test_missing_environment_control_fails_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(f3_compact_publication, "_collect_code_identity", lambda: _CODE)
    for name in f3_compact_publication._ENVIRONMENT_CONTROL_NAMES:
        monkeypatch.setenv(name, "1")
    monkeypatch.delenv("PYOSV_ACCEL")

    assert f3_compact_publication.main(_generation_arguments(tmp_path)) == 1
    assert "error: compact generation requires environment controls: PYOSV_ACCEL" in (
        capsys.readouterr().err
    )


def test_existing_output_fails_before_source_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "publication").mkdir()
    (tmp_path / "uv.lock").write_text("lock-version = 1\n", encoding="utf-8")
    monkeypatch.setattr(f3_compact_publication, "_collect_code_identity", lambda: _CODE)
    monkeypatch.setattr(f3_compact_publication, "_collect_environment_controls", lambda: _CONTROLS)

    assert f3_compact_publication.main(_generation_arguments(tmp_path)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: compact publication output already exists:")


def test_validate_only_imports_no_source_visualization_or_runtime_stack(tmp_path: Path) -> None:
    output = _minimal_bundle(tmp_path / "publication")
    source_root = Path(__file__).resolve().parents[2] / "src"
    script = r"""
import builtins
import os
import subprocess
import sys

original_import = builtins.__import__
forbidden_prefixes = (
    "matplotlib",
    "numba",
    "pyosv.viz",
    "pyosv.evaluation.f3_compact_publication.source",
    "pyosv.evaluation.f3_compact_publication.figures",
    "pyosv.evaluation.f3d_mode_comparison",
    "pyosv.evaluation.mode_comparison_publication",
)

def guarded_import(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes):
        raise AssertionError("validate-only imported forbidden module: " + name)
    return original_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from pyosv.cli.f3_compact_publication import main

def forbidden_command(*args, **kwargs):
    raise AssertionError("validate-only inspected Git")

original_environment = os.environ
control_names = {
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_DISABLE_JIT",
    "NUMBA_NUM_THREADS",
    "PYOSV_ACCEL",
}

class ForbiddenEnvironment:
    def get(self, key, *args, **kwargs):
        if key in control_names:
            raise AssertionError("validate-only read environment controls")
        return original_environment.get(key, *args, **kwargs)

    def __getitem__(self, key):
        if key in control_names:
            raise AssertionError("validate-only read environment controls")
        return original_environment[key]

subprocess.run = forbidden_command
os.environ = ForbiddenEnvironment()
assert main(["--validate-only", "--output-dir", sys.argv[1]]) == 0
assert not any(
    name == prefix or name.startswith(prefix + ".")
    for name in sys.modules
    for prefix in forbidden_prefixes
)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source_root)

    result = subprocess.run(
        [sys.executable, "-c", script, str(output)],
        cwd=source_root.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_thin_example_calls_cli_main(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[None] = []

    def fake_main() -> int:
        calls.append(None)
        return 17

    monkeypatch.setattr(f3_compact_publication, "main", fake_main)
    example = Path(__file__).resolve().parents[2] / "examples/report_3d_f3_compact_publication.py"

    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(example), run_name="__main__")

    assert error.value.code == 17
    assert calls == [None]
