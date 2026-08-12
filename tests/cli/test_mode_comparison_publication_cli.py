from __future__ import annotations

from pathlib import Path

import pytest

from pyosv.cli import mode_comparison_publication
from pyosv.evaluation import publication_manifest_io
from pyosv.evaluation.mode_comparison_publication import v1_bundle


def test_parser_has_single_publication_contract() -> None:
    args = mode_comparison_publication.build_parser().parse_args(["--output-dir", "output"])
    assert not hasattr(args, "publication_contract")


def test_parser_rejects_removed_contract_option() -> None:
    with pytest.raises(SystemExit):
        mode_comparison_publication.build_parser().parse_args(
            ["--publication-contract", "v1", "--output-dir", "output"]
        )


def test_normal_generation_requires_all_source_arguments(tmp_path: Path) -> None:
    assert mode_comparison_publication.main(["--output-dir", str(tmp_path / "out")]) == 1


def test_generation_routes_code_controls_and_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "publication"
    lock_file = tmp_path / "uv.lock"
    code = {"repository": "ozw4/pyosv", "git_commit": "a" * 40, "dirty": True}
    controls = {name: "value" for name in mode_comparison_publication._ENVIRONMENT_CONTROL_NAMES}
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(mode_comparison_publication, "_collect_code_identity", lambda: code)
    monkeypatch.setattr(
        mode_comparison_publication, "_collect_environment_controls", lambda: controls
    )
    monkeypatch.setattr(
        v1_bundle,
        "generate_publication_bundle_v1",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    result = mode_comparison_publication.main(
        [
            "--pretty",
            "--synthetic-bundle",
            str(tmp_path / "synthetic"),
            "--f3-bundle",
            str(tmp_path / "f3"),
            "--f3-data-root",
            str(tmp_path / "data"),
            "--environment-lock",
            str(lock_file),
            "--output-dir",
            str(output),
        ]
    )

    assert result == 0
    assert calls == [
        (
            (tmp_path / "synthetic", tmp_path / "f3", tmp_path / "data", output),
            {
                "environment_lock": lock_file,
                "code": code,
                "environment_controls": controls,
                "pretty": True,
            },
        )
    ]


def test_validate_only_routes_only_to_directory_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "publication"
    calls: list[Path] = []

    def forbidden() -> dict[str, object]:
        raise AssertionError("validate-only collected generation identity")

    monkeypatch.setattr(mode_comparison_publication, "_collect_code_identity", forbidden)
    monkeypatch.setattr(mode_comparison_publication, "_collect_environment_controls", forbidden)
    monkeypatch.setattr(
        publication_manifest_io,
        "validate_publication_directory",
        lambda root: calls.append(Path(root)) or {},
    )
    monkeypatch.setattr(
        v1_bundle,
        "generate_publication_bundle_v1",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("validate-only attempted generation")
        ),
    )

    assert mode_comparison_publication.main(["--validate-only", "--output-dir", str(output)]) == 0
    assert calls == [output]


def test_generation_requires_environment_lock(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        mode_comparison_publication.main(
            [
                "--synthetic-bundle",
                str(tmp_path / "synthetic"),
                "--f3-bundle",
                str(tmp_path / "f3"),
                "--f3-data-root",
                str(tmp_path / "data"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
        == 1
    )
    assert "generation requires --environment-lock" in capsys.readouterr().err


def test_git_failure_is_reported_by_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import subprocess

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )
    assert (
        mode_comparison_publication.main(
            [
                "--synthetic-bundle",
                str(tmp_path / "synthetic"),
                "--f3-bundle",
                str(tmp_path / "f3"),
                "--f3-data-root",
                str(tmp_path / "data"),
                "--environment-lock",
                str(tmp_path / "uv.lock"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
        == 1
    )


def test_missing_environment_control_fails_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        mode_comparison_publication,
        "_collect_code_identity",
        lambda: {"repository": "ozw4/pyosv", "git_commit": "a" * 40, "dirty": False},
    )
    for name in mode_comparison_publication._ENVIRONMENT_CONTROL_NAMES:
        monkeypatch.setenv(name, "value")
    monkeypatch.delenv("PYOSV_ACCEL")

    assert (
        mode_comparison_publication.main(
            [
                "--synthetic-bundle",
                str(tmp_path / "synthetic"),
                "--f3-bundle",
                str(tmp_path / "f3"),
                "--f3-data-root",
                str(tmp_path / "data"),
                "--environment-lock",
                str(tmp_path / "uv.lock"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
        == 1
    )
