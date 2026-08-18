from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

from scripts import check_qqual_public_snapshot as checker
from scripts import export_qqual_public_snapshot as exporter


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit_all(root: Path, message: str = "fixture") -> str:
    _git(root, "add", "--all")
    _git(root, "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _initialize_repository(root: Path) -> None:
    root.mkdir(parents=True)
    _git(root, "init", "--initial-branch=main")
    _git(root, "config", "user.name", "Public Export Test")
    _git(root, "config", "user.email", "public-export@example.invalid")


@pytest.fixture
def source_repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "source"
    _initialize_repository(root)
    _write(root / "public_release/template/README.md", "public readme\n")
    _write(
        root / "public_release/template/pyproject.toml",
        "[project]\nname = \"fixture\"\nversion = \"0.0.0\"\n",
    )
    _write(root / "src/pyosv/__init__.py", '__version__ = "0.0.0"\n')
    _write(
        root / "public_release/qqual_public_files.txt",
        "public_release/template/README.md\n"
        "public_release/template/pyproject.toml\n"
        "src/pyosv/__init__.py\n",
    )
    return root, _commit_all(root)


def _export(source: Path, commit: str, destination: Path) -> dict[str, object]:
    return exporter.export_snapshot(
        source_root=source,
        source_commit=commit,
        destination=destination,
    )


def test_temp_source_fixture_exports_allowlisted_files(
    source_repository: tuple[Path, str],
    tmp_path: Path,
) -> None:
    source, commit = source_repository
    destination = tmp_path / "public"

    manifest = _export(source, commit, destination)

    assert manifest["schema"] == exporter.SNAPSHOT_SCHEMA
    assert manifest["source_repository"] == "ozw4/pyosv"
    assert manifest["source_commit"] == commit
    assert [record["path"] for record in manifest["files"]] == [
        "README.md",
        "pyproject.toml",
        "src/pyosv/__init__.py",
    ]
    assert not (destination / ".git").exists()
    assert {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    } == {
        "README.md",
        "SOURCE_SNAPSHOT.json",
        "pyproject.toml",
        "src/pyosv/__init__.py",
    }


def test_existing_destination_is_rejected(
    source_repository: tuple[Path, str],
    tmp_path: Path,
) -> None:
    source, commit = source_repository
    destination = tmp_path / "existing"
    destination.mkdir()

    with pytest.raises(exporter.SnapshotExportError, match="already exists"):
        _export(source, commit, destination)


def test_dirty_source_is_rejected(
    source_repository: tuple[Path, str],
    tmp_path: Path,
) -> None:
    source, commit = source_repository
    _write(source / "untracked.txt", "dirty\n")

    with pytest.raises(exporter.SnapshotExportError, match="must be clean"):
        _export(source, commit, tmp_path / "public")


def test_commit_mismatch_is_rejected(
    source_repository: tuple[Path, str],
    tmp_path: Path,
) -> None:
    source, commit = source_repository
    different = ("0" if commit[0] != "0" else "1") + commit[1:]

    with pytest.raises(exporter.SnapshotExportError, match="does not match HEAD"):
        _export(source, different, tmp_path / "public")


def test_symlink_source_is_rejected(
    source_repository: tuple[Path, str],
    tmp_path: Path,
) -> None:
    source, _commit = source_repository
    readme = source / "public_release/template/README.md"
    readme.unlink()
    readme.symlink_to("pyproject.toml")
    commit = _commit_all(source, "symlink")

    with pytest.raises(exporter.SnapshotExportError, match="symlink"):
        _export(source, commit, tmp_path / "public")


@pytest.mark.parametrize("unsafe", ["../secret.txt", "/absolute.txt", "src/../secret.txt"])
def test_unsafe_allowlist_path_is_rejected(
    source_repository: tuple[Path, str],
    tmp_path: Path,
    unsafe: str,
) -> None:
    source, _commit = source_repository
    _write(source / "public_release/qqual_public_files.txt", f"{unsafe}\n")
    commit = _commit_all(source, "unsafe allowlist")

    with pytest.raises(exporter.SnapshotExportError, match="unsafe allowlist path"):
        _export(source, commit, tmp_path / "public")


def test_copied_bytes_sizes_and_hashes_match_source(
    source_repository: tuple[Path, str],
    tmp_path: Path,
) -> None:
    source, commit = source_repository
    destination = tmp_path / "public"

    manifest = _export(source, commit, destination)

    for record in manifest["files"]:
        output = destination / record["path"]
        payload = output.read_bytes()
        assert record["size"] == len(payload)
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
    assert (destination / "README.md").read_bytes() == (
        source / "public_release/template/README.md"
    ).read_bytes()
    assert (destination / "src/pyosv/__init__.py").read_bytes() == (
        source / "src/pyosv/__init__.py"
    ).read_bytes()


def test_source_snapshot_is_deterministic(
    source_repository: tuple[Path, str],
    tmp_path: Path,
) -> None:
    source, commit = source_repository
    first = tmp_path / "first"
    second = tmp_path / "second"

    _export(source, commit, first)
    _export(source, commit, second)

    first_files = {
        path.relative_to(first).as_posix(): path.read_bytes()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(second).as_posix(): path.read_bytes()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files
    snapshot = json.loads((first / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    assert "timestamp" not in snapshot
    assert snapshot["files"] == sorted(snapshot["files"], key=lambda record: record["path"])
    assert all(record["path"] != "SOURCE_SNAPSHOT.json" for record in snapshot["files"])


def _copy_real_allowlist_to_clean_repository(root: Path) -> str:
    _initialize_repository(root)
    allowlist = REPOSITORY_ROOT / exporter.DEFAULT_ALLOWLIST
    entries = exporter.load_allowlist(allowlist)
    for relative in entries:
        source = REPOSITORY_ROOT.joinpath(*relative.parts)
        destination = root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    destination_allowlist = root / exporter.DEFAULT_ALLOWLIST
    destination_allowlist.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(allowlist, destination_allowlist)
    return _commit_all(root, "public source fixture")


def test_public_dependency_closure_smoke_layout_and_console_scripts(tmp_path: Path) -> None:
    source = tmp_path / "clean-source"
    commit = _copy_real_allowlist_to_clean_repository(source)
    destination = tmp_path / "public"
    _export(source, commit, destination)

    result = checker.check_snapshot(destination)

    assert result["entry_points"] == list(checker.PUBLIC_ENTRY_POINTS)
    assert result["entry_module_count"] > len(checker.PUBLIC_ENTRY_POINTS)
    assert result["smoke_module_count"] >= result["entry_module_count"]
    assert not ({path.name for path in destination.iterdir()} & checker.DISALLOWED_TOP_LEVEL)
    assert not list(destination.rglob("*.dat"))
    project = tomllib.loads((destination / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"] == checker.PUBLIC_CONSOLE_SCRIPTS
    assert len(project["project"]["scripts"]) == 2
    facade = destination / "src/pyosv/evaluation/synthetic_quality/__init__.py"
    assert (
        facade.read_bytes()
        == (
            REPOSITORY_ROOT
            / "public_release/template/src/pyosv/evaluation/synthetic_quality/__init__.py"
        ).read_bytes()
    )
    assert (
        facade.read_bytes()
        != (REPOSITORY_ROOT / "src/pyosv/evaluation/synthetic_quality/__init__.py").read_bytes()
    )
    manifest = json.loads((destination / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) == len(
        exporter.load_allowlist(REPOSITORY_ROOT / exporter.DEFAULT_ALLOWLIST)
    )
    assert not os.path.lexists(destination / ".git")
