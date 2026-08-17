#!/usr/bin/env python
"""Export the fixed Q-QUAL public file set from a clean Git commit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath


SOURCE_REPOSITORY = "ozw4/pyosv"
SNAPSHOT_SCHEMA = "pyosv.qqual_public_source_snapshot.v1"
SNAPSHOT_FILENAME = "SOURCE_SNAPSHOT.json"
DEFAULT_ALLOWLIST = Path("public_release/qqual_public_files.txt")
TEMPLATE_PREFIX = PurePosixPath("public_release/template")
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


class SnapshotExportError(ValueError):
    """Raised when a source tree cannot be exported safely."""


def _run_git(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "Git command failed"
        raise SnapshotExportError(detail)
    return result.stdout.strip()


def _validate_allowlist_path(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise SnapshotExportError(f"unsafe allowlist path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SnapshotExportError(f"unsafe allowlist path: {value!r}")
    return path


def load_allowlist(path: Path) -> tuple[PurePosixPath, ...]:
    """Load and validate one-file-per-line source paths."""

    if path.is_symlink() or not path.is_file():
        raise SnapshotExportError(f"allowlist must be a regular non-symlink file: {path}")
    entries: list[PurePosixPath] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw_line:
            raise SnapshotExportError(f"blank allowlist line at {path}:{line_number}")
        if raw_line != raw_line.strip():
            raise SnapshotExportError(f"whitespace in allowlist path at {path}:{line_number}")
        entries.append(_validate_allowlist_path(raw_line))
    if not entries:
        raise SnapshotExportError("allowlist must contain at least one path")
    if len(set(entries)) != len(entries):
        raise SnapshotExportError("allowlist contains duplicate paths")
    return tuple(entries)


def public_path(source_path: PurePosixPath) -> PurePosixPath:
    """Map a source allowlist path to its public repository path."""

    try:
        relative = source_path.relative_to(TEMPLATE_PREFIX)
    except ValueError:
        return source_path
    if not relative.parts:
        raise SnapshotExportError("template directory cannot be an allowlist entry")
    return relative


def _validate_source_file(source_root: Path, relative: PurePosixPath) -> Path:
    current = source_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SnapshotExportError(f"source file must not traverse a symlink: {relative}")
    try:
        mode = current.stat().st_mode
    except FileNotFoundError as error:
        raise SnapshotExportError(f"allowlisted source file is missing: {relative}") from error
    if not stat.S_ISREG(mode):
        raise SnapshotExportError(f"allowlisted source path is not a regular file: {relative}")
    return current


def _source_files(
    source_root: Path,
    entries: tuple[PurePosixPath, ...],
) -> tuple[tuple[PurePosixPath, Path, PurePosixPath], ...]:
    records = tuple(
        (entry, _validate_source_file(source_root, entry), public_path(entry)) for entry in entries
    )
    destinations = [record[2] for record in records]
    if len(set(destinations)) != len(destinations):
        raise SnapshotExportError("allowlist paths collide after public template mapping")
    if PurePosixPath(SNAPSHOT_FILENAME) in destinations:
        raise SnapshotExportError(f"{SNAPSHOT_FILENAME} is generated and must not be allowlisted")
    return tuple(sorted(records, key=lambda record: record[2].as_posix()))


def _snapshot_bytes(source_commit: str, files: list[dict[str, int | str]]) -> bytes:
    manifest = {
        "schema": SNAPSHOT_SCHEMA,
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": source_commit,
        "files": files,
    }
    return (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _regular_files(root: Path) -> set[str]:
    result: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SnapshotExportError(f"export contains a symlink: {path.relative_to(root)}")
        if path.is_file():
            result.add(path.relative_to(root).as_posix())
    return result


def export_snapshot(
    *,
    source_root: Path,
    source_commit: str,
    destination: Path,
    allowlist_path: Path | None = None,
) -> dict[str, object]:
    """Export the allowlisted files and return the generated snapshot manifest."""

    source_root = source_root.resolve()
    destination = destination.absolute()
    if os.path.lexists(destination):
        raise SnapshotExportError(f"destination already exists: {destination}")
    if not _COMMIT_PATTERN.fullmatch(source_commit):
        raise SnapshotExportError("source commit must be a 40-character lowercase Git SHA")
    head = _run_git(source_root, "rev-parse", "HEAD")
    if source_commit != head:
        raise SnapshotExportError(
            f"requested source commit {source_commit} does not match HEAD {head}"
        )
    if _run_git(source_root, "status", "--porcelain", "--untracked-files=all"):
        raise SnapshotExportError("source repository must be clean")

    selected_allowlist = (
        (source_root / DEFAULT_ALLOWLIST) if allowlist_path is None else allowlist_path.absolute()
    )
    entries = load_allowlist(selected_allowlist)
    records = _source_files(source_root, entries)

    destination.mkdir(parents=True)
    file_records: list[dict[str, int | str]] = []
    try:
        for _source_relative, source, output_relative in records:
            output = destination.joinpath(*output_relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = source.read_bytes()
            output.write_bytes(payload)
            copied = output.read_bytes()
            if copied != payload:
                raise SnapshotExportError(f"copied bytes do not match: {output_relative}")
            file_records.append(
                {
                    "path": output_relative.as_posix(),
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        snapshot_path = destination / SNAPSHOT_FILENAME
        snapshot_path.write_bytes(_snapshot_bytes(source_commit, file_records))
        expected = {str(record["path"]) for record in file_records} | {SNAPSHOT_FILENAME}
        actual = _regular_files(destination)
        if actual != expected:
            raise SnapshotExportError(
                f"exported regular file set differs from allowlist: {sorted(actual ^ expected)}"
            )
    except BaseException:
        shutil.rmtree(destination)
        raise
    return json.loads((destination / SNAPSHOT_FILENAME).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        manifest = export_snapshot(
            source_root=arguments.source_root,
            source_commit=arguments.source_commit,
            destination=arguments.destination,
        )
    except (OSError, UnicodeError, SnapshotExportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"exported {len(manifest['files'])} files to {arguments.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
