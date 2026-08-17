#!/usr/bin/env python
"""Check one exported Q-QUAL public snapshot in isolated subprocesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath


SNAPSHOT_SCHEMA = "pyosv.qqual_public_source_snapshot.v1"
SNAPSHOT_FILENAME = "SOURCE_SNAPSHOT.json"
SOURCE_REPOSITORY = "ozw4/pyosv"
PUBLIC_ENTRY_POINTS = (
    "pyosv.qqual3d",
    "pyosv.cli.qqual3d",
    "pyosv.compact_publication_validation",
    "pyosv.cli.validate_compact_publication",
)
PUBLIC_CONSOLE_SCRIPTS = {
    "pyosv-qqual3d": "pyosv.cli.qqual3d:main",
    "pyosv-validate-compact": "pyosv.cli.validate_compact_publication:main",
}
DISALLOWED_TOP_LEVEL = {
    ".devcontainer",
    ".git",
    ".github",
    ".issue_forge",
    "AGENTS.md",
    "benchmarks",
    "evidence",
    "outputs",
    "reference_osv",
    "scripts",
    "vendor",
}
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class SnapshotCheckError(ValueError):
    """Raised when an exported public snapshot violates its contract."""


def _safe_relative(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SnapshotCheckError(f"unsafe snapshot path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise SnapshotCheckError(f"unsafe snapshot path: {value!r}")
    return path


def _read_manifest(root: Path) -> dict[str, object]:
    path = root / SNAPSHOT_FILENAME
    if path.is_symlink() or not path.is_file():
        raise SnapshotCheckError(f"missing regular {SNAPSHOT_FILENAME}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise SnapshotCheckError(f"invalid {SNAPSHOT_FILENAME}: {error}") from error
    if not isinstance(value, dict):
        raise SnapshotCheckError("snapshot manifest must be an object")
    if set(value) != {"schema", "source_repository", "source_commit", "files"}:
        raise SnapshotCheckError("snapshot manifest has an invalid field set")
    if value["schema"] != SNAPSHOT_SCHEMA:
        raise SnapshotCheckError("snapshot manifest schema is invalid")
    if value["source_repository"] != SOURCE_REPOSITORY:
        raise SnapshotCheckError("snapshot source repository is invalid")
    if not isinstance(value["source_commit"], str) or not _COMMIT_PATTERN.fullmatch(
        value["source_commit"]
    ):
        raise SnapshotCheckError("snapshot source commit is invalid")
    if not isinstance(value["files"], list):
        raise SnapshotCheckError("snapshot files must be a list")
    return value


def _check_files(root: Path, manifest: dict[str, object]) -> int:
    records = manifest["files"]
    assert isinstance(records, list)
    listed: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise SnapshotCheckError("snapshot file record is invalid")
        relative = _safe_relative(record["path"])
        relative_name = relative.as_posix()
        listed.append(relative_name)
        if not isinstance(record["size"], int) or isinstance(record["size"], bool):
            raise SnapshotCheckError(f"invalid recorded size: {relative_name}")
        if record["size"] < 0:
            raise SnapshotCheckError(f"invalid recorded size: {relative_name}")
        if not isinstance(record["sha256"], str) or not _SHA256_PATTERN.fullmatch(record["sha256"]):
            raise SnapshotCheckError(f"invalid recorded SHA-256: {relative_name}")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise SnapshotCheckError(f"listed file is not regular: {relative_name}")
        payload = path.read_bytes()
        if len(payload) != record["size"]:
            raise SnapshotCheckError(f"file size differs from manifest: {relative_name}")
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise SnapshotCheckError(f"file SHA-256 differs from manifest: {relative_name}")
    if listed != sorted(listed) or len(listed) != len(set(listed)):
        raise SnapshotCheckError("snapshot files are not uniquely sorted by path")

    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise SnapshotCheckError(f"snapshot contains a symlink: {path.relative_to(root)}")
        if path.is_file() and path.name != SNAPSHOT_FILENAME:
            actual.add(path.relative_to(root).as_posix())
    if actual != set(listed):
        raise SnapshotCheckError(
            f"snapshot contains missing or unlisted files: {sorted(actual ^ set(listed))}"
        )
    return len(listed)


def _check_public_layout(root: Path) -> None:
    present = {path.name for path in root.iterdir()}
    disallowed = sorted(present & DISALLOWED_TOP_LEVEL)
    if disallowed:
        raise SnapshotCheckError(f"disallowed top-level paths: {disallowed}")
    dat_files = [path.relative_to(root).as_posix() for path in root.rglob("*.dat")]
    if dat_files:
        raise SnapshotCheckError(f"snapshot contains DAT files: {dat_files}")
    project_path = root / "pyproject.toml"
    if project_path.is_symlink() or not project_path.is_file():
        raise SnapshotCheckError("public pyproject.toml is missing")
    project = tomllib.loads(project_path.read_text(encoding="utf-8"))
    metadata = project.get("project")
    if not isinstance(metadata, dict):
        raise SnapshotCheckError("public project metadata is missing")
    if metadata.get("name") != "pyosv-qqual-poc":
        raise SnapshotCheckError("public distribution name is invalid")
    if metadata.get("requires-python") != ">=3.10":
        raise SnapshotCheckError("public Python requirement is invalid")
    if metadata.get("scripts") != PUBLIC_CONSOLE_SCRIPTS:
        raise SnapshotCheckError("public console scripts differ from the fixed two-script contract")


_IMPORT_INSPECTION = r'''
import importlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
sys.path.insert(0, str(root / "src"))
for name in sys.argv[2:]:
    importlib.import_module(name)
loaded = {}
for name, module in sys.modules.items():
    if name == "pyosv" or name.startswith("pyosv."):
        filename = getattr(module, "__file__", None)
        if filename is None:
            raise RuntimeError(f"imported module has no __file__: {name}")
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"imported module escaped snapshot: {name} -> {path}")
        loaded[name] = path.relative_to(root).as_posix()
print(json.dumps(loaded, sort_keys=True))
'''


_SMOKE_INSPECTION = r'''
import contextlib
import io
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.dont_write_bytecode = True
sys.path[:0] = [str(root / "src"), str(root)]
import pytest

stdout = io.StringIO()
stderr = io.StringIO()
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    status = pytest.main(["-q", "-p", "no:cacheprovider", str(root / "tests")])
if status:
    sys.stderr.write(stdout.getvalue())
    sys.stderr.write(stderr.getvalue())
    raise SystemExit(status)
loaded = {}
for name, module in sys.modules.items():
    if name == "pyosv" or name.startswith("pyosv."):
        filename = getattr(module, "__file__", None)
        if filename is None:
            raise RuntimeError(f"imported module has no __file__: {name}")
        path = Path(filename).resolve()
        if not path.is_relative_to(root):
            raise RuntimeError(f"imported module escaped snapshot: {name} -> {path}")
        loaded[name] = path.relative_to(root).as_posix()
print(json.dumps({"modules": loaded, "pytest_output": stdout.getvalue()}, sort_keys=True))
'''


def _isolated_python(root: Path, script: str, *arguments: str) -> dict[str, object]:
    environment = os.environ.copy()
    environment.update(
        {
            "NUMBA_DISABLE_JIT": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYOSV_ACCEL": "off",
        }
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(root), *arguments],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "isolated check failed"
        raise SnapshotCheckError(detail)
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise SnapshotCheckError(
            f"isolated check returned invalid JSON: {result.stdout!r}"
        ) from error
    if not isinstance(value, dict):
        raise SnapshotCheckError("isolated check returned a non-object result")
    return value


def check_snapshot(root: Path) -> dict[str, object]:
    """Validate provenance, layout, entry imports, and public smoke tests."""

    root = root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise SnapshotCheckError(f"snapshot root must be a directory: {root}")
    manifest = _read_manifest(root)
    file_count = _check_files(root, manifest)
    _check_public_layout(root)
    imports = _isolated_python(root, _IMPORT_INSPECTION, *PUBLIC_ENTRY_POINTS)
    smoke = _isolated_python(root, _SMOKE_INSPECTION)
    modules = smoke.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise SnapshotCheckError("public smoke tests did not import any pyosv modules")
    _check_files(root, manifest)
    return {
        "file_count": file_count,
        "entry_points": list(PUBLIC_ENTRY_POINTS),
        "entry_module_count": len(imports),
        "smoke_module_count": len(modules),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        result = check_snapshot(arguments.snapshot_root)
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, SnapshotCheckError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
