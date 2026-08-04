"""Standard-library helpers for publication artifact validation tests.

The helpers deliberately do not import matplotlib (or any production writer).
They are used to emulate a party that can rewrite ``completion.json`` after
changing an artifact, so the tests exercise semantic validation rather than
only the byte-level completion hash.
"""

from __future__ import annotations

import csv
import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any, Iterable, Mapping


def read_csv_rows(path: Path) -> tuple[tuple[str, ...], list[dict[str, str]]]:
    """Read one test CSV while retaining its deterministic header order."""

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise AssertionError(f"test CSV has no header: {path}")
        return tuple(reader.fieldnames), [dict(row) for row in reader]


def write_csv_rows(
    path: Path,
    header: Iterable[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Rewrite one CSV using the publication writer's newline convention."""

    fieldnames = tuple(header)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: _csv_value(row.get(name)) for name in fieldnames} for row in rows)


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
        + "\n",
        encoding="utf-8",
    )


def rewrite_completion(root: Path) -> None:
    """Refresh completion hashes after a deliberate semantic mutation."""

    completion_path = root / "completion.json"
    completion = read_json(completion_path)
    if not isinstance(completion, dict):
        raise AssertionError("test completion must be an object")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name != "completion.json"
    )
    records = []
    for path in files:
        payload = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    completion["required_files"] = [record["path"] for record in records]
    completion["files"] = records
    write_json(completion_path, completion)


def assert_completion_matches(root: Path) -> None:
    """Assert only the byte-level completion contract for a test artifact."""

    completion = read_json(root / "completion.json")
    if not isinstance(completion, dict):
        raise AssertionError("test completion must be an object")
    expected = completion.get("files")
    if not isinstance(expected, list):
        raise AssertionError("test completion must list files")
    actual_paths = []
    for record in expected:
        if not isinstance(record, dict):
            raise AssertionError("test completion record must be an object")
        path = root / str(record["path"])
        payload = path.read_bytes()
        assert record["size"] == len(payload)
        assert record["sha256"] == hashlib.sha256(payload).hexdigest()
        actual_paths.append(record["path"])
    assert completion.get("required_files") == actual_paths


def write_png(path: Path, *, width: int, height: int) -> None:
    """Write a small valid RGB PNG with only standard-library primitives."""

    if width <= 0 or height <= 0:
        raise AssertionError("test PNG dimensions must be positive")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    # One filter byte followed by width RGB samples for each scan line.
    rows = b"".join(b"\x00" + (b"\x00\x00\x00" * width) for _ in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(payload)
