"""Canonical parsing and cross-file validation for F3 skin artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from pyosv.synthetic_metrics import skin_topology_metrics

_ROOT_FIELDS = {"format_version", "skinning_enabled", "skin_count", "skins"}
_SKIN_FIELDS = {"skin_index", "cell_count", "cells"}
_CELL_FIELDS = {"x1", "x2", "x3", "i1", "i2", "i3", "fl", "fp", "ft"}
_INDEX_FIELDS = ("i1", "i2", "i3")
_SCALAR_FIELDS = ("x1", "x2", "x3", "fl", "fp", "ft")
_TOPOLOGY_FIELDS = (
    "skin_count",
    "cell_count",
    "unique_cell_count",
    "duplicate_cell_count",
    "largest_skin_size",
    "largest_skin_fraction",
    "small_skin_size",
    "small_skin_count",
    "small_skin_cell_count",
    "small_skin_cell_fraction",
)
_INTEGER_TOPOLOGY_FIELDS = frozenset(
    {
        "skin_count",
        "cell_count",
        "unique_cell_count",
        "duplicate_cell_count",
        "largest_skin_size",
        "small_skin_size",
        "small_skin_count",
        "small_skin_cell_count",
    }
)
_DAT_DTYPE = np.dtype(">f4")
_MASK_SLAB_VOXELS = 1_000_000
_INDEX_CHUNK_SIZE = 100_000


class SkinArtifactValidationError(ValueError):
    """Raised when canonical F3 skin artifacts disagree."""


@dataclass(frozen=True, slots=True)
class SkinCellRecord:
    """Immutable cell values retained exactly from ``skins.json``."""

    x1: float
    x2: float
    x3: float
    i1: int
    i2: int
    i3: int
    fl: float
    fp: float
    ft: float


@dataclass(frozen=True, slots=True)
class ParsedSkinArtifacts:
    """Canonical skins with duplicate voxel occurrences preserved."""

    skins: tuple[tuple[SkinCellRecord, ...], ...]

    @property
    def cell_count(self) -> int:
        return sum(len(skin) for skin in self.skins)

    @property
    def unique_indices(self) -> frozenset[tuple[int, int, int]]:
        return frozenset((cell.i1, cell.i2, cell.i3) for skin in self.skins for cell in skin)

    @property
    def duplicate_cell_count(self) -> int:
        return self.cell_count - len(self.unique_indices)


def parse_skins_json(
    path: str | Path,
    shape: tuple[int, int, int],
) -> ParsedSkinArtifacts:
    """Parse one canonical ``skins.json`` and validate its low-cost schema."""

    volume_shape = _volume_shape(shape)
    artifact_path = Path(path)
    payload = _read_json_object(artifact_path, "skins.json")
    if set(payload) != _ROOT_FIELDS:
        raise SkinArtifactValidationError("skins.json root field set mismatch")
    if payload["format_version"] != 1 or isinstance(payload["format_version"], bool):
        raise SkinArtifactValidationError("skins.json format_version must be 1")
    if payload["skinning_enabled"] is not True:
        raise SkinArtifactValidationError("skins.json skinning_enabled must be true")
    skins_value = payload["skins"]
    if not isinstance(skins_value, list):
        raise SkinArtifactValidationError("skins.json skins must be an array")
    skin_count = _integer(payload["skin_count"], "skins.json skin_count")
    if skin_count != len(skins_value):
        raise SkinArtifactValidationError("skins.json skin_count mismatch")

    parsed_skins: list[tuple[SkinCellRecord, ...]] = []
    for expected_skin_index, skin_value in enumerate(skins_value):
        if not isinstance(skin_value, dict) or set(skin_value) != _SKIN_FIELDS:
            raise SkinArtifactValidationError("skins.json skin field set mismatch")
        skin_index = _integer(skin_value["skin_index"], "skins.json skin_index")
        if skin_index != expected_skin_index:
            raise SkinArtifactValidationError("skins.json skin_index mismatch")
        cells_value = skin_value["cells"]
        if not isinstance(cells_value, list):
            raise SkinArtifactValidationError("skins.json cells must be an array")
        cell_count = _integer(skin_value["cell_count"], "skins.json cell_count")
        if cell_count != len(cells_value):
            raise SkinArtifactValidationError("skins.json cell_count mismatch")

        cells: list[SkinCellRecord] = []
        previous_key: tuple[int, int, int] | None = None
        for cell_value in cells_value:
            if not isinstance(cell_value, dict) or set(cell_value) != _CELL_FIELDS:
                raise SkinArtifactValidationError("skins.json cell field set mismatch")
            indices = tuple(
                _integer(cell_value[name], f"skins.json cell {name}") for name in _INDEX_FIELDS
            )
            _validate_bounds(indices, volume_shape)
            scalars = {
                name: _finite_scalar(cell_value[name], f"skins.json cell {name}")
                for name in _SCALAR_FIELDS
            }
            canonical_key = (indices[2], indices[1], indices[0])
            if previous_key is not None and canonical_key < previous_key:
                raise SkinArtifactValidationError("skins.json cell order is not canonical")
            previous_key = canonical_key
            cells.append(
                SkinCellRecord(
                    x1=scalars["x1"],
                    x2=scalars["x2"],
                    x3=scalars["x3"],
                    i1=indices[0],
                    i2=indices[1],
                    i3=indices[2],
                    fl=scalars["fl"],
                    fp=scalars["fp"],
                    ft=scalars["ft"],
                )
            )
        parsed_skins.append(tuple(cells))
    return ParsedSkinArtifacts(tuple(parsed_skins))


def validate_skin_artifact_semantics(
    stage_path: str | Path,
    shape: tuple[int, int, int],
    *,
    small_skin_size: int,
    parsed: ParsedSkinArtifacts | None = None,
) -> ParsedSkinArtifacts:
    """Cross-check ``skins.json``, ``skin_mask.dat``, and report topology."""

    volume_shape = _volume_shape(shape)
    root = Path(stage_path)
    skin_data = parsed if parsed is not None else parse_skins_json(root / "skins.json", shape)
    if not isinstance(skin_data, ParsedSkinArtifacts):
        raise TypeError("parsed must be ParsedSkinArtifacts or None")
    if isinstance(small_skin_size, bool) or not isinstance(small_skin_size, int):
        raise SkinArtifactValidationError("small_skin_size must be an integer")
    if small_skin_size < 0:
        raise SkinArtifactValidationError("small_skin_size must be non-negative")

    _validate_skin_mask(root / "skin_mask.dat", volume_shape, skin_data.unique_indices)
    topology = skin_topology_metrics(
        skin_data.skins,
        volume_shape,
        small_skin_size=small_skin_size,
    )
    report = _read_json_object(root / "report.json", "skinning report")
    stored_topology = report.get("topology")
    if not isinstance(stored_topology, Mapping):
        raise SkinArtifactValidationError("skinning report topology must be an object")
    for name in _TOPOLOGY_FIELDS:
        if name not in stored_topology:
            raise SkinArtifactValidationError(f"skinning report topology missing {name}")
        stored = stored_topology[name]
        if name in _INTEGER_TOPOLOGY_FIELDS:
            stored = _integer(stored, f"skinning report topology {name}")
        else:
            stored = _finite_scalar(stored, f"skinning report topology {name}")
        if stored != topology[name]:
            raise SkinArtifactValidationError(f"skinning report topology mismatch: {name}")
    _validate_final_diagnostic_mapping(report.get("diagnostics"), topology)
    return skin_data


def _validate_skin_mask(
    path: Path,
    shape: tuple[int, int, int],
    unique_indices: frozenset[tuple[int, int, int]],
) -> None:
    if not path.is_file() or path.is_symlink():
        raise SkinArtifactValidationError("skin_mask.dat must be a regular non-symlink file")
    expected_bytes = math.prod(shape) * _DAT_DTYPE.itemsize
    try:
        if path.stat().st_size != expected_bytes:
            raise SkinArtifactValidationError("skin_mask.dat shape or dtype size mismatch")
        mask = np.memmap(path, dtype=_DAT_DTYPE, mode="r", shape=shape)
    except OSError as error:
        raise SkinArtifactValidationError("skin_mask.dat is unreadable") from error
    try:
        plane_size = shape[1] * shape[2]
        slab_depth = max(1, _MASK_SLAB_VOXELS // plane_size)
        one_count = 0
        for start in range(0, shape[0], slab_depth):
            slab = mask[start : start + slab_depth]
            if not bool(np.isfinite(slab).all()):
                raise SkinArtifactValidationError("skin_mask.dat contains non-finite values")
            if not bool(np.logical_or(slab == 0.0, slab == 1.0).all()):
                raise SkinArtifactValidationError("skin_mask.dat values must be exactly 0.0 or 1.0")
            one_count += int(np.count_nonzero(slab))
        index_iterator = iter(unique_indices)
        while chunk := tuple(islice(index_iterator, _INDEX_CHUNK_SIZE)):
            i1 = np.fromiter((index[0] for index in chunk), dtype=np.intp)
            i2 = np.fromiter((index[1] for index in chunk), dtype=np.intp)
            i3 = np.fromiter((index[2] for index in chunk), dtype=np.intp)
            if not bool((mask[i3, i2, i1] == 1.0).all()):
                raise SkinArtifactValidationError(
                    "skin_mask.dat does not match skins.json voxel mask"
                )
        if one_count != len(unique_indices):
            raise SkinArtifactValidationError("skin_mask.dat does not match skins.json voxel mask")
    finally:
        mapping = getattr(mask, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()


def _validate_final_diagnostic_mapping(
    value: Any,
    topology: Mapping[str, float | int],
) -> None:
    if not isinstance(value, Mapping):
        raise SkinArtifactValidationError("skinning report diagnostics must be an object")
    fallback_used = value.get("fallback_used")
    if not isinstance(fallback_used, bool):
        raise SkinArtifactValidationError("skinning report diagnostics fallback_used must be bool")
    fields = (
        (("fallback_skin_count", "skin_count"), ("fallback_cell_count", "cell_count"))
        if fallback_used
        else (("accepted_skin_count", "skin_count"), ("accepted_cell_count", "cell_count"))
    )
    for diagnostic_name, topology_name in fields:
        diagnostic_value = _integer(
            value.get(diagnostic_name),
            f"skinning report diagnostics {diagnostic_name}",
        )
        if diagnostic_value != topology[topology_name]:
            raise SkinArtifactValidationError(
                f"skinning report diagnostics mismatch: {diagnostic_name}"
            )


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SkinArtifactValidationError(f"{context} must be a regular non-symlink file")
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: _raise_nonfinite(token),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SkinArtifactValidationError(f"{context} is not strict JSON") from error
    if not isinstance(value, dict):
        raise SkinArtifactValidationError(f"{context} must be an object")
    return value


def _raise_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def _volume_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if (
        not isinstance(shape, tuple)
        or len(shape) != 3
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
    ):
        raise TypeError("shape must contain exactly three positive integers")
    return shape


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SkinArtifactValidationError(f"{context} must be an integer")
    return value


def _finite_scalar(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SkinArtifactValidationError(f"{context} must be a finite scalar")
    try:
        result = float(value)
    except OverflowError as error:
        raise SkinArtifactValidationError(f"{context} must be finite") from error
    if not math.isfinite(result):
        raise SkinArtifactValidationError(f"{context} must be finite")
    return result


def _validate_bounds(
    index: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> None:
    i1, i2, i3 = index
    n3, n2, n1 = shape
    if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
        raise SkinArtifactValidationError("skins.json cell index is out of bounds")


__all__ = [
    "ParsedSkinArtifacts",
    "SkinArtifactValidationError",
    "SkinCellRecord",
    "parse_skins_json",
    "validate_skin_artifact_semantics",
]
