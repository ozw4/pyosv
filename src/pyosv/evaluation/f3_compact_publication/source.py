"""Resolve validated source handles for the F3 compact publication."""

from __future__ import annotations

import hashlib
import stat
from math import prod
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from pyosv.candidate_volume import NONZERO_EPSILON, positive_candidate_mask

from ..f3d_mode_comparison.metrics import (
    F3_REFERENCE_STAGE_FILES,
    F3_REFERENCE_STAGE_ROLES,
)
from ..f3d_mode_comparison.runner import (
    thinning_stage_artifacts,
    voting_stage_artifacts,
)
from ..f3d_mode_comparison.scanner import scanner_stage_artifacts
from ..mode_comparison_publication.figures import build_f3_ridge_threshold_contract
from ..mode_comparison_publication.loaders import load_f3_source
from .config import (
    AMPLITUDE_DTYPE,
    AMPLITUDE_FILENAME,
    AMPLITUDE_ROLE,
    DISPLAY_CELL,
    SECTION_GROUPS,
    SECTION_SELECTION_POLICY,
    SECTIONS_PER_AXIS,
    STAGE_ORDER,
)
from .models import (
    AmplitudeIdentity,
    CompactSourceContext,
    RidgeStageThresholds,
    SelectedSection,
    SourceRidgeThresholdContract,
    StageSource,
)

_CANDIDATE_KIND_BY_STAGE = dict(zip(STAGE_ORDER, ("scanner", "voting", "thinning")))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot checksum source file: {path}") from error
    return digest.hexdigest()


def _require_volume_file(path: Path, expected_size: int, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing or inaccessible: {path}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    if metadata.st_size != expected_size:
        raise ValueError(
            f"{label} size mismatch: expected {expected_size} bytes, got {metadata.st_size}"
        )


def _select_q_qual(cells: Iterable[Any]) -> Any:
    matches = tuple(cell for cell in cells if getattr(cell, "label", None) == DISPLAY_CELL)
    if len(matches) != 1:
        raise ValueError(
            f"F3 result must contain exactly one {DISPLAY_CELL!r} cell; found {len(matches)}"
        )
    return matches[0]


def _artifact_for_stage(
    stage: str,
    shape: tuple[int, int, int],
    q_qual_cell: Any,
) -> tuple[str, Any]:
    kind = _CANDIDATE_KIND_BY_STAGE[stage]
    if kind == "scanner":
        artifacts = scanner_stage_artifacts(shape, q_qual_cell.backend)
    elif kind == "voting":
        artifacts = voting_stage_artifacts(shape)
    else:
        artifacts = thinning_stage_artifacts(shape)
    filename = f"{stage}.dat"
    matches = tuple(artifact for artifact in artifacts if artifact.filename == filename)
    if len(matches) != 1:
        raise ValueError(f"{kind} artifact contract must contain exactly one {filename!r}")
    return kind, matches[0]


def _identity_for_role(dataset_identity: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    files = dataset_identity.get("files")
    if not isinstance(files, list):
        raise ValueError("validated F3 dataset identity has no file list")
    matches = tuple(
        item for item in files if isinstance(item, Mapping) and item.get("role") == role
    )
    if len(matches) != 1:
        raise ValueError(f"validated F3 dataset identity must contain exactly one {role!r} file")
    return matches[0]


def _resolve_amplitude(source: Any, shape: tuple[int, int, int]) -> AmplitudeIdentity:
    path = source.data_root / AMPLITUDE_FILENAME
    dtype = np.dtype(AMPLITUDE_DTYPE)
    expected_size = prod(shape) * dtype.itemsize
    _require_volume_file(path, expected_size, "F3 amplitude input")
    return AmplitudeIdentity(
        role=AMPLITUDE_ROLE,
        filename=AMPLITUDE_FILENAME,
        resolved_path=path,
        shape=shape,
        storage_dtype=dtype.str,
        size=expected_size,
        sha256=_sha256(path),
    )


def _resolve_stage_sources(
    source: Any,
    q_qual_cell: Any,
    shape: tuple[int, int, int],
) -> tuple[StageSource, ...]:
    expected_size = prod(shape) * np.dtype(source.result.storage_dtype).itemsize
    output = []
    for stage in STAGE_ORDER:
        public_role = F3_REFERENCE_STAGE_ROLES[stage]
        public_filename = F3_REFERENCE_STAGE_FILES[stage]
        public_path = source.data_root / public_filename
        _require_volume_file(public_path, expected_size, f"public {stage} reference")
        public_identity = _identity_for_role(source.dataset_identity, public_role)
        if (
            public_identity.get("size") != expected_size
            or public_identity.get("shape") != list(shape)
            or public_identity.get("storage_dtype") != np.dtype(">f4").str
        ):
            raise ValueError(f"validated public {stage} reference layout is inconsistent")
        public_sha256 = public_identity.get("sha256")
        if not isinstance(public_sha256, str):
            raise ValueError(f"validated public {stage} reference SHA-256 is missing")

        kind, artifact = _artifact_for_stage(stage, shape, q_qual_cell)
        fingerprint = getattr(q_qual_cell.stages, kind)
        candidate_path = source.path / "stages" / kind / fingerprint / artifact.filename
        artifact_size = prod(artifact.shape) * np.dtype(artifact.dtype).itemsize
        if artifact_size != expected_size:
            raise ValueError(f"{kind} artifact layout does not match the F3 volume")
        _require_volume_file(candidate_path, expected_size, f"Q-QUAL {stage} artifact")
        output.append(
            StageSource(
                stage=stage,
                public_reference_role=public_role,
                public_reference_filename=public_filename,
                public_reference_path=public_path,
                public_reference_sha256=public_sha256,
                candidate_source_kind=kind,
                candidate_fingerprint=fingerprint,
                candidate_filename=artifact.filename,
                candidate_path=candidate_path,
            )
        )
    return tuple(output)


def _ridge_thresholds(source: Any) -> SourceRidgeThresholdContract:
    contract = build_f3_ridge_threshold_contract(source)
    stages = tuple(
        RidgeStageThresholds(
            stage=stage,
            public_reference_threshold=contract.stages[stage].reference_threshold,
            q_qual_threshold=contract.stages[stage].candidate_thresholds[DISPLAY_CELL],
        )
        for stage in STAGE_ORDER
    )
    return SourceRidgeThresholdContract(
        selection=contract.selection,
        percentile=contract.percentile,
        buffer_radius=contract.buffer_radius,
        stages=stages,
    )


def _axis_length(shape: tuple[int, int, int], axis: str) -> int:
    positions = {"i1": 2, "i3": 0}
    try:
        return shape[positions[axis]]
    except KeyError as error:
        raise ValueError(f"unsupported compact section axis: {axis!r}") from error


def _section(volume: np.ndarray, axis: str, index: int) -> np.ndarray:
    if axis == "i1":
        return volume[:, :, index]
    if axis == "i3":
        return volume[index, :, :]
    raise ValueError(f"unsupported compact section axis: {axis!r}")


def _select_public_fvt_sections(
    path: Path,
    shape: tuple[int, int, int],
    storage_dtype: str,
    threshold: float,
    *,
    section_group: str,
    axis: str,
    count: int,
) -> tuple[SelectedSection, ...]:
    axis_length = _axis_length(shape, axis)
    if axis_length < count:
        raise ValueError(
            f"compact section axis {axis!r} length must be at least {count}; got {axis_length}"
        )
    volume: np.memmap | None = None
    try:
        volume = np.memmap(path, dtype=storage_dtype, mode="r", shape=shape, order="C")
        selected = []
        for bin_index in range(count):
            start = bin_index * axis_length // count
            stop = (bin_index + 1) * axis_length // count
            best_index = start
            best_score = -1
            for index in range(start, stop):
                sample = np.asarray(_section(volume, axis, index))
                mask = positive_candidate_mask(sample, epsilon=NONZERO_EPSILON)
                mask &= sample >= threshold
                score = int(np.count_nonzero(mask))
                if score > best_score:
                    best_index = index
                    best_score = score
            selected.append(
                SelectedSection(
                    section_group=section_group,
                    axis=axis,
                    bin_index=bin_index,
                    index=best_index,
                    policy=SECTION_SELECTION_POLICY,
                    ridge_count_score=best_score,
                )
            )
        return tuple(selected)
    finally:
        if volume is not None:
            mapping = getattr(volume, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()


def load_compact_source(
    f3_bundle: str | Path,
    f3_data_root: str | Path,
) -> CompactSourceContext:
    """Validate and resolve all read-only inputs for one compact F3 publication."""

    source = load_f3_source(f3_bundle, f3_data_root)
    shape = tuple(source.result.volume_shape)
    if shape != tuple(source.dataset_spec.shape):
        raise ValueError("validated F3 result shape does not match its dataset contract")
    if (
        np.dtype(source.result.storage_dtype).str != AMPLITUDE_DTYPE
        or np.dtype(source.dataset_spec.storage_dtype).str != AMPLITUDE_DTYPE
    ):
        raise ValueError("validated F3 source storage dtype must be big-endian float32")
    q_qual_cell = _select_q_qual(source.result.cells)
    amplitude = _resolve_amplitude(source, shape)
    stage_sources = _resolve_stage_sources(source, q_qual_cell, shape)
    ridge_threshold_contract = _ridge_thresholds(source)
    fvt_threshold = ridge_threshold_contract.stages[-1].public_reference_threshold
    fvt_source = stage_sources[-1]
    selected_sections = tuple(
        selected
        for section_group, axis in SECTION_GROUPS
        for selected in _select_public_fvt_sections(
            fvt_source.public_reference_path,
            shape,
            source.result.storage_dtype,
            fvt_threshold,
            section_group=section_group,
            axis=axis,
            count=SECTIONS_PER_AXIS,
        )
    )
    return CompactSourceContext(
        f3=source,
        amplitude=amplitude,
        q_qual_cell=q_qual_cell,
        stage_sources=stage_sources,
        ridge_threshold_contract=ridge_threshold_contract,
        selected_sections=selected_sections,
    )


__all__ = ["load_compact_source"]
