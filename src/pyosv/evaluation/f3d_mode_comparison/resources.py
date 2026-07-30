"""Runtime, peak-RSS, and storage diagnostics for canonical F3 runs."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .artifacts import F3RunWorkspace
from .runner import F3StageRuntime
from .scanner import F3ScannerStageResult

F3_RESOURCE_SCHEMA_VERSION = 1
F3_RESOURCE_INTERPRETATION = "within_run_attribution_not_an_isolated_benchmark_or_accuracy_evidence"

ResourceStatus = Literal["available", "unavailable"]
StageState = Literal["computed", "reused"]
RSSProbe = Callable[[], int | None]


@dataclass(frozen=True, slots=True)
class RSSSnapshot:
    """A process-peak or stage-boundary RSS observation normalized to bytes."""

    schema_version: int
    scope: str
    point: str
    value_bytes: int | None
    status: ResourceStatus
    source: str
    semantics: str

    def __post_init__(self) -> None:
        if self.schema_version != F3_RESOURCE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {F3_RESOURCE_SCHEMA_VERSION}")
        if self.scope not in {"process_peak", "stage_snapshot"}:
            raise ValueError("scope must be 'process_peak' or 'stage_snapshot'")
        if self.status == "available":
            if (
                isinstance(self.value_bytes, bool)
                or not isinstance(self.value_bytes, int)
                or self.value_bytes < 0
            ):
                raise ValueError("available RSS values must be non-negative bytes")
        elif self.status == "unavailable":
            if self.value_bytes is not None:
                raise ValueError("unavailable RSS values must be None")
        else:
            raise ValueError("unknown RSS status")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StageResourceRow:
    """Resource attribution for one computed or reused stage-use event."""

    schema_version: int
    stage_kind: str
    fingerprint: str
    computed: bool
    state: StageState
    cell_consumers: tuple[str, ...]
    cell: str
    elapsed_seconds: float
    elapsed_semantics: str
    input_bytes: int
    output_bytes: int
    voxel_count: int
    voxel_throughput_per_second: float | None
    interpretation: str = F3_RESOURCE_INTERPRETATION

    def __post_init__(self) -> None:
        if self.schema_version != F3_RESOURCE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {F3_RESOURCE_SCHEMA_VERSION}")
        if self.state not in {"computed", "reused"}:
            raise ValueError("state must be 'computed' or 'reused'")
        if self.computed != (self.state == "computed"):
            raise ValueError("computed must agree with state")
        if not np.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        for name in ("input_bytes", "output_bytes", "voxel_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        expected = "compute" if self.computed else "load_validation"
        if self.elapsed_semantics != expected:
            raise ValueError(f"elapsed_semantics must be {expected!r}")
        expected_throughput = (
            self.voxel_count / self.elapsed_seconds
            if self.voxel_count and self.elapsed_seconds > 0.0
            else None
        )
        if self.voxel_throughput_per_second != expected_throughput:
            raise ValueError("voxel throughput is inconsistent with elapsed time")
        if self.interpretation != F3_RESOURCE_INTERPRETATION:
            raise ValueError("resource interpretation is not canonical")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StorageRow:
    """Storage used by a stage or by the whole workspace."""

    schema_version: int
    scope: str
    stage_kind: str | None
    fingerprint: str | None
    logical_bytes: int
    actual_file_bytes: int
    allocated_bytes: int | None
    file_count: int
    reference_files_are_not_dereferenced: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != F3_RESOURCE_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {F3_RESOURCE_SCHEMA_VERSION}")
        if self.scope not in {"workspace", "stage"}:
            raise ValueError("scope must be 'workspace' or 'stage'")
        if self.scope == "stage" and (self.stage_kind is None or self.fingerprint is None):
            raise ValueError("stage rows require stage_kind and fingerprint")
        if self.scope == "workspace" and (
            self.stage_kind is not None or self.fingerprint is not None
        ):
            raise ValueError("workspace rows must not identify a stage")
        for name in ("logical_bytes", "actual_file_bytes", "file_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.allocated_bytes is not None and (
            isinstance(self.allocated_bytes, bool)
            or not isinstance(self.allocated_bytes, int)
            or self.allocated_bytes < 0
        ):
            raise ValueError("allocated_bytes must be non-negative or None")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResourceExtraction:
    """Complete non-accuracy resource evidence for one run."""

    stage_rows: tuple[StageResourceRow, ...]
    rss_snapshots: tuple[RSSSnapshot, ...]
    storage_rows: tuple[StorageRow, ...]
    interpretation: str = F3_RESOURCE_INTERPRETATION

    def as_dict(self) -> dict[str, Any]:
        return {
            "resource_schema_version": F3_RESOURCE_SCHEMA_VERSION,
            "interpretation": self.interpretation,
            "stage_runtime": [row.as_dict() for row in self.stage_rows],
            "rss": [row.as_dict() for row in self.rss_snapshots],
            "storage": [row.as_dict() for row in self.storage_rows],
        }


class PeakRSSRecorder:
    """Record explicit process-peak RSS snapshots without a sampler thread."""

    def __init__(
        self,
        probe: RSSProbe | None = None,
        *,
        source: str | None = None,
        semantics: str | None = None,
    ) -> None:
        if probe is None:
            self._probe = _linux_peak_rss_bytes
            self._source = "resource.getrusage(RUSAGE_SELF).ru_maxrss"
            self._semantics = "cumulative_process_peak_rss_bytes"
        else:
            if not callable(probe):
                raise TypeError("probe must be callable")
            self._probe = probe
            self._source = source or "injected_probe"
            self._semantics = semantics or "injected_peak_rss_bytes"
        self._snapshots: list[RSSSnapshot] = []
        self._point_counts: dict[tuple[str, str], int] = {}
        self._recorded_points: set[tuple[str, str]] = set()

    @property
    def snapshots(self) -> tuple[RSSSnapshot, ...]:
        return tuple(self._snapshots)

    def snapshot(self, point: str, *, scope: str = "stage_snapshot") -> RSSSnapshot:
        """Take one exception-safe observation; probe failures become unavailable."""

        if not isinstance(point, str) or not point:
            raise ValueError("point must be a non-empty string")
        key = (scope, point)
        occurrence = self._point_counts.get(key, 0) + 1
        recorded_point = point if occurrence == 1 else f"{point}:occurrence={occurrence}"
        while (scope, recorded_point) in self._recorded_points:
            occurrence += 1
            recorded_point = f"{point}:occurrence={occurrence}"
        self._point_counts[key] = occurrence
        self._recorded_points.add((scope, recorded_point))
        try:
            value = self._probe()
            if value is None:
                raise RuntimeError("RSS probe is unavailable")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("RSS probe must return non-negative bytes or None")
        except Exception:
            row = RSSSnapshot(
                F3_RESOURCE_SCHEMA_VERSION,
                scope,
                recorded_point,
                None,
                "unavailable",
                self._source,
                self._semantics,
            )
        else:
            row = RSSSnapshot(
                F3_RESOURCE_SCHEMA_VERSION,
                scope,
                recorded_point,
                value,
                "available",
                self._source,
                self._semantics,
            )
        self._snapshots.append(row)
        return row

    def process_peak(self, point: str = "process_end") -> RSSSnapshot:
        return self.snapshot(point, scope="process_peak")

    def stage_before(
        self,
        stage_kind: str,
        fingerprint: str,
        *,
        phase: str | None = None,
    ) -> RSSSnapshot:
        return self.snapshot(_stage_snapshot_point(stage_kind, fingerprint, phase, "before"))

    def stage_after(
        self,
        stage_kind: str,
        fingerprint: str,
        *,
        phase: str | None = None,
    ) -> RSSSnapshot:
        return self.snapshot(_stage_snapshot_point(stage_kind, fingerprint, phase, "after"))


def stage_resource_row(
    runtime: F3StageRuntime,
    *,
    voxel_count: int,
) -> StageResourceRow:
    """Normalize one runner event into the stable resource schema."""

    if not isinstance(runtime, F3StageRuntime):
        raise TypeError("runtime must be an F3StageRuntime")
    if isinstance(voxel_count, bool) or not isinstance(voxel_count, int) or voxel_count < 0:
        raise ValueError("voxel_count must be a non-negative integer")
    computed = runtime.state == "computed"
    elapsed = float(runtime.elapsed_seconds)
    throughput = voxel_count / elapsed if voxel_count and elapsed > 0.0 else None
    return StageResourceRow(
        F3_RESOURCE_SCHEMA_VERSION,
        runtime.kind,
        runtime.fingerprint,
        computed,
        runtime.state,
        tuple(runtime.shared_consumers),
        runtime.cell,
        elapsed,
        "compute" if computed else "load_validation",
        int(runtime.source_bytes),
        int(runtime.output_bytes),
        voxel_count,
        throughput,
    )


def extract_stage_resources(
    runtime_events: Iterable[F3StageRuntime],
    *,
    shape: tuple[int, int, int],
) -> tuple[StageResourceRow, ...]:
    """Normalize all runtime hooks with a common full-volume voxel count."""

    if (
        not isinstance(shape, tuple)
        or len(shape) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape
        )
    ):
        raise ValueError("shape must contain exactly three positive integers")
    voxel_count = int(np.prod(shape))
    return tuple(stage_resource_row(event, voxel_count=voxel_count) for event in runtime_events)


def scanner_stage_resource_rows(
    stages: Sequence[F3ScannerStageResult] | Mapping[str, F3ScannerStageResult],
) -> tuple[StageResourceRow, ...]:
    """Normalize scanner results, including reuse validation time."""

    stage_rows = tuple(stages.values()) if isinstance(stages, Mapping) else tuple(stages)
    output = []
    consumers = {
        "reference-like": ("RL-REF", "RL-QUAL"),
        "quality": ("Q-REF", "Q-QUAL"),
    }
    for stage in stage_rows:
        if not isinstance(stage, F3ScannerStageResult):
            raise TypeError("stages must contain F3ScannerStageResult values")
        voxel_count = int(np.prod(stage.shape))
        elapsed = float(stage.elapsed_seconds)
        output.append(
            StageResourceRow(
                F3_RESOURCE_SCHEMA_VERSION,
                "scanner",
                stage.fingerprint,
                not stage.reused,
                "reused" if stage.reused else "computed",
                consumers[stage.backend],
                consumers[stage.backend][0],
                elapsed,
                "load_validation" if stage.reused else "compute",
                int(stage.input_bytes),
                int(stage.output_bytes),
                voxel_count,
                voxel_count / elapsed if voxel_count and elapsed > 0.0 else None,
            )
        )
    return tuple(output)


def storage_report(workspace: F3RunWorkspace | str | os.PathLike[str]) -> tuple[StorageRow, ...]:
    """Count each stage and the stable, pre-publication workspace exactly once."""

    root = workspace.path if isinstance(workspace, F3RunWorkspace) else Path(workspace)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("workspace must be a non-symlink directory")
    rows: list[StorageRow] = []
    stages_root = root / "stages"
    if stages_root.is_dir() and not stages_root.is_symlink():
        for kind_path in sorted(stages_root.iterdir(), key=lambda path: path.name):
            if not kind_path.is_dir() or kind_path.is_symlink():
                continue
            for stage_path in sorted(kind_path.iterdir(), key=lambda path: path.name):
                if not stage_path.is_dir() or stage_path.is_symlink():
                    continue
                logical, actual, allocated, count = _directory_storage(stage_path)
                rows.append(
                    StorageRow(
                        F3_RESOURCE_SCHEMA_VERSION,
                        "stage",
                        kind_path.name,
                        stage_path.name,
                        logical,
                        actual,
                        allocated,
                        count,
                    )
                )
    completion_temporaries = tuple(root.glob(".completion.json.tmp-*"))
    logical, actual, allocated, count = _directory_storage(
        root,
        excluded_roots=(
            root / "reports",
            # Dedicated post-publication evidence is not source-bundle storage.
            root / "reskin_policy_comparison",
        ),
        excluded_files=(root / "completion.json", *completion_temporaries),
    )
    rows.append(
        StorageRow(
            F3_RESOURCE_SCHEMA_VERSION,
            "workspace",
            None,
            None,
            logical,
            actual,
            allocated,
            count,
        )
    )
    return tuple(rows)


def extract_f3d_resources(
    runtime_events: Sequence[F3StageRuntime],
    *,
    shape: tuple[int, int, int],
    workspace: F3RunWorkspace | str | os.PathLike[str],
    scanner_stages: Sequence[F3ScannerStageResult] | Mapping[str, F3ScannerStageResult] = (),
    rss_recorder: PeakRSSRecorder,
) -> ResourceExtraction:
    """Build the complete within-run resource diagnostic result."""

    if not isinstance(rss_recorder, PeakRSSRecorder):
        raise TypeError("rss_recorder must be a PeakRSSRecorder used during stage execution")
    if not any(snapshot.scope == "stage_snapshot" for snapshot in rss_recorder.snapshots):
        raise ValueError("rss_recorder has no stage-boundary snapshots")
    if not any(snapshot.scope == "process_peak" for snapshot in rss_recorder.snapshots):
        raise ValueError("rss_recorder has no explicit process-peak snapshot")
    stage_rows = (
        *scanner_stage_resource_rows(scanner_stages),
        *extract_stage_resources(runtime_events, shape=shape),
    )
    referenced_stages = {(row.stage_kind, row.fingerprint) for row in stage_rows}
    storage_rows = tuple(
        row
        for row in storage_report(workspace)
        if row.scope == "workspace" or (row.stage_kind, row.fingerprint) in referenced_stages
    )
    return ResourceExtraction(
        stage_rows,
        rss_recorder.snapshots,
        storage_rows,
    )


def _linux_peak_rss_bytes() -> int | None:
    """Return Linux ``ru_maxrss`` normalized from documented KiB to bytes."""

    if not sys.platform.startswith("linux"):
        return None
    try:
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, OSError, ValueError):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not np.isfinite(value) or value < 0 or not float(value).is_integer():
        return None
    # Linux getrusage(2) defines ru_maxrss in KiB.
    return int(value) * 1024


def _stage_snapshot_point(
    stage_kind: str,
    fingerprint: str,
    phase: str | None,
    boundary: str,
) -> str:
    for name, value in (
        ("stage_kind", stage_kind),
        ("fingerprint", fingerprint),
        ("boundary", boundary),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
    if phase is not None and (not isinstance(phase, str) or not phase):
        raise ValueError("phase must be a non-empty string or None")
    parts = (stage_kind, fingerprint, *((phase,) if phase is not None else ()), boundary)
    return ":".join(parts)


def _directory_storage(
    path: Path,
    *,
    excluded_roots: tuple[Path, ...] = (),
    excluded_files: tuple[Path, ...] = (),
) -> tuple[int, int, int | None, int]:
    logical = 0
    actual = 0
    allocated = 0
    allocated_supported = True
    count = 0
    seen_inodes: set[tuple[int, int]] = set()
    for candidate in sorted(path.rglob("*")):
        if candidate in excluded_files or any(
            candidate.is_relative_to(root) for root in excluded_roots
        ):
            continue
        if candidate.is_symlink() or not candidate.is_file():
            continue
        stat_result = candidate.stat(follow_symlinks=False)
        size = int(stat_result.st_size)
        logical += size
        count += 1
        identity = (stat_result.st_dev, stat_result.st_ino)
        if identity in seen_inodes:
            continue
        seen_inodes.add(identity)
        actual += size
        blocks = getattr(stat_result, "st_blocks", None)
        if blocks is None:
            allocated_supported = False
        else:
            allocated += int(blocks) * 512
    return logical, actual, allocated if allocated_supported else None, count


__all__ = [
    "F3_RESOURCE_INTERPRETATION",
    "F3_RESOURCE_SCHEMA_VERSION",
    "PeakRSSRecorder",
    "RSSSnapshot",
    "ResourceExtraction",
    "StageResourceRow",
    "StorageRow",
    "extract_f3d_resources",
    "extract_stage_resources",
    "scanner_stage_resource_rows",
    "stage_resource_row",
    "storage_report",
]
