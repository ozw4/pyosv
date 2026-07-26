from __future__ import annotations

from pathlib import Path

import pytest

from pyosv.evaluation.f3d_mode_comparison.resources import (
    F3_RESOURCE_INTERPRETATION,
    PeakRSSRecorder,
    extract_f3d_resources,
    extract_stage_resources,
    storage_report,
)
from pyosv.evaluation.f3d_mode_comparison.runner import F3StageRuntime


def _runtime(*, state: str, elapsed: float) -> F3StageRuntime:
    return F3StageRuntime(
        kind="voting",
        fingerprint="a" * 64,
        state=state,  # type: ignore[arg-type]
        elapsed_seconds=elapsed,
        source_bytes=120,
        output_bytes=80,
        cell_owner="RL-REF",
        shared_consumers=("RL-REF", "RL-QUAL"),
        cell="RL-REF",
    )


def test_stage_resources_distinguish_compute_and_reuse_elapsed() -> None:
    computed, reused = extract_stage_resources(
        (_runtime(state="computed", elapsed=2.0), _runtime(state="reused", elapsed=0.25)),
        shape=(2, 3, 4),
    )

    assert computed.computed is True
    assert computed.elapsed_semantics == "compute"
    assert computed.voxel_throughput_per_second == 12.0
    assert reused.computed is False
    assert reused.elapsed_semantics == "load_validation"
    assert reused.elapsed_seconds == 0.25
    assert reused.voxel_throughput_per_second == 96.0
    assert reused.input_bytes == 120
    assert reused.output_bytes == 80
    assert reused.interpretation == F3_RESOURCE_INTERPRETATION


def test_peak_rss_probe_and_exception_policy_are_stable() -> None:
    recorder = PeakRSSRecorder(lambda: 4096)
    before = recorder.stage_before("voting", "a" * 64)
    repeated = recorder.stage_before("voting", "a" * 64)
    colliding_name = recorder.snapshot(repeated.point)
    peak = recorder.process_peak()

    assert before.scope == "stage_snapshot"
    assert before.status == "available"
    assert before.value_bytes == 4096
    assert repeated.point == f"{before.point}:occurrence=2"
    assert len({before.point, repeated.point, colliding_name.point}) == 3
    assert peak.scope == "process_peak"

    def fail() -> int:
        raise RuntimeError("probe failure")

    unavailable = PeakRSSRecorder(fail).process_peak()
    assert unavailable.status == "unavailable"
    assert unavailable.value_bytes is None


def test_resource_extraction_requires_explicit_execution_and_process_peak_snapshots(
    tmp_path: Path,
) -> None:
    recorder = PeakRSSRecorder(lambda: 4096)
    with pytest.raises(ValueError, match="no stage-boundary snapshots"):
        extract_f3d_resources(
            (_runtime(state="computed", elapsed=2.0),),
            shape=(2, 3, 4),
            workspace=tmp_path,
            rss_recorder=recorder,
        )

    recorder.stage_before("voting", "a" * 64, phase="compute")
    recorder.stage_after("voting", "a" * 64, phase="compute")
    with pytest.raises(ValueError, match="no explicit process-peak snapshot"):
        extract_f3d_resources(
            (_runtime(state="computed", elapsed=2.0),),
            shape=(2, 3, 4),
            workspace=tmp_path,
            rss_recorder=recorder,
        )

    recorder.process_peak()
    active_stage = tmp_path / "stages" / "voting" / ("a" * 64)
    stale_stage = tmp_path / "stages" / "voting" / ("b" * 64)
    active_stage.mkdir(parents=True)
    stale_stage.mkdir()
    (active_stage / "complete.json").write_text("{}\n", encoding="utf-8")
    (stale_stage / "complete.json").write_text("{}\n", encoding="utf-8")
    extraction = extract_f3d_resources(
        (_runtime(state="computed", elapsed=2.0),),
        shape=(2, 3, 4),
        workspace=tmp_path,
        rss_recorder=recorder,
    )

    assert [snapshot.scope for snapshot in extraction.rss_snapshots] == [
        "stage_snapshot",
        "stage_snapshot",
        "process_peak",
    ]
    assert [(row.scope, row.stage_kind, row.fingerprint) for row in extraction.storage_rows] == [
        ("stage", "voting", "a" * 64),
        ("workspace", None, None),
    ]


def test_storage_counts_stage_once_and_does_not_dereference_cell_json(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stages" / "voting" / ("a" * 64)
    stage.mkdir(parents=True)
    (stage / "fv.dat").write_bytes(b"12345678")
    (stage / "report.json").write_text("{}", encoding="utf-8")
    cells = tmp_path / "cells"
    cells.mkdir()
    (cells / "RL-REF.json").write_text(
        '{"stages":{"voting":"' + "a" * 64 + '"}}',
        encoding="utf-8",
    )
    (cells / "RL-QUAL.json").write_text(
        '{"stages":{"voting":"' + "a" * 64 + '"}}',
        encoding="utf-8",
    )

    stage_row, workspace_row = storage_report(tmp_path)

    assert stage_row.scope == "stage"
    assert stage_row.file_count == 2
    assert stage_row.actual_file_bytes == 10
    assert workspace_row.file_count == 4
    assert workspace_row.actual_file_bytes == sum(
        path.stat().st_size for path in tmp_path.rglob("*") if path.is_file()
    )
    assert workspace_row.reference_files_are_not_dereferenced is True

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "runtime.csv").write_text("runtime\n", encoding="utf-8")
    (tmp_path / "completion.json").write_text("{}\n", encoding="utf-8")

    _, completed_workspace_row = storage_report(tmp_path)
    assert completed_workspace_row == workspace_row
