from __future__ import annotations

import inspect
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

import pyosv.evaluation.f3d_mode_comparison.runner as runner_module
from pyosv.evaluation.f3d_mode_comparison import (
    F3DatasetIdentity,
    F3DatasetSpec,
    F3FileIdentity,
    F3ModeComparisonConfig,
    F3RunWorkspace,
    F3ScannerStageResult,
    F3StageCorruptionError,
    F3VolumeSource,
    F3VotingControls,
    F3WorkspaceMismatchError,
    PeakRSSRecorder,
    build_f3d_cell_stage_fingerprints,
    build_f3d_mode_comparison_plan,
    canonical_fingerprint,
    canonical_json_bytes,
    extract_f3d_diagnostics,
    extract_f3d_metric_rows,
    extract_stage_resources,
    load_f3d_mode_comparison_cells,
    run_f3d_mode_comparison_cells,
    scanner_stage_artifacts,
    scanner_stage_resolved_settings,
    stage_fingerprint,
)
from pyosv.evaluation.synthetic_quality import SyntheticSkinningConfig


def test_run_cells_appends_rss_recorder_after_existing_parameters() -> None:
    parameters = list(inspect.signature(run_f3d_mode_comparison_cells).parameters)

    assert parameters[-2:] == ["variant_spec", "rss_recorder"]


class _LoadedScanner:
    def __init__(
        self,
        stage: F3ScannerStageResult,
        closed: list[str],
    ) -> None:
        self.backend = stage.backend
        self.path = stage.path
        self.fingerprint = stage.fingerprint
        self.shape = stage.shape
        self.report = {}
        self.closed = False
        self.ft = np.full(stage.shape, 0.8, dtype=np.float32)
        self.pt = np.full(stage.shape, 20.0, dtype=np.float32)
        self.tt = np.full(stage.shape, 70.0, dtype=np.float32)
        self.fet = self.ft.copy()
        self.fpt = self.pt.copy()
        self.ftt = self.tt.copy()
        self.confidence = None
        self._closed_backends = closed

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._closed_backends.append(self.backend)


def _workspace(path: Path) -> F3RunWorkspace:
    for kind in ("scanner", "voting", "thinning", "skinning"):
        (path / "stages" / kind).mkdir(parents=True)
    (path / "cells").mkdir()
    (path / "reports").mkdir()
    source = _fixture_volume_source(path.parent / "data")
    computation = {
        "artifact_schema_version": 1,
        "stage_contract_version": 1,
        "fingerprint_contract_version": 1,
        "plan": {"name": "runner-fixture"},
        "dataset_identity": source.identity.computation_identity,
        "implementation_identity": {"name": "test"},
    }
    fingerprint = canonical_fingerprint(computation)
    manifest = {
        **computation,
        "run_fingerprint": fingerprint,
        "provenance": {},
    }
    (path / "run_manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return F3RunWorkspace(path, fingerprint, manifest, resumed=False)


def _fixture_volume_source(
    data_root: Path,
    *,
    input_fingerprint: str = "3" * 64,
) -> F3VolumeSource:
    shape = (3, 4, 5)
    spec = F3DatasetSpec(
        dataset_id="runner-fixture",
        shape=shape,
        files={"input": "ep.dat"},
        expected_bytes=int(np.prod(shape)) * np.dtype(">f4").itemsize,
    )
    identity = F3DatasetIdentity(
        dataset_id=spec.dataset_id,
        data_root=data_root.absolute(),
        files=(
            F3FileIdentity(
                role="input",
                filename="ep.dat",
                resolved_path=(data_root / "ep.dat").absolute(),
                size=spec.expected_bytes,
                sha256=input_fingerprint,
                shape=shape,
                storage_dtype=spec.storage_dtype,
            ),
        ),
    )
    source = object.__new__(F3VolumeSource)
    source._spec = spec
    source._identity = identity
    return source


def _scanner_stages(
    workspace: F3RunWorkspace,
    shape: tuple[int, int, int] = (3, 4, 5),
) -> dict[str, F3ScannerStageResult]:
    return {
        backend: _scanner_stage(workspace, backend, shape=shape)
        for backend in ("reference-like", "quality")
    }


def _scanner_stage(
    workspace: F3RunWorkspace,
    backend: str,
    *,
    shape: tuple[int, int, int] = (3, 4, 5),
    input_fingerprint: str = "3" * 64,
) -> F3ScannerStageResult:
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())
    config = plan.scanner_config_for(backend)  # type: ignore[arg-type]
    settings = scanner_stage_resolved_settings(config, shape)
    artifacts = scanner_stage_artifacts(shape, backend)  # type: ignore[arg-type]
    fingerprint = stage_fingerprint(
        "scanner",
        run_fingerprint_value=workspace.fingerprint,
        input_fingerprints={"ep.dat": input_fingerprint},
        resolved_settings=settings,
        artifacts=artifacts,
    )
    report = {
        "fingerprint": fingerprint,
        "backend": backend,
        "shape": list(shape),
        "input_fingerprint": {"sha256": input_fingerprint},
        "resolved_config": asdict(config),
        "resolved_stage_settings": settings,
    }

    def writer(path: Path) -> None:
        for artifact in artifacts:
            artifact_path = path / artifact.filename
            if artifact.format == "dat":
                np.zeros(shape, dtype=">f4").tofile(artifact_path)
            else:
                artifact_path.write_bytes(canonical_json_bytes(report) + b"\n")

    stage = workspace.write_or_reuse_stage(
        "scanner",
        input_fingerprints={"ep.dat": input_fingerprint},
        resolved_settings=settings,
        artifacts=artifacts,
        writer=writer,
        fingerprint=fingerprint,
    )
    return F3ScannerStageResult(
        backend=backend,  # type: ignore[arg-type]
        path=stage.path,
        fingerprint=stage.fingerprint,
        reused=stage.reused,
        shape=shape,
        input_fingerprint=input_fingerprint,
        report=MappingProxyType(report),
    )


def test_canonical_cells_share_voting_and_release_scanner_handles(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))
    loaded: list[str] = []
    closed: list[str] = []
    caches = []
    recorder = PeakRSSRecorder(lambda: 4096)

    def loader(stage: F3ScannerStageResult) -> _LoadedScanner:
        loaded.append(stage.backend)
        return _LoadedScanner(stage, closed)

    def workflow_runner(**kwargs: Any) -> Any:
        assert all(
            not cache._seeds
            and not cache._voting
            and not cache._thinning
            and not cache._final_thinning
            and not cache._primary_skinning
            for cache in caches
        )
        caches.append(kwargs["stage_cache"])
        return runner_module.execute_workflow3d(**kwargs)

    result = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        workflow_runner=workflow_runner,
        scanner_loader=loader,  # type: ignore[arg-type]
        rss_recorder=recorder,
    )

    assert [cell.label for cell in result.cells] == [
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    ]
    assert loaded == ["reference-like", "quality"]
    assert closed == loaded
    assert len({id(cache) for cache in caches}) == 4
    assert result.cell_for("RL-REF").stages.voting == result.cell_for("RL-QUAL").stages.voting
    assert result.cell_for("Q-REF").stages.voting == result.cell_for("Q-QUAL").stages.voting
    assert result.cell_for("RL-REF").stages.thinning != result.cell_for("RL-QUAL").stages.thinning
    assert not any((workspace.path / "stages" / "skinning").iterdir())
    assert {event.kind for event in result.stage_runtime} == {"voting", "thinning"}
    assert {path.name for path in (workspace.path / "cells").iterdir()} == {
        "RL-REF.json",
        "RL-QUAL.json",
        "Q-REF.json",
        "Q-QUAL.json",
    }
    points = [snapshot.point for snapshot in recorder.snapshots]
    assert sum(point.endswith(":before") for point in points) == sum(
        point.endswith(":after") for point in points
    )
    assert any(":compute:" in point for point in points)
    assert any(":load_validation:" in point for point in points)


def test_fake_clock_fixes_computed_reused_bytes_and_throughput(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))
    tick = 0.25
    clock_value = 0.0

    def fake_clock() -> float:
        nonlocal clock_value
        clock_value += tick
        return clock_value

    monkeypatch.setattr(runner_module.time, "perf_counter", fake_clock)
    result = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: _LoadedScanner(stage, []),  # type: ignore[arg-type]
    )

    events = {(event.cell, event.kind): event for event in result.stage_runtime}
    computed = events["RL-REF", "voting"]
    reused = events["RL-QUAL", "voting"]
    voxel_count = int(np.prod(scanners["reference-like"].shape))
    expected_input_bytes = 3 * voxel_count * np.dtype(">f4").itemsize
    stage_path = workspace.stage_path("voting", computed.fingerprint)
    expected_output_bytes = sum(
        (stage_path / filename).stat().st_size
        for filename in ("fv.dat", "vp.dat", "vt.dat", "report.json")
    )

    assert computed.state == "computed"
    assert reused.state == "reused"
    assert computed.fingerprint == reused.fingerprint
    assert computed.elapsed_seconds == reused.elapsed_seconds == tick
    assert computed.source_bytes == reused.source_bytes == expected_input_bytes
    assert computed.output_bytes == reused.output_bytes == expected_output_bytes

    rows = {
        (row.cell, row.stage_kind): row
        for row in extract_stage_resources(
            result.stage_runtime, shape=scanners["reference-like"].shape
        )
    }
    computed_row = rows["RL-REF", "voting"]
    reused_row = rows["RL-QUAL", "voting"]
    assert computed_row.elapsed_semantics == "compute"
    assert reused_row.elapsed_semantics == "load_validation"
    assert computed_row.voxel_throughput_per_second == voxel_count / tick
    assert reused_row.voxel_throughput_per_second == voxel_count / tick


def test_resolved_common_thin_override_shares_thinning_per_backend(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(
        F3ModeComparisonConfig(
            skinning_enabled=False,
            voter_thin_mode_override="reference",
        )
    )
    workflow_calls: list[str] = []

    def workflow_runner(**kwargs: Any) -> Any:
        workflow_calls.append(kwargs["attribute_identity"].backend)
        return runner_module.execute_workflow3d(**kwargs)

    result = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        workflow_runner=workflow_runner,
        scanner_loader=lambda stage: _LoadedScanner(stage, []),  # type: ignore[arg-type]
    )

    assert workflow_calls == ["reference-like", "quality"]
    for backend, labels in (
        ("reference-like", ("RL-REF", "RL-QUAL")),
        ("quality", ("Q-REF", "Q-QUAL")),
    ):
        first, second = (result.cell_for(label) for label in labels)
        assert first.backend == second.backend == backend
        assert first.stages.voting == second.stages.voting
        assert first.stages.thinning == second.stages.thinning
    events = {(event.cell, event.kind): event.state for event in result.stage_runtime}
    assert events["RL-QUAL", "thinning"] == "reused"
    assert events["Q-QUAL", "thinning"] == "reused"


def test_dependent_fingerprints_change_only_from_their_semantic_inputs(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    base_plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())
    base = build_f3d_cell_stage_fingerprints(workspace, base_plan, scanners)

    changed_scanners = {
        **scanners,
        "reference-like": _scanner_stage(
            workspace,
            "reference-like",
            input_fingerprint="4" * 64,
        ),
    }
    source_changed = build_f3d_cell_stage_fingerprints(
        workspace,
        base_plan,
        changed_scanners,
    )
    for label in ("RL-REF", "RL-QUAL"):
        assert source_changed[label].scanner != base[label].scanner
        assert source_changed[label].voting != base[label].voting
        assert source_changed[label].thinning != base[label].thinning
        assert source_changed[label].skinning != base[label].skinning
    for label in ("Q-REF", "Q-QUAL"):
        assert source_changed[label] == base[label]

    voting_plan = build_f3d_mode_comparison_plan(
        F3ModeComparisonConfig(
            voting_controls=replace(F3VotingControls(), seed_threshold=0.4),
        )
    )
    voting_changed = build_f3d_cell_stage_fingerprints(workspace, voting_plan, scanners)
    for label in base:
        assert voting_changed[label].scanner == base[label].scanner
        assert voting_changed[label].voting != base[label].voting
        assert voting_changed[label].thinning != base[label].thinning
        assert voting_changed[label].skinning != base[label].skinning

    skinning_plan = build_f3d_mode_comparison_plan(
        F3ModeComparisonConfig(
            skinning_template=replace(SyntheticSkinningConfig(), small_skin_size=11),
        )
    )
    skinning_changed = build_f3d_cell_stage_fingerprints(workspace, skinning_plan, scanners)
    for label in base:
        assert skinning_changed[label].scanner == base[label].scanner
        assert skinning_changed[label].voting == base[label].voting
        assert skinning_changed[label].thinning == base[label].thinning
        assert skinning_changed[label].skinning != base[label].skinning


def test_resume_uses_all_stages_without_loading_scanner(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))
    first_closed: list[str] = []
    run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: _LoadedScanner(stage, first_closed),  # type: ignore[arg-type]
    )

    def fail_loader(stage: F3ScannerStageResult) -> _LoadedScanner:
        raise AssertionError(f"unexpected scanner load: {stage.backend}")

    resumed = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=fail_loader,  # type: ignore[arg-type]
    )

    assert all(cell.reused for cell in resumed.cells)
    assert all(event.state == "reused" for event in resumed.stage_runtime)
    assert len(load_f3d_mode_comparison_cells(workspace, plan, scanners)) == 4


def test_resume_computes_only_a_missing_stage_chain(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))
    closed: list[str] = []
    first = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: _LoadedScanner(stage, closed),  # type: ignore[arg-type]
    )
    missing = first.cell_for("RL-QUAL").stages.thinning
    shutil.rmtree(workspace.stage_path("thinning", missing))
    loaded: list[str] = []

    resumed = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: loaded.append(stage.backend) or _LoadedScanner(stage, closed),  # type: ignore[arg-type]
    )

    assert loaded == ["reference-like"]
    events = {(event.cell, event.kind): event.state for event in resumed.stage_runtime}
    assert events["RL-QUAL", "voting"] == "reused"
    assert events["RL-QUAL", "thinning"] == "computed"
    assert all(
        events[label, kind] == "reused"
        for label in ("RL-REF", "Q-REF", "Q-QUAL")
        for kind in ("voting", "thinning")
    )


def test_resume_missing_skinning_hydrates_final_thinning_identity(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())
    first = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: _LoadedScanner(stage, []),  # type: ignore[arg-type]
    )
    cell = first.cell_for("RL-REF")
    skinning_path = workspace.stage_path("skinning", cell.stages.skinning)
    expected = {
        filename: (skinning_path / filename).read_bytes()
        for filename in ("skin_mask.dat", "skins.json", "report.json")
    }
    shutil.rmtree(skinning_path)
    caches = []

    def workflow_runner(**kwargs: Any) -> Any:
        cache = kwargs["stage_cache"]
        caches.append(cache)
        assert not cache._thinning
        assert len(cache._final_thinning) == 1
        return runner_module.execute_workflow3d(**kwargs)

    resumed = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        workflow_runner=workflow_runner,
        scanner_loader=lambda stage: _LoadedScanner(stage, []),  # type: ignore[arg-type]
    )

    assert len(caches) == 1
    assert resumed.cell_for("RL-REF").reused
    assert {filename: (skinning_path / filename).read_bytes() for filename in expected} == expected


def test_reverse_execution_keeps_cell_mapping_and_volume_bytes(
    tmp_path: Path,
) -> None:
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))
    runs = []
    for name, order in (
        ("forward", None),
        ("reverse", tuple(reversed([cell.label for cell in plan.cells]))),
    ):
        workspace = _workspace(tmp_path / name)
        scanners = _scanner_stages(workspace)
        result = run_f3d_mode_comparison_cells(
            workspace,
            plan,
            scanners,
            cell_order=order,
            scanner_loader=lambda stage: _LoadedScanner(stage, []),  # type: ignore[arg-type]
        )
        runs.append((workspace, result))

    forward_workspace, forward = runs[0]
    reverse_workspace, reverse = runs[1]
    for label in (cell.label for cell in plan.cells):
        forward_cell = forward.cell_for(label)
        reverse_cell = reverse.cell_for(label)
        assert forward_cell.stages == reverse_cell.stages
        assert forward_cell.path.read_bytes() == reverse_cell.path.read_bytes()
        for kind, filename in (("voting", "fv.dat"), ("thinning", "fvt.dat")):
            fingerprint = getattr(forward_cell.stages, kind)
            assert (forward_workspace.stage_path(kind, fingerprint) / filename).read_bytes() == (
                reverse_workspace.stage_path(kind, fingerprint) / filename
            ).read_bytes()


def test_enabled_skinning_writes_only_the_fixed_stage_artifacts(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())
    result = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: _LoadedScanner(stage, []),  # type: ignore[arg-type]
    )

    for cell in result.cells:
        path = workspace.stage_path("skinning", cell.stages.skinning)
        assert {item.name for item in path.iterdir()} == {
            "skin_mask.dat",
            "skins.json",
            "report.json",
            "stage_manifest.json",
            "complete.json",
        }
        mask = np.memmap(path / "skin_mask.dat", dtype=">f4", mode="r", shape=(3, 4, 5))
        try:
            assert mask.dtype == np.dtype(">f4")
            assert np.all(np.isfinite(mask))
        finally:
            mask._mmap.close()
        report = json.loads((path / "report.json").read_text())
        assert report["enabled"] is True
        assert "geological_accuracy" not in report


def test_corrupt_stage_and_unknown_cell_fingerprint_are_rejected(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))
    closed: list[str] = []
    result = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: _LoadedScanner(stage, closed),  # type: ignore[arg-type]
    )
    voting = workspace.stage_path("voting", result.cells[0].stages.voting)
    with (voting / "fv.dat").open("r+b") as stream:
        stream.write(b"\xff")

    with pytest.raises(F3StageCorruptionError, match="hash or size"):
        run_f3d_mode_comparison_cells(
            workspace,
            plan,
            scanners,
            scanner_loader=lambda stage: _LoadedScanner(stage, closed),  # type: ignore[arg-type]
        )

    # Cell identity is checked before referenced stages are opened.
    cell_path = workspace.path / "cells" / "RL-REF.json"
    cell = json.loads(cell_path.read_text())
    cell["stages"]["voting"] = "f" * 64
    cell_path.write_text(json.dumps(cell))
    with pytest.raises(F3StageCorruptionError, match="unknown stage fingerprint"):
        load_f3d_mode_comparison_cells(workspace, plan, scanners)


@pytest.mark.parametrize("extractor", ("metrics", "diagnostics"))
def test_extractors_validate_stage_completion_hashes(
    tmp_path: Path,
    extractor: str,
) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))
    result = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: _LoadedScanner(stage, []),  # type: ignore[arg-type]
    )
    artifact = scanners["reference-like"].path / "ft.dat"
    with artifact.open("r+b") as stream:
        stream.write(np.asarray(0.25, dtype=">f4").tobytes())

    source = _fixture_volume_source(tmp_path / "data")
    with pytest.raises(F3StageCorruptionError, match="hash or size"):
        if extractor == "metrics":
            extract_f3d_metric_rows(source, result.cells, slab_depth=1)
        else:
            extract_f3d_diagnostics(source, result.cells, boundary_margin=0)


def test_extractor_rejects_same_shape_source_from_another_dataset(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))
    result = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: _LoadedScanner(stage, []),  # type: ignore[arg-type]
    )

    other = _fixture_volume_source(tmp_path / "other-data", input_fingerprint="4" * 64)
    with pytest.raises(F3WorkspaceMismatchError, match="dataset identity"):
        extract_f3d_metric_rows(other, result.cells, slab_depth=1)


def test_resume_and_load_reject_corrupt_scanner_stage(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))
    run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: _LoadedScanner(stage, []),  # type: ignore[arg-type]
    )
    scanner_path = scanners["reference-like"].path
    with (scanner_path / "fet.dat").open("r+b") as stream:
        stream.write(b"\xff")

    with pytest.raises(F3StageCorruptionError, match="hash or size"):
        run_f3d_mode_comparison_cells(
            workspace,
            plan,
            scanners,
            scanner_loader=lambda stage: _LoadedScanner(stage, []),  # type: ignore[arg-type]
        )
    with pytest.raises(F3StageCorruptionError, match="hash or size"):
        load_f3d_mode_comparison_cells(workspace, plan, scanners)


def test_scanner_stage_must_belong_to_current_workspace(tmp_path: Path) -> None:
    first_workspace = _workspace(tmp_path / "first")
    second_workspace = _workspace(tmp_path / "second")
    first_scanners = _scanner_stages(first_workspace)
    second_scanners = _scanner_stages(second_workspace)
    second_scanners["reference-like"] = first_scanners["reference-like"]
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())

    with pytest.raises(F3StageCorruptionError, match="current workspace"):
        build_f3d_cell_stage_fingerprints(second_workspace, plan, second_scanners)


def test_wrong_parent_fingerprint_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))
    result = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: _LoadedScanner(stage, []),  # type: ignore[arg-type]
    )
    cell = result.cell_for("RL-REF")
    manifest_path = workspace.stage_path("thinning", cell.stages.thinning) / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parent_fingerprints"] = [cell.stages.scanner]
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(F3StageCorruptionError, match="parent_fingerprints"):
        run_f3d_mode_comparison_cells(
            workspace,
            plan,
            scanners,
            scanner_loader=lambda stage: _LoadedScanner(stage, []),  # type: ignore[arg-type]
        )


def test_stage_write_exception_removes_partial_artifacts_and_releases_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))
    closed: list[str] = []
    caches = []
    original_write_dat = runner_module._write_dat
    write_count = 0

    def workflow_runner(**kwargs: Any) -> Any:
        caches.append(kwargs["stage_cache"])
        return runner_module.execute_workflow3d(**kwargs)

    def failing_write_dat(path: Path, values: np.ndarray) -> None:
        nonlocal write_count
        write_count += 1
        if write_count == 2:
            raise RuntimeError("injected stage write failure")
        original_write_dat(path, values)

    monkeypatch.setattr(runner_module, "_write_dat", failing_write_dat)

    with pytest.raises(RuntimeError, match="injected stage write failure"):
        run_f3d_mode_comparison_cells(
            workspace,
            plan,
            scanners,
            workflow_runner=workflow_runner,
            scanner_loader=lambda stage: _LoadedScanner(stage, closed),  # type: ignore[arg-type]
        )

    assert closed == ["reference-like"]
    assert len(caches) == 1
    assert all(
        not cache._seeds
        and not cache._voting
        and not cache._thinning
        and not cache._final_thinning
        and not cache._primary_skinning
        for cache in caches
    )
    assert not any((workspace.path / "cells").iterdir())
    for kind in ("voting", "thinning", "skinning"):
        assert not any((workspace.path / "stages" / kind).iterdir())


def test_resume_exception_closes_hydrated_memmaps_and_releases_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path / "run")
    scanners = _scanner_stages(workspace)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))
    first = run_f3d_mode_comparison_cells(
        workspace,
        plan,
        scanners,
        scanner_loader=lambda stage: _LoadedScanner(stage, []),  # type: ignore[arg-type]
    )
    missing = first.cell_for("RL-QUAL").stages.thinning
    shutil.rmtree(workspace.stage_path("thinning", missing))
    opened: list[np.memmap] = []
    caches = []
    closed: list[str] = []
    original_open_dat = runner_module._open_dat

    def tracking_open_dat(path: Path, shape: tuple[int, int, int]) -> np.memmap:
        array = original_open_dat(path, shape)
        opened.append(array)
        return array

    def failing_workflow_runner(**kwargs: Any) -> Any:
        caches.append(kwargs["stage_cache"])
        raise RuntimeError("injected workflow failure")

    monkeypatch.setattr(runner_module, "_open_dat", tracking_open_dat)

    with pytest.raises(RuntimeError, match="injected workflow failure"):
        run_f3d_mode_comparison_cells(
            workspace,
            plan,
            scanners,
            workflow_runner=failing_workflow_runner,
            scanner_loader=lambda stage: _LoadedScanner(stage, closed),  # type: ignore[arg-type]
        )

    assert closed == ["reference-like"]
    assert len(caches) == 1
    assert not caches[0]._voting
    assert not caches[0]._final_thinning
    assert opened
    assert all(array._mmap.closed for array in opened)
