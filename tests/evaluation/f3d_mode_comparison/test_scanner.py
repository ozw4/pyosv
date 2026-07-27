from __future__ import annotations

import hashlib
import json
import shutil
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import pyosv.evaluation.f3d_mode_comparison.scanner as scanner_module
from pyosv.evaluation.f3d_mode_comparison import (
    F3DatasetIdentity,
    F3DatasetSpec,
    F3FileIdentity,
    F3ModeComparisonConfig,
    F3RunWorkspace,
    F3ScannerConfig,
    F3VolumeSource,
    F3WorkspaceMismatchError,
    PeakRSSRecorder,
    build_f3d_mode_comparison_plan,
    load_scanner_stage,
    run_f3d_mode_comparison_cells,
    run_scanner_stages,
    scanner_stage_artifacts,
    scanner_stage_fingerprint,
)
from pyosv.io import write_dat

_IMPLEMENTATION = "scanner-test-implementation"


class _Source:
    def __init__(self, root: Path, values: np.ndarray) -> None:
        shape = values.shape
        identity = F3FileIdentity(
            role="input",
            filename="ep.dat",
            resolved_path=(root / "ep.dat").absolute(),
            size=values.size * np.dtype(">f4").itemsize,
            sha256=hashlib.sha256(values.tobytes()).hexdigest(),
            shape=shape,
            storage_dtype=">f4",
        )
        self.identity = F3DatasetIdentity(
            dataset_id="fixture",
            files=(identity,),
            data_root=root.absolute(),
        )
        self.values = values
        self.read_count = 0

    def read_native_volume(self, role: str) -> np.ndarray:
        assert role == "input"
        self.read_count += 1
        result = np.array(self.values, dtype=np.float32, order="C", copy=True)
        result.flags.writeable = False
        return result


class _Scanner:
    def __init__(self, calls: dict[str, object]) -> None:
        self.calls = calls

    def scan(self, *args: object, **kwargs: object):
        image = args[4]
        self._observe_input(image)
        self.calls["reference_scan"] = int(self.calls["reference_scan"]) + 1
        return self._raw(image, likelihood=0.5)

    def scan_quality(self, *args: object, **kwargs: object):
        image = args[4]
        self._observe_input(image)
        self.calls["quality_scan"] = int(self.calls["quality_scan"]) + 1
        ft, pt, tt = self._raw(image, likelihood=0.75)
        return ft, pt, tt, np.full(image.shape, 0.8, dtype=np.float32)

    def thin(self, ft: np.ndarray, pt: np.ndarray, tt: np.ndarray, **kwargs: object):
        self.calls["thin"] = int(self.calls["thin"]) + 1
        assert kwargs == {
            "mode": "reference",
            "reference_sigma": 1.0,
            "remove_edge_effects": True,
        }
        return ft.copy(), pt.copy(), tt.copy()

    def reference_like_strike_sampling(self, *args: object) -> np.ndarray:
        return np.array([0.0, 20.0], dtype=np.float32)

    def reference_like_dip_sampling(self, *args: object) -> np.ndarray:
        return np.array([65.0, 70.0], dtype=np.float32)

    def refined_reference_like_strike_sampling(self, *args: object, **kwargs: object) -> np.ndarray:
        return np.array([0.0, 10.0, 20.0], dtype=np.float32)

    def refined_reference_like_dip_sampling(self, *args: object, **kwargs: object) -> np.ndarray:
        return np.array([65.0, 67.5, 70.0], dtype=np.float32)

    def _observe_input(self, image: np.ndarray) -> None:
        self.calls.setdefault("input_ids", []).append(id(image))
        assert not image.flags.writeable
        with pytest.raises(ValueError):
            image.flags.writeable = True

    @staticmethod
    def _raw(
        image: np.ndarray,
        *,
        likelihood: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = image.shape
        return (
            np.full(shape, likelihood, dtype=np.float32),
            np.full(shape, 20.0, dtype=np.float32),
            np.full(shape, 70.0, dtype=np.float32),
        )


def _workspace(path: Path, source: _Source | F3VolumeSource) -> F3RunWorkspace:
    (path / "stages" / "scanner").mkdir(parents=True)
    return F3RunWorkspace(
        path,
        "0" * 64,
        {"dataset_identity": source.identity.computation_identity},
        resumed=False,
    )


def _factory(calls: dict[str, object]):
    def make(sigma1: float, sigma2: float) -> _Scanner:
        assert sigma1 == 8.0
        assert sigma2 == 8.0
        calls["construction"] = int(calls["construction"]) + 1
        return _Scanner(calls)

    return make


def _calls() -> dict[str, object]:
    return {
        "construction": 0,
        "reference_scan": 0,
        "quality_scan": 0,
        "thin": 0,
    }


def test_runner_reads_once_shares_immutable_input_and_reuses(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source = _Source(tmp_path / "data", values)
    workspace = _workspace(tmp_path / "run", source)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())
    calls = _calls()
    recorder = PeakRSSRecorder(lambda: 4096)

    first = run_scanner_stages(
        workspace,
        source,  # type: ignore[arg-type]
        plan,
        scanner_factory=_factory(calls),  # type: ignore[arg-type]
        implementation_identity=_IMPLEMENTATION,
        rss_recorder=recorder,
    )

    assert source.read_count == 1
    assert calls == {
        "construction": 2,
        "reference_scan": 1,
        "quality_scan": 1,
        "thin": 2,
        "input_ids": [calls["input_ids"][0], calls["input_ids"][0]],
    }
    assert "confidence.dat" not in {item.name for item in first["reference-like"].path.iterdir()}
    assert (first["quality"].path / "confidence.dat").is_file()
    for backend, stage in first.items():
        expected_output_bytes = sum(
            (stage.path / artifact.filename).stat().st_size
            for artifact in scanner_stage_artifacts(values.shape, backend)
        )
        assert stage.output_bytes == expected_output_bytes
    assert [snapshot.point.rsplit(":", 1)[-1] for snapshot in recorder.snapshots] == [
        "before",
        "after",
        "before",
        "after",
    ]
    assert all(":compute_or_load_validation:" in row.point for row in recorder.snapshots)

    reuse_source = _Source(tmp_path / "other-data", values)
    reuse_calls = _calls()
    reuse_recorder = PeakRSSRecorder(lambda: 8192)
    reused = run_scanner_stages(
        workspace,
        reuse_source,  # type: ignore[arg-type]
        plan,
        scanner_factory=_factory(reuse_calls),  # type: ignore[arg-type]
        implementation_identity=_IMPLEMENTATION,
        rss_recorder=reuse_recorder,
    )
    assert all(stage.reused for stage in reused.values())
    assert reuse_source.read_count == 0
    assert reuse_calls == _calls()
    assert len(reuse_recorder.snapshots) == 4


def test_array_summary_uses_and_records_nonzero_epsilon() -> None:
    values = np.array([0.0, 5.0e-8, -5.0e-8, 2.0e-6, -2.0e-6], dtype=np.float32)

    summary = scanner_module._array_summary(values)

    assert summary["nonzero_epsilon"] == 1.0e-6
    assert summary["nonzero_count"] == 2
    assert summary["nonzero_fraction"] == pytest.approx(2 / 5)


def test_runner_rejects_input_not_bound_to_workspace_before_compute(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source = _Source(tmp_path / "data", values)
    workspace = _workspace(tmp_path / "run", source)
    changed_source = _Source(tmp_path / "changed-data", values + np.float32(1.0))
    calls = _calls()

    with pytest.raises(F3WorkspaceMismatchError, match="run manifest dataset identity"):
        run_scanner_stages(
            workspace,
            changed_source,  # type: ignore[arg-type]
            build_f3d_mode_comparison_plan(F3ModeComparisonConfig()),
            scanner_factory=_factory(calls),  # type: ignore[arg-type]
            implementation_identity=_IMPLEMENTATION,
        )

    assert changed_source.read_count == 0
    assert calls == _calls()
    assert not any((workspace.path / "stages" / "scanner").iterdir())


def test_reverse_order_and_one_backend_resume(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())
    source = _Source(tmp_path / "data", values)
    workspace = _workspace(tmp_path / "run", source)
    calls = _calls()
    first = run_scanner_stages(
        workspace,
        source,  # type: ignore[arg-type]
        plan,
        backend_order=("quality", "reference-like"),
        scanner_factory=_factory(calls),  # type: ignore[arg-type]
        implementation_identity=_IMPLEMENTATION,
    )

    forward_source = _Source(tmp_path / "forward-data", values)
    forward = run_scanner_stages(
        _workspace(tmp_path / "forward-run", forward_source),
        forward_source,  # type: ignore[arg-type]
        plan,
        scanner_factory=_factory(_calls()),  # type: ignore[arg-type]
        implementation_identity=_IMPLEMENTATION,
    )
    for backend in ("reference-like", "quality"):
        filenames = ("ft.dat", "pt.dat", "tt.dat", "fet.dat", "fpt.dat", "ftt.dat", "report.json")
        if backend == "quality":
            filenames = (*filenames, "confidence.dat")
        for filename in filenames:
            assert (first[backend].path / filename).read_bytes() == (
                forward[backend].path / filename
            ).read_bytes()

    shutil.rmtree(first["quality"].path)

    resumed_source = _Source(tmp_path / "resumed-data", values)
    resumed_calls = _calls()
    resumed = run_scanner_stages(
        workspace,
        resumed_source,  # type: ignore[arg-type]
        plan,
        scanner_factory=_factory(resumed_calls),  # type: ignore[arg-type]
        implementation_identity=_IMPLEMENTATION,
    )
    assert resumed["reference-like"].reused is True
    assert resumed["quality"].reused is False
    assert resumed_source.read_count == 1
    assert resumed_calls["construction"] == 1
    assert resumed_calls["reference_scan"] == 0
    assert resumed_calls["quality_scan"] == 1
    assert resumed_calls["thin"] == 1


@pytest.mark.parametrize(
    ("invalid", "match"),
    [
        ("shape", "ft shape"),
        ("dtype", "ft dtype"),
        ("nonfinite", "ft must contain only finite"),
        ("out-of-range", "closed unit interval"),
    ],
)
def test_invalid_scanner_outputs_are_rejected_before_publication(
    tmp_path: Path,
    invalid: str,
    match: str,
) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    calls = _calls()

    class InvalidScanner(_Scanner):
        def scan(self, *args: object, **kwargs: object):
            ft, pt, tt = super().scan(*args, **kwargs)
            if invalid == "shape":
                ft = ft[:, :, :-1]
            elif invalid == "dtype":
                ft = ft.astype(np.float64)
            elif invalid == "nonfinite":
                ft[0, 0, 0] = np.nan
            else:
                ft[0, 0, 0] = np.float32(1.01)
            return ft, pt, tt

    def factory(sigma1: float, sigma2: float) -> InvalidScanner:
        calls["construction"] = int(calls["construction"]) + 1
        return InvalidScanner(calls)

    run_root = tmp_path / "run"
    source = _Source(tmp_path / "data", values)
    with pytest.raises(ValueError, match=match):
        run_scanner_stages(
            _workspace(run_root, source),
            source,  # type: ignore[arg-type]
            build_f3d_mode_comparison_plan(F3ModeComparisonConfig()),
            scanner_factory=factory,  # type: ignore[arg-type]
            implementation_identity=_IMPLEMENTATION,
        )

    assert not any((run_root / "stages" / "scanner").iterdir())


def test_scan_exception_does_not_publish_partial_stage(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    calls = _calls()

    class FailingScanner(_Scanner):
        def scan(self, *args: object, **kwargs: object):
            raise RuntimeError("injected scan failure")

    def factory(sigma1: float, sigma2: float) -> FailingScanner:
        calls["construction"] = int(calls["construction"]) + 1
        return FailingScanner(calls)

    run_root = tmp_path / "run"
    source = _Source(tmp_path / "data", values)
    with pytest.raises(RuntimeError, match="injected scan failure"):
        run_scanner_stages(
            _workspace(run_root, source),
            source,  # type: ignore[arg-type]
            build_f3d_mode_comparison_plan(F3ModeComparisonConfig()),
            scanner_factory=factory,  # type: ignore[arg-type]
            implementation_identity=_IMPLEMENTATION,
        )

    assert not any((run_root / "stages" / "scanner").iterdir())


def test_write_exception_does_not_publish_partial_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    original_write = scanner_module._write_big_endian_dat

    def fail_after_write(path: Path, output: np.ndarray) -> None:
        original_write(path, output)
        raise OSError("injected write failure")

    monkeypatch.setattr(scanner_module, "_write_big_endian_dat", fail_after_write)
    run_root = tmp_path / "run"
    source = _Source(tmp_path / "data", values)
    with pytest.raises(OSError, match="injected write failure"):
        run_scanner_stages(
            _workspace(run_root, source),
            source,  # type: ignore[arg-type]
            build_f3d_mode_comparison_plan(F3ModeComparisonConfig()),
            scanner_factory=_factory(_calls()),  # type: ignore[arg-type]
            implementation_identity=_IMPLEMENTATION,
        )

    assert not any((run_root / "stages" / "scanner").iterdir())


def test_backend_arrays_are_released_before_next_backend(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    calls = _calls()
    references: list[weakref.ReferenceType[np.ndarray]] = []

    class LifecycleScanner(_Scanner):
        def scan(self, *args: object, **kwargs: object):
            output = super().scan(*args, **kwargs)
            references.extend(weakref.ref(array) for array in output)
            return output

        def scan_quality(self, *args: object, **kwargs: object):
            assert references
            assert all(reference() is None for reference in references)
            output = super().scan_quality(*args, **kwargs)
            references.extend(weakref.ref(array) for array in output)
            return output

        def thin(self, ft: np.ndarray, pt: np.ndarray, tt: np.ndarray, **kwargs: object):
            output = super().thin(ft, pt, tt, **kwargs)
            references.extend(weakref.ref(array) for array in output)
            return output

    def factory(sigma1: float, sigma2: float) -> LifecycleScanner:
        calls["construction"] = int(calls["construction"]) + 1
        return LifecycleScanner(calls)

    source = _Source(tmp_path / "data", values)
    run_scanner_stages(
        _workspace(tmp_path / "run", source),
        source,  # type: ignore[arg-type]
        build_f3d_mode_comparison_plan(F3ModeComparisonConfig()),
        scanner_factory=factory,  # type: ignore[arg-type]
        implementation_identity=_IMPLEMENTATION,
    )

    assert references
    assert all(reference() is None for reference in references)


def test_fingerprint_tracks_config_and_input_content(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source = _Source(tmp_path / "data", values)
    changed_source = _Source(tmp_path / "changed", values + np.float32(1.0))
    workspace = _workspace(tmp_path / "run", source)
    config = F3ScannerConfig()

    baseline = scanner_stage_fingerprint(
        workspace,
        source.identity.file_for("input"),
        config,
        implementation_identity=_IMPLEMENTATION,
    )
    assert (
        scanner_stage_fingerprint(
            workspace,
            source.identity.file_for("input"),
            replace(config, sigma1=9.0),
            implementation_identity=_IMPLEMENTATION,
        )
        != baseline
    )
    assert (
        scanner_stage_fingerprint(
            _workspace(tmp_path / "changed-run", changed_source),
            changed_source.identity.file_for("input"),
            config,
            implementation_identity=_IMPLEMENTATION,
        )
        != baseline
    )


def test_injected_scanner_factory_has_distinct_stage_identity(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source = _Source(tmp_path / "data", values)
    workspace = _workspace(tmp_path / "run", source)
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())
    stages = run_scanner_stages(
        workspace,
        source,  # type: ignore[arg-type]
        plan,
        scanner_factory=_factory(_calls()),  # type: ignore[arg-type]
    )

    stage = stages["reference-like"]
    identity = stage.report["resolved_stage_settings"]["scanner_stage_implementation_identity"]
    assert "scanner_factory" in identity
    assert stage.fingerprint != scanner_stage_fingerprint(
        workspace,
        source.identity.file_for("input"),
        plan.scanner_config_for("reference-like"),
    )


def test_default_fingerprint_tracks_scanner_algorithm_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source = _Source(tmp_path / "data", values)
    workspace = _workspace(tmp_path / "run", source)
    config = F3ScannerConfig()

    settings = scanner_module.scanner_stage_resolved_settings(config, values.shape)
    identity = settings["scanner_stage_implementation_identity"]
    assert isinstance(identity, dict)
    assert {
        "pyosv/_orient3d/scanner.py",
        "pyosv/thinning3d.py",
    } <= set(identity["algorithm_modules"])

    algorithm_source = tmp_path / "scanner_algorithm.py"
    algorithm_source.write_text("first implementation\n", encoding="utf-8")
    monkeypatch.setattr(
        scanner_module,
        "_scanner_implementation_source_files",
        lambda: {"pyosv/_orient3d/scanner.py": algorithm_source},
    )
    baseline = scanner_stage_fingerprint(
        workspace,
        source.identity.file_for("input"),
        config,
    )
    algorithm_source.write_text("other implementation\n", encoding="utf-8")

    assert (
        scanner_stage_fingerprint(
            workspace,
            source.identity.file_for("input"),
            config,
        )
        != baseline
    )


def test_loader_opens_big_endian_volumes_and_closes_maps(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    source = _Source(tmp_path / "data", values)
    workspace = _workspace(tmp_path / "run", source)
    calls = _calls()
    stages = run_scanner_stages(
        workspace,
        source,  # type: ignore[arg-type]
        build_f3d_mode_comparison_plan(F3ModeComparisonConfig()),
        scanner_factory=_factory(calls),  # type: ignore[arg-type]
        implementation_identity=_IMPLEMENTATION,
    )
    completion = json.loads((stages["quality"].path / "complete.json").read_text(encoding="utf-8"))
    assert completion["files"]["ft.dat"]["shape"] == list(values.shape)
    assert completion["files"]["ft.dat"]["dtype"] == ">f4"

    loaded = load_scanner_stage(stages["quality"])
    assert loaded.ft.dtype == np.dtype(">f4")
    assert loaded.ft.shape == values.shape
    assert not loaded.ft.flags.writeable
    np.testing.assert_allclose(loaded.ft, 0.75)
    np.testing.assert_allclose(loaded.confidence, 0.8)
    loaded.close()
    assert loaded.closed


def test_actual_scanners_smoke_on_small_volume(tmp_path: Path) -> None:
    shape = (3, 4, 5)
    data_root = tmp_path / "data"
    data_root.mkdir()
    values = np.linspace(0.0, 1.0, np.prod(shape), dtype=np.float32).reshape(shape)
    write_dat(data_root / "ep.dat", values)

    spec = F3DatasetSpec(
        dataset_id="fixture",
        shape=shape,
        files={"input": "ep.dat"},
        expected_bytes=values.size * np.dtype(">f4").itemsize,
    )
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))
    with F3VolumeSource(data_root, spec=spec) as source:
        workspace = _workspace(tmp_path / "run", source)
        stages = run_scanner_stages(
            workspace,
            source,
            plan,
            implementation_identity=_IMPLEMENTATION,
        )

    assert set(stages) == {"reference-like", "quality"}
    assert all(not stage.reused for stage in stages.values())
    for name in ("voting", "thinning", "skinning"):
        (workspace.path / "stages" / name).mkdir(parents=True, exist_ok=True)
    (workspace.path / "cells").mkdir()
    (workspace.path / "reports").mkdir()

    result = run_f3d_mode_comparison_cells(workspace, plan, stages)

    assert tuple(cell.label for cell in result.cells) == (
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    )
