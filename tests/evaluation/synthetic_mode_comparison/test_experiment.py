"""Tests for sequential synthetic mode-comparison experiments."""

from __future__ import annotations

import gc
import json
import weakref
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from pyosv.evaluation.synthetic_mode_comparison import (
    MetricRow,
    RuntimeRow,
    SyntheticModeComparisonConfig,
    run_mode_comparison,
)
from pyosv.evaluation.synthetic_quality import (
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
)
from pyosv.evaluation.synthetic_quality.stage_cache import PipelineStageCacheStats


def _small_result():
    return run_mode_comparison(
        SyntheticModeComparisonConfig(
            case_ids=("single_vertical_plane",),
            shape=(9, 9, 9),
        )
    )


def test_small_experiment_returns_json_safe_scalar_evidence() -> None:
    result = _small_result()

    json.dumps(result.as_dict(), allow_nan=False)
    assert len(result.trial_metadata) == 1
    assert len(result.cell_reports) == 1
    assert result.cell_reports[0]["trial_id"] == result.trial_metadata[0]["trial_id"]
    assert result.cache_stats[0]["trial_id"] == result.trial_metadata[0]["trial_id"]
    assert {row.trial_id for row in result.metric_rows} == {"single_vertical_plane"}
    assert {row.trial_id for row in result.contrast_rows} == {"single_vertical_plane"}
    assert all(np.isfinite(row.value) for row in result.metric_rows)
    assert all(np.isfinite(row.raw_value) for row in result.contrast_rows)
    assert all(
        np.isfinite(row.mean) for row in (*result.metric_aggregates, *result.contrast_aggregates)
    )


def test_positive_width_dipping_case_returns_finite_metrics_and_contrasts() -> None:
    result = run_mode_comparison(
        SyntheticModeComparisonConfig(
            case_ids=("single_dipping_plane",),
            shape=(10, 10, 10),
            skinning_config=SyntheticSkinningConfig(enabled=False),
            truth_metric_config=SyntheticTruthMetricConfig(
                truth_surface_half_width=0.5,
            ),
        )
    )

    assert result.metric_rows
    assert result.contrast_rows
    assert all(np.isfinite(row.value) for row in result.metric_rows)
    assert all(np.isfinite(row.raw_value) for row in result.contrast_rows)


def test_runtime_stage_order_and_shared_scanner_costs() -> None:
    result = _small_result()
    stages = tuple(row.stage for row in result.runtime_rows)

    assert stages == (
        "case_generation",
        "scanner_input_generation",
        "scanner_scan_thinning",
        "scanner_scan_thinning",
        "scanner_scalar_evidence",
        "scanner_scalar_evidence",
        *("cell_execution",) * 8,
        "metric_extraction",
        "contrast_extraction",
        "trial_total",
        "experiment_total",
    )
    scanner_rows = [row for row in result.runtime_rows if row.stage.startswith("scanner_")]
    assert tuple(row.scanner_backend for row in scanner_rows) == (
        None,
        "reference-like",
        "quality",
        "reference-like",
        "quality",
    )
    assert all(row.shared_stage for row in scanner_rows)
    assert all(row.call_count == 1 for row in result.runtime_rows)
    assert all(row.elapsed_seconds >= 0.0 for row in result.runtime_rows)


def test_result_payload_and_runtime_rows_are_recursively_frozen() -> None:
    result = _small_result()

    with pytest.raises(FrozenInstanceError):
        result.runtime_rows = ()
    with pytest.raises(FrozenInstanceError):
        result.runtime_rows[0].elapsed_seconds = 0.0
    with pytest.raises(TypeError):
        result.plan_metadata["shape"] = [1, 1, 1]  # type: ignore[index]
    with pytest.raises(TypeError):
        result.trial_metadata[0]["trial_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.cell_reports[0]["trial_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.cell_reports[0]["cells"]["RL-SCAN"]["case_id"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.cache_stats[0]["seed_hits"] = 100  # type: ignore[index]

    serialized = result.as_dict()
    serialized["plan_metadata"]["shape"][0] = 1
    assert result.plan_metadata["shape"][0] == 9


@pytest.mark.parametrize("elapsed", (-1.0, float("nan"), float("inf")))
def test_runtime_row_rejects_invalid_elapsed(elapsed: float) -> None:
    with pytest.raises(ValueError, match="elapsed_seconds"):
        RuntimeRow(
            case_id="case",
            trial_id="trial",
            seed=None,
            stage="stage",
            cell_label=None,
            scanner_backend=None,
            elapsed_seconds=elapsed,
            call_count=1,
            shared_stage=True,
        )


def test_backwards_clock_fails_without_returning_a_result() -> None:
    values = iter((0.0, 2.0, 1.0, 0.0))

    with pytest.raises(ValueError, match="clock moved backwards"):
        run_mode_comparison(
            SyntheticModeComparisonConfig(
                case_ids=("single_vertical_plane",),
                shape=(9, 9, 9),
            ),
            clock=lambda: next(values),
        )


def test_clock_regression_between_timed_stages_fails() -> None:
    values = iter((0.0, 0.0, 10.0, 11.0, 5.0))
    zero_stats = PipelineStageCacheStats(*(0,) * 8)

    class FakeEvaluation:
        pass

    def fake_runner(plan, trial, *, clock, runtime_recorder):
        evaluation = FakeEvaluation()
        evaluation.trial = trial
        evaluation.report_payload = {cell.label: {} for cell in plan.cells}
        evaluation.stage_cache_stats = zero_stats
        return evaluation

    with pytest.raises(ValueError, match="clock moved backwards"):
        run_mode_comparison(
            SyntheticModeComparisonConfig(
                case_ids=("single_vertical_plane",),
                shape=(9, 9, 9),
            ),
            clock=lambda: next(values),
            trial_runner=fake_runner,
            metric_extractor=lambda evaluation: (),
        )


def test_invalid_fake_runner_preserves_trial_order_and_releases_each_evaluation() -> None:
    calls: list[tuple[str, int | None]] = []
    references: list[weakref.ReferenceType[Any]] = []
    clock_values = iter(float(value) for value in range(100))
    zero_stats = PipelineStageCacheStats(*(0,) * 8)

    class FakeEvaluation:
        pass

    def fake_runner(plan, trial, *, clock, runtime_recorder):
        if references:
            assert references[-1]() is None
        calls.append((trial.case_id, trial.seed))
        evaluation = FakeEvaluation()
        evaluation.trial = trial
        evaluation.report_payload = {cell.label: {"scalar": np.float32(1.0)} for cell in plan.cells}
        evaluation.artifacts = {"temporary": np.ones((2, 2))}
        evaluation.stage_cache_stats = zero_stats
        references.append(weakref.ref(evaluation))
        runtime_recorder.record(
            stage="case_generation",
            elapsed_seconds=0.25,
            shared_stage=True,
        )
        for cell in plan.cells:
            runtime_recorder.record(
                stage="cell_execution",
                elapsed_seconds=0.5,
                cell_label=cell.label,
                scanner_backend=cell.scanner_backend,
                shared_stage=False,
            )
        return evaluation

    with pytest.raises(ValueError, match="runtime_rows"):
        run_mode_comparison(
            SyntheticModeComparisonConfig(
                case_ids=("single_vertical_plane", "single_dipping_plane", "weak_noisy_plane"),
                trial_seeds=(3, 5, 7),
                shape=(9, 9, 9),
            ),
            clock=lambda: next(clock_values),
            trial_runner=fake_runner,
            metric_extractor=lambda evaluation: (),
        )

    assert calls == [
        ("single_vertical_plane", None),
        ("single_dipping_plane", None),
        ("weak_noisy_plane", 3),
        ("weak_noisy_plane", 5),
        ("weak_noisy_plane", 7),
    ]
    assert references[-1]() is None
    assert references[-1]() is None


def test_array_in_cell_report_is_rejected_instead_of_retained_as_lists() -> None:
    zero_stats = PipelineStageCacheStats(*(0,) * 8)

    class FakeEvaluation:
        pass

    def fake_runner(plan, trial, *, clock, runtime_recorder):
        evaluation = FakeEvaluation()
        evaluation.trial = trial
        evaluation.report_payload = {cell.label: {"volume": np.ones((2, 2))} for cell in plan.cells}
        evaluation.stage_cache_stats = zero_stats
        return evaluation

    with pytest.raises(ValueError, match="scalar-only.*NumPy arrays"):
        run_mode_comparison(
            SyntheticModeComparisonConfig(
                case_ids=("single_vertical_plane",),
                shape=(9, 9, 9),
            ),
            trial_runner=fake_runner,
            metric_extractor=lambda evaluation: (),
        )


def test_result_rejects_array_in_injected_metric_row() -> None:
    zero_stats = PipelineStageCacheStats(*(0,) * 8)
    row = MetricRow(
        schema_version=1,
        case_id="single_vertical_plane",
        trial_id="single_vertical_plane",
        seed=None,
        scope="scanner_only",
        cell_label="RL-SCAN",
        input_mode="scanner",
        scanner_backend="reference-like",
        scanner_refinement_factor=None,
        scanner_thin_mode="reference",
        workflow_mode=None,
        voter_thin_mode=None,
        skinner_method=None,
        variant="current_default",
        stage="scanner_raw",
        selection="all",
        metric="array_nonzero_fraction",
        value=0.5,
        unit="fraction",
        direction="neutral",
        contrast_eligible=True,
    )
    object.__setattr__(row, "scanner_backend", np.asarray(["reference-like"]))

    class FakeEvaluation:
        pass

    def fake_runner(plan, trial, *, clock, runtime_recorder):
        evaluation = FakeEvaluation()
        evaluation.trial = trial
        evaluation.report_payload = {cell.label: {} for cell in plan.cells}
        evaluation.stage_cache_stats = zero_stats
        return evaluation

    with pytest.raises(ValueError, match="scalar-only.*NumPy arrays"):
        run_mode_comparison(
            SyntheticModeComparisonConfig(
                case_ids=("single_vertical_plane",),
                shape=(9, 9, 9),
            ),
            trial_runner=fake_runner,
            metric_extractor=lambda evaluation: (row,),
            contrast_builder=lambda rows: (),
            metric_aggregator=lambda rows: (),
        )


@pytest.mark.parametrize("failure_stage", ("metric", "contrast"))
def test_processing_failure_releases_trial_evaluation(failure_stage: str) -> None:
    zero_stats = PipelineStageCacheStats(*(0,) * 8)
    references: list[weakref.ReferenceType[Any]] = []

    class FakeEvaluation:
        pass

    def fake_runner(plan, trial, *, clock, runtime_recorder):
        evaluation = FakeEvaluation()
        evaluation.trial = trial
        evaluation.report_payload = {cell.label: {} for cell in plan.cells}
        evaluation.artifacts = {"temporary": np.ones((2, 2))}
        evaluation.stage_cache_stats = zero_stats
        references.append(weakref.ref(evaluation))
        return evaluation

    def raise_processing_error(values):
        raise RuntimeError(f"{failure_stage} failed")

    def run_failure() -> None:
        kwargs = (
            {"metric_extractor": raise_processing_error}
            if failure_stage == "metric"
            else {
                "metric_extractor": lambda evaluation: (),
                "contrast_builder": raise_processing_error,
            }
        )
        with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
            run_mode_comparison(
                SyntheticModeComparisonConfig(
                    case_ids=("single_vertical_plane",),
                    shape=(9, 9, 9),
                ),
                trial_runner=fake_runner,
                **kwargs,
            )

    run_failure()
    gc.collect()
    assert references[0]() is None
