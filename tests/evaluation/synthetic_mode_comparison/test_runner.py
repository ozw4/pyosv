"""Tests for shared-stage synthetic mode-comparison trial execution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from pyosv.evaluation.synthetic_mode_comparison import (
    SyntheticModeComparisonConfig,
    build_mode_comparison_plan,
    run_synthetic_trial,
)
from pyosv.evaluation.synthetic_mode_comparison import runner as comparison_runner
from pyosv.evaluation.synthetic_quality import SyntheticSkinningConfig
from pyosv.evaluation.synthetic_quality.models import PipelineArtifacts
from pyosv.evaluation.synthetic_quality.runner import prepare_case_inputs, run_case_variant


def _plan():
    return build_mode_comparison_plan(SyntheticModeComparisonConfig(shape=(9, 9, 9)))


def test_trial_runs_canonical_cells_with_expected_shared_stage_cache_stats() -> None:
    plan = _plan()

    result = run_synthetic_trial(plan, plan.trials[0])

    expected_labels = tuple(cell.label for cell in plan.cells)
    assert tuple(cell.cell.label for cell in result.cells) == expected_labels
    assert tuple(result.report_payload) == expected_labels
    assert result.stage_cache_stats.seed_misses == 3
    assert result.stage_cache_stats.seed_hits == 3
    assert result.stage_cache_stats.voting_misses == 3
    assert result.stage_cache_stats.voting_hits == 3


def test_scanner_cells_reuse_prepared_arrays_without_pipeline_artifacts() -> None:
    plan = _plan()

    result = run_synthetic_trial(plan, plan.trials[0])
    by_label = {evaluation.cell.label: evaluation for evaluation in result.cells}

    assert not isinstance(by_label["RL-SCAN"].artifacts, PipelineArtifacts)
    assert not isinstance(by_label["Q-SCAN"].artifacts, PipelineArtifacts)
    assert isinstance(by_label["RL-REF"].artifacts, PipelineArtifacts)
    assert isinstance(by_label["Q-QUAL"].artifacts, PipelineArtifacts)
    for scanner_label, downstream_label in (
        ("RL-SCAN", "RL-REF"),
        ("RL-SCAN", "RL-QUAL"),
        ("Q-SCAN", "Q-REF"),
        ("Q-SCAN", "Q-QUAL"),
    ):
        scanner_volumes = by_label[scanner_label].artifacts
        downstream = by_label[downstream_label].artifacts
        assert isinstance(downstream, PipelineArtifacts)
        assert scanner_volumes["scanner_fet"] is downstream.volumes["scanner_fet"]
        assert scanner_volumes["scanner_fpt"] is downstream.volumes["scanner_fpt"]
        assert scanner_volumes["scanner_ftt"] is downstream.volumes["scanner_ftt"]


def test_execution_order_does_not_change_results_or_return_order(monkeypatch) -> None:
    plan = _plan()
    baseline = run_synthetic_trial(plan, plan.trials[0])
    monkeypatch.setattr(comparison_runner, "_execution_cells", lambda plan: plan.cells[::-1])

    reversed_result = run_synthetic_trial(plan, plan.trials[0])

    assert tuple(reversed_result.report_payload) == tuple(cell.label for cell in plan.cells)
    baseline_by_label = {cell.cell.label: cell for cell in baseline.cells}
    for cell in reversed_result.cells:
        expected = baseline_by_label[cell.cell.label]
        if isinstance(cell.artifacts, PipelineArtifacts):
            assert isinstance(expected.artifacts, PipelineArtifacts)
            assert cell.report_payload == expected.report_payload
            for name, volume in cell.artifacts.volumes.items():
                assert np.array_equal(volume, expected.artifacts.volumes[name])
        else:
            for name, volume in cell.artifacts.items():
                assert np.array_equal(volume, expected.artifacts[name])


def test_downstream_cells_match_standalone_case_variant_runs() -> None:
    plan = _plan()
    shared = run_synthetic_trial(plan, plan.trials[0])
    case = comparison_runner._build_trial_case(plan, plan.trials[0])
    prepared = prepare_case_inputs(
        case,
        scanner_config=plan.scanner_template,
        input_mode="both",
        scanner_backend_matrix=False,
        scanner_backends=("reference-like", "quality"),
    )

    for cell in plan.cells[2:]:
        settings = (
            plan.reference_workflow_settings
            if cell.workflow_mode == "reference"
            else plan.quality_workflow_settings
        )
        scanner_config = (
            plan.scanner_template
            if cell.scanner_backend is None
            else replace(plan.scanner_template, backend=cell.scanner_backend)
        )
        standalone = run_case_variant(
            case,
            voting_config=settings.voting_config,
            scanner_config=scanner_config,
            truth_metric_config=plan.truth_metric_config,
            skinning_config=settings.skinning_config,
            variant=plan.comparison_variant,
            input_mode=cell.input_mode,
            scanner_backend_matrix=False,
            include_thinning_diagnostic=False,
            include_scanner_downstream_diagnostics=False,
            prepared_inputs=prepared,
        )
        actual = next(item for item in shared.cells if item.cell.label == cell.label)
        assert actual.report_payload == standalone.report_payload
        assert isinstance(actual.artifacts, PipelineArtifacts)
        for name, volume in standalone.artifacts.volumes.items():
            assert np.array_equal(actual.artifacts.volumes[name], volume)


def test_all_cells_complete_with_skinning_disabled_contract() -> None:
    plan = build_mode_comparison_plan(
        SyntheticModeComparisonConfig(
            shape=(9, 9, 9),
            skinning_config=SyntheticSkinningConfig(enabled=False),
        )
    )

    result = run_synthetic_trial(plan, plan.trials[0])

    assert len(result.cells) == len(plan.cells)
    for cell in result.cells[2:]:
        assert cell.report_payload["skinning"]["enabled"] is False
        assert cell.report_payload["quality"]["skin"] is None
        assert isinstance(cell.artifacts, PipelineArtifacts)
        assert not np.any(cell.artifacts.volumes["skin_mask_py"])


def test_case_mismatch_fails_before_scanner_preparation(monkeypatch) -> None:
    plan = _plan()
    case = comparison_runner._build_trial_case(plan, plan.trials[0])
    object.__setattr__(case, "case_id", "single_dipping_plane")
    prepared = False

    def fail_if_prepared(*args, **kwargs):
        nonlocal prepared
        prepared = True
        raise AssertionError("scanner preparation must not run")

    monkeypatch.setattr(comparison_runner, "_build_trial_case", lambda plan, trial: case)
    monkeypatch.setattr(comparison_runner, "prepare_case_inputs", fail_if_prepared)

    with pytest.raises(ValueError, match="expected trial case"):
        run_synthetic_trial(plan, plan.trials[0])
    assert not prepared


def test_case_shape_mismatch_fails_before_scanner_preparation(monkeypatch) -> None:
    plan = _plan()
    case = comparison_runner._build_trial_case(plan, plan.trials[0])
    object.__setattr__(case, "shape", (11, 9, 9))
    prepared = False

    def fail_if_prepared(*args, **kwargs):
        nonlocal prepared
        prepared = True
        raise AssertionError("scanner preparation must not run")

    monkeypatch.setattr(comparison_runner, "_build_trial_case", lambda plan, trial: case)
    monkeypatch.setattr(comparison_runner, "prepare_case_inputs", fail_if_prepared)

    with pytest.raises(ValueError, match="expected trial shape"):
        run_synthetic_trial(plan, plan.trials[0])
    assert not prepared


def test_trial_shape_mismatch_fails_before_scanner_preparation(monkeypatch) -> None:
    plan = _plan()
    trial = replace(plan.trials[0], shape=(11, 9, 9))
    object.__setattr__(plan, "trials", (trial, *plan.trials[1:]))
    prepared = False

    def fail_if_prepared(*args, **kwargs):
        nonlocal prepared
        prepared = True
        raise AssertionError("scanner preparation must not run")

    monkeypatch.setattr(comparison_runner, "prepare_case_inputs", fail_if_prepared)

    with pytest.raises(ValueError, match="does not match plan shape"):
        run_synthetic_trial(plan, trial)
    assert not prepared


def test_cell_failure_clears_case_local_cache(monkeypatch) -> None:
    plan = _plan()
    caches = []

    class TrackingCache(comparison_runner.PipelineStageCache):
        def __post_init__(self, case) -> None:
            super().__post_init__(case)
            caches.append(self)

    monkeypatch.setattr(comparison_runner, "PipelineStageCache", TrackingCache)
    monkeypatch.setattr(
        comparison_runner,
        "run_case_variant",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cell failed")),
    )

    with pytest.raises(RuntimeError, match="cell failed"):
        run_synthetic_trial(plan, plan.trials[0])

    assert len(caches) == 1
    assert caches[0]._case is None
    assert not caches[0]._seeds
    assert not caches[0]._voting


def test_trial_and_cell_evaluations_are_frozen() -> None:
    plan = _plan()
    result = run_synthetic_trial(plan, plan.trials[0])

    with pytest.raises(FrozenInstanceError):
        result.trial = plan.trials[0]
    with pytest.raises(FrozenInstanceError):
        result.cells[0].artifacts = {}
