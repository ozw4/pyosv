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
from pyosv.evaluation.synthetic_quality import (
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
)
from pyosv.evaluation.synthetic_quality import pipeline as quality_pipeline
from pyosv.evaluation.synthetic_quality import runner as quality_runner
from pyosv.evaluation.synthetic_quality import scanner as quality_scanner
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


def test_trial_prepares_each_shared_scanner_stage_once(monkeypatch) -> None:
    plan = _plan()
    case_factory_calls = 0
    scanner_input_calls = 0
    scan_backends: list[str] = []
    scanner_thin_calls = 0
    scanner_cell_active = False
    voter_call_phases: list[bool] = []
    skinner_call_phases: list[bool] = []

    original_case_factory = comparison_runner._build_trial_case
    original_scanner_input = quality_runner.make_scanner_input_from_case
    original_backend_scan = quality_scanner.scan_backend_attributes
    original_scanner_thin = quality_scanner.FaultOrientScanner3.thin
    original_scanner_cell = comparison_runner._evaluate_scanner_cell
    original_voter_apply = quality_pipeline.OptimalSurfaceVoter.apply_voting_from_seeds
    original_skinner = quality_pipeline.find_synthetic_skins

    def counted_case_factory(*args, **kwargs):
        nonlocal case_factory_calls
        case_factory_calls += 1
        return original_case_factory(*args, **kwargs)

    def counted_scanner_input(*args, **kwargs):
        nonlocal scanner_input_calls
        scanner_input_calls += 1
        return original_scanner_input(*args, **kwargs)

    def counted_backend_scan(scanner, scanner_config, scanner_input, backend):
        scan_backends.append(backend)
        return original_backend_scan(scanner, scanner_config, scanner_input, backend)

    def counted_scanner_thin(*args, **kwargs):
        nonlocal scanner_thin_calls
        scanner_thin_calls += 1
        return original_scanner_thin(*args, **kwargs)

    def tracked_scanner_cell(*args, **kwargs):
        nonlocal scanner_cell_active
        scanner_cell_active = True
        try:
            return original_scanner_cell(*args, **kwargs)
        finally:
            scanner_cell_active = False

    def tracked_voter_apply(*args, **kwargs):
        voter_call_phases.append(scanner_cell_active)
        return original_voter_apply(*args, **kwargs)

    def tracked_skinner(*args, **kwargs):
        skinner_call_phases.append(scanner_cell_active)
        return original_skinner(*args, **kwargs)

    monkeypatch.setattr(comparison_runner, "_build_trial_case", counted_case_factory)
    monkeypatch.setattr(quality_runner, "make_scanner_input_from_case", counted_scanner_input)
    monkeypatch.setattr(quality_scanner, "scan_backend_attributes", counted_backend_scan)
    monkeypatch.setattr(quality_scanner.FaultOrientScanner3, "thin", counted_scanner_thin)
    monkeypatch.setattr(comparison_runner, "_evaluate_scanner_cell", tracked_scanner_cell)
    monkeypatch.setattr(
        quality_pipeline.OptimalSurfaceVoter,
        "apply_voting_from_seeds",
        tracked_voter_apply,
    )
    monkeypatch.setattr(quality_pipeline, "find_synthetic_skins", tracked_skinner)

    run_synthetic_trial(plan, plan.trials[0])

    assert case_factory_calls == 1
    assert scanner_input_calls == 1
    assert scan_backends == ["reference-like", "quality"]
    assert scanner_thin_calls == 2
    assert voter_call_phases and not any(voter_call_phases)
    assert skinner_call_phases and not any(skinner_call_phases)


def test_empty_truth_surface_fails_before_all_expensive_stages(monkeypatch) -> None:
    plan = build_mode_comparison_plan(
        SyntheticModeComparisonConfig(
            case_ids=("single_dipping_plane",),
            shape=(10, 10, 10),
            truth_metric_config=SyntheticTruthMetricConfig(
                truth_surface_half_width=0.0,
            ),
        )
    )
    calls = {
        "case_factory": 0,
        "stage_cache": 0,
        "scanner_input": 0,
        "scanner_constructor": 0,
        "reference-like_scan": 0,
        "quality_scan": 0,
        "scanner_thinning": 0,
        "voter": 0,
        "skinner": 0,
    }
    original_case_factory = comparison_runner._build_trial_case
    original_stage_cache = comparison_runner.PipelineStageCache
    original_scanner_input = quality_runner.make_scanner_input_from_case
    original_scanner_type = quality_scanner.FaultOrientScanner3
    original_backend_scan = quality_scanner.scan_backend_attributes
    original_scanner_thin = original_scanner_type.thin
    original_voter_apply = quality_pipeline.OptimalSurfaceVoter.apply_voting_from_seeds
    original_skinner = quality_pipeline.find_synthetic_skins

    def counted_case_factory(*args, **kwargs):
        calls["case_factory"] += 1
        return original_case_factory(*args, **kwargs)

    def counted_stage_cache(*args, **kwargs):
        calls["stage_cache"] += 1
        return original_stage_cache(*args, **kwargs)

    def counted_scanner_input(*args, **kwargs):
        calls["scanner_input"] += 1
        return original_scanner_input(*args, **kwargs)

    def counted_scanner_constructor(*args, **kwargs):
        calls["scanner_constructor"] += 1
        return original_scanner_type(*args, **kwargs)

    def counted_backend_scan(scanner, scanner_config, scanner_input, backend):
        calls[f"{backend}_scan"] += 1
        return original_backend_scan(scanner, scanner_config, scanner_input, backend)

    def counted_scanner_thin(*args, **kwargs):
        calls["scanner_thinning"] += 1
        return original_scanner_thin(*args, **kwargs)

    def counted_voter(*args, **kwargs):
        calls["voter"] += 1
        return original_voter_apply(*args, **kwargs)

    def counted_skinner(*args, **kwargs):
        calls["skinner"] += 1
        return original_skinner(*args, **kwargs)

    monkeypatch.setattr(comparison_runner, "_build_trial_case", counted_case_factory)
    monkeypatch.setattr(comparison_runner, "PipelineStageCache", counted_stage_cache)
    monkeypatch.setattr(quality_runner, "make_scanner_input_from_case", counted_scanner_input)
    monkeypatch.setattr(quality_scanner, "FaultOrientScanner3", counted_scanner_constructor)
    monkeypatch.setattr(quality_scanner, "scan_backend_attributes", counted_backend_scan)
    monkeypatch.setattr(original_scanner_type, "thin", counted_scanner_thin)
    monkeypatch.setattr(
        quality_pipeline.OptimalSurfaceVoter,
        "apply_voting_from_seeds",
        counted_voter,
    )
    monkeypatch.setattr(quality_pipeline, "find_synthetic_skins", counted_skinner)

    with pytest.raises(ValueError) as error:
        run_synthetic_trial(plan, plan.trials[0])

    message = str(error.value)
    assert "empty truth-surface support" in message
    assert "case_id='single_dipping_plane'" in message
    assert "trial_id='single_dipping_plane'" in message
    assert "shape=(10, 10, 10)" in message
    assert "truth_surface_half_width=0.0" in message
    assert calls == {
        "case_factory": 1,
        "stage_cache": 0,
        "scanner_input": 0,
        "scanner_constructor": 0,
        "reference-like_scan": 0,
        "quality_scan": 0,
        "scanner_thinning": 0,
        "voter": 0,
        "skinner": 0,
    }


def test_zero_width_truth_surface_runs_when_exact_support_exists() -> None:
    plan = build_mode_comparison_plan(
        SyntheticModeComparisonConfig(
            case_ids=("single_vertical_plane",),
            shape=(9, 9, 9),
            skinning_config=SyntheticSkinningConfig(enabled=False),
            truth_metric_config=SyntheticTruthMetricConfig(
                truth_surface_half_width=0.0,
            ),
        )
    )

    result = run_synthetic_trial(plan, plan.trials[0])

    assert len(result.cells) == len(plan.cells)


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
