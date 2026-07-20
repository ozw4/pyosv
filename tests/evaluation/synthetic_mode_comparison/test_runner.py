"""Tests for shared-stage synthetic mode-comparison trial execution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import numpy as np
import pytest

from pyosv.evaluation.synthetic_mode_comparison import (
    SyntheticModeComparisonConfig,
    build_mode_comparison_plan,
    run_synthetic_trial,
)
from pyosv.evaluation.synthetic_mode_comparison import experiment as comparison_experiment
from pyosv.evaluation.synthetic_mode_comparison import metrics as comparison_metrics
from pyosv.evaluation.synthetic_mode_comparison import runner as comparison_runner
from pyosv.evaluation.synthetic_mode_comparison.validation import (
    _resolved_stage_keys_for_cell,
)
from pyosv.evaluation.synthetic_quality import (
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from pyosv.evaluation.synthetic_quality import pipeline as quality_pipeline
from pyosv.evaluation.synthetic_quality import runner as quality_runner
from pyosv.evaluation.synthetic_quality import scanner as quality_scanner
from pyosv.evaluation.synthetic_quality.models import PipelineArtifacts
from pyosv.evaluation.synthetic_quality.runner import prepare_case_inputs, run_case_variant
from pyosv.evaluation.synthetic_quality.stage_cache import DownstreamScalarEvidenceCacheStats


def _plan():
    return build_mode_comparison_plan(SyntheticModeComparisonConfig(shape=(9, 9, 9)))


def _prepared_scalar_evidence(plan, case, prepared, backend="reference-like"):
    assert prepared.scanner is not None
    attributes = prepared.scanner.by_backend[backend]
    return comparison_runner._prepare_scanner_scalar_evidence(
        case=case,
        scanner_backend=backend,
        scanner_config=replace(plan.scanner_template, backend=backend),
        truth_metric_config=plan.truth_metric_config,
        scanner_report=attributes.report,
        scanner_volumes=attributes.volumes,
        metric_evidence_builder=comparison_metrics.build_scanner_metric_evidence,
    )


def _assert_type_sensitive_equal(actual, expected) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, dict):
        assert tuple(actual) == tuple(expected)
        for key in expected:
            _assert_type_sensitive_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_type_sensitive_equal(actual_item, expected_item)
    else:
        assert actual == expected


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


def test_trial_scalar_evidence_counts_follow_unique_semantic_keys(monkeypatch) -> None:
    plan = build_mode_comparison_plan(
        SyntheticModeComparisonConfig(
            shape=(9, 9, 9),
            skinning_config=SyntheticSkinningConfig(enabled=False),
        )
    )
    caches = []
    metric_calls = {
        name: 0
        for name in (
            "top_truth",
            "top_positive_truth",
            "overlap",
            "distance",
            "orientation",
            "edge",
        )
    }
    originals = {
        "top_truth": quality_pipeline.top_truth_count_mask,
        "top_positive_truth": quality_pipeline.top_positive_truth_count_mask,
        "overlap": quality_pipeline.buffered_surface_overlap,
        "distance": quality_pipeline.surface_distance_metrics,
        "orientation": quality_pipeline.masked_orientation_error,
        "edge": quality_pipeline.edge_false_positive_ratio,
    }

    class TrackingCache(comparison_runner.DownstreamScalarEvidenceCache):
        def __post_init__(self, case) -> None:
            super().__post_init__(case)
            caches.append(self)

    def counted(name):
        def operation(*args, **kwargs):
            metric_calls[name] += 1
            return originals[name](*args, **kwargs)

        return operation

    monkeypatch.setattr(comparison_runner, "DownstreamScalarEvidenceCache", TrackingCache)
    for name, attribute in (
        ("top_truth", "top_truth_count_mask"),
        ("top_positive_truth", "top_positive_truth_count_mask"),
        ("overlap", "buffered_surface_overlap"),
        ("distance", "surface_distance_metrics"),
        ("orientation", "masked_orientation_error"),
        ("edge", "edge_false_positive_ratio"),
    ):
        monkeypatch.setattr(quality_pipeline, attribute, counted(name))

    result = run_synthetic_trial(plan, plan.trials[0])

    assert len(caches) == 1
    assert caches[0].stats == DownstreamScalarEvidenceCacheStats(
        voting_builds=3,
        voting_reuses=3,
        thinning_builds=6,
        thinning_reuses=0,
    )
    assert metric_calls == {
        "top_truth": 9,
        "top_positive_truth": 9,
        "overlap": 18,
        "distance": 18,
        "orientation": 18,
        "edge": 18,
    }
    assert caches[0]._case is None
    assert not caches[0]._voting
    assert not caches[0]._thinning

    cells = result.report_payload
    for left, right in (
        ("ORACLE-REF", "ORACLE-QUAL"),
        ("RL-REF", "RL-QUAL"),
        ("Q-REF", "Q-QUAL"),
    ):
        _assert_type_sensitive_equal(cells[left]["pyosv"]["fv"], cells[right]["pyosv"]["fv"])
        for name in ("fv_top_truth_count", "fv_positive_top_truth_count"):
            _assert_type_sensitive_equal(
                cells[left]["quality"][name],
                cells[right]["quality"][name],
            )


def test_downstream_scalar_evidence_runtime_is_shared_and_cell_exclusive() -> None:
    plan = build_mode_comparison_plan(
        SyntheticModeComparisonConfig(
            shape=(9, 9, 9),
            skinning_config=SyntheticSkinningConfig(enabled=False),
        )
    )
    rows = []
    clock_value = -1.0

    class Recorder:
        def record_batch(self, batch) -> None:
            rows.extend(batch)

    def clock() -> float:
        nonlocal clock_value
        clock_value += 1.0
        return clock_value

    run_synthetic_trial(
        plan,
        plan.trials[0],
        clock=clock,
        runtime_recorder=Recorder(),
    )

    shared = {
        row["stage"]: row
        for row in rows
        if row["stage"] in {"voting_scalar_evidence", "thinning_scalar_evidence"}
    }
    assert shared["voting_scalar_evidence"] == {
        "stage": "voting_scalar_evidence",
        "elapsed_seconds": 3.0,
        "cell_label": None,
        "scanner_backend": None,
        "call_count": 3,
        "shared_stage": True,
    }
    assert shared["thinning_scalar_evidence"] == {
        "stage": "thinning_scalar_evidence",
        "elapsed_seconds": 6.0,
        "cell_label": None,
        "scanner_backend": None,
        "call_count": 6,
        "shared_stage": True,
    }
    cell_rows = [row for row in rows if row["stage"] == "cell_execution"]
    assert tuple(row["cell_label"] for row in cell_rows) == tuple(cell.label for cell in plan.cells)
    assert sum(row["elapsed_seconds"] for row in cell_rows) == 17.0


def test_matching_workflow_thinning_config_reuses_scalar_evidence(monkeypatch) -> None:
    plan = build_mode_comparison_plan(
        SyntheticModeComparisonConfig(
            shape=(9, 9, 9),
            voting_config=SyntheticVotingConfig(voter_thin_mode="reference"),
            skinning_config=SyntheticSkinningConfig(enabled=False),
        )
    )
    caches = []

    class TrackingCache(comparison_runner.DownstreamScalarEvidenceCache):
        def __post_init__(self, case) -> None:
            super().__post_init__(case)
            caches.append(self)

    monkeypatch.setattr(comparison_runner, "DownstreamScalarEvidenceCache", TrackingCache)

    rows = []

    class Recorder:
        def record_batch(self, batch) -> None:
            rows.extend(batch)

    result = run_synthetic_trial(
        plan,
        plan.trials[0],
        runtime_recorder=Recorder(),
    )

    assert caches[0].stats == DownstreamScalarEvidenceCacheStats(
        voting_builds=3,
        voting_reuses=3,
        thinning_builds=3,
        thinning_reuses=3,
    )
    thinning_runtime = next(row for row in rows if row["stage"] == "thinning_scalar_evidence")
    assert thinning_runtime["call_count"] == 3
    for left, right in (
        ("ORACLE-REF", "ORACLE-QUAL"),
        ("RL-REF", "RL-QUAL"),
        ("Q-REF", "Q-QUAL"),
    ):
        _assert_type_sensitive_equal(
            result.report_payload[left]["pyosv"]["fvt"],
            result.report_payload[right]["pyosv"]["fvt"],
        )


def test_without_oracle_isolation_voting_evidence_uses_two_attribute_keys(monkeypatch) -> None:
    plan = build_mode_comparison_plan(
        SyntheticModeComparisonConfig(
            shape=(9, 9, 9),
            include_oracle_workflow_isolation=False,
            skinning_config=SyntheticSkinningConfig(enabled=False),
        )
    )
    caches = []

    class TrackingCache(comparison_runner.DownstreamScalarEvidenceCache):
        def __post_init__(self, case) -> None:
            super().__post_init__(case)
            caches.append(self)

    monkeypatch.setattr(comparison_runner, "DownstreamScalarEvidenceCache", TrackingCache)

    rows = []

    class Recorder:
        def record_batch(self, batch) -> None:
            rows.extend(batch)

    run_synthetic_trial(
        plan,
        plan.trials[0],
        runtime_recorder=Recorder(),
    )

    assert caches[0].stats.voting_builds == 2
    assert caches[0].stats.voting_reuses == 2
    voting_runtime = next(row for row in rows if row["stage"] == "voting_scalar_evidence")
    assert voting_runtime["call_count"] == 2


def test_trial_records_one_immutable_truth_report_before_scanner_input(monkeypatch) -> None:
    plan = _plan()
    original = comparison_runner.quality_metrics.truth_report
    calls = 0

    def tracking_truth_report(case, truth_metric_config):
        nonlocal calls
        calls += 1
        return original(case, truth_metric_config)

    monkeypatch.setattr(comparison_runner.quality_metrics, "truth_report", tracking_truth_report)

    result = run_synthetic_trial(plan, plan.trials[0])

    assert calls == 1
    assert tuple(result.truth_evidence) == ("fault_voxel_count", "surface_voxel_count")
    assert result.truth_evidence["surface_voxel_count"] > 0
    with pytest.raises(TypeError):
        result.truth_evidence["surface_voxel_count"] = 0  # type: ignore[index]


def test_invalid_trial_truth_report_fails_before_scanner_input(monkeypatch) -> None:
    plan = _plan()
    prepared = False

    def fail_if_prepared(*args, **kwargs):
        nonlocal prepared
        prepared = True
        raise AssertionError("scanner preparation must not run")

    monkeypatch.setattr(
        comparison_runner.quality_metrics,
        "truth_report",
        lambda case, config: {"fault_voxel_count": 1, "surface_voxel_count": 0},
    )
    monkeypatch.setattr(comparison_runner, "prepare_case_inputs", fail_if_prepared)

    with pytest.raises(ValueError, match="empty truth-surface support"):
        run_synthetic_trial(plan, plan.trials[0])
    assert not prepared


def test_runtime_looks_up_the_resolved_keys_for_each_downstream_cell(monkeypatch) -> None:
    plan = _plan()
    looked_up = {stage: [] for stage in ("seed", "voting", "thinning", "primary_skinning")}

    class TrackingCache(comparison_runner.PipelineStageCache):
        def get_seed(self, key):
            looked_up["seed"].append(key)
            return super().get_seed(key)

        def get_voting(self, key):
            looked_up["voting"].append(key)
            return super().get_voting(key)

        def get_thinning(self, key):
            looked_up["thinning"].append(key)
            return super().get_thinning(key)

        def get_primary_skinning(self, key):
            looked_up["primary_skinning"].append(key)
            return super().get_primary_skinning(key)

    monkeypatch.setattr(comparison_runner, "PipelineStageCache", TrackingCache)

    run_synthetic_trial(plan, plan.trials[0])

    expected = tuple(
        _resolved_stage_keys_for_cell(plan, cell)
        for cell in plan.cells
        if cell.workflow_mode is not None
    )
    for stage in looked_up:
        assert looked_up[stage] == [getattr(keys, stage) for keys in expected]


def test_trial_prepares_each_shared_scanner_stage_once(monkeypatch) -> None:
    plan = _plan()
    case_factory_calls = 0
    scanner_input_calls = 0
    scan_backends: list[str] = []
    scanner_thin_calls = 0
    evidence_backends: list[str] = []
    scanner_cell_active = False
    voter_call_phases: list[bool] = []
    skinner_call_phases: list[bool] = []
    scanner_quality_calls = 0
    metric_calls = {name: 0 for name in ("top_k", "overlap", "distance", "orientation", "edge")}

    original_case_factory = comparison_runner._build_trial_case
    original_scanner_input = quality_runner.make_scanner_input_from_case
    original_backend_scan = quality_scanner.scan_backend_attributes
    original_scanner_thin = quality_scanner.FaultOrientScanner3.thin
    original_scanner_cell = comparison_runner._evaluate_scanner_cell
    original_evidence = comparison_metrics.build_scanner_metric_evidence
    original_voter_apply = quality_pipeline.OptimalSurfaceVoter.apply_voting_from_seeds
    original_skinner = quality_pipeline.find_synthetic_skins
    original_scanner_quality = quality_runner.quality_metrics.scanner_truth_quality
    original_metric_operations = {
        "top_k": comparison_metrics.top_truth_count_mask,
        "overlap": comparison_metrics.buffered_surface_overlap,
        "distance": comparison_metrics.surface_distance_metrics,
        "orientation": comparison_metrics.masked_orientation_error,
        "edge": comparison_metrics.edge_false_positive_ratio,
    }

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

    def counted_evidence(*args, **kwargs):
        evidence_backends.append(kwargs["scanner_backend"])
        return original_evidence(*args, **kwargs)

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

    def counted_scanner_quality(*args, **kwargs):
        nonlocal scanner_quality_calls
        scanner_quality_calls += 1
        return original_scanner_quality(*args, **kwargs)

    def counted_metric(name):
        def operation(*args, **kwargs):
            metric_calls[name] += 1
            return original_metric_operations[name](*args, **kwargs)

        return operation

    monkeypatch.setattr(comparison_runner, "_build_trial_case", counted_case_factory)
    monkeypatch.setattr(quality_runner, "make_scanner_input_from_case", counted_scanner_input)
    monkeypatch.setattr(quality_scanner, "scan_backend_attributes", counted_backend_scan)
    monkeypatch.setattr(quality_scanner.FaultOrientScanner3, "thin", counted_scanner_thin)
    monkeypatch.setattr(comparison_runner, "_evaluate_scanner_cell", tracked_scanner_cell)
    monkeypatch.setattr(comparison_metrics, "build_scanner_metric_evidence", counted_evidence)
    monkeypatch.setattr(
        quality_pipeline.OptimalSurfaceVoter,
        "apply_voting_from_seeds",
        tracked_voter_apply,
    )
    monkeypatch.setattr(quality_pipeline, "find_synthetic_skins", tracked_skinner)
    monkeypatch.setattr(
        quality_runner.quality_metrics,
        "scanner_truth_quality",
        counted_scanner_quality,
    )
    for name, attribute in (
        ("top_k", "top_truth_count_mask"),
        ("overlap", "buffered_surface_overlap"),
        ("distance", "surface_distance_metrics"),
        ("orientation", "masked_orientation_error"),
        ("edge", "edge_false_positive_ratio"),
    ):
        monkeypatch.setattr(comparison_metrics, attribute, counted_metric(name))

    run_synthetic_trial(plan, plan.trials[0])

    assert case_factory_calls == 1
    assert scanner_input_calls == 1
    assert scan_backends == ["reference-like", "quality"]
    assert scanner_thin_calls == 2
    assert evidence_backends == ["reference-like", "quality"]
    assert scanner_quality_calls == 0
    assert metric_calls == {name: 4 for name in metric_calls}
    assert voter_call_phases and not any(voter_call_phases)
    assert skinner_call_phases and not any(skinner_call_phases)


def test_invalid_truth_metric_config_prevents_all_trial_work(monkeypatch) -> None:
    case_factory = Mock()
    scanner_input = Mock()
    scanner_constructor = Mock()
    backend_scan = Mock()
    scanner_thinning = Mock()
    voter = Mock()
    skinner = Mock()
    result_constructor = Mock()
    scanner_type = quality_scanner.FaultOrientScanner3

    monkeypatch.setattr(comparison_runner, "_build_trial_case", case_factory)
    monkeypatch.setattr(quality_runner, "make_scanner_input_from_case", scanner_input)
    monkeypatch.setattr(scanner_type, "thin", scanner_thinning)
    monkeypatch.setattr(quality_scanner, "FaultOrientScanner3", scanner_constructor)
    monkeypatch.setattr(quality_scanner, "scan_backend_attributes", backend_scan)
    monkeypatch.setattr(
        quality_pipeline.OptimalSurfaceVoter,
        "apply_voting_from_seeds",
        voter,
    )
    monkeypatch.setattr(quality_pipeline, "find_synthetic_skins", skinner)
    monkeypatch.setattr(
        comparison_experiment,
        "SyntheticModeComparisonResult",
        result_constructor,
    )

    with pytest.raises(ValueError, match="^buffer_radius must be non-negative$"):
        comparison_experiment.run_mode_comparison(
            SyntheticModeComparisonConfig(
                truth_metric_config=SyntheticTruthMetricConfig(buffer_radius=-0.1),
            )
        )

    for operation in (
        case_factory,
        scanner_input,
        scanner_constructor,
        backend_scan,
        scanner_thinning,
        voter,
        skinner,
        result_constructor,
    ):
        operation.assert_not_called()


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
        actual_payload = dict(actual.report_payload)
        evidence = actual_payload.pop("scanner_metric_evidence", None)
        assert actual_payload == standalone.report_payload
        assert (evidence is not None) == (cell.scanner_backend is not None)
        assert isinstance(actual.artifacts, PipelineArtifacts)
        for name, volume in standalone.artifacts.volumes.items():
            assert np.array_equal(actual.artifacts.volumes[name], volume)


def test_prepared_scanner_scalar_evidence_matches_legacy_pipeline() -> None:
    plan = _plan()
    case = comparison_runner._build_trial_case(plan, plan.trials[0])
    prepared = prepare_case_inputs(
        case,
        scanner_config=plan.scanner_template,
        input_mode="scanner",
        scanner_backend_matrix=False,
    )
    evidence = _prepared_scalar_evidence(plan, case, prepared)
    settings = plan.reference_workflow_settings
    common = {
        "voting_config": settings.voting_config,
        "scanner_config": plan.scanner_template,
        "truth_metric_config": plan.truth_metric_config,
        "skinning_config": settings.skinning_config,
        "variant": plan.comparison_variant,
        "input_mode": "scanner",
        "scanner_backend_matrix": False,
        "include_thinning_diagnostic": False,
        "include_scanner_downstream_diagnostics": False,
        "prepared_inputs": prepared,
    }

    legacy = run_case_variant(case, **common)
    injected = run_case_variant(
        case,
        prepared_scanner_scalar_evidence=evidence,
        **common,
    )

    assert injected.report_payload == legacy.report_payload
    assert injected.report_payload["scanner"] is evidence.scanner_report
    assert injected.report_payload["scanner_quality"] is evidence.scanner_quality_report
    for name, volume in legacy.artifacts.volumes.items():
        assert np.array_equal(injected.artifacts.volumes[name], volume)


@pytest.mark.parametrize(
    ("evidence_changes", "message"),
    (
        ({"case_id": "other-case"}, "case"),
        ({"case_token": 0}, "case identity"),
        ({"shape": (8, 9, 9)}, "shape"),
        (
            {
                "scanner_backend": "quality",
                "scanner_config": replace(_plan().scanner_template, backend="quality"),
            },
            "backend",
        ),
        (
            {"truth_metric_config": SyntheticTruthMetricConfig(buffer_radius=1.0)},
            "truth metric config",
        ),
    ),
)
def test_prepared_scanner_scalar_evidence_mismatch_fails_before_voting(
    monkeypatch, evidence_changes, message
) -> None:
    plan = _plan()
    case = comparison_runner._build_trial_case(plan, plan.trials[0])
    prepared = prepare_case_inputs(
        case,
        scanner_config=plan.scanner_template,
        input_mode="both",
        scanner_backend_matrix=False,
    )
    evidence = replace(_prepared_scalar_evidence(plan, case, prepared), **evidence_changes)
    voting = Mock()
    monkeypatch.setattr(quality_runner, "run_voting_from_attributes", voting)

    with pytest.raises(ValueError, match=message):
        run_case_variant(
            case,
            voting_config=plan.reference_workflow_settings.voting_config,
            scanner_config=plan.scanner_template,
            truth_metric_config=plan.truth_metric_config,
            skinning_config=plan.reference_workflow_settings.skinning_config,
            variant=plan.comparison_variant,
            input_mode="both",
            scanner_backend_matrix=False,
            include_thinning_diagnostic=False,
            include_scanner_downstream_diagnostics=False,
            prepared_inputs=prepared,
            prepared_scanner_scalar_evidence=evidence,
        )

    voting.assert_not_called()


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


def test_scalar_evidence_failure_clears_case_local_caches(monkeypatch) -> None:
    plan = _plan()
    stage_caches = []
    scalar_caches = []
    runtime_batches = []

    class Recorder:
        def record_batch(self, batch) -> None:
            runtime_batches.append(batch)

    class TrackingStageCache(comparison_runner.PipelineStageCache):
        def __post_init__(self, case) -> None:
            super().__post_init__(case)
            stage_caches.append(self)

    class TrackingScalarCache(comparison_runner.DownstreamScalarEvidenceCache):
        def __post_init__(self, case) -> None:
            super().__post_init__(case)
            scalar_caches.append(self)

    monkeypatch.setattr(comparison_runner, "PipelineStageCache", TrackingStageCache)
    monkeypatch.setattr(comparison_runner, "DownstreamScalarEvidenceCache", TrackingScalarCache)
    monkeypatch.setattr(
        quality_pipeline,
        "build_thinning_scalar_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("evidence failed")),
    )

    with pytest.raises(RuntimeError, match="evidence failed"):
        run_synthetic_trial(
            plan,
            plan.trials[0],
            runtime_recorder=Recorder(),
        )

    assert runtime_batches == []
    assert len(stage_caches) == len(scalar_caches) == 1
    assert stage_caches[0]._case is None
    assert not stage_caches[0]._seeds
    assert not stage_caches[0]._voting
    assert not stage_caches[0]._thinning
    assert scalar_caches[0].stats.voting_builds == 1
    assert scalar_caches[0].stats.thinning_builds == 0
    assert scalar_caches[0]._case is None
    assert not scalar_caches[0]._voting
    assert not scalar_caches[0]._thinning


def test_trial_and_cell_evaluations_are_frozen() -> None:
    plan = _plan()
    result = run_synthetic_trial(plan, plan.trials[0])

    with pytest.raises(FrozenInstanceError):
        result.trial = plan.trials[0]
    with pytest.raises(FrozenInstanceError):
        result.cells[0].artifacts = {}
