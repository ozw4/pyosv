from __future__ import annotations

import gc
import hashlib
import io
import json
import weakref
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest

from pyosv import synthetic_metrics
from pyosv.cells import FaultCell
from pyosv.evaluation.synthetic_mode_comparison import (
    SyntheticModeComparisonConfig,
    SyntheticModeComparisonResult,
    aggregate_contrast_rows,
    aggregate_metric_rows,
    build_mode_comparison_plan,
    compute_contrast_rows,
    run_mode_comparison,
    validate_completed_bundle,
    validate_mode_comparison_result,
    write_artifact_bundle,
)
from pyosv.evaluation.synthetic_mode_comparison import artifacts as comparison_artifacts
from pyosv.evaluation.synthetic_mode_comparison import metrics as comparison_metrics
from pyosv.evaluation.synthetic_mode_comparison import runner as comparison_runner
from pyosv.evaluation.synthetic_mode_comparison.metrics import scanner_metric_definitions
from pyosv.evaluation.synthetic_mode_comparison.scalar_algebra import volume_diagonal
from pyosv.evaluation.synthetic_mode_comparison.validation import (
    _expected_cache_counters,
    _resolved_stage_keys_for_cell,
)
from pyosv.evaluation.synthetic_quality import (
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from pyosv.evaluation.synthetic_quality import pipeline as quality_pipeline
from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.evaluation.synthetic_quality import runner as quality_runner
from pyosv.evaluation.synthetic_quality import scanner as quality_scanner
from pyosv.evaluation.synthetic_quality.cases import CASE_IDS
from pyosv.evaluation.synthetic_quality.stage_cache import PipelineStageCacheStats
from pyosv.orient3d import FaultOrientScanner3
from pyosv.skin import FaultSkin

SHAPE = (9, 9, 9)
SEEDS = (20260707, 20260708)
SHARED_CONTRASTS = {
    "oracle_workflow_effect",
    "workflow_effect_rl",
    "workflow_effect_q",
}


def _fixed_clock() -> float:
    return 0.0


@pytest.fixture(scope="module")
def default_config() -> SyntheticModeComparisonConfig:
    return SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=SHAPE,
    )


@pytest.fixture(scope="module")
def default_result(default_config) -> SyntheticModeComparisonResult:
    return run_mode_comparison(default_config, clock=_fixed_clock)


@pytest.fixture(scope="module")
def shared_config() -> SyntheticModeComparisonConfig:
    return SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=SHAPE,
        voting_config=SyntheticVotingConfig(),
        skinner_method_explicit=True,
    )


@pytest.fixture(scope="module")
def shared_result(shared_config) -> SyntheticModeComparisonResult:
    return run_mode_comparison(shared_config, clock=_fixed_clock)


@pytest.fixture(scope="module")
def topology_config() -> SyntheticModeComparisonConfig:
    return SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane", "parallel_planes", "crossing_planes"),
        shape=SHAPE,
    )


@pytest.fixture(scope="module")
def topology_result(topology_config) -> SyntheticModeComparisonResult:
    return run_mode_comparison(topology_config, clock=_fixed_clock)


@pytest.fixture(scope="module")
def empty_skin_config() -> SyntheticModeComparisonConfig:
    return SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=SHAPE,
        skinning_config=SyntheticSkinningConfig(min_likelihood=1.5),
    )


@pytest.fixture(scope="module")
def empty_skin_result(empty_skin_config) -> SyntheticModeComparisonResult:
    return run_mode_comparison(empty_skin_config, clock=_fixed_clock)


@pytest.fixture(scope="module")
def radius_zero_config() -> SyntheticModeComparisonConfig:
    return SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=SHAPE,
        truth_metric_config=SyntheticTruthMetricConfig(buffer_radius=0.0),
    )


@pytest.fixture(scope="module")
def radius_zero_result(radius_zero_config) -> SyntheticModeComparisonResult:
    return run_mode_comparison(radius_zero_config, clock=_fixed_clock)


@pytest.fixture(scope="module")
def fractional_radius_config() -> SyntheticModeComparisonConfig:
    return SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=SHAPE,
        skinning_config=SyntheticSkinningConfig(enabled=False),
        truth_metric_config=SyntheticTruthMetricConfig(buffer_radius=0.5),
    )


@pytest.fixture(scope="module")
def fractional_radius_result(fractional_radius_config) -> SyntheticModeComparisonResult:
    return run_mode_comparison(fractional_radius_config, clock=_fixed_clock)


@pytest.fixture(scope="module")
def full_volume_radius_config() -> SyntheticModeComparisonConfig:
    return SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=SHAPE,
        skinning_config=SyntheticSkinningConfig(enabled=False),
        truth_metric_config=SyntheticTruthMetricConfig(buffer_radius=volume_diagonal(SHAPE)),
    )


@pytest.fixture(scope="module")
def full_volume_radius_result(full_volume_radius_config) -> SyntheticModeComparisonResult:
    return run_mode_comparison(full_volume_radius_config, clock=_fixed_clock)


def test_extended_smoke_fixes_trial_cell_scanner_and_cache_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = SyntheticModeComparisonConfig(
        case_set="extended",
        trial_seeds=SEEDS,
        shape=SHAPE,
        skinning_config=SyntheticSkinningConfig(enabled=False),
    )
    plan = build_mode_comparison_plan(config)
    calls: dict[str, Counter[str]] = defaultdict(Counter)
    active_trial = ""
    active_backend = ""
    original_case_factory = comparison_runner._build_trial_case
    original_scanner_input = quality_runner.make_scanner_input_from_case
    original_backend_scan = quality_scanner.scan_backend_attributes
    original_ensemble_scan = quality_scanner.scan_ensemble_attributes
    original_scanner_thin = FaultOrientScanner3.thin
    original_scanner_evidence = comparison_metrics.build_scanner_metric_evidence
    original_voting_evidence = quality_pipeline.build_voting_scalar_evidence
    original_thinning_evidence = quality_pipeline.build_thinning_scalar_evidence

    def counted_case_factory(plan, trial):
        nonlocal active_trial
        active_trial = trial.trial_id
        calls[active_trial]["case_factory"] += 1
        return original_case_factory(plan, trial)

    def counted_scanner_input(*args, **kwargs):
        calls[active_trial]["scanner_input"] += 1
        return original_scanner_input(*args, **kwargs)

    def counted_backend_scan(scanner, scanner_config, scanner_input, backend):
        nonlocal active_backend
        active_backend = backend
        calls[active_trial][f"{backend}_scan"] += 1
        return original_backend_scan(scanner, scanner_config, scanner_input, backend)

    def counted_ensemble_scan(*args, **kwargs):
        calls[active_trial]["ensemble_scan"] += 1
        return original_ensemble_scan(*args, **kwargs)

    def counted_scanner_thin(*args, **kwargs):
        calls[active_trial][f"{active_backend}_thin"] += 1
        return original_scanner_thin(*args, **kwargs)

    def counted_scanner_evidence(*args, **kwargs):
        backend = kwargs["scanner_backend"]
        calls[active_trial][f"{backend}_evidence"] += 1
        return original_scanner_evidence(*args, **kwargs)

    def counted_voting_evidence(*args, **kwargs):
        calls[active_trial]["voting_evidence"] += 1
        return original_voting_evidence(*args, **kwargs)

    def counted_thinning_evidence(*args, **kwargs):
        calls[active_trial]["thinning_evidence"] += 1
        return original_thinning_evidence(*args, **kwargs)

    monkeypatch.setattr(comparison_runner, "_build_trial_case", counted_case_factory)
    monkeypatch.setattr(quality_runner, "make_scanner_input_from_case", counted_scanner_input)
    monkeypatch.setattr(quality_scanner, "scan_backend_attributes", counted_backend_scan)
    monkeypatch.setattr(quality_scanner, "scan_ensemble_attributes", counted_ensemble_scan)
    monkeypatch.setattr(FaultOrientScanner3, "thin", counted_scanner_thin)
    monkeypatch.setattr(
        comparison_metrics,
        "build_scanner_metric_evidence",
        counted_scanner_evidence,
    )
    monkeypatch.setattr(
        quality_pipeline,
        "build_voting_scalar_evidence",
        counted_voting_evidence,
    )
    monkeypatch.setattr(
        quality_pipeline,
        "build_thinning_scalar_evidence",
        counted_thinning_evidence,
    )

    result = run_mode_comparison(config, clock=_fixed_clock)

    trial_counts = Counter(trial.case_id for trial in plan.trials)
    assert trial_counts == Counter({case_id: 1 for case_id in CASE_IDS}) + Counter(
        {"weak_noisy_plane": 1}
    )
    assert len(plan.trials) == len(CASE_IDS) + 1
    assert (
        tuple(trial.seed for trial in plan.trials if trial.case_id == "weak_noisy_plane") == SEEDS
    )
    assert all(
        tuple(report["cells"]) == tuple(cell.label for cell in plan.cells)
        for report in result.cell_reports
    )
    for trial in plan.trials:
        assert calls[trial.trial_id] == Counter(
            {
                "case_factory": 1,
                "scanner_input": 1,
                "reference-like_scan": 1,
                "reference-like_thin": 1,
                "reference-like_evidence": 1,
                "quality_scan": 1,
                "quality_thin": 1,
                "quality_evidence": 1,
                "voting_evidence": 3,
                "thinning_evidence": 6,
            }
        )
    _assert_scanner_publication_identity(result)
    expected_counters = {
        "seed_hits": 3,
        "seed_misses": 3,
        "voting_hits": 3,
        "voting_misses": 3,
        "thinning_hits": 0,
        "thinning_misses": 6,
        "primary_skinning_hits": 0,
        "primary_skinning_misses": 0,
    }
    assert _expected_cache_counters(plan) == expected_counters
    assert all(
        {name: row[name] for name in expected_counters} == expected_counters
        for row in result.cache_stats
    )
    expected_trial_stages = (
        "case_generation",
        "scanner_input_generation",
        "scanner_scan_thinning",
        "scanner_scan_thinning",
        "scanner_scalar_evidence",
        "scanner_scalar_evidence",
        "voting_scalar_evidence",
        "thinning_scalar_evidence",
        *("cell_execution" for _ in plan.cells),
        "metric_extraction",
        "contrast_extraction",
        "trial_total",
    )
    for trial in plan.trials:
        rows = tuple(row for row in result.runtime_rows if row.trial_id == trial.trial_id)
        assert tuple(row.stage for row in rows) == expected_trial_stages
        assert tuple(
            row.scanner_backend
            for row in rows
            if row.stage in {"scanner_scan_thinning", "scanner_scalar_evidence"}
        ) == ("reference-like", "quality", "reference-like", "quality")
        assert tuple(row.cell_label for row in rows if row.stage == "cell_execution") == tuple(
            cell.label for cell in plan.cells
        )
        evidence_calls = {
            row.stage: row.call_count
            for row in rows
            if row.stage in {"voting_scalar_evidence", "thinning_scalar_evidence"}
        }
        assert evidence_calls == {
            "voting_scalar_evidence": 3,
            "thinning_scalar_evidence": 6,
        }
        assert all(row.shared_stage for row in rows[:8])
        assert all(not row.shared_stage for row in rows[8 : 8 + len(plan.cells)])
        assert all(row.shared_stage for row in rows[8 + len(plan.cells) :])
    assert result.runtime_rows[-1].stage == "experiment_total"
    assert result.runtime_rows[-1].trial_id is None
    assert result.runtime_rows[-1].shared_stage is True
    assert len(result.runtime_rows) == len(plan.trials) * len(expected_trial_stages) + 1
    _assert_publication_scalar_contract(result)
    _assert_trial_truth_targets(result)
    _assert_scalar_only(result)

    calls_before_validation = {key: Counter(value) for key, value in calls.items()}
    validate_mode_comparison_result(result, config)
    bundle = write_artifact_bundle(result, tmp_path / "extended-smoke", config=config)
    assert validate_completed_bundle(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_schema_version"] == comparison_artifacts.ARTIFACT_SCHEMA_VERSION == 3
    assert (
        manifest["scalar_evidence_contract_version"]
        == comparison_artifacts.SCALAR_EVIDENCE_CONTRACT_VERSION
        == 3
    )
    assert (
        manifest["runtime_contract_version"] == comparison_artifacts.RUNTIME_CONTRACT_VERSION == 2
    )
    assert tuple(sorted(path.name for path in bundle.iterdir())) == tuple(
        sorted(comparison_artifacts.REQUIRED_BUNDLE_FILES)
    )
    assert calls == calls_before_validation


def test_skinning_cases_round_trip_complete_topology_algebra(
    topology_config: SyntheticModeComparisonConfig,
    topology_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    assert tuple(report["case_id"] for report in topology_result.cell_reports) == (
        "single_vertical_plane",
        "parallel_planes",
        "crossing_planes",
    )
    for report in topology_result.cell_reports:
        for label in (
            "ORACLE-REF",
            "ORACLE-QUAL",
            "RL-REF",
            "RL-QUAL",
            "Q-REF",
            "Q-QUAL",
        ):
            payload = report["cells"][label]
            assert payload["skinning"]["enabled"] is True
            assert payload["quality"]["skin"] is not None
            assert payload["pyosv"]["skins"] == payload["quality"]["skin"]["topology"]

    _assert_publication_scalar_contract(topology_result)
    _assert_trial_truth_targets(topology_result)

    validate_mode_comparison_result(topology_result, topology_config)
    bundle = write_artifact_bundle(
        topology_result,
        tmp_path / "topology-cases",
        config=topology_config,
    )
    assert validate_completed_bundle(bundle)


def test_buffer_radius_regimes_round_trip_with_deterministic_numerators(
    radius_zero_config: SyntheticModeComparisonConfig,
    radius_zero_result: SyntheticModeComparisonResult,
    fractional_radius_config: SyntheticModeComparisonConfig,
    fractional_radius_result: SyntheticModeComparisonResult,
    full_volume_radius_config: SyntheticModeComparisonConfig,
    full_volume_radius_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    regimes = (
        ("zero", radius_zero_config, radius_zero_result, "intersection"),
        (
            "fractional",
            fractional_radius_config,
            fractional_radius_result,
            "intersection",
        ),
        (
            "full-volume",
            full_volume_radius_config,
            full_volume_radius_result,
            "source",
        ),
    )
    for name, config, result, expected in regimes:
        for overlap in _downstream_overlap_reports(result):
            if expected == "intersection":
                assert overlap["candidate_in_truth_buffer_count"] == overlap["intersection_count"]
                assert overlap["truth_in_candidate_buffer_count"] == overlap["intersection_count"]
            else:
                assert overlap["candidate_in_truth_buffer_count"] == overlap["candidate_count"]
                assert overlap["truth_in_candidate_buffer_count"] == overlap["truth_count"]
        validate_mode_comparison_result(result, config)
        bundle = write_artifact_bundle(result, tmp_path / name, config=config)
        assert validate_completed_bundle(bundle)


@pytest.mark.parametrize(
    ("fixture_prefix", "message"),
    (
        ("fractional_radius", "fractional-radius buffered overlap counts"),
        ("full_volume_radius", "full-volume buffered overlap counts"),
    ),
)
def test_buffer_radius_regime_tampering_is_rejected_everywhere(
    fixture_prefix: str,
    message: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    config = request.getfixturevalue(f"{fixture_prefix}_config")
    result = request.getfixturevalue(f"{fixture_prefix}_result")
    reports = result.as_dict()["cell_reports"]
    payload = reports[0]["cells"]["Q-QUAL"]
    for downstream in (payload, payload["pipelines"][payload["active_pipeline"]]):
        overlap = downstream["quality"]["fv_top_truth_count"]["buffered_overlap_radius2"]
        if fixture_prefix == "fractional_radius":
            assert overlap["intersection_count"] < overlap["candidate_count"]
            overlap["candidate_in_truth_buffer_count"] = overlap["intersection_count"] + 1
        else:
            assert overlap["candidate_count"] > 0
            overlap["candidate_in_truth_buffer_count"] = overlap["candidate_count"] - 1
        overlap["buffered_precision"] = (
            overlap["candidate_in_truth_buffer_count"] / overlap["candidate_count"]
        )
        overlap["buffered_f1"] = _f1(overlap["buffered_precision"], overlap["buffered_recall"])

    _assert_report_tamper_rejected_in_memory_and_bundle(
        config=config,
        result=result,
        reports=reports,
        message=message,
        bundle_path=tmp_path / fixture_prefix,
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("stage_fault", "truth_evidence.fault_voxel_count"),
        ("stage_surface", "truth_evidence.surface_voxel_count"),
        ("trial_fault", "truth_evidence.fault_voxel_count"),
        ("trial_surface", "truth_evidence.surface_voxel_count"),
    ),
)
def test_truth_target_tampering_is_rejected_everywhere(
    tamper: str,
    message: str,
    default_config: SyntheticModeComparisonConfig,
    default_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    reports = default_result.as_dict()["cell_reports"]
    if tamper.startswith("trial_"):
        reports[0]["truth_evidence"][f"{tamper.removeprefix('trial_')}_voxel_count"] += 1
    else:
        payload = reports[0]["cells"]["Q-QUAL"]
        for downstream in (payload, payload["pipelines"][payload["active_pipeline"]]):
            if tamper == "stage_fault":
                quality_report = downstream["quality"]["fv_top_truth_count"]
                _increment_overlap_truth_count(quality_report["buffered_overlap_radius2"])
            else:
                quality_report = downstream["quality"]["fv_positive_top_truth_count"]
                quality_report["surface_distance"]["truth_count"] += 1

    _assert_report_tamper_rejected_in_memory_and_bundle(
        config=default_config,
        result=default_result,
        reports=reports,
        message=message,
        bundle_path=tmp_path / tamper,
    )


@pytest.mark.parametrize(
    "tamper",
    ("missing_shared", "extra_shared", "wrong_call_count", "cell_double_attribution"),
)
def test_runtime_contract_tampering_is_rejected_everywhere(
    tamper: str,
    default_config: SyntheticModeComparisonConfig,
    default_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    rows = list(default_result.runtime_rows)
    index = next(index for index, row in enumerate(rows) if row.stage == "voting_scalar_evidence")
    if tamper == "missing_shared":
        rows.pop(index)
    elif tamper == "extra_shared":
        rows.insert(index, rows[index])
    elif tamper == "wrong_call_count":
        rows[index] = replace(rows[index], call_count=rows[index].call_count + 1)
    else:
        cell = build_mode_comparison_plan(default_config).cells[0]
        duplicated_runtime = replace(
            rows[index],
            stage="cell_execution",
            cell_label=cell.label,
            scanner_backend=cell.scanner_backend,
            call_count=1,
            shared_stage=False,
        )
        cell_index = next(
            row_index
            for row_index, row in enumerate(rows)
            if row.stage == "cell_execution" and row.cell_label == cell.label
        )
        rows.insert(cell_index, duplicated_runtime)
    tampered_rows = tuple(rows)

    with pytest.raises(ValueError, match="runtime_rows"):
        validate_mode_comparison_result(
            replace(default_result, runtime_rows=tampered_rows), default_config
        )

    bundle = write_artifact_bundle(
        default_result,
        tmp_path / tamper,
        config=default_config,
    )
    runtime_path = bundle / comparison_artifacts.RUNTIME_FILE
    runtime_path.write_bytes(
        comparison_artifacts._csv_bytes(tampered_rows, comparison_artifacts.RuntimeRow)
    )
    _rehash(bundle, comparison_artifacts.RUNTIME_FILE)
    with pytest.raises(ValueError, match="runtime_rows"):
        validate_completed_bundle(bundle)


def test_empty_skin_control_preserves_overlap_degeneracy_and_round_trip(
    empty_skin_config: SyntheticModeComparisonConfig,
    empty_skin_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    empty_labels = []
    for label, payload in empty_skin_result.cell_reports[0]["cells"].items():
        if "quality" not in payload:
            continue
        skin = payload["quality"]["skin"]
        if skin["topology"]["skin_count"]:
            continue
        empty_labels.append(label)
        overlap = skin["buffered_overlap_radius2"]
        assert overlap["candidate_count"] == 0
        assert overlap["candidate_in_truth_buffer_count"] == 0
        assert overlap["truth_in_candidate_buffer_count"] == 0
        assert overlap["buffered_recall"] == 0.0
        assert overlap["buffered_f1"] == 0.0
        assert skin["component_topology"]["skins"] == ()
    assert empty_labels
    _assert_publication_scalar_contract(empty_skin_result)
    validate_mode_comparison_result(empty_skin_result, empty_skin_config)
    bundle = write_artifact_bundle(
        empty_skin_result,
        tmp_path / "empty-skin",
        config=empty_skin_config,
    )
    assert validate_completed_bundle(bundle)


def test_runtime_and_validator_use_resolved_default_and_explicit_cache_keys(
    default_config: SyntheticModeComparisonConfig,
    default_result: SyntheticModeComparisonResult,
    shared_config: SyntheticModeComparisonConfig,
    shared_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    expected_by_config = (
        (
            default_config,
            default_result,
            {
                "seed_hits": 3,
                "seed_misses": 3,
                "voting_hits": 3,
                "voting_misses": 3,
                "thinning_hits": 0,
                "thinning_misses": 6,
                "primary_skinning_hits": 0,
                "primary_skinning_misses": 6,
            },
            (3, 6),
        ),
        (
            shared_config,
            shared_result,
            {
                "seed_hits": 3,
                "seed_misses": 3,
                "voting_hits": 3,
                "voting_misses": 3,
                "thinning_hits": 3,
                "thinning_misses": 3,
                "primary_skinning_hits": 3,
                "primary_skinning_misses": 3,
            },
            (3, 3),
        ),
    )
    for index, (config, result, expected, evidence_calls) in enumerate(expected_by_config):
        plan = build_mode_comparison_plan(config)
        assert _expected_cache_counters(plan) == expected
        assert {name: result.cache_stats[0][name] for name in expected} == expected
        assert (
            tuple(
                row.call_count
                for row in result.runtime_rows
                if row.stage in {"voting_scalar_evidence", "thinning_scalar_evidence"}
            )
            == evidence_calls
        )
        validate_mode_comparison_result(result, config)
        bundle = write_artifact_bundle(result, tmp_path / f"cache-{index}", config=config)
        assert validate_completed_bundle(bundle)

    shared_plan = build_mode_comparison_plan(shared_config)
    assert (
        shared_plan.reference_workflow_settings.voting_config
        == shared_plan.quality_workflow_settings.voting_config
    )
    assert (
        shared_plan.reference_workflow_settings.skinning_config
        == shared_plan.quality_workflow_settings.skinning_config
    )


def test_shared_stage_evidence_and_zero_contrasts_follow_semantic_keys(
    default_config: SyntheticModeComparisonConfig,
    default_result: SyntheticModeComparisonResult,
    shared_config: SyntheticModeComparisonConfig,
    shared_result: SyntheticModeComparisonResult,
) -> None:
    default_cells = default_result.cell_reports[0]["cells"]
    for prefix in ("RL", "Q"):
        scanner_evidence = _scanner_evidence(default_cells[f"{prefix}-SCAN"])
        assert scanner_evidence == _scanner_evidence(default_cells[f"{prefix}-REF"])
        assert scanner_evidence == _scanner_evidence(default_cells[f"{prefix}-QUAL"])

    source_pairs = (
        ("ORACLE-REF", "ORACLE-QUAL"),
        ("RL-REF", "RL-QUAL"),
        ("Q-REF", "Q-QUAL"),
    )
    for left, right in source_pairs:
        assert _downstream_evidence(default_cells[left], "fv") == _downstream_evidence(
            default_cells[right], "fv"
        )

    default_plan = build_mode_comparison_plan(default_config)
    shared_plan = build_mode_comparison_plan(shared_config)
    shared_cells = shared_result.cell_reports[0]["cells"]
    for left, right in source_pairs:
        default_left = next(cell for cell in default_plan.cells if cell.label == left)
        default_right = next(cell for cell in default_plan.cells if cell.label == right)
        assert (
            _resolved_stage_keys_for_cell(default_plan, default_left).thinning
            != _resolved_stage_keys_for_cell(default_plan, default_right).thinning
        )

        shared_left = next(cell for cell in shared_plan.cells if cell.label == left)
        shared_right = next(cell for cell in shared_plan.cells if cell.label == right)
        assert (
            _resolved_stage_keys_for_cell(shared_plan, shared_left).thinning
            == _resolved_stage_keys_for_cell(shared_plan, shared_right).thinning
        )
        assert _downstream_evidence(shared_cells[left], "fvt") == _downstream_evidence(
            shared_cells[right], "fvt"
        )

    default_fv = tuple(
        row
        for row in default_result.contrast_rows
        if row.stage == "fv" and row.contrast_name in SHARED_CONTRASTS
    )
    shared_fvt = tuple(
        row
        for row in shared_result.contrast_rows
        if row.stage == "fvt" and row.contrast_name in SHARED_CONTRASTS
    )
    assert default_fv and all(row.raw_value == 0.0 for row in default_fv)
    assert shared_fvt and all(row.raw_value == 0.0 for row in shared_fvt)


@pytest.mark.parametrize(
    ("stage", "label", "message"),
    (
        ("scanner", "RL-QUAL", "shared attribute stage evidence"),
        ("fv", "Q-QUAL", "shared voting stage evidence"),
        ("fvt", "RL-QUAL", "shared thinning stage evidence"),
    ),
)
def test_coherent_cross_cell_tamper_is_rejected_in_memory_and_after_rehash(
    stage: str,
    label: str,
    message: str,
    default_config: SyntheticModeComparisonConfig,
    default_result: SyntheticModeComparisonResult,
    shared_config: SyntheticModeComparisonConfig,
    shared_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    config, result = (
        (shared_config, shared_result) if stage == "fvt" else (default_config, default_result)
    )
    reports = result.as_dict()["cell_reports"]
    _tamper_shared_subtree(reports, label=label, stage=stage)
    tampered = replace(result, cell_reports=tuple(reports))
    with pytest.raises(ValueError, match=message):
        validate_mode_comparison_result(tampered, config)

    bundle = write_artifact_bundle(result, tmp_path / stage, config=config)
    reports_path = bundle / "cell_reports.json"
    persisted = json.loads(reports_path.read_text(encoding="utf-8"))
    _tamper_shared_subtree(persisted, label=label, stage=stage)
    _write_json(reports_path, persisted)
    _rehash(bundle, "cell_reports.json")
    with pytest.raises(ValueError, match=message):
        validate_completed_bundle(bundle)


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        ("summary", "with-truth and without-truth skin counts"),
        ("per_truth", r"truth_components\[0\]\.recall"),
        ("per_skin", "truth and background cell counts"),
        ("topology_skin_count", "skin_count does not match skin topology"),
        ("topology_cell_count", "per-skin cell count does not match skin topology"),
    ),
)
def test_component_topology_tampering_is_rejected_in_memory_and_after_rehash(
    tamper: str,
    message: str,
    topology_config: SyntheticModeComparisonConfig,
    topology_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    reports = topology_result.as_dict()["cell_reports"]
    _tamper_component_topology(reports, tamper)
    tampered = replace(topology_result, cell_reports=tuple(reports))
    with pytest.raises(ValueError, match=message):
        validate_mode_comparison_result(tampered, topology_config)

    bundle = write_artifact_bundle(
        topology_result,
        tmp_path / f"component-{tamper}",
        config=topology_config,
    )
    reports_path = bundle / "cell_reports.json"
    persisted = json.loads(reports_path.read_text(encoding="utf-8"))
    _tamper_component_topology(persisted, tamper)
    _write_json(reports_path, persisted)
    _rehash(bundle, "cell_reports.json")
    with pytest.raises(ValueError, match=message):
        validate_completed_bundle(bundle)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("RL-SCAN", "scanner", "input", "finite_count"), 730),
        (("RL-SCAN", "scanner", "input", "finite_fraction"), 1.1),
        (("RL-SCAN", "scanner", "input", "mean"), -1.0),
        (
            (
                "RL-SCAN",
                "scanner_quality",
                "ft_top_truth_count",
                "buffered_overlap_radius2",
                "candidate_count",
            ),
            -1,
        ),
        (
            (
                "RL-SCAN",
                "scanner_quality",
                "ft_top_truth_count",
                "surface_distance",
                "candidate_to_truth_mean",
            ),
            -1.0,
        ),
        (
            (
                "RL-REF",
                "quality",
                "edge_false_positive",
                "fv_top_truth_count",
                "edge_candidate_fraction",
            ),
            1.1,
        ),
    ),
)
def test_rehashed_bundle_rejects_array_summary_and_report_scalar_constraints(
    path: tuple[str, ...],
    replacement: int | float,
    default_config: SyntheticModeComparisonConfig,
    default_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    bundle = write_artifact_bundle(default_result, tmp_path / "scalar", config=default_config)
    reports_path = bundle / "cell_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    _set_nested(reports[0]["cells"], path, replacement)
    _write_json(reports_path, reports)
    _rehash(bundle, "cell_reports.json")

    with pytest.raises(ValueError):
        validate_completed_bundle(bundle)


def test_rehashed_bundle_rejects_candidate_count_family_mismatch(
    default_config: SyntheticModeComparisonConfig,
    default_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    bundle = write_artifact_bundle(default_result, tmp_path / "count-family", config=default_config)
    reports_path = bundle / "cell_reports.json"
    reports = json.loads(reports_path.read_text(encoding="utf-8"))
    skin = reports[0]["cells"]["RL-REF"]["quality"]["skin"]
    mismatched = skin["orientation_error"]["count"] + 1
    skin["orientation_error"]["count"] = mismatched
    skin["topology"]["cell_count"] = mismatched
    _write_json(reports_path, reports)
    _rehash(bundle, "cell_reports.json")

    with pytest.raises(ValueError, match="candidate counts must match"):
        validate_completed_bundle(bundle)


def test_coherent_top_truth_count_family_tamper_is_rejected_everywhere(
    default_config: SyntheticModeComparisonConfig,
    default_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    tampered = _tamper_top_truth_count_family(default_result)

    with pytest.raises(ValueError, match="top_truth_count candidate_count must equal"):
        validate_mode_comparison_result(tampered, default_config)

    bundle = write_artifact_bundle(
        default_result,
        tmp_path / "coherent-count-family",
        config=default_config,
    )
    _replace_bundle_result_tables(bundle, tampered)
    with pytest.raises(ValueError, match="top_truth_count candidate_count must equal"):
        validate_completed_bundle(bundle)


def test_empty_candidate_positive_buffered_recall_tamper_is_rejected_everywhere(
    empty_skin_config: SyntheticModeComparisonConfig,
    empty_skin_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    reports = empty_skin_result.as_dict()["cell_reports"]
    payload = next(
        payload
        for payload in reports[0]["cells"].values()
        if "pipelines" in payload and payload["quality"]["skin"]["topology"]["skin_count"] == 0
    )
    for downstream in (payload, payload["pipelines"][payload["active_pipeline"]]):
        overlap = downstream["quality"]["skin"]["buffered_overlap_radius2"]
        overlap["truth_in_candidate_buffer_count"] = 1
        overlap["buffered_recall"] = 1.0 / overlap["truth_count"]
        overlap["buffered_f1"] = _f1(overlap["buffered_precision"], overlap["buffered_recall"])

    _assert_report_tamper_rejected_in_memory_and_bundle(
        config=empty_skin_config,
        result=empty_skin_result,
        reports=reports,
        message="nonempty candidate mask",
        bundle_path=tmp_path / "empty-buffered",
    )


def test_radius_zero_buffered_numerator_tamper_is_rejected_everywhere(
    radius_zero_config: SyntheticModeComparisonConfig,
    radius_zero_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    reports = radius_zero_result.as_dict()["cell_reports"]
    payload = reports[0]["cells"]["Q-QUAL"]
    for downstream in (payload, payload["pipelines"][payload["active_pipeline"]]):
        overlap = downstream["quality"]["fv_top_truth_count"]["buffered_overlap_radius2"]
        assert overlap["intersection_count"] < overlap["candidate_count"]
        overlap["candidate_in_truth_buffer_count"] = overlap["intersection_count"] + 1
        overlap["buffered_precision"] = (
            overlap["candidate_in_truth_buffer_count"] / overlap["candidate_count"]
        )
        overlap["buffered_f1"] = _f1(overlap["buffered_precision"], overlap["buffered_recall"])

    _assert_report_tamper_rejected_in_memory_and_bundle(
        config=radius_zero_config,
        result=radius_zero_result,
        reports=reports,
        message="radius-zero buffered overlap counts",
        bundle_path=tmp_path / "radius-zero",
    )


def test_nonempty_distance_above_volume_diagonal_is_rejected_everywhere(
    default_config: SyntheticModeComparisonConfig,
    default_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    reports = default_result.as_dict()["cell_reports"]
    payload = reports[0]["cells"]["Q-QUAL"]
    for downstream in (payload, payload["pipelines"][payload["active_pipeline"]]):
        distance = downstream["quality"]["fv_top_truth_count"]["surface_distance"]
        assert distance["candidate_count"] > 0
        assert distance["truth_count"] > 0
        distance["candidate_to_truth_p95"] = volume_diagonal(SHAPE) + 1.0

    _assert_report_tamper_rejected_in_memory_and_bundle(
        config=default_config,
        result=default_result,
        reports=reports,
        message="exceeds the volume diagonal",
        bundle_path=tmp_path / "distance-diagonal",
    )


def test_coherent_skin_summary_tamper_is_rejected_everywhere(
    default_config: SyntheticModeComparisonConfig,
    default_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    reports = default_result.as_dict()["cell_reports"]
    payload = reports[0]["cells"]["Q-QUAL"]
    for downstream in (payload, payload["pipelines"][payload["active_pipeline"]]):
        skin = downstream["quality"]["skin"]
        topology = skin["topology"]
        assert topology["largest_skin_size"] > 1
        topology["largest_skin_size"] -= 1
        topology["largest_skin_fraction"] = topology["largest_skin_size"] / topology["cell_count"]
        downstream["pyosv"]["skins"].update(topology)

    _assert_report_tamper_rejected_in_memory_and_bundle(
        config=default_config,
        result=default_result,
        reports=reports,
        message="does not match per-skin cell counts",
        bundle_path=tmp_path / "skin-summary",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("truth_surface_half_width", -0.1, "truth_surface_half_width must be non-negative"),
        ("truth_surface_half_width", np.inf, "truth_surface_half_width must be finite"),
        ("buffer_radius", np.nan, "buffer_radius must be finite"),
    ),
)
def test_invalid_truth_metric_scalar_prevents_work_and_artifact_creation(
    field: str,
    value: float,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operations = (
        Mock(),
        Mock(),
        Mock(),
        Mock(),
        Mock(),
    )
    monkeypatch.setattr(comparison_runner, "_build_trial_case", operations[0])
    monkeypatch.setattr(quality_runner, "make_scanner_input_from_case", operations[1])
    monkeypatch.setattr(quality_scanner, "scan_backend_attributes", operations[2])
    monkeypatch.setattr(
        quality_pipeline.OptimalSurfaceVoter, "apply_voting_from_seeds", operations[3]
    )
    monkeypatch.setattr(quality_pipeline, "find_synthetic_skins", operations[4])
    output = tmp_path / "invalid"

    def execute() -> None:
        config = SyntheticModeComparisonConfig(
            truth_metric_config=SyntheticTruthMetricConfig(**{field: value})
        )
        result = run_mode_comparison(config)
        write_artifact_bundle(result, output, config=config)

    with pytest.raises(ValueError, match=f"^{message}$"):
        execute()

    assert all(operation.call_count == 0 for operation in operations)
    assert not output.exists()
    assert not (output / "completion.json").exists()
    assert not list(tmp_path.glob(".invalid.tmp-*"))


def test_semantic_and_artifact_validation_never_reexecutes_volume_stages(
    monkeypatch: pytest.MonkeyPatch,
    default_config: SyntheticModeComparisonConfig,
    default_result: SyntheticModeComparisonResult,
    tmp_path: Path,
) -> None:
    operations = (
        (comparison_runner, "_build_trial_case"),
        (quality_runner, "make_scanner_input_from_case"),
        (quality_scanner, "scan_backend_attributes"),
        (comparison_metrics, "build_scanner_metric_evidence"),
        (quality_metrics, "scanner_truth_quality"),
        (synthetic_metrics, "distance_transform_edt"),
        (quality_pipeline.OptimalSurfaceVoter, "apply_voting_from_seeds"),
        (quality_pipeline.OptimalSurfaceVoter, "thin"),
        (quality_pipeline, "build_voting_scalar_evidence"),
        (quality_pipeline, "build_thinning_scalar_evidence"),
        (quality_pipeline, "find_synthetic_skins"),
        (comparison_metrics, "extract_trial_metric_rows"),
    )
    mocks = []
    for owner, name in operations:
        operation = Mock(side_effect=AssertionError(f"validation called {name}"))
        monkeypatch.setattr(owner, name, operation)
        mocks.append(operation)

    validate_mode_comparison_result(default_result, default_config)
    bundle = write_artifact_bundle(
        default_result,
        tmp_path / "validation-only",
        config=default_config,
    )
    assert validate_completed_bundle(bundle)
    assert all(operation.call_count == 0 for operation in mocks)


def test_volume_bearing_trial_evaluations_remain_sequential() -> None:
    live: weakref.WeakSet[Any] = weakref.WeakSet()
    maximum_live = 0
    zero_stats = PipelineStageCacheStats(*(0,) * 8)

    class FakeEvaluation:
        pass

    def fake_runner(plan, trial, *, clock, runtime_recorder):
        nonlocal maximum_live
        gc.collect()
        evaluation = FakeEvaluation()
        evaluation.trial = trial
        evaluation.report_payload = {cell.label: {} for cell in plan.cells}
        evaluation.stage_cache_stats = zero_stats
        evaluation.truth_evidence = {"fault_voxel_count": 1, "surface_voxel_count": 1}
        evaluation.volume = np.ones(SHAPE, dtype=np.float32)
        live.add(evaluation)
        maximum_live = max(maximum_live, len(live))
        return evaluation

    with pytest.raises(ValueError, match="runtime_rows"):
        run_mode_comparison(
            SyntheticModeComparisonConfig(
                case_ids=("weak_noisy_plane",),
                trial_seeds=SEEDS,
                shape=SHAPE,
            ),
            clock=_fixed_clock,
            trial_runner=fake_runner,
            metric_extractor=lambda evaluation: (),
        )

    gc.collect()
    assert maximum_live == 1
    assert len(live) == 0


def _assert_publication_scalar_contract(result: SyntheticModeComparisonResult) -> None:
    maximum_distance = volume_diagonal(SHAPE)
    distance_names = (
        "candidate_to_truth_mean",
        "candidate_to_truth_median",
        "candidate_to_truth_p90",
        "candidate_to_truth_p95",
        "truth_to_candidate_mean",
        "truth_to_candidate_median",
        "truth_to_candidate_p90",
        "truth_to_candidate_p95",
        "symmetric_chamfer_mean",
        "hausdorff_p95",
    )

    def assert_distance(distance: Mapping[str, Any]) -> None:
        assert all(0.0 <= distance[name] <= maximum_distance for name in distance_names)

    for report in result.cell_reports:
        for payload in report["cells"].values():
            for evidence in payload.get("scanner_metric_evidence", ()):
                quality = evidence.get("quality_report")
                if quality is None:
                    continue
                overlap = quality["buffered_overlap_radius2"]
                distance = quality["surface_distance"]
                assert overlap["candidate_count"] == distance["truth_count"]
                assert_distance(distance)

            quality = payload.get("quality")
            if quality is None:
                continue
            for stage in ("fv", "fvt"):
                top = quality[f"{stage}_top_truth_count"]
                positive = quality[f"{stage}_positive_top_truth_count"]
                assert (
                    top["buffered_overlap_radius2"]["candidate_count"]
                    == top["surface_distance"]["truth_count"]
                )
                assert (
                    positive["buffered_overlap_radius2"]["candidate_count"]
                    <= positive["surface_distance"]["truth_count"]
                )
                assert_distance(top["surface_distance"])
                assert_distance(positive["surface_distance"])

            skin = quality["skin"]
            if skin is None:
                continue
            assert_distance(skin["surface_distance"])
            topology = skin["topology"]
            sizes = [item["cell_count"] for item in skin["component_topology"]["skins"]]
            threshold = payload["config"]["skinning"]["small_skin_size"]
            small_sizes = [size for size in sizes if size < threshold]
            total_size = sum(sizes)
            assert topology["small_skin_size"] == threshold
            assert topology["cell_count"] == total_size
            assert topology["largest_skin_size"] == max(sizes, default=0)
            assert topology["largest_skin_fraction"] == (
                topology["largest_skin_size"] / total_size if total_size else 0.0
            )
            assert topology["small_skin_count"] == len(small_sizes)
            assert topology["small_skin_cell_count"] == sum(small_sizes)
            assert topology["small_skin_cell_fraction"] == (
                sum(small_sizes) / total_size if total_size else 0.0
            )


def _assert_trial_truth_targets(result: SyntheticModeComparisonResult) -> None:
    assert len(result.cell_reports) == len(result.trial_metadata)
    for report in result.cell_reports:
        assert tuple(report["truth_evidence"]) == (
            "fault_voxel_count",
            "surface_voxel_count",
        )
        evidence = report["truth_evidence"]
        for payload in report["cells"].values():
            scanner_quality = payload.get("scanner_quality")
            if scanner_quality is not None:
                _assert_report_truth_targets(scanner_quality["ft_top_truth_count"], evidence)
            for entry in payload.get("scanner_metric_evidence", ()):
                quality_report = entry.get("quality_report")
                if quality_report is not None:
                    _assert_report_truth_targets(quality_report, evidence)
            quality = payload.get("quality")
            if quality is None:
                continue
            for stage in ("fv", "fvt"):
                for selection in ("top_truth_count", "positive_top_truth_count"):
                    _assert_report_truth_targets(quality[f"{stage}_{selection}"], evidence)
            if quality["skin"] is not None:
                _assert_report_truth_targets(quality["skin"], evidence)


def _assert_report_truth_targets(report: Mapping[str, Any], evidence: Mapping[str, int]) -> None:
    assert report["buffered_overlap_radius2"]["truth_count"] == evidence["fault_voxel_count"]
    assert report["surface_distance"]["truth_count"] == evidence["surface_voxel_count"]


def _downstream_overlap_reports(
    result: SyntheticModeComparisonResult,
) -> tuple[Mapping[str, Any], ...]:
    reports = []
    for trial_report in result.cell_reports:
        for payload in trial_report["cells"].values():
            quality = payload.get("quality")
            if quality is None:
                continue
            for stage in ("fv", "fvt"):
                for selection in ("top_truth_count", "positive_top_truth_count"):
                    reports.append(quality[f"{stage}_{selection}"]["buffered_overlap_radius2"])
            if quality["skin"] is not None:
                reports.append(quality["skin"]["buffered_overlap_radius2"])
    return tuple(reports)


def _tamper_top_truth_count_family(
    result: SyntheticModeComparisonResult,
) -> SyntheticModeComparisonResult:
    reports = result.as_dict()["cell_reports"]
    payload = reports[0]["cells"]["Q-QUAL"]
    metric_values: dict[str, float] = {}
    for downstream in (payload, payload["pipelines"][payload["active_pipeline"]]):
        quality = downstream["quality"]
        selection = quality["fv_top_truth_count"]
        overlap = selection["buffered_overlap_radius2"]
        overlap["candidate_count"] += 1
        overlap["union_count"] += 1
        overlap["precision"] = overlap["intersection_count"] / overlap["candidate_count"]
        overlap["f1"] = _f1(overlap["precision"], overlap["recall"])
        overlap["jaccard"] = overlap["intersection_count"] / overlap["union_count"]
        overlap["buffered_precision"] = (
            overlap["candidate_in_truth_buffer_count"] / overlap["candidate_count"]
        )
        overlap["buffered_f1"] = _f1(overlap["buffered_precision"], overlap["buffered_recall"])
        selection["surface_distance"]["candidate_count"] += 1
        selection["orientation_error"]["count"] += 1
        edge = quality["edge_false_positive"]["fv_top_truth_count"]
        edge["candidate_count"] += 1
        edge["edge_candidate_fraction"] = edge["edge_candidate_count"] / edge["candidate_count"]
        edge["edge_false_positive_fraction_of_candidates"] = (
            edge["edge_false_positive_count"] / edge["candidate_count"]
        )
        metric_values = {
            "candidate_count": float(overlap["candidate_count"]),
            "buffered_precision": overlap["buffered_precision"],
            "buffered_f1": overlap["buffered_f1"],
            "edge_false_positive_fraction_of_candidates": edge[
                "edge_false_positive_fraction_of_candidates"
            ],
        }

    rows = tuple(
        replace(row, value=metric_values[row.metric])
        if (
            row.cell_label == "Q-QUAL"
            and row.stage == "fv"
            and row.selection == "top_truth_count"
            and row.metric in metric_values
        )
        else row
        for row in result.metric_rows
    )
    contrasts = compute_contrast_rows(rows)
    return replace(
        result,
        cell_reports=tuple(reports),
        metric_rows=rows,
        contrast_rows=contrasts,
        metric_aggregates=aggregate_metric_rows(rows),
        contrast_aggregates=aggregate_contrast_rows(contrasts),
    )


def _assert_report_tamper_rejected_in_memory_and_bundle(
    *,
    config: SyntheticModeComparisonConfig,
    result: SyntheticModeComparisonResult,
    reports: list[dict[str, Any]],
    message: str,
    bundle_path: Path,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_mode_comparison_result(replace(result, cell_reports=tuple(reports)), config)

    bundle = write_artifact_bundle(result, bundle_path, config=config)
    reports_path = bundle / "cell_reports.json"
    _write_json(reports_path, reports)
    _rehash(bundle, "cell_reports.json")
    with pytest.raises(ValueError, match=message):
        validate_completed_bundle(bundle)


def _replace_bundle_result_tables(bundle: Path, result: SyntheticModeComparisonResult) -> None:
    _write_json(bundle / "cell_reports.json", result.as_dict()["cell_reports"])
    rows_by_filename = {
        comparison_artifacts.METRICS_FILE: result.metric_rows,
        comparison_artifacts.METRIC_AGGREGATES_FILE: result.metric_aggregates,
        comparison_artifacts.CONTRASTS_FILE: result.contrast_rows,
        comparison_artifacts.CONTRAST_AGGREGATES_FILE: result.contrast_aggregates,
    }
    for filename, rows in rows_by_filename.items():
        model = comparison_artifacts._CSV_MODELS[filename]
        (bundle / filename).write_bytes(comparison_artifacts._csv_bytes(rows, model))
    for filename in ("cell_reports.json", *rows_by_filename):
        _rehash(bundle, filename)


def _f1(left: float, right: float) -> float:
    return 2.0 * left * right / (left + right) if left + right else 0.0


def _increment_overlap_truth_count(overlap: dict[str, Any]) -> None:
    overlap["truth_count"] += 1
    overlap["union_count"] = (
        overlap["candidate_count"] + overlap["truth_count"] - overlap["intersection_count"]
    )
    overlap["recall"] = overlap["intersection_count"] / overlap["truth_count"]
    overlap["f1"] = _f1(overlap["precision"], overlap["recall"])
    overlap["jaccard"] = overlap["intersection_count"] / overlap["union_count"]
    overlap["buffered_recall"] = overlap["truth_in_candidate_buffer_count"] / overlap["truth_count"]
    overlap["buffered_f1"] = _f1(overlap["buffered_precision"], overlap["buffered_recall"])


def _scanner_evidence(payload: Mapping[str, Any]) -> tuple[Any, Any]:
    return payload["scanner"], payload["scanner_quality"]


def _assert_scanner_publication_identity(result: SyntheticModeComparisonResult) -> None:
    labels_by_backend = {
        "reference-like": ("RL-SCAN", "RL-REF", "RL-QUAL"),
        "quality": ("Q-SCAN", "Q-REF", "Q-QUAL"),
    }
    for report in result.cell_reports:
        trial_id = report["trial_id"]
        cells = report["cells"]
        for backend, labels in labels_by_backend.items():
            definitions = scanner_metric_definitions(backend)
            expected = tuple(
                (definition.stage, definition.selection, definition.metric)
                for definition in definitions
            )
            evidence_by_label = tuple(cells[label]["scanner_metric_evidence"] for label in labels)
            assert all(evidence == evidence_by_label[0] for evidence in evidence_by_label[1:])
            for evidence in evidence_by_label:
                evidence_identities = tuple(
                    (entry["stage"], entry["selection"], entry["metric"]) for entry in evidence
                )
                assert evidence_identities == expected
            scanner_label = labels[0]
            row_identities = tuple(
                (row.stage, row.selection, row.metric)
                for row in result.metric_rows
                if row.trial_id == trial_id
                and row.cell_label == scanner_label
                and row.stage.startswith("scanner")
            )
            assert row_identities == expected

            confidence = tuple(
                identity for identity in expected if identity[0] == "scanner_confidence"
            )
            if backend == "reference-like":
                assert confidence == ()
            else:
                assert confidence == tuple(
                    ("scanner_confidence", selection, f"confidence_{summary}")
                    for selection in ("finite", "raw_top_truth_count")
                    for summary in ("mean", "median", "p95")
                )


def _downstream_evidence(payload: Mapping[str, Any], stage: str) -> tuple[Any, ...]:
    selections = (f"{stage}_top_truth_count", f"{stage}_positive_top_truth_count")
    evidence: tuple[Any, ...] = (
        payload["pyosv"][stage],
        *(payload["quality"][selection] for selection in selections),
        *(payload["quality"]["edge_false_positive"][selection] for selection in selections),
    )
    if stage == "fv":
        evidence += (payload["pyosv"]["voting"],)
    return evidence


def _tamper_shared_subtree(reports: list[dict[str, Any]], *, label: str, stage: str) -> None:
    payload = reports[0]["cells"][label]
    if stage == "scanner":
        payload["scanner"]["fet"]["mean"] += 0.01
        payload["pipelines"]["scanner"]["scanner"]["fet"]["mean"] += 0.01
        return
    payload["pyosv"][stage]["mean"] += 0.01
    payload["pipelines"]["scanner"]["pyosv"][stage]["mean"] += 0.01


def _tamper_component_topology(reports: list[dict[str, Any]], tamper: str) -> None:
    payload = reports[0]["cells"]["ORACLE-REF"]
    skin = payload["quality"]["skin"]
    component = skin["component_topology"]
    if tamper == "summary":
        component["skin_without_truth_count"] += 1
    elif tamper == "per_truth":
        truth = component["truth_components"][0]
        truth["recall"] = 0.0 if truth["recall"] else 0.5
    elif tamper == "per_skin":
        component["skins"][0]["background_cell_count"] += 1
    elif tamper == "topology_skin_count":
        skin_count = skin["topology"]["skin_count"] + 1
        skin["topology"]["skin_count"] = skin_count
        payload["pyosv"]["skins"]["skin_count"] = skin_count
    elif tamper == "topology_cell_count":
        item = component["skins"][0]
        item["cell_count"] += 1
        item["background_cell_count"] += 1
        item["purity"] = item["dominant_truth_cell_count"] / item["cell_count"]
        purities = [entry["purity"] for entry in component["skins"]]
        component["mean_skin_purity"] = sum(purities) / len(purities)
        component["min_skin_purity"] = min(purities)
    else:
        raise AssertionError(f"unknown component topology tamper: {tamper}")


def _set_nested(value: dict[str, Any], path: tuple[str, ...], replacement: Any) -> None:
    for name in path[:-1]:
        value = value[name]
    value[path[-1]] = replacement


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _rehash(bundle: Path, filename: str) -> None:
    completion_path = bundle / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    payload = (bundle / filename).read_bytes()
    completion["files"][filename] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    _write_json(completion_path, completion)


def _assert_scalar_only(value: Any) -> None:
    prohibited = (np.ndarray, FaultSkin, FaultCell, io.IOBase)
    seen: set[int] = set()

    def visit(item: Any) -> None:
        assert not isinstance(item, prohibited)
        if item is None or isinstance(item, (str, bytes, bool, int, float, np.generic)):
            return
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, Mapping):
            for key, nested in item.items():
                visit(key)
                visit(nested)
        elif isinstance(item, Sequence):
            for nested in item:
                visit(nested)

    visit(value)
