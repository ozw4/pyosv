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

from pyosv.cells import FaultCell
from pyosv.evaluation.synthetic_mode_comparison import (
    SyntheticModeComparisonConfig,
    SyntheticModeComparisonResult,
    build_mode_comparison_plan,
    run_mode_comparison,
    validate_completed_bundle,
    validate_mode_comparison_result,
    write_artifact_bundle,
)
from pyosv.evaluation.synthetic_mode_comparison import metrics as comparison_metrics
from pyosv.evaluation.synthetic_mode_comparison import runner as comparison_runner
from pyosv.evaluation.synthetic_mode_comparison.metrics import scanner_metric_definitions
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
    _assert_scalar_only(result)

    calls_before_validation = {key: Counter(value) for key, value in calls.items()}
    validate_mode_comparison_result(result, config)
    bundle = write_artifact_bundle(result, tmp_path / "extended-smoke", config=config)
    assert validate_completed_bundle(bundle)
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

    validate_mode_comparison_result(topology_result, topology_config)
    bundle = write_artifact_bundle(
        topology_result,
        tmp_path / "topology-cases",
        config=topology_config,
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
        ),
    )
    for index, (config, result, expected) in enumerate(expected_by_config):
        plan = build_mode_comparison_plan(config)
        assert _expected_cache_counters(plan) == expected
        assert {name: result.cache_stats[0][name] for name in expected} == expected
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
        (quality_pipeline.OptimalSurfaceVoter, "apply_voting_from_seeds"),
        (quality_pipeline.OptimalSurfaceVoter, "thin"),
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
