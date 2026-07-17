from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
import weakref
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyosv.cells import FaultCell
from pyosv.cli import synthetic_mode_comparison as comparison_cli
from pyosv.evaluation.synthetic_mode_comparison import (
    CONTRAST_DEFINITIONS,
    AggregateRow,
    ContrastRow,
    MetricRow,
    RuntimeRow,
    SyntheticModeComparisonConfig,
    build_mode_comparison_plan,
    run_mode_comparison,
    validate_completed_bundle,
    write_artifact_bundle,
)
from pyosv.evaluation.synthetic_mode_comparison import artifacts as comparison_artifacts
from pyosv.evaluation.synthetic_mode_comparison import runner as comparison_runner
from pyosv.evaluation.synthetic_quality import SyntheticSkinningConfig
from pyosv.evaluation.synthetic_quality import runner as quality_runner
from pyosv.evaluation.synthetic_quality import scanner as quality_scanner
from pyosv.evaluation.synthetic_quality.stage_cache import PipelineStageCacheStats
from pyosv.orient3d import FaultOrientScanner3
from pyosv.skin import FaultSkin

CASES = ("single_vertical_plane", "weak_noisy_plane")
SEEDS = (20260707, 20260708, 20260709)
SHAPE = (9, 9, 9)
EXPECTED_HASHED_BUNDLE_FILES = (
    "manifest.json",
    "cell_reports.json",
    "metrics_long.csv",
    "metric_aggregates.csv",
    "contrasts.csv",
    "contrast_aggregates.csv",
    "runtime.csv",
)
EXPECTED_BUNDLE_FILES = (*EXPECTED_HASHED_BUNDLE_FILES, "completion.json")


def _config() -> SyntheticModeComparisonConfig:
    return SyntheticModeComparisonConfig(case_ids=CASES, trial_seeds=SEEDS, shape=SHAPE)


def _fixed_clock() -> float:
    return 0.0


def test_public_experiment_integrates_shared_stages_metrics_and_aggregation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Counter[str]] = defaultdict(Counter)
    active_trial = ""
    active_backend = ""
    original_case_factory = comparison_runner._build_trial_case
    original_scanner_input = quality_runner.make_scanner_input_from_case
    original_backend_scan = quality_scanner.scan_backend_attributes
    original_ensemble_scan = quality_scanner.scan_ensemble_attributes
    original_thin = FaultOrientScanner3.thin

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

    def counted_thin(*args, **kwargs):
        calls[active_trial][f"{active_backend}_thin"] += 1
        return original_thin(*args, **kwargs)

    monkeypatch.setattr(comparison_runner, "_build_trial_case", counted_case_factory)
    monkeypatch.setattr(quality_runner, "make_scanner_input_from_case", counted_scanner_input)
    monkeypatch.setattr(quality_scanner, "scan_backend_attributes", counted_backend_scan)
    monkeypatch.setattr(quality_scanner, "scan_ensemble_attributes", counted_ensemble_scan)
    monkeypatch.setattr(FaultOrientScanner3, "thin", counted_thin)

    result = run_mode_comparison(_config(), clock=_fixed_clock)

    expected_trials = (
        "single_vertical_plane",
        "weak_noisy_plane__seed_20260707",
        "weak_noisy_plane__seed_20260708",
        "weak_noisy_plane__seed_20260709",
    )
    assert tuple(row["trial_id"] for row in result.trial_metadata) == expected_trials
    assert [len(report["cells"]) for report in result.cell_reports] == [8, 8, 8, 8]
    for trial_id in expected_trials:
        assert calls[trial_id] == Counter(
            {
                "case_factory": 1,
                "scanner_input": 1,
                "reference-like_scan": 1,
                "reference-like_thin": 1,
                "quality_scan": 1,
                "quality_thin": 1,
            }
        )

    for stats in result.cache_stats:
        assert (stats["seed_misses"], stats["seed_hits"]) == (3, 3)
        assert (stats["voting_misses"], stats["voting_hits"]) == (3, 3)
        assert (stats["thinning_misses"], stats["thinning_hits"]) == (6, 0)
        assert (stats["primary_skinning_misses"], stats["primary_skinning_hits"]) == (6, 0)

    metric_identities = {
        (row.case_id, row.trial_id, row.cell_label, row.stage, row.selection, row.metric)
        for row in result.metric_rows
    }
    assert len(metric_identities) == len(result.metric_rows)
    allowed_stages = {
        "scanner-only": {"scanner_raw", "scanner_thinned", "scanner_confidence"},
        "oracle-workflow-isolation": {"fv", "fvt", "skin"},
        "end-to-end": {"fv", "fvt", "skin"},
    }
    assert all(row.stage in allowed_stages[row.scope] for row in result.metric_rows)

    eligible_cells: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in result.metric_rows:
        if row.contrast_eligible:
            eligible_cells[
                (
                    row.case_id,
                    row.trial_id,
                    row.seed,
                    row.stage,
                    row.selection,
                    row.metric,
                    row.unit,
                    row.direction,
                )
            ].add(row.cell_label)
    expected_contrast_identities: Counter[tuple[Any, ...]] = Counter()
    for metric_identity, cells in eligible_cells.items():
        for definition in CONTRAST_DEFINITIONS:
            if set(definition.component_cells) <= cells:
                expected_contrast_identities[
                    (definition.name, *metric_identity, definition.component_cells)
                ] += 1
    actual_contrast_identities = Counter(
        (
            row.contrast_name,
            row.case_id,
            row.trial_id,
            row.seed,
            row.stage,
            row.selection,
            row.metric,
            row.unit,
            row.direction,
            row.component_cells,
        )
        for row in result.contrast_rows
    )
    assert actual_contrast_identities == expected_contrast_identities

    expected_n = {"single_vertical_plane": 1, "weak_noisy_plane": 3}
    for aggregate in (*result.metric_aggregates, *result.contrast_aggregates):
        assert aggregate.n == expected_n[aggregate.case_id]
    contrast_groups: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in result.contrast_rows:
        contrast_groups[
            (row.case_id, row.contrast_name, row.stage, row.selection, row.metric)
        ].append(row.raw_value)
    for aggregate in result.contrast_aggregates:
        key = (
            aggregate.case_id,
            aggregate.contrast_name,
            aggregate.stage,
            aggregate.selection,
            aggregate.metric,
        )
        assert aggregate.mean == pytest.approx(np.mean(contrast_groups[key]))

    _assert_scalar_only(result)


@pytest.mark.parametrize(
    ("include_oracle_workflow_isolation", "attribute_source_count"),
    [(True, 3), (False, 2)],
)
def test_cache_keys_separate_differing_thinning_and_skinning_configurations(
    include_oracle_workflow_isolation: bool,
    attribute_source_count: int,
) -> None:
    config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=SHAPE,
        include_oracle_workflow_isolation=include_oracle_workflow_isolation,
        skinning_config=SyntheticSkinningConfig(small_skin_size=2),
    )
    plan = build_mode_comparison_plan(config)
    assert (
        plan.reference_workflow_settings.voting_config.voter_thin_mode
        != plan.quality_workflow_settings.voting_config.voter_thin_mode
    )
    assert (
        plan.reference_workflow_settings.skinning_config
        != plan.quality_workflow_settings.skinning_config
    )

    result = run_mode_comparison(config, clock=_fixed_clock)

    assert len(result.cache_stats) == 1
    stats = result.cache_stats[0]
    assert (stats["seed_misses"], stats["seed_hits"]) == (
        attribute_source_count,
        attribute_source_count,
    )
    assert (stats["voting_misses"], stats["voting_hits"]) == (
        attribute_source_count,
        attribute_source_count,
    )
    assert (stats["thinning_misses"], stats["thinning_hits"]) == (
        attribute_source_count * 2,
        0,
    )
    assert (stats["primary_skinning_misses"], stats["primary_skinning_hits"]) == (
        attribute_source_count * 2,
        0,
    )


def test_cli_bundle_is_complete_valid_and_reproducible_with_a_fixed_clock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = tmp_path / "bundle"
    captured_configs: list[SyntheticModeComparisonConfig] = []
    captured_results: list[Any] = []

    def fixed_run(config: SyntheticModeComparisonConfig):
        captured_configs.append(config)
        result = run_mode_comparison(config, clock=_fixed_clock)
        captured_results.append(result)
        return result

    monkeypatch.setattr(comparison_cli, "run_mode_comparison", fixed_run)
    cli_args = [
        "--case-ids",
        ",".join(CASES),
        "--shape",
        ",".join(map(str, SHAPE)),
        "--trial-seeds",
        ",".join(map(str, SEEDS)),
    ]
    assert comparison_cli.main(["--output-dir", str(bundle), *cli_args]) == 0
    assert capsys.readouterr().out == f"{bundle}\n"
    assert {path.name for path in bundle.iterdir()} == set(EXPECTED_BUNDLE_FILES)
    assert validate_completed_bundle(bundle)

    result = captured_results[0]
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["input_config"]["case_ids"] == list(CASES)
    assert manifest["resolved_plan"]["shape"] == list(SHAPE)
    assert [cell["label"] for cell in manifest["canonical_cells"]] == [
        "RL-SCAN",
        "Q-SCAN",
        "ORACLE-REF",
        "ORACLE-QUAL",
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    ]
    assert [trial["case_generation_seed"] for trial in manifest["trials"]] == [
        None,
        *SEEDS,
    ]

    csv_contracts = {
        "metrics_long.csv": (MetricRow, result.metric_rows),
        "metric_aggregates.csv": (AggregateRow, result.metric_aggregates),
        "contrasts.csv": (ContrastRow, result.contrast_rows),
        "contrast_aggregates.csv": (AggregateRow, result.contrast_aggregates),
        "runtime.csv": (RuntimeRow, result.runtime_rows),
    }
    for filename, (model, expected_rows) in csv_contracts.items():
        with (bundle / filename).open(encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        assert rows[0] == [field.name for field in fields(model)]
        assert len(rows) - 1 == len(expected_rows)

    completion = json.loads((bundle / "completion.json").read_text(encoding="utf-8"))
    for filename in EXPECTED_HASHED_BUNDLE_FILES:
        payload = (bundle / filename).read_bytes()
        assert completion["files"][filename] == {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }

    second = tmp_path / "second"
    assert comparison_cli.main(["--output-dir", str(second), *cli_args]) == 0
    assert capsys.readouterr().out == f"{second}\n"
    assert captured_configs[0] == captured_configs[1]
    assert captured_results[0] is not captured_results[1]
    assert validate_completed_bundle(second)
    assert all(
        (bundle / filename).read_bytes() == (second / filename).read_bytes()
        for filename in EXPECTED_HASHED_BUNDLE_FILES
    )

    with (bundle / "runtime.csv").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="size mismatch"):
        validate_completed_bundle(bundle)


def test_volume_bearing_trial_evaluations_are_released_sequentially() -> None:
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

    result = run_mode_comparison(
        _config(),
        clock=_fixed_clock,
        trial_runner=fake_runner,
        metric_extractor=lambda evaluation: (),
    )

    gc.collect()
    assert maximum_live == 1
    assert len(live) == 0
    _assert_scalar_only(result)


def test_failures_do_not_publish_partial_experiment_or_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=SHAPE,
    )

    trial_calls = 0

    def fail_during_experiment(*args, **kwargs):
        nonlocal trial_calls
        trial_calls += 1
        if trial_calls == 2:
            raise RuntimeError("experiment failed")
        plan, trial = args[:2]
        evaluation = type("FakeEvaluation", (), {})()
        evaluation.trial = trial
        evaluation.report_payload = {cell.label: {} for cell in plan.cells}
        evaluation.stage_cache_stats = PipelineStageCacheStats(*(0,) * 8)
        return evaluation

    with pytest.raises(RuntimeError, match="experiment failed"):
        run_mode_comparison(
            _config(),
            clock=_fixed_clock,
            trial_runner=fail_during_experiment,
            metric_extractor=lambda evaluation: (),
        )
    assert trial_calls == 2

    with pytest.raises(RuntimeError, match="metric extraction failed"):
        run_mode_comparison(
            config,
            clock=_fixed_clock,
            metric_extractor=lambda evaluation: (_ for _ in ()).throw(
                RuntimeError("metric extraction failed")
            ),
        )

    result = run_mode_comparison(config, clock=_fixed_clock)
    output = tmp_path / "failed-bundle"
    original_write = comparison_artifacts._write_bytes
    writes = 0

    def failing_write(path: Path, payload: bytes) -> None:
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError("artifact write failed")
        original_write(path, payload)

    monkeypatch.setattr(comparison_artifacts, "_write_bytes", failing_write)
    with pytest.raises(OSError, match="artifact write failed"):
        write_artifact_bundle(result, output, config=config)

    assert not output.exists()
    assert not (output / "completion.json").exists()
    assert not list(tmp_path.glob(".failed-bundle.tmp-*"))


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
