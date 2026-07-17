"""Tests for paired synthetic comparison contrasts and trial summaries."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pyosv.evaluation.synthetic_mode_comparison import (
    CONTRAST_DEFINITIONS,
    MetricRow,
    SyntheticModeComparisonConfig,
    aggregate_contrast_rows,
    aggregate_metric_rows,
    build_mode_comparison_plan,
    compute_contrast_rows,
    extract_trial_metric_rows,
    run_synthetic_trial,
)


def _row(
    cell_label: str,
    value: float,
    *,
    trial_id: str = "case__seed_1",
    seed: int | None = 1,
    stage: str = "fvt",
    selection: str = "top_truth_count",
    metric: str = "buffered_f1",
    unit: str = "fraction",
    direction: str = "higher",
    contrast_eligible: bool = True,
) -> MetricRow:
    axes = {
        "RL-SCAN": ("scanner-only", "scanner", "reference-like", None),
        "Q-SCAN": ("scanner-only", "scanner", "quality", None),
        "ORACLE-REF": ("oracle-workflow-isolation", "oracle", None, "reference"),
        "ORACLE-QUAL": ("oracle-workflow-isolation", "oracle", None, "quality"),
        "RL-REF": ("end-to-end", "scanner", "reference-like", "reference"),
        "RL-QUAL": ("end-to-end", "scanner", "reference-like", "quality"),
        "Q-REF": ("end-to-end", "scanner", "quality", "reference"),
        "Q-QUAL": ("end-to-end", "scanner", "quality", "quality"),
    }
    scope, input_mode, scanner_backend, workflow_mode = axes.get(
        cell_label, ("end-to-end", "scanner", "quality", "quality")
    )
    return MetricRow(
        schema_version=1,
        case_id="case",
        trial_id=trial_id,
        seed=seed,
        scope=scope,
        cell_label=cell_label,
        input_mode=input_mode,
        scanner_backend=scanner_backend,
        scanner_refinement_factor=2 if scanner_backend is not None else None,
        scanner_thin_mode="reference" if scanner_backend is not None else None,
        workflow_mode=workflow_mode,
        voter_thin_mode=None,
        skinner_method=None,
        variant="current_default",
        stage=stage,
        selection=selection,
        metric=metric,
        value=value,
        unit=unit,
        direction=direction,  # type: ignore[arg-type]
        contrast_eligible=contrast_eligible,
    )


def _all_cell_rows(*, direction: str = "higher") -> tuple[MetricRow, ...]:
    values = {
        "RL-SCAN": 1.0,
        "Q-SCAN": 4.0,
        "ORACLE-REF": 10.0,
        "ORACLE-QUAL": 13.0,
        "RL-REF": 2.0,
        "RL-QUAL": 8.0,
        "Q-REF": 6.0,
        "Q-QUAL": 16.0,
    }
    return tuple(_row(cell, value, direction=direction) for cell, value in values.items())


def test_all_fixed_contrast_formulas_and_component_order() -> None:
    rows = compute_contrast_rows(_all_cell_rows())

    assert tuple(row.contrast_name for row in rows) == tuple(
        definition.name for definition in CONTRAST_DEFINITIONS
    )
    assert {row.contrast_name: row.raw_value for row in rows} == {
        "scanner_only_effect": 3.0,
        "oracle_workflow_effect": 3.0,
        "scanner_effect_ref": 4.0,
        "scanner_effect_qual": 8.0,
        "workflow_effect_rl": 6.0,
        "workflow_effect_q": 10.0,
        "end_to_end_delta": 14.0,
        "scanner_main_effect": 6.0,
        "workflow_main_effect": 8.0,
        "scanner_workflow_interaction": 4.0,
    }
    assert all(row.raw_value == row.improvement_value for row in rows)
    assert tuple(row.component_cells for row in rows) == tuple(
        definition.component_cells for definition in CONTRAST_DEFINITIONS
    )


@pytest.mark.parametrize(
    ("direction", "expected"),
    (("higher", 3.0), ("lower", -3.0), ("neutral", None)),
)
def test_improvement_value_respects_metric_direction(
    direction: str, expected: float | None
) -> None:
    rows = compute_contrast_rows(
        (
            _row("RL-SCAN", 1.0, direction=direction),
            _row("Q-SCAN", 4.0, direction=direction),
        )
    )

    assert rows[0].raw_value == 3.0
    assert rows[0].improvement_value == expected


def test_stage_and_selection_are_never_paired() -> None:
    rows = (
        _row("RL-SCAN", 1.0, stage="scanner_raw", selection="top_truth_count"),
        _row("Q-SCAN", 2.0, stage="scanner_thinned", selection="top_truth_count"),
    )

    with pytest.raises(ValueError, match="missing required cell"):
        compute_contrast_rows(rows)


@pytest.mark.parametrize(
    "rows",
    (
        (_row("RL-SCAN", 1.0),),
        (_row("RL-SCAN", 1.0), _row("RL-SCAN", 2.0)),
        (
            _row("RL-SCAN", 1.0, seed=1, trial_id="case__seed_1"),
            _row("Q-SCAN", 2.0, seed=2, trial_id="case__seed_2"),
        ),
        (_row("RL-SCAN", 1.0), _row("Q-SCAN", 2.0, unit="voxel")),
        (_row("RL-SCAN", 1.0), _row("Q-SCAN", 2.0, direction="lower")),
    ),
)
def test_incomplete_duplicate_or_inconsistent_pairs_fail(rows: tuple[MetricRow, ...]) -> None:
    with pytest.raises(ValueError):
        compute_contrast_rows(rows)


def test_noneligible_backend_rows_are_ignored() -> None:
    row = _row("Q-SCAN", 0.5, contrast_eligible=False)

    assert compute_contrast_rows((row,)) == ()


def test_unknown_cells_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown cell"):
        compute_contrast_rows((_row("UNKNOWN", 1.0),))


def test_absolute_aggregate_uses_trials_as_replicates_and_fixed_quantiles() -> None:
    rows = tuple(
        _row("Q-QUAL", value, seed=seed, trial_id=f"case__seed_{seed}")
        for seed, value in zip((1, 2, 3, 4), (1.0, 2.0, 4.0, 8.0), strict=True)
    )

    aggregate = aggregate_metric_rows(rows)[0]

    assert aggregate.n == 4
    assert aggregate.mean == pytest.approx(3.75)
    assert aggregate.median == pytest.approx(3.0)
    assert aggregate.std == pytest.approx(np.std([1.0, 2.0, 4.0, 8.0], ddof=1))
    assert aggregate.min == 1.0
    assert aggregate.max == 8.0
    assert aggregate.q25 == pytest.approx(1.75)
    assert aggregate.q75 == pytest.approx(5.0)


def test_single_deterministic_trial_has_finite_zero_std_and_quartiles() -> None:
    aggregate = aggregate_metric_rows((_row("RL-REF", 2.5, trial_id="case", seed=None),))[0]

    assert aggregate.n == 1
    assert aggregate.std == 0.0
    assert aggregate.q25 == 2.5
    assert aggregate.q75 == 2.5
    assert all(
        np.isfinite(getattr(aggregate, name))
        for name in ("mean", "median", "std", "min", "max", "q25", "q75")
    )


def test_contrast_aggregate_is_mean_of_paired_trial_deltas() -> None:
    metric_rows = (
        _row("RL-SCAN", 0.0, seed=1, trial_id="case__seed_1"),
        _row("Q-SCAN", 10.0, seed=1, trial_id="case__seed_1"),
        _row("RL-SCAN", 100.0, seed=2, trial_id="case__seed_2"),
        _row("Q-SCAN", 102.0, seed=2, trial_id="case__seed_2"),
        _row("RL-SCAN", 1.0, seed=3, trial_id="case__seed_3"),
        _row("Q-SCAN", 5.0, seed=3, trial_id="case__seed_3"),
    )

    paired = compute_contrast_rows(metric_rows)
    aggregate = aggregate_contrast_rows(paired)[0]

    assert [row.raw_value for row in paired] == [10.0, 2.0, 4.0]
    assert aggregate.n == 3
    assert aggregate.mean == pytest.approx(16.0 / 3.0)
    assert aggregate.median == 4.0
    assert aggregate.std == pytest.approx(np.std([10.0, 2.0, 4.0], ddof=1))


def test_aggregate_order_does_not_depend_on_input_order() -> None:
    rows = (
        _row("Q-SCAN", 2.0, stage="scanner_raw"),
        _row("RL-SCAN", 1.0, stage="scanner_raw"),
        _row("Q-SCAN", 4.0, stage="scanner_thinned"),
        _row("RL-SCAN", 3.0, stage="scanner_thinned"),
    )
    contrasts = compute_contrast_rows(rows)

    assert aggregate_metric_rows(rows) == aggregate_metric_rows(tuple(reversed(rows)))
    assert aggregate_contrast_rows(contrasts) == aggregate_contrast_rows(tuple(reversed(contrasts)))


def test_duplicate_trials_and_mixed_aggregate_semantics_are_rejected() -> None:
    metric = _row("Q-QUAL", 1.0)
    with pytest.raises(ValueError, match="duplicate metric row"):
        aggregate_metric_rows((metric, replace(metric, value=2.0)))
    with pytest.raises(ValueError, match="mixed units or directions"):
        aggregate_metric_rows(
            (
                metric,
                _row("Q-QUAL", 2.0, seed=2, trial_id="case__seed_2", unit="voxel"),
            )
        )
    with pytest.raises(ValueError, match="duplicate trial seed"):
        aggregate_metric_rows((metric, _row("Q-QUAL", 2.0, seed=1, trial_id="case__other_seed_1")))


@pytest.mark.parametrize(
    ("case_id", "seeds", "expected_n"),
    (("single_vertical_plane", (7,), 1), ("weak_noisy_plane", (1, 2, 3), 3)),
)
def test_real_trials_have_complete_finite_contrasts_and_case_aggregates(
    case_id: str, seeds: tuple[int, ...], expected_n: int
) -> None:
    plan = build_mode_comparison_plan(
        SyntheticModeComparisonConfig(case_ids=(case_id,), trial_seeds=seeds, shape=(9, 9, 9))
    )
    metric_rows = tuple(
        row
        for trial in plan.trials
        for row in extract_trial_metric_rows(run_synthetic_trial(plan, trial))
    )

    contrasts = compute_contrast_rows(metric_rows)
    aggregates = aggregate_contrast_rows(contrasts)

    assert contrasts
    assert aggregates
    assert all(np.isfinite(row.raw_value) for row in contrasts)
    assert all(row.n == expected_n for row in aggregates)
    assert all(
        np.isfinite(getattr(row, name))
        for row in aggregates
        for name in ("mean", "median", "std", "min", "max", "q25", "q75")
    )
