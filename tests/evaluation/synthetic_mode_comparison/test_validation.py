from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pyosv.evaluation.synthetic_mode_comparison import (
    SyntheticModeComparisonConfig,
    aggregate_metric_rows,
    compute_contrast_rows,
    extract_trial_metric_rows,
    run_mode_comparison,
    validate_mode_comparison_result,
)


@pytest.fixture(scope="module")
def config() -> SyntheticModeComparisonConfig:
    return SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=(9, 9, 9),
    )


@pytest.fixture(scope="module")
def result(config):
    return run_mode_comparison(config)


def test_complete_result_passes_without_mutation(result, config) -> None:
    before = result.as_dict()

    assert validate_mode_comparison_result(result, config) is None
    assert result.as_dict() == before


def test_plan_metadata_numeric_type_tampering_is_rejected(result, config) -> None:
    plan_metadata = result.as_dict()["plan_metadata"]
    plan_metadata["shape"][0] = 9.0

    with pytest.raises(ValueError, match="plan_metadata does not match the canonical plan"):
        validate_mode_comparison_result(replace(result, plan_metadata=plan_metadata), config)


@pytest.mark.parametrize("value", (1.25, -0.5))
def test_fraction_value_outside_closed_unit_interval_is_rejected(result, config, value) -> None:
    index = next(
        index
        for index, row in enumerate(result.metric_rows)
        if row.metric == "array_nonzero_fraction"
    )
    rows = list(result.metric_rows)
    rows[index] = replace(rows[index], value=value)

    with pytest.raises(ValueError, match="closed unit interval"):
        validate_mode_comparison_result(replace(result, metric_rows=tuple(rows)), config)


@pytest.mark.parametrize("value", (-1.0, 1.5))
def test_invalid_count_value_is_rejected(result, config, value) -> None:
    index = next(
        index for index, row in enumerate(result.metric_rows) if row.metric == "candidate_count"
    )
    rows = list(result.metric_rows)
    rows[index] = replace(rows[index], value=value)

    with pytest.raises(ValueError, match="integer-valued count"):
        validate_mode_comparison_result(replace(result, metric_rows=tuple(rows)), config)


@pytest.mark.parametrize(
    ("field", "value"),
    (("unit", "score"), ("direction", "higher"), ("contrast_eligible", False)),
)
def test_metric_registry_tampering_is_rejected(result, config, field, value) -> None:
    rows = list(result.metric_rows)
    rows[0] = replace(rows[0], **{field: value})

    with pytest.raises(ValueError, match="metric row semantics"):
        validate_mode_comparison_result(replace(result, metric_rows=tuple(rows)), config)


def test_metric_metadata_tampering_is_rejected(result, config) -> None:
    index = next(
        index for index, row in enumerate(result.metric_rows) if row.cell_label == "RL-SCAN"
    )
    rows = list(result.metric_rows)
    rows[index] = replace(rows[index], workflow_mode="quality")

    with pytest.raises(ValueError, match="metric row metadata"):
        validate_mode_comparison_result(replace(result, metric_rows=tuple(rows)), config)


@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_metric_coverage_tampering_is_rejected(result, config, mutation) -> None:
    rows = (
        result.metric_rows[1:]
        if mutation == "missing"
        else (result.metric_rows[0], *result.metric_rows)
    )

    with pytest.raises(ValueError, match="metric_rows.*coverage"):
        validate_mode_comparison_result(replace(result, metric_rows=tuple(rows)), config)


def test_stale_contrast_after_metric_change_is_rejected(result, config) -> None:
    index = next(
        index for index, row in enumerate(result.metric_rows) if row.metric == "candidate_count"
    )
    rows = list(result.metric_rows)
    rows[index] = replace(rows[index], value=rows[index].value + 1.0)

    with pytest.raises(ValueError, match="contrast_rows"):
        validate_mode_comparison_result(replace(result, metric_rows=tuple(rows)), config)


def test_aggregate_tampering_is_rejected(result, config) -> None:
    aggregates = list(result.metric_aggregates)
    aggregates[0] = replace(aggregates[0], q25=aggregates[0].q25 + 1.0)

    with pytest.raises(ValueError, match="metric_aggregates"):
        validate_mode_comparison_result(
            replace(result, metric_aggregates=tuple(aggregates)), config
        )


def test_confidence_metric_and_aggregate_tampering_is_rejected(result, config) -> None:
    rows = list(result.metric_rows)
    index = next(
        index
        for index, row in enumerate(rows)
        if (
            row.cell_label,
            row.stage,
            row.selection,
            row.metric,
        )
        == ("Q-SCAN", "scanner_confidence", "finite", "confidence_mean")
    )
    rows[index] = replace(rows[index], value=rows[index].value + 0.01)
    tampered_rows = tuple(rows)

    with pytest.raises(ValueError, match="scalar evidence in cell_reports"):
        validate_mode_comparison_result(
            replace(
                result,
                metric_rows=tampered_rows,
                metric_aggregates=aggregate_metric_rows(tampered_rows),
            ),
            config,
        )


@pytest.mark.parametrize("field", ("cell_reports", "cache_stats", "runtime_rows"))
def test_top_level_coverage_tampering_is_rejected(result, config, field) -> None:
    bad = replace(result, **{field: getattr(result, field)[1:]})

    with pytest.raises(ValueError, match=field):
        validate_mode_comparison_result(bad, config)


def test_unknown_nested_cell_report_field_is_rejected(result, config) -> None:
    reports = result.as_dict()["cell_reports"]
    reports[0]["cells"]["RL-SCAN"]["scanner"]["unexpected"] = 1

    with pytest.raises(ValueError, match="scalar evidence in cell_reports.*unknown"):
        validate_mode_comparison_result(
            replace(result, cell_reports=tuple(reports)),
            config,
        )


def test_injected_numpy_array_in_cell_report_is_rejected(result, config) -> None:
    reports = result.as_dict()["cell_reports"]
    reports[0]["cells"]["RL-SCAN"]["scanner"]["input"]["mean"] = np.zeros(1)
    tampered = replace(result)
    object.__setattr__(tampered, "cell_reports", tuple(reports))

    with pytest.raises(ValueError, match="scalar evidence in cell_reports.*finite number"):
        validate_mode_comparison_result(tampered, config)


def test_run_rejects_invalid_custom_metric_extractor(config) -> None:
    def invalid_extractor(evaluation):
        rows = list(extract_trial_metric_rows(evaluation))
        rows[0] = replace(rows[0], value=1.25)
        return tuple(rows)

    with pytest.raises(ValueError, match="closed unit interval"):
        run_mode_comparison(config, metric_extractor=invalid_extractor)


def test_run_rejects_invalid_custom_contrast_builder(config) -> None:
    def invalid_builder(rows):
        contrasts = list(compute_contrast_rows(rows))
        contrasts[0] = replace(contrasts[0], raw_value=contrasts[0].raw_value + 1.0)
        return tuple(contrasts)

    with pytest.raises(ValueError, match="contrast_rows"):
        run_mode_comparison(config, contrast_builder=invalid_builder)


def test_run_rejects_invalid_custom_aggregator(config) -> None:
    def invalid_aggregator(rows):
        aggregates = list(aggregate_metric_rows(rows))
        aggregates[0] = replace(aggregates[0], mean=aggregates[0].mean + 1.0)
        return tuple(aggregates)

    with pytest.raises(ValueError, match="metric_aggregates"):
        run_mode_comparison(config, metric_aggregator=invalid_aggregator)
