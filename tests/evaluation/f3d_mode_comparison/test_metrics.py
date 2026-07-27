from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pytest

from pyosv.candidate_volume import (
    NONZERO_EPSILON,
    nonzero_count,
    nonzero_fraction,
    nonzero_mask,
    positive_candidate_mask,
)
from pyosv.evaluation.f3d_mode_comparison.metrics import (
    CONTRAST_DEFINITIONS,
    F3_METRIC_ROW_FIELDS,
    F3_METRIC_SCHEMA_VERSION,
    ContrastRow,
    MetricEvidence,
    MetricRow,
    VoxelwiseContrastSummary,
    compute_contrast_rows,
    compute_reference_metric_rows,
    compute_skin_metric_rows,
    compute_voxelwise_contrast_summaries,
)
from pyosv.metrics import (
    buffered_ridge_overlap,
    sparse_ridge_distance_metrics,
    top_percentile_overlap,
)

_FINGERPRINTS = {
    "RL-REF": "a" * 64,
    "RL-QUAL": "b" * 64,
    "Q-REF": "c" * 64,
    "Q-QUAL": "d" * 64,
}
_AXES = {
    "RL-REF": ("reference-like", "reference"),
    "RL-QUAL": ("reference-like", "quality"),
    "Q-REF": ("quality", "reference"),
    "Q-QUAL": ("quality", "quality"),
}


def _reference_rows(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    cell_label: str = "RL-REF",
    stage: str = "ft",
) -> tuple[tuple[MetricRow, ...], tuple[MetricEvidence, ...]]:
    scanner_backend, workflow_mode = _AXES[cell_label]
    return compute_reference_metric_rows(
        dataset_id="fixture",
        cell_label=cell_label,
        scanner_backend=scanner_backend,
        workflow_mode=workflow_mode,
        stage=stage,
        reference_file={"ft": "fl.dat", "fv": "fv.dat", "fvt": "fvt.dat"}[stage],
        candidate=candidate,
        reference=reference,
        source_stage_fingerprint=_FINGERPRINTS[cell_label],
        reference_sha256="e" * 64,
        slab_depth=1,
    )


def _value(
    rows: tuple[MetricRow, ...],
    metric: str,
    *,
    selection: str = "all",
) -> float | None:
    return next(row.value for row in rows if row.selection == selection and row.metric == metric)


def _metric_row(
    cell_label: str,
    value: float,
    *,
    direction: str = "higher",
    unit: str = "correlation",
    stage: str = "fvt",
    selection: str = "all",
    metric: str = "normalized_correlation",
) -> MetricRow:
    scanner_backend, workflow_mode = _AXES[cell_label]
    return MetricRow(
        F3_METRIC_SCHEMA_VERSION,
        "fixture",
        cell_label,
        scanner_backend,
        workflow_mode,
        stage,
        "full",
        selection,
        {"ft": "fl.dat", "fv": "fv.dat", "fvt": "fvt.dat"}[stage],
        metric,
        value,
        unit,
        direction,  # type: ignore[arg-type]
        True,
    )


def test_metric_row_field_order_and_full_region_contract() -> None:
    assert tuple(field.name for field in fields(MetricRow)) == F3_METRIC_ROW_FIELDS
    row = _metric_row("RL-REF", 0.5)

    assert tuple(row.as_dict()) == F3_METRIC_ROW_FIELDS
    assert row.region == "full"
    assert "crop" not in row.as_dict()
    assert "tile" not in row.as_dict()


def test_metric_artifacts_use_integer_schema_version_2_only() -> None:
    values = np.zeros((1, 1, 2), dtype=np.float32)
    metric_rows, evidence = _reference_rows(values, values)
    contrast_rows = compute_contrast_rows(
        tuple(_metric_row(cell, value) for cell, value in zip(_AXES, range(4), strict=True))
    )
    voxel_rows = compute_voxelwise_contrast_summaries(
        dataset_id="fixture",
        stage="fvt",
        volumes={cell: values for cell in _AXES},
        stage_fingerprints=_FINGERPRINTS,
        registration_id="fixture-registration",
        epsilon=0.0,
        slab_depth=1,
    )
    artifacts = (metric_rows[0], evidence[0], contrast_rows[0], voxel_rows[0])

    assert F3_METRIC_SCHEMA_VERSION == 2
    assert all(type(item.schema_version) is int and item.schema_version == 2 for item in artifacts)
    for item in artifacts:
        for invalid in (1, True, 2.0, "2"):
            with pytest.raises(ValueError, match="schema_version"):
                replace(item, schema_version=invalid)


def test_all_voxel_accumulator_matches_direct_numpy() -> None:
    reference = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    candidate = reference.copy()
    candidate[0, 0, 0] = 2.0
    candidate[1, 2, 3] = 20.0
    rows, evidence = _reference_rows(candidate, reference)
    difference = candidate.astype(np.float64) - reference
    absolute = np.abs(difference)

    assert _value(rows, "normalized_correlation") == pytest.approx(
        np.corrcoef(candidate.ravel(), reference.ravel())[0, 1]
    )
    assert _value(rows, "mean_absolute_difference") == pytest.approx(np.mean(absolute))
    assert _value(rows, "root_mean_square_difference") == pytest.approx(
        np.sqrt(np.mean(difference**2))
    )
    assert _value(rows, "absolute_difference_median") == pytest.approx(np.percentile(absolute, 50))
    assert _value(rows, "absolute_difference_p90") == pytest.approx(np.percentile(absolute, 90))
    assert _value(rows, "absolute_difference_p95") == pytest.approx(np.percentile(absolute, 95))
    assert _value(rows, "absolute_difference_p99") == pytest.approx(np.percentile(absolute, 99))
    assert _value(rows, "absolute_difference_max") == pytest.approx(np.max(absolute))
    all_evidence = next(item for item in evidence if item.selection == "all")
    assert dict(all_evidence.counts)["voxel_count"] == reference.size
    assert dict(all_evidence.accumulators)["absolute_difference_sum"] == pytest.approx(
        np.sum(absolute)
    )
    assert dict(all_evidence.accumulators)["absolute_difference_p95"] == pytest.approx(
        np.percentile(absolute, 95)
    )
    assert dict(all_evidence.accumulators)["absolute_difference_max"] == pytest.approx(
        np.max(absolute)
    )


def test_continuous_candidate_contract_and_all_voxel_evidence() -> None:
    epsilon = np.float32(NONZERO_EPSILON)
    above = np.nextafter(epsilon, np.float32(np.inf))
    values = np.array([0.0, epsilon, above, -epsilon, -above], dtype=np.float32)

    np.testing.assert_array_equal(nonzero_mask(values), [False, False, True, False, True])
    np.testing.assert_array_equal(
        positive_candidate_mask(values), [False, False, True, False, False]
    )
    assert nonzero_count(values) == 2
    assert nonzero_fraction(values) == 2 / 5

    candidate = np.array([0.0, 5.0e-8, -5.0e-8, 2.0e-6, -2.0e-6], dtype=np.float32)
    reference = np.array([0.0, 5.0e-8, -5.0e-8, 2.0e-6, 0.0], dtype=np.float32)
    rows, evidence = _reference_rows(candidate.reshape(1, 1, 5), reference.reshape(1, 1, 5))

    assert _value(rows, "candidate_nonzero_count") == 2
    assert _value(rows, "reference_nonzero_count") == 1
    assert _value(rows, "candidate_nonzero_fraction") == pytest.approx(2 / 5)
    assert _value(rows, "reference_nonzero_fraction") == pytest.approx(1 / 5)
    assert _value(rows, "nonzero_fraction_ratio") == 2.0
    all_evidence = next(item for item in evidence if item.selection == "all")
    assert dict(all_evidence.thresholds) == {"nonzero_epsilon": NONZERO_EPSILON}
    assert dict(all_evidence.counts)["candidate_nonzero_count"] == 2
    assert dict(all_evidence.counts)["reference_nonzero_count"] == 1


def test_nonzero_count_preserves_integer_and_boolean_semantics() -> None:
    assert nonzero_count(np.array([0, 1, -1], dtype=np.int32)) == 2
    assert nonzero_count(np.array([False, True, True], dtype=bool)) == 2
    np.testing.assert_array_equal(
        positive_candidate_mask(np.array([-1, 0, 1], dtype=np.int32)),
        [False, False, True],
    )
    np.testing.assert_array_equal(
        positive_candidate_mask(np.array([False, True], dtype=bool)),
        [False, True],
    )


def test_ridge_metrics_match_existing_helpers() -> None:
    reference = np.zeros((3, 4, 5), dtype=np.float32)
    candidate = np.zeros_like(reference)
    reference[1, 1, 1:4] = (1.0, 2.0, 3.0)
    candidate[1, 1, 2:5] = (1.0, 2.0, 3.0)
    rows, evidence = _reference_rows(candidate, reference)

    top = top_percentile_overlap(reference, candidate, 99, positive_only=True)
    assert _value(rows, "reference_count", selection="positive_p99") == top["a_count"]
    assert _value(rows, "candidate_count", selection="positive_p99") == top["b_count"]
    assert _value(rows, "jaccard", selection="positive_p99") == top["jaccard"]

    buffered = buffered_ridge_overlap(
        reference, candidate, percentile=99, radius=2, positive_only=True
    )
    for metric in (
        "candidate_in_reference_buffer_count",
        "reference_in_candidate_buffer_count",
        "buffered_precision",
        "buffered_recall",
        "buffered_f1",
    ):
        assert _value(rows, metric, selection="positive_p99_radius2") == pytest.approx(
            buffered[metric]
        )

    distance = sparse_ridge_distance_metrics(
        reference, candidate, percentile=99, positive_only=True
    )
    for metric in (
        "candidate_to_reference_mean",
        "candidate_to_reference_median",
        "candidate_to_reference_p90",
        "candidate_to_reference_p95",
        "reference_to_candidate_mean",
        "reference_to_candidate_median",
        "reference_to_candidate_p90",
        "reference_to_candidate_p95",
    ):
        assert _value(rows, metric, selection="positive_p99_distance") == pytest.approx(
            distance[metric]
        )
    distance_evidence = next(item for item in evidence if item.selection == "positive_p99_distance")
    distance_accumulators = dict(distance_evidence.accumulators)
    assert distance_accumulators["candidate_to_reference_distance_sum"] == pytest.approx(
        distance["candidate_to_reference_mean"] * distance["candidate_count"]
    )
    assert distance_accumulators["reference_to_candidate_p95"] == pytest.approx(
        distance["reference_to_candidate_p95"]
    )


def test_empty_and_constant_contracts() -> None:
    zeros = np.zeros((2, 3, 4), dtype=np.float32)
    rows, _ = _reference_rows(zeros, zeros)

    assert _value(rows, "normalized_correlation") == 0.0
    assert _value(rows, "candidate_to_reference_mean", selection="positive_p99_distance") is None
    distance_row = next(
        row
        for row in rows
        if row.selection == "positive_p99_distance" and row.metric == "candidate_to_reference_mean"
    )
    assert distance_row.as_dict()["value"] is None
    assert distance_row.as_dict(csv=True)["value"] == ""
    assert not distance_row.contrast_eligible


def test_wrong_mapping_shape_and_nonfinite_are_rejected() -> None:
    values = np.zeros((2, 3, 4), dtype=np.float32)
    kwargs = {
        "dataset_id": "fixture",
        "cell_label": "RL-REF",
        "scanner_backend": "reference-like",
        "workflow_mode": "reference",
        "stage": "ft",
        "reference_file": "fl.dat",
        "candidate": values,
        "reference": values,
        "source_stage_fingerprint": "a" * 64,
        "reference_sha256": "e" * 64,
    }
    with pytest.raises(ValueError, match="requires reference_file"):
        compute_reference_metric_rows(**{**kwargs, "reference_file": "fv.dat"})
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_reference_metric_rows(
            **{**kwargs, "reference": np.zeros((2, 3, 5), dtype=np.float32)}
        )
    nonfinite = values.copy()
    nonfinite[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_reference_metric_rows(**{**kwargs, "candidate": nonfinite})


def test_all_eight_scalar_contrasts_and_direction_transform() -> None:
    values = {
        "RL-REF": 2.0,
        "RL-QUAL": 8.0,
        "Q-REF": 6.0,
        "Q-QUAL": 16.0,
    }
    rows = tuple(_metric_row(cell, value) for cell, value in values.items())
    contrasts = compute_contrast_rows(rows)

    assert tuple(row.contrast_name for row in contrasts) == tuple(
        definition.name for definition in CONTRAST_DEFINITIONS
    )
    assert {row.contrast_name: row.raw_value for row in contrasts} == {
        "scanner_effect_ref": 4.0,
        "scanner_effect_qual": 8.0,
        "workflow_effect_rl": 6.0,
        "workflow_effect_q": 10.0,
        "end_to_end_delta": 14.0,
        "scanner_main_effect": 6.0,
        "workflow_main_effect": 8.0,
        "scanner_workflow_interaction": 4.0,
    }
    assert all(row.improvement_value == row.raw_value for row in contrasts)

    lower = compute_contrast_rows(
        tuple(
            _metric_row(
                cell,
                value,
                direction="lower",
                unit="value",
                metric="mean_absolute_difference",
            )
            for cell, value in values.items()
        )
    )
    assert all(row.improvement_value == -row.raw_value for row in lower)

    neutral = compute_contrast_rows(
        tuple(
            _metric_row(
                cell,
                value,
                direction="neutral",
                unit="value",
                metric="candidate_mean",
            )
            for cell, value in values.items()
        )
    )
    assert all(row.improvement_value is None for row in neutral)


def test_contrast_pairing_rejects_incomplete_or_mismatched_identity() -> None:
    with pytest.raises(ValueError, match="missing required"):
        compute_contrast_rows((_metric_row("RL-REF", 1.0),))

    rows = tuple(
        _metric_row(
            cell,
            value,
            selection="all" if cell != "Q-QUAL" else "positive_p99",
            metric=("normalized_correlation" if cell != "Q-QUAL" else "f1"),
            unit="correlation" if cell != "Q-QUAL" else "fraction",
        )
        for cell, value in zip(_AXES, (1.0, 2.0, 3.0, 4.0), strict=True)
    )
    with pytest.raises(ValueError, match="missing required"):
        compute_contrast_rows(rows)


def test_contrasts_skip_nullable_distance_group_with_mixed_empty_ridges() -> None:
    rows = tuple(
        MetricRow(
            F3_METRIC_SCHEMA_VERSION,
            "fixture",
            cell,
            *_AXES[cell],
            "fvt",
            "full",
            "positive_p99_distance",
            "fvt.dat",
            "candidate_to_reference_mean",
            None if cell == "RL-REF" else value,
            "voxel",
            "lower",
            cell != "RL-REF",
        )
        for cell, value in zip(_AXES, (0.0, 2.0, 3.0, 4.0), strict=True)
    )

    assert compute_contrast_rows(rows) == ()


def test_voxelwise_contrast_summary_matches_direct_numpy() -> None:
    base = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    volumes = {
        "RL-REF": base,
        "RL-QUAL": base + np.float32(1.0),
        "Q-REF": base * np.float32(2.0),
        "Q-QUAL": base * np.float32(2.0) + np.float32(3.0),
    }
    summaries = compute_voxelwise_contrast_summaries(
        dataset_id="fixture",
        stage="fvt",
        volumes=volumes,
        stage_fingerprints=_FINGERPRINTS,
        registration_id="fixture-registration",
        epsilon=0.5,
        slab_depth=1,
    )

    for definition, summary in zip(CONTRAST_DEFINITIONS, summaries, strict=True):
        direct = sum(
            coefficient * volumes[cell].astype(np.float64)
            for cell, coefficient in definition.coefficients
        )
        assert summary.mean == pytest.approx(np.mean(direct))
        assert summary.std == pytest.approx(np.std(direct))
        assert summary.mean_absolute == pytest.approx(np.mean(np.abs(direct)))
        assert summary.p95_absolute == pytest.approx(np.percentile(np.abs(direct), 95))
        assert summary.max_absolute == pytest.approx(np.max(np.abs(direct)))
        assert summary.epsilon_nonzero_fraction == pytest.approx(np.mean(np.abs(direct) > 0.5))
        assert not hasattr(summary, "volume")
        assert tuple(field.name for field in fields(summary)) == tuple(
            field.name for field in fields(VoxelwiseContrastSummary)
        )


def test_skin_rows_are_descriptive_and_have_no_reference_file() -> None:
    report = {
        "fingerprint": "a" * 64,
        "shape": [2, 3, 4],
        "enabled": True,
        "topology": {
            "skin_count": 2,
            "cell_count": 7,
            "unique_cell_count": 6,
            "duplicate_cell_count": 1,
            "largest_skin_size": 5,
            "largest_skin_fraction": 5 / 7,
            "small_skin_cell_count": 2,
            "small_skin_cell_fraction": 2 / 7,
        },
        "diagnostics": {
            "accepted_skin_count": 0,
            "fallback_enabled": True,
            "fallback_used": True,
            "fallback_skin_count": 2,
            "fallback_cell_count": 7,
        },
    }
    rows, evidence = compute_skin_metric_rows(
        dataset_id="fixture",
        cell_label="RL-REF",
        scanner_backend="reference-like",
        workflow_mode="reference",
        report=report,
        source_stage_fingerprint="a" * 64,
        shape=(2, 3, 4),
    )

    assert all(row.reference_file is None for row in rows)
    assert evidence[0].reference_file is None
    assert evidence[0].reference_sha256 is None
    assert dict(evidence[0].counts)["largest_skin_size"] == 5
    assert dict(evidence[0].counts)["small_skin_cell_count"] == 2
    assert not any("accuracy" in row.metric or "truth" in row.metric for row in rows)
    assert _value(rows, "duplicate_cell_count", selection="descriptive") == 1.0


def test_contrast_row_does_not_use_effect_or_accuracy_claim_fields() -> None:
    assert "accuracy" not in {field.name for field in fields(ContrastRow)}
