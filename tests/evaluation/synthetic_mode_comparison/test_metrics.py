"""Tests for canonical synthetic mode-comparison metric rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from typing import Any

import numpy as np
import pytest

from pyosv.cells import FaultCell
from pyosv.evaluation.synthetic_mode_comparison import (
    METRIC_REGISTRY,
    MetricRow,
    SyntheticModeComparisonConfig,
    build_mode_comparison_plan,
    extract_trial_metric_rows,
    run_synthetic_trial,
)
from pyosv.evaluation.synthetic_quality import PipelineArtifacts, SyntheticSkinningConfig
from pyosv.evaluation.synthetic_quality.quality_metrics import EDGE_FALSE_POSITIVE_MARGIN
from pyosv.skin import FaultSkin
from pyosv.synthetic_metrics import (
    buffered_surface_overlap,
    edge_false_positive_ratio,
    masked_orientation_error,
    skin_mask_from_skins,
    skin_truth_metrics,
    surface_distance_metrics,
    top_positive_truth_count_mask,
    top_truth_count_mask,
)


def _evaluation(*, skinning_enabled: bool = True):
    plan = build_mode_comparison_plan(
        SyntheticModeComparisonConfig(
            case_ids=("single_vertical_plane",),
            shape=(9, 9, 9),
            skinning_config=SyntheticSkinningConfig(enabled=skinning_enabled),
        )
    )
    return run_synthetic_trial(plan, plan.trials[0])


def _quality_metrics(
    candidate: np.ndarray,
    *,
    truth_fault: np.ndarray,
    truth_surface: np.ndarray,
    predicted_strike: np.ndarray,
    predicted_dip: np.ndarray,
    truth_strike: np.ndarray,
    truth_dip: np.ndarray,
    buffer_radius: float,
) -> dict[str, Mapping[str, float | int]]:
    return {
        "buffered_overlap_radius2": buffered_surface_overlap(
            candidate, truth_fault, radius=buffer_radius
        ),
        "surface_distance": surface_distance_metrics(candidate, truth_surface),
        "orientation_error": masked_orientation_error(
            predicted_strike,
            predicted_dip,
            truth_strike,
            truth_dip,
            candidate,
        ),
    }


def _quality_row_values(
    quality: Mapping[str, Mapping[str, float | int]],
    edge: Mapping[str, float | int],
) -> dict[str, float]:
    overlap = quality["buffered_overlap_radius2"]
    distance = quality["surface_distance"]
    orientation = quality["orientation_error"]
    sources = {
        "candidate_count": overlap,
        "buffered_precision": overlap,
        "buffered_recall": overlap,
        "buffered_f1": overlap,
        "candidate_to_truth_median": distance,
        "candidate_to_truth_p95": distance,
        "truth_to_candidate_median": distance,
        "truth_to_candidate_p95": distance,
        "hausdorff_p95": distance,
        "strike_median": orientation,
        "strike_p95": orientation,
        "dip_median": orientation,
        "dip_p95": orientation,
        "edge_false_positive_fraction_of_candidates": edge,
    }
    return {metric: float(source[metric]) for metric, source in sources.items()}


def _skin(indices: Sequence[tuple[int, int, int]]) -> FaultSkin:
    return FaultSkin.from_cells(
        FaultCell(float(i1), float(i2), float(i3), 0.8, 179.0, 45.0) for i1, i2, i3 in indices
    )


def _handcrafted_evaluation(*, empty_skins: bool = False):
    evaluation = _evaluation()
    shape = evaluation.trial.shape
    buffer_radius = evaluation.truth_metric_config.buffer_radius

    truth_distance = np.full(shape, 4.0, dtype=np.float32)
    truth_fault = np.zeros(shape, dtype=np.float32)
    truth_fault_id = np.zeros(shape, dtype=np.int32)
    component_one = ((4, 4, 4), (5, 4, 4))
    component_two = ((4, 5, 4), (5, 5, 4))
    for i1, i2, i3 in (*component_one, *component_two):
        truth_distance[i3, i2, i1] = 0.0
        truth_fault[i3, i2, i1] = 1.0
    for i1, i2, i3 in component_one:
        truth_fault_id[i3, i2, i1] = 1
    for i1, i2, i3 in component_two:
        truth_fault_id[i3, i2, i1] = 2
    truth_surface = truth_distance == 0.0
    truth_strike = np.ones(shape, dtype=np.float32)
    truth_dip = np.full(shape, 40.0, dtype=np.float32)
    predicted_strike = np.full(shape, 179.0, dtype=np.float32)
    predicted_dip = np.full(shape, 45.0, dtype=np.float32)

    scanner_ft = np.zeros(shape, dtype=np.float32)
    scanner_fet = np.zeros(shape, dtype=np.float32)
    scanner_ft[0, 0, 0] = 10.0
    scanner_ft[4, 4, 4] = 9.0
    scanner_ft[4, 4, 5] = 8.0
    scanner_ft[4, 5, 4] = 7.0
    scanner_fet[8, 8, 8] = 10.0
    scanner_fet[4, 4, 4] = 9.0
    scanner_fet[4, 4, 5] = 8.0
    scanner_fet[4, 5, 5] = 7.0
    confidence = np.linspace(0.0, 1.0, num=np.prod(shape), dtype=np.float32).reshape(shape)
    scanner_artifacts = {
        "scanner_ft": scanner_ft,
        "scanner_pt": predicted_strike,
        "scanner_tt": predicted_dip,
        "scanner_fet": scanner_fet,
        "scanner_fpt": predicted_strike,
        "scanner_ftt": predicted_dip,
        "scanner_confidence": confidence,
    }
    scanner_cell = replace(evaluation.cells[1], artifacts=scanner_artifacts)

    fv = np.zeros(shape, dtype=np.float32)
    for value, (i1, i2, i3) in enumerate(((0, 0, 0), *component_one, *component_two), start=1):
        fv[i3, i2, i1] = float(value)
    fvt = np.zeros(shape, dtype=np.float32)
    stage_arrays = {"fv": fv, "fvt": fvt}
    quality: dict[str, Any] = {"edge_false_positive": {}}
    expected: dict[str, dict[tuple[str, str, str], float]] = {
        scanner_cell.cell.label: {},
        "Q-QUAL": {},
    }

    for stage, values in stage_arrays.items():
        expected["Q-QUAL"][(stage, "all", "array_nonzero_fraction")] = float(
            np.count_nonzero(values) / values.size
        )
        for selection, candidate in (
            ("top_truth_count", top_truth_count_mask(values, truth_surface)),
            (
                "positive_top_truth_count",
                top_positive_truth_count_mask(values, truth_surface),
            ),
        ):
            block = _quality_metrics(
                candidate,
                truth_fault=truth_fault.astype(bool),
                truth_surface=truth_surface,
                predicted_strike=predicted_strike,
                predicted_dip=predicted_dip,
                truth_strike=truth_strike,
                truth_dip=truth_dip,
                buffer_radius=buffer_radius,
            )
            edge = edge_false_positive_ratio(
                candidate,
                truth_surface,
                edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
                truth_buffer_radius=buffer_radius,
            )
            report_key = f"{stage}_{selection}"
            quality[report_key] = block
            quality["edge_false_positive"][report_key] = edge
            expected["Q-QUAL"].update(
                {
                    (stage, selection, metric): value
                    for metric, value in _quality_row_values(block, edge).items()
                }
            )

    skins = (
        []
        if empty_skins
        else [
            _skin(((4, 4, 4), (4, 5, 4), (0, 0, 0))),
            _skin(((5, 4, 4),)),
            _skin(((5, 5, 4),)),
        ]
    )
    skin_metrics = skin_truth_metrics(
        skins,
        shape=shape,
        truth_fault_mask=truth_fault.astype(bool),
        truth_surface_mask=truth_surface,
        truth_strike=truth_strike,
        truth_dip=truth_dip,
        buffer_radius=buffer_radius,
        small_skin_size=10,
        truth_fault_id=truth_fault_id,
    )
    skin_edge = edge_false_positive_ratio(
        skin_mask_from_skins(skins, shape),
        truth_surface,
        edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        truth_buffer_radius=buffer_radius,
    )
    quality["skin"] = skin_metrics
    quality["edge_false_positive"]["skin"] = skin_edge
    expected["Q-QUAL"].update(
        {
            ("skin", "skin_cells", metric): value
            for metric, value in _quality_row_values(skin_metrics, skin_edge).items()
        }
    )
    for source in (skin_metrics["topology"], skin_metrics["component_topology"]):
        expected["Q-QUAL"].update(
            {
                ("skin", "skin_cells", metric): float(source[metric])
                for metric in (
                    "skin_count",
                    "largest_skin_fraction",
                    "small_skin_cell_fraction",
                    "duplicate_cell_count",
                    "covered_truth_component_count",
                    "uncovered_truth_component_count",
                    "over_merge_skin_count",
                    "over_split_truth_component_count",
                    "mean_skin_purity",
                    "min_skin_purity",
                    "mean_truth_component_recall",
                    "min_truth_component_recall",
                )
                if metric in source
            }
        )

    volumes = {
        "truth_fault_mask": truth_fault,
        "truth_distance": truth_distance,
        "truth_strike": truth_strike,
        "truth_dip": truth_dip,
        "fv_py": fv,
        "vp_py": predicted_strike,
        "vt_py": predicted_dip,
        "fvt_py": fvt,
        "skin_mask_py": skin_mask_from_skins(skins, shape).astype(np.float32),
    }
    downstream = next(cell for cell in evaluation.cells if cell.cell.label == "Q-QUAL")
    downstream_cell = replace(
        downstream,
        report_payload={
            "quality": quality,
            "skinning": {"enabled": True},
        },
        artifacts=PipelineArtifacts(volumes=volumes, skins_payload={}),
    )

    for stage, values in (("scanner_raw", scanner_ft), ("scanner_thinned", scanner_fet)):
        candidate = top_truth_count_mask(values, truth_surface)
        block = _quality_metrics(
            candidate,
            truth_fault=truth_fault.astype(bool),
            truth_surface=truth_surface,
            predicted_strike=predicted_strike,
            predicted_dip=predicted_dip,
            truth_strike=truth_strike,
            truth_dip=truth_dip,
            buffer_radius=buffer_radius,
        )
        edge = edge_false_positive_ratio(
            candidate,
            truth_surface,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        )
        expected[scanner_cell.cell.label][(stage, "all", "array_nonzero_fraction")] = float(
            np.count_nonzero(values) / values.size
        )
        expected[scanner_cell.cell.label].update(
            {
                (stage, "top_truth_count", metric): value
                for metric, value in _quality_row_values(block, edge).items()
            }
        )
    raw_support = top_truth_count_mask(scanner_ft, truth_surface)
    for selection, samples in (
        ("finite", confidence.ravel()),
        ("raw_top_truth_count", confidence[raw_support]),
    ):
        for summary, value in (
            ("mean", np.mean(samples)),
            ("median", np.median(samples)),
            ("p95", np.percentile(samples, 95.0)),
        ):
            expected[scanner_cell.cell.label][
                ("scanner_confidence", selection, f"confidence_{summary}")
            ] = float(value)

    return replace(evaluation, cells=(scanner_cell, downstream_cell)), expected


def test_metric_row_field_order_is_canonical() -> None:
    assert tuple(field.name for field in fields(MetricRow)) == (
        "schema_version",
        "case_id",
        "trial_id",
        "seed",
        "scope",
        "cell_label",
        "input_mode",
        "scanner_backend",
        "scanner_refinement_factor",
        "scanner_thin_mode",
        "workflow_mode",
        "voter_thin_mode",
        "skinner_method",
        "variant",
        "stage",
        "selection",
        "metric",
        "value",
        "unit",
        "direction",
        "contrast_eligible",
    )


@pytest.mark.parametrize(
    "field_name",
    (
        "seed",
        "scanner_backend",
        "scanner_refinement_factor",
        "scanner_thin_mode",
        "workflow_mode",
        "voter_thin_mode",
        "skinner_method",
    ),
)
def test_metric_row_rejects_array_optional_metadata(field_name: str) -> None:
    row = extract_trial_metric_rows(_evaluation())[0]

    with pytest.raises(ValueError, match=field_name):
        replace(row, **{field_name: np.asarray([1])})


def test_runner_rows_are_finite_unique_and_follow_cell_registry_order() -> None:
    evaluation = _evaluation()

    rows = extract_trial_metric_rows(evaluation)

    assert all(np.isfinite(row.value) for row in rows)
    identities = [
        (row.case_id, row.trial_id, row.cell_label, row.stage, row.selection, row.metric)
        for row in rows
    ]
    assert len(identities) == len(set(identities))
    labels = tuple(dict.fromkeys(row.cell_label for row in rows))
    assert labels == tuple(cell.cell.label for cell in evaluation.cells)
    registry_order = {
        (definition.stage, definition.selection, definition.metric): index
        for index, definition in enumerate(METRIC_REGISTRY)
    }
    for label in labels:
        positions = [
            registry_order[(row.stage, row.selection, row.metric)]
            for row in rows
            if row.cell_label == label
        ]
        assert positions == sorted(positions)
    assert rows == extract_trial_metric_rows(evaluation)


def test_scope_rows_and_effective_metadata_are_not_inferred_across_axes() -> None:
    rows = extract_trial_metric_rows(_evaluation())
    by_label = {
        label: [row for row in rows if row.cell_label == label]
        for label in {r.cell_label for r in rows}
    }

    assert {row.stage for row in by_label["RL-SCAN"]} == {"scanner_raw", "scanner_thinned"}
    assert {row.stage for row in by_label["Q-SCAN"]} == {
        "scanner_raw",
        "scanner_thinned",
        "scanner_confidence",
    }
    assert not any(
        row.contrast_eligible for row in by_label["Q-SCAN"] if row.stage == "scanner_confidence"
    )
    assert not any(row.stage.startswith("scanner") for row in by_label["ORACLE-REF"])
    oracle = by_label["ORACLE-REF"][0]
    assert oracle.scanner_backend is None
    assert oracle.scanner_refinement_factor is None
    assert oracle.scanner_thin_mode is None
    assert oracle.workflow_mode == "reference"
    quality = by_label["Q-QUAL"][0]
    assert quality.scanner_backend == "quality"
    assert quality.scanner_refinement_factor == 2
    assert quality.workflow_mode == "quality"
    assert quality.voter_thin_mode == "hybrid_v2"
    assert quality.skinner_method == "quality"


def test_downstream_rows_reuse_report_metric_values() -> None:
    evaluation = _evaluation()
    rows = extract_trial_metric_rows(evaluation)
    cell = next(cell for cell in evaluation.cells if cell.cell.label == "Q-QUAL")
    expected = cell.report_payload["quality"]["fvt_top_truth_count"]

    actual = {
        row.metric: row.value
        for row in rows
        if row.cell_label == "Q-QUAL" and row.stage == "fvt" and row.selection == "top_truth_count"
    }
    assert actual["buffered_f1"] == expected["buffered_overlap_radius2"]["buffered_f1"]
    assert (
        actual["candidate_to_truth_p95"] == expected["surface_distance"]["candidate_to_truth_p95"]
    )
    assert actual["strike_median"] == expected["orientation_error"]["strike_median"]


def test_handcrafted_rows_match_existing_metric_implementations() -> None:
    evaluation, expected = _handcrafted_evaluation()

    rows = extract_trial_metric_rows(evaluation)
    actual = {
        label: {
            (row.stage, row.selection, row.metric): row.value
            for row in rows
            if row.cell_label == label
        }
        for label in expected
    }

    assert actual.keys() == expected.keys()
    for label in expected:
        assert actual[label].keys() == expected[label].keys()
        for identity, value in expected[label].items():
            assert actual[label][identity] == pytest.approx(value)

    assert actual["Q-SCAN"][("scanner_raw", "top_truth_count", "strike_median")] == 2.0
    assert actual["Q-SCAN"][("scanner_raw", "top_truth_count", "strike_p95")] == 2.0
    assert (
        actual["Q-SCAN"][
            ("scanner_raw", "top_truth_count", "edge_false_positive_fraction_of_candidates")
        ]
        > 0.0
    )
    assert actual["Q-QUAL"][("fvt", "positive_top_truth_count", "candidate_count")] == 0.0
    assert actual["Q-QUAL"][("skin", "skin_cells", "over_merge_skin_count")] == 1.0
    assert actual["Q-QUAL"][("skin", "skin_cells", "over_split_truth_component_count")] == 2.0


def test_empty_skin_rows_preserve_finite_zero_and_penalty_contracts() -> None:
    evaluation, expected = _handcrafted_evaluation(empty_skins=True)

    rows = [row for row in extract_trial_metric_rows(evaluation) if row.stage == "skin"]
    actual = {(row.stage, row.selection, row.metric): row.value for row in rows}

    assert actual.keys() == {identity for identity in expected["Q-QUAL"] if identity[0] == "skin"}
    assert all(np.isfinite(value) for value in actual.values())
    assert actual[("skin", "skin_cells", "candidate_count")] == 0.0
    assert actual[("skin", "skin_cells", "skin_count")] == 0.0
    assert actual[("skin", "skin_cells", "uncovered_truth_component_count")] == 2.0
    assert actual[("skin", "skin_cells", "hausdorff_p95")] == pytest.approx(
        np.sqrt(sum((size - 1) ** 2 for size in evaluation.trial.shape))
    )


def test_disabled_skinning_omits_skin_rows() -> None:
    rows = extract_trial_metric_rows(_evaluation(skinning_enabled=False))

    assert not any(row.stage == "skin" for row in rows)


@pytest.mark.parametrize(
    "case_id",
    ("single_vertical_plane", "boundary_plane", "parallel_planes", "crossing_planes"),
)
def test_real_runner_case_rows_have_complete_finite_contract(case_id: str) -> None:
    plan = build_mode_comparison_plan(
        SyntheticModeComparisonConfig(case_ids=(case_id,), shape=(9, 9, 9))
    )

    rows = extract_trial_metric_rows(run_synthetic_trial(plan, plan.trials[0]))

    assert rows
    assert all(row.case_id == case_id for row in rows)
    assert all(np.isfinite(row.value) for row in rows)
    assert all(row.unit and row.direction in {"higher", "lower", "neutral"} for row in rows)
    assert all(row.scope and row.cell_label and row.input_mode and row.variant for row in rows)


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_artifact_fails_the_whole_extraction(invalid: float) -> None:
    evaluation = _evaluation()
    scanner_cell = evaluation.cells[0]
    artifacts = dict(scanner_cell.artifacts)
    ft = artifacts["scanner_ft"].copy()
    ft[0, 0, 0] = invalid
    artifacts["scanner_ft"] = ft
    invalid_cell = replace(scanner_cell, artifacts=artifacts)
    invalid_evaluation = replace(evaluation, cells=(invalid_cell, *evaluation.cells[1:]))

    with pytest.raises(ValueError, match="scanner_ft must contain only finite values"):
        extract_trial_metric_rows(invalid_evaluation)


def test_shape_mismatch_fails_the_whole_extraction() -> None:
    evaluation = _evaluation()
    scanner_cell = evaluation.cells[0]
    artifacts = dict(scanner_cell.artifacts)
    artifacts["scanner_pt"] = artifacts["scanner_pt"][:-1]
    invalid_cell = replace(scanner_cell, artifacts=artifacts)

    with pytest.raises(ValueError, match="scanner_pt must have shape"):
        extract_trial_metric_rows(replace(evaluation, cells=(invalid_cell, *evaluation.cells[1:])))
