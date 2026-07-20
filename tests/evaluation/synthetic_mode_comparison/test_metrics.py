"""Tests for canonical synthetic mode-comparison metric rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
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
from pyosv.evaluation.synthetic_mode_comparison.metrics import (
    build_scanner_metric_evidence,
    scanner_metric_definitions,
)
from pyosv.evaluation.synthetic_mode_comparison.scalar_algebra import (
    validate_quality_scalar_algebra,
    validate_selection_cardinality,
    validate_surface_distance_algebra,
    volume_diagonal,
)
from pyosv.evaluation.synthetic_quality import PipelineArtifacts, SyntheticSkinningConfig
from pyosv.evaluation.synthetic_quality.quality_metrics import (
    EDGE_FALSE_POSITIVE_MARGIN,
    array_nonzero_fraction,
)
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


def _algebra_reports() -> tuple[dict[str, Any], dict[str, Any]]:
    shape = (3, 4, 5)
    candidate = np.zeros(shape, dtype=bool)
    truth = np.zeros(shape, dtype=bool)
    candidate[1, 1, 1] = True
    candidate[1, 1, 2] = True
    truth[1, 1, 2] = True
    truth[1, 1, 3] = True
    zeros = np.zeros(shape, dtype=np.float32)
    quality = _quality_metrics(
        candidate,
        truth_fault=truth,
        truth_surface=truth,
        predicted_strike=zeros,
        predicted_dip=zeros,
        truth_strike=zeros,
        truth_dip=zeros,
        buffer_radius=1.0,
    )
    edge = edge_false_positive_ratio(
        candidate,
        truth,
        edge_margin=0,
        truth_buffer_radius=1.0,
    )
    return quality, edge


def _skin(indices: Sequence[tuple[int, int, int]]) -> FaultSkin:
    return FaultSkin.from_cells(
        FaultCell(float(i1), float(i2), float(i3), 0.8, 179.0, 45.0) for i1, i2, i3 in indices
    )


def test_quality_scalar_algebra_accepts_canonical_reports_and_rounding_noise() -> None:
    quality, edge = _algebra_reports()
    distance = quality["surface_distance"]
    distance["symmetric_chamfer_mean"] += 5.0e-13

    validate_quality_scalar_algebra(
        overlap=quality["buffered_overlap_radius2"],
        distance=distance,
        orientation=quality["orientation_error"],
        edge=edge,
        shape=(3, 4, 5),
        context="quality.fixture",
    )


def test_volume_diagonal_uses_voxel_index_extents() -> None:
    assert volume_diagonal((9, 9, 9)) == pytest.approx(8.0 * np.sqrt(3.0))


@pytest.mark.parametrize(
    "value",
    (
        0.0,
        8.0 * np.sqrt(3.0) - 1.0e-9,
        8.0 * np.sqrt(3.0) + 5.0e-13,
    ),
)
def test_surface_distance_algebra_accepts_reachable_nonempty_distances(value: float) -> None:
    report = {
        "candidate_count": 1,
        "truth_count": 1,
        **{
            name: value
            for name in (
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
        },
    }

    validate_surface_distance_algebra(report, (9, 9, 9), "quality.surface_distance")


@pytest.mark.parametrize(
    "field",
    (
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
    ),
)
def test_surface_distance_algebra_rejects_each_unreachable_summary(field: str) -> None:
    report = {
        "candidate_count": 1,
        "truth_count": 1,
        **{
            name: 0.0
            for name in (
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
        },
    }
    report[field] = volume_diagonal((9, 9, 9)) + 1.0e-6

    with pytest.raises(ValueError, match=field):
        validate_surface_distance_algebra(report, (9, 9, 9), "quality.surface_distance")


@pytest.mark.parametrize(
    ("selection", "candidate_count", "truth_count"),
    (
        ("top_truth_count", 4, 4),
        ("positive_top_truth_count", 4, 4),
        ("positive_top_truth_count", 3, 4),
        ("top_truth_count", 0, 0),
    ),
)
def test_selection_cardinality_accepts_canonical_counts(
    selection: str, candidate_count: int, truth_count: int
) -> None:
    validate_selection_cardinality(
        selection=selection,
        candidate_count=candidate_count,
        truth_count=truth_count,
        context="quality.fixture",
    )


@pytest.mark.parametrize(
    ("selection", "candidate_count", "truth_count"),
    (
        ("top_truth_count", 3, 4),
        ("top_truth_count", 5, 4),
        ("positive_top_truth_count", 5, 4),
    ),
)
def test_selection_cardinality_rejects_impossible_counts(
    selection: str, candidate_count: int, truth_count: int
) -> None:
    with pytest.raises(ValueError, match="surface_distance.truth_count"):
        validate_selection_cardinality(
            selection=selection,
            candidate_count=candidate_count,
            truth_count=truth_count,
            context="quality.fixture",
        )


def test_overlap_monotonicity_accepts_rounding_noise_within_derived_tolerance() -> None:
    quality, edge = _algebra_reports()
    overlap = quality["buffered_overlap_radius2"]
    overlap["candidate_in_truth_buffer_count"] = overlap["intersection_count"]
    overlap["truth_in_candidate_buffer_count"] = overlap["intersection_count"]
    overlap["buffered_precision"] = overlap["precision"] - 5.0e-13
    overlap["buffered_recall"] = overlap["recall"]
    overlap["buffered_f1"] = overlap["f1"]

    validate_quality_scalar_algebra(
        overlap=overlap,
        distance=quality["surface_distance"],
        orientation=quality["orientation_error"],
        edge=edge,
        shape=(3, 4, 5),
        context="quality.fixture",
    )


@pytest.mark.parametrize(
    ("block_name", "field", "replacement"),
    (
        ("buffered_overlap_radius2", "union_count", 4),
        ("buffered_overlap_radius2", "precision", 0.75),
        ("buffered_overlap_radius2", "candidate_in_truth_buffer_count", 0),
        ("surface_distance", "candidate_to_truth_p90", -1.0),
        ("surface_distance", "symmetric_chamfer_mean", 2.0),
        ("surface_distance", "hausdorff_p95", 2.0),
        ("orientation_error", "strike_p90", -1.0),
        ("edge", "edge_candidate_count", 3),
        ("edge", "edge_candidate_fraction", 0.5),
    ),
)
def test_quality_scalar_algebra_rejects_inconsistent_reports(
    block_name: str, field: str, replacement: float | int
) -> None:
    quality, edge = _algebra_reports()
    quality = deepcopy(quality)
    edge = deepcopy(edge)
    block = edge if block_name == "edge" else quality[block_name]
    block[field] = replacement

    with pytest.raises(ValueError):
        validate_quality_scalar_algebra(
            overlap=quality["buffered_overlap_radius2"],
            distance=quality["surface_distance"],
            orientation=quality["orientation_error"],
            edge=edge,
            shape=(3, 4, 5),
            context="quality.fixture",
        )


def test_quality_scalar_algebra_rejects_meaningful_derived_rounding_change() -> None:
    quality, edge = _algebra_reports()
    quality["surface_distance"]["symmetric_chamfer_mean"] += 1.0e-8

    with pytest.raises(ValueError, match="symmetric_chamfer_mean"):
        validate_quality_scalar_algebra(
            overlap=quality["buffered_overlap_radius2"],
            distance=quality["surface_distance"],
            orientation=quality["orientation_error"],
            edge=edge,
            shape=(3, 4, 5),
            context="quality.fixture",
        )


@pytest.mark.parametrize(
    ("block_name", "field"),
    (
        ("surface_distance", "candidate_to_truth_mean"),
        ("orientation_error", "dip_mean"),
    ),
)
def test_quality_scalar_algebra_enforces_empty_distance_and_orientation_conventions(
    block_name: str, field: str
) -> None:
    empty = np.zeros((3, 4, 5), dtype=bool)
    zeros = np.zeros_like(empty, dtype=np.float32)
    quality = _quality_metrics(
        empty,
        truth_fault=empty,
        truth_surface=empty,
        predicted_strike=zeros,
        predicted_dip=zeros,
        truth_strike=zeros,
        truth_dip=zeros,
        buffer_radius=1.0,
    )
    quality[block_name][field] = 1.0

    with pytest.raises(ValueError):
        validate_quality_scalar_algebra(
            overlap=quality["buffered_overlap_radius2"],
            distance=quality["surface_distance"],
            orientation=quality["orientation_error"],
            shape=(3, 4, 5),
            context="quality.empty",
        )


def test_quality_scalar_algebra_enforces_one_empty_volume_diagonal_penalty() -> None:
    shape = (3, 4, 5)
    empty = np.zeros(shape, dtype=bool)
    truth = np.zeros(shape, dtype=bool)
    truth[1, 1, 1] = True
    zeros = np.zeros(shape, dtype=np.float32)
    quality = _quality_metrics(
        empty,
        truth_fault=truth,
        truth_surface=truth,
        predicted_strike=zeros,
        predicted_dip=zeros,
        truth_strike=zeros,
        truth_dip=zeros,
        buffer_radius=1.0,
    )
    quality["surface_distance"]["truth_to_candidate_mean"] = 0.0
    quality["surface_distance"]["symmetric_chamfer_mean"] *= 0.5

    with pytest.raises(ValueError, match="truth_to_candidate_mean"):
        validate_quality_scalar_algebra(
            overlap=quality["buffered_overlap_radius2"],
            distance=quality["surface_distance"],
            orientation=quality["orientation_error"],
            shape=shape,
            context="quality.one_empty",
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
    scanner_ft[0, 0, 1] = 5.0e-8
    scanner_fet[8, 8, 8] = 10.0
    scanner_fet[4, 4, 4] = 9.0
    scanner_fet[4, 4, 5] = 8.0
    scanner_fet[4, 5, 5] = 7.0
    scanner_fet[0, 0, 1] = -5.0e-8
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
    fv[0, 0, 1] = 5.0e-8
    fvt = np.zeros(shape, dtype=np.float32)
    fvt[0, 0, 1] = -5.0e-8
    stage_arrays = {"fv": fv, "fvt": fvt}
    quality: dict[str, Any] = {"edge_false_positive": {}}
    expected: dict[str, dict[tuple[str, str, str], float]] = {
        scanner_cell.cell.label: {},
        "Q-QUAL": {},
    }

    for stage, values in stage_arrays.items():
        expected["Q-QUAL"][(stage, "all", "array_nonzero_fraction")] = array_nonzero_fraction(
            values
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
            "pyosv": {
                stage: {"nonzero_fraction": array_nonzero_fraction(values)}
                for stage, values in stage_arrays.items()
            },
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
        expected[scanner_cell.cell.label][(stage, "all", "array_nonzero_fraction")] = (
            array_nonzero_fraction(values)
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

    scanner_report = dict(scanner_cell.report_payload)
    scanner_report["scanner"] = {
        **scanner_report["scanner"],
        "ft": {"nonzero_fraction": array_nonzero_fraction(scanner_ft)},
        "fet": {"nonzero_fraction": array_nonzero_fraction(scanner_fet)},
    }
    scanner_report["scanner_metric_evidence"] = build_scanner_metric_evidence(
        scanner_backend="quality",
        scanner_volumes=scanner_artifacts,
        scanner_report=scanner_report["scanner"],
        shape=shape,
        truth_fault=truth_fault,
        truth_distance=truth_distance,
        truth_strike=truth_strike,
        truth_dip=truth_dip,
        truth_surface_half_width=evaluation.truth_metric_config.truth_surface_half_width,
        buffer_radius=buffer_radius,
    )
    scanner_cell = replace(scanner_cell, report_payload=scanner_report)

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


def test_scanner_evidence_is_complete_registry_ordered_and_shared_by_backend() -> None:
    evaluation = _evaluation()
    cells = {cell.cell.label: cell for cell in evaluation.cells}

    for backend, labels in (
        ("reference-like", ("RL-SCAN", "RL-REF", "RL-QUAL")),
        ("quality", ("Q-SCAN", "Q-REF", "Q-QUAL")),
    ):
        evidence = cells[labels[0]].report_payload["scanner_metric_evidence"]
        definitions = scanner_metric_definitions(backend)
        assert tuple(
            (entry["stage"], entry["selection"], entry["metric"]) for entry in evidence
        ) == tuple(
            (definition.stage, definition.selection, definition.metric)
            for definition in definitions
        )
        assert tuple((entry["unit"], entry["direction"]) for entry in evidence) == tuple(
            (definition.unit, definition.direction) for definition in definitions
        )
        quality_entries = tuple(entry for entry in evidence if "quality_report" in entry)
        assert tuple(
            (entry["stage"], entry["selection"], entry["metric"]) for entry in quality_entries
        ) == (
            ("scanner_raw", "top_truth_count", "candidate_count"),
            ("scanner_thinned", "top_truth_count", "candidate_count"),
        )
        for entry in quality_entries:
            assert set(entry["quality_report"]) == {
                "buffered_overlap_radius2",
                "surface_distance",
                "orientation_error",
                "edge_false_positive",
            }
            with pytest.raises(TypeError):
                entry["quality_report"]["buffered_overlap_radius2"]["candidate_count"] = 0
        assert all(
            cells[label].report_payload["scanner_metric_evidence"] is evidence for label in labels
        )
        for report_name in ("scanner", "scanner_quality"):
            report = cells[labels[0]].report_payload[report_name]
            assert all(cells[label].report_payload[report_name] is report for label in labels)

    assert all(
        "scanner_metric_evidence" not in cells[label].report_payload
        for label in ("ORACLE-REF", "ORACLE-QUAL")
    )


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
def test_scanner_extraction_uses_evidence_without_reading_volumes(invalid: float) -> None:
    evaluation = _evaluation()
    expected = extract_trial_metric_rows(evaluation)
    scanner_cell = evaluation.cells[0]
    artifacts = dict(scanner_cell.artifacts)
    ft = artifacts["scanner_ft"].copy()
    ft[0, 0, 0] = invalid
    artifacts["scanner_ft"] = ft
    invalid_cell = replace(scanner_cell, artifacts=artifacts)
    invalid_evaluation = replace(evaluation, cells=(invalid_cell, *evaluation.cells[1:]))

    assert extract_trial_metric_rows(invalid_evaluation) == expected


def test_scanner_extraction_ignores_volume_shape_after_evidence_generation() -> None:
    evaluation = _evaluation()
    expected = extract_trial_metric_rows(evaluation)
    scanner_cell = evaluation.cells[0]
    artifacts = dict(scanner_cell.artifacts)
    artifacts["scanner_pt"] = artifacts["scanner_pt"][:-1]
    invalid_cell = replace(scanner_cell, artifacts=artifacts)

    assert (
        extract_trial_metric_rows(replace(evaluation, cells=(invalid_cell, *evaluation.cells[1:])))
        == expected
    )
