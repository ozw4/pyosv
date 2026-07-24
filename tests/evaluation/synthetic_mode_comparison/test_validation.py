from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pyosv.evaluation.synthetic_mode_comparison import (
    SyntheticModeComparisonConfig,
    aggregate_contrast_rows,
    aggregate_metric_rows,
    build_mode_comparison_plan,
    compute_contrast_rows,
    extract_trial_metric_rows,
    run_mode_comparison,
    validate_mode_comparison_result,
)
from pyosv.evaluation.synthetic_mode_comparison.validation import (
    _expected_cache_counters,
    _validate_downstream_topology_algebra,
)
from pyosv.evaluation.synthetic_mode_comparison.scalar_algebra import (
    validate_component_topology_algebra,
    validate_component_topology_evidence,
    validate_overlap_algebra,
    validate_skin_report_topology_algebra,
    validate_skin_topology_algebra,
)
from pyosv.evaluation.synthetic_quality import SyntheticSkinningConfig, SyntheticVotingConfig
from pyosv.synthetic_metrics import buffered_surface_overlap


@pytest.fixture(scope="module")
def config() -> SyntheticModeComparisonConfig:
    return SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=(9, 9, 9),
    )


@pytest.fixture(scope="module")
def result(config):
    return run_mode_comparison(config)


def _valid_topology_reports():
    topology = {
        "skin_count": 3,
        "cell_count": 5,
        "unique_cell_count": 4,
        "duplicate_cell_count": 1,
        "largest_skin_size": 3,
        "largest_skin_fraction": 0.6,
        "small_skin_size": 2,
        "small_skin_count": 2,
        "small_skin_cell_count": 2,
        "small_skin_cell_fraction": 0.4,
    }
    component = {
        "qualification_min_fraction": 0.05,
        "truth_component_count": 2,
        "covered_truth_component_count": 2,
        "uncovered_truth_component_count": 0,
        "skin_count": 3,
        "skin_with_truth_count": 2,
        "skin_without_truth_count": 1,
        "over_merge_skin_count": 0,
        "over_split_truth_component_count": 0,
        "max_truth_components_per_skin": 1,
        "max_skins_per_truth_component": 1,
        "mean_skin_purity": 5.0 / 9.0,
        "min_skin_purity": 0.0,
        "mean_truth_component_recall": 0.75,
        "min_truth_component_recall": 0.5,
        "truth_components": [
            {
                "truth_id": 1,
                "truth_cell_count": 2,
                "covered_cell_count": 2,
                "recall": 1.0,
                "skin_count_touching": 1,
                "dominant_skin_index": 0,
                "dominant_skin_cell_count": 2,
                "dominant_skin_fraction_of_truth": 1.0,
                "skin_cell_counts": [{"skin_index": 0, "covered_cell_count": 2}],
                "qualifying_skin_count": 1,
            },
            {
                "truth_id": 2,
                "truth_cell_count": 2,
                "covered_cell_count": 1,
                "recall": 0.5,
                "skin_count_touching": 1,
                "dominant_skin_index": 1,
                "dominant_skin_cell_count": 1,
                "dominant_skin_fraction_of_truth": 0.5,
                "skin_cell_counts": [{"skin_index": 1, "covered_cell_count": 1}],
                "qualifying_skin_count": 1,
            },
        ],
        "skins": [
            {
                "skin_index": 0,
                "cell_count": 3,
                "truth_cell_count": 2,
                "background_cell_count": 1,
                "truth_component_count_touching": 1,
                "dominant_truth_id": 1,
                "dominant_truth_cell_count": 2,
                "purity": 2.0 / 3.0,
                "truth_component_cell_counts": [{"truth_id": 1, "cell_count": 2}],
                "qualifying_truth_component_count": 1,
            },
            {
                "skin_index": 1,
                "cell_count": 1,
                "truth_cell_count": 1,
                "background_cell_count": 0,
                "truth_component_count_touching": 1,
                "dominant_truth_id": 2,
                "dominant_truth_cell_count": 1,
                "purity": 1.0,
                "truth_component_cell_counts": [{"truth_id": 2, "cell_count": 1}],
                "qualifying_truth_component_count": 1,
            },
            {
                "skin_index": 2,
                "cell_count": 1,
                "truth_cell_count": 0,
                "background_cell_count": 1,
                "truth_component_count_touching": 0,
                "dominant_truth_id": None,
                "dominant_truth_cell_count": 0,
                "purity": 0.0,
                "truth_component_cell_counts": [],
                "qualifying_truth_component_count": 0,
            },
        ],
    }
    return topology, component


def _fragmentation_reports(cell_counts, small_skin_size):
    total_cell_count = sum(cell_counts)
    small_counts = [count for count in cell_counts if count < small_skin_size]
    largest_skin_size = max(cell_counts, default=0)
    topology = {
        "skin_count": len(cell_counts),
        "cell_count": total_cell_count,
        "unique_cell_count": total_cell_count,
        "duplicate_cell_count": 0,
        "largest_skin_size": largest_skin_size,
        "largest_skin_fraction": (
            largest_skin_size / total_cell_count if total_cell_count else 0.0
        ),
        "small_skin_size": small_skin_size,
        "small_skin_count": len(small_counts),
        "small_skin_cell_count": sum(small_counts),
        "small_skin_cell_fraction": (
            sum(small_counts) / total_cell_count if total_cell_count else 0.0
        ),
    }
    component = {
        "qualification_min_fraction": 0.05,
        "truth_component_count": 0,
        "covered_truth_component_count": 0,
        "uncovered_truth_component_count": 0,
        "skin_count": len(cell_counts),
        "skin_with_truth_count": 0,
        "skin_without_truth_count": len(cell_counts),
        "over_merge_skin_count": 0,
        "over_split_truth_component_count": 0,
        "max_truth_components_per_skin": 0,
        "max_skins_per_truth_component": 0,
        "mean_skin_purity": 0.0,
        "min_skin_purity": 0.0,
        "mean_truth_component_recall": 0.0,
        "min_truth_component_recall": 0.0,
        "truth_components": [],
        "skins": [
            {
                "skin_index": index,
                "cell_count": count,
                "truth_cell_count": 0,
                "background_cell_count": count,
                "truth_component_count_touching": 0,
                "dominant_truth_id": None,
                "dominant_truth_cell_count": 0,
                "purity": 0.0,
                "truth_component_cell_counts": [],
                "qualifying_truth_component_count": 0,
            }
            for index, count in enumerate(cell_counts)
        ],
    }
    return topology, component


def _replace_nested(value, path, replacement) -> None:
    for name in path[:-1]:
        value = value[name]
    value[path[-1]] = replacement


@pytest.mark.parametrize(
    ("candidate", "truth", "numerator"),
    (
        ([False, False], [True, False], "truth_in_candidate_buffer_count"),
        ([True, False], [False, False], "candidate_in_truth_buffer_count"),
    ),
)
def test_overlap_algebra_rejects_buffered_hits_against_an_empty_mask(
    candidate, truth, numerator
) -> None:
    report = buffered_surface_overlap(np.array(candidate), np.array(truth), radius=1.0)
    report[numerator] = 1
    report["buffered_precision"] = (
        report["candidate_in_truth_buffer_count"] / report["candidate_count"]
        if report["candidate_count"]
        else 1.0
    )
    report["buffered_recall"] = (
        report["truth_in_candidate_buffer_count"] / report["truth_count"]
        if report["truth_count"]
        else 1.0
    )
    left = report["buffered_precision"]
    right = report["buffered_recall"]
    report["buffered_f1"] = 2.0 * left * right / (left + right) if left + right else 0.0

    with pytest.raises(ValueError, match="nonempty .* mask"):
        validate_overlap_algebra(report, (1, 1, 2), "overlap")


def test_overlap_algebra_rejects_radius_zero_buffered_hits_beyond_intersection() -> None:
    candidate = np.array([True, False])
    truth = np.array([False, True])
    report = buffered_surface_overlap(candidate, truth, radius=1.0)
    report["radius"] = 0.0

    with pytest.raises(ValueError, match="radius-zero"):
        validate_overlap_algebra(report, (1, 1, 2), "overlap")


def test_overlap_algebra_accepts_positive_radius_buffered_hits_beyond_intersection() -> None:
    candidate = np.array([True, False])
    truth = np.array([False, True])
    report = buffered_surface_overlap(candidate, truth, radius=1.0)

    assert report["candidate_in_truth_buffer_count"] > report["intersection_count"]
    assert report["truth_in_candidate_buffer_count"] > report["intersection_count"]
    validate_overlap_algebra(report, (1, 1, 2), "overlap")


@pytest.mark.parametrize("radius", (0.0, 0.5, np.nextafter(1.0, 0.0)))
def test_overlap_algebra_requires_exact_overlap_for_fractional_radius(radius: float) -> None:
    candidate = np.zeros((1, 1, 2), dtype=bool)
    truth = np.zeros_like(candidate)
    candidate[0, 0, 0] = True
    truth[0, 0, 1] = True
    report = buffered_surface_overlap(candidate, truth, radius=float(radius))

    assert report["candidate_in_truth_buffer_count"] == report["intersection_count"]
    assert report["truth_in_candidate_buffer_count"] == report["intersection_count"]
    validate_overlap_algebra(report, candidate.shape, "overlap")


@pytest.mark.parametrize("radius", (0.5, np.nextafter(1.0, 0.0)))
def test_overlap_algebra_rejects_fractional_buffered_hits_beyond_intersection(
    radius: float,
) -> None:
    candidate = np.zeros((1, 1, 2), dtype=bool)
    truth = np.zeros_like(candidate)
    candidate[0, 0, 0] = True
    truth[0, 0, 1] = True
    report = buffered_surface_overlap(candidate, truth, radius=1.0)
    report["radius"] = radius

    with pytest.raises(ValueError, match="fractional-radius"):
        validate_overlap_algebra(report, candidate.shape, "overlap")


@pytest.mark.parametrize("offset", (0.0, 1.0))
def test_overlap_algebra_requires_source_counts_at_volume_diagonal(offset: float) -> None:
    candidate = np.zeros((2, 2, 2), dtype=bool)
    truth = np.zeros_like(candidate)
    candidate[0, 0, 0] = True
    truth[1, 1, 1] = True
    radius = np.sqrt(3.0) + offset
    report = buffered_surface_overlap(candidate, truth, radius=radius)

    assert report["candidate_in_truth_buffer_count"] == report["candidate_count"]
    assert report["truth_in_candidate_buffer_count"] == report["truth_count"]
    validate_overlap_algebra(report, candidate.shape, "overlap")


def test_overlap_algebra_rejects_incomplete_hits_at_volume_diagonal() -> None:
    candidate = np.zeros((2, 2, 2), dtype=bool)
    truth = np.zeros_like(candidate)
    candidate[0, 0, 0] = True
    truth[1, 1, 1] = True
    diagonal = np.sqrt(3.0)
    report = buffered_surface_overlap(candidate, truth, radius=np.nextafter(diagonal, 0.0))
    report["radius"] = diagonal

    with pytest.raises(ValueError, match="full-volume"):
        validate_overlap_algebra(report, candidate.shape, "overlap")


def test_overlap_algebra_does_not_expand_exact_radius_boundaries() -> None:
    candidate = np.zeros((2, 2, 2), dtype=bool)
    truth = np.zeros_like(candidate)
    candidate[0, 0, 0] = True
    truth[1, 1, 1] = True
    diagonal = np.sqrt(3.0)

    radius_one = buffered_surface_overlap(candidate, truth, radius=1.0)
    validate_overlap_algebra(radius_one, candidate.shape, "overlap")

    below_diagonal = buffered_surface_overlap(candidate, truth, radius=np.nextafter(diagonal, 0.0))
    validate_overlap_algebra(below_diagonal, candidate.shape, "overlap")


@pytest.mark.parametrize(
    ("candidate_value", "truth_value"),
    ((True, True), (False, True), (True, False), (False, False)),
)
def test_overlap_algebra_handles_singleton_volume_and_empty_masks(
    candidate_value: bool, truth_value: bool
) -> None:
    candidate = np.array([[[candidate_value]]])
    truth = np.array([[[truth_value]]])
    report = buffered_surface_overlap(candidate, truth, radius=0.0)

    validate_overlap_algebra(report, candidate.shape, "overlap")


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("duplicate_cell_count",), 0),
        (("largest_skin_fraction",), 0.5),
        (("small_skin_cell_fraction",), 0.5),
        (("unique_cell_count",), 5),
        (("largest_skin_size",), 5),
        (("small_skin_count",), 4),
    ),
)
def test_skin_topology_algebra_rejects_inconsistent_summary(path, replacement) -> None:
    topology, _ = _valid_topology_reports()
    _replace_nested(topology, path, replacement)

    with pytest.raises(ValueError):
        validate_skin_topology_algebra(topology, "topology", shape=(9, 9, 9))


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("truth_component_count",), 3),
        (("covered_truth_component_count",), 1),
        (("skin_with_truth_count",), 1),
        (("truth_components", 1, "recall"), 0.75),
        (("truth_components", 1, "dominant_skin_index"), 3),
        (("skins", 1, "background_cell_count"), 1),
        (("skins", 0, "purity"), 0.5),
        (("skins", 2, "truth_component_count_touching"), 1),
        (("skins", 2, "dominant_truth_id"), 1),
        (("skins", 2, "dominant_truth_cell_count"), 1),
        (("skins", 2, "purity"), 0.5),
        (("mean_skin_purity",), 0.5),
        (("max_skins_per_truth_component",), 2),
        (("over_merge_skin_count",), 1),
        (("over_split_truth_component_count",), 1),
        (("skin_count",), 1),
    ),
)
def test_component_topology_algebra_rejects_inconsistent_report(path, replacement) -> None:
    topology, component = _valid_topology_reports()
    _replace_nested(component, path, replacement)

    with pytest.raises(ValueError):
        validate_component_topology_algebra(component, topology, "component_topology")


def test_topology_algebra_accepts_valid_duplicate_and_background_reports() -> None:
    topology, component = _valid_topology_reports()

    validate_skin_topology_algebra(topology, "topology", shape=(9, 9, 9))
    validate_component_topology_algebra(component, topology, "component_topology")


@pytest.mark.parametrize(
    ("fault_voxel_count", "intersection_count", "message"),
    (
        (5, 3, "fault_voxel_count"),
        (4, 2, "intersection_count"),
    ),
)
def test_component_topology_evidence_binds_trial_truth_and_skin_overlap(
    fault_voxel_count,
    intersection_count,
    message,
) -> None:
    _, component = _valid_topology_reports()
    truth_evidence = {"fault_voxel_count": fault_voxel_count}
    overlap = {"intersection_count": intersection_count}

    with pytest.raises(ValueError, match=message):
        validate_component_topology_evidence(
            component,
            truth_evidence,
            overlap,
            "component_topology",
        )


def test_skin_topology_caps_only_unique_cells_to_volume_capacity() -> None:
    topology, _ = _valid_topology_reports()

    validate_skin_topology_algebra(topology, "topology", shape=(1, 2, 2))

    with pytest.raises(ValueError, match="unique_cell_count exceeds volume voxel count"):
        validate_skin_topology_algebra(topology, "topology", shape=(1, 1, 3))


@pytest.mark.parametrize("cell_counts", ((), (2,), (1, 2, 3)))
def test_skin_report_topology_algebra_recomputes_fragmentation_from_per_skin_sizes(
    cell_counts,
) -> None:
    topology, component = _fragmentation_reports(cell_counts, small_skin_size=2)

    validate_skin_report_topology_algebra(
        topology,
        component,
        "skin",
        small_skin_size=2,
        shape=(9, 9, 9),
    )


def test_skin_report_topology_algebra_requires_effective_small_skin_size() -> None:
    topology, component = _fragmentation_reports((1, 2, 3), small_skin_size=2)

    with pytest.raises(ValueError, match="small_skin_size does not match"):
        validate_skin_report_topology_algebra(
            topology,
            component,
            "skin",
            small_skin_size=3,
            shape=(9, 9, 9),
        )


@pytest.mark.parametrize(
    ("name", "replacement"),
    (
        ("largest_skin_size", 2),
        ("largest_skin_fraction", 0.4),
        ("small_skin_count", 2),
        ("small_skin_cell_count", 2),
        ("small_skin_cell_fraction", 0.5),
    ),
)
def test_skin_report_topology_algebra_rejects_fragmentation_summary_tampering(
    name, replacement
) -> None:
    topology, component = _fragmentation_reports((1, 2, 3), small_skin_size=2)
    topology[name] = replacement

    with pytest.raises(ValueError):
        validate_skin_report_topology_algebra(
            topology,
            component,
            "skin",
            small_skin_size=2,
            shape=(9, 9, 9),
        )


@pytest.mark.parametrize(
    "updates",
    (
        {"largest_skin_size": 2, "largest_skin_fraction": 2.0 / 6.0},
        {
            "small_skin_count": 2,
            "small_skin_cell_count": 2,
            "small_skin_cell_fraction": 2.0 / 6.0,
        },
    ),
)
def test_skin_report_topology_algebra_rejects_internally_coherent_summary_tampering(
    updates,
) -> None:
    topology, component = _fragmentation_reports((1, 2, 3), small_skin_size=2)
    topology.update(updates)

    with pytest.raises(ValueError, match="does not match per-skin cell counts"):
        validate_skin_report_topology_algebra(
            topology,
            component,
            "skin",
            small_skin_size=2,
            shape=(9, 9, 9),
        )


def test_skin_report_topology_algebra_rejects_per_skin_redistribution() -> None:
    topology, component = _fragmentation_reports((1, 4), small_skin_size=2)
    for item, count in zip(component["skins"], (2, 3), strict=True):
        item["cell_count"] = count
        item["background_cell_count"] = count

    with pytest.raises(ValueError, match="does not match per-skin cell counts"):
        validate_skin_report_topology_algebra(
            topology,
            component,
            "skin",
            small_skin_size=2,
            shape=(9, 9, 9),
        )


def test_complete_result_passes_without_mutation(result, config) -> None:
    before = result.as_dict()

    assert validate_mode_comparison_result(result, config) is None
    assert result.as_dict() == before
    shared_workflow_contrasts = {
        "oracle_workflow_effect",
        "workflow_effect_rl",
        "workflow_effect_q",
    }
    fv_contrasts = tuple(
        row
        for row in result.contrast_rows
        if row.stage == "fv" and row.contrast_name in shared_workflow_contrasts
    )
    assert fv_contrasts
    assert all(row.raw_value == 0.0 for row in fv_contrasts)


@pytest.mark.parametrize(
    ("config_overrides", "expected"),
    (
        (
            {},
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
            {"voting_config": SyntheticVotingConfig()},
            {
                "seed_hits": 3,
                "seed_misses": 3,
                "voting_hits": 3,
                "voting_misses": 3,
                "thinning_hits": 3,
                "thinning_misses": 3,
                "primary_skinning_hits": 0,
                "primary_skinning_misses": 6,
            },
        ),
        (
            {
                "voting_config": SyntheticVotingConfig(),
                "skinner_method_explicit": True,
                "include_oracle_workflow_isolation": False,
            },
            {
                "seed_hits": 2,
                "seed_misses": 2,
                "voting_hits": 2,
                "voting_misses": 2,
                "thinning_hits": 2,
                "thinning_misses": 2,
                "primary_skinning_hits": 2,
                "primary_skinning_misses": 2,
            },
        ),
        (
            {"skinning_config": SyntheticSkinningConfig(enabled=False)},
            {
                "seed_hits": 3,
                "seed_misses": 3,
                "voting_hits": 3,
                "voting_misses": 3,
                "thinning_hits": 0,
                "thinning_misses": 6,
                "primary_skinning_hits": 0,
                "primary_skinning_misses": 0,
            },
        ),
    ),
)
def test_expected_cache_counters_follow_resolved_stage_keys(config_overrides, expected) -> None:
    plan = build_mode_comparison_plan(
        SyntheticModeComparisonConfig(
            case_ids=("single_vertical_plane",),
            shape=(9, 9, 9),
            **config_overrides,
        )
    )

    assert _expected_cache_counters(plan) == expected


def test_shared_explicit_stage_keys_pass_runtime_semantic_validation() -> None:
    explicit_config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=(9, 9, 9),
        voting_config=SyntheticVotingConfig(),
        skinner_method_explicit=True,
    )

    explicit_result = run_mode_comparison(explicit_config)

    assert (
        explicit_result.cache_stats[0]["thinning_misses"],
        explicit_result.cache_stats[0]["thinning_hits"],
    ) == (3, 3)
    assert (
        explicit_result.cache_stats[0]["primary_skinning_misses"],
        explicit_result.cache_stats[0]["primary_skinning_hits"],
    ) == (3, 3)
    assert validate_mode_comparison_result(explicit_result, explicit_config) is None
    fvt_workflow_contrasts = tuple(
        row
        for row in explicit_result.contrast_rows
        if row.stage == "fvt"
        and row.contrast_name
        in {"oracle_workflow_effect", "workflow_effect_rl", "workflow_effect_q"}
    )
    assert fvt_workflow_contrasts
    assert all(row.raw_value == 0.0 for row in fvt_workflow_contrasts)

    reports = explicit_result.as_dict()["cell_reports"]
    payload = reports[0]["cells"]["RL-QUAL"]
    payload["pyosv"]["fvt"]["mean"] += 0.01
    payload["pipelines"]["scanner"]["pyosv"]["fvt"]["mean"] += 0.01

    with pytest.raises(ValueError, match="shared thinning stage evidence"):
        validate_mode_comparison_result(
            replace(explicit_result, cell_reports=tuple(reports)),
            explicit_config,
        )


def test_shared_scanner_input_tampering_is_rejected(result, config) -> None:
    reports = result.as_dict()["cell_reports"]
    reports[0]["cells"]["Q-SCAN"]["scanner"]["input"]["mean"] += 0.01

    with pytest.raises(ValueError, match="shared scanner input evidence"):
        validate_mode_comparison_result(replace(result, cell_reports=tuple(reports)), config)


@pytest.mark.parametrize("field", ("fault_voxel_count", "surface_voxel_count"))
def test_trial_truth_evidence_tampering_is_rejected(result, config, field: str) -> None:
    reports = result.as_dict()["cell_reports"]
    reports[0]["truth_evidence"][field] += 1

    with pytest.raises(ValueError, match=rf"truth_evidence.{field}"):
        validate_mode_comparison_result(replace(result, cell_reports=tuple(reports)), config)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_field", "missing: surface_voxel_count"),
        ("wrong_order", "fields do not match the canonical schema and order"),
        ("boolean_count", "must be an integer"),
    ),
)
def test_trial_truth_evidence_schema_is_strict_in_memory(
    result, config, mutation: str, message: str
) -> None:
    reports = result.as_dict()["cell_reports"]
    evidence = reports[0]["truth_evidence"]
    if mutation == "missing_field":
        del evidence["surface_voxel_count"]
    elif mutation == "wrong_order":
        reports[0]["truth_evidence"] = {
            "surface_voxel_count": evidence["surface_voxel_count"],
            "fault_voxel_count": evidence["fault_voxel_count"],
        }
    else:
        evidence["fault_voxel_count"] = True

    with pytest.raises(ValueError, match=message):
        validate_mode_comparison_result(replace(result, cell_reports=tuple(reports)), config)


def _increment_overlap_truth_count(report: dict) -> None:
    overlap = report["buffered_overlap_radius2"]
    overlap["truth_count"] += 1
    overlap["union_count"] = (
        overlap["candidate_count"] + overlap["truth_count"] - overlap["intersection_count"]
    )
    overlap["recall"] = overlap["intersection_count"] / overlap["truth_count"]
    overlap["f1"] = (
        2.0 * overlap["precision"] * overlap["recall"] / (overlap["precision"] + overlap["recall"])
    )
    overlap["jaccard"] = overlap["intersection_count"] / overlap["union_count"]
    overlap["buffered_recall"] = overlap["truth_in_candidate_buffer_count"] / overlap["truth_count"]
    overlap["buffered_f1"] = (
        2.0
        * overlap["buffered_precision"]
        * overlap["buffered_recall"]
        / (overlap["buffered_precision"] + overlap["buffered_recall"])
    )


@pytest.mark.parametrize("stage", ("scanner_raw", "scanner_thinned"))
def test_scanner_truth_targets_are_joined_to_trial_evidence_in_memory(
    result, config, stage: str
) -> None:
    reports = result.as_dict()["cell_reports"]
    cell = reports[0]["cells"]["RL-SCAN"]
    entry = next(
        entry
        for entry in cell["scanner_metric_evidence"]
        if entry["stage"] == stage and "quality_report" in entry
    )
    _increment_overlap_truth_count(entry["quality_report"])
    overlap = entry["quality_report"]["buffered_overlap_radius2"]
    for metric in ("buffered_recall", "buffered_f1"):
        metric_entry = next(
            item
            for item in cell["scanner_metric_evidence"]
            if item["stage"] == stage and item["metric"] == metric
        )
        metric_entry["value"] = overlap[metric]
    if stage == "scanner_raw":
        _increment_overlap_truth_count(cell["scanner_quality"]["ft_top_truth_count"])

    with pytest.raises(ValueError, match="truth_evidence.fault_voxel_count"):
        validate_mode_comparison_result(replace(result, cell_reports=tuple(reports)), config)


@pytest.mark.parametrize(
    "quality_name",
    (
        "fv_top_truth_count",
        "fv_positive_top_truth_count",
        "fvt_top_truth_count",
        "fvt_positive_top_truth_count",
        "skin",
    ),
)
def test_downstream_truth_targets_are_joined_to_trial_evidence_in_memory(
    result, config, quality_name: str
) -> None:
    reports = result.as_dict()["cell_reports"]
    cell = reports[0]["cells"]["RL-QUAL"]
    for payload in (cell, cell["pipelines"][cell["active_pipeline"]]):
        _increment_overlap_truth_count(payload["quality"][quality_name])

    with pytest.raises(ValueError, match="truth_evidence.fault_voxel_count"):
        validate_mode_comparison_result(replace(result, cell_reports=tuple(reports)), config)


def test_coherent_end_to_end_scanner_tampering_is_rejected(result, config) -> None:
    reports = result.as_dict()["cell_reports"]
    payload = reports[0]["cells"]["RL-QUAL"]
    payload["scanner"]["fet"]["mean"] += 0.01
    payload["pipelines"]["scanner"]["scanner"]["fet"]["mean"] += 0.01

    with pytest.raises(ValueError, match="shared attribute stage evidence"):
        validate_mode_comparison_result(replace(result, cell_reports=tuple(reports)), config)


def test_coherent_voting_summary_tampering_is_rejected(result, config) -> None:
    reports = result.as_dict()["cell_reports"]
    payload = reports[0]["cells"]["Q-QUAL"]
    payload["pyosv"]["fv"]["mean"] += 0.01
    payload["pipelines"]["scanner"]["pyosv"]["fv"]["mean"] += 0.01

    with pytest.raises(ValueError, match="shared voting stage evidence"):
        validate_mode_comparison_result(replace(result, cell_reports=tuple(reports)), config)


def test_coherent_voting_metric_tampering_is_rejected(result, config) -> None:
    reports = result.as_dict()["cell_reports"]
    payload = reports[0]["cells"]["Q-QUAL"]
    for quality in (payload["quality"], payload["pipelines"]["scanner"]["quality"]):
        quality["fv_top_truth_count"]["buffered_overlap_radius2"]["candidate_count"] += 1

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
        == ("Q-QUAL", "fv", "top_truth_count", "candidate_count")
    )
    rows[index] = replace(rows[index], value=rows[index].value + 1.0)
    metric_rows = tuple(rows)
    contrast_rows = compute_contrast_rows(metric_rows)

    with pytest.raises(ValueError, match="candidate counts"):
        validate_mode_comparison_result(
            replace(
                result,
                cell_reports=tuple(reports),
                metric_rows=metric_rows,
                contrast_rows=contrast_rows,
                metric_aggregates=aggregate_metric_rows(metric_rows),
                contrast_aggregates=aggregate_contrast_rows(contrast_rows),
            ),
            config,
        )


def test_in_memory_validation_applies_shared_overlap_algebra(result, config) -> None:
    reports = result.as_dict()["cell_reports"]
    overlap = reports[0]["cells"]["RL-SCAN"]["scanner_quality"]["ft_top_truth_count"][
        "buffered_overlap_radius2"
    ]
    overlap["union_count"] += 1

    with pytest.raises(ValueError, match="union_count is inconsistent"):
        validate_mode_comparison_result(replace(result, cell_reports=tuple(reports)), config)


def test_in_memory_validation_applies_skin_topology_algebra(result, config) -> None:
    reports = result.as_dict()["cell_reports"]
    payload = reports[0]["cells"]["RL-REF"]
    for downstream in (payload, payload["pipelines"][payload["active_pipeline"]]):
        downstream["pyosv"]["skins"]["largest_skin_fraction"] -= 0.1
        downstream["quality"]["skin"]["topology"]["largest_skin_fraction"] -= 0.1

    with pytest.raises(ValueError, match="largest_skin_fraction is inconsistent"):
        validate_mode_comparison_result(replace(result, cell_reports=tuple(reports)), config)


def test_in_memory_validation_requires_matching_skin_topologies(result, config) -> None:
    reports = result.as_dict()["cell_reports"]
    payload = reports[0]["cells"]["RL-REF"]
    payload["pyosv"]["skins"]["largest_skin_fraction"] = 1

    with pytest.raises(ValueError, match="pyosv.skins does not match quality.skin.topology"):
        validate_mode_comparison_result(replace(result, cell_reports=tuple(reports)), config)


def test_in_memory_validation_requires_empty_skin_topology_when_disabled() -> None:
    topology, _ = _valid_topology_reports()
    payload = {"pyosv": {"skins": topology}, "quality": {"skin": None}}

    with pytest.raises(ValueError, match="must be empty when skinning is disabled"):
        _validate_downstream_topology_algebra(
            payload,
            SyntheticSkinningConfig(enabled=False, small_skin_size=2),
            (9, 9, 9),
            "cell",
            {"fault_voxel_count": 1},
        )


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


@pytest.mark.parametrize("tampering", ("call_count", "shared_stage", "order"))
def test_downstream_runtime_attribution_tampering_is_rejected(result, config, tampering) -> None:
    rows = list(result.runtime_rows)
    index = next(index for index, row in enumerate(rows) if row.stage == "voting_scalar_evidence")
    if tampering == "call_count":
        rows[index] = replace(rows[index], call_count=rows[index].call_count + 1)
    elif tampering == "shared_stage":
        rows[index] = replace(rows[index], shared_stage=False)
    else:
        rows[index], rows[index + 1] = rows[index + 1], rows[index]

    with pytest.raises(ValueError, match="runtime_rows"):
        validate_mode_comparison_result(replace(result, runtime_rows=tuple(rows)), config)


@pytest.mark.parametrize(
    ("stage", "message"),
    (
        ("case_generation", "disjoint elapsed exceeds trial_total"),
        ("trial_total", "disjoint elapsed exceeds trial_total"),
        ("experiment_total", "trial_total sum exceeds experiment_total"),
    ),
)
def test_runtime_elapsed_upper_bound_tampering_is_rejected(result, config, stage, message) -> None:
    rows = list(result.runtime_rows)
    index = next(index for index, row in enumerate(rows) if row.stage == stage)
    if stage == "case_generation":
        trial_total = next(row.elapsed_seconds for row in rows if row.stage == "trial_total")
        rows[index] = replace(rows[index], elapsed_seconds=trial_total + 1.0)
    else:
        rows[index] = replace(rows[index], elapsed_seconds=0.0)

    with pytest.raises(ValueError, match=message):
        validate_mode_comparison_result(replace(result, runtime_rows=tuple(rows)), config)


def test_runtime_elapsed_algebra_accepts_canonical_rounding_tolerance(result, config) -> None:
    rows = list(result.runtime_rows)
    trial_total_index = next(index for index, row in enumerate(rows) if row.stage == "trial_total")
    disjoint_total = sum(row.elapsed_seconds for row in rows[:trial_total_index])
    rows[trial_total_index] = replace(
        rows[trial_total_index],
        elapsed_seconds=disjoint_total - 5.0e-13,
    )

    validate_mode_comparison_result(replace(result, runtime_rows=tuple(rows)), config)


def test_zero_call_shared_runtime_requires_zero_elapsed() -> None:
    config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        shape=(9, 9, 9),
        skinning_config=SyntheticSkinningConfig(enabled=False),
    )
    result = run_mode_comparison(config)
    rows = list(result.runtime_rows)
    index = next(index for index, row in enumerate(rows) if row.stage == "primary_skinning")
    assert rows[index].call_count == 0
    rows[index] = replace(rows[index], elapsed_seconds=1.0e-6)

    with pytest.raises(ValueError, match="zero-call shared stage"):
        validate_mode_comparison_result(replace(result, runtime_rows=tuple(rows)), config)


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
