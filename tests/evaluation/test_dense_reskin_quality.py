import csv
import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pyosv.evaluation.dense_reskin_quality as dense_reskin_quality
import pyosv.evaluation.f3d_mode_comparison.reskin_policy_comparison as reskin_policy_comparison
import pyosv.evaluation.f3d_mode_comparison.result as f3_result_module
import pyosv.synthetic_metrics as synthetic_metrics
from pyosv.cells import FaultCell
from pyosv.evaluation.dense_reskin_quality import (
    build_dense_reskin_promotion_gate,
    controlled_dense_reskin_cases,
    write_dense_reskin_evidence,
)
from pyosv.evaluation.f3d_mode_comparison.reskin_policy_comparison import (
    F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE,
    compare_reskin_policies_from_bundle,
    compare_reskin_policies_from_parent,
    validate_f3_reskin_policy_comparison,
    write_f3_reskin_policy_comparison,
)
from pyosv.evaluation.synthetic_quality import (
    SyntheticSkinningConfig,
    resolve_workflow_settings,
)
from pyosv.skin import FaultSkin
from pyosv.synthetic_metrics import reskin_generation_metrics, skin_link_topology_metrics


def _set_nested(mapping, path, value) -> None:
    target = mapping
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value


@pytest.fixture(scope="module")
def passing_gate_inputs():
    gate = build_dense_reskin_promotion_gate()
    results = gate["case_results"]
    aggregate = dense_reskin_quality._aggregate(results)
    aggregate.update(
        {
            "candidate_buffered_recall_mean": aggregate["baseline_buffered_recall_mean"],
            "buffered_recall_delta": 0.0,
            "buffered_precision_delta": -0.02,
            "symmetric_chamfer_mean_delta": 0.25,
            "small_skin_cell_fraction_delta": 0.05,
        }
    )
    assert (
        dense_reskin_quality._gate_reasons(
            results,
            aggregate,
            deterministic=True,
        )
        == []
    )
    return results, aggregate


def test_controlled_dense_reskin_gate_applies_fixed_contract_to_every_case() -> None:
    gate = build_dense_reskin_promotion_gate()
    case_ids = [case.case_id for case in controlled_dense_reskin_cases()]

    assert gate["schema_version"] == 2
    assert gate["promotion_status"] == "promotion_candidate"
    assert gate["artifacts"] == {"figures": []}
    assert gate["passed"] is True
    assert gate["reasons"] == []
    assert gate["aggregate"]["deterministic_reexecution"] is True
    assert gate["aggregate"]["case_count"] == len(case_ids)
    assert gate["aggregate"]["case_ids"] == case_ids
    assert {result["case_id"] for result in gate["case_results"]} == {
        *case_ids,
    }
    assert len(gate["case_results"]) == 10


def test_dense_gate_required_hole_and_safety_evidence() -> None:
    by_id = {
        result["case_id"]: result for result in build_dense_reskin_promotion_gate()["case_results"]
    }

    one = by_id["plane_3x3_center_hole"]
    assert one["candidate"]["truth"]["truth_hole_recovered_count"] == 1
    assert one["candidate"]["generation"]["reskin_generated_cell_count"] == 1
    block = by_id["plane_4x4_internal_2x2_hole"]
    assert (
        block["candidate"]["truth"]["overlap"]["recall"]
        > block["baseline"]["truth"]["overlap"]["recall"]
    )
    assert by_id["low_support_gap"]["candidate"]["generation"]["reskin_generated_cell_count"] == 0
    valid_mask = by_id["valid_mask_barrier"]["candidate"]
    assert valid_mask["generation"]["reskin_rejected_invalid_mask_count"] > 0
    assert valid_mask["generated_on_invalid_mask_count"] == 0
    assert valid_mask["generation"]["reskin_generated_cell_count"] == 0
    prior = by_id["prior_occupancy_barrier"]["candidate"]
    assert prior["generation"]["reskin_rejected_prior_skin_collision_count"] > 0
    assert prior["generated_on_prior_occupancy_count"] == 0
    assert prior["generation"]["reskin_generated_cell_count"] == 0
    boundary = by_id["volume_boundary_surface"]["candidate"]
    assert boundary["truth"]["truth_hole_recovered_count"] == 1
    assert boundary["generation"]["reskin_generated_cell_count"] == 1
    assert boundary["out_of_volume_cell_count"] == 0
    assert boundary["clamped_artifact_count"] == 0
    for case_id in ("parallel_surfaces", "corner_touch_orientation_boundary"):
        assert by_id[case_id]["surface_count"] == 2
        assert by_id[case_id]["skinning_invocation_count"] == 1
        assert by_id[case_id]["candidate"]["truth_fault_id_bridge_count"] == 0


def test_controlled_cases_use_configured_quality_skinning_phase(monkeypatch) -> None:
    original = dense_reskin_quality.find_synthetic_skins
    calls = []

    def tracked(*args, skinning_config, valid_mask=None, **kwargs):
        calls.append(
            (
                skinning_config.method,
                skinning_config.reskin,
                skinning_config.reskin_policy,
                valid_mask is not None,
            )
        )
        return original(
            *args,
            skinning_config=skinning_config,
            valid_mask=valid_mask,
            **kwargs,
        )

    monkeypatch.setattr(dense_reskin_quality, "find_synthetic_skins", tracked)

    for case in controlled_dense_reskin_cases():
        dense_reskin_quality.evaluate_dense_reskin_case(case)

    assert len(calls) == 2 * len(controlled_dense_reskin_cases())
    assert {method for method, _, _, _ in calls} == {"quality"}
    assert all(reskin for _, reskin, _, _ in calls)
    assert [policy for _, _, policy, _ in calls] == [
        policy
        for _ in controlled_dense_reskin_cases()
        for policy in ("existing_cells_v1", "reference_dense_v1")
    ]
    assert sum(has_valid_mask for _, _, _, has_valid_mask in calls) == 2


def test_link_topology_helper_reports_violations_without_mutation() -> None:
    left = FaultCell(1, 1, 1, 1, 0, 90)
    right = FaultCell(2, 1, 1, 1, 0, 90)
    object.__setattr__(left, "cr", right)
    skins = (FaultSkin.from_cells((left,)), FaultSkin.from_cells((right,)))

    metrics = skin_link_topology_metrics(skins)

    assert metrics["reciprocal_link_violation_count"] == 1
    assert metrics["cross_skin_link_count"] == 1
    assert metrics["isolated_cell_count"] == 0
    assert left.cr is right
    assert right.cl is None


def test_reskin_generation_metrics_reject_inconsistent_observed_count() -> None:
    observed = FaultCell(1, 1, 1, 1, 0, 90)
    generated = FaultCell(
        2,
        1,
        1,
        1,
        0,
        90,
        generation="dense_reskin_generated",
        reskin_support=0.5,
    )

    with pytest.raises(ValueError, match="observed_output_cell_count does not match"):
        reskin_generation_metrics(
            (FaultSkin.from_cells((observed, generated)),),
            diagnostics={
                "input_cell_count": 1,
                "output_cell_count": 2,
                "observed_output_cell_count": 2,
                "generated_cell_count": 1,
            },
        )


def test_reskin_generation_metrics_exclude_connected_component_fallback_cells() -> None:
    fallback = FaultCell(
        1,
        1,
        1,
        1,
        0,
        90,
        generation="connected_component",
    )

    metrics = reskin_generation_metrics((FaultSkin.from_cells((fallback,)),))

    assert metrics["reskin_input_cell_count"] == 0
    assert metrics["reskin_output_cell_count"] == 0
    assert metrics["reskin_observed_output_cell_count"] == 0
    assert metrics["reskin_generated_cell_count"] == 0
    assert metrics["reskin_generated_cell_fraction"] is None
    assert metrics["reskin_generated_cell_fraction_status"] == "zero_output_cells"


def test_dense_gate_writers_keep_nulls_explicit(tmp_path) -> None:
    gate = build_dense_reskin_promotion_gate()

    json_path, csv_path, markdown_path = write_dense_reskin_evidence(gate, tmp_path)

    assert json.loads(json_path.read_text())["passed"] is True
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    assert len(rows) == 30
    low_baseline = next(
        row for row in rows if row["case_id"] == "low_support_gap" and row["policy"] == "baseline"
    )
    assert low_baseline["generated_cell_fraction"] == ""
    assert low_baseline["support_min"] == ""
    assert "Gate reasons" in markdown_path.read_text()


def test_dense_gate_save_figures_writes_file_list_and_metadata(
    tmp_path,
    monkeypatch,
) -> None:
    captured_panels = []

    def fake_save_slice_panel(path, panels, **kwargs):
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"figure")
        captured_panels.append(
            (
                [name for name, _ in panels],
                [np.asarray(values).shape for _, values in panels],
                kwargs,
            )
        )
        return output

    monkeypatch.setattr("pyosv.viz.save_slice_panel", fake_save_slice_panel)
    gate = build_dense_reskin_promotion_gate()

    json_path, _, markdown_path = write_dense_reskin_evidence(
        gate,
        tmp_path,
        save_figures=True,
    )

    payload = json.loads(json_path.read_text())
    figures = payload["artifacts"]["figures"]
    case_ids = [result["case_id"] for result in gate["case_results"]]
    assert [item["case_id"] for item in figures] == case_ids
    assert [item["path"] for item in figures] == [
        f"figures/{case_id}_dense_reskin_i2_projection.png" for case_id in case_ids
    ]
    assert all(item["projection_axis"] == "i2" for item in figures)
    assert all(
        item["panels"]
        == [
            "baseline_cells",
            "candidate_observed_cells",
            "candidate_generated_cells",
            "truth_surface",
        ]
        for item in figures
    )
    assert all((tmp_path / item["path"]).is_file() for item in figures)
    assert len(captured_panels) == len(case_ids)
    assert all(
        names
        == [
            "baseline cells",
            "candidate observed cells",
            "candidate generated cells",
            "truth surface",
        ]
        and shapes == [(21, 21)] * 4
        and options["clip_percentiles"] == (0.0, 100.0)
        for names, shapes, options in captured_panels
    )
    assert "Promotion status: `promotion_candidate`" in markdown_path.read_text()


def test_dense_gate_cli_forwards_save_figures(tmp_path, monkeypatch) -> None:
    calls = []
    gate = build_dense_reskin_promotion_gate()

    monkeypatch.setattr(
        dense_reskin_quality,
        "build_dense_reskin_promotion_gate",
        lambda: gate,
    )

    def fake_write(evidence, output_dir, *, save_figures):
        calls.append((evidence, output_dir, save_figures))
        return (tmp_path / "gate.json", tmp_path / "gate.csv", tmp_path / "gate.md")

    monkeypatch.setattr(dense_reskin_quality, "write_dense_reskin_evidence", fake_write)

    assert (
        dense_reskin_quality.main(
            [
                "--output-dir",
                str(tmp_path),
                "--save-figures",
            ]
        )
        == 0
    )
    assert calls == [(gate, tmp_path, True)]


def test_dense_gate_reports_baseline_and_candidate_nonfinite_metrics(tmp_path, monkeypatch) -> None:
    evaluate = dense_reskin_quality.evaluate_dense_reskin_case

    def evaluate_with_nonfinite(case):
        result = evaluate(case)
        if case.case_id == "plane_3x3_center_hole":
            result["baseline"]["generation"]["reskin_support_mean"] = float("nan")
            result["baseline"]["finite_failure_count"] = 1
            result["candidate"]["generation"]["reskin_support_mean"] = float("inf")
            result["candidate"]["finite_failure_count"] = 1
        return result

    monkeypatch.setattr(dense_reskin_quality, "evaluate_dense_reskin_case", evaluate_with_nonfinite)

    gate = build_dense_reskin_promotion_gate()

    assert gate["passed"] is False
    assert "plane_3x3_center_hole:baseline_nonfinite_metric" in gate["reasons"]
    assert "plane_3x3_center_hole:nonfinite_metric" in gate["reasons"]
    first = gate["case_results"][0]
    assert first["baseline"]["generation"]["reskin_support_mean"] is None
    assert first["candidate"]["generation"]["reskin_support_mean"] is None
    json.dumps(gate, allow_nan=False)
    json_path, _, _ = write_dense_reskin_evidence(gate, tmp_path)
    assert json.loads(json_path.read_text())["passed"] is False


def test_out_of_volume_cell_becomes_machine_readable_gate_failure(
    passing_gate_inputs,
) -> None:
    case = controlled_dense_reskin_cases()[0]
    observed = FaultCell(
        7,
        7,
        7,
        0.9,
        0,
        90,
        generation="dense_reskin_observed",
        reskin_support=0.8,
    )
    out_of_volume = FaultCell(
        -1,
        7,
        7,
        0.9,
        0,
        90,
        generation="dense_reskin_generated",
        reskin_support=0.8,
    )
    metrics = dense_reskin_quality._policy_metrics(
        case,
        (FaultSkin.from_cells((observed, out_of_volume)),),
        {
            "input_cell_count": 1,
            "output_cell_count": 2,
            "observed_output_cell_count": 1,
            "generated_cell_count": 1,
        },
    )

    assert metrics["out_of_volume_cell_count"] == 1
    results, aggregate = copy.deepcopy(passing_gate_inputs)
    by_id = {result["case_id"]: result for result in results}
    by_id[case.case_id]["candidate"] = metrics
    assert f"{case.case_id}:out_of_volume_cell_count" in dense_reskin_quality._gate_reasons(
        results,
        aggregate,
        deterministic=True,
    )


def test_dense_gate_requires_valid_mask_rejection_evidence(monkeypatch) -> None:
    evaluate = dense_reskin_quality.evaluate_dense_reskin_case

    def evaluate_without_mask_rejection(case):
        result = evaluate(case)
        if case.case_id == "valid_mask_barrier":
            result["candidate"]["generation"]["reskin_rejected_invalid_mask_count"] = 0
        return result

    monkeypatch.setattr(
        dense_reskin_quality,
        "evaluate_dense_reskin_case",
        evaluate_without_mask_rejection,
    )

    gate = build_dense_reskin_promotion_gate()

    assert "valid_mask_barrier:invalid_mask_rejection_not_exercised" in gate["reasons"]


@pytest.mark.parametrize(
    ("case_id", "policy", "path", "value", "expected_reason"),
    (
        (
            "plane_3x3_center_hole",
            "baseline",
            ("finite_failure_count",),
            1,
            "plane_3x3_center_hole:baseline_nonfinite_metric",
        ),
        (
            "plane_3x3_center_hole",
            "candidate",
            ("finite_failure_count",),
            1,
            "plane_3x3_center_hole:nonfinite_metric",
        ),
        (
            "plane_3x3_center_hole",
            "candidate",
            ("duplicate_rounded_cell_index_count",),
            1,
            "plane_3x3_center_hole:duplicate_rounded_cell_index",
        ),
        (
            "plane_3x3_center_hole",
            "candidate",
            ("link_topology", "reciprocal_link_violation_count"),
            1,
            "plane_3x3_center_hole:reciprocal_link_violation_count",
        ),
        (
            "plane_3x3_center_hole",
            "candidate",
            ("link_topology", "cross_skin_link_count"),
            1,
            "plane_3x3_center_hole:cross_skin_link_count",
        ),
        (
            "plane_3x3_center_hole",
            "candidate",
            ("link_topology", "self_link_count"),
            1,
            "plane_3x3_center_hole:self_link_count",
        ),
        (
            "valid_mask_barrier",
            "candidate",
            ("generated_on_invalid_mask_count",),
            1,
            "valid_mask_barrier:generated_on_invalid_mask_count",
        ),
        (
            "prior_occupancy_barrier",
            "candidate",
            ("generated_on_prior_occupancy_count",),
            1,
            "prior_occupancy_barrier:generated_on_prior_occupancy_count",
        ),
        (
            "volume_boundary_surface",
            "candidate",
            ("out_of_volume_cell_count",),
            1,
            "volume_boundary_surface:out_of_volume_cell_count",
        ),
        (
            "volume_boundary_surface",
            "candidate",
            ("clamped_artifact_count",),
            1,
            "volume_boundary_surface:clamped_artifact_count",
        ),
        (
            "plane_3x3_center_hole",
            "candidate",
            ("truth", "truth_hole_recovered_count"),
            0,
            "plane_3x3_center_hole:hole_not_recovered",
        ),
        (
            "plane_3x3_center_hole",
            "candidate",
            ("truth", "truth_buffer_outside_generated_cell_count"),
            1,
            "plane_3x3_center_hole:generated_outside_truth_buffer",
        ),
        (
            "plane_4x4_internal_2x2_hole",
            "candidate",
            ("truth", "overlap", "recall"),
            "baseline_recall",
            "plane_4x4_internal_2x2_hole:recall_not_improved",
        ),
        (
            "plane_4x4_internal_2x2_hole",
            "candidate",
            ("truth", "truth_buffer_outside_generated_cell_count"),
            1,
            "plane_4x4_internal_2x2_hole:generated_outside_truth_buffer",
        ),
        (
            "low_support_gap",
            "candidate",
            ("generation", "reskin_generated_cell_count"),
            1,
            "low_support_gap:generated_cell",
        ),
        (
            "valid_mask_barrier",
            "candidate",
            ("generation", "reskin_rejected_invalid_mask_count"),
            0,
            "valid_mask_barrier:invalid_mask_rejection_not_exercised",
        ),
        (
            "prior_occupancy_barrier",
            "candidate",
            ("generation", "reskin_rejected_prior_skin_collision_count"),
            0,
            "prior_occupancy_barrier:collision_not_exercised",
        ),
        (
            "volume_boundary_surface",
            "candidate",
            ("truth", "truth_hole_recovered_count"),
            0,
            "volume_boundary_surface:boundary_hole_not_recovered",
        ),
        (
            "volume_boundary_surface",
            "candidate",
            ("generation", "reskin_generated_cell_count"),
            0,
            "volume_boundary_surface:dense_generation_not_exercised",
        ),
        (
            "parallel_surfaces",
            "candidate",
            ("truth_fault_id_bridge_count",),
            1,
            "parallel_surfaces:truth_fault_id_bridge",
        ),
        (
            "corner_touch_orientation_boundary",
            "candidate",
            ("truth_fault_id_bridge_count",),
            1,
            "corner_touch_orientation_boundary:truth_fault_id_bridge",
        ),
    ),
)
def test_every_dense_hard_gate_failure_reason(
    passing_gate_inputs,
    case_id,
    policy,
    path,
    value,
    expected_reason,
) -> None:
    base_results, base_aggregate = passing_gate_inputs
    results = copy.deepcopy(base_results)
    aggregate = copy.deepcopy(base_aggregate)
    by_id = {result["case_id"]: result for result in results}
    if value == "baseline_recall":
        value = by_id[case_id]["baseline"]["truth"]["overlap"]["recall"]
    _set_nested(by_id[case_id][policy], path, value)

    reasons = dense_reskin_quality._gate_reasons(
        results,
        aggregate,
        deterministic=True,
    )

    assert reasons == [expected_reason]


def test_dense_gate_rejects_nondeterministic_payload(passing_gate_inputs) -> None:
    results, aggregate = passing_gate_inputs

    reasons = dense_reskin_quality._gate_reasons(
        results,
        aggregate,
        deterministic=False,
    )

    assert reasons == ["canonical_payload_not_deterministic"]


@pytest.mark.parametrize(
    ("field", "boundary", "failure_direction", "expected_reason"),
    (
        (
            "candidate_buffered_recall_mean",
            "baseline",
            -1.0,
            "aggregate:buffered_recall_regressed",
        ),
        (
            "buffered_precision_delta",
            -0.02,
            -1.0,
            "aggregate:buffered_precision_regressed_over_0.02",
        ),
        (
            "symmetric_chamfer_mean_delta",
            0.25,
            1.0,
            "aggregate:symmetric_chamfer_regressed_over_0.25",
        ),
        (
            "small_skin_cell_fraction_delta",
            0.05,
            1.0,
            "aggregate:small_skin_cell_fraction_increased_over_0.05",
        ),
    ),
)
def test_dense_aggregate_gate_passes_at_boundary_and_fails_beyond_it(
    passing_gate_inputs,
    field,
    boundary,
    failure_direction,
    expected_reason,
) -> None:
    results, base_aggregate = passing_gate_inputs
    aggregate = copy.deepcopy(base_aggregate)
    if boundary == "baseline":
        boundary = aggregate["baseline_buffered_recall_mean"]
    aggregate[field] = boundary
    assert (
        dense_reskin_quality._gate_reasons(
            results,
            aggregate,
            deterministic=True,
        )
        == []
    )

    aggregate[field] = np.nextafter(boundary, failure_direction * np.inf)
    assert dense_reskin_quality._gate_reasons(
        results,
        aggregate,
        deterministic=True,
    ) == [expected_reason]


def test_f3_skin_only_pair_shares_parent_and_branches_policy(monkeypatch) -> None:
    shape = (15, 15, 15)
    fv = np.zeros(shape, dtype=np.float32)
    vp = np.zeros(shape, dtype=np.float32)
    vt = np.full(shape, 90.0, dtype=np.float32)
    for i3 in range(6, 9):
        for i1 in range(6, 9):
            if (i1, i3) != (7, 7):
                fv[i3, 7, i1] = 0.9
    config = SyntheticSkinningConfig(
        method="quality",
        growth_source="pre_thin",
        min_likelihood=0.5,
        min_skin_size=1,
        ru=3,
        rv=5,
        rw=5,
        reskin=True,
    )
    distance_transform = synthetic_metrics.distance_transform_edt
    transform_shapes = []

    def counted_distance_transform(values):
        transform_shapes.append(values.shape)
        return distance_transform(values)

    monkeypatch.setattr(
        synthetic_metrics,
        "distance_transform_edt",
        counted_distance_transform,
    )

    report = compare_reskin_policies_from_parent(
        fv=fv,
        fvt=fv.copy(),
        vp=vp,
        vt=vt,
        skinning_config=config,
    )

    assert report["upstream_parent_fingerprint_identical"] is True
    assert transform_shapes == [shape, shape, shape]
    assert report["contrast"]["generated_cell_count_delta"] == 1
    assert set(report["policies"]) == {"existing_cells_v1", "reference_dense_v1"}
    assert (
        report["policies"]["reference_dense_v1"]["parent_ridge_surface"]["overlap"][
            "buffered_recall"
        ]
        == 1.0
    )


def test_f3_parent_fingerprint_includes_scanner_target_mask() -> None:
    shape = (5, 5, 5)
    values = np.zeros(shape, dtype=np.float32)
    scanner_mask = np.zeros(shape, dtype=np.bool_)
    config = SyntheticSkinningConfig(
        method="quality",
        min_skin_size=1,
        reskin=True,
    )

    without_target = compare_reskin_policies_from_parent(
        fv=values,
        fvt=values,
        vp=values,
        vt=values,
        skinning_config=config,
        scanner_target_positive_mask=scanner_mask,
    )
    scanner_mask[2, 2, 2] = True
    with_target = compare_reskin_policies_from_parent(
        fv=values,
        fvt=values,
        vp=values,
        vt=values,
        skinning_config=config,
        scanner_target_positive_mask=scanner_mask,
    )

    assert (
        without_target["upstream_parent_fingerprint"] != with_target["upstream_parent_fingerprint"]
    )


def test_f3_pair_writes_json_csv_markdown_and_canonical_skins(tmp_path) -> None:
    shape = (15, 15, 15)
    fv = np.zeros(shape, dtype=np.float32)
    vp = np.zeros(shape, dtype=np.float32)
    vt = np.full(shape, 90.0, dtype=np.float32)
    fv[6:9, 7, 6:9] = 0.9
    config = SyntheticSkinningConfig(
        method="quality",
        growth_source="pre_thin",
        min_likelihood=0.5,
        min_skin_size=1,
        ru=3,
        rv=5,
        rw=5,
        reskin=True,
    )
    report = compare_reskin_policies_from_parent(
        fv=fv,
        fvt=fv.copy(),
        vp=vp,
        vt=vt,
        skinning_config=config,
    )

    paths = write_f3_reskin_policy_comparison(report, tmp_path)

    assert {path.name for path in paths} == {
        "reskin_policy_comparison.json",
        "reskin_policy_metrics.csv",
        "reskin_policy_comparison.md",
        "existing_cells_v1_skins.json",
        "reference_dense_v1_skins.json",
    }
    assert json.loads(paths[0].read_text())["upstream_parent_fingerprint_identical"] is True
    with paths[1].open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == [
            "schema_version",
            "row_type",
            "policy",
            "reskin_policy",
            "parent_fingerprint",
            "metric",
            "value",
            "status",
        ]
        rows = list(reader)
    assert any(row["metric"] == "parent_ridge_overlap.buffered_recall" for row in rows)
    assert "Ridge buffered precision/recall" in paths[2].read_text()
    assert json.loads(paths[3].read_text())["format_version"] == 2
    assert json.loads(paths[4].read_text())["format_version"] == 2


def test_f3_bundle_pair_reads_q_qual_parent_artifacts(tmp_path, monkeypatch) -> None:
    shape = (15, 15, 15)
    fv = np.zeros(shape, dtype=np.float32)
    fvt = np.zeros(shape, dtype=np.float32)
    vp = np.zeros(shape, dtype=np.float32)
    vt = np.full(shape, 90.0, dtype=np.float32)
    fvt[6:9, 7, 6:9] = 0.9
    config = resolve_workflow_settings(
        workflow_mode="quality",
        skinning_config=SyntheticSkinningConfig(min_skin_size=1, reskin=True),
    ).skinning_config
    assert config.boundary_skinner_fallback is True
    scanner_fingerprint = "c" * 64
    voting_fingerprint = "a" * 64
    thinning_fingerprint = "b" * 64
    scanner = tmp_path / "stages" / "scanner" / scanner_fingerprint
    voting = tmp_path / "stages" / "voting" / voting_fingerprint
    thinning = tmp_path / "stages" / "thinning" / thinning_fingerprint
    scanner.mkdir(parents=True)
    voting.mkdir(parents=True)
    thinning.mkdir(parents=True)
    for path, values in (
        (scanner / "ft.dat", fvt),
        (voting / "fv.dat", fv),
        (voting / "vp.dat", vp),
        (voting / "vt.dat", vt),
        (thinning / "fvt.dat", fvt),
    ):
        np.asarray(values, dtype=">f4").tofile(path)
    report = json.dumps({"shape": list(shape)})
    (scanner / "report.json").write_text(report, encoding="utf-8")
    (voting / "report.json").write_text(report, encoding="utf-8")
    (thinning / "report.json").write_text(report, encoding="utf-8")
    cell = SimpleNamespace(
        label="Q-QUAL",
        backend="quality",
        workflow="quality",
        skinning_enabled=True,
        resolved_config={"skinning": asdict(config)},
        stages=SimpleNamespace(
            scanner=scanner_fingerprint,
            voting=voting_fingerprint,
            thinning=thinning_fingerprint,
        ),
    )
    monkeypatch.setattr(
        f3_result_module,
        "load_f3d_mode_comparison_result",
        lambda *args, **kwargs: SimpleNamespace(cells=(cell,)),
    )

    paths = compare_reskin_policies_from_bundle(tmp_path)

    payload = json.loads(paths[0].read_text())
    assert payload["source_cell"] == "Q-QUAL"
    assert payload["upstream_parent_stage_fingerprints"] == {
        "scanner": scanner_fingerprint,
        "voting": voting_fingerprint,
        "thinning": thinning_fingerprint,
    }
    for policy in ("existing_cells_v1", "reference_dense_v1"):
        policy_payload = payload["policies"][policy]
        assert policy_payload["diagnostics"]["fallback_enabled"] is True
        assert policy_payload["diagnostics"]["fallback_used"] is True
        assert (
            policy_payload["diagnostics"]["skin_scanner_target_positive_edge_shell_fraction"]
            is not None
        )
        assert policy_payload["generation"]["reskin_input_cell_count"] == 0
        assert policy_payload["generation"]["reskin_output_cell_count"] == 0
        assert policy_payload["generation"]["reskin_observed_output_cell_count"] == 0
        assert {
            cell["generation"]
            for skin in policy_payload["canonical_skin_artifact"]["skins"]
            for cell in skin["cells"]
        } == {"connected_component"}
    assert paths[0].parent == tmp_path / "reskin_policy_comparison"
    comparison_root = paths[0].parent
    completion_path = comparison_root / F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE
    assert completion_path.is_file()
    snapshots = {path: path.read_bytes() for path in paths}
    completion_snapshot = completion_path.read_bytes()

    assert compare_reskin_policies_from_bundle(tmp_path, resume=True) == paths
    assert {path: path.read_bytes() for path in paths} == snapshots
    assert validate_f3_reskin_policy_comparison(tmp_path, deep=True) == paths

    tamper_cases = (
        (
            lambda report: report["policies"]["existing_cells_v1"]["generation"].__setitem__(
                "reskin_output_cell_count",
                report["policies"]["existing_cells_v1"]["generation"]["reskin_output_cell_count"]
                + 1,
            ),
            "generation metrics mismatch",
        ),
        (
            lambda report: report["policies"]["existing_cells_v1"]["skin_topology"].__setitem__(
                "cell_count",
                report["policies"]["existing_cells_v1"]["skin_topology"]["cell_count"] + 1,
            ),
            "skin topology metrics mismatch",
        ),
        (
            lambda report: report["policies"]["existing_cells_v1"].__setitem__(
                "duplicate_rounded_cell_index_count",
                report["policies"]["existing_cells_v1"]["duplicate_rounded_cell_index_count"] + 1,
            ),
            "duplicate count mismatch",
        ),
        (
            lambda report: report["policies"]["existing_cells_v1"]["link_topology"].__setitem__(
                "linked_component_count",
                report["policies"]["existing_cells_v1"]["link_topology"]["linked_component_count"]
                + 1,
            ),
            "linked component count mismatch",
        ),
        (
            lambda report: report["contrast"].__setitem__(
                "generated_cell_count_delta",
                report["contrast"]["generated_cell_count_delta"] + 1,
            ),
            "contrast metrics mismatch",
        ),
    )
    for tamper, expected_error in tamper_cases:
        report_payload = json.loads(snapshots[paths[0]])
        tamper(report_payload)
        paths[0].write_text(
            reskin_policy_comparison._canonical_json(report_payload) + "\n",
            encoding="utf-8",
        )
        paths[1].write_text(
            reskin_policy_comparison._comparison_csv(report_payload),
            encoding="utf-8",
        )
        paths[2].write_text(
            reskin_policy_comparison._comparison_markdown(report_payload),
            encoding="utf-8",
        )
        completion = json.loads(completion_snapshot)
        completion["artifact_files"] = {
            path.name: reskin_policy_comparison.artifact_file_metadata(path) for path in paths
        }
        completion_path.write_text(
            reskin_policy_comparison._canonical_json(completion) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=expected_error):
            validate_f3_reskin_policy_comparison(tmp_path)
        for path, content in snapshots.items():
            path.write_bytes(content)
        completion_path.write_bytes(completion_snapshot)

    paths[4].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        compare_reskin_policies_from_bundle(tmp_path, resume=True)
