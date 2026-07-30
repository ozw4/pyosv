import csv
import copy
import itertools
import json
import weakref
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
from pyosv.evaluation.synthetic_quality.variants import SkinningPatch, VariantSpec
from pyosv.skin import FaultSkin
from pyosv.synthetic_metrics import reskin_generation_metrics, skin_link_topology_metrics


def _set_nested(mapping, path, value) -> None:
    target = mapping
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = value


def _write_rehashed_comparison_report(
    report,
    paths,
    completion_path,
    completion,
) -> None:
    paths[0].write_text(
        reskin_policy_comparison._canonical_json(report) + "\n",
        encoding="utf-8",
    )
    paths[1].write_text(
        reskin_policy_comparison._comparison_csv(report),
        encoding="utf-8",
    )
    paths[2].write_text(
        reskin_policy_comparison._comparison_markdown(report),
        encoding="utf-8",
    )
    completion["report_semantic_evidence_sha256"] = (
        reskin_policy_comparison._report_semantic_evidence_digest(report)
    )
    completion["artifact_files"] = {
        path.name: reskin_policy_comparison.artifact_file_metadata(path) for path in paths
    }
    completion_path.write_text(
        reskin_policy_comparison._canonical_json(completion) + "\n",
        encoding="utf-8",
    )


def _publication_runtime_identity():
    identity = copy.deepcopy(reskin_policy_comparison.numerical_runtime_identity())
    identity.update(
        {
            "requested_acceleration_mode": "auto",
            "pyosv_accel": "auto",
            "numba_available": True,
            "numba_version": "test-numba",
            "numba_jit": {"status": "enabled", "enabled": True},
            "effective_acceleration_state": "numba_jit_enabled",
            "python_hash_seed": "0",
            "numpy_build": {
                "status": "available",
                "sha256": "a" * 64,
            },
            "numpy_runtime_cpu": {
                "status": "available",
                "features": ["TEST_CPU_FEATURE"],
            },
            "numpy_runtime_blas": {
                "status": "available",
                "libraries": [
                    {
                        "implementation": "test-blas",
                        "version": "1",
                        "threading_layer": "pthreads",
                        "architecture": "test-architecture",
                        "effective_thread_count": 1,
                    }
                ],
            },
            "scipy_build": {
                "status": "available",
                "sha256": "b" * 64,
            },
        }
    )
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        identity["thread_environment"][name] = "1"
    identity["numba_environment"]["NUMBA_DISABLE_JIT"] = "0"
    identity["numba_environment"]["NUMBA_NUM_THREADS"] = "1"
    return identity


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
        report["policies"]["reference_dense_v1"]["parent_ridge_surface"]["metrics"]["overlap"][
            "buffered_recall"
        ]
        == 1.0
    )


def test_f3_skin_only_pair_applies_variant_skinning_patch(monkeypatch) -> None:
    shape = (5, 5, 5)
    values = np.zeros(shape, dtype=np.float32)
    config = SyntheticSkinningConfig(
        method="quality",
        min_likelihood=0.5,
        min_skin_size=1,
        reskin=True,
    )
    variant = VariantSpec(
        "patched-skinning",
        skinning=SkinningPatch(
            min_likelihood=0.75,
            override_min_likelihood=True,
            accepted_occupancy_radius=0,
        ),
        experimental=False,
    )
    original_execute = reskin_policy_comparison.execute_skinning_phase3d
    executed_configs = {}

    def tracked_execute(**kwargs):
        settings = kwargs["skinning_settings"]
        executed_configs[settings.reskin_policy] = settings
        return original_execute(**kwargs)

    monkeypatch.setattr(
        reskin_policy_comparison,
        "execute_skinning_phase3d",
        tracked_execute,
    )

    report = compare_reskin_policies_from_parent(
        fv=values,
        fvt=values,
        vp=values,
        vt=values,
        skinning_config=config,
        variant_spec=variant,
    )

    effective = report["resolved_config"]["effective_skinning_configs"]
    assert set(executed_configs) == {"existing_cells_v1", "reference_dense_v1"}
    for policy, settings in executed_configs.items():
        assert settings.min_likelihood == 0.75
        assert settings.accepted_occupancy_radius == 0
        assert effective[policy] == asdict(settings)


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


@pytest.mark.parametrize(
    ("candidate_indices", "reference_indices", "status"),
    (
        ((), (), "both_empty"),
        ((), (0,), "candidate_empty"),
        ((0,), (), "reference_empty"),
    ),
)
def test_f3_parent_surface_evidence_preserves_empty_semantics(
    candidate_indices,
    reference_indices,
    status,
) -> None:
    candidate = np.zeros(7, dtype=bool)
    reference = np.zeros_like(candidate)
    candidate[list(candidate_indices)] = True
    reference[list(reference_indices)] = True

    result = synthetic_metrics.surface_comparison_metrics_with_evidence(
        {"policy": candidate},
        reference,
        radius=2.0,
        positive_epsilon=1.0e-6,
    )["policy"]

    assert result["evidence"]["surface_distance"]["status"] == status
    assert synthetic_metrics.metrics_from_surface_evidence(result["evidence"]) == result["metrics"]


def test_surface_comparison_accepts_rounding_at_constant_distance_bound() -> None:
    candidate = np.zeros((2, 29), dtype=bool)
    reference = np.zeros_like(candidate)
    candidate[0, 0::3] = True
    reference[1, 1::3] = True

    metrics = synthetic_metrics.surface_comparison_metrics(
        {"policy": candidate},
        reference,
        radius=2.0,
    )["policy"]["surface_distance"]

    assert metrics["candidate_to_truth_mean"] == pytest.approx(np.sqrt(2.0))
    assert metrics["truth_to_candidate_mean"] == pytest.approx(np.sqrt(2.0))


@pytest.mark.parametrize(
    ("candidate_indices", "expected_lower", "expected_upper", "expected_weight"),
    (
        ((0, 1, 3), 1, 1, 0.0),
        ((0, 1, 3, 6), 1, 2, 0.5),
    ),
)
def test_f3_parent_surface_quantiles_use_bounded_order_statistic_evidence(
    candidate_indices,
    expected_lower,
    expected_upper,
    expected_weight,
) -> None:
    candidate = np.zeros(7, dtype=bool)
    reference = np.zeros_like(candidate)
    candidate[list(candidate_indices)] = True
    reference[0] = True

    result = synthetic_metrics.surface_comparison_metrics_with_evidence(
        {"policy": candidate},
        reference,
        radius=2.0,
        positive_epsilon=1.0e-6,
    )["policy"]
    median = result["evidence"]["surface_distance"]["candidate_to_reference"]["quantiles"]["median"]

    assert median["lower_rank"] == expected_lower
    assert median["upper_rank"] == expected_upper
    assert median["interpolation_weight"] == expected_weight
    median["rank"] += 0.25
    with pytest.raises(ValueError, match="rank mismatch"):
        synthetic_metrics.metrics_from_surface_evidence(result["evidence"])


@pytest.mark.parametrize(
    ("path", "value", "expected_error"),
    (
        (
            ("overlap", "exact_intersection_count"),
            3,
            "intersection count is inconsistent",
        ),
        (("overlap", "exact_union_count"), 1, "union count is inconsistent"),
        (
            ("overlap", "candidate_in_reference_buffer_count"),
            3,
            "candidate buffer count is inconsistent",
        ),
        (
            ("overlap", "candidate_in_reference_buffer_count"),
            0,
            "candidate buffer count is inconsistent",
        ),
        (
            ("overlap", "reference_in_candidate_buffer_count"),
            0,
            "reference buffer count is inconsistent",
        ),
        (
            (
                "surface_distance",
                "candidate_to_reference",
                "mean_numerator",
            ),
            2.0,
            "mean accumulator is inconsistent",
        ),
        (
            (
                "surface_distance",
                "candidate_to_reference",
                "quantiles",
                "p90",
                "interpolation_weight",
            ),
            0.25,
            "interpolation weight mismatch",
        ),
    ),
)
def test_f3_parent_surface_rejects_invalid_scalar_evidence(
    path,
    value,
    expected_error,
) -> None:
    candidate = np.array([True, True, False, False])
    reference = np.array([True, False, False, True])
    evidence = synthetic_metrics.surface_comparison_metrics_with_evidence(
        {"policy": candidate},
        reference,
        radius=1.0,
        positive_epsilon=1.0e-6,
    )["policy"]["evidence"]
    _set_nested(evidence, path, value)

    with pytest.raises(ValueError, match=expected_error):
        synthetic_metrics.metrics_from_surface_evidence(evidence)


def test_surface_comparison_releases_each_candidate_distance_field_before_next(
    monkeypatch,
) -> None:
    masks = {
        "baseline": np.array([True, False, False]),
        "candidate": np.array([False, False, True]),
    }
    reference = np.array([False, True, False])
    original = synthetic_metrics._surface_distance_field
    first_candidate_distance = None
    call_count = 0

    def tracked(mask):
        nonlocal call_count, first_candidate_distance
        call_count += 1
        if call_count == 3:
            assert first_candidate_distance is not None
            assert first_candidate_distance() is None
        result = original(mask)
        if call_count == 2:
            first_candidate_distance = weakref.ref(result)
        return result

    monkeypatch.setattr(synthetic_metrics, "_surface_distance_field", tracked)

    synthetic_metrics.surface_comparison_metrics_with_evidence(
        masks,
        reference,
        radius=1.0,
        positive_epsilon=0.0,
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
    tampered_diagnostics = copy.deepcopy(report)
    tampered_diagnostics["policies"]["existing_cells_v1"]["diagnostics"]["reskin"]["attempted"][
        "processed_skin_count"
    ] += 1
    assert reskin_policy_comparison._report_semantic_evidence_digest(
        tampered_diagnostics
    ) != reskin_policy_comparison._report_semantic_evidence_digest(report)

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
            "comparison_config_fingerprint",
            "metric_evidence_schema_version",
            "validation_level",
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
        resolved_config={
            "skinning": asdict(config),
            "variant": asdict(VariantSpec("f3-canonical", experimental=False)),
        },
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
    runtime_identity = _publication_runtime_identity()
    monkeypatch.setattr(
        reskin_policy_comparison,
        "numerical_runtime_identity",
        lambda: runtime_identity,
    )
    dataset_file_size = int(np.prod(shape)) * np.dtype(">f4").itemsize
    computation = {
        "artifact_schema_version": 1,
        "stage_contract_version": 1,
        "fingerprint_contract_version": 4,
        "plan": {"fixture": "small-reskin-comparison"},
        "dataset_identity": {
            "dataset_id": reskin_policy_comparison.F3_DATASET_ID,
            "files": [
                {
                    "role": "input",
                    "size": dataset_file_size,
                    "sha256": "d" * 64,
                    "shape": list(shape),
                    "storage_dtype": ">f4",
                }
            ],
        },
        "implementation_identity": {"fixture": "small-reskin-comparison-v1"},
        "runtime_identity": runtime_identity,
    }
    source_run_fingerprint = reskin_policy_comparison.canonical_fingerprint(computation)
    (tmp_path / "run_manifest.json").write_text(
        reskin_policy_comparison._canonical_json(
            {
                **computation,
                "run_fingerprint": source_run_fingerprint,
                "provenance": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    paths = compare_reskin_policies_from_bundle(tmp_path)

    payload = json.loads(paths[0].read_text())
    assert payload["source_cell"] == "Q-QUAL"
    assert payload["schema_version"] == 4
    assert payload["metric_evidence_schema_version"] == 1
    assert payload["validation_level"] == "shallow"
    assert payload["comparison_config_fingerprint"] == (
        reskin_policy_comparison.canonical_fingerprint(payload["resolved_config"])
    )
    assert payload["source_run_fingerprint"] == source_run_fingerprint
    assert payload["source_runtime_identity_schema_version"] == 3
    assert (
        payload["source_runtime_identity_sha256"] == payload["comparison_runtime_identity_sha256"]
    )
    assert payload["comparison_implementation_identity"]["algorithm_modules"]
    assert {
        "_skinner/reference.py",
        "evaluation/synthetic_quality/quality_metrics.py",
        "experimental/boundary_thinning.py",
        "filters.py",
        "geometry.py",
    } <= set(payload["comparison_implementation_identity"]["algorithm_modules"])
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
    completion_payload = json.loads(completion_path.read_text())
    assert completion_payload["validation_level"] == "shallow"
    with pytest.raises(ValueError, match="deep validation required"):
        validate_f3_reskin_policy_comparison(tmp_path, require_deep=True)
    snapshots = {path: path.read_bytes() for path in paths}
    completion_snapshot = completion_path.read_bytes()
    manifest_path = tmp_path / "run_manifest.json"
    manifest_snapshot = manifest_path.read_bytes()

    def fail_validation():
        raise ValueError("injected final validation failure")

    def successful_validation():
        return paths

    original_atomic_write = reskin_policy_comparison.atomic_write_artifact

    def fail_markdown_write(path, payload, *, temporary_prefix):
        if Path(path).name == "reskin_policy_comparison.md":
            raise OSError("injected promotion write failure")
        return original_atomic_write(path, payload, temporary_prefix=temporary_prefix)

    def fail_staged_validation(*args, **kwargs):
        raise ValueError("injected staged validation failure")

    for existing, failure in itertools.product((False, True), ("write", "validation")):
        transaction_root = tmp_path / f"transaction-{existing}-{failure}"
        if existing:
            transaction_root.mkdir()
            (transaction_root / "reskin_policy_metrics.csv").write_bytes(b"existing\n")
            (transaction_root / ".complete.json.tmp-interrupted").write_bytes(b"temporary\n")
        before_paths = (
            {path.relative_to(transaction_root).as_posix() for path in transaction_root.rglob("*")}
            if existing
            else set()
        )
        before_files = (
            {
                path.relative_to(transaction_root).as_posix(): path.read_bytes()
                for path in transaction_root.rglob("*")
                if path.is_file()
            }
            if existing
            else {}
        )
        with monkeypatch.context() as transaction_patch:
            if failure == "write":
                transaction_patch.setattr(
                    reskin_policy_comparison,
                    "atomic_write_artifact",
                    fail_markdown_write,
                )
                expected_error = OSError
            else:
                transaction_patch.setattr(
                    reskin_policy_comparison,
                    "validate_f3_reskin_policy_comparison",
                    fail_staged_validation,
                )
                expected_error = ValueError
            with pytest.raises(expected_error, match="injected"):
                compare_reskin_policies_from_bundle(
                    tmp_path,
                    output_dir=transaction_root,
                    resume=existing,
                )
        if existing:
            assert {
                path.relative_to(transaction_root).as_posix()
                for path in transaction_root.rglob("*")
            } == before_paths
            assert {
                path.relative_to(transaction_root).as_posix(): path.read_bytes()
                for path in transaction_root.rglob("*")
                if path.is_file()
            } == before_files
        else:
            assert not transaction_root.exists()
        assert not tuple(transaction_root.parent.glob(f".{transaction_root.name}.generation-tmp-*"))

    def fail_temporary_directory_creation(*_args, **_kwargs):
        raise OSError("injected temporary-directory failure")

    missing_parent = tmp_path / "missing-output-parent"
    with monkeypatch.context() as temporary_patch:
        temporary_patch.setattr(
            reskin_policy_comparison.tempfile,
            "mkdtemp",
            fail_temporary_directory_creation,
        )
        with pytest.raises(OSError, match="injected temporary-directory failure"):
            compare_reskin_policies_from_bundle(
                tmp_path,
                output_dir=missing_parent / "comparison",
            )
    assert not missing_parent.exists()

    for failure in ("write", "validation"):
        with monkeypatch.context() as promotion_patch:
            validator = fail_validation
            if failure == "write":
                promotion_patch.setattr(
                    reskin_policy_comparison,
                    "atomic_write_artifact",
                    fail_markdown_write,
                )
                validator = successful_validation
            with pytest.raises((OSError, ValueError), match="injected"):
                reskin_policy_comparison._promote_comparison_completion(
                    comparison_root,
                    validator=validator,
                )
        assert {path: path.read_bytes() for path in paths} == snapshots
        assert completion_path.read_bytes() == completion_snapshot
        assert not tuple(comparison_root.parent.glob(f".{comparison_root.name}.promotion-tmp-*"))

    monkeypatch.setattr(
        reskin_policy_comparison,
        "numerical_runtime_identity",
        lambda: pytest.fail("shallow comparison resume queried current runtime"),
    )
    assert compare_reskin_policies_from_bundle(tmp_path, resume=True) == paths
    assert validate_f3_reskin_policy_comparison(tmp_path) == paths
    assert {path: path.read_bytes() for path in paths} == snapshots
    monkeypatch.setattr(
        reskin_policy_comparison,
        "numerical_runtime_identity",
        lambda: runtime_identity,
    )
    assert validate_f3_reskin_policy_comparison(tmp_path, deep=True) == paths

    invalid_manifest = json.loads(manifest_snapshot)
    invalid_manifest["runtime_identity"]["python_hash_seed"] = "invalid"
    invalid_computation = {
        name: invalid_manifest[name] for name in reskin_policy_comparison._RUN_COMPUTATION_FIELDS
    }
    invalid_manifest["run_fingerprint"] = reskin_policy_comparison.canonical_fingerprint(
        invalid_computation
    )
    manifest_path.write_text(
        reskin_policy_comparison._canonical_json(invalid_manifest) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="publication runtime identity is invalid"):
        validate_f3_reskin_policy_comparison(tmp_path)
    assert {path: path.read_bytes() for path in paths} == snapshots
    assert completion_path.read_bytes() == completion_snapshot
    manifest_path.write_bytes(manifest_snapshot)

    original_open_parent_dat = reskin_policy_comparison._open_parent_dat
    monkeypatch.setattr(
        reskin_policy_comparison,
        "_open_parent_dat",
        lambda *args, **kwargs: pytest.fail("runtime mismatch opened a parent DAT"),
    )

    def change_acceleration(identity):
        identity["numba_available"] = False
        identity["numba_version"] = None
        identity["numba_jit"] = {"status": "not_applicable", "enabled": None}
        identity["effective_acceleration_state"] = "python_only"

    def change_jit(identity):
        identity["numba_jit"] = {"status": "disabled", "enabled": False}
        identity["effective_acceleration_state"] = "numba_jit_disabled"
        identity["numba_environment"]["NUMBA_DISABLE_JIT"] = "1"

    runtime_mismatches = {
        "acceleration": change_acceleration,
        "jit": change_jit,
        "cpu": lambda identity: identity["numpy_runtime_cpu"]["features"].append(
            "TEST_CPU_FEATURE_2"
        ),
        "blas": lambda identity: identity["numpy_runtime_blas"]["libraries"][0].__setitem__(
            "version", "2"
        ),
        "blas-threads": lambda identity: identity["numpy_runtime_blas"]["libraries"][0].__setitem__(
            "effective_thread_count", 2
        ),
        "scipy": lambda identity: identity["scipy_build"].__setitem__("sha256", "c" * 64),
        "hash": lambda identity: identity.__setitem__("python_hash_seed", "different"),
        "threads": lambda identity: identity["thread_environment"].__setitem__(
            "OMP_NUM_THREADS", "2"
        ),
    }
    for name, mutate_runtime in runtime_mismatches.items():
        changed_runtime = copy.deepcopy(runtime_identity)
        mutate_runtime(changed_runtime)
        monkeypatch.setattr(
            reskin_policy_comparison,
            "numerical_runtime_identity",
            lambda changed_runtime=changed_runtime: changed_runtime,
        )
        mismatch_root = tmp_path / f"{name}-mismatch-comparison"
        with pytest.raises(
            ValueError,
            match="current (?:publication runtime identity is invalid|runtime identity does not match)",
        ):
            compare_reskin_policies_from_bundle(tmp_path, output_dir=mismatch_root)
        assert not mismatch_root.exists()
        with pytest.raises(
            ValueError,
            match="current (?:publication runtime identity is invalid|runtime identity does not match)",
        ):
            validate_f3_reskin_policy_comparison(tmp_path, deep=True)
        assert {path: path.read_bytes() for path in paths} == snapshots
        assert completion_path.read_bytes() == completion_snapshot

    def fail_runtime_identity():
        raise RuntimeError("injected runtime identity failure")

    monkeypatch.setattr(
        reskin_policy_comparison,
        "numerical_runtime_identity",
        fail_runtime_identity,
    )
    getter_failure_root = tmp_path / "getter-failure-comparison"
    with pytest.raises(ValueError, match="current publication runtime identity is invalid"):
        compare_reskin_policies_from_bundle(tmp_path, output_dir=getter_failure_root)
    assert not getter_failure_root.exists()
    with pytest.raises(ValueError, match="current publication runtime identity is invalid"):
        validate_f3_reskin_policy_comparison(tmp_path, deep=True)
    assert {path: path.read_bytes() for path in paths} == snapshots
    assert completion_path.read_bytes() == completion_snapshot

    monkeypatch.setattr(
        reskin_policy_comparison,
        "numerical_runtime_identity",
        lambda: runtime_identity,
    )
    monkeypatch.setattr(
        reskin_policy_comparison,
        "_open_parent_dat",
        original_open_parent_dat,
    )
    separate_root = tmp_path / "separate-comparison"
    separate_paths = compare_reskin_policies_from_bundle(
        tmp_path,
        output_dir=separate_root,
        deep=True,
    )
    separate_payload = json.loads(separate_paths[0].read_text())
    assert separate_payload["validation_level"] == "deep"
    assert (
        validate_f3_reskin_policy_comparison(
            tmp_path,
            output_dir=separate_root,
            require_deep=True,
        )
        == separate_paths
    )
    assert separate_payload["source_run_fingerprint"] == source_run_fingerprint
    assert (
        separate_payload["source_runtime_identity_sha256"]
        == payload["source_runtime_identity_sha256"]
    )

    source_config = copy.deepcopy(cell.resolved_config)
    cell.resolved_config["skinning"]["min_skin_size"] += 1
    with pytest.raises(ValueError, match="resolved config mismatch"):
        validate_f3_reskin_policy_comparison(tmp_path)
    cell.resolved_config = source_config

    config_tamper_cases = (
        (
            lambda report, completion: report["resolved_config"]["effective_skinning_configs"][
                "reference_dense_v1"
            ].__setitem__(
                "min_skin_size",
                report["resolved_config"]["effective_skinning_configs"]["reference_dense_v1"][
                    "min_skin_size"
                ]
                + 1,
            ),
            "differ outside reskin_policy",
        ),
        (
            lambda report, completion: report["policies"]["existing_cells_v1"].__setitem__(
                "reskin_policy",
                "reference_dense_v1",
            ),
            "policy identity mismatch",
        ),
        (
            lambda report, completion: (
                report["resolved_config"]["shared_skinning_config"].__setitem__(
                    "min_skin_size",
                    report["resolved_config"]["shared_skinning_config"]["min_skin_size"] + 1,
                ),
                report.__setitem__(
                    "comparison_config_fingerprint",
                    reskin_policy_comparison.canonical_fingerprint(report["resolved_config"]),
                ),
                completion.__setitem__(
                    "comparison_config_fingerprint",
                    report["comparison_config_fingerprint"],
                ),
            ),
            "resolved config mismatch",
        ),
        (
            lambda report, completion: (
                report.__setitem__("comparison_config_fingerprint", "e" * 64),
                completion.__setitem__("comparison_config_fingerprint", "e" * 64),
            ),
            "config fingerprint mismatch",
        ),
    )
    for tamper, expected_error in config_tamper_cases:
        report_payload = json.loads(snapshots[paths[0]])
        completion = json.loads(completion_snapshot)
        tamper(report_payload, completion)
        _write_rehashed_comparison_report(
            report_payload,
            paths,
            completion_path,
            completion,
        )
        with pytest.raises(ValueError, match=expected_error):
            validate_f3_reskin_policy_comparison(tmp_path)
        for path, content in snapshots.items():
            path.write_bytes(content)
        completion_path.write_bytes(completion_snapshot)

    deep_report = json.loads(separate_paths[0].read_text())
    deep_completion_path = separate_root / F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE
    deep_completion = json.loads(deep_completion_path.read_text())
    surface = deep_report["policies"]["existing_cells_v1"]["parent_ridge_surface"]
    surface["evidence"]["overlap"]["buffer_radius"] += 1.0
    surface["metrics"] = synthetic_metrics.metrics_from_surface_evidence(surface["evidence"])
    deep_report["contrast"] = reskin_policy_comparison._comparison_contrast(deep_report["policies"])
    _write_rehashed_comparison_report(
        deep_report,
        separate_paths,
        deep_completion_path,
        deep_completion,
    )
    assert (
        validate_f3_reskin_policy_comparison(
            tmp_path,
            output_dir=separate_root,
        )
        == separate_paths
    )
    with pytest.raises(ValueError, match="does not exactly match"):
        validate_f3_reskin_policy_comparison(
            tmp_path,
            output_dir=separate_root,
            deep=True,
        )

    def tamper_parent_surface_metric(report):
        report["policies"]["existing_cells_v1"]["parent_ridge_surface"]["metrics"]["overlap"][
            "buffered_precision"
        ] = 0.5
        report["contrast"] = reskin_policy_comparison._comparison_contrast(report["policies"])

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
        (
            tamper_parent_surface_metric,
            "metrics/evidence mismatch",
        ),
        (
            lambda report: report["policies"]["existing_cells_v1"]["diagnostics"]["reskin"][
                "attempted"
            ].__setitem__(
                "output_cell_count",
                report["policies"]["existing_cells_v1"]["diagnostics"]["reskin"]["attempted"][
                    "output_cell_count"
                ]
                + 1,
            ),
            "reskin diagnostics are invalid",
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
        completion["report_semantic_evidence_sha256"] = (
            reskin_policy_comparison._report_semantic_evidence_digest(report_payload)
        )
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

    provenance_tamper_cases = (
        ("report", "source_run_fingerprint", "e" * 64, "source run fingerprint"),
        ("completion", "source_run_fingerprint", "e" * 64, "source run fingerprint"),
        ("report", "source_runtime_identity_sha256", "e" * 64, "source runtime digest"),
        (
            "completion",
            "comparison_runtime_identity_sha256",
            "e" * 64,
            "runtime digest",
        ),
        (
            "completion",
            "comparison_implementation_identity_sha256",
            "e" * 64,
            "implementation digest",
        ),
    )
    for target, field, value, expected_error in provenance_tamper_cases:
        report_payload = json.loads(snapshots[paths[0]])
        completion = json.loads(completion_snapshot)
        if target == "report":
            report_payload[field] = value
            paths[0].write_text(
                reskin_policy_comparison._canonical_json(report_payload) + "\n",
                encoding="utf-8",
            )
            completion["artifact_files"][paths[0].name] = (
                reskin_policy_comparison.artifact_file_metadata(paths[0])
            )
        else:
            completion[field] = value
        completion_path.write_text(
            reskin_policy_comparison._canonical_json(completion) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=expected_error):
            validate_f3_reskin_policy_comparison(tmp_path)
        paths[0].write_bytes(snapshots[paths[0]])
        completion_path.write_bytes(completion_snapshot)

    legacy_completion = json.loads(completion_snapshot)
    legacy_completion["completion_schema_version"] = 3
    completion_path.write_text(
        reskin_policy_comparison._canonical_json(legacy_completion) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported.*completion schema"):
        validate_f3_reskin_policy_comparison(tmp_path)
    completion_path.write_bytes(completion_snapshot)

    legacy_report = json.loads(snapshots[paths[0]])
    legacy_report["schema_version"] = 3
    _write_rehashed_comparison_report(
        legacy_report,
        paths,
        completion_path,
        json.loads(completion_snapshot),
    )
    with pytest.raises(ValueError, match="report schema mismatch"):
        validate_f3_reskin_policy_comparison(tmp_path)
    for path, content in snapshots.items():
        path.write_bytes(content)
    completion_path.write_bytes(completion_snapshot)

    paths[4].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        compare_reskin_policies_from_bundle(tmp_path, resume=True)
