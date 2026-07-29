import csv
import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

import pyosv.evaluation.dense_reskin_quality as dense_reskin_quality
import pyosv.evaluation.f3d_mode_comparison.result as f3_result_module
from pyosv.cells import FaultCell
from pyosv.evaluation.dense_reskin_quality import (
    build_dense_reskin_promotion_gate,
    controlled_dense_reskin_cases,
    write_dense_reskin_evidence,
)
from pyosv.evaluation.f3d_mode_comparison.reskin_policy_comparison import (
    compare_reskin_policies_from_bundle,
    compare_reskin_policies_from_parent,
    write_f3_reskin_policy_comparison,
)
from pyosv.evaluation.synthetic_quality import SyntheticSkinningConfig
from pyosv.skin import FaultSkin
from pyosv.synthetic_metrics import skin_link_topology_metrics


def test_controlled_dense_reskin_gate_applies_fixed_contract_to_every_case() -> None:
    gate = build_dense_reskin_promotion_gate()
    case_ids = [case.case_id for case in controlled_dense_reskin_cases()]

    assert gate["schema_version"] == 1
    assert gate["passed"] is False
    assert gate["reasons"] == ["aggregate:buffered_recall_regressed"]
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
    assert boundary["generation"]["reskin_rejected_out_of_bounds_count"] > 0
    assert boundary["out_of_volume_cell_count"] == 0
    assert boundary["clamped_artifact_count"] == 0
    for case_id in ("parallel_surfaces", "corner_touch_orientation_boundary"):
        assert by_id[case_id]["surface_count"] == 2
        assert by_id[case_id]["skinning_invocation_count"] == 1
        assert by_id[case_id]["candidate"]["truth_fault_id_bridge_count"] == 0


def test_link_topology_helper_reports_violations_without_mutation() -> None:
    left = FaultCell(1, 1, 1, 1, 0, 90)
    right = FaultCell(2, 1, 1, 1, 0, 90)
    object.__setattr__(left, "cr", right)
    skins = (FaultSkin.from_cells((left,)), FaultSkin.from_cells((right,)))

    metrics = skin_link_topology_metrics(skins)

    assert metrics["reciprocal_link_violation_count"] == 1
    assert metrics["cross_skin_link_count"] == 1
    assert left.cr is right
    assert right.cl is None


def test_dense_gate_writers_keep_nulls_explicit(tmp_path) -> None:
    gate = build_dense_reskin_promotion_gate()

    json_path, csv_path, markdown_path = write_dense_reskin_evidence(gate, tmp_path)

    assert json.loads(json_path.read_text())["passed"] is False
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    assert len(rows) == 30
    low_baseline = next(
        row for row in rows if row["case_id"] == "low_support_gap" and row["policy"] == "baseline"
    )
    assert low_baseline["generated_cell_fraction"] == "0.0"
    assert low_baseline["support_min"] == ""
    assert "Gate reasons" in markdown_path.read_text()


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


def test_dense_gate_requires_boundary_rejection_evidence(monkeypatch) -> None:
    evaluate = dense_reskin_quality.evaluate_dense_reskin_case

    def evaluate_without_boundary_rejection(case):
        result = evaluate(case)
        if case.case_id == "volume_boundary_surface":
            result["candidate"]["generation"]["reskin_generated_cell_count"] = 0
            result["candidate"]["generation"]["reskin_rejected_out_of_bounds_count"] = 0
        return result

    monkeypatch.setattr(
        dense_reskin_quality,
        "evaluate_dense_reskin_case",
        evaluate_without_boundary_rejection,
    )

    gate = build_dense_reskin_promotion_gate()

    assert "volume_boundary_surface:dense_generation_not_exercised" in gate["reasons"]
    assert "volume_boundary_surface:out_of_bounds_rejection_not_exercised" in gate["reasons"]


def test_f3_skin_only_pair_shares_parent_and_branches_policy() -> None:
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

    report = compare_reskin_policies_from_parent(
        fv=fv,
        fvt=fv.copy(),
        vp=vp,
        vt=vt,
        skinning_config=config,
    )

    assert report["upstream_parent_fingerprint_identical"] is True
    assert report["contrast"]["generated_cell_count_delta"] == 1
    assert set(report["policies"]) == {"existing_cells_v1", "reference_dense_v1"}
    assert (
        report["policies"]["reference_dense_v1"]["parent_ridge_surface"]["overlap"][
            "buffered_recall"
        ]
        == 1.0
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
    voting_fingerprint = "a" * 64
    thinning_fingerprint = "b" * 64
    voting = tmp_path / "stages" / "voting" / voting_fingerprint
    thinning = tmp_path / "stages" / "thinning" / thinning_fingerprint
    voting.mkdir(parents=True)
    thinning.mkdir(parents=True)
    for path, values in (
        (voting / "fv.dat", fv),
        (voting / "vp.dat", vp),
        (voting / "vt.dat", vt),
        (thinning / "fvt.dat", fv),
    ):
        np.asarray(values, dtype=">f4").tofile(path)
    report = json.dumps({"shape": list(shape)})
    (voting / "report.json").write_text(report, encoding="utf-8")
    (thinning / "report.json").write_text(report, encoding="utf-8")
    cell = SimpleNamespace(
        label="Q-QUAL",
        backend="quality",
        workflow="quality",
        skinning_enabled=True,
        resolved_config={"skinning": asdict(config)},
        stages=SimpleNamespace(
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
        "voting": voting_fingerprint,
        "thinning": thinning_fingerprint,
    }
    assert paths[0].parent == tmp_path / "reskin_policy_comparison"
