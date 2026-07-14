from __future__ import annotations

import json
from pathlib import Path


EVIDENCE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "f3d_scanner_thinning_policy"
    / "quality_reference_like_normal_v1_evidence.json"
)


def test_failed_f3_policy_evidence_preserves_gate_and_review_status() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert evidence["evidence_type"] == "f3_scanner_thinning_policy_external_smoke"
    assert evidence["validation_role"] == "truthless_external_smoke"
    assert evidence["status"] == {
        "reference_like_backend_49_cubed_synthetic_gate": "passed",
        "f3_shared_scan_policy_validation": "failed_64x3",
        "large_crop": "not_run_prerequisite_failed",
        "manual_review": "pending_human_review",
        "quality_workflow_default_promotion": "blocked_unchanged",
    }

    run = evidence["run_64x3"]
    assert run["exit_code"] == 2
    assert run["crop_shape"] == [64, 64, 64]
    assert len(run["selected_centers"]) == 3
    assert len(run["effective_crop_bounds"]) == 3
    assert run["scanner_execution_count"] == 3

    validation = evidence["policy_validation"]
    assert validation["passed"] is False
    assert len(validation["checks"]) == 8
    failed_checks = [name for name, check in validation["checks"].items() if not check["passed"]]
    assert failed_checks == ["public_fvt_sparse_distance_p95"]
    sparse_check = validation["checks"]["public_fvt_sparse_distance_p95"]
    assert (
        sparse_check["maximum_crop_delta_samples"] > sparse_check["maximum_allowed_delta_samples"]
    )

    assert evidence["large_crop"]["status"] == "not_run_prerequisite_failed"
    assert evidence["manual_review"]["status"] == "pending_human_review"
    assert evidence["manual_review"]["decision"] is None
    assert evidence["machine_preliminary_visual_screen"]["status"].endswith("not_a_manual_review")

    for artifact in evidence["artifacts"].values():
        assert artifact["committed"] is False
        assert len(artifact["sha256"]) == 64
        int(artifact["sha256"], 16)
