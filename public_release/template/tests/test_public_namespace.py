from __future__ import annotations


def test_synthetic_quality_public_facade_exposes_no_internal_evaluation_api() -> None:
    import pyosv.evaluation.synthetic_quality as synthetic_quality

    assert synthetic_quality.__all__ == [
        "ResolvedWorkflowSettings",
        "SyntheticSkinningConfig",
        "SyntheticVotingConfig",
        "quality_metrics",
        "resolve_workflow_settings",
    ]

    for name in (
        "build_report",
        "run_case",
        "build_dense_reskin_promotion_gate",
        "controlled_dense_reskin_cases",
        "write_dense_reskin_evidence",
        "write_dense_reskin_figures",
    ):
        assert not hasattr(synthetic_quality, name)
