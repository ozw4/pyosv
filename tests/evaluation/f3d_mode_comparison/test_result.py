from dataclasses import fields

from pyosv.evaluation.f3d_mode_comparison import (
    F3_REPORT_FILES,
    F3_RESULT_INTERPRETATION,
    F3ModeComparisonResult,
)


def test_result_model_is_scalar_reference_only_and_report_set_is_fixed() -> None:
    assert tuple(field.name for field in fields(F3ModeComparisonResult)) == (
        "run_fingerprint",
        "dataset_id",
        "volume_shape",
        "storage_dtype",
        "cells",
        "metric_rows",
        "metric_evidence",
        "contrast_rows",
        "voxelwise_contrasts",
        "regional_rows",
        "orientation_rows",
        "runtime_rows",
        "rss_snapshots",
        "storage_rows",
        "resource_interpretation",
        "interpretation",
    )
    assert F3_REPORT_FILES == (
        "cells.json",
        "metrics_long.csv",
        "metric_evidence.json",
        "contrasts.csv",
        "voxel_contrast_summaries.csv",
        "regional_metrics.csv",
        "orientation_diagnostics.csv",
        "runtime.csv",
        "resources.json",
    )
    assert "geological_truth" in F3_RESULT_INTERPRETATION
