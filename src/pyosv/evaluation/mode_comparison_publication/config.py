"""Fixed contracts for the derived mode-comparison publication bundle."""

from __future__ import annotations

from typing import Final

CANONICAL_CELL_ORDER: Final = ("RL-REF", "RL-QUAL", "Q-REF", "Q-QUAL")
SYNTHETIC_SCANNER_CELL_ORDER: Final = ("RL-SCAN", "Q-SCAN")
CANONICAL_STAGE_ORDER: Final = ("ft", "fv", "fvt")
SYNTHETIC_STAGE_ORDER: Final = ("scanner_raw", "fvt", "skin")

PUBLICATION_METRICS_HEADER: Final = (
    "dataset",
    "evaluation_semantics",
    "case_or_region",
    "trial_id",
    "seed",
    "cell_label",
    "scanner_backend",
    "workflow_mode",
    "stage",
    "selection",
    "metric",
    "value",
    "unit",
    "direction",
    "source_artifact",
)

PUBLICATION_CONTRASTS_HEADER: Final = (
    "dataset",
    "evaluation_semantics",
    "case_or_region",
    "trial_id",
    "seed",
    "contrast_name",
    "stage",
    "selection",
    "metric",
    "raw_value",
    "improvement_value",
    "unit",
    "direction",
    "component_cells",
    "source_artifact",
)

PUBLICATION_SUMMARY_HEADER: Final = (
    "dataset",
    "evaluation_semantics",
    "case_or_region",
    "stage",
    "selection",
    "metric",
    "cell_label",
    "n",
    "mean",
    "median",
    "minimum",
    "maximum",
    "q25",
    "q75",
    "unit",
    "direction",
)

F3_REGIONAL_SUMMARY_HEADER: Final = (
    "dataset",
    "evaluation_semantics",
    "case_or_region",
    "stage",
    "cell_label",
    "scanner_backend",
    "workflow_mode",
    "region",
    "metric",
    "display_label",
    "value",
    "unit",
    "source_artifact",
)

F3_ORIENTATION_SUMMARY_HEADER: Final = (
    "dataset",
    "evaluation_semantics",
    "case_or_region",
    "stage",
    "left_cell",
    "right_cell",
    "support_contract",
    "support_count",
    "metric",
    "display_label",
    "value",
    "unit",
    "source_artifact",
)

RUNTIME_SUMMARY_HEADER: Final = (
    "dataset",
    "evaluation_semantics",
    "case_or_region",
    "trial_id",
    "seed",
    "stage",
    "fingerprint",
    "scanner_backend",
    "call_count",
    "cell_label",
    "cell_consumers",
    "state",
    "elapsed_seconds",
    "elapsed_semantics",
    "shared_stage",
    "attribution",
    "source_artifact",
)

FIGURE_DATA_HEADER: Final = (
    "figure_id",
    "dataset",
    "evaluation_semantics",
    "source_metric",
    "source_stage",
    "case_or_region",
    "trial_id",
    "seed",
    "cell_label",
    "panel_label",
    "metric",
    "value",
    "raw_improvement",
    "normalized_value",
    "unit",
    "direction",
    "axis",
    "slice_index",
    "slice_selection_policy",
    "slice_score",
    "selection_threshold",
    "candidate_selection_threshold",
    "vmin",
    "vmax",
    "scale_policy",
    "colormap",
    "difference_limit",
    "difference_vmin",
    "difference_vmax",
)

SYNTHETIC_SKIN_FIGURE_OMISSION_REASON: Final = "source synthetic skinning is disabled"

FIGURE_SELECTION_POLICY: Final = {
    "slice_policies": (
        "center",
        "public_reference_peak",
        "end_to_end_difference_peak",
    ),
    "center_definition": "i3=n3//2; i2=n2//2; i1=n1//2",
    "public_reference_peak_definition": (
        "maximum positive-p99 public-reference ridge count in the slice; ties use the "
        "smallest index"
    ),
    "end_to_end_difference_peak_definition": (
        "maximum slice sum of abs(Q-QUAL - RL-REF) for fvt; ties use the smallest index"
    ),
    "ridge_selection": "positive-only; source F3 evidence percentile and radius are reused",
    "shared_scale": "validated full-volume min/max across all displayed normal panels",
    "difference_scale": "symmetric around zero using the displayed signed-difference samples",
}

ROOT_TABLE_FILES: Final = (
    "publication_metrics.csv",
    "publication_contrasts.csv",
    "publication_summary.csv",
    "f3_regional_summary.csv",
    "f3_orientation_summary.csv",
    "runtime_summary.csv",
)

SYNTHETIC_SEMANTICS: Final = "synthetic_truth"
F3_SEMANTICS: Final = "f3_public_reference_agreement"

CONTRAST_NAMES: Final = (
    "scanner_effect_ref",
    "scanner_effect_qual",
    "workflow_effect_rl",
    "workflow_effect_q",
    "scanner_main_effect",
    "workflow_main_effect",
    "scanner_workflow_interaction",
    "end_to_end_delta",
)

F3_PUBLIC_REFERENCE_LABEL: Final = "PUBLIC-REF"
