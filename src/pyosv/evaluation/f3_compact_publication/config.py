"""Fixed source contracts for the F3 compact publication."""

from __future__ import annotations

from typing import Final

DISPLAY_CELL: Final = "Q-QUAL"
STAGE_ORDER: Final = ("ft", "fv", "fvt")
SECTION_SELECTION_POLICY: Final = "public_fvt_positive_p99_peak_per_equal_bin"
SECTIONS_PER_AXIS: Final = 5
SECTION_GROUPS: Final = (
    ("time_slices", "i1"),
    ("inline_sections", "i3"),
)
AMPLITUDE_ROLE: Final = "seismic_amplitude"
AMPLITUDE_FILENAME: Final = "xs.dat"
AMPLITUDE_DTYPE: Final = ">f4"
SUMMARY_FILENAME: Final = "f3_q_qual_vs_public_ref_summary.csv"
SUMMARY_HEADER: Final = (
    "stage",
    "public_reference_file",
    "q_qual_stage_fingerprint",
    "normalized_correlation",
    "mean_absolute_difference",
    "nonzero_fraction_ratio",
    "buffered_f1",
    "candidate_to_reference_p95_voxel",
    "reference_to_candidate_p95_voxel",
)
EXPERIMENT_SCHEMA: Final = "pyosv.f3_compact_publication_experiment.v1"
PUBLIC_REFERENCE_LABEL: Final = "PUBLIC-REF"
AMPLITUDE_PERCENTILE: Final = 99.0
ATTRIBUTE_COLORMAP: Final = "inferno"
ATTRIBUTE_ALPHA_MIN: Final = 0.12
ATTRIBUTE_ALPHA_MAX: Final = 0.85
ATTRIBUTE_ALPHA_GAMMA: Final = 2.0
DIFFERENCE_COLORMAP: Final = "coolwarm"
DIFFERENCE_PERCENTILE: Final = 99.0
FIGURE_DATA_HEADER: Final = (
    "figure_id",
    "stage",
    "section_group",
    "axis",
    "bin_index",
    "section_index",
    "selection_policy",
    "ridge_count_score",
    "panel_label",
    "source_label",
    "source_file",
    "source_sha256",
    "source_stage_fingerprint",
    "selection_threshold",
    "amplitude_file",
    "amplitude_sha256",
    "amplitude_limit",
    "overlay_vmin",
    "overlay_vmax",
    "alpha_min",
    "alpha_max",
    "alpha_gamma",
    "colormap",
    "difference_limit",
)


__all__ = [
    "AMPLITUDE_DTYPE",
    "AMPLITUDE_FILENAME",
    "AMPLITUDE_PERCENTILE",
    "AMPLITUDE_ROLE",
    "ATTRIBUTE_ALPHA_GAMMA",
    "ATTRIBUTE_ALPHA_MAX",
    "ATTRIBUTE_ALPHA_MIN",
    "ATTRIBUTE_COLORMAP",
    "DIFFERENCE_COLORMAP",
    "DIFFERENCE_PERCENTILE",
    "DISPLAY_CELL",
    "EXPERIMENT_SCHEMA",
    "FIGURE_DATA_HEADER",
    "PUBLIC_REFERENCE_LABEL",
    "SECTION_GROUPS",
    "SECTION_SELECTION_POLICY",
    "SECTIONS_PER_AXIS",
    "STAGE_ORDER",
    "SUMMARY_FILENAME",
    "SUMMARY_HEADER",
]
