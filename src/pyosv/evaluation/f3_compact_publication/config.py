"""Fixed source contracts for the F3 compact publication."""

from __future__ import annotations

from typing import Final

DISPLAY_CELL: Final = "Q-QUAL"
STAGE_ORDER: Final = ("ft", "fv", "fvt")
SLICE_AXIS: Final = "i2"
SLICE_POLICY: Final = "public_fvt_positive_p99_peak"
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
AMPLITUDE_ALPHA_MAX: Final = 0.75
ATTRIBUTE_COLORMAP: Final = "magma"
DIFFERENCE_COLORMAP: Final = "coolwarm"
DIFFERENCE_PERCENTILE: Final = 99.0
FIGURE_DATA_HEADER: Final = (
    "figure_id",
    "stage",
    "axis",
    "slice_index",
    "slice_selection_policy",
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
    "alpha_max",
    "colormap",
    "difference_limit",
)


__all__ = [
    "AMPLITUDE_DTYPE",
    "AMPLITUDE_FILENAME",
    "AMPLITUDE_PERCENTILE",
    "AMPLITUDE_ROLE",
    "AMPLITUDE_ALPHA_MAX",
    "ATTRIBUTE_COLORMAP",
    "DIFFERENCE_COLORMAP",
    "DIFFERENCE_PERCENTILE",
    "DISPLAY_CELL",
    "EXPERIMENT_SCHEMA",
    "FIGURE_DATA_HEADER",
    "PUBLIC_REFERENCE_LABEL",
    "SLICE_AXIS",
    "SLICE_POLICY",
    "STAGE_ORDER",
    "SUMMARY_FILENAME",
    "SUMMARY_HEADER",
]
