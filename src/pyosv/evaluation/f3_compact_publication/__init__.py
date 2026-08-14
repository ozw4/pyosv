"""F3-only compact publication source contracts."""

from .config import (
    AMPLITUDE_DTYPE,
    AMPLITUDE_FILENAME,
    AMPLITUDE_PERCENTILE,
    AMPLITUDE_ROLE,
    AMPLITUDE_ALPHA_MAX,
    ATTRIBUTE_COLORMAP,
    DIFFERENCE_COLORMAP,
    DIFFERENCE_PERCENTILE,
    DISPLAY_CELL,
    EXPERIMENT_SCHEMA,
    FIGURE_DATA_HEADER,
    PUBLIC_REFERENCE_LABEL,
    SLICE_AXIS,
    SLICE_POLICY,
    STAGE_ORDER,
    SUMMARY_FILENAME,
    SUMMARY_HEADER,
)
from .figures import generate_figures
from .models import (
    AmplitudeIdentity,
    CompactSourceContext,
    RidgeStageThresholds,
    SelectedSlice,
    SourceRidgeThresholdContract,
    StageSource,
)
from .source import load_compact_source
from .summary import (
    build_experiment,
    build_summary_rows,
    experiment_json_bytes,
    summary_csv_bytes,
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
    "AmplitudeIdentity",
    "CompactSourceContext",
    "RidgeStageThresholds",
    "SelectedSlice",
    "SourceRidgeThresholdContract",
    "StageSource",
    "build_experiment",
    "build_summary_rows",
    "experiment_json_bytes",
    "generate_figures",
    "load_compact_source",
    "summary_csv_bytes",
]
