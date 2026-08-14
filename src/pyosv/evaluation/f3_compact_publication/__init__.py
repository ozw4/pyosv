"""F3-only compact publication contracts."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AMPLITUDE_DTYPE": ("config", "AMPLITUDE_DTYPE"),
    "AMPLITUDE_FILENAME": ("config", "AMPLITUDE_FILENAME"),
    "AMPLITUDE_PERCENTILE": ("config", "AMPLITUDE_PERCENTILE"),
    "AMPLITUDE_ROLE": ("config", "AMPLITUDE_ROLE"),
    "AMPLITUDE_ALPHA_MAX": ("config", "AMPLITUDE_ALPHA_MAX"),
    "ATTRIBUTE_COLORMAP": ("config", "ATTRIBUTE_COLORMAP"),
    "DIFFERENCE_COLORMAP": ("config", "DIFFERENCE_COLORMAP"),
    "DIFFERENCE_PERCENTILE": ("config", "DIFFERENCE_PERCENTILE"),
    "DISPLAY_CELL": ("config", "DISPLAY_CELL"),
    "EXPERIMENT_SCHEMA": ("config", "EXPERIMENT_SCHEMA"),
    "FIGURE_DATA_HEADER": ("config", "FIGURE_DATA_HEADER"),
    "PUBLIC_REFERENCE_LABEL": ("config", "PUBLIC_REFERENCE_LABEL"),
    "SLICE_AXIS": ("config", "SLICE_AXIS"),
    "SLICE_POLICY": ("config", "SLICE_POLICY"),
    "STAGE_ORDER": ("config", "STAGE_ORDER"),
    "SUMMARY_FILENAME": ("config", "SUMMARY_FILENAME"),
    "SUMMARY_HEADER": ("config", "SUMMARY_HEADER"),
    "AmplitudeIdentity": ("models", "AmplitudeIdentity"),
    "CompactSourceContext": ("models", "CompactSourceContext"),
    "RidgeStageThresholds": ("models", "RidgeStageThresholds"),
    "SelectedSlice": ("models", "SelectedSlice"),
    "SourceRidgeThresholdContract": ("models", "SourceRidgeThresholdContract"),
    "StageSource": ("models", "StageSource"),
    "load_compact_source": ("source", "load_compact_source"),
    "build_experiment": ("summary", "build_experiment"),
    "build_summary_rows": ("summary", "build_summary_rows"),
    "experiment_json_bytes": ("summary", "experiment_json_bytes"),
    "summary_csv_bytes": ("summary", "summary_csv_bytes"),
    "generate_figures": ("figures", "generate_figures"),
    "F3_COMPACT_PUBLICATION_MANIFEST_SCHEMA": (
        "manifest",
        "F3_COMPACT_PUBLICATION_MANIFEST_SCHEMA",
    ),
    "PUBLICATION_MANIFEST_FILENAME": ("manifest", "PUBLICATION_MANIFEST_FILENAME"),
    "build_manifest": ("manifest", "build_manifest"),
    "compute_publication_id": ("manifest", "compute_publication_id"),
    "validate_manifest": ("manifest", "validate_manifest"),
    "validate_publication_directory": ("manifest", "validate_publication_directory"),
    "write_manifest": ("manifest", "write_manifest"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
