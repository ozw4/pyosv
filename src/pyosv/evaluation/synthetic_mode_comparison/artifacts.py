"""Atomic artifact bundles for completed synthetic mode comparisons."""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields, replace
from math import isclose
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import scipy

import pyosv
from pyosv.synthetic3d import SyntheticScannerInputConfig

from ..synthetic_quality import (
    ResolvedWorkflowSettings,
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from ..synthetic_quality.variants import effective_skinning_config, get_variant_spec
from ..synthetic_quality.stage_cache import SCALAR_EVIDENCE_CONTRACT_VERSION
from .builder import build_mode_comparison_plan
from .config import SyntheticModeComparisonConfig
from .contrasts import AggregateRow, ContrastRow
from .experiment import RuntimeRow, SyntheticModeComparisonResult
from .metrics import (
    METRIC_SCHEMA_VERSION,
    MetricRow,
    scanner_metric_definitions,
    validate_metric_value,
)
from .models import (
    ORACLE_WORKFLOW_ISOLATION_SCOPE,
    SCANNER_ONLY_SCOPE,
    ModeCellSpec,
    SyntheticModeComparisonPlan,
)
from .scalar_algebra import (
    derived_scalars_close,
    validate_downstream_quality_scalar_algebra,
    validate_quality_scalar_algebra,
    validate_scanner_quality_scalar_algebra,
    validate_selection_cardinality,
    validate_skin_report_topology_algebra,
    validate_skin_topology_algebra,
    volume_voxel_count,
)
from .validation import validate_mode_comparison_result

ARTIFACT_SCHEMA_VERSION = 3
RUNTIME_CONTRACT_VERSION = 3
COMPLETION_SCHEMA_VERSION = 1
METRIC_REGISTRY_ID = "pyosv.synthetic_mode_comparison.metrics"
METRIC_REGISTRY_DEFINITION_VERSION = 1
CONTRAST_DEFINITION_ID = "pyosv.synthetic_mode_comparison.contrasts"
CONTRAST_FORMULA_VERSION = 1

MANIFEST_FILE = "manifest.json"
CELL_REPORTS_FILE = "cell_reports.json"
METRICS_FILE = "metrics_long.csv"
METRIC_AGGREGATES_FILE = "metric_aggregates.csv"
CONTRASTS_FILE = "contrasts.csv"
CONTRAST_AGGREGATES_FILE = "contrast_aggregates.csv"
RUNTIME_FILE = "runtime.csv"
COMPLETION_FILE = "completion.json"

HASHED_BUNDLE_FILES = (
    MANIFEST_FILE,
    CELL_REPORTS_FILE,
    METRICS_FILE,
    METRIC_AGGREGATES_FILE,
    CONTRASTS_FILE,
    CONTRAST_AGGREGATES_FILE,
    RUNTIME_FILE,
)
REQUIRED_BUNDLE_FILES = (*HASHED_BUNDLE_FILES, COMPLETION_FILE)

_CSV_MODELS = {
    METRICS_FILE: MetricRow,
    METRIC_AGGREGATES_FILE: AggregateRow,
    CONTRASTS_FILE: ContrastRow,
    CONTRAST_AGGREGATES_FILE: AggregateRow,
    RUNTIME_FILE: RuntimeRow,
}
_CSV_INTEGER_FIELDS = {
    METRICS_FILE: {"schema_version", "seed", "scanner_refinement_factor"},
    METRIC_AGGREGATES_FILE: {"n"},
    CONTRASTS_FILE: {"seed"},
    CONTRAST_AGGREGATES_FILE: {"n"},
    RUNTIME_FILE: {"seed", "call_count"},
}
_CSV_FLOAT_FIELDS = {
    METRICS_FILE: {"value"},
    METRIC_AGGREGATES_FILE: {"mean", "median", "std", "min", "max", "q25", "q75"},
    CONTRASTS_FILE: {"raw_value", "improvement_value"},
    CONTRAST_AGGREGATES_FILE: {
        "mean",
        "median",
        "std",
        "min",
        "max",
        "q25",
        "q75",
    },
    RUNTIME_FILE: {"elapsed_seconds"},
}
_CSV_NULLABLE_NUMERIC_FIELDS = {
    METRICS_FILE: {"seed", "scanner_refinement_factor"},
    METRIC_AGGREGATES_FILE: set(),
    CONTRASTS_FILE: {"seed", "improvement_value"},
    CONTRAST_AGGREGATES_FILE: set(),
    RUNTIME_FILE: {"seed"},
}
_CSV_BOOLEAN_FIELDS = {
    METRICS_FILE: {"contrast_eligible"},
    METRIC_AGGREGATES_FILE: set(),
    CONTRASTS_FILE: set(),
    CONTRAST_AGGREGATES_FILE: set(),
    RUNTIME_FILE: {"shared_stage"},
}
_CSV_NULLABLE_STRING_FIELDS = {
    METRICS_FILE: {
        "scanner_backend",
        "scanner_thin_mode",
        "workflow_mode",
        "voter_thin_mode",
        "skinner_method",
    },
    METRIC_AGGREGATES_FILE: {"cell_label", "contrast_name"},
    CONTRASTS_FILE: set(),
    CONTRAST_AGGREGATES_FILE: {"cell_label", "contrast_name"},
    RUNTIME_FILE: {"case_id", "trial_id", "cell_label", "scanner_backend"},
}
_CSV_TUPLE_FIELDS = {CONTRASTS_FILE: {"component_cells"}}

_MANIFEST_FIELDS = {
    "artifact_schema_version",
    "scalar_evidence_contract_version",
    "runtime_contract_version",
    "metric_schema_version",
    "canonical_cells",
    "input_config",
    "resolved_plan",
    "case_order",
    "cases",
    "trials",
    "shape",
    "variant",
    "oracle_workflow_isolation",
    "metric_registry",
    "contrast_definition",
    "cache_stats",
    "software_versions",
    "software_version_status",
    "source_provenance",
}
_CACHE_COUNTERS = (
    "seed_hits",
    "seed_misses",
    "voting_hits",
    "voting_misses",
    "thinning_hits",
    "thinning_misses",
    "primary_skinning_hits",
    "primary_skinning_misses",
)
_SCANNER_REPORT_FIELDS = {"config", "input", "ft", "fet", "pt", "fpt", "tt", "ftt"}
_SCANNER_CONFIG_REPORT_FIELDS = {
    "backend",
    "phi_min",
    "phi_max",
    "theta_min",
    "theta_max",
    "sigma1",
    "sigma2",
    "refinement_factor",
    "scanner_thin_mode",
    "remove_edge_effects",
    "input",
}
_SCANNER_INPUT_CONFIG_REPORT_FIELDS = {
    "background",
    "fault_contrast",
    "noise_sigma",
    "seed",
    "clip_min",
    "clip_max",
}
_ARRAY_SUMMARY_FIELDS = {
    "shape",
    "finite_count",
    "finite_fraction",
    "min",
    "max",
    "mean",
    "nonzero_fraction",
}
_SCANNER_QUALITY_REPORT_FIELDS = {
    "ft_top_truth_count",
    "orientation_error",
    "input_association",
}
_DOWNSTREAM_REPORT_FIELDS = {"config", "skinning", "pyosv", "quality"}
_END_TO_END_REPORT_FIELDS = {
    *_DOWNSTREAM_REPORT_FIELDS,
    "scanner",
    "scanner_quality",
    "scanner_metric_evidence",
    "active_pipeline",
    "pipelines",
}
_BUFFERED_OVERLAP_REPORT_SCHEMA = {
    **dict.fromkeys(
        (
            "candidate_count",
            "truth_count",
            "intersection_count",
            "union_count",
            "candidate_in_truth_buffer_count",
            "truth_in_candidate_buffer_count",
        ),
        "nonnegative_integer",
    ),
    **dict.fromkeys(
        (
            "precision",
            "recall",
            "f1",
            "jaccard",
            "buffered_precision",
            "buffered_recall",
            "buffered_f1",
        ),
        "closed_unit_interval_number",
    ),
    "radius": "nonnegative_number",
}
_SURFACE_DISTANCE_REPORT_SCHEMA = {
    **dict.fromkeys(("candidate_count", "truth_count"), "nonnegative_integer"),
    **dict.fromkeys(
        (
            "candidate_to_truth_mean",
            "candidate_to_truth_median",
            "candidate_to_truth_p90",
            "candidate_to_truth_p95",
            "truth_to_candidate_mean",
            "truth_to_candidate_median",
            "truth_to_candidate_p90",
            "truth_to_candidate_p95",
            "symmetric_chamfer_mean",
            "hausdorff_p95",
        ),
        "nonnegative_number",
    ),
}
_ORIENTATION_ERROR_REPORT_SCHEMA = {
    "count": "nonnegative_integer",
    **dict.fromkeys(
        (
            "strike_mean",
            "strike_median",
            "strike_p90",
            "strike_p95",
            "dip_mean",
            "dip_median",
            "dip_p90",
            "dip_p95",
        ),
        "nonnegative_number",
    ),
}
_INPUT_ASSOCIATION_REPORT_SCHEMA = dict.fromkeys(
    ("truth_surface_mean", "far_from_truth_mean", "contrast"), "number"
)
_SKINNING_CONFIG_REPORT_SCHEMA = {
    **dict.fromkeys(
        (
            "enabled",
            "adaptive_min_likelihood",
            "reskin",
            "boundary_skinner_fallback",
        ),
        "boolean",
    ),
    **dict.fromkeys(
        (
            "method",
            "growth_source",
            "seed_planarity_source",
            "boundary_skinner_fallback_policy",
        ),
        "string",
    ),
    "min_likelihood": "optional_nonnegative_number",
    "seed_min_ep": "optional_closed_unit_interval_number",
    **dict.fromkeys(
        ("min_skin_size", "rv", "rw", "accepted_occupancy_radius"),
        "optional_nonnegative_integer",
    ),
    **dict.fromkeys(
        (
            "d",
            "ru",
            "max_steps",
            "effective_accepted_occupancy_radius",
            "small_skin_size",
        ),
        "nonnegative_integer",
    ),
    **dict.fromkeys(("du", "max_delta_strike"), "nonnegative_number"),
}
_PRIMARY_SKINNER_DIAGNOSTIC_FIELDS = {
    "seed_candidate_count_before_spacing": "nonnegative_integer",
    "seed_count_after_spacing": "nonnegative_integer",
    "seed_count_rejected_by_occupied": "nonnegative_integer",
    "grow_attempt_count": "nonnegative_integer",
    "grown_skin_count_before_min_size": "nonnegative_integer",
    "discarded_empty_skin_count": "nonnegative_integer",
    "discarded_small_skin_count": "nonnegative_integer",
    "accepted_skin_count": "nonnegative_integer",
    "accepted_cell_count": "nonnegative_integer",
    "accepted_occupancy_radius": "nonnegative_integer",
    "seed_min_ep": "closed_unit_interval_number",
    "seed_threshold": "nonnegative_number",
    "grow_threshold": "nonnegative_number",
}
_SKINNING_DIAGNOSTIC_REPORT_SCHEMA = {
    **dict.fromkeys(
        (
            "skin_primary_count",
            "skin_primary_cell_count",
            "skin_primary_unique_cell_count",
            "skin_primary_largest_size",
            "skin_primary_small_count",
            "fallback_skin_count",
            "fallback_cell_count",
            "fallback_primary_skin_count",
            "fallback_primary_cell_count",
            "fallback_candidate_count",
            "skin_fallback_raw_component_cell_count",
            "skin_fallback_pruned_component_cell_count",
            "skin_fallback_largest_component_size_before_pruning",
            "skin_fallback_largest_component_size_after_pruning",
            "skin_fallback_pruning_removed_cell_count",
            "skin_fallback_component_count",
            "skin_fallback_candidate_cell_count",
            "skin_fallback_largest_component_size",
            "skin_fallback_top3_component_cell_count",
            "skin_fallback_small_component_count",
            "skin_fallback_accepted_component_count",
            "skin_fallback_discarded_component_count",
            "skin_fallback_accepted_component_cell_count",
            "skin_fallback_filter_min_component_size",
            "skin_fallback_filter_max_components",
        ),
        "nonnegative_integer",
    ),
    **dict.fromkeys(
        (
            "skin_primary_largest_fraction",
            "skin_primary_small_cell_fraction",
            "skin_primary_edge_shell_fraction",
            "skin_fvt_positive_edge_shell_fraction",
            "skin_fallback_pruned_fraction",
            "skin_fallback_largest_component_fraction",
            "skin_fallback_top3_component_fraction",
            "skin_fallback_filter_min_component_fraction_of_largest",
        ),
        "closed_unit_interval_number",
    ),
    **dict.fromkeys(
        (
            "skin_primary_cell_coverage_of_fvt_positive",
            "skin_primary_largest_coverage_of_fvt_positive",
            "fallback_coverage_before",
            "fallback_coverage_after",
        ),
        "nonnegative_number",
    ),
    **dict.fromkeys(
        ("skin_scanner_target_positive_edge_shell_fraction",),
        "optional_closed_unit_interval_number",
    ),
    "skin_fvt_to_scanner_target_distance_p95": "optional_nonnegative_number",
    **dict.fromkeys(
        (
            "skin_primary_degraded_candidate",
            "fallback_enabled",
            "fallback_used",
            "fallback_triggered_by_degraded_primary",
            "fallback_replaced_primary",
            "skin_primary_boundary_degraded_candidate",
        ),
        "boolean",
    ),
    **dict.fromkeys(
        (
            "fallback_policy",
            "fallback_reason",
            "fallback_method",
            "fallback_input",
            "skin_fallback_pruning_method",
            "skin_fallback_skeletonization_axis_mode",
        ),
        "optional_string",
    ),
    "skin_fallback_component_policy": "string",
    **dict.fromkeys(
        (
            "skin_primary_degraded_reasons",
            "fallback_degraded_reasons",
            "skin_primary_boundary_degraded_reasons",
        ),
        "string_array",
    ),
}
_GUARDRAIL_REPORT_SCHEMA = {
    "enabled": "boolean",
    "passed": "boolean",
    "reasons": "string_array",
    "max_skin_count": "nonnegative_integer",
    "fallback_skin_count": "nonnegative_integer",
    **dict.fromkeys(
        (
            "coverage_of_fvt_positive",
            "min_coverage_of_fvt_positive",
            "max_coverage_of_fvt_positive",
        ),
        "nonnegative_number",
    ),
    **dict.fromkeys(
        (
            "small_skin_cell_fraction",
            "max_small_skin_cell_fraction",
            "largest_skin_fraction",
            "min_largest_skin_fraction",
            "pruned_fraction",
            "max_pruned_fraction",
        ),
        "closed_unit_interval_number",
    ),
}
_VOTING_REPORT_SCHEMA = {
    "surface_voting_boundary_policy": "string",
    "surface_support_min_fraction": "closed_unit_interval_number",
    "surface_support_exponent": "nonnegative_number",
}
_VOTING_DIAGNOSTIC_REPORT_SCHEMA = {
    "policy": "string",
    **dict.fromkeys(
        (
            "seed_count",
            "boundary_affected_seed_count",
            "voted_seed_count",
            "skipped_seed_count",
            "surface_projection_count",
            "selected_invalid_sample_count",
            "face_center_vote_count",
        ),
        "nonnegative_integer",
    ),
    **dict.fromkeys(
        ("support_fraction_min", "support_fraction_mean"),
        "closed_unit_interval_number",
    ),
}
_SKIN_TOPOLOGY_REPORT_SCHEMA = {
    **dict.fromkeys(
        (
            "skin_count",
            "cell_count",
            "unique_cell_count",
            "duplicate_cell_count",
            "largest_skin_size",
            "small_skin_size",
            "small_skin_count",
            "small_skin_cell_count",
        ),
        "nonnegative_integer",
    ),
    **dict.fromkeys(
        ("largest_skin_fraction", "small_skin_cell_fraction"),
        "closed_unit_interval_number",
    ),
}
_EDGE_FALSE_POSITIVE_REPORT_SCHEMA = {
    **dict.fromkeys(
        ("candidate_count", "edge_candidate_count", "edge_false_positive_count", "edge_margin"),
        "nonnegative_integer",
    ),
    **dict.fromkeys(
        (
            "edge_candidate_fraction",
            "edge_false_positive_fraction_of_candidates",
            "edge_false_positive_fraction_of_edge_candidates",
        ),
        "closed_unit_interval_number",
    ),
    "truth_buffer_radius": "nonnegative_number",
}
_COMPONENT_TOPOLOGY_SUMMARY_SCHEMA = {
    **dict.fromkeys(
        (
            "truth_component_count",
            "covered_truth_component_count",
            "uncovered_truth_component_count",
            "skin_count",
            "skin_with_truth_count",
            "skin_without_truth_count",
            "over_merge_skin_count",
            "over_split_truth_component_count",
            "max_truth_components_per_skin",
            "max_skins_per_truth_component",
        ),
        "nonnegative_integer",
    ),
    **dict.fromkeys(
        (
            "mean_skin_purity",
            "min_skin_purity",
            "mean_truth_component_recall",
            "min_truth_component_recall",
        ),
        "closed_unit_interval_number",
    ),
}
_TRUTH_COMPONENT_REPORT_SCHEMA = {
    **dict.fromkeys(
        (
            "truth_id",
            "truth_cell_count",
            "covered_cell_count",
            "skin_count_touching",
            "dominant_skin_cell_count",
        ),
        "nonnegative_integer",
    ),
    "dominant_skin_index": "optional_nonnegative_integer",
    **dict.fromkeys(
        ("recall", "dominant_skin_fraction_of_truth"),
        "closed_unit_interval_number",
    ),
}
_SKIN_COMPONENT_REPORT_SCHEMA = {
    **dict.fromkeys(
        (
            "skin_index",
            "cell_count",
            "truth_cell_count",
            "background_cell_count",
            "truth_component_count_touching",
            "dominant_truth_cell_count",
        ),
        "nonnegative_integer",
    ),
    "dominant_truth_id": "optional_nonnegative_integer",
    "purity": "closed_unit_interval_number",
}


def write_artifact_bundle(
    result: SyntheticModeComparisonResult,
    output_dir: str | PathLike[str],
    *,
    config: SyntheticModeComparisonConfig,
    pretty: bool = False,
) -> Path:
    """Atomically write one complete, self-validating result bundle."""

    if not isinstance(result, SyntheticModeComparisonResult):
        raise ValueError("result must be a SyntheticModeComparisonResult")
    if not isinstance(config, SyntheticModeComparisonConfig):
        raise ValueError("config must be a SyntheticModeComparisonConfig")
    if not isinstance(pretty, bool):
        raise ValueError("pretty must be bool")

    final_path = Path(output_dir)
    if os.path.lexists(final_path):
        raise FileExistsError(f"artifact output already exists: {final_path}")
    validate_mode_comparison_result(result, config)

    source = _source_provenance()
    manifest = _build_manifest(result, config, source)
    cell_reports = _json_value(result.cell_reports)
    csv_rows: dict[str, Sequence[Any]] = {
        METRICS_FILE: result.metric_rows,
        METRIC_AGGREGATES_FILE: result.metric_aggregates,
        CONTRASTS_FILE: result.contrast_rows,
        CONTRAST_AGGREGATES_FILE: result.contrast_aggregates,
        RUNTIME_FILE: result.runtime_rows,
    }

    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{final_path.name}.tmp-", dir=final_path.parent)
    )
    finalized = False
    try:
        file_bytes: dict[str, bytes] = {
            MANIFEST_FILE: _json_bytes(manifest, pretty=pretty),
            CELL_REPORTS_FILE: _json_bytes(cell_reports, pretty=pretty),
        }
        file_bytes.update(
            {
                filename: _csv_bytes(rows, _CSV_MODELS[filename])
                for filename, rows in csv_rows.items()
            }
        )

        metadata: dict[str, dict[str, Any]] = {}
        for filename in HASHED_BUNDLE_FILES:
            payload = file_bytes[filename]
            artifact_path = temporary_path / filename
            _write_bytes(artifact_path, payload)
            metadata[filename] = _file_metadata(artifact_path)

        completion = {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "status": "complete",
            "required_files": list(REQUIRED_BUNDLE_FILES),
            "files": metadata,
        }
        _write_bytes(
            temporary_path / COMPLETION_FILE,
            _json_bytes(completion, pretty=pretty),
        )
        _fsync_directory(temporary_path)
        _finalize_bundle(temporary_path, final_path)
        finalized = True
        _fsync_directory(final_path.parent)
    except BaseException as error:
        _cleanup_path(final_path if finalized else temporary_path, error)
        raise
    return final_path


def validate_completed_bundle(path: str | PathLike[str]) -> bool:
    """Reject an incomplete, changed, non-finite, or malformed artifact bundle."""

    bundle = Path(path)
    if not bundle.is_dir() or bundle.is_symlink():
        raise ValueError("artifact bundle must be a directory")
    entries = {entry.name: entry for entry in bundle.iterdir()}
    expected = set(REQUIRED_BUNDLE_FILES)
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        unexpected = sorted(set(entries) - expected)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        raise ValueError("invalid artifact file set (" + "; ".join(details) + ")")
    if any(not entry.is_file() or entry.is_symlink() for entry in entries.values()):
        raise ValueError("artifact bundle entries must be regular files")

    completion = _object(
        _read_json(entries[COMPLETION_FILE]),
        {"schema_version", "status", "required_files", "files"},
        "completion.json",
    )
    completion_schema_version = _integer(
        completion.get("schema_version"), "completion.json.schema_version"
    )
    if completion_schema_version != COMPLETION_SCHEMA_VERSION:
        raise ValueError("unsupported completion schema version")
    if completion.get("status") != "complete":
        raise ValueError("completion status must be 'complete'")
    if completion.get("required_files") != list(REQUIRED_BUNDLE_FILES):
        raise ValueError("completion required_files do not match the bundle contract")
    metadata = completion.get("files")
    if not isinstance(metadata, dict) or set(metadata) != set(HASHED_BUNDLE_FILES):
        raise ValueError("completion file metadata do not match the bundle contract")

    verified_payloads: dict[str, bytes] = {}
    for filename in HASHED_BUNDLE_FILES:
        record = metadata[filename]
        if not isinstance(record, dict) or set(record) != {"sha256", "size"}:
            raise ValueError(f"invalid completion metadata for {filename}")
        expected_size = record["size"]
        expected_hash = record["sha256"]
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            raise ValueError(f"invalid recorded size for {filename}")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"invalid recorded SHA-256 for {filename}")
        payload = entries[filename].read_bytes()
        if len(payload) != expected_size:
            raise ValueError(f"size mismatch for {filename}")
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f"SHA-256 mismatch for {filename}")
        verified_payloads[filename] = payload

    config, result = _load_bundle_objects(verified_payloads)
    validate_mode_comparison_result(result, config)
    return True


def _load_bundle_objects(
    payloads: Mapping[str, bytes],
) -> tuple[SyntheticModeComparisonConfig, SyntheticModeComparisonResult]:
    manifest_value = _read_json_bytes(payloads[MANIFEST_FILE], MANIFEST_FILE)
    if not isinstance(manifest_value, dict):
        raise ValueError("manifest.json must be an object")
    if "artifact_schema_version" not in manifest_value:
        raise ValueError("manifest artifact_schema_version is required")
    artifact_schema_version = _integer(
        manifest_value["artifact_schema_version"], "manifest artifact_schema_version"
    )
    if artifact_schema_version != ARTIFACT_SCHEMA_VERSION:
        if artifact_schema_version == 1:
            raise ValueError(
                "unsupported artifact schema version 1: legacy bundle does not contain "
                "complete scanner metric evidence"
            )
        if artifact_schema_version == 2:
            raise ValueError(
                "unsupported artifact schema version 2: legacy bundle does not uniquely "
                "identify runtime coverage"
            )
        raise ValueError("unsupported artifact schema version")

    _manifest_contract_version(
        manifest_value,
        "scalar_evidence_contract_version",
        SCALAR_EVIDENCE_CONTRACT_VERSION,
        "scalar evidence",
        legacy_messages={
            1: (
                "unsupported scalar evidence contract version 1: legacy schema-v3 bundle "
                "does not contain trial truth evidence"
            ),
            2: (
                "unsupported scalar evidence contract version 2: legacy schema-v3 bundle "
                "does not validate buffered overlap radius regimes"
            ),
            3: (
                "unsupported scalar evidence contract version 3: legacy schema-v3 bundle "
                "does not constrain mask-derived counts to the canonical volume capacity"
            ),
        },
    )
    _manifest_contract_version(
        manifest_value,
        "runtime_contract_version",
        RUNTIME_CONTRACT_VERSION,
        "runtime",
        legacy_messages={
            1: (
                "unsupported runtime contract version 1: legacy schema-v3 bundle "
                "does not contain downstream shared scalar evidence attribution"
            ),
            2: (
                "unsupported runtime contract version 2: legacy schema-v3 bundle "
                "does not contain shared volume-stage attribution or elapsed algebra"
            ),
        },
    )

    manifest = _object(
        manifest_value,
        _MANIFEST_FIELDS,
        "manifest.json",
    )
    metric_schema_version = _integer(
        manifest["metric_schema_version"], "manifest metric_schema_version"
    )
    if metric_schema_version != METRIC_SCHEMA_VERSION:
        raise ValueError("unsupported metric schema version")

    config = _load_input_config(manifest["input_config"])
    plan = build_mode_comparison_plan(config)
    expected_plan = _json_value(asdict(plan))
    if not _exact_json_value(manifest["resolved_plan"], expected_plan):
        raise ValueError("manifest resolved_plan does not match input_config")
    _validate_manifest_plan_records(manifest, plan)
    cache_stats = _load_cache_stats(manifest["cache_stats"])
    _validate_manifest_environment(manifest)

    reports = _load_cell_reports(
        _read_json_bytes(payloads[CELL_REPORTS_FILE], CELL_REPORTS_FILE),
        plan,
    )
    csv_rows = {filename: _load_csv(payloads[filename], filename) for filename in _CSV_MODELS}
    result = SyntheticModeComparisonResult(
        plan_metadata=manifest["resolved_plan"],
        trial_metadata=tuple(_json_value(asdict(trial)) for trial in plan.trials),
        cell_reports=reports,
        metric_rows=csv_rows[METRICS_FILE],
        contrast_rows=csv_rows[CONTRASTS_FILE],
        metric_aggregates=csv_rows[METRIC_AGGREGATES_FILE],
        contrast_aggregates=csv_rows[CONTRAST_AGGREGATES_FILE],
        cache_stats=cache_stats,
        runtime_rows=csv_rows[RUNTIME_FILE],
    )
    return config, result


def _load_input_config(value: Any) -> SyntheticModeComparisonConfig:
    data = _dataclass_object(value, SyntheticModeComparisonConfig, "manifest input_config")
    case_set = _optional_string(data["case_set"], "input_config.case_set")
    case_ids_value = data["case_ids"]
    case_ids = (
        None if case_ids_value is None else _string_tuple(case_ids_value, "input_config.case_ids")
    )
    return SyntheticModeComparisonConfig(
        case_set=case_set,
        case_ids=case_ids,
        trial_seeds=_integer_tuple(data["trial_seeds"], "input_config.trial_seeds"),
        shape=_shape(data["shape"], "input_config.shape"),
        scanner_template=_load_scanner_config(data["scanner_template"]),
        voting_config=(
            None if data["voting_config"] is None else _load_voting_config(data["voting_config"])
        ),
        skinning_config=_load_skinning_config(data["skinning_config"]),
        truth_metric_config=_load_truth_metric_config(data["truth_metric_config"]),
        include_oracle_workflow_isolation=_boolean(
            data["include_oracle_workflow_isolation"],
            "input_config.include_oracle_workflow_isolation",
        ),
        comparison_variant=_string(data["comparison_variant"], "input_config.comparison_variant"),
        skinner_method_explicit=_boolean(
            data["skinner_method_explicit"], "input_config.skinner_method_explicit"
        ),
        skinner_min_likelihood_explicit=_boolean(
            data["skinner_min_likelihood_explicit"],
            "input_config.skinner_min_likelihood_explicit",
        ),
        skinner_growth_source_explicit=_boolean(
            data["skinner_growth_source_explicit"],
            "input_config.skinner_growth_source_explicit",
        ),
        skinner_accepted_occupancy_radius_explicit=_boolean(
            data["skinner_accepted_occupancy_radius_explicit"],
            "input_config.skinner_accepted_occupancy_radius_explicit",
        ),
        skinner_boundary_fallback_explicit=_boolean(
            data["skinner_boundary_fallback_explicit"],
            "input_config.skinner_boundary_fallback_explicit",
        ),
    )


def _load_scanner_config(value: Any) -> SyntheticScannerConfig:
    data = _dataclass_object(value, SyntheticScannerConfig, "input_config.scanner_template")
    input_data = _dataclass_object(
        data["input_config"],
        SyntheticScannerInputConfig,
        "input_config.scanner_template.input_config",
    )
    scanner_input = SyntheticScannerInputConfig(
        background=_number(input_data["background"], "scanner input background"),
        fault_contrast=_number(input_data["fault_contrast"], "scanner input fault_contrast"),
        noise_sigma=_number(input_data["noise_sigma"], "scanner input noise_sigma"),
        seed=_integer(input_data["seed"], "scanner input seed"),
        clip_min=_number(input_data["clip_min"], "scanner input clip_min"),
        clip_max=_number(input_data["clip_max"], "scanner input clip_max"),
    )
    return SyntheticScannerConfig(
        backend=_string(data["backend"], "scanner backend"),
        phi_min=_number(data["phi_min"], "scanner phi_min"),
        phi_max=_number(data["phi_max"], "scanner phi_max"),
        theta_min=_number(data["theta_min"], "scanner theta_min"),
        theta_max=_number(data["theta_max"], "scanner theta_max"),
        sigma1=_number(data["sigma1"], "scanner sigma1"),
        sigma2=_number(data["sigma2"], "scanner sigma2"),
        refinement_factor=_integer(data["refinement_factor"], "scanner refinement_factor"),
        scanner_thin_mode=_string(data["scanner_thin_mode"], "scanner thin mode"),
        remove_edge_effects=_boolean(data["remove_edge_effects"], "scanner remove_edge_effects"),
        input_config=scanner_input,
    )


def _load_voting_config(value: Any) -> SyntheticVotingConfig:
    data = _dataclass_object(value, SyntheticVotingConfig, "input_config.voting_config")
    return SyntheticVotingConfig(
        ru=_integer(data["ru"], "voting ru"),
        rv=_integer(data["rv"], "voting rv"),
        rw=_integer(data["rw"], "voting rw"),
        seed_distance=_integer(data["seed_distance"], "voting seed_distance"),
        seed_threshold=_number(data["seed_threshold"], "voting seed_threshold"),
        attribute_smoothing=_integer(data["attribute_smoothing"], "voting attribute_smoothing"),
        voter_thin_mode=_string(data["voter_thin_mode"], "voting voter_thin_mode"),
        reference_thin_sigma=_number(data["reference_thin_sigma"], "voting reference_thin_sigma"),
        surface_support_min_fraction=_number(
            data["surface_support_min_fraction"], "voting surface_support_min_fraction"
        ),
        surface_support_exponent=_number(
            data["surface_support_exponent"], "voting surface_support_exponent"
        ),
    )


def _load_skinning_config(value: Any) -> SyntheticSkinningConfig:
    data = _dataclass_object(value, SyntheticSkinningConfig, "input_config.skinning_config")
    return SyntheticSkinningConfig(
        enabled=_boolean(data["enabled"], "skinning enabled"),
        method=_string(data["method"], "skinning method"),
        growth_source=_string(data["growth_source"], "skinning growth_source"),
        min_likelihood=_optional_number(data["min_likelihood"], "skinning min_likelihood"),
        min_skin_size=_optional_integer(data["min_skin_size"], "skinning min_skin_size"),
        d=_integer(data["d"], "skinning d"),
        ru=_integer(data["ru"], "skinning ru"),
        rv=_optional_integer(data["rv"], "skinning rv"),
        rw=_optional_integer(data["rw"], "skinning rw"),
        max_steps=_integer(data["max_steps"], "skinning max_steps"),
        du=_number(data["du"], "skinning du"),
        max_delta_strike=_number(data["max_delta_strike"], "skinning max_delta_strike"),
        reskin=_boolean(data["reskin"], "skinning reskin"),
        accepted_occupancy_radius=_optional_integer(
            data["accepted_occupancy_radius"], "skinning accepted_occupancy_radius"
        ),
        small_skin_size=_integer(data["small_skin_size"], "skinning small_skin_size"),
        boundary_skinner_fallback=_boolean(
            data["boundary_skinner_fallback"], "skinning boundary_skinner_fallback"
        ),
        boundary_skinner_fallback_policy=_string(
            data["boundary_skinner_fallback_policy"],
            "skinning boundary_skinner_fallback_policy",
        ),
    )


def _load_truth_metric_config(value: Any) -> SyntheticTruthMetricConfig:
    data = _dataclass_object(value, SyntheticTruthMetricConfig, "input_config.truth_metric_config")
    return SyntheticTruthMetricConfig(
        truth_surface_half_width=_number(
            data["truth_surface_half_width"], "truth metric truth_surface_half_width"
        ),
        buffer_radius=_number(data["buffer_radius"], "truth metric buffer_radius"),
    )


def _validate_manifest_plan_records(manifest: Mapping[str, Any], plan: Any) -> None:
    expected_cells = [
        {"order": order, "label": cell.label, "scope": cell.scope}
        for order, cell in enumerate(plan.cells)
    ]
    expected_case_order = list(plan.case_ids)
    case_stochastic = {trial.case_id: trial.seed is not None for trial in plan.trials}
    expected_cases = [
        {
            "order": order,
            "case_id": case_id,
            "stochastic": case_stochastic.get(case_id, False),
        }
        for order, case_id in enumerate(plan.case_ids)
    ]
    scanner_seed = plan.scanner_template.input_config.seed
    expected_trials = [
        {
            "order": order,
            "case_id": trial.case_id,
            "trial_id": trial.trial_id,
            "stochastic": case_stochastic[trial.case_id],
            "case_generation_seed": trial.seed,
            "scanner_input_seed": scanner_seed,
        }
        for order, trial in enumerate(plan.trials)
    ]
    expected = {
        "canonical_cells": expected_cells,
        "case_order": expected_case_order,
        "cases": expected_cases,
        "trials": expected_trials,
        "shape": list(plan.shape),
        "variant": plan.comparison_variant,
        "oracle_workflow_isolation": plan.include_oracle_workflow_isolation,
        "metric_registry": {
            "id": METRIC_REGISTRY_ID,
            "definition_version": METRIC_REGISTRY_DEFINITION_VERSION,
        },
        "contrast_definition": {
            "id": CONTRAST_DEFINITION_ID,
            "formula_version": CONTRAST_FORMULA_VERSION,
        },
    }
    for name, value in expected.items():
        if not _exact_json_value(manifest[name], value):
            raise ValueError(f"manifest {name} does not match the canonical plan")


def _load_cell_reports(
    value: Any,
    plan: SyntheticModeComparisonPlan,
) -> tuple[Mapping[str, Any], ...]:
    reports = _array(value, "cell_reports.json")
    if len(reports) != len(plan.trials):
        raise ValueError("cell_reports must contain exactly one report per canonical trial")
    expected_cells = {cell.label: cell for cell in plan.cells}
    output = []
    for index, report_value in enumerate(reports):
        context = f"cell_reports[{index}]"
        report = _object(
            report_value,
            {"case_id", "trial_id", "seed", "truth_evidence", "cells"},
            context,
        )
        if tuple(report) != ("case_id", "trial_id", "seed", "truth_evidence", "cells"):
            raise ValueError(f"{context} fields do not match the canonical schema and order")
        trial = plan.trials[index]
        metadata = (
            _string(report["case_id"], f"{context}.case_id"),
            _string(report["trial_id"], f"{context}.trial_id"),
            _optional_integer(report["seed"], f"{context}.seed"),
        )
        if metadata != (trial.case_id, trial.trial_id, trial.seed):
            raise ValueError(f"{context} trial metadata does not match the canonical trial")
        truth_evidence = _load_trial_truth_evidence(
            report["truth_evidence"],
            plan.shape,
            f"{context}.truth_evidence",
        )
        cells = _object(report["cells"], set(expected_cells), f"{context}.cells")
        if tuple(cells) != tuple(expected_cells):
            raise ValueError("cell report cells do not match the canonical cells and order")
        loaded_cells = {
            label: _load_cell_payload(
                payload,
                expected_cells[label],
                plan,
                f"{context}.cells.{label}",
            )
            for label, payload in cells.items()
        }
        _validate_trial_truth_targets(
            loaded_cells,
            truth_evidence=truth_evidence,
            context=context,
        )
        output.append({**report, "truth_evidence": truth_evidence, "cells": loaded_cells})
    return tuple(output)


def _load_trial_truth_evidence(value: Any, shape: Sequence[int], context: str) -> dict[str, int]:
    fields = ("fault_voxel_count", "surface_voxel_count")
    evidence = _object(value, set(fields), context)
    if tuple(evidence) != fields:
        raise ValueError(f"{context} fields do not match the canonical schema and order")
    voxel_count = volume_voxel_count(shape)
    validated = {name: _nonnegative_integer(evidence[name], f"{context}.{name}") for name in fields}
    for name, count in validated.items():
        if not 1 <= count <= voxel_count:
            raise ValueError(
                f"{context}.{name} must be between 1 and volume voxel count {voxel_count}"
            )
    return validated


def _validate_trial_truth_targets(
    cells: Mapping[str, Mapping[str, Any]],
    *,
    truth_evidence: Mapping[str, int],
    context: str,
) -> None:
    fault_count = truth_evidence["fault_voxel_count"]
    surface_count = truth_evidence["surface_voxel_count"]
    for label, payload in cells.items():
        cell_context = f"{context}.cells.{label}"
        if "scanner_quality" in payload:
            scanner_quality = payload["scanner_quality"]["ft_top_truth_count"]
            _require_truth_target_counts(
                scanner_quality,
                fault_count=fault_count,
                surface_count=surface_count,
                context=f"{cell_context}.scanner_quality.ft_top_truth_count",
            )
            for entry in payload["scanner_metric_evidence"]:
                if "quality_report" in entry:
                    _require_truth_target_counts(
                        entry["quality_report"],
                        fault_count=fault_count,
                        surface_count=surface_count,
                        context=(
                            f"{cell_context}.scanner_metric_evidence."
                            f"{entry['stage']}.{entry['selection']}"
                        ),
                    )
        if "quality" not in payload:
            continue
        quality = payload["quality"]
        for stage in ("fv", "fvt"):
            for selection in ("top_truth_count", "positive_top_truth_count"):
                name = f"{stage}_{selection}"
                _require_truth_target_counts(
                    quality[name],
                    fault_count=fault_count,
                    surface_count=surface_count,
                    context=f"{cell_context}.quality.{name}",
                )
        if quality["skin"] is not None:
            _require_truth_target_counts(
                quality["skin"],
                fault_count=fault_count,
                surface_count=surface_count,
                context=f"{cell_context}.quality.skin",
            )


def _require_truth_target_counts(
    report: Mapping[str, Any],
    *,
    fault_count: int,
    surface_count: int,
    context: str,
) -> None:
    overlap_count = report["buffered_overlap_radius2"]["truth_count"]
    if overlap_count != fault_count:
        raise ValueError(
            f"{context}.buffered_overlap_radius2.truth_count does not match "
            "trial truth_evidence.fault_voxel_count"
        )
    distance_count = report["surface_distance"]["truth_count"]
    if distance_count != surface_count:
        raise ValueError(
            f"{context}.surface_distance.truth_count does not match "
            "trial truth_evidence.surface_voxel_count"
        )


def _load_cell_payload(
    value: Any,
    cell: ModeCellSpec,
    plan: SyntheticModeComparisonPlan,
    context: str,
) -> dict[str, Any]:
    if cell.scope == SCANNER_ONLY_SCOPE:
        payload = _object(
            value,
            {"scanner", "scanner_quality", "scanner_metric_evidence"},
            context,
        )
        scanner_config = replace(plan.scanner_template, backend=cell.scanner_backend)
        scanner_report = _load_scanner_report(
            payload["scanner"],
            scanner_config,
            plan.shape,
            f"{context}.scanner",
        )
        scanner_quality = _load_scanner_quality_report(
            payload["scanner_quality"],
            buffer_radius=plan.truth_metric_config.buffer_radius,
            shape=plan.shape,
            context=f"{context}.scanner_quality",
        )
        payload["scanner_metric_evidence"] = _load_scanner_metric_evidence(
            payload["scanner_metric_evidence"],
            scanner_backend=scanner_config.backend,
            scanner_report=scanner_report,
            scanner_quality=scanner_quality,
            buffer_radius=plan.truth_metric_config.buffer_radius,
            shape=plan.shape,
            context=f"{context}.scanner_metric_evidence",
        )
        return payload

    settings = (
        plan.reference_workflow_settings
        if cell.workflow_mode == "reference"
        else plan.quality_workflow_settings
    )
    expected_fields = (
        _DOWNSTREAM_REPORT_FIELDS
        if cell.scope == ORACLE_WORKFLOW_ISOLATION_SCOPE
        else _END_TO_END_REPORT_FIELDS
    )
    payload = _object(value, expected_fields, context)
    _load_downstream_sections(payload, plan, settings, context)
    if cell.scope == ORACLE_WORKFLOW_ISOLATION_SCOPE:
        return payload

    active_pipeline = _string(payload["active_pipeline"], f"{context}.active_pipeline")
    if active_pipeline != cell.input_mode:
        raise ValueError(f"{context}.active_pipeline does not match the canonical cell")
    scanner_config = replace(plan.scanner_template, backend=cell.scanner_backend)
    scanner_report = _load_scanner_report(
        payload["scanner"], scanner_config, plan.shape, f"{context}.scanner"
    )
    scanner_quality = _load_scanner_quality_report(
        payload["scanner_quality"],
        buffer_radius=plan.truth_metric_config.buffer_radius,
        shape=plan.shape,
        context=f"{context}.scanner_quality",
    )
    payload["scanner_metric_evidence"] = _load_scanner_metric_evidence(
        payload["scanner_metric_evidence"],
        scanner_backend=scanner_config.backend,
        scanner_report=scanner_report,
        scanner_quality=scanner_quality,
        buffer_radius=plan.truth_metric_config.buffer_radius,
        shape=plan.shape,
        context=f"{context}.scanner_metric_evidence",
    )
    pipelines = _object(payload["pipelines"], {cell.input_mode}, f"{context}.pipelines")
    pipeline = _object(
        pipelines[cell.input_mode],
        {*_DOWNSTREAM_REPORT_FIELDS, "scanner", "scanner_quality"},
        f"{context}.pipelines.{cell.input_mode}",
    )
    pipeline_context = f"{context}.pipelines.{cell.input_mode}"
    _load_downstream_sections(pipeline, plan, settings, pipeline_context)
    _load_scanner_report(
        pipeline["scanner"],
        scanner_config,
        plan.shape,
        f"{pipeline_context}.scanner",
    )
    _load_scanner_quality_report(
        pipeline["scanner_quality"],
        buffer_radius=plan.truth_metric_config.buffer_radius,
        shape=plan.shape,
        context=f"{pipeline_context}.scanner_quality",
    )
    for name in (*_DOWNSTREAM_REPORT_FIELDS, "scanner", "scanner_quality"):
        if not _exact_json_value(payload[name], pipeline[name]):
            raise ValueError(f"{context}.{name} does not match the active pipeline duplicate")
    return payload


def _load_downstream_sections(
    payload: Mapping[str, Any],
    plan: SyntheticModeComparisonPlan,
    settings: ResolvedWorkflowSettings,
    context: str,
) -> None:
    effective_skinning = effective_skinning_config(
        get_variant_spec(plan.comparison_variant), settings.skinning_config
    )
    config_context = f"{context}.config"
    config = _object(payload["config"], {"skinning"}, config_context)
    skinning_config = _load_scalar_report_object(
        config["skinning"],
        _SKINNING_CONFIG_REPORT_SCHEMA,
        f"{config_context}.skinning",
    )
    if not _exact_json_value(skinning_config, effective_skinning.as_report_dict()):
        raise ValueError(
            f"{config_context}.skinning does not match the resolved workflow configuration"
        )
    enabled = skinning_config["enabled"]

    skinning_context = f"{context}.skinning"
    skinning_fields = {"enabled", "diagnostics"} if enabled else {"enabled"}
    skinning = _object(payload["skinning"], skinning_fields, skinning_context)
    if _boolean(skinning["enabled"], f"{skinning_context}.enabled") != enabled:
        raise ValueError(f"{skinning_context}.enabled does not match config.skinning.enabled")
    if enabled:
        _load_skinning_diagnostics(
            skinning["diagnostics"],
            method=skinning_config["method"],
            context=f"{skinning_context}.diagnostics",
        )

    pyosv_context = f"{context}.pyosv"
    pyosv_report = _object(payload["pyosv"], {"fv", "fvt", "voting", "skins"}, pyosv_context)
    _load_array_summary(pyosv_report["fv"], plan.shape, f"{pyosv_context}.fv")
    _load_array_summary(pyosv_report["fvt"], plan.shape, f"{pyosv_context}.fvt")
    _load_voting_report(
        pyosv_report["voting"],
        settings.voting_config,
        f"{pyosv_context}.voting",
    )
    pyosv_topology = _load_scalar_report_object(
        pyosv_report["skins"],
        _SKIN_TOPOLOGY_REPORT_SCHEMA,
        f"{pyosv_context}.skins",
    )
    validate_skin_topology_algebra(
        pyosv_topology,
        f"{pyosv_context}.skins",
        shape=plan.shape,
        require_empty=not enabled,
    )

    quality = _load_downstream_quality_report(
        payload["quality"],
        enabled=enabled,
        small_skin_size=effective_skinning.small_skin_size,
        buffer_radius=plan.truth_metric_config.buffer_radius,
        shape=plan.shape,
        context=f"{context}.quality",
    )
    if enabled and not _exact_json_value(pyosv_topology, quality["skin"]["topology"]):
        raise ValueError(f"{context}.pyosv.skins does not match quality.skin.topology")


def _load_scanner_report(
    value: Any,
    scanner_config: SyntheticScannerConfig,
    expected_shape: tuple[int, int, int],
    context: str,
) -> dict[str, Any]:
    expected_fields = set(_SCANNER_REPORT_FIELDS)
    if scanner_config.backend == "quality":
        expected_fields.add("confidence")
    payload = _object(value, expected_fields, context)
    config_context = f"{context}.config"
    config = _object(payload["config"], _SCANNER_CONFIG_REPORT_FIELDS, config_context)
    if _string(config["backend"], f"{config_context}.backend") != scanner_config.backend:
        raise ValueError(f"{config_context}.backend does not match the canonical cell")
    for name in ("phi_min", "phi_max", "theta_min", "theta_max", "sigma1", "sigma2"):
        _number(config[name], f"{config_context}.{name}")
    _integer(config["refinement_factor"], f"{config_context}.refinement_factor")
    _string(config["scanner_thin_mode"], f"{config_context}.scanner_thin_mode")
    _boolean(config["remove_edge_effects"], f"{config_context}.remove_edge_effects")
    input_context = f"{config_context}.input"
    scanner_input = _object(
        config["input"],
        _SCANNER_INPUT_CONFIG_REPORT_FIELDS,
        input_context,
    )
    for name in ("background", "fault_contrast", "noise_sigma", "clip_min", "clip_max"):
        _number(scanner_input[name], f"{input_context}.{name}")
    _integer(scanner_input["seed"], f"{input_context}.seed")
    if not _exact_json_value(config, scanner_config.as_report_dict()):
        raise ValueError(f"{config_context} does not match the resolved scanner configuration")
    for name in expected_fields - {"config"}:
        _load_array_summary(
            payload[name],
            expected_shape,
            f"{context}.{name}",
            values_in_closed_unit_interval=name == "confidence",
        )
    return payload


def _load_array_summary(
    value: Any,
    expected_shape: tuple[int, int, int],
    context: str,
    *,
    values_in_closed_unit_interval: bool = False,
) -> dict[str, Any]:
    summary = _object(value, _ARRAY_SUMMARY_FIELDS, context)
    shape = _shape(summary["shape"], f"{context}.shape")
    if shape != expected_shape:
        raise ValueError(f"{context}.shape does not match the canonical plan shape")
    finite_count = _nonnegative_integer(summary["finite_count"], f"{context}.finite_count")
    voxel_count = expected_shape[0] * expected_shape[1] * expected_shape[2]
    if finite_count != voxel_count:
        raise ValueError(f"{context}.finite_count must equal the shape voxel count")
    finite_fraction = _number(summary["finite_fraction"], f"{context}.finite_fraction")
    if finite_fraction != 1.0:
        raise ValueError(f"{context}.finite_fraction must equal 1.0")
    minimum = _number(summary["min"], f"{context}.min")
    maximum = _number(summary["max"], f"{context}.max")
    mean = _number(summary["mean"], f"{context}.mean")
    _closed_unit_interval_number(summary["nonzero_fraction"], f"{context}.nonzero_fraction")
    if values_in_closed_unit_interval:
        for name in ("min", "mean", "max"):
            _closed_unit_interval_number(summary[name], f"{context}.{name}")
    if not minimum <= mean <= maximum:
        raise ValueError(f"{context} must satisfy min <= mean <= max")
    return summary


def _load_scanner_quality_report(
    value: Any, *, buffer_radius: float, shape: tuple[int, int, int], context: str
) -> dict[str, Any]:
    payload = _object(value, _SCANNER_QUALITY_REPORT_FIELDS, context)
    truth_count = _object(
        payload["ft_top_truth_count"],
        {"buffered_overlap_radius2", "surface_distance"},
        f"{context}.ft_top_truth_count",
    )
    overlap = _load_scalar_report_object(
        truth_count["buffered_overlap_radius2"],
        _BUFFERED_OVERLAP_REPORT_SCHEMA,
        f"{context}.ft_top_truth_count.buffered_overlap_radius2",
    )
    _validate_report_radius(
        overlap["radius"],
        buffer_radius,
        f"{context}.ft_top_truth_count.buffered_overlap_radius2.radius",
    )
    _load_scalar_report_object(
        truth_count["surface_distance"],
        _SURFACE_DISTANCE_REPORT_SCHEMA,
        f"{context}.ft_top_truth_count.surface_distance",
    )
    orientation = _object(
        payload["orientation_error"],
        {"raw_scan_top_truth_count", "used_attributes_top_truth_count"},
        f"{context}.orientation_error",
    )
    for name, item in orientation.items():
        _load_scalar_report_object(
            item,
            _ORIENTATION_ERROR_REPORT_SCHEMA,
            f"{context}.orientation_error.{name}",
        )
    validate_scanner_quality_scalar_algebra(payload, shape, context)
    _load_scalar_report_object(
        payload["input_association"],
        _INPUT_ASSOCIATION_REPORT_SCHEMA,
        f"{context}.input_association",
    )
    return payload


def _load_scanner_metric_evidence(
    value: Any,
    *,
    scanner_backend: str,
    scanner_report: Mapping[str, Any],
    scanner_quality: Mapping[str, Any],
    buffer_radius: float,
    shape: tuple[int, int, int],
    context: str,
) -> list[dict[str, Any]]:
    entries = _array(value, context)
    definitions = scanner_metric_definitions(scanner_backend)
    if len(entries) != len(definitions):
        raise ValueError(f"{context} must contain exactly every applicable scanner registry metric")
    loaded: list[dict[str, Any]] = []
    base_fields = {"stage", "selection", "metric", "value", "unit", "direction"}
    for index, (entry_value, definition) in enumerate(zip(entries, definitions, strict=True)):
        entry_context = f"{context}[{index}]"
        fields = (
            {*base_fields, "quality_report"}
            if definition.metric == "candidate_count"
            else base_fields
        )
        entry = _object(entry_value, fields, entry_context)
        identity = tuple(
            _string(entry[name], f"{entry_context}.{name}")
            for name in ("stage", "selection", "metric")
        )
        expected_identity = (definition.stage, definition.selection, definition.metric)
        if identity != expected_identity:
            raise ValueError(
                f"{entry_context} identity does not match the applicable metric registry order"
            )
        unit = _string(entry["unit"], f"{entry_context}.unit")
        direction = _string(entry["direction"], f"{entry_context}.direction")
        if unit != definition.unit or direction != definition.direction:
            raise ValueError(f"{entry_context} unit or direction does not match the registry")
        try:
            metric_value = validate_metric_value(
                definition,
                _number(entry["value"], f"{entry_context}.value"),
            )
        except ValueError as error:
            raise ValueError(
                f"{entry_context}.value violates the metric registry: {error}"
            ) from error
        loaded_entry = {
            "stage": identity[0],
            "selection": identity[1],
            "metric": identity[2],
            "value": metric_value,
            "unit": unit,
            "direction": direction,
        }
        if definition.metric == "candidate_count":
            loaded_entry["quality_report"] = _load_scanner_evidence_quality_report(
                entry["quality_report"],
                selection=definition.selection,
                buffer_radius=buffer_radius,
                shape=shape,
                context=f"{entry_context}.quality_report",
            )
        loaded.append(loaded_entry)
    _validate_scanner_metric_evidence_consistency(
        loaded,
        scanner_report=scanner_report,
        scanner_quality=scanner_quality,
        context=context,
    )
    return loaded


def _validate_scanner_metric_evidence_consistency(
    evidence: Sequence[Mapping[str, Any]],
    *,
    scanner_report: Mapping[str, Any],
    scanner_quality: Mapping[str, Any],
    context: str,
) -> None:
    truth_count = scanner_quality["ft_top_truth_count"]
    overlap = truth_count["buffered_overlap_radius2"]
    distance = truth_count["surface_distance"]
    orientations = scanner_quality["orientation_error"]
    raw_orientation = orientations["raw_scan_top_truth_count"]
    thinned_orientation = orientations["used_attributes_top_truth_count"]
    quality_reports = {
        (entry["stage"], entry["selection"]): entry["quality_report"]
        for entry in evidence
        if "quality_report" in entry
    }
    expected_quality_identities = {
        ("scanner_raw", "top_truth_count"),
        ("scanner_thinned", "top_truth_count"),
    }
    if set(quality_reports) != expected_quality_identities:
        raise ValueError(f"{context} must contain one quality report for each scanner stage")
    raw_quality = quality_reports[("scanner_raw", "top_truth_count")]
    thinned_quality = quality_reports[("scanner_thinned", "top_truth_count")]
    for field in ("candidate_count", "truth_count"):
        raw_report = (
            raw_quality["buffered_overlap_radius2"]
            if field == "candidate_count"
            else raw_quality["surface_distance"]
        )
        thinned_report = (
            thinned_quality["buffered_overlap_radius2"]
            if field == "candidate_count"
            else thinned_quality["surface_distance"]
        )
        if raw_report[field] != thinned_report[field]:
            raise ValueError(f"{context} scanner raw and thinned {field} must match")
    for evidence_report, legacy_report, report_name in (
        (
            raw_quality["buffered_overlap_radius2"],
            overlap,
            "scanner_raw buffered overlap",
        ),
        (raw_quality["surface_distance"], distance, "scanner_raw surface distance"),
        (raw_quality["orientation_error"], raw_orientation, "scanner_raw orientation"),
        (
            thinned_quality["orientation_error"],
            thinned_orientation,
            "scanner_thinned orientation",
        ),
    ):
        if not _exact_json_value(evidence_report, legacy_report):
            raise ValueError(f"{context} {report_name} does not match scanner_quality")
    expected_values = {
        ("scanner_raw", "all", "array_nonzero_fraction"): scanner_report["ft"]["nonzero_fraction"],
        ("scanner_thinned", "all", "array_nonzero_fraction"): scanner_report["fet"][
            "nonzero_fraction"
        ],
    }
    if "confidence" in scanner_report:
        expected_values[("scanner_confidence", "finite", "confidence_mean")] = scanner_report[
            "confidence"
        ]["mean"]
    for (stage, selection), report in quality_reports.items():
        overlap_report = report["buffered_overlap_radius2"]
        distance_report = report["surface_distance"]
        orientation_report = report["orientation_error"]
        edge_report = report["edge_false_positive"]
        sources = {
            "candidate_count": overlap_report,
            "buffered_precision": overlap_report,
            "buffered_recall": overlap_report,
            "buffered_f1": overlap_report,
            "candidate_to_truth_median": distance_report,
            "candidate_to_truth_p95": distance_report,
            "truth_to_candidate_median": distance_report,
            "truth_to_candidate_p95": distance_report,
            "hausdorff_p95": distance_report,
            "strike_median": orientation_report,
            "strike_p95": orientation_report,
            "dip_median": orientation_report,
            "dip_p95": orientation_report,
            "edge_false_positive_fraction_of_candidates": edge_report,
        }
        expected_values.update(
            {(stage, selection, metric): source[metric] for metric, source in sources.items()}
        )

    for entry in evidence:
        identity = tuple(entry[name] for name in ("stage", "selection", "metric"))
        expected = expected_values.get(identity)
        if expected is None:
            continue
        metric = entry["metric"]
        actual = entry["value"]
        if identity == ("scanner_confidence", "finite", "confidence_mean"):
            # The evidence mean uses NumPy's float32 reduction while the legacy
            # scanner summary promotes samples to float64 before reduction.
            matches = isclose(float(actual), float(expected), rel_tol=1.0e-7, abs_tol=1.0e-9)
        elif metric == "candidate_count":
            matches = actual == float(expected)
        else:
            matches = derived_scalars_close(float(actual), float(expected))
        if not matches:
            raise ValueError(
                f"{context} {identity[0]}/{identity[1]}/{metric} "
                "does not match scanner_quality/scanner evidence"
            )


def _load_scanner_evidence_quality_report(
    value: Any,
    *,
    selection: str,
    buffer_radius: float,
    shape: tuple[int, int, int],
    context: str,
) -> dict[str, Any]:
    payload = _object(
        value,
        {
            "buffered_overlap_radius2",
            "surface_distance",
            "orientation_error",
            "edge_false_positive",
        },
        context,
    )
    _load_quality_stage_report(
        {
            name: payload[name]
            for name in (
                "buffered_overlap_radius2",
                "surface_distance",
                "orientation_error",
            )
        },
        buffer_radius=buffer_radius,
        context=context,
    )
    edge = _load_scalar_report_object(
        payload["edge_false_positive"],
        _EDGE_FALSE_POSITIVE_REPORT_SCHEMA,
        f"{context}.edge_false_positive",
    )
    _validate_report_radius(
        edge["truth_buffer_radius"],
        buffer_radius,
        f"{context}.edge_false_positive.truth_buffer_radius",
    )
    validate_quality_scalar_algebra(
        overlap=payload["buffered_overlap_radius2"],
        distance=payload["surface_distance"],
        orientation=payload["orientation_error"],
        edge=edge,
        shape=shape,
        context=context,
    )
    validate_selection_cardinality(
        selection=selection,
        candidate_count=payload["buffered_overlap_radius2"]["candidate_count"],
        truth_count=payload["surface_distance"]["truth_count"],
        context=context,
    )
    return payload


def _load_skinning_diagnostics(value: Any, *, method: str, context: str) -> dict[str, Any]:
    schema = dict(_SKINNING_DIAGNOSTIC_REPORT_SCHEMA)
    if method != "connected_component":
        schema.update(_PRIMARY_SKINNER_DIAGNOSTIC_FIELDS)
    payload = _object(value, {*schema, "fallback_v5_guardrail"}, context)
    _load_scalar_report_fields(payload, schema, context)
    _load_scalar_report_object(
        payload["fallback_v5_guardrail"],
        _GUARDRAIL_REPORT_SCHEMA,
        f"{context}.fallback_v5_guardrail",
    )
    return payload


def _load_voting_report(
    value: Any, voting_config: SyntheticVotingConfig, context: str
) -> dict[str, Any]:
    payload = _object(value, {*_VOTING_REPORT_SCHEMA, "diagnostic_summary"}, context)
    _load_scalar_report_fields(payload, _VOTING_REPORT_SCHEMA, context)
    diagnostic = _load_scalar_report_object(
        payload["diagnostic_summary"],
        _VOTING_DIAGNOSTIC_REPORT_SCHEMA,
        f"{context}.diagnostic_summary",
    )
    expected = {
        "surface_voting_boundary_policy": "reference",
        "surface_support_min_fraction": float(voting_config.surface_support_min_fraction),
        "surface_support_exponent": float(voting_config.surface_support_exponent),
    }
    if any(not _exact_json_value(payload[name], value) for name, value in expected.items()):
        raise ValueError(f"{context} does not match the resolved voting configuration")
    if diagnostic["policy"] != payload["surface_voting_boundary_policy"]:
        raise ValueError(f"{context}.diagnostic_summary.policy does not match voting policy")
    return payload


def _load_downstream_quality_report(
    value: Any,
    *,
    enabled: bool,
    small_skin_size: int,
    buffer_radius: float,
    shape: tuple[int, int, int],
    context: str,
) -> dict[str, Any]:
    stage_names = (
        "fv_top_truth_count",
        "fvt_top_truth_count",
        "fv_positive_top_truth_count",
        "fvt_positive_top_truth_count",
    )
    payload = _object(value, {*stage_names, "edge_false_positive", "skin"}, context)
    for name in stage_names:
        _load_quality_stage_report(
            payload[name],
            buffer_radius=buffer_radius,
            context=f"{context}.{name}",
        )

    edge_names = {*stage_names, *(('skin',) if enabled else ())}
    edge = _object(payload["edge_false_positive"], edge_names, f"{context}.edge_false_positive")
    for name, item in edge.items():
        report = _load_scalar_report_object(
            item,
            _EDGE_FALSE_POSITIVE_REPORT_SCHEMA,
            f"{context}.edge_false_positive.{name}",
        )
        _validate_report_radius(
            report["truth_buffer_radius"],
            buffer_radius,
            f"{context}.edge_false_positive.{name}.truth_buffer_radius",
        )

    if not enabled:
        if payload["skin"] is not None:
            raise ValueError(f"{context}.skin must be null when skinning is disabled")
    else:
        _load_skin_quality_report(
            payload["skin"],
            buffer_radius=buffer_radius,
            context=f"{context}.skin",
        )
    validate_downstream_quality_scalar_algebra(payload, shape, context)
    if enabled:
        skin = payload["skin"]
        topology = skin["topology"]
        validate_skin_report_topology_algebra(
            topology,
            skin["component_topology"],
            f"{context}.skin",
            small_skin_size=small_skin_size,
            shape=shape,
        )
    return payload


def _load_quality_stage_report(value: Any, *, buffer_radius: float, context: str) -> dict[str, Any]:
    payload = _object(
        value,
        {"buffered_overlap_radius2", "surface_distance", "orientation_error"},
        context,
    )
    overlap = _load_scalar_report_object(
        payload["buffered_overlap_radius2"],
        _BUFFERED_OVERLAP_REPORT_SCHEMA,
        f"{context}.buffered_overlap_radius2",
    )
    _validate_report_radius(
        overlap["radius"],
        buffer_radius,
        f"{context}.buffered_overlap_radius2.radius",
    )
    _load_scalar_report_object(
        payload["surface_distance"],
        _SURFACE_DISTANCE_REPORT_SCHEMA,
        f"{context}.surface_distance",
    )
    _load_scalar_report_object(
        payload["orientation_error"],
        _ORIENTATION_ERROR_REPORT_SCHEMA,
        f"{context}.orientation_error",
    )
    return payload


def _load_skin_quality_report(
    value: Any,
    *,
    buffer_radius: float,
    context: str,
) -> dict[str, Any]:
    payload = _object(
        value,
        {
            "topology",
            "buffered_overlap_radius2",
            "surface_distance",
            "orientation_error",
            "component_topology",
        },
        context,
    )
    topology = _load_scalar_report_object(
        payload["topology"], _SKIN_TOPOLOGY_REPORT_SCHEMA, f"{context}.topology"
    )
    overlap = _load_scalar_report_object(
        payload["buffered_overlap_radius2"],
        _BUFFERED_OVERLAP_REPORT_SCHEMA,
        f"{context}.buffered_overlap_radius2",
    )
    _validate_report_radius(
        overlap["radius"],
        buffer_radius,
        f"{context}.buffered_overlap_radius2.radius",
    )
    _load_scalar_report_object(
        payload["surface_distance"],
        _SURFACE_DISTANCE_REPORT_SCHEMA,
        f"{context}.surface_distance",
    )
    orientation = _load_scalar_report_object(
        payload["orientation_error"],
        _ORIENTATION_ERROR_REPORT_SCHEMA,
        f"{context}.orientation_error",
    )
    if overlap["candidate_count"] != topology["unique_cell_count"]:
        raise ValueError(f"{context} candidate count must match the unique skin cell count")
    if orientation["count"] != topology["cell_count"]:
        raise ValueError(f"{context} orientation count must match the skin cell count")
    _load_component_topology_report(payload["component_topology"], f"{context}.component_topology")
    return payload


def _validate_report_radius(value: Any, expected: float, context: str) -> None:
    if not _exact_json_value(value, float(expected)):
        raise ValueError(f"{context} does not match the truth metric configuration")


def _load_component_topology_report(value: Any, context: str) -> dict[str, Any]:
    payload = _object(
        value,
        {*_COMPONENT_TOPOLOGY_SUMMARY_SCHEMA, "truth_components", "skins"},
        context,
    )
    _load_scalar_report_fields(payload, _COMPONENT_TOPOLOGY_SUMMARY_SCHEMA, context)
    for name, schema in (
        ("truth_components", _TRUTH_COMPONENT_REPORT_SCHEMA),
        ("skins", _SKIN_COMPONENT_REPORT_SCHEMA),
    ):
        items = _array(payload[name], f"{context}.{name}")
        for index, item in enumerate(items):
            _load_scalar_report_object(item, schema, f"{context}.{name}[{index}]")
    return payload


def _load_scalar_report_object(
    value: Any, schema: Mapping[str, str], context: str
) -> dict[str, Any]:
    payload = _object(value, set(schema), context)
    _load_scalar_report_fields(payload, schema, context)
    return payload


def _load_scalar_report_fields(
    payload: Mapping[str, Any], schema: Mapping[str, str], context: str
) -> None:
    for name, kind in schema.items():
        value = payload[name]
        field_context = f"{context}.{name}"
        if kind == "integer":
            _integer(value, field_context)
        elif kind == "nonnegative_integer":
            _nonnegative_integer(value, field_context)
        elif kind == "optional_integer":
            _optional_integer(value, field_context)
        elif kind == "optional_nonnegative_integer":
            _optional_nonnegative_integer(value, field_context)
        elif kind == "number":
            _number(value, field_context)
        elif kind == "nonnegative_number":
            _nonnegative_number(value, field_context)
        elif kind == "closed_unit_interval_number":
            _closed_unit_interval_number(value, field_context)
        elif kind == "optional_number":
            _optional_number(value, field_context)
        elif kind == "optional_nonnegative_number":
            _optional_nonnegative_number(value, field_context)
        elif kind == "optional_closed_unit_interval_number":
            _optional_closed_unit_interval_number(value, field_context)
        elif kind == "boolean":
            _boolean(value, field_context)
        elif kind == "string":
            _string(value, field_context)
        elif kind == "optional_string":
            _optional_string(value, field_context)
        elif kind == "string_array":
            _string_tuple(value, field_context)
        else:
            raise AssertionError(f"unknown scalar report field kind: {kind}")


def _load_cache_stats(value: Any) -> tuple[Mapping[str, Any], ...]:
    rows = _array(value, "manifest cache_stats")
    expected_fields = {"case_id", "trial_id", "seed", *_CACHE_COUNTERS}
    output = []
    for index, row_value in enumerate(rows):
        row = _object(row_value, expected_fields, f"cache_stats[{index}]")
        _string(row["case_id"], f"cache_stats[{index}].case_id")
        _string(row["trial_id"], f"cache_stats[{index}].trial_id")
        _optional_integer(row["seed"], f"cache_stats[{index}].seed")
        for name in _CACHE_COUNTERS:
            counter = _integer(row[name], f"cache_stats[{index}].{name}")
            if counter < 0:
                raise ValueError(f"cache_stats[{index}].{name} must be non-negative")
        output.append(row)
    return tuple(output)


def _validate_manifest_environment(manifest: Mapping[str, Any]) -> None:
    versions = _object(
        manifest["software_versions"],
        {"python", "pyosv", "numpy", "scipy"},
        "manifest software_versions",
    )
    statuses = _object(
        manifest["software_version_status"],
        {"python", "pyosv", "numpy", "scipy"},
        "manifest software_version_status",
    )
    for name in versions:
        if statuses[name] not in {"available", "not_available"}:
            raise ValueError(f"invalid software version status for {name}")
        if statuses[name] == "available":
            _string(versions[name], f"software_versions.{name}")
        elif versions[name] is not None:
            raise ValueError(f"unavailable software version {name} must be null")

    provenance = _object(
        manifest["source_provenance"],
        {"status", "method", "commit", "dirty"},
        "manifest source_provenance",
    )
    if provenance["method"] != "git_cli":
        raise ValueError("invalid source provenance method")
    if provenance["status"] == "available":
        commit = provenance["commit"]
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ValueError("invalid source provenance commit")
        _boolean(provenance["dirty"], "source_provenance.dirty")
    elif provenance["status"] == "not_available":
        if provenance["commit"] is not None or provenance["dirty"] is not None:
            raise ValueError("unavailable source provenance fields must be null")
    else:
        raise ValueError("invalid source provenance status")


def _build_manifest(
    result: SyntheticModeComparisonResult,
    config: SyntheticModeComparisonConfig,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    plan = _json_value(result.plan_metadata)
    trials = _json_value(result.trial_metadata)
    if not isinstance(plan, dict) or not isinstance(trials, list):
        raise ValueError("result metadata must be JSON objects")
    cells = plan.get("cells")
    if not isinstance(cells, list):
        raise ValueError("result plan metadata must contain ordered cells")
    scanner = plan.get("scanner_template")
    if not isinstance(scanner, dict) or not isinstance(scanner.get("input_config"), dict):
        raise ValueError("result plan metadata must contain resolved scanner input settings")
    scanner_input_seed = scanner["input_config"].get("seed")

    case_stochastic = {
        trial.get("case_id"): trial.get("seed") is not None
        for trial in trials
        if isinstance(trial, dict)
    }
    trial_records = []
    for order, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ValueError("trial metadata entries must be objects")
        case_id = trial.get("case_id")
        trial_records.append(
            {
                "order": order,
                "case_id": case_id,
                "trial_id": trial.get("trial_id"),
                "stochastic": case_stochastic.get(case_id, False),
                "case_generation_seed": trial.get("seed"),
                "scanner_input_seed": scanner_input_seed,
            }
        )

    software_versions, software_version_status = _software_versions()
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "scalar_evidence_contract_version": SCALAR_EVIDENCE_CONTRACT_VERSION,
        "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "canonical_cells": [
            {
                "order": order,
                "label": cell.get("label"),
                "scope": cell.get("scope"),
            }
            for order, cell in enumerate(cells)
            if isinstance(cell, dict)
        ],
        "input_config": _json_value(asdict(config)),
        "resolved_plan": plan,
        "case_order": list(plan.get("case_ids", [])),
        "cases": [
            {
                "order": order,
                "case_id": case_id,
                "stochastic": case_stochastic.get(case_id, False),
            }
            for order, case_id in enumerate(plan.get("case_ids", []))
        ],
        "trials": trial_records,
        "shape": plan.get("shape"),
        "variant": plan.get("comparison_variant"),
        "oracle_workflow_isolation": plan.get("include_oracle_workflow_isolation"),
        "metric_registry": {
            "id": METRIC_REGISTRY_ID,
            "definition_version": METRIC_REGISTRY_DEFINITION_VERSION,
        },
        "contrast_definition": {
            "id": CONTRAST_DEFINITION_ID,
            "formula_version": CONTRAST_FORMULA_VERSION,
        },
        "cache_stats": _json_value(result.cache_stats),
        "software_versions": software_versions,
        "software_version_status": software_version_status,
        "source_provenance": _json_value(source),
    }


def _source_provenance() -> dict[str, Any]:
    unavailable = {
        "status": "not_available",
        "method": "git_cli",
        "commit": None,
        "dirty": None,
    }
    try:
        source_file = Path(__file__).resolve()
        root_text = subprocess.run(
            ["git", "-C", str(source_file.parent), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        if not root_text:
            return unavailable
        root = Path(root_text).resolve()
        source_relative = source_file.relative_to(root).as_posix()
        tracked_source = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", source_relative],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()
        if tracked_source != [source_relative]:
            return unavailable
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            return unavailable
    except Exception:
        return unavailable
    return {
        "status": "available",
        "method": "git_cli",
        "commit": commit,
        "dirty": bool(status),
    }


def _software_versions() -> tuple[dict[str, str | None], dict[str, str]]:
    getters = {
        "python": platform.python_version,
        "pyosv": lambda: pyosv.__version__,
        "numpy": lambda: np.__version__,
        "scipy": lambda: scipy.__version__,
    }
    versions: dict[str, str | None] = {}
    status: dict[str, str] = {}
    for name, getter in getters.items():
        try:
            value = getter()
            if not isinstance(value, str) or not value:
                raise ValueError("version must be a non-empty string")
        except Exception:
            versions[name] = None
            status[name] = "not_available"
        else:
            versions[name] = value
            status[name] = "available"
    return versions, status


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
    normalized = _json_value(value)
    options: dict[str, Any] = {"allow_nan": False, "ensure_ascii": False}
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return (json.dumps(normalized, **options) + "\n").encode("utf-8")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("artifact JSON values must be finite")
        return value
    if isinstance(value, np.generic) or isinstance(value, np.ndarray):
        raise ValueError("artifact JSON must not implicitly convert NumPy objects")
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("artifact JSON object keys must be strings")
            output[key] = _json_value(item)
        return output
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise ValueError(f"artifact value is not JSON-safe: {type(value).__name__}")


def _exact_json_value(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's bool/int or int/float coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact_json_value(actual[name], value) for name, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact_json_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _csv_bytes(rows: Sequence[Any], model: type[Any]) -> bytes:
    field_names = tuple(field.name for field in fields(model))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(field_names)
    for row in rows:
        if not isinstance(row, model):
            raise ValueError(f"CSV rows must contain only {model.__name__} values")
        mapping = row.as_dict()
        writer.writerow(_csv_value(mapping[name]) for name in field_names)
    return stream.getvalue().encode("utf-8")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("artifact CSV values must be finite")
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list, Mapping)):
        return json.dumps(
            _json_value(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    raise ValueError(f"artifact CSV value has unsupported type {type(value).__name__}")


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        written = stream.write(payload)
        if written != len(payload):
            raise OSError(f"short artifact write for {path.name}")
        stream.flush()
        os.fsync(stream.fileno())


def _file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {"sha256": digest.hexdigest(), "size": size}


def _finalize_bundle(temporary_path: Path, final_path: Path) -> None:
    _rename_noreplace(temporary_path, final_path)


def _rename_noreplace(source: Path, destination: Path) -> None:
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        try:
            renameat2 = library.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace rename is not supported",
                destination,
            ) from error
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,  # AT_FDCWD
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                os.strerror(error_number),
                destination,
            )
        return

    if sys.platform == "darwin":
        library = ctypes.CDLL(None, use_errno=True)
        renamex_np = library.renamex_np
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        result = renamex_np(
            os.fsencode(source),
            os.fsencode(destination),
            4,  # RENAME_EXCL
        )
        if result != 0:
            error_number = ctypes.get_errno()
            raise OSError(
                error_number,
                os.strerror(error_number),
                destination,
            )
        return

    if os.name == "nt":
        os.rename(source, destination)
        return

    raise OSError(
        errno.ENOTSUP,
        "atomic no-replace rename is not supported",
        destination,
    )


def _cleanup_path(path: Path, original_error: BaseException) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except BaseException as cleanup_error:
        add_note = getattr(original_error, "add_note", None)
        if add_note is not None:
            add_note(f"artifact cleanup also failed for {path}: {cleanup_error!r}")


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> Any:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"malformed JSON artifact: {path.name}") from error
    return _read_json_bytes(payload, path.name)


def _read_json_bytes(payload: bytes, filename: str) -> Any:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda constant: _raise_nonfinite_json(constant),
            object_pairs_hook=_json_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed JSON artifact: {filename}") from error
    return _json_value(value)


def _raise_nonfinite_json(constant: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {constant}")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, value in pairs:
        if name in output:
            raise ValueError(f"duplicate JSON object field: {name}")
        output[name] = value
    return output


def _load_csv(payload: bytes, filename: str) -> tuple[Any, ...]:
    try:
        text = payload.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except (UnicodeError, csv.Error) as error:
        raise ValueError(f"malformed CSV artifact: {filename}") from error
    header = [field.name for field in fields(_CSV_MODELS[filename])]
    if not rows or rows[0] != header:
        raise ValueError(f"invalid CSV header for {filename}")
    output = []
    for row in rows[1:]:
        if len(row) != len(header):
            raise ValueError(f"malformed CSV row in {filename}")
        values = dict(zip(header, row, strict=True))
        typed: dict[str, Any] = {}
        for name, raw_value in values.items():
            nullable_numeric = name in _CSV_NULLABLE_NUMERIC_FIELDS[filename]
            if raw_value == "" and nullable_numeric:
                typed[name] = None
            elif name in _CSV_INTEGER_FIELDS[filename]:
                try:
                    typed[name] = int(raw_value, 10)
                except ValueError as error:
                    raise ValueError(f"invalid integer in {filename}: {name}") from error
                if (
                    filename == METRICS_FILE
                    and name == "schema_version"
                    and typed[name] != METRIC_SCHEMA_VERSION
                ):
                    raise ValueError("unsupported metric schema version in metrics_long.csv")
            elif name in _CSV_FLOAT_FIELDS[filename]:
                try:
                    typed[name] = float(raw_value)
                except ValueError as error:
                    raise ValueError(f"invalid number in {filename}: {name}") from error
                if not np.isfinite(typed[name]):
                    raise ValueError(f"non-finite number in {filename}: {name}")
            elif name in _CSV_BOOLEAN_FIELDS[filename]:
                if raw_value not in {"true", "false"}:
                    raise ValueError(f"invalid boolean in {filename}: {name}")
                typed[name] = raw_value == "true"
            elif name in _CSV_TUPLE_FIELDS.get(filename, set()):
                try:
                    tuple_value = json.loads(raw_value)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid {name} in {filename}") from error
                if not isinstance(tuple_value, list) or any(
                    not isinstance(item, str) for item in tuple_value
                ):
                    raise ValueError(f"invalid {name} in {filename}")
                typed[name] = tuple(tuple_value)
            elif raw_value == "" and name in _CSV_NULLABLE_STRING_FIELDS[filename]:
                typed[name] = None
            else:
                typed[name] = raw_value

            if _csv_value(typed[name]) != raw_value:
                raise ValueError(f"noncanonical value in {filename}: {name}")
        try:
            output.append(_CSV_MODELS[filename](**typed))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid typed row in {filename}: {error}") from error
    return tuple(output)


def _dataclass_object(value: Any, model: type[Any], context: str) -> dict[str, Any]:
    return _object(value, {field.name for field in fields(model)}, context)


def _object(value: Any, expected_fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    actual_fields = set(value)
    if actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        unknown = sorted(actual_fields - expected_fields)
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown: {', '.join(unknown)}")
        raise ValueError(f"invalid fields in {context} ({'; '.join(details)})")
    return value


def _array(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be an array")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _optional_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context)


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a boolean")
    return value


def _integer(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    return value


def _manifest_contract_version(
    manifest: Mapping[str, Any],
    field: str,
    supported_version: int,
    contract_name: str,
    *,
    legacy_messages: Mapping[int, str] | None = None,
) -> int:
    if field not in manifest:
        raise ValueError(f"manifest {field} is required")
    version = _integer(manifest[field], f"manifest {field}")
    if version != supported_version:
        if legacy_messages is not None and version in legacy_messages:
            raise ValueError(legacy_messages[version])
        raise ValueError(f"unsupported {contract_name} contract version")
    return version


def _optional_integer(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _integer(value, context)


def _nonnegative_integer(value: Any, context: str) -> int:
    integer = _integer(value, context)
    if integer < 0:
        raise ValueError(f"{context} must be a non-negative integer")
    return integer


def _optional_nonnegative_integer(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, context)


def _number(value: Any, context: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    if not np.isfinite(value):
        raise ValueError(f"{context} must be a finite number")
    return value


def _optional_number(value: Any, context: str) -> int | float | None:
    if value is None:
        return None
    return _number(value, context)


def _nonnegative_number(value: Any, context: str) -> int | float:
    number = _number(value, context)
    if number < 0:
        raise ValueError(f"{context} must be a finite non-negative number")
    return number


def _optional_nonnegative_number(value: Any, context: str) -> int | float | None:
    if value is None:
        return None
    return _nonnegative_number(value, context)


def _closed_unit_interval_number(value: Any, context: str) -> int | float:
    number = _number(value, context)
    if not 0 <= number <= 1:
        raise ValueError(f"{context} must be a number in the closed unit interval")
    return number


def _optional_closed_unit_interval_number(value: Any, context: str) -> int | float | None:
    if value is None:
        return None
    return _closed_unit_interval_number(value, context)


def _string_tuple(value: Any, context: str) -> tuple[str, ...]:
    items = _array(value, context)
    return tuple(_string(item, f"{context} item") for item in items)


def _integer_tuple(value: Any, context: str) -> tuple[int, ...]:
    items = _array(value, context)
    return tuple(_integer(item, f"{context} item") for item in items)


def _shape(value: Any, context: str) -> tuple[int, int, int]:
    items = _integer_tuple(value, context)
    if len(items) != 3:
        raise ValueError(f"{context} must contain exactly three integers")
    return items


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "COMPLETION_SCHEMA_VERSION",
    "HASHED_BUNDLE_FILES",
    "REQUIRED_BUNDLE_FILES",
    "RUNTIME_CONTRACT_VERSION",
    "SCALAR_EVIDENCE_CONTRACT_VERSION",
    "validate_completed_bundle",
    "write_artifact_bundle",
]
