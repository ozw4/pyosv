"""Completed, self-validating result bundles for canonical F3 comparisons."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from pyosv.candidate_volume import NONZERO_EPSILON, positive_candidate_mask
from pyosv.experimental.boundary_skinning import apply_boundary_skinner_fallback
from pyosv.orient3d import FaultOrientScanner3

from .artifacts import (
    F3_ARTIFACT_SCHEMA_VERSION,
    F3_FINGERPRINT_CONTRACT_VERSION,
    RUN_COMPLETION_FILE,
    RUN_MANIFEST_FILE,
    STAGE_COMPLETION_FILE,
    STAGE_MANIFEST_FILE,
    F3ArtifactError,
    F3RunWorkspace,
    artifact_file_metadata,
    atomic_write_artifact,
    canonical_fingerprint,
    canonical_json_bytes,
    validate_stage,
)
from .data import F3_DATASET_ID, F3DatasetSpec, OFFICIAL_F3_DATASET_SPEC
from .diagnostics import (
    F3_DIAGNOSTIC_REGIONS,
    F3_ORIENTATION_PAIRS,
    F3_REGION_SEMANTICS,
    OrientationDiagnosticRow,
    RegionalDiagnosticRow,
    compute_orientation_pair_diagnostic,
    compute_regional_reference_diagnostics,
)
from .metrics import (
    CONTRAST_DEFINITIONS,
    F3_REFERENCE_STAGE_FILES,
    F3_REFERENCE_STAGE_ROLES,
    METRIC_REGISTRY,
    ContrastRow,
    MetricEvidence,
    MetricRow,
    VoxelwiseContrastSummary,
    compute_contrast_rows,
    compute_reference_metric_rows,
    compute_skin_metric_rows,
    validate_shared_stage_metrics,
)
from .models import F3ScannerBackend, canonical_f3_cells
from .config import F3ScannerConfig
from .resources import (
    F3_RESOURCE_INTERPRETATION,
    F3_RESOURCE_SCHEMA_VERSION,
    RSSSnapshot,
    StageResourceRow,
    StorageRow,
    storage_report,
)
from .runner import (
    F3_CELL_RUNNER_CONTRACT_VERSION,
    F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
    F3_SKINNING_STAGE_IMPLEMENTATION,
    F3_THINNING_STAGE_IMPLEMENTATION,
    F3_VOTING_STAGE_IMPLEMENTATION,
    F3CellReference,
    F3CellStageFingerprints,
    skinning_stage_artifacts,
    thinning_stage_artifacts,
    voting_stage_artifacts,
)
from .scanner import (
    F3_SCANNER_STAGE_CONTRACT_VERSION,
    scanner_array_summary,
    scanner_sampling_count,
    scanner_stage_artifacts,
    scanner_stage_resolved_settings,
)
from .skin_artifacts import (
    ParsedSkinArtifacts,
    SkinArtifactValidationError,
    canonical_skins_payload,
    parse_skins_json,
    resolve_skin_parent_volume_contract,
    validate_skin_artifact_semantics,
)
from .runtime_identity import (
    numerical_runtime_identity,
    validate_numerical_runtime_identity,
    validate_publication_runtime_identity,
)
from ..synthetic_quality.config import SyntheticSkinningConfig, SyntheticVotingConfig
from ..synthetic_quality.stage_keys import (
    build_final_thinning_stage_key,
    build_seed_stage_key,
    build_thinning_stage_key,
    build_voting_stage_key,
)
from ..synthetic_quality.variants import SkinningPatch, VariantSpec, VotingPatch
from ..workflow3d import (
    PreparedAttributeIdentity,
    VolumeVotingControls,
    execute_skinning_phase3d,
)

F3_RESULT_SCHEMA_VERSION = 1
F3_COMPLETION_SCHEMA_VERSION = 1
F3_RESULT_INTERPRETATION = (
    "internal_scalar_and_artifact_consistency_not_a_cryptographic_signature_"
    "or_independent_geological_truth"
)

CELLS_REPORT_FILE = "cells.json"
METRICS_REPORT_FILE = "metrics_long.csv"
METRIC_EVIDENCE_REPORT_FILE = "metric_evidence.json"
CONTRASTS_REPORT_FILE = "contrasts.csv"
VOXEL_CONTRASTS_REPORT_FILE = "voxel_contrast_summaries.csv"
REGIONAL_REPORT_FILE = "regional_metrics.csv"
ORIENTATION_REPORT_FILE = "orientation_diagnostics.csv"
RUNTIME_REPORT_FILE = "runtime.csv"
RESOURCES_REPORT_FILE = "resources.json"

F3_REPORT_FILES = (
    CELLS_REPORT_FILE,
    METRICS_REPORT_FILE,
    METRIC_EVIDENCE_REPORT_FILE,
    CONTRASTS_REPORT_FILE,
    VOXEL_CONTRASTS_REPORT_FILE,
    REGIONAL_REPORT_FILE,
    ORIENTATION_REPORT_FILE,
    RUNTIME_REPORT_FILE,
    RESOURCES_REPORT_FILE,
)

_RUN_COMPUTATION_FIELDS = (
    "artifact_schema_version",
    "stage_contract_version",
    "fingerprint_contract_version",
    "plan",
    "dataset_identity",
    "implementation_identity",
    "runtime_identity",
)
_STAGE_COMPUTATION_FIELDS = (
    "artifact_schema_version",
    "stage_contract_version",
    "kind",
    "run_fingerprint",
    "parent_fingerprints",
    "input_fingerprints",
    "resolved_settings",
    "artifact_schema",
)
_SCANNER_REPORT_FIELDS = {
    "scanner_stage_contract_version",
    "fingerprint",
    "backend",
    "shape",
    "input_fingerprint",
    "resolved_config",
    "resolved_stage_settings",
    "sampling_count",
    "requested_remove_edge_effects",
    "effective_remove_edge_effects",
    "raw",
    "thinned",
}
_SCANNER_SUMMARY_FIELDS = {
    "shape",
    "dtype",
    "finite_count",
    "min",
    "max",
    "mean",
    "nonzero_epsilon",
    "nonzero_count",
    "nonzero_fraction",
}
_SCANNER_THINNED_NAMES = {"fet", "fpt", "ftt"}
_CELL_ORDER = tuple(cell.label for cell in canonical_f3_cells())
_CELL_AXES = {
    cell.label: (cell.scanner_backend, cell.workflow_mode) for cell in canonical_f3_cells()
}
_REFERENCE_ROLE_BY_FILE = {
    filename: F3_REFERENCE_STAGE_ROLES[stage]
    for stage, filename in F3_REFERENCE_STAGE_FILES.items()
}
_CSV_MODELS = {
    METRICS_REPORT_FILE: MetricRow,
    CONTRASTS_REPORT_FILE: ContrastRow,
    VOXEL_CONTRASTS_REPORT_FILE: VoxelwiseContrastSummary,
    REGIONAL_REPORT_FILE: RegionalDiagnosticRow,
    ORIENTATION_REPORT_FILE: OrientationDiagnosticRow,
    RUNTIME_REPORT_FILE: StageResourceRow,
}
_SHA256_LENGTH = 64
_SKIN_RECOMPUTE_CHUNK_VOXELS = 1_000_000
_FLOAT_REL_TOL = 1.0e-9
_FLOAT_ABS_TOL = 1.0e-12
_MISSING = object()
_ORIENTATION_STAGE_FILES = {
    "scanner": ("scanner", "scanner", ("ft.dat", "pt.dat", "tt.dat")),
    "voting": ("voting", "voting", ("fv.dat", "vp.dat", "vt.dat")),
}


class F3ResultValidationError(F3ArtifactError):
    """Raised when a completed result is structurally or semantically invalid."""


@dataclass(frozen=True, slots=True)
class F3ModeComparisonResult:
    """Scalar/reference-only result for one complete F3 evaluation unit."""

    run_fingerprint: str
    dataset_id: str
    volume_shape: tuple[int, int, int]
    storage_dtype: str
    cells: tuple[F3CellReference, ...]
    metric_rows: tuple[MetricRow, ...]
    metric_evidence: tuple[MetricEvidence, ...]
    contrast_rows: tuple[ContrastRow, ...]
    voxelwise_contrasts: tuple[VoxelwiseContrastSummary, ...]
    regional_rows: tuple[RegionalDiagnosticRow, ...]
    orientation_rows: tuple[OrientationDiagnosticRow, ...]
    runtime_rows: tuple[StageResourceRow, ...]
    rss_snapshots: tuple[RSSSnapshot, ...]
    storage_rows: tuple[StorageRow, ...]
    resource_interpretation: str = F3_RESOURCE_INTERPRETATION
    interpretation: str = F3_RESULT_INTERPRETATION

    def __post_init__(self) -> None:
        _sha256(self.run_fingerprint, "run_fingerprint")
        if not isinstance(self.dataset_id, str) or not self.dataset_id:
            raise ValueError("dataset_id must be a non-empty string")
        shape = _shape3(self.volume_shape)
        object.__setattr__(self, "volume_shape", shape)
        try:
            dtype = np.dtype(self.storage_dtype)
        except TypeError as error:
            raise ValueError("storage_dtype must be a valid dtype") from error
        if dtype.str != ">f4":
            raise ValueError("storage_dtype must be big-endian float32")
        object.__setattr__(self, "storage_dtype", dtype.str)
        for name, item_type in (
            ("cells", F3CellReference),
            ("metric_rows", MetricRow),
            ("metric_evidence", MetricEvidence),
            ("contrast_rows", ContrastRow),
            ("voxelwise_contrasts", VoxelwiseContrastSummary),
            ("regional_rows", RegionalDiagnosticRow),
            ("orientation_rows", OrientationDiagnosticRow),
            ("runtime_rows", StageResourceRow),
            ("rss_snapshots", RSSSnapshot),
            ("storage_rows", StorageRow),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, item_type) for value in values
            ):
                raise ValueError(f"{name} must be a tuple of {item_type.__name__} values")
        if self.resource_interpretation != F3_RESOURCE_INTERPRETATION:
            raise ValueError("resource_interpretation is not canonical")
        if self.interpretation != F3_RESULT_INTERPRETATION:
            raise ValueError("interpretation is not canonical")
        _reject_volume_bearing_values(self)

    @classmethod
    def from_extractions(
        cls,
        *,
        workspace: F3RunWorkspace,
        cells: Sequence[F3CellReference],
        metrics: Any,
        diagnostics: Any,
        resources: Any,
        _dataset_spec: F3DatasetSpec | None = None,
    ) -> F3ModeComparisonResult:
        """Build the result from the stable extraction objects."""

        dataset = _dataset_contract(workspace.manifest, _dataset_spec)
        return cls(
            run_fingerprint=workspace.fingerprint,
            dataset_id=dataset["dataset_id"],
            volume_shape=dataset["shape"],
            storage_dtype=dataset["storage_dtype"],
            cells=tuple(cells),
            metric_rows=tuple(metrics.metric_rows),
            metric_evidence=tuple(metrics.metric_evidence),
            contrast_rows=tuple(metrics.contrast_rows),
            voxelwise_contrasts=tuple(metrics.voxelwise_contrasts),
            regional_rows=tuple(diagnostics.regional_rows),
            orientation_rows=tuple(diagnostics.orientation_rows),
            runtime_rows=tuple(resources.stage_rows),
            rss_snapshots=tuple(resources.rss_snapshots),
            storage_rows=tuple(resources.storage_rows),
            resource_interpretation=resources.interpretation,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe scalar representation."""

        return {
            "result_schema_version": F3_RESULT_SCHEMA_VERSION,
            "run_fingerprint": self.run_fingerprint,
            "dataset_id": self.dataset_id,
            "volume_shape": list(self.volume_shape),
            "storage_dtype": self.storage_dtype,
            "cells": [cell.as_dict() for cell in self.cells],
            "metric_rows": [row.as_dict() for row in self.metric_rows],
            "metric_evidence": [row.as_dict() for row in self.metric_evidence],
            "contrast_rows": [row.as_dict() for row in self.contrast_rows],
            "voxelwise_contrasts": [row.as_dict() for row in self.voxelwise_contrasts],
            "regional_rows": [row.as_dict() for row in self.regional_rows],
            "orientation_rows": [row.as_dict() for row in self.orientation_rows],
            "runtime_rows": [row.as_dict() for row in self.runtime_rows],
            "rss_snapshots": [row.as_dict() for row in self.rss_snapshots],
            "storage_rows": [row.as_dict() for row in self.storage_rows],
            "resource_interpretation": self.resource_interpretation,
            "interpretation": self.interpretation,
        }


def finalize_f3d_bundle(
    workspace: F3RunWorkspace | str | os.PathLike[str],
    result: F3ModeComparisonResult | None = None,
    *,
    resume: bool = False,
    deep: bool = False,
    pretty: bool = False,
    _dataset_spec: F3DatasetSpec | None = None,
) -> F3ModeComparisonResult:
    """Write all reports and publish root completion as the final operation."""

    if not isinstance(resume, bool) or not isinstance(deep, bool) or not isinstance(pretty, bool):
        raise TypeError("resume, deep, and pretty must be bool")
    root = _workspace_path(workspace)
    completion_path = root / RUN_COMPLETION_FILE
    if completion_path.exists() or completion_path.is_symlink():
        if not resume:
            raise FileExistsError(f"completed F3 bundle already exists: {root}")
        return load_f3d_mode_comparison_result(
            root,
            deep=deep,
            _dataset_spec=_dataset_spec,
        )
    if result is None:
        raise F3ResultValidationError("incomplete workspace has no result to finalize")

    validate_f3d_mode_comparison_result(
        root,
        result,
        deep=False,
        _dataset_spec=_dataset_spec,
    )
    report_payloads = _serialize_reports(result)
    reports = root / "reports"
    _require_directory(reports, "reports directory")
    _cleanup_report_temporaries(reports)
    for item in root.glob(".completion.json.tmp-*"):
        if item.is_file() and not item.is_symlink():
            item.unlink()
    try:
        for filename in F3_REPORT_FILES:
            atomic_write_artifact(
                reports / filename,
                report_payloads[filename],
                temporary_prefix=f".{filename}.tmp-",
            )
        loaded = _load_reports(root)
        validate_f3d_mode_comparison_result(
            root,
            loaded,
            deep=deep,
            _dataset_spec=_dataset_spec,
        )
        report_metadata = {
            filename: artifact_file_metadata(reports / filename) for filename in F3_REPORT_FILES
        }
        stage_metadata = _referenced_stage_completion_metadata(root, loaded.cells)
        completion = {
            "completion_schema_version": F3_COMPLETION_SCHEMA_VERSION,
            "artifact_schema_version": F3_ARTIFACT_SCHEMA_VERSION,
            "result_schema_version": F3_RESULT_SCHEMA_VERSION,
            "status": "complete",
            "run_fingerprint": loaded.run_fingerprint,
            "report_files": report_metadata,
            "stage_completions": stage_metadata,
            "interpretation": F3_RESULT_INTERPRETATION,
        }
        atomic_write_artifact(
            completion_path,
            _completion_json_bytes(completion, pretty=pretty),
            temporary_prefix=".completion.json.tmp-",
        )
        validate_completed_f3d_bundle(
            root,
            deep=deep,
            _dataset_spec=_dataset_spec,
        )
    except BaseException:
        completion_path.unlink(missing_ok=True)
        _cleanup_report_temporaries(reports)
        raise
    return loaded


def load_f3d_mode_comparison_result(
    path: str | os.PathLike[str],
    *,
    deep: bool = False,
    _dataset_spec: F3DatasetSpec | None = None,
) -> F3ModeComparisonResult:
    """Strictly load a completed bundle after all hashes and semantics pass."""

    validate_completed_f3d_bundle(path, deep=deep, _dataset_spec=_dataset_spec)
    return _load_reports(Path(path))


def validate_completed_f3d_bundle(
    path: str | os.PathLike[str],
    deep: bool = False,
    *,
    _dataset_spec: F3DatasetSpec | None = None,
) -> bool:
    """Validate completion, reports, stages, and cross-file semantic relations."""

    if not isinstance(deep, bool):
        raise TypeError("deep must be bool")
    root = Path(path)
    _require_directory(root, "F3 bundle")
    if deep and _dataset_spec is None and _manifest_is_official(root):
        _validate_current_publication_runtime(root)
    completion = _read_json_object(root / RUN_COMPLETION_FILE, "completion.json")
    expected_fields = {
        "completion_schema_version",
        "artifact_schema_version",
        "result_schema_version",
        "status",
        "run_fingerprint",
        "report_files",
        "stage_completions",
        "interpretation",
    }
    if set(completion) != expected_fields:
        raise F3ResultValidationError("completion field set mismatch")
    if completion["completion_schema_version"] != F3_COMPLETION_SCHEMA_VERSION:
        raise F3ResultValidationError("unsupported completion schema version")
    if completion["artifact_schema_version"] != F3_ARTIFACT_SCHEMA_VERSION:
        raise F3ResultValidationError("completion artifact schema mismatch")
    if completion["result_schema_version"] != F3_RESULT_SCHEMA_VERSION:
        raise F3ResultValidationError("completion result schema mismatch")
    if completion["status"] != "complete":
        raise F3ResultValidationError("completion status must be 'complete'")
    if completion["interpretation"] != F3_RESULT_INTERPRETATION:
        raise F3ResultValidationError("completion interpretation mismatch")

    reports = root / "reports"
    _require_directory(reports, "reports directory")
    report_entries = {item.name for item in reports.iterdir()}
    if report_entries != set(F3_REPORT_FILES):
        raise F3ResultValidationError("report file set mismatch")
    report_metadata = completion["report_files"]
    if not isinstance(report_metadata, dict) or set(report_metadata) != set(F3_REPORT_FILES):
        raise F3ResultValidationError("completion report metadata mismatch")
    for filename in F3_REPORT_FILES:
        _verify_file_metadata(reports / filename, report_metadata[filename], filename)

    manifest = _validated_run_manifest(root)
    _reject_crop_semantics(manifest)
    if completion["run_fingerprint"] != manifest["run_fingerprint"]:
        raise F3ResultValidationError("completion run fingerprint mismatch")
    result = _load_reports(root)
    if result.run_fingerprint != completion["run_fingerprint"]:
        raise F3ResultValidationError("result run fingerprint mismatch")
    validate_f3d_mode_comparison_result(
        root,
        result,
        deep=deep,
        _dataset_spec=_dataset_spec,
    )
    actual_stages = _referenced_stage_completion_metadata(root, result.cells)
    if completion["stage_completions"] != actual_stages:
        raise F3ResultValidationError("referenced stage completion digest mismatch")
    return True


def validate_f3d_mode_comparison_result(
    workspace: F3RunWorkspace | str | os.PathLike[str],
    result: F3ModeComparisonResult,
    *,
    deep: bool = False,
    _dataset_spec: F3DatasetSpec | None = None,
) -> bool:
    """Validate one in-memory result against its immutable run workspace."""

    if not isinstance(result, F3ModeComparisonResult):
        raise TypeError("result must be an F3ModeComparisonResult")
    if not isinstance(deep, bool):
        raise TypeError("deep must be bool")
    if _dataset_spec is not None and not isinstance(_dataset_spec, F3DatasetSpec):
        raise TypeError("_dataset_spec must be an F3DatasetSpec or None")
    root = _workspace_path(workspace)
    manifest = _validated_run_manifest(root)
    if (
        deep
        and _dataset_spec is None
        and manifest["dataset_identity"].get("dataset_id") == F3_DATASET_ID
    ):
        _validate_current_publication_runtime(root)
    dataset = _dataset_contract(
        manifest,
        OFFICIAL_F3_DATASET_SPEC if _dataset_spec is None else _dataset_spec,
    )
    if (
        result.run_fingerprint != manifest["run_fingerprint"]
        or result.dataset_id != dataset["dataset_id"]
        or result.volume_shape != dataset["shape"]
        or result.storage_dtype != dataset["storage_dtype"]
    ):
        raise F3ResultValidationError("result identity does not match the run manifest")
    _reject_crop_semantics(result.as_dict())
    parsed_skins = _validate_cells_and_stages(root, result, dataset, manifest["plan"])
    _validate_metrics(result, dataset)
    _validate_voxelwise_contrasts(result)
    _validate_regional_rows(result, manifest["plan"])
    _validate_orientation_rows(result)
    _validate_resource_rows(root, result)
    if deep:
        _deep_validate_scanner_stages(root, result, manifest["plan"])
        _deep_validate_skin_artifacts(root, result, parsed_skins)
        _deep_validate_reference_metrics(root, result, dataset)
        _deep_validate_voxelwise_contrasts(root, result)
        _deep_validate_orientation_diagnostics(root, result)
    return True


# Public spellings kept close to the issue terminology.
finalize_f3d_mode_comparison_result = finalize_f3d_bundle
load_f3d_result_bundle = load_f3d_mode_comparison_result
validate_f3d_result = validate_f3d_mode_comparison_result


def _completion_json_bytes(value: Mapping[str, Any], *, pretty: bool) -> bytes:
    if not pretty:
        return canonical_json_bytes(value) + b"\n"
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _serialize_reports(result: F3ModeComparisonResult) -> dict[str, bytes]:
    identity = _result_identity(result)
    cells = {
        **identity,
        "cells": [cell.as_dict() for cell in result.cells],
        "interpretation": result.interpretation,
    }
    evidence = {
        **identity,
        "metric_evidence": [row.as_dict() for row in result.metric_evidence],
    }
    resources = {
        **identity,
        "resource_schema_version": F3_RESOURCE_SCHEMA_VERSION,
        "interpretation": result.resource_interpretation,
        "rss": [row.as_dict() for row in result.rss_snapshots],
        "storage": [row.as_dict() for row in result.storage_rows],
    }
    return {
        CELLS_REPORT_FILE: canonical_json_bytes(cells) + b"\n",
        METRICS_REPORT_FILE: _csv_bytes(result.metric_rows, MetricRow),
        METRIC_EVIDENCE_REPORT_FILE: canonical_json_bytes(evidence) + b"\n",
        CONTRASTS_REPORT_FILE: _csv_bytes(result.contrast_rows, ContrastRow),
        VOXEL_CONTRASTS_REPORT_FILE: _csv_bytes(
            result.voxelwise_contrasts, VoxelwiseContrastSummary
        ),
        REGIONAL_REPORT_FILE: _csv_bytes(result.regional_rows, RegionalDiagnosticRow),
        ORIENTATION_REPORT_FILE: _csv_bytes(result.orientation_rows, OrientationDiagnosticRow),
        RUNTIME_REPORT_FILE: _csv_bytes(result.runtime_rows, StageResourceRow),
        RESOURCES_REPORT_FILE: canonical_json_bytes(resources) + b"\n",
    }


def _load_reports(root: Path) -> F3ModeComparisonResult:
    reports = root / "reports"
    cells_payload = _read_json_object(reports / CELLS_REPORT_FILE, CELLS_REPORT_FILE)
    evidence_payload = _read_json_object(
        reports / METRIC_EVIDENCE_REPORT_FILE, METRIC_EVIDENCE_REPORT_FILE
    )
    resources_payload = _read_json_object(reports / RESOURCES_REPORT_FILE, RESOURCES_REPORT_FILE)
    identity = _parse_result_identity(cells_payload, CELLS_REPORT_FILE)
    if _parse_result_identity(evidence_payload, METRIC_EVIDENCE_REPORT_FILE) != identity:
        raise F3ResultValidationError("metric evidence identity mismatch")
    if _parse_result_identity(resources_payload, RESOURCES_REPORT_FILE) != identity:
        raise F3ResultValidationError("resource identity mismatch")
    if set(cells_payload) != {
        "result_schema_version",
        "run_fingerprint",
        "dataset_id",
        "volume_shape",
        "storage_dtype",
        "cells",
        "interpretation",
    }:
        raise F3ResultValidationError("cells.json field set mismatch")
    if cells_payload["interpretation"] != F3_RESULT_INTERPRETATION:
        raise F3ResultValidationError("cells.json interpretation mismatch")
    if set(evidence_payload) != {
        "result_schema_version",
        "run_fingerprint",
        "dataset_id",
        "volume_shape",
        "storage_dtype",
        "metric_evidence",
    }:
        raise F3ResultValidationError("metric_evidence.json field set mismatch")
    if set(resources_payload) != {
        "result_schema_version",
        "run_fingerprint",
        "dataset_id",
        "volume_shape",
        "storage_dtype",
        "resource_schema_version",
        "interpretation",
        "rss",
        "storage",
    }:
        raise F3ResultValidationError("resources.json field set mismatch")
    if (
        resources_payload["resource_schema_version"] != F3_RESOURCE_SCHEMA_VERSION
        or resources_payload["interpretation"] != F3_RESOURCE_INTERPRETATION
    ):
        raise F3ResultValidationError("resources.json contract mismatch")

    cells = tuple(
        _cell_from_dict(root, item) for item in _list(cells_payload["cells"], "cells.json.cells")
    )
    metric_rows = tuple(
        _metric_row(item) for item in _read_csv(reports / METRICS_REPORT_FILE, MetricRow)
    )
    metric_evidence = tuple(
        _metric_evidence(item)
        for item in _list(
            evidence_payload["metric_evidence"],
            "metric_evidence.json.metric_evidence",
        )
    )
    contrast_rows = tuple(
        _contrast_row(item) for item in _read_csv(reports / CONTRASTS_REPORT_FILE, ContrastRow)
    )
    voxelwise = tuple(
        _voxelwise_row(item)
        for item in _read_csv(
            reports / VOXEL_CONTRASTS_REPORT_FILE,
            VoxelwiseContrastSummary,
        )
    )
    regional = tuple(
        _regional_row(item)
        for item in _read_csv(reports / REGIONAL_REPORT_FILE, RegionalDiagnosticRow)
    )
    orientation = tuple(
        _orientation_row(item)
        for item in _read_csv(
            reports / ORIENTATION_REPORT_FILE,
            OrientationDiagnosticRow,
        )
    )
    runtime = tuple(
        _runtime_row(item) for item in _read_csv(reports / RUNTIME_REPORT_FILE, StageResourceRow)
    )
    rss = tuple(_rss_row(item) for item in _list(resources_payload["rss"], "resources.rss"))
    storage = tuple(
        _storage_row(item) for item in _list(resources_payload["storage"], "resources.storage")
    )
    return F3ModeComparisonResult(
        identity["run_fingerprint"],
        identity["dataset_id"],
        identity["volume_shape"],
        identity["storage_dtype"],
        cells,
        metric_rows,
        metric_evidence,
        contrast_rows,
        voxelwise,
        regional,
        orientation,
        runtime,
        rss,
        storage,
    )


def _validated_run_manifest(root: Path) -> dict[str, Any]:
    manifest = _read_json_object(root / RUN_MANIFEST_FILE, RUN_MANIFEST_FILE)
    if set(manifest) != {*_RUN_COMPUTATION_FIELDS, "run_fingerprint", "provenance"}:
        raise F3ResultValidationError("run manifest field set mismatch")
    computation = {name: manifest[name] for name in _RUN_COMPUTATION_FIELDS}
    fingerprint = _sha256(manifest["run_fingerprint"], "run manifest fingerprint")
    if canonical_fingerprint(computation) != fingerprint:
        raise F3ResultValidationError("run manifest fingerprint mismatch")
    if manifest["fingerprint_contract_version"] != F3_FINGERPRINT_CONTRACT_VERSION:
        raise F3ResultValidationError("run manifest fingerprint contract mismatch")
    try:
        validate_numerical_runtime_identity(manifest["runtime_identity"])
    except ValueError as error:
        raise F3ResultValidationError("run manifest runtime identity is invalid") from error
    return manifest


def _validate_current_publication_runtime(root: Path) -> None:
    """Reject deep recomputation before reading any numerical stage artifact."""

    manifest = _read_json_object(root / RUN_MANIFEST_FILE, RUN_MANIFEST_FILE)
    recorded = manifest.get("runtime_identity")
    try:
        normalized_recorded = validate_numerical_runtime_identity(recorded)
        current = validate_publication_runtime_identity(numerical_runtime_identity())
    except ValueError as error:
        raise F3ResultValidationError(
            "deep validation publication runtime contract mismatch"
        ) from error
    if current != normalized_recorded:
        raise F3ResultValidationError(
            "deep validation current runtime identity does not match run manifest"
        )


def _manifest_is_official(root: Path) -> bool:
    manifest = _read_json_object(root / RUN_MANIFEST_FILE, RUN_MANIFEST_FILE)
    dataset = manifest.get("dataset_identity")
    return isinstance(dataset, Mapping) and dataset.get("dataset_id") == F3_DATASET_ID


def _dataset_contract(
    manifest: Mapping[str, Any],
    expected_spec: F3DatasetSpec | None = None,
) -> dict[str, Any]:
    if expected_spec is None:
        expected_spec = OFFICIAL_F3_DATASET_SPEC
    if not isinstance(expected_spec, F3DatasetSpec):
        raise TypeError("expected_spec must be an F3DatasetSpec")
    identity = manifest.get("dataset_identity")
    if not isinstance(identity, Mapping) or set(identity) != {"dataset_id", "files"}:
        raise F3ResultValidationError("run manifest dataset identity is invalid")
    dataset_id = identity["dataset_id"]
    if not isinstance(dataset_id, str) or not dataset_id:
        raise F3ResultValidationError("dataset ID is invalid")
    files = _list(identity["files"], "dataset identity files")
    if not files:
        raise F3ResultValidationError("dataset identity has no files")
    roles: list[str] = []
    layout: tuple[tuple[int, int, int], str, int] | None = None
    by_role: dict[str, Mapping[str, Any]] = {}
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {
            "role",
            "size",
            "sha256",
            "shape",
            "storage_dtype",
        }:
            raise F3ResultValidationError("dataset file identity is invalid")
        role = item["role"]
        if not isinstance(role, str) or not role or role in by_role:
            raise F3ResultValidationError("dataset file roles are invalid")
        shape = _shape3_list(item["shape"])
        try:
            dtype = np.dtype(item["storage_dtype"]).str
        except TypeError as error:
            raise F3ResultValidationError("dataset storage dtype is invalid") from error
        size = item["size"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size != int(np.prod(shape)) * np.dtype(dtype).itemsize
        ):
            raise F3ResultValidationError("dataset file size/layout mismatch")
        if dtype != ">f4":
            raise F3ResultValidationError("dataset storage dtype must be big-endian float32")
        _sha256(item["sha256"], f"dataset file {role} digest")
        current = (shape, dtype, size)
        if layout is None:
            layout = current
        elif layout != current:
            raise F3ResultValidationError("dataset files have mixed layouts")
        roles.append(role)
        by_role[role] = item
    assert layout is not None
    plan = manifest.get("plan")
    if isinstance(plan, Mapping) and "dataset_spec" in plan:
        spec = plan["dataset_spec"]
        if not isinstance(spec, Mapping):
            raise F3ResultValidationError("plan dataset spec is invalid")
        if (
            spec.get("dataset_id") != dataset_id
            or tuple(spec.get("shape", ())) != layout[0]
            or np.dtype(spec.get("storage_dtype", "")).str != layout[1]
        ):
            raise F3ResultValidationError("plan and dataset identity layout mismatch")
    official = expected_spec == OFFICIAL_F3_DATASET_SPEC
    if dataset_id != expected_spec.dataset_id:
        message = (
            "dataset ID must be the official F3 dataset ID"
            if official
            else "dataset ID does not match the injected dataset spec"
        )
        raise F3ResultValidationError(message)
    if layout[0] != expected_spec.shape:
        message = (
            f"dataset shape must be the official F3 shape {expected_spec.shape}"
            if official
            else "dataset shape does not match the injected dataset spec"
        )
        raise F3ResultValidationError(message)
    if layout[1] != expected_spec.storage_dtype:
        message = (
            f"dataset dtype must be the official F3 dtype {expected_spec.storage_dtype}"
            if official
            else "dataset dtype does not match the injected dataset spec"
        )
        raise F3ResultValidationError(message)
    if tuple(roles) != expected_spec.roles:
        message = (
            "official dataset role coverage mismatch"
            if official
            else "dataset role coverage does not match the injected dataset spec"
        )
        raise F3ResultValidationError(message)
    return {
        "dataset_id": dataset_id,
        "shape": layout[0],
        "storage_dtype": layout[1],
        "files": MappingProxyType(by_role),
    }


def _validate_cells_and_stages(
    root: Path,
    result: F3ModeComparisonResult,
    dataset: Mapping[str, Any],
    plan: Any,
) -> dict[str, ParsedSkinArtifacts]:
    if tuple(cell.label for cell in result.cells) != _CELL_ORDER:
        raise F3ResultValidationError("cells must have exact canonical coverage and order")
    cell_dir = root / "cells"
    _require_directory(cell_dir, "cells directory")
    if {item.name for item in cell_dir.iterdir()} != {f"{label}.json" for label in _CELL_ORDER}:
        raise F3ResultValidationError("cell reference file set mismatch")
    input_file = dataset["files"].get("input")
    input_digest = input_file["sha256"] if input_file is not None else None
    validated: dict[tuple[str, str], Mapping[str, Any]] = {}
    scanner_report_backends: dict[str, str] = {}
    for cell in result.cells:
        scanner_report_backend = scanner_report_backends.setdefault(
            cell.stages.scanner,
            cell.backend,
        )
        if scanner_report_backend != cell.backend:
            raise F3ResultValidationError("scanner stage backend reuse mismatch")
    validated_scanner_reports: set[str] = set()
    parsed_skins: dict[str, ParsedSkinArtifacts] = {}
    workflow_identity: Any = _MISSING
    for cell in result.cells:
        expected_axes = _CELL_AXES[cell.label]
        if (cell.backend, cell.workflow) != expected_axes:
            raise F3ResultValidationError("cell axes mismatch")
        expected_config, skinning_enabled = _resolved_cell_contract(plan, cell.workflow)
        if (
            dict(cell.resolved_config) != expected_config
            or cell.skinning_enabled is not skinning_enabled
        ):
            raise F3ResultValidationError("cell resolved_config does not match the run plan")
        expected_path = cell_dir / f"{cell.label}.json"
        if cell.path.absolute() != expected_path.absolute():
            raise F3ResultValidationError("cell reference path mismatch")
        if _read_json_object(expected_path, expected_path.name) != cell.as_dict():
            raise F3ResultValidationError(f"cell reference mismatch: {cell.label}")
        chain = (
            ("scanner", cell.stages.scanner, (), {"ep.dat": input_digest}),
            ("voting", cell.stages.voting, (cell.stages.scanner,), {}),
            ("thinning", cell.stages.thinning, (cell.stages.voting,), {}),
        )
        if cell.skinning_enabled:
            chain = (
                *chain,
                ("skinning", cell.stages.skinning, (cell.stages.thinning,), {}),
            )
        for kind, fingerprint, parents, inputs in chain:
            key = (kind, fingerprint)
            if key not in validated:
                validated[key] = _validate_referenced_stage(
                    root,
                    kind,
                    fingerprint,
                    result.run_fingerprint,
                    parents,
                    inputs,
                    result.volume_shape,
                )
            workflow_identity = _validate_stage_resolved_settings(
                kind,
                validated[key],
                cell,
                plan,
                result.run_fingerprint,
                result.volume_shape,
                workflow_identity,
            )
        if cell.stages.scanner not in validated_scanner_reports:
            scanner_name = (
                "reference_like_scanner_config"
                if cell.backend == "reference-like"
                else "quality_scanner_config"
            )
            scanner_config_value = dict(
                _object(
                    _object(plan, "run plan")[scanner_name],
                    f"run plan {scanner_name}",
                )
            )
            try:
                scanner_config = F3ScannerConfig(**scanner_config_value)
            except (TypeError, ValueError) as error:
                raise F3ResultValidationError(
                    "scanner report config cannot be derived from the run plan"
                ) from error
            scanner_report = _read_json_object(
                root / "stages" / "scanner" / cell.stages.scanner / "report.json",
                "scanner report",
            )
            _validate_scanner_report_contract(
                scanner_report,
                fingerprint=cell.stages.scanner,
                backend=cell.backend,
                shape=result.volume_shape,
                input_identity=dict(input_file) if input_file is not None else None,
                config=scanner_config,
                settings=validated[("scanner", cell.stages.scanner)],
            )
            validated_scanner_reports.add(cell.stages.scanner)
        if cell.skinning_enabled and cell.stages.skinning not in parsed_skins:
            try:
                scanner_settings = validated[("scanner", cell.stages.scanner)]
                angle_range = _object(
                    scanner_settings["angle_range"],
                    f"{cell.label} scanner angle range",
                )
                parsed_skins[cell.stages.skinning] = parse_skins_json(
                    root / "stages" / "skinning" / cell.stages.skinning / "skins.json",
                    result.volume_shape,
                    strike_range=(
                        angle_range["phi_min"],
                        angle_range["phi_max"],
                    ),
                    dip_range=(
                        angle_range["theta_min"],
                        angle_range["theta_max"],
                    ),
                )
            except (KeyError, TypeError, SkinArtifactValidationError) as error:
                raise F3ResultValidationError(f"invalid skin artifact schema: {error}") from error
        if not cell.skinning_enabled:
            if workflow_identity is _MISSING:
                raise F3ResultValidationError("workflow runner identity is missing")
            skinning_settings = _canonical_skinning_stage_settings(
                cell.resolved_config,
                workflow_identity,
            )
            expected_skinning_fingerprint = canonical_fingerprint(
                {
                    "run_fingerprint": result.run_fingerprint,
                    "kind": "skinning",
                    "parent_fingerprint": cell.stages.thinning,
                    "resolved_settings": skinning_settings,
                    "artifact_schema": {},
                }
            )
            if cell.stages.skinning != expected_skinning_fingerprint:
                raise F3ResultValidationError("skinning stage fingerprint mismatch")
        _validate_cell_stage_reports(
            root,
            cell,
            result.volume_shape,
        )
    by_label = {cell.label: cell for cell in result.cells}
    for left, right in (("RL-REF", "RL-QUAL"), ("Q-REF", "Q-QUAL")):
        if by_label[left].stages.scanner != by_label[right].stages.scanner:
            raise F3ResultValidationError("paired workflows must share the scanner stage")
        if by_label[left].stages.voting != by_label[right].stages.voting:
            raise F3ResultValidationError("default paired workflows must share the voting stage")
    for left, right in (("RL-REF", "Q-REF"), ("RL-QUAL", "Q-QUAL")):
        if by_label[left].resolved_config != by_label[right].resolved_config:
            raise F3ResultValidationError("scanner cells differ outside the scanner-owned stage")
    return parsed_skins


def _resolved_cell_contract(plan_value: Any, workflow: str) -> tuple[dict[str, Any], bool]:
    try:
        plan = _object(plan_value, "run plan")
        workflow_settings = _object(
            plan[f"{workflow}_workflow_settings"],
            f"{workflow} workflow settings",
        )
        voting = _object(plan["voting_controls"], "run plan voting controls")
        controls = VolumeVotingControls(
            strain_max1=voting["strain_max1"],
            strain_max2=voting["strain_max2"],
            surface_smoothing1=voting["surface_smoothing1"],
            surface_smoothing2=voting["surface_smoothing2"],
            boundary_policy=voting["surface_voting_boundary_policy"],
            support_min_fraction=voting["surface_support_min_fraction"],
            support_exponent=voting["surface_support_exponent"],
            orientation_smoothing=voting["surface_orientation_smoothing"],
            final_normalization_smoothing=voting["final_normalization_smoothing"],
        )
        skinning_enabled = _boolean(plan["skinning_enabled"], "run plan skinning_enabled")
        expected = {
            "workflow_mode": workflow,
            "voting": dict(_object(workflow_settings["voting_config"], "workflow voting config")),
            "voting_controls": asdict(controls),
            "skinning": dict(
                _object(workflow_settings["skinning_config"], "workflow skinning config")
            ),
            "variant": asdict(VariantSpec("f3-canonical", experimental=False)),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise F3ResultValidationError("run plan cannot resolve canonical cell settings") from error
    normalized = json.loads(canonical_json_bytes(expected))
    if not isinstance(normalized, dict):
        raise AssertionError("resolved cell contract must be an object")
    return normalized, skinning_enabled


def _validate_cell_stage_reports(
    root: Path,
    cell: F3CellReference,
    shape: tuple[int, int, int],
) -> None:
    contracts = (
        (
            "voting",
            cell.stages.voting,
            {
                "fingerprint": cell.stages.voting,
                "scanner_stage_fingerprint": cell.stages.scanner,
            },
        ),
        (
            "thinning",
            cell.stages.thinning,
            {
                "fingerprint": cell.stages.thinning,
                "voting_stage_fingerprint": cell.stages.voting,
            },
        ),
    )
    if cell.skinning_enabled:
        contracts = (
            *contracts,
            (
                "skinning",
                cell.stages.skinning,
                {
                    "fingerprint": cell.stages.skinning,
                    "thinning_stage_fingerprint": cell.stages.thinning,
                    "enabled": True,
                },
            ),
        )
    for kind, fingerprint, expected in contracts:
        report = _read_json_object(
            root / "stages" / kind / fingerprint / "report.json",
            f"{kind} report",
        )
        _reject_crop_semantics(report)
        if tuple(report.get("shape", ())) != shape:
            raise F3ResultValidationError(f"{kind} report shape mismatch")
        for name, value in expected.items():
            if report.get(name) == value:
                continue
            if name in {"resolved_config", "resolved_stage_settings"}:
                raise F3ResultValidationError(f"{kind} report {name} mismatch")
            raise F3ResultValidationError(f"{kind} report source identity mismatch")


def _validate_scanner_report_contract(
    report: Mapping[str, Any],
    *,
    fingerprint: str,
    backend: str,
    shape: tuple[int, int, int],
    input_identity: Mapping[str, Any] | None,
    config: F3ScannerConfig,
    settings: Mapping[str, Any],
) -> None:
    if set(report) != _SCANNER_REPORT_FIELDS:
        raise F3ResultValidationError("scanner report field set mismatch")
    version = report["scanner_stage_contract_version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != F3_SCANNER_STAGE_CONTRACT_VERSION
    ):
        raise F3ResultValidationError("scanner report contract version mismatch")
    if (
        report["fingerprint"] != fingerprint
        or report["backend"] != backend
        or report["shape"] != list(shape)
        or input_identity is None
        or report["input_fingerprint"] != dict(input_identity)
    ):
        raise F3ResultValidationError("scanner report source identity mismatch")
    if report["resolved_config"] != asdict(config):
        raise F3ResultValidationError("scanner report resolved_config mismatch")
    if report["resolved_stage_settings"] != dict(settings):
        raise F3ResultValidationError("scanner report resolved_stage_settings mismatch")
    if report["requested_remove_edge_effects"] is not config.remove_edge_effects:
        raise F3ResultValidationError("scanner report requested edge removal mismatch")
    if report["effective_remove_edge_effects"] is not config.effective_remove_edge_effects:
        raise F3ResultValidationError("scanner report effective edge removal mismatch")

    sampling_count = _object(report["sampling_count"], "scanner sampling_count")
    if set(sampling_count) != {"strike", "dip", "orientations"}:
        raise F3ResultValidationError("scanner sampling_count field set mismatch")
    sampling = _integer_mapping(sampling_count, "scanner sampling_count")
    if (
        sampling["strike"] <= 0
        or sampling["dip"] <= 0
        or sampling["orientations"] != sampling["strike"] * sampling["dip"]
    ):
        raise F3ResultValidationError("scanner sampling_count relation mismatch")

    expected_raw = {"ft", "pt", "tt"}
    if backend == "quality":
        expected_raw.add("confidence")
    raw = _object(report["raw"], "scanner raw summaries")
    thinned = _object(report["thinned"], "scanner thinned summaries")
    if set(raw) != expected_raw:
        raise F3ResultValidationError("scanner raw summary key set mismatch")
    if set(thinned) != _SCANNER_THINNED_NAMES:
        raise F3ResultValidationError("scanner thinned summary key set mismatch")
    for name, value in raw.items():
        _validate_scanner_summary(value, name=name, shape=shape, config=config, thinned=False)
    for name, value in thinned.items():
        _validate_scanner_summary(value, name=name, shape=shape, config=config, thinned=True)


def _validate_scanner_summary(
    value: Any,
    *,
    name: str,
    shape: tuple[int, int, int],
    config: F3ScannerConfig,
    thinned: bool,
) -> None:
    summary = _object(value, f"scanner {name} summary")
    if set(summary) != _SCANNER_SUMMARY_FIELDS:
        raise F3ResultValidationError(f"scanner {name} summary field set mismatch")
    if summary["shape"] != list(shape):
        raise F3ResultValidationError(f"scanner {name} summary shape mismatch")
    if summary["dtype"] != "float32":
        raise F3ResultValidationError(f"scanner {name} summary dtype mismatch")

    finite_count = summary["finite_count"]
    nonzero_count = summary["nonzero_count"]
    if (
        isinstance(finite_count, bool)
        or not isinstance(finite_count, int)
        or finite_count != math.prod(shape)
    ):
        raise F3ResultValidationError(f"scanner {name} finite_count mismatch")
    if (
        isinstance(nonzero_count, bool)
        or not isinstance(nonzero_count, int)
        or not 0 <= nonzero_count <= finite_count
    ):
        raise F3ResultValidationError(f"scanner {name} nonzero_count is invalid")

    minimum = _scanner_summary_number(summary["min"], name, "min")
    maximum = _scanner_summary_number(summary["max"], name, "max")
    mean = _scanner_summary_number(summary["mean"], name, "mean")
    epsilon = _scanner_summary_number(
        summary["nonzero_epsilon"],
        name,
        "nonzero_epsilon",
    )
    fraction = _scanner_summary_number(
        summary["nonzero_fraction"],
        name,
        "nonzero_fraction",
    )
    if epsilon != NONZERO_EPSILON:
        raise F3ResultValidationError(f"scanner {name} nonzero_epsilon mismatch")
    if fraction != nonzero_count / finite_count or not 0.0 <= fraction <= 1.0:
        raise F3ResultValidationError(f"scanner {name} nonzero_fraction mismatch")
    if not minimum <= mean <= maximum:
        raise F3ResultValidationError(f"scanner {name} summary extrema are invalid")

    if name in {"ft", "fet", "confidence"} and (minimum < 0.0 or maximum > 1.0):
        raise F3ResultValidationError(f"scanner {name} summary range is invalid")
    if name in {"pt", "fpt"}:
        _validate_scanner_summary_angle_range(
            name,
            minimum,
            maximum,
            config.phi_min,
            config.phi_max,
            allow_zero=thinned,
        )
    if name in {"tt", "ftt"}:
        _validate_scanner_summary_angle_range(
            name,
            minimum,
            maximum,
            config.theta_min,
            config.theta_max,
            allow_zero=thinned,
        )


def _scanner_summary_number(value: Any, name: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F3ResultValidationError(f"scanner {name} summary {field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise F3ResultValidationError(f"scanner {name} summary {field} must be finite")
    return result


def _validate_scanner_summary_angle_range(
    name: str,
    observed_minimum: float,
    observed_maximum: float,
    configured_minimum: float,
    configured_maximum: float,
    *,
    allow_zero: bool,
) -> None:
    tolerance = (
        8.0
        * float(np.finfo(np.float32).eps)
        * max(
            1.0,
            abs(configured_minimum),
            abs(configured_maximum),
        )
    )
    minimum = min(configured_minimum, 0.0) if allow_zero else configured_minimum
    maximum = max(configured_maximum, 0.0) if allow_zero else configured_maximum
    if observed_minimum < minimum - tolerance or observed_maximum > maximum + tolerance:
        raise F3ResultValidationError(f"scanner {name} summary angle range is invalid")


def _validate_referenced_stage(
    root: Path,
    kind: str,
    fingerprint: str,
    run_fingerprint: str,
    parents: tuple[str, ...],
    inputs: Mapping[str, Any],
    shape: tuple[int, int, int],
) -> Mapping[str, Any]:
    _sha256(fingerprint, f"{kind} stage fingerprint")
    stage_path = root / "stages" / kind / fingerprint
    manifest = _read_json_object(stage_path / STAGE_MANIFEST_FILE, "stage manifest")
    try:
        computation = {name: manifest[name] for name in _STAGE_COMPUTATION_FIELDS}
    except KeyError as error:
        raise F3ResultValidationError("stage computation identity is incomplete") from error
    if (
        computation["kind"] != kind
        or computation["run_fingerprint"] != run_fingerprint
        or computation["parent_fingerprints"] != list(parents)
    ):
        raise F3ResultValidationError(f"{kind} stage parent/run chain mismatch")
    expected_inputs = {name: value for name, value in inputs.items() if value is not None}
    if expected_inputs and computation["input_fingerprints"] != expected_inputs:
        raise F3ResultValidationError(f"{kind} stage input fingerprint mismatch")
    if canonical_fingerprint(computation) != fingerprint:
        raise F3ResultValidationError(f"{kind} stage fingerprint mismatch")
    _reject_crop_semantics(computation)
    schema = computation.get("artifact_schema")
    if not isinstance(schema, Mapping) or not schema:
        raise F3ResultValidationError(f"{kind} artifact schema is invalid")
    settings = computation["resolved_settings"]
    if not isinstance(settings, Mapping):
        raise F3ResultValidationError(f"{kind} stage resolved_settings is invalid")
    try:
        if kind == "scanner":
            backend = settings.get("backend")
            if backend not in {"reference-like", "quality"}:
                raise ValueError("scanner backend is invalid")
            canonical_artifacts = scanner_stage_artifacts(shape, backend)
        elif kind == "voting":
            canonical_artifacts = voting_stage_artifacts(shape)
        elif kind == "thinning":
            canonical_artifacts = thinning_stage_artifacts(shape)
        elif kind == "skinning":
            canonical_artifacts = skinning_stage_artifacts(shape, enabled=True)
        else:
            raise ValueError("stage kind is invalid")
    except (TypeError, ValueError) as error:
        raise F3ResultValidationError(f"{kind} canonical artifact schema is invalid") from error
    canonical_schema = {artifact.filename: artifact.as_dict() for artifact in canonical_artifacts}
    if schema != canonical_schema:
        raise F3ResultValidationError(f"{kind} canonical artifact schema mismatch")
    for descriptor in schema.values():
        if not isinstance(descriptor, Mapping):
            raise F3ResultValidationError(f"{kind} artifact descriptor is invalid")
        artifact_shape = descriptor.get("shape")
        if artifact_shape is not None and tuple(artifact_shape) != shape:
            raise F3ResultValidationError(f"{kind} stage is cropped or has the wrong shape")
        if descriptor.get("format") == "dat" and descriptor.get("dtype") != ">f4":
            raise F3ResultValidationError(f"{kind} stage dtype mismatch")
    validate_stage(stage_path, computation, fingerprint)
    return settings


def _validate_stage_resolved_settings(
    kind: str,
    settings: Mapping[str, Any],
    cell: F3CellReference,
    plan_value: Any,
    run_fingerprint: str,
    shape: tuple[int, int, int],
    workflow_identity: Any,
) -> Any:
    try:
        plan = _object(plan_value, "run plan")
        scanner_name = (
            "reference_like_scanner_config"
            if cell.backend == "reference-like"
            else "quality_scanner_config"
        )
        scanner_config = F3ScannerConfig(
            **dict(_object(plan[scanner_name], f"run plan {scanner_name}"))
        )
        if kind == "scanner":
            expected = scanner_stage_resolved_settings(
                scanner_config,
                shape,
                implementation_identity=settings.get("scanner_stage_implementation_identity"),
            )
        else:
            current_identity = settings.get("workflow_runner_identity")
            if current_identity is None:
                raise ValueError("workflow_runner_identity is missing")
            if workflow_identity is _MISSING:
                workflow_identity = current_identity
            elif current_identity != workflow_identity:
                raise ValueError("workflow_runner_identity is inconsistent")

            resolved = dict(cell.resolved_config)
            voting_config = SyntheticVotingConfig(
                **dict(_object(resolved["voting"], "cell voting config"))
            )
            controls = VolumeVotingControls(
                **dict(_object(resolved["voting_controls"], "cell voting controls"))
            )
            variant = VariantSpec("f3-canonical", experimental=False)
            attribute = PreparedAttributeIdentity(
                dataset_fingerprint=run_fingerprint,
                stage_fingerprint=cell.stages.scanner,
                shape=shape,
                backend=cell.backend,
                scanner_thin_mode=scanner_config.scanner_thin_mode,
                edge_policy=bool(scanner_config.effective_remove_edge_effects),
            )
            seed_key = build_seed_stage_key(
                attribute_key=attribute.stage_key,
                voting_config=voting_config,
                variant_spec=variant,
                target_source="scanner_fet",
            )
            voting_key = build_voting_stage_key(
                seed_key=seed_key,
                voting_config=voting_config,
                variant_spec=variant,
                voting_controls=controls,
            )
            thinning_key = build_thinning_stage_key(
                voting_key=voting_key,
                voting_config=voting_config,
                variant_spec=variant,
            )
            final_thinning_key = build_final_thinning_stage_key(
                thinning_key=thinning_key,
                variant_spec=variant,
                target_source="scanner_fet",
            )
            if voting_key is None or final_thinning_key is None:
                raise ValueError("canonical workflow stage keys are missing")
            if kind == "voting":
                expected = {
                    "cell_runner_contract_version": F3_CELL_RUNNER_CONTRACT_VERSION,
                    "implementation_contract": F3_VOTING_STAGE_IMPLEMENTATION,
                    "workflow_runner_identity": workflow_identity,
                    "attribute_identity": asdict(attribute),
                    "semantic_key": asdict(voting_key),
                }
            elif kind == "thinning":
                expected = {
                    "cell_runner_contract_version": F3_CELL_RUNNER_CONTRACT_VERSION,
                    "implementation_contract": F3_THINNING_STAGE_IMPLEMENTATION,
                    "workflow_runner_identity": workflow_identity,
                    "semantic_key": asdict(final_thinning_key),
                }
            elif kind == "skinning":
                expected = _canonical_skinning_stage_settings(
                    resolved,
                    workflow_identity,
                )
            else:
                raise ValueError(f"unknown stage kind: {kind}")
        normalized = json.loads(canonical_json_bytes(expected))
    except (KeyError, TypeError, ValueError) as error:
        raise F3ResultValidationError(
            f"{kind} stage resolved_settings cannot be derived from the run plan"
        ) from error
    if settings != normalized:
        raise F3ResultValidationError(
            f"{kind} stage resolved_settings does not match the run plan/cell config"
        )
    return workflow_identity


def _canonical_skinning_stage_settings(
    resolved_config: Mapping[str, Any],
    workflow_identity: Any,
) -> dict[str, Any]:
    skinning = dict(_object(resolved_config["skinning"], "cell skinning config"))
    settings = {
        "cell_runner_contract_version": F3_CELL_RUNNER_CONTRACT_VERSION,
        "implementation_contract": F3_SKINNING_STAGE_IMPLEMENTATION,
        "workflow_runner_identity": workflow_identity,
        "enabled": skinning["enabled"],
        "resolved_skinner_config": skinning,
        "growth_source": skinning["growth_source"],
        "fallback_policy": {
            "enabled": skinning["boundary_skinner_fallback"],
            "policy": skinning["boundary_skinner_fallback_policy"],
        },
        "primary_skinner_identity": ("pyosv.experimental.boundary_skinning.find_synthetic_skins"),
    }
    if skinning["enabled"]:
        settings["skin_artifact_semantic_contract_version"] = (
            F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION
        )
    normalized = json.loads(canonical_json_bytes(settings))
    if not isinstance(normalized, dict):
        raise AssertionError("skinning stage settings must be an object")
    return normalized


def _validate_metrics(
    result: F3ModeComparisonResult,
    dataset: Mapping[str, Any],
) -> None:
    rows = result.metric_rows
    evidence = result.metric_evidence
    if len({row.identity for row in rows}) != len(rows):
        raise F3ResultValidationError("duplicate metric row identity")
    if len({item.identity for item in evidence}) != len(evidence):
        raise F3ResultValidationError("duplicate metric evidence identity")
    by_evidence = {item.identity: item for item in evidence}
    by_cell = {cell.label: cell for cell in result.cells}
    registry = {(item.stage, item.selection, item.metric): item for item in METRIC_REGISTRY}
    expected_row_ids: set[tuple[str, str, str, str]] = set()
    for cell in result.cells:
        for definition in METRIC_REGISTRY:
            if definition.stage != "skin" or cell.skinning_enabled:
                expected_row_ids.add(
                    (cell.label, definition.stage, definition.selection, definition.metric)
                )
    actual_row_ids = {(row.cell_label, row.stage, row.selection, row.metric) for row in rows}
    if actual_row_ids != expected_row_ids:
        raise F3ResultValidationError("metric registry coverage mismatch")
    expected_row_order = tuple(
        (cell.label, definition.stage, definition.selection, definition.metric)
        for cell in result.cells
        for definition in METRIC_REGISTRY
        if definition.stage != "skin" or cell.skinning_enabled
    )
    actual_row_order = tuple((row.cell_label, row.stage, row.selection, row.metric) for row in rows)
    if actual_row_order != expected_row_order:
        raise F3ResultValidationError("metric row order mismatch")
    expected_evidence_ids = {
        (
            result.dataset_id,
            cell.label,
            definition.stage,
            "full",
            definition.selection,
            F3_REFERENCE_STAGE_FILES.get(definition.stage) or "",
        )
        for cell in result.cells
        for definition in METRIC_REGISTRY
        if definition.stage != "skin" or cell.skinning_enabled
    }
    if set(by_evidence) != expected_evidence_ids:
        raise F3ResultValidationError("metric evidence coverage mismatch")
    for row in rows:
        if row.dataset_id != result.dataset_id:
            raise F3ResultValidationError("metric dataset mismatch")
        owning_cell = by_cell[row.cell_label]
        if (row.scanner_backend, row.workflow_mode) != (
            owning_cell.backend,
            owning_cell.workflow,
        ):
            raise F3ResultValidationError("metric row axes do not match owning cell")
        definition = registry.get((row.stage, row.selection, row.metric))
        if definition is None or (
            row.unit,
            row.direction,
            row.contrast_eligible,
        ) != (
            definition.unit,
            definition.direction,
            definition.contrast_eligible and row.value is not None,
        ):
            raise F3ResultValidationError("metric registry semantics mismatch")
        item = by_evidence.get(row.identity[:-1])
        if item is None:
            raise F3ResultValidationError("metric row has no evidence")
        expected_fingerprint = {
            "ft": by_cell[row.cell_label].stages.scanner,
            "fv": by_cell[row.cell_label].stages.voting,
            "fvt": by_cell[row.cell_label].stages.thinning,
            "skin": by_cell[row.cell_label].stages.skinning,
        }[row.stage]
        if item.source_stage_fingerprint != expected_fingerprint:
            raise F3ResultValidationError("metric source stage fingerprint mismatch")
        if item.shape != result.volume_shape:
            raise F3ResultValidationError("metric evidence shape mismatch")
        if item.dataset_id != result.dataset_id:
            raise F3ResultValidationError("metric evidence dataset mismatch")
        _validate_metric_evidence_fields(item)
        if item.reference_file is not None:
            role = _REFERENCE_ROLE_BY_FILE[item.reference_file]
            reference = dataset["files"].get(role)
            if reference is None or item.reference_sha256 != reference["sha256"]:
                raise F3ResultValidationError("metric reference file fingerprint mismatch")
        expected_value = _metric_from_evidence(item, row.metric)
        if not _optional_number_equal(row.value, expected_value):
            raise F3ResultValidationError(
                f"metric evidence scalar mismatch: {row.cell_label}/{row.stage}/"
                f"{row.selection}/{row.metric}"
            )
    try:
        validate_shared_stage_metrics(rows, evidence)
        expected_contrasts = compute_contrast_rows(rows, evidence)
    except ValueError as error:
        raise F3ResultValidationError(str(error)) from error
    if result.contrast_rows != expected_contrasts:
        raise F3ResultValidationError("contrast rows do not match metric rows")


def _validate_metric_evidence_fields(item: MetricEvidence) -> None:
    counts = set(dict(item.counts))
    accumulators = set(dict(item.accumulators))
    thresholds = dict(item.thresholds)
    if item.stage == "skin":
        expected_counts = {
            "skin_count",
            "cell_count",
            "unique_cell_count",
            "duplicate_cell_count",
            "largest_skin_size",
            "small_skin_cell_count",
            "accepted_skin_count",
            "fallback_enabled",
            "fallback_used",
            "fallback_skin_count",
            "fallback_cell_count",
        }
        expected_accumulators: set[str] = set()
        expected_thresholds: set[str] = set()
    elif item.selection == "all":
        expected_counts = {
            "voxel_count",
            "candidate_finite_count",
            "reference_finite_count",
            "candidate_nonzero_count",
            "reference_nonzero_count",
        }
        expected_accumulators = {
            "candidate_sum",
            "candidate_sum_square",
            "reference_sum",
            "reference_sum_square",
            "cross_product_sum",
            "absolute_difference_sum",
            "squared_difference_sum",
            "candidate_min",
            "candidate_max",
            "reference_min",
            "reference_max",
            "absolute_difference_median",
            "absolute_difference_p90",
            "absolute_difference_p95",
            "absolute_difference_p99",
            "absolute_difference_max",
        }
        expected_thresholds = {"nonzero_epsilon"}
    elif item.selection.endswith("_distance"):
        expected_counts = {"reference_count", "candidate_count"}
        expected_accumulators = set()
        if all(dict(item.counts)[name] for name in expected_counts):
            for prefix in ("candidate_to_reference", "reference_to_candidate"):
                expected_accumulators.update(
                    {
                        f"{prefix}_distance_sum",
                        f"{prefix}_median",
                        f"{prefix}_p90",
                        f"{prefix}_p95",
                    }
                )
        expected_thresholds = {
            "percentile",
            "reference_threshold",
            "candidate_threshold",
        }
    else:
        expected_counts = {
            "reference_count",
            "candidate_count",
            "intersection_count",
            "union_count",
        }
        if item.selection.endswith("_radius2"):
            expected_counts.update(
                {
                    "candidate_in_reference_buffer_count",
                    "reference_in_candidate_buffer_count",
                }
            )
            expected_thresholds = {
                "percentile",
                "reference_threshold",
                "candidate_threshold",
                "radius",
            }
        else:
            expected_thresholds = {
                "percentile",
                "reference_threshold",
                "candidate_threshold",
            }
        expected_accumulators = set()
    if (
        counts != expected_counts
        or accumulators != expected_accumulators
        or set(thresholds) != expected_thresholds
    ):
        raise F3ResultValidationError(
            "metric evidence field coverage mismatch: "
            f"{item.stage}/{item.selection}; counts={sorted(counts)}; "
            f"accumulators={sorted(accumulators)}; thresholds={sorted(thresholds)}"
        )
    if item.selection == "all":
        if thresholds["nonzero_epsilon"] != NONZERO_EPSILON:
            raise F3ResultValidationError("metric evidence nonzero epsilon mismatch")
    elif thresholds:
        expected_percentile = {
            "positive_p95": 95.0,
            "positive_p99": 99.0,
            "positive_p99_5": 99.5,
            "positive_p99_radius2": 99.0,
            "positive_p99_distance": 99.0,
        }[item.selection]
        if thresholds["percentile"] != expected_percentile:
            raise F3ResultValidationError("metric evidence percentile mismatch")
        if thresholds["reference_threshold"] < 0.0 or thresholds["candidate_threshold"] < 0.0:
            raise F3ResultValidationError("metric evidence threshold must be non-negative")
        if "radius" in thresholds and thresholds["radius"] != 2.0:
            raise F3ResultValidationError("metric evidence radius mismatch")
    count_values = dict(item.counts)
    if item.stage == "skin":
        if (
            count_values["unique_cell_count"] > count_values["cell_count"]
            or count_values["duplicate_cell_count"]
            != count_values["cell_count"] - count_values["unique_cell_count"]
            or count_values["largest_skin_size"] > count_values["cell_count"]
            or count_values["small_skin_cell_count"] > count_values["cell_count"]
            or count_values["fallback_enabled"] not in {0, 1}
            or count_values["fallback_used"] not in {0, 1}
            or count_values["fallback_used"] > count_values["fallback_enabled"]
        ):
            raise F3ResultValidationError("skin metric evidence counts are inconsistent")
    elif item.selection == "all":
        voxel_count = int(np.prod(item.shape))
        if (
            count_values["voxel_count"] != voxel_count
            or count_values["candidate_finite_count"] != voxel_count
            or count_values["reference_finite_count"] != voxel_count
            or count_values["candidate_nonzero_count"] > voxel_count
            or count_values["reference_nonzero_count"] > voxel_count
        ):
            raise F3ResultValidationError("all-voxel evidence counts are inconsistent")
    elif not item.selection.endswith("_distance"):
        reference = count_values["reference_count"]
        candidate = count_values["candidate_count"]
        intersection = count_values["intersection_count"]
        union = count_values["union_count"]
        if (
            intersection > min(reference, candidate)
            or union != reference + candidate - intersection
        ):
            raise F3ResultValidationError("overlap evidence counts are inconsistent")
        if item.selection.endswith("_radius2") and (
            count_values["candidate_in_reference_buffer_count"] > candidate
            or count_values["reference_in_candidate_buffer_count"] > reference
        ):
            raise F3ResultValidationError("buffered overlap counts are inconsistent")


def _metric_from_evidence(item: MetricEvidence, metric: str) -> float | None:
    counts = dict(item.counts)
    accumulators = dict(item.accumulators)
    if item.stage == "skin":
        if metric in counts:
            return float(counts[metric])
        cell_count = counts["cell_count"]
        if metric == "largest_skin_fraction":
            return counts["largest_skin_size"] / cell_count if cell_count else 0.0
        if metric == "small_skin_cell_fraction":
            return counts["small_skin_cell_count"] / cell_count if cell_count else 0.0
        raise F3ResultValidationError(f"skin evidence cannot derive metric {metric!r}")
    if item.selection == "all":
        count = counts["voxel_count"]
        candidate_count = counts["candidate_finite_count"]
        reference_count = counts["reference_finite_count"]
        candidate_sum = accumulators["candidate_sum"]
        reference_sum = accumulators["reference_sum"]
        candidate_square = accumulators["candidate_sum_square"]
        reference_square = accumulators["reference_sum_square"]
        covariance = accumulators["cross_product_sum"] - candidate_sum * reference_sum / count
        candidate_variance_sum = max(0.0, candidate_square - candidate_sum**2 / count)
        reference_variance_sum = max(0.0, reference_square - reference_sum**2 / count)
        denominator = math.sqrt(candidate_variance_sum * reference_variance_sum)
        values: dict[str, float] = {
            "voxel_count": float(count),
            "candidate_finite_count": float(candidate_count),
            "reference_finite_count": float(reference_count),
            "candidate_finite_fraction": candidate_count / count,
            "reference_finite_fraction": reference_count / count,
            "candidate_min": accumulators["candidate_min"],
            "candidate_max": accumulators["candidate_max"],
            "candidate_mean": candidate_sum / count,
            "candidate_std": math.sqrt(candidate_variance_sum / count),
            "reference_min": accumulators["reference_min"],
            "reference_max": accumulators["reference_max"],
            "reference_mean": reference_sum / count,
            "reference_std": math.sqrt(reference_variance_sum / count),
            "candidate_nonzero_count": float(counts["candidate_nonzero_count"]),
            "reference_nonzero_count": float(counts["reference_nonzero_count"]),
            "candidate_nonzero_fraction": counts["candidate_nonzero_count"] / count,
            "reference_nonzero_fraction": counts["reference_nonzero_count"] / count,
            "nonzero_fraction_ratio": (
                counts["candidate_nonzero_count"] / counts["reference_nonzero_count"]
                if counts["reference_nonzero_count"]
                else 0.0
            ),
            "normalized_correlation": 0.0 if denominator == 0.0 else covariance / denominator,
            "mean_absolute_difference": accumulators["absolute_difference_sum"] / count,
            "root_mean_square_difference": math.sqrt(
                accumulators["squared_difference_sum"] / count
            ),
            "absolute_difference_mean": accumulators["absolute_difference_sum"] / count,
            "absolute_difference_median": accumulators["absolute_difference_median"],
            "absolute_difference_p90": accumulators["absolute_difference_p90"],
            "absolute_difference_p95": accumulators["absolute_difference_p95"],
            "absolute_difference_p99": accumulators["absolute_difference_p99"],
            "absolute_difference_max": accumulators["absolute_difference_max"],
        }
        return values[metric]
    if item.selection.endswith("_distance"):
        if metric in counts:
            return float(counts[metric])
        prefix = metric.rsplit("_", 1)[0]
        suffix = metric.rsplit("_", 1)[1]
        count_name = "candidate_count" if prefix == "candidate_to_reference" else "reference_count"
        count = counts[count_name]
        if count == 0 or not counts["candidate_count"] or not counts["reference_count"]:
            return None
        if suffix == "mean":
            return accumulators[f"{prefix}_distance_sum"] / count
        return accumulators[metric]
    reference_count = counts["reference_count"]
    candidate_count = counts["candidate_count"]
    intersection = counts["intersection_count"]
    union = counts["union_count"]
    precision = intersection / candidate_count if candidate_count else 0.0
    recall = intersection / reference_count if reference_count else 0.0
    values = {
        "reference_count": float(reference_count),
        "candidate_count": float(candidate_count),
        "intersection_count": float(intersection),
        "union_count": float(union),
        "precision": precision,
        "recall": recall,
        "f1": 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall),
        "jaccard": intersection / union if union else 0.0,
    }
    if item.selection.endswith("_radius2"):
        candidate_buffer = counts["candidate_in_reference_buffer_count"]
        reference_buffer = counts["reference_in_candidate_buffer_count"]
        buffered_precision = candidate_buffer / candidate_count if candidate_count else 0.0
        buffered_recall = reference_buffer / reference_count if reference_count else 0.0
        values.update(
            {
                "candidate_in_reference_buffer_count": float(candidate_buffer),
                "reference_in_candidate_buffer_count": float(reference_buffer),
                "buffered_precision": buffered_precision,
                "buffered_recall": buffered_recall,
                "buffered_f1": (
                    0.0
                    if buffered_precision + buffered_recall == 0.0
                    else 2.0
                    * buffered_precision
                    * buffered_recall
                    / (buffered_precision + buffered_recall)
                ),
            }
        )
    return values[metric]


def _validate_voxelwise_contrasts(result: F3ModeComparisonResult) -> None:
    definitions = {item.name: item for item in CONTRAST_DEFINITIONS}
    expected = {
        (stage, definition.name)
        for stage in F3_REFERENCE_STAGE_FILES
        for definition in CONTRAST_DEFINITIONS
    }
    actual = {(row.stage, row.contrast_name) for row in result.voxelwise_contrasts}
    if actual != expected or len(actual) != len(result.voxelwise_contrasts):
        raise F3ResultValidationError("voxel contrast coverage mismatch")
    expected_order = tuple(
        (stage, definition.name)
        for stage in F3_REFERENCE_STAGE_FILES
        for definition in CONTRAST_DEFINITIONS
    )
    actual_order = tuple((row.stage, row.contrast_name) for row in result.voxelwise_contrasts)
    if actual_order != expected_order:
        raise F3ResultValidationError("voxel contrast row order mismatch")
    by_cell = {cell.label: cell for cell in result.cells}
    for row in result.voxelwise_contrasts:
        if row.dataset_id != result.dataset_id or row.shape != result.volume_shape:
            raise F3ResultValidationError("voxel contrast identity mismatch")
        if row.registration_id != f"{result.dataset_id}:{result.volume_shape}":
            raise F3ResultValidationError("voxel contrast registration mismatch")
        definition = definitions[row.contrast_name]
        if row.component_cells != definition.component_cells:
            raise F3ResultValidationError("voxel contrast formula mismatch")
        expected_fingerprints = tuple(
            (
                cell,
                {
                    "ft": by_cell[cell].stages.scanner,
                    "fv": by_cell[cell].stages.voting,
                    "fvt": by_cell[cell].stages.thinning,
                }[row.stage],
            )
            for cell in definition.component_cells
        )
        if row.component_stage_fingerprints != expected_fingerprints:
            raise F3ResultValidationError("voxel contrast source fingerprint mismatch")
        if (
            row.std < 0.0
            or row.mean_absolute < 0.0
            or row.p95_absolute < 0.0
            or row.max_absolute < 0.0
            or row.p95_absolute > row.max_absolute
            or row.mean_absolute > row.max_absolute
            or abs(row.mean) > row.mean_absolute + _FLOAT_ABS_TOL
        ):
            raise F3ResultValidationError("voxel contrast summary is inconsistent")
        if row.stage in {"ft", "fv"} and row.contrast_name in {
            "workflow_effect_rl",
            "workflow_effect_q",
            "workflow_main_effect",
            "scanner_workflow_interaction",
        }:
            if any(
                value != 0.0
                for value in (
                    row.mean,
                    row.std,
                    row.mean_absolute,
                    row.p95_absolute,
                    row.max_absolute,
                    row.epsilon_nonzero_fraction,
                )
            ):
                raise F3ResultValidationError("shared-stage voxel contrast must be exactly zero")


def _validate_regional_rows(result: F3ModeComparisonResult, plan_value: Any) -> None:
    plan = _object(plan_value, "run plan")
    boundary_margin = plan.get("boundary_diagnostic_margin")
    if (
        isinstance(boundary_margin, bool)
        or not isinstance(boundary_margin, int)
        or boundary_margin < 0
    ):
        raise F3ResultValidationError("run plan boundary diagnostic margin is invalid")
    by_cell = {cell.label: cell for cell in result.cells}
    expected = {
        (cell, stage, region)
        for cell in _CELL_ORDER
        for stage in F3_REFERENCE_STAGE_FILES
        for region in F3_DIAGNOSTIC_REGIONS
    }
    actual = {(row.cell_label, row.stage, row.region) for row in result.regional_rows}
    if actual != expected or len(actual) != len(result.regional_rows):
        raise F3ResultValidationError("regional diagnostic coverage mismatch")
    expected_order = tuple(
        (cell, stage, region)
        for stage in F3_REFERENCE_STAGE_FILES
        for cell in _CELL_ORDER
        for region in F3_DIAGNOSTIC_REGIONS
    )
    actual_order = tuple((row.cell_label, row.stage, row.region) for row in result.regional_rows)
    if actual_order != expected_order:
        raise F3ResultValidationError("regional diagnostic row order mismatch")
    groups: dict[tuple[str, str], dict[str, RegionalDiagnosticRow]] = {}
    for row in result.regional_rows:
        owning_cell = by_cell[row.cell_label]
        if (
            row.dataset_id != result.dataset_id
            or row.volume_shape != result.volume_shape
            or row.region_semantics != F3_REGION_SEMANTICS
        ):
            raise F3ResultValidationError("regional diagnostic identity mismatch")
        if (row.scanner_backend, row.workflow_mode) != (
            owning_cell.backend,
            owning_cell.workflow,
        ):
            raise F3ResultValidationError("regional row axes do not match owning cell")
        if row.source_stage_fingerprint != _cell_stage_fingerprint(
            owning_cell,
            row.stage,
        ):
            raise F3ResultValidationError("regional source stage fingerprint mismatch")
        groups.setdefault((row.cell_label, row.stage), {})[row.region] = row
        _validate_regional_metric_algebra(row.metrics)
    voxel_count = math.prod(result.volume_shape)
    interior_voxel_count = math.prod(size - 2 * boundary_margin for size in result.volume_shape)
    boundary_voxel_count = voxel_count - interior_voxel_count
    margins = {row.boundary_margin for row in result.regional_rows}
    if len(margins) != 1:
        raise F3ResultValidationError("regional rows have mixed boundary margins")
    if margins != {boundary_margin}:
        raise F3ResultValidationError("regional boundary margin does not match the run manifest")
    full_counts = {
        (item.cell_label, item.stage): _count_value(dict(item.counts), "voxel_count")
        for item in result.metric_evidence
        if item.stage in F3_REFERENCE_STAGE_FILES and item.selection == "all"
    }
    if set(full_counts) != set(groups):
        raise F3ResultValidationError("regional rows have no matching full-volume evidence")
    for identity, rows in groups.items():
        full_count = full_counts[identity]
        interior_count = int(rows["interior"].metrics["voxel_count"])
        boundary_count = int(rows["boundary_shell"].metrics["voxel_count"])
        if (
            full_count != voxel_count
            or interior_count != interior_voxel_count
            or boundary_count != boundary_voxel_count
        ):
            raise F3ResultValidationError("regional counts do not partition the full volume")


def _validate_regional_metric_algebra(metrics: Mapping[str, Any]) -> None:
    _finite_json_value(dict(metrics), "regional metrics", allow_none=True)
    voxel_count = _count_value(metrics, "voxel_count")
    for name in ("candidate_nonzero_fraction", "reference_nonzero_fraction"):
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise F3ResultValidationError(f"regional {name} is invalid")
    for name in ("mean_absolute_difference", "root_mean_square_difference"):
        value = metrics.get(name)
        if voxel_count == 0:
            if value is not None:
                raise F3ResultValidationError("empty region must have null difference metrics")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0.0:
            raise F3ResultValidationError(f"regional {name} is invalid")
    correlation = metrics.get("normalized_correlation")
    if voxel_count == 0:
        if correlation is not None:
            raise F3ResultValidationError("empty region must have null correlation")
    elif (
        isinstance(correlation, bool)
        or not isinstance(correlation, (int, float))
        or not -1.0 - _FLOAT_ABS_TOL <= float(correlation) <= 1.0 + _FLOAT_ABS_TOL
    ):
        raise F3ResultValidationError("regional correlation is invalid")
    for prefix in _metric_prefixes(metrics, "_intersection_count"):
        reference = _count_value(metrics, f"{prefix}_reference_count")
        candidate = _count_value(metrics, f"{prefix}_candidate_count")
        intersection = _count_value(metrics, f"{prefix}_intersection_count")
        union = _count_value(metrics, f"{prefix}_union_count")
        if (
            intersection > min(reference, candidate)
            or union != reference + candidate - intersection
        ):
            raise F3ResultValidationError("regional overlap counts are inconsistent")
        _require_close(
            metrics[f"{prefix}_precision"], intersection / candidate if candidate else 0.0
        )
        _require_close(metrics[f"{prefix}_recall"], intersection / reference if reference else 0.0)
        _require_close(metrics[f"{prefix}_jaccard"], intersection / union if union else 0.0)
        precision = float(metrics[f"{prefix}_precision"])
        recall = float(metrics[f"{prefix}_recall"])
        _require_close(
            metrics[f"{prefix}_f1"],
            0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall),
        )
        buffer_candidate_name = f"{prefix}_candidate_in_reference_buffer_count"
        if buffer_candidate_name in metrics:
            candidate_buffer = _count_value(metrics, buffer_candidate_name)
            reference_buffer = _count_value(
                metrics, f"{prefix}_reference_in_candidate_buffer_count"
            )
            buffered_precision = candidate_buffer / candidate if candidate else 0.0
            buffered_recall = reference_buffer / reference if reference else 0.0
            _require_close(metrics[f"{prefix}_buffered_precision"], buffered_precision)
            _require_close(metrics[f"{prefix}_buffered_recall"], buffered_recall)
            _require_close(
                metrics[f"{prefix}_buffered_f1"],
                (
                    0.0
                    if buffered_precision + buffered_recall == 0.0
                    else 2.0
                    * buffered_precision
                    * buffered_recall
                    / (buffered_precision + buffered_recall)
                ),
            )


def _validate_orientation_rows(result: F3ModeComparisonResult) -> None:
    by_cell = {cell.label: cell for cell in result.cells}
    expected = {
        (stage, left, right)
        for stage in ("scanner", "voting")
        for left, right in F3_ORIENTATION_PAIRS
    }
    actual = {(row.stage, row.left_cell, row.right_cell) for row in result.orientation_rows}
    if actual != expected or len(actual) != len(result.orientation_rows):
        raise F3ResultValidationError("orientation diagnostic coverage mismatch")
    expected_order = tuple(
        (stage, left, right)
        for stage in ("scanner", "voting")
        for left, right in F3_ORIENTATION_PAIRS
    )
    actual_order = tuple(
        (row.stage, row.left_cell, row.right_cell) for row in result.orientation_rows
    )
    if actual_order != expected_order:
        raise F3ResultValidationError("orientation diagnostic row order mismatch")
    for row in result.orientation_rows:
        if row.dataset_id != result.dataset_id:
            raise F3ResultValidationError("orientation diagnostic dataset mismatch")
        if row.left_source_stage_fingerprint != _cell_stage_fingerprint(
            by_cell[row.left_cell], row.stage
        ) or row.right_source_stage_fingerprint != _cell_stage_fingerprint(
            by_cell[row.right_cell], row.stage
        ):
            raise F3ResultValidationError("orientation source stage fingerprint mismatch")
        for summary in (
            row.strike_circular_absolute_difference,
            row.dip_absolute_difference,
            row.normal_vector_angular_difference,
        ):
            _validate_summary(summary, row.support_count)


def _validate_summary(summary: Mapping[str, Any], support_count: int) -> None:
    if summary["count"] != support_count:
        raise F3ResultValidationError("orientation support count mismatch")
    values = [summary[name] for name in ("mean", "median", "p90", "p95")]
    if support_count == 0:
        if any(value is not None for value in values):
            raise F3ResultValidationError("empty orientation support has non-null summary")
        return
    if any(value is None for value in values):
        raise F3ResultValidationError("orientation summary is incomplete")
    _finite_json_value(values, "orientation summary")
    if float(summary["p90"]) > float(summary["p95"]):
        raise F3ResultValidationError("orientation percentile order mismatch")


def _validate_resource_rows(root: Path, result: F3ModeComparisonResult) -> None:
    referenced = _referenced_stage_keys(result.cells)
    actual = {(row.stage_kind, row.fingerprint) for row in result.runtime_rows}
    if actual != referenced:
        raise F3ResultValidationError("runtime stage coverage mismatch")
    identities = [(row.stage_kind, row.fingerprint, row.cell) for row in result.runtime_rows]
    if len(identities) != len(set(identities)):
        raise F3ResultValidationError("duplicate runtime stage row")
    expected_identities = {
        ("scanner", fingerprint, consumers[0])
        for fingerprint in {cell.stages.scanner for cell in result.cells}
        if (
            consumers := tuple(
                cell.label for cell in result.cells if cell.stages.scanner == fingerprint
            )
        )
    }
    expected_identities.update(
        (kind, getattr(cell.stages, kind), cell.label)
        for cell in result.cells
        for kind in ("voting", "thinning", "skinning")
        if kind != "skinning" or cell.skinning_enabled
    )
    if set(identities) != expected_identities:
        raise F3ResultValidationError("runtime stage-use coverage mismatch")
    expected_order = [
        ("scanner", fingerprint, consumers[0])
        for fingerprint in dict.fromkeys(cell.stages.scanner for cell in result.cells)
        if (
            consumers := tuple(
                cell.label for cell in result.cells if cell.stages.scanner == fingerprint
            )
        )
    ]
    expected_order.extend(
        (kind, getattr(cell.stages, kind), cell.label)
        for cell in result.cells
        for kind in ("voting", "thinning", "skinning")
        if kind != "skinning" or cell.skinning_enabled
    )
    if identities != expected_order:
        raise F3ResultValidationError("runtime row order mismatch")
    for row in result.runtime_rows:
        if row.stage_kind not in {"scanner", "voting", "thinning", "skinning"}:
            raise F3ResultValidationError("unknown runtime stage kind")
        expected_consumers = tuple(
            cell.label
            for cell in result.cells
            if getattr(cell.stages, row.stage_kind) == row.fingerprint
            and (row.stage_kind != "skinning" or cell.skinning_enabled)
        )
        if row.cell_consumers != expected_consumers or row.cell not in expected_consumers:
            raise F3ResultValidationError("runtime consumer mapping mismatch")
        if row.voxel_count != int(np.prod(result.volume_shape)):
            raise F3ResultValidationError("runtime voxel count mismatch")
    for key in referenced:
        rows = [row for row in result.runtime_rows if (row.stage_kind, row.fingerprint) == key]
        states = tuple(row.state for row in rows)
        if states not in {
            ("reused",) * len(rows),
            ("computed", *(("reused",) * (len(rows) - 1))),
        }:
            raise F3ResultValidationError(
                "runtime computed/reused stage multiplicity is infeasible"
            )
    stage_storage = {
        (row.stage_kind, row.fingerprint) for row in result.storage_rows if row.scope == "stage"
    }
    storage_identities = [
        (row.scope, row.stage_kind, row.fingerprint) for row in result.storage_rows
    ]
    if len(storage_identities) != len(set(storage_identities)):
        raise F3ResultValidationError("duplicate resource storage row")
    if stage_storage != referenced:
        raise F3ResultValidationError("resource storage stage coverage mismatch")
    if sum(row.scope == "workspace" for row in result.storage_rows) != 1:
        raise F3ResultValidationError("resource storage requires one workspace row")
    current_storage = storage_report(root)
    actual_stage_storage = {
        (row.stage_kind, row.fingerprint): row
        for row in current_storage
        if row.scope == "stage" and (row.stage_kind, row.fingerprint) in referenced
    }
    reported_stage_storage = {
        (row.stage_kind, row.fingerprint): row
        for row in result.storage_rows
        if row.scope == "stage"
    }
    if reported_stage_storage != actual_stage_storage:
        raise F3ResultValidationError("resource stage storage values mismatch")
    workspace_row = next(row for row in result.storage_rows if row.scope == "workspace")
    actual_workspace_row = next(row for row in current_storage if row.scope == "workspace")
    if workspace_row != actual_workspace_row:
        raise F3ResultValidationError("resource workspace storage values mismatch")
    rss_points = [(row.scope, row.point) for row in result.rss_snapshots]
    if len(rss_points) != len(set(rss_points)):
        raise F3ResultValidationError("duplicate RSS snapshot")
    if not any(row.scope == "process_peak" for row in result.rss_snapshots):
        raise F3ResultValidationError("resource RSS requires a process-peak snapshot")
    stage_boundaries: dict[tuple[str, str], set[str]] = {}
    for row in result.rss_snapshots:
        if row.scope != "stage_snapshot":
            continue
        stage_key, boundary = _rss_stage_boundary(row.point)
        if stage_key not in referenced:
            raise F3ResultValidationError("resource RSS references an unknown stage")
        stage_boundaries.setdefault(stage_key, set()).add(boundary)
    if set(stage_boundaries) != referenced or any(
        boundaries != {"before", "after"} for boundaries in stage_boundaries.values()
    ):
        raise F3ResultValidationError("resource RSS stage-boundary coverage mismatch")


def _rss_stage_boundary(point: str) -> tuple[tuple[str, str], str]:
    parts = point.split(":")
    if parts[-1].startswith("occurrence="):
        parts = parts[:-1]
    if len(parts) < 3 or parts[-1] not in {"before", "after"}:
        raise F3ResultValidationError("resource RSS stage point is not canonical")
    stage_kind, fingerprint = parts[:2]
    if stage_kind not in {"scanner", "voting", "thinning", "skinning"}:
        raise F3ResultValidationError("resource RSS has an unknown stage kind")
    try:
        _sha256(fingerprint, "resource RSS stage fingerprint")
    except ValueError as error:
        raise F3ResultValidationError("resource RSS stage fingerprint is invalid") from error
    return (stage_kind, fingerprint), parts[-1]


def _deep_validate_scanner_stages(
    root: Path,
    result: F3ModeComparisonResult,
    plan_value: Any,
) -> None:
    plan = _object(plan_value, "run plan")
    stages: dict[str, F3ScannerBackend] = {}
    for cell in result.cells:
        previous = stages.setdefault(cell.stages.scanner, cell.backend)
        if previous != cell.backend:
            raise F3ResultValidationError("scanner stage backend reuse mismatch")

    for fingerprint, backend in stages.items():
        scanner_name = (
            "reference_like_scanner_config"
            if backend == "reference-like"
            else "quality_scanner_config"
        )
        try:
            config = F3ScannerConfig(
                **dict(_object(plan[scanner_name], f"run plan {scanner_name}"))
            )
            scanner = FaultOrientScanner3(config.sigma1, config.sigma2)
            expected_sampling = scanner_sampling_count(
                scanner,
                config,
                backend,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise F3ResultValidationError(
                "deep scanner sampling contract cannot be derived"
            ) from error

        stage_path = root / "stages" / "scanner" / fingerprint
        report = _read_json_object(stage_path / "report.json", "scanner report")
        if report["sampling_count"] != expected_sampling:
            raise F3ResultValidationError("deep scanner sampling_count mismatch")

        groups = (
            ("raw", ("ft", "pt", "tt", *(("confidence",) if backend == "quality" else ()))),
            ("thinned", ("fet", "fpt", "ftt")),
        )
        expected_bytes = math.prod(result.volume_shape) * np.dtype(">f4").itemsize
        for group, names in groups:
            summaries = _object(report[group], f"scanner {group} summaries")
            for name in names:
                path = stage_path / f"{name}.dat"
                array: np.memmap | None = None
                try:
                    if path.stat().st_size != expected_bytes:
                        raise ValueError("storage size mismatch")
                    array = np.memmap(
                        path,
                        dtype=">f4",
                        mode="r",
                        shape=result.volume_shape,
                        order="C",
                    )
                    if array.dtype.str != ">f4" or array.shape != result.volume_shape:
                        raise ValueError("storage layout mismatch")
                    actual = scanner_array_summary(array)
                    _deep_validate_scanner_array_range(
                        array,
                        name=name,
                        config=config,
                        thinned=group == "thinned",
                    )
                except (OSError, ValueError) as error:
                    raise F3ResultValidationError(
                        f"deep scanner array validation failed: {name}"
                    ) from error
                finally:
                    if array is not None:
                        mapping = getattr(array, "_mmap", None)
                        if mapping is not None and not mapping.closed:
                            mapping.close()
                        del array
                if summaries[name] != actual:
                    raise F3ResultValidationError(f"deep scanner summary mismatch: {name}")


def _deep_validate_scanner_array_range(
    values: np.ndarray,
    *,
    name: str,
    config: F3ScannerConfig,
    thinned: bool,
) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError("array contains non-finite values")
    if name in {"ft", "fet", "confidence"}:
        if float(np.min(values)) < 0.0 or float(np.max(values)) > 1.0:
            raise ValueError("unit-interval array is out of range")
        return
    if name in {"pt", "fpt"}:
        minimum, maximum = config.phi_min, config.phi_max
    elif name in {"tt", "ftt"}:
        minimum, maximum = config.theta_min, config.theta_max
    else:
        raise ValueError("unknown scanner array")
    tolerance = (
        8.0
        * float(np.finfo(np.float32).eps)
        * max(
            1.0,
            abs(minimum),
            abs(maximum),
        )
    )
    valid = (values >= minimum - tolerance) & (values <= maximum + tolerance)
    if thinned:
        valid |= values == np.float32(0.0)
    if not np.all(valid):
        raise ValueError("angle array is out of range")


def _deep_validate_skin_artifacts(
    root: Path,
    result: F3ModeComparisonResult,
    parsed_skins: Mapping[str, ParsedSkinArtifacts],
) -> None:
    """Validate and exactly reproduce each referenced final skin collection once."""

    validated: set[str] = set()
    for cell in result.cells:
        fingerprint = cell.stages.skinning
        if not cell.skinning_enabled or fingerprint in validated:
            continue
        try:
            parsed = parsed_skins[fingerprint]
            skinning_config = _object(
                cell.resolved_config["skinning"],
                f"{cell.label} skinning config",
            )
            validate_skin_artifact_semantics(
                root / "stages" / "skinning" / fingerprint,
                result.volume_shape,
                small_skin_size=skinning_config["small_skin_size"],
                parsed=parsed,
            )
            skinning_report = _read_json_object(
                root / "stages" / "skinning" / fingerprint / "report.json",
                "skinning report",
            )
            diagnostics = _object(
                skinning_report["diagnostics"],
                f"{cell.label} skinning diagnostics",
            )
            fallback_used = diagnostics.get("fallback_used")
            if not isinstance(fallback_used, bool):
                raise SkinArtifactValidationError("skinning report fallback_used must be bool")
            parent_contract = resolve_skin_parent_volume_contract(
                cell.stages.as_dict(),
                skinning_config,
                _object(
                    cell.resolved_config["variant"],
                    f"{cell.label} variant config",
                ),
                fallback_used=fallback_used,
            )
            _validate_skin_parent_samples(
                root,
                parsed,
                result.volume_shape,
                parent_contract.likelihood,
                parent_contract.strike,
                parent_contract.dip,
            )
            _recompute_skin_artifacts(
                root,
                cell,
                parsed,
                result.volume_shape,
                skinning_config,
                fallback_used,
                parent_contract.scanner_target,
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise F3ResultValidationError(
                f"deep skin artifact mismatch: {cell.label}: {error}"
            ) from error
        validated.add(fingerprint)


def _validate_skin_parent_samples(
    root: Path,
    parsed: ParsedSkinArtifacts,
    shape: tuple[int, int, int],
    likelihood: tuple[str, str, str],
    strike: tuple[str, str, str],
    dip: tuple[str, str, str],
) -> None:
    for field_name, source in (
        ("fl", likelihood),
        ("fp", strike),
        ("ft", dip),
    ):
        values = _open_parent_volume(root, source, shape)
        try:
            for skin in parsed.skins:
                for cell in skin:
                    expected = np.float32(values[cell.i3, cell.i2, cell.i1])
                    if np.float32(getattr(cell, field_name)) != expected:
                        raise SkinArtifactValidationError(
                            f"skins.json cell {field_name} does not match parent volume"
                        )
        finally:
            _close_memmap(values)


def _recompute_skin_artifacts(
    root: Path,
    cell: F3CellReference,
    parsed: ParsedSkinArtifacts,
    shape: tuple[int, int, int],
    skinning_config: Mapping[str, Any],
    fallback_used: bool,
    scanner_target: tuple[str, str, str] | None,
) -> None:
    voting = ("voting", cell.stages.voting)
    thinning = ("thinning", cell.stages.thinning)
    sources = (
        (*voting, "fv.dat"),
        (*thinning, "fvt.dat"),
        (*voting, "vp.dat"),
        (*voting, "vt.dat"),
    )
    recomputed_payload: dict[str, Any] | None = None
    recomputed = None
    volume_size = math.prod(shape)
    float_bytes = volume_size * np.dtype(np.float32).itemsize
    mask_bytes = (volume_size + 7) // 8 if scanner_target is not None else 0
    with tempfile.TemporaryFile(dir=root) as staging:
        staging.truncate(len(sources) * float_bytes + mask_bytes)
        backing = np.memmap(
            staging,
            dtype=np.uint8,
            mode="r+",
            shape=(len(sources) * float_bytes + mask_bytes,),
        )
        parent_views: list[np.ndarray] = []
        scanner_mask_bits: np.ndarray | None = None
        try:
            for index, source in enumerate(sources):
                parent = np.ndarray(
                    shape,
                    dtype=np.float32,
                    buffer=backing,
                    offset=index * float_bytes,
                    order="C",
                )
                _copy_parent_volume(root, source, shape, parent)
                parent.flags.writeable = False
                parent_views.append(parent)
            if scanner_target is not None:
                scanner_mask_bits = np.ndarray(
                    (mask_bytes,),
                    dtype=np.uint8,
                    buffer=backing,
                    offset=len(sources) * float_bytes,
                    order="C",
                )
                _copy_parent_positive_mask(root, scanner_target, shape, scanner_mask_bits)
                scanner_mask_bits.flags.writeable = False
            backing.flush()
            backing.flags.writeable = False
            fv, fvt, vp, vt = parent_views

            def rerun_boundary_fallback(*args: Any, **kwargs: Any) -> None:
                scanner_mask = (
                    None
                    if scanner_mask_bits is None
                    else np.unpackbits(
                        scanner_mask_bits,
                        count=volume_size,
                    )
                    .reshape(shape)
                    .view(np.bool_)
                )
                if scanner_mask is not None:
                    scanner_mask.flags.writeable = False
                kwargs["scanner_target_positive_mask"] = scanner_mask
                try:
                    apply_boundary_skinner_fallback(*args, **kwargs)
                finally:
                    scanner_mask = None

            recomputed = execute_skinning_phase3d(
                fv=fv,
                fvt=fvt,
                vp=vp,
                vt=vt,
                skinning_settings=SyntheticSkinningConfig(**dict(skinning_config)),
                variant_spec=_resolved_variant_spec(cell.resolved_config["variant"]),
                scanner_target_positive_mask=None,
                boundary_fallback_runner=rerun_boundary_fallback,
            )
            if recomputed.diagnostics.get("fallback_used") is not fallback_used:
                raise SkinArtifactValidationError(
                    "recomputed fallback state does not match skinning report"
                )
            recomputed_payload = canonical_skins_payload(recomputed.skins)
        finally:
            scanner_mask_bits = None
            parent_views.clear()
            recomputed = None
            _close_memmap(backing)
    if recomputed_payload != canonical_skins_payload(parsed.skins):
        raise SkinArtifactValidationError(
            "skins.json does not exactly match skin-only recomputation"
        )


def _copy_parent_volume(
    root: Path,
    source: tuple[str, str, str],
    shape: tuple[int, int, int],
    target: np.ndarray,
) -> None:
    values = _open_parent_volume(root, source, shape)
    try:
        source_flat = values.reshape(-1)
        target_flat = target.reshape(-1)
        for start in range(0, source_flat.size, _SKIN_RECOMPUTE_CHUNK_VOXELS):
            stop = min(source_flat.size, start + _SKIN_RECOMPUTE_CHUNK_VOXELS)
            target_flat[start:stop] = source_flat[start:stop]
    finally:
        _close_memmap(values)


def _copy_parent_positive_mask(
    root: Path,
    source: tuple[str, str, str],
    shape: tuple[int, int, int],
    target: np.ndarray,
) -> None:
    values = _open_parent_volume(root, source, shape)
    try:
        source_flat = values.reshape(-1)
        target_offset = 0
        for start in range(0, source_flat.size, _SKIN_RECOMPUTE_CHUNK_VOXELS):
            stop = min(source_flat.size, start + _SKIN_RECOMPUTE_CHUNK_VOXELS)
            packed = np.packbits(positive_candidate_mask(source_flat[start:stop]))
            target[target_offset : target_offset + packed.size] = packed
            target_offset += packed.size
    finally:
        _close_memmap(values)


def _resolved_variant_spec(value: Any) -> VariantSpec:
    variant = _object(value, "cell variant")
    return VariantSpec(
        name=variant["name"],
        voting=VotingPatch(**dict(_object(variant["voting"], "cell variant voting patch"))),
        seed_policy=variant["seed_policy"],
        thinning_policy=variant["thinning_policy"],
        post_thinning_policy=variant["post_thinning_policy"],
        skinning=SkinningPatch(**dict(_object(variant["skinning"], "cell variant skinning patch"))),
        experimental=variant["experimental"],
        presets=tuple(variant["presets"]),
        baseline=variant["baseline"],
    )


def _open_parent_volume(
    root: Path,
    source: tuple[str, str, str],
    shape: tuple[int, int, int],
) -> np.memmap:
    kind, fingerprint, filename = source
    values = np.memmap(
        root / "stages" / kind / fingerprint / filename,
        dtype=">f4",
        mode="r",
        shape=shape,
        order="C",
    )
    values.flags.writeable = False
    return values


def _deep_validate_reference_metrics(
    root: Path,
    result: F3ModeComparisonResult,
    dataset: Mapping[str, Any],
) -> None:
    source_paths = _dataset_source_paths(root, dataset)
    by_rows: dict[tuple[str, str], tuple[MetricRow, ...]] = {}
    by_evidence: dict[tuple[str, str], tuple[MetricEvidence, ...]] = {}
    for cell in result.cells:
        for stage, filename in F3_REFERENCE_STAGE_FILES.items():
            role = F3_REFERENCE_STAGE_ROLES[stage]
            reference_path = source_paths[role]
            candidate_kind, candidate_fingerprint, candidate_filename = {
                "ft": ("scanner", cell.stages.scanner, "ft.dat"),
                "fv": ("voting", cell.stages.voting, "fv.dat"),
                "fvt": ("thinning", cell.stages.thinning, "fvt.dat"),
            }[stage]
            candidate_path = (
                root / "stages" / candidate_kind / candidate_fingerprint / candidate_filename
            )
            reference: np.memmap | None = None
            candidate: np.memmap | None = None
            try:
                reference = np.memmap(
                    reference_path,
                    dtype=result.storage_dtype,
                    mode="r",
                    shape=result.volume_shape,
                    order="C",
                )
                candidate = np.memmap(
                    candidate_path,
                    dtype=result.storage_dtype,
                    mode="r",
                    shape=result.volume_shape,
                    order="C",
                )
                stored_all_evidence = next(
                    item
                    for item in result.metric_evidence
                    if item.cell_label == cell.label
                    and item.stage == stage
                    and item.selection == "all"
                )
                computed_rows, computed_evidence = compute_reference_metric_rows(
                    dataset_id=result.dataset_id,
                    cell_label=cell.label,
                    scanner_backend=cell.backend,
                    workflow_mode=cell.workflow,
                    stage=stage,
                    reference_file=filename,
                    candidate=candidate,
                    reference=reference,
                    source_stage_fingerprint=candidate_fingerprint,
                    reference_sha256=dataset["files"][role]["sha256"],
                    nonzero_epsilon=dict(stored_all_evidence.thresholds)["nonzero_epsilon"],
                )
                computed_regional = compute_regional_reference_diagnostics(
                    dataset_id=result.dataset_id,
                    cell_label=cell.label,
                    scanner_backend=cell.backend,
                    workflow_mode=cell.workflow,
                    stage=stage,
                    source_stage_fingerprint=candidate_fingerprint,
                    candidate=candidate,
                    reference=reference,
                    margin=result.regional_rows[0].boundary_margin,
                )
            finally:
                _close_memmap(candidate)
                _close_memmap(reference)
            by_rows[(cell.label, stage)] = computed_rows
            by_evidence[(cell.label, stage)] = computed_evidence
            stored_regional = tuple(
                row
                for row in result.regional_rows
                if row.cell_label == cell.label and row.stage == stage
            )
            if computed_regional != stored_regional:
                raise F3ResultValidationError("deep regional metric mismatch")
        if cell.skinning_enabled:
            stage_path = root / "stages" / "skinning" / cell.stages.skinning
            report = _read_json_object(
                stage_path / "report.json",
                f"{cell.label} skinning report",
            )
            try:
                computed_rows, computed_evidence = compute_skin_metric_rows(
                    dataset_id=result.dataset_id,
                    cell_label=cell.label,
                    scanner_backend=cell.backend,
                    workflow_mode=cell.workflow,
                    report=report,
                    source_stage_fingerprint=cell.stages.skinning,
                    shape=result.volume_shape,
                )
            except ValueError as error:
                raise F3ResultValidationError(
                    f"deep skin metric report mismatch: {cell.label}"
                ) from error
            by_rows[(cell.label, "skin")] = computed_rows
            by_evidence[(cell.label, "skin")] = computed_evidence
    for key, computed in by_rows.items():
        stored = tuple(row for row in result.metric_rows if (row.cell_label, row.stage) == key)
        if len(computed) != len(stored) or any(
            left.identity != right.identity
            or left.as_dict().keys() != right.as_dict().keys()
            or {name: value for name, value in left.as_dict().items() if name != "value"}
            != {name: value for name, value in right.as_dict().items() if name != "value"}
            or not _optional_number_equal(left.value, right.value)
            for left, right in zip(computed, stored, strict=True)
        ):
            raise F3ResultValidationError("deep metric row mismatch")
    for key, computed in by_evidence.items():
        stored = tuple(row for row in result.metric_evidence if (row.cell_label, row.stage) == key)
        if len(computed) != len(stored) or any(
            not _metric_evidence_close(left, right)
            for left, right in zip(computed, stored, strict=True)
        ):
            raise F3ResultValidationError("deep metric evidence mismatch")


def _deep_validate_voxelwise_contrasts(
    root: Path,
    result: F3ModeComparisonResult,
) -> None:
    by_cell = {cell.label: cell for cell in result.cells}
    rows = {(row.stage, row.contrast_name): row for row in result.voxelwise_contrasts}
    size = int(np.prod(result.volume_shape))
    slab_depth = min(8, result.volume_shape[0])
    stage_sources = {
        "ft": ("scanner", "scanner", "ft.dat"),
        "fv": ("voting", "voting", "fv.dat"),
        "fvt": ("thinning", "thinning", "fvt.dat"),
    }
    with tempfile.TemporaryDirectory(dir=root) as temporary_directory:
        temporary_root = Path(temporary_directory)
        for stage in F3_REFERENCE_STAGE_FILES:
            kind, fingerprint_field, filename = stage_sources[stage]
            for definition in CONTRAST_DEFINITIONS:
                stored = rows[(stage, definition.name)]
                contrast_path = temporary_root / f"{stage}-{definition.name}.float64"
                contrast = np.memmap(
                    contrast_path,
                    dtype=np.float64,
                    mode="w+",
                    shape=result.volume_shape,
                    order="C",
                )
                try:
                    for component_index, (cell_label, coefficient) in enumerate(
                        definition.coefficients
                    ):
                        fingerprint = getattr(
                            by_cell[cell_label].stages,
                            fingerprint_field,
                        )
                        source: np.memmap | None = None
                        try:
                            source = np.memmap(
                                root / "stages" / kind / fingerprint / filename,
                                dtype=result.storage_dtype,
                                mode="r",
                                shape=result.volume_shape,
                                order="C",
                            )
                            for start in range(0, result.volume_shape[0], slab_depth):
                                stop = min(result.volume_shape[0], start + slab_depth)
                                component = np.asarray(source[start:stop], dtype=np.float64)
                                if component_index == 0:
                                    contrast[start:stop] = coefficient * component
                                else:
                                    contrast[start:stop] += coefficient * component
                        finally:
                            _close_memmap(source)
                    contrast.flush()
                    actual = _summarize_deep_contrast(
                        contrast,
                        size=size,
                        epsilon=stored.epsilon,
                        slab_depth=slab_depth,
                    )
                finally:
                    _close_memmap(contrast)
                for name, value in actual.items():
                    if not _optional_number_equal(value, getattr(stored, name)):
                        raise F3ResultValidationError(
                            "deep voxel contrast summary mismatch: "
                            f"{stage}/{definition.name}/{name}"
                        )


def _deep_validate_orientation_diagnostics(
    root: Path,
    result: F3ModeComparisonResult,
) -> None:
    """Recompute each orientation pair while retaining only one pair at a time."""

    by_cell = {cell.label: cell for cell in result.cells}
    stored_rows = {
        (row.stage, row.left_cell, row.right_cell): row for row in result.orientation_rows
    }
    for stage, (kind, fingerprint_field, filenames) in _ORIENTATION_STAGE_FILES.items():
        likelihood_name, strike_name, dip_name = filenames
        for left_label, right_label in F3_ORIENTATION_PAIRS:
            opened: list[np.memmap] = []
            fields_by_cell: list[tuple[np.memmap, np.memmap, np.memmap]] = []
            try:
                for label in (left_label, right_label):
                    fingerprint = getattr(by_cell[label].stages, fingerprint_field)
                    stage_path = root / "stages" / kind / fingerprint
                    cell_fields: list[np.memmap] = []
                    for filename in (likelihood_name, strike_name, dip_name):
                        array = np.memmap(
                            stage_path / filename,
                            dtype=result.storage_dtype,
                            mode="r",
                            shape=result.volume_shape,
                            order="C",
                        )
                        opened.append(array)
                        cell_fields.append(array)
                    fields_by_cell.append((cell_fields[0], cell_fields[1], cell_fields[2]))
                left_fields, right_fields = fields_by_cell
                computed = compute_orientation_pair_diagnostic(
                    dataset_id=result.dataset_id,
                    stage=stage,
                    left_cell=left_label,
                    right_cell=right_label,
                    left_source_stage_fingerprint=_cell_stage_fingerprint(
                        by_cell[left_label],
                        stage,
                    ),
                    right_source_stage_fingerprint=_cell_stage_fingerprint(
                        by_cell[right_label],
                        stage,
                    ),
                    left_likelihood=left_fields[0],
                    left_strike=left_fields[1],
                    left_dip=left_fields[2],
                    right_likelihood=right_fields[0],
                    right_strike=right_fields[1],
                    right_dip=right_fields[2],
                )
            finally:
                fields_by_cell.clear()
                for array in opened:
                    _close_memmap(array)
            stored = stored_rows[(stage, left_label, right_label)]
            if not _orientation_row_close(computed, stored):
                raise F3ResultValidationError(
                    f"deep orientation diagnostic mismatch: {stage}/{left_label}/{right_label}"
                )


def _orientation_row_close(
    left: OrientationDiagnosticRow,
    right: OrientationDiagnosticRow,
) -> bool:
    identity_fields = (
        "schema_version",
        "dataset_id",
        "stage",
        "left_cell",
        "right_cell",
        "left_source_stage_fingerprint",
        "right_source_stage_fingerprint",
        "support_contract",
        "support_count",
    )
    if any(getattr(left, name) != getattr(right, name) for name in identity_fields):
        return False
    for name in (
        "strike_circular_absolute_difference",
        "dip_absolute_difference",
        "normal_vector_angular_difference",
    ):
        left_summary = dict(getattr(left, name))
        right_summary = dict(getattr(right, name))
        if set(left_summary) != set(right_summary) or any(
            not _optional_number_equal(value, right_summary[key])
            for key, value in left_summary.items()
        ):
            return False
    return True


def _summarize_deep_contrast(
    contrast: np.memmap,
    *,
    size: int,
    epsilon: float,
    slab_depth: int,
) -> dict[str, float]:
    count = 0
    mean = 0.0
    m2 = 0.0
    absolute_parts: list[float] = []
    epsilon_count = 0
    maximum = 0.0
    for start in range(0, contrast.shape[0], slab_depth):
        stop = min(contrast.shape[0], start + slab_depth)
        values = np.asarray(contrast[start:stop], dtype=np.float64)
        slab_count = int(values.size)
        slab_mean = float(np.mean(values, dtype=np.float64))
        centered = values - slab_mean
        slab_m2 = float(np.dot(centered.ravel(), centered.ravel()))
        if count:
            combined = count + slab_count
            delta = slab_mean - mean
            m2 += slab_m2 + delta * delta * count * slab_count / combined
            mean += delta * slab_count / combined
            count = combined
        else:
            count = slab_count
            mean = slab_mean
            m2 = slab_m2
        absolute = np.abs(values)
        absolute_parts.append(float(np.sum(absolute, dtype=np.float64)))
        epsilon_count += int(np.count_nonzero(absolute > epsilon))
        maximum = max(maximum, float(np.max(absolute)))
        contrast[start:stop] = absolute
    contrast.flush()
    position = (size - 1) * 0.95
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    flat = contrast.reshape(-1)
    flat.partition(sorted({lower, upper}))
    weight = position - lower
    p95 = float(float(flat[lower]) * (1.0 - weight) + float(flat[upper]) * weight)
    return {
        "mean": mean,
        "std": math.sqrt(max(0.0, m2 / count)),
        "mean_absolute": math.fsum(absolute_parts) / size,
        "p95_absolute": p95,
        "max_absolute": maximum,
        "epsilon_nonzero_fraction": epsilon_count / size,
    }


def _metric_evidence_close(left: MetricEvidence, right: MetricEvidence) -> bool:
    if (
        left.identity != right.identity
        or left.source_stage_fingerprint != right.source_stage_fingerprint
        or left.reference_sha256 != right.reference_sha256
        or left.shape != right.shape
        or dict(left.counts) != dict(right.counts)
        or set(dict(left.thresholds)) != set(dict(right.thresholds))
        or set(dict(left.accumulators)) != set(dict(right.accumulators))
    ):
        return False
    return all(
        _optional_number_equal(value, dict(right.thresholds)[name])
        for name, value in left.thresholds
    ) and all(
        _optional_number_equal(value, dict(right.accumulators)[name])
        for name, value in left.accumulators
    )


def _cell_stage_fingerprint(cell: F3CellReference, stage: str) -> str:
    return {
        "ft": cell.stages.scanner,
        "fv": cell.stages.voting,
        "fvt": cell.stages.thinning,
        "scanner": cell.stages.scanner,
        "voting": cell.stages.voting,
    }[stage]


def _dataset_source_paths(
    root: Path,
    dataset: Mapping[str, Any],
) -> dict[str, Path]:
    manifest = _read_json_object(root / RUN_MANIFEST_FILE, RUN_MANIFEST_FILE)
    provenance = manifest.get("provenance")
    entries = provenance.get("dataset_files") if isinstance(provenance, Mapping) else None
    if not isinstance(entries, list):
        raise F3ResultValidationError("deep validation requires dataset source provenance")
    result: dict[str, Path] = {}
    for item in entries:
        if not isinstance(item, Mapping):
            raise F3ResultValidationError("dataset source provenance is invalid")
        role = item.get("role")
        path = item.get("resolved_path")
        if role in dataset["files"] and isinstance(path, str):
            source = Path(path)
            if not source.is_file() or source.is_symlink():
                raise F3ResultValidationError(f"deep source file is unavailable: {role}")
            expected_metadata = {
                "sha256": dataset["files"][role]["sha256"],
                "size": dataset["files"][role]["size"],
            }
            if artifact_file_metadata(source) != expected_metadata:
                raise F3ResultValidationError(f"deep source file hash or size mismatch: {role}")
            result[role] = source
    required = set(F3_REFERENCE_STAGE_ROLES.values())
    if set(result) & required != required:
        raise F3ResultValidationError("deep validation source coverage mismatch")
    return result


def _referenced_stage_completion_metadata(
    root: Path,
    cells: Sequence[F3CellReference],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for kind, fingerprint in sorted(_referenced_stage_keys(cells)):
        key = f"{kind}/{fingerprint}"
        output[key] = artifact_file_metadata(
            root / "stages" / kind / fingerprint / STAGE_COMPLETION_FILE
        )
    return output


def _referenced_stage_keys(
    cells: Sequence[F3CellReference],
) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for cell in cells:
        result.update(
            {
                ("scanner", cell.stages.scanner),
                ("voting", cell.stages.voting),
                ("thinning", cell.stages.thinning),
            }
        )
        if cell.skinning_enabled:
            result.add(("skinning", cell.stages.skinning))
    return result


def _result_identity(result: F3ModeComparisonResult) -> dict[str, Any]:
    return {
        "result_schema_version": F3_RESULT_SCHEMA_VERSION,
        "run_fingerprint": result.run_fingerprint,
        "dataset_id": result.dataset_id,
        "volume_shape": list(result.volume_shape),
        "storage_dtype": result.storage_dtype,
    }


def _parse_result_identity(payload: Mapping[str, Any], context: str) -> dict[str, Any]:
    try:
        version = payload["result_schema_version"]
        fingerprint = payload["run_fingerprint"]
        dataset_id = payload["dataset_id"]
        shape = payload["volume_shape"]
        dtype = payload["storage_dtype"]
    except KeyError as error:
        raise F3ResultValidationError(f"{context} result identity is incomplete") from error
    if version != F3_RESULT_SCHEMA_VERSION:
        raise F3ResultValidationError(f"{context} result schema mismatch")
    _sha256(fingerprint, f"{context} run fingerprint")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise F3ResultValidationError(f"{context} dataset ID is invalid")
    parsed_shape = _shape3_list(shape)
    try:
        parsed_dtype = np.dtype(dtype).str
    except TypeError as error:
        raise F3ResultValidationError(f"{context} dtype is invalid") from error
    return {
        "run_fingerprint": fingerprint,
        "dataset_id": dataset_id,
        "volume_shape": parsed_shape,
        "storage_dtype": parsed_dtype,
    }


def _cell_from_dict(root: Path, value: Any) -> F3CellReference:
    item = _object(value, "cell")
    if set(item) != {
        "cell_reference_schema_version",
        "label",
        "backend",
        "workflow",
        "resolved_config",
        "stages",
        "skinning",
    }:
        raise F3ResultValidationError("cell field set mismatch")
    if item["cell_reference_schema_version"] != 1:
        raise F3ResultValidationError("cell reference schema mismatch")
    stages = _object(item["stages"], "cell stages")
    if set(stages) != {"scanner", "voting", "thinning", "skinning"}:
        raise F3ResultValidationError("cell stage field set mismatch")
    skinning = _object(item["skinning"], "cell skinning")
    if set(skinning) != {"enabled", "state"}:
        raise F3ResultValidationError("cell skinning field set mismatch")
    enabled = _boolean(skinning["enabled"], "cell skinning enabled")
    if skinning["state"] != ("enabled" if enabled else "disabled"):
        raise F3ResultValidationError("cell skinning state mismatch")
    label = item["label"]
    if not isinstance(label, str):
        raise F3ResultValidationError("cell label must be a string")
    resolved = _object(item["resolved_config"], "cell resolved_config")
    return F3CellReference(
        label,
        item["backend"],
        item["workflow"],
        MappingProxyType(dict(resolved)),
        F3CellStageFingerprints(
            _sha256(stages["scanner"], "scanner fingerprint"),
            _sha256(stages["voting"], "voting fingerprint"),
            _sha256(stages["thinning"], "thinning fingerprint"),
            _sha256(stages["skinning"], "skinning fingerprint"),
        ),
        enabled,
        root / "cells" / f"{label}.json",
        True,
    )


def _metric_row(item: Mapping[str, str]) -> MetricRow:
    return MetricRow(
        _integer_text(item["schema_version"], "metric schema_version"),
        item["dataset_id"],
        item["cell_label"],
        item["scanner_backend"],
        item["workflow_mode"],
        item["stage"],
        item["region"],
        item["selection"],
        item["reference_file"] or None,
        item["metric"],
        _optional_float_text(item["value"], "metric value"),
        item["unit"],
        item["direction"],  # type: ignore[arg-type]
        _bool_text(item["contrast_eligible"], "metric contrast_eligible"),
    )


def _metric_evidence(value: Any) -> MetricEvidence:
    item = _object(value, "metric evidence")
    expected = {
        "schema_version",
        "dataset_id",
        "cell_label",
        "stage",
        "region",
        "selection",
        "reference_file",
        "source_stage_fingerprint",
        "reference_sha256",
        "shape",
        "thresholds",
        "counts",
        "accumulators",
    }
    if set(item) != expected:
        raise F3ResultValidationError("metric evidence field set mismatch")
    return MetricEvidence(
        item["schema_version"],
        item["dataset_id"],
        item["cell_label"],
        item["stage"],
        item["region"],
        item["selection"],
        item["reference_file"],
        item["source_stage_fingerprint"],
        item["reference_sha256"],
        _shape3_list(item["shape"]),
        tuple(_number_mapping(item["thresholds"], "thresholds").items()),
        tuple(_integer_mapping(item["counts"], "counts").items()),
        tuple(_number_mapping(item["accumulators"], "accumulators").items()),
    )


def _contrast_row(item: Mapping[str, str]) -> ContrastRow:
    return ContrastRow(
        _integer_text(item["schema_version"], "contrast schema_version"),
        item["dataset_id"],
        item["contrast_name"],
        item["stage"],
        item["region"],
        item["selection"],
        item["reference_file"] or None,
        item["metric"],
        item["unit"],
        item["direction"],  # type: ignore[arg-type]
        tuple(_json_csv_list(item["component_cells"], "component_cells")),
        _float_text(item["raw_value"], "raw_value"),
        _optional_float_text(item["improvement_value"], "improvement_value"),
    )


def _voxelwise_row(item: Mapping[str, str]) -> VoxelwiseContrastSummary:
    fingerprint_value = _json_csv(
        item["component_stage_fingerprints"], "component_stage_fingerprints"
    )
    if isinstance(fingerprint_value, dict):
        fingerprints = tuple(fingerprint_value.items())
    elif isinstance(fingerprint_value, list) and all(
        isinstance(pair, list) and len(pair) == 2 for pair in fingerprint_value
    ):
        fingerprints = tuple((pair[0], pair[1]) for pair in fingerprint_value)
    else:
        raise F3ResultValidationError("component_stage_fingerprints must encode ordered pairs")
    return VoxelwiseContrastSummary(
        _integer_text(item["schema_version"], "voxel schema_version"),
        item["dataset_id"],
        item["contrast_name"],
        item["stage"],
        item["region"],
        tuple(_json_csv_list(item["shape"], "shape")),  # type: ignore[arg-type]
        item["registration_id"],
        tuple(_json_csv_list(item["component_cells"], "component_cells")),
        fingerprints,
        _float_text(item["mean"], "mean"),
        _float_text(item["std"], "std"),
        _float_text(item["mean_absolute"], "mean_absolute"),
        _float_text(item["p95_absolute"], "p95_absolute"),
        _float_text(item["max_absolute"], "max_absolute"),
        _float_text(item["epsilon"], "epsilon"),
        _float_text(item["epsilon_nonzero_fraction"], "epsilon_nonzero_fraction"),
    )


def _regional_row(item: Mapping[str, str]) -> RegionalDiagnosticRow:
    return RegionalDiagnosticRow(
        _integer_text(item["schema_version"], "regional schema_version"),
        item["dataset_id"],
        item["cell_label"],
        item["scanner_backend"],
        item["workflow_mode"],
        item["stage"],
        item["source_stage_fingerprint"],
        tuple(_json_csv_list(item["volume_shape"], "volume_shape")),  # type: ignore[arg-type]
        _integer_text(item["boundary_margin"], "boundary_margin"),
        item["region"],
        item["region_semantics"],
        _json_csv_object(item["metrics"], "regional metrics"),
    )


def _orientation_row(item: Mapping[str, str]) -> OrientationDiagnosticRow:
    return OrientationDiagnosticRow(
        _integer_text(item["schema_version"], "orientation schema_version"),
        item["dataset_id"],
        item["stage"],
        item["left_cell"],
        item["right_cell"],
        item["left_source_stage_fingerprint"],
        item["right_source_stage_fingerprint"],
        item["support_contract"],
        _integer_text(item["support_count"], "support_count"),
        _json_csv_object(
            item["strike_circular_absolute_difference"],
            "strike summary",
        ),
        _json_csv_object(item["dip_absolute_difference"], "dip summary"),
        _json_csv_object(
            item["normal_vector_angular_difference"],
            "normal summary",
        ),
    )


def _runtime_row(item: Mapping[str, str]) -> StageResourceRow:
    return StageResourceRow(
        _integer_text(item["schema_version"], "runtime schema_version"),
        item["stage_kind"],
        item["fingerprint"],
        _bool_text(item["computed"], "runtime computed"),
        item["state"],  # type: ignore[arg-type]
        tuple(_json_csv_list(item["cell_consumers"], "cell_consumers")),
        item["cell"],
        _float_text(item["elapsed_seconds"], "elapsed_seconds"),
        item["elapsed_semantics"],
        _integer_text(item["input_bytes"], "input_bytes"),
        _integer_text(item["output_bytes"], "output_bytes"),
        _integer_text(item["voxel_count"], "voxel_count"),
        _optional_float_text(
            item["voxel_throughput_per_second"],
            "voxel_throughput_per_second",
        ),
        item["interpretation"],
    )


def _rss_row(value: Any) -> RSSSnapshot:
    item = _object(value, "RSS row")
    if set(item) != {field.name for field in fields(RSSSnapshot)}:
        raise F3ResultValidationError("RSS row field set mismatch")
    return RSSSnapshot(
        item["schema_version"],
        item["scope"],
        item["point"],
        item["value_bytes"],
        item["status"],
        item["source"],
        item["semantics"],
    )


def _storage_row(value: Any) -> StorageRow:
    item = _object(value, "storage row")
    if set(item) != {field.name for field in fields(StorageRow)}:
        raise F3ResultValidationError("storage row field set mismatch")
    return StorageRow(
        item["schema_version"],
        item["scope"],
        item["stage_kind"],
        item["fingerprint"],
        item["logical_bytes"],
        item["actual_file_bytes"],
        item["allocated_bytes"],
        item["file_count"],
        item["reference_files_are_not_dereferenced"],
    )


def _csv_bytes(rows: Sequence[Any], model: type[Any]) -> bytes:
    output = io.StringIO(newline="")
    names = tuple(field.name for field in fields(model))
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(names)
    for row in rows:
        if not isinstance(row, model):
            raise TypeError(f"CSV rows must contain only {model.__name__}")
        writer.writerow(_csv_value(getattr(row, name)) for name in names)
    return output.getvalue().encode("utf-8")


def _csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("CSV numeric values must be finite")
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (tuple, list, dict, Mapping)):
        return canonical_json_bytes(value).decode("utf-8")
    raise TypeError(f"unsupported CSV value: {type(value).__name__}")


def _read_csv(path: Path, model: type[Any]) -> tuple[dict[str, str], ...]:
    if not path.is_file() or path.is_symlink():
        raise F3ResultValidationError(f"{path.name} must be a regular non-symlink file")
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise F3ResultValidationError(f"{path.name} is not UTF-8") from error
    if "\r" in text or (text and not text.endswith("\n")):
        raise F3ResultValidationError(f"{path.name} must use LF and end with LF")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    expected = [field.name for field in fields(model)]
    if reader.fieldnames != expected:
        raise F3ResultValidationError(f"{path.name} CSV header mismatch")
    rows = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise F3ResultValidationError(f"{path.name} malformed CSV row")
        rows.append(dict(row))
    return tuple(rows)


def _verify_file_metadata(path: Path, value: Any, context: str) -> None:
    record = _object(value, f"{context} metadata")
    if set(record) != {"sha256", "size"}:
        raise F3ResultValidationError(f"{context} metadata field set mismatch")
    _sha256(record["sha256"], f"{context} digest")
    size = record["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise F3ResultValidationError(f"{context} recorded size is invalid")
    if artifact_file_metadata(path) != record:
        raise F3ResultValidationError(f"{context} hash or size mismatch")


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise F3ResultValidationError(f"{context} must be a regular non-symlink file")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise F3ResultValidationError(f"{context} is missing or unreadable") from error
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: _raise_nonfinite(token),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise F3ResultValidationError(f"{context} is not strict JSON") from error
    if not isinstance(value, dict):
        raise F3ResultValidationError(f"{context} must be an object")
    _finite_json_value(value, context, allow_none=True)
    return value


def _raise_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def _finite_json_value(value: Any, context: str, *, allow_none: bool = False) -> None:
    if value is None:
        if allow_none:
            return
        raise F3ResultValidationError(f"{context} contains null")
    if isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise F3ResultValidationError(f"{context} contains a non-finite number")
        return
    if isinstance(value, list | tuple):
        for item in value:
            _finite_json_value(item, context, allow_none=allow_none)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise F3ResultValidationError(f"{context} has a non-string key")
            _finite_json_value(item, context, allow_none=allow_none)
        return
    raise F3ResultValidationError(f"{context} contains unsupported values")


def _reject_crop_semantics(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _has_forbidden_volume_semantics(key):
                raise F3ResultValidationError("crop/tile/center dimensions are forbidden")
            _reject_crop_semantics(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_crop_semantics(item)
    elif isinstance(value, str) and not value.startswith(("/", "./", "../")):
        if _has_forbidden_volume_semantics(value, key=False):
            raise F3ResultValidationError("crop/tile/center dimensions are forbidden")


def _has_forbidden_volume_semantics(value: Any, *, key: bool = True) -> bool:
    if not isinstance(value, str):
        return False
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    tokens = {token for token in re.split(r"[^a-z0-9]+", camel_split.lower()) if token}
    if tokens & {"crop", "cropped", "cropping", "tile", "tiled", "tiling"}:
        return True
    if "center" in tokens and (len(tokens) == 1 or bool(tokens & {"dimension", "dimensions"})):
        return True
    return key and "replicate" in tokens and "index" in tokens


def _reject_volume_bearing_values(result: F3ModeComparisonResult) -> None:
    _reject_forbidden_object(result.as_dict())
    for name in (
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
    ):
        for value in getattr(result, name):
            if isinstance(value, (np.ndarray, np.memmap, io.IOBase)):
                raise ValueError("result object graph must not retain arrays or open files")


def _reject_forbidden_object(value: Any) -> None:
    if isinstance(value, (np.ndarray, np.memmap, io.IOBase)):
        raise ValueError("result object graph must not retain arrays or open files")
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_forbidden_object(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_forbidden_object(item)


def _workspace_path(value: F3RunWorkspace | str | os.PathLike[str]) -> Path:
    root = value.path if isinstance(value, F3RunWorkspace) else Path(value)
    _require_directory(root, "run workspace")
    return root


def _require_directory(path: Path, context: str) -> None:
    if not path.is_dir() or path.is_symlink():
        raise F3ResultValidationError(f"{context} must be a non-symlink directory")


def _cleanup_report_temporaries(path: Path) -> None:
    if not path.is_dir():
        return
    for item in path.iterdir():
        if any(item.name.startswith(f".{filename}.tmp-") for filename in F3_REPORT_FILES):
            item.unlink(missing_ok=True)


def _shape3(value: Any) -> tuple[int, int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value)
    ):
        raise ValueError("volume_shape must contain three positive integers")
    return value


def _shape3_list(value: Any) -> tuple[int, int, int]:
    if not isinstance(value, list):
        raise F3ResultValidationError("shape must be a JSON array")
    try:
        return _shape3(tuple(value))
    except ValueError as error:
        raise F3ResultValidationError(str(error)) from error


def _sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise F3ResultValidationError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise F3ResultValidationError(f"{context} must be an object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise F3ResultValidationError(f"{context} must be an array")
    return value


def _boolean(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise F3ResultValidationError(f"{context} must be bool")
    return value


def _number_mapping(value: Any, context: str) -> dict[str, float]:
    item = _object(value, context)
    output = {}
    for name, number in item.items():
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise F3ResultValidationError(f"{context} values must be numbers")
        normalized = float(number)
        if not math.isfinite(normalized):
            raise F3ResultValidationError(f"{context} values must be finite")
        output[name] = normalized
    return output


def _integer_mapping(value: Any, context: str) -> dict[str, int]:
    item = _object(value, context)
    output = {}
    for name, number in item.items():
        if isinstance(number, bool) or not isinstance(number, int) or number < 0:
            raise F3ResultValidationError(f"{context} values must be non-negative integers")
        output[name] = number
    return output


def _integer_text(value: str, context: str) -> int:
    if not value or value.strip() != value:
        raise F3ResultValidationError(f"{context} is not a canonical integer")
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise F3ResultValidationError(f"{context} is not an integer") from error
    if str(parsed) != value:
        raise F3ResultValidationError(f"{context} is not a canonical integer")
    return parsed


def _float_text(value: str, context: str) -> float:
    if not value or value.strip() != value:
        raise F3ResultValidationError(f"{context} is not a canonical float")
    try:
        parsed = float(value)
    except ValueError as error:
        raise F3ResultValidationError(f"{context} is not a float") from error
    if not math.isfinite(parsed) or repr(parsed) != value:
        raise F3ResultValidationError(f"{context} is not a canonical finite float")
    return parsed


def _optional_float_text(value: str, context: str) -> float | None:
    return None if value == "" else _float_text(value, context)


def _bool_text(value: str, context: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise F3ResultValidationError(f"{context} is not a canonical boolean")


def _json_csv_list(value: str, context: str) -> list[Any]:
    parsed = _json_csv(value, context)
    if not isinstance(parsed, list):
        raise F3ResultValidationError(f"{context} must encode an array")
    return parsed


def _json_csv_object(value: str, context: str) -> dict[str, Any]:
    parsed = _json_csv(value, context)
    if not isinstance(parsed, dict):
        raise F3ResultValidationError(f"{context} must encode an object")
    return parsed


def _json_csv(value: str, context: str) -> Any:
    try:
        parsed = json.loads(value, parse_constant=lambda token: _raise_nonfinite(token))
    except (json.JSONDecodeError, ValueError) as error:
        raise F3ResultValidationError(f"{context} is not canonical JSON") from error
    if canonical_json_bytes(parsed).decode("utf-8") != value:
        raise F3ResultValidationError(f"{context} is not canonical JSON")
    return parsed


def _optional_number_equal(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(
        left,
        right,
        rel_tol=_FLOAT_REL_TOL,
        abs_tol=_FLOAT_ABS_TOL,
    )


def _require_close(actual: Any, expected: float) -> None:
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isclose(
            float(actual),
            expected,
            rel_tol=_FLOAT_REL_TOL,
            abs_tol=_FLOAT_ABS_TOL,
        )
    ):
        raise F3ResultValidationError("derived regional scalar mismatch")


def _count_value(metrics: Mapping[str, Any], name: str) -> int:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise F3ResultValidationError(f"{name} must be a non-negative integer")
    return value


def _metric_prefixes(metrics: Mapping[str, Any], suffix: str) -> tuple[str, ...]:
    return tuple(name[: -len(suffix)] for name in metrics if name.endswith(suffix))


def _close_memmap(value: np.memmap | None) -> None:
    if value is None:
        return
    mapping = getattr(value, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


__all__ = [
    "CELLS_REPORT_FILE",
    "CONTRASTS_REPORT_FILE",
    "F3_COMPLETION_SCHEMA_VERSION",
    "F3_REPORT_FILES",
    "F3_RESULT_INTERPRETATION",
    "F3_RESULT_SCHEMA_VERSION",
    "F3ModeComparisonResult",
    "F3ResultValidationError",
    "METRICS_REPORT_FILE",
    "METRIC_EVIDENCE_REPORT_FILE",
    "ORIENTATION_REPORT_FILE",
    "REGIONAL_REPORT_FILE",
    "RESOURCES_REPORT_FILE",
    "RUNTIME_REPORT_FILE",
    "VOXEL_CONTRASTS_REPORT_FILE",
    "finalize_f3d_bundle",
    "finalize_f3d_mode_comparison_result",
    "load_f3d_mode_comparison_result",
    "load_f3d_result_bundle",
    "validate_completed_f3d_bundle",
    "validate_f3d_mode_comparison_result",
    "validate_f3d_result",
]
