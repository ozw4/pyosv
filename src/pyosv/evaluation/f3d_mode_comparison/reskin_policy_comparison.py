"""Skin-only F3 comparison from one immutable upstream parent."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pyosv.candidate_volume import NONZERO_EPSILON, positive_candidate_mask
from pyosv.evaluation.synthetic_quality import SyntheticSkinningConfig
from pyosv.evaluation.synthetic_quality.variants import (
    SkinningPatch,
    VariantSpec,
    VotingPatch,
    effective_skinning_config,
)
from pyosv.evaluation.workflow3d import execute_skinning_phase3d
from pyosv.skinner import (
    RESKIN_POLICY_EXISTING_CELLS_V1,
    RESKIN_POLICY_REFERENCE_DENSE_V1,
)
from pyosv.synthetic_metrics import (
    SURFACE_METRIC_EVIDENCE_SCHEMA_VERSION,
    metrics_from_surface_evidence,
    reskin_generation_metrics,
    rounded_duplicate_cell_count,
    skin_mask_from_skins,
    skin_link_topology_metrics,
    skin_topology_metrics,
    surface_comparison_metrics_with_evidence,
)

from .artifacts import (
    F3_FINGERPRINT_CONTRACT_VERSION,
    RUN_MANIFEST_FILE,
    artifact_file_metadata,
    atomic_write_artifact,
    canonical_fingerprint,
    canonical_json_bytes,
    implementation_identity,
)
from .data import F3_DATASET_ID
from .runtime_identity import (
    numerical_runtime_identity,
    validate_numerical_runtime_identity,
    validate_publication_runtime_identity,
)
from .skin_artifacts import (
    CONNECTED_COMPONENT_FALLBACK,
    F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
    PRIMARY_DENSE_RESKINNED,
    PRIMARY_EXISTING_CELLS_RESKINNED,
    ParsedSkinArtifacts,
    canonical_skins_payload,
    parse_skins_json,
    validate_skin_generation_provenance,
    validate_reskin_diagnostics_contract,
)

F3_RESKIN_POLICY_COMPARISON_SCHEMA_VERSION = 4
F3_RESKIN_POLICY_COMPARISON_COMPLETION_SCHEMA_VERSION = 4
F3_RESKIN_POLICY_CONFIG_SCHEMA_VERSION = 1
F3_RESKIN_POLICY_COMPARISON_FILES = (
    "reskin_policy_comparison.json",
    "reskin_policy_metrics.csv",
    "reskin_policy_comparison.md",
    "existing_cells_v1_skins.json",
    "reference_dense_v1_skins.json",
)
F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE = "complete.json"
F3_RESKIN_POLICY_COMPARISON_DIR = "reskin_policy_comparison"
F3_RESKIN_POLICY_COMPARISON_CELL = "Q-QUAL"
_F3_CANONICAL_VARIANT = VariantSpec("f3-canonical", experimental=False)
_CONTRAST_FIELDS = (
    "generated_cell_count_delta",
    "output_cell_count_delta",
    "small_skin_cell_fraction_delta",
    "parent_ridge_buffered_precision_delta",
    "parent_ridge_buffered_recall_delta",
    "parent_ridge_symmetric_chamfer_mean_delta",
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
_SHA256_LENGTH = 64


@dataclass(frozen=True, slots=True)
class _SourceBundleIdentity:
    run_fingerprint: str
    dataset_identity: dict[str, Any]
    runtime_identity: dict[str, Any]
    runtime_identity_schema_version: int
    runtime_identity_sha256: str
    official: bool


def compare_reskin_policies_from_parent(
    *,
    fv: np.ndarray,
    fvt: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    skinning_config: SyntheticSkinningConfig,
    variant_spec: VariantSpec = _F3_CANONICAL_VARIANT,
    scanner_target_positive_mask: np.ndarray | None = None,
    scanner_target_mask_source: str | None = None,
) -> dict[str, Any]:
    """Branch only skinning policy while retaining one exact parent fingerprint."""

    arrays = _validated_parent_arrays(fv=fv, fvt=fvt, vp=vp, vt=vt)
    if not isinstance(skinning_config, SyntheticSkinningConfig):
        raise ValueError("skinning_config must be a SyntheticSkinningConfig")
    if not skinning_config.enabled:
        raise ValueError("skinning_config.enabled must be true")
    if skinning_config.method != "quality":
        raise ValueError("skinning_config.method must be 'quality'")
    if not skinning_config.reskin:
        raise ValueError("skinning_config.reskin must be true")
    if not isinstance(variant_spec, VariantSpec):
        raise ValueError("variant_spec must be a VariantSpec")
    if scanner_target_mask_source is None:
        scanner_target_mask_source = (
            "provided_parent_mask" if scanner_target_positive_mask is not None else "not_provided"
        )
    if not isinstance(scanner_target_mask_source, str) or not scanner_target_mask_source:
        raise ValueError("scanner_target_mask_source must be a non-empty string")
    if scanner_target_positive_mask is None and scanner_target_mask_source != "not_provided":
        raise ValueError("scanner target mask source requires a provided mask")
    if scanner_target_positive_mask is not None and scanner_target_mask_source == "not_provided":
        raise ValueError("provided scanner target mask requires its source")
    effective_configs = _effective_comparison_skinning_configs(
        skinning_config,
        variant_spec,
    )
    resolved_config = _resolved_comparison_config(
        skinning_config,
        variant_spec,
        scanner_target_mask_source=scanner_target_mask_source,
        effective_configs=effective_configs,
    )
    comparison_config_fingerprint = canonical_fingerprint(resolved_config)

    scanner_mask = _validated_parent_mask(
        scanner_target_positive_mask,
        shape=arrays["fv"].shape,
    )
    parent_fingerprint = _parent_fingerprint(
        arrays,
        scanner_target_positive_mask=scanner_mask,
    )
    parent_ridge = positive_candidate_mask(
        arrays["fvt"],
        epsilon=NONZERO_EPSILON,
    )
    policies: dict[str, Any] = {}
    skin_masks: dict[str, np.ndarray] = {}
    for policy in (
        RESKIN_POLICY_EXISTING_CELLS_V1,
        RESKIN_POLICY_REFERENCE_DENSE_V1,
    ):
        skinning_result = execute_skinning_phase3d(
            fv=arrays["fv"],
            fvt=arrays["fvt"],
            vp=arrays["vp"],
            vt=arrays["vt"],
            skinning_settings=effective_configs[policy],
            variant_spec=variant_spec,
            scanner_target_positive_mask=scanner_mask,
        )
        skins = list(skinning_result.skins)
        diagnostics = _mutable_scalar_evidence(skinning_result.diagnostics)
        reskin_diagnostics = _final_reskin_diagnostics(diagnostics)
        skin_mask = skin_mask_from_skins(skins, arrays["fv"].shape)
        skin_masks[policy] = skin_mask
        policies[policy] = {
            "reskin_policy": policy,
            "parent_fingerprint": parent_fingerprint,
            "canonical_skin_artifact": canonical_skins_payload(skins),
            "generation": reskin_generation_metrics(
                skins,
                diagnostics=reskin_diagnostics,
            ),
            "link_topology": skin_link_topology_metrics(skins),
            "skin_topology": skin_topology_metrics(
                skins,
                arrays["fv"].shape,
                small_skin_size=skinning_config.small_skin_size,
            ),
            "parent_ridge_surface": {
                "reference": "shared_parent_fvt_positive",
                "positive_epsilon": NONZERO_EPSILON,
                "ridge_voxel_count": int(np.count_nonzero(parent_ridge)),
            },
            "duplicate_rounded_cell_index_count": rounded_duplicate_cell_count(skins),
            "diagnostics": diagnostics,
        }
    surface_results = surface_comparison_metrics_with_evidence(
        skin_masks,
        parent_ridge,
        radius=2.0,
        positive_epsilon=NONZERO_EPSILON,
    )
    for policy, result in surface_results.items():
        policies[policy]["parent_ridge_surface"].update(result)

    baseline = policies[RESKIN_POLICY_EXISTING_CELLS_V1]
    candidate = policies[RESKIN_POLICY_REFERENCE_DENSE_V1]
    return {
        "schema_version": F3_RESKIN_POLICY_COMPARISON_SCHEMA_VERSION,
        "baseline": RESKIN_POLICY_EXISTING_CELLS_V1,
        "candidate": RESKIN_POLICY_REFERENCE_DENSE_V1,
        "upstream_parent_fingerprint": parent_fingerprint,
        "upstream_parent_fingerprint_identical": (
            baseline["parent_fingerprint"] == candidate["parent_fingerprint"]
        ),
        "resolved_config": resolved_config,
        "comparison_config_fingerprint": comparison_config_fingerprint,
        "metric_evidence_schema_version": SURFACE_METRIC_EVIDENCE_SCHEMA_VERSION,
        "validation_level": "shallow",
        "policies": policies,
        "contrast": _comparison_contrast(policies),
    }


def write_f3_reskin_policy_comparison(
    report: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Persist the dedicated comparison and both canonical skin artifacts."""

    if report.get("schema_version") != F3_RESKIN_POLICY_COMPARISON_SCHEMA_VERSION:
        raise ValueError(
            f"report schema_version must be {F3_RESKIN_POLICY_COMPARISON_SCHEMA_VERSION}"
        )
    policies = report.get("policies")
    if not isinstance(policies, Mapping) or set(policies) != {
        RESKIN_POLICY_EXISTING_CELLS_V1,
        RESKIN_POLICY_REFERENCE_DENSE_V1,
    }:
        raise ValueError("report must contain the fixed baseline and candidate policies")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / F3_RESKIN_POLICY_COMPARISON_FILES[0]
    csv_path = root / F3_RESKIN_POLICY_COMPARISON_FILES[1]
    markdown_path = root / F3_RESKIN_POLICY_COMPARISON_FILES[2]
    baseline_skin_path = root / F3_RESKIN_POLICY_COMPARISON_FILES[3]
    candidate_skin_path = root / F3_RESKIN_POLICY_COMPARISON_FILES[4]
    atomic_write_artifact(
        json_path,
        canonical_json_bytes(report) + b"\n",
        temporary_prefix=".reskin_policy_comparison.json.tmp-",
    )
    atomic_write_artifact(
        csv_path,
        _comparison_csv(report).encode("utf-8"),
        temporary_prefix=".reskin_policy_metrics.csv.tmp-",
    )
    atomic_write_artifact(
        markdown_path,
        _comparison_markdown(report).encode("utf-8"),
        temporary_prefix=".reskin_policy_comparison.md.tmp-",
    )
    atomic_write_artifact(
        baseline_skin_path,
        canonical_json_bytes(policies[RESKIN_POLICY_EXISTING_CELLS_V1]["canonical_skin_artifact"])
        + b"\n",
        temporary_prefix=".existing_cells_v1_skins.json.tmp-",
    )
    atomic_write_artifact(
        candidate_skin_path,
        canonical_json_bytes(policies[RESKIN_POLICY_REFERENCE_DENSE_V1]["canonical_skin_artifact"])
        + b"\n",
        temporary_prefix=".reference_dense_v1_skins.json.tmp-",
    )
    return (
        json_path,
        csv_path,
        markdown_path,
        baseline_skin_path,
        candidate_skin_path,
    )


def compare_reskin_policies_from_bundle(
    bundle: str | Path,
    *,
    output_dir: str | Path | None = None,
    resume: bool = False,
    deep: bool = False,
) -> tuple[Path, Path, Path, Path, Path]:
    """Run or reuse the fixed skin-only pair from canonical Q-QUAL parents.

    A deep-complete ``resume=True, deep=True`` checks saved deep evidence only;
    explicit :func:`validate_f3_reskin_policy_comparison` with ``deep=True``
    remains the current-runtime exact-replay path.
    """

    # Keep result loading local: package __init__ imports this module before result.py.
    from .result import load_f3d_mode_comparison_result

    if not isinstance(resume, bool) or not isinstance(deep, bool):
        raise TypeError("resume and deep must be bool")
    root = Path(bundle)
    result = load_f3d_mode_comparison_result(root, deep=False)
    source_identity = _load_source_bundle_identity(root)
    destination = (
        Path(output_dir) if output_dir is not None else root / F3_RESKIN_POLICY_COMPARISON_DIR
    )
    completion_path = destination / F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE
    if completion_path.is_file() and not completion_path.is_symlink():
        if not resume:
            raise FileExistsError(f"completed F3 reskin policy comparison exists: {destination}")
        paths = validate_f3_reskin_policy_comparison(
            root,
            output_dir=destination,
            result=result,
        )
        if deep:
            completion = _read_json_object(
                completion_path,
                "reskin policy comparison completion",
            )
            if completion["validation_level"] == "deep":
                validate_f3_reskin_policy_comparison(
                    root,
                    output_dir=destination,
                    require_deep=True,
                    result=result,
                )
                return paths
            validate_f3_reskin_policy_comparison(
                root,
                output_dir=destination,
                deep=True,
                result=result,
            )
            return _promote_comparison_completion(
                destination,
                validator=lambda: validate_f3_reskin_policy_comparison(
                    root,
                    output_dir=destination,
                    result=result,
                ),
            )
        return paths
    comparison_runtime_sha256 = _validate_current_comparison_runtime(source_identity)
    comparison_implementation = _comparison_implementation_identity()
    if destination.exists() and (not destination.is_dir() or destination.is_symlink()):
        raise ValueError("reskin policy comparison output must be a directory")
    if destination.is_dir() and any(destination.iterdir()):
        if not resume:
            raise FileExistsError(f"reskin policy comparison output exists: {destination}")
        allowed_partial_files = {
            *F3_RESKIN_POLICY_COMPARISON_FILES,
            F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE,
        }
        for item in destination.iterdir():
            is_owned_completion_temporary = item.name.startswith(".complete.json.tmp-")
            if (
                (item.name not in allowed_partial_files and not is_owned_completion_temporary)
                or not item.is_file()
                or item.is_symlink()
            ):
                raise ValueError("incomplete reskin policy comparison file set is invalid")

    report, cell, shape = _comparison_report_from_bundle(
        root,
        result,
        source_identity=source_identity,
        comparison_runtime_sha256=comparison_runtime_sha256,
        comparison_implementation=comparison_implementation,
    )
    created_parents = _create_missing_parents(destination.parent)
    staging: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{destination.name}.generation-tmp-",
                dir=destination.parent,
            )
        )
        staged_paths = write_f3_reskin_policy_comparison(report, staging)
        comparison_implementation_sha256 = canonical_fingerprint(comparison_implementation)
        completion = {
            "completion_schema_version": F3_RESKIN_POLICY_COMPARISON_COMPLETION_SCHEMA_VERSION,
            "comparison_schema_version": F3_RESKIN_POLICY_COMPARISON_SCHEMA_VERSION,
            "status": "complete",
            "source_cell": F3_RESKIN_POLICY_COMPARISON_CELL,
            "source_run_fingerprint": source_identity.run_fingerprint,
            "source_runtime_identity_sha256": source_identity.runtime_identity_sha256,
            "comparison_runtime_identity_sha256": comparison_runtime_sha256,
            "comparison_implementation_identity_sha256": comparison_implementation_sha256,
            "comparison_config_fingerprint": report["comparison_config_fingerprint"],
            "metric_evidence_schema_version": SURFACE_METRIC_EVIDENCE_SCHEMA_VERSION,
            "report_semantic_evidence_sha256": _report_semantic_evidence_digest(report),
            "validation_level": report["validation_level"],
            "upstream_parent_stage_fingerprints": {
                "scanner": cell.stages.scanner,
                "voting": cell.stages.voting,
                "thinning": cell.stages.thinning,
            },
            "artifact_files": {path.name: artifact_file_metadata(path) for path in staged_paths},
        }
        atomic_write_artifact(
            staging / F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE,
            canonical_json_bytes(completion) + b"\n",
            temporary_prefix=".complete.json.tmp-",
        )
        validate_f3_reskin_policy_comparison(
            root,
            output_dir=staging,
            deep=deep,
            result=result,
            _shape=shape,
        )
        if deep:
            _promote_comparison_completion(
                staging,
                validator=lambda: validate_f3_reskin_policy_comparison(
                    root,
                    output_dir=staging,
                    result=result,
                    _shape=shape,
                ),
            )
        _publish_comparison_directory(staging, destination)
        return tuple(destination / name for name in F3_RESKIN_POLICY_COMPARISON_FILES)
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        for parent in created_parents:
            try:
                parent.rmdir()
            except OSError:
                break


def _create_missing_parents(parent: Path) -> tuple[Path, ...]:
    missing: list[Path] = []
    current = parent
    while not current.exists():
        missing.append(current)
        current = current.parent
    parent.mkdir(parents=True, exist_ok=True)
    return tuple(missing)


def _publish_comparison_directory(staging: Path, destination: Path) -> None:
    if not destination.exists():
        os.rename(staging, destination)
        return
    backup = staging.with_name(f"{staging.name}-previous")
    os.rename(destination, backup)
    try:
        os.rename(staging, destination)
    except BaseException:
        os.rename(backup, destination)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def validate_f3_reskin_policy_comparison(
    bundle: str | Path,
    *,
    output_dir: str | Path | None = None,
    deep: bool = False,
    require_deep: bool = False,
    result: Any | None = None,
    _shape: tuple[int, int, int] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    """Validate completion and optionally exact-replay both skin artifacts."""

    if not isinstance(deep, bool) or not isinstance(require_deep, bool):
        raise TypeError("deep and require_deep must be bool")
    root = Path(bundle)
    destination = (
        Path(output_dir) if output_dir is not None else root / F3_RESKIN_POLICY_COMPARISON_DIR
    )
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError("reskin policy comparison directory is invalid")
    expected_entries = {
        *F3_RESKIN_POLICY_COMPARISON_FILES,
        F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE,
    }
    if {item.name for item in destination.iterdir()} != expected_entries:
        raise ValueError("reskin policy comparison file set mismatch")

    completion_path = destination / F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE
    completion = _read_json_object(completion_path, "reskin policy comparison completion")
    expected_completion_fields = {
        "completion_schema_version",
        "comparison_schema_version",
        "status",
        "source_cell",
        "source_run_fingerprint",
        "source_runtime_identity_sha256",
        "comparison_runtime_identity_sha256",
        "comparison_implementation_identity_sha256",
        "comparison_config_fingerprint",
        "metric_evidence_schema_version",
        "report_semantic_evidence_sha256",
        "validation_level",
        "upstream_parent_stage_fingerprints",
        "artifact_files",
    }
    if set(completion) != expected_completion_fields:
        raise ValueError("reskin policy comparison completion field set mismatch")
    if (
        completion["completion_schema_version"]
        != F3_RESKIN_POLICY_COMPARISON_COMPLETION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported reskin policy comparison completion schema")
    if completion["comparison_schema_version"] != F3_RESKIN_POLICY_COMPARISON_SCHEMA_VERSION:
        raise ValueError("reskin policy comparison schema mismatch")
    if completion["status"] != "complete":
        raise ValueError("reskin policy comparison status must be 'complete'")
    if completion["source_cell"] != F3_RESKIN_POLICY_COMPARISON_CELL:
        raise ValueError("reskin policy comparison source cell mismatch")
    if completion["validation_level"] not in {"shallow", "deep"}:
        raise ValueError("reskin policy comparison validation level is invalid")
    if require_deep and completion["validation_level"] != "deep":
        raise ValueError(
            "reskin policy comparison is not publication-ready: deep validation required"
        )
    if completion["metric_evidence_schema_version"] != SURFACE_METRIC_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("reskin policy comparison metric evidence schema mismatch")

    source_identity = _load_source_bundle_identity(root)
    if result is None:
        from .result import load_f3d_mode_comparison_result

        result = load_f3d_mode_comparison_result(root, deep=False)
    cell, config, variant = _comparison_cell_and_config(result)
    expected_parent_stages = {
        "scanner": cell.stages.scanner,
        "voting": cell.stages.voting,
        "thinning": cell.stages.thinning,
    }
    if completion["upstream_parent_stage_fingerprints"] != expected_parent_stages:
        raise ValueError("reskin policy comparison parent stage mismatch")

    paths = tuple(destination / filename for filename in F3_RESKIN_POLICY_COMPARISON_FILES)
    report = _read_json_object(paths[0], "reskin policy comparison report")
    comparison_implementation = _comparison_implementation_identity()
    comparison_implementation_sha256 = canonical_fingerprint(comparison_implementation)
    _validate_comparison_provenance(
        report,
        completion,
        source_identity=source_identity,
        comparison_implementation=comparison_implementation,
        comparison_implementation_sha256=comparison_implementation_sha256,
    )

    metadata = completion["artifact_files"]
    if not isinstance(metadata, Mapping) or set(metadata) != set(F3_RESKIN_POLICY_COMPARISON_FILES):
        raise ValueError("reskin policy comparison artifact metadata mismatch")
    for path in paths:
        if artifact_file_metadata(path) != metadata[path.name]:
            raise ValueError(
                f"reskin policy comparison artifact hash or size mismatch: {path.name}"
            )

    expected_report_fields = {
        "schema_version",
        "baseline",
        "candidate",
        "upstream_parent_fingerprint",
        "upstream_parent_fingerprint_identical",
        "resolved_config",
        "comparison_config_fingerprint",
        "metric_evidence_schema_version",
        "validation_level",
        "policies",
        "contrast",
        "source_cell",
        "source_run_fingerprint",
        "source_runtime_identity_schema_version",
        "source_runtime_identity_sha256",
        "comparison_runtime_identity_sha256",
        "comparison_implementation_identity",
        "upstream_parent_stage_fingerprints",
    }
    if set(report) != expected_report_fields:
        raise ValueError("reskin policy comparison report field set mismatch")
    if report["schema_version"] != F3_RESKIN_POLICY_COMPARISON_SCHEMA_VERSION:
        raise ValueError("reskin policy comparison report schema mismatch")
    if (
        report["baseline"] != RESKIN_POLICY_EXISTING_CELLS_V1
        or report["candidate"] != RESKIN_POLICY_REFERENCE_DENSE_V1
    ):
        raise ValueError("reskin policy comparison pair mismatch")
    if report["source_cell"] != F3_RESKIN_POLICY_COMPARISON_CELL:
        raise ValueError("reskin policy comparison report source cell mismatch")
    if report["upstream_parent_stage_fingerprints"] != expected_parent_stages:
        raise ValueError("reskin policy comparison report parent stage mismatch")
    if report["upstream_parent_fingerprint_identical"] is not True:
        raise ValueError("reskin policy comparison parent fingerprint differs")
    if report["validation_level"] not in {"shallow", "deep"}:
        raise ValueError("reskin policy comparison report validation level is invalid")
    if report["validation_level"] != completion["validation_level"]:
        raise ValueError("reskin policy comparison validation level mismatch")
    expected_resolved_config = _resolved_comparison_config(config, variant)
    resolved_config = report["resolved_config"]
    if not isinstance(resolved_config, Mapping):
        raise ValueError("reskin policy comparison resolved config must be an object")
    if resolved_config.get("schema_version") != F3_RESKIN_POLICY_CONFIG_SCHEMA_VERSION:
        raise ValueError("reskin policy comparison resolved config schema mismatch")
    _validate_effective_skinning_configs(resolved_config.get("effective_skinning_configs"))
    if report["resolved_config"] != expected_resolved_config:
        raise ValueError("reskin policy comparison resolved config mismatch")
    expected_config_fingerprint = canonical_fingerprint(expected_resolved_config)
    if report["comparison_config_fingerprint"] != expected_config_fingerprint:
        raise ValueError("reskin policy comparison config fingerprint mismatch")
    if completion["comparison_config_fingerprint"] != expected_config_fingerprint:
        raise ValueError("reskin policy comparison completion config fingerprint mismatch")
    if report["metric_evidence_schema_version"] != SURFACE_METRIC_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("reskin policy comparison report metric evidence schema mismatch")
    if completion["report_semantic_evidence_sha256"] != _report_semantic_evidence_digest(report):
        raise ValueError("reskin policy comparison semantic evidence digest mismatch")
    contrast = report.get("contrast")
    if not isinstance(contrast, Mapping) or set(contrast) != set(_CONTRAST_FIELDS):
        raise ValueError("reskin policy comparison contrast field set mismatch")

    policies = report.get("policies")
    if not isinstance(policies, Mapping) or set(policies) != {
        RESKIN_POLICY_EXISTING_CELLS_V1,
        RESKIN_POLICY_REFERENCE_DENSE_V1,
    }:
        raise ValueError("reskin policy comparison policies mismatch")
    shape = _shape
    if shape is None:
        voting_path = root / "stages" / "voting" / cell.stages.voting
        shape = _stage_shape(voting_path / "report.json")
    for policy, path, provenance in (
        (
            RESKIN_POLICY_EXISTING_CELLS_V1,
            paths[3],
            PRIMARY_EXISTING_CELLS_RESKINNED,
        ),
        (
            RESKIN_POLICY_REFERENCE_DENSE_V1,
            paths[4],
            PRIMARY_DENSE_RESKINNED,
        ),
    ):
        item = policies[policy]
        if not isinstance(item, Mapping):
            raise ValueError("reskin policy comparison policy payload must be an object")
        if item.get("reskin_policy") != policy:
            raise ValueError("reskin policy comparison policy identity mismatch")
        if resolved_config["effective_skinning_configs"][policy]["reskin_policy"] != item.get(
            "reskin_policy"
        ):
            raise ValueError("reskin policy comparison config/policy identity mismatch")
        if item.get("parent_fingerprint") != report["upstream_parent_fingerprint"]:
            raise ValueError("reskin policy comparison parent fingerprint mismatch")
        canonical_artifact = item.get("canonical_skin_artifact")
        if canonical_artifact != _read_json_object(path, f"{policy} canonical skins"):
            raise ValueError("reskin policy comparison canonical skin artifact mismatch")
        parsed = parse_skins_json(path, shape)
        diagnostics = item.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ValueError("reskin policy comparison diagnostics must be an object")
        _validate_reported_policy_metrics(
            item,
            parsed,
            shape=shape,
            small_skin_size=config.small_skin_size,
        )
        fallback_used = diagnostics.get("fallback_used")
        if not isinstance(fallback_used, bool):
            raise ValueError("reskin policy comparison fallback_used must be a bool")
        reskin_diagnostics = diagnostics.get("reskin")
        if not isinstance(reskin_diagnostics, Mapping):
            raise ValueError("reskin policy comparison reskin diagnostics must be an object")
        validate_skin_generation_provenance(
            parsed,
            CONNECTED_COMPONENT_FALLBACK if fallback_used else provenance,
            semantic_contract_version=F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
            reskin_diagnostics=reskin_diagnostics,
        )

    if dict(contrast) != _comparison_contrast(policies):
        raise ValueError("reskin policy comparison contrast metrics mismatch")
    if paths[1].read_text(encoding="utf-8") != _comparison_csv(report):
        raise ValueError("reskin policy comparison CSV does not match report")
    if paths[2].read_text(encoding="utf-8") != _comparison_markdown(report):
        raise ValueError("reskin policy comparison Markdown does not match report")

    # ``require_deep`` attests to persisted completion evidence. Only the
    # explicit ``deep`` request performs a current-runtime replay.
    if deep:
        comparison_runtime_sha256 = _validate_current_comparison_runtime(source_identity)
        recomputed, _, _ = _comparison_report_from_bundle(
            root,
            result,
            source_identity=source_identity,
            comparison_runtime_sha256=comparison_runtime_sha256,
            comparison_implementation=comparison_implementation,
        )
        recomputed["validation_level"] = report["validation_level"]
        if canonical_json_bytes(recomputed) != canonical_json_bytes(report):
            raise ValueError(
                "reskin policy comparison does not exactly match skin-only recomputation"
            )
    return paths


def _comparison_cell_and_config(
    result: Any,
) -> tuple[Any, SyntheticSkinningConfig, VariantSpec]:
    cell = next(
        (item for item in result.cells if item.label == F3_RESKIN_POLICY_COMPARISON_CELL),
        None,
    )
    if cell is None:
        raise ValueError(f"bundle has no canonical {F3_RESKIN_POLICY_COMPARISON_CELL} cell")
    if cell.backend != "quality" or cell.workflow != "quality":
        raise ValueError("canonical reskin comparison cell must use quality scanner/workflow")
    if not cell.skinning_enabled:
        raise ValueError("canonical reskin comparison requires a skinning-enabled bundle")

    skinning = cell.resolved_config.get("skinning")
    variant = cell.resolved_config.get("variant")
    if not isinstance(skinning, Mapping) or not isinstance(variant, Mapping):
        raise ValueError("canonical comparison cell has no resolved skinning/variant config")
    config = SyntheticSkinningConfig(**dict(skinning))
    try:
        variant_spec = VariantSpec(
            name=variant["name"],
            voting=VotingPatch(**dict(variant["voting"])),
            seed_policy=variant["seed_policy"],
            thinning_policy=variant["thinning_policy"],
            post_thinning_policy=variant["post_thinning_policy"],
            skinning=SkinningPatch(**dict(variant["skinning"])),
            experimental=variant["experimental"],
            presets=tuple(variant["presets"]),
            baseline=variant["baseline"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("canonical comparison cell resolved variant is invalid") from error
    if config.method != "quality" or not config.reskin:
        raise ValueError(
            "canonical Q-QUAL skinning config must use method='quality' and reskin=True"
        )
    return cell, config, variant_spec


def _comparison_report_from_bundle(
    root: Path,
    result: Any,
    *,
    source_identity: _SourceBundleIdentity | None = None,
    comparison_runtime_sha256: str | None = None,
    comparison_implementation: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], Any, tuple[int, int, int]]:
    if source_identity is None:
        source_identity = _load_source_bundle_identity(root)
    if comparison_runtime_sha256 is None:
        comparison_runtime_sha256 = _validate_current_comparison_runtime(source_identity)
    if comparison_implementation is None:
        comparison_implementation = _comparison_implementation_identity()
    cell, config, variant = _comparison_cell_and_config(result)
    scanner_path = root / "stages" / "scanner" / cell.stages.scanner
    voting_path = root / "stages" / "voting" / cell.stages.voting
    thinning_path = root / "stages" / "thinning" / cell.stages.thinning
    shape = _stage_shape(voting_path / "report.json")
    if _stage_shape(scanner_path / "report.json") != shape:
        raise ValueError("scanner and voting parent shapes do not match")
    if _stage_shape(thinning_path / "report.json") != shape:
        raise ValueError("voting and thinning parent shapes do not match")

    arrays = (
        _open_parent_dat(voting_path / "fv.dat", shape),
        _open_parent_dat(thinning_path / "fvt.dat", shape),
        _open_parent_dat(voting_path / "vp.dat", shape),
        _open_parent_dat(voting_path / "vt.dat", shape),
        _open_parent_dat(scanner_path / "ft.dat", shape),
    )
    try:
        scanner_target_positive_mask = positive_candidate_mask(
            arrays[4],
            epsilon=NONZERO_EPSILON,
        )
        scanner_target_positive_mask.flags.writeable = False
        report = compare_reskin_policies_from_parent(
            fv=arrays[0],
            fvt=arrays[1],
            vp=arrays[2],
            vt=arrays[3],
            skinning_config=config,
            variant_spec=variant,
            scanner_target_positive_mask=scanner_target_positive_mask,
            scanner_target_mask_source="source_scanner_ft_positive",
        )
    finally:
        scanner_target_positive_mask = None
        for array in arrays:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()

    report["source_cell"] = F3_RESKIN_POLICY_COMPARISON_CELL
    report["source_run_fingerprint"] = source_identity.run_fingerprint
    report["source_runtime_identity_schema_version"] = (
        source_identity.runtime_identity_schema_version
    )
    report["source_runtime_identity_sha256"] = source_identity.runtime_identity_sha256
    report["comparison_runtime_identity_sha256"] = comparison_runtime_sha256
    report["comparison_implementation_identity"] = dict(comparison_implementation)
    report["upstream_parent_stage_fingerprints"] = {
        "scanner": cell.stages.scanner,
        "voting": cell.stages.voting,
        "thinning": cell.stages.thinning,
    }
    return report, cell, shape


def _resolved_comparison_config(
    skinning_config: SyntheticSkinningConfig,
    variant_spec: VariantSpec,
    *,
    scanner_target_mask_source: str = "source_scanner_ft_positive",
    effective_configs: Mapping[str, SyntheticSkinningConfig] | None = None,
) -> dict[str, Any]:
    shared = _mutable_scalar_evidence(asdict(skinning_config))
    if effective_configs is None:
        effective_configs = _effective_comparison_skinning_configs(
            skinning_config,
            variant_spec,
        )
    effective = {
        policy: _mutable_scalar_evidence(asdict(effective_configs[policy]))
        for policy in (
            RESKIN_POLICY_EXISTING_CELLS_V1,
            RESKIN_POLICY_REFERENCE_DENSE_V1,
        )
    }
    _validate_effective_skinning_configs(effective)
    return {
        "schema_version": F3_RESKIN_POLICY_CONFIG_SCHEMA_VERSION,
        "shared_skinning_config": shared,
        "resolved_variant": _mutable_scalar_evidence(asdict(variant_spec)),
        "scanner_target_mask": {
            "source": scanner_target_mask_source,
            "positive_epsilon": NONZERO_EPSILON,
        },
        "effective_skinning_configs": effective,
    }


def _effective_comparison_skinning_configs(
    skinning_config: SyntheticSkinningConfig,
    variant_spec: VariantSpec,
) -> dict[str, SyntheticSkinningConfig]:
    variant_effective = effective_skinning_config(variant_spec, skinning_config)
    if not variant_effective.enabled:
        raise ValueError("effective skinning config must be enabled")
    if variant_effective.method != "quality":
        raise ValueError("effective skinning config method must be 'quality'")
    if not variant_effective.reskin:
        raise ValueError("effective skinning config reskin must be true")
    return {
        policy: replace(variant_effective, reskin_policy=policy)
        for policy in (
            RESKIN_POLICY_EXISTING_CELLS_V1,
            RESKIN_POLICY_REFERENCE_DENSE_V1,
        )
    }


def _validate_effective_skinning_configs(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        RESKIN_POLICY_EXISTING_CELLS_V1,
        RESKIN_POLICY_REFERENCE_DENSE_V1,
    }:
        raise ValueError("effective skinning config policy coverage mismatch")
    without_policy: list[dict[str, Any]] = []
    for policy in (
        RESKIN_POLICY_EXISTING_CELLS_V1,
        RESKIN_POLICY_REFERENCE_DENSE_V1,
    ):
        config = value[policy]
        if not isinstance(config, Mapping):
            raise ValueError("effective skinning config must be an object")
        config_copy = dict(config)
        if config_copy.pop("reskin_policy", None) != policy:
            raise ValueError("effective skinning config policy identity mismatch")
        without_policy.append(config_copy)
    if without_policy[0] != without_policy[1]:
        raise ValueError("effective skinning configs differ outside reskin_policy")


def _report_semantic_evidence_digest(report: Mapping[str, Any]) -> str:
    policies = report["policies"]
    semantic_policies = {
        policy: {
            name: policies[policy][name]
            for name in (
                "reskin_policy",
                "parent_fingerprint",
                "generation",
                "link_topology",
                "skin_topology",
                "parent_ridge_surface",
                "duplicate_rounded_cell_index_count",
                "diagnostics",
            )
        }
        for policy in (
            RESKIN_POLICY_EXISTING_CELLS_V1,
            RESKIN_POLICY_REFERENCE_DENSE_V1,
        )
    }
    return canonical_fingerprint(
        {
            "schema_version": report["schema_version"],
            "baseline": report["baseline"],
            "candidate": report["candidate"],
            "upstream_parent_fingerprint": report["upstream_parent_fingerprint"],
            "resolved_config": report["resolved_config"],
            "comparison_config_fingerprint": report["comparison_config_fingerprint"],
            "metric_evidence_schema_version": report["metric_evidence_schema_version"],
            "validation_level": report["validation_level"],
            "policies": semantic_policies,
            "contrast": report["contrast"],
        }
    )


def _promote_comparison_completion(
    destination: Path,
    *,
    validator: Callable[[], tuple[Path, Path, Path, Path, Path]],
) -> tuple[Path, Path, Path, Path, Path]:
    paths = tuple(destination / name for name in F3_RESKIN_POLICY_COMPARISON_FILES)
    completion_path = destination / F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.promotion-tmp-",
            dir=destination.parent,
        )
    )
    backup = staging / "previous"
    backup.mkdir()
    changed_paths = paths[:3]
    try:
        report = _read_json_object(paths[0], "reskin policy comparison report")
        report["validation_level"] = "deep"
        staged_paths = tuple(staging / path.name for path in changed_paths)
        atomic_write_artifact(
            staged_paths[0],
            canonical_json_bytes(report) + b"\n",
            temporary_prefix=".reskin_policy_comparison.json.tmp-",
        )
        atomic_write_artifact(
            staged_paths[1],
            _comparison_csv(report).encode("utf-8"),
            temporary_prefix=".reskin_policy_metrics.csv.tmp-",
        )
        atomic_write_artifact(
            staged_paths[2],
            _comparison_markdown(report).encode("utf-8"),
            temporary_prefix=".reskin_policy_comparison.md.tmp-",
        )
        completion = _read_json_object(
            completion_path,
            "reskin policy comparison completion",
        )
        completion["validation_level"] = "deep"
        completion["report_semantic_evidence_sha256"] = _report_semantic_evidence_digest(report)
        completion["artifact_files"] = {
            **{
                path.name: artifact_file_metadata(staged)
                for path, staged in zip(changed_paths, staged_paths, strict=True)
            },
            **{path.name: artifact_file_metadata(path) for path in paths[3:]},
        }
        staged_completion = staging / completion_path.name
        atomic_write_artifact(
            staged_completion,
            canonical_json_bytes(completion) + b"\n",
            temporary_prefix=".complete.json.tmp-",
        )

        os.replace(completion_path, backup / completion_path.name)
        for path in changed_paths:
            os.replace(path, backup / path.name)
        for staged, path in zip(staged_paths, changed_paths, strict=True):
            os.replace(staged, path)
        os.replace(staged_completion, completion_path)
        validated = validator()
    except BaseException as error:
        previous_completion = backup / completion_path.name
        if previous_completion.is_file():
            completion_path.unlink(missing_ok=True)
            restore_failed = False
            for path in changed_paths:
                previous = backup / path.name
                if previous.is_file():
                    try:
                        os.replace(previous, path)
                    except BaseException as restore_error:
                        restore_failed = True
                        add_note = getattr(error, "add_note", None)
                        if add_note is not None:
                            add_note(f"comparison artifact restore also failed: {restore_error!r}")
            if not restore_failed:
                try:
                    os.replace(previous_completion, completion_path)
                except BaseException as restore_error:
                    add_note = getattr(error, "add_note", None)
                    if add_note is not None:
                        add_note(f"comparison completion restore also failed: {restore_error!r}")
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return validated


def _comparison_contrast(policies: Mapping[str, Any]) -> dict[str, float | int]:
    baseline = policies[RESKIN_POLICY_EXISTING_CELLS_V1]
    candidate = policies[RESKIN_POLICY_REFERENCE_DENSE_V1]
    return {
        "generated_cell_count_delta": (
            candidate["generation"]["reskin_generated_cell_count"]
            - baseline["generation"]["reskin_generated_cell_count"]
        ),
        "output_cell_count_delta": (
            candidate["generation"]["reskin_output_cell_count"]
            - baseline["generation"]["reskin_output_cell_count"]
        ),
        "small_skin_cell_fraction_delta": (
            candidate["skin_topology"]["small_skin_cell_fraction"]
            - baseline["skin_topology"]["small_skin_cell_fraction"]
        ),
        "parent_ridge_buffered_precision_delta": (
            candidate["parent_ridge_surface"]["metrics"]["overlap"]["buffered_precision"]
            - baseline["parent_ridge_surface"]["metrics"]["overlap"]["buffered_precision"]
        ),
        "parent_ridge_buffered_recall_delta": (
            candidate["parent_ridge_surface"]["metrics"]["overlap"]["buffered_recall"]
            - baseline["parent_ridge_surface"]["metrics"]["overlap"]["buffered_recall"]
        ),
        "parent_ridge_symmetric_chamfer_mean_delta": (
            candidate["parent_ridge_surface"]["metrics"]["surface_distance"][
                "symmetric_chamfer_mean"
            ]
            - baseline["parent_ridge_surface"]["metrics"]["surface_distance"][
                "symmetric_chamfer_mean"
            ]
        ),
    }


def _validate_reported_link_topology(
    links: Any,
    *,
    parsed: ParsedSkinArtifacts,
    expected_topology: Mapping[str, Any],
) -> None:
    """Validate link-topology algebra provable from serialized skin evidence.

    Canonical skin artifacts preserve skin membership and cell values, but not
    the live ``FaultCell.ca/cb/cl/cr`` edges.  Consequently this helper checks
    scalar safety and graph bounds without deriving component counts from
    policy provenance or serialized skin containers.  Exact component and
    isolation counts remain the responsibility of current-runtime deep replay.
    """

    expected_link_fields = {
        "reciprocal_link_violation_count",
        "cross_skin_link_count",
        "self_link_count",
        "linked_component_count",
        "isolated_cell_count",
        "quad_closure_candidate_count",
        "quad_closure_match_count",
        "quad_closure_mismatch_count",
    }
    if not isinstance(links, Mapping) or set(links) != expected_link_fields:
        raise ValueError("reskin policy comparison link topology field set mismatch")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in links.values()
    ):
        raise ValueError(
            "reskin policy comparison link topology counts must be non-negative integers"
        )

    if any(
        links[name] != 0
        for name in (
            "reciprocal_link_violation_count",
            "cross_skin_link_count",
            "self_link_count",
        )
    ):
        raise ValueError("reskin policy comparison link topology contains a safety violation")

    quad_candidates = links["quad_closure_candidate_count"]
    quad_matches = links["quad_closure_match_count"]
    quad_mismatches = links["quad_closure_mismatch_count"]
    if quad_candidates != quad_matches + quad_mismatches:
        raise ValueError("reskin policy comparison quad closure counts mismatch")

    try:
        cell_count = expected_topology["cell_count"]
        skin_count = expected_topology["skin_count"]
    except (KeyError, TypeError) as error:
        raise ValueError("reskin policy comparison skin topology counts are incomplete") from error
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (cell_count, skin_count)
    ):
        raise ValueError("reskin policy comparison skin topology counts are invalid")
    if cell_count != parsed.cell_count or skin_count != len(parsed.skins):
        raise ValueError("reskin policy comparison serialized skin topology counts mismatch")

    if quad_candidates > cell_count:
        raise ValueError("reskin policy comparison quad closure candidate count exceeds cells")
    if quad_matches > quad_candidates or quad_mismatches > quad_candidates:
        raise ValueError("reskin policy comparison quad closure count bounds mismatch")

    linked_components = links["linked_component_count"]
    isolated_cells = links["isolated_cell_count"]
    single_cell_skin_count = sum(len(skin) == 1 for skin in parsed.skins)

    if cell_count == 0:
        if skin_count != 0 or linked_components != 0 or isolated_cells != 0:
            raise ValueError("reskin policy comparison empty link topology mismatch")
        return

    if not 1 <= skin_count <= linked_components <= cell_count:
        raise ValueError("reskin policy comparison link topology component bounds mismatch")
    if not single_cell_skin_count <= isolated_cells <= linked_components:
        raise ValueError("reskin policy comparison link topology isolated bounds mismatch")
    if 2 * (linked_components - isolated_cells) > cell_count - isolated_cells:
        raise ValueError("reskin policy comparison link topology component size algebra mismatch")
    if linked_components == cell_count and isolated_cells != cell_count:
        raise ValueError("reskin policy comparison all-components isolation mismatch")
    if isolated_cells == cell_count and linked_components != cell_count:
        raise ValueError("reskin policy comparison all-isolated component mismatch")


def _validate_reported_policy_metrics(
    item: Mapping[str, Any],
    parsed: ParsedSkinArtifacts,
    *,
    shape: tuple[int, int, int],
    small_skin_size: int,
) -> None:
    expected_fields = {
        "reskin_policy",
        "parent_fingerprint",
        "canonical_skin_artifact",
        "generation",
        "link_topology",
        "skin_topology",
        "parent_ridge_surface",
        "duplicate_rounded_cell_index_count",
        "diagnostics",
    }
    if set(item) != expected_fields:
        raise ValueError("reskin policy comparison policy field set mismatch")
    diagnostics = item["diagnostics"]
    reskin_diagnostics = diagnostics.get("reskin")
    if not isinstance(reskin_diagnostics, Mapping):
        raise ValueError("reskin policy comparison reskin diagnostics must be an object")
    try:
        validate_reskin_diagnostics_contract(
            reskin_diagnostics,
            semantic_contract_version=F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
            skin_count=len(parsed.skins),
            final_provenance=(
                CONNECTED_COMPONENT_FALLBACK
                if diagnostics.get("fallback_used") is True
                else (
                    PRIMARY_EXISTING_CELLS_RESKINNED
                    if item["reskin_policy"] == RESKIN_POLICY_EXISTING_CELLS_V1
                    else PRIMARY_DENSE_RESKINNED
                )
            ),
        )
    except ValueError as error:
        raise ValueError(
            f"reskin policy comparison reskin diagnostics are invalid: {error}"
        ) from error
    if reskin_diagnostics.get("reskin_policy") != item["reskin_policy"]:
        raise ValueError("reskin policy comparison diagnostics policy identity mismatch")
    expected_generation = reskin_generation_metrics(
        parsed.skins,
        diagnostics=_final_reskin_diagnostics(diagnostics),
    )
    if item["generation"] != expected_generation:
        raise ValueError("reskin policy comparison generation metrics mismatch")

    expected_topology = skin_topology_metrics(
        parsed.skins,
        shape,
        small_skin_size=small_skin_size,
    )
    if item["skin_topology"] != expected_topology:
        raise ValueError("reskin policy comparison skin topology metrics mismatch")
    expected_duplicates = rounded_duplicate_cell_count(parsed.skins)
    if item["duplicate_rounded_cell_index_count"] != expected_duplicates:
        raise ValueError("reskin policy comparison duplicate count mismatch")

    fallback_used = diagnostics.get("fallback_used")
    if not isinstance(fallback_used, bool):
        raise ValueError("reskin policy comparison fallback_used must be a bool")
    expected_count_fields = (
        {
            "fallback_skin_count": expected_topology["skin_count"],
            "fallback_cell_count": expected_topology["cell_count"],
        }
        if fallback_used
        else {
            "skin_primary_count": expected_topology["skin_count"],
            "skin_primary_cell_count": expected_topology["cell_count"],
        }
    )
    for name, expected in expected_count_fields.items():
        if diagnostics.get(name) != expected:
            raise ValueError(f"reskin policy comparison diagnostics mismatch: {name}")

    links = item.get("link_topology")
    _validate_reported_link_topology(
        links,
        parsed=parsed,
        expected_topology=expected_topology,
    )

    surface = item.get("parent_ridge_surface")
    if not isinstance(surface, Mapping) or set(surface) != {
        "reference",
        "positive_epsilon",
        "ridge_voxel_count",
        "metrics",
        "evidence",
    }:
        raise ValueError("reskin policy comparison parent ridge surface field set mismatch")
    if surface["reference"] != "shared_parent_fvt_positive":
        raise ValueError("reskin policy comparison parent ridge reference mismatch")
    if surface["positive_epsilon"] != NONZERO_EPSILON:
        raise ValueError("reskin policy comparison parent ridge epsilon mismatch")
    ridge_count = surface["ridge_voxel_count"]
    if isinstance(ridge_count, bool) or not isinstance(ridge_count, int) or ridge_count < 0:
        raise ValueError("reskin policy comparison ridge voxel count is invalid")
    evidence = surface["evidence"]
    try:
        derived = metrics_from_surface_evidence(evidence)
    except ValueError as error:
        raise ValueError(
            f"reskin policy comparison parent ridge metric evidence is invalid: {error}"
        ) from error
    if surface["metrics"] != derived:
        raise ValueError("reskin policy comparison parent ridge metrics/evidence mismatch")
    if evidence["overlap"]["positive_epsilon"] != NONZERO_EPSILON:
        raise ValueError("reskin policy comparison metric evidence epsilon mismatch")
    if evidence["surface_distance"]["volume_shape"] != list(shape):
        raise ValueError("reskin policy comparison metric evidence shape mismatch")
    for name in ("overlap", "surface_distance"):
        metrics = derived[name]
        if not isinstance(metrics, Mapping):
            raise ValueError(f"reskin policy comparison {name} must be an object")
        if metrics.get("candidate_count") != expected_topology["unique_cell_count"]:
            raise ValueError(f"reskin policy comparison {name} candidate count mismatch")
        if metrics.get("truth_count") != ridge_count:
            raise ValueError(f"reskin policy comparison {name} truth count mismatch")


def _final_reskin_diagnostics(diagnostics: Mapping[str, Any]) -> Mapping[str, Any] | None:
    fallback_used = diagnostics.get("fallback_used")
    if not isinstance(fallback_used, bool):
        raise ValueError("reskin policy comparison fallback_used must be a bool")
    if fallback_used:
        return None
    reskin = diagnostics.get("reskin")
    if not isinstance(reskin, Mapping):
        raise ValueError("reskin policy comparison reskin diagnostics must be an object")
    return reskin


def _mutable_scalar_evidence(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _mutable_scalar_evidence(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_mutable_scalar_evidence(item) for item in value]
    return value


def _comparison_csv(report: Mapping[str, Any]) -> str:
    fields = (
        "schema_version",
        "row_type",
        "policy",
        "reskin_policy",
        "parent_fingerprint",
        "comparison_config_fingerprint",
        "metric_evidence_schema_version",
        "validation_level",
        "metric",
        "value",
        "status",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for policy in (
        RESKIN_POLICY_EXISTING_CELLS_V1,
        RESKIN_POLICY_REFERENCE_DENSE_V1,
    ):
        item = report["policies"][policy]
        metric_groups = (
            ("generation", item["generation"]),
            ("link_topology", item["link_topology"]),
            ("skin_topology", item["skin_topology"]),
            (
                "parent_ridge_overlap",
                item["parent_ridge_surface"]["metrics"]["overlap"],
            ),
            (
                "parent_ridge_surface_distance",
                item["parent_ridge_surface"]["metrics"]["surface_distance"],
            ),
        )
        for group, metrics in metric_groups:
            for metric in sorted(metrics):
                value = metrics[metric]
                if isinstance(value, (str, Mapping, list, tuple)):
                    continue
                writer.writerow(
                    {
                        "schema_version": report["schema_version"],
                        "row_type": "policy",
                        "policy": (
                            "baseline" if policy == RESKIN_POLICY_EXISTING_CELLS_V1 else "candidate"
                        ),
                        "reskin_policy": policy,
                        "parent_fingerprint": item["parent_fingerprint"],
                        "comparison_config_fingerprint": report["comparison_config_fingerprint"],
                        "metric_evidence_schema_version": report["metric_evidence_schema_version"],
                        "validation_level": report["validation_level"],
                        "metric": f"{group}.{metric}",
                        "value": value,
                        "status": (
                            _generation_metric_status(item["generation"], metric)
                            if group == "generation"
                            else ""
                        ),
                    }
                )
    for metric in _CONTRAST_FIELDS:
        value = report["contrast"][metric]
        writer.writerow(
            {
                "schema_version": report["schema_version"],
                "row_type": "contrast",
                "policy": "candidate_minus_baseline",
                "reskin_policy": (
                    f"{RESKIN_POLICY_REFERENCE_DENSE_V1}_minus_{RESKIN_POLICY_EXISTING_CELLS_V1}"
                ),
                "parent_fingerprint": report["upstream_parent_fingerprint"],
                "comparison_config_fingerprint": report["comparison_config_fingerprint"],
                "metric_evidence_schema_version": report["metric_evidence_schema_version"],
                "validation_level": report["validation_level"],
                "metric": metric,
                "value": value,
            }
        )
    return stream.getvalue()


def _comparison_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# F3 dense reskin same-parent comparison",
        "",
        f"- Shared parent fingerprint: `{report['upstream_parent_fingerprint']}`",
        f"- Parent fingerprint identical: `{str(report['upstream_parent_fingerprint_identical']).lower()}`",
        f"- Comparison config fingerprint: `{report['comparison_config_fingerprint']}`",
        f"- Metric evidence schema: `{report['metric_evidence_schema_version']}`",
        f"- Validation level: `{report['validation_level']}`",
        "",
        "| Policy | Cells | Generated | Generated fraction | Support min/mean/max | "
        "Ridge buffered precision/recall | Ridge chamfer | Link violations |",
        "|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for policy in (
        RESKIN_POLICY_EXISTING_CELLS_V1,
        RESKIN_POLICY_REFERENCE_DENSE_V1,
    ):
        item = report["policies"][policy]
        generation = item["generation"]
        overlap = item["parent_ridge_surface"]["metrics"]["overlap"]
        distance = item["parent_ridge_surface"]["metrics"]["surface_distance"]
        links = item["link_topology"]
        support = "/".join(
            _display(generation[name])
            for name in (
                "reskin_support_min",
                "reskin_support_mean",
                "reskin_support_max",
            )
        )
        violations = sum(
            int(links[name])
            for name in (
                "reciprocal_link_violation_count",
                "cross_skin_link_count",
                "self_link_count",
            )
        )
        lines.append(
            f"| `{policy}` | {generation['reskin_output_cell_count']} | "
            f"{generation['reskin_generated_cell_count']} | "
            f"{_display(generation['reskin_generated_cell_fraction'])} | {support} | "
            f"{_display(overlap['buffered_precision'])}/"
            f"{_display(overlap['buffered_recall'])} | "
            f"{_display(distance['symmetric_chamfer_mean'])} | {violations} |"
        )
    lines.extend(
        [
            "",
            "The parent FVT ridge is comparison evidence, not independent geological truth.",
            "",
        ]
    )
    return "\n".join(lines)


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _generation_metric_status(generation: Mapping[str, Any], metric: str) -> str:
    if metric == "reskin_generated_cell_fraction":
        return str(generation["reskin_generated_cell_fraction_status"])
    if metric in {
        "reskin_support_min",
        "reskin_support_mean",
        "reskin_support_max",
    }:
        return str(generation["reskin_support_status"])
    if metric in {
        "reskin_final_likelihood_min_for_generated",
        "reskin_final_likelihood_mean_for_generated",
    }:
        return str(generation["reskin_generated_likelihood_status"])
    return ""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{context} must be a regular non-symlink file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_raise_nonfinite_json,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{context} is not strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _load_source_bundle_identity(root: Path) -> _SourceBundleIdentity:
    manifest = _read_json_object(root / RUN_MANIFEST_FILE, RUN_MANIFEST_FILE)
    return _source_bundle_identity_from_manifest(manifest)


def _source_bundle_identity_from_manifest(
    manifest: Mapping[str, Any],
) -> _SourceBundleIdentity:
    """Return canonical numerical provenance from one source run manifest."""

    if not isinstance(manifest, Mapping):
        raise ValueError("source run manifest must be an object")
    expected_fields = {*_RUN_COMPUTATION_FIELDS, "run_fingerprint", "provenance"}
    if set(manifest) != expected_fields:
        raise ValueError("source run manifest field set mismatch")
    computation = {name: manifest[name] for name in _RUN_COMPUTATION_FIELDS}
    run_fingerprint = _validated_sha256(
        manifest["run_fingerprint"],
        "source run fingerprint",
    )
    if canonical_fingerprint(computation) != run_fingerprint:
        raise ValueError("source run manifest fingerprint mismatch")
    if manifest["fingerprint_contract_version"] != F3_FINGERPRINT_CONTRACT_VERSION:
        raise ValueError("source run manifest fingerprint contract mismatch")

    dataset_identity = _validated_source_dataset_identity(manifest["dataset_identity"])
    official = dataset_identity["dataset_id"] == F3_DATASET_ID
    try:
        runtime_identity = (
            validate_publication_runtime_identity(manifest["runtime_identity"])
            if official
            else validate_numerical_runtime_identity(manifest["runtime_identity"])
        )
    except ValueError as error:
        contract = "publication runtime" if official else "runtime"
        raise ValueError(f"source run manifest {contract} identity is invalid: {error}") from error
    schema_version = runtime_identity["runtime_identity_schema_version"]
    return _SourceBundleIdentity(
        run_fingerprint=run_fingerprint,
        dataset_identity=dataset_identity,
        runtime_identity=runtime_identity,
        runtime_identity_schema_version=schema_version,
        runtime_identity_sha256=canonical_fingerprint(runtime_identity),
        official=official,
    )


def _validated_source_dataset_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"dataset_id", "files"}:
        raise ValueError("source run manifest dataset identity is invalid")
    dataset_id = value["dataset_id"]
    files = value["files"]
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("source run manifest dataset ID is invalid")
    if not isinstance(files, list) or not files:
        raise ValueError("source run manifest dataset files are invalid")
    roles: set[str] = set()
    normalized_files: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {
            "role",
            "size",
            "sha256",
            "shape",
            "storage_dtype",
        }:
            raise ValueError("source run manifest dataset file identity is invalid")
        role = item["role"]
        size = item["size"]
        shape = item["shape"]
        if not isinstance(role, str) or not role or role in roles:
            raise ValueError("source run manifest dataset file roles are invalid")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("source run manifest dataset file size is invalid")
        if (
            not isinstance(shape, list)
            or len(shape) != 3
            or any(
                isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0
                for dimension in shape
            )
        ):
            raise ValueError("source run manifest dataset file shape is invalid")
        try:
            dtype = np.dtype(item["storage_dtype"])
        except TypeError as error:
            raise ValueError("source run manifest dataset file dtype is invalid") from error
        if dtype.str != ">f4" or size != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
            raise ValueError("source run manifest dataset file layout is invalid")
        roles.add(role)
        normalized_files.append(
            {
                "role": role,
                "size": size,
                "sha256": _validated_sha256(
                    item["sha256"],
                    f"source dataset file {role} digest",
                ),
                "shape": list(shape),
                "storage_dtype": dtype.str,
            }
        )
    return {"dataset_id": dataset_id, "files": normalized_files}


def _validate_current_comparison_runtime(source: _SourceBundleIdentity) -> str:
    """Require exact source/current numerical identity before recomputation."""

    try:
        current_raw = numerical_runtime_identity()
        current = (
            validate_publication_runtime_identity(current_raw)
            if source.official
            else validate_numerical_runtime_identity(current_raw)
        )
    except (RuntimeError, ValueError) as error:
        contract = "publication runtime" if source.official else "runtime"
        raise ValueError(f"comparison current {contract} identity is invalid: {error}") from error
    current_sha256 = canonical_fingerprint(current)
    if current_sha256 != source.runtime_identity_sha256 or canonical_json_bytes(
        current
    ) != canonical_json_bytes(source.runtime_identity):
        raise ValueError("comparison current runtime identity does not match source run manifest")
    return current_sha256


def _comparison_implementation_identity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[2]
    relative_sources = (
        "_accel.py",
        "_skinner/candidate_path.py",
        "_skinner/candidate_sampling.py",
        "_skinner/connected.py",
        "_skinner/grid.py",
        "candidate_volume.py",
        "cells.py",
        "filters.py",
        "geometry.py",
        "skin.py",
        "skinner.py",
        "_skinner/growth.py",
        "_skinner/models.py",
        "_skinner/occupancy.py",
        "_skinner/reference.py",
        "_skinner/reskin.py",
        "_skinner/seeds.py",
        "_skinner/transforms.py",
        "_skinner/validation.py",
        "evaluation/f3d_mode_comparison/artifacts.py",
        # The comparison reads completed bundles through result.py and uses
        # data.py's official dataset ID to select the publication runtime
        # contract, so both modules are part of comparison input provenance.
        "evaluation/f3d_mode_comparison/data.py",
        "evaluation/f3d_mode_comparison/result.py",
        "evaluation/f3d_mode_comparison/runtime_identity.py",
        "evaluation/f3d_mode_comparison/skin_artifacts.py",
        "evaluation/workflow3d.py",
        "evaluation/f3d_mode_comparison/reskin_policy_comparison.py",
        "evaluation/synthetic_quality/config.py",
        "evaluation/synthetic_quality/quality_metrics.py",
        "evaluation/synthetic_quality/stage_cache.py",
        "evaluation/synthetic_quality/stage_keys.py",
        "evaluation/synthetic_quality/variants.py",
        "experimental/boundary_skinning.py",
        "experimental/boundary_thinning.py",
        "experimental/skin_diagnostics.py",
        "synthetic_metrics.py",
    )
    return implementation_identity(
        source_files={name: package_root / name for name in relative_sources}
    )


def _validate_comparison_provenance(
    report: Mapping[str, Any],
    completion: Mapping[str, Any],
    *,
    source_identity: _SourceBundleIdentity,
    comparison_implementation: Mapping[str, Any],
    comparison_implementation_sha256: str,
) -> None:
    source_runtime_sha256 = source_identity.runtime_identity_sha256
    if (
        report.get("source_run_fingerprint") != source_identity.run_fingerprint
        or completion["source_run_fingerprint"] != source_identity.run_fingerprint
    ):
        raise ValueError("reskin policy comparison source run fingerprint mismatch")
    if (
        report.get("source_runtime_identity_schema_version")
        != source_identity.runtime_identity_schema_version
    ):
        raise ValueError("reskin policy comparison source runtime schema mismatch")
    if (
        report.get("source_runtime_identity_sha256") != source_runtime_sha256
        or completion["source_runtime_identity_sha256"] != source_runtime_sha256
    ):
        raise ValueError("reskin policy comparison source runtime digest mismatch")
    if (
        report.get("comparison_runtime_identity_sha256") != source_runtime_sha256
        or completion["comparison_runtime_identity_sha256"] != source_runtime_sha256
    ):
        raise ValueError("reskin policy comparison runtime digest mismatch")
    if report.get("comparison_implementation_identity") != comparison_implementation:
        raise ValueError("reskin policy comparison implementation identity mismatch")
    if (
        completion["comparison_implementation_identity_sha256"] != comparison_implementation_sha256
        or canonical_fingerprint(report["comparison_implementation_identity"])
        != comparison_implementation_sha256
    ):
        raise ValueError("reskin policy comparison implementation digest mismatch")


def _validated_sha256(value: Any, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _raise_nonfinite_json(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError(f"duplicate JSON key: {name}")
        value[name] = item
    return value


def _validated_parent_arrays(**arrays: np.ndarray) -> dict[str, np.ndarray]:
    validated: dict[str, np.ndarray] = {}
    shape: tuple[int, ...] | None = None
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.ndim != 3:
            raise ValueError(f"{name} must be a 3D array")
        if array.dtype.kind != "f" or array.dtype.itemsize != np.dtype(np.float32).itemsize:
            raise TypeError(f"{name} must have dtype float32")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise ValueError(f"{name} shape must match {shape}")
        for index in range(array.shape[0]):
            if not np.all(np.isfinite(array[index])):
                raise ValueError(f"{name} must contain only finite values")
        view = array.view()
        view.flags.writeable = False
        validated[name] = view
    return validated


def _validated_parent_mask(
    mask: np.ndarray | None,
    *,
    shape: tuple[int, ...],
) -> np.ndarray | None:
    if mask is None:
        return None
    array = np.asarray(mask)
    if array.dtype != np.dtype(np.bool_):
        raise TypeError("scanner_target_positive_mask must have dtype bool")
    if array.shape != shape:
        raise ValueError(f"scanner_target_positive_mask shape must match {shape}")
    view = array.view()
    view.flags.writeable = False
    return view


def _parent_fingerprint(
    arrays: dict[str, np.ndarray],
    *,
    scanner_target_positive_mask: np.ndarray | None,
) -> str:
    digest = hashlib.sha256()
    for name in ("fv", "fvt", "vp", "vt"):
        array = arrays[name]
        digest.update(name.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        for index in range(array.shape[0]):
            digest.update(np.ascontiguousarray(array[index]).tobytes())
    digest.update(b"scanner_target_positive_mask")
    if scanner_target_positive_mask is None:
        digest.update(b"none")
    else:
        digest.update(
            json.dumps(
                scanner_target_positive_mask.shape,
                separators=(",", ":"),
            ).encode("ascii")
        )
        for index in range(scanner_target_positive_mask.shape[0]):
            digest.update(np.ascontiguousarray(scanner_target_positive_mask[index]).tobytes())
    return digest.hexdigest()


def _stage_shape(report_path: Path) -> tuple[int, int, int]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read parent stage report: {report_path}") from error
    shape = report.get("shape") if isinstance(report, Mapping) else None
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape
        )
    ):
        raise ValueError(f"parent stage report has invalid shape: {report_path}")
    return tuple(shape)


def _open_parent_dat(path: Path, shape: tuple[int, int, int]) -> np.memmap:
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * np.dtype(">f4").itemsize
    try:
        actual_bytes = path.stat().st_size
    except OSError as error:
        raise ValueError(f"cannot stat parent stage artifact: {path}") from error
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"parent stage artifact size mismatch for {path}: "
            f"expected {expected_bytes}, got {actual_bytes}"
        )
    return np.memmap(path, dtype=">f4", mode="r", shape=shape, order="C")


__all__ = [
    "F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE",
    "F3_RESKIN_POLICY_COMPARISON_COMPLETION_SCHEMA_VERSION",
    "F3_RESKIN_POLICY_COMPARISON_FILES",
    "F3_RESKIN_POLICY_COMPARISON_CELL",
    "F3_RESKIN_POLICY_COMPARISON_DIR",
    "F3_RESKIN_POLICY_COMPARISON_SCHEMA_VERSION",
    "compare_reskin_policies_from_bundle",
    "compare_reskin_policies_from_parent",
    "validate_f3_reskin_policy_comparison",
    "write_f3_reskin_policy_comparison",
]
