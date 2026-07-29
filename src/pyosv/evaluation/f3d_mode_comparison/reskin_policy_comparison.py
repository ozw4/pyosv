"""Skin-only F3 comparison from one immutable upstream parent."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from pyosv.candidate_volume import NONZERO_EPSILON, positive_candidate_mask
from pyosv.evaluation.synthetic_quality import SyntheticSkinningConfig
from pyosv.evaluation.synthetic_quality.variants import VariantSpec
from pyosv.evaluation.workflow3d import execute_skinning_phase3d
from pyosv.skinner import (
    RESKIN_POLICY_EXISTING_CELLS_V1,
    RESKIN_POLICY_REFERENCE_DENSE_V1,
)
from pyosv.synthetic_metrics import (
    reskin_generation_metrics,
    rounded_duplicate_cell_count,
    skin_mask_from_skins,
    skin_link_topology_metrics,
    skin_topology_metrics,
    surface_comparison_metrics,
)

from .artifacts import artifact_file_metadata, atomic_write_artifact, canonical_json_bytes
from .skin_artifacts import (
    CONNECTED_COMPONENT_FALLBACK,
    F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
    PRIMARY_DENSE_RESKINNED,
    PRIMARY_EXISTING_CELLS_RESKINNED,
    ParsedSkinArtifacts,
    canonical_skins_payload,
    parse_skins_json,
    validate_skin_generation_provenance,
)

F3_RESKIN_POLICY_COMPARISON_SCHEMA_VERSION = 1
F3_RESKIN_POLICY_COMPARISON_COMPLETION_SCHEMA_VERSION = 1
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


def compare_reskin_policies_from_parent(
    *,
    fv: np.ndarray,
    fvt: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    skinning_config: SyntheticSkinningConfig,
    variant_spec: VariantSpec = _F3_CANONICAL_VARIANT,
    scanner_target_positive_mask: np.ndarray | None = None,
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

    parent_fingerprint = _parent_fingerprint(arrays)
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
            skinning_settings=replace(skinning_config, reskin_policy=policy),
            variant_spec=variant_spec,
            scanner_target_positive_mask=scanner_target_positive_mask,
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
    surface_metrics = surface_comparison_metrics(
        skin_masks,
        parent_ridge,
        radius=2.0,
    )
    for policy, metrics in surface_metrics.items():
        policies[policy]["parent_ridge_surface"].update(metrics)

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
    json_path.write_text(_canonical_json(report) + "\n", encoding="utf-8")
    csv_path.write_text(_comparison_csv(report), encoding="utf-8")
    markdown_path.write_text(_comparison_markdown(report), encoding="utf-8")
    baseline_skin_path.write_text(
        _canonical_json(policies[RESKIN_POLICY_EXISTING_CELLS_V1]["canonical_skin_artifact"])
        + "\n",
        encoding="utf-8",
    )
    candidate_skin_path.write_text(
        _canonical_json(policies[RESKIN_POLICY_REFERENCE_DENSE_V1]["canonical_skin_artifact"])
        + "\n",
        encoding="utf-8",
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
    """Run or reuse the fixed skin-only pair from canonical Q-QUAL parents."""

    # Keep result loading local: package __init__ imports this module before result.py.
    from .result import load_f3d_mode_comparison_result

    if not isinstance(resume, bool) or not isinstance(deep, bool):
        raise TypeError("resume and deep must be bool")
    root = Path(bundle)
    result = load_f3d_mode_comparison_result(root, deep=False)
    destination = (
        Path(output_dir) if output_dir is not None else root / F3_RESKIN_POLICY_COMPARISON_DIR
    )
    completion_path = destination / F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE
    if completion_path.is_file() and not completion_path.is_symlink():
        if not resume:
            raise FileExistsError(f"completed F3 reskin policy comparison exists: {destination}")
        return validate_f3_reskin_policy_comparison(
            root,
            output_dir=destination,
            deep=deep,
            result=result,
        )
    if destination.exists() and (not destination.is_dir() or destination.is_symlink()):
        raise ValueError("reskin policy comparison output must be a directory")
    if resume and destination.is_dir():
        for item in destination.glob(".complete.json.tmp-*"):
            if item.is_file() and not item.is_symlink():
                item.unlink()
    if destination.is_dir() and any(destination.iterdir()):
        if not resume:
            raise FileExistsError(f"reskin policy comparison output exists: {destination}")
        allowed_partial_files = {
            *F3_RESKIN_POLICY_COMPARISON_FILES,
            F3_RESKIN_POLICY_COMPARISON_COMPLETION_FILE,
        }
        for item in destination.iterdir():
            if item.name not in allowed_partial_files or not item.is_file() or item.is_symlink():
                raise ValueError("incomplete reskin policy comparison file set is invalid")

    report, cell, shape = _comparison_report_from_bundle(root, result)
    paths = write_f3_reskin_policy_comparison(report, destination)
    completion = {
        "completion_schema_version": F3_RESKIN_POLICY_COMPARISON_COMPLETION_SCHEMA_VERSION,
        "comparison_schema_version": F3_RESKIN_POLICY_COMPARISON_SCHEMA_VERSION,
        "status": "complete",
        "source_cell": F3_RESKIN_POLICY_COMPARISON_CELL,
        "upstream_parent_stage_fingerprints": {
            "voting": cell.stages.voting,
            "thinning": cell.stages.thinning,
        },
        "artifact_files": {path.name: artifact_file_metadata(path) for path in paths},
    }
    atomic_write_artifact(
        completion_path,
        canonical_json_bytes(completion) + b"\n",
        temporary_prefix=".complete.json.tmp-",
    )
    try:
        return validate_f3_reskin_policy_comparison(
            root,
            output_dir=destination,
            deep=deep,
            result=result,
            _shape=shape,
        )
    except BaseException:
        completion_path.unlink(missing_ok=True)
        raise


def validate_f3_reskin_policy_comparison(
    bundle: str | Path,
    *,
    output_dir: str | Path | None = None,
    deep: bool = False,
    result: Any | None = None,
    _shape: tuple[int, int, int] | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    """Validate completion and optionally exact-replay both skin artifacts."""

    if not isinstance(deep, bool):
        raise TypeError("deep must be bool")
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

    if result is None:
        from .result import load_f3d_mode_comparison_result

        result = load_f3d_mode_comparison_result(root, deep=False)
    cell, config = _comparison_cell_and_config(result)
    expected_parent_stages = {
        "voting": cell.stages.voting,
        "thinning": cell.stages.thinning,
    }
    if completion["upstream_parent_stage_fingerprints"] != expected_parent_stages:
        raise ValueError("reskin policy comparison parent stage mismatch")

    paths = tuple(destination / filename for filename in F3_RESKIN_POLICY_COMPARISON_FILES)
    metadata = completion["artifact_files"]
    if not isinstance(metadata, Mapping) or set(metadata) != set(F3_RESKIN_POLICY_COMPARISON_FILES):
        raise ValueError("reskin policy comparison artifact metadata mismatch")
    for path in paths:
        if artifact_file_metadata(path) != metadata[path.name]:
            raise ValueError(
                f"reskin policy comparison artifact hash or size mismatch: {path.name}"
            )

    report = _read_json_object(paths[0], "reskin policy comparison report")
    expected_report_fields = {
        "schema_version",
        "baseline",
        "candidate",
        "upstream_parent_fingerprint",
        "upstream_parent_fingerprint_identical",
        "policies",
        "contrast",
        "source_cell",
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
        validate_skin_generation_provenance(
            parsed,
            CONNECTED_COMPONENT_FALLBACK if fallback_used else provenance,
            semantic_contract_version=F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
            reskin_diagnostics=_final_reskin_diagnostics(diagnostics),
        )

    if dict(contrast) != _comparison_contrast(policies):
        raise ValueError("reskin policy comparison contrast metrics mismatch")
    if paths[1].read_text(encoding="utf-8") != _comparison_csv(report):
        raise ValueError("reskin policy comparison CSV does not match report")
    if paths[2].read_text(encoding="utf-8") != _comparison_markdown(report):
        raise ValueError("reskin policy comparison Markdown does not match report")

    if deep:
        recomputed, _, _ = _comparison_report_from_bundle(root, result)
        if canonical_json_bytes(recomputed) != canonical_json_bytes(report):
            raise ValueError(
                "reskin policy comparison does not exactly match skin-only recomputation"
            )
    return paths


def _comparison_cell_and_config(result: Any) -> tuple[Any, SyntheticSkinningConfig]:
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
    if not isinstance(skinning, Mapping):
        raise ValueError("canonical comparison cell has no resolved skinning config")
    config = SyntheticSkinningConfig(**dict(skinning))
    if config.method != "quality" or not config.reskin:
        raise ValueError(
            "canonical Q-QUAL skinning config must use method='quality' and reskin=True"
        )
    return cell, config


def _comparison_report_from_bundle(
    root: Path,
    result: Any,
) -> tuple[dict[str, Any], Any, tuple[int, int, int]]:
    cell, config = _comparison_cell_and_config(result)
    voting_path = root / "stages" / "voting" / cell.stages.voting
    thinning_path = root / "stages" / "thinning" / cell.stages.thinning
    shape = _stage_shape(voting_path / "report.json")
    if _stage_shape(thinning_path / "report.json") != shape:
        raise ValueError("voting and thinning parent shapes do not match")

    arrays = (
        _open_parent_dat(voting_path / "fv.dat", shape),
        _open_parent_dat(thinning_path / "fvt.dat", shape),
        _open_parent_dat(voting_path / "vp.dat", shape),
        _open_parent_dat(voting_path / "vt.dat", shape),
    )
    try:
        report = compare_reskin_policies_from_parent(
            fv=arrays[0],
            fvt=arrays[1],
            vp=arrays[2],
            vt=arrays[3],
            skinning_config=config,
        )
    finally:
        for array in arrays:
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and not mapping.closed:
                mapping.close()

    report["source_cell"] = F3_RESKIN_POLICY_COMPARISON_CELL
    report["upstream_parent_stage_fingerprints"] = {
        "voting": cell.stages.voting,
        "thinning": cell.stages.thinning,
    }
    return report, cell, shape


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
            candidate["parent_ridge_surface"]["overlap"]["buffered_precision"]
            - baseline["parent_ridge_surface"]["overlap"]["buffered_precision"]
        ),
        "parent_ridge_buffered_recall_delta": (
            candidate["parent_ridge_surface"]["overlap"]["buffered_recall"]
            - baseline["parent_ridge_surface"]["overlap"]["buffered_recall"]
        ),
        "parent_ridge_symmetric_chamfer_mean_delta": (
            candidate["parent_ridge_surface"]["surface_distance"]["symmetric_chamfer_mean"]
            - baseline["parent_ridge_surface"]["surface_distance"]["symmetric_chamfer_mean"]
        ),
    }


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
    if links["quad_closure_candidate_count"] != (
        links["quad_closure_match_count"] + links["quad_closure_mismatch_count"]
    ):
        raise ValueError("reskin policy comparison quad closure counts mismatch")
    expected_linked_components = (
        expected_topology["cell_count"] if fallback_used else expected_topology["skin_count"]
    )
    expected_isolated = (
        expected_topology["cell_count"]
        if fallback_used
        else sum(len(skin) == 1 for skin in parsed.skins)
    )
    if links["linked_component_count"] != expected_linked_components:
        raise ValueError("reskin policy comparison linked component count mismatch")
    if links["isolated_cell_count"] != expected_isolated:
        raise ValueError("reskin policy comparison isolated cell count mismatch")

    surface = item.get("parent_ridge_surface")
    if not isinstance(surface, Mapping) or set(surface) != {
        "reference",
        "positive_epsilon",
        "ridge_voxel_count",
        "overlap",
        "surface_distance",
    }:
        raise ValueError("reskin policy comparison parent ridge surface field set mismatch")
    if surface["reference"] != "shared_parent_fvt_positive":
        raise ValueError("reskin policy comparison parent ridge reference mismatch")
    if surface["positive_epsilon"] != NONZERO_EPSILON:
        raise ValueError("reskin policy comparison parent ridge epsilon mismatch")
    ridge_count = surface["ridge_voxel_count"]
    if isinstance(ridge_count, bool) or not isinstance(ridge_count, int) or ridge_count < 0:
        raise ValueError("reskin policy comparison ridge voxel count is invalid")
    for name in ("overlap", "surface_distance"):
        metrics = surface[name]
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
            ("parent_ridge_overlap", item["parent_ridge_surface"]["overlap"]),
            (
                "parent_ridge_surface_distance",
                item["parent_ridge_surface"]["surface_distance"],
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
        overlap = item["parent_ridge_surface"]["overlap"]
        distance = item["parent_ridge_surface"]["surface_distance"]
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


def _parent_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("fv", "fvt", "vp", "vt"):
        array = arrays[name]
        digest.update(name.encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        for index in range(array.shape[0]):
            digest.update(np.ascontiguousarray(array[index]).tobytes())
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
