"""Canonical parsing and cross-file validation for F3 skin artifacts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from pyosv.cells import (
    FAULT_CELL_GENERATION_CONNECTED_COMPONENT,
    FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED,
    FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED,
    FAULT_CELL_GENERATION_EXISTING_CELLS_RESKINNED,
    FAULT_CELL_GENERATION_GROWN,
)
from pyosv.skinner import (
    RESKIN_POLICY_EXISTING_CELLS_V1,
    RESKIN_POLICY_REFERENCE_DENSE_V1,
    validate_reskin_policy,
)
from pyosv.synthetic_metrics import skin_topology_metrics

_ROOT_FIELDS = {"format_version", "skinning_enabled", "skin_count", "skins"}
_SKIN_FIELDS = {"skin_index", "cell_count", "cells"}
_CELL_FIELDS_V1 = {"x1", "x2", "x3", "i1", "i2", "i3", "fl", "fp", "ft"}
_CELL_FIELDS_V2 = {*_CELL_FIELDS_V1, "generation", "reskin_support"}
_INDEX_FIELDS = ("i1", "i2", "i3")
_SCALAR_FIELDS = ("x1", "x2", "x3", "fl", "fp", "ft")
_TOPOLOGY_FIELDS = (
    "skin_count",
    "cell_count",
    "unique_cell_count",
    "duplicate_cell_count",
    "largest_skin_size",
    "largest_skin_fraction",
    "small_skin_size",
    "small_skin_count",
    "small_skin_cell_count",
    "small_skin_cell_fraction",
)
_INTEGER_TOPOLOGY_FIELDS = frozenset(
    {
        "skin_count",
        "cell_count",
        "unique_cell_count",
        "duplicate_cell_count",
        "largest_skin_size",
        "small_skin_size",
        "small_skin_count",
        "small_skin_cell_count",
    }
)
_DAT_DTYPE = np.dtype(">f4")
_MASK_SLAB_VOXELS = 1_000_000
_INDEX_CHUNK_SIZE = 100_000
F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION = 4
F3_LEGACY_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION = 3
PRIMARY_NEAREST_SAMPLE = "primary_nearest_sample"
PRIMARY_RESKINNED = "primary_reskinned"
PRIMARY_EXISTING_CELLS_RESKINNED = "primary_existing_cells_reskinned"
PRIMARY_DENSE_RESKINNED = "primary_dense_reskinned"
PRIMARY_CONNECTED_COMPONENT = "primary_connected_component"
CONNECTED_COMPONENT_FALLBACK = "connected_component_fallback"
_CELL_GENERATIONS = {
    FAULT_CELL_GENERATION_GROWN,
    FAULT_CELL_GENERATION_EXISTING_CELLS_RESKINNED,
    FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED,
    FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED,
    FAULT_CELL_GENERATION_CONNECTED_COMPONENT,
}
_DENSE_GENERATIONS = {
    FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED,
    FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED,
}


class SkinArtifactValidationError(ValueError):
    """Raised when canonical F3 skin artifacts disagree."""


@dataclass(frozen=True, slots=True)
class SkinCellRecord:
    """Immutable cell values retained exactly from ``skins.json``."""

    x1: float
    x2: float
    x3: float
    i1: int
    i2: int
    i3: int
    fl: float
    fp: float
    ft: float
    generation: str | None = None
    reskin_support: float | None = None


@dataclass(frozen=True, slots=True)
class SkinParentVolumeContract:
    """Persisted parent volumes that define final skin cell values."""

    provenance: str
    likelihood: tuple[str, str, str]
    strike: tuple[str, str, str]
    dip: tuple[str, str, str]
    scanner_target: tuple[str, str, str] | None


@dataclass(frozen=True, slots=True)
class ParsedSkinArtifacts:
    """Canonical skins with duplicate voxel occurrences preserved."""

    skins: tuple[tuple[SkinCellRecord, ...], ...]
    format_version: int = 1

    @property
    def cell_count(self) -> int:
        return sum(len(skin) for skin in self.skins)

    @property
    def unique_indices(self) -> frozenset[tuple[int, int, int]]:
        return frozenset((cell.i1, cell.i2, cell.i3) for skin in self.skins for cell in skin)

    @property
    def duplicate_cell_count(self) -> int:
        return self.cell_count - len(self.unique_indices)


def canonical_skins_payload(
    skins: Sequence[Any],
) -> dict[str, Any]:
    """Serialize skins with the canonical format-version-2 contract."""

    return _canonical_skins_payload_for_format(skins, format_version=2)


def _canonical_skins_payload_for_format(
    skins: Sequence[Any],
    *,
    format_version: int,
) -> dict[str, Any]:
    """Serialize current v2 or historical v1 payloads for exact validation."""

    materialized = tuple(tuple(skin) for skin in skins)
    if isinstance(format_version, bool) or format_version not in {1, 2}:
        raise ValueError("format_version must be 1 or 2")

    def cell_payload(cell: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "x1": float(cell.x1),
            "x2": float(cell.x2),
            "x3": float(cell.x3),
            "i1": int(cell.i1),
            "i2": int(cell.i2),
            "i3": int(cell.i3),
            "fl": float(cell.fl),
            "fp": float(cell.fp),
            "ft": float(cell.ft),
        }
        if format_version == 2:
            generation = getattr(cell, "generation", None)
            support = getattr(cell, "reskin_support", None)
            _validate_generation_support(generation, support)
            payload["generation"] = generation
            payload["reskin_support"] = None if support is None else float(support)
        return payload

    serialized = []
    for skin_index, skin in enumerate(materialized):
        cells = sorted(skin, key=lambda cell: (cell.i3, cell.i2, cell.i1))
        serialized.append(
            {
                "skin_index": skin_index,
                "cell_count": len(cells),
                "cells": [cell_payload(cell) for cell in cells],
            }
        )
    return {
        "format_version": format_version,
        "skinning_enabled": True,
        "skin_count": len(serialized),
        "skins": serialized,
    }


def resolve_skin_parent_volume_contract(
    stages: Mapping[str, str],
    resolved_skinning: Mapping[str, Any],
    resolved_variant: Mapping[str, Any],
    *,
    fallback_used: bool,
    resolved_stage_settings: Mapping[str, Any] | None = None,
) -> SkinParentVolumeContract:
    """Purely resolve final cell sources from recorded stage and skin settings."""

    provenance = resolve_final_skin_cell_value_provenance(
        resolved_skinning,
        resolved_variant,
        fallback_used=fallback_used,
        resolved_stage_settings=resolved_stage_settings,
    )
    if not isinstance(fallback_used, bool):
        raise TypeError("fallback_used must be bool")
    try:
        scanner = stages["scanner"]
        voting = stages["voting"]
        thinning = stages["thinning"]
        growth_source = resolved_skinning["growth_source"]
        fallback_enabled = resolved_skinning["boundary_skinner_fallback"]
        post_thinning_policy = resolved_variant["post_thinning_policy"]
    except (KeyError, TypeError) as error:
        raise SkinArtifactValidationError("skin parent volume contract is incomplete") from error
    if any(not isinstance(value, str) or not value for value in (scanner, voting, thinning)):
        raise SkinArtifactValidationError("skin parent stage fingerprint is invalid")
    if growth_source not in {"pre_thin", "thinned"}:
        raise SkinArtifactValidationError("skin growth_source is invalid")
    if not isinstance(fallback_enabled, bool):
        raise SkinArtifactValidationError("skin boundary fallback setting is invalid")
    if not isinstance(post_thinning_policy, str) or post_thinning_policy not in {
        "none",
        "recenter_scanner_target",
        "boundary_edge_thin_v1",
    }:
        raise SkinArtifactValidationError("skin post-thinning policy is invalid")
    if fallback_used and not fallback_enabled:
        raise SkinArtifactValidationError("skin fallback state conflicts with resolved settings")
    if fallback_used:
        likelihood = ("thinning", thinning, "fvt.dat")
    elif growth_source == "pre_thin":
        likelihood = ("voting", voting, "fv.dat")
    else:
        # The thinning artifact stores the final fvt: base thinning for "none",
        # otherwise the output of the resolved post-thinning policy.
        assert growth_source == "thinned"
        assert post_thinning_policy in {
            "none",
            "recenter_scanner_target",
            "boundary_edge_thin_v1",
        }
        likelihood = ("thinning", thinning, "fvt.dat")
    return SkinParentVolumeContract(
        provenance=provenance,
        likelihood=likelihood,
        strike=("voting", voting, "vp.dat"),
        dip=("voting", voting, "vt.dat"),
        scanner_target=("scanner", scanner, "ft.dat") if fallback_enabled else None,
    )


def resolve_final_skin_cell_value_provenance(
    resolved_skinning: Mapping[str, Any],
    resolved_variant: Mapping[str, Any],
    *,
    fallback_used: bool,
    resolved_stage_settings: Mapping[str, Any] | None = None,
) -> str:
    """Resolve the final cell-value path from the complete final-phase contract."""

    if not isinstance(fallback_used, bool):
        raise TypeError("fallback_used must be bool")
    if not isinstance(resolved_skinning, Mapping) or not isinstance(resolved_variant, Mapping):
        raise SkinArtifactValidationError("skin provenance contract is incomplete")
    try:
        enabled = resolved_skinning["enabled"]
        method = resolved_skinning["method"]
        reskin = resolved_skinning["reskin"]
        growth_source = resolved_skinning["growth_source"]
        fallback_enabled = resolved_skinning["boundary_skinner_fallback"]
        fallback_policy = resolved_skinning["boundary_skinner_fallback_policy"]
        variant_name = resolved_variant["name"]
        post_thinning_policy = resolved_variant["post_thinning_policy"]
        variant_skinning = resolved_variant["skinning"]
    except KeyError as error:
        raise SkinArtifactValidationError("skin provenance contract is incomplete") from error
    if enabled is not True:
        raise SkinArtifactValidationError("skin provenance requires enabled skinning")
    if not isinstance(reskin, bool) or not isinstance(fallback_enabled, bool):
        raise SkinArtifactValidationError("skin provenance boolean setting is invalid")
    if method not in {"reference", "quality", "connected_component"}:
        raise SkinArtifactValidationError("skin method is invalid")
    if growth_source not in {"pre_thin", "thinned"}:
        raise SkinArtifactValidationError("skin growth_source is invalid")
    if not isinstance(fallback_policy, str) or not fallback_policy:
        raise SkinArtifactValidationError("skin fallback policy is invalid")
    if not isinstance(variant_name, str) or not variant_name:
        raise SkinArtifactValidationError("skin variant name is invalid")
    if not isinstance(post_thinning_policy, str) or not post_thinning_policy:
        raise SkinArtifactValidationError("skin post-thinning policy is invalid")
    if not isinstance(variant_skinning, Mapping):
        raise SkinArtifactValidationError("skin variant patch is invalid")
    semantic_contract_version = _semantic_contract_version(
        resolved_skinning,
        resolved_stage_settings,
    )
    if semantic_contract_version == F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION:
        try:
            reskin_policy = validate_reskin_policy(resolved_skinning["reskin_policy"])
        except (KeyError, TypeError, ValueError) as error:
            raise SkinArtifactValidationError("skin reskin_policy is invalid") from error
    else:
        reskin_policy = RESKIN_POLICY_EXISTING_CELLS_V1
    for name, resolved_value in (
        ("method", method),
        ("growth_source", growth_source),
        ("boundary_skinner_fallback", fallback_enabled),
        ("boundary_skinner_fallback_policy", fallback_policy),
        ("reskin_policy", reskin_policy),
    ):
        patched_value = variant_skinning.get(name)
        if patched_value is not None and patched_value != resolved_value:
            raise SkinArtifactValidationError(
                f"skin variant {name} conflicts with resolved settings"
            )
    if fallback_used and not fallback_enabled:
        raise SkinArtifactValidationError("skin fallback state conflicts with resolved settings")

    if resolved_stage_settings is not None:
        if not isinstance(resolved_stage_settings, Mapping):
            raise SkinArtifactValidationError("skinning stage settings must be an object")
        if (
            resolved_stage_settings.get("enabled") is not True
            or resolved_stage_settings.get("resolved_skinner_config") != dict(resolved_skinning)
            or resolved_stage_settings.get("growth_source") != growth_source
            or resolved_stage_settings.get("fallback_policy")
            != {
                "enabled": fallback_enabled,
                "policy": fallback_policy,
            }
            or resolved_stage_settings.get("skin_artifact_semantic_contract_version")
            != semantic_contract_version
        ):
            raise SkinArtifactValidationError(
                "skinning stage settings conflict with provenance inputs"
            )

    if fallback_used:
        return CONNECTED_COMPONENT_FALLBACK
    if reskin and method != "connected_component":
        if semantic_contract_version == F3_LEGACY_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION:
            return PRIMARY_RESKINNED
        if reskin_policy == RESKIN_POLICY_REFERENCE_DENSE_V1:
            return PRIMARY_DENSE_RESKINNED
        return PRIMARY_EXISTING_CELLS_RESKINNED
    if (
        method == "connected_component"
        and semantic_contract_version == F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION
    ):
        return PRIMARY_CONNECTED_COMPONENT
    return PRIMARY_NEAREST_SAMPLE


def _semantic_contract_version(
    resolved_skinning: Mapping[str, Any],
    resolved_stage_settings: Mapping[str, Any] | None,
) -> int:
    if resolved_stage_settings is None:
        version = (
            F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION
            if "reskin_policy" in resolved_skinning
            else F3_LEGACY_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION
        )
    else:
        version = resolved_stage_settings.get("skin_artifact_semantic_contract_version")
    if isinstance(version, bool) or version not in {
        F3_LEGACY_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
        F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
    }:
        raise SkinArtifactValidationError("skin artifact semantic contract version is invalid")
    return int(version)


def parse_skins_json(
    path: str | Path,
    shape: tuple[int, int, int],
) -> ParsedSkinArtifacts:
    """Parse one canonical ``skins.json`` and validate its low-cost schema."""

    volume_shape = _volume_shape(shape)
    artifact_path = Path(path)
    payload = _read_json_object(artifact_path, "skins.json")
    if set(payload) != _ROOT_FIELDS:
        raise SkinArtifactValidationError("skins.json root field set mismatch")
    format_version = payload["format_version"]
    if isinstance(format_version, bool) or format_version not in {1, 2}:
        raise SkinArtifactValidationError("skins.json format_version must be 1 or 2")
    cell_fields = _CELL_FIELDS_V1 if format_version == 1 else _CELL_FIELDS_V2
    if payload["skinning_enabled"] is not True:
        raise SkinArtifactValidationError("skins.json skinning_enabled must be true")
    skins_value = payload["skins"]
    if not isinstance(skins_value, list):
        raise SkinArtifactValidationError("skins.json skins must be an array")
    skin_count = _integer(payload["skin_count"], "skins.json skin_count")
    if skin_count != len(skins_value):
        raise SkinArtifactValidationError("skins.json skin_count mismatch")

    parsed_skins: list[tuple[SkinCellRecord, ...]] = []
    for expected_skin_index, skin_value in enumerate(skins_value):
        if not isinstance(skin_value, dict) or set(skin_value) != _SKIN_FIELDS:
            raise SkinArtifactValidationError("skins.json skin field set mismatch")
        skin_index = _integer(skin_value["skin_index"], "skins.json skin_index")
        if skin_index != expected_skin_index:
            raise SkinArtifactValidationError("skins.json skin_index mismatch")
        cells_value = skin_value["cells"]
        if not isinstance(cells_value, list):
            raise SkinArtifactValidationError("skins.json cells must be an array")
        cell_count = _integer(skin_value["cell_count"], "skins.json cell_count")
        if cell_count != len(cells_value):
            raise SkinArtifactValidationError("skins.json cell_count mismatch")

        cells: list[SkinCellRecord] = []
        previous_key: tuple[int, int, int] | None = None
        for cell_value in cells_value:
            if not isinstance(cell_value, dict) or set(cell_value) != cell_fields:
                raise SkinArtifactValidationError("skins.json cell field set mismatch")
            indices = tuple(
                _integer(cell_value[name], f"skins.json cell {name}") for name in _INDEX_FIELDS
            )
            _validate_bounds(indices, volume_shape)
            scalars = {
                name: _finite_scalar(cell_value[name], f"skins.json cell {name}")
                for name in _SCALAR_FIELDS
            }
            _validate_cell_values(
                indices,
                scalars,
                volume_shape,
            )
            generation: str | None = None
            reskin_support: float | None = None
            if format_version == 2:
                generation = cell_value["generation"]
                support_value = cell_value["reskin_support"]
                if support_value is not None:
                    reskin_support = _finite_scalar(
                        support_value,
                        "skins.json cell reskin_support",
                    )
                _validate_generation_support(generation, reskin_support)
            canonical_key = (indices[2], indices[1], indices[0])
            if previous_key is not None and canonical_key < previous_key:
                raise SkinArtifactValidationError("skins.json cell order is not canonical")
            previous_key = canonical_key
            cells.append(
                SkinCellRecord(
                    x1=scalars["x1"],
                    x2=scalars["x2"],
                    x3=scalars["x3"],
                    i1=indices[0],
                    i2=indices[1],
                    i3=indices[2],
                    fl=scalars["fl"],
                    fp=scalars["fp"],
                    ft=scalars["ft"],
                    generation=generation,
                    reskin_support=reskin_support,
                )
            )
        parsed_skins.append(tuple(cells))
    parsed = ParsedSkinArtifacts(tuple(parsed_skins), format_version=format_version)
    if _canonical_json_bytes(payload) != _canonical_json_bytes(
        _canonical_skins_payload_for_format(
            parsed.skins,
            format_version=format_version,
        )
    ):
        raise SkinArtifactValidationError(
            "skins.json values do not match the canonical skin serializer"
        )
    return parsed


def _validate_generation_support(generation: Any, support: Any) -> None:
    if not isinstance(generation, str) or generation not in _CELL_GENERATIONS:
        raise SkinArtifactValidationError("skins.json cell generation is invalid")
    if generation in _DENSE_GENERATIONS:
        if (
            isinstance(support, (bool, np.bool_))
            or not isinstance(support, (int, float, np.integer, np.floating))
            or not math.isfinite(float(support))
            or not 0.0 <= float(support) <= 1.0
        ):
            raise SkinArtifactValidationError(
                "skins.json dense cell reskin_support must be finite and in [0, 1]"
            )
    elif support is not None:
        raise SkinArtifactValidationError("skins.json non-dense cell reskin_support must be null")


def validate_skin_artifact_semantics(
    stage_path: str | Path,
    shape: tuple[int, int, int],
    *,
    small_skin_size: int,
    parsed: ParsedSkinArtifacts | None = None,
) -> ParsedSkinArtifacts:
    """Cross-check ``skins.json``, ``skin_mask.dat``, and report topology."""

    volume_shape = _volume_shape(shape)
    root = Path(stage_path)
    skin_data = parsed if parsed is not None else parse_skins_json(root / "skins.json", shape)
    if not isinstance(skin_data, ParsedSkinArtifacts):
        raise TypeError("parsed must be ParsedSkinArtifacts or None")
    if isinstance(small_skin_size, bool) or not isinstance(small_skin_size, int):
        raise SkinArtifactValidationError("small_skin_size must be an integer")
    if small_skin_size < 0:
        raise SkinArtifactValidationError("small_skin_size must be non-negative")

    _validate_skin_mask(root / "skin_mask.dat", volume_shape, skin_data.unique_indices)
    topology = skin_topology_metrics(
        skin_data.skins,
        volume_shape,
        small_skin_size=small_skin_size,
    )
    report = _read_json_object(root / "report.json", "skinning report")
    stored_topology = report.get("topology")
    if not isinstance(stored_topology, Mapping):
        raise SkinArtifactValidationError("skinning report topology must be an object")
    for name in _TOPOLOGY_FIELDS:
        if name not in stored_topology:
            raise SkinArtifactValidationError(f"skinning report topology missing {name}")
        stored = stored_topology[name]
        if name in _INTEGER_TOPOLOGY_FIELDS:
            stored = _integer(stored, f"skinning report topology {name}")
        else:
            stored = _finite_scalar(stored, f"skinning report topology {name}")
        if stored != topology[name]:
            raise SkinArtifactValidationError(f"skinning report topology mismatch: {name}")
    _validate_final_diagnostic_mapping(report.get("diagnostics"), topology)
    return skin_data


def validate_skin_generation_provenance(
    parsed: ParsedSkinArtifacts,
    provenance: str,
    *,
    semantic_contract_version: int,
    reskin_diagnostics: Mapping[str, Any] | None = None,
) -> None:
    """Cross-check v2 cell generations against the resolved final phase."""

    if semantic_contract_version == F3_LEGACY_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION:
        if parsed.format_version != 1:
            raise SkinArtifactValidationError("contract 3 requires skins.json format version 1")
        return
    if semantic_contract_version != F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION:
        raise SkinArtifactValidationError("skin artifact semantic contract version is invalid")
    if parsed.format_version != 2:
        raise SkinArtifactValidationError("contract 4 requires skins.json format version 2")

    generations = {cell.generation for skin in parsed.skins for cell in skin}
    allowed = {
        PRIMARY_NEAREST_SAMPLE: {
            FAULT_CELL_GENERATION_GROWN,
        },
        PRIMARY_CONNECTED_COMPONENT: {
            FAULT_CELL_GENERATION_CONNECTED_COMPONENT,
        },
        PRIMARY_EXISTING_CELLS_RESKINNED: {
            FAULT_CELL_GENERATION_GROWN,
            FAULT_CELL_GENERATION_EXISTING_CELLS_RESKINNED,
        },
        PRIMARY_DENSE_RESKINNED: {
            FAULT_CELL_GENERATION_GROWN,
            FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED,
            FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED,
        },
        CONNECTED_COMPONENT_FALLBACK: {
            FAULT_CELL_GENERATION_CONNECTED_COMPONENT,
        },
    }.get(provenance)
    if allowed is None:
        raise SkinArtifactValidationError("skin cell provenance is invalid")
    if not generations.issubset(allowed):
        raise SkinArtifactValidationError(
            "skins.json cell generation conflicts with final provenance"
        )
    if reskin_diagnostics is None:
        return
    if provenance not in {
        PRIMARY_EXISTING_CELLS_RESKINNED,
        PRIMARY_DENSE_RESKINNED,
    }:
        raise SkinArtifactValidationError(
            "reskin diagnostics conflict with final provenance"
        )
    if not isinstance(reskin_diagnostics, Mapping):
        raise SkinArtifactValidationError("reskin diagnostics must be an object")

    expected_policy = (
        RESKIN_POLICY_EXISTING_CELLS_V1
        if provenance == PRIMARY_EXISTING_CELLS_RESKINNED
        else RESKIN_POLICY_REFERENCE_DENSE_V1
    )
    if reskin_diagnostics.get("reskin_policy") != expected_policy:
        raise SkinArtifactValidationError(
            "reskin diagnostics policy conflicts with final provenance"
        )
    if any(
        cell.generation == FAULT_CELL_GENERATION_GROWN
        for skin in parsed.skins
        if len(skin) > 1
        for cell in skin
    ):
        raise SkinArtifactValidationError(
            "multi-cell reskinned skin cannot retain grown cell generation"
        )
    reskin_applied = reskin_diagnostics.get("reskin_applied")
    if not isinstance(reskin_applied, bool):
        raise SkinArtifactValidationError("reskin diagnostics reskin_applied must be bool")
    policy_generations = (
        {FAULT_CELL_GENERATION_EXISTING_CELLS_RESKINNED}
        if provenance == PRIMARY_EXISTING_CELLS_RESKINNED
        else _DENSE_GENERATIONS
    )
    if reskin_applied != bool(generations & policy_generations):
        raise SkinArtifactValidationError(
            "reskin diagnostics reskin_applied does not match skins.json generations"
        )

    generated_count = sum(
        cell.generation == FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED
        for skin in parsed.skins
        for cell in skin
    )
    expected_counts = {
        "output_cell_count": parsed.cell_count,
        "observed_output_cell_count": parsed.cell_count - generated_count,
        "generated_cell_count": generated_count,
    }
    for name, expected in expected_counts.items():
        actual = _integer(
            reskin_diagnostics.get(name),
            f"reskin diagnostics {name}",
        )
        if actual < 0:
            raise SkinArtifactValidationError(
                f"reskin diagnostics {name} must be non-negative"
            )
        if actual != expected:
            raise SkinArtifactValidationError(
                f"reskin diagnostics {name} does not match skins.json generations"
            )


def _validate_skin_mask(
    path: Path,
    shape: tuple[int, int, int],
    unique_indices: frozenset[tuple[int, int, int]],
) -> None:
    if not path.is_file() or path.is_symlink():
        raise SkinArtifactValidationError("skin_mask.dat must be a regular non-symlink file")
    expected_bytes = math.prod(shape) * _DAT_DTYPE.itemsize
    try:
        if path.stat().st_size != expected_bytes:
            raise SkinArtifactValidationError("skin_mask.dat shape or dtype size mismatch")
        mask = np.memmap(path, dtype=_DAT_DTYPE, mode="r", shape=shape)
    except OSError as error:
        raise SkinArtifactValidationError("skin_mask.dat is unreadable") from error
    try:
        plane_size = shape[1] * shape[2]
        slab_depth = max(1, _MASK_SLAB_VOXELS // plane_size)
        one_count = 0
        for start in range(0, shape[0], slab_depth):
            slab = mask[start : start + slab_depth]
            if not bool(np.isfinite(slab).all()):
                raise SkinArtifactValidationError("skin_mask.dat contains non-finite values")
            if not bool(np.logical_or(slab == 0.0, slab == 1.0).all()):
                raise SkinArtifactValidationError("skin_mask.dat values must be exactly 0.0 or 1.0")
            one_count += int(np.count_nonzero(slab))
        index_iterator = iter(unique_indices)
        while chunk := tuple(islice(index_iterator, _INDEX_CHUNK_SIZE)):
            i1 = np.fromiter((index[0] for index in chunk), dtype=np.intp)
            i2 = np.fromiter((index[1] for index in chunk), dtype=np.intp)
            i3 = np.fromiter((index[2] for index in chunk), dtype=np.intp)
            if not bool((mask[i3, i2, i1] == 1.0).all()):
                raise SkinArtifactValidationError(
                    "skin_mask.dat does not match skins.json voxel mask"
                )
        if one_count != len(unique_indices):
            raise SkinArtifactValidationError("skin_mask.dat does not match skins.json voxel mask")
    finally:
        mapping = getattr(mask, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()


def _validate_final_diagnostic_mapping(
    value: Any,
    topology: Mapping[str, float | int],
) -> None:
    if not isinstance(value, Mapping):
        raise SkinArtifactValidationError("skinning report diagnostics must be an object")
    fallback_used = value.get("fallback_used")
    if not isinstance(fallback_used, bool):
        raise SkinArtifactValidationError("skinning report diagnostics fallback_used must be bool")
    fields = (
        (("fallback_skin_count", "skin_count"), ("fallback_cell_count", "cell_count"))
        if fallback_used
        else (("accepted_skin_count", "skin_count"), ("accepted_cell_count", "cell_count"))
    )
    for diagnostic_name, topology_name in fields:
        diagnostic_value = _integer(
            value.get(diagnostic_name),
            f"skinning report diagnostics {diagnostic_name}",
        )
        if diagnostic_value != topology[topology_name]:
            raise SkinArtifactValidationError(
                f"skinning report diagnostics mismatch: {diagnostic_name}"
            )


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise SkinArtifactValidationError(f"{context} must be a regular non-symlink file")
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda token: _raise_nonfinite(token),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SkinArtifactValidationError(f"{context} is not strict JSON") from error
    if not isinstance(value, dict):
        raise SkinArtifactValidationError(f"{context} must be an object")
    return value


def _raise_nonfinite(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _volume_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if (
        not isinstance(shape, tuple)
        or len(shape) != 3
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
    ):
        raise TypeError("shape must contain exactly three positive integers")
    return shape


def _integer(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SkinArtifactValidationError(f"{context} must be an integer")
    return value


def _finite_scalar(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SkinArtifactValidationError(f"{context} must be a finite scalar")
    try:
        result = float(value)
    except OverflowError as error:
        raise SkinArtifactValidationError(f"{context} must be finite") from error
    if not math.isfinite(result):
        raise SkinArtifactValidationError(f"{context} must be finite")
    return result


def _validate_bounds(
    index: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> None:
    i1, i2, i3 = index
    n3, n2, n1 = shape
    if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
        raise SkinArtifactValidationError("skins.json cell index is out of bounds")


def _validate_cell_values(
    indices: tuple[int, int, int],
    scalars: Mapping[str, float],
    shape: tuple[int, int, int],
) -> None:
    n3, n2, n1 = shape
    coordinates = (scalars["x1"], scalars["x2"], scalars["x3"])
    for axis, (coordinate, index, size) in enumerate(
        zip(coordinates, indices, (n1, n2, n3), strict=True),
        start=1,
    ):
        if not (-0.5 <= coordinate < size - 0.5):
            raise SkinArtifactValidationError(
                f"skins.json cell x{axis} is outside the voxel-center domain"
            )
        if math.floor(coordinate + 0.5) != index:
            raise SkinArtifactValidationError(
                f"skins.json cell x{axis}/i{axis} Java rounding mismatch"
            )
    if not 0.0 <= scalars["fl"] <= 1.0:
        raise SkinArtifactValidationError("skins.json cell fl is outside [0, 1]")
    if not 0.0 <= scalars["fp"] < 360.0:
        raise SkinArtifactValidationError(
            "skins.json cell fp is outside canonical strike orientation domain"
        )
    if not 0.0 <= scalars["ft"] <= 90.0:
        raise SkinArtifactValidationError(
            "skins.json cell ft is outside canonical dip orientation domain"
        )


__all__ = [
    "ParsedSkinArtifacts",
    "F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION",
    "SkinArtifactValidationError",
    "SkinCellRecord",
    "SkinParentVolumeContract",
    "canonical_skins_payload",
    "parse_skins_json",
    "resolve_final_skin_cell_value_provenance",
    "resolve_skin_parent_volume_contract",
    "validate_skin_artifact_semantics",
]
