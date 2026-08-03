from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyosv.evaluation.f3d_mode_comparison.skin_artifacts import (
    F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
    ParsedSkinArtifacts,
    SkinCellRecord,
    SkinArtifactValidationError,
    canonical_skins_payload,
    parse_skins_json,
    resolve_final_skin_cell_value_provenance,
    resolve_skin_parent_volume_contract,
    validate_skin_artifact_semantics,
    validate_skin_generation_provenance,
    validate_reskin_diagnostics_contract,
)

_SHAPE = (2, 3, 4)


def _cell(i1: int, i2: int, i3: int, *, fl: float = 0.8) -> dict[str, Any]:
    return {
        "x1": float(i1),
        "x2": float(i2),
        "x3": float(i3),
        "i1": i1,
        "i2": i2,
        "i3": i3,
        "fl": fl,
        "fp": 25.0,
        "ft": 70.0,
    }


def _payload() -> dict[str, Any]:
    cells = [_cell(0, 0, 0), _cell(0, 0, 0)]
    return {
        "format_version": 1,
        "skinning_enabled": True,
        "skin_count": 2,
        "skins": [
            {"skin_index": 0, "cell_count": len(cells), "cells": cells},
            {"skin_index": 1, "cell_count": 1, "cells": [_cell(2, 1, 1)]},
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _single_cell_payload(cell: dict[str, Any]) -> dict[str, Any]:
    return {
        "format_version": 1,
        "skinning_enabled": True,
        "skin_count": 1,
        "skins": [{"skin_index": 0, "cell_count": 1, "cells": [cell]}],
    }


def _resolved_skinning(
    *,
    growth_source: str = "thinned",
    method: str = "reference",
    reskin: bool = False,
    fallback_enabled: bool = False,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "method": method,
        "reskin": reskin,
        "growth_source": growth_source,
        "boundary_skinner_fallback": fallback_enabled,
        "boundary_skinner_fallback_policy": "empty_primary",
    }


def _resolved_variant(post_thinning_policy: str = "none") -> dict[str, Any]:
    return {
        "name": "f3-canonical",
        "post_thinning_policy": post_thinning_policy,
        "skinning": {},
    }


def _write_stage(path: Path) -> None:
    path.mkdir()
    _write_json(path / "skins.json", _payload())
    mask = np.zeros(_SHAPE, dtype=">f4")
    mask[0, 0, 0] = 1.0
    mask[1, 1, 2] = 1.0
    mask.tofile(path / "skin_mask.dat")
    _write_json(
        path / "report.json",
        {
            "diagnostics": {
                "accepted_skin_count": 2,
                "accepted_cell_count": 3,
                "fallback_used": False,
                "fallback_skin_count": 0,
                "fallback_cell_count": 0,
            },
            "topology": {
                "skin_count": 2,
                "cell_count": 3,
                "unique_cell_count": 2,
                "duplicate_cell_count": 1,
                "largest_skin_size": 2,
                "largest_skin_fraction": 2 / 3,
                "small_skin_size": 2,
                "small_skin_count": 1,
                "small_skin_cell_count": 1,
                "small_skin_cell_fraction": 1 / 3,
            },
        },
    )


def _mutate_payload(payload: dict[str, Any], case: str) -> None:
    first_skin = payload["skins"][0]
    first_cell = first_skin["cells"][0]
    if case == "root_field":
        payload["unknown"] = 1
    elif case == "cell_field":
        first_cell.pop("ft")
    elif case == "bounds":
        first_cell["i1"] = _SHAPE[2]
    elif case == "bool_index":
        first_cell["i1"] = True
    elif case == "nonfinite":
        first_cell["fl"] = float("nan")
    elif case == "skin_count":
        payload["skin_count"] += 1
    elif case == "cell_count":
        first_skin["cell_count"] += 1
    elif case == "skin_index":
        first_skin["skin_index"] = 1
    elif case == "order":
        first_skin["cells"] = [_cell(1, 0, 0), _cell(0, 0, 0)]
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    (
        "root_field",
        "cell_field",
        "bounds",
        "bool_index",
        "nonfinite",
        "skin_count",
        "cell_count",
        "skin_index",
        "order",
    ),
)
def test_parse_skins_json_rejects_noncanonical_payload(
    tmp_path: Path,
    case: str,
) -> None:
    payload = copy.deepcopy(_payload())
    _mutate_payload(payload, case)
    path = tmp_path / "skins.json"
    _write_json(path, payload)

    with pytest.raises(SkinArtifactValidationError):
        parse_skins_json(path, _SHAPE)


def test_duplicate_cells_are_retained_and_cross_file_validation_passes(
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    _write_stage(stage)

    parsed = parse_skins_json(stage / "skins.json", _SHAPE)

    assert parsed.cell_count == 3
    assert len(parsed.unique_indices) == 2
    assert parsed.duplicate_cell_count == 1
    assert (
        validate_skin_artifact_semantics(
            stage,
            _SHAPE,
            small_skin_size=2,
            parsed=parsed,
        )
        is parsed
    )


def test_canonical_serializer_preserves_cell_attributes() -> None:
    cell = SkinCellRecord(
        0.5,
        0.0,
        0.0,
        1,
        0,
        0,
        0.1,
        10.0,
        65.0,
        "grown",
        None,
    )

    payload = canonical_skins_payload(((cell,),))

    assert payload["skins"][0]["cells"][0] == {
        **_cell(1, 0, 0, fl=0.1),
        "x1": 0.5,
        "fp": 10.0,
        "ft": 65.0,
        "generation": "grown",
        "reskin_support": None,
    }
    assert payload["format_version"] == 2


def test_canonical_serializer_rejects_empty_skin_but_accepts_zero_skins() -> None:
    assert canonical_skins_payload([]) == {
        "format_version": 2,
        "skinning_enabled": True,
        "skin_count": 0,
        "skins": [],
    }

    with pytest.raises(
        SkinArtifactValidationError,
        match="canonical skin artifacts must not contain empty skins",
    ):
        canonical_skins_payload([()])

    cell = SkinCellRecord(
        0.0,
        0.0,
        0.0,
        0,
        0,
        0,
        0.8,
        25.0,
        70.0,
        "grown",
        None,
    )
    assert canonical_skins_payload([(cell,)])["skin_count"] == 1


def test_parser_rejects_empty_canonical_skin(tmp_path: Path) -> None:
    path = tmp_path / "skins.json"
    _write_json(
        path,
        {
            "format_version": 2,
            "skinning_enabled": True,
            "skin_count": 1,
            "skins": [
                {
                    "skin_index": 0,
                    "cell_count": 0,
                    "cells": [],
                }
            ],
        },
    )

    with pytest.raises(SkinArtifactValidationError, match="empty skins"):
        parse_skins_json(path, _SHAPE)


def test_parser_accepts_zero_skin_canonical_artifact(tmp_path: Path) -> None:
    path = tmp_path / "skins.json"
    _write_json(
        path,
        {
            "format_version": 2,
            "skinning_enabled": True,
            "skin_count": 0,
            "skins": [],
        },
    )

    parsed = parse_skins_json(path, _SHAPE)

    assert parsed.skins == ()
    assert parsed.cell_count == 0


def test_canonical_v2_round_trip_preserves_generation_and_support(tmp_path: Path) -> None:
    cells = (
        SkinCellRecord(
            0.0,
            0.0,
            0.0,
            0,
            0,
            0,
            0.8,
            25.0,
            70.0,
            "dense_reskin_observed",
            0.75,
        ),
        SkinCellRecord(
            1.0,
            0.0,
            0.0,
            1,
            0,
            0,
            0.7,
            25.0,
            70.0,
            "dense_reskin_generated",
            0.5,
        ),
    )
    payload = canonical_skins_payload((cells,))
    path = tmp_path / "skins.json"
    _write_json(path, payload)

    parsed = parse_skins_json(path, _SHAPE)

    assert payload["format_version"] == 2
    assert parsed.format_version == 2
    assert [cell.generation for cell in parsed.skins[0]] == [
        "dense_reskin_observed",
        "dense_reskin_generated",
    ]
    assert [cell.reskin_support for cell in parsed.skins[0]] == [0.75, 0.5]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("generation", "unknown"),
        ("reskin_support", None),
        ("reskin_support", -0.1),
        ("reskin_support", 1.1),
    ),
)
def test_v2_parser_rejects_invalid_dense_metadata(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    cell = {
        **_cell(0, 0, 0),
        "generation": "dense_reskin_generated",
        "reskin_support": 0.5,
    }
    cell[field] = value
    payload = {
        "format_version": 2,
        "skinning_enabled": True,
        "skin_count": 1,
        "skins": [{"skin_index": 0, "cell_count": 1, "cells": [cell]}],
    }
    path = tmp_path / "skins.json"
    _write_json(path, payload)

    with pytest.raises(SkinArtifactValidationError):
        parse_skins_json(path, _SHAPE)


def test_v2_parser_rejects_versioned_field_set_mismatch(tmp_path: Path) -> None:
    payload = _single_cell_payload(_cell(0, 0, 0))
    payload["format_version"] = 2
    path = tmp_path / "skins.json"
    _write_json(path, payload)

    with pytest.raises(SkinArtifactValidationError, match="field set"):
        parse_skins_json(path, _SHAPE)


@pytest.mark.parametrize(
    ("growth_source", "post_thinning_policy", "fallback_used", "expected"),
    (
        ("pre_thin", "none", False, ("voting", "vote", "fv.dat")),
        (
            "pre_thin",
            "recenter_scanner_target",
            False,
            ("voting", "vote", "fv.dat"),
        ),
        ("thinned", "none", False, ("thinning", "thin", "fvt.dat")),
        (
            "thinned",
            "boundary_edge_thin_v1",
            False,
            ("thinning", "thin", "fvt.dat"),
        ),
        (
            "pre_thin",
            "recenter_scanner_target",
            True,
            ("thinning", "thin", "fvt.dat"),
        ),
    ),
)
def test_parent_likelihood_uses_resolved_execution_contract(
    growth_source: str,
    post_thinning_policy: str,
    fallback_used: bool,
    expected: tuple[str, str, str],
) -> None:
    contract = resolve_skin_parent_volume_contract(
        {"scanner": "scan", "voting": "vote", "thinning": "thin"},
        _resolved_skinning(
            growth_source=growth_source,
            fallback_enabled=fallback_used,
        ),
        _resolved_variant(post_thinning_policy),
        fallback_used=fallback_used,
    )

    assert contract.likelihood == expected


def test_parent_volume_contract_rejects_unknown_post_thinning_policy() -> None:
    with pytest.raises(SkinArtifactValidationError, match="post-thinning"):
        resolve_skin_parent_volume_contract(
            {"scanner": "scan", "voting": "vote", "thinning": "thin"},
            _resolved_skinning(),
            _resolved_variant("unknown"),
            fallback_used=False,
        )


@pytest.mark.parametrize(
    ("reskin", "fallback_used", "expected"),
    (
        (False, False, "primary_nearest_sample"),
        (True, False, "primary_reskinned"),
        (False, True, "connected_component_fallback"),
        (True, True, "connected_component_fallback"),
    ),
)
def test_final_cell_value_provenance_follows_final_phase_contract(
    reskin: bool,
    fallback_used: bool,
    expected: str,
) -> None:
    assert (
        resolve_final_skin_cell_value_provenance(
            _resolved_skinning(
                reskin=reskin,
                fallback_enabled=fallback_used,
            ),
            _resolved_variant(),
            fallback_used=fallback_used,
        )
        == expected
    )


def test_connected_component_primary_ignores_reskin_setting_for_provenance() -> None:
    assert (
        resolve_final_skin_cell_value_provenance(
            _resolved_skinning(method="connected_component", reskin=True),
            _resolved_variant(),
            fallback_used=False,
        )
        == "primary_nearest_sample"
    )


def test_contract4_connected_component_has_method_specific_provenance() -> None:
    assert (
        resolve_final_skin_cell_value_provenance(
            {
                **_resolved_skinning(method="connected_component", reskin=True),
                "reskin_policy": "existing_cells_v1",
            },
            _resolved_variant(),
            fallback_used=False,
        )
        == "primary_connected_component"
    )


@pytest.mark.parametrize(
    ("provenance", "generation", "valid"),
    (
        ("primary_nearest_sample", "grown", True),
        ("primary_nearest_sample", "connected_component", False),
        ("primary_connected_component", "connected_component", True),
        ("primary_connected_component", "grown", False),
    ),
)
def test_contract4_generation_validation_is_method_specific(
    provenance: str,
    generation: str,
    valid: bool,
) -> None:
    parsed = ParsedSkinArtifacts(
        skins=(
            (
                SkinCellRecord(
                    0.0,
                    0.0,
                    0.0,
                    0,
                    0,
                    0,
                    0.8,
                    25.0,
                    70.0,
                    generation,
                    None,
                ),
            ),
        ),
        format_version=2,
    )

    if valid:
        validate_skin_generation_provenance(
            parsed,
            provenance,
            semantic_contract_version=4,
        )
    else:
        with pytest.raises(SkinArtifactValidationError, match="generation conflicts"):
            validate_skin_generation_provenance(
                parsed,
                provenance,
                semantic_contract_version=4,
            )


def _contract5_reskin_diagnostics() -> dict[str, Any]:
    counts = {
        "reskin_applied": True,
        "processed_skin_count": 1,
        "input_cell_count": 1,
        "output_cell_count": 1,
        "observed_output_cell_count": 1,
        "generated_cell_count": 0,
        "dropped_input_cell_count": 0,
        "projected_local_duplicate_count": 0,
        "candidate_local_key_count": 0,
        "rejected_support_count": 0,
        "rejected_invalid_mask_count": 0,
        "rejected_prior_skin_collision_count": 0,
        "rejected_out_of_bounds_count": 0,
        "rejected_duplicate_world_index_count": 0,
        "max_generated_chebyshev_distance_from_observed": 0,
    }
    return {
        "reskin_diagnostics_contract_version": 2,
        "reskin_policy": "existing_cells_v1",
        **counts,
        "attempted": dict(counts),
    }


def test_contract4_dense_generation_must_match_reskin_diagnostics() -> None:
    parsed = ParsedSkinArtifacts(
        skins=(
            (
                SkinCellRecord(
                    0.0,
                    0.0,
                    0.0,
                    0,
                    0,
                    0,
                    0.8,
                    25.0,
                    70.0,
                    "grown",
                    None,
                ),
            ),
        ),
        format_version=2,
    )
    diagnostics = _contract5_reskin_diagnostics()
    diagnostics.pop("reskin_diagnostics_contract_version")
    diagnostics.pop("attempted")
    diagnostics.update(
        {
            "reskin_policy": "reference_dense_v1",
            "observed_output_cell_count": 0,
            "generated_cell_count": 1,
        }
    )

    with pytest.raises(SkinArtifactValidationError, match="does not match skins.json generations"):
        validate_skin_generation_provenance(
            parsed,
            "primary_dense_reskinned",
            semantic_contract_version=4,
            reskin_diagnostics=diagnostics,
        )


def _contract5_parsed_skin() -> ParsedSkinArtifacts:
    return ParsedSkinArtifacts(
        skins=(
            (
                SkinCellRecord(
                    0.0,
                    0.0,
                    0.0,
                    0,
                    0,
                    0,
                    0.8,
                    25.0,
                    70.0,
                    "existing_cells_reskinned",
                    None,
                ),
            ),
        ),
        format_version=2,
    )


def test_contract5_validates_final_and_attempted_reskin_diagnostics() -> None:
    validate_skin_generation_provenance(
        _contract5_parsed_skin(),
        "primary_existing_cells_reskinned",
        semantic_contract_version=F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
        reskin_diagnostics=_contract5_reskin_diagnostics(),
    )

    historical = _contract5_reskin_diagnostics()
    historical.pop("reskin_diagnostics_contract_version")
    historical.pop("attempted")
    validate_skin_generation_provenance(
        _contract5_parsed_skin(),
        "primary_existing_cells_reskinned",
        semantic_contract_version=4,
        reskin_diagnostics=historical,
    )


def test_contract5_fallback_keeps_primary_attempted_counts_separate() -> None:
    """Fallback final counts need not be bounded by primary attempt counts."""

    parsed = _contract5_parsed_skin()
    cell = parsed.skins[0][0]
    fallback = ParsedSkinArtifacts(
        skins=((replace(cell, generation="connected_component"),),),
        format_version=parsed.format_version,
    )
    diagnostics = _contract5_reskin_diagnostics()
    diagnostics["reskin_applied"] = False
    diagnostics["attempted"] = {
        **diagnostics["attempted"],
        "reskin_applied": True,
        "processed_skin_count": 1,
        "input_cell_count": 1,
        "output_cell_count": 1,
        "observed_output_cell_count": 1,
    }

    validate_skin_generation_provenance(
        fallback,
        "connected_component_fallback",
        semantic_contract_version=F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
        reskin_diagnostics=diagnostics,
    )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("reskin_policy", "invalid", "policy is invalid"),
        ("reskin_applied", 1, "reskin_applied must be bool"),
    ),
)
def test_contract5_rejects_invalid_final_reskin_identity(
    field: str,
    value: Any,
    match: str,
) -> None:
    diagnostics = _contract5_reskin_diagnostics()
    diagnostics[field] = value

    with pytest.raises(SkinArtifactValidationError, match=match):
        validate_reskin_diagnostics_contract(
            diagnostics,
            semantic_contract_version=F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
            skin_count=1,
        )


@pytest.mark.parametrize(
    ("provenance", "generation"),
    (
        ("primary_nearest_sample", "grown"),
        ("primary_connected_component", "connected_component"),
        ("connected_component_fallback", "connected_component"),
    ),
)
def test_contract5_non_reskinned_provenance_matches_final_diagnostics(
    provenance: str,
    generation: str,
) -> None:
    cell = replace(_contract5_parsed_skin().skins[0][0], generation=generation)
    parsed = ParsedSkinArtifacts(skins=((cell,),), format_version=2)
    diagnostics = _contract5_reskin_diagnostics()
    diagnostics["reskin_applied"] = False
    diagnostics["attempted"] = {
        name: False if name == "reskin_applied" else 0 for name in diagnostics["attempted"]
    }

    validate_skin_generation_provenance(
        parsed,
        provenance,
        semantic_contract_version=F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
        reskin_diagnostics=diagnostics,
    )

    diagnostics["processed_skin_count"] = 2
    with pytest.raises(SkinArtifactValidationError, match="processed_skin_count"):
        validate_skin_generation_provenance(
            parsed,
            provenance,
            semantic_contract_version=F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
            reskin_diagnostics=diagnostics,
        )


def test_contract4_rejects_diagnostics_v2_fields() -> None:
    with pytest.raises(SkinArtifactValidationError, match="v1 field set mismatch"):
        validate_skin_generation_provenance(
            _contract5_parsed_skin(),
            "primary_existing_cells_reskinned",
            semantic_contract_version=4,
            reskin_diagnostics=_contract5_reskin_diagnostics(),
        )


def test_contract5_allows_zero_attempted_counts_when_reskin_phase_is_not_reached() -> None:
    diagnostics = _contract5_reskin_diagnostics()
    diagnostics["reskin_applied"] = False
    diagnostics["attempted"] = {
        name: False if name == "reskin_applied" else 0 for name in diagnostics["attempted"]
    }
    parsed = _contract5_parsed_skin()
    cell = parsed.skins[0][0]
    parsed = ParsedSkinArtifacts(
        skins=((replace(cell, generation="grown"),),),
        format_version=parsed.format_version,
    )

    validate_skin_generation_provenance(
        parsed,
        "primary_existing_cells_reskinned",
        semantic_contract_version=F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
        reskin_diagnostics=diagnostics,
    )


@pytest.mark.parametrize(
    "mutate",
    (
        lambda diagnostics: diagnostics.__setitem__(
            "reskin_diagnostics_contract_version",
            1,
        ),
        lambda diagnostics: diagnostics.__setitem__("output_cell_count", 2),
        lambda diagnostics: diagnostics["attempted"].__setitem__(
            "processed_skin_count",
            0,
        ),
        lambda diagnostics: diagnostics.__setitem__("unexpected", 0),
        lambda diagnostics: diagnostics["attempted"].__setitem__("unexpected", 0),
    ),
)
def test_contract5_rejects_reskin_diagnostics_tamper(mutate: Any) -> None:
    diagnostics = _contract5_reskin_diagnostics()
    mutate(diagnostics)

    with pytest.raises(SkinArtifactValidationError):
        validate_skin_generation_provenance(
            _contract5_parsed_skin(),
            "primary_existing_cells_reskinned",
            semantic_contract_version=F3_SKIN_ARTIFACT_SEMANTIC_CONTRACT_VERSION,
            reskin_diagnostics=diagnostics,
        )


@pytest.mark.parametrize(
    ("policy", "expected"),
    (
        ("existing_cells_v1", "primary_existing_cells_reskinned"),
        ("reference_dense_v1", "primary_dense_reskinned"),
    ),
)
def test_contract4_provenance_distinguishes_reskin_policy(
    policy: str,
    expected: str,
) -> None:
    skinning = {
        **_resolved_skinning(reskin=True),
        "reskin_policy": policy,
    }

    assert (
        resolve_final_skin_cell_value_provenance(
            skinning,
            _resolved_variant(),
            fallback_used=False,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("x1", "i1", "valid"),
    (
        (-0.5, 0, True),
        (float(np.nextafter(np.float32(0.5), np.float32(-np.inf))), 0, True),
        (0.5, 1, True),
        (float(np.nextafter(np.float32(-0.5), np.float32(-np.inf))), 0, False),
        (_SHAPE[2] - 0.5, _SHAPE[2] - 1, False),
        (0.5, 0, False),
    ),
)
def test_cell_coordinates_use_java_rounding_and_half_open_volume_bounds(
    tmp_path: Path,
    x1: float,
    i1: int,
    valid: bool,
) -> None:
    path = tmp_path / "skins.json"
    cell = _cell(i1, 0, 0)
    cell["x1"] = x1
    _write_json(path, _single_cell_payload(cell))

    if valid:
        assert parse_skins_json(path, _SHAPE).cell_count == 1
    else:
        with pytest.raises(SkinArtifactValidationError):
            parse_skins_json(path, _SHAPE)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("fl", 2.0, "fl"),
        ("fp", 360.0, "strike"),
        ("ft", 91.0, "dip"),
    ),
)
def test_cell_attributes_respect_final_cell_domains(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    path = tmp_path / "skins.json"
    cell = _cell(0, 0, 0)
    cell[field] = value
    _write_json(path, _single_cell_payload(cell))

    with pytest.raises(SkinArtifactValidationError, match=message):
        parse_skins_json(path, _SHAPE)


def test_cell_orientation_is_not_limited_to_scanner_search_range(
    tmp_path: Path,
) -> None:
    path = tmp_path / "skins.json"
    cell = _cell(0, 0, 0)
    cell["fp"] = 200.0
    cell["ft"] = 20.0
    _write_json(path, _single_cell_payload(cell))

    assert parse_skins_json(path, _SHAPE).cell_count == 1


@pytest.mark.parametrize("level", ("root", "skin", "cell"))
def test_duplicate_json_keys_are_rejected_at_every_object_level(
    tmp_path: Path,
    level: str,
) -> None:
    text = json.dumps(_single_cell_payload(_cell(0, 0, 0)), separators=(",", ":"))
    old, new = {
        "root": ('"format_version":1', '"format_version":1,"format_version":1'),
        "skin": ('"skin_index":0', '"skin_index":0,"skin_index":0'),
        "cell": ('"x1":0.0', '"x1":0.0,"x1":0.0'),
    }[level]
    path = tmp_path / "skins.json"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(SkinArtifactValidationError, match="strict JSON"):
        parse_skins_json(path, _SHAPE)


@pytest.mark.parametrize(
    "case",
    (
        "cell_only",
        "mask_only",
        "coordinated_without_report",
        "report",
        "diagnostics",
        "nonbinary",
    ),
)
def test_cross_file_validation_rejects_semantic_tamper(
    tmp_path: Path,
    case: str,
) -> None:
    stage = tmp_path / "stage"
    _write_stage(stage)
    if case == "cell_only":
        payload = _payload()
        payload["skins"][1]["cells"][0] = _cell(3, 1, 1)
        _write_json(stage / "skins.json", payload)
    elif case == "mask_only":
        mask = np.fromfile(stage / "skin_mask.dat", dtype=">f4").reshape(_SHAPE)
        mask[1, 1, 2] = 0.0
        mask.tofile(stage / "skin_mask.dat")
    elif case == "coordinated_without_report":
        payload = _payload()
        payload["skins"][1]["cells"].append(_cell(3, 1, 1))
        payload["skins"][1]["cell_count"] += 1
        _write_json(stage / "skins.json", payload)
        mask = np.fromfile(stage / "skin_mask.dat", dtype=">f4").reshape(_SHAPE)
        mask[1, 1, 3] = 1.0
        mask.tofile(stage / "skin_mask.dat")
    elif case == "report":
        report = json.loads((stage / "report.json").read_text(encoding="utf-8"))
        report["topology"]["skin_count"] += 1
        _write_json(stage / "report.json", report)
    elif case == "diagnostics":
        report = json.loads((stage / "report.json").read_text(encoding="utf-8"))
        report["diagnostics"]["accepted_skin_count"] += 1
        _write_json(stage / "report.json", report)
    elif case == "nonbinary":
        mask = np.fromfile(stage / "skin_mask.dat", dtype=">f4").reshape(_SHAPE)
        mask[0, 0, 1] = 0.5
        mask.tofile(stage / "skin_mask.dat")
    else:
        raise AssertionError(case)

    with pytest.raises(SkinArtifactValidationError):
        validate_skin_artifact_semantics(stage, _SHAPE, small_skin_size=2)
