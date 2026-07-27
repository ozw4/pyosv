from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyosv.evaluation.f3d_mode_comparison.skin_artifacts import (
    SkinCellRecord,
    SkinArtifactValidationError,
    canonical_skins_payload,
    parse_skins_json,
    resolve_skin_parent_volume_contract,
    validate_skin_artifact_semantics,
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
    cell = SkinCellRecord(0.5, 0.0, 0.0, 1, 0, 0, 0.1, 10.0, 65.0)

    payload = canonical_skins_payload(((cell,),))

    assert payload["skins"][0]["cells"][0] == {
        **_cell(1, 0, 0, fl=0.1),
        "x1": 0.5,
        "fp": 10.0,
        "ft": 65.0,
    }


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
        {
            "growth_source": growth_source,
            "boundary_skinner_fallback": fallback_used,
        },
        {"post_thinning_policy": post_thinning_policy},
        fallback_used=fallback_used,
    )

    assert contract.likelihood == expected


def test_parent_volume_contract_rejects_unknown_post_thinning_policy() -> None:
    with pytest.raises(SkinArtifactValidationError, match="post-thinning"):
        resolve_skin_parent_volume_contract(
            {"scanner": "scan", "voting": "vote", "thinning": "thin"},
            {
                "growth_source": "thinned",
                "boundary_skinner_fallback": False,
            },
            {"post_thinning_policy": "unknown"},
            fallback_used=False,
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
        ("fp", 60.0, "strike"),
        ("ft", 81.0, "dip"),
    ),
)
def test_cell_attributes_respect_resolved_ranges(
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
        parse_skins_json(
            path,
            _SHAPE,
            strike_range=(20.0, 50.0),
            dip_range=(65.0, 80.0),
        )


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
