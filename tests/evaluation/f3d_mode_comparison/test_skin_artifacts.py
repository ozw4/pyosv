from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyosv.evaluation.f3d_mode_comparison.skin_artifacts import (
    SkinArtifactValidationError,
    parse_skins_json,
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
