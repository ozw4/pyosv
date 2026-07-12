from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pyosv.evaluation.reporting import artifacts


_SHA256_FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "reporting" / "artifact_sha256.json"
)


def _volumes(*, scanner: bool = False) -> dict[str, np.ndarray]:
    values = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    result = {name: values.copy() for name in artifacts.VOLUME_NAMES}
    if scanner:
        result.update({name: values.copy() for name, _ in artifacts.SCANNER_VOLUME_NAMES})
    return result


def _stage_diagnostic_volumes() -> dict[str, np.ndarray]:
    mask = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.float32).reshape(2, 2, 2)
    return {
        source_name: mask.copy()
        for source_name, _ in artifacts.SCANNER_BOUNDARY_STAGE_DIAGNOSTIC_VOLUME_NAMES
    }


def test_write_case_volumes_preserves_single_and_multiple_variant_layout(tmp_path: Path) -> None:
    single = {"geometry/plane": {"current_default": _volumes()}}
    artifacts.write_case_volumes(single, tmp_path / "single")
    assert {
        path.relative_to(tmp_path / "single").as_posix()
        for path in (tmp_path / "single").rglob("*.dat")
    } == {f"geometry/plane/{name}.dat" for name in artifacts.VOLUME_NAMES}

    multiple = {
        "plane": {
            "current_default": _volumes(),
            "boundary_aware_voter_v1": _volumes(),
        }
    }
    artifacts.write_case_volumes(multiple, tmp_path / "multiple")
    assert {
        path.relative_to(tmp_path / "multiple").as_posix()
        for path in (tmp_path / "multiple").rglob("*.dat")
    } == {
        f"plane/{variant}/{name}.dat"
        for variant in multiple["plane"]
        for name in artifacts.VOLUME_NAMES
    }


def test_write_case_volumes_preserves_both_pipeline_layout(tmp_path: Path) -> None:
    outputs = {
        "plane": {
            "current_default": {
                artifacts.PIPELINE_OUTPUTS_KEY: {
                    "oracle": _volumes(),
                    "scanner": _volumes(scanner=True),
                }
            }
        }
    }
    artifacts.write_case_volumes(outputs, tmp_path)
    expected = {
        *(f"plane/oracle/{name}.dat" for name in artifacts.VOLUME_NAMES),
        *(f"plane/scanner/{name}.dat" for name in artifacts.VOLUME_NAMES),
        *(f"plane/scanner/{output_name}.dat" for _, output_name in artifacts.SCANNER_VOLUME_NAMES),
    }
    assert {path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*.dat")} == expected


def test_write_case_volumes_writes_scanner_boundary_stage_diagnostics(tmp_path: Path) -> None:
    from pyosv.io import read_dat

    volumes = _volumes(scanner=True)
    volumes.update(_stage_diagnostic_volumes())
    paths = artifacts.write_case_volumes({"plane": {"current_default": volumes}}, tmp_path)
    diagnostic_dir = tmp_path / "plane" / "scanner_boundary_stage_diagnostics"
    expected_names = {
        f"{output_name}.dat"
        for _, output_name in artifacts.SCANNER_BOUNDARY_STAGE_DIAGNOSTIC_VOLUME_NAMES
    }
    assert {path.name for path in paths if path.parent == diagnostic_dir} == expected_names
    actual = read_dat(diagnostic_dir / "seed_selected.dat", (2, 2, 2))
    np.testing.assert_array_equal(
        actual,
        volumes["scanner_boundary_stage_seed_selected"],
    )
    assert actual.dtype == np.float32
    assert set(np.unique(actual)) <= {0.0, 1.0}


def test_write_case_volumes_skips_absent_stage_diagnostics(tmp_path: Path) -> None:
    artifacts.write_case_volumes({"plane": {"current_default": _volumes(scanner=True)}}, tmp_path)
    assert not (tmp_path / "plane" / "scanner_boundary_stage_diagnostics").exists()


def test_write_case_volumes_rejects_incomplete_stage_diagnostics(tmp_path: Path) -> None:
    volumes = _volumes(scanner=True)
    volumes["scanner_boundary_stage_seed_candidate"] = np.zeros((2, 2, 2), dtype=np.float32)
    with pytest.raises(KeyError, match="incomplete scanner boundary stage diagnostic volumes"):
        artifacts.write_case_volumes({"plane": {"current_default": volumes}}, tmp_path)


def test_write_case_volumes_writes_stage_diagnostics_only_for_scanner_pipeline(
    tmp_path: Path,
) -> None:
    scanner = _volumes(scanner=True)
    scanner.update(_stage_diagnostic_volumes())
    outputs = {
        "plane": {
            "current_default": {
                artifacts.PIPELINE_OUTPUTS_KEY: {"oracle": _volumes(), "scanner": scanner}
            }
        }
    }
    artifacts.write_case_volumes(outputs, tmp_path)
    assert not (tmp_path / "plane" / "oracle" / "scanner_boundary_stage_diagnostics").exists()
    assert (tmp_path / "plane" / "scanner" / "scanner_boundary_stage_diagnostics").is_dir()


def test_write_case_skins_json_preserves_payload_and_does_not_mutate_it(tmp_path: Path) -> None:
    payload = {
        "format_version": 1,
        "skinning_enabled": True,
        "skin_count": 1,
        "skins": [{"skin_index": 0, "cell_count": 0, "cells": []}],
    }
    before = json.loads(json.dumps(payload))
    paths = artifacts.write_case_skins_json(
        {"plane": {"current_default": payload}},
        tmp_path,
    )
    assert paths == [tmp_path / "plane" / "skins.json"]
    assert json.loads(paths[0].read_text(encoding="utf-8")) == before
    assert payload == before


def test_dat_and_skins_json_match_sha256_fixture(tmp_path: Path) -> None:
    artifacts.write_case_volumes(
        {"plane": {"current_default": _volumes()}},
        tmp_path,
    )
    artifacts.write_case_skins_json(
        {
            "plane": {
                "current_default": {
                    "format_version": 1,
                    "skinning_enabled": True,
                    "skin_count": 1,
                    "skins": [{"skin_index": 0, "cell_count": 0, "cells": []}],
                }
            }
        },
        tmp_path,
    )

    expected = json.loads(_SHA256_FIXTURE.read_text(encoding="utf-8"))
    expected = {path: digest for path, digest in expected.items() if path != "visual_report.md"}
    actual = {
        path.relative_to(tmp_path).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(tmp_path.rglob("*"))
        if path.is_file()
    }
    assert actual == expected


@pytest.mark.parametrize("case_id", ("../escape", "/absolute", "case/../../escape"))
def test_artifact_writers_reject_case_path_traversal(tmp_path: Path, case_id: str) -> None:
    with pytest.raises(ValueError, match="relative path inside output_dir"):
        artifacts.write_case_volumes({case_id: {"current_default": _volumes()}}, tmp_path)


def test_write_case_figures_writes_expected_png_set(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    artifacts.write_case_figures({"plane": {"current_default": _volumes()}}, tmp_path)
    expected = {
        *(
            f"{name}_{axis}_center.png"
            for axis in ("i3", "i2", "i1")
            for name in artifacts.FIGURE_VOLUME_NAMES
        ),
        *(f"truth_vs_fvt_overlay_{axis}_center.png" for axis in ("i3", "i2", "i1")),
        *(f"truth_vs_skin_overlay_{axis}_center.png" for axis in ("i3", "i2", "i1")),
        "skin_mask_py_i3_center.png",
    }
    figures = tmp_path / "plane" / "figures"
    assert {path.name for path in figures.glob("*.png")} == expected
    assert all(path.stat().st_size > 0 for path in figures.glob("*.png"))
