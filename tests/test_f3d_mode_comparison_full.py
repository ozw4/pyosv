from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pyosv.cli import f3d_mode_comparison
from pyosv.evaluation.f3d_mode_comparison import (
    OFFICIAL_F3_DATASET_SPEC,
    load_f3d_mode_comparison_result,
    validate_completed_f3d_bundle,
)


def _required_environment() -> tuple[Path, Path]:
    if os.environ.get("PYOSV_RUN_F3D_MODE_COMPARISON") != "1":
        pytest.skip("set PYOSV_RUN_F3D_MODE_COMPARISON=1 for the official F3 run")
    data_value = os.environ.get("PYOSV_F3D_DATA_ROOT")
    output_value = os.environ.get("PYOSV_F3D_MODE_COMPARISON_OUTPUT_DIR")
    if not data_value or not output_value:
        pytest.skip("official F3 data-root and output-dir environment variables are required")
    data_root = Path(data_value)
    output_root = Path(output_value)
    missing = [
        filename
        for filename in OFFICIAL_F3_DATASET_SPEC.required_files
        if not (data_root / filename).is_file()
    ]
    if missing:
        pytest.skip(f"official F3 files are unavailable: {', '.join(missing)}")
    if output_root.exists() and (not output_root.is_dir() or output_root.is_symlink()):
        pytest.skip(f"output path is unavailable: {output_root}")
    if not output_root.exists() and not output_root.parent.is_dir():
        pytest.skip(f"output parent is unavailable: {output_root.parent}")
    return data_root, output_root


def test_official_f3_full_volume_mode_comparison() -> None:
    data_root, output_root = _required_environment()
    deep = os.environ.get("PYOSV_F3D_MODE_COMPARISON_DEEP_VALIDATE") == "1"
    arguments = [
        "--data-root",
        str(data_root),
        "--output-dir",
        str(output_root),
    ]
    if output_root.exists():
        arguments.append("--resume")
    if deep:
        arguments.append("--deep-validate")

    assert f3d_mode_comparison.main(arguments) == 0
    assert validate_completed_f3d_bundle(output_root, deep=deep)
    result = load_f3d_mode_comparison_result(output_root, deep=deep)
    assert result.dataset_id == OFFICIAL_F3_DATASET_SPEC.dataset_id
    assert result.volume_shape == OFFICIAL_F3_DATASET_SPEC.shape
    assert [cell.label for cell in result.cells] == [
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    ]
    assert len({cell.stages.scanner for cell in result.cells}) == 2
    assert len({cell.stages.voting for cell in result.cells}) == 2
    assert len({cell.stages.thinning for cell in result.cells}) == 4
    assert len({cell.stages.skinning for cell in result.cells}) == 4

    manifest = json.loads((output_root / "run_manifest.json").read_text())
    identities = manifest["dataset_identity"]["files"]
    assert [item["role"] for item in identities] == list(OFFICIAL_F3_DATASET_SPEC.roles)
    assert all(len(item["sha256"]) == 64 for item in identities)
    completion = json.loads((output_root / "completion.json").read_text())
    assert len(completion["stage_completions"]) == 12
    assert all(
        len(metadata["sha256"]) == 64 for metadata in completion["stage_completions"].values()
    )
