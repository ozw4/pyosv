from __future__ import annotations

import builtins
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from pyosv.evaluation import f3d_mode_comparison, synthetic_mode_comparison
from pyosv.evaluation.mode_comparison_publication import (
    generate_publication_bundle,
    validate_publication_bundle,
)
from pyosv.evaluation.mode_comparison_publication import artifacts as publication_artifacts
from pyosv.evaluation.mode_comparison_publication import validation as publication_validation
from pyosv.evaluation.synthetic_mode_comparison import (
    SyntheticModeComparisonConfig,
    run_mode_comparison,
    write_artifact_bundle,
)
from pyosv.evaluation.synthetic_quality import SyntheticSkinningConfig


def test_generation_preserves_sources_and_has_fixed_artifact_set(
    publication_bundle: tuple[Path, dict[str, Any]],
) -> None:
    output, sources = publication_bundle
    assert validate_publication_bundle(output)
    assert {item.name for item in output.iterdir()} == {
        "manifest.json",
        "publication_metrics.csv",
        "publication_contrasts.csv",
        "publication_summary.csv",
        "f3_regional_summary.csv",
        "f3_orientation_summary.csv",
        "runtime_summary.csv",
        "figure_manifest.json",
        "report.md",
        "figure_data",
        "figures",
        "completion.json",
    }
    assert sources["synthetic_snapshot"] == _snapshot(sources["synthetic"])
    assert sources["f3_snapshot"] == _snapshot(sources["f3"])
    assert sources["data_snapshot"] == _snapshot(sources["data_root"])
    completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
    assert "completion.json" not in completion["required_files"]
    assert all(set(item) == {"path", "size", "sha256"} for item in completion["files"])


def test_publication_source_runner_functions_are_never_called(
    source_bundles: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("publication generation called a source runner")

    monkeypatch.setattr(synthetic_mode_comparison, "run_mode_comparison", fail)
    monkeypatch.setattr(synthetic_mode_comparison, "run_synthetic_trial", fail)
    monkeypatch.setattr(f3d_mode_comparison, "run_scanner_stages", fail)
    monkeypatch.setattr(f3d_mode_comparison, "run_f3d_mode_comparison", fail, raising=False)
    workflow_module = __import__("pyosv.evaluation.workflow3d", fromlist=["execute_workflow3d"])
    monkeypatch.setattr(workflow_module, "execute_workflow3d", fail)

    output = tmp_path / "publication"
    generate_publication_bundle(
        source_bundles["synthetic"],
        source_bundles["f3"],
        source_bundles["data_root"],
        output,
    )
    assert validate_publication_bundle(output)


def test_existing_output_and_failed_generation_leave_no_completed_bundle(
    source_bundles: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "publication"
    output.mkdir()
    with pytest.raises(FileExistsError):
        generate_publication_bundle(
            source_bundles["synthetic"],
            source_bundles["f3"],
            source_bundles["data_root"],
            output,
        )
    output.rmdir()

    original = publication_artifacts.validate_publication_bundle

    def fail_validation(path: Path) -> bool:
        del path
        raise ValueError("forced publication validation failure")

    monkeypatch.setattr(publication_artifacts, "validate_publication_bundle", fail_validation)
    with pytest.raises(ValueError, match="forced publication"):
        generate_publication_bundle(
            source_bundles["synthetic"],
            source_bundles["f3"],
            source_bundles["data_root"],
            output,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".publication.tmp-*"))
    monkeypatch.setattr(publication_artifacts, "validate_publication_bundle", original)


def test_completion_is_present_before_atomic_rename(
    source_bundles: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, bool] = {}
    original = publication_artifacts._rename_new

    def check_then_rename(source: Path, destination: Path) -> None:
        observed["completion"] = (source / "completion.json").is_file()
        observed["destination_absent"] = not destination.exists()
        original(source, destination)

    monkeypatch.setattr(publication_artifacts, "_rename_new", check_then_rename)
    generate_publication_bundle(
        source_bundles["synthetic"],
        source_bundles["f3"],
        source_bundles["data_root"],
        tmp_path / "publication",
    )
    assert observed == {"completion": True, "destination_absent": True}


def test_validate_only_needs_no_matplotlib_or_sources(
    publication_bundle: tuple[Path, dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _sources = publication_bundle
    imported: list[str] = []
    original_import = builtins.__import__

    def track_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("matplotlib"):
            imported.append(name)
            raise AssertionError("publication validator imported matplotlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", track_import)
    assert validate_publication_bundle(output)
    assert imported == []


def test_data_identity_mismatch_fails_before_output(
    source_bundles: dict[str, Any],
    tmp_path: Path,
) -> None:
    bad_data_root = tmp_path / "bad-f3-data"
    shutil.copytree(source_bundles["data_root"], bad_data_root)
    target = bad_data_root / "fl.dat"
    payload = bytearray(target.read_bytes())
    payload[0] ^= 1
    target.write_bytes(payload)
    output = tmp_path / "publication"
    with pytest.raises(ValueError, match="identity|SHA-256|checksum"):
        generate_publication_bundle(
            source_bundles["synthetic"],
            source_bundles["f3"],
            bad_data_root,
            output,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("filename", "mutation"),
    (
        ("publication_metrics.csv", lambda path: path.write_bytes(path.read_bytes() + b"\n")),
        (
            "figure_manifest.json",
            lambda path: path.write_bytes(
                path.read_bytes().replace(b'"main"', b'"supplementary"', 1)
            ),
        ),
    ),
)
def test_completion_detects_publication_tampering(
    publication_bundle: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    filename: str,
    mutation: Any,
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / "tampered"
    shutil.copytree(source, output)
    mutation(output / filename)
    with pytest.raises(ValueError, match="hash|size|completion"):
        validate_publication_bundle(output)


def test_png_set_tampering_is_rejected(
    publication_bundle: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    source, _sources = publication_bundle
    output = tmp_path / "tampered"
    shutil.copytree(source, output)
    png = next((output / "figures").glob("*.png"))
    png.unlink()
    with pytest.raises(ValueError):
        validate_publication_bundle(output)


def test_disabled_synthetic_skinning_omits_skin_figure(
    source_bundles: dict[str, Any],
    tmp_path: Path,
) -> None:
    config = SyntheticModeComparisonConfig(
        case_ids=("single_vertical_plane",),
        trial_seeds=(20260707,),
        shape=(9, 9, 9),
        skinning_config=SyntheticSkinningConfig(enabled=False),
    )
    source = write_artifact_bundle(
        run_mode_comparison(config),
        tmp_path / "synthetic-disabled",
        config=config,
    )
    output = tmp_path / "publication"
    generate_publication_bundle(
        source,
        source_bundles["f3"],
        source_bundles["data_root"],
        output,
    )
    manifest = json.loads((output / "figure_manifest.json").read_text(encoding="utf-8"))
    omitted = next(
        item
        for item in manifest["figures"]
        if item["figure_id"] == "synthetic_skin_buffered_f1_by_case"
    )
    assert omitted["omitted"] is True
    assert not (output / "figures" / "synthetic_skin_buffered_f1_by_case.png").exists()
    assert validate_publication_bundle(output)


def _snapshot(root: Path) -> dict[str, tuple[bytes, int, int, str]]:
    from tests.evaluation.mode_comparison_publication.conftest import snapshot_files

    return snapshot_files(root)


def test_publication_validator_has_no_source_runner_imports() -> None:
    source = Path(publication_validation.__file__).read_text(encoding="utf-8")
    assert "run_mode_comparison" not in source
    assert "run_f3d_mode_comparison" not in source
