from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pyosv.evaluation.f3d_mode_comparison import (
    F3ArtifactError,
    F3DatasetIdentity,
    F3FileIdentity,
    F3ModeComparisonConfig,
    F3StageArtifact,
    F3StageCorruptionError,
    F3WorkspaceMismatchError,
    build_f3d_mode_comparison_plan,
    canonical_fingerprint,
    canonical_json_bytes,
    implementation_identity,
    prepare_run_workspace,
    run_fingerprint,
)
from pyosv.evaluation.f3d_mode_comparison import artifacts as artifacts_module

_VERSIONS = {
    "pyosv": "test-pyosv",
    "python": "test-python",
    "numpy": "test-numpy",
    "scipy": "test-scipy",
}
_IMPLEMENTATION = {
    "software_versions": _VERSIONS,
    "algorithm_modules": {
        "scanner.py": {
            "sha256": hashlib.sha256(b"scanner").hexdigest(),
            "size": 7,
        }
    },
}


def _plan():
    return build_f3d_mode_comparison_plan(F3ModeComparisonConfig())


def _identity(root: Path, content: bytes = b"dataset") -> F3DatasetIdentity:
    digest = hashlib.sha256(content).hexdigest()
    return F3DatasetIdentity(
        dataset_id="fixture",
        data_root=root,
        files=(
            F3FileIdentity(
                role="input",
                filename="ep.dat",
                resolved_path=root / "ep.dat",
                size=len(content),
                sha256=digest,
                shape=(2, 3, 4),
                storage_dtype=">f4",
            ),
        ),
    )


def _workspace(path: Path):
    return prepare_run_workspace(
        path,
        _plan(),
        _identity(path.parent / "data"),
        resume=False,
        implementation=_IMPLEMENTATION,
        created_at="2026-01-01T00:00:00+00:00",
        source_provenance={
            "status": "not_available",
            "method": "git_cli",
            "commit": None,
            "dirty": None,
        },
    )


def _write_array(path: Path, values: np.ndarray | None = None) -> None:
    if values is None:
        values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    np.save(path / "fv.npy", values)


def _stage(workspace, writer=_write_array):
    return workspace.write_or_reuse_stage(
        "voting",
        parent_fingerprints=("0" * 64,),
        input_fingerprints={"scanner": "1" * 64},
        resolved_settings={"radius": 10, "weight": 0.5},
        artifacts=(F3StageArtifact("fv.npy", (2, 3, 4)),),
        writer=writer,
    )


def test_canonical_json_and_fingerprint_are_stable_and_finite() -> None:
    first = {"β": [1, 2.5], "a": {"y": True, "x": None}}
    second = {"a": {"x": None, "y": True}, "β": [1, 2.5]}

    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert canonical_fingerprint(first) == canonical_fingerprint(second)
    assert canonical_json_bytes(first).decode("utf-8").startswith('{"a":')
    with pytest.raises(ValueError, match="finite"):
        canonical_json_bytes({"bad": float("nan")})


def test_implementation_identity_hashes_source_and_versions(tmp_path: Path) -> None:
    source = tmp_path / "scanner.py"
    source.write_bytes(b"first")
    first = implementation_identity(
        software_versions=_VERSIONS,
        source_files={"scanner.py": source},
    )

    source.write_bytes(b"second")
    second = implementation_identity(
        software_versions=_VERSIONS,
        source_files={"scanner.py": source},
    )

    assert first["software_versions"] == _VERSIONS
    assert (
        first["algorithm_modules"]["scanner.py"]["sha256"]
        != (second["algorithm_modules"]["scanner.py"]["sha256"])
    )


def test_default_implementation_identity_hashes_f3_execution_sources() -> None:
    modules = implementation_identity()["algorithm_modules"]

    assert {
        "evaluation/f3d_mode_comparison/artifacts.py",
        "evaluation/f3d_mode_comparison/builder.py",
        "evaluation/f3d_mode_comparison/data.py",
        "evaluation/f3d_mode_comparison/models.py",
    } <= modules.keys()


def test_run_fingerprint_excludes_paths_but_includes_content_and_implementation(
    tmp_path: Path,
) -> None:
    plan = _plan()
    first = _identity(tmp_path / "first")
    moved = _identity(tmp_path / "moved")
    changed = _identity(tmp_path / "first", b"changed")

    baseline = run_fingerprint(plan, first, implementation=_IMPLEMENTATION)
    assert run_fingerprint(plan, moved, implementation=_IMPLEMENTATION) == baseline
    assert run_fingerprint(plan, changed, implementation=_IMPLEMENTATION) != baseline

    changed_versions = {
        **_IMPLEMENTATION,
        "software_versions": {**_VERSIONS, "numpy": "different"},
    }
    changed_source = {
        **_IMPLEMENTATION,
        "algorithm_modules": {
            "scanner.py": {
                "sha256": hashlib.sha256(b"different").hexdigest(),
                "size": 9,
            }
        },
    }
    assert run_fingerprint(plan, first, implementation=changed_versions) != baseline
    assert run_fingerprint(plan, first, implementation=changed_source) != baseline


def test_workspace_new_resume_and_manifest_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "run"
    first = _workspace(path)

    assert first.resumed is False
    assert set(item.name for item in path.iterdir()) == {
        "run_manifest.json",
        "stages",
        "cells",
        "reports",
    }
    assert set(item.name for item in (path / "stages").iterdir()) == {
        "scanner",
        "voting",
        "thinning",
        "skinning",
    }
    assert not (path / "completion.json").exists()
    with pytest.raises(FileExistsError):
        _workspace(path)

    resumed = prepare_run_workspace(
        path,
        _plan(),
        _identity(tmp_path / "moved-data"),
        resume=True,
        implementation=_IMPLEMENTATION,
    )
    assert resumed.resumed is True
    assert resumed.fingerprint == first.fingerprint
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["provenance"]["data_root"].endswith("/data")

    changed = {**_IMPLEMENTATION, "software_versions": {**_VERSIONS, "scipy": "changed"}}
    with pytest.raises(F3WorkspaceMismatchError, match="implementation_identity"):
        prepare_run_workspace(
            path,
            _plan(),
            _identity(tmp_path / "data"),
            resume=True,
            implementation=changed,
        )


@pytest.mark.parametrize(
    ("resume", "error_type"),
    ((False, FileExistsError), (True, F3ArtifactError)),
)
def test_workspace_rejects_symlink_output_path(
    tmp_path: Path,
    resume: bool,
    error_type: type[Exception],
) -> None:
    target = tmp_path / "target"
    workspace = _workspace(target)
    link = tmp_path / "run-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    original_manifest = (target / "run_manifest.json").read_bytes()

    with pytest.raises(error_type):
        prepare_run_workspace(
            link,
            _plan(),
            _identity(tmp_path / "data"),
            resume=resume,
            implementation=_IMPLEMENTATION,
        )

    assert link.is_symlink()
    assert workspace.path == target
    assert (target / "run_manifest.json").read_bytes() == original_manifest


@pytest.mark.parametrize(
    ("resume", "error_type"),
    ((False, FileExistsError), (True, F3ArtifactError)),
)
def test_workspace_rejects_dangling_symlink_output_path(
    tmp_path: Path,
    resume: bool,
    error_type: type[Exception],
) -> None:
    target = tmp_path / "missing-target"
    link = tmp_path / "run-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(error_type):
        prepare_run_workspace(
            link,
            _plan(),
            _identity(tmp_path / "data"),
            resume=resume,
            implementation=_IMPLEMENTATION,
        )

    assert link.is_symlink()
    assert not target.exists()


def test_stage_is_completed_last_validated_and_reused_without_writer(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")
    result = _stage(workspace)

    assert result.reused is False
    assert result.path == workspace.path / "stages" / "voting" / result.fingerprint
    assert set(item.name for item in result.path.iterdir()) == {
        "fv.npy",
        "stage_manifest.json",
        "complete.json",
    }
    completion = json.loads((result.path / "complete.json").read_text(encoding="utf-8"))
    assert set(completion["files"]) == {"fv.npy", "stage_manifest.json"}
    assert completion["files"]["fv.npy"]["size"] == (result.path / "fv.npy").stat().st_size

    calls = 0

    def should_not_run(path: Path) -> None:
        nonlocal calls
        calls += 1

    reused = _stage(workspace, should_not_run)
    assert reused.reused is True
    assert calls == 0

    shutil.rmtree(result.path)
    assert not result.path.exists()


@pytest.mark.parametrize("corruption", ("missing", "extra", "hash", "symlink", "shape", "dtype"))
def test_corrupt_stage_is_rejected_without_compute(
    tmp_path: Path,
    corruption: str,
) -> None:
    workspace = _workspace(tmp_path / "run")
    result = _stage(workspace)
    artifact = result.path / "fv.npy"
    if corruption == "missing":
        (result.path / "complete.json").unlink()
    elif corruption == "extra":
        (result.path / "extra.bin").write_bytes(b"extra")
    elif corruption == "hash":
        artifact.write_bytes(artifact.read_bytes() + b"changed")
    elif corruption == "symlink":
        artifact.unlink()
        artifact.symlink_to(result.path / "stage_manifest.json")
    elif corruption in {"shape", "dtype"}:
        values = np.zeros(
            (2, 3, 5) if corruption == "shape" else (2, 3, 4),
            dtype=np.float32 if corruption == "shape" else np.float64,
        )
        np.save(artifact, values)
        _rehash_stage_file(result.path, "fv.npy")
    else:
        raise AssertionError(corruption)
    before = sorted(item.name for item in result.path.iterdir())
    calls = 0

    def should_not_run(path: Path) -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(F3StageCorruptionError):
        _stage(workspace, should_not_run)
    assert calls == 0
    assert sorted(item.name for item in result.path.iterdir()) == before


def _rehash_stage_file(stage: Path, filename: str) -> None:
    payload = (stage / filename).read_bytes()
    metadata = {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    manifest_path = stage / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][filename] = metadata
    manifest_payload = canonical_json_bytes(manifest) + b"\n"
    manifest_path.write_bytes(manifest_payload)
    completion_path = stage / "complete.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["files"][filename] = metadata
    completion["files"]["stage_manifest.json"] = {
        "sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "size": len(manifest_payload),
    }
    completion_path.write_bytes(canonical_json_bytes(completion) + b"\n")


@pytest.mark.parametrize("fault", ("write", "hash", "fsync", "rename"))
def test_stage_fault_does_not_publish_or_leave_temporary_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    workspace = _workspace(tmp_path / "run")
    parent = workspace.path / "stages" / "voting"

    if fault == "write":
        monkeypatch.setattr(
            artifacts_module,
            "_write_bytes",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("write fault")),
        )
    elif fault == "hash":
        monkeypatch.setattr(
            artifacts_module,
            "_file_metadata",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("hash fault")),
        )
    elif fault == "fsync":
        monkeypatch.setattr(
            artifacts_module,
            "_fsync_directory",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("fsync fault")),
        )
    elif fault == "rename":
        monkeypatch.setattr(
            artifacts_module,
            "_rename_noreplace",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("rename fault")),
        )

    with pytest.raises(OSError, match="fault"):
        _stage(workspace)
    assert list(parent.iterdir()) == []


def test_resume_cleans_only_owned_nonsymlink_temporary_directories(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")
    parent = workspace.path / "stages" / "scanner"
    owned = parent / ".pyosv-stage-tmp-deadbeef"
    owned.mkdir()
    unrelated = parent / ".other-temp"
    unrelated.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    symlink = parent / ".pyosv-stage-tmp-symlink"
    try:
        symlink.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    prepare_run_workspace(
        workspace.path,
        _plan(),
        _identity(tmp_path / "data"),
        resume=True,
        implementation=_IMPLEMENTATION,
    )

    assert not owned.exists()
    assert unrelated.exists()
    assert symlink.is_symlink()
    assert target.exists()


def test_existing_wrong_fingerprint_manifest_is_not_overwritten(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")
    manifest_path = workspace.path / "run_manifest.json"
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    manifest["run_fingerprint"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    tampered = manifest_path.read_bytes()

    with pytest.raises(F3WorkspaceMismatchError, match="run_fingerprint"):
        prepare_run_workspace(
            workspace.path,
            _plan(),
            _identity(tmp_path / "data"),
            resume=True,
            implementation=_IMPLEMENTATION,
        )
    assert manifest_path.read_bytes() == tampered
    assert manifest_path.read_bytes() != original


def test_stage_writer_rejects_mixed_generation_files(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "run")

    def writer(path: Path) -> None:
        _write_array(path)
        (path / "old-generation.npy").write_bytes(b"old")

    with pytest.raises(F3StageCorruptionError, match="extra"):
        _stage(workspace, writer)
    assert list((workspace.path / "stages" / "voting").iterdir()) == []


def test_workspace_manifest_atomic_replace_fault_cleans_new_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "run"
    real_replace = os.replace

    def fail_replace(source: Path, destination: Path) -> None:
        if Path(destination).name == "run_manifest.json":
            raise OSError("replace fault")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace fault"):
        _workspace(path)
    assert not path.exists()


def test_config_change_changes_run_fingerprint(tmp_path: Path) -> None:
    baseline = _plan()
    changed = replace(
        baseline,
        boundary_diagnostic_margin=baseline.boundary_diagnostic_margin + 1,
    )

    assert run_fingerprint(
        baseline,
        _identity(tmp_path / "data"),
        implementation=_IMPLEMENTATION,
    ) != run_fingerprint(
        changed,
        _identity(tmp_path / "data"),
        implementation=_IMPLEMENTATION,
    )
