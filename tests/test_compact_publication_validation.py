from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import pyosv.compact_publication_validation as validation_module
from pyosv.compact_publication_validation import validate_compact_publication

_CONTROLS = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NUMBA_DISABLE_JIT": "0",
    "NUMBA_NUM_THREADS": "1",
    "PYOSV_ACCEL": "auto",
}
_DATASET_FILES = [
    {"role": role, "filename": filename, "size": 4, "sha256": digit * 64}
    for role, filename, digit in (
        ("input", "ep.dat", "1"),
        ("reference_fault_likelihood", "fl.dat", "2"),
        ("reference_fault_votes", "fv.dat", "3"),
        ("reference_thinned_fault_votes", "fvt.dat", "4"),
        ("seismic_amplitude", "xs.dat", "5"),
    )
]


def _artifact(path: str, content: bytes, role: str) -> dict[str, object]:
    return {
        "path": path,
        "tier": "primary",
        "role": role,
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _prepare_compact_publication(root: Path) -> Path:
    root.mkdir()
    experiment_content = b'{"schema":"fixture"}\n'
    lock_content = b"lock-version = 1\n"
    (root / "experiment.json").write_bytes(experiment_content)
    (root / "uv.lock").write_bytes(lock_content)
    experiment = _artifact("experiment.json", experiment_content, "resolved_experiment")
    lock = _artifact("uv.lock", lock_content, "environment_lock")
    manifest = validation_module.build_manifest(
        created_at_utc="2026-08-16T00:00:00Z",
        code={"repository": "ozw4/pyosv", "git_commit": "a" * 40, "dirty": False},
        environment={
            "python": "3.10.14",
            "lock_file": "uv.lock",
            "lock_sha256": lock["sha256"],
            "controls": _CONTROLS,
        },
        source={"f3_completion_sha256": "b" * 64},
        dataset={
            "dataset_id": "standalone-validator-fixture",
            "shape": [1, 1, 1],
            "storage_dtype": ">f4",
            "files": _DATASET_FILES,
        },
        experiment={
            "config_file": "experiment.json",
            "config_sha256": experiment["sha256"],
        },
        semantics={
            "evaluation": "f3_public_reference_agreement",
            "public_reference_is_geological_truth": False,
            "evaluation_units": 1,
            "displayed_condition": "Q-QUAL",
            "stage_order": ["ft", "fv", "fvt"],
        },
        artifacts=[experiment, lock],
    )
    validation_module.write_manifest(root, manifest)
    return root


def test_valid_compact_publication_returns_normalized_manifest(tmp_path: Path) -> None:
    root = _prepare_compact_publication(tmp_path / "publication")

    manifest = validate_compact_publication(root)

    assert manifest["schema"] == "pyosv.f3_compact_publication_manifest.v1"
    dataset = manifest["dataset"]
    assert isinstance(dataset, dict)
    assert dataset["dataset_id"] == "standalone-validator-fixture"
    assert validation_module.__all__ == ["validate_compact_publication"]


def test_rejects_artifact_tampering(tmp_path: Path) -> None:
    root = _prepare_compact_publication(tmp_path / "publication")
    (root / "experiment.json").write_bytes(b"tampered\n")

    with pytest.raises(ValueError, match="artifact (size|SHA-256)"):
        validate_compact_publication(root)


def test_rejects_missing_artifact(tmp_path: Path) -> None:
    root = _prepare_compact_publication(tmp_path / "publication")
    (root / "uv.lock").unlink()

    with pytest.raises(ValueError, match="artifact must be a regular"):
        validate_compact_publication(root)


def test_rejects_extra_file(tmp_path: Path) -> None:
    root = _prepare_compact_publication(tmp_path / "publication")
    (root / "extra.txt").write_text("extra\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid regular file set"):
        validate_compact_publication(root)


def test_rejects_symlink(tmp_path: Path) -> None:
    root = _prepare_compact_publication(tmp_path / "publication")
    (root / "link").symlink_to("uv.lock")

    with pytest.raises(ValueError, match="must not contain symlinks"):
        validate_compact_publication(root)


def test_rejects_publication_id_mismatch(tmp_path: Path) -> None:
    root = _prepare_compact_publication(tmp_path / "publication")
    manifest_path = root / "publication_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["publication_id"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="publication_id does not match"):
        validate_compact_publication(root)


def test_validates_archive_after_extraction(tmp_path: Path) -> None:
    root = _prepare_compact_publication(tmp_path / "publication")
    archive = tmp_path / "publication.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(root, arcname="publication")
    extracted_root = tmp_path / "extracted"
    extracted_root.mkdir()
    with tarfile.open(archive, "r:gz") as stream:
        stream.extractall(extracted_root, filter="data")

    manifest = validate_compact_publication(extracted_root / "publication")

    assert manifest["publication_id"]


def test_validator_runs_when_numerical_and_internal_modules_are_blocked(tmp_path: Path) -> None:
    root = _prepare_compact_publication(tmp_path / "publication")
    package_root = Path(__file__).resolve().parents[1] / "src"
    script = r"""
import importlib.abc
import sys

BLOCKED = ("numpy", "scipy", "matplotlib", "numba")
BLOCKED_PARTS = (
    "pyosv.viz",
    "f3d_mode_comparison",
    "mode_comparison_publication",
    "f3_compact_publication.source",
    "f3_compact_publication.figures",
)

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + ".") for name in BLOCKED):
            raise RuntimeError(f"blocked optional import: {fullname}")
        if any(part in fullname for part in BLOCKED_PARTS):
            raise RuntimeError(f"blocked internal import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
from pyosv.compact_publication_validation import validate_compact_publication
validate_compact_publication(sys.argv[1])
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(package_root)

    result = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
