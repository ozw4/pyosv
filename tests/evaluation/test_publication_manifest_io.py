from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from pyosv.evaluation import publication_manifest_io
from pyosv.evaluation.publication_manifest import (
    build_publication_manifest,
    canonical_json_bytes,
)
from pyosv.evaluation.publication_manifest_io import (
    artifact_file_record,
    validate_publication_directory,
    write_publication_manifest,
)

SHA_A = "a" * 64
SHA_B = "b" * 64


def _write_artifacts(root: Path) -> list[dict[str, object]]:
    files = {
        "experiment.json": b'{"experiment":"minimal"}\n',
        "figures/overview.png": b"derived image bytes",
        "synthetic/metrics.csv": b"metric,value\nscore,1.0\n",
        "uv.lock": b"lock file bytes\n",
    }
    details = {
        "experiment.json": ("primary", "experiment_config"),
        "figures/overview.png": ("derived", "figure"),
        "synthetic/metrics.csv": ("primary", "metric_table"),
        "uv.lock": ("primary", "environment_lock"),
    }
    records: list[dict[str, object]] = []
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        tier, role = details[relative_path]
        records.append(artifact_file_record(root, relative_path, tier=tier, role=role))
    return records


def _build_manifest(
    records: list[dict[str, object]],
    *,
    lock_sha256: str | None = None,
    config_sha256: str | None = None,
) -> dict[str, object]:
    by_path = {record["path"]: record for record in records}
    return build_publication_manifest(
        created_at_utc="2026-08-12T00:00:00Z",
        code={"repository": "ozw4/pyosv", "git_commit": "1" * 40, "dirty": False},
        environment={
            "python": "3.10.14",
            "lock_file": "uv.lock",
            "lock_sha256": lock_sha256 or by_path["uv.lock"]["sha256"],
            "controls": {
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "NUMBA_DISABLE_JIT": "0",
                "NUMBA_NUM_THREADS": "1",
                "PYOSV_ACCEL": "auto",
            },
        },
        datasets={
            "f3": {
                "dataset_id": "f3d-official-v1",
                "shape": [420, 400, 100],
                "dtype": ">f4",
                "files": [
                    {
                        "role": "input",
                        "filename": "ep.dat",
                        "size": 67_200_000,
                        "sha256": SHA_A,
                    }
                ],
            }
        },
        experiment={
            "config_file": "experiment.json",
            "config_sha256": config_sha256 or by_path["experiment.json"]["sha256"],
            "source_runs": {
                "synthetic": {"completion_sha256": SHA_A},
                "f3": {"completion_sha256": SHA_B},
            },
        },
        semantics={
            "synthetic": "known_truth",
            "f3": "public_reference_agreement",
            "f3_public_reference_is_geological_truth": False,
            "f3_evaluation_units": 1,
        },
        artifacts=records,
    )


def _prepare_bundle(root: Path) -> dict[str, object]:
    root.mkdir()
    return _build_manifest(_write_artifacts(root))


def _write_raw_manifest(root: Path, manifest: dict[str, object]) -> None:
    (root / "publication_manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")


def test_artifact_file_record_hashes_primary_derived_and_nested_files(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    root.mkdir()
    records = _write_artifacts(root)
    by_path = {record["path"]: record for record in records}

    assert by_path["synthetic/metrics.csv"] == {
        "path": "synthetic/metrics.csv",
        "tier": "primary",
        "role": "metric_table",
        "size": len(b"metric,value\nscore,1.0\n"),
        "sha256": hashlib.sha256(b"metric,value\nscore,1.0\n").hexdigest(),
    }
    assert by_path["figures/overview.png"]["tier"] == "derived"


@pytest.mark.parametrize("pretty", [False, True])
def test_atomic_write_and_directory_validation_succeed(tmp_path: Path, pretty: bool) -> None:
    root = tmp_path / f"publication-{pretty}"
    manifest = _prepare_bundle(root)

    destination = write_publication_manifest(root, manifest, pretty=pretty)

    assert destination == root / "publication_manifest.json"
    assert destination.read_bytes().endswith(b"\n")
    assert validate_publication_directory(root) == manifest
    if pretty:
        assert b'\n  "artifacts": [' in destination.read_bytes()
    else:
        assert destination.read_bytes().count(b"\n") == 1


@pytest.mark.parametrize(
    "relative_path",
    ["/metrics.csv", "../metrics.csv", "a\\b.csv", "publication_manifest.json"],
)
def test_artifact_file_record_rejects_unsafe_paths(tmp_path: Path, relative_path: str) -> None:
    with pytest.raises(ValueError):
        artifact_file_record(tmp_path, relative_path, tier="primary", role="metric_table")


def test_directory_validation_rejects_missing_artifact(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    manifest = _prepare_bundle(root)
    write_publication_manifest(root, manifest)
    (root / "synthetic/metrics.csv").unlink()

    with pytest.raises(ValueError, match="regular non-symlink file"):
        validate_publication_directory(root)


def test_directory_validation_rejects_modified_artifact(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    manifest = _prepare_bundle(root)
    write_publication_manifest(root, manifest)
    path = root / "synthetic/metrics.csv"
    path.write_bytes(b"X" * path.stat().st_size)

    with pytest.raises(ValueError, match="SHA-256"):
        validate_publication_directory(root)


def test_directory_validation_rejects_artifact_size_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    root.mkdir()
    records = _write_artifacts(root)
    changed = deepcopy(records)
    metrics = next(record for record in changed if record["path"] == "synthetic/metrics.csv")
    metrics["size"] = int(metrics["size"]) + 1
    manifest = _build_manifest(changed)
    _write_raw_manifest(root, manifest)

    with pytest.raises(ValueError, match="size"):
        validate_publication_directory(root)


def test_directory_validation_rejects_unlisted_extra_file(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    manifest = _prepare_bundle(root)
    write_publication_manifest(root, manifest)
    (root / "unexpected.log").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid regular file set"):
        validate_publication_directory(root)


def test_directory_validation_rejects_symlink_artifact(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    manifest = _prepare_bundle(root)
    write_publication_manifest(root, manifest)
    external = tmp_path / "external.csv"
    external.write_bytes((root / "synthetic/metrics.csv").read_bytes())
    artifact = root / "synthetic/metrics.csv"
    artifact.unlink()
    try:
        artifact.symlink_to(external)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="regular non-symlink file"):
        validate_publication_directory(root)


@pytest.mark.parametrize(
    ("section", "kwargs", "message"),
    [
        ("lock", {"lock_sha256": SHA_A}, "environment lock file SHA-256"),
        ("config", {"config_sha256": SHA_B}, "experiment config file SHA-256"),
    ],
)
def test_directory_validation_links_lock_and_config_hashes(
    tmp_path: Path,
    section: str,
    kwargs: dict[str, str],
    message: str,
) -> None:
    root = tmp_path / section
    root.mkdir()
    records = _write_artifacts(root)
    manifest = _build_manifest(records, **kwargs)
    _write_raw_manifest(root, manifest)

    with pytest.raises(ValueError, match=message):
        validate_publication_directory(root)


def test_directory_validation_rejects_rewritten_manifest_id(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    manifest = _prepare_bundle(root)
    write_publication_manifest(root, manifest)
    changed = deepcopy(manifest)
    changed["publication_id"] = SHA_A
    _write_raw_manifest(root, changed)

    with pytest.raises(ValueError, match="publication_id"):
        validate_publication_directory(root)


@pytest.mark.parametrize("content", [b"{", b'{"value":NaN}\n', b'{"value":Infinity}\n'])
def test_directory_validation_rejects_malformed_or_nonfinite_json(
    tmp_path: Path, content: bytes
) -> None:
    (tmp_path / "publication_manifest.json").write_bytes(content)

    with pytest.raises(ValueError):
        validate_publication_directory(tmp_path)


def test_writer_rejects_existing_manifest(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    manifest = _prepare_bundle(root)
    write_publication_manifest(root, manifest)

    with pytest.raises(FileExistsError):
        write_publication_manifest(root, manifest)


def test_write_failure_removes_final_and_temporary_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "publication"
    manifest = _prepare_bundle(root)

    def fail_write(path: Path, value: bytes) -> None:
        path.write_bytes(value[:10])
        raise OSError("injected write failure")

    monkeypatch.setattr(publication_manifest_io, "_write_manifest_bytes", fail_write)

    with pytest.raises(OSError, match="injected write failure"):
        write_publication_manifest(root, manifest)

    assert not (root / "publication_manifest.json").exists()
    assert list(root.glob(".publication_manifest.json.*.tmp")) == []


def test_artifact_validation_failure_does_not_create_manifest(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    manifest = _prepare_bundle(root)
    artifact = root / "synthetic/metrics.csv"
    artifact.write_bytes(b"X" * artifact.stat().st_size)

    with pytest.raises(ValueError, match="SHA-256"):
        write_publication_manifest(root, manifest)

    assert not (root / "publication_manifest.json").exists()
    assert list(root.glob(".publication_manifest.json.*.tmp")) == []
