from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from pyosv.evaluation.f3_compact_publication.manifest import (
    F3_COMPACT_PUBLICATION_MANIFEST_SCHEMA,
    PUBLICATION_MANIFEST_FILENAME,
    build_manifest,
    compute_publication_id,
    validate_manifest,
    validate_publication_directory,
    write_manifest,
)
from pyosv.evaluation.publication_manifest_io import artifact_file_record

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64

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
    {"role": "input", "filename": "ep.dat", "size": 96, "sha256": SHA_A},
    {
        "role": "reference_fault_likelihood",
        "filename": "fl.dat",
        "size": 96,
        "sha256": SHA_B,
    },
    {
        "role": "reference_fault_votes",
        "filename": "fv.dat",
        "size": 96,
        "sha256": SHA_C,
    },
    {
        "role": "reference_thinned_fault_votes",
        "filename": "fvt.dat",
        "size": 96,
        "sha256": SHA_D,
    },
    {"role": "seismic_amplitude", "filename": "xs.dat", "size": 96, "sha256": SHA_E},
]
_ARTIFACT_DETAILS = {
    "experiment.json": ("primary", "resolved_experiment", b'{"schema":"fixture"}\n'),
    "f3_q_qual_vs_public_ref_summary.csv": (
        "primary",
        "summary_table",
        b"stage,value\nft,1\n",
    ),
    "figure_data/f3_ft.csv": ("primary", "figure_data", b"panel,value\nPUBLIC-REF,1\n"),
    "figure_data/f3_fv.csv": ("primary", "figure_data", b"panel,value\nQ-QUAL,1\n"),
    "figure_data/f3_fvt.csv": ("primary", "figure_data", b"panel,value\ndifference,0\n"),
    "figures/f3_ft.png": ("derived", "figure", b"png-ft"),
    "figures/f3_fv.png": ("derived", "figure", b"png-fv"),
    "figures/f3_fvt.png": ("derived", "figure", b"png-fvt"),
    "report.md": ("derived", "report", b"# Compact report\n"),
    "uv.lock": ("primary", "environment_lock", b"lock-version = 1\n"),
}


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _placeholder_artifacts() -> list[dict[str, object]]:
    return [
        {
            "path": path,
            "tier": tier,
            "role": role,
            "size": len(content),
            "sha256": _digest(content),
        }
        for path, (tier, role, content) in _ARTIFACT_DETAILS.items()
    ]


def _parts(artifacts: list[dict[str, object]] | None = None) -> dict[str, object]:
    records = _placeholder_artifacts() if artifacts is None else artifacts
    by_path = {str(record["path"]): record for record in records}
    return {
        "created_at_utc": "2026-08-14T00:00:00Z",
        "code": {"repository": "ozw4/pyosv", "git_commit": "1" * 40, "dirty": False},
        "environment": {
            "python": "3.10.14",
            "lock_file": "uv.lock",
            "lock_sha256": by_path["uv.lock"]["sha256"],
            "controls": dict(_CONTROLS),
        },
        "source": {"f3_completion_sha256": SHA_A},
        "dataset": {
            "dataset_id": "fixture-f3",
            "shape": [2, 3, 4],
            "storage_dtype": ">f4",
            "files": list(reversed(deepcopy(_DATASET_FILES))),
        },
        "experiment": {
            "config_file": "experiment.json",
            "config_sha256": by_path["experiment.json"]["sha256"],
        },
        "semantics": {
            "evaluation": "f3_public_reference_agreement",
            "public_reference_is_geological_truth": False,
            "evaluation_units": 1,
            "displayed_condition": "Q-QUAL",
            "stage_order": ["ft", "fv", "fvt"],
        },
        "artifacts": records,
    }


def _build(parts: dict[str, object] | None = None) -> dict[str, object]:
    return build_manifest(**(_parts() if parts is None else parts))  # type: ignore[arg-type]


def _write_artifacts(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative_path, (tier, role, content) in _ARTIFACT_DETAILS.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        records.append(artifact_file_record(root, relative_path, tier=tier, role=role))
    return records


def _prepare_directory(root: Path) -> dict[str, object]:
    root.mkdir()
    manifest = _build(_parts(_write_artifacts(root)))
    write_manifest(root, manifest)
    return manifest


def _write_raw_manifest(root: Path, manifest: dict[str, object]) -> None:
    (root / PUBLICATION_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def test_build_validate_round_trip_uses_exact_contract_and_stable_orders() -> None:
    parts = _parts()
    original = deepcopy(parts)

    manifest = _build(parts)

    assert parts == original
    assert tuple(manifest) == (
        "schema",
        "publication_id",
        "created_at_utc",
        "code",
        "environment",
        "source",
        "dataset",
        "experiment",
        "semantics",
        "artifacts",
    )
    assert manifest["schema"] == F3_COMPACT_PUBLICATION_MANIFEST_SCHEMA
    assert validate_manifest(manifest) == manifest
    assert [item["role"] for item in manifest["dataset"]["files"]] == sorted(  # type: ignore[index]
        item["role"] for item in _DATASET_FILES
    )
    assert [item["path"] for item in manifest["artifacts"]] == sorted(  # type: ignore[index]
        _ARTIFACT_DETAILS
    )
    without_id = {key: value for key, value in manifest.items() if key != "publication_id"}
    assert compute_publication_id(without_id) == manifest["publication_id"]


def test_artifact_input_order_does_not_change_publication_id() -> None:
    first = _parts()
    second = deepcopy(first)
    second["artifacts"].reverse()  # type: ignore[union-attr]

    assert _build(first)["publication_id"] == _build(second)["publication_id"]


def test_created_time_does_not_change_publication_id() -> None:
    original = _build()
    changed = _parts()
    changed["created_at_utc"] = "2026-08-14T12:34:56Z"

    assert _build(changed)["publication_id"] == original["publication_id"]


def test_derived_record_does_not_change_publication_id() -> None:
    original = _build()
    changed = _parts()
    figure = next(
        record
        for record in changed["artifacts"]
        if record["path"] == "figures/f3_ft.png"  # type: ignore[union-attr]
    )
    figure["size"] = 99
    figure["sha256"] = SHA_C

    assert _build(changed)["publication_id"] == original["publication_id"]


def test_primary_record_changes_publication_id() -> None:
    original = _build()
    changed = _parts()
    summary = next(
        record
        for record in changed["artifacts"]  # type: ignore[union-attr]
        if record["path"] == "f3_q_qual_vs_public_ref_summary.csv"
    )
    summary["sha256"] = SHA_C

    assert _build(changed)["publication_id"] != original["publication_id"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unknown": 1}),
        lambda value: value.pop("source"),
        lambda value: value.update({"schema": "pyosv.publication_manifest.v1"}),
        lambda value: value["code"].update({"git_commit": "A" * 40}),
        lambda value: value["environment"]["controls"].pop("PYOSV_ACCEL"),
        lambda value: value["source"].update({"f3_completion_sha256": "invalid"}),
        lambda value: value["dataset"].update({"shape": [2, 0, 4]}),
        lambda value: value["dataset"].update({"storage_dtype": "float32"}),
        lambda value: value["dataset"]["files"].pop(),
        lambda value: value["dataset"]["files"][0].update({"filename": "other.dat"}),
        lambda value: value["semantics"].update({"evaluation": "known_truth"}),
        lambda value: value["semantics"].update({"public_reference_is_geological_truth": True}),
        lambda value: value["semantics"].update({"displayed_condition": "Q-REF"}),
        lambda value: value["semantics"].update({"stage_order": ["fvt", "fv", "ft"]}),
        lambda value: value["artifacts"][0].update({"role": "metric_table"}),
        lambda value: value["artifacts"][0].update({"tier": "diagnostic"}),
        lambda value: value["artifacts"][0].update({"size": 0}),
    ],
)
def test_rejects_wrong_schema_fields_and_values(mutation: object) -> None:
    manifest = _build()
    mutation(manifest)  # type: ignore[operator]

    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_rejects_duplicate_dataset_role() -> None:
    manifest = _build()
    manifest["dataset"]["files"].append(deepcopy(manifest["dataset"]["files"][0]))  # type: ignore[index]

    with pytest.raises(ValueError, match="roles must be unique"):
        validate_manifest(manifest)


def test_rejects_duplicate_artifact_path() -> None:
    manifest = _build()
    manifest["artifacts"].append(deepcopy(manifest["artifacts"][0]))  # type: ignore[union-attr,index]

    with pytest.raises(ValueError, match="paths must be unique"):
        validate_manifest(manifest)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("environment", "lock_file", "/uv.lock"),
        ("environment", "lock_file", "locks/uv.lock"),
        ("experiment", "config_file", "../experiment.json"),
        ("artifact", "path", "figures\\f3.png"),
        ("artifact", "path", "figure_data/../f3.csv"),
        ("dataset_file", "filename", "data/ep.dat"),
    ],
)
def test_rejects_unsafe_paths(section: str, field: str, value: str) -> None:
    manifest = _build()
    if section == "artifact":
        manifest["artifacts"][0][field] = value  # type: ignore[index]
    elif section == "dataset_file":
        manifest["dataset"]["files"][0][field] = value  # type: ignore[index]
    else:
        manifest[section][field] = value  # type: ignore[index]

    with pytest.raises(ValueError):
        validate_manifest(manifest)


@pytest.mark.parametrize("pretty", [False, True])
def test_write_and_validate_complete_directory(tmp_path: Path, pretty: bool) -> None:
    root = tmp_path / f"publication-{pretty}"
    root.mkdir()
    manifest = _build(_parts(_write_artifacts(root)))

    destination = write_manifest(root, manifest, pretty=pretty)

    assert destination == root / PUBLICATION_MANIFEST_FILENAME
    assert destination.read_bytes().endswith(b"\n")
    assert validate_publication_directory(root) == manifest


def test_directory_validator_rejects_file_tampering(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    _prepare_directory(root)
    path = root / "report.md"
    path.write_bytes(b"X" * path.stat().st_size)

    with pytest.raises(ValueError, match="SHA-256"):
        validate_publication_directory(root)


def test_directory_validator_rejects_unrecorded_file(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    _prepare_directory(root)
    (root / "unexpected.txt").write_text("extra\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid regular file set"):
        validate_publication_directory(root)


def test_directory_validator_rejects_missing_file(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    _prepare_directory(root)
    (root / "figure_data/f3_ft.csv").unlink()

    with pytest.raises(ValueError, match="regular non-symlink file"):
        validate_publication_directory(root)


def test_directory_validator_rejects_symlink_artifact(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    _prepare_directory(root)
    artifact = root / "figures/f3_ft.png"
    external = tmp_path / "external.png"
    external.write_bytes(artifact.read_bytes())
    artifact.unlink()
    try:
        artifact.symlink_to(external)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="regular non-symlink file"):
        validate_publication_directory(root)


def test_directory_validator_rejects_unrecorded_symlink(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    _prepare_directory(root)
    try:
        (root / "unexpected-link").symlink_to(root / "report.md")
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="must not contain symlinks"):
        validate_publication_directory(root)


@pytest.mark.parametrize(
    ("link", "replacement", "message"),
    [
        ("lock", SHA_A, "environment lock SHA-256"),
        ("experiment", SHA_B, "experiment config SHA-256"),
    ],
)
def test_directory_validator_rejects_link_hash_mismatch(
    tmp_path: Path,
    link: str,
    replacement: str,
    message: str,
) -> None:
    root = tmp_path / link
    root.mkdir()
    records = _write_artifacts(root)
    parts = _parts(records)
    if link == "lock":
        parts["environment"]["lock_sha256"] = replacement  # type: ignore[index]
    else:
        parts["experiment"]["config_sha256"] = replacement  # type: ignore[index]
    manifest = _build(parts)
    _write_raw_manifest(root, manifest)

    with pytest.raises(ValueError, match=message):
        validate_publication_directory(root)


def test_directory_validator_rejects_link_role_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "publication"
    root.mkdir()
    records = _write_artifacts(root)
    lock = next(record for record in records if record["path"] == "uv.lock")
    lock["role"] = "summary_table"
    manifest = _build(_parts(records))
    _write_raw_manifest(root, manifest)

    with pytest.raises(ValueError, match="environment_lock"):
        validate_publication_directory(root)


def test_validate_only_imports_neither_source_module_nor_optional_runtimes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "publication"
    _prepare_directory(root)
    package_root = Path(__file__).resolve().parents[3] / "src"
    script = """
import importlib.abc
import sys

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.endswith(".f3_compact_publication.source"):
            raise RuntimeError(f"blocked source import: {fullname}")
        if fullname == "matplotlib" or fullname.startswith("matplotlib."):
            raise RuntimeError(f"blocked matplotlib import: {fullname}")
        if fullname == "numba" or fullname.startswith("numba."):
            raise RuntimeError(f"blocked numba import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
from pyosv.evaluation.f3_compact_publication.manifest import validate_publication_directory
validate_publication_directory(sys.argv[1])
"""
    task_environment = os.environ.copy()
    task_environment["PYTHONPATH"] = str(package_root)

    result = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        env=task_environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
