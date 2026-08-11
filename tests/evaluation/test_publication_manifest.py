from __future__ import annotations

from copy import deepcopy

import pytest

from pyosv.evaluation.publication_manifest import (
    PUBLICATION_MANIFEST_SCHEMA,
    build_publication_manifest,
    canonical_json_bytes,
    compute_publication_id,
    validate_publication_manifest,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _parts() -> dict[str, object]:
    return {
        "created_at_utc": "2026-08-09T00:00:00Z",
        "code": {"repository": "ozw4/pyosv", "git_commit": "1" * 40, "dirty": False},
        "environment": {
            "python": "3.10.14",
            "lock_file": "uv.lock",
            "lock_sha256": SHA_A,
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
        "datasets": {
            "f3": {
                "dataset_id": "f3d-official-v1",
                "shape": [420, 400, 100],
                "dtype": ">f4",
                "files": [
                    {"role": "input", "filename": "ep.dat", "size": 67_200_000, "sha256": SHA_B}
                ],
            }
        },
        "experiment": {
            "config_file": "experiment.json",
            "config_sha256": SHA_C,
            "source_runs": {
                "synthetic": {"completion_sha256": SHA_A},
                "f3": {"completion_sha256": SHA_B},
            },
        },
        "semantics": {
            "synthetic": "known_truth",
            "f3": "public_reference_agreement",
            "f3_public_reference_is_geological_truth": False,
            "f3_evaluation_units": 1,
        },
        "artifacts": [
            {
                "path": "synthetic/metrics.csv",
                "tier": "primary",
                "role": "metric_table",
                "size": 1234,
                "sha256": SHA_A,
            },
            {
                "path": "synthetic/report.png",
                "tier": "derived",
                "role": "figure",
                "size": 5678,
                "sha256": SHA_B,
            },
        ],
    }


def _build(parts: dict[str, object] | None = None) -> dict[str, object]:
    values = _parts() if parts is None else parts
    return build_publication_manifest(**values)  # type: ignore[arg-type]


def test_builds_and_validates_minimal_manifest_without_mutating_input() -> None:
    parts = _parts()
    original = deepcopy(parts)

    manifest = _build(parts)

    assert parts == original
    assert manifest["schema"] == PUBLICATION_MANIFEST_SCHEMA
    assert validate_publication_manifest(manifest) == manifest
    assert validate_publication_manifest(manifest) is not manifest
    without_id = {key: value for key, value in manifest.items() if key != "publication_id"}
    assert compute_publication_id(without_id) == manifest["publication_id"]


def test_canonical_json_is_independent_of_dict_insertion_order() -> None:
    assert canonical_json_bytes({"b": 2, "a": {"d": 4, "c": 3}}) == canonical_json_bytes(
        {"a": {"c": 3, "d": 4}, "b": 2}
    )


def test_builder_sorts_artifacts_and_input_order_does_not_change_output() -> None:
    first = _parts()
    second = deepcopy(first)
    second["artifacts"] = list(reversed(second["artifacts"]))  # type: ignore[arg-type]

    assert _build(first) == _build(second)


@pytest.mark.parametrize("field", ["created_at_utc", "derived"])
def test_timestamp_and_derived_artifacts_do_not_change_publication_id(field: str) -> None:
    original = _build()
    changed = _parts()
    if field == "created_at_utc":
        changed["created_at_utc"] = "2026-08-10T12:34:56Z"
    else:
        changed["artifacts"][1]["sha256"] = SHA_C  # type: ignore[index]
        changed["artifacts"][1]["size"] = 9999  # type: ignore[index]

    assert _build(changed)["publication_id"] == original["publication_id"]


@pytest.mark.parametrize("field", ["primary", "git_commit", "f3_sha256", "config_sha256"])
def test_primary_artifact_and_provenance_change_publication_id(field: str) -> None:
    original = _build()
    changed = _parts()
    if field == "primary":
        changed["artifacts"][0]["sha256"] = SHA_C  # type: ignore[index]
    elif field == "git_commit":
        changed["code"]["git_commit"] = "2" * 40  # type: ignore[index]
    elif field == "f3_sha256":
        changed["datasets"]["f3"]["files"][0]["sha256"] = SHA_C  # type: ignore[index]
    else:
        changed["experiment"]["config_sha256"] = SHA_A  # type: ignore[index]

    assert _build(changed)["publication_id"] != original["publication_id"]


def _invalid_manifest() -> dict[str, object]:
    return _build()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unknown": 1}),
        lambda value: value.pop("code"),
        lambda value: value["code"].update({"git_commit": "A" * 40}),
        lambda value: value["environment"].update({"lock_sha256": "not-a-sha"}),
        lambda value: value["environment"].update({"lock_file": "/uv.lock"}),
        lambda value: value["experiment"].update({"config_file": "../experiment.json"}),
        lambda value: value["artifacts"][0].update({"path": "synthetic\\metrics.csv"}),
        lambda value: value["artifacts"].append(deepcopy(value["artifacts"][0])),
        lambda value: value["artifacts"].reverse(),
        lambda value: value["datasets"]["f3"]["files"].append(
            {"role": "input", "filename": "other.dat", "size": 1, "sha256": SHA_A}
        ),
        lambda value: value["datasets"]["f3"]["files"][0].update({"filename": "f3/ep.dat"}),
        lambda value: value["artifacts"][0].update({"size": 0}),
        lambda value: value["datasets"]["f3"]["files"][0].update({"size": -1}),
        lambda value: value["artifacts"][0].update({"size": float("nan")}),
        lambda value: value["artifacts"][0].update({"size": float("inf")}),
        lambda value: value["datasets"]["f3"].update({"shape": [420, 0, 100]}),
        lambda value: value["datasets"]["f3"].update({"dtype": "float32"}),
        lambda value: value["semantics"].update({"synthetic": "reference_agreement"}),
        lambda value: value["semantics"].update({"f3_public_reference_is_geological_truth": True}),
        lambda value: value.update({"publication_id": SHA_C}),
        lambda value: value["artifacts"].insert(
            0,
            {
                "path": "publication_manifest.json",
                "tier": "primary",
                "role": "manifest",
                "size": 1,
                "sha256": SHA_A,
            },
        ),
    ],
)
def test_rejects_invalid_manifests(mutation: object) -> None:
    manifest = _invalid_manifest()
    mutation(manifest)  # type: ignore[operator]

    with pytest.raises(ValueError):
        validate_publication_manifest(manifest)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": value})


def test_canonical_json_rejects_non_standard_scalar() -> None:
    class IntegerLike(int):
        pass

    with pytest.raises(TypeError):
        canonical_json_bytes({"value": IntegerLike(1)})


def test_validate_false_allows_provisional_publication_id_only() -> None:
    manifest = _build()
    manifest["publication_id"] = "pending"

    assert (
        validate_publication_manifest(manifest, verify_publication_id=False)["publication_id"]
        == "pending"
    )
    with pytest.raises(ValueError):
        validate_publication_manifest(manifest)
