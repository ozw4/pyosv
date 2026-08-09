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

_SHA = "a" * 64
_GIT_SHA = "b" * 40


def _controls() -> dict[str, str]:
    return {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "NUMBA_DISABLE_JIT": "0",
        "NUMBA_NUM_THREADS": "1",
        "PYOSV_ACCEL": "auto",
    }


def _inputs() -> dict[str, object]:
    return {
        "created_at_utc": "2026-08-09T00:00:00Z",
        "code": {
            "repository": "ozw4/pyosv",
            "git_commit": _GIT_SHA,
            "dirty": False,
        },
        "environment": {
            "python": "3.10.14",
            "lock_file": "uv.lock",
            "lock_sha256": _SHA,
            "controls": _controls(),
        },
        "datasets": {
            "f3": {
                "dataset_id": "f3d-official-v1",
                "shape": [420, 400, 100],
                "dtype": ">f4",
                "files": [
                    {
                        "role": "input",
                        "filename": "ep.dat",
                        "size": 67_200_000,
                        "sha256": _SHA,
                    }
                ],
            }
        },
        "experiment": {
            "config_file": "experiment.json",
            "config_sha256": _SHA,
            "source_runs": {
                "synthetic": {"completion_sha256": _SHA},
                "f3": {"completion_sha256": _SHA},
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
                "size": 123,
                "sha256": "c" * 64,
            },
            {
                "path": "figures/example.png",
                "tier": "derived",
                "role": "figure",
                "size": 456,
                "sha256": "d" * 64,
            },
        ],
    }


def _build(**overrides: object) -> dict[str, object]:
    values = _inputs()
    values.update(overrides)
    return build_publication_manifest(**values)  # type: ignore[arg-type]


def test_build_and_validate_minimal_manifest() -> None:
    manifest = _build()

    assert manifest["schema"] == PUBLICATION_MANIFEST_SCHEMA
    assert validate_publication_manifest(manifest) == manifest
    assert compute_publication_id(manifest) == manifest["publication_id"]


def test_canonical_json_ignores_mapping_insertion_order() -> None:
    left = {"b": 2, "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "b": 2}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)


def test_builder_sorts_artifacts_and_input_order_does_not_change_identity() -> None:
    forward = _inputs()["artifacts"]
    reverse = list(reversed(forward))  # type: ignore[arg-type]

    left = _build(artifacts=forward)
    right = _build(artifacts=reverse)

    assert left == right
    assert [item["path"] for item in left["artifacts"]] == [  # type: ignore[index]
        "figures/example.png",
        "synthetic/metrics.csv",
    ]


def test_timestamp_and_derived_artifact_do_not_change_publication_id() -> None:
    baseline = _build()
    changed_time = _build(created_at_utc="2026-08-10T01:02:03Z")
    changed_artifacts = deepcopy(_inputs()["artifacts"])
    changed_artifacts[0]["sha256"] = "e" * 64  # type: ignore[index]
    changed_primary = _build(artifacts=changed_artifacts)
    changed_artifacts = deepcopy(_inputs()["artifacts"])
    changed_artifacts[1]["sha256"] = "e" * 64  # type: ignore[index]
    changed_derived = _build(artifacts=changed_artifacts)

    assert baseline["publication_id"] == changed_time["publication_id"]
    assert baseline["publication_id"] == changed_derived["publication_id"]
    assert baseline["publication_id"] != changed_primary["publication_id"]


@pytest.mark.parametrize(
    "mutation",
    [
        "git_commit",
        "dataset_sha",
        "config_sha",
    ],
)
def test_stable_inputs_change_publication_id(mutation: str) -> None:
    baseline = _build()
    inputs = _inputs()
    if mutation == "git_commit":
        inputs["code"]["git_commit"] = "c" * 40  # type: ignore[index]
    elif mutation == "dataset_sha":
        inputs["datasets"]["f3"]["files"][0]["sha256"] = "f" * 64  # type: ignore[index]
    else:
        inputs["experiment"]["config_sha256"] = "f" * 64  # type: ignore[index]

    changed = build_publication_manifest(**inputs)  # type: ignore[arg-type]
    assert baseline["publication_id"] != changed["publication_id"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.__setitem__("unknown", 1), "field set mismatch"),
        (lambda value: value.pop("semantics"), "field set mismatch"),
        (
            lambda value: value["code"].__setitem__("git_commit", "ABC"),
            "Git SHA",
        ),
        (
            lambda value: value["environment"].__setitem__("lock_sha256", "ABC"),
            "SHA-256",
        ),
        (
            lambda value: value["artifacts"][0].__setitem__("path", "/tmp/x"),
            "safe POSIX relative path",
        ),
        (
            lambda value: value["artifacts"][0].__setitem__("path", "../x"),
            "safe POSIX relative path",
        ),
        (
            lambda value: value["artifacts"][0].__setitem__("path", "a\\b"),
            "safe POSIX relative path",
        ),
        (
            lambda value: value["datasets"]["f3"].__setitem__("dtype", "<f4"),
            "dtype",
        ),
        (
            lambda value: value["datasets"]["f3"].__setitem__("shape", [420, 0, 100]),
            "positive integer",
        ),
        (
            lambda value: value["semantics"].__setitem__(
                "f3_public_reference_is_geological_truth", True
            ),
            "geological truth",
        ),
        (
            lambda value: value.__setitem__("publication_id", "f" * 64),
            "publication_id",
        ),
    ],
)
def test_invalid_manifests_are_rejected(mutation, match: str) -> None:
    value = _build()
    mutation(value)

    with pytest.raises(ValueError, match=match):
        validate_publication_manifest(value)


def test_duplicate_f3_role_is_rejected() -> None:
    value = _build()
    duplicate = deepcopy(value["datasets"]["f3"]["files"][0])  # type: ignore[index]
    value["datasets"]["f3"]["files"].append(duplicate)  # type: ignore[index]

    with pytest.raises(ValueError, match="duplicate F3 file role"):
        validate_publication_manifest(value, verify_publication_id=False)


def test_non_basename_f3_filename_is_rejected() -> None:
    value = _build()
    value["datasets"]["f3"]["files"][0]["filename"] = "data/ep.dat"  # type: ignore[index]

    with pytest.raises(ValueError, match="basename"):
        validate_publication_manifest(value, verify_publication_id=False)


@pytest.mark.parametrize("size", [0, -1, True])
def test_non_positive_artifact_size_is_rejected(size: object) -> None:
    value = _build()
    value["artifacts"][0]["size"] = size  # type: ignore[index]

    with pytest.raises(ValueError, match="positive integer"):
        validate_publication_manifest(value, verify_publication_id=False)


def test_duplicate_artifact_path_is_rejected() -> None:
    value = _build()
    value["artifacts"][1]["path"] = value["artifacts"][0]["path"]  # type: ignore[index]

    with pytest.raises(ValueError, match="duplicate artifact path"):
        validate_publication_manifest(value, verify_publication_id=False)


def test_unsorted_artifacts_are_rejected_by_validator() -> None:
    value = _build()
    value["artifacts"] = list(reversed(value["artifacts"]))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="sorted by path"):
        validate_publication_manifest(value, verify_publication_id=False)


def test_manifest_cannot_list_itself_as_artifact() -> None:
    artifacts = deepcopy(_inputs()["artifacts"])
    artifacts.append(
        {
            "path": "publication_manifest.json",
            "tier": "derived",
            "role": "manifest",
            "size": 1,
            "sha256": "e" * 64,
        }
    )

    with pytest.raises(ValueError, match="must not list itself"):
        _build(artifacts=artifacts)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        canonical_json_bytes({"value": value})


def test_canonical_json_rejects_non_json_scalar() -> None:
    class IntLike(int):
        pass

    with pytest.raises(ValueError, match="non-JSON"):
        canonical_json_bytes({"value": IntLike(1)})
