from __future__ import annotations

import json
from copy import deepcopy

import pytest

from pyosv.evaluation.publication_experiment import (
    PUBLICATION_CONDITION_ORDER,
    PUBLICATION_EXPERIMENT_SCHEMA,
    build_publication_experiment,
    publication_experiment_bytes,
    validate_publication_experiment,
)


def _parts() -> dict[str, object]:
    return {
        "synthetic": {
            "shape": [49, 49, 49],
            "case_order": ["case-b", "case-a"],
            "trials": [
                {"case_id": "case-a", "trial_id": "trial-2", "seed": 20260708},
                {"case_id": "case-b", "trial_id": "trial-1", "seed": 20260707},
                {"case_id": "case-b", "trial_id": "trial-1", "seed": None},
            ],
            "skinning_enabled": True,
            "resolved_plan": {
                "scanner": {"sigma": 1.0, "enabled": True},
                "variant": "quality",
            },
        },
        "f3": {
            "dataset_id": "f3d-official-v1",
            "shape": [420, 400, 100],
            "storage_dtype": ">f4",
            "resolved_plan": {
                "mode": "reference",
                "thresholds": [0.25, 0.5],
            },
        },
        "publication": {
            "stage_order": ["ft", "fv", "fvt"],
            "metric_keys": [
                "synthetic/fvt/all/precision",
                "f3/fv/all/agreement",
            ],
            "slice_selection_policy": {
                "f3": {"axis": "n3", "index": 210},
                "synthetic": {"axis": "n2", "index": 24},
            },
        },
    }


def _build(parts: dict[str, object] | None = None) -> dict[str, object]:
    values = _parts() if parts is None else parts
    return build_publication_experiment(**values)  # type: ignore[arg-type]


def test_builds_and_validates_minimal_experiment_without_mutating_input() -> None:
    parts = _parts()
    original = deepcopy(parts)

    experiment = _build(parts)

    assert parts == original
    assert experiment["schema"] == PUBLICATION_EXPERIMENT_SCHEMA
    assert experiment["condition_order"] == list(PUBLICATION_CONDITION_ORDER)
    assert validate_publication_experiment(experiment) == experiment
    assert validate_publication_experiment(experiment) is not experiment


def test_mapping_insertion_order_does_not_change_compact_bytes() -> None:
    first = _build()
    second_parts = deepcopy(_parts())
    for section in ("synthetic", "f3", "publication"):
        value = second_parts[section]
        second_parts[section] = dict(reversed(list(value.items())))  # type: ignore[union-attr]
    second_parts["synthetic"]["resolved_plan"] = {  # type: ignore[index]
        "variant": "quality",
        "scanner": {"enabled": True, "sigma": 1.0},
    }
    second = dict(reversed(list(_build(second_parts).items())))

    assert publication_experiment_bytes(first) == publication_experiment_bytes(second)


def test_trial_input_order_is_canonicalized() -> None:
    first = _parts()
    second = deepcopy(first)
    second["synthetic"]["trials"].reverse()  # type: ignore[index]

    first_experiment = _build(first)
    second_experiment = _build(second)

    assert first_experiment == second_experiment
    assert publication_experiment_bytes(first_experiment) == publication_experiment_bytes(
        second_experiment
    )
    assert first_experiment["synthetic"]["trials"] == [  # type: ignore[index]
        {"case_id": "case-b", "trial_id": "trial-1", "seed": None},
        {"case_id": "case-b", "trial_id": "trial-1", "seed": 20260707},
        {"case_id": "case-a", "trial_id": "trial-2", "seed": 20260708},
    ]


def test_validator_normalizes_trial_order() -> None:
    experiment = _build()
    reordered = deepcopy(experiment)
    reordered["synthetic"]["trials"].reverse()  # type: ignore[index]

    assert validate_publication_experiment(reordered) == experiment
    assert publication_experiment_bytes(reordered) == publication_experiment_bytes(experiment)


def test_metric_key_input_order_is_canonicalized() -> None:
    first = _parts()
    second = deepcopy(first)
    second["publication"]["metric_keys"].reverse()  # type: ignore[index]

    assert _build(first) == _build(second)
    assert publication_experiment_bytes(_build(first)) == publication_experiment_bytes(
        _build(second)
    )


def test_case_and_stage_order_are_preserved() -> None:
    experiment = _build()

    assert experiment["synthetic"]["case_order"] == ["case-b", "case-a"]  # type: ignore[index]
    assert experiment["publication"]["stage_order"] == ["ft", "fv", "fvt"]  # type: ignore[index]


def test_pretty_and_compact_bytes_have_the_same_json_value() -> None:
    experiment = _build()
    compact = publication_experiment_bytes(experiment)
    pretty = publication_experiment_bytes(experiment, pretty=True)

    assert compact.endswith(b"\n")
    assert pretty.endswith(b"\n")
    assert compact.count(b"\n") == 1
    assert b'\n  "condition_order": [' in pretty
    assert json.loads(compact) == json.loads(pretty) == experiment


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unknown": 1}),
        lambda value: value.pop("f3"),
        lambda value: value["synthetic"].update({"unknown": 1}),
        lambda value: value["publication"].pop("stage_order"),
    ],
)
def test_rejects_unknown_and_missing_fields(mutation: object) -> None:
    experiment = _build()
    mutation(experiment)  # type: ignore[operator]

    with pytest.raises(ValueError):
        validate_publication_experiment(experiment)


@pytest.mark.parametrize(
    ("section", "shape"),
    [
        ("synthetic", [49, 49]),
        ("synthetic", [49, 0, 49]),
        ("f3", [420, True, 100]),
    ],
)
def test_rejects_invalid_shape(section: str, shape: list[object]) -> None:
    parts = _parts()
    parts[section]["shape"] = shape  # type: ignore[index]

    with pytest.raises(ValueError, match="shape"):
        _build(parts)


def test_rejects_duplicate_case() -> None:
    parts = _parts()
    parts["synthetic"]["case_order"] = ["case-a", "case-a"]  # type: ignore[index]

    with pytest.raises(ValueError, match="duplicates"):
        _build(parts)


def test_rejects_duplicate_trial_identity() -> None:
    parts = _parts()
    trials = parts["synthetic"]["trials"]  # type: ignore[index]
    trials.append(deepcopy(trials[0]))

    with pytest.raises(ValueError, match="trial identities"):
        _build(parts)


def test_rejects_trial_for_unknown_case() -> None:
    parts = _parts()
    parts["synthetic"]["trials"][0]["case_id"] = "unknown"  # type: ignore[index]

    with pytest.raises(ValueError, match="case_order"):
        _build(parts)


def test_rejects_bool_seed() -> None:
    parts = _parts()
    parts["synthetic"]["trials"][0]["seed"] = True  # type: ignore[index]

    with pytest.raises(ValueError, match="seed"):
        _build(parts)


def test_rejects_duplicate_metric_key() -> None:
    parts = _parts()
    metrics = parts["publication"]["metric_keys"]  # type: ignore[index]
    metrics.append(metrics[0])

    with pytest.raises(ValueError, match="duplicates"):
        _build(parts)


def test_validator_rejects_unsorted_metric_keys() -> None:
    experiment = _build()
    experiment["publication"]["metric_keys"].reverse()  # type: ignore[index]

    with pytest.raises(ValueError, match="sorted"):
        validate_publication_experiment(experiment)


def test_rejects_invalid_storage_dtype() -> None:
    parts = _parts()
    parts["f3"]["storage_dtype"] = "float32"  # type: ignore[index]

    with pytest.raises(ValueError, match="storage_dtype"):
        _build(parts)


@pytest.mark.parametrize("section", ["synthetic", "f3"])
def test_rejects_non_object_resolved_plan(section: str) -> None:
    parts = _parts()
    parts[section]["resolved_plan"] = []  # type: ignore[index]

    with pytest.raises(ValueError, match="resolved_plan"):
        _build(parts)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_nested_json(value: float) -> None:
    parts = _parts()
    parts["publication"]["slice_selection_policy"] = {"value": value}  # type: ignore[index]

    with pytest.raises(ValueError, match="finite"):
        _build(parts)


def test_rejects_changed_condition_order() -> None:
    experiment = _build()
    experiment["condition_order"].reverse()  # type: ignore[union-attr]

    with pytest.raises(ValueError, match="condition_order"):
        validate_publication_experiment(experiment)
