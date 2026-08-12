"""Pure data contract for deterministic publication experiment snapshots."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from pyosv.evaluation.publication_manifest import canonical_json_bytes

PUBLICATION_EXPERIMENT_SCHEMA = "pyosv.publication_experiment.v1"
PUBLICATION_CONDITION_ORDER = (
    "RL-REF",
    "RL-QUAL",
    "Q-REF",
    "Q-QUAL",
)

_TOP_LEVEL_FIELDS = {"schema", "condition_order", "synthetic", "f3", "publication"}
_SYNTHETIC_FIELDS = {
    "shape",
    "case_order",
    "trials",
    "skinning_enabled",
    "resolved_plan",
}
_F3_FIELDS = {"dataset_id", "shape", "storage_dtype", "resolved_plan"}
_PUBLICATION_FIELDS = {
    "stage_order",
    "metric_keys",
    "slice_selection_policy",
}

__all__ = [
    "PUBLICATION_CONDITION_ORDER",
    "PUBLICATION_EXPERIMENT_SCHEMA",
    "build_publication_experiment",
    "publication_experiment_bytes",
    "validate_publication_experiment",
]


def build_publication_experiment(
    *,
    synthetic: Mapping[str, object],
    f3: Mapping[str, object],
    publication: Mapping[str, object],
) -> dict[str, object]:
    """Build and normalize a publication experiment from explicit inputs."""
    candidate: dict[str, object] = {
        "schema": PUBLICATION_EXPERIMENT_SCHEMA,
        "condition_order": list(PUBLICATION_CONDITION_ORDER),
        "synthetic": _normalize_synthetic(synthetic),
        "f3": _normalize_f3(f3),
        "publication": _normalize_publication(publication, require_sorted_metrics=False),
    }
    return validate_publication_experiment(candidate)


def validate_publication_experiment(
    experiment: Mapping[str, object],
) -> dict[str, object]:
    """Strictly validate and normalize a publication experiment."""
    source = _require_mapping(experiment, "experiment")
    _require_fields(source, _TOP_LEVEL_FIELDS, "experiment")

    schema = _require_string(source["schema"], "schema")
    if schema != PUBLICATION_EXPERIMENT_SCHEMA:
        raise ValueError(f"schema must be {PUBLICATION_EXPERIMENT_SCHEMA!r}")

    condition_order = source["condition_order"]
    if (
        type(condition_order) is not list
        or any(type(item) is not str for item in condition_order)
        or tuple(condition_order) != PUBLICATION_CONDITION_ORDER
    ):
        raise ValueError("condition_order must match PUBLICATION_CONDITION_ORDER")

    return {
        "schema": schema,
        "condition_order": list(PUBLICATION_CONDITION_ORDER),
        "synthetic": _normalize_synthetic(source["synthetic"]),
        "f3": _normalize_f3(source["f3"]),
        "publication": _normalize_publication(source["publication"], require_sorted_metrics=True),
    }


def publication_experiment_bytes(
    experiment: Mapping[str, object],
    *,
    pretty: bool = False,
) -> bytes:
    """Serialize a validated experiment to deterministic UTF-8 JSON bytes."""
    if type(pretty) is not bool:
        raise ValueError("pretty must be a bool")
    normalized = validate_publication_experiment(experiment)
    if not pretty:
        return canonical_json_bytes(normalized) + b"\n"
    return (
        json.dumps(
            normalized,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{path} keys must be strings")
    return value


def _require_fields(value: Mapping[str, object], expected: set[str], path: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{path} has invalid fields; missing={missing}, unknown={unknown}")


def _require_string(value: object, path: str) -> str:
    if type(value) is not str:
        raise ValueError(f"{path} must be a string")
    return value


def _require_nonempty_string(value: object, path: str) -> str:
    result = _require_string(value, path)
    if not result:
        raise ValueError(f"{path} must not be empty")
    return result


def _normalize_shape(value: object, path: str) -> list[int]:
    if type(value) is not list or len(value) != 3:
        raise ValueError(f"{path} must be a three-element array")
    normalized: list[int] = []
    for index, dimension in enumerate(value):
        if type(dimension) is not int or dimension <= 0:
            raise ValueError(f"{path}[{index}] must be a positive integer")
        normalized.append(dimension)
    return normalized


def _normalize_unique_strings(value: object, path: str) -> list[str]:
    if type(value) is not list or not value:
        raise ValueError(f"{path} must be a non-empty array")
    normalized = [
        _require_nonempty_string(item, f"{path}[{index}]") for index, item in enumerate(value)
    ]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{path} must not contain duplicates")
    return normalized


def _normalize_json_object(value: object, path: str) -> dict[str, object]:
    source = _require_mapping(value, path)
    try:
        normalized = json.loads(canonical_json_bytes(dict(source)))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must contain only finite standard JSON values") from error
    return cast(dict[str, object], normalized)


def _normalize_synthetic(value: object) -> dict[str, object]:
    source = _require_mapping(value, "synthetic")
    _require_fields(source, _SYNTHETIC_FIELDS, "synthetic")
    case_order = _normalize_unique_strings(source["case_order"], "synthetic.case_order")
    case_positions = {case_id: index for index, case_id in enumerate(case_order)}

    trials_value = source["trials"]
    if type(trials_value) is not list:
        raise ValueError("synthetic.trials must be an array")
    trials: list[dict[str, object]] = []
    identities: set[tuple[str, str, int | None]] = set()
    for index, value in enumerate(trials_value):
        path = f"synthetic.trials[{index}]"
        trial = _require_mapping(value, path)
        _require_fields(trial, {"case_id", "trial_id", "seed"}, path)
        case_id = _require_nonempty_string(trial["case_id"], f"{path}.case_id")
        if case_id not in case_positions:
            raise ValueError(f"{path}.case_id must be present in synthetic.case_order")
        trial_id = _require_nonempty_string(trial["trial_id"], f"{path}.trial_id")
        seed = trial["seed"]
        if seed is not None and type(seed) is not int:
            raise ValueError(f"{path}.seed must be an integer or null")
        identity = (case_id, trial_id, seed)
        if identity in identities:
            raise ValueError("synthetic trial identities must be unique")
        identities.add(identity)
        trials.append({"case_id": case_id, "trial_id": trial_id, "seed": seed})

    trials.sort(key=lambda trial: _trial_sort_key(trial, case_positions))
    if type(source["skinning_enabled"]) is not bool:
        raise ValueError("synthetic.skinning_enabled must be a bool")
    return {
        "shape": _normalize_shape(source["shape"], "synthetic.shape"),
        "case_order": case_order,
        "trials": trials,
        "skinning_enabled": source["skinning_enabled"],
        "resolved_plan": _normalize_json_object(source["resolved_plan"], "synthetic.resolved_plan"),
    }


def _trial_sort_key(
    trial: Mapping[str, object],
    case_positions: Mapping[str, int],
) -> tuple[int, str, bool, int]:
    case_id = cast(str, trial["case_id"])
    trial_id = cast(str, trial["trial_id"])
    seed = cast(int | None, trial["seed"])
    return (
        case_positions[case_id],
        trial_id,
        seed is not None,
        seed if seed is not None else 0,
    )


def _normalize_f3(value: object) -> dict[str, object]:
    source = _require_mapping(value, "f3")
    _require_fields(source, _F3_FIELDS, "f3")
    storage_dtype = source["storage_dtype"]
    if type(storage_dtype) is not str or storage_dtype != ">f4":
        raise ValueError("f3.storage_dtype must be '>f4'")
    return {
        "dataset_id": _require_nonempty_string(source["dataset_id"], "f3.dataset_id"),
        "shape": _normalize_shape(source["shape"], "f3.shape"),
        "storage_dtype": ">f4",
        "resolved_plan": _normalize_json_object(source["resolved_plan"], "f3.resolved_plan"),
    }


def _normalize_publication(
    value: object,
    *,
    require_sorted_metrics: bool,
) -> dict[str, object]:
    source = _require_mapping(value, "publication")
    _require_fields(source, _PUBLICATION_FIELDS, "publication")
    stage_order = _normalize_unique_strings(source["stage_order"], "publication.stage_order")
    metric_keys = _normalize_unique_strings(source["metric_keys"], "publication.metric_keys")
    sorted_metric_keys = sorted(metric_keys)
    if require_sorted_metrics and metric_keys != sorted_metric_keys:
        raise ValueError("publication.metric_keys must be sorted")
    return {
        "stage_order": stage_order,
        "metric_keys": sorted_metric_keys,
        "slice_selection_policy": _normalize_json_object(
            source["slice_selection_policy"], "publication.slice_selection_policy"
        ),
    }
