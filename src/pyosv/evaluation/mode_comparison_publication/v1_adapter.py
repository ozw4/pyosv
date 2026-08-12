"""Read-only adapter from validated source bundles to publication v1 inputs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import PurePosixPath

from pyosv.evaluation.publication_experiment import build_publication_experiment

from .config import CANONICAL_STAGE_ORDER, FIGURE_SELECTION_POLICY
from .models import PublicationReport
from .registry import PUBLICATION_METRIC_REGISTRY

__all__ = ["adapt_publication_sources"]


def adapt_publication_sources(report: PublicationReport) -> dict[str, object]:
    """Copy validated source identities into path-independent publication v1 inputs."""
    dataset = _adapt_f3_dataset(report)
    experiment = build_publication_experiment(
        synthetic=_synthetic_experiment(report),
        f3={
            "dataset_id": dataset["f3"]["dataset_id"],
            "shape": list(report.f3.result.volume_shape),
            "storage_dtype": report.f3.result.storage_dtype,
            "resolved_plan": _required_mapping(report.f3.run_manifest, "plan", "F3"),
        },
        publication={
            "stage_order": list(CANONICAL_STAGE_ORDER),
            "metric_keys": ["/".join(entry.identity) for entry in PUBLICATION_METRIC_REGISTRY],
            "slice_selection_policy": json.loads(
                json.dumps(FIGURE_SELECTION_POLICY, allow_nan=False)
            ),
        },
    )
    return {
        "datasets": dataset,
        "source_runs": {
            "synthetic": {"completion_sha256": report.synthetic.completion_sha256},
            "f3": {"completion_sha256": report.f3.completion_sha256},
        },
        "experiment": experiment,
    }


def _adapt_f3_dataset(report: PublicationReport) -> dict[str, object]:
    identity = report.f3.dataset_identity
    if not isinstance(identity, Mapping):
        raise ValueError("F3 dataset identity must be an object")
    dataset_id = identity.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("F3 dataset identity is missing dataset_id")
    files_value = identity.get("files")
    if not isinstance(files_value, list) or not files_value:
        raise ValueError("F3 dataset identity files must be a non-empty array")

    files: list[dict[str, object]] = []
    layout: tuple[tuple[int, int, int], str] | None = None
    roles: set[str] = set()
    for index, value in enumerate(files_value):
        if not isinstance(value, Mapping):
            raise ValueError(f"F3 dataset file {index} must be an object")
        role = value.get("role")
        shape = value.get("shape")
        storage_dtype = value.get("storage_dtype")
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(shape, list)
            or len(shape) != 3
            or any(type(dimension) is not int or dimension <= 0 for dimension in shape)
            or not isinstance(storage_dtype, str)
        ):
            raise ValueError(f"F3 dataset file {index} has an invalid role or layout")
        if role in roles:
            raise ValueError(f"duplicate F3 dataset file role {role!r}")
        roles.add(role)
        current_layout = ((shape[0], shape[1], shape[2]), storage_dtype)
        if layout is None:
            layout = current_layout
        elif current_layout != layout:
            raise ValueError("F3 dataset files have inconsistent shape or storage dtype")

        try:
            filename = report.f3.dataset_spec.filename_for(role)
        except (KeyError, ValueError) as error:
            raise ValueError(f"F3 dataset role {role!r} has no filename mapping") from error
        if (
            not isinstance(filename, str)
            or not filename
            or "\\" in filename
            or PurePosixPath(filename).name != filename
        ):
            raise ValueError(f"F3 dataset role {role!r} has an invalid filename mapping")
        files.append(
            {
                "role": role,
                "filename": filename,
                "size": value.get("size"),
                "sha256": value.get("sha256"),
            }
        )

    assert layout is not None
    result_layout = (tuple(report.f3.result.volume_shape), report.f3.result.storage_dtype)
    if layout != result_layout:
        raise ValueError("F3 dataset identity shape or storage dtype does not match the result")
    if getattr(report.f3.result, "dataset_id", dataset_id) != dataset_id:
        raise ValueError("F3 dataset identity does not match the result dataset_id")

    files.sort(key=lambda item: item["role"])
    return {
        "f3": {
            "dataset_id": dataset_id,
            "shape": list(layout[0]),
            "dtype": layout[1],
            "files": files,
        }
    }


def _synthetic_experiment(report: PublicationReport) -> dict[str, object]:
    manifest = report.synthetic.manifest
    resolved_plan = _required_mapping(manifest, "resolved_plan", "synthetic")
    trials_value = manifest.get("trials")
    if not isinstance(trials_value, list):
        raise ValueError("synthetic manifest trials must be an array")
    trials: list[dict[str, object]] = []
    for index, value in enumerate(trials_value):
        if not isinstance(value, Mapping):
            raise ValueError(f"synthetic trial {index} must be an object")
        missing = {"case_id", "trial_id", "case_generation_seed"} - set(value)
        if missing:
            raise ValueError(f"synthetic trial {index} is missing fields {sorted(missing)}")
        trials.append(
            {
                "case_id": value["case_id"],
                "trial_id": value["trial_id"],
                "seed": value["case_generation_seed"],
            }
        )
    if "shape" not in manifest:
        raise ValueError("synthetic manifest is missing shape")
    return {
        "shape": manifest["shape"],
        "case_order": list(report.synthetic.case_order),
        "trials": trials,
        "skinning_enabled": report.synthetic.skinning_enabled,
        "resolved_plan": resolved_plan,
    }


def _required_mapping(source: Mapping[str, object], field: str, label: str) -> Mapping[str, object]:
    if field not in source:
        raise ValueError(f"{label} manifest is missing {field}")
    value = source[field]
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} manifest {field} must be an object")
    return value
