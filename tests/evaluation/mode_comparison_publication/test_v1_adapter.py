from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyosv.evaluation.f3d_mode_comparison import F3DatasetSpec
from pyosv.evaluation.mode_comparison_publication.config import (
    CANONICAL_STAGE_ORDER,
    FIGURE_SELECTION_POLICY,
)
from pyosv.evaluation.mode_comparison_publication.models import (
    F3SourceBundle,
    PublicationReport,
    SyntheticSourceBundle,
)
from pyosv.evaluation.mode_comparison_publication.registry import (
    PUBLICATION_METRIC_REGISTRY,
)
from pyosv.evaluation.mode_comparison_publication.v1_adapter import (
    adapt_publication_sources,
)
from pyosv.evaluation.publication_experiment import validate_publication_experiment


def _report(root: Path) -> PublicationReport:
    shape = (2, 3, 4)
    size = 96
    synthetic_manifest = {
        "shape": [9, 9, 9],
        "resolved_plan": {
            "shape": [9, 9, 9],
            "case_ids": ["case-b", "case-a"],
            "threshold": 0.5,
        },
        "trials": [
            {
                "order": 1,
                "case_id": "case-a",
                "trial_id": "trial-2",
                "case_generation_seed": 20260708,
                "scanner_input_seed": 17,
            },
            {
                "order": 0,
                "case_id": "case-b",
                "trial_id": "trial-1",
                "case_generation_seed": 20260707,
                "scanner_input_seed": 17,
            },
        ],
        "source_path": str(root / "private-synthetic-source"),
        "runtime_identity": {"hostname": "private-host"},
    }
    synthetic = SyntheticSourceBundle(
        path=root / "synthetic-source",
        manifest=synthetic_manifest,
        completion_sha256="a" * 64,
        metric_rows=(),
        contrast_rows=(),
        runtime_rows=(),
        skinning_enabled=True,
        case_order=("case-b", "case-a"),
    )

    spec = F3DatasetSpec(
        dataset_id="fixture-f3",
        shape=shape,
        storage_dtype=">f4",
        files=(("reference", "reference.dat"), ("input", "input.dat")),
        expected_bytes=size,
    )
    dataset_identity = {
        "dataset_id": "fixture-f3",
        "files": [
            {
                "role": "reference",
                "size": size,
                "sha256": "d" * 64,
                "shape": list(shape),
                "storage_dtype": ">f4",
            },
            {
                "role": "input",
                "size": size,
                "sha256": "e" * 64,
                "shape": list(shape),
                "storage_dtype": ">f4",
            },
        ],
    }
    f3_manifest = {
        "plan": {"cells": ["RL-REF", "Q-QUAL"], "stages": ["ft", "fv", "fvt"]},
        "runtime_identity": {"hostname": "private-host"},
        "source_path": str(root / "private-f3-source"),
    }
    result = SimpleNamespace(
        dataset_id="fixture-f3",
        volume_shape=shape,
        storage_dtype=">f4",
    )
    f3 = F3SourceBundle(
        path=root / "f3-source",
        data_root=root / "private-f3-data",
        dataset_spec=spec,
        run_manifest=f3_manifest,
        completion_sha256="f" * 64,
        result=result,
        metric_evidence=(),
        dataset_identity=dataset_identity,
    )
    return PublicationReport(synthetic=synthetic, f3=f3, tables={})


def test_adapts_validated_sources_to_path_independent_v1_inputs(tmp_path: Path) -> None:
    report = _report(tmp_path)

    adapted = adapt_publication_sources(report)

    assert set(adapted) == {"datasets", "source_runs", "experiment"}
    assert adapted["datasets"] == {
        "f3": {
            "dataset_id": "fixture-f3",
            "shape": [2, 3, 4],
            "dtype": ">f4",
            "files": [
                {
                    "role": "input",
                    "filename": "input.dat",
                    "size": 96,
                    "sha256": "e" * 64,
                },
                {
                    "role": "reference",
                    "filename": "reference.dat",
                    "size": 96,
                    "sha256": "d" * 64,
                },
            ],
        }
    }
    assert adapted["source_runs"] == {
        "synthetic": {"completion_sha256": "a" * 64},
        "f3": {"completion_sha256": "f" * 64},
    }

    experiment = validate_publication_experiment(adapted["experiment"])
    assert experiment == adapted["experiment"]
    assert experiment["synthetic"] == {
        "shape": [9, 9, 9],
        "case_order": ["case-b", "case-a"],
        "trials": [
            {"case_id": "case-b", "trial_id": "trial-1", "seed": 20260707},
            {"case_id": "case-a", "trial_id": "trial-2", "seed": 20260708},
        ],
        "skinning_enabled": True,
        "resolved_plan": {
            "case_ids": ["case-b", "case-a"],
            "shape": [9, 9, 9],
            "threshold": 0.5,
        },
    }
    assert experiment["f3"]["resolved_plan"] == report.f3.run_manifest["plan"]
    metric_keys = experiment["publication"]["metric_keys"]
    assert metric_keys == sorted(set(metric_keys))
    assert experiment["publication"] == {
        "stage_order": list(CANONICAL_STAGE_ORDER),
        "metric_keys": sorted("/".join(entry.identity) for entry in PUBLICATION_METRIC_REGISTRY),
        "slice_selection_policy": json.loads(json.dumps(FIGURE_SELECTION_POLICY)),
    }

    serialized = json.dumps(adapted, sort_keys=True)
    for path in (report.synthetic.path, report.f3.path, report.f3.data_root, tmp_path):
        assert str(path) not in serialized
    assert "runtime_identity" not in serialized


def test_adapter_does_not_mutate_sources_and_returns_new_nested_values(tmp_path: Path) -> None:
    report = _report(tmp_path)
    before = deepcopy(
        {
            "synthetic_manifest": report.synthetic.manifest,
            "f3_manifest": report.f3.run_manifest,
            "dataset_identity": report.f3.dataset_identity,
        }
    )

    adapted = adapt_publication_sources(report)
    adapted["experiment"]["synthetic"]["resolved_plan"]["threshold"] = 0.75
    adapted["datasets"]["f3"]["files"][0]["size"] = 1

    assert report.synthetic.manifest == before["synthetic_manifest"]
    assert report.f3.run_manifest == before["f3_manifest"]
    assert report.f3.dataset_identity == before["dataset_identity"]


@pytest.mark.parametrize("source", ["synthetic", "f3"])
def test_rejects_missing_resolved_plan(tmp_path: Path, source: str) -> None:
    report = _report(tmp_path)
    if source == "synthetic":
        manifest = dict(report.synthetic.manifest)
        manifest.pop("resolved_plan")
        report = replace(report, synthetic=replace(report.synthetic, manifest=manifest))
    else:
        manifest = dict(report.f3.run_manifest)
        manifest.pop("plan")
        report = replace(report, f3=replace(report.f3, run_manifest=manifest))

    with pytest.raises(ValueError, match="missing (resolved_plan|plan)"):
        adapt_publication_sources(report)


def test_rejects_mixed_f3_file_layouts(tmp_path: Path) -> None:
    report = _report(tmp_path)
    identity = deepcopy(report.f3.dataset_identity)
    identity["files"][1]["shape"] = [2, 3, 5]
    report = replace(report, f3=replace(report.f3, dataset_identity=identity))

    with pytest.raises(ValueError, match="inconsistent shape or storage dtype"):
        adapt_publication_sources(report)


def test_rejects_trial_missing_required_identity_field(tmp_path: Path) -> None:
    report = _report(tmp_path)
    manifest = deepcopy(report.synthetic.manifest)
    manifest["trials"][0].pop("case_generation_seed")
    report = replace(report, synthetic=replace(report.synthetic, manifest=manifest))

    with pytest.raises(ValueError, match="case_generation_seed"):
        adapt_publication_sources(report)
