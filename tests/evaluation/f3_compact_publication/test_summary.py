from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyosv.evaluation.f3_compact_publication import (
    AMPLITUDE_DTYPE,
    AMPLITUDE_FILENAME,
    AMPLITUDE_ROLE,
    DISPLAY_CELL,
    EXPERIMENT_SCHEMA,
    PUBLIC_REFERENCE_LABEL,
    STAGE_ORDER,
    SUMMARY_HEADER,
    AmplitudeIdentity,
    CompactSourceContext,
    RidgeStageThresholds,
    SelectedSlice,
    SourceRidgeThresholdContract,
    StageSource,
    build_experiment,
    build_summary_rows,
    experiment_json_bytes,
    summary_csv_bytes,
)
from pyosv.evaluation.f3d_mode_comparison import (
    F3_METRIC_SCHEMA_VERSION,
    F3_REFERENCE_STAGE_FILES,
    F3DatasetSpec,
    METRIC_REGISTRY,
    MetricRow,
)
from pyosv.evaluation.f3d_mode_comparison.metrics import F3_REFERENCE_STAGE_ROLES
from pyosv.evaluation.mode_comparison_publication.models import F3SourceBundle

_SHAPE = (2, 3, 4)
_SIZE = int(np.prod(_SHAPE)) * np.dtype(AMPLITUDE_DTYPE).itemsize
_METRICS = (
    ("normalized_correlation", "all", "normalized_correlation"),
    ("mean_absolute_difference", "all", "mean_absolute_difference"),
    ("nonzero_fraction_ratio", "all", "nonzero_fraction_ratio"),
    ("buffered_f1", "positive_p99_radius2", "buffered_f1"),
    (
        "candidate_to_reference_p95_voxel",
        "positive_p99_distance",
        "candidate_to_reference_p95",
    ),
    (
        "reference_to_candidate_p95_voxel",
        "positive_p99_distance",
        "reference_to_candidate_p95",
    ),
)
_DEFINITIONS = {(item.stage, item.selection, item.metric): item for item in METRIC_REGISTRY}


@dataclass(frozen=True)
class _SummaryFixture:
    context: CompactSourceContext
    values: dict[tuple[str, str], float | None]


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _metric_rows() -> tuple[tuple[MetricRow, ...], dict[tuple[str, str], float | None]]:
    rows = []
    values: dict[tuple[str, str], float | None] = {}
    for stage_index, stage in enumerate(STAGE_ORDER):
        stage_values = (
            0.91 - stage_index * 0.01,
            0.11 + stage_index * 0.01,
            1.01 + stage_index * 0.01,
            0.81 - stage_index * 0.01,
            None if stage == "fv" else 1.5 + stage_index,
            2.5 + stage_index,
        )
        for (_, selection, metric), value in zip(_METRICS, stage_values, strict=True):
            definition = _DEFINITIONS[(stage, selection, metric)]
            rows.append(
                MetricRow(
                    schema_version=F3_METRIC_SCHEMA_VERSION,
                    dataset_id="compact-summary-fixture",
                    cell_label=DISPLAY_CELL,
                    scanner_backend="quality",
                    workflow_mode="quality",
                    stage=stage,
                    region="full",
                    selection=selection,
                    reference_file=F3_REFERENCE_STAGE_FILES[stage],
                    metric=metric,
                    value=value,
                    unit=definition.unit,
                    direction=definition.direction,
                    contrast_eligible=definition.contrast_eligible and value is not None,
                )
            )
            values[(stage, metric)] = value
    return tuple(rows), values


@pytest.fixture
def summary_fixture(tmp_path: Path) -> _SummaryFixture:
    spec = F3DatasetSpec(
        dataset_id="compact-summary-fixture",
        shape=_SHAPE,
        storage_dtype=AMPLITUDE_DTYPE,
        files=(
            ("input", "ep.dat"),
            ("reference_fault_likelihood", "fl.dat"),
            ("reference_fault_votes", "fv.dat"),
            ("reference_thinned_fault_votes", "fvt.dat"),
        ),
        expected_bytes=_SIZE,
    )
    identities = [
        {
            "role": role,
            "size": _SIZE,
            "sha256": _digest(filename),
            "shape": list(_SHAPE),
            "storage_dtype": AMPLITUDE_DTYPE,
        }
        for role, filename in spec.files
    ]
    rows, values = _metric_rows()
    result = SimpleNamespace(
        dataset_id=spec.dataset_id,
        volume_shape=_SHAPE,
        storage_dtype=AMPLITUDE_DTYPE,
        metric_rows=rows,
    )
    bundle = tmp_path / "source" / "bundle"
    data_root = tmp_path / "source" / "data"
    source = F3SourceBundle(
        path=bundle,
        data_root=data_root,
        dataset_spec=spec,
        run_manifest={},
        completion_sha256=_digest("completion"),
        result=result,
        metric_evidence=(),
        dataset_identity={"dataset_id": spec.dataset_id, "files": identities},
    )
    fingerprints = tuple(_digest(kind) for kind in ("scanner", "voting", "thinning"))
    kinds = ("scanner", "voting", "thinning")
    stage_sources = tuple(
        StageSource(
            stage=stage,
            public_reference_role=F3_REFERENCE_STAGE_ROLES[stage],
            public_reference_filename=F3_REFERENCE_STAGE_FILES[stage],
            public_reference_path=data_root / F3_REFERENCE_STAGE_FILES[stage],
            public_reference_sha256=_digest(F3_REFERENCE_STAGE_FILES[stage]),
            candidate_source_kind=kind,
            candidate_fingerprint=fingerprint,
            candidate_filename=f"{stage}.dat",
            candidate_path=bundle / "stages" / kind / fingerprint / f"{stage}.dat",
        )
        for stage, kind, fingerprint in zip(STAGE_ORDER, kinds, fingerprints, strict=True)
    )
    ridge = SourceRidgeThresholdContract(
        selection="positive_p99_radius2",
        percentile=99.0,
        buffer_radius=2.0,
        stages=tuple(
            RidgeStageThresholds(
                stage=stage,
                public_reference_threshold=0.7 - index * 0.1,
                q_qual_threshold=0.6 - index * 0.1,
            )
            for index, stage in enumerate(STAGE_ORDER)
        ),
    )
    amplitude = AmplitudeIdentity(
        role=AMPLITUDE_ROLE,
        filename=AMPLITUDE_FILENAME,
        resolved_path=data_root / AMPLITUDE_FILENAME,
        shape=_SHAPE,
        storage_dtype=AMPLITUDE_DTYPE,
        size=_SIZE,
        sha256=_digest(AMPLITUDE_FILENAME),
    )
    context = CompactSourceContext(
        f3=source,
        amplitude=amplitude,
        q_qual_cell=SimpleNamespace(label=DISPLAY_CELL),
        stage_sources=stage_sources,
        ridge_threshold_contract=ridge,
        selected_slice=SelectedSlice(
            axis="i2",
            index=1,
            policy="public_fvt_positive_p99_peak",
            public_fvt_reference_threshold=0.5,
            ridge_count_score=7,
        ),
    )
    return _SummaryFixture(context=context, values=values)


def _with_rows(
    context: CompactSourceContext,
    rows: tuple[object, ...],
) -> CompactSourceContext:
    fields = vars(context.f3.result).copy()
    fields["metric_rows"] = rows
    result = SimpleNamespace(**fields)
    return replace(context, f3=replace(context.f3, result=result))


def test_summary_uses_fixed_stage_order_and_exact_source_metrics(
    summary_fixture: _SummaryFixture,
) -> None:
    rows = build_summary_rows(summary_fixture.context)

    assert tuple(row["stage"] for row in rows) == STAGE_ORDER
    assert tuple(rows[0]) == SUMMARY_HEADER
    for row in rows:
        stage = str(row["stage"])
        for field, _, metric in _METRICS:
            assert row[field] == summary_fixture.values[(stage, metric)]
    assert {str(row["public_reference_file"]) for row in rows} == {"fl.dat", "fv.dat", "fvt.dat"}


def test_nullable_distance_is_an_empty_csv_field(summary_fixture: _SummaryFixture) -> None:
    payload = summary_csv_bytes(build_summary_rows(summary_fixture.context))
    rows = tuple(csv.DictReader(io.StringIO(payload.decode("utf-8"))))

    assert tuple(rows[0]) == SUMMARY_HEADER
    assert rows[1]["candidate_to_reference_p95_voxel"] == ""
    assert rows[1]["reference_to_candidate_p95_voxel"] == "3.5"


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_summary_serializer_requires_the_fixed_header(
    summary_fixture: _SummaryFixture,
    change: str,
) -> None:
    rows = [dict(row) for row in build_summary_rows(summary_fixture.context)]
    if change == "missing":
        rows[0].pop("buffered_f1")
    else:
        rows[0]["unexpected"] = 1.0

    with pytest.raises(ValueError, match="fixed CSV header"):
        summary_csv_bytes(rows)


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_source_metric_must_appear_exactly_once(
    summary_fixture: _SummaryFixture,
    change: str,
) -> None:
    rows = summary_fixture.context.f3.result.metric_rows
    target = rows[0]
    changed = rows[1:] if change == "missing" else (target, *rows)

    with pytest.raises(ValueError, match="exactly one metric row"):
        build_summary_rows(_with_rows(summary_fixture.context, changed))


@pytest.mark.parametrize("field,value", [("unit", "wrong"), ("direction", "neutral")])
def test_source_metric_unit_and_direction_follow_registry(
    summary_fixture: _SummaryFixture,
    field: str,
    value: str,
) -> None:
    rows = summary_fixture.context.f3.result.metric_rows
    changed_row = replace(rows[0], **{field: value})

    with pytest.raises(ValueError, match="semantics do not match"):
        build_summary_rows(_with_rows(summary_fixture.context, (changed_row, *rows[1:])))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_summary_serializer_rejects_nonfinite_values(
    summary_fixture: _SummaryFixture,
    value: float,
) -> None:
    rows = [dict(row) for row in build_summary_rows(summary_fixture.context)]
    rows[0]["normalized_correlation"] = value

    with pytest.raises(ValueError, match="must be finite"):
        summary_csv_bytes(rows)


def test_experiment_serialization_is_deterministic_and_pretty_is_semantic_only(
    summary_fixture: _SummaryFixture,
) -> None:
    experiment = build_experiment(summary_fixture.context)
    first = experiment_json_bytes(experiment)
    second = experiment_json_bytes(build_experiment(summary_fixture.context))
    pretty = experiment_json_bytes(experiment, pretty=True)

    assert first == second
    assert json.loads(first) == json.loads(pretty)
    assert tuple(experiment) == (
        "schema",
        "source",
        "dataset",
        "display",
        "stages",
        "slice",
        "ridge_thresholds",
        "visualization",
    )
    assert experiment["schema"] == EXPERIMENT_SCHEMA


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_experiment_serializer_rejects_nonfinite_values(
    summary_fixture: _SummaryFixture,
    value: float,
) -> None:
    experiment = dict(build_experiment(summary_fixture.context))
    visualization = dict(experiment["visualization"])
    visualization["amplitude_percentile"] = value
    experiment["visualization"] = visualization

    with pytest.raises(ValueError, match="finite"):
        experiment_json_bytes(experiment)


def _all_strings(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)


def test_experiment_is_path_independent_and_has_only_public_display_labels(
    summary_fixture: _SummaryFixture,
) -> None:
    experiment = build_experiment(summary_fixture.context)
    strings = tuple(_all_strings(experiment))

    assert not any(str(summary_fixture.context.f3.path) in value for value in strings)
    assert not any(str(summary_fixture.context.f3.data_root) in value for value in strings)
    assert experiment["display"] == {
        "public_reference_label": PUBLIC_REFERENCE_LABEL,
        "candidate_label": DISPLAY_CELL,
        "stage_order": list(STAGE_ORDER),
    }
    assert all(
        label not in experiment_json_bytes(experiment).decode("utf-8")
        for label in ("RL-REF", "RL-QUAL", "Q-REF")
    )


def test_experiment_contains_official_and_amplitude_file_identities(
    summary_fixture: _SummaryFixture,
) -> None:
    experiment = build_experiment(summary_fixture.context)
    files = experiment["dataset"]["files"]

    assert tuple(item["role"] for item in files) == tuple(
        sorted((*summary_fixture.context.f3.dataset_spec.roles, AMPLITUDE_ROLE))
    )
    amplitude = next(item for item in files if item["role"] == AMPLITUDE_ROLE)
    assert amplitude == {
        "role": AMPLITUDE_ROLE,
        "filename": AMPLITUDE_FILENAME,
        "shape": list(_SHAPE),
        "storage_dtype": AMPLITUDE_DTYPE,
        "size": _SIZE,
        "sha256": summary_fixture.context.amplitude.sha256,
    }
    assert {item["filename"] for item in files} == {
        "ep.dat",
        "fl.dat",
        "fv.dat",
        "fvt.dat",
        AMPLITUDE_FILENAME,
    }


def test_experiment_slice_and_thresholds_match_context(
    summary_fixture: _SummaryFixture,
) -> None:
    experiment = build_experiment(summary_fixture.context)
    selected = summary_fixture.context.selected_slice

    assert experiment["slice"] == {
        "axis": selected.axis,
        "index": selected.index,
        "selection_policy": selected.policy,
        "score": selected.ridge_count_score,
        "public_fvt_reference_threshold": selected.public_fvt_reference_threshold,
    }
    assert experiment["ridge_thresholds"] == [
        {
            "stage": item.stage,
            "public_reference_threshold": item.public_reference_threshold,
            "q_qual_candidate_threshold": item.q_qual_threshold,
        }
        for item in summary_fixture.context.ridge_threshold_contract.stages
    ]
    assert [item["q_qual_stage_semantics"] for item in experiment["stages"]] == [
        "quality scanner output in Q-QUAL lineage",
        "quality scanner voting output in Q-QUAL lineage",
        "Q-QUAL thinned voting output",
    ]
