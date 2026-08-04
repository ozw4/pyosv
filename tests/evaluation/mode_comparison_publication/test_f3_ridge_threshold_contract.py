from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from pyosv.evaluation.f3d_mode_comparison import (
    F3_BUFFERED_PERCENTILE,
    F3_BUFFER_RADIUS,
    F3_METRIC_SCHEMA_VERSION,
    F3_REFERENCE_STAGE_FILES,
    MetricEvidence,
)
from pyosv.evaluation.mode_comparison_publication.config import (
    CANONICAL_CELL_ORDER,
    CANONICAL_STAGE_ORDER,
)
from pyosv.evaluation.mode_comparison_publication.figures import (
    build_f3_ridge_threshold_contract,
)


_REFERENCE_THRESHOLDS = {"ft": 0.80, "fv": 0.70, "fvt": 0.60}
_CANDIDATE_THRESHOLDS = {
    "ft": {"RL-REF": 0.40, "RL-QUAL": 0.41, "Q-REF": 0.42, "Q-QUAL": 0.43},
    "fv": {"RL-REF": 0.44, "RL-QUAL": 0.45, "Q-REF": 0.46, "Q-QUAL": 0.47},
    "fvt": {"RL-REF": 0.40, "RL-QUAL": 0.45, "Q-REF": 0.50, "Q-QUAL": 0.55},
}


def _evidence(
    stage: str,
    cell: str,
    *,
    selection: str = "positive_p99_radius2",
    percentile: float = F3_BUFFERED_PERCENTILE,
    radius: float = F3_BUFFER_RADIUS,
    reference_threshold: float | None = None,
    candidate_threshold: float | None = None,
    include_candidate_threshold: bool = True,
) -> MetricEvidence:
    thresholds: dict[str, float] = {
        "percentile": percentile,
        "radius": radius,
        "reference_threshold": (
            _REFERENCE_THRESHOLDS[stage] if reference_threshold is None else reference_threshold
        ),
    }
    if include_candidate_threshold:
        thresholds["candidate_threshold"] = (
            _CANDIDATE_THRESHOLDS[stage][cell]
            if candidate_threshold is None
            else candidate_threshold
        )
    return MetricEvidence(
        schema_version=F3_METRIC_SCHEMA_VERSION,
        dataset_id="publication-fixture",
        cell_label=cell,
        stage=stage,
        region="full",
        selection=selection,
        reference_file=F3_REFERENCE_STAGE_FILES[stage],
        source_stage_fingerprint="a" * 64,
        reference_sha256="b" * 64,
        shape=(1, 1, 1),
        thresholds=tuple(thresholds.items()),
    )


def _all_evidence() -> list[MetricEvidence]:
    return [
        _evidence(stage, cell) for stage in CANONICAL_STAGE_ORDER for cell in CANONICAL_CELL_ORDER
    ]


def _source(evidence: list[MetricEvidence]) -> SimpleNamespace:
    return SimpleNamespace(metric_evidence=tuple(evidence))


def _set_threshold(evidence: MetricEvidence, name: str, value: float) -> MetricEvidence:
    return replace(
        evidence,
        thresholds=tuple(
            (key, value if key == name else current) for key, current in evidence.thresholds
        ),
    )


def test_f3_ridge_threshold_contract_uses_source_cell_thresholds_in_canonical_order() -> None:
    evidence = _all_evidence()
    contract = build_f3_ridge_threshold_contract(_source(list(reversed(evidence))))

    assert tuple(contract.stages) == CANONICAL_STAGE_ORDER
    serialized = contract.as_dict()
    assert serialized["selection"] == "positive_p99_radius2"
    assert serialized["percentile"] == F3_BUFFERED_PERCENTILE
    assert serialized["buffer_radius"] == F3_BUFFER_RADIUS
    assert tuple(serialized["stages"]) == CANONICAL_STAGE_ORDER
    for stage in CANONICAL_STAGE_ORDER:
        stage_contract = serialized["stages"][stage]
        assert stage_contract["reference_threshold"] == _REFERENCE_THRESHOLDS[stage]
        assert tuple(stage_contract["candidate_thresholds"]) == CANONICAL_CELL_ORDER
        assert stage_contract["candidate_thresholds"] == _CANDIDATE_THRESHOLDS[stage]
    assert serialized["stages"]["fvt"]["candidate_thresholds"] == {
        "RL-REF": 0.40,
        "RL-QUAL": 0.45,
        "Q-REF": 0.50,
        "Q-QUAL": 0.55,
    }


def test_f3_ridge_threshold_contract_rejects_missing_cell() -> None:
    evidence = _all_evidence()
    evidence = [
        item for item in evidence if not (item.stage == "fv" and item.cell_label == "Q-QUAL")
    ]
    with pytest.raises(ValueError):
        build_f3_ridge_threshold_contract(_source(evidence))


def test_f3_ridge_threshold_contract_rejects_duplicate_evidence() -> None:
    evidence = _all_evidence()
    evidence.append(evidence[0])
    with pytest.raises(ValueError):
        build_f3_ridge_threshold_contract(_source(evidence))


def test_f3_ridge_threshold_contract_rejects_unknown_cell() -> None:
    evidence = _all_evidence()
    invalid = evidence[0]
    object.__setattr__(invalid, "cell_label", "PUBLIC-REF")
    with pytest.raises(ValueError):
        build_f3_ridge_threshold_contract(_source(evidence))


def test_f3_ridge_threshold_contract_rejects_missing_candidate_threshold() -> None:
    evidence = _all_evidence()
    evidence[0] = _evidence("ft", "RL-REF", include_candidate_threshold=False)
    with pytest.raises(ValueError):
        build_f3_ridge_threshold_contract(_source(evidence))


def test_f3_ridge_threshold_contract_rejects_percentile_and_radius_mismatches() -> None:
    evidence = _all_evidence()
    evidence[0] = _set_threshold(evidence[0], "percentile", 98.0)
    with pytest.raises(ValueError):
        build_f3_ridge_threshold_contract(_source(evidence))

    evidence = _all_evidence()
    evidence[0] = _set_threshold(evidence[0], "radius", 3.0)
    with pytest.raises(ValueError):
        build_f3_ridge_threshold_contract(_source(evidence))


def test_f3_ridge_threshold_contract_rejects_mixed_stage_reference_thresholds() -> None:
    evidence = _all_evidence()
    index = next(
        index
        for index, item in enumerate(evidence)
        if item.stage == "fvt" and item.cell_label == "Q-QUAL"
    )
    evidence[index] = _set_threshold(evidence[index], "reference_threshold", 0.80)
    with pytest.raises(ValueError):
        build_f3_ridge_threshold_contract(_source(evidence))


def test_f3_ridge_threshold_contract_does_not_mix_stages() -> None:
    evidence = _all_evidence()
    stage_index = next(index for index, item in enumerate(evidence) if item.stage == "ft")
    evidence[stage_index] = replace(
        evidence[stage_index],
        stage="skin",
        reference_file=None,
        reference_sha256=None,
    )
    with pytest.raises(ValueError):
        build_f3_ridge_threshold_contract(_source(evidence))


def test_f3_ridge_threshold_contract_rejects_wrong_selection_and_nonfinite_threshold() -> None:
    evidence = _all_evidence()
    evidence[0] = replace(evidence[0], selection="positive_p99")
    with pytest.raises(ValueError):
        build_f3_ridge_threshold_contract(_source(evidence))

    evidence = _all_evidence()
    invalid = evidence[0]
    object.__setattr__(
        invalid,
        "thresholds",
        tuple(
            (key, float("inf") if key == "candidate_threshold" else value)
            for key, value in invalid.thresholds
        ),
    )
    with pytest.raises(ValueError):
        build_f3_ridge_threshold_contract(_source(evidence))


def test_f3_ridge_threshold_contract_requires_metric_evidence_instances() -> None:
    invalid = SimpleNamespace(
        stage="ft",
        cell_label="RL-REF",
        selection="positive_p99_radius2",
        thresholds={},
    )
    with pytest.raises(ValueError):
        build_f3_ridge_threshold_contract(SimpleNamespace(metric_evidence=(invalid,)))
