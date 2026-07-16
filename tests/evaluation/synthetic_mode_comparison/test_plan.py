from __future__ import annotations

from dataclasses import replace

import pytest

from pyosv.evaluation.synthetic_mode_comparison import (
    ModeCellSpec,
    SyntheticModeComparisonConfig,
    build_mode_comparison_plan,
)
from pyosv.evaluation.synthetic_quality import SyntheticScannerConfig
from pyosv.evaluation.synthetic_quality.cases import CASE_IDS


def _labels(config: SyntheticModeComparisonConfig) -> tuple[str, ...]:
    return tuple(cell.label for cell in build_mode_comparison_plan(config).cells)


def test_default_plan_has_canonical_cell_order_and_resolved_workflows() -> None:
    plan = build_mode_comparison_plan(SyntheticModeComparisonConfig())

    assert tuple(cell.label for cell in plan.cells) == (
        "RL-SCAN",
        "Q-SCAN",
        "ORACLE-REF",
        "ORACLE-QUAL",
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    )
    assert plan.comparison_variant == "current_default"
    assert plan.trial_seeds == (20260707,)
    assert tuple((trial.trial_id, trial.seed) for trial in plan.trials) == (
        ("single_vertical_plane", None),
    )
    assert plan.reference_workflow_settings.voting_config.voter_thin_mode == "reference"
    assert plan.quality_workflow_settings.voting_config.voter_thin_mode == "hybrid_v2"


def test_oracle_isolation_can_be_omitted() -> None:
    config = SyntheticModeComparisonConfig(include_oracle_workflow_isolation=False)

    assert _labels(config) == (
        "RL-SCAN",
        "Q-SCAN",
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("scanner_thin_mode", "normal", "scanner_template.scanner_thin_mode"),
        ("remove_edge_effects", False, "scanner_template.remove_edge_effects"),
        ("refinement_factor", 1, "scanner_template.refinement_factor"),
        ("backend", "quality", "scanner_template.backend"),
        ("backend", "fast", "scanner_template.backend"),
        ("backend", "ensemble", "scanner_template.backend"),
    ),
)
def test_plan_rejects_noncanonical_scanner_template(
    field: str, value: object, message: str
) -> None:
    scanner = replace(SyntheticScannerConfig(), **{field: value})

    with pytest.raises(ValueError, match=message):
        build_mode_comparison_plan(SyntheticModeComparisonConfig(scanner_template=scanner))


def test_plan_rejects_noncanonical_variant() -> None:
    with pytest.raises(ValueError, match="comparison_variant must be 'current_default'"):
        SyntheticModeComparisonConfig(comparison_variant="boundary_aware_voter_v1")


def test_case_set_and_explicit_case_ids_preserve_validator_order() -> None:
    geometry = build_mode_comparison_plan(SyntheticModeComparisonConfig(case_set="geometry"))
    explicit_ids = (CASE_IDS[2], CASE_IDS[0])
    explicit = build_mode_comparison_plan(SyntheticModeComparisonConfig(case_ids=explicit_ids))

    assert geometry.case_ids == CASE_IDS[:3]
    assert explicit.case_ids == explicit_ids


def test_shape_uses_existing_shape_validation() -> None:
    with pytest.raises(ValueError, match="shape must be a 3D"):
        SyntheticModeComparisonConfig(shape=(17, 17))  # type: ignore[arg-type]


@pytest.mark.parametrize("case_ids", (("missing",), (CASE_IDS[0], CASE_IDS[0])))
def test_explicit_case_ids_use_existing_validation(case_ids: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        build_mode_comparison_plan(SyntheticModeComparisonConfig(case_ids=case_ids))


def test_unknown_case_set_uses_existing_validation() -> None:
    with pytest.raises(ValueError, match="^unknown case_set: missing$"):
        build_mode_comparison_plan(SyntheticModeComparisonConfig(case_set="missing"))


@pytest.mark.parametrize(
    "changes",
    (
        {"scope": "missing"},
        {"scanner_backend": "fast"},
        {"workflow_mode": "diagnostic"},
        {"input_mode": "both"},
        {"workflow_mode": "reference"},
    ),
)
def test_cell_rejects_unknown_or_inconsistent_axes(changes: dict[str, object]) -> None:
    fields: dict[str, object] = {
        "label": "RL-SCAN",
        "scope": "scanner-only",
        "input_mode": "scanner",
        "scanner_backend": "reference-like",
        "workflow_mode": None,
    }
    fields.update(changes)

    with pytest.raises(ValueError):
        ModeCellSpec(**fields)  # type: ignore[arg-type]


def test_plan_rejects_duplicate_cell_labels() -> None:
    plan = build_mode_comparison_plan(SyntheticModeComparisonConfig())

    with pytest.raises(ValueError, match="duplicate mode cell label"):
        replace(plan, cells=(plan.cells[0], plan.cells[0], *plan.cells[2:]))
