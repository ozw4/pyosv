from __future__ import annotations

from dataclasses import replace

import pytest

from pyosv.evaluation.synthetic_mode_comparison import (
    ModeCellSpec,
    SyntheticModeComparisonConfig,
    build_mode_comparison_plan,
)
from pyosv.evaluation.synthetic_quality import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticVotingConfig,
)
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


def test_plan_rejects_workflow_settings_from_different_voting_config() -> None:
    plan = build_mode_comparison_plan(SyntheticModeComparisonConfig())

    with pytest.raises(ValueError, match="quality_workflow_settings must match"):
        replace(plan, voting_config=SyntheticVotingConfig())


def test_plan_rejects_workflow_settings_from_different_skinning_or_explicit_flags() -> None:
    plan = build_mode_comparison_plan(SyntheticModeComparisonConfig())

    with pytest.raises(ValueError, match="workflow_settings must match"):
        replace(plan, skinning_config=SyntheticSkinningConfig(min_likelihood=0.7))
    with pytest.raises(ValueError, match="workflow_settings must match"):
        replace(plan, skinner_min_likelihood_explicit=True)


def test_plan_normalizes_case_ids_and_trial_seeds_to_immutable_tuples() -> None:
    plan = build_mode_comparison_plan(SyntheticModeComparisonConfig())
    mutable_case_ids = list(plan.case_ids)
    mutable_trial_seeds = list(plan.trial_seeds)

    normalized = replace(
        plan,
        case_ids=mutable_case_ids,  # type: ignore[arg-type]
        trial_seeds=mutable_trial_seeds,  # type: ignore[arg-type]
    )
    mutable_case_ids.append("single_dipping_plane")
    mutable_trial_seeds.append(1)

    assert normalized.case_ids == plan.case_ids
    assert normalized.trial_seeds == plan.trial_seeds
    assert isinstance(normalized.case_ids, tuple)
    assert isinstance(normalized.trial_seeds, tuple)


@pytest.mark.parametrize(
    "field",
    (
        "include_oracle_workflow_isolation",
        "skinner_method_explicit",
        "skinner_min_likelihood_explicit",
        "skinner_growth_source_explicit",
        "skinner_accepted_occupancy_radius_explicit",
        "skinner_boundary_fallback_explicit",
    ),
)
def test_plan_rejects_non_boolean_flags(field: str) -> None:
    plan = build_mode_comparison_plan(SyntheticModeComparisonConfig())

    with pytest.raises(ValueError, match=rf"^{field} must be a bool$"):
        replace(plan, **{field: 1})


def test_plan_rejects_invalid_truth_metric_config() -> None:
    plan = build_mode_comparison_plan(SyntheticModeComparisonConfig())

    with pytest.raises(
        ValueError,
        match="^truth_metric_config must be a SyntheticTruthMetricConfig$",
    ):
        replace(plan, truth_metric_config=object())
