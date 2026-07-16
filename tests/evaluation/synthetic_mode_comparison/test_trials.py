from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pyosv.evaluation.synthetic_mode_comparison import (
    SyntheticModeComparisonConfig,
    SyntheticTrialSpec,
    build_mode_comparison_plan,
    expand_synthetic_trials,
)
from pyosv.evaluation.synthetic_quality import SyntheticScannerConfig
from pyosv.evaluation.synthetic_quality.cases import EXTENDED_CASES
from pyosv.synthetic3d import SyntheticScannerInputConfig


def test_extended_cases_expand_only_stochastic_case_for_each_seed() -> None:
    seeds = (5, 3, 8, 1, 2)

    trials = expand_synthetic_trials(EXTENDED_CASES, seeds)

    assert tuple(trial.case_id for trial in trials[:6]) == tuple(
        definition.case_id for definition in EXTENDED_CASES[:6]
    )
    assert tuple(trial.seed for trial in trials[:6]) == (None,) * 6
    assert tuple(trial.seed for trial in trials[6:]) == seeds
    assert tuple(trial.trial_id for trial in trials[6:]) == tuple(
        f"weak_noisy_plane__seed_{seed}" for seed in seeds
    )
    assert len(trials) == 11
    assert len({trial.trial_id for trial in trials}) == len(trials)


def test_trial_expansion_and_plan_order_are_repeatable() -> None:
    config = SyntheticModeComparisonConfig(case_set="extended", trial_seeds=(13, 7))

    first = build_mode_comparison_plan(config)
    second = build_mode_comparison_plan(config)

    assert first.trials == second.trials
    assert first.trial_seeds == (13, 7)
    assert first.trials == expand_synthetic_trials(EXTENDED_CASES, (13, 7))


@pytest.mark.parametrize(
    "trial_seeds",
    ((), (1, 1), (True,), (-1,), (1.5,)),
)
def test_config_rejects_invalid_trial_seeds(trial_seeds: tuple[object, ...]) -> None:
    with pytest.raises(ValueError):
        SyntheticModeComparisonConfig(trial_seeds=trial_seeds)  # type: ignore[arg-type]


def test_empty_seed_list_fails_before_trial_expansion() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        expand_synthetic_trials(EXTENDED_CASES, ())


def test_trial_spec_is_immutable() -> None:
    trial = SyntheticTrialSpec("weak_noisy_plane__seed_3", "weak_noisy_plane", 3)

    with pytest.raises(FrozenInstanceError):
        trial.seed = 4  # type: ignore[misc]


def test_trial_seed_does_not_replace_scanner_input_seed() -> None:
    scanner = SyntheticScannerConfig(input_config=SyntheticScannerInputConfig(seed=123))

    plan = build_mode_comparison_plan(
        SyntheticModeComparisonConfig(
            case_set="extended",
            trial_seeds=(456,),
            scanner_template=scanner,
        )
    )

    assert plan.trials[-1].seed == 456
    assert plan.scanner_template.input_config.seed == 123
