"""Seeded trial contracts for synthetic mode comparisons."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

from ..synthetic_quality.cases import SyntheticQualityCaseDefinition

_FILESYSTEM_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True, slots=True)
class SyntheticTrialSpec:
    """One deterministic or seeded synthetic case realization."""

    trial_id: str
    case_id: str
    seed: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not _FILESYSTEM_SAFE_ID.fullmatch(self.case_id):
            raise ValueError("case_id must be a non-empty filesystem-safe ID")
        if self.seed is not None:
            if not isinstance(self.seed, Integral) or isinstance(self.seed, bool) or self.seed < 0:
                raise ValueError("seed must be None or a non-negative integer")
            object.__setattr__(self, "seed", int(self.seed))
        expected_id = _trial_id(self.case_id, self.seed)
        if self.trial_id != expected_id:
            raise ValueError(f"trial_id must be {expected_id!r} for case_id and seed")


def validate_trial_seeds(trial_seeds: Sequence[int]) -> tuple[int, ...]:
    """Validate and normalize a non-empty sequence of unique trial seeds."""

    try:
        values = tuple(trial_seeds)
    except TypeError as error:
        raise ValueError("trial_seeds must be a sequence of non-negative integers") from error
    if not values:
        raise ValueError("trial_seeds must include at least one seed")
    if any(not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0 for seed in values):
        raise ValueError("trial_seeds must contain only non-negative integers")
    normalized = tuple(int(seed) for seed in values)
    duplicates = {seed for seed in normalized if normalized.count(seed) > 1}
    if duplicates:
        joined = ",".join(str(seed) for seed in sorted(duplicates))
        raise ValueError(f"duplicate trial seed(s): {joined}")
    return normalized


def expand_synthetic_trials(
    case_definitions: Sequence[SyntheticQualityCaseDefinition],
    trial_seeds: Sequence[int],
) -> tuple[SyntheticTrialSpec, ...]:
    """Expand cases in case order, repeating only stochastic cases by seed."""

    seeds = validate_trial_seeds(trial_seeds)
    trials: list[SyntheticTrialSpec] = []
    for definition in case_definitions:
        if not isinstance(definition, SyntheticQualityCaseDefinition):
            raise ValueError("case_definitions must contain SyntheticQualityCaseDefinition values")
        if definition.is_stochastic:
            trials.extend(
                SyntheticTrialSpec(
                    trial_id=_trial_id(definition.case_id, seed),
                    case_id=definition.case_id,
                    seed=seed,
                )
                for seed in seeds
            )
        else:
            trials.append(
                SyntheticTrialSpec(
                    trial_id=_trial_id(definition.case_id, None),
                    case_id=definition.case_id,
                    seed=None,
                )
            )

    trial_ids = tuple(trial.trial_id for trial in trials)
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("expanded trial IDs must be unique")
    return tuple(trials)


def _trial_id(case_id: str, seed: int | None) -> str:
    if seed is None:
        return case_id
    return f"{case_id}__seed_{seed}"
