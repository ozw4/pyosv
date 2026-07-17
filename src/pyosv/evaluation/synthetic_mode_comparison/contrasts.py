"""Paired contrasts and descriptive summaries for synthetic comparisons."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np

from .metrics import METRIC_REGISTRY, MetricDirection, MetricRow

AggregateSource = Literal["metric", "contrast"]

_CANONICAL_CELLS = frozenset(
    {
        "RL-SCAN",
        "Q-SCAN",
        "ORACLE-REF",
        "ORACLE-QUAL",
        "RL-REF",
        "RL-QUAL",
        "Q-REF",
        "Q-QUAL",
    }
)


@dataclass(frozen=True, slots=True)
class ContrastDefinition:
    """One fixed linear contrast over canonical comparison cells."""

    name: str
    coefficients: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not self.coefficients:
            raise ValueError("coefficients must not be empty")
        cells = tuple(cell for cell, _ in self.coefficients)
        if len(cells) != len(set(cells)):
            raise ValueError("contrast cells must be unique")
        unknown = set(cells) - _CANONICAL_CELLS
        if unknown:
            raise ValueError(f"unknown contrast cell(s): {','.join(sorted(unknown))}")
        normalized: list[tuple[str, float]] = []
        for cell, coefficient in self.coefficients:
            if isinstance(coefficient, bool) or not isinstance(coefficient, Real):
                raise ValueError("contrast coefficients must be finite numbers")
            value = float(coefficient)
            if not np.isfinite(value) or value == 0.0:
                raise ValueError("contrast coefficients must be finite and nonzero")
            normalized.append((cell, value))
        object.__setattr__(self, "coefficients", tuple(normalized))

    @property
    def component_cells(self) -> tuple[str, ...]:
        """Return component cell labels in formula order."""

        return tuple(cell for cell, _ in self.coefficients)


CONTRAST_DEFINITIONS = (
    ContrastDefinition("scanner_only_effect", (("Q-SCAN", 1.0), ("RL-SCAN", -1.0))),
    ContrastDefinition("oracle_workflow_effect", (("ORACLE-QUAL", 1.0), ("ORACLE-REF", -1.0))),
    ContrastDefinition("scanner_effect_ref", (("Q-REF", 1.0), ("RL-REF", -1.0))),
    ContrastDefinition("scanner_effect_qual", (("Q-QUAL", 1.0), ("RL-QUAL", -1.0))),
    ContrastDefinition("workflow_effect_rl", (("RL-QUAL", 1.0), ("RL-REF", -1.0))),
    ContrastDefinition("workflow_effect_q", (("Q-QUAL", 1.0), ("Q-REF", -1.0))),
    ContrastDefinition("end_to_end_delta", (("Q-QUAL", 1.0), ("RL-REF", -1.0))),
    ContrastDefinition(
        "scanner_main_effect",
        (("Q-REF", 0.5), ("RL-REF", -0.5), ("Q-QUAL", 0.5), ("RL-QUAL", -0.5)),
    ),
    ContrastDefinition(
        "workflow_main_effect",
        (("RL-QUAL", 0.5), ("RL-REF", -0.5), ("Q-QUAL", 0.5), ("Q-REF", -0.5)),
    ),
    ContrastDefinition(
        "scanner_workflow_interaction",
        (("Q-QUAL", 1.0), ("Q-REF", -1.0), ("RL-QUAL", -1.0), ("RL-REF", 1.0)),
    ),
)

_DEFINITION_NAMES = frozenset(definition.name for definition in CONTRAST_DEFINITIONS)
_METRIC_ORDER = {
    (definition.stage, definition.selection, definition.metric): index
    for index, definition in enumerate(METRIC_REGISTRY)
}


@dataclass(frozen=True, slots=True)
class ContrastRow:
    """One within-trial paired contrast observation."""

    contrast_name: str
    case_id: str
    trial_id: str
    seed: int | None
    stage: str
    selection: str
    metric: str
    unit: str
    direction: MetricDirection
    component_cells: tuple[str, ...]
    raw_value: float
    improvement_value: float | None

    def __post_init__(self) -> None:
        if self.contrast_name not in _DEFINITION_NAMES:
            raise ValueError(f"unknown contrast: {self.contrast_name}")
        for name in ("case_id", "trial_id", "stage", "selection", "metric", "unit"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(self, "seed", _validated_seed(self.seed))
        definition = _definition_by_name(self.contrast_name)
        if self.component_cells != definition.component_cells:
            raise ValueError("component_cells must match the contrast definition")
        if self.direction not in {"higher", "lower", "neutral"}:
            raise ValueError("direction must be 'higher', 'lower', or 'neutral'")
        raw_value = _finite_number(self.raw_value, "raw_value")
        object.__setattr__(self, "raw_value", raw_value)
        expected = _improvement(raw_value, self.direction)
        if self.improvement_value is None:
            if expected is not None:
                raise ValueError("improvement_value is required for directional metrics")
        else:
            improvement = _finite_number(self.improvement_value, "improvement_value")
            if expected is None or improvement != expected:
                raise ValueError("improvement_value is inconsistent with direction and raw_value")
            object.__setattr__(self, "improvement_value", improvement)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping in canonical field order."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class AggregateRow:
    """Descriptive statistics over trial-level absolute or contrast values."""

    source: AggregateSource
    case_id: str
    cell_label: str | None
    contrast_name: str | None
    stage: str
    selection: str
    metric: str
    unit: str
    direction: MetricDirection
    n: int
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float

    def __post_init__(self) -> None:
        if self.source not in {"metric", "contrast"}:
            raise ValueError("source must be 'metric' or 'contrast'")
        if self.source == "metric":
            if self.cell_label not in _CANONICAL_CELLS or self.contrast_name is not None:
                raise ValueError("metric aggregates require one canonical cell_label")
        elif self.cell_label is not None or self.contrast_name not in _DEFINITION_NAMES:
            raise ValueError("contrast aggregates require one known contrast_name")
        for name in ("case_id", "stage", "selection", "metric", "unit"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.direction not in {"higher", "lower", "neutral"}:
            raise ValueError("direction must be 'higher', 'lower', or 'neutral'")
        if not isinstance(self.n, int) or isinstance(self.n, bool) or self.n < 1:
            raise ValueError("n must be a positive integer")
        for name in ("mean", "median", "std", "min", "max", "q25", "q75"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), name))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping in canonical field order."""

        return asdict(self)


_MetricPairKey = tuple[str, str, int | None, str, str, str]


def compute_contrast_rows(rows: Sequence[MetricRow]) -> tuple[ContrastRow, ...]:
    """Compute all applicable fixed contrasts within each exact trial pairing."""

    metric_rows = _validate_metric_rows(rows)
    grouped: dict[_MetricPairKey, dict[str, MetricRow]] = defaultdict(dict)
    for row in metric_rows:
        if not row.contrast_eligible:
            continue
        key = (row.case_id, row.trial_id, row.seed, row.stage, row.selection, row.metric)
        if row.cell_label in grouped[key]:
            raise ValueError(f"duplicate metric row for pairing key and cell {row.cell_label!r}")
        grouped[key][row.cell_label] = row

    output: list[ContrastRow] = []
    case_trials = _case_trials(metric_rows)
    for definition in CONTRAST_DEFINITIONS:
        required = frozenset(definition.component_cells)
        applicable = {
            (case_id, stage, selection, metric)
            for (case_id, _trial_id, _seed, stage, selection, metric), by_cell in grouped.items()
            if required.intersection(by_cell)
        }
        for case_id, stage, selection, metric in sorted(applicable):
            for trial_id, seed in case_trials[case_id]:
                key = (case_id, trial_id, seed, stage, selection, metric)
                present = required.intersection(grouped.get(key, {}))
                if present != required:
                    missing = required - present
                    raise ValueError(
                        f"contrast {definition.name!r} is missing required cell(s) in "
                        f"trial {trial_id!r}: {','.join(sorted(missing))}"
                    )
        for key in sorted(grouped, key=_sortable_pair_key):
            by_cell = grouped[key]
            present = required.intersection(by_cell)
            if present and present != required:
                missing = required - present
                raise ValueError(
                    f"contrast {definition.name!r} is missing required cell(s): "
                    f"{','.join(sorted(missing))}"
                )
            if not present:
                continue
            components = [by_cell[cell] for cell in definition.component_cells]
            units = {row.unit for row in components}
            directions = {row.direction for row in components}
            if len(units) != 1:
                raise ValueError(f"contrast {definition.name!r} has mixed units")
            if len(directions) != 1:
                raise ValueError(f"contrast {definition.name!r} has mixed directions")
            raw_value = float(
                sum(
                    coefficient * by_cell[cell].value
                    for cell, coefficient in definition.coefficients
                )
            )
            if not np.isfinite(raw_value):
                raise ValueError(f"contrast {definition.name!r} produced a nonfinite value")
            first = components[0]
            output.append(
                ContrastRow(
                    contrast_name=definition.name,
                    case_id=first.case_id,
                    trial_id=first.trial_id,
                    seed=first.seed,
                    stage=first.stage,
                    selection=first.selection,
                    metric=first.metric,
                    unit=first.unit,
                    direction=first.direction,
                    component_cells=definition.component_cells,
                    raw_value=raw_value,
                    improvement_value=_improvement(raw_value, first.direction),
                )
            )
    return tuple(output)


def aggregate_metric_rows(rows: Sequence[MetricRow]) -> tuple[AggregateRow, ...]:
    """Aggregate absolute metric values across trials within each case and cell."""

    metric_rows = _validate_metric_rows(rows)
    groups: dict[tuple[str, str, str, str, str, str, MetricDirection], list[MetricRow]] = (
        defaultdict(list)
    )
    for row in metric_rows:
        key = (
            row.case_id,
            row.cell_label,
            row.stage,
            row.selection,
            row.metric,
            row.unit,
            row.direction,
        )
        groups[key].append(row)

    output: list[AggregateRow] = []
    case_trials = _case_trials(metric_rows)
    for key in sorted(groups):
        group = groups[key]
        _reject_duplicate_trials(group)
        _require_all_case_trials(group, case_trials[key[0]])
        stats = _describe([row.value for row in group])
        case_id, cell_label, stage, selection, metric, unit, direction = key
        output.append(
            AggregateRow(
                source="metric",
                case_id=case_id,
                cell_label=cell_label,
                contrast_name=None,
                stage=stage,
                selection=selection,
                metric=metric,
                unit=unit,
                direction=direction,
                **stats,
            )
        )
    return tuple(output)


def aggregate_contrast_rows(rows: Sequence[ContrastRow]) -> tuple[AggregateRow, ...]:
    """Aggregate already-paired raw contrast values across trials within each case."""

    contrast_rows = _validate_contrast_rows(rows)
    groups: dict[tuple[str, str, str, str, str, str, MetricDirection], list[ContrastRow]] = (
        defaultdict(list)
    )
    for row in contrast_rows:
        key = (
            row.case_id,
            row.contrast_name,
            row.stage,
            row.selection,
            row.metric,
            row.unit,
            row.direction,
        )
        groups[key].append(row)

    definition_order = {
        definition.name: index for index, definition in enumerate(CONTRAST_DEFINITIONS)
    }
    ordered_keys = sorted(groups, key=lambda key: (key[0], definition_order[key[1]], *key[2:]))
    output: list[AggregateRow] = []
    case_trials = _case_trials(contrast_rows)
    for key in ordered_keys:
        group = groups[key]
        _reject_duplicate_trials(group)
        _require_all_case_trials(group, case_trials[key[0]])
        stats = _describe([row.raw_value for row in group])
        case_id, contrast_name, stage, selection, metric, unit, direction = key
        output.append(
            AggregateRow(
                source="contrast",
                case_id=case_id,
                cell_label=None,
                contrast_name=contrast_name,
                stage=stage,
                selection=selection,
                metric=metric,
                unit=unit,
                direction=direction,
                **stats,
            )
        )
    return tuple(output)


def _validate_metric_rows(rows: Sequence[MetricRow]) -> tuple[MetricRow, ...]:
    try:
        normalized = tuple(rows)
    except TypeError as error:
        raise ValueError("rows must be a sequence of MetricRow values") from error
    if any(not isinstance(row, MetricRow) for row in normalized):
        raise ValueError("rows must contain only MetricRow values")
    unknown = {row.cell_label for row in normalized} - _CANONICAL_CELLS
    if unknown:
        raise ValueError(f"unknown cell label(s): {','.join(sorted(unknown))}")

    identities: set[tuple[Any, ...]] = set()
    trial_seeds: dict[tuple[str, str], int | None] = {}
    seed_trials: dict[tuple[str, int | None], str] = {}
    semantics: dict[tuple[str, str, str, str, str], tuple[str, MetricDirection]] = {}
    for row in normalized:
        seed = _validated_seed(row.seed)
        identity = (
            row.case_id,
            row.trial_id,
            row.seed,
            row.cell_label,
            row.stage,
            row.selection,
            row.metric,
        )
        if identity in identities:
            raise ValueError("duplicate metric row identity")
        identities.add(identity)
        trial_key = (row.case_id, row.trial_id)
        if trial_key in trial_seeds and trial_seeds[trial_key] != seed:
            raise ValueError("trial_id has inconsistent seed values")
        trial_seeds[trial_key] = seed
        seed_key = (row.case_id, seed)
        if seed_key in seed_trials and seed_trials[seed_key] != row.trial_id:
            raise ValueError("duplicate trial seed maps to multiple trial IDs")
        seed_trials[seed_key] = row.trial_id
        semantic_key = (row.case_id, row.cell_label, row.stage, row.selection, row.metric)
        semantic = (row.unit, row.direction)
        if semantic_key in semantics and semantics[semantic_key] != semantic:
            raise ValueError("metric aggregate identity has mixed units or directions")
        semantics[semantic_key] = semantic
    return normalized


def _validate_contrast_rows(rows: Sequence[ContrastRow]) -> tuple[ContrastRow, ...]:
    try:
        normalized = tuple(rows)
    except TypeError as error:
        raise ValueError("rows must be a sequence of ContrastRow values") from error
    if any(not isinstance(row, ContrastRow) for row in normalized):
        raise ValueError("rows must contain only ContrastRow values")
    trial_seeds: dict[tuple[str, str], int | None] = {}
    seed_trials: dict[tuple[str, int | None], str] = {}
    identities: set[tuple[Any, ...]] = set()
    semantics: dict[tuple[str, str, str, str, str], tuple[str, MetricDirection]] = {}
    for row in normalized:
        seed = _validated_seed(row.seed)
        identity = (
            row.case_id,
            row.trial_id,
            row.seed,
            row.contrast_name,
            row.stage,
            row.selection,
            row.metric,
        )
        if identity in identities:
            raise ValueError("duplicate contrast row identity")
        identities.add(identity)
        trial_key = (row.case_id, row.trial_id)
        if trial_key in trial_seeds and trial_seeds[trial_key] != seed:
            raise ValueError("trial_id has inconsistent seed values")
        trial_seeds[trial_key] = seed
        seed_key = (row.case_id, seed)
        if seed_key in seed_trials and seed_trials[seed_key] != row.trial_id:
            raise ValueError("duplicate trial seed maps to multiple trial IDs")
        seed_trials[seed_key] = row.trial_id
        semantic_key = (row.case_id, row.contrast_name, row.stage, row.selection, row.metric)
        semantic = (row.unit, row.direction)
        if semantic_key in semantics and semantics[semantic_key] != semantic:
            raise ValueError("contrast aggregate identity has mixed units or directions")
        semantics[semantic_key] = semantic
    return normalized


def _reject_duplicate_trials(rows: Sequence[MetricRow | ContrastRow]) -> None:
    trials = [(row.trial_id, row.seed) for row in rows]
    if len(trials) != len(set(trials)):
        raise ValueError("duplicate trial in aggregate group")


def _case_trials(
    rows: Sequence[MetricRow | ContrastRow],
) -> dict[str, tuple[tuple[str, int | None], ...]]:
    by_case: dict[str, set[tuple[str, int | None]]] = defaultdict(set)
    for row in rows:
        by_case[row.case_id].add((row.trial_id, row.seed))
    return {
        case_id: tuple(
            sorted(trials, key=lambda item: (item[0], -1 if item[1] is None else item[1]))
        )
        for case_id, trials in by_case.items()
    }


def _require_all_case_trials(
    rows: Sequence[MetricRow | ContrastRow], expected: Sequence[tuple[str, int | None]]
) -> None:
    actual = {(row.trial_id, row.seed) for row in rows}
    missing = set(expected) - actual
    if missing:
        trial_ids = ",".join(sorted(trial_id for trial_id, _seed in missing))
        raise ValueError(f"aggregate group is missing completed trial(s): {trial_ids}")


def _describe(values: Sequence[float]) -> dict[str, int | float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("aggregate values must be a non-empty finite sequence")
    std = 0.0 if array.size == 1 else float(np.std(array, ddof=1))
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": std,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "q25": float(np.quantile(array, 0.25, method="linear")),
        "q75": float(np.quantile(array, 0.75, method="linear")),
    }


def _finite_number(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _validated_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be None or a non-negative integer")
    return int(seed)


def _improvement(raw_value: float, direction: MetricDirection) -> float | None:
    if direction == "higher":
        return raw_value
    if direction == "lower":
        return -raw_value
    return None


def _definition_by_name(name: str) -> ContrastDefinition:
    return next(definition for definition in CONTRAST_DEFINITIONS if definition.name == name)


def _sortable_pair_key(key: _MetricPairKey) -> tuple[Any, ...]:
    case_id, trial_id, seed, stage, selection, metric = key
    metric_identity = (stage, selection, metric)
    return (
        case_id,
        trial_id,
        -1 if seed is None else seed,
        _METRIC_ORDER.get(metric_identity, len(_METRIC_ORDER)),
        metric_identity,
    )
