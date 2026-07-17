"""Sequential orchestration for canonical synthetic mode comparisons."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from time import perf_counter
from typing import Any

import numpy as np

from ..reporting.models import freeze_report_value
from .builder import build_mode_comparison_plan
from .config import SyntheticModeComparisonConfig
from .contrasts import (
    AggregateRow,
    ContrastRow,
    aggregate_contrast_rows,
    aggregate_metric_rows,
    compute_contrast_rows,
)
from .metrics import MetricRow, extract_trial_metric_rows
from .models import SyntheticModeComparisonPlan
from .runner import SyntheticTrialEvaluation, TrialRuntimeRecorder, run_synthetic_trial
from .trials import SyntheticTrialSpec

Clock = Callable[[], float]
TrialRunner = Callable[..., SyntheticTrialEvaluation]
MetricExtractor = Callable[[SyntheticTrialEvaluation], Sequence[MetricRow]]
ContrastBuilder = Callable[[Sequence[MetricRow]], Sequence[ContrastRow]]
MetricAggregator = Callable[[Sequence[MetricRow]], Sequence[AggregateRow]]
ContrastAggregator = Callable[[Sequence[ContrastRow]], Sequence[AggregateRow]]


class _MonotonicClock:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._previous: float | None = None

    def __call__(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("clock must return a finite number")
        normalized = float(value)
        if not np.isfinite(normalized):
            raise ValueError("clock must return a finite number")
        if self._previous is not None and normalized < self._previous:
            raise ValueError("clock moved backwards")
        self._previous = normalized
        return normalized


@dataclass(frozen=True, slots=True)
class RuntimeRow:
    """One validated runtime observation for an experiment stage."""

    case_id: str | None
    trial_id: str | None
    seed: int | None
    stage: str
    cell_label: str | None
    scanner_backend: str | None
    elapsed_seconds: float
    call_count: int
    shared_stage: bool

    def __post_init__(self) -> None:
        if (self.case_id is None) != (self.trial_id is None):
            raise ValueError("case_id and trial_id must either both be set or both be None")
        for name in ("case_id", "trial_id", "cell_label", "scanner_backend"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be None or a non-empty string")
        if not isinstance(self.stage, str) or not self.stage:
            raise ValueError("stage must be a non-empty string")
        if self.seed is not None:
            if not isinstance(self.seed, Integral) or isinstance(self.seed, bool) or self.seed < 0:
                raise ValueError("seed must be None or a non-negative integer")
            object.__setattr__(self, "seed", int(self.seed))
        if isinstance(self.elapsed_seconds, bool) or not isinstance(self.elapsed_seconds, Real):
            raise ValueError("elapsed_seconds must be a finite non-negative number")
        elapsed = float(self.elapsed_seconds)
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("elapsed_seconds must be a finite non-negative number")
        object.__setattr__(self, "elapsed_seconds", elapsed)
        if (
            not isinstance(self.call_count, Integral)
            or isinstance(self.call_count, bool)
            or self.call_count < 0
        ):
            raise ValueError("call_count must be a non-negative integer")
        object.__setattr__(self, "call_count", int(self.call_count))
        if not isinstance(self.shared_stage, bool):
            raise ValueError("shared_stage must be bool")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping in canonical field order."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class SyntheticModeComparisonResult:
    """Scalar-only result of one fully completed mode-comparison experiment."""

    plan_metadata: Mapping[str, Any]
    trial_metadata: tuple[Mapping[str, Any], ...]
    cell_reports: tuple[Mapping[str, Any], ...]
    metric_rows: tuple[MetricRow, ...]
    contrast_rows: tuple[ContrastRow, ...]
    metric_aggregates: tuple[AggregateRow, ...]
    contrast_aggregates: tuple[AggregateRow, ...]
    cache_stats: tuple[Mapping[str, Any], ...]
    runtime_rows: tuple[RuntimeRow, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "plan_metadata",
            freeze_report_value(_json_safe(self.plan_metadata)),
        )
        for name in ("trial_metadata", "cell_reports", "cache_stats"):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, Mapping) for value in values
            ):
                raise ValueError(f"{name} must be a tuple of mappings")
            object.__setattr__(
                self,
                name,
                tuple(freeze_report_value(_json_safe(value)) for value in values),
            )
        for name, item_type in (
            ("metric_rows", MetricRow),
            ("contrast_rows", ContrastRow),
            ("metric_aggregates", AggregateRow),
            ("contrast_aggregates", AggregateRow),
            ("runtime_rows", RuntimeRow),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(
                not isinstance(value, item_type) for value in values
            ):
                raise ValueError(f"{name} must be a tuple of {item_type.__name__} values")
        if not isinstance(self.plan_metadata, Mapping):
            raise ValueError("plan_metadata must be a mapping")

    def as_dict(self) -> dict[str, Any]:
        """Serialize the complete scalar evidence in deterministic field order."""

        return {
            "plan_metadata": _json_safe(self.plan_metadata),
            "trial_metadata": _json_safe(self.trial_metadata),
            "cell_reports": _json_safe(self.cell_reports),
            "metric_rows": [row.as_dict() for row in self.metric_rows],
            "contrast_rows": [row.as_dict() for row in self.contrast_rows],
            "metric_aggregates": [row.as_dict() for row in self.metric_aggregates],
            "contrast_aggregates": [row.as_dict() for row in self.contrast_aggregates],
            "cache_stats": _json_safe(self.cache_stats),
            "runtime_rows": [row.as_dict() for row in self.runtime_rows],
        }


class _ExperimentRuntimeRecorder(TrialRuntimeRecorder):
    def __init__(self, trial: SyntheticTrialSpec, output: list[RuntimeRow]) -> None:
        self._trial = trial
        self._output = output

    def record(
        self,
        *,
        stage: str,
        elapsed_seconds: float,
        cell_label: str | None = None,
        scanner_backend: str | None = None,
        call_count: int = 1,
        shared_stage: bool,
    ) -> None:
        self._output.append(
            RuntimeRow(
                case_id=self._trial.case_id,
                trial_id=self._trial.trial_id,
                seed=self._trial.seed,
                stage=stage,
                cell_label=cell_label,
                scanner_backend=scanner_backend,
                elapsed_seconds=elapsed_seconds,
                call_count=call_count,
                shared_stage=shared_stage,
            )
        )


def run_mode_comparison(
    config: SyntheticModeComparisonConfig,
    *,
    clock: Clock = perf_counter,
    trial_runner: TrialRunner = run_synthetic_trial,
    metric_extractor: MetricExtractor = extract_trial_metric_rows,
    contrast_builder: ContrastBuilder = compute_contrast_rows,
    metric_aggregator: MetricAggregator = aggregate_metric_rows,
    contrast_aggregator: ContrastAggregator = aggregate_contrast_rows,
) -> SyntheticModeComparisonResult:
    """Run every planned trial sequentially and return scalar-only evidence."""

    monitored_clock = _MonotonicClock(clock)
    experiment_start = _clock_value(monitored_clock, "experiment_total")
    plan = build_mode_comparison_plan(config)
    runtime_rows: list[RuntimeRow] = []
    trial_metadata: list[Mapping[str, Any]] = []
    cell_reports: list[Mapping[str, Any]] = []
    cache_stats: list[Mapping[str, Any]] = []
    metric_rows: list[MetricRow] = []
    contrast_rows: list[ContrastRow] = []
    for trial in plan.trials:
        recorder = _ExperimentRuntimeRecorder(trial, runtime_rows)
        trial_start = _clock_value(monitored_clock, "trial_total")
        evaluation: SyntheticTrialEvaluation | None = None
        try:
            evaluation = trial_runner(
                plan,
                trial,
                clock=monitored_clock,
                runtime_recorder=recorder,
            )
            report = _cell_report(evaluation, plan, trial)
            trial_metrics = _timed_processing(
                "metric_extraction",
                lambda: tuple(metric_extractor(evaluation)),
                clock=monitored_clock,
                recorder=recorder,
            )
            trial_contrasts = _timed_processing(
                "contrast_extraction",
                lambda: tuple(contrast_builder(trial_metrics)),
                clock=monitored_clock,
                recorder=recorder,
            )
            trial_cache = _cache_stats(evaluation, trial)
            trial_end = _clock_value(monitored_clock, "trial_total")
            if trial_end < trial_start:
                raise ValueError("clock moved backwards while timing 'trial_total'")
            recorder.record(
                stage="trial_total",
                elapsed_seconds=trial_end - trial_start,
                shared_stage=True,
            )

            trial_metadata.append(_trial_metadata(trial))
            cell_reports.append(report)
            cache_stats.append(trial_cache)
            metric_rows.extend(trial_metrics)
            contrast_rows.extend(trial_contrasts)
        finally:
            evaluation = None

    metric_aggregates = tuple(metric_aggregator(tuple(metric_rows)))
    contrast_aggregates = tuple(contrast_aggregator(tuple(contrast_rows)))
    experiment_end = _clock_value(monitored_clock, "experiment_total")
    if experiment_end < experiment_start:
        raise ValueError("clock moved backwards while timing 'experiment_total'")
    runtime_rows.append(
        RuntimeRow(
            case_id=None,
            trial_id=None,
            seed=None,
            stage="experiment_total",
            cell_label=None,
            scanner_backend=None,
            elapsed_seconds=experiment_end - experiment_start,
            call_count=1,
            shared_stage=True,
        )
    )
    return SyntheticModeComparisonResult(
        plan_metadata=_plan_metadata(plan),
        trial_metadata=tuple(trial_metadata),
        cell_reports=tuple(cell_reports),
        metric_rows=tuple(metric_rows),
        contrast_rows=tuple(contrast_rows),
        metric_aggregates=metric_aggregates,
        contrast_aggregates=contrast_aggregates,
        cache_stats=tuple(cache_stats),
        runtime_rows=tuple(runtime_rows),
    )


def _timed_processing(
    stage: str,
    operation: Callable[[], Any],
    *,
    clock: Clock,
    recorder: _ExperimentRuntimeRecorder,
) -> Any:
    start = _clock_value(clock, stage)
    result = operation()
    end = _clock_value(clock, stage)
    if end < start:
        raise ValueError(f"clock moved backwards while timing {stage!r}")
    recorder.record(
        stage=stage,
        elapsed_seconds=end - start,
        shared_stage=True,
    )
    return result


def _clock_value(clock: Clock, stage: str) -> float:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"clock must return a finite number while timing {stage!r}")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"clock must return a finite number while timing {stage!r}")
    return normalized


def _plan_metadata(plan: SyntheticModeComparisonPlan) -> Mapping[str, Any]:
    return _json_safe(asdict(plan))


def _trial_metadata(trial: SyntheticTrialSpec) -> Mapping[str, Any]:
    return _json_safe(asdict(trial))


def _cell_report(
    evaluation: SyntheticTrialEvaluation,
    plan: SyntheticModeComparisonPlan,
    trial: SyntheticTrialSpec,
) -> Mapping[str, Any]:
    if evaluation.trial != trial:
        raise ValueError("trial runner returned evaluation metadata for a different trial")
    expected_labels = tuple(cell.label for cell in plan.cells)
    if tuple(evaluation.report_payload) != expected_labels:
        raise ValueError("trial runner returned cell reports outside canonical plan order")
    return {
        "case_id": trial.case_id,
        "trial_id": trial.trial_id,
        "seed": trial.seed,
        "cells": _json_safe(evaluation.report_payload),
    }


def _cache_stats(
    evaluation: SyntheticTrialEvaluation, trial: SyntheticTrialSpec
) -> Mapping[str, Any]:
    return {
        "case_id": trial.case_id,
        "trial_id": trial.trial_id,
        "seed": trial.seed,
        **_json_safe(asdict(evaluation.stage_cache_stats)),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("JSON-safe experiment values must be finite")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        raise ValueError("scalar-only experiment results cannot contain NumPy arrays")
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("JSON-safe experiment mappings require string keys")
            output[key] = _json_safe(item)
        return output
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    raise ValueError(f"experiment result contains a non-JSON-safe {type(value).__name__}")
