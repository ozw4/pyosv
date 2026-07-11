"""Typed, immutable structure for synthetic-quality reports."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


class _FrozenList(tuple[Any, ...]):
    """Tuple storage that remembers that its wire representation is a list."""


def freeze_report_value(value: Any) -> Any:
    """Copy a JSON-like value into recursively read-only storage."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_report_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(freeze_report_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(freeze_report_value(item) for item in value)
    return value


def thaw_report_value(value: Any) -> Any:
    """Return a detached value with the original dict/list/tuple representation."""

    if isinstance(value, Mapping):
        return {key: thaw_report_value(item) for key, item in value.items()}
    if isinstance(value, _FrozenList):
        return [thaw_report_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(thaw_report_value(item) for item in value)
    return value


def _frozen_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return freeze_report_value(value)


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Reference to a report artifact without prescribing an artifact schema."""

    path: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class VariantComparison:
    """Comparison payload and its explicitly typed baseline relationship."""

    baseline_variant: str | None
    variants: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "variants", _frozen_mapping(self.variants))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VariantComparison:
        return cls(payload.get("baseline_variant"), payload.get("variants", {}))


@dataclass(frozen=True, slots=True)
class VariantReport:
    """One variant's metrics, with optional per-input pipeline reports."""

    metrics: Mapping[str, Any]
    pipelines: Mapping[str, Mapping[str, Any]]
    active_pipeline: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metrics", _frozen_mapping(self.metrics))
        object.__setattr__(self, "pipelines", _frozen_mapping(self.pipelines))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VariantReport:
        return cls(
            metrics={key: value for key, value in payload.items() if key != "pipelines"},
            pipelines=payload.get("pipelines", {}),
            active_pipeline=payload.get("active_pipeline"),
        )

    def to_dict(self) -> dict[str, Any]:
        result = thaw_report_value(self.metrics)
        if self.pipelines:
            result["pipelines"] = thaw_report_value(self.pipelines)
        return result


@dataclass(frozen=True, slots=True)
class PipelineReport:
    """Named variants and their comparison within one input pipeline."""

    variants: Mapping[str, VariantReport]
    variant_comparison: VariantComparison

    def __post_init__(self) -> None:
        object.__setattr__(self, "variants", MappingProxyType(dict(self.variants)))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PipelineReport:
        return cls(
            variants={
                name: VariantReport.from_dict(report)
                for name, report in payload["variants"].items()
            },
            variant_comparison=VariantComparison.from_dict(payload["variant_comparison"]),
        )


@dataclass(frozen=True, slots=True)
class CaseReport:
    """Typed case identity, variants, and pipeline relationships."""

    case_id: str
    shape: tuple[int, int, int]
    truth: Mapping[str, Any]
    variants: Mapping[str, VariantReport]
    pipelines: Mapping[str, PipelineReport]
    variant_comparison: Mapping[str, Any]
    aliases: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "truth", _frozen_mapping(self.truth))
        object.__setattr__(self, "variants", MappingProxyType(dict(self.variants)))
        object.__setattr__(self, "pipelines", MappingProxyType(dict(self.pipelines)))
        object.__setattr__(self, "variant_comparison", _frozen_mapping(self.variant_comparison))
        object.__setattr__(self, "aliases", _frozen_mapping(self.aliases))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CaseReport:
        structural = {"case_id", "shape", "truth", "variants", "pipelines", "variant_comparison"}
        return cls(
            case_id=payload["case_id"],
            shape=tuple(payload["shape"]),
            truth=payload["truth"],
            variants={
                name: VariantReport.from_dict(report)
                for name, report in payload["variants"].items()
            },
            pipelines={
                name: PipelineReport.from_dict(report)
                for name, report in payload["pipelines"].items()
            },
            variant_comparison=payload["variant_comparison"],
            aliases={key: value for key, value in payload.items() if key not in structural},
        )


@dataclass(frozen=True, slots=True)
class ReportConfig:
    """Read-only report configuration with stable legacy keys."""

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", _frozen_mapping(self.values))


@dataclass(frozen=True, slots=True)
class Report:
    """Top-level typed report model."""

    config: ReportConfig
    cases: tuple[CaseReport, ...]
    format_version: int = 1
    artifact_references: tuple[ArtifactReference, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Report:
        return cls(
            format_version=payload["format_version"],
            config=ReportConfig(payload["config"]),
            cases=tuple(CaseReport.from_dict(case) for case in payload["cases"]),
        )
