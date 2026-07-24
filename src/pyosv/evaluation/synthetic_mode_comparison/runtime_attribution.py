"""Pure semantic-key planning for cacheable runtime attribution."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any

from ..synthetic_quality.stage_cache import SCALAR_EVIDENCE_CONTRACT_VERSION
from ..synthetic_quality.stage_keys import (
    PipelineStageKeys,
    build_final_thinning_stage_key,
    build_oracle_attribute_stage_key,
    build_primary_skinning_stage_key,
    build_scanner_attribute_stage_key,
    build_seed_stage_key,
    build_thinning_scalar_evidence_key,
    build_thinning_stage_key,
    build_voting_scalar_evidence_key,
    build_voting_stage_key,
)
from ..synthetic_quality.variants import effective_skinning_config, get_variant_spec
from ..workflow3d import VolumeVotingControls
from .models import SCANNER_ONLY_SCOPE, ModeCellSpec, SyntheticModeComparisonPlan
from .trials import SyntheticTrialSpec

CACHEABLE_RUNTIME_STAGES = (
    "seed_selection",
    "voting_volume",
    "base_thinning",
    "primary_skinning",
    "voting_scalar_evidence",
    "thinning_scalar_evidence",
)


@dataclass(frozen=True, slots=True)
class RuntimeStageKeys:
    """All cacheable semantic keys looked up by one canonical cell."""

    seed_selection: Any | None
    voting_volume: Any | None
    base_thinning: Any | None
    primary_skinning: Any | None
    voting_scalar_evidence: Any | None
    thinning_scalar_evidence: Any | None

    def key_for(self, stage: str) -> Any | None:
        if stage not in CACHEABLE_RUNTIME_STAGES:
            raise ValueError(f"unsupported cacheable runtime stage {stage!r}")
        return getattr(self, stage)


@dataclass(frozen=True, slots=True)
class RuntimeStageAttribution:
    """Reference-count attribution for one canonical cell and stage."""

    cell_label: str
    scanner_backend: str | None
    stage: str
    semantic_key: Any | None
    reference_count: int

    @property
    def shared(self) -> bool:
        return self.semantic_key is not None and self.reference_count >= 2

    @property
    def cell_owned(self) -> bool:
        return self.semantic_key is not None and self.reference_count == 1


@dataclass(frozen=True, slots=True)
class RuntimeAttributionPlan:
    """Canonical runtime ownership derived without executing pipeline stages."""

    entries: tuple[RuntimeStageAttribution, ...]

    def attribution_for(self, cell_label: str, stage: str) -> RuntimeStageAttribution:
        for entry in self.entries:
            if entry.cell_label == cell_label and entry.stage == stage:
                return entry
        raise ValueError(f"runtime attribution is missing cell {cell_label!r} stage {stage!r}")

    def shared_keys(self, stage: str) -> tuple[Any, ...]:
        keys: list[Any] = []
        for entry in self.entries:
            if entry.stage == stage and entry.shared and entry.semantic_key not in keys:
                keys.append(entry.semantic_key)
        return tuple(keys)

    def cell_owned_entries(self) -> tuple[RuntimeStageAttribution, ...]:
        return tuple(entry for entry in self.entries if entry.cell_owned)


def resolved_stage_keys_for_cell(
    plan: SyntheticModeComparisonPlan,
    cell: ModeCellSpec,
    trial: SyntheticTrialSpec | None = None,
) -> PipelineStageKeys:
    """Derive the volume-stage cache keys looked up by one canonical cell."""

    if cell.scope == SCANNER_ONLY_SCOPE:
        return PipelineStageKeys(None, None, None, None, None)

    trial = plan.trials[0] if trial is None else trial
    if cell.input_mode == "oracle":
        attribute_key = build_oracle_attribute_stage_key(
            case_id=trial.case_id,
            shape=trial.shape,
        )
        target_source = "oracle_ft"
    elif cell.input_mode == "scanner" and cell.scanner_backend is not None:
        scanner_config = replace(plan.scanner_template, backend=cell.scanner_backend)
        attribute_key = build_scanner_attribute_stage_key(
            case_id=trial.case_id,
            shape=trial.shape,
            scanner_config=scanner_config,
        )
        target_source = "scanner_fet"
    else:
        raise ValueError("downstream cell must have a canonical attribute source")

    settings = _workflow_settings(plan, cell)
    variant_spec = get_variant_spec(plan.comparison_variant)
    skinning_config = effective_skinning_config(variant_spec, settings.skinning_config)
    seed_key = build_seed_stage_key(
        attribute_key=attribute_key,
        voting_config=settings.voting_config,
        variant_spec=variant_spec,
        target_source=target_source,
    )
    voting_key = build_voting_stage_key(
        seed_key=seed_key,
        voting_config=settings.voting_config,
        variant_spec=variant_spec,
        voting_controls=VolumeVotingControls.resolve(
            settings.voting_config,
            variant_spec,
        ),
    )
    thinning_key = build_thinning_stage_key(
        voting_key=voting_key,
        voting_config=settings.voting_config,
        variant_spec=variant_spec,
    )
    primary_skinning_key = build_primary_skinning_stage_key(
        thinning_key=thinning_key,
        skinning_config=skinning_config,
        variant_spec=variant_spec,
        target_source=target_source,
    )
    return PipelineStageKeys(
        attribute=attribute_key,
        seed=seed_key,
        voting=voting_key,
        thinning=thinning_key,
        primary_skinning=primary_skinning_key,
    )


def resolve_runtime_stage_keys_for_cell(
    plan: SyntheticModeComparisonPlan,
    cell: ModeCellSpec,
    trial: SyntheticTrialSpec,
    *,
    case_token: int = 0,
) -> RuntimeStageKeys:
    """Derive all six cacheable runtime keys for one cell."""

    keys = resolved_stage_keys_for_cell(plan, cell, trial)
    if cell.scope == SCANNER_ONLY_SCOPE:
        return RuntimeStageKeys(None, None, None, None, None, None)
    target_source = "oracle_ft" if cell.input_mode == "oracle" else "scanner_fet"
    variant_spec = get_variant_spec(plan.comparison_variant)
    final_thinning_key = build_final_thinning_stage_key(
        thinning_key=keys.thinning,
        variant_spec=variant_spec,
        target_source=target_source,
    )
    return RuntimeStageKeys(
        seed_selection=keys.seed,
        voting_volume=keys.voting,
        base_thinning=keys.thinning,
        primary_skinning=keys.primary_skinning,
        voting_scalar_evidence=build_voting_scalar_evidence_key(
            case_id=trial.case_id,
            case_token=case_token,
            shape=trial.shape,
            voting_key=keys.voting,
            truth_metric_config=plan.truth_metric_config,
            contract_version=SCALAR_EVIDENCE_CONTRACT_VERSION,
        ),
        thinning_scalar_evidence=build_thinning_scalar_evidence_key(
            case_id=trial.case_id,
            case_token=case_token,
            shape=trial.shape,
            final_thinning_key=final_thinning_key,
            truth_metric_config=plan.truth_metric_config,
            contract_version=SCALAR_EVIDENCE_CONTRACT_VERSION,
        ),
    )


def build_runtime_attribution_plan(
    plan: SyntheticModeComparisonPlan,
    trial: SyntheticTrialSpec,
    *,
    case_token: int = 0,
) -> RuntimeAttributionPlan:
    """Count canonical semantic-key references and assign runtime ownership."""

    keys_by_cell = {
        cell.label: resolve_runtime_stage_keys_for_cell(
            plan,
            cell,
            trial,
            case_token=case_token,
        )
        for cell in plan.cells
    }
    counts = {
        stage: Counter(
            key for keys in keys_by_cell.values() if (key := keys.key_for(stage)) is not None
        )
        for stage in CACHEABLE_RUNTIME_STAGES
    }
    return RuntimeAttributionPlan(
        entries=tuple(
            RuntimeStageAttribution(
                cell_label=cell.label,
                scanner_backend=cell.scanner_backend,
                stage=stage,
                semantic_key=(key := keys_by_cell[cell.label].key_for(stage)),
                reference_count=0 if key is None else counts[stage][key],
            )
            for cell in plan.cells
            for stage in CACHEABLE_RUNTIME_STAGES
        )
    )


def _workflow_settings(plan: SyntheticModeComparisonPlan, cell: ModeCellSpec):
    if cell.workflow_mode == "reference":
        return plan.reference_workflow_settings
    if cell.workflow_mode == "quality":
        return plan.quality_workflow_settings
    raise ValueError("downstream cell must have a canonical workflow mode")
