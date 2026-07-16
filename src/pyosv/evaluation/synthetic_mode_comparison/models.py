"""Immutable models for canonical synthetic mode-comparison plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pyosv.synthetic3d import validate_shape3

from ..synthetic_quality import (
    ResolvedWorkflowSettings,
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
    resolve_workflow_settings,
)
from ..synthetic_quality.cases import EXTENDED_CASES, validate_case_ids
from .trials import SyntheticTrialSpec, expand_synthetic_trials, validate_trial_seeds

ModeCellScope = Literal["scanner-only", "oracle-workflow-isolation", "end-to-end"]
ModeInputMode = Literal["scanner", "oracle"]
CanonicalScannerBackend = Literal["reference-like", "quality"]
CanonicalWorkflowMode = Literal["reference", "quality"]

SCANNER_ONLY_SCOPE = "scanner-only"
ORACLE_WORKFLOW_ISOLATION_SCOPE = "oracle-workflow-isolation"
END_TO_END_SCOPE = "end-to-end"


@dataclass(frozen=True, slots=True)
class ModeCellSpec:
    """One cell in the canonical scanner-backend/workflow comparison."""

    label: str
    scope: ModeCellScope
    input_mode: ModeInputMode
    scanner_backend: CanonicalScannerBackend | None
    workflow_mode: CanonicalWorkflowMode | None

    def __post_init__(self) -> None:
        if self.scope not in {
            SCANNER_ONLY_SCOPE,
            ORACLE_WORKFLOW_ISOLATION_SCOPE,
            END_TO_END_SCOPE,
        }:
            raise ValueError(f"unknown mode cell scope: {self.scope}")
        if self.input_mode not in {"scanner", "oracle"}:
            raise ValueError(f"unknown mode cell input_mode: {self.input_mode}")
        if self.scanner_backend not in {None, "reference-like", "quality"}:
            raise ValueError(f"unknown scanner backend: {self.scanner_backend}")
        if self.workflow_mode not in {None, "reference", "quality"}:
            raise ValueError(f"unknown workflow mode: {self.workflow_mode}")

        expected_label = _label_for_axes(
            scope=self.scope,
            input_mode=self.input_mode,
            scanner_backend=self.scanner_backend,
            workflow_mode=self.workflow_mode,
        )
        if self.label != expected_label:
            raise ValueError(
                f"mode cell label {self.label!r} is inconsistent with its scope and axes; "
                f"expected {expected_label!r}"
            )


def _label_for_axes(
    *,
    scope: str,
    input_mode: str,
    scanner_backend: str | None,
    workflow_mode: str | None,
) -> str:
    if scope == SCANNER_ONLY_SCOPE:
        if input_mode != "scanner" or scanner_backend is None or workflow_mode is not None:
            raise ValueError(
                "scanner-only cells require input_mode='scanner', a scanner backend, "
                "and workflow_mode=None"
            )
        return "RL-SCAN" if scanner_backend == "reference-like" else "Q-SCAN"
    if scope == ORACLE_WORKFLOW_ISOLATION_SCOPE:
        if input_mode != "oracle" or scanner_backend is not None or workflow_mode is None:
            raise ValueError(
                "oracle-workflow-isolation cells require input_mode='oracle', "
                "scanner_backend=None, and a workflow mode"
            )
        return "ORACLE-REF" if workflow_mode == "reference" else "ORACLE-QUAL"
    if input_mode != "scanner" or scanner_backend is None or workflow_mode is None:
        raise ValueError(
            "end-to-end cells require input_mode='scanner', a scanner backend, and a workflow mode"
        )
    backend_label = "RL" if scanner_backend == "reference-like" else "Q"
    workflow_label = "REF" if workflow_mode == "reference" else "QUAL"
    return f"{backend_label}-{workflow_label}"


@dataclass(frozen=True, slots=True)
class SyntheticModeComparisonPlan:
    """Validated, execution-free canonical synthetic comparison plan."""

    case_ids: tuple[str, ...]
    trial_seeds: tuple[int, ...]
    trials: tuple[SyntheticTrialSpec, ...]
    shape: tuple[int, int, int]
    scanner_template: SyntheticScannerConfig
    voting_config: SyntheticVotingConfig | None
    skinning_config: SyntheticSkinningConfig
    truth_metric_config: SyntheticTruthMetricConfig
    include_oracle_workflow_isolation: bool
    comparison_variant: str
    skinner_method_explicit: bool
    skinner_min_likelihood_explicit: bool
    skinner_growth_source_explicit: bool
    skinner_accepted_occupancy_radius_explicit: bool
    skinner_boundary_fallback_explicit: bool
    reference_workflow_settings: ResolvedWorkflowSettings
    quality_workflow_settings: ResolvedWorkflowSettings
    cells: tuple[ModeCellSpec, ...]

    def __post_init__(self) -> None:
        validate_case_ids(self.case_ids)
        seeds = validate_trial_seeds(self.trial_seeds)
        definitions_by_id = {definition.case_id: definition for definition in EXTENDED_CASES}
        expected_trials = expand_synthetic_trials(
            tuple(definitions_by_id[case_id] for case_id in self.case_ids),
            seeds,
        )
        if self.trials != expected_trials:
            raise ValueError("trials must match the selected cases and trial seeds")
        validate_shape3(self.shape)
        if self.comparison_variant != "current_default":
            raise ValueError("comparison_variant must be 'current_default'")
        if self.scanner_template.backend != "reference-like":
            raise ValueError("scanner_template.backend must be 'reference-like'")
        if self.scanner_template.scanner_thin_mode != "reference":
            raise ValueError("scanner_template.scanner_thin_mode must be 'reference'")
        if not self.scanner_template.remove_edge_effects:
            raise ValueError("scanner_template.remove_edge_effects must be True")
        if self.scanner_template.refinement_factor != 2:
            raise ValueError("scanner_template.refinement_factor must be 2")
        if self.reference_workflow_settings.workflow_mode != "reference":
            raise ValueError("reference_workflow_settings must use the reference workflow")
        if self.quality_workflow_settings.workflow_mode != "quality":
            raise ValueError("quality_workflow_settings must use the quality workflow")
        workflow_kwargs = {
            "voting_config": self.voting_config,
            "skinning_config": self.skinning_config,
            "skinner_method_explicit": self.skinner_method_explicit,
            "skinner_min_likelihood_explicit": self.skinner_min_likelihood_explicit,
            "skinner_growth_source_explicit": self.skinner_growth_source_explicit,
            "skinner_accepted_occupancy_radius_explicit": (
                self.skinner_accepted_occupancy_radius_explicit
            ),
            "skinner_boundary_fallback_explicit": self.skinner_boundary_fallback_explicit,
        }
        expected_reference_settings = resolve_workflow_settings(
            workflow_mode="reference", **workflow_kwargs
        )
        expected_quality_settings = resolve_workflow_settings(
            workflow_mode="quality", **workflow_kwargs
        )
        if self.reference_workflow_settings != expected_reference_settings:
            raise ValueError(
                "reference_workflow_settings must match the plan workflow configuration"
            )
        if self.quality_workflow_settings != expected_quality_settings:
            raise ValueError("quality_workflow_settings must match the plan workflow configuration")
        if (
            self.reference_workflow_settings.include_thinning_diagnostic
            or self.quality_workflow_settings.include_thinning_diagnostic
        ):
            raise ValueError("diagnostic workflow settings are not canonical comparison cells")
        labels = tuple(cell.label for cell in self.cells)
        duplicates = {label for label in labels if labels.count(label) > 1}
        if duplicates:
            raise ValueError(f"duplicate mode cell label(s): {','.join(sorted(duplicates))}")
        expected_cells = canonical_mode_cells(
            include_oracle_workflow_isolation=self.include_oracle_workflow_isolation
        )
        if self.cells != expected_cells:
            raise ValueError("cells must match the canonical mode-comparison cells and order")


def canonical_mode_cells(
    *, include_oracle_workflow_isolation: bool = True
) -> tuple[ModeCellSpec, ...]:
    """Return the fixed canonical cells in execution order."""

    scanner_only = (
        ModeCellSpec("RL-SCAN", SCANNER_ONLY_SCOPE, "scanner", "reference-like", None),
        ModeCellSpec("Q-SCAN", SCANNER_ONLY_SCOPE, "scanner", "quality", None),
    )
    oracle = (
        ModeCellSpec(
            "ORACLE-REF",
            ORACLE_WORKFLOW_ISOLATION_SCOPE,
            "oracle",
            None,
            "reference",
        ),
        ModeCellSpec(
            "ORACLE-QUAL",
            ORACLE_WORKFLOW_ISOLATION_SCOPE,
            "oracle",
            None,
            "quality",
        ),
    )
    end_to_end = (
        ModeCellSpec("RL-REF", END_TO_END_SCOPE, "scanner", "reference-like", "reference"),
        ModeCellSpec("RL-QUAL", END_TO_END_SCOPE, "scanner", "reference-like", "quality"),
        ModeCellSpec("Q-REF", END_TO_END_SCOPE, "scanner", "quality", "reference"),
        ModeCellSpec("Q-QUAL", END_TO_END_SCOPE, "scanner", "quality", "quality"),
    )
    return scanner_only + (oracle if include_oracle_workflow_isolation else ()) + end_to_end
