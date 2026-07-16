"""Configuration for canonical synthetic mode-comparison plans."""

from __future__ import annotations

from dataclasses import dataclass

from pyosv.synthetic3d import validate_shape3

from ..synthetic_quality import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from .trials import validate_trial_seeds

CANONICAL_COMPARISON_VARIANT = "current_default"


@dataclass(frozen=True, slots=True)
class SyntheticModeComparisonConfig:
    """Inputs used to construct a canonical synthetic mode-comparison plan."""

    case_set: str = "minimal"
    case_ids: tuple[str, ...] | None = None
    trial_seeds: tuple[int, ...] = (20260707,)
    shape: tuple[int, int, int] = (49, 49, 49)
    scanner_template: SyntheticScannerConfig = SyntheticScannerConfig()
    voting_config: SyntheticVotingConfig | None = None
    skinning_config: SyntheticSkinningConfig = SyntheticSkinningConfig()
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig()
    include_oracle_workflow_isolation: bool = True
    comparison_variant: str = CANONICAL_COMPARISON_VARIANT
    skinner_method_explicit: bool = False
    skinner_min_likelihood_explicit: bool = False
    skinner_growth_source_explicit: bool = False
    skinner_accepted_occupancy_radius_explicit: bool = False
    skinner_boundary_fallback_explicit: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", validate_shape3(self.shape))
        if self.case_ids is not None:
            object.__setattr__(self, "case_ids", tuple(self.case_ids))
        object.__setattr__(self, "trial_seeds", validate_trial_seeds(self.trial_seeds))
        if not isinstance(self.scanner_template, SyntheticScannerConfig):
            raise ValueError("scanner_template must be a SyntheticScannerConfig")
        if self.voting_config is not None and not isinstance(
            self.voting_config, SyntheticVotingConfig
        ):
            raise ValueError("voting_config must be a SyntheticVotingConfig or None")
        if not isinstance(self.skinning_config, SyntheticSkinningConfig):
            raise ValueError("skinning_config must be a SyntheticSkinningConfig")
        if not isinstance(self.truth_metric_config, SyntheticTruthMetricConfig):
            raise ValueError("truth_metric_config must be a SyntheticTruthMetricConfig")
        _validate_bool(
            self.include_oracle_workflow_isolation,
            "include_oracle_workflow_isolation",
        )
        for name in (
            "skinner_method_explicit",
            "skinner_min_likelihood_explicit",
            "skinner_growth_source_explicit",
            "skinner_accepted_occupancy_radius_explicit",
            "skinner_boundary_fallback_explicit",
        ):
            _validate_bool(getattr(self, name), name)
        if self.comparison_variant != CANONICAL_COMPARISON_VARIANT:
            raise ValueError(
                "comparison_variant must be 'current_default' for a canonical mode-comparison plan"
            )


def _validate_bool(value: bool, name: str) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")
