"""Typed variant registry for synthetic quality evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Sequence

from .config import SyntheticSkinningConfig, SyntheticVotingConfig


@dataclass(frozen=True, slots=True)
class VotingPatch:
    """Declarative changes applied to the voting stage."""

    boundary_policy: str | None = None
    support_min_fraction: float | None = None
    support_exponent: float | None = None
    orientation_smoothing: float | None = None
    final_normalization_smoothing: float | None = None


@dataclass(frozen=True, slots=True)
class SkinningPatch:
    """Declarative changes applied to ``SyntheticSkinningConfig``."""

    method: str | None = None
    min_likelihood: float | None = None
    override_min_likelihood: bool = False
    accepted_occupancy_radius: int | None = None
    growth_source: str | None = None
    boundary_skinner_fallback: bool | None = None
    boundary_skinner_fallback_policy: str | None = None
    reskin_policy: str | None = None


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """Serializable-like declaration of one synthetic quality variant."""

    name: str
    voting: VotingPatch = VotingPatch()
    seed_policy: str = "default"
    thinning_policy: str | None = None
    post_thinning_policy: str = "none"
    skinning: SkinningPatch = SkinningPatch()
    experimental: bool = True
    presets: tuple[str, ...] = ()
    baseline: bool = False


_QUALITY_MATRIX = ("quality-matrix",)
_QUALITY_SKINNER = SkinningPatch(
    method="quality",
    min_likelihood=None,
    override_min_likelihood=True,
    accepted_occupancy_radius=1,
    growth_source="pre_thin",
)

VARIANT_SPECS = (
    VariantSpec(
        "current_default",
        experimental=False,
        presets=("default", "quality-matrix"),
        baseline=True,
    ),
    VariantSpec(
        "boundary_aware_voter_v1",
        voting=VotingPatch(boundary_policy="masked_in_bounds"),
    ),
    VariantSpec(
        "no_surface_orientation_smoothing",
        voting=VotingPatch(orientation_smoothing=0.0),
        presets=_QUALITY_MATRIX,
    ),
    VariantSpec(
        "final_norm_smoothing_1",
        voting=VotingPatch(final_normalization_smoothing=1.0),
        presets=_QUALITY_MATRIX,
    ),
    VariantSpec("voter_thin_normal", thinning_policy="normal", presets=_QUALITY_MATRIX),
    VariantSpec("voter_thin_hybrid", thinning_policy="hybrid", presets=_QUALITY_MATRIX),
    VariantSpec("voter_thin_hybrid_v2", thinning_policy="hybrid_v2", presets=_QUALITY_MATRIX),
    VariantSpec(
        "voter_thin_hybrid_v2_recenter_scanner_target",
        thinning_policy="hybrid_v2",
        post_thinning_policy="recenter_scanner_target",
    ),
    VariantSpec(
        "boundary_edge_thin_v1",
        thinning_policy="hybrid_v2",
        post_thinning_policy="boundary_edge_thin_v1",
    ),
    VariantSpec("boundary_seed_retention_v1", seed_policy="boundary_seed_retention_v1"),
    VariantSpec(
        "voter_thin_normal_plateau",
        thinning_policy="normal_plateau",
        presets=_QUALITY_MATRIX,
    ),
    VariantSpec(
        "surface_support_weighted",
        voting=VotingPatch(support_min_fraction=0.5, support_exponent=1.0),
        presets=_QUALITY_MATRIX,
    ),
    VariantSpec("quality_skinner_v2", skinning=_QUALITY_SKINNER, presets=_QUALITY_MATRIX),
    VariantSpec(
        "quality_boundary_skinner_fallback",
        skinning=SkinningPatch(boundary_skinner_fallback=True),
        presets=_QUALITY_MATRIX,
    ),
    VariantSpec(
        "quality_boundary_skinner_fallback_v2",
        skinning=SkinningPatch(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary",
        ),
        presets=_QUALITY_MATRIX,
    ),
    VariantSpec(
        "quality_boundary_skinner_fallback_v3",
        skinning=SkinningPatch(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary_filtered",
        ),
        presets=_QUALITY_MATRIX,
    ),
    VariantSpec(
        "quality_boundary_skinner_fallback_v4",
        skinning=SkinningPatch(
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary_skeletonized",
        ),
        presets=_QUALITY_MATRIX,
    ),
    VariantSpec(
        "quality_boundary_skinner_fallback_v5",
        skinning=replace(
            _QUALITY_SKINNER,
            boundary_skinner_fallback=True,
            boundary_skinner_fallback_policy="degraded_primary_topology_guarded",
        ),
    ),
)

VARIANT_REGISTRY: Mapping[str, VariantSpec] = MappingProxyType(
    {spec.name: spec for spec in VARIANT_SPECS}
)
VARIANT_NAMES = tuple(VARIANT_REGISTRY)
VARIANT_PRESETS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        preset: tuple(spec.name for spec in VARIANT_SPECS if preset in spec.presets)
        for preset in ("default", "quality-matrix")
    }
)
DEFAULT_VARIANTS = VARIANT_PRESETS["default"]
QUALITY_MATRIX_VARIANTS = VARIANT_PRESETS["quality-matrix"]
BASELINE_VARIANT = next(spec.name for spec in VARIANT_SPECS if spec.baseline)


def get_variant_spec(name: str) -> VariantSpec:
    """Resolve a variant name or raise the report's established error."""

    try:
        return VARIANT_REGISTRY[name]
    except KeyError as error:
        raise ValueError(f"unknown variant: {name}") from error


def validate_variants(variants: Sequence[str]) -> tuple[str, ...]:
    valid_variants = tuple(variants)
    if not valid_variants:
        raise ValueError("variants must include at least one variant")
    unknown = sorted(set(valid_variants).difference(VARIANT_REGISTRY))
    if unknown:
        raise ValueError(f"unknown variant(s): {','.join(unknown)}")
    duplicates = {name for name in valid_variants if valid_variants.count(name) > 1}
    if duplicates:
        raise ValueError(f"duplicate variant(s): {','.join(sorted(duplicates))}")
    return valid_variants


def validate_variant_preset(variant_preset: str) -> str:
    if variant_preset not in VARIANT_PRESETS:
        raise ValueError("variant_preset must be one of: " + ", ".join(sorted(VARIANT_PRESETS)))
    return variant_preset


def resolve_variants(*, variants: Sequence[str] | None, variant_preset: str) -> tuple[str, ...]:
    if variants is not None:
        return validate_variants(variants)
    return validate_variants(VARIANT_PRESETS[validate_variant_preset(variant_preset)])


def effective_thin_mode(spec: VariantSpec, voting_config: SyntheticVotingConfig) -> str:
    return spec.thinning_policy or voting_config.voter_thin_mode


def effective_skinning_config(
    spec: VariantSpec, skinning_config: SyntheticSkinningConfig
) -> SyntheticSkinningConfig:
    patch = spec.skinning
    changes: dict[str, object] = {}
    for field in (
        "method",
        "accepted_occupancy_radius",
        "growth_source",
        "boundary_skinner_fallback",
        "boundary_skinner_fallback_policy",
        "reskin_policy",
    ):
        value = getattr(patch, field)
        if value is not None:
            changes[field] = value
    if patch.override_min_likelihood:
        changes["min_likelihood"] = patch.min_likelihood
    return replace(skinning_config, **changes) if changes else skinning_config


def effective_variant_config(
    spec: VariantSpec,
    *,
    voting_config: SyntheticVotingConfig,
    skinning_config: SyntheticSkinningConfig,
) -> dict[str, object]:
    """Return the effective declarative settings for tests and diagnostics."""

    voting = spec.voting
    return {
        "name": spec.name,
        "voting": {
            "boundary_policy": voting.boundary_policy or "legacy",
            "support_min_fraction": (
                voting_config.surface_support_min_fraction
                if voting.support_min_fraction is None
                else voting.support_min_fraction
            ),
            "support_exponent": (
                voting_config.surface_support_exponent
                if voting.support_exponent is None
                else voting.support_exponent
            ),
            "orientation_smoothing": voting.orientation_smoothing,
            "final_normalization_smoothing": voting.final_normalization_smoothing,
        },
        "seed_policy": spec.seed_policy,
        "thin_mode": effective_thin_mode(spec, voting_config),
        "post_thinning_policy": spec.post_thinning_policy,
        "skinning": effective_skinning_config(spec, skinning_config).as_report_dict(),
    }
