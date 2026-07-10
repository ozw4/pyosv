from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pyosv.evaluation.synthetic_quality.config import (
    SyntheticSkinningConfig,
    SyntheticVotingConfig,
)
from pyosv.evaluation.synthetic_quality.variants import (
    BASELINE_VARIANT,
    DEFAULT_VARIANTS,
    QUALITY_MATRIX_VARIANTS,
    VARIANT_NAMES,
    VARIANT_PRESETS,
    VARIANT_REGISTRY,
    VARIANT_SPECS,
    effective_skinning_config,
    effective_thin_mode,
    effective_variant_config,
    get_variant_spec,
    resolve_variants,
    validate_variants,
)


EXPECTED_NAMES = (
    "current_default",
    "boundary_aware_voter_v1",
    "no_surface_orientation_smoothing",
    "final_norm_smoothing_1",
    "voter_thin_normal",
    "voter_thin_hybrid",
    "voter_thin_hybrid_v2",
    "voter_thin_hybrid_v2_recenter_scanner_target",
    "boundary_edge_thin_v1",
    "boundary_seed_retention_v1",
    "voter_thin_normal_plateau",
    "surface_support_weighted",
    "quality_skinner_v2",
    "quality_boundary_skinner_fallback",
    "quality_boundary_skinner_fallback_v2",
    "quality_boundary_skinner_fallback_v3",
    "quality_boundary_skinner_fallback_v4",
    "quality_boundary_skinner_fallback_v5",
)


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_every_variant_has_consistent_effective_configuration(name: str) -> None:
    spec = get_variant_spec(name)
    voting_config = SyntheticVotingConfig(voter_thin_mode="reference")
    skinning_config = SyntheticSkinningConfig()

    assert spec.name == name
    assert effective_thin_mode(spec, voting_config) == (
        spec.thinning_policy or voting_config.voter_thin_mode
    )
    assert effective_skinning_config(spec, skinning_config).boundary_skinner_fallback_policy == (
        spec.skinning.boundary_skinner_fallback_policy or "empty_primary"
    )
    assert (name in DEFAULT_VARIANTS) == ("default" in spec.presets)
    assert (name in QUALITY_MATRIX_VARIANTS) == ("quality-matrix" in spec.presets)


def test_registry_name_order_presets_and_baseline_are_derived() -> None:
    assert tuple(spec.name for spec in VARIANT_SPECS) == EXPECTED_NAMES
    assert VARIANT_NAMES == EXPECTED_NAMES
    assert tuple(VARIANT_REGISTRY) == EXPECTED_NAMES
    assert DEFAULT_VARIANTS == ("current_default",)
    assert (
        QUALITY_MATRIX_VARIANTS == EXPECTED_NAMES[:1] + EXPECTED_NAMES[2:7] + EXPECTED_NAMES[10:17]
    )
    assert VARIANT_PRESETS == {
        "default": DEFAULT_VARIANTS,
        "quality-matrix": QUALITY_MATRIX_VARIANTS,
    }
    assert BASELINE_VARIANT == "current_default"


def test_registry_models_and_mapping_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        get_variant_spec("current_default").experimental = True  # type: ignore[misc]
    with pytest.raises(TypeError):
        VARIANT_REGISTRY["new"] = get_variant_spec("current_default")  # type: ignore[index]


def test_current_default_and_boundary_aware_effective_config_dicts() -> None:
    voting_config = SyntheticVotingConfig()
    skinning_config = SyntheticSkinningConfig()
    current = effective_variant_config(
        get_variant_spec("current_default"),
        voting_config=voting_config,
        skinning_config=skinning_config,
    )
    boundary = effective_variant_config(
        get_variant_spec("boundary_aware_voter_v1"),
        voting_config=voting_config,
        skinning_config=skinning_config,
    )

    assert current == {
        "name": "current_default",
        "voting": {
            "boundary_policy": "legacy",
            "support_min_fraction": 0.0,
            "support_exponent": 0.0,
            "orientation_smoothing": None,
            "final_normalization_smoothing": None,
        },
        "seed_policy": "default",
        "thin_mode": "reference",
        "post_thinning_policy": "none",
        "skinning": skinning_config.as_report_dict(),
    }
    assert boundary == {
        **current,
        "name": "boundary_aware_voter_v1",
        "voting": {**current["voting"], "boundary_policy": "masked_in_bounds"},
    }


@pytest.mark.parametrize(
    ("variants", "message"),
    [
        ((), "variants must include at least one variant"),
        (("unknown",), "unknown variant(s): unknown"),
        (("current_default", "current_default"), "duplicate variant(s): current_default"),
    ],
)
def test_validate_variants_preserves_rejection_behavior(
    variants: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        validate_variants(variants)


def test_resolve_variants_prefers_explicit_input_and_validates_presets() -> None:
    assert resolve_variants(variants=("boundary_aware_voter_v1",), variant_preset="default") == (
        "boundary_aware_voter_v1",
    )
    assert resolve_variants(variants=None, variant_preset="default") == DEFAULT_VARIANTS
    with pytest.raises(ValueError, match="variant_preset must be one of"):
        resolve_variants(variants=None, variant_preset="missing")
