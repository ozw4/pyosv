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


BASE_VOTING = {
    "boundary_policy": "legacy",
    "support_min_fraction": 0.0,
    "support_exponent": 0.0,
    "orientation_smoothing": None,
    "final_normalization_smoothing": None,
}
BASE_SKINNING = {
    "enabled": True,
    "method": "reference",
    "growth_source": "thinned",
    "min_likelihood": 0.5,
    "adaptive_min_likelihood": False,
    "seed_min_ep": 0.8,
    "seed_planarity_source": "fvt",
    "min_skin_size": 1,
    "d": 1,
    "ru": 10,
    "rv": None,
    "rw": None,
    "max_steps": 10,
    "du": 5.0,
    "max_delta_strike": 30.0,
    "reskin": True,
    "accepted_occupancy_radius": None,
    "effective_accepted_occupancy_radius": 5,
    "small_skin_size": 10,
    "boundary_skinner_fallback": False,
    "boundary_skinner_fallback_policy": "empty_primary",
}
QUALITY_SKINNING = {
    "method": "quality",
    "growth_source": "pre_thin",
    "min_likelihood": None,
    "adaptive_min_likelihood": True,
    "seed_min_ep": 0.5,
    "accepted_occupancy_radius": 1,
    "effective_accepted_occupancy_radius": 1,
}
QUALITY_MATRIX = ("quality-matrix",)
EXPECTED_VARIANTS = {
    "current_default": {
        "voting": {},
        "seed_policy": "default",
        "thin_mode": "reference",
        "post_thinning_policy": "none",
        "skinning": {},
        "experimental": False,
        "presets": ("default", "quality-matrix"),
        "baseline": True,
    },
    "boundary_aware_voter_v1": {
        "voting": {"boundary_policy": "masked_in_bounds"},
        "seed_policy": "default",
        "thin_mode": "reference",
        "post_thinning_policy": "none",
        "skinning": {},
        "experimental": True,
        "presets": (),
        "baseline": False,
    },
    "no_surface_orientation_smoothing": {
        "voting": {"orientation_smoothing": 0.0},
        "seed_policy": "default",
        "thin_mode": "reference",
        "post_thinning_policy": "none",
        "skinning": {},
        "experimental": True,
        "presets": QUALITY_MATRIX,
        "baseline": False,
    },
    "final_norm_smoothing_1": {
        "voting": {"final_normalization_smoothing": 1.0},
        "seed_policy": "default",
        "thin_mode": "reference",
        "post_thinning_policy": "none",
        "skinning": {},
        "experimental": True,
        "presets": QUALITY_MATRIX,
        "baseline": False,
    },
    "voter_thin_normal": {
        "voting": {},
        "seed_policy": "default",
        "thin_mode": "normal",
        "post_thinning_policy": "none",
        "skinning": {},
        "experimental": True,
        "presets": QUALITY_MATRIX,
        "baseline": False,
    },
    "voter_thin_hybrid": {
        "voting": {},
        "seed_policy": "default",
        "thin_mode": "hybrid",
        "post_thinning_policy": "none",
        "skinning": {},
        "experimental": True,
        "presets": QUALITY_MATRIX,
        "baseline": False,
    },
    "voter_thin_hybrid_v2": {
        "voting": {},
        "seed_policy": "default",
        "thin_mode": "hybrid_v2",
        "post_thinning_policy": "none",
        "skinning": {},
        "experimental": True,
        "presets": QUALITY_MATRIX,
        "baseline": False,
    },
    "voter_thin_hybrid_v2_recenter_scanner_target": {
        "voting": {},
        "seed_policy": "default",
        "thin_mode": "hybrid_v2",
        "post_thinning_policy": "recenter_scanner_target",
        "skinning": {},
        "experimental": True,
        "presets": (),
        "baseline": False,
    },
    "boundary_edge_thin_v1": {
        "voting": {},
        "seed_policy": "default",
        "thin_mode": "hybrid_v2",
        "post_thinning_policy": "boundary_edge_thin_v1",
        "skinning": {},
        "experimental": True,
        "presets": (),
        "baseline": False,
    },
    "boundary_seed_retention_v1": {
        "voting": {},
        "seed_policy": "boundary_seed_retention_v1",
        "thin_mode": "reference",
        "post_thinning_policy": "none",
        "skinning": {},
        "experimental": True,
        "presets": (),
        "baseline": False,
    },
    "voter_thin_normal_plateau": {
        "voting": {},
        "seed_policy": "default",
        "thin_mode": "normal_plateau",
        "post_thinning_policy": "none",
        "skinning": {},
        "experimental": True,
        "presets": QUALITY_MATRIX,
        "baseline": False,
    },
    "surface_support_weighted": {
        "voting": {"support_min_fraction": 0.5, "support_exponent": 1.0},
        "seed_policy": "default",
        "thin_mode": "reference",
        "post_thinning_policy": "none",
        "skinning": {},
        "experimental": True,
        "presets": QUALITY_MATRIX,
        "baseline": False,
    },
    "quality_skinner_v2": {
        "voting": {},
        "seed_policy": "default",
        "thin_mode": "reference",
        "post_thinning_policy": "none",
        "skinning": QUALITY_SKINNING,
        "experimental": True,
        "presets": QUALITY_MATRIX,
        "baseline": False,
    },
    "quality_boundary_skinner_fallback": {
        "voting": {},
        "seed_policy": "default",
        "thin_mode": "reference",
        "post_thinning_policy": "none",
        "skinning": {"boundary_skinner_fallback": True},
        "experimental": True,
        "presets": QUALITY_MATRIX,
        "baseline": False,
    },
    "quality_boundary_skinner_fallback_v2": {
        "voting": {},
        "seed_policy": "default",
        "thin_mode": "reference",
        "post_thinning_policy": "none",
        "skinning": {
            "boundary_skinner_fallback": True,
            "boundary_skinner_fallback_policy": "degraded_primary",
        },
        "experimental": True,
        "presets": QUALITY_MATRIX,
        "baseline": False,
    },
    "quality_boundary_skinner_fallback_v3": {
        "voting": {},
        "seed_policy": "default",
        "thin_mode": "reference",
        "post_thinning_policy": "none",
        "skinning": {
            "boundary_skinner_fallback": True,
            "boundary_skinner_fallback_policy": "degraded_primary_filtered",
        },
        "experimental": True,
        "presets": QUALITY_MATRIX,
        "baseline": False,
    },
    "quality_boundary_skinner_fallback_v4": {
        "voting": {},
        "seed_policy": "default",
        "thin_mode": "reference",
        "post_thinning_policy": "none",
        "skinning": {
            "boundary_skinner_fallback": True,
            "boundary_skinner_fallback_policy": "degraded_primary_skeletonized",
        },
        "experimental": True,
        "presets": QUALITY_MATRIX,
        "baseline": False,
    },
    "quality_boundary_skinner_fallback_v5": {
        "voting": {},
        "seed_policy": "default",
        "thin_mode": "reference",
        "post_thinning_policy": "none",
        "skinning": {
            **QUALITY_SKINNING,
            "boundary_skinner_fallback": True,
            "boundary_skinner_fallback_policy": "degraded_primary_topology_guarded",
        },
        "experimental": True,
        "presets": (),
        "baseline": False,
    },
}


@pytest.mark.parametrize("name", EXPECTED_NAMES)
def test_every_variant_has_expected_effective_configuration(name: str) -> None:
    spec = get_variant_spec(name)
    voting_config = SyntheticVotingConfig(voter_thin_mode="reference")
    skinning_config = SyntheticSkinningConfig()
    expected = EXPECTED_VARIANTS[name]

    assert spec.name == name
    assert spec.experimental is expected["experimental"]
    assert spec.presets == expected["presets"]
    assert spec.baseline is expected["baseline"]
    assert effective_variant_config(
        spec,
        voting_config=voting_config,
        skinning_config=skinning_config,
    ) == {
        "name": name,
        "voting": {**BASE_VOTING, **expected["voting"]},
        "seed_policy": expected["seed_policy"],
        "thin_mode": expected["thin_mode"],
        "post_thinning_policy": expected["post_thinning_policy"],
        "skinning": {**BASE_SKINNING, **expected["skinning"]},
    }


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
