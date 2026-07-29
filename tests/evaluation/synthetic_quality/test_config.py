from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pyosv.evaluation.synthetic_quality import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)


def test_config_defaults() -> None:
    assert SyntheticVotingConfig() == SyntheticVotingConfig(
        ru=1,
        rv=2,
        rw=2,
        seed_distance=3,
        seed_threshold=0.5,
        attribute_smoothing=0,
        voter_thin_mode="reference",
        reference_thin_sigma=1.0,
        surface_support_min_fraction=0.0,
        surface_support_exponent=0.0,
    )
    assert SyntheticScannerConfig() == SyntheticScannerConfig(
        backend="reference-like",
        phi_min=0.0,
        phi_max=180.0,
        theta_min=45.0,
        theta_max=90.0,
        sigma1=2.0,
        sigma2=2.0,
        refinement_factor=2,
        scanner_thin_mode="reference",
        remove_edge_effects=True,
    )
    assert SyntheticTruthMetricConfig() == SyntheticTruthMetricConfig(
        truth_surface_half_width=0.5,
        buffer_radius=2.0,
    )
    assert SyntheticSkinningConfig() == SyntheticSkinningConfig(
        enabled=True,
        method="reference",
        growth_source="thinned",
        min_likelihood=0.5,
        min_skin_size=1,
        d=1,
        ru=10,
        rv=None,
        rw=None,
        max_steps=10,
        du=5.0,
        max_delta_strike=30.0,
        reskin=True,
        reskin_policy="existing_cells_v1",
        accepted_occupancy_radius=None,
        small_skin_size=10,
        boundary_skinner_fallback=False,
        boundary_skinner_fallback_policy="empty_primary",
    )


def test_default_report_dicts() -> None:
    assert SyntheticVotingConfig().as_report_dict() == {
        "ru": 1,
        "rv": 2,
        "rw": 2,
        "seed_distance": 3,
        "seed_threshold": 0.5,
        "attribute_smoothing": 0,
        "voter_thin_mode": "reference",
        "reference_thin_sigma": 1.0,
        "surface_support_min_fraction": 0.0,
        "surface_support_exponent": 0.0,
    }
    assert SyntheticScannerConfig().as_report_dict() == {
        "backend": "reference-like",
        "phi_min": 0.0,
        "phi_max": 180.0,
        "theta_min": 45.0,
        "theta_max": 90.0,
        "sigma1": 2.0,
        "sigma2": 2.0,
        "refinement_factor": 2,
        "scanner_thin_mode": "reference",
        "remove_edge_effects": True,
        "input": {
            "background": 1.0,
            "fault_contrast": 0.85,
            "noise_sigma": 0.0,
            "seed": 20260706,
            "clip_min": 0.0,
            "clip_max": 1.0,
        },
    }
    assert SyntheticTruthMetricConfig().as_report_dict() == {
        "truth_surface_half_width": 0.5,
        "buffer_radius": 2.0,
    }
    assert SyntheticSkinningConfig().as_report_dict() == {
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
        "reskin_policy": "existing_cells_v1",
        "accepted_occupancy_radius": None,
        "effective_accepted_occupancy_radius": 5,
        "small_skin_size": 10,
        "boundary_skinner_fallback": False,
        "boundary_skinner_fallback_policy": "empty_primary",
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sigma1": float("nan")}, "scanner_sigma1 must be finite"),
        ({"sigma2": 0.0}, "scanner_sigma2 must be positive"),
        ({"refinement_factor": True}, "scanner_refinement_factor must be an integer"),
        ({"refinement_factor": 5}, "scanner_refinement_factor must be an integer"),
    ],
)
def test_scanner_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SyntheticScannerConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_likelihood": -0.1}, "skinner_min_likelihood must be non-negative"),
        ({"min_skin_size": 1.5}, "skinner_min_skin_size must be a non-negative integer"),
        ({"reskin_policy": "unknown"}, "reskin_policy"),
        ({"ru": 1}, "skinner_ru must be at least 2"),
        ({"rv": 1}, "skinner_rv must be at least 2"),
    ],
)
def test_skinning_config_rejects_invalid_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SyntheticSkinningConfig(**kwargs)


def test_skinning_config_optional_fields_preserve_none_semantics() -> None:
    config = SyntheticSkinningConfig(
        method="quality",
        min_likelihood=None,
        min_skin_size=None,
        rv=None,
        rw=None,
        accepted_occupancy_radius=None,
    )

    assert config.as_report_dict()["adaptive_min_likelihood"] is True
    assert config.as_report_dict()["effective_accepted_occupancy_radius"] == 5

    with pytest.raises(ValueError, match="skinner_ru must be a non-negative integer"):
        SyntheticSkinningConfig(ru=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("config", "field"),
    [
        (SyntheticVotingConfig(), "ru"),
        (SyntheticScannerConfig(), "backend"),
        (SyntheticTruthMetricConfig(), "buffer_radius"),
        (SyntheticSkinningConfig(), "enabled"),
    ],
)
def test_configs_are_immutable(config: object, field: str) -> None:
    with pytest.raises(FrozenInstanceError):
        config.__setattr__(field, None)
