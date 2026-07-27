from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from pyosv.evaluation.f3d_mode_comparison.artifacts import canonical_json_bytes
from pyosv.evaluation.f3d_mode_comparison import (
    F3CellSpec,
    F3DatasetSpec,
    F3ModeComparisonConfig,
    F3ScannerConfig,
    F3VotingControls,
    build_f3d_mode_comparison_plan,
)
from pyosv.evaluation.synthetic_quality import SyntheticSkinningConfig
from pyosv.f3d_reference import F3D_DTYPE, F3D_EXPECTED_BYTES, F3D_SHAPE


def test_default_plan_fixes_dataset_cells_and_full_runner_controls() -> None:
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())

    assert plan.dataset_spec.shape == F3D_SHAPE
    assert plan.dataset_spec.input_file == "ep.dat"
    assert plan.dataset_spec.dtype == F3D_DTYPE
    assert plan.dataset_spec.expected_bytes == F3D_EXPECTED_BYTES
    assert tuple((cell.label, cell.scanner_backend, cell.workflow_mode) for cell in plan.cells) == (
        ("RL-REF", "reference-like", "reference"),
        ("RL-QUAL", "reference-like", "quality"),
        ("Q-REF", "quality", "reference"),
        ("Q-QUAL", "quality", "quality"),
    )
    assert plan.reference_like_scanner_config == F3ScannerConfig()
    assert plan.quality_scanner_config == replace(F3ScannerConfig(), backend="quality")
    assert plan.voting_controls == F3VotingControls()
    assert plan.boundary_diagnostic_margin == 16
    assert plan.fixed_control_evidence.full_volume_evaluation_units == 1


def test_scanner_and_workflow_are_independent_axes() -> None:
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())

    for cell in plan.cells:
        scanner = plan.scanner_config_for(cell.scanner_backend)
        workflow = plan.workflow_settings_for(cell.workflow_mode)
        assert scanner.backend == cell.scanner_backend
        assert workflow.workflow_mode == cell.workflow_mode
        assert scanner.scanner_thin_mode == "reference"
        assert scanner.remove_edge_effects is True
        assert scanner.effective_remove_edge_effects is True
        assert scanner.refinement_factor == 2

    assert plan.scanner_config_for("reference-like") is plan.scanner_config_for(
        plan.cells[1].scanner_backend
    )
    assert plan.scanner_config_for("quality") is plan.scanner_config_for(
        plan.cells[3].scanner_backend
    )


def test_default_resolved_workflow_differences_are_owned_by_workflow() -> None:
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())
    reference = plan.reference_workflow_settings
    quality = plan.quality_workflow_settings

    assert reference.voting_config.voter_thin_mode == "reference"
    assert quality.voting_config.voter_thin_mode == "hybrid_v2"
    assert (
        replace(
            quality.voting_config,
            voter_thin_mode="reference",
        )
        == reference.voting_config
    )
    assert reference.skinning_config.method == "reference"
    assert quality.skinning_config.method == "quality"
    assert reference.skinning_config.min_likelihood == 0.5
    assert quality.skinning_config.min_likelihood is None
    assert reference.skinning_config.growth_source == "thinned"
    assert quality.skinning_config.growth_source == "pre_thin"
    assert reference.skinning_config.accepted_occupancy_radius is None
    assert quality.skinning_config.accepted_occupancy_radius == 1
    assert reference.skinning_config.boundary_skinner_fallback is False
    assert quality.skinning_config.boundary_skinner_fallback is True
    assert quality.skinning_config.boundary_skinner_fallback_policy == ("empty_primary")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("scanner_thin_mode", "normal", "scanner_template.scanner_thin_mode"),
        ("remove_edge_effects", False, "scanner_template.remove_edge_effects"),
        ("refinement_factor", 1, "scanner_template.refinement_factor"),
        ("backend", "quality", "scanner_template.backend"),
    ),
)
def test_builder_rejects_noncanonical_scanner(field: str, value: object, message: str) -> None:
    scanner = replace(F3ScannerConfig(), **{field: value})

    with pytest.raises(ValueError, match=message):
        build_f3d_mode_comparison_plan(F3ModeComparisonConfig(scanner_template=scanner))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (
            {
                "interpolation_backend": "structured_linear",
                "interpolation_order": 3,
            },
            "interpolation_backend='structured_linear' requires interpolation_order=1",
        ),
        (
            {
                "interpolation_backend": "structured_linear",
                "orientation_backend": "directional",
            },
            "interpolation_backend='structured_linear' requires backend='rotate_shear'",
        ),
    ),
)
def test_scanner_config_rejects_cross_field_combinations(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        F3ScannerConfig(**kwargs)


def test_builder_rejects_noncanonical_dataset() -> None:
    with pytest.raises(ValueError, match="official F3 shape"):
        build_f3d_mode_comparison_plan(F3ModeComparisonConfig(shape=(421, 400, 100)))
    with pytest.raises(ValueError, match="input_file must be 'ep.dat'"):
        build_f3d_mode_comparison_plan(F3ModeComparisonConfig(input_file="xs.dat"))


def test_plan_model_rejects_nonofficial_fixture_dataset() -> None:
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())
    fixture_spec = F3DatasetSpec(
        dataset_id="fixture",
        shape=(2, 3, 4),
        files={"input": "ep.dat"},
        expected_bytes=96,
    )

    with pytest.raises(ValueError, match="official F3 dataset spec"):
        replace(plan, dataset_spec=fixture_spec)


@pytest.mark.parametrize(
    "fields",
    (
        {
            "label": "BAD",
            "scanner_backend": "reference-like",
            "workflow_mode": "reference",
        },
        {
            "label": "RL-REF",
            "scanner_backend": "fast",
            "workflow_mode": "reference",
        },
        {
            "label": "RL-REF",
            "scanner_backend": "reference-like",
            "workflow_mode": "diagnostic",
        },
    ),
)
def test_cell_rejects_inconsistent_or_unknown_axes(
    fields: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        F3CellSpec(**fields)  # type: ignore[arg-type]


def test_global_skinning_disable_changes_only_resolved_skinning() -> None:
    enabled = build_f3d_mode_comparison_plan(F3ModeComparisonConfig())
    disabled = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(skinning_enabled=False))

    assert disabled.reference_like_scanner_config == (enabled.reference_like_scanner_config)
    assert disabled.quality_scanner_config == enabled.quality_scanner_config
    assert disabled.voting_controls == enabled.voting_controls
    assert disabled.reference_workflow_settings.voting_config == (
        enabled.reference_workflow_settings.voting_config
    )
    assert disabled.quality_workflow_settings.voting_config == (
        enabled.quality_workflow_settings.voting_config
    )
    assert disabled.reference_workflow_settings.skinning_config.enabled is False
    assert disabled.quality_workflow_settings.skinning_config.enabled is False


def test_explicit_skinner_overrides_apply_to_both_workflows() -> None:
    skinning = SyntheticSkinningConfig(
        method="reference",
        min_likelihood=0.7,
        growth_source="thinned",
        accepted_occupancy_radius=3,
        boundary_skinner_fallback=False,
    )
    plan = build_f3d_mode_comparison_plan(
        F3ModeComparisonConfig(
            skinning_template=skinning,
            skinner_method_explicit=True,
            skinner_min_likelihood_explicit=True,
            skinner_growth_source_explicit=True,
            skinner_accepted_occupancy_radius_explicit=True,
            skinner_boundary_fallback_explicit=True,
        )
    )

    assert plan.reference_workflow_settings.skinning_config == skinning
    assert plan.quality_workflow_settings.skinning_config == skinning
    assert plan.skinner_method_explicit is True
    assert plan.skinner_min_likelihood_explicit is True
    assert plan.skinner_growth_source_explicit is True
    assert plan.skinner_accepted_occupancy_radius_explicit is True
    assert plan.skinner_boundary_fallback_explicit is True


def test_common_voting_override_is_identical_between_workflows() -> None:
    controls = replace(
        F3VotingControls(),
        seed_threshold=0.4,
        strain_max1=0.2,
        surface_support_min_fraction=0.25,
        surface_support_exponent=2.0,
    )
    plan = build_f3d_mode_comparison_plan(F3ModeComparisonConfig(voting_controls=controls))

    assert plan.voting_controls is controls
    assert (
        replace(
            plan.quality_workflow_settings.voting_config,
            voter_thin_mode="reference",
        )
        == plan.reference_workflow_settings.voting_config
    )


def test_common_voter_thin_override_is_identical_between_workflows() -> None:
    plan = build_f3d_mode_comparison_plan(
        F3ModeComparisonConfig(voter_thin_mode_override="reference")
    )

    assert plan.voter_thin_mode_override == "reference"
    assert (
        plan.reference_workflow_settings.voting_config.voter_thin_mode
        == plan.quality_workflow_settings.voting_config.voter_thin_mode
        == "reference"
    )


def test_common_voter_thin_override_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="voter_thin_mode_override"):
        F3ModeComparisonConfig(voter_thin_mode_override="normal")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("strain_max1", 0.0),
        ("strain_max1", 1.01),
        ("strain_max2", 0.0),
        ("strain_max2", 1.01),
    ),
)
def test_voting_controls_reject_unexecutable_strain(
    field: str,
    value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        replace(F3VotingControls(), **{field: value})


def test_plan_serialization_is_deterministic_and_json_safe() -> None:
    first = build_f3d_mode_comparison_plan(F3ModeComparisonConfig()).as_dict()
    second = build_f3d_mode_comparison_plan(F3ModeComparisonConfig()).as_dict()

    assert first == second
    assert first["dataset_spec"] == {
        "dataset_id": "f3d-official-v1",
        "shape": F3D_SHAPE,
        "storage_dtype": F3D_DTYPE,
        "files": (
            ("input", "ep.dat"),
            ("reference_fault_likelihood", "fl.dat"),
            ("reference_fault_votes", "fv.dat"),
            ("reference_thinned_fault_votes", "fvt.dat"),
        ),
        "expected_bytes": F3D_EXPECTED_BYTES,
    }
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(
        second,
        sort_keys=True,
        allow_nan=False,
    )


def test_numpy_numeric_config_values_are_normalized_before_serialization() -> None:
    scanner = replace(
        F3ScannerConfig(),
        phi_min=np.float32(0.0),
        refinement_factor=np.int64(2),
        interpolation_order=np.int64(1),
    )
    voting = replace(
        F3VotingControls(),
        ru=np.int64(10),
        seed_threshold=np.float32(0.3),
        strain_max1=np.float32(0.25),
    )
    plan = build_f3d_mode_comparison_plan(
        F3ModeComparisonConfig(
            scanner_template=scanner,
            voting_controls=voting,
        )
    )

    assert type(scanner.phi_min) is float
    assert type(scanner.refinement_factor) is int
    assert type(scanner.interpolation_order) is int
    assert type(voting.ru) is int
    assert type(voting.seed_threshold) is float
    assert type(voting.strain_max1) is float
    canonical_json_bytes(plan.as_dict())
