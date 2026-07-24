from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pyosv.cells import FaultCell
from pyosv.evaluation.synthetic_quality import pipeline
from pyosv.evaluation.synthetic_quality.config import (
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from pyosv.evaluation.synthetic_quality.models import (
    OrientationField3D,
    PipelineEvaluation,
    PipelineStageTrace3D,
)
from pyosv.evaluation.synthetic_quality.pipeline import run_voting_from_attributes
from pyosv.evaluation.synthetic_quality.variants import get_variant_spec
from pyosv.synthetic3d import make_single_vertical_plane_case


def test_orientation_field_validates_without_copying() -> None:
    values = np.zeros((3, 4, 5), dtype=np.float32)
    field = OrientationField3D(values, values, values)
    assert field.ft is values
    with pytest.raises(TypeError, match="float32"):
        OrientationField3D(values.astype(np.float64), values, values)


@pytest.mark.parametrize("variant", ("current_default", "boundary_aware_voter_v1"))
@pytest.mark.parametrize("skinning_enabled", (False, True))
def test_pipeline_stage_arrays_and_inputs(variant: str, skinning_enabled: bool) -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    inputs = tuple(array.copy() for array in (case.ft_oracle, case.pt_oracle, case.tt_oracle))
    result = run_voting_from_attributes(
        case,
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
        voting_config=SyntheticVotingConfig(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=skinning_enabled),
        variant_spec=get_variant_spec(variant),
    )
    assert isinstance(result, PipelineEvaluation)
    assert result.artifacts.stage_trace is None
    for name in ("fv_py", "vp_py", "vt_py", "fvt_py", "skin_mask_py"):
        volume = result.artifacts.volumes[name]
        assert volume.shape == case.shape
        assert volume.dtype == np.float32
        assert np.isfinite(volume).all()
    for actual, original in zip((case.ft_oracle, case.pt_oracle, case.tt_oracle), inputs):
        np.testing.assert_array_equal(actual, original)


def test_skinning_diagnostics_keep_truth_count_selected_fvt_candidate_count() -> None:
    base_case = make_single_vertical_plane_case((9, 9, 9))
    truth_distance = np.full(base_case.shape, 10.0, dtype=np.float32)
    truth_distance[4, 4, 4] = 0.0
    case = replace(base_case, truth_distance=truth_distance)

    result = run_voting_from_attributes(
        case,
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
        voting_config=SyntheticVotingConfig(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=True),
        variant_spec=get_variant_spec("current_default"),
    )

    fvt = result.artifacts.volumes["fvt_py"]
    selected_count = result.report_payload["quality"]["fvt_positive_top_truth_count"][
        "buffered_overlap_radius2"
    ]["candidate_count"]
    diagnostics = result.report_payload["skinning"]["diagnostics"]
    unique_cell_count = diagnostics["skin_primary_unique_cell_count"]
    largest_size = diagnostics["skin_primary_largest_size"]

    assert np.count_nonzero(fvt > pipeline.NONZERO_EPSILON) > selected_count
    assert diagnostics["skin_primary_cell_coverage_of_fvt_positive"] == (
        unique_cell_count / selected_count
    )
    assert diagnostics["skin_primary_largest_coverage_of_fvt_positive"] == (
        largest_size / selected_count
    )
    assert diagnostics["skin_primary_degraded_candidate"] is False
    assert diagnostics["skin_primary_degraded_reasons"] == []


def test_pipeline_keeps_truth_metric_validation_after_workflow_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    truth_metric_config = SyntheticTruthMetricConfig(buffer_radius=-0.1)
    calls = {"voting": 0, "thinning": 0, "skinning": 0, "fallback": 0}
    original_voting = pipeline.OptimalSurfaceVoter.apply_voting_from_seeds
    original_thinning = pipeline.OptimalSurfaceVoter.thin
    original_skinning = pipeline.find_synthetic_skins
    original_fallback = pipeline.apply_boundary_skinner_fallback

    def counted_voting(self, *args, **kwargs):
        calls["voting"] += 1
        return original_voting(self, *args, **kwargs)

    def counted_thinning(self, *args, **kwargs):
        calls["thinning"] += 1
        return original_thinning(self, *args, **kwargs)

    def counted_skinning(*args, **kwargs):
        calls["skinning"] += 1
        return original_skinning(*args, **kwargs)

    def counted_fallback(*args, **kwargs):
        calls["fallback"] += 1
        return original_fallback(*args, **kwargs)

    monkeypatch.setattr(
        pipeline.OptimalSurfaceVoter,
        "apply_voting_from_seeds",
        counted_voting,
    )
    monkeypatch.setattr(pipeline.OptimalSurfaceVoter, "thin", counted_thinning)
    monkeypatch.setattr(pipeline, "find_synthetic_skins", counted_skinning)
    monkeypatch.setattr(pipeline, "apply_boundary_skinner_fallback", counted_fallback)

    assert truth_metric_config.buffer_radius == -0.1
    with pytest.raises(ValueError, match="^buffer_radius must be non-negative$"):
        run_voting_from_attributes(
            case,
            ft=case.ft_oracle,
            pt=case.pt_oracle,
            tt=case.tt_oracle,
            voting_config=SyntheticVotingConfig(),
            truth_metric_config=truth_metric_config,
            skinning_config=SyntheticSkinningConfig(enabled=True),
            variant_spec=get_variant_spec("current_default"),
        )

    assert calls == {"voting": 1, "thinning": 1, "skinning": 1, "fallback": 1}


def _run_pipeline_with_trace(
    *, skinning_enabled: bool, variant: str = "current_default"
) -> PipelineEvaluation:
    case = make_single_vertical_plane_case((9, 9, 9))
    return run_voting_from_attributes(
        case,
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
        voting_config=SyntheticVotingConfig(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=skinning_enabled),
        variant_spec=get_variant_spec(variant),
        capture_stage_trace=True,
    )


def test_stage_trace_opt_in_does_not_change_report_or_volume_keys() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    kwargs = {
        "ft": case.ft_oracle,
        "pt": case.pt_oracle,
        "tt": case.tt_oracle,
        "voting_config": SyntheticVotingConfig(),
        "truth_metric_config": SyntheticTruthMetricConfig(),
        "skinning_config": SyntheticSkinningConfig(enabled=False),
        "variant_spec": get_variant_spec("current_default"),
    }
    plain = run_voting_from_attributes(case, **kwargs)
    captured = run_voting_from_attributes(case, **kwargs, capture_stage_trace=True)

    assert plain.artifacts.stage_trace is None
    assert plain.report_payload == captured.report_payload
    assert plain.artifacts.volumes.keys() == captured.artifacts.volumes.keys()


def test_stage_trace_captures_voting_and_disabled_skinning_masks() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    result = _run_pipeline_with_trace(skinning_enabled=False)
    trace = result.artifacts.stage_trace
    assert trace is not None
    for mask in (
        trace.seed_candidate_mask,
        trace.seed_selected_mask,
        trace.fv_positive_mask,
        trace.fvt_positive_mask,
        trace.primary_skin_mask,
        trace.fallback_skin_mask,
        trace.final_skin_mask,
    ):
        assert mask.shape == case.shape
        assert mask.dtype == np.bool_
    np.testing.assert_array_equal(
        trace.seed_candidate_mask,
        case.ft_oracle > np.float32(SyntheticVotingConfig().seed_threshold),
    )
    np.testing.assert_array_equal(
        trace.fv_positive_mask,
        result.artifacts.volumes["fv_py"] > pipeline.NONZERO_EPSILON,
    )
    np.testing.assert_array_equal(
        trace.fvt_positive_mask,
        result.artifacts.volumes["fvt_py"] > pipeline.NONZERO_EPSILON,
    )
    assert np.count_nonzero(trace.seed_selected_mask) == 9
    assert not trace.skinning_enabled
    assert not trace.fallback_used
    assert not trace.primary_skin_mask.any()
    assert not trace.fallback_skin_mask.any()
    assert not trace.final_skin_mask.any()


def test_stage_trace_primary_skin_is_final_when_fallback_is_unused() -> None:
    trace = _run_pipeline_with_trace(skinning_enabled=True).artifacts.stage_trace
    assert trace is not None
    assert trace.primary_skin_mask.any()
    np.testing.assert_array_equal(trace.primary_skin_mask, trace.final_skin_mask)
    assert not trace.fallback_used
    assert not trace.fallback_skin_mask.any()


def test_stage_trace_separates_primary_and_adopted_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def replace_with_fallback(
        skins: list[object],
        *args: object,
        diagnostics: dict[str, object],
        **kwargs: object,
    ) -> None:
        skins[:] = [[FaultCell(0, 0, 0, 1.0, 0.0, 90.0)]]
        diagnostics["fallback_used"] = True

    monkeypatch.setattr(pipeline, "apply_boundary_skinner_fallback", replace_with_fallback)
    trace = _run_pipeline_with_trace(skinning_enabled=True).artifacts.stage_trace
    assert trace is not None
    assert trace.fallback_used
    assert trace.primary_skin_mask.any()
    assert not np.array_equal(trace.primary_skin_mask, trace.final_skin_mask)
    np.testing.assert_array_equal(trace.fallback_skin_mask, trace.final_skin_mask)
    assert trace.final_skin_mask[0, 0, 0]


def test_boundary_seed_variant_trace_uses_selected_variant_seeds() -> None:
    result = _run_pipeline_with_trace(skinning_enabled=False, variant="boundary_seed_retention_v1")
    trace = result.artifacts.stage_trace
    assert trace is not None
    diagnostics = result.report_payload["boundary_seed_retention"]
    assert np.count_nonzero(trace.seed_selected_mask) == diagnostics["total_seed_count"]
    assert np.count_nonzero(trace.seed_selected_mask) > diagnostics["default_seed_count"]


def test_pipeline_stage_trace_validates_and_copies_masks() -> None:
    mask = np.zeros((2, 3, 4), dtype=bool)
    trace = PipelineStageTrace3D(
        mask,
        mask,
        mask,
        mask,
        mask,
        mask,
        mask,
        False,
        False,
    )
    mask[:] = True
    assert not trace.seed_candidate_mask.any()

    mismatched = np.zeros((2, 3, 5), dtype=bool)
    with pytest.raises(ValueError, match="must have shape"):
        PipelineStageTrace3D(mask, mismatched, mask, mask, mask, mask, mask, False, False)
    non_bool = np.zeros((2, 3, 4), dtype=np.uint8)
    with pytest.raises(TypeError, match="dtype bool"):
        PipelineStageTrace3D(non_bool, mask, mask, mask, mask, mask, mask, False, False)
