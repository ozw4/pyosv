from __future__ import annotations

import numpy as np
import pytest

from pyosv.evaluation.synthetic_quality.config import (
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from pyosv.evaluation.synthetic_quality.models import OrientationField3D, PipelineEvaluation
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
    for name in ("fv_py", "vp_py", "vt_py", "fvt_py", "skin_mask_py"):
        volume = result.artifacts.volumes[name]
        assert volume.shape == case.shape
        assert volume.dtype == np.float32
        assert np.isfinite(volume).all()
    for actual, original in zip((case.ft_oracle, case.pt_oracle, case.tt_oracle), inputs):
        np.testing.assert_array_equal(actual, original)
