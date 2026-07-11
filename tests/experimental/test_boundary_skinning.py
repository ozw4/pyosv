from __future__ import annotations

import numpy as np
import pytest

from pyosv.evaluation.synthetic_quality.config import SyntheticSkinningConfig
from pyosv.evaluation.synthetic_quality.variants import get_variant_spec
from pyosv.experimental.boundary_skinning import (
    apply_boundary_skinner_fallback,
    fallback_component_diagnostics,
    positive_mask_components,
    skeletonize_fallback_components,
)


def test_component_ordering_is_deterministic() -> None:
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[0, 2, 0:2] = True
    mask[0:2, 0, 2] = True
    assert positive_mask_components(mask, connectivity="edge") == [
        [(0, 0, 2), (1, 0, 2)],
        [(0, 2, 0), (0, 2, 1)],
    ]


def test_filtered_policy_keeps_largest_when_all_components_are_small() -> None:
    mask = np.zeros((1, 1, 10), dtype=np.float32)
    mask[0, 0, 0:4] = 1.0
    mask[0, 0, 7:10] = 1.0
    diagnostics = fallback_component_diagnostics(
        mask,
        min_skin_size=1,
        small_component_size=3,
        connectivity="edge",
        component_policy="degraded_primary_filtered",
    )
    assert diagnostics["skin_fallback_accepted_component_count"] == 1
    assert diagnostics["skin_fallback_accepted_component_cell_count"] == 4


def test_skeleton_subset_and_tie_are_deterministic() -> None:
    fvt = np.ones((1, 4, 1), dtype=np.float32)
    vp = np.zeros_like(fvt)
    vt = np.full_like(fvt, 90.0)
    component = [(0, i2, 0) for i2 in range(4)]
    first, first_diagnostics = skeletonize_fallback_components(fvt, vp, vt, [component])
    second, second_diagnostics = skeletonize_fallback_components(fvt, vp, vt, [component])
    np.testing.assert_array_equal(first, second)
    assert first_diagnostics == second_diagnostics
    assert {tuple(index) for index in np.argwhere(first)} == {(0, 1, 0)}


def test_fallback_uses_fvt_mask_and_mutates_skins_in_place() -> None:
    fvt = np.zeros((2, 2, 3), dtype=np.float32)
    fvt[0, 0, 0:2] = 1.0
    vp = np.zeros_like(fvt)
    vt = np.ones_like(fvt)
    skins: list[object] = []
    diagnostics: dict[str, object] = {}
    apply_boundary_skinner_fallback(
        skins,
        fvt,
        vp,
        vt,
        skinning_config=SyntheticSkinningConfig(boundary_skinner_fallback=True, min_skin_size=1),
        variant_spec=get_variant_spec("quality_boundary_skinner_fallback"),
        diagnostics=diagnostics,
    )
    assert diagnostics["skin_fallback_candidate_cell_count"] == 2
    assert diagnostics["fallback_cell_count"] == 2
    assert len(skins) == 1


@pytest.mark.parametrize(
    ("variant", "policy"),
    [
        ("quality_boundary_skinner_fallback_v2", "degraded_primary"),
        ("quality_boundary_skinner_fallback_v3", "degraded_primary_filtered"),
        ("quality_boundary_skinner_fallback_v4", "degraded_primary_skeletonized"),
        ("quality_boundary_skinner_fallback_v5", "degraded_primary_topology_guarded"),
    ],
)
def test_v2_to_v5_resolved_policy_is_recorded(variant: str, policy: str) -> None:
    empty = np.zeros((1, 1, 1), dtype=np.float32)
    diagnostics: dict[str, object] = {}
    apply_boundary_skinner_fallback(
        [],
        empty,
        empty,
        empty,
        skinning_config=SyntheticSkinningConfig(
            boundary_skinner_fallback=True, boundary_skinner_fallback_policy=policy
        ),
        variant_spec=get_variant_spec(variant),
        diagnostics=diagnostics,
    )
    assert diagnostics["fallback_policy"] == policy
    assert diagnostics["fallback_reason"] == "empty_primary_skin_without_positive_fvt"
