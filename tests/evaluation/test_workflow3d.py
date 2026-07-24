from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from pyosv.cells import FaultCell
from pyosv.evaluation.synthetic_quality.config import (
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from pyosv.evaluation.synthetic_quality.pipeline import run_voting_from_attributes
from pyosv.evaluation.synthetic_quality.stage_cache import PipelineStageCache
from pyosv.evaluation.synthetic_quality.variants import VariantSpec, get_variant_spec
from pyosv.evaluation.workflow3d import (
    PreparedAttributeIdentity,
    VolumeVotingControls,
    execute_workflow3d,
)
from pyosv.synthetic3d import make_single_vertical_plane_case


def _attributes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (5, 5, 5)
    ft = np.zeros(shape, dtype=np.float32)
    ft[2, 2, 2] = 1.0
    pt = np.zeros(shape, dtype=np.float32)
    tt = np.full(shape, 90.0, dtype=np.float32)
    return ft, pt, tt


def _identity(
    stage: str = "prepared-v1",
    shape: tuple[int, int, int] = (5, 5, 5),
) -> PreparedAttributeIdentity:
    return PreparedAttributeIdentity(
        dataset_fingerprint="fixture-v1",
        stage_fingerprint=stage,
        shape=shape,
        backend="fixture",
        scanner_thin_mode="none",
        edge_policy="keep",
    )


def _execute(
    *,
    identity: PreparedAttributeIdentity | None = None,
    controls: VolumeVotingControls | None = None,
    cache: PipelineStageCache | None = None,
):
    ft, pt, tt = _attributes()
    voting = SyntheticVotingConfig(
        ru=0,
        rv=1,
        rw=1,
        seed_distance=0,
        voter_thin_mode="normal",
    )
    variant = VariantSpec("fixture", experimental=False)
    return execute_workflow3d(
        ft=ft,
        pt=pt,
        tt=tt,
        attribute_identity=_identity() if identity is None else identity,
        voting_settings=voting,
        voting_controls=(
            VolumeVotingControls.resolve(voting, variant) if controls is None else controls
        ),
        skinning_settings=SyntheticSkinningConfig(enabled=False),
        variant_spec=variant,
        stage_cache=cache,
    )


def test_external_identity_and_controls_separate_semantic_keys() -> None:
    voting = SyntheticVotingConfig(ru=0, rv=1, rw=1)
    variant = VariantSpec("fixture", experimental=False)
    controls = VolumeVotingControls.resolve(voting, variant)

    first = _execute(identity=_identity("prepared-v1"), controls=controls)
    changed_source = _execute(identity=_identity("prepared-v2"), controls=controls)
    changed_control = _execute(
        identity=_identity("prepared-v1"),
        controls=replace(controls, final_normalization_smoothing=1.0),
    )

    assert first.stage_keys.attribute != changed_source.stage_keys.attribute
    assert first.stage_keys.voting != changed_control.stage_keys.voting
    assert not np.array_equal(first.fv, changed_control.fv)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("strain_max1", 0.0),
        ("strain_max2", 1.1),
        ("surface_smoothing1", -1.0),
        ("surface_smoothing2", float("nan")),
        ("support_min_fraction", -0.1),
        ("support_min_fraction", 1.1),
        ("support_exponent", float("inf")),
        ("orientation_smoothing", -1.0),
        ("final_normalization_smoothing", float("nan")),
        ("boundary_policy", "unknown"),
        ("orientation_backend", "unknown"),
    ),
)
def test_volume_voting_controls_reject_invalid_values(name: str, value: object) -> None:
    with pytest.raises(ValueError, match=name):
        replace(VolumeVotingControls(), **{name: value})


def test_cache_hits_return_independent_read_only_volumes() -> None:
    cache = PipelineStageCache()

    first = _execute(cache=cache)
    second = _execute(cache=cache)

    assert cache.stats.seed_hits == 1
    assert cache.stats.voting_hits == 1
    assert cache.stats.thinning_hits == 1
    assert np.array_equal(first.fv, second.fv)
    assert first.fv is not second.fv
    assert not first.fv.flags.writeable
    assert not second.fv.flags.writeable


def test_missing_identity_bypasses_cache_build_timer() -> None:
    def unexpected_timer(*args: object) -> object:
        raise AssertionError("unsafe cache path must not call the cache timer")

    cache = PipelineStageCache(build_timer=unexpected_timer)
    ft, pt, tt = _attributes()
    voting = SyntheticVotingConfig(ru=0, rv=1, rw=1, seed_distance=0)
    variant = VariantSpec("fixture", experimental=False)

    result = execute_workflow3d(
        ft=ft,
        pt=pt,
        tt=tt,
        attribute_identity=None,
        voting_settings=voting,
        voting_controls=VolumeVotingControls.resolve(voting, variant),
        skinning_settings=SyntheticSkinningConfig(enabled=False),
        variant_spec=variant,
        stage_cache=cache,
    )

    assert result.stage_keys.attribute is None
    assert cache.stats.seed_hits == cache.stats.seed_misses == 0


@pytest.mark.parametrize(
    ("variant_name", "skinning_enabled"),
    (
        ("current_default", False),
        ("current_default", True),
        ("boundary_aware_voter_v1", False),
        ("voter_thin_hybrid", False),
        ("boundary_seed_retention_v1", False),
    ),
)
def test_generic_core_matches_synthetic_path(
    variant_name: str,
    skinning_enabled: bool,
) -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    voting = SyntheticVotingConfig()
    skinning = SyntheticSkinningConfig(enabled=skinning_enabled)
    variant = get_variant_spec(variant_name)
    identity = _identity(shape=case.shape)

    core = execute_workflow3d(
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
        attribute_identity=identity,
        voting_settings=voting,
        voting_controls=VolumeVotingControls.resolve(voting, variant),
        skinning_settings=skinning,
        variant_spec=variant,
    )
    synthetic = run_voting_from_attributes(
        case,
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
        voting_config=voting,
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=skinning,
        variant_spec=variant,
        attribute_stage_key=identity.stage_key,
    )

    for core_name, synthetic_name in (
        ("fv", "fv_py"),
        ("vp", "vp_py"),
        ("vt", "vt_py"),
        ("fvt", "fvt_py"),
    ):
        np.testing.assert_array_equal(
            getattr(core, core_name),
            synthetic.artifacts.volumes[synthetic_name],
        )
    core_cells = [
        sorted(
            (
                float(cell.x1),
                float(cell.x2),
                float(cell.x3),
                int(cell.i1),
                int(cell.i2),
                int(cell.i3),
                float(cell.fl),
                float(cell.fp),
                float(cell.ft),
            )
            for cell in skin
        )
        for skin in core.skins
    ]
    synthetic_cells = [
        sorted(
            (
                cell["x1"],
                cell["x2"],
                cell["x3"],
                cell["i1"],
                cell["i2"],
                cell["i3"],
                cell["fl"],
                cell["fp"],
                cell["ft"],
            )
            for cell in skin["cells"]
        )
        for skin in synthetic.artifacts.skins_payload["skins"]
    ]
    assert core_cells == synthetic_cells
    assert len(core.skins) == synthetic.artifacts.skins_payload["skin_count"]
    assert (
        dict(core.diagnostics.voting)
        == (synthetic.report_payload["pyosv"]["voting"]["diagnostic_summary"])
    )
    if skinning_enabled:
        assert (
            dict(core.diagnostics.skinning) == (synthetic.report_payload["skinning"]["diagnostics"])
        )
    if variant.seed_policy != "default":
        assert (
            dict(core.diagnostics.seed or {})
            == (synthetic.report_payload["boundary_seed_retention"])
        )


def test_generic_core_matches_pre_extraction_golden() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    voting = SyntheticVotingConfig()
    skinning = SyntheticSkinningConfig(enabled=True)
    variant = get_variant_spec("current_default")

    result = execute_workflow3d(
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
        attribute_identity=_identity(shape=case.shape),
        voting_settings=voting,
        voting_controls=VolumeVotingControls.resolve(voting, variant),
        skinning_settings=skinning,
        variant_spec=variant,
    )

    # Fixed from the synthetic pipeline before the generic-core extraction.
    expected_volume_sha256 = {
        "fv": "2e9867e0a46d3022d695543513d4e491db3fbdfeb065881926f246f20c37ae8a",
        "vp": "3dae92d06f364fa6ea8e434c89a3d9d40cb995af8e46e50fff5ff91bd28fe112",
        "vt": "1485b0fb89803162cb8a7c8128d9c4189efd7407550984ad62fb406a1f006c67",
        "fvt": "01b222b444bbc9d0b7b355123a2d746489039e0d16f621bd07f17a69953c1e65",
    }
    assert {
        name: hashlib.sha256(
            np.asarray(getattr(result, name), dtype="<f4", order="C").tobytes()
        ).hexdigest()
        for name in expected_volume_sha256
    } == expected_volume_sha256

    cells = [
        sorted(
            [
                float(cell.x1),
                float(cell.x2),
                float(cell.x3),
                int(cell.i1),
                int(cell.i2),
                int(cell.i3),
                float(cell.fl),
                float(cell.fp),
                float(cell.ft),
            ]
            for cell in skin
        )
        for skin in result.skins
    ]

    def json_sha256(value: object) -> str:
        payload = json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    assert json_sha256(cells) == (
        "d69c93830b1b67e88fa1f5332cc464f830b4a3bcc8fbeb77c0facb10a8d4739b"
    )
    assert json_sha256(dict(result.diagnostics.voting)) == (
        "5882a130a398d273211c1947ad93a3b031d53a5bf3a4f7e05762f2a993d2ac9a"
    )


def test_skinning_preserves_primary_payload_before_boundary_fallback() -> None:
    events: list[str] = []

    def primary_skinner(
        *args: object,
        diagnostics: dict[str, object],
        **kwargs: object,
    ) -> list[list[FaultCell]]:
        events.append("primary")
        diagnostics["primary_marker"] = 1
        return [[FaultCell(2, 2, 2, 1.0, 0.0, 90.0)]]

    def boundary_fallback(
        skins: list[object],
        *args: object,
        skinning_config: SyntheticSkinningConfig,
        diagnostics: dict[str, object],
        **kwargs: object,
    ) -> None:
        events.append("fallback")
        assert skinning_config.boundary_skinner_fallback
        assert diagnostics["primary_marker"] == 1
        skins[:] = [[FaultCell(0, 0, 0, 1.0, 0.0, 90.0)]]
        diagnostics["fallback_used"] = True

    ft, pt, tt = _attributes()
    voting = SyntheticVotingConfig(ru=0, rv=1, rw=1, seed_distance=0)
    variant = VariantSpec("fixture", experimental=False)
    result = execute_workflow3d(
        ft=ft,
        pt=pt,
        tt=tt,
        attribute_identity=_identity(),
        voting_settings=voting,
        voting_controls=VolumeVotingControls.resolve(voting, variant),
        skinning_settings=SyntheticSkinningConfig(
            enabled=True,
            boundary_skinner_fallback=True,
        ),
        variant_spec=variant,
        primary_skinner=primary_skinner,
        boundary_fallback_runner=boundary_fallback,
    )

    assert events == ["primary", "fallback"]
    assert result.skin.primary_mask[2, 2, 2]
    assert not result.skin.primary_mask[0, 0, 0]
    assert [(cell.i3, cell.i2, cell.i1) for cell in result.skins[0]] == [(0, 0, 0)]
    assert result.diagnostics.skinning["fallback_used"] is True


def test_timer_exception_does_not_cache_partial_voting_result() -> None:
    cache = PipelineStageCache()
    ft, pt, tt = _attributes()
    voting = SyntheticVotingConfig(ru=0, rv=1, rw=1, seed_distance=0)
    variant = VariantSpec("fixture", experimental=False)
    kwargs = {
        "ft": ft,
        "pt": pt,
        "tt": tt,
        "attribute_identity": _identity(),
        "voting_settings": voting,
        "voting_controls": VolumeVotingControls.resolve(voting, variant),
        "skinning_settings": SyntheticSkinningConfig(enabled=False),
        "variant_spec": variant,
        "stage_cache": cache,
    }

    def fail_voting_timer(
        stage: str,
        semantic_key: object,
        builder: Callable[[], object],
    ) -> object:
        if stage == "voting_volume":
            raise RuntimeError("timer failed")
        return builder()

    with pytest.raises(RuntimeError, match="timer failed"):
        execute_workflow3d(**kwargs, stage_timer=fail_voting_timer)

    assert cache.stats.seed_misses == 1
    assert cache.stats.voting_misses == 1
    assert cache.stats.thinning_misses == 0
    execute_workflow3d(**kwargs)
    assert cache.stats.seed_hits == 1
    assert cache.stats.voting_misses == 2
    assert cache.stats.thinning_misses == 1


def test_injected_skinner_exception_does_not_cache_partial_result() -> None:
    cache = PipelineStageCache()
    ft, pt, tt = _attributes()
    voting = SyntheticVotingConfig(ru=0, rv=1, rw=1, seed_distance=0)
    variant = VariantSpec("fixture", experimental=False)
    kwargs = {
        "ft": ft,
        "pt": pt,
        "tt": tt,
        "attribute_identity": _identity(),
        "voting_settings": voting,
        "voting_controls": VolumeVotingControls.resolve(voting, variant),
        "skinning_settings": SyntheticSkinningConfig(enabled=True),
        "variant_spec": variant,
        "stage_cache": cache,
    }

    def fail_primary(*args: object, **kwargs: object) -> list[object]:
        raise RuntimeError("builder failed")

    with pytest.raises(RuntimeError, match="builder failed"):
        execute_workflow3d(**kwargs, primary_skinner=fail_primary)

    assert cache.stats.primary_skinning_misses == 0

    def empty_primary(*args: object, **kwargs: object) -> list[object]:
        return []

    execute_workflow3d(**kwargs, primary_skinner=empty_primary)
    execute_workflow3d(**kwargs, primary_skinner=empty_primary)
    assert cache.stats.primary_skinning_hits == cache.stats.primary_skinning_misses == 0

    with pytest.raises(RuntimeError, match="builder failed"):
        execute_workflow3d(**kwargs, primary_skinner=fail_primary)
    assert cache.stats.primary_skinning_misses == 0


def test_external_seed_target_hides_unsafe_downstream_keys() -> None:
    cache = PipelineStageCache()
    ft, pt, tt = _attributes()
    voting = SyntheticVotingConfig(ru=0, rv=1, rw=1, seed_distance=0)
    variant = get_variant_spec("boundary_seed_retention_v1")

    result = execute_workflow3d(
        ft=ft,
        pt=pt,
        tt=tt,
        attribute_identity=_identity(),
        voting_settings=voting,
        voting_controls=VolumeVotingControls.resolve(voting, variant),
        skinning_settings=SyntheticSkinningConfig(enabled=True),
        variant_spec=variant,
        stage_cache=cache,
        fvt_recenter_target=np.zeros(ft.shape, dtype=np.float32),
        fvt_recenter_target_source="custom_target",
    )

    assert result.stage_keys.attribute is not None
    assert result.stage_keys.seed is None
    assert result.stage_keys.voting is None
    assert result.stage_keys.thinning is None
    assert result.stage_keys.final_thinning is None
    assert result.stage_keys.primary_skinning is None
    assert cache.stats.seed_hits == cache.stats.seed_misses == 0
    assert cache.stats.primary_skinning_hits == cache.stats.primary_skinning_misses == 0


def test_external_recenter_target_hides_only_affected_keys() -> None:
    cache = PipelineStageCache()
    ft, pt, tt = _attributes()
    voting = SyntheticVotingConfig(ru=0, rv=1, rw=1, seed_distance=0)
    variant = get_variant_spec("voter_thin_hybrid_v2_recenter_scanner_target")
    common = {
        "ft": ft,
        "pt": pt,
        "tt": tt,
        "attribute_identity": _identity(),
        "voting_settings": voting,
        "voting_controls": VolumeVotingControls.resolve(voting, variant),
        "skinning_settings": SyntheticSkinningConfig(enabled=True),
        "variant_spec": variant,
        "stage_cache": cache,
        "fvt_recenter_target_source": "custom_target",
    }

    first = execute_workflow3d(
        **common,
        fvt_recenter_target=np.zeros(ft.shape, dtype=np.float32),
    )
    second = execute_workflow3d(
        **common,
        fvt_recenter_target=np.ones(ft.shape, dtype=np.float32),
    )

    assert first.stage_keys.voting == second.stage_keys.voting
    assert first.stage_keys.thinning == second.stage_keys.thinning
    assert first.stage_keys.final_thinning is None
    assert second.stage_keys.final_thinning is None
    assert first.stage_keys.primary_skinning is None
    assert second.stage_keys.primary_skinning is None
    assert cache.stats.voting_hits == 1
    assert cache.stats.thinning_hits == 1
    assert cache.stats.primary_skinning_hits == cache.stats.primary_skinning_misses == 0


def test_stateful_injected_primary_skinner_bypasses_cache() -> None:
    cache = PipelineStageCache()
    ft, pt, tt = _attributes()
    voting = SyntheticVotingConfig(ru=0, rv=1, rw=1, seed_distance=0)
    variant = VariantSpec("fixture", experimental=False)
    common = {
        "ft": ft,
        "pt": pt,
        "tt": tt,
        "attribute_identity": _identity(),
        "voting_settings": voting,
        "voting_controls": VolumeVotingControls.resolve(voting, variant),
        "skinning_settings": SyntheticSkinningConfig(enabled=True),
        "variant_spec": variant,
        "stage_cache": cache,
    }

    class StatefulSkinner:
        location = 1

        def __call__(
            self,
            *args: object,
            **kwargs: object,
        ) -> list[list[FaultCell]]:
            i = self.location
            return [[FaultCell(i, i, i, 1.0, 0.0, 90.0)]]

    skinner = StatefulSkinner()
    first = execute_workflow3d(**common, primary_skinner=skinner)
    skinner.location = 3
    second = execute_workflow3d(**common, primary_skinner=skinner)

    assert first.stage_keys.primary_skinning is None
    assert second.stage_keys.primary_skinning is None
    assert [(cell.i3, cell.i2, cell.i1) for cell in first.skins[0]] == [(1, 1, 1)]
    assert [(cell.i3, cell.i2, cell.i1) for cell in second.skins[0]] == [(3, 3, 3)]
    assert cache.stats.primary_skinning_hits == cache.stats.primary_skinning_misses == 0


def test_injected_primary_skinner_with_explicit_identity_uses_cache() -> None:
    cache = PipelineStageCache()
    calls = 0

    def primary_skinner(*args: object, **kwargs: object) -> list[object]:
        nonlocal calls
        calls += 1
        return []

    ft, pt, tt = _attributes()
    voting = SyntheticVotingConfig(ru=0, rv=1, rw=1, seed_distance=0)
    variant = VariantSpec("fixture", experimental=False)
    common = {
        "ft": ft,
        "pt": pt,
        "tt": tt,
        "attribute_identity": _identity(),
        "voting_settings": voting,
        "voting_controls": VolumeVotingControls.resolve(voting, variant),
        "skinning_settings": SyntheticSkinningConfig(enabled=True),
        "variant_spec": variant,
        "stage_cache": cache,
        "primary_skinner": primary_skinner,
        "primary_skinner_identity": "test.primary-skinner.v1",
    }

    first = execute_workflow3d(**common)
    second = execute_workflow3d(**common)

    assert first.stage_keys.primary_skinning == second.stage_keys.primary_skinning
    assert calls == 1
    assert cache.stats.primary_skinning_misses == 1
    assert cache.stats.primary_skinning_hits == 1


@pytest.mark.parametrize(
    ("control_name", "change"),
    (
        (
            "strain",
            {"strain_max1": 0.5, "strain_max2": 0.5},
        ),
        (
            "surface_smoothing",
            {"surface_smoothing1": 0.0, "surface_smoothing2": 0.0},
        ),
        ("orientation_smoothing", {"orientation_smoothing": 0.0}),
        ("final_normalization", {"final_normalization_smoothing": 1.0}),
    ),
)
def test_each_voting_control_changes_key_and_numerical_path(
    control_name: str,
    change: dict[str, float],
) -> None:
    rng = np.random.default_rng(492)
    shape = (9, 9, 9)
    ft = rng.random(shape, dtype=np.float32)
    ft[ft < np.float32(0.82)] = np.float32(0.0)
    pt = rng.uniform(0.0, 360.0, shape).astype(np.float32)
    tt = rng.uniform(60.0, 120.0, shape).astype(np.float32)
    voting = SyntheticVotingConfig(
        ru=2,
        rv=2,
        rw=2,
        seed_distance=1,
        seed_threshold=0.8,
        attribute_smoothing=1,
    )
    variant = VariantSpec("fixture", experimental=False)
    controls = VolumeVotingControls.resolve(voting, variant)
    common = {
        "ft": ft,
        "pt": pt,
        "tt": tt,
        "attribute_identity": _identity(shape=shape),
        "voting_settings": voting,
        "skinning_settings": SyntheticSkinningConfig(enabled=False),
        "variant_spec": variant,
    }

    baseline = execute_workflow3d(**common, voting_controls=controls)
    changed = execute_workflow3d(
        **common,
        voting_controls=replace(controls, **change),
    )

    assert baseline.stage_keys.voting != changed.stage_keys.voting, control_name
    assert any(
        not np.array_equal(getattr(baseline, name), getattr(changed, name))
        for name in ("fv", "vp", "vt", "fvt")
    ), control_name
