from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import weakref

import numpy as np
import pytest

from pyosv.evaluation.synthetic_quality import application, pipeline
from pyosv.evaluation.synthetic_quality.config import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from pyosv.evaluation.synthetic_quality.models import OrientationField3D
from pyosv.evaluation.synthetic_quality.runner import (
    PreparedCaseInputs,
    PreparedScannerInput,
    prepare_case_inputs,
    run_case_variant,
)
from pyosv.evaluation.synthetic_quality.stage_cache import (
    AttributeStageKey,
    DownstreamScalarEvidence,
    DownstreamScalarEvidenceCache,
    PipelineStageCache,
    PrimarySkinningStageResult,
    SeedStageResult,
    ThinningStageResult,
    VotingStageResult,
)
from pyosv.evaluation.synthetic_quality.stage_keys import (
    build_final_thinning_stage_key,
    build_primary_skinning_stage_key,
    build_seed_stage_key,
    build_thinning_scalar_evidence_key,
    build_thinning_stage_key,
    build_voting_scalar_evidence_key,
    build_voting_stage_key,
)
from pyosv.evaluation.synthetic_quality.variants import get_variant_spec
from pyosv.evaluation.workflow3d import VolumeVotingControls
from pyosv.evaluation.reporting.artifacts import write_case_skins_json, write_case_volumes
from pyosv.evaluation.reporting.csv_v1 import write_summary_csv
from pyosv.evaluation.reporting.json_v1 import write_metrics_json
from pyosv.synthetic3d import Synthetic3DCase
from pyosv.synthetic3d import make_single_vertical_plane_case
from pyosv.cells import (
    FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED,
    FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED,
    FaultCell,
)
from pyosv.skin import FaultSkin


def _run_variant(
    variant: str,
    *,
    cache: PipelineStageCache | None,
    scalar_cache: DownstreamScalarEvidenceCache | None = None,
    case: Synthetic3DCase | None = None,
    voting_config: SyntheticVotingConfig = SyntheticVotingConfig(),
    skinning_config: SyntheticSkinningConfig = SyntheticSkinningConfig(enabled=False),
):
    if case is None:
        case = make_single_vertical_plane_case((9, 9, 9))
    return run_case_variant(
        case,
        voting_config=voting_config,
        scanner_config=SyntheticScannerConfig(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=skinning_config,
        variant=variant,
        input_mode="oracle",
        scanner_backend_matrix=False,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=False,
        stage_cache=cache,
        scalar_evidence_cache=scalar_cache,
    )


def _scalar_stage_keys(case: Synthetic3DCase, *, target_source: str = "oracle_ft"):
    voting_config = SyntheticVotingConfig()
    variant_spec = get_variant_spec("current_default")
    attribute = AttributeStageKey(case.case_id, case.shape, "oracle")
    seed = build_seed_stage_key(
        attribute_key=attribute,
        voting_config=voting_config,
        variant_spec=variant_spec,
        target_source=target_source,
    )
    voting = build_voting_stage_key(
        seed_key=seed,
        voting_config=voting_config,
        variant_spec=variant_spec,
        voting_controls=VolumeVotingControls.resolve(voting_config, variant_spec),
    )
    thinning = build_thinning_stage_key(
        voting_key=voting,
        voting_config=voting_config,
        variant_spec=variant_spec,
    )
    final_thinning = build_final_thinning_stage_key(
        thinning_key=thinning,
        variant_spec=variant_spec,
        target_source=target_source,
    )
    assert voting is not None
    assert thinning is not None
    assert final_thinning is not None
    return voting, thinning, final_thinning


def test_identical_seed_and_voting_stages_are_called_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"seed": 0, "voting": 0}
    original_seed = pipeline.OptimalSurfaceVoter.pick_seeds
    original_voting = pipeline.OptimalSurfaceVoter.apply_voting_from_seeds

    def count_seed(self, *args, **kwargs):
        calls["seed"] += 1
        return original_seed(self, *args, **kwargs)

    def count_voting(self, *args, **kwargs):
        calls["voting"] += 1
        return original_voting(self, *args, **kwargs)

    monkeypatch.setattr(pipeline.OptimalSurfaceVoter, "pick_seeds", count_seed)
    monkeypatch.setattr(pipeline.OptimalSurfaceVoter, "apply_voting_from_seeds", count_voting)
    case = make_single_vertical_plane_case((9, 9, 9))
    common = dict(
        voting_config=SyntheticVotingConfig(),
        scanner_config=SyntheticScannerConfig(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        input_mode="oracle",
        scanner_backend_matrix=False,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=False,
    )
    run_case_variant(case, variant="current_default", stage_cache=None, **common)
    run_case_variant(case, variant="voter_thin_normal", stage_cache=None, **common)
    assert calls == {"seed": 2, "voting": 2}

    calls.update(seed=0, voting=0)
    cache = PipelineStageCache(case)
    run_case_variant(case, variant="current_default", stage_cache=cache, **common)
    run_case_variant(case, variant="voter_thin_normal", stage_cache=cache, **common)

    assert calls == {"seed": 1, "voting": 1}
    assert cache.stats.seed_hits == 1
    assert cache.stats.voting_hits == 1


def test_pipeline_stage_build_timer_wraps_shared_misses_in_stage_order() -> None:
    stages: list[str] = []

    def record_build(stage, semantic_key, operation):
        assert semantic_key is not None
        stages.append(stage)
        return operation()

    case = make_single_vertical_plane_case((9, 9, 9))
    cache = PipelineStageCache(case, build_timer=record_build)
    skinning_config = SyntheticSkinningConfig()
    _run_variant(
        "current_default",
        cache=cache,
        case=case,
        skinning_config=skinning_config,
    )
    _run_variant(
        "quality_boundary_skinner_fallback",
        cache=cache,
        case=case,
        skinning_config=skinning_config,
    )

    assert stages == [
        "seed_selection",
        "voting_volume",
        "base_thinning",
        "primary_skinning",
    ]
    assert cache.stats.seed_misses == cache.stats.seed_hits == 1
    assert cache.stats.voting_misses == cache.stats.voting_hits == 1
    assert cache.stats.thinning_misses == 1
    assert cache.stats.thinning_hits == 0
    assert len(cache._final_thinning) == 1
    assert cache.stats.primary_skinning_misses == cache.stats.primary_skinning_hits == 1


def test_pipeline_stage_build_timer_skips_primary_when_skinning_is_disabled() -> None:
    stages: list[str] = []

    def record_build(stage, semantic_key, operation):
        assert semantic_key is not None
        stages.append(stage)
        return operation()

    cache = PipelineStageCache(build_timer=record_build)
    _run_variant("current_default", cache=cache)

    assert stages == ["seed_selection", "voting_volume", "base_thinning"]
    assert cache.stats.primary_skinning_misses == 0


def test_pipeline_stage_build_exceptions_do_not_register_partial_entries() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    voting_key, thinning_key, _ = _scalar_stage_keys(case)
    primary_key = build_primary_skinning_stage_key(
        thinning_key=thinning_key,
        skinning_config=SyntheticSkinningConfig(),
        variant_spec=get_variant_spec("current_default"),
        target_source="oracle_ft",
    )
    assert primary_key is not None
    shape = (1, 1, 1)
    builds = (
        (
            "get_or_build_seed",
            voting_key.seed,
            SeedStageResult(seeds=()),
            "_seeds",
        ),
        (
            "get_or_build_voting",
            voting_key,
            VotingStageResult(
                fv=np.zeros(shape, dtype=np.float32),
                vp=np.zeros(shape, dtype=np.float32),
                vt=np.zeros(shape, dtype=np.float32),
                diagnostic_items=(),
            ),
            "_voting",
        ),
        (
            "get_or_build_thinning",
            thinning_key,
            ThinningStageResult(fvt=np.zeros(shape, dtype=np.float32)),
            "_thinning",
        ),
        (
            "get_or_build_primary_skinning",
            primary_key,
            PrimarySkinningStageResult.from_skins([], {}),
            "_primary_skinning",
        ),
    )

    for method_name, key, completed, entries_name in builds:
        cache = PipelineStageCache(case)
        method = getattr(cache, method_name)

        def fail():
            raise RuntimeError("stage failed")

        with pytest.raises(RuntimeError, match="stage failed"):
            method(key, fail)

        assert not getattr(cache, entries_name)
        assert method(key, lambda: completed) is completed
        assert len(getattr(cache, entries_name)) == 1


def test_pipeline_stage_timer_exception_does_not_register_partial_entry() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    voting_key, _, _ = _scalar_stage_keys(case)
    operation_called = False

    def fail_timer(stage, semantic_key, operation):
        assert stage == "seed_selection"
        assert semantic_key == voting_key.seed
        raise RuntimeError("timer failed")

    cache = PipelineStageCache(case, build_timer=fail_timer)

    def build_seed():
        nonlocal operation_called
        operation_called = True
        return SeedStageResult(seeds=())

    with pytest.raises(RuntimeError, match="timer failed"):
        cache.get_or_build_seed(voting_key.seed, build_seed)

    assert not operation_called
    assert not cache._seeds
    cache.build_timer = None
    assert cache.get_or_build_seed(voting_key.seed, build_seed).seeds == ()
    assert cache.stats.seed_misses == 2


def test_scalar_evidence_keys_include_every_invalidation_dimension() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    voting, thinning, final_thinning = _scalar_stage_keys(case)
    truth_config = SyntheticTruthMetricConfig()
    voting_kwargs = {
        "case_id": case.case_id,
        "case_token": id(case),
        "shape": case.shape,
        "voting_key": voting,
        "truth_metric_config": truth_config,
        "contract_version": 3,
    }
    voting_keys = {
        build_voting_scalar_evidence_key(**voting_kwargs),
        build_voting_scalar_evidence_key(**{**voting_kwargs, "case_id": "other_case"}),
        build_voting_scalar_evidence_key(**{**voting_kwargs, "case_token": id(case) + 1}),
        build_voting_scalar_evidence_key(**{**voting_kwargs, "shape": (11, 9, 9)}),
        build_voting_scalar_evidence_key(
            **{
                **voting_kwargs,
                "truth_metric_config": replace(truth_config, buffer_radius=3.0),
            }
        ),
        build_voting_scalar_evidence_key(**{**voting_kwargs, "contract_version": 4}),
    }
    assert None not in voting_keys
    assert len(voting_keys) == 6

    recenter_spec = get_variant_spec("voter_thin_hybrid_v2_recenter_scanner_target")
    recentered_scanner = build_final_thinning_stage_key(
        thinning_key=thinning,
        variant_spec=recenter_spec,
        target_source="scanner_fet",
    )
    recentered_custom = build_final_thinning_stage_key(
        thinning_key=thinning,
        variant_spec=recenter_spec,
        target_source="custom_target",
    )
    thinning_kwargs = {
        "case_id": case.case_id,
        "case_token": id(case),
        "shape": case.shape,
        "truth_metric_config": truth_config,
        "contract_version": 3,
    }
    thinning_keys = {
        build_thinning_scalar_evidence_key(
            **thinning_kwargs,
            final_thinning_key=final_thinning,
        ),
        build_thinning_scalar_evidence_key(
            **thinning_kwargs,
            final_thinning_key=recentered_scanner,
        ),
        build_thinning_scalar_evidence_key(
            **thinning_kwargs,
            final_thinning_key=recentered_custom,
        ),
    }
    assert None not in thinning_keys
    assert len(thinning_keys) == 3


def test_downstream_scalar_evidence_is_scalar_only_and_recursively_immutable() -> None:
    evidence = DownstreamScalarEvidence(
        array_summary={"shape": [9, 9, 9], "mean": np.float32(0.25)},
        top_truth_count={"overlap": {"value": 1.0}},
        positive_top_truth_count={"distance": [0.0, 1.0]},
        edge_top_truth_count={"ratio": 0.0},
        edge_positive_top_truth_count={"ratio": 0.5},
    )

    assert json.loads(json.dumps(evidence.array_summary)) == {
        "shape": [9, 9, 9],
        "mean": 0.25,
    }
    with pytest.raises(TypeError, match="immutable"):
        evidence.array_summary["mean"] = 1.0
    with pytest.raises(TypeError, match="immutable"):
        evidence.array_summary["shape"][0] = 1
    with pytest.raises(TypeError, match="immutable"):
        evidence.top_truth_count["overlap"]["value"] = 0.0
    with pytest.raises(ValueError, match="scalar-only"):
        DownstreamScalarEvidence(
            array_summary={"array": np.zeros(1, dtype=np.float32)},
            top_truth_count={},
            positive_top_truth_count={},
            edge_top_truth_count={},
            edge_positive_top_truth_count={},
        )


def test_evidence_builder_exception_does_not_register_partial_cache_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    cache = DownstreamScalarEvidenceCache(case)
    monkeypatch.setattr(
        pipeline,
        "build_voting_scalar_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("evidence failed")),
    )

    with pytest.raises(RuntimeError, match="evidence failed"):
        _run_variant("current_default", cache=None, scalar_cache=cache, case=case)

    assert cache.stats.voting_builds == 0
    assert cache.stats.thinning_builds == 0
    assert not cache._voting
    assert not cache._thinning


def test_replaced_prepared_oracle_bypasses_stage_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = pipeline.OptimalSurfaceVoter.pick_seeds

    def count_seed(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline.OptimalSurfaceVoter, "pick_seeds", count_seed)
    case = make_single_vertical_plane_case((9, 9, 9))
    prepared = prepare_case_inputs(
        case,
        scanner_config=SyntheticScannerConfig(),
        input_mode="oracle",
        scanner_backend_matrix=False,
    )
    custom_oracle = OrientationField3D(
        ft=np.zeros_like(prepared.oracle.ft),
        pt=prepared.oracle.pt.copy(),
        tt=prepared.oracle.tt.copy(),
    )
    common = dict(
        voting_config=SyntheticVotingConfig(),
        scanner_config=SyntheticScannerConfig(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        variant="current_default",
        input_mode="oracle",
        scanner_backend_matrix=False,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=False,
    )
    stages: list[str] = []

    def record_build(stage, semantic_key, operation):
        assert semantic_key is not None
        stages.append(stage)
        return operation()

    cache = PipelineStageCache(case, build_timer=record_build)

    run_case_variant(case, prepared_inputs=prepared, stage_cache=cache, **common)
    bypassed = run_case_variant(
        case,
        prepared_inputs=PreparedCaseInputs(case, custom_oracle, None),
        stage_cache=cache,
        **common,
    )
    legacy = run_case_variant(
        case,
        prepared_inputs=PreparedCaseInputs(case, custom_oracle, None),
        stage_cache=None,
        **common,
    )

    assert calls == 3
    assert stages == ["seed_selection", "voting_volume", "base_thinning"]
    _assert_nested_equal(bypassed.report_payload, legacy.report_payload)
    _assert_nested_equal(bypassed.artifacts.volumes, legacy.artifacts.volumes)


def test_replaced_prepared_scanner_bypasses_stage_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = pipeline.OptimalSurfaceVoter.pick_seeds

    def count_seed(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline.OptimalSurfaceVoter, "pick_seeds", count_seed)
    case = make_single_vertical_plane_case((9, 9, 9))
    scanner_config = SyntheticScannerConfig()
    prepared = prepare_case_inputs(
        case,
        scanner_config=scanner_config,
        input_mode="scanner",
        scanner_backend_matrix=False,
    )
    assert prepared.scanner is not None
    custom_volumes = dict(prepared.scanner.selected.volumes)
    custom_volumes["scanner_fet"] = np.zeros_like(custom_volumes["scanner_fet"])
    custom_attributes = replace(prepared.scanner.selected, volumes=custom_volumes)
    custom_scanner = PreparedScannerInput(
        config=scanner_config,
        selected=custom_attributes,
        by_backend={scanner_config.backend: custom_attributes},
    )
    common = dict(
        voting_config=SyntheticVotingConfig(),
        scanner_config=scanner_config,
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        input_mode="scanner",
        scanner_backend_matrix=False,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=False,
    )
    cache = PipelineStageCache(case)

    run_case_variant(
        case,
        variant="current_default",
        prepared_inputs=prepared,
        stage_cache=cache,
        **common,
    )
    run_case_variant(
        case,
        variant="voter_thin_normal",
        prepared_inputs=prepared,
        stage_cache=cache,
        **common,
    )
    assert calls == 1
    run_case_variant(
        case,
        variant="current_default",
        prepared_inputs=replace(prepared, scanner=custom_scanner),
        stage_cache=cache,
        **common,
    )

    assert calls == 2


def test_identical_base_thinning_is_called_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = pipeline.OptimalSurfaceVoter.thin

    def count_thinning(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pipeline.OptimalSurfaceVoter, "thin", count_thinning)
    case = make_single_vertical_plane_case((9, 9, 9))
    cache = PipelineStageCache(case)
    _run_variant("current_default", cache=cache, case=case)
    _run_variant("quality_boundary_skinner_fallback", cache=cache, case=case)

    assert calls == 1
    assert cache.stats.thinning_misses == 1
    assert cache.stats.thinning_hits == 0
    assert len(cache._final_thinning) == 1


def test_thinning_key_distinguishes_mode_and_sigma_but_not_post_policy() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    cache = PipelineStageCache(case)
    _run_variant("current_default", cache=cache, case=case)
    _run_variant("voter_thin_normal", cache=cache, case=case)
    _run_variant(
        "current_default",
        cache=cache,
        case=case,
        voting_config=SyntheticVotingConfig(reference_thin_sigma=2.0),
    )

    assert cache.stats.thinning_misses == 3

    post_cache = PipelineStageCache(case)
    _run_variant("voter_thin_hybrid_v2", cache=post_cache, case=case)
    _run_variant(
        "voter_thin_hybrid_v2_recenter_scanner_target",
        cache=post_cache,
        case=case,
    )
    assert post_cache.stats.thinning_misses == 1
    assert post_cache.stats.thinning_hits == 1


def test_fallback_only_variants_share_primary_skinning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = pipeline.find_synthetic_skins

    def count_primary(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "find_synthetic_skins", count_primary)
    case = make_single_vertical_plane_case((9, 9, 9))
    cache = PipelineStageCache(case)
    for variant in (
        "current_default",
        "quality_boundary_skinner_fallback",
        "quality_boundary_skinner_fallback_v2",
    ):
        _run_variant(
            variant,
            cache=cache,
            case=case,
            skinning_config=SyntheticSkinningConfig(),
        )

    assert calls == 1
    assert cache.stats.primary_skinning_misses == 1
    assert cache.stats.primary_skinning_hits == 2


@pytest.mark.parametrize(
    "changed",
    (
        {"ru": 11},
        {"min_likelihood": 0.4},
        {"min_skin_size": 2},
        {"reskin": False},
        {"reskin_policy": "reference_dense_v1"},
    ),
)
def test_primary_skinning_key_distinguishes_growth_settings(changed: dict[str, object]) -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    cache = PipelineStageCache(case)
    config = SyntheticSkinningConfig()
    _run_variant("current_default", cache=cache, case=case, skinning_config=config)
    _run_variant(
        "current_default",
        cache=cache,
        case=case,
        skinning_config=replace(config, **changed),
    )

    assert cache.stats.primary_skinning_misses == 2


def test_post_thinning_policy_does_not_share_primary_skinning() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    cache = PipelineStageCache(case)
    config = SyntheticSkinningConfig()
    _run_variant("voter_thin_hybrid_v2", cache=cache, case=case, skinning_config=config)
    _run_variant(
        "voter_thin_hybrid_v2_recenter_scanner_target",
        cache=cache,
        case=case,
        skinning_config=config,
    )

    assert cache.stats.thinning_hits == 1
    assert cache.stats.primary_skinning_misses == 2


def test_custom_post_thinning_target_bypasses_primary_skinning_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = pipeline.find_synthetic_skins

    def count_primary(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline, "find_synthetic_skins", count_primary)
    case = make_single_vertical_plane_case((9, 9, 9))
    cache = PipelineStageCache(case)
    common = dict(
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
        voting_config=SyntheticVotingConfig(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(),
        variant_spec=get_variant_spec("voter_thin_hybrid_v2_recenter_scanner_target"),
        fvt_recenter_target_source="custom_target",
        stage_cache=cache,
        attribute_stage_key=AttributeStageKey(case.case_id, case.shape, "oracle"),
    )

    pipeline.run_voting_from_attributes(
        case,
        fvt_recenter_target=np.zeros(case.shape, dtype=np.float32),
        **common,
    )
    pipeline.run_voting_from_attributes(
        case,
        fvt_recenter_target=np.ones(case.shape, dtype=np.float32),
        **common,
    )

    assert calls == 2
    assert cache.stats.voting_hits == 1


def test_primary_skinning_snapshot_clones_cells_links_skins_and_diagnostics() -> None:
    above = FaultCell(
        1,
        2,
        3,
        0.9,
        20,
        70,
        generation=FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED,
        reskin_support=0.7,
    )
    below = FaultCell(
        1,
        2,
        4,
        0.8,
        21,
        69,
        generation=FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED,
        reskin_support=0.6,
    )
    object.__setattr__(above, "cb", below)
    object.__setattr__(below, "ca", above)
    result = PrimarySkinningStageResult.from_skins(
        [FaultSkin.from_cells((above, below))],
        {"nested": ["clean"]},
    )

    first_skins, first_diagnostics = result.clone()
    second_skins, second_diagnostics = result.clone()
    first_skins[0].append(FaultCell(0, 0, 0, 0.1, 0, 90))
    object.__setattr__(next(iter(first_skins[0])), "cb", None)
    first_diagnostics["nested"].append("mutated")

    second_cells = list(second_skins[0])
    assert len(second_cells) == 2
    assert second_cells[0].cb is second_cells[1]
    assert second_cells[1].ca is second_cells[0]
    assert [cell.generation for cell in second_cells] == [
        FAULT_CELL_GENERATION_DENSE_RESKIN_OBSERVED,
        FAULT_CELL_GENERATION_DENSE_RESKIN_GENERATED,
    ]
    assert [cell.reskin_support for cell in second_cells] == [0.7, 0.6]
    assert first_skins[0].cells[0] is not second_cells[0]
    assert first_skins[0].cells[1] is not second_cells[1]
    assert second_diagnostics == {"nested": ["clean"]}


@pytest.mark.parametrize(
    "variant",
    (
        "boundary_aware_voter_v1",
        "no_surface_orientation_smoothing",
        "final_norm_smoothing_1",
        "surface_support_weighted",
    ),
)
def test_voting_output_field_changes_produce_distinct_keys(variant: str) -> None:
    cache = PipelineStageCache()
    case = make_single_vertical_plane_case((9, 9, 9))
    common = dict(
        voting_config=SyntheticVotingConfig(),
        scanner_config=SyntheticScannerConfig(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        input_mode="oracle",
        scanner_backend_matrix=False,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=False,
        stage_cache=cache,
    )
    run_case_variant(case, variant="current_default", **common)
    run_case_variant(case, variant=variant, **common)

    assert cache.stats.seed_hits == 1
    assert cache.stats.voting_misses == 2


def test_seed_policy_change_produces_distinct_seed_and_voting_keys() -> None:
    cache = PipelineStageCache()
    case = make_single_vertical_plane_case((9, 9, 9))
    common = dict(
        voting_config=SyntheticVotingConfig(),
        scanner_config=SyntheticScannerConfig(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        input_mode="oracle",
        scanner_backend_matrix=False,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=False,
        stage_cache=cache,
    )
    run_case_variant(case, variant="current_default", **common)
    run_case_variant(case, variant="boundary_seed_retention_v1", **common)

    assert cache.stats.seed_misses == 2
    assert cache.stats.voting_misses == 2


def _assert_nested_equal(actual: object, expected: object) -> None:
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, Mapping):
        assert isinstance(actual, Mapping)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    assert actual == expected


def test_scalar_cache_reuse_is_numerically_identical_to_legacy_execution() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    legacy = _run_variant("current_default", cache=None, case=case)
    scalar_cache = DownstreamScalarEvidenceCache(case)

    first = _run_variant(
        "current_default",
        cache=None,
        scalar_cache=scalar_cache,
        case=case,
    )
    reused = _run_variant(
        "current_default",
        cache=None,
        scalar_cache=scalar_cache,
        case=case,
    )

    assert scalar_cache.stats.voting_builds == scalar_cache.stats.thinning_builds == 1
    assert scalar_cache.stats.voting_reuses == scalar_cache.stats.thinning_reuses == 1
    for evaluation in (first, reused):
        _assert_nested_equal(evaluation.report_payload, legacy.report_payload)
        _assert_nested_equal(evaluation.artifacts.volumes, legacy.artifacts.volumes)


def test_external_boundary_seed_target_bypasses_scalar_evidence_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    cache = DownstreamScalarEvidenceCache(case)
    calls = {"voting": 0, "thinning": 0}
    original_voting = pipeline.build_voting_scalar_evidence
    original_thinning = pipeline.build_thinning_scalar_evidence

    def counted_voting(*args, **kwargs):
        calls["voting"] += 1
        return original_voting(*args, **kwargs)

    def counted_thinning(*args, **kwargs):
        calls["thinning"] += 1
        return original_thinning(*args, **kwargs)

    monkeypatch.setattr(pipeline, "build_voting_scalar_evidence", counted_voting)
    monkeypatch.setattr(pipeline, "build_thinning_scalar_evidence", counted_thinning)
    common = dict(
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
        voting_config=SyntheticVotingConfig(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        variant_spec=get_variant_spec("boundary_seed_retention_v1"),
        fvt_recenter_target_source="custom_target",
        scalar_evidence_cache=cache,
        attribute_stage_key=AttributeStageKey(case.case_id, case.shape, "oracle"),
    )

    pipeline.run_voting_from_attributes(
        case,
        fvt_recenter_target=np.zeros(case.shape, dtype=np.float32),
        **common,
    )
    pipeline.run_voting_from_attributes(
        case,
        fvt_recenter_target=np.ones(case.shape, dtype=np.float32),
        **common,
    )

    assert calls == {"voting": 2, "thinning": 2}
    assert cache.stats.voting_builds == cache.stats.voting_reuses == 0
    assert cache.stats.thinning_builds == cache.stats.thinning_reuses == 0
    assert not cache._voting
    assert not cache._thinning


def test_unavailable_semantic_key_bypasses_scalar_cache_and_matches_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    custom_oracle = OrientationField3D(
        ft=case.ft_oracle.copy(),
        pt=case.pt_oracle.copy(),
        tt=case.tt_oracle.copy(),
    )
    prepared = PreparedCaseInputs(case, custom_oracle, None)
    cache = DownstreamScalarEvidenceCache(case)
    calls = {"voting": 0, "thinning": 0}
    original_voting = pipeline.build_voting_scalar_evidence
    original_thinning = pipeline.build_thinning_scalar_evidence

    def counted_voting(*args, **kwargs):
        calls["voting"] += 1
        return original_voting(*args, **kwargs)

    def counted_thinning(*args, **kwargs):
        calls["thinning"] += 1
        return original_thinning(*args, **kwargs)

    monkeypatch.setattr(pipeline, "build_voting_scalar_evidence", counted_voting)
    monkeypatch.setattr(pipeline, "build_thinning_scalar_evidence", counted_thinning)
    common = dict(
        voting_config=SyntheticVotingConfig(),
        scanner_config=SyntheticScannerConfig(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        variant="current_default",
        input_mode="oracle",
        scanner_backend_matrix=False,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=False,
        prepared_inputs=prepared,
    )

    legacy = run_case_variant(case, scalar_evidence_cache=None, **common)
    bypassed = run_case_variant(case, scalar_evidence_cache=cache, **common)

    assert calls == {"voting": 2, "thinning": 2}
    assert cache.stats.voting_builds == cache.stats.voting_reuses == 0
    assert cache.stats.thinning_builds == cache.stats.thinning_reuses == 0
    assert not cache._voting
    assert not cache._thinning
    _assert_nested_equal(bypassed.report_payload, legacy.report_payload)
    _assert_nested_equal(bypassed.artifacts.volumes, legacy.artifacts.volumes)


def _write_report_outputs(
    output_dir: Path,
    outputs: tuple[dict, dict, dict],
) -> None:
    report, volumes, skins = outputs
    write_metrics_json(report, output_dir)
    write_summary_csv(report, output_dir)
    write_case_volumes(volumes, output_dir)
    write_case_skins_json(skins, output_dir)


def _artifact_hashes(output_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(output_dir).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and (path.suffix == ".dat" or path.name == "skins.json")
    }


def _output_bytes(output_dir: Path) -> dict[str, bytes]:
    return {
        path.relative_to(output_dir).as_posix(): path.read_bytes()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    }


def test_report_outputs_are_identical_with_and_without_cache(tmp_path: Path) -> None:
    kwargs = dict(
        case_set="minimal",
        shape=(9, 9, 9),
        variants=("current_default", "voter_thin_normal", "quality_skinner_v2"),
        skinning_config=SyntheticSkinningConfig(enabled=False),
    )
    cached = application._build_report_outputs(**kwargs, use_stage_cache=True)
    uncached = application._build_report_outputs(**kwargs, use_stage_cache=False)
    _assert_nested_equal(cached[0], uncached[0])
    assert json.dumps(cached[0]) == json.dumps(uncached[0])
    _assert_nested_equal(cached[1], uncached[1])
    _assert_nested_equal(cached[2], uncached[2])

    cached_dir = tmp_path / "cached"
    uncached_dir = tmp_path / "uncached"
    _write_report_outputs(cached_dir, cached)
    _write_report_outputs(uncached_dir, uncached)
    assert (cached_dir / "metrics.json").read_bytes() == (
        uncached_dir / "metrics.json"
    ).read_bytes()
    assert (cached_dir / "summary.csv").read_bytes() == (uncached_dir / "summary.csv").read_bytes()
    assert _artifact_hashes(cached_dir) == _artifact_hashes(uncached_dir)


def test_fresh_scanner_report_runs_are_byte_identical(tmp_path: Path) -> None:
    kwargs = dict(
        case_set="minimal",
        shape=(9, 9, 9),
        variants=("current_default", "boundary_aware_voter_v1"),
        scanner_config=SyntheticScannerConfig(backend="quality", refinement_factor=2),
        input_mode="both",
        workflow_mode="quality",
        include_scanner_downstream_diagnostics=True,
    )
    first = application._build_report_outputs(**kwargs)
    second = application._build_report_outputs(**kwargs)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _write_report_outputs(first_dir, first)
    _write_report_outputs(second_dir, second)

    assert _output_bytes(first_dir) == _output_bytes(second_dir)


def test_case_and_variant_order_do_not_change_stage_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_definitions = application.validate_case_set("extended")[:2]
    case_orders = iter((case_definitions, tuple(reversed(case_definitions))))
    monkeypatch.setattr(application, "validate_case_set", lambda _case_set: next(case_orders))
    variants = ("current_default", "boundary_aware_voter_v1")
    common = dict(
        case_set="extended",
        shape=(9, 9, 9),
        scanner_config=SyntheticScannerConfig(backend="quality", refinement_factor=2),
        input_mode="both",
        workflow_mode="quality",
        include_scanner_downstream_diagnostics=True,
    )
    forward = application._build_report_outputs(**common, variants=variants)
    reverse = application._build_report_outputs(**common, variants=tuple(reversed(variants)))
    forward_cases = {case["case_id"]: case for case in forward[0]["cases"]}
    reverse_cases = {case["case_id"]: case for case in reverse[0]["cases"]}
    for case_id in forward_cases:
        for variant in variants:
            _assert_nested_equal(
                forward_cases[case_id]["variants"][variant],
                reverse_cases[case_id]["variants"][variant],
            )
            _assert_nested_equal(forward[1][case_id][variant], reverse[1][case_id][variant])
            _assert_nested_equal(forward[2][case_id][variant], reverse[2][case_id][variant])


def test_skinning_outputs_are_identical_with_and_without_cache(tmp_path: Path) -> None:
    kwargs = dict(
        case_set="minimal",
        shape=(9, 9, 9),
        variants=(
            "current_default",
            "quality_boundary_skinner_fallback",
            "quality_boundary_skinner_fallback_v2",
        ),
        skinning_config=SyntheticSkinningConfig(),
    )
    cached = application._build_report_outputs(**kwargs, use_stage_cache=True)
    uncached = application._build_report_outputs(**kwargs, use_stage_cache=False)
    for cached_part, uncached_part in zip(cached, uncached):
        _assert_nested_equal(cached_part, uncached_part)

    cached_dir = tmp_path / "cached-skinning"
    uncached_dir = tmp_path / "uncached-skinning"
    _write_report_outputs(cached_dir, cached)
    _write_report_outputs(uncached_dir, uncached)
    assert (cached_dir / "metrics.json").read_bytes() == (
        uncached_dir / "metrics.json"
    ).read_bytes()
    assert (cached_dir / "summary.csv").read_bytes() == (uncached_dir / "summary.csv").read_bytes()
    assert _artifact_hashes(cached_dir) == _artifact_hashes(uncached_dir)


def test_cached_voting_arrays_are_read_only_and_diagnostics_are_copied() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    cache = PipelineStageCache()
    first = _run_variant("current_default", cache=cache, case=case)
    first_summary = first.report_payload["pyosv"]["voting"]["diagnostic_summary"]
    first_summary["seed_count"] = -1
    second = _run_variant("voter_thin_normal", cache=cache, case=case)

    assert second.report_payload["pyosv"]["voting"]["diagnostic_summary"]["seed_count"] >= 0
    with pytest.raises(ValueError, match="read-only"):
        first.artifacts.volumes["fv_py"][0, 0, 0] = np.float32(1.0)


def test_separate_case_caches_do_not_share_entries() -> None:
    first = PipelineStageCache()
    second = PipelineStageCache()
    _run_variant("current_default", cache=first)
    _run_variant("current_default", cache=second)
    assert first.stats.voting_misses == 1
    assert second.stats.voting_misses == 1
    assert first.stats.voting_hits == second.stats.voting_hits == 0


def test_one_cache_cannot_be_reused_for_distinct_cases_with_same_metadata() -> None:
    cache = PipelineStageCache()
    first_case = make_single_vertical_plane_case((9, 9, 9))
    second_case = make_single_vertical_plane_case((9, 9, 9))
    _run_variant("current_default", cache=cache, case=first_case)

    with pytest.raises(
        ValueError,
        match="pipeline stage cache must not be reused across synthetic cases",
    ):
        _run_variant("current_default", cache=cache, case=second_case)


def test_clear_releases_cached_stage_arrays() -> None:
    cache = PipelineStageCache()
    evaluation = _run_variant("current_default", cache=cache)
    cached_fvt = next(iter(cache._thinning.values())).fvt
    reference = weakref.ref(cached_fvt)

    del cached_fvt
    del evaluation
    cache.clear()
    gc.collect()

    assert reference() is None
    assert not cache._seeds
    assert not cache._voting
    assert not cache._thinning
    assert not cache._final_thinning
    assert not cache._primary_skinning
