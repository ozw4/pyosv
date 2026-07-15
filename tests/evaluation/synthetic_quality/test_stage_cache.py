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
from pyosv.evaluation.synthetic_quality.runner import run_case_variant
from pyosv.evaluation.synthetic_quality.stage_cache import PipelineStageCache
from pyosv.evaluation.synthetic_quality.stage_cache import PrimarySkinningStageResult
from pyosv.evaluation.reporting.artifacts import write_case_skins_json, write_case_volumes
from pyosv.evaluation.reporting.csv_v1 import write_summary_csv
from pyosv.evaluation.reporting.json_v1 import write_metrics_json
from pyosv.synthetic3d import Synthetic3DCase
from pyosv.synthetic3d import make_single_vertical_plane_case
from pyosv.cells import FaultCell
from pyosv.skin import FaultSkin


def _run_variant(
    variant: str,
    *,
    cache: PipelineStageCache | None,
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
    )


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
    assert cache.stats.thinning_hits == 1


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


def test_primary_skinning_snapshot_clones_cells_links_skins_and_diagnostics() -> None:
    above = FaultCell(1, 2, 3, 0.9, 20, 70)
    below = FaultCell(1, 2, 4, 0.8, 21, 69)
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


def test_variant_order_does_not_change_each_variant_result() -> None:
    variants = (
        "current_default",
        "quality_boundary_skinner_fallback",
        "quality_boundary_skinner_fallback_v2",
    )
    common = dict(
        case_set="minimal",
        shape=(9, 9, 9),
        skinning_config=SyntheticSkinningConfig(),
    )
    forward = application._build_report_outputs(**common, variants=variants)
    reverse = application._build_report_outputs(**common, variants=tuple(reversed(variants)))
    forward_case = forward[0]["cases"][0]
    reverse_case = reverse[0]["cases"][0]
    for variant in variants:
        _assert_nested_equal(
            forward_case["variants"][variant],
            reverse_case["variants"][variant],
        )
        _assert_nested_equal(
            forward[1]["single_vertical_plane"][variant],
            reverse[1]["single_vertical_plane"][variant],
        )
        _assert_nested_equal(
            forward[2]["single_vertical_plane"][variant],
            reverse[2]["single_vertical_plane"][variant],
        )


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
    assert not cache._primary_skinning
