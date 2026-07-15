from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

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
from pyosv.evaluation.reporting.artifacts import write_case_skins_json, write_case_volumes
from pyosv.evaluation.reporting.csv_v1 import write_summary_csv
from pyosv.evaluation.reporting.json_v1 import write_metrics_json
from pyosv.synthetic3d import Synthetic3DCase
from pyosv.synthetic3d import make_single_vertical_plane_case


def _run_variant(
    variant: str,
    *,
    cache: PipelineStageCache | None,
    case: Synthetic3DCase | None = None,
    voting_config: SyntheticVotingConfig = SyntheticVotingConfig(),
):
    if case is None:
        case = make_single_vertical_plane_case((9, 9, 9))
    return run_case_variant(
        case,
        voting_config=voting_config,
        scanner_config=SyntheticScannerConfig(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
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
    variants = ("current_default", "voter_thin_normal", "quality_skinner_v2")
    common = dict(
        case_set="minimal",
        shape=(9, 9, 9),
        skinning_config=SyntheticSkinningConfig(enabled=False),
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
