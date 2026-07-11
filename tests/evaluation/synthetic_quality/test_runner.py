from __future__ import annotations

from collections.abc import Mapping
import json

import numpy as np
import pytest

from pyosv.evaluation.synthetic_quality.cases import MINIMAL_CASES
from pyosv.evaluation.synthetic_quality.application import build_report
from pyosv.evaluation.synthetic_quality.config import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from pyosv.evaluation.synthetic_quality.runner import (
    case_pipeline_reports,
    case_variant_comparison_alias,
    prepare_case_inputs,
    run_case,
    run_case_variant,
    run_scanner_pipeline,
)
from pyosv.evaluation.synthetic_quality import runner
from pyosv.cli import synthetic_quality
from pyosv.synthetic3d import make_single_vertical_plane_case


def _load_example_module() -> object:
    return synthetic_quality


def _assert_nested_arrays_equal(actual: object, expected: object) -> None:
    if isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, expected)
        return
    assert isinstance(actual, Mapping)
    assert isinstance(expected, Mapping)
    assert actual.keys() == expected.keys()
    for key in expected:
        _assert_nested_arrays_equal(actual[key], expected[key])


@pytest.mark.parametrize(
    ("input_mode", "expected", "active"),
    (
        ("oracle", {"oracle"}, None),
        ("scanner", {"scanner"}, "scanner"),
        ("both", {"oracle", "scanner"}, "oracle"),
    ),
)
def test_runner_pipeline_aliases(input_mode: str, expected: set[str], active: str | None) -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    result = run_case_variant(
        case,
        voting_config=SyntheticVotingConfig(),
        scanner_config=SyntheticScannerConfig(
            backend="fast",
            phi_min=0.0,
            phi_max=0.0,
            theta_min=90.0,
            theta_max=90.0,
            sigma1=1.0,
            sigma2=1.0,
            scanner_thin_mode="none",
        ),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        variant="current_default",
        input_mode=input_mode,
        scanner_backend_matrix=False,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=False,
    )
    if input_mode == "oracle":
        assert "pipelines" not in result.report_payload
    else:
        assert set(result.report_payload["pipelines"]) == expected
        assert result.report_payload["active_pipeline"] == active


def test_run_case_builds_active_pipeline_and_variant_aliases() -> None:
    result = run_case(
        MINIMAL_CASES[0], shape=(9, 9, 9), skinning_config=SyntheticSkinningConfig(enabled=False)
    )
    assert set(result.report_payload["pipelines"]) == {"oracle"}
    assert set(result.report_payload["variants"]) == {"current_default"}
    assert (
        result.report_payload["config"]
        == result.report_payload["variants"]["current_default"]["config"]
    )


def test_example_wrapper_matches_package_report_volumes_and_skins() -> None:
    example = _load_example_module()
    case = make_single_vertical_plane_case((9, 9, 9))
    kwargs = {
        "voting_config": SyntheticVotingConfig(),
        "scanner_config": SyntheticScannerConfig(
            backend="fast",
            phi_min=0.0,
            phi_max=0.0,
            theta_min=90.0,
            theta_max=90.0,
            sigma1=1.0,
            sigma2=1.0,
            scanner_thin_mode="none",
        ),
        "truth_metric_config": SyntheticTruthMetricConfig(),
        "skinning_config": SyntheticSkinningConfig(enabled=True),
        "variant": "current_default",
        "input_mode": "both",
        "scanner_backend_matrix": False,
        "include_thinning_diagnostic": False,
        "include_scanner_downstream_diagnostics": False,
    }
    package_result = run_case_variant(case, **kwargs)
    wrapper_report, wrapper_volumes, wrapper_skins = example._run_case_variant(case, **kwargs)

    assert wrapper_report == package_result.report_payload
    _assert_nested_arrays_equal(wrapper_volumes, package_result.artifacts.volumes)
    assert wrapper_skins == package_result.artifacts.skins_payload


def test_multiple_variants_build_pipeline_comparison_aliases() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    common = {
        "voting_config": SyntheticVotingConfig(),
        "scanner_config": SyntheticScannerConfig(),
        "truth_metric_config": SyntheticTruthMetricConfig(),
        "skinning_config": SyntheticSkinningConfig(enabled=False),
        "input_mode": "oracle",
        "scanner_backend_matrix": False,
        "include_thinning_diagnostic": False,
        "include_scanner_downstream_diagnostics": False,
    }
    reports = {
        variant: run_case_variant(case, variant=variant, **common).report_payload
        for variant in ("current_default", "boundary_aware_voter_v1")
    }

    pipelines = case_pipeline_reports(reports, "oracle")
    assert set(pipelines["oracle"]["variants"]) == {
        "current_default",
        "boundary_aware_voter_v1",
    }
    assert (
        case_variant_comparison_alias(pipelines, "oracle")
        == pipelines["oracle"]["variant_comparison"]
    )


def _fast_scanner_config() -> SyntheticScannerConfig:
    return SyntheticScannerConfig(
        backend="fast",
        phi_min=0.0,
        phi_max=0.0,
        theta_min=90.0,
        theta_max=90.0,
        sigma1=1.0,
        sigma2=1.0,
        scanner_thin_mode="none",
    )


def test_scanner_pipeline_preserves_captured_stage_trace() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    result = run_scanner_pipeline(
        case,
        voting_config=SyntheticVotingConfig(),
        scanner_config=_fast_scanner_config(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        variant_spec=runner.get_variant_spec("current_default"),
        scanner_backend_matrix=False,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=False,
        capture_stage_trace=True,
    )

    assert result.artifacts.stage_trace is not None
    assert result.artifacts.stage_trace.fv_positive_mask.shape == case.shape


def test_scanner_boundary_stage_diagnostics_is_independent_opt_in() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    common = dict(
        voting_config=SyntheticVotingConfig(),
        scanner_config=_fast_scanner_config(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        variant_spec=runner.get_variant_spec("current_default"),
        scanner_backend_matrix=False,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=False,
    )

    plain = run_scanner_pipeline(case, **common)
    detailed = run_scanner_pipeline(case, include_scanner_boundary_stage_diagnostics=True, **common)

    assert "scanner_boundary_stage_diagnostics" not in plain.report_payload
    assert not any(name.startswith("scanner_boundary_stage_") for name in plain.artifacts.volumes)
    assert "scanner_boundary_stage_diagnostics" in detailed.report_payload
    assert detailed.artifacts.stage_trace is not None
    assert (
        sum(name.startswith("scanner_boundary_stage_") for name in detailed.artifacts.volumes) == 10
    )


@pytest.mark.parametrize(
    ("include_downstream", "include_boundary"),
    ((False, False), (True, False), (False, True), (True, True)),
)
def test_scanner_diagnostic_runners_are_independent(
    include_downstream: bool,
    include_boundary: bool,
) -> None:
    case = make_single_vertical_plane_case((5, 5, 5))
    calls: list[str] = []

    def downstream(**kwargs: object) -> dict[str, bool]:
        calls.append("downstream")
        return {"called": True}

    def stage_loss(**kwargs: object) -> dict[str, bool]:
        calls.append("stage_loss")
        return {"called": True}

    def boundary(**kwargs: object) -> tuple[dict[str, bool], dict[str, np.ndarray]]:
        calls.append("boundary")
        return {"called": True}, {}

    result = run_scanner_pipeline(
        case,
        voting_config=SyntheticVotingConfig(),
        scanner_config=_fast_scanner_config(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        variant_spec=runner.get_variant_spec("current_default"),
        scanner_backend_matrix=False,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=include_downstream,
        include_scanner_boundary_stage_diagnostics=include_boundary,
        scanner_downstream_diagnostic_runner=downstream,
        scanner_stage_loss_diagnostic_runner=stage_loss,
        scanner_boundary_stage_diagnostic_runner=boundary,
    )

    expected_calls = []
    if include_downstream:
        expected_calls.extend(("downstream", "stage_loss"))
    if include_boundary:
        expected_calls.append("boundary")
    assert calls == expected_calls
    assert ("scanner_downstream" in result.report_payload) is include_downstream
    assert ("scanner_stage_loss" in result.report_payload) is include_downstream
    assert ("scanner_boundary_stage_diagnostics" in result.report_payload) is include_boundary


def test_boundary_diagnostics_apply_only_to_scanner_pipeline_in_both_mode() -> None:
    case = make_single_vertical_plane_case((5, 5, 5))
    common = {
        "voting_config": SyntheticVotingConfig(),
        "scanner_config": _fast_scanner_config(),
        "truth_metric_config": SyntheticTruthMetricConfig(),
        "skinning_config": SyntheticSkinningConfig(enabled=False),
        "variant": "current_default",
        "scanner_backend_matrix": False,
        "include_thinning_diagnostic": False,
        "include_scanner_downstream_diagnostics": False,
        "include_scanner_boundary_stage_diagnostics": True,
    }

    oracle = run_case_variant(case, input_mode="oracle", **common)
    both = run_case_variant(case, input_mode="both", **common)

    assert "scanner_boundary_stage_diagnostics" not in oracle.report_payload
    assert "scanner_boundary_stage_diagnostics" not in both.report_payload["pipelines"]["oracle"]
    scanner_report = both.report_payload["pipelines"]["scanner"]
    assert "scanner_boundary_stage_diagnostics" in scanner_report
    json.dumps(both.report_payload, allow_nan=False)


def test_boundary_seed_report_uses_the_captured_selected_seed_mask() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    result = run_scanner_pipeline(
        case,
        voting_config=SyntheticVotingConfig(),
        scanner_config=_fast_scanner_config(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        variant_spec=runner.get_variant_spec("boundary_seed_retention_v1"),
        scanner_backend_matrix=False,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=False,
        include_scanner_boundary_stage_diagnostics=True,
    )

    trace = result.artifacts.stage_trace
    assert trace is not None
    selected_count = int(np.count_nonzero(trace.seed_selected_mask))
    diagnostic = result.report_payload["scanner_boundary_stage_diagnostics"]
    assert diagnostic["stages"]["seed_selected"]["candidate_count"] == selected_count
    assert selected_count == result.report_payload["boundary_seed_retention"]["total_seed_count"]


def test_backend_matrix_calls_boundary_builder_only_for_selected_backend() -> None:
    case = make_single_vertical_plane_case((5, 5, 5))
    calls = 0

    def boundary(**kwargs: object) -> tuple[dict[str, bool], dict[str, np.ndarray]]:
        nonlocal calls
        calls += 1
        return {"called": True}, {}

    result = run_scanner_pipeline(
        case,
        voting_config=SyntheticVotingConfig(),
        scanner_config=_fast_scanner_config(),
        truth_metric_config=SyntheticTruthMetricConfig(),
        skinning_config=SyntheticSkinningConfig(enabled=False),
        variant_spec=runner.get_variant_spec("current_default"),
        scanner_backend_matrix=True,
        include_thinning_diagnostic=False,
        include_scanner_downstream_diagnostics=False,
        include_scanner_boundary_stage_diagnostics=True,
        scanner_boundary_stage_diagnostic_runner=boundary,
    )

    assert calls == 1
    backends = result.report_payload["scanner_backend_matrix"]["backends"]
    assert "scanner_boundary_stage_diagnostics" in backends["fast"]
    assert "scanner_boundary_stage_diagnostics" not in backends["reference-like"]
    assert "scanner_boundary_stage_diagnostics" not in backends["quality"]


def test_report_config_adds_boundary_diagnostics_only_when_requested() -> None:
    common = {
        "case_set": "minimal",
        "shape": (5, 5, 5),
        "variants": ("current_default",),
        "skinning_config": SyntheticSkinningConfig(enabled=False),
    }

    default_report = build_report(input_mode="oracle", **common)
    oracle_requested = build_report(
        input_mode="oracle",
        include_scanner_boundary_stage_diagnostics=True,
        **common,
    )
    scanner_requested = build_report(
        input_mode="scanner",
        scanner_config=_fast_scanner_config(),
        include_scanner_boundary_stage_diagnostics=True,
        **common,
    )

    assert "scanner_boundary_stage_diagnostics" not in default_report["config"]
    assert oracle_requested["config"]["scanner_boundary_stage_diagnostics"] == {"enabled": False}
    assert scanner_requested["config"]["scanner_boundary_stage_diagnostics"] == {"enabled": True}
    oracle_variant = oracle_requested["cases"][0]["pipelines"]["oracle"]["variants"][
        "current_default"
    ]
    assert "scanner_boundary_stage_diagnostics" not in oracle_variant


def test_scanner_cache_report_build_scans_once_per_case_across_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    example = _load_example_module()
    calls = 0
    implementation = runner.scanner_attributes_from_case

    def counting_scanner(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return implementation(*args, **kwargs)

    monkeypatch.setattr(runner, "scanner_attributes_from_case", counting_scanner)
    example.build_report(
        case_set="extended",
        shape=(9, 9, 9),
        scanner_config=_fast_scanner_config(),
        variants=("current_default", "boundary_aware_voter_v1"),
        input_mode="both",
        skinning_config=SyntheticSkinningConfig(enabled=False),
    )

    assert calls == 7


def test_scanner_cache_oracle_input_does_not_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    case = make_single_vertical_plane_case((9, 9, 9))

    def unexpected_scan(*args: object, **kwargs: object) -> object:
        raise AssertionError("oracle input must not invoke the scanner")

    monkeypatch.setattr(runner, "scanner_attributes_from_case", unexpected_scan)
    prepared = prepare_case_inputs(
        case,
        scanner_config=_fast_scanner_config(),
        input_mode="oracle",
        scanner_backend_matrix=True,
    )

    assert prepared.scanner is None


def test_prepared_scanner_variant_order_is_independent_and_shared_arrays_are_not_mutated() -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    scanner_config = _fast_scanner_config()
    prepared = prepare_case_inputs(
        case,
        scanner_config=scanner_config,
        input_mode="scanner",
        scanner_backend_matrix=False,
    )
    assert prepared.scanner is not None
    original = {name: volume.copy() for name, volume in prepared.scanner.selected.volumes.items()}
    common = {
        "voting_config": SyntheticVotingConfig(),
        "scanner_config": scanner_config,
        "truth_metric_config": SyntheticTruthMetricConfig(),
        "skinning_config": SyntheticSkinningConfig(enabled=False),
        "input_mode": "scanner",
        "scanner_backend_matrix": False,
        "include_thinning_diagnostic": False,
        "include_scanner_downstream_diagnostics": False,
        "prepared_inputs": prepared,
    }
    variants = ("current_default", "boundary_aware_voter_v1")
    forward = {variant: run_case_variant(case, variant=variant, **common) for variant in variants}
    reverse = {
        variant: run_case_variant(case, variant=variant, **common) for variant in reversed(variants)
    }

    for variant in variants:
        assert forward[variant].report_payload == reverse[variant].report_payload
        _assert_nested_arrays_equal(
            forward[variant].artifacts.volumes,
            reverse[variant].artifacts.volumes,
        )
    for name, expected in original.items():
        np.testing.assert_array_equal(prepared.scanner.selected.volumes[name], expected)


def test_prepared_backend_matrix_scans_each_backend_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_single_vertical_plane_case((9, 9, 9))
    backends: list[str] = []
    implementation = runner.scanner_attributes_from_case

    def counting_scanner(
        case_arg: object, config: SyntheticScannerConfig, **kwargs: object
    ) -> object:
        backends.append(config.backend)
        return implementation(case_arg, config, **kwargs)

    monkeypatch.setattr(runner, "scanner_attributes_from_case", counting_scanner)
    prepared = prepare_case_inputs(
        case,
        scanner_config=_fast_scanner_config(),
        input_mode="scanner",
        scanner_backend_matrix=True,
    )

    assert prepared.scanner is not None
    assert backends == ["fast", "reference-like", "quality"]
    assert set(prepared.scanner.by_backend) == {"reference-like", "quality", "fast"}
