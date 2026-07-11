from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from pyosv.evaluation.synthetic_quality.cases import MINIMAL_CASES
from pyosv.evaluation.synthetic_quality.config import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from pyosv.evaluation.synthetic_quality.runner import (
    case_pipeline_reports,
    case_variant_comparison_alias,
    run_case,
    run_case_variant,
)
from pyosv.synthetic3d import make_single_vertical_plane_case


def _load_example_module() -> object:
    script = Path(__file__).resolve().parents[3] / "examples" / "report_3d_synthetic_quality.py"
    spec = importlib.util.spec_from_file_location("issue_351_report_3d_synthetic_quality", script)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
