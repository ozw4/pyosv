"""Public application API for building synthetic-quality reports."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from pyosv.evaluation.reporting.json_v1 import LegacyReportV1Adapter
from pyosv.evaluation.reporting.models import Report, ReportConfig
from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.evaluation.synthetic_quality.cases import (
    SyntheticQualityCaseDefinition,
    validate_case_ids,
    validate_case_set,
)
from pyosv.evaluation.synthetic_quality.config import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from pyosv.evaluation.synthetic_quality.profiles import (
    _default_surface_support_policy_for_workflow,
    _default_voter_thin_mode_for_workflow,
    _effective_include_thinning_diagnostic,
    _effective_skinning_config_for_workflow,
    _validate_workflow_mode,
)
from pyosv.evaluation.synthetic_quality.runner import (
    build_case_report_model,
    prepare_case_inputs,
    run_case as _run_case,
    run_case_variant,
    validate_input_mode,
)
from pyosv.evaluation.synthetic_quality.variants import (
    DEFAULT_VARIANTS,
    validate_variant_preset,
    validate_variants,
)
from pyosv.synthetic3d import validate_shape3

DEFAULT_VARIANT_PRESET = "default"
DEFAULT_THINNING_DIAGNOSTIC_CASES = ("curved_surface",)


def run_case(
    case_definition: SyntheticQualityCaseDefinition,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one case and return its legacy v1 report and volume payloads."""

    evaluation = _run_case(case_definition, **kwargs)
    return dict(evaluation.report_payload), dict(evaluation.artifacts.volumes)


def build_report(
    *,
    case_set: str,
    shape: tuple[int, int, int],
    voting_config: SyntheticVotingConfig | None = None,
    scanner_config: SyntheticScannerConfig = SyntheticScannerConfig(),
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig(),
    variants: Sequence[str] = DEFAULT_VARIANTS,
    variant_preset: str = DEFAULT_VARIANT_PRESET,
    skinning_config: SyntheticSkinningConfig = SyntheticSkinningConfig(),
    input_mode: str = "oracle",
    scanner_backend_matrix: bool = False,
    workflow_mode: str = "reference",
    skinner_method_explicit: bool = False,
    skinner_min_likelihood_explicit: bool = False,
    skinner_growth_source_explicit: bool = False,
    skinner_accepted_occupancy_radius_explicit: bool = False,
    include_thinning_diagnostic: bool = False,
    include_scanner_downstream_diagnostics: bool = False,
    thinning_diagnostic_cases: Sequence[str] = DEFAULT_THINNING_DIAGNOSTIC_CASES,
) -> dict[str, Any]:
    """Build a legacy v1 synthetic-quality report without writing artifacts."""

    report, _, _ = _build_report_outputs(
        case_set=case_set,
        shape=shape,
        voting_config=voting_config,
        scanner_config=scanner_config,
        truth_metric_config=truth_metric_config,
        variants=variants,
        variant_preset=variant_preset,
        skinning_config=skinning_config,
        input_mode=input_mode,
        scanner_backend_matrix=scanner_backend_matrix,
        workflow_mode=workflow_mode,
        skinner_method_explicit=skinner_method_explicit,
        skinner_min_likelihood_explicit=skinner_min_likelihood_explicit,
        skinner_growth_source_explicit=skinner_growth_source_explicit,
        skinner_accepted_occupancy_radius_explicit=(skinner_accepted_occupancy_radius_explicit),
        include_thinning_diagnostic=include_thinning_diagnostic,
        include_scanner_downstream_diagnostics=(include_scanner_downstream_diagnostics),
        thinning_diagnostic_cases=thinning_diagnostic_cases,
    )
    return report


def _build_report_outputs(
    *,
    case_set: str,
    shape: tuple[int, int, int],
    voting_config: SyntheticVotingConfig | None = None,
    scanner_config: SyntheticScannerConfig = SyntheticScannerConfig(),
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig(),
    variants: Sequence[str] = DEFAULT_VARIANTS,
    variant_preset: str = DEFAULT_VARIANT_PRESET,
    skinning_config: SyntheticSkinningConfig = SyntheticSkinningConfig(),
    input_mode: str = "oracle",
    scanner_backend_matrix: bool = False,
    workflow_mode: str = "reference",
    skinner_method_explicit: bool = False,
    skinner_min_likelihood_explicit: bool = False,
    skinner_growth_source_explicit: bool = False,
    skinner_accepted_occupancy_radius_explicit: bool = False,
    include_thinning_diagnostic: bool = False,
    include_scanner_downstream_diagnostics: bool = False,
    thinning_diagnostic_cases: Sequence[str] = DEFAULT_THINNING_DIAGNOSTIC_CASES,
    thinning_diagnostic_runner: Callable[..., Any] | None = None,
    recenter_distance_diagnostic_runner: Callable[..., Any] | None = None,
    scanner_downstream_diagnostic_runner: Callable[..., Any] | None = None,
    scanner_stage_loss_diagnostic_runner: Callable[..., Any] | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, dict[str, Any]]],
]:
    """Build a report and the volume/skin payloads consumed by artifact writers."""

    valid_shape = validate_shape3(shape)
    valid_variants = validate_variants(variants)
    valid_variant_preset = validate_variant_preset(variant_preset)
    valid_input_mode = validate_input_mode(input_mode)
    effective_scanner_backend_matrix = bool(scanner_backend_matrix and valid_input_mode != "oracle")
    effective_scanner_downstream_diagnostics = bool(
        include_scanner_downstream_diagnostics and valid_input_mode != "oracle"
    )
    valid_workflow_mode = _validate_workflow_mode(workflow_mode)
    skinning_config = _effective_skinning_config_for_workflow(
        workflow_mode=valid_workflow_mode,
        skinning_config=skinning_config,
        skinner_method_explicit=skinner_method_explicit,
        skinner_min_likelihood_explicit=skinner_min_likelihood_explicit,
        skinner_growth_source_explicit=skinner_growth_source_explicit,
        skinner_accepted_occupancy_radius_explicit=(skinner_accepted_occupancy_radius_explicit),
    )
    if voting_config is None:
        support_min_fraction, support_exponent = _default_surface_support_policy_for_workflow(
            valid_workflow_mode
        )
        voting_config = SyntheticVotingConfig(
            voter_thin_mode=_default_voter_thin_mode_for_workflow(valid_workflow_mode),
            surface_support_min_fraction=support_min_fraction,
            surface_support_exponent=support_exponent,
        )
    include_thinning_diagnostic = _effective_include_thinning_diagnostic(
        workflow_mode=valid_workflow_mode,
        include_thinning_diagnostic=include_thinning_diagnostic,
    )
    diagnostic_case_ids = set(
        validate_case_ids(
            thinning_diagnostic_cases,
            description="thinning diagnostic",
            sequence_name="thinning_diagnostic_cases",
        )
    )
    case_definitions = validate_case_set(case_set)

    cases = []
    volume_outputs = {}
    skin_outputs = {}
    for case_definition in case_definitions:
        case = case_definition.factory(valid_shape)
        if case.case_id != case_definition.case_id:
            raise ValueError(
                f"case factory returned {case.case_id!r}, expected {case_definition.case_id!r}"
            )
        variant_reports = {}
        variant_volumes = {}
        variant_skins = {}
        prepared_inputs = prepare_case_inputs(
            case,
            scanner_config=scanner_config,
            input_mode=valid_input_mode,
            scanner_backend_matrix=effective_scanner_backend_matrix,
        )
        for variant in valid_variants:
            diagnostic_runners = {
                name: runner
                for name, runner in (
                    ("thinning_diagnostic_runner", thinning_diagnostic_runner),
                    (
                        "recenter_distance_diagnostic_runner",
                        recenter_distance_diagnostic_runner,
                    ),
                    (
                        "scanner_downstream_diagnostic_runner",
                        scanner_downstream_diagnostic_runner,
                    ),
                    (
                        "scanner_stage_loss_diagnostic_runner",
                        scanner_stage_loss_diagnostic_runner,
                    ),
                )
                if runner is not None
            }
            evaluation = run_case_variant(
                case,
                voting_config=voting_config,
                scanner_config=scanner_config,
                truth_metric_config=truth_metric_config,
                skinning_config=skinning_config,
                variant=variant,
                input_mode=valid_input_mode,
                scanner_backend_matrix=effective_scanner_backend_matrix,
                include_thinning_diagnostic=(
                    include_thinning_diagnostic and case.case_id in diagnostic_case_ids
                ),
                include_scanner_downstream_diagnostics=(effective_scanner_downstream_diagnostics),
                prepared_inputs=prepared_inputs,
                **diagnostic_runners,
            )
            variant_reports[variant] = dict(evaluation.report_payload)
            variant_volumes[variant] = dict(evaluation.artifacts.volumes)
            variant_skins[variant] = dict(evaluation.artifacts.skins_payload)
        case_model = build_case_report_model(
            case_id=case.case_id,
            shape=case.shape,
            truth=quality_metrics.truth_report(case, truth_metric_config),
            variant_reports=variant_reports,
            input_mode=valid_input_mode,
        )
        cases.append(case_model)
        volume_outputs[case_definition.case_id] = variant_volumes
        skin_outputs[case_definition.case_id] = variant_skins

    config: dict[str, Any] = {
        "case_set": case_set,
        "workflow_mode": valid_workflow_mode,
        "variant_preset": valid_variant_preset,
        "shape": [int(size) for size in valid_shape],
        "variants": list(valid_variants),
        "voting": voting_config.as_report_dict(),
        "truth_metrics": truth_metric_config.as_report_dict(),
        "skinning": skinning_config.as_report_dict(),
        "scanner_backend_matrix": effective_scanner_backend_matrix,
        "scanner_downstream_diagnostics": effective_scanner_downstream_diagnostics,
    }
    if valid_input_mode != "oracle":
        config["input_mode"] = valid_input_mode
        config["scanner"] = scanner_config.as_report_dict()
    if include_thinning_diagnostic:
        config["thinning_diagnostic"] = {"enabled": True}

    report_model = Report(config=ReportConfig(config), cases=tuple(cases))
    report = LegacyReportV1Adapter().to_dict(report_model)
    return report, volume_outputs, skin_outputs
