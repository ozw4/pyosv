"""Application-level orchestration for synthetic-quality evaluations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.evaluation.synthetic_quality.boundary_stage_diagnostics import (
    build_scanner_boundary_stage_diagnostics,
)
from pyosv.evaluation.synthetic_quality.cases import (
    SyntheticQualityCaseDefinition,
    validate_case_ids,
)
from pyosv.evaluation.synthetic_quality.config import (
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from pyosv.evaluation.synthetic_quality.diagnostics import (
    _run_voter_thinning_diagnostic,
    _scanner_downstream_diagnostics,
    _scanner_stage_loss_diagnostics,
)
from pyosv.evaluation.synthetic_quality.models import (
    OrientationField3D,
    PipelineArtifacts,
    PipelineEvaluation,
)
from pyosv.evaluation.synthetic_quality.pipeline import run_voting_from_attributes
from pyosv.evaluation.synthetic_quality.scanner import (
    ScannerAttributes,
    scanner_attributes_from_case,
)
from pyosv.evaluation.synthetic_quality.variants import (
    BASELINE_VARIANT,
    VariantSpec,
    effective_skinning_config,
    get_variant_spec,
)
from pyosv.evaluation.reporting.json_v1 import LegacyReportV1Adapter
from pyosv.evaluation.reporting.models import CaseReport
from pyosv.experimental.boundary_seed_selection import select_boundary_seed_retention_v1
from pyosv.experimental.boundary_thinning import fvt_recenter_target_distance_diagnostics
from pyosv.synthetic3d import Synthetic3DCase

PIPELINE_OUTPUTS_KEY = "__pipelines__"
SCANNER_BACKEND_MATRIX_BACKENDS = ("reference-like", "quality", "fast")
DEFAULT_THINNING_DIAGNOSTIC_CASES = ("curved_surface",)


@dataclass(frozen=True, slots=True)
class PreparedScannerInput:
    """Scanner attributes generated once for one case and scanner configuration."""

    config: SyntheticScannerConfig
    selected: ScannerAttributes
    by_backend: Mapping[str, ScannerAttributes]


@dataclass(frozen=True, slots=True)
class PreparedCaseInputs:
    """Variant-independent inputs for one synthetic case."""

    case: Synthetic3DCase
    oracle: OrientationField3D
    scanner: PreparedScannerInput | None


def validate_input_mode(input_mode: str) -> str:
    if input_mode not in {"oracle", "scanner", "both"}:
        raise ValueError("input_mode must be 'oracle', 'scanner', or 'both'")
    return input_mode


def prepare_case_inputs(
    case: Synthetic3DCase,
    *,
    scanner_config: SyntheticScannerConfig,
    input_mode: str,
    scanner_backend_matrix: bool,
) -> PreparedCaseInputs:
    """Generate inputs shared by every variant in a single report build."""

    valid_input_mode = validate_input_mode(input_mode)
    oracle = OrientationField3D(ft=case.ft_oracle, pt=case.pt_oracle, tt=case.tt_oracle)
    if valid_input_mode == "oracle":
        return PreparedCaseInputs(case=case, oracle=oracle, scanner=None)

    selected = scanner_attributes_from_case(case, scanner_config)
    by_backend = {scanner_config.backend: selected}
    if scanner_backend_matrix:
        backends = SCANNER_BACKEND_MATRIX_BACKENDS
        if scanner_config.backend not in backends:
            backends = (*backends, scanner_config.backend)
        for backend in backends:
            if backend not in by_backend:
                by_backend[backend] = scanner_attributes_from_case(
                    case, replace(scanner_config, backend=backend)
                )
    return PreparedCaseInputs(
        case=case,
        oracle=oracle,
        scanner=PreparedScannerInput(
            config=scanner_config,
            selected=selected,
            by_backend=by_backend,
        ),
    )


def run_oracle_pipeline(
    case: Synthetic3DCase,
    *,
    voting_config: SyntheticVotingConfig,
    truth_metric_config: SyntheticTruthMetricConfig,
    skinning_config: SyntheticSkinningConfig,
    variant_spec: VariantSpec,
    include_thinning_diagnostic: bool,
    thinning_diagnostic_runner: Callable[..., Any] = _run_voter_thinning_diagnostic,
    recenter_distance_diagnostic_runner: Callable[..., Any] = (
        fvt_recenter_target_distance_diagnostics
    ),
    prepared_oracle: OrientationField3D | None = None,
) -> PipelineEvaluation:
    oracle = (
        OrientationField3D(ft=case.ft_oracle, pt=case.pt_oracle, tt=case.tt_oracle)
        if prepared_oracle is None
        else prepared_oracle
    )
    return run_voting_from_attributes(
        case,
        ft=oracle.ft,
        pt=oracle.pt,
        tt=oracle.tt,
        voting_config=voting_config,
        truth_metric_config=truth_metric_config,
        skinning_config=skinning_config,
        variant_spec=variant_spec,
        include_thinning_diagnostic=include_thinning_diagnostic,
        fvt_recenter_target=oracle.ft,
        fvt_recenter_target_source="oracle_ft",
        thinning_diagnostic_runner=thinning_diagnostic_runner,
        recenter_distance_diagnostic_runner=recenter_distance_diagnostic_runner,
    )


def run_scanner_pipeline(
    case: Synthetic3DCase,
    *,
    voting_config: SyntheticVotingConfig,
    scanner_config: SyntheticScannerConfig,
    truth_metric_config: SyntheticTruthMetricConfig,
    skinning_config: SyntheticSkinningConfig,
    variant_spec: VariantSpec,
    scanner_backend_matrix: bool,
    include_thinning_diagnostic: bool,
    include_scanner_downstream_diagnostics: bool,
    include_scanner_boundary_stage_diagnostics: bool = False,
    capture_stage_trace: bool = False,
    thinning_diagnostic_runner: Callable[..., Any] = _run_voter_thinning_diagnostic,
    recenter_distance_diagnostic_runner: Callable[..., Any] = (
        fvt_recenter_target_distance_diagnostics
    ),
    scanner_downstream_diagnostic_runner: Callable[..., Any] = (_scanner_downstream_diagnostics),
    scanner_stage_loss_diagnostic_runner: Callable[..., Any] = _scanner_stage_loss_diagnostics,
    scanner_boundary_stage_diagnostic_runner: Callable[..., Any] = (
        build_scanner_boundary_stage_diagnostics
    ),
    prepared_scanner: PreparedScannerInput | None = None,
) -> PipelineEvaluation:
    scanner = (
        scanner_attributes_from_case(case, scanner_config)
        if prepared_scanner is None
        else _prepared_scanner_attributes(prepared_scanner, scanner_config)
    )
    scanner_volumes = dict(scanner.volumes)
    evaluation = run_voting_from_attributes(
        case,
        ft=scanner_volumes["scanner_fet"],
        pt=scanner_volumes["scanner_fpt"],
        tt=scanner_volumes["scanner_ftt"],
        voting_config=voting_config,
        truth_metric_config=truth_metric_config,
        skinning_config=skinning_config,
        variant_spec=variant_spec,
        include_thinning_diagnostic=include_thinning_diagnostic,
        capture_stage_trace=(capture_stage_trace or include_scanner_boundary_stage_diagnostics),
        scanner_target_positive_mask=quality_metrics.positive_candidate_mask(
            scanner_volumes["scanner_ft"]
        ),
        fvt_recenter_target=scanner_volumes["scanner_fet"],
        fvt_recenter_target_source="scanner_fet",
        thinning_diagnostic_runner=thinning_diagnostic_runner,
        recenter_distance_diagnostic_runner=recenter_distance_diagnostic_runner,
    )
    report = dict(evaluation.report_payload)
    volumes = dict(evaluation.artifacts.volumes)
    report["scanner"] = dict(scanner.report)
    report["scanner_quality"] = quality_metrics.scanner_truth_quality(
        case,
        scanner_volumes=scanner_volumes,
        truth_metric_config=truth_metric_config,
    )
    if include_scanner_downstream_diagnostics:
        report["scanner_downstream"] = scanner_downstream_diagnostic_runner(
            case=case,
            scanner_config=scanner_config,
            voting_config=voting_config,
            variant_spec=variant_spec,
            report=report,
            scanner_volumes=scanner_volumes,
            fv=volumes["fv_py"],
            vp=volumes["vp_py"],
            vt=volumes["vt_py"],
            fvt=volumes["fvt_py"],
            truth_metric_config=truth_metric_config,
        )
        report["scanner_stage_loss"] = scanner_stage_loss_diagnostic_runner(
            case=case,
            voting_config=voting_config,
            variant_spec=variant_spec,
            scanner_volumes=scanner_volumes,
            fv=volumes["fv_py"],
            fvt=volumes["fvt_py"],
            skin_mask=volumes["skin_mask_py"],
            truth_metric_config=truth_metric_config,
            boundary_seed_selector=_boundary_seed_retention_v1_seeds,
        )
    if include_scanner_boundary_stage_diagnostics:
        stage_trace = evaluation.artifacts.stage_trace
        if stage_trace is None:
            raise RuntimeError("scanner boundary stage diagnostics require a pipeline stage trace")
        diagnostic_report, diagnostic_volumes = scanner_boundary_stage_diagnostic_runner(
            case=case,
            scanner_volumes=scanner_volumes,
            stage_trace=stage_trace,
            truth_metric_config=truth_metric_config,
            skinning_diagnostics=report["skinning"].get("diagnostics"),
        )
        report["scanner_boundary_stage_diagnostics"] = diagnostic_report
        volumes.update(diagnostic_volumes)
    if scanner_backend_matrix:
        report["scanner_backend_matrix"] = _scanner_backend_matrix_report(
            case,
            voting_config=voting_config,
            scanner_config=scanner_config,
            truth_metric_config=truth_metric_config,
            skinning_config=skinning_config,
            variant_spec=variant_spec,
            include_thinning_diagnostic=include_thinning_diagnostic,
            selected_report=report,
            thinning_diagnostic_runner=thinning_diagnostic_runner,
            recenter_distance_diagnostic_runner=recenter_distance_diagnostic_runner,
            prepared_scanner=prepared_scanner,
        )
    volumes.update(scanner_volumes)
    return PipelineEvaluation(
        report_payload=report,
        artifacts=PipelineArtifacts(
            volumes=volumes,
            skins_payload=evaluation.artifacts.skins_payload,
            stage_trace=evaluation.artifacts.stage_trace,
        ),
    )


def run_case_variant(
    case: Synthetic3DCase,
    *,
    voting_config: SyntheticVotingConfig,
    scanner_config: SyntheticScannerConfig,
    truth_metric_config: SyntheticTruthMetricConfig,
    skinning_config: SyntheticSkinningConfig,
    variant: str,
    input_mode: str,
    scanner_backend_matrix: bool,
    include_thinning_diagnostic: bool,
    include_scanner_downstream_diagnostics: bool,
    include_scanner_boundary_stage_diagnostics: bool = False,
    thinning_diagnostic_runner: Callable[..., Any] = _run_voter_thinning_diagnostic,
    recenter_distance_diagnostic_runner: Callable[..., Any] = (
        fvt_recenter_target_distance_diagnostics
    ),
    scanner_downstream_diagnostic_runner: Callable[..., Any] = (_scanner_downstream_diagnostics),
    scanner_stage_loss_diagnostic_runner: Callable[..., Any] = _scanner_stage_loss_diagnostics,
    scanner_boundary_stage_diagnostic_runner: Callable[..., Any] = (
        build_scanner_boundary_stage_diagnostics
    ),
    prepared_inputs: PreparedCaseInputs | None = None,
) -> PipelineEvaluation:
    variant_spec = get_variant_spec(variant)
    valid_input_mode = validate_input_mode(input_mode)
    effective_skinning = effective_skinning_config(variant_spec, skinning_config)
    if prepared_inputs is None:
        prepared_inputs = prepare_case_inputs(
            case,
            scanner_config=scanner_config,
            input_mode=valid_input_mode,
            scanner_backend_matrix=scanner_backend_matrix,
        )
    elif prepared_inputs.case is not case:
        raise ValueError("prepared inputs must belong to the evaluated case")
    if valid_input_mode == "oracle":
        return run_oracle_pipeline(
            case,
            voting_config=voting_config,
            truth_metric_config=truth_metric_config,
            skinning_config=effective_skinning,
            variant_spec=variant_spec,
            include_thinning_diagnostic=include_thinning_diagnostic,
            thinning_diagnostic_runner=thinning_diagnostic_runner,
            recenter_distance_diagnostic_runner=recenter_distance_diagnostic_runner,
            prepared_oracle=prepared_inputs.oracle,
        )

    outputs: dict[str, PipelineEvaluation] = {}
    if valid_input_mode == "both":
        outputs["oracle"] = run_oracle_pipeline(
            case,
            voting_config=voting_config,
            truth_metric_config=truth_metric_config,
            skinning_config=effective_skinning,
            variant_spec=variant_spec,
            include_thinning_diagnostic=include_thinning_diagnostic,
            thinning_diagnostic_runner=thinning_diagnostic_runner,
            recenter_distance_diagnostic_runner=recenter_distance_diagnostic_runner,
            prepared_oracle=prepared_inputs.oracle,
        )
    outputs["scanner"] = run_scanner_pipeline(
        case,
        voting_config=voting_config,
        scanner_config=scanner_config,
        truth_metric_config=truth_metric_config,
        skinning_config=effective_skinning,
        variant_spec=variant_spec,
        scanner_backend_matrix=scanner_backend_matrix,
        include_thinning_diagnostic=include_thinning_diagnostic,
        include_scanner_downstream_diagnostics=include_scanner_downstream_diagnostics,
        include_scanner_boundary_stage_diagnostics=(include_scanner_boundary_stage_diagnostics),
        thinning_diagnostic_runner=thinning_diagnostic_runner,
        recenter_distance_diagnostic_runner=recenter_distance_diagnostic_runner,
        scanner_downstream_diagnostic_runner=scanner_downstream_diagnostic_runner,
        scanner_stage_loss_diagnostic_runner=scanner_stage_loss_diagnostic_runner,
        scanner_boundary_stage_diagnostic_runner=scanner_boundary_stage_diagnostic_runner,
        prepared_scanner=prepared_inputs.scanner,
    )
    active = "scanner" if valid_input_mode == "scanner" else "oracle"
    active_output = outputs[active]
    report = dict(active_output.report_payload)
    report["active_pipeline"] = active
    report["pipelines"] = {name: output.report_payload for name, output in outputs.items()}
    if valid_input_mode == "both":
        volumes: Mapping[str, Any] = {
            PIPELINE_OUTPUTS_KEY: {
                name: output.artifacts.volumes for name, output in outputs.items()
            }
        }
        skins: Mapping[str, Any] = {
            PIPELINE_OUTPUTS_KEY: {
                name: output.artifacts.skins_payload for name, output in outputs.items()
            }
        }
    else:
        volumes = active_output.artifacts.volumes
        skins = active_output.artifacts.skins_payload
    return PipelineEvaluation(report, PipelineArtifacts(volumes, skins))


def case_pipeline_reports(
    variant_reports: Mapping[str, Mapping[str, Any]], input_mode: str
) -> dict[str, dict[str, Any]]:
    names = {"oracle": ("oracle",), "scanner": ("scanner",), "both": ("oracle", "scanner")}[
        validate_input_mode(input_mode)
    ]
    result = {}
    for name in names:
        reports = {
            variant: variant_pipeline_report(report, name)
            for variant, report in variant_reports.items()
        }
        result[name] = {
            "variants": reports,
            "variant_comparison": quality_metrics.variant_comparison(reports),
        }
    return result


def variant_pipeline_report(report: Mapping[str, Any], pipeline: str) -> Mapping[str, Any]:
    pipelines = report.get("pipelines")
    if isinstance(pipelines, Mapping) and pipeline in pipelines:
        value = pipelines[pipeline]
        if not isinstance(value, Mapping):
            raise TypeError(f"pipeline report must be a mapping: {pipeline}")
        return value
    if pipeline == "oracle" and "scanner_quality" not in report:
        return report
    if pipeline == "scanner" and "scanner_quality" in report:
        return report
    raise KeyError(f"missing pipeline report: {pipeline}")


def case_variant_comparison_alias(
    pipelines: Mapping[str, Mapping[str, Any]], input_mode: str
) -> dict[str, Any]:
    valid = validate_input_mode(input_mode)
    if valid == "both":
        return {
            "pipelines": {name: report["variant_comparison"] for name, report in pipelines.items()}
        }
    active = "scanner" if valid == "scanner" else "oracle"
    return dict(pipelines[active]["variant_comparison"])


def build_case_report_model(
    *,
    case_id: str,
    shape: Sequence[int],
    truth: Mapping[str, Any],
    variant_reports: Mapping[str, Mapping[str, Any]],
    input_mode: str,
    include_baseline_config_alias: bool = False,
) -> CaseReport:
    """Build the typed case-level report while retaining legacy baseline aliases."""

    pipelines = case_pipeline_reports(variant_reports, input_mode)
    payload: dict[str, Any] = {
        "case_id": case_id,
        "shape": [int(size) for size in shape],
        "truth": truth,
        "variants": variant_reports,
        "pipelines": pipelines,
        "variant_comparison": case_variant_comparison_alias(pipelines, input_mode),
    }
    if BASELINE_VARIANT in variant_reports:
        payload.update(
            {
                key: value
                for key, value in variant_reports[BASELINE_VARIANT].items()
                if key != "pipelines" and (include_baseline_config_alias or key != "config")
            }
        )
        payload["pipelines"] = pipelines
        payload["variant_comparison"] = case_variant_comparison_alias(pipelines, input_mode)
    return CaseReport.from_dict(payload)


def run_case(
    case_definition: SyntheticQualityCaseDefinition,
    *,
    shape: tuple[int, int, int],
    voting_config: SyntheticVotingConfig = SyntheticVotingConfig(),
    scanner_config: SyntheticScannerConfig = SyntheticScannerConfig(),
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig(),
    skinning_config: SyntheticSkinningConfig = SyntheticSkinningConfig(),
    variant: str = "current_default",
    input_mode: str = "oracle",
    scanner_backend_matrix: bool = False,
    include_thinning_diagnostic: bool = False,
    include_scanner_downstream_diagnostics: bool = False,
    include_scanner_boundary_stage_diagnostics: bool = False,
    thinning_diagnostic_cases: Sequence[str] = DEFAULT_THINNING_DIAGNOSTIC_CASES,
    thinning_diagnostic_runner: Callable[..., Any] = _run_voter_thinning_diagnostic,
    recenter_distance_diagnostic_runner: Callable[..., Any] = (
        fvt_recenter_target_distance_diagnostics
    ),
    scanner_downstream_diagnostic_runner: Callable[..., Any] = (_scanner_downstream_diagnostics),
    scanner_stage_loss_diagnostic_runner: Callable[..., Any] = _scanner_stage_loss_diagnostics,
    scanner_boundary_stage_diagnostic_runner: Callable[..., Any] = (
        build_scanner_boundary_stage_diagnostics
    ),
) -> PipelineEvaluation:
    case = case_definition.factory(shape)
    if case.case_id != case_definition.case_id:
        raise ValueError(
            f"case factory returned {case.case_id!r}, expected {case_definition.case_id!r}"
        )
    diagnostic_ids = set(
        validate_case_ids(
            thinning_diagnostic_cases,
            description="thinning diagnostic",
            sequence_name="thinning_diagnostic_cases",
        )
    )
    evaluation = run_case_variant(
        case,
        voting_config=voting_config,
        scanner_config=scanner_config,
        truth_metric_config=truth_metric_config,
        skinning_config=skinning_config,
        variant=variant,
        input_mode=input_mode,
        scanner_backend_matrix=scanner_backend_matrix,
        include_thinning_diagnostic=(
            include_thinning_diagnostic and case.case_id in diagnostic_ids
        ),
        include_scanner_downstream_diagnostics=include_scanner_downstream_diagnostics,
        include_scanner_boundary_stage_diagnostics=(include_scanner_boundary_stage_diagnostics),
        thinning_diagnostic_runner=thinning_diagnostic_runner,
        recenter_distance_diagnostic_runner=recenter_distance_diagnostic_runner,
        scanner_downstream_diagnostic_runner=scanner_downstream_diagnostic_runner,
        scanner_stage_loss_diagnostic_runner=scanner_stage_loss_diagnostic_runner,
        scanner_boundary_stage_diagnostic_runner=scanner_boundary_stage_diagnostic_runner,
    )
    variant_report = evaluation.report_payload
    report_model = build_case_report_model(
        case_id=case.case_id,
        shape=case.shape,
        truth=quality_metrics.truth_report(case, truth_metric_config),
        variant_reports={variant: variant_report},
        input_mode=input_mode,
        include_baseline_config_alias=True,
    )
    # PipelineEvaluation remains the example-facing compatibility API.
    report = LegacyReportV1Adapter().case_to_dict(report_model)
    return PipelineEvaluation(report, evaluation.artifacts)


def _boundary_seed_retention_v1_seeds(**kwargs: Any) -> tuple[list[Any], list[Any], dict[str, Any]]:
    result = select_boundary_seed_retention_v1(**kwargs)
    return list(result.default_seeds), list(result.selected_seeds), result.diagnostics


def _scanner_backend_matrix_report(
    case: Synthetic3DCase,
    *,
    voting_config: SyntheticVotingConfig,
    scanner_config: SyntheticScannerConfig,
    truth_metric_config: SyntheticTruthMetricConfig,
    skinning_config: SyntheticSkinningConfig,
    variant_spec: VariantSpec,
    include_thinning_diagnostic: bool,
    selected_report: Mapping[str, Any],
    thinning_diagnostic_runner: Callable[..., Any],
    recenter_distance_diagnostic_runner: Callable[..., Any],
    prepared_scanner: PreparedScannerInput | None,
) -> dict[str, Any]:
    reports = {}
    backends = SCANNER_BACKEND_MATRIX_BACKENDS
    if scanner_config.backend not in backends:
        backends = (*backends, scanner_config.backend)
    for backend in backends:
        if backend == scanner_config.backend:
            reports[backend] = dict(selected_report)
        else:
            reports[backend] = dict(
                run_scanner_pipeline(
                    case,
                    voting_config=voting_config,
                    scanner_config=replace(scanner_config, backend=backend),
                    truth_metric_config=truth_metric_config,
                    skinning_config=skinning_config,
                    variant_spec=variant_spec,
                    scanner_backend_matrix=False,
                    include_thinning_diagnostic=include_thinning_diagnostic,
                    include_scanner_downstream_diagnostics=False,
                    include_scanner_boundary_stage_diagnostics=False,
                    thinning_diagnostic_runner=thinning_diagnostic_runner,
                    recenter_distance_diagnostic_runner=recenter_distance_diagnostic_runner,
                    prepared_scanner=prepared_scanner,
                ).report_payload
            )
    return {
        "backends": reports,
        "comparison": _scanner_backend_matrix_comparison(
            reports, selected_backend=scanner_config.backend
        ),
    }


def _prepared_scanner_attributes(
    prepared: PreparedScannerInput,
    scanner_config: SyntheticScannerConfig,
) -> ScannerAttributes:
    expected_config = replace(prepared.config, backend=scanner_config.backend)
    if scanner_config != expected_config:
        raise ValueError("prepared scanner input does not match scanner configuration")
    try:
        return prepared.by_backend[scanner_config.backend]
    except KeyError as error:
        raise ValueError(
            f"prepared scanner input is missing backend {scanner_config.backend!r}"
        ) from error


def _scanner_backend_matrix_comparison(
    reports: Mapping[str, Mapping[str, Any]], *, selected_backend: str
) -> dict[str, Any]:
    values = {
        name: _scanner_backend_matrix_metric_values(report) for name, report in reports.items()
    }
    selected = values.get(selected_backend, {})
    return {
        "selected_backend": selected_backend,
        "metric_values": values,
        "deltas_vs_selected_backend": {
            name: {
                metric: _metric_delta(value, selected.get(metric))
                for metric, value in metrics.items()
            }
            for name, metrics in values.items()
        },
        "best_fvt_positive_buffered_f1_backend": _best_backend(
            values, "fvt_positive_buffered_f1", higher_is_better=True
        ),
        "best_skin_buffered_f1_backend": _best_backend(
            values, "skin_buffered_f1", higher_is_better=True
        ),
        "best_boundary_edge_fp_backend": _best_backend(
            values, "fvt_positive_edge_false_positive_fraction", higher_is_better=False
        ),
    }


def _scanner_backend_matrix_metric_values(report: Mapping[str, Any]) -> dict[str, float | None]:
    quality = report["quality"]
    overlap = quality["fvt_positive_top_truth_count"]["buffered_overlap_radius2"]
    edge = quality["edge_false_positive"]["fvt_positive_top_truth_count"]
    skin = quality["skin"]
    return {
        "fvt_positive_buffered_f1": _finite_metric_or_none(overlap["buffered_f1"]),
        "skin_buffered_f1": (
            None
            if skin is None
            else _finite_metric_or_none(skin["buffered_overlap_radius2"]["buffered_f1"])
        ),
        "fvt_positive_edge_false_positive_fraction": _finite_metric_or_none(
            edge["edge_false_positive_fraction_of_candidates"]
        ),
    }


def _finite_metric_or_none(value: object) -> float | None:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    return metric if np.isfinite(metric) else None


def _metric_delta(value: float | None, baseline: float | None) -> float | None:
    return None if value is None or baseline is None else float(value - baseline)


def _best_backend(
    values: Mapping[str, Mapping[str, float | None]], metric: str, *, higher_is_better: bool
) -> str | None:
    candidates = [
        (name, metrics[metric])
        for name, metrics in values.items()
        if metrics.get(metric) is not None
    ]
    if not candidates:
        return None
    key = (lambda item: item[1]) if higher_is_better else (lambda item: -item[1])
    return max(candidates, key=key)[0]
