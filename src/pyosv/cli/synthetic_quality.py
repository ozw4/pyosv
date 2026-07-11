r"""Build, write, and run controlled 3D synthetic truth quality reports.

Example:
    PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
      --case-set extended \
      --shape 33,33,33 \
      --variants current_default \
      --output-dir outputs/3d/synthetic_quality/extended_001 \
      --pretty \
      --save-figures \
      --write-markdown-index
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import replace
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from pyosv.experimental.boundary_seed_selection import select_boundary_seed_retention_v1
from pyosv.experimental.boundary_thinning import (
    apply_boundary_edge_thin_v1,
    fvt_recenter_target_distance_diagnostics,
    recenter_edge_fvt_to_target,
)
from pyosv.synthetic3d import (
    Synthetic3DCase,
    make_boundary_plane_case,  # noqa: F401 - compatibility export
    validate_shape3,
)
from pyosv.evaluation.synthetic_quality.config import (
    SKINNER_GROWTH_SOURCES,
    SKINNER_METHODS,
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
)
from pyosv.evaluation.synthetic_quality import quality_metrics
from pyosv.evaluation.synthetic_quality.boundary_stage_diagnostics import (
    build_scanner_boundary_stage_diagnostics,
)
from pyosv.evaluation.synthetic_quality.application import (
    _build_report_outputs as _package_build_report_outputs,
    run_case as _package_application_run_case,
)
from pyosv.evaluation.synthetic_quality.diagnostics import (
    _run_voter_thinning_diagnostic,
    _scanner_downstream_diagnostics,
    _scanner_stage_loss_diagnostics,
)
from pyosv.evaluation.synthetic_quality.scanner import (
    SCANNER_BACKENDS,
    SCANNER_ENSEMBLE_COMPONENT_BACKENDS,  # noqa: F401 - compatibility export
    SCANNER_ENSEMBLE_PRIORS,  # noqa: F401 - compatibility export
    SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_BASE,  # noqa: F401 - compatibility export
    SCANNER_ENSEMBLE_QUALITY_CONFIDENCE_SCALE,  # noqa: F401 - compatibility export
    scan_backend_attributes,
    scan_ensemble_attributes,
    scanner_attributes_from_case,
    unit_range_normalize,
)
from pyosv.evaluation.synthetic_quality.pipeline import (
    run_voting_from_attributes as _package_run_voting_from_attributes,
)
from pyosv.evaluation.synthetic_quality.runner import (
    case_pipeline_reports as _package_case_pipeline_reports,
    case_variant_comparison_alias as _package_case_variant_comparison_alias,
    run_case_variant as _package_run_case_variant,
    run_oracle_pipeline as _package_run_oracle_pipeline,
    run_scanner_pipeline as _package_run_scanner_pipeline,
    validate_input_mode as _package_validate_input_mode,
    variant_pipeline_report as _package_variant_pipeline_report,
)
from pyosv.evaluation.synthetic_quality.cases import (
    CASE_IDS,  # noqa: F401 - compatibility export
    CASE_SETS,
    EXTENDED_CASES,  # noqa: F401 - compatibility export
    GEOMETRY_CASES,  # noqa: F401 - compatibility export
    MINIMAL_CASES,  # noqa: F401 - compatibility export
    SyntheticQualityCaseDefinition as SyntheticQualityCaseDefinition,
    validate_case_ids,
)
from pyosv.evaluation.synthetic_quality.profiles import (
    WORKFLOW_MODES,
    _default_skinner_method_for_workflow,  # noqa: F401 - compatibility export
    _default_skinner_min_likelihood_for_method,  # noqa: F401 - compatibility export
    _effective_include_thinning_diagnostic,
    _effective_skinner_method,
    _effective_skinner_min_likelihood,
    _effective_surface_support_policy,
    _effective_voter_thin_mode,
    _validate_workflow_mode,
)
from pyosv.evaluation.reporting.json_v1 import (
    report_to_json as _report_to_json,
    write_metrics_json as _write_metrics_json,
)
from pyosv.evaluation.reporting.csv_v1 import (
    _empty_summary_csv_skin_fallback_v5_guardrail_row as _empty_summary_csv_skin_fallback_v5_guardrail_row,
    _iter_pipeline_reports as _iter_pipeline_reports,
    _summary_csv_boundary_edge_thin_row as _summary_csv_boundary_edge_thin_row,
    _summary_csv_boundary_seed_retention_row as _summary_csv_boundary_seed_retention_row,
    _summary_csv_comparison_row as _summary_csv_comparison_row,
    _summary_csv_fvt_recenter_row as _summary_csv_fvt_recenter_row,
    _summary_csv_scanner_backend_matrix_row as _summary_csv_scanner_backend_matrix_row,
    _summary_csv_scanner_downstream_row as _summary_csv_scanner_downstream_row,
    _summary_csv_scanner_row as _summary_csv_scanner_row,
    _summary_csv_scanner_stage_loss_row as _summary_csv_scanner_stage_loss_row,
    _summary_csv_skin_component_topology_row as _summary_csv_skin_component_topology_row,
    _summary_csv_skin_diagnostics_row as _summary_csv_skin_diagnostics_row,
    _summary_csv_skin_fallback_v5_guardrail_row as _summary_csv_skin_fallback_v5_guardrail_row,
    _summary_csv_skin_row as _summary_csv_skin_row,
    _summary_csv_thinning_diagnostic_row as _summary_csv_thinning_diagnostic_row,
    _summary_csv_voting_row as _summary_csv_voting_row,
    write_summary_csv,
)
from pyosv.evaluation.reporting.artifacts import (
    write_case_figures,
    write_case_skins_json,
    write_case_volumes,
)
from pyosv.evaluation.reporting.markdown_v1 import write_visual_report_markdown
from pyosv.evaluation.synthetic_quality.variants import (
    DEFAULT_VARIANTS,
    QUALITY_MATRIX_VARIANTS,  # noqa: F401 - compatibility export
    VARIANT_NAMES,
    VARIANT_PRESETS,
    VariantSpec,
    effective_skinning_config,
    effective_thin_mode,
    get_variant_spec,
    resolve_variants,
    validate_variant_preset,
    validate_variants,
)
from pyosv.cells import FaultCell
from pyosv.voting3d import OptimalSurfaceVoter  # noqa: F401 - compatibility export

DEFAULT_SHAPE = (33, 33, 33)
FORMAT_VERSION = 1
EDGE_FALSE_POSITIVE_MARGIN = quality_metrics.EDGE_FALSE_POSITIVE_MARGIN
NONZERO_EPSILON = quality_metrics.NONZERO_EPSILON
DEFAULT_VARIANT_PRESET = "default"
DEFAULT_THINNING_DIAGNOSTIC_CASES = ("curved_surface",)
SCANNER_BACKEND_MATRIX_BACKENDS = ("reference-like", "quality", "fast")
VARIANT_COMPARISON_METRICS = quality_metrics.VARIANT_COMPARISON_METRICS


def parse_workflow_mode(text: str) -> str:
    try:
        return _validate_workflow_mode(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _effective_skinning_config_for_variant(
    *,
    skinning_config: SyntheticSkinningConfig,
    variant: str,
) -> SyntheticSkinningConfig:
    return effective_skinning_config(get_variant_spec(variant), skinning_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run controlled 3D synthetic truth-quality reports.",
        epilog=(
            "Example:\n"
            "  PYTHONPATH=src python examples/report_3d_synthetic_quality.py \\\n"
            "    --case-set extended \\\n"
            "    --shape 33,33,33 \\\n"
            "    --variants current_default,boundary_aware_voter_v1,"
            "no_surface_orientation_smoothing,"
            "final_norm_smoothing_1,voter_thin_normal,voter_thin_hybrid,"
            "voter_thin_hybrid_v2,voter_thin_normal_plateau,"
            "surface_support_weighted,quality_skinner_v2,"
            "quality_boundary_skinner_fallback,"
            "quality_boundary_skinner_fallback_v2,"
            "quality_boundary_skinner_fallback_v3,"
            "quality_boundary_skinner_fallback_v4,"
            "quality_boundary_skinner_fallback_v5 \\\n"
            "    --output-dir outputs/3d/synthetic_quality/extended_001 \\\n"
            "    --pretty \\\n"
            "    --save-figures \\\n"
            "    --write-markdown-index"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--case-set",
        choices=tuple(CASE_SETS),
        default="minimal",
        help="Synthetic case set to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where metrics.json and summary.csv are written.",
    )
    parser.add_argument(
        "--shape",
        type=parse_shape3,
        default=DEFAULT_SHAPE,
        help="Synthetic volume shape in n3,n2,n1 order.",
    )
    parser.add_argument("--pretty", action="store_true", help="Write indented JSON.")
    parser.add_argument(
        "--save-volumes",
        action="store_true",
        help="Write truth, oracle, and Python DAT volumes under each case directory.",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Write static PNG center-slice figures under each case directory.",
    )
    parser.add_argument(
        "--write-markdown-index",
        action="store_true",
        help="Write visual_report.md under OUTPUT_DIR.",
    )
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=None,
        help=(
            "Comma-separated diagnostic variants to run; overrides --variant-preset when provided."
        ),
    )
    parser.add_argument(
        "--variant-preset",
        choices=tuple(VARIANT_PRESETS),
        default=DEFAULT_VARIANT_PRESET,
        help="Named variant preset used when --variants is omitted.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("oracle", "scanner", "both"),
        default="oracle",
        help=(
            "Input path: oracle evaluates voting/thinning/skinning from truth "
            "attributes, scanner evaluates scanner plus downstream stages, both runs both."
        ),
    )
    parser.add_argument(
        "--workflow-mode",
        type=parse_workflow_mode,
        default="reference",
        metavar="{" + ",".join(WORKFLOW_MODES) + "}",
        help=(
            "Workflow defaults: reference keeps reference-like voter thinning, "
            "quality uses hybrid_v2 voter thinning with support-aware voting inactive, "
            "diagnostic keeps reference thinning and enables reference-vs-normal "
            "diagnostics."
        ),
    )
    parser.add_argument(
        "--scanner-backend",
        choices=SCANNER_BACKENDS,
        default="reference-like",
        help=(
            "FaultOrientScanner3 backend for scanner/both mode: reference-like scan, "
            "fast scan, refined quality scan, or diagnostic ensemble scan."
        ),
    )
    parser.add_argument(
        "--scanner-backend-matrix",
        action="store_true",
        help=(
            "Run reference-like, quality, and fast scanner backends as an opt-in "
            "diagnostic matrix for scanner/both input mode."
        ),
    )
    parser.add_argument(
        "--scanner-downstream-diagnostics",
        action="store_true",
        help=(
            "Add scanner downstream fv/fvt retention and thinning-mode diagnostics "
            "for scanner/both input mode."
        ),
    )
    parser.add_argument(
        "--scanner-boundary-stage-diagnostics",
        action="store_true",
        help=(
            "Add detailed scanner boundary stage profiles and transitions for "
            "scanner/both input mode."
        ),
    )
    parser.add_argument(
        "--scanner-phi-min",
        type=float,
        default=0.0,
        help="Minimum scanner strike angle in degrees.",
    )
    parser.add_argument(
        "--scanner-phi-max",
        type=float,
        default=180.0,
        help="Maximum scanner strike angle in degrees.",
    )
    parser.add_argument(
        "--scanner-theta-min",
        type=float,
        default=45.0,
        help="Minimum scanner dip angle in degrees.",
    )
    parser.add_argument(
        "--scanner-theta-max",
        type=float,
        default=90.0,
        help="Maximum scanner dip angle in degrees.",
    )
    parser.add_argument(
        "--scanner-sigma1",
        type=float,
        default=2.0,
        help="Scanner sigma1 control.",
    )
    parser.add_argument(
        "--scanner-sigma2",
        type=float,
        default=2.0,
        help="Scanner sigma2 control.",
    )
    parser.add_argument(
        "--scanner-refinement-factor",
        type=int,
        default=2,
        help="Refined sampling factor used by --scanner-backend quality.",
    )
    parser.add_argument(
        "--scanner-thin-mode",
        choices=("none", "reference", "normal"),
        default="reference",
        help="Scanner attribute thinning before voting: none, reference strike-bin, or normal.",
    )
    parser.add_argument(
        "--keep-scanner-edge-effects",
        action="store_true",
        help="Keep scanner reference-thin edge effects for diagnostics.",
    )
    parser.add_argument("--ru", type=int, default=1, help="Voting shift radius in u.")
    parser.add_argument("--rv", type=int, default=2, help="Voting shift radius in v.")
    parser.add_argument("--rw", type=int, default=2, help="Voting shift radius in w.")
    parser.add_argument(
        "--seed-distance",
        type=int,
        default=3,
        help="Minimum seed spacing used by the voter.",
    )
    parser.add_argument(
        "--seed-threshold",
        type=float,
        default=0.5,
        help="Oracle ft threshold used for seed selection.",
    )
    parser.add_argument(
        "--attribute-smoothing",
        type=int,
        default=0,
        help="Number of voter attribute smoothing passes.",
    )
    parser.add_argument(
        "--voter-thin-mode",
        choices=("reference", "normal", "hybrid", "hybrid_v2", "normal_plateau"),
        default=None,
        help="Thinning mode passed to OptimalSurfaceVoter.thin(); defaults by workflow mode.",
    )
    parser.add_argument(
        "--reference-thin-sigma",
        type=float,
        default=1.0,
        help="Smoothing sigma used by reference-like thinning.",
    )
    parser.add_argument(
        "--surface-support-min-fraction",
        type=float,
        default=None,
        help=(
            "Explicit minimum valid surface-vote support fraction override; omitted "
            "workflow defaults leave support-aware voting inactive."
        ),
    )
    parser.add_argument(
        "--surface-support-exponent",
        type=float,
        default=None,
        help=(
            "Explicit support-fraction vote down-weighting exponent override; omitted "
            "workflow defaults leave support-aware voting inactive."
        ),
    )
    parser.add_argument(
        "--thinning-diagnostics",
        "--include-thinning-diagnostic",
        dest="include_thinning_diagnostic",
        action="store_true",
        help="Add reference-vs-normal voter thinning diagnostics to metrics.json.",
    )
    parser.add_argument(
        "--thinning-diagnostic-cases",
        type=parse_thinning_diagnostic_cases,
        default=DEFAULT_THINNING_DIAGNOSTIC_CASES,
        help="Comma-separated case IDs for thinning diagnostics.",
    )
    parser.add_argument(
        "--truth-surface-half-width",
        type=float,
        default=0.5,
        help="Half-width around the truth surface used for thin-surface metrics.",
    )
    parser.add_argument(
        "--buffer-radius",
        type=float,
        default=2.0,
        help="Distance radius used for buffered overlap metrics.",
    )
    parser.add_argument(
        "--skip-skinning",
        action="store_true",
        help="Skip FaultSkinner skin extraction and skin truth metrics.",
    )
    parser.add_argument(
        "--skinner-min-likelihood",
        type=float,
        default=None,
        help=(
            "Minimum thinned vote likelihood for FaultSkinner; defaults by workflow/skinner method."
        ),
    )
    parser.add_argument(
        "--skinner-method",
        choices=SKINNER_METHODS,
        default=None,
        help="FaultSkinner backend: reference, quality, or connected_component.",
    )
    parser.add_argument(
        "--skinner-growth-source",
        choices=SKINNER_GROWTH_SOURCES,
        default="thinned",
        help="FaultSkinner growth source: thinned or pre_thin.",
    )
    parser.add_argument(
        "--skinner-min-skin-size",
        type=parse_optional_nonnegative_int,
        default=1,
        help="Minimum skin size kept by FaultSkinner, or 'none'.",
    )
    parser.add_argument("--skinner-d", type=int, default=1, help="Skinner seed distance.")
    parser.add_argument(
        "--skinner-ru",
        type=int,
        default=10,
        help="Synthetic report skinner u search radius.",
    )
    parser.add_argument(
        "--skinner-rv",
        type=parse_optional_nonnegative_int,
        default=None,
        help="Skinner v search radius, or 'none' to use the backend default.",
    )
    parser.add_argument(
        "--skinner-rw",
        type=parse_optional_nonnegative_int,
        default=None,
        help="Skinner w search radius, or 'none' to use the backend default.",
    )
    parser.add_argument(
        "--skinner-max-steps",
        type=int,
        default=10,
        help="Maximum local skin growth steps.",
    )
    parser.add_argument(
        "--skinner-du",
        type=float,
        default=5.0,
        help="Maximum local u displacement between linked skin cells.",
    )
    parser.add_argument(
        "--skinner-max-delta-strike",
        type=float,
        default=30.0,
        help="Maximum strike change between linked skin cells in degrees.",
    )
    parser.add_argument(
        "--no-skinner-reskin",
        action="store_true",
        help="Disable reference-like reskin smoothing/reorientation.",
    )
    parser.add_argument(
        "--skinner-accepted-occupancy-radius",
        type=parse_optional_nonnegative_int,
        default=None,
        help=(
            "Accepted skin occupancy radius for reference-like FaultSkinner, "
            "or 'none' for the backend default."
        ),
    )
    parser.add_argument(
        "--small-skin-size",
        type=int,
        default=10,
        help="Skin size threshold for small-skin topology metrics.",
    )
    return parser


def parse_shape3(text: str) -> tuple[int, int, int]:
    """Parse a 3D shape string as ``(n3, n2, n1)``."""
    try:
        parts = tuple(int(part) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be three comma-separated integers") from error
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be three comma-separated integers")
    try:
        return validate_shape3(parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_variants(text: str) -> tuple[str, ...]:
    """Parse a comma-separated diagnostic variant list."""

    variants = tuple(part.strip() for part in text.split(",") if part.strip())
    if not variants:
        raise argparse.ArgumentTypeError("variants must include at least one variant")
    unknown = sorted(set(variants).difference(VARIANT_NAMES))
    if unknown:
        valid = ",".join(VARIANT_NAMES)
        raise argparse.ArgumentTypeError(
            f"unknown variant(s): {','.join(unknown)}; choices: {valid}"
        )
    duplicates = {variant for variant in variants if variants.count(variant) > 1}
    if duplicates:
        raise argparse.ArgumentTypeError(f"duplicate variant(s): {','.join(sorted(duplicates))}")
    return variants


def parse_thinning_diagnostic_cases(text: str) -> tuple[str, ...]:
    """Parse a comma-separated thinning diagnostic case list."""

    case_ids = tuple(part.strip() for part in text.split(",") if part.strip())
    try:
        return _validate_thinning_diagnostic_cases(case_ids)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_optional_nonnegative_int(text: str) -> int | None:
    """Parse a non-negative integer or a textual None value."""

    if text.lower() in {"none", "null"}:
        return None
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "value must be a non-negative integer or 'none'"
        ) from error
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer or 'none'")
    return value


def run_case(
    case_definition: SyntheticQualityCaseDefinition,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    kwargs.setdefault("thinning_diagnostic_runner", _run_voter_thinning_diagnostic)
    kwargs.setdefault(
        "recenter_distance_diagnostic_runner", fvt_recenter_target_distance_diagnostics
    )
    kwargs.setdefault("scanner_downstream_diagnostic_runner", _scanner_downstream_diagnostics)
    kwargs.setdefault("scanner_stage_loss_diagnostic_runner", _scanner_stage_loss_diagnostics)
    kwargs.setdefault(
        "scanner_boundary_stage_diagnostic_runner",
        build_scanner_boundary_stage_diagnostics,
    )
    return _package_application_run_case(case_definition, **kwargs)


def _run_case_variant(
    case: Synthetic3DCase,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    kwargs.setdefault("thinning_diagnostic_runner", _run_voter_thinning_diagnostic)
    kwargs.setdefault(
        "recenter_distance_diagnostic_runner", fvt_recenter_target_distance_diagnostics
    )
    kwargs.setdefault("scanner_downstream_diagnostic_runner", _scanner_downstream_diagnostics)
    kwargs.setdefault("scanner_stage_loss_diagnostic_runner", _scanner_stage_loss_diagnostics)
    kwargs.setdefault(
        "scanner_boundary_stage_diagnostic_runner",
        build_scanner_boundary_stage_diagnostics,
    )
    evaluation = _package_run_case_variant(case, **kwargs)
    return (
        dict(evaluation.report_payload),
        dict(evaluation.artifacts.volumes),
        dict(evaluation.artifacts.skins_payload),
    )


def _validate_input_mode(input_mode: str) -> str:
    return _package_validate_input_mode(input_mode)


def _case_pipeline_reports(
    variant_reports: Mapping[str, Mapping[str, Any]],
    input_mode: str,
) -> dict[str, dict[str, Any]]:
    return _package_case_pipeline_reports(variant_reports, input_mode)


def _variant_pipeline_report(
    variant_report: Mapping[str, Any],
    pipeline: str,
) -> Mapping[str, Any]:
    return _package_variant_pipeline_report(variant_report, pipeline)


def _case_variant_comparison_alias(
    pipelines: Mapping[str, Mapping[str, Any]],
    input_mode: str,
) -> dict[str, Any]:
    return _package_case_variant_comparison_alias(pipelines, input_mode)


def _run_oracle_pipeline(
    case: Synthetic3DCase,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    kwargs.setdefault("thinning_diagnostic_runner", _run_voter_thinning_diagnostic)
    kwargs.setdefault(
        "recenter_distance_diagnostic_runner", fvt_recenter_target_distance_diagnostics
    )
    evaluation = _package_run_oracle_pipeline(case, **kwargs)
    return (
        dict(evaluation.report_payload),
        dict(evaluation.artifacts.volumes),
        dict(evaluation.artifacts.skins_payload),
    )


def _run_scanner_pipeline(
    case: Synthetic3DCase,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    kwargs.setdefault("thinning_diagnostic_runner", _run_voter_thinning_diagnostic)
    kwargs.setdefault(
        "recenter_distance_diagnostic_runner", fvt_recenter_target_distance_diagnostics
    )
    kwargs.setdefault("scanner_downstream_diagnostic_runner", _scanner_downstream_diagnostics)
    kwargs.setdefault("scanner_stage_loss_diagnostic_runner", _scanner_stage_loss_diagnostics)
    kwargs.setdefault(
        "scanner_boundary_stage_diagnostic_runner",
        build_scanner_boundary_stage_diagnostics,
    )
    evaluation = _package_run_scanner_pipeline(case, **kwargs)
    return (
        dict(evaluation.report_payload),
        dict(evaluation.artifacts.volumes),
        dict(evaluation.artifacts.skins_payload),
    )


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
) -> dict[str, Any]:
    backend_reports: dict[str, Any] = {}
    matrix_backends = SCANNER_BACKEND_MATRIX_BACKENDS
    if scanner_config.backend not in matrix_backends:
        matrix_backends = (*matrix_backends, scanner_config.backend)
    for backend in matrix_backends:
        if backend == scanner_config.backend:
            backend_reports[backend] = dict(selected_report)
        else:
            backend_report, _, _ = _run_scanner_pipeline(
                case,
                voting_config=voting_config,
                scanner_config=replace(scanner_config, backend=backend),
                truth_metric_config=truth_metric_config,
                skinning_config=skinning_config,
                variant_spec=variant_spec,
                scanner_backend_matrix=False,
                include_thinning_diagnostic=include_thinning_diagnostic,
                include_scanner_downstream_diagnostics=False,
            )
            backend_reports[backend] = backend_report
    return {
        "backends": backend_reports,
        "comparison": _scanner_backend_matrix_comparison(
            backend_reports,
            selected_backend=scanner_config.backend,
        ),
    }


def _scanner_backend_matrix_comparison(
    backend_reports: Mapping[str, Mapping[str, Any]],
    *,
    selected_backend: str,
) -> dict[str, Any]:
    metric_values = {
        backend: _scanner_backend_matrix_metric_values(report)
        for backend, report in backend_reports.items()
    }
    selected_values = metric_values.get(selected_backend, {})
    deltas_vs_selected = {
        backend: {
            metric: _metric_delta(value, selected_values.get(metric))
            for metric, value in values.items()
        }
        for backend, values in metric_values.items()
    }
    return {
        "selected_backend": selected_backend,
        "metric_values": metric_values,
        "deltas_vs_selected_backend": deltas_vs_selected,
        "best_fvt_positive_buffered_f1_backend": _best_backend(
            metric_values,
            "fvt_positive_buffered_f1",
            higher_is_better=True,
        ),
        "best_skin_buffered_f1_backend": _best_backend(
            metric_values,
            "skin_buffered_f1",
            higher_is_better=True,
        ),
        "best_boundary_edge_fp_backend": _best_backend(
            metric_values,
            "fvt_positive_edge_false_positive_fraction",
            higher_is_better=False,
        ),
    }


def _scanner_backend_matrix_metric_values(report: Mapping[str, Any]) -> dict[str, float | None]:
    quality = report["quality"]
    fvt_positive_overlap = quality["fvt_positive_top_truth_count"]["buffered_overlap_radius2"]
    edge_false_positive = quality["edge_false_positive"]["fvt_positive_top_truth_count"]
    skin_quality = quality["skin"]
    skin_buffered_f1 = None
    if skin_quality is not None:
        skin_buffered_f1 = _finite_metric_or_none(
            skin_quality["buffered_overlap_radius2"]["buffered_f1"]
        )
    return {
        "fvt_positive_buffered_f1": _finite_metric_or_none(fvt_positive_overlap["buffered_f1"]),
        "skin_buffered_f1": skin_buffered_f1,
        "fvt_positive_edge_false_positive_fraction": _finite_metric_or_none(
            edge_false_positive["edge_false_positive_fraction_of_candidates"]
        ),
    }


def _finite_metric_or_none(value: object) -> float | None:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(metric):
        return None
    return metric


def _metric_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value - baseline)


def _best_backend(
    metric_values: Mapping[str, Mapping[str, float | None]],
    metric: str,
    *,
    higher_is_better: bool,
) -> str | None:
    candidates = [
        (backend, values[metric])
        for backend, values in metric_values.items()
        if values.get(metric) is not None
    ]
    if not candidates:
        return None
    return sorted(
        candidates,
        key=lambda item: item[1] if higher_is_better else -item[1],
        reverse=True,
    )[0][0]


def _scanner_attributes_from_case(
    case: Synthetic3DCase,
    scanner_config: SyntheticScannerConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    attributes = scanner_attributes_from_case(
        case,
        scanner_config,
        backend_scan=_scan_backend_attributes,
        ensemble_scan=_scan_ensemble_attributes,
    )
    return dict(attributes.report), dict(attributes.volumes)


def _scan_backend_attributes(
    scanner: Any,
    scanner_config: SyntheticScannerConfig,
    scanner_input: np.ndarray,
    backend: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    return scan_backend_attributes(scanner, scanner_config, scanner_input, backend)


def _scan_ensemble_attributes(
    scanner: Any,
    scanner_config: SyntheticScannerConfig,
    scanner_input: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, None, dict[str, Any]]:
    return scan_ensemble_attributes(
        scanner,
        scanner_config,
        scanner_input,
        backend_scan=_scan_backend_attributes,
    )


def _unit_range_normalize(array: np.ndarray) -> np.ndarray:
    return unit_range_normalize(array)


def _scanner_truth_quality(
    case: Synthetic3DCase,
    *,
    scanner_volumes: Mapping[str, np.ndarray],
    truth_metric_config: SyntheticTruthMetricConfig,
) -> dict[str, Any]:
    return quality_metrics.scanner_truth_quality(
        case,
        scanner_volumes=scanner_volumes,
        truth_metric_config=truth_metric_config,
    )


def _top_truth_count_quality(
    candidate_mask: np.ndarray,
    *,
    truth_fault_mask: np.ndarray,
    truth_surface_mask: np.ndarray,
    buffer_radius: float,
) -> dict[str, Any]:
    return quality_metrics.top_truth_count_quality(
        candidate_mask,
        truth_fault_mask=truth_fault_mask,
        truth_surface_mask=truth_surface_mask,
        buffer_radius=buffer_radius,
    )


def _boundary_seed_retention_v1_seeds(
    **kwargs: Any,
) -> tuple[list[FaultCell], list[FaultCell], dict[str, Any]]:
    result = select_boundary_seed_retention_v1(**kwargs)
    return list(result.default_seeds), list(result.selected_seeds), result.diagnostics


def _thin_mode_for_variant(variant: str, voting_config: SyntheticVotingConfig) -> str:
    return effective_thin_mode(get_variant_spec(variant), voting_config)


def _positive_candidate_count(array: np.ndarray) -> int:
    return quality_metrics.positive_candidate_count(array)


def _positive_candidate_mask(array: np.ndarray) -> np.ndarray:
    return quality_metrics.positive_candidate_mask(array)


def _candidate_count(mask: np.ndarray) -> int:
    return quality_metrics.candidate_count(mask)


def _positive_pair_overlap(
    *,
    candidate_name: str,
    reference_name: str,
    candidate_mask: np.ndarray,
    reference_mask: np.ndarray,
    buffer_radius: float,
) -> dict[str, str | float | int]:
    return quality_metrics.positive_pair_overlap(
        candidate_name=candidate_name,
        reference_name=reference_name,
        candidate_mask=candidate_mask,
        reference_mask=reference_mask,
        buffer_radius=buffer_radius,
    )


def _fraction_or_zero(numerator: int, denominator: int) -> float:
    return quality_metrics.fraction_or_zero(numerator, denominator)


def _edge_candidate_fraction(candidate_mask: np.ndarray, *, edge_margin: int) -> float:
    return quality_metrics.edge_candidate_fraction(candidate_mask, edge_margin=edge_margin)


def _edge_mask(shape: tuple[int, ...], margin: int) -> np.ndarray:
    return quality_metrics.edge_mask(shape, margin)


def _recenter_edge_fvt_to_target(*args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
    result = recenter_edge_fvt_to_target(*args, **kwargs)
    return result.output, result.diagnostics


def _fvt_recenter_target_distance_diagnostics(**kwargs: Any) -> dict[str, float | None]:
    return fvt_recenter_target_distance_diagnostics(**kwargs)


def _apply_boundary_edge_thin_v1(*args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
    result = apply_boundary_edge_thin_v1(*args, **kwargs)
    return result.output, result.diagnostics


def _scanner_input_association(
    scanner_input: np.ndarray,
    *,
    truth_surface_mask: np.ndarray,
    far_from_truth_mask: np.ndarray,
) -> dict[str, float]:
    return quality_metrics.scanner_input_association(
        scanner_input,
        truth_surface_mask=truth_surface_mask,
        far_from_truth_mask=far_from_truth_mask,
    )


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    return quality_metrics.masked_mean(values, mask)


def _run_voting_from_attributes(
    case: Synthetic3DCase,
    **kwargs: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    kwargs.setdefault("thinning_diagnostic_runner", _run_voter_thinning_diagnostic)
    kwargs.setdefault(
        "recenter_distance_diagnostic_runner", fvt_recenter_target_distance_diagnostics
    )
    evaluation = _package_run_voting_from_attributes(case, **kwargs)
    return (
        dict(evaluation.report_payload),
        dict(evaluation.artifacts.volumes),
        dict(evaluation.artifacts.skins_payload),
    )


def _skins_json_payload(skins: Sequence[Any]) -> dict[str, Any]:
    serialized_skins = []
    for skin_index, skin in enumerate(skins):
        cells = sorted(skin, key=lambda cell: (int(cell.i3), int(cell.i2), int(cell.i1)))
        serialized_skins.append(
            {
                "skin_index": int(skin_index),
                "cell_count": int(len(cells)),
                "cells": [_skin_cell_json(cell) for cell in cells],
            }
        )
    return {
        "format_version": FORMAT_VERSION,
        "skinning_enabled": True,
        "skin_count": int(len(serialized_skins)),
        "skins": serialized_skins,
    }


def _disabled_skins_json_payload() -> dict[str, Any]:
    return {
        "format_version": FORMAT_VERSION,
        "skinning_enabled": False,
        "skin_count": 0,
        "skins": [],
    }


def _skin_cell_json(cell: Any) -> dict[str, float | int]:
    return {
        "x1": float(cell.x1),
        "x2": float(cell.x2),
        "x3": float(cell.x3),
        "i1": int(cell.i1),
        "i2": int(cell.i2),
        "i3": int(cell.i3),
        "fl": float(cell.fl),
        "fp": float(cell.fp),
        "ft": float(cell.ft),
    }


def _argv_has_long_option(argv: Sequence[str], option: str) -> bool:
    option_with_value = f"{option}="
    return any(arg == option or arg.startswith(option_with_value) for arg in argv)


def _truth_report(
    case: Synthetic3DCase,
    truth_metric_config: SyntheticTruthMetricConfig,
) -> dict[str, int]:
    return quality_metrics.truth_report(case, truth_metric_config)


def _build_report_and_volumes(
    **kwargs: Any,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, dict[str, Any]]],
]:
    kwargs.setdefault("thinning_diagnostic_runner", _run_voter_thinning_diagnostic)
    kwargs.setdefault(
        "recenter_distance_diagnostic_runner", fvt_recenter_target_distance_diagnostics
    )
    kwargs.setdefault("scanner_downstream_diagnostic_runner", _scanner_downstream_diagnostics)
    kwargs.setdefault("scanner_stage_loss_diagnostic_runner", _scanner_stage_loss_diagnostics)
    kwargs.setdefault(
        "scanner_boundary_stage_diagnostic_runner",
        build_scanner_boundary_stage_diagnostics,
    )
    return _package_build_report_outputs(**kwargs)


def build_report(**kwargs: Any) -> dict[str, Any]:
    report, _, _ = _build_report_and_volumes(**kwargs)
    return report


def _validate_variants(variants: Sequence[str]) -> tuple[str, ...]:
    return validate_variants(variants)


def _validate_variant_preset(variant_preset: str) -> str:
    return validate_variant_preset(variant_preset)


def _resolve_variants(
    *,
    variants: Sequence[str] | None,
    variant_preset: str,
) -> tuple[str, ...]:
    return resolve_variants(variants=variants, variant_preset=variant_preset)


def _validate_thinning_diagnostic_cases(case_ids: Sequence[str]) -> tuple[str, ...]:
    return validate_case_ids(
        case_ids,
        description="thinning diagnostic",
        sequence_name="thinning_diagnostic_cases",
    )


def _variant_comparison(
    variant_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return quality_metrics.variant_comparison(variant_reports)


def _metric_value(report: Mapping[str, Any], path: Sequence[str]) -> float | None:
    return quality_metrics.metric_value(report, path)


def _delta_or_none(value: float | None, baseline_value: float | None) -> float | None:
    return quality_metrics.delta_or_none(value, baseline_value)


def report_to_json(report: Mapping[str, Any], *, pretty: bool = False) -> str:
    return _report_to_json(report, pretty=pretty)


def write_metrics_json(
    report: Mapping[str, Any],
    output_dir: str | PathLike[str],
    *,
    pretty: bool = False,
) -> Path:
    return _write_metrics_json(report, output_dir, pretty=pretty)


def _normalize_report_skin_metric_keys(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return quality_metrics.normalize_report_skin_metric_keys(metrics)


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    values = np.asarray(array)
    finite = np.isfinite(values)
    finite_values = values[finite].astype(np.float64, copy=False)
    if finite_values.size:
        minimum = float(np.min(finite_values))
        maximum = float(np.max(finite_values))
        mean = float(np.mean(finite_values))
    else:
        minimum = float("nan")
        maximum = float("nan")
        mean = float("nan")

    return {
        "shape": [int(size) for size in values.shape],
        "finite_count": int(np.count_nonzero(finite)),
        "finite_fraction": (float(np.count_nonzero(finite) / values.size) if values.size else 0.0),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "nonzero_fraction": (
            float(np.count_nonzero(np.abs(values) > NONZERO_EPSILON) / values.size)
            if values.size
            else 0.0
        ),
    }


def run_example(
    *,
    output_dir: str | PathLike[str],
    case_set: str = "minimal",
    shape: tuple[int, int, int] = DEFAULT_SHAPE,
    voting_config: SyntheticVotingConfig | None = None,
    scanner_config: SyntheticScannerConfig = SyntheticScannerConfig(),
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig(),
    skinning_config: SyntheticSkinningConfig = SyntheticSkinningConfig(),
    variants: Sequence[str] = DEFAULT_VARIANTS,
    variant_preset: str = DEFAULT_VARIANT_PRESET,
    input_mode: str = "oracle",
    scanner_backend_matrix: bool = False,
    workflow_mode: str = "reference",
    skinner_method_explicit: bool = False,
    skinner_min_likelihood_explicit: bool = False,
    skinner_growth_source_explicit: bool = False,
    skinner_accepted_occupancy_radius_explicit: bool = False,
    pretty: bool = False,
    save_volumes: bool = False,
    save_figures: bool = False,
    write_markdown_index: bool = False,
    include_thinning_diagnostic: bool = False,
    include_scanner_downstream_diagnostics: bool = False,
    include_scanner_boundary_stage_diagnostics: bool = False,
    thinning_diagnostic_cases: Sequence[str] = DEFAULT_THINNING_DIAGNOSTIC_CASES,
) -> dict[str, Any]:
    report, volume_outputs, skin_outputs = _build_report_and_volumes(
        case_set=case_set,
        shape=shape,
        voting_config=voting_config,
        scanner_config=scanner_config,
        truth_metric_config=truth_metric_config,
        skinning_config=skinning_config,
        variants=variants,
        variant_preset=variant_preset,
        input_mode=input_mode,
        scanner_backend_matrix=scanner_backend_matrix,
        workflow_mode=workflow_mode,
        skinner_method_explicit=skinner_method_explicit,
        skinner_min_likelihood_explicit=skinner_min_likelihood_explicit,
        skinner_growth_source_explicit=skinner_growth_source_explicit,
        skinner_accepted_occupancy_radius_explicit=skinner_accepted_occupancy_radius_explicit,
        include_thinning_diagnostic=include_thinning_diagnostic,
        include_scanner_downstream_diagnostics=include_scanner_downstream_diagnostics,
        include_scanner_boundary_stage_diagnostics=(include_scanner_boundary_stage_diagnostics),
        thinning_diagnostic_cases=thinning_diagnostic_cases,
    )
    write_metrics_json(report, output_dir, pretty=pretty)
    write_summary_csv(report, output_dir)
    if save_volumes:
        write_case_volumes(volume_outputs, output_dir)
        write_case_skins_json(skin_outputs, output_dir)
    if save_figures:
        write_case_figures(
            volume_outputs,
            output_dir,
            buffer_radius=truth_metric_config.buffer_radius,
        )
    if write_markdown_index:
        write_visual_report_markdown(report, output_dir)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    voter_thin_mode = _effective_voter_thin_mode(
        workflow_mode=args.workflow_mode,
        voter_thin_mode=args.voter_thin_mode,
    )
    surface_support_min_fraction, surface_support_exponent = _effective_surface_support_policy(
        workflow_mode=args.workflow_mode,
        min_fraction=args.surface_support_min_fraction,
        exponent=args.surface_support_exponent,
    )
    include_thinning_diagnostic = _effective_include_thinning_diagnostic(
        workflow_mode=args.workflow_mode,
        include_thinning_diagnostic=args.include_thinning_diagnostic,
    )
    skinner_method = _effective_skinner_method(
        workflow_mode=args.workflow_mode,
        skinner_method=args.skinner_method,
    )
    skinner_min_likelihood = _effective_skinner_min_likelihood(
        skinner_method=skinner_method,
        min_likelihood=args.skinner_min_likelihood,
    )
    variants = _resolve_variants(
        variants=args.variants,
        variant_preset=args.variant_preset,
    )

    try:
        run_example(
            case_set=args.case_set,
            output_dir=args.output_dir,
            shape=args.shape,
            voting_config=SyntheticVotingConfig(
                ru=args.ru,
                rv=args.rv,
                rw=args.rw,
                seed_distance=args.seed_distance,
                seed_threshold=args.seed_threshold,
                attribute_smoothing=args.attribute_smoothing,
                voter_thin_mode=voter_thin_mode,
                reference_thin_sigma=args.reference_thin_sigma,
                surface_support_min_fraction=surface_support_min_fraction,
                surface_support_exponent=surface_support_exponent,
            ),
            scanner_config=SyntheticScannerConfig(
                backend=args.scanner_backend,
                phi_min=args.scanner_phi_min,
                phi_max=args.scanner_phi_max,
                theta_min=args.scanner_theta_min,
                theta_max=args.scanner_theta_max,
                sigma1=args.scanner_sigma1,
                sigma2=args.scanner_sigma2,
                refinement_factor=args.scanner_refinement_factor,
                scanner_thin_mode=args.scanner_thin_mode,
                remove_edge_effects=not args.keep_scanner_edge_effects,
            ),
            truth_metric_config=SyntheticTruthMetricConfig(
                truth_surface_half_width=args.truth_surface_half_width,
                buffer_radius=args.buffer_radius,
            ),
            skinning_config=SyntheticSkinningConfig(
                enabled=not args.skip_skinning,
                method=skinner_method,
                growth_source=args.skinner_growth_source,
                min_likelihood=skinner_min_likelihood,
                min_skin_size=args.skinner_min_skin_size,
                d=args.skinner_d,
                ru=args.skinner_ru,
                rv=args.skinner_rv,
                rw=args.skinner_rw,
                max_steps=args.skinner_max_steps,
                du=args.skinner_du,
                max_delta_strike=args.skinner_max_delta_strike,
                reskin=not args.no_skinner_reskin,
                accepted_occupancy_radius=args.skinner_accepted_occupancy_radius,
                small_skin_size=args.small_skin_size,
            ),
            variants=variants,
            variant_preset=args.variant_preset,
            input_mode=args.input_mode,
            scanner_backend_matrix=args.scanner_backend_matrix,
            workflow_mode=args.workflow_mode,
            skinner_method_explicit=args.skinner_method is not None,
            skinner_min_likelihood_explicit=_argv_has_long_option(
                raw_argv,
                "--skinner-min-likelihood",
            ),
            skinner_growth_source_explicit=_argv_has_long_option(
                raw_argv,
                "--skinner-growth-source",
            ),
            skinner_accepted_occupancy_radius_explicit=(
                _argv_has_long_option(raw_argv, "--skinner-accepted-occupancy-radius")
            ),
            include_thinning_diagnostic=include_thinning_diagnostic,
            include_scanner_downstream_diagnostics=args.scanner_downstream_diagnostics,
            include_scanner_boundary_stage_diagnostics=(args.scanner_boundary_stage_diagnostics),
            thinning_diagnostic_cases=args.thinning_diagnostic_cases,
            pretty=args.pretty,
            save_volumes=args.save_volumes,
            save_figures=args.save_figures,
            write_markdown_index=args.write_markdown_index,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
