r"""Report controlled 3D synthetic truth quality metrics.

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
import csv
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from os import PathLike
from pathlib import Path, PurePosixPath
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
    prepare_case_inputs as _package_prepare_case_inputs,
    run_case as _package_run_case,
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
    SyntheticQualityCaseDefinition,
    validate_case_ids,
    validate_case_set,
)
from pyosv.evaluation.synthetic_quality.profiles import (
    WORKFLOW_MODES,
    _default_skinner_method_for_workflow,  # noqa: F401 - compatibility export
    _default_skinner_min_likelihood_for_method,  # noqa: F401 - compatibility export
    _default_surface_support_policy_for_workflow,
    _default_voter_thin_mode_for_workflow,
    _effective_include_thinning_diagnostic,
    _effective_skinner_method,
    _effective_skinner_min_likelihood,
    _effective_skinning_config_for_workflow,
    _effective_surface_support_policy,
    _effective_voter_thin_mode,
    _validate_workflow_mode,
)
from pyosv.evaluation.synthetic_quality.variants import (
    BASELINE_VARIANT,
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
VOLUME_NAMES = (
    "truth_fault_mask",
    "truth_distance",
    "truth_strike",
    "truth_dip",
    "ft_oracle",
    "pt_oracle",
    "tt_oracle",
    "fv_py",
    "vp_py",
    "vt_py",
    "fvt_py",
    "skin_mask_py",
)
FIGURE_VOLUME_NAMES = ("ft_oracle", "fv_py", "fvt_py")
SCANNER_VOLUME_NAMES = (
    ("scanner_input", "scanner_input"),
    ("scanner_ft", "ft_scan"),
    ("scanner_pt", "pt_scan"),
    ("scanner_tt", "tt_scan"),
    ("scanner_fet", "ft_used"),
    ("scanner_fpt", "pt_used"),
    ("scanner_ftt", "tt_used"),
    ("scanner_confidence", "scanner_confidence"),
)
SCANNER_FIGURE_VOLUME_NAMES = (
    ("scanner_input", "scanner_input"),
    ("scanner_ft", "ft_scan"),
    ("scanner_fet", "ft_used"),
)
THINNING_DIAGNOSTIC_VOLUME_NAMES = (
    ("fvt_reference_thinning_diagnostic", "fvt_reference"),
    ("fvt_normal_thinning_diagnostic", "fvt_normal"),
    ("keep_reference_thinning_diagnostic", "keep_reference"),
    ("keep_normal_thinning_diagnostic", "keep_normal"),
    ("keep_both_thinning_diagnostic", "keep_both"),
    ("keep_reference_only_thinning_diagnostic", "keep_reference_only"),
    ("keep_normal_only_thinning_diagnostic", "keep_normal_only"),
)
PIPELINE_OUTPUTS_KEY = "__pipelines__"
PIPELINE_NAMES = ("oracle", "scanner")
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


CSV_VARIANT_COMPARISON_FIELDS = (
    (
        "fvt_buffered_f1_delta_vs_baseline",
        "fvt_buffered_f1_r2_delta_vs_current",
    ),
    (
        "fvt_distance_p95_delta_vs_baseline",
        "fvt_candidate_to_truth_p95_delta_vs_current",
    ),
    (
        "fvt_strike_median_error_delta_vs_baseline",
        "fvt_strike_median_error_delta_vs_current",
    ),
    (
        "fvt_dip_median_error_delta_vs_baseline",
        "fvt_dip_median_error_delta_vs_current",
    ),
    (
        "skin_buffered_f1_delta_vs_baseline",
        "skin_buffered_f1_r2_delta_vs_current",
    ),
    (
        "skin_distance_p95_delta_vs_baseline",
        "skin_candidate_to_truth_p95_delta_vs_current",
    ),
    (
        "skin_strike_median_error_delta_vs_baseline",
        "skin_strike_median_error_delta_vs_current",
    ),
    (
        "skin_dip_median_error_delta_vs_baseline",
        "skin_dip_median_error_delta_vs_current",
    ),
    (
        "skin_count_delta_vs_baseline",
        "skin_count_delta_vs_current",
    ),
)


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
    evaluation = _package_run_case(case_definition, **kwargs)
    return dict(evaluation.report_payload), dict(evaluation.artifacts.volumes)


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
    report, _, _ = _build_report_and_volumes(
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
        skinner_accepted_occupancy_radius_explicit=skinner_accepted_occupancy_radius_explicit,
        include_thinning_diagnostic=include_thinning_diagnostic,
        include_scanner_downstream_diagnostics=include_scanner_downstream_diagnostics,
        thinning_diagnostic_cases=thinning_diagnostic_cases,
    )
    return report


def _build_report_and_volumes(
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
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, dict[str, Any]]],
]:
    valid_shape = validate_shape3(shape)
    valid_variants = _validate_variants(variants)
    valid_variant_preset = _validate_variant_preset(variant_preset)
    valid_input_mode = _validate_input_mode(input_mode)
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
        skinner_accepted_occupancy_radius_explicit=skinner_accepted_occupancy_radius_explicit,
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
    diagnostic_case_ids = set(_validate_thinning_diagnostic_cases(thinning_diagnostic_cases))
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
        prepared_inputs = _package_prepare_case_inputs(
            case,
            scanner_config=scanner_config,
            input_mode=valid_input_mode,
            scanner_backend_matrix=effective_scanner_backend_matrix,
        )
        for variant in valid_variants:
            variant_report, volumes, skins_output = _run_case_variant(
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
            )
            variant_reports[variant] = variant_report
            variant_volumes[variant] = volumes
            variant_skins[variant] = skins_output
        pipelines = _case_pipeline_reports(variant_reports, valid_input_mode)
        case_report = {
            "case_id": case.case_id,
            "shape": [int(size) for size in case.shape],
            "truth": _truth_report(case, truth_metric_config),
            "variants": variant_reports,
            "pipelines": pipelines,
            "variant_comparison": _case_variant_comparison_alias(pipelines, valid_input_mode),
        }
        if BASELINE_VARIANT in variant_reports:
            case_report.update(
                {
                    key: value
                    for key, value in variant_reports[BASELINE_VARIANT].items()
                    if key not in {"config", "pipelines"}
                }
            )
            case_report["pipelines"] = pipelines
            case_report["variant_comparison"] = _case_variant_comparison_alias(
                pipelines,
                valid_input_mode,
            )
        cases.append(case_report)
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

    report = {
        "format_version": FORMAT_VERSION,
        "config": config,
        "cases": cases,
    }
    return report, volume_outputs, skin_outputs


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
    indent = 2 if pretty else None
    return json.dumps(report, indent=indent, sort_keys=True) + "\n"


def write_metrics_json(
    report: Mapping[str, Any],
    output_dir: str | PathLike[str],
    *,
    pretty: bool = False,
) -> Path:
    output_path = Path(output_dir) / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_to_json(report, pretty=pretty), encoding="utf-8")
    return output_path


def write_summary_csv(report: Mapping[str, Any], output_dir: str | PathLike[str]) -> Path:
    output_path = Path(output_dir) / "summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "case_id",
                "pipeline",
                "variant",
                "baseline_variant",
                "input_mode",
                "workflow_mode",
                "scanner_backend",
                "scanner_refinement_factor",
                "scanner_ensemble_reference_like_fraction",
                "scanner_ensemble_quality_fraction",
                "scanner_ensemble_fast_fraction",
                "scanner_thin_mode",
                "shape_n3",
                "shape_n2",
                "shape_n1",
                "surface_voting_boundary_policy",
                "voter_boundary_affected_seed_count",
                "voter_skipped_seed_count",
                "voter_support_fraction_mean",
                "voter_support_fraction_min",
                "voter_surface_projection_count",
                "voter_selected_invalid_sample_count",
                "voter_face_center_vote_count",
                "fv_max",
                "fv_mean",
                "fv_nonzero_fraction",
                "fv_buffered_f1_r2",
                "fv_distance_p95",
                "fv_edge_false_positive_fraction",
                "fv_positive_candidate_count",
                "fv_strike_median_error",
                "fv_dip_median_error",
                "fvt_max",
                "fvt_mean",
                "fvt_nonzero_fraction",
                "fvt_buffered_f1_r2",
                "fvt_distance_p95",
                "fvt_edge_false_positive_fraction",
                "fvt_positive_candidate_count",
                "fvt_positive_buffered_f1_r2",
                "fvt_positive_distance_p95",
                "fvt_positive_edge_false_positive_fraction",
                "fvt_strike_median_error",
                "fvt_dip_median_error",
                "skinning_enabled",
                "skin_enabled",
                "skin_count",
                "skin_cell_count",
                "skin_unique_cell_count",
                "skin_duplicate_cell_count",
                "skin_largest_size",
                "skin_largest_fraction",
                "skin_small_count",
                "skin_small_cell_fraction",
                "skin_seed_candidate_count_before_spacing",
                "skin_seed_count_after_spacing",
                "skin_seed_rejected_by_occupied",
                "skin_grow_attempt_count",
                "skin_discarded_empty_count",
                "skin_discarded_small_count",
                "skin_accepted_count",
                "skin_fallback_enabled",
                "skin_fallback_policy",
                "skin_fallback_used",
                "skin_fallback_reason",
                "skin_fallback_method",
                "skin_fallback_input",
                "skin_fallback_skin_count",
                "skin_fallback_cell_count",
                "skin_fallback_triggered_by_degraded_primary",
                "skin_fallback_degraded_reasons",
                "skin_fallback_replaced_primary",
                "skin_fallback_primary_skin_count",
                "skin_fallback_primary_cell_count",
                "skin_fallback_candidate_count",
                "skin_fallback_component_count",
                "skin_fallback_candidate_cell_count",
                "skin_fallback_largest_component_size",
                "skin_fallback_largest_component_fraction",
                "skin_fallback_top3_component_cell_count",
                "skin_fallback_top3_component_fraction",
                "skin_fallback_small_component_count",
                "skin_fallback_component_policy",
                "skin_fallback_accepted_component_count",
                "skin_fallback_discarded_component_count",
                "skin_fallback_accepted_component_cell_count",
                "skin_fallback_filter_min_component_size",
                "skin_fallback_filter_min_component_fraction_of_largest",
                "skin_fallback_filter_max_components",
                "skin_fallback_pruning_method",
                "skin_fallback_raw_component_cell_count",
                "skin_fallback_pruned_component_cell_count",
                "skin_fallback_pruned_fraction",
                "skin_fallback_largest_component_size_before_pruning",
                "skin_fallback_largest_component_size_after_pruning",
                "skin_fallback_pruning_removed_cell_count",
                "skin_fallback_skeletonization_axis_mode",
                "skin_fallback_coverage_before",
                "skin_fallback_coverage_after",
                "skin_fallback_v5_guardrail_enabled",
                "skin_fallback_v5_guardrail_passed",
                "skin_fallback_v5_guardrail_reasons",
                "skin_fallback_v5_guardrail_fallback_skin_count",
                "skin_fallback_v5_guardrail_coverage_of_fvt_positive",
                "skin_fallback_v5_guardrail_largest_skin_fraction",
                "skin_fallback_v5_guardrail_small_skin_cell_fraction",
                "skin_fallback_v5_guardrail_pruned_fraction",
                "skin_primary_count",
                "skin_primary_cell_count",
                "skin_primary_unique_cell_count",
                "skin_primary_largest_size",
                "skin_primary_largest_fraction",
                "skin_primary_small_count",
                "skin_primary_small_cell_fraction",
                "skin_primary_cell_coverage_of_fvt_positive",
                "skin_primary_largest_coverage_of_fvt_positive",
                "skin_primary_edge_shell_fraction",
                "skin_fvt_positive_edge_shell_fraction",
                "skin_scanner_target_positive_edge_shell_fraction",
                "skin_fvt_to_scanner_target_distance_p95",
                "skin_primary_degraded_candidate",
                "skin_primary_degraded_reasons",
                "skin_primary_boundary_degraded_candidate",
                "skin_primary_boundary_degraded_reasons",
                "skin_buffered_f1_r2",
                "skin_buffered_precision_r2",
                "skin_buffered_recall_r2",
                "skin_distance_p95",
                "skin_distance_candidate_to_truth_p95",
                "skin_distance_truth_to_candidate_p95",
                "skin_distance_hausdorff_p95",
                "skin_strike_median_error",
                "skin_dip_median_error",
                "skin_truth_component_count",
                "skin_covered_truth_component_count",
                "skin_uncovered_truth_component_count",
                "skin_over_merge_count",
                "skin_over_split_count",
                "skin_max_truth_components_per_skin",
                "skin_max_skins_per_truth_component",
                "skin_mean_purity",
                "skin_min_purity",
                "skin_mean_truth_component_recall",
                "skin_min_truth_component_recall",
                "scanner_ft_buffered_f1_r2",
                "scanner_ft_distance_p95",
                "scanner_strike_median_error",
                "scanner_dip_median_error",
                "scanner_input_contrast",
                "scanner_matrix_best_fvt_positive_buffered_f1_backend",
                "scanner_matrix_best_skin_buffered_f1_backend",
                "scanner_matrix_best_boundary_edge_fp_backend",
                "scanner_downstream_ft_positive_candidate_count",
                "scanner_downstream_fet_positive_candidate_count",
                "scanner_downstream_ft_to_fet_retention_fraction",
                "scanner_downstream_fv_positive_candidate_count",
                "scanner_downstream_fvt_positive_candidate_count",
                "scanner_downstream_fvt_to_fv_positive_fraction",
                "scanner_downstream_fvt_positive_edge_candidate_fraction",
                "scanner_downstream_fvt_positive_edge_false_positive_fraction",
                "scanner_downstream_scanner_ft_positive_count",
                "scanner_downstream_scanner_fet_positive_count",
                "scanner_downstream_fv_positive_count",
                "scanner_downstream_fvt_positive_count",
                "scanner_downstream_ft_to_fvt_overlap_f1",
                "scanner_downstream_fvt_to_ft_distance_p95",
                "scanner_downstream_fvt_edge_shell_fraction",
                "scanner_downstream_fv_to_fvt_positive_ratio",
                "scanner_downstream_voter_thin_mode",
                "scanner_downstream_plateau_tie_breaker_source",
                "scanner_downstream_scanner_thin_mode",
                "scanner_downstream_reference_fvt_positive_buffered_f1_r2",
                "scanner_downstream_hybrid_fvt_positive_buffered_f1_r2",
                "scanner_downstream_hybrid_v2_fvt_positive_buffered_f1_r2",
                "scanner_downstream_normal_plateau_fvt_positive_buffered_f1_r2",
                "scanner_stage_ft_positive_count",
                "scanner_stage_fet_positive_count",
                "scanner_stage_seed_candidate_count",
                "scanner_stage_seed_selected_count",
                "scanner_stage_fv_positive_count",
                "scanner_stage_fvt_positive_count",
                "scanner_stage_skin_count",
                "scanner_stage_ft_truth_f1_r2",
                "scanner_stage_fet_truth_f1_r2",
                "scanner_stage_seed_selected_truth_f1_r2",
                "scanner_stage_fv_truth_f1_r2",
                "scanner_stage_fvt_truth_f1_r2",
                "scanner_stage_skin_truth_f1_r2",
                "scanner_stage_ft_to_fet_ratio",
                "scanner_stage_fet_to_seed_selected_f1_r2",
                "scanner_stage_seed_selected_to_fv_f1_r2",
                "scanner_stage_fv_to_fvt_ratio",
                "scanner_stage_fvt_to_skin_f1_r2",
                "scanner_stage_fvt_to_skin_distance_p95",
                "fvt_recenter_enabled",
                "fvt_recenter_target_source",
                "fvt_recenter_candidate_count",
                "fvt_recenter_moved_count",
                "fvt_recenter_collision_count",
                "fvt_recenter_mean_shift",
                "fvt_recenter_p95_shift",
                "fvt_recenter_max_shift",
                "fvt_recenter_edge_shell_only",
                "fvt_recenter_positive_count_before",
                "fvt_recenter_positive_count_after",
                "fvt_recenter_to_target_distance_p95_before",
                "fvt_recenter_to_target_distance_p95_after",
                "boundary_edge_thin_enabled",
                "boundary_edge_thin_target_source",
                "boundary_edge_thin_adopted_candidate_count",
                "boundary_edge_thin_replaced_candidate_count",
                "boundary_edge_thin_positive_count_before",
                "boundary_edge_thin_positive_count_after",
                "boundary_edge_thin_to_target_distance_p95_before",
                "boundary_edge_thin_to_target_distance_p95_after",
                "boundary_seed_retention_enabled",
                "boundary_seed_retention_target_source",
                "boundary_seed_retention_default_seed_count",
                "boundary_seed_retention_boundary_candidate_count",
                "boundary_seed_retention_added_seed_count",
                "boundary_seed_retention_total_seed_count",
                "thinning_diag_reference_fvt_buffered_f1_r2",
                "thinning_diag_normal_fvt_buffered_f1_r2",
                "thinning_diag_normal_minus_reference_fvt_buffered_f1_r2",
                "thinning_diag_reference_fvt_distance_p95",
                "thinning_diag_normal_fvt_distance_p95",
                "thinning_diag_normal_minus_reference_fvt_distance_p95",
                "thinning_diag_reference_count",
                "thinning_diag_normal_count",
                "thinning_diag_intersection_count",
                "thinning_diag_reference_only_count",
                "thinning_diag_normal_only_count",
                "thinning_diag_jaccard",
                "fvt_buffered_f1_delta_vs_baseline",
                "fvt_distance_p95_delta_vs_baseline",
                "fvt_strike_median_error_delta_vs_baseline",
                "fvt_dip_median_error_delta_vs_baseline",
                "skin_buffered_f1_delta_vs_baseline",
                "skin_distance_p95_delta_vs_baseline",
                "skin_strike_median_error_delta_vs_baseline",
                "skin_dip_median_error_delta_vs_baseline",
                "skin_count_delta_vs_baseline",
            ),
        )
        writer.writeheader()
        config = report.get("config", {})
        input_mode = str(config.get("input_mode", "oracle"))
        workflow_mode = str(config.get("workflow_mode", "reference"))
        for case in report["cases"]:
            n3, n2, n1 = case["shape"]
            for pipeline, pipeline_report in _iter_pipeline_reports(case["pipelines"]):
                variant_comparison = pipeline_report["variant_comparison"]
                baseline_variant = variant_comparison["baseline_variant"]
                comparison_variants = variant_comparison["variants"]
                for variant, variant_report in pipeline_report["variants"].items():
                    pyosv = variant_report["pyosv"]
                    fv = pyosv["fv"]
                    fvt = pyosv["fvt"]
                    quality = variant_report["quality"]
                    fv_quality = quality["fv_top_truth_count"]
                    fvt_quality = quality["fvt_top_truth_count"]
                    fv_positive_quality = quality["fv_positive_top_truth_count"]
                    fvt_positive_quality = quality["fvt_positive_top_truth_count"]
                    edge_false_positive = quality["edge_false_positive"]
                    skinning = variant_report["skinning"]
                    skin_quality = quality["skin"]
                    skin_summary = _summary_csv_skin_row(
                        enabled=bool(skinning["enabled"]),
                        quality=skin_quality,
                        diagnostics=skinning.get("diagnostics"),
                    )
                    comparison_row = _summary_csv_comparison_row(
                        comparison_variants.get(variant, {}),
                    )
                    scanner_row = _summary_csv_scanner_row(
                        variant_report=variant_report,
                        input_mode=input_mode,
                    )
                    scanner_matrix_row = _summary_csv_scanner_backend_matrix_row(
                        variant_report=variant_report,
                    )
                    scanner_downstream_row = _summary_csv_scanner_downstream_row(
                        variant_report=variant_report,
                    )
                    scanner_stage_loss_row = _summary_csv_scanner_stage_loss_row(
                        variant_report=variant_report,
                    )
                    fvt_recenter_row = _summary_csv_fvt_recenter_row(
                        variant_report=variant_report,
                    )
                    boundary_edge_thin_row = _summary_csv_boundary_edge_thin_row(
                        variant_report=variant_report,
                    )
                    boundary_seed_retention_row = _summary_csv_boundary_seed_retention_row(
                        variant_report=variant_report,
                    )
                    thinning_diagnostic_row = _summary_csv_thinning_diagnostic_row(
                        variant_report.get("thinning_diagnostic"),
                    )
                    voting_row = _summary_csv_voting_row(variant_report=variant_report)
                    writer.writerow(
                        {
                            "case_id": case["case_id"],
                            "pipeline": pipeline,
                            "variant": variant,
                            "baseline_variant": baseline_variant,
                            "input_mode": input_mode,
                            "workflow_mode": workflow_mode,
                            "shape_n3": n3,
                            "shape_n2": n2,
                            "shape_n1": n1,
                            **voting_row,
                            "fv_max": fv["max"],
                            "fv_mean": fv["mean"],
                            "fv_nonzero_fraction": fv["nonzero_fraction"],
                            "fv_buffered_f1_r2": fv_quality["buffered_overlap_radius2"][
                                "buffered_f1"
                            ],
                            "fv_distance_p95": fv_quality["surface_distance"][
                                "candidate_to_truth_p95"
                            ],
                            "fv_edge_false_positive_fraction": edge_false_positive[
                                "fv_top_truth_count"
                            ]["edge_false_positive_fraction_of_candidates"],
                            "fv_positive_candidate_count": fv_positive_quality[
                                "buffered_overlap_radius2"
                            ]["candidate_count"],
                            "fv_strike_median_error": fv_quality["orientation_error"][
                                "strike_median"
                            ],
                            "fv_dip_median_error": fv_quality["orientation_error"]["dip_median"],
                            "fvt_max": fvt["max"],
                            "fvt_mean": fvt["mean"],
                            "fvt_nonzero_fraction": fvt["nonzero_fraction"],
                            "fvt_buffered_f1_r2": fvt_quality["buffered_overlap_radius2"][
                                "buffered_f1"
                            ],
                            "fvt_distance_p95": fvt_quality["surface_distance"][
                                "candidate_to_truth_p95"
                            ],
                            "fvt_edge_false_positive_fraction": edge_false_positive[
                                "fvt_top_truth_count"
                            ]["edge_false_positive_fraction_of_candidates"],
                            "fvt_positive_candidate_count": fvt_positive_quality[
                                "buffered_overlap_radius2"
                            ]["candidate_count"],
                            "fvt_positive_buffered_f1_r2": fvt_positive_quality[
                                "buffered_overlap_radius2"
                            ]["buffered_f1"],
                            "fvt_positive_distance_p95": fvt_positive_quality["surface_distance"][
                                "candidate_to_truth_p95"
                            ],
                            "fvt_positive_edge_false_positive_fraction": edge_false_positive[
                                "fvt_positive_top_truth_count"
                            ]["edge_false_positive_fraction_of_candidates"],
                            "fvt_strike_median_error": fvt_quality["orientation_error"][
                                "strike_median"
                            ],
                            "fvt_dip_median_error": fvt_quality["orientation_error"]["dip_median"],
                            **skin_summary,
                            **scanner_row,
                            **scanner_matrix_row,
                            **scanner_downstream_row,
                            **scanner_stage_loss_row,
                            **fvt_recenter_row,
                            **boundary_edge_thin_row,
                            **boundary_seed_retention_row,
                            **thinning_diagnostic_row,
                            **comparison_row,
                        }
                    )
    return output_path


def _summary_csv_voting_row(*, variant_report: Mapping[str, Any]) -> dict[str, str | float | int]:
    """Return stable scalar voter diagnostics, including for pre-diagnostic reports."""

    pyosv = variant_report.get("pyosv")
    voting = pyosv.get("voting") if isinstance(pyosv, Mapping) else None
    if not isinstance(voting, Mapping):
        voting = {}
    diagnostic = voting.get("diagnostic_summary")
    if not isinstance(diagnostic, Mapping):
        diagnostic = {}

    projection_count = diagnostic.get(
        "surface_projection_count",
        diagnostic.get("projection_count", 0),
    )
    return {
        "surface_voting_boundary_policy": str(
            voting.get("surface_voting_boundary_policy", "reference")
        ),
        "voter_boundary_affected_seed_count": int(
            diagnostic.get("boundary_affected_seed_count", 0)
        ),
        "voter_skipped_seed_count": int(diagnostic.get("skipped_seed_count", 0)),
        "voter_support_fraction_mean": float(diagnostic.get("support_fraction_mean", 1.0)),
        "voter_support_fraction_min": float(diagnostic.get("support_fraction_min", 1.0)),
        "voter_surface_projection_count": int(projection_count),
        "voter_selected_invalid_sample_count": int(
            diagnostic.get("selected_invalid_sample_count", 0)
        ),
        "voter_face_center_vote_count": int(diagnostic.get("face_center_vote_count", 0)),
    }


def _summary_csv_scanner_row(
    *,
    variant_report: Mapping[str, Any],
    input_mode: str,
) -> dict[str, str | float | None]:
    empty_row: dict[str, str | float | None] = {
        "scanner_backend": None,
        "scanner_refinement_factor": None,
        "scanner_ensemble_reference_like_fraction": None,
        "scanner_ensemble_quality_fraction": None,
        "scanner_ensemble_fast_fraction": None,
        "scanner_thin_mode": None,
        "scanner_ft_buffered_f1_r2": None,
        "scanner_ft_distance_p95": None,
        "scanner_strike_median_error": None,
        "scanner_dip_median_error": None,
        "scanner_input_contrast": None,
    }
    if input_mode == "oracle":
        return empty_row

    scanner_report = variant_report
    if "pipelines" in variant_report:
        scanner_report = variant_report["pipelines"].get("scanner", {})
    scanner = scanner_report.get("scanner")
    scanner_quality = scanner_report.get("scanner_quality")
    if not scanner or not scanner_quality:
        return empty_row

    ft_quality = scanner_quality["ft_top_truth_count"]
    orientation_error = scanner_quality["orientation_error"]["raw_scan_top_truth_count"]
    input_association = scanner_quality["input_association"]
    selection_fraction = scanner.get("selection_fraction_by_backend")
    if not isinstance(selection_fraction, Mapping):
        selection_fraction = {}
    return {
        "scanner_backend": scanner["config"]["backend"],
        "scanner_refinement_factor": scanner["config"]["refinement_factor"],
        "scanner_ensemble_reference_like_fraction": selection_fraction.get("reference-like"),
        "scanner_ensemble_quality_fraction": selection_fraction.get("quality"),
        "scanner_ensemble_fast_fraction": selection_fraction.get("fast"),
        "scanner_thin_mode": scanner["config"]["scanner_thin_mode"],
        "scanner_ft_buffered_f1_r2": ft_quality["buffered_overlap_radius2"]["buffered_f1"],
        "scanner_ft_distance_p95": ft_quality["surface_distance"]["candidate_to_truth_p95"],
        "scanner_strike_median_error": orientation_error["strike_median"],
        "scanner_dip_median_error": orientation_error["dip_median"],
        "scanner_input_contrast": input_association["contrast"],
    }


def _summary_csv_scanner_backend_matrix_row(
    *,
    variant_report: Mapping[str, Any],
) -> dict[str, str | None]:
    empty_row = {
        "scanner_matrix_best_fvt_positive_buffered_f1_backend": None,
        "scanner_matrix_best_skin_buffered_f1_backend": None,
        "scanner_matrix_best_boundary_edge_fp_backend": None,
    }
    matrix = variant_report.get("scanner_backend_matrix")
    if not isinstance(matrix, Mapping):
        return empty_row
    comparison = matrix.get("comparison")
    if not isinstance(comparison, Mapping):
        return empty_row
    return {
        "scanner_matrix_best_fvt_positive_buffered_f1_backend": comparison.get(
            "best_fvt_positive_buffered_f1_backend"
        ),
        "scanner_matrix_best_skin_buffered_f1_backend": comparison.get(
            "best_skin_buffered_f1_backend"
        ),
        "scanner_matrix_best_boundary_edge_fp_backend": comparison.get(
            "best_boundary_edge_fp_backend"
        ),
    }


def _summary_csv_scanner_downstream_row(
    *,
    variant_report: Mapping[str, Any],
) -> dict[str, str | int | float | None]:
    empty_row: dict[str, str | int | float | None] = {
        "scanner_downstream_ft_positive_candidate_count": None,
        "scanner_downstream_fet_positive_candidate_count": None,
        "scanner_downstream_ft_to_fet_retention_fraction": None,
        "scanner_downstream_fv_positive_candidate_count": None,
        "scanner_downstream_fvt_positive_candidate_count": None,
        "scanner_downstream_fvt_to_fv_positive_fraction": None,
        "scanner_downstream_fvt_positive_edge_candidate_fraction": None,
        "scanner_downstream_fvt_positive_edge_false_positive_fraction": None,
        "scanner_downstream_scanner_ft_positive_count": None,
        "scanner_downstream_scanner_fet_positive_count": None,
        "scanner_downstream_fv_positive_count": None,
        "scanner_downstream_fvt_positive_count": None,
        "scanner_downstream_ft_to_fvt_overlap_f1": None,
        "scanner_downstream_fvt_to_ft_distance_p95": None,
        "scanner_downstream_fvt_edge_shell_fraction": None,
        "scanner_downstream_fv_to_fvt_positive_ratio": None,
        "scanner_downstream_voter_thin_mode": None,
        "scanner_downstream_plateau_tie_breaker_source": None,
        "scanner_downstream_scanner_thin_mode": None,
        "scanner_downstream_reference_fvt_positive_buffered_f1_r2": None,
        "scanner_downstream_hybrid_fvt_positive_buffered_f1_r2": None,
        "scanner_downstream_hybrid_v2_fvt_positive_buffered_f1_r2": None,
        "scanner_downstream_normal_plateau_fvt_positive_buffered_f1_r2": None,
    }
    diagnostic = variant_report.get("scanner_downstream")
    if not isinstance(diagnostic, Mapping):
        return empty_row
    thinning_modes = diagnostic.get("thinning_modes")
    if not isinstance(thinning_modes, Mapping):
        thinning_modes = {}
    ft_to_fvt_overlap = diagnostic.get("scanner_ft_vs_fvt_positive_buffered_overlap_radius2")
    if not isinstance(ft_to_fvt_overlap, Mapping):
        ft_to_fvt_overlap = {}

    def thinning_f1(mode: str) -> float | None:
        mode_report = thinning_modes.get(mode)
        if not isinstance(mode_report, Mapping):
            return None
        value = mode_report.get("fvt_positive_buffered_f1_r2")
        return None if value is None else float(value)

    return {
        "scanner_downstream_ft_positive_candidate_count": diagnostic.get(
            "scanner_ft_positive_candidate_count"
        ),
        "scanner_downstream_fet_positive_candidate_count": diagnostic.get(
            "scanner_fet_positive_candidate_count"
        ),
        "scanner_downstream_ft_to_fet_retention_fraction": diagnostic.get(
            "scanner_ft_to_fet_retention_fraction"
        ),
        "scanner_downstream_fv_positive_candidate_count": diagnostic.get(
            "fv_positive_candidate_count"
        ),
        "scanner_downstream_fvt_positive_candidate_count": diagnostic.get(
            "fvt_positive_candidate_count"
        ),
        "scanner_downstream_fvt_to_fv_positive_fraction": diagnostic.get(
            "fvt_to_fv_positive_fraction"
        ),
        "scanner_downstream_fvt_positive_edge_candidate_fraction": diagnostic.get(
            "fvt_positive_edge_candidate_fraction"
        ),
        "scanner_downstream_fvt_positive_edge_false_positive_fraction": diagnostic.get(
            "fvt_positive_edge_false_positive_fraction"
        ),
        "scanner_downstream_scanner_ft_positive_count": diagnostic.get(
            "scanner_ft_positive_candidate_count"
        ),
        "scanner_downstream_scanner_fet_positive_count": diagnostic.get(
            "scanner_fet_positive_candidate_count"
        ),
        "scanner_downstream_fv_positive_count": diagnostic.get("fv_positive_candidate_count"),
        "scanner_downstream_fvt_positive_count": diagnostic.get("fvt_positive_candidate_count"),
        "scanner_downstream_ft_to_fvt_overlap_f1": ft_to_fvt_overlap.get("buffered_f1"),
        "scanner_downstream_fvt_to_ft_distance_p95": diagnostic.get(
            "fvt_candidate_to_scanner_ft_distance_p95"
        ),
        "scanner_downstream_fvt_edge_shell_fraction": diagnostic.get(
            "fvt_positive_edge_shell_fraction"
        ),
        "scanner_downstream_fv_to_fvt_positive_ratio": diagnostic.get(
            "fv_to_fvt_positive_candidate_count_ratio"
        ),
        "scanner_downstream_voter_thin_mode": diagnostic.get("voter_thin_mode"),
        "scanner_downstream_plateau_tie_breaker_source": diagnostic.get(
            "plateau_tie_breaker_source"
        ),
        "scanner_downstream_scanner_thin_mode": diagnostic.get("scanner_thin_mode"),
        "scanner_downstream_reference_fvt_positive_buffered_f1_r2": thinning_f1("reference"),
        "scanner_downstream_hybrid_fvt_positive_buffered_f1_r2": thinning_f1("hybrid"),
        "scanner_downstream_hybrid_v2_fvt_positive_buffered_f1_r2": thinning_f1("hybrid_v2"),
        "scanner_downstream_normal_plateau_fvt_positive_buffered_f1_r2": thinning_f1(
            "normal_plateau"
        ),
    }


def _summary_csv_scanner_stage_loss_row(
    *,
    variant_report: Mapping[str, Any],
) -> dict[str, int | float | None]:
    empty_row: dict[str, int | float | None] = {
        "scanner_stage_ft_positive_count": None,
        "scanner_stage_fet_positive_count": None,
        "scanner_stage_seed_candidate_count": None,
        "scanner_stage_seed_selected_count": None,
        "scanner_stage_fv_positive_count": None,
        "scanner_stage_fvt_positive_count": None,
        "scanner_stage_skin_count": None,
        "scanner_stage_ft_truth_f1_r2": None,
        "scanner_stage_fet_truth_f1_r2": None,
        "scanner_stage_seed_selected_truth_f1_r2": None,
        "scanner_stage_fv_truth_f1_r2": None,
        "scanner_stage_fvt_truth_f1_r2": None,
        "scanner_stage_skin_truth_f1_r2": None,
        "scanner_stage_ft_to_fet_ratio": None,
        "scanner_stage_fet_to_seed_selected_f1_r2": None,
        "scanner_stage_seed_selected_to_fv_f1_r2": None,
        "scanner_stage_fv_to_fvt_ratio": None,
        "scanner_stage_fvt_to_skin_f1_r2": None,
        "scanner_stage_fvt_to_skin_distance_p95": None,
    }
    diagnostic = variant_report.get("scanner_stage_loss")
    if not isinstance(diagnostic, Mapping):
        return empty_row
    stages = diagnostic.get("stages")
    transitions = diagnostic.get("transitions")
    if not isinstance(stages, Mapping) or not isinstance(transitions, Mapping):
        return empty_row

    def stage_value(stage: str, key: str) -> int | float | None:
        stage_report = stages.get(stage)
        if not isinstance(stage_report, Mapping):
            return None
        value = stage_report.get(key)
        return None if value is None else value

    def transition_value(transition: str, key: str) -> int | float | None:
        transition_report = transitions.get(transition)
        if not isinstance(transition_report, Mapping):
            return None
        value = transition_report.get(key)
        return None if value is None else value

    return {
        "scanner_stage_ft_positive_count": stage_value("scanner_ft_positive", "candidate_count"),
        "scanner_stage_fet_positive_count": stage_value("scanner_fet_positive", "candidate_count"),
        "scanner_stage_seed_candidate_count": stage_value("seed_candidate", "candidate_count"),
        "scanner_stage_seed_selected_count": stage_value("seed_selected", "candidate_count"),
        "scanner_stage_fv_positive_count": stage_value("fv_positive", "candidate_count"),
        "scanner_stage_fvt_positive_count": stage_value("fvt_positive", "candidate_count"),
        "scanner_stage_skin_count": stage_value("skin", "candidate_count"),
        "scanner_stage_ft_truth_f1_r2": stage_value("scanner_ft_positive", "truth_buffered_f1_r2"),
        "scanner_stage_fet_truth_f1_r2": stage_value(
            "scanner_fet_positive", "truth_buffered_f1_r2"
        ),
        "scanner_stage_seed_selected_truth_f1_r2": stage_value(
            "seed_selected", "truth_buffered_f1_r2"
        ),
        "scanner_stage_fv_truth_f1_r2": stage_value("fv_positive", "truth_buffered_f1_r2"),
        "scanner_stage_fvt_truth_f1_r2": stage_value("fvt_positive", "truth_buffered_f1_r2"),
        "scanner_stage_skin_truth_f1_r2": stage_value("skin", "truth_buffered_f1_r2"),
        "scanner_stage_ft_to_fet_ratio": transition_value(
            "scanner_ft_positive_to_scanner_fet_positive",
            "target_to_source_count_ratio",
        ),
        "scanner_stage_fet_to_seed_selected_f1_r2": transition_value(
            "scanner_fet_positive_to_seed_selected",
            "buffered_f1_r2",
        ),
        "scanner_stage_seed_selected_to_fv_f1_r2": transition_value(
            "seed_selected_to_fv_positive",
            "buffered_f1_r2",
        ),
        "scanner_stage_fv_to_fvt_ratio": transition_value(
            "fv_positive_to_fvt_positive",
            "target_to_source_count_ratio",
        ),
        "scanner_stage_fvt_to_skin_f1_r2": transition_value(
            "fvt_positive_to_skin",
            "buffered_f1_r2",
        ),
        "scanner_stage_fvt_to_skin_distance_p95": transition_value(
            "fvt_positive_to_skin",
            "target_to_source_distance_p95",
        ),
    }


def _summary_csv_fvt_recenter_row(
    *,
    variant_report: Mapping[str, Any],
) -> dict[str, str | bool | int | float | None]:
    keys = (
        "fvt_recenter_enabled",
        "fvt_recenter_target_source",
        "fvt_recenter_candidate_count",
        "fvt_recenter_moved_count",
        "fvt_recenter_collision_count",
        "fvt_recenter_mean_shift",
        "fvt_recenter_p95_shift",
        "fvt_recenter_max_shift",
        "fvt_recenter_edge_shell_only",
        "fvt_recenter_positive_count_before",
        "fvt_recenter_positive_count_after",
        "fvt_recenter_to_target_distance_p95_before",
        "fvt_recenter_to_target_distance_p95_after",
    )
    diagnostic = variant_report.get("fvt_recenter")
    if not isinstance(diagnostic, Mapping):
        return {key: None for key in keys}
    return {key: diagnostic.get(key) for key in keys}


def _summary_csv_boundary_edge_thin_row(
    *,
    variant_report: Mapping[str, Any],
) -> dict[str, str | bool | int | float | None]:
    keys = {
        "boundary_edge_thin_enabled": "enabled",
        "boundary_edge_thin_target_source": "target_source",
        "boundary_edge_thin_adopted_candidate_count": "adopted_candidate_count",
        "boundary_edge_thin_replaced_candidate_count": "replaced_candidate_count",
        "boundary_edge_thin_positive_count_before": "positive_count_before",
        "boundary_edge_thin_positive_count_after": "positive_count_after",
        "boundary_edge_thin_to_target_distance_p95_before": "to_target_distance_p95_before",
        "boundary_edge_thin_to_target_distance_p95_after": "to_target_distance_p95_after",
    }
    diagnostic = variant_report.get("boundary_edge_thin")
    if not isinstance(diagnostic, Mapping):
        return {key: False if key == "boundary_edge_thin_enabled" else None for key in keys}
    return {key: diagnostic.get(source_key) for key, source_key in keys.items()}


def _summary_csv_boundary_seed_retention_row(
    *,
    variant_report: Mapping[str, Any],
) -> dict[str, str | bool | int | None]:
    keys = {
        "boundary_seed_retention_enabled": "enabled",
        "boundary_seed_retention_target_source": "target_source",
        "boundary_seed_retention_default_seed_count": "default_seed_count",
        "boundary_seed_retention_boundary_candidate_count": "boundary_candidate_count",
        "boundary_seed_retention_added_seed_count": "added_seed_count",
        "boundary_seed_retention_total_seed_count": "total_seed_count",
    }
    diagnostic = variant_report.get("boundary_seed_retention")
    if not isinstance(diagnostic, Mapping):
        return {key: False if key == "boundary_seed_retention_enabled" else None for key in keys}
    return {key: diagnostic.get(source_key) for key, source_key in keys.items()}


def _iter_pipeline_reports(
    pipelines: Mapping[str, Mapping[str, Any]],
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    unknown = sorted(set(pipelines).difference(PIPELINE_NAMES))
    if unknown:
        raise ValueError(f"unknown pipeline(s): {','.join(unknown)}")
    return tuple(
        (pipeline, pipelines[pipeline]) for pipeline in PIPELINE_NAMES if pipeline in pipelines
    )


def _summary_csv_thinning_diagnostic_row(
    diagnostic: Mapping[str, Any] | None,
) -> dict[str, float | int | None]:
    empty_row: dict[str, float | int | None] = {
        "thinning_diag_reference_fvt_buffered_f1_r2": None,
        "thinning_diag_normal_fvt_buffered_f1_r2": None,
        "thinning_diag_normal_minus_reference_fvt_buffered_f1_r2": None,
        "thinning_diag_reference_fvt_distance_p95": None,
        "thinning_diag_normal_fvt_distance_p95": None,
        "thinning_diag_normal_minus_reference_fvt_distance_p95": None,
        "thinning_diag_reference_count": None,
        "thinning_diag_normal_count": None,
        "thinning_diag_intersection_count": None,
        "thinning_diag_reference_only_count": None,
        "thinning_diag_normal_only_count": None,
        "thinning_diag_jaccard": None,
    }
    if diagnostic is None:
        return empty_row

    reference_quality = diagnostic["reference"]["quality"]["fvt_top_truth_count"]
    normal_quality = diagnostic["normal"]["quality"]["fvt_top_truth_count"]
    delta = diagnostic["delta"]["normal_minus_reference"]
    keep_mask = diagnostic["keep_mask"]
    return {
        "thinning_diag_reference_fvt_buffered_f1_r2": reference_quality["buffered_overlap_radius2"][
            "buffered_f1"
        ],
        "thinning_diag_normal_fvt_buffered_f1_r2": normal_quality["buffered_overlap_radius2"][
            "buffered_f1"
        ],
        "thinning_diag_normal_minus_reference_fvt_buffered_f1_r2": delta["fvt_buffered_f1_r2"],
        "thinning_diag_reference_fvt_distance_p95": reference_quality["surface_distance"][
            "candidate_to_truth_p95"
        ],
        "thinning_diag_normal_fvt_distance_p95": normal_quality["surface_distance"][
            "candidate_to_truth_p95"
        ],
        "thinning_diag_normal_minus_reference_fvt_distance_p95": delta[
            "fvt_candidate_to_truth_p95"
        ],
        "thinning_diag_reference_count": keep_mask["reference_count"],
        "thinning_diag_normal_count": keep_mask["normal_count"],
        "thinning_diag_intersection_count": keep_mask["intersection_count"],
        "thinning_diag_reference_only_count": keep_mask["reference_only_count"],
        "thinning_diag_normal_only_count": keep_mask["normal_only_count"],
        "thinning_diag_jaccard": keep_mask["jaccard"],
    }


def _summary_csv_skin_row(
    *,
    enabled: bool,
    quality: Mapping[str, Any] | None,
    diagnostics: Mapping[str, Any] | None,
) -> dict[str, bool | int | float | None]:
    diagnostic_row = _summary_csv_skin_diagnostics_row(enabled=enabled, diagnostics=diagnostics)
    component_topology_row = _summary_csv_skin_component_topology_row(None)
    if quality is None:
        return {
            "skinning_enabled": enabled,
            "skin_enabled": enabled,
            "skin_count": 0,
            "skin_cell_count": 0,
            "skin_unique_cell_count": 0,
            "skin_duplicate_cell_count": 0,
            "skin_largest_size": 0,
            "skin_largest_fraction": 0.0,
            "skin_small_count": 0,
            "skin_small_cell_fraction": 0.0,
            **diagnostic_row,
            "skin_buffered_f1_r2": None,
            "skin_buffered_precision_r2": None,
            "skin_buffered_recall_r2": None,
            "skin_distance_p95": None,
            "skin_distance_candidate_to_truth_p95": None,
            "skin_distance_truth_to_candidate_p95": None,
            "skin_distance_hausdorff_p95": None,
            "skin_strike_median_error": None,
            "skin_dip_median_error": None,
            **component_topology_row,
        }

    topology = quality["topology"]
    overlap = quality["buffered_overlap_radius2"]
    distance = quality["surface_distance"]
    orientation = quality["orientation_error"]
    component_topology_row = _summary_csv_skin_component_topology_row(
        quality.get("component_topology")
    )
    return {
        "skinning_enabled": enabled,
        "skin_enabled": enabled,
        "skin_count": topology["skin_count"],
        "skin_cell_count": topology["cell_count"],
        "skin_unique_cell_count": topology["unique_cell_count"],
        "skin_duplicate_cell_count": topology["duplicate_cell_count"],
        "skin_largest_size": topology["largest_skin_size"],
        "skin_largest_fraction": topology["largest_skin_fraction"],
        "skin_small_count": topology["small_skin_count"],
        "skin_small_cell_fraction": topology["small_skin_cell_fraction"],
        **diagnostic_row,
        "skin_buffered_f1_r2": overlap["buffered_f1"],
        "skin_buffered_precision_r2": overlap["buffered_precision"],
        "skin_buffered_recall_r2": overlap["buffered_recall"],
        "skin_distance_p95": distance["candidate_to_truth_p95"],
        "skin_distance_candidate_to_truth_p95": distance["candidate_to_truth_p95"],
        "skin_distance_truth_to_candidate_p95": distance["truth_to_candidate_p95"],
        "skin_distance_hausdorff_p95": distance["hausdorff_p95"],
        "skin_strike_median_error": orientation["strike_median"],
        "skin_dip_median_error": orientation["dip_median"],
        **component_topology_row,
    }


def _summary_csv_skin_component_topology_row(
    component_topology: Mapping[str, Any] | None,
) -> dict[str, int | float | None]:
    if component_topology is None:
        return {
            "skin_truth_component_count": None,
            "skin_covered_truth_component_count": None,
            "skin_uncovered_truth_component_count": None,
            "skin_over_merge_count": None,
            "skin_over_split_count": None,
            "skin_max_truth_components_per_skin": None,
            "skin_max_skins_per_truth_component": None,
            "skin_mean_purity": None,
            "skin_min_purity": None,
            "skin_mean_truth_component_recall": None,
            "skin_min_truth_component_recall": None,
        }
    return {
        "skin_truth_component_count": component_topology["truth_component_count"],
        "skin_covered_truth_component_count": component_topology["covered_truth_component_count"],
        "skin_uncovered_truth_component_count": component_topology[
            "uncovered_truth_component_count"
        ],
        "skin_over_merge_count": component_topology["over_merge_skin_count"],
        "skin_over_split_count": component_topology["over_split_truth_component_count"],
        "skin_max_truth_components_per_skin": component_topology["max_truth_components_per_skin"],
        "skin_max_skins_per_truth_component": component_topology["max_skins_per_truth_component"],
        "skin_mean_purity": component_topology["mean_skin_purity"],
        "skin_min_purity": component_topology["min_skin_purity"],
        "skin_mean_truth_component_recall": component_topology["mean_truth_component_recall"],
        "skin_min_truth_component_recall": component_topology["min_truth_component_recall"],
    }


def _empty_summary_csv_skin_fallback_v5_guardrail_row(
    *,
    enabled: bool | None,
) -> dict[str, bool | int | float | str | None]:
    return {
        "skin_fallback_v5_guardrail_enabled": enabled,
        "skin_fallback_v5_guardrail_passed": None,
        "skin_fallback_v5_guardrail_reasons": None,
        "skin_fallback_v5_guardrail_fallback_skin_count": None,
        "skin_fallback_v5_guardrail_coverage_of_fvt_positive": None,
        "skin_fallback_v5_guardrail_largest_skin_fraction": None,
        "skin_fallback_v5_guardrail_small_skin_cell_fraction": None,
        "skin_fallback_v5_guardrail_pruned_fraction": None,
    }


def _summary_csv_skin_fallback_v5_guardrail_row(
    diagnostics: Mapping[str, Any],
) -> dict[str, bool | int | float | str | None]:
    guardrail = diagnostics.get("fallback_v5_guardrail")
    if not isinstance(guardrail, Mapping):
        return _empty_summary_csv_skin_fallback_v5_guardrail_row(enabled=False)
    reasons = guardrail.get("reasons")
    if isinstance(reasons, list):
        reasons = ",".join(str(reason) for reason in reasons)
    return {
        "skin_fallback_v5_guardrail_enabled": guardrail.get("enabled"),
        "skin_fallback_v5_guardrail_passed": guardrail.get("passed"),
        "skin_fallback_v5_guardrail_reasons": reasons,
        "skin_fallback_v5_guardrail_fallback_skin_count": guardrail.get("fallback_skin_count"),
        "skin_fallback_v5_guardrail_coverage_of_fvt_positive": guardrail.get(
            "coverage_of_fvt_positive"
        ),
        "skin_fallback_v5_guardrail_largest_skin_fraction": guardrail.get("largest_skin_fraction"),
        "skin_fallback_v5_guardrail_small_skin_cell_fraction": guardrail.get(
            "small_skin_cell_fraction"
        ),
        "skin_fallback_v5_guardrail_pruned_fraction": guardrail.get("pruned_fraction"),
    }


def _summary_csv_skin_diagnostics_row(
    *,
    enabled: bool,
    diagnostics: Mapping[str, Any] | None,
) -> dict[str, bool | int | float | str | None]:
    if not enabled:
        return {
            "skin_seed_candidate_count_before_spacing": 0,
            "skin_seed_count_after_spacing": 0,
            "skin_seed_rejected_by_occupied": 0,
            "skin_grow_attempt_count": 0,
            "skin_discarded_empty_count": 0,
            "skin_discarded_small_count": 0,
            "skin_accepted_count": 0,
            "skin_fallback_enabled": False,
            "skin_fallback_policy": None,
            "skin_fallback_used": False,
            "skin_fallback_reason": None,
            "skin_fallback_method": None,
            "skin_fallback_input": None,
            "skin_fallback_skin_count": 0,
            "skin_fallback_cell_count": 0,
            "skin_fallback_triggered_by_degraded_primary": False,
            "skin_fallback_degraded_reasons": "",
            "skin_fallback_replaced_primary": False,
            "skin_fallback_primary_skin_count": 0,
            "skin_fallback_primary_cell_count": 0,
            "skin_fallback_candidate_count": 0,
            "skin_fallback_component_count": 0,
            "skin_fallback_candidate_cell_count": 0,
            "skin_fallback_largest_component_size": 0,
            "skin_fallback_largest_component_fraction": 0.0,
            "skin_fallback_top3_component_cell_count": 0,
            "skin_fallback_top3_component_fraction": 0.0,
            "skin_fallback_small_component_count": 0,
            "skin_fallback_component_policy": "all",
            "skin_fallback_accepted_component_count": 0,
            "skin_fallback_discarded_component_count": 0,
            "skin_fallback_accepted_component_cell_count": 0,
            "skin_fallback_filter_min_component_size": 0,
            "skin_fallback_filter_min_component_fraction_of_largest": 0.0,
            "skin_fallback_filter_max_components": 0,
            "skin_fallback_pruning_method": None,
            "skin_fallback_raw_component_cell_count": 0,
            "skin_fallback_pruned_component_cell_count": 0,
            "skin_fallback_pruned_fraction": 0.0,
            "skin_fallback_largest_component_size_before_pruning": 0,
            "skin_fallback_largest_component_size_after_pruning": 0,
            "skin_fallback_pruning_removed_cell_count": 0,
            "skin_fallback_skeletonization_axis_mode": None,
            "skin_fallback_coverage_before": 0.0,
            "skin_fallback_coverage_after": 0.0,
            **_empty_summary_csv_skin_fallback_v5_guardrail_row(enabled=False),
            "skin_primary_count": 0,
            "skin_primary_cell_count": 0,
            "skin_primary_unique_cell_count": 0,
            "skin_primary_largest_size": 0,
            "skin_primary_largest_fraction": 0.0,
            "skin_primary_small_count": 0,
            "skin_primary_small_cell_fraction": 0.0,
            "skin_primary_cell_coverage_of_fvt_positive": 0.0,
            "skin_primary_largest_coverage_of_fvt_positive": 0.0,
            "skin_primary_edge_shell_fraction": 0.0,
            "skin_fvt_positive_edge_shell_fraction": 0.0,
            "skin_scanner_target_positive_edge_shell_fraction": None,
            "skin_fvt_to_scanner_target_distance_p95": None,
            "skin_primary_degraded_candidate": False,
            "skin_primary_degraded_reasons": "",
            "skin_primary_boundary_degraded_candidate": False,
            "skin_primary_boundary_degraded_reasons": "",
        }
    if diagnostics is None:
        return {
            "skin_seed_candidate_count_before_spacing": None,
            "skin_seed_count_after_spacing": None,
            "skin_seed_rejected_by_occupied": None,
            "skin_grow_attempt_count": None,
            "skin_discarded_empty_count": None,
            "skin_discarded_small_count": None,
            "skin_accepted_count": None,
            "skin_fallback_enabled": None,
            "skin_fallback_policy": None,
            "skin_fallback_used": None,
            "skin_fallback_reason": None,
            "skin_fallback_method": None,
            "skin_fallback_input": None,
            "skin_fallback_skin_count": None,
            "skin_fallback_cell_count": None,
            "skin_fallback_triggered_by_degraded_primary": None,
            "skin_fallback_degraded_reasons": None,
            "skin_fallback_replaced_primary": None,
            "skin_fallback_primary_skin_count": None,
            "skin_fallback_primary_cell_count": None,
            "skin_fallback_candidate_count": None,
            "skin_fallback_component_count": None,
            "skin_fallback_candidate_cell_count": None,
            "skin_fallback_largest_component_size": None,
            "skin_fallback_largest_component_fraction": None,
            "skin_fallback_top3_component_cell_count": None,
            "skin_fallback_top3_component_fraction": None,
            "skin_fallback_small_component_count": None,
            "skin_fallback_component_policy": None,
            "skin_fallback_accepted_component_count": None,
            "skin_fallback_discarded_component_count": None,
            "skin_fallback_accepted_component_cell_count": None,
            "skin_fallback_filter_min_component_size": None,
            "skin_fallback_filter_min_component_fraction_of_largest": None,
            "skin_fallback_filter_max_components": None,
            "skin_fallback_pruning_method": None,
            "skin_fallback_raw_component_cell_count": None,
            "skin_fallback_pruned_component_cell_count": None,
            "skin_fallback_pruned_fraction": None,
            "skin_fallback_largest_component_size_before_pruning": None,
            "skin_fallback_largest_component_size_after_pruning": None,
            "skin_fallback_pruning_removed_cell_count": None,
            "skin_fallback_skeletonization_axis_mode": None,
            "skin_fallback_coverage_before": None,
            "skin_fallback_coverage_after": None,
            **_empty_summary_csv_skin_fallback_v5_guardrail_row(enabled=None),
            "skin_primary_count": None,
            "skin_primary_cell_count": None,
            "skin_primary_unique_cell_count": None,
            "skin_primary_largest_size": None,
            "skin_primary_largest_fraction": None,
            "skin_primary_small_count": None,
            "skin_primary_small_cell_fraction": None,
            "skin_primary_cell_coverage_of_fvt_positive": None,
            "skin_primary_largest_coverage_of_fvt_positive": None,
            "skin_primary_edge_shell_fraction": None,
            "skin_fvt_positive_edge_shell_fraction": None,
            "skin_scanner_target_positive_edge_shell_fraction": None,
            "skin_fvt_to_scanner_target_distance_p95": None,
            "skin_primary_degraded_candidate": None,
            "skin_primary_degraded_reasons": None,
            "skin_primary_boundary_degraded_candidate": None,
            "skin_primary_boundary_degraded_reasons": None,
        }
    degraded_reasons = diagnostics.get("skin_primary_degraded_reasons")
    if isinstance(degraded_reasons, list):
        degraded_reasons = ",".join(str(reason) for reason in degraded_reasons)
    boundary_degraded_reasons = diagnostics.get("skin_primary_boundary_degraded_reasons")
    if isinstance(boundary_degraded_reasons, list):
        boundary_degraded_reasons = ",".join(str(reason) for reason in boundary_degraded_reasons)
    fallback_degraded_reasons = diagnostics.get("fallback_degraded_reasons")
    if isinstance(fallback_degraded_reasons, list):
        fallback_degraded_reasons = ",".join(str(reason) for reason in fallback_degraded_reasons)
    return {
        "skin_seed_candidate_count_before_spacing": diagnostics.get(
            "seed_candidate_count_before_spacing"
        ),
        "skin_seed_count_after_spacing": diagnostics.get("seed_count_after_spacing"),
        "skin_seed_rejected_by_occupied": diagnostics.get("seed_count_rejected_by_occupied"),
        "skin_grow_attempt_count": diagnostics.get("grow_attempt_count"),
        "skin_discarded_empty_count": diagnostics.get("discarded_empty_skin_count"),
        "skin_discarded_small_count": diagnostics.get("discarded_small_skin_count"),
        "skin_accepted_count": diagnostics.get("accepted_skin_count"),
        "skin_fallback_enabled": diagnostics.get("fallback_enabled"),
        "skin_fallback_policy": diagnostics.get("fallback_policy"),
        "skin_fallback_used": diagnostics.get("fallback_used"),
        "skin_fallback_reason": diagnostics.get("fallback_reason"),
        "skin_fallback_method": diagnostics.get("fallback_method"),
        "skin_fallback_input": diagnostics.get("fallback_input"),
        "skin_fallback_skin_count": diagnostics.get("fallback_skin_count"),
        "skin_fallback_cell_count": diagnostics.get("fallback_cell_count"),
        "skin_fallback_triggered_by_degraded_primary": diagnostics.get(
            "fallback_triggered_by_degraded_primary"
        ),
        "skin_fallback_degraded_reasons": fallback_degraded_reasons,
        "skin_fallback_replaced_primary": diagnostics.get("fallback_replaced_primary"),
        "skin_fallback_primary_skin_count": diagnostics.get("fallback_primary_skin_count"),
        "skin_fallback_primary_cell_count": diagnostics.get("fallback_primary_cell_count"),
        "skin_fallback_candidate_count": diagnostics.get("fallback_candidate_count"),
        "skin_fallback_component_count": diagnostics.get("skin_fallback_component_count"),
        "skin_fallback_candidate_cell_count": diagnostics.get("skin_fallback_candidate_cell_count"),
        "skin_fallback_largest_component_size": diagnostics.get(
            "skin_fallback_largest_component_size"
        ),
        "skin_fallback_largest_component_fraction": diagnostics.get(
            "skin_fallback_largest_component_fraction"
        ),
        "skin_fallback_top3_component_cell_count": diagnostics.get(
            "skin_fallback_top3_component_cell_count"
        ),
        "skin_fallback_top3_component_fraction": diagnostics.get(
            "skin_fallback_top3_component_fraction"
        ),
        "skin_fallback_small_component_count": diagnostics.get(
            "skin_fallback_small_component_count"
        ),
        "skin_fallback_component_policy": diagnostics.get("skin_fallback_component_policy"),
        "skin_fallback_accepted_component_count": diagnostics.get(
            "skin_fallback_accepted_component_count"
        ),
        "skin_fallback_discarded_component_count": diagnostics.get(
            "skin_fallback_discarded_component_count"
        ),
        "skin_fallback_accepted_component_cell_count": diagnostics.get(
            "skin_fallback_accepted_component_cell_count"
        ),
        "skin_fallback_filter_min_component_size": diagnostics.get(
            "skin_fallback_filter_min_component_size"
        ),
        "skin_fallback_filter_min_component_fraction_of_largest": diagnostics.get(
            "skin_fallback_filter_min_component_fraction_of_largest"
        ),
        "skin_fallback_filter_max_components": diagnostics.get(
            "skin_fallback_filter_max_components"
        ),
        "skin_fallback_pruning_method": diagnostics.get("skin_fallback_pruning_method"),
        "skin_fallback_raw_component_cell_count": diagnostics.get(
            "skin_fallback_raw_component_cell_count"
        ),
        "skin_fallback_pruned_component_cell_count": diagnostics.get(
            "skin_fallback_pruned_component_cell_count"
        ),
        "skin_fallback_pruned_fraction": diagnostics.get("skin_fallback_pruned_fraction"),
        "skin_fallback_largest_component_size_before_pruning": diagnostics.get(
            "skin_fallback_largest_component_size_before_pruning"
        ),
        "skin_fallback_largest_component_size_after_pruning": diagnostics.get(
            "skin_fallback_largest_component_size_after_pruning"
        ),
        "skin_fallback_pruning_removed_cell_count": diagnostics.get(
            "skin_fallback_pruning_removed_cell_count"
        ),
        "skin_fallback_skeletonization_axis_mode": diagnostics.get(
            "skin_fallback_skeletonization_axis_mode"
        ),
        "skin_fallback_coverage_before": diagnostics.get("fallback_coverage_before"),
        "skin_fallback_coverage_after": diagnostics.get("fallback_coverage_after"),
        **_summary_csv_skin_fallback_v5_guardrail_row(diagnostics),
        "skin_primary_count": diagnostics.get("skin_primary_count"),
        "skin_primary_cell_count": diagnostics.get("skin_primary_cell_count"),
        "skin_primary_unique_cell_count": diagnostics.get("skin_primary_unique_cell_count"),
        "skin_primary_largest_size": diagnostics.get("skin_primary_largest_size"),
        "skin_primary_largest_fraction": diagnostics.get("skin_primary_largest_fraction"),
        "skin_primary_small_count": diagnostics.get("skin_primary_small_count"),
        "skin_primary_small_cell_fraction": diagnostics.get("skin_primary_small_cell_fraction"),
        "skin_primary_cell_coverage_of_fvt_positive": diagnostics.get(
            "skin_primary_cell_coverage_of_fvt_positive"
        ),
        "skin_primary_largest_coverage_of_fvt_positive": diagnostics.get(
            "skin_primary_largest_coverage_of_fvt_positive"
        ),
        "skin_primary_edge_shell_fraction": diagnostics.get("skin_primary_edge_shell_fraction"),
        "skin_fvt_positive_edge_shell_fraction": diagnostics.get(
            "skin_fvt_positive_edge_shell_fraction"
        ),
        "skin_scanner_target_positive_edge_shell_fraction": diagnostics.get(
            "skin_scanner_target_positive_edge_shell_fraction"
        ),
        "skin_fvt_to_scanner_target_distance_p95": diagnostics.get(
            "skin_fvt_to_scanner_target_distance_p95"
        ),
        "skin_primary_degraded_candidate": diagnostics.get("skin_primary_degraded_candidate"),
        "skin_primary_degraded_reasons": degraded_reasons,
        "skin_primary_boundary_degraded_candidate": diagnostics.get(
            "skin_primary_boundary_degraded_candidate"
        ),
        "skin_primary_boundary_degraded_reasons": boundary_degraded_reasons,
    }


def _normalize_report_skin_metric_keys(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return quality_metrics.normalize_report_skin_metric_keys(metrics)


def _summary_csv_comparison_row(comparison: Mapping[str, Any]) -> dict[str, float | None]:
    return {
        csv_field: comparison.get(json_field)
        for csv_field, json_field in CSV_VARIANT_COMPARISON_FIELDS
    }


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


def write_case_volumes(
    volume_outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    output_dir: str | PathLike[str],
) -> list[Path]:
    from pyosv.io import write_dat

    written = []
    output_root = Path(output_dir)
    for case_id, variants in volume_outputs.items():
        case_dir = _case_output_dir(output_root, case_id)
        for variant, volumes in variants.items():
            output_dir_for_variant = _variant_output_dir(case_dir, variant, len(variants) == 1)
            for pipeline, pipeline_volumes in _iter_pipeline_volume_outputs(volumes):
                volume_dir = _pipeline_output_dir(output_dir_for_variant, pipeline)
                written.extend(_write_pipeline_volumes(volume_dir, pipeline_volumes, write_dat))
    return written


def _iter_pipeline_volume_outputs(
    volumes: Mapping[str, Any],
) -> Sequence[tuple[str | None, Mapping[str, np.ndarray]]]:
    pipeline_outputs = volumes.get(PIPELINE_OUTPUTS_KEY)
    if isinstance(pipeline_outputs, Mapping):
        unknown = sorted(set(pipeline_outputs).difference(PIPELINE_NAMES))
        if unknown:
            raise ValueError(f"unknown pipeline(s): {','.join(unknown)}")
        return tuple(
            (pipeline, pipeline_outputs[pipeline])
            for pipeline in PIPELINE_NAMES
            if pipeline in pipeline_outputs
        )
    return ((None, volumes),)


def _write_pipeline_volumes(
    output_dir: Path,
    volumes: Mapping[str, np.ndarray],
    write_dat: Callable[[str | PathLike[str], np.ndarray], Path],
) -> list[Path]:
    written = []
    for name in VOLUME_NAMES:
        written.append(write_dat(output_dir / f"{name}.dat", volumes[name]))
    for source_name, output_name in SCANNER_VOLUME_NAMES:
        if source_name in volumes:
            written.append(write_dat(output_dir / f"{output_name}.dat", volumes[source_name]))
    written.extend(_write_thinning_diagnostic_volumes(output_dir, volumes, write_dat))
    return written


def _write_thinning_diagnostic_volumes(
    output_dir: Path,
    volumes: Mapping[str, np.ndarray],
    write_dat: Callable[[str | PathLike[str], np.ndarray], Path],
) -> list[Path]:
    if "fvt_reference_thinning_diagnostic" not in volumes:
        return []

    diagnostic_dir = output_dir / "thinning_diagnostic"
    return [
        write_dat(diagnostic_dir / f"{output_name}.dat", volumes[source_name])
        for source_name, output_name in THINNING_DIAGNOSTIC_VOLUME_NAMES
    ]


def write_case_skins_json(
    skin_outputs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    output_dir: str | PathLike[str],
) -> list[Path]:
    written = []
    output_root = Path(output_dir)
    for case_id, variants in skin_outputs.items():
        case_dir = _case_output_dir(output_root, case_id)
        for variant, skins_output in variants.items():
            output_dir_for_variant = _variant_output_dir(case_dir, variant, len(variants) == 1)
            for pipeline, pipeline_skins in _iter_pipeline_skin_outputs(skins_output):
                skin_dir = _pipeline_output_dir(output_dir_for_variant, pipeline)
                output_path = skin_dir / "skins.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    json.dumps(pipeline_skins, sort_keys=True) + "\n", encoding="utf-8"
                )
                written.append(output_path)
    return written


def _iter_pipeline_skin_outputs(
    skins_output: Mapping[str, Any],
) -> Sequence[tuple[str | None, Mapping[str, Any]]]:
    pipeline_outputs = skins_output.get(PIPELINE_OUTPUTS_KEY)
    if isinstance(pipeline_outputs, Mapping):
        unknown = sorted(set(pipeline_outputs).difference(PIPELINE_NAMES))
        if unknown:
            raise ValueError(f"unknown pipeline(s): {','.join(unknown)}")
        return tuple(
            (pipeline, pipeline_outputs[pipeline])
            for pipeline in PIPELINE_NAMES
            if pipeline in pipeline_outputs
        )
    return ((None, skins_output),)


def _pipeline_output_dir(output_dir_for_variant: Path, pipeline: str | None) -> Path:
    if pipeline is None:
        return output_dir_for_variant
    if pipeline not in {"oracle", "scanner"}:
        raise ValueError(f"unknown pipeline: {pipeline}")
    return output_dir_for_variant / pipeline


def write_case_figures(
    volume_outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    output_dir: str | PathLike[str],
    *,
    buffer_radius: float = 2.0,
) -> list[Path]:
    written = []
    output_root = Path(output_dir)
    for case_id, variants in volume_outputs.items():
        case_dir = _case_output_dir(output_root, case_id)
        for variant, volumes in variants.items():
            output_dir_for_variant = _variant_output_dir(case_dir, variant, len(variants) == 1)
            for pipeline, pipeline_volumes in _iter_pipeline_volume_outputs(volumes):
                figure_dir = _pipeline_output_dir(output_dir_for_variant, pipeline) / "figures"
                written.extend(
                    _write_pipeline_figures(
                        pipeline_volumes,
                        figure_dir,
                        case_id=case_id,
                        variant=variant,
                        pipeline=pipeline,
                        buffer_radius=buffer_radius,
                    )
                )
    return written


def _write_pipeline_figures(
    volumes: Mapping[str, np.ndarray],
    figures_dir: Path,
    *,
    case_id: str,
    variant: str,
    pipeline: str | None,
    buffer_radius: float,
) -> list[Path]:
    from pyosv import viz

    written = []
    title_parts = [case_id, variant]
    if pipeline is not None:
        title_parts.append(pipeline)
    title_prefix = " ".join(title_parts)
    indices = viz.select_center_slices(np.asarray(volumes["fvt_py"]).shape)
    for axis in ("i3", "i2", "i1"):
        index = indices[axis]
        for name in FIGURE_VOLUME_NAMES:
            figure_path = figures_dir / f"{name}_{axis}_center.png"
            written.append(
                viz.save_slice_panel(
                    figure_path,
                    [(name, viz.slice_2d(volumes[name], axis, index))],
                    title=f"{title_prefix} {name} {axis}=center",
                )
            )
        if "scanner_input" in volumes:
            for source_name, output_name in SCANNER_FIGURE_VOLUME_NAMES:
                figure_path = figures_dir / f"{output_name}_{axis}_center.png"
                written.append(
                    viz.save_slice_panel(
                        figure_path,
                        [
                            (
                                output_name,
                                viz.slice_2d(volumes[source_name], axis, index),
                            )
                        ],
                        title=f"{title_prefix} {output_name} {axis}=center",
                    )
                )
        if axis == "i3":
            written.append(
                viz.save_slice_panel(
                    figures_dir / "skin_mask_py_i3_center.png",
                    [
                        (
                            "skin_mask_py",
                            viz.slice_2d(volumes["skin_mask_py"], axis, index),
                        )
                    ],
                    title=f"{title_prefix} skin_mask_py {axis}=center",
                    clip_percentiles=(0.0, 100.0),
                )
            )
        written.append(
            viz.save_ridge_overlay_slice(
                figures_dir / f"truth_vs_fvt_overlay_{axis}_center.png",
                reference=volumes["truth_fault_mask"],
                candidate=volumes["fvt_py"],
                axis=axis,
                index=index,
                percentile=99.0,
                buffer_radius=buffer_radius,
                title=f"{title_prefix} truth vs fvt {axis}=center",
            )
        )
        written.append(
            viz.save_ridge_overlay_slice(
                figures_dir / f"truth_vs_skin_overlay_{axis}_center.png",
                reference=volumes["truth_fault_mask"],
                candidate=volumes["skin_mask_py"].astype(np.float32),
                axis=axis,
                index=index,
                percentile=99.0,
                buffer_radius=buffer_radius,
                title=f"{title_prefix} truth vs skin {axis}=center",
            )
        )
        if "scanner_ft" in volumes:
            written.append(
                viz.save_ridge_overlay_slice(
                    figures_dir / f"truth_vs_ft_scan_overlay_{axis}_center.png",
                    reference=volumes["truth_fault_mask"],
                    candidate=volumes["scanner_ft"],
                    axis=axis,
                    index=index,
                    percentile=99.0,
                    buffer_radius=buffer_radius,
                    title=f"{title_prefix} truth vs ft_scan {axis}=center",
                )
            )
            written.append(
                viz.save_ridge_overlay_slice(
                    figures_dir / f"truth_vs_ft_used_overlay_{axis}_center.png",
                    reference=volumes["truth_fault_mask"],
                    candidate=volumes["scanner_fet"],
                    axis=axis,
                    index=index,
                    percentile=99.0,
                    buffer_radius=buffer_radius,
                    title=f"{title_prefix} truth vs ft_used {axis}=center",
                )
            )
    written.extend(
        _write_thinning_diagnostic_figures(
            volumes,
            figures_dir.parent / "thinning_diagnostic",
            title_prefix=title_prefix,
            buffer_radius=buffer_radius,
        )
    )
    return written


def _write_thinning_diagnostic_figures(
    volumes: Mapping[str, np.ndarray],
    diagnostic_dir: Path,
    *,
    title_prefix: str,
    buffer_radius: float,
) -> list[Path]:
    if "fvt_reference_thinning_diagnostic" not in volumes:
        return []

    from pyosv import viz

    written = []
    indices = viz.select_center_slices(
        np.asarray(volumes["fvt_reference_thinning_diagnostic"]).shape
    )
    for axis in ("i3", "i2", "i1"):
        index = indices[axis]
        for source_name, output_name in (
            ("fvt_reference_thinning_diagnostic", "fvt_reference"),
            ("fvt_normal_thinning_diagnostic", "fvt_normal"),
            ("keep_reference_only_thinning_diagnostic", "keep_reference_only"),
            ("keep_normal_only_thinning_diagnostic", "keep_normal_only"),
        ):
            written.append(
                viz.save_ridge_overlay_slice(
                    diagnostic_dir / f"truth_vs_{output_name}_overlay_{axis}_center.png",
                    reference=volumes["truth_fault_mask"],
                    candidate=volumes[source_name],
                    axis=axis,
                    index=index,
                    percentile=99.0,
                    buffer_radius=buffer_radius,
                    title=f"{title_prefix} truth vs {output_name} {axis}=center",
                )
            )
        reference_slice = viz.slice_2d(volumes["fvt_reference_thinning_diagnostic"], axis, index)
        normal_slice = viz.slice_2d(volumes["fvt_normal_thinning_diagnostic"], axis, index)
        written.append(
            viz.save_slice_panel(
                diagnostic_dir / f"fvt_reference_vs_normal_{axis}_center.png",
                [
                    ("fvt_reference", reference_slice),
                    ("fvt_normal", normal_slice),
                    ("normal - reference", normal_slice - reference_slice),
                ],
                title=f"{title_prefix} fvt reference vs normal {axis}=center",
            )
        )
    return written


def write_visual_report_markdown(
    report: Mapping[str, Any],
    output_dir: str | PathLike[str],
) -> Path:
    output_path = Path(output_dir) / "visual_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(visual_report_markdown(report), encoding="utf-8")
    return output_path


def visual_report_markdown(report: Mapping[str, Any]) -> str:
    lines = ["# Controlled Synthetic Quality Report", ""]
    input_mode = str(report.get("config", {}).get("input_mode", "oracle"))
    for case in report["cases"]:
        case_id = str(case["case_id"])
        lines.extend(
            [
                f"## {case_id}",
                "",
            ]
        )
        variants = case["variants"]
        for variant, variant_report in variants.items():
            pipelines = variant_report.get("pipelines", {})
            lines.extend([f"### {variant}", ""])
            if input_mode == "both" and isinstance(pipelines, Mapping):
                oracle_report = pipelines["oracle"]
                scanner_report = pipelines["scanner"]
                lines.extend(
                    _pipeline_comparison_table(
                        oracle_report=oracle_report,
                        scanner_report=scanner_report,
                    )
                )
                lines.extend(
                    _visual_pipeline_section(
                        "oracle",
                        oracle_report,
                        case_id=case_id,
                        variant=variant,
                        variant_count=len(variants),
                        path_pipeline="oracle",
                        include_scanner=False,
                    )
                )
                lines.extend(
                    _visual_pipeline_section(
                        "scanner",
                        scanner_report,
                        case_id=case_id,
                        variant=variant,
                        variant_count=len(variants),
                        path_pipeline="scanner",
                        include_scanner=True,
                    )
                )
            else:
                lines.extend(
                    _visual_pipeline_section(
                        None,
                        variant_report,
                        case_id=case_id,
                        variant=variant,
                        variant_count=len(variants),
                        path_pipeline=None,
                        include_scanner="scanner_quality" in variant_report,
                    )
                )
    return "\n".join(lines).rstrip() + "\n"


def _visual_pipeline_section(
    pipeline_label: str | None,
    pipeline_report: Mapping[str, Any],
    *,
    case_id: str,
    variant: str,
    variant_count: int,
    path_pipeline: str | None,
    include_scanner: bool,
) -> list[str]:
    quality = pipeline_report["quality"]["fvt_top_truth_count"]
    overlap = quality["buffered_overlap_radius2"]
    distance = quality["surface_distance"]
    orientation = quality["orientation_error"]
    overlay_path = _figure_path(
        case_id,
        variant=variant,
        variant_count=variant_count,
        pipeline=path_pipeline,
        filename="truth_vs_fvt_overlay_i3_center.png",
    )
    skin_overlay_path = _figure_path(
        case_id,
        variant=variant,
        variant_count=variant_count,
        pipeline=path_pipeline,
        filename="truth_vs_skin_overlay_i3_center.png",
    )
    lines: list[str] = []
    if pipeline_label is not None:
        lines.extend([f"#### {pipeline_label} pipeline", ""])
    lines.extend(
        [
            f"- buffered_f1_r2: {_format_markdown_metric(overlap['buffered_f1'])}",
            f"- distance_p95: {_format_markdown_metric(distance['candidate_to_truth_p95'])}",
            f"- strike_median_error: {_format_markdown_metric(orientation['strike_median'])}",
            f"- dip_median_error: {_format_markdown_metric(orientation['dip_median'])}",
        ]
    )
    if include_scanner:
        lines.extend(_scanner_markdown_metrics(pipeline_report))
    if "thinning_diagnostic" in pipeline_report:
        lines.extend(
            _thinning_diagnostic_markdown(
                pipeline_report["thinning_diagnostic"],
                case_id=case_id,
                variant=variant,
                variant_count=variant_count,
                pipeline=path_pipeline,
            )
        )
    lines.extend(["", f"![fvt overlay]({overlay_path.as_posix()})", ""])
    if include_scanner:
        scanner_scan_overlay_path = _figure_path(
            case_id,
            variant=variant,
            variant_count=variant_count,
            pipeline=path_pipeline,
            filename="truth_vs_ft_scan_overlay_i3_center.png",
        )
        scanner_used_overlay_path = _figure_path(
            case_id,
            variant=variant,
            variant_count=variant_count,
            pipeline=path_pipeline,
            filename="truth_vs_ft_used_overlay_i3_center.png",
        )
        lines.extend(
            [
                f"![scanner ft scan overlay]({scanner_scan_overlay_path.as_posix()})",
                "",
                f"![scanner ft used overlay]({scanner_used_overlay_path.as_posix()})",
                "",
            ]
        )
    if bool(pipeline_report["skinning"]["enabled"]):
        skin_quality = pipeline_report["quality"]["skin"]
        skin_topology = skin_quality["topology"]
        skin_overlap = skin_quality["buffered_overlap_radius2"]
        skin_distance = skin_quality["surface_distance"]
        skin_orientation = skin_quality["orientation_error"]
        lines.extend(
            [
                f"- skin_count: {_format_markdown_metric(skin_topology['skin_count'])}",
                f"- skin_cell_count: {_format_markdown_metric(skin_topology['cell_count'])}",
                f"- skin_buffered_f1_r2: {_format_markdown_metric(skin_overlap['buffered_f1'])}",
                "- skin_distance_p95: "
                f"{_format_markdown_metric(skin_distance['candidate_to_truth_p95'])}",
                "- skin_strike_median_error: "
                f"{_format_markdown_metric(skin_orientation['strike_median'])}",
                "- skin_dip_median_error: "
                f"{_format_markdown_metric(skin_orientation['dip_median'])}",
                "",
                f"![skin overlay]({skin_overlay_path.as_posix()})",
                "",
            ]
        )
    else:
        lines.extend(["- skinning disabled", ""])
    return lines


def _thinning_diagnostic_markdown(
    diagnostic: Mapping[str, Any],
    *,
    case_id: str,
    variant: str,
    variant_count: int,
    pipeline: str | None,
) -> list[str]:
    reference_quality = diagnostic["reference"]["quality"]["fvt_top_truth_count"]
    normal_quality = diagnostic["normal"]["quality"]["fvt_top_truth_count"]
    delta = diagnostic["delta"]["normal_minus_reference"]
    keep_mask = diagnostic["keep_mask"]
    links = [
        (
            "reference overlay",
            "truth_vs_fvt_reference_overlay_i3_center.png",
        ),
        (
            "normal overlay",
            "truth_vs_fvt_normal_overlay_i3_center.png",
        ),
        (
            "reference-only overlay",
            "truth_vs_keep_reference_only_overlay_i3_center.png",
        ),
        (
            "normal-only overlay",
            "truth_vs_keep_normal_only_overlay_i3_center.png",
        ),
        (
            "reference vs normal",
            "fvt_reference_vs_normal_i3_center.png",
        ),
    ]
    lines = [
        "",
        "##### thinning diagnostic",
        "",
        "- reference buffered F1: "
        f"{_format_markdown_metric(reference_quality['buffered_overlap_radius2']['buffered_f1'])}",
        "- normal buffered F1: "
        f"{_format_markdown_metric(normal_quality['buffered_overlap_radius2']['buffered_f1'])}",
        f"- normal-minus-reference delta: {_format_markdown_metric(delta['fvt_buffered_f1_r2'])}",
        f"- keep-mask Jaccard: {_format_markdown_metric(keep_mask['jaccard'])}",
    ]
    for label, filename in links:
        path = _thinning_diagnostic_figure_path(
            case_id,
            variant=variant,
            variant_count=variant_count,
            pipeline=pipeline,
            filename=filename,
        )
        lines.append(f"- [{label}]({path.as_posix()})")
    return lines


def _scanner_markdown_metrics(pipeline_report: Mapping[str, Any]) -> list[str]:
    scanner_quality = pipeline_report["scanner_quality"]
    input_association = scanner_quality["input_association"]
    ft_quality = scanner_quality["ft_top_truth_count"]
    ft_overlap = ft_quality["buffered_overlap_radius2"]
    ft_distance = ft_quality["surface_distance"]
    orientation = scanner_quality["orientation_error"]["raw_scan_top_truth_count"]
    return [
        f"- scanner input contrast: {_format_markdown_metric(input_association['contrast'])}",
        f"- scanner ft buffered_f1: {_format_markdown_metric(ft_overlap['buffered_f1'])}",
        "- scanner ft distance_p95: "
        f"{_format_markdown_metric(ft_distance['candidate_to_truth_p95'])}",
        f"- scanner strike median error: {_format_markdown_metric(orientation['strike_median'])}",
        f"- scanner dip median error: {_format_markdown_metric(orientation['dip_median'])}",
    ]


def _pipeline_comparison_table(
    *,
    oracle_report: Mapping[str, Any],
    scanner_report: Mapping[str, Any],
) -> list[str]:
    headers = (
        "pipeline",
        "scanner input contrast",
        "scanner ft buffered_f1",
        "scanner ft distance_p95",
        "scanner strike median error",
        "scanner dip median error",
        "fvt buffered_f1",
        "skin buffered_f1",
    )
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        _pipeline_comparison_row("oracle", oracle_report, include_scanner=False),
        _pipeline_comparison_row("scanner", scanner_report, include_scanner=True),
        "",
    ]


def _pipeline_comparison_row(
    pipeline: str,
    pipeline_report: Mapping[str, Any],
    *,
    include_scanner: bool,
) -> str:
    fvt_overlap = pipeline_report["quality"]["fvt_top_truth_count"]["buffered_overlap_radius2"][
        "buffered_f1"
    ]
    skin_quality = pipeline_report["quality"]["skin"]
    skin_overlap = (
        skin_quality["buffered_overlap_radius2"]["buffered_f1"]
        if skin_quality is not None
        else "skinning disabled"
    )
    scanner_values: tuple[object, ...]
    if include_scanner:
        scanner_quality = pipeline_report["scanner_quality"]
        scanner_ft_quality = scanner_quality["ft_top_truth_count"]
        scanner_orientation = scanner_quality["orientation_error"]["raw_scan_top_truth_count"]
        scanner_values = (
            scanner_quality["input_association"]["contrast"],
            scanner_ft_quality["buffered_overlap_radius2"]["buffered_f1"],
            scanner_ft_quality["surface_distance"]["candidate_to_truth_p95"],
            scanner_orientation["strike_median"],
            scanner_orientation["dip_median"],
        )
    else:
        scanner_values = ("n/a", "n/a", "n/a", "n/a", "n/a")
    values = (pipeline, *scanner_values, fvt_overlap, skin_overlap)
    return "| " + " | ".join(_format_markdown_metric(value) for value in values) + " |"


def _figure_path(
    case_id: str,
    *,
    variant: str,
    variant_count: int,
    pipeline: str | None,
    filename: str,
) -> PurePosixPath:
    parts = [case_id]
    if variant_count > 1:
        parts.append(variant)
    if pipeline is not None:
        parts.append(pipeline)
    parts.extend(("figures", filename))
    return PurePosixPath(*parts)


def _thinning_diagnostic_figure_path(
    case_id: str,
    *,
    variant: str,
    variant_count: int,
    pipeline: str | None,
    filename: str,
) -> PurePosixPath:
    parts = [case_id]
    if variant_count > 1:
        parts.append(variant)
    if pipeline is not None:
        parts.append(pipeline)
    parts.extend(("thinning_diagnostic", filename))
    return PurePosixPath(*parts)


def _format_markdown_metric(value: object) -> str:
    if isinstance(value, int | float | np.floating | np.integer):
        return f"{float(value):.6g}"
    return str(value)


def _case_output_dir(output_dir: Path, case_id: str) -> Path:
    relative_case_path = PurePosixPath(case_id)
    if (
        relative_case_path.is_absolute()
        or not relative_case_path.parts
        or any(part in {"", ".", ".."} for part in relative_case_path.parts)
    ):
        raise ValueError(f"case_id must be a relative path inside output_dir: {case_id!r}")
    return output_dir.joinpath(*relative_case_path.parts)


def _variant_output_dir(case_dir: Path, variant: str, is_single_variant: bool) -> Path:
    if is_single_variant:
        return case_dir
    if variant not in VARIANT_NAMES:
        raise ValueError(f"unknown variant: {variant}")
    return case_dir / variant


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
