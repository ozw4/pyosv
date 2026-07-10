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
import math
import sys
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from pyosv.synthetic3d import (
    Synthetic3DCase,
    make_boundary_plane_case,  # noqa: F401 - compatibility export
    validate_shape3,
)
from pyosv.synthetic_metrics import (
    buffered_surface_overlap,
    edge_false_positive_ratio,
    masked_orientation_error,
    skin_mask_from_skins,
    skin_topology_metrics,
    skin_truth_metrics,
    surface_distance_metrics,
    top_positive_truth_count_mask,
    top_truth_count_mask,
)
from pyosv.evaluation.synthetic_quality.config import (
    SKINNER_GROWTH_SOURCES,
    SKINNER_METHODS,
    SyntheticScannerConfig,
    SyntheticSkinningConfig,
    SyntheticTruthMetricConfig,
    SyntheticVotingConfig,
    _validate_nonnegative_finite_scalar,
    _validate_nonnegative_int,
)
from pyosv.evaluation.synthetic_quality import quality_metrics
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
from pyosv.skinner import FaultSkinner, find_connected_component_skins
from pyosv.voting3d import OptimalSurfaceVoter

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
SKIN_PRIMARY_DEGRADED_MIN_CELL_COVERAGE = 0.50
SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_SKIN_COUNT = 8
SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_LARGEST_FRACTION = 0.75
SKIN_PRIMARY_DEGRADED_MAX_SMALL_CELL_FRACTION = 0.25
# Boundary degraded-primary fallback is intentionally stricter than the generic
# degraded-primary diagnostics. These thresholds require boundary-local evidence
# before diagnostic degraded-primary variants replace a non-empty primary skin.
SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_FVT_EDGE_SHELL_FRACTION = 0.25
SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_SCANNER_TARGET_DISTANCE_P95 = 2.0
SKIN_PRIMARY_BOUNDARY_DEGRADED_MAX_CELL_COVERAGE = 0.50
SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_EDGE_LOCAL_FVT_FRACTION = 0.15
SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_EDGE_LOCAL_PRIMARY_FRACTION = 0.05
SKIN_FALLBACK_FILTER_MAX_COMPONENTS = 3
SKIN_FALLBACK_FILTER_MIN_COMPONENT_SIZE_FLOOR = 8
SKIN_FALLBACK_FILTER_MIN_COMPONENT_FRACTION = 0.05
SKIN_FALLBACK_FILTER_MIN_COMPONENT_FRACTION_OF_LARGEST = 0.10
SKIN_FALLBACK_V5_MAX_SKIN_COUNT = 3
SKIN_FALLBACK_V5_MIN_COVERAGE_OF_FVT_POSITIVE = 0.75
SKIN_FALLBACK_V5_MAX_COVERAGE_OF_FVT_POSITIVE = 1.25
SKIN_FALLBACK_V5_MAX_SMALL_SKIN_CELL_FRACTION = 0.20
SKIN_FALLBACK_V5_MIN_LARGEST_SKIN_FRACTION = 0.50
SKIN_FALLBACK_V5_MAX_PRUNED_FRACTION = 0.60
DEFAULT_VARIANT_PRESET = "default"
DEFAULT_THINNING_DIAGNOSTIC_CASES = ("curved_surface",)
FVT_RECENTER_MAX_SHIFT = 3
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
    thinning_diagnostic_cases: Sequence[str] = DEFAULT_THINNING_DIAGNOSTIC_CASES,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case = case_definition.factory(shape)
    if case.case_id != case_definition.case_id:
        raise ValueError(
            f"case factory returned {case.case_id!r}, expected {case_definition.case_id!r}"
        )
    diagnostic_case_ids = set(_validate_thinning_diagnostic_cases(thinning_diagnostic_cases))
    variant_report, volumes, _ = _run_case_variant(
        case,
        voting_config=voting_config,
        scanner_config=scanner_config,
        truth_metric_config=truth_metric_config,
        skinning_config=skinning_config,
        variant=variant,
        input_mode=input_mode,
        scanner_backend_matrix=scanner_backend_matrix,
        include_thinning_diagnostic=(
            include_thinning_diagnostic and case.case_id in diagnostic_case_ids
        ),
        include_scanner_downstream_diagnostics=include_scanner_downstream_diagnostics,
    )
    report = {
        "case_id": case.case_id,
        "shape": [int(size) for size in case.shape],
        "truth": _truth_report(case, truth_metric_config),
        "variants": {variant: variant_report},
    }
    if variant == BASELINE_VARIANT:
        report.update(
            {key: value for key, value in variant_report.items() if key not in {"pipelines"}}
        )
    pipelines = _case_pipeline_reports({variant: variant_report}, input_mode)
    report["pipelines"] = pipelines
    report["variant_comparison"] = _case_variant_comparison_alias(pipelines, input_mode)
    return report, volumes


def _run_case_variant(
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
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    variant_spec = get_variant_spec(variant)
    valid_input_mode = _validate_input_mode(input_mode)
    skinning_config = effective_skinning_config(variant_spec, skinning_config)

    if valid_input_mode == "oracle":
        return _run_oracle_pipeline(
            case,
            voting_config=voting_config,
            truth_metric_config=truth_metric_config,
            skinning_config=skinning_config,
            variant_spec=variant_spec,
            include_thinning_diagnostic=include_thinning_diagnostic,
        )

    pipelines = {}
    pipeline_outputs = {}
    if valid_input_mode == "both":
        oracle_report, oracle_volumes, oracle_skins = _run_oracle_pipeline(
            case,
            voting_config=voting_config,
            truth_metric_config=truth_metric_config,
            skinning_config=skinning_config,
            variant_spec=variant_spec,
            include_thinning_diagnostic=include_thinning_diagnostic,
        )
        pipelines["oracle"] = oracle_report
        pipeline_outputs["oracle"] = (oracle_report, oracle_volumes, oracle_skins)

    scanner_report, scanner_volumes, scanner_skins = _run_scanner_pipeline(
        case,
        voting_config=voting_config,
        scanner_config=scanner_config,
        truth_metric_config=truth_metric_config,
        skinning_config=skinning_config,
        variant_spec=variant_spec,
        scanner_backend_matrix=scanner_backend_matrix,
        include_thinning_diagnostic=include_thinning_diagnostic,
        include_scanner_downstream_diagnostics=include_scanner_downstream_diagnostics,
    )
    pipelines["scanner"] = scanner_report
    pipeline_outputs["scanner"] = (scanner_report, scanner_volumes, scanner_skins)

    active_pipeline = "scanner" if valid_input_mode == "scanner" else "oracle"
    active_report, active_volumes, active_skins = pipeline_outputs[active_pipeline]
    report = dict(active_report)
    report["active_pipeline"] = active_pipeline
    report["pipelines"] = pipelines
    if valid_input_mode == "both":
        pipeline_volumes = {pipeline: outputs[1] for pipeline, outputs in pipeline_outputs.items()}
        pipeline_skins = {pipeline: outputs[2] for pipeline, outputs in pipeline_outputs.items()}
        return (
            report,
            {PIPELINE_OUTPUTS_KEY: pipeline_volumes},
            {PIPELINE_OUTPUTS_KEY: pipeline_skins},
        )
    return report, active_volumes, active_skins


def _validate_input_mode(input_mode: str) -> str:
    if input_mode not in {"oracle", "scanner", "both"}:
        raise ValueError("input_mode must be 'oracle', 'scanner', or 'both'")
    return input_mode


def _case_pipeline_reports(
    variant_reports: Mapping[str, Mapping[str, Any]],
    input_mode: str,
) -> dict[str, dict[str, Any]]:
    pipeline_names = {
        "oracle": ("oracle",),
        "scanner": ("scanner",),
        "both": ("oracle", "scanner"),
    }[_validate_input_mode(input_mode)]
    return {
        pipeline: {
            "variants": {
                variant: _variant_pipeline_report(variant_report, pipeline)
                for variant, variant_report in variant_reports.items()
            },
            "variant_comparison": _variant_comparison(
                {
                    variant: _variant_pipeline_report(variant_report, pipeline)
                    for variant, variant_report in variant_reports.items()
                }
            ),
        }
        for pipeline in pipeline_names
    }


def _variant_pipeline_report(
    variant_report: Mapping[str, Any],
    pipeline: str,
) -> Mapping[str, Any]:
    pipelines = variant_report.get("pipelines")
    if isinstance(pipelines, Mapping) and pipeline in pipelines:
        pipeline_report = pipelines[pipeline]
        if not isinstance(pipeline_report, Mapping):
            raise TypeError(f"pipeline report must be a mapping: {pipeline}")
        return pipeline_report
    if pipeline == "oracle" and "scanner_quality" not in variant_report:
        return variant_report
    if pipeline == "scanner" and "scanner_quality" in variant_report:
        return variant_report
    raise KeyError(f"missing pipeline report: {pipeline}")


def _case_variant_comparison_alias(
    pipelines: Mapping[str, Mapping[str, Any]],
    input_mode: str,
) -> dict[str, Any]:
    valid_input_mode = _validate_input_mode(input_mode)
    if valid_input_mode == "both":
        return {
            "pipelines": {
                pipeline: pipeline_report["variant_comparison"]
                for pipeline, pipeline_report in pipelines.items()
            }
        }
    active_pipeline = "scanner" if valid_input_mode == "scanner" else "oracle"
    return dict(pipelines[active_pipeline]["variant_comparison"])


def _run_oracle_pipeline(
    case: Synthetic3DCase,
    *,
    voting_config: SyntheticVotingConfig,
    truth_metric_config: SyntheticTruthMetricConfig,
    skinning_config: SyntheticSkinningConfig,
    variant_spec: VariantSpec,
    include_thinning_diagnostic: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    return _run_voting_from_attributes(
        case,
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
        voting_config=voting_config,
        truth_metric_config=truth_metric_config,
        skinning_config=skinning_config,
        variant_spec=variant_spec,
        include_thinning_diagnostic=include_thinning_diagnostic,
        fvt_recenter_target=case.ft_oracle,
        fvt_recenter_target_source="oracle_ft",
    )


def _run_scanner_pipeline(
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
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    scanner_report, scanner_volumes = _scanner_attributes_from_case(case, scanner_config)
    report, volumes, skins_output = _run_voting_from_attributes(
        case,
        ft=scanner_volumes["scanner_fet"],
        pt=scanner_volumes["scanner_fpt"],
        tt=scanner_volumes["scanner_ftt"],
        voting_config=voting_config,
        truth_metric_config=truth_metric_config,
        skinning_config=skinning_config,
        variant_spec=variant_spec,
        include_thinning_diagnostic=include_thinning_diagnostic,
        scanner_target_positive_mask=_positive_candidate_mask(scanner_volumes["scanner_ft"]),
        fvt_recenter_target=scanner_volumes["scanner_fet"],
        fvt_recenter_target_source="scanner_fet",
    )
    report["scanner"] = scanner_report
    report["scanner_quality"] = _scanner_truth_quality(
        case,
        scanner_volumes=scanner_volumes,
        truth_metric_config=truth_metric_config,
    )
    if include_scanner_downstream_diagnostics:
        report["scanner_downstream"] = _scanner_downstream_diagnostics(
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
        report["scanner_stage_loss"] = _scanner_stage_loss_diagnostics(
            case=case,
            voting_config=voting_config,
            variant_spec=variant_spec,
            scanner_volumes=scanner_volumes,
            fv=volumes["fv_py"],
            fvt=volumes["fvt_py"],
            skin_mask=volumes["skin_mask_py"],
            truth_metric_config=truth_metric_config,
        )
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
        )
    volumes.update(scanner_volumes)
    return report, volumes, skins_output


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


def _scanner_downstream_diagnostics(
    *,
    case: Synthetic3DCase,
    scanner_config: SyntheticScannerConfig,
    voting_config: SyntheticVotingConfig,
    variant_spec: VariantSpec,
    report: Mapping[str, Any],
    scanner_volumes: Mapping[str, np.ndarray],
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    fvt: np.ndarray,
    truth_metric_config: SyntheticTruthMetricConfig,
) -> dict[str, Any]:
    scanner_ft_positive = _positive_candidate_mask(scanner_volumes["scanner_ft"])
    scanner_fet_positive = _positive_candidate_mask(scanner_volumes["scanner_fet"])
    fv_positive = _positive_candidate_mask(fv)
    fvt_positive = _positive_candidate_mask(fvt)
    scanner_ft_positive_count = _candidate_count(scanner_ft_positive)
    scanner_fet_positive_count = _candidate_count(scanner_fet_positive)
    fv_positive_count = _candidate_count(fv_positive)
    fvt_positive_count = _candidate_count(fvt_positive)
    fvt_to_scanner_ft_distance = surface_distance_metrics(
        fvt_positive,
        scanner_ft_positive,
    )
    fvt_to_fv_distance = surface_distance_metrics(
        fvt_positive,
        fv_positive,
    )
    buffer_radius = _validate_nonnegative_finite_scalar(
        truth_metric_config.buffer_radius,
        "buffer_radius",
    )
    voter_thin_mode = effective_thin_mode(variant_spec, voting_config)
    plateau_source = "scanner_fet" if voter_thin_mode in {"hybrid_v2", "normal_plateau"} else None

    diagnostic = {
        "scanner_ft_positive_candidate_count": scanner_ft_positive_count,
        "scanner_fet_positive_candidate_count": scanner_fet_positive_count,
        "scanner_ft_to_fv_positive_candidate_count_ratio": _fraction_or_zero(
            fv_positive_count,
            scanner_ft_positive_count,
        ),
        "scanner_ft_to_fvt_positive_candidate_count_ratio": _fraction_or_zero(
            fvt_positive_count,
            scanner_ft_positive_count,
        ),
        "scanner_ft_to_fet_retention_fraction": _fraction_or_zero(
            scanner_fet_positive_count,
            scanner_ft_positive_count,
        ),
        "fv_positive_candidate_count": fv_positive_count,
        "fvt_positive_candidate_count": fvt_positive_count,
        "fv_to_fvt_positive_candidate_count_ratio": _fraction_or_zero(
            fvt_positive_count,
            fv_positive_count,
        ),
        "fvt_to_fv_positive_fraction": _fraction_or_zero(
            fvt_positive_count,
            fv_positive_count,
        ),
        "scanner_ft_vs_fv_positive_buffered_overlap_radius2": _positive_pair_overlap(
            candidate_name="scanner_ft",
            reference_name="fv",
            candidate_mask=scanner_ft_positive,
            reference_mask=fv_positive,
            buffer_radius=buffer_radius,
        ),
        "scanner_ft_vs_fvt_positive_buffered_overlap_radius2": _positive_pair_overlap(
            candidate_name="scanner_ft",
            reference_name="fvt",
            candidate_mask=scanner_ft_positive,
            reference_mask=fvt_positive,
            buffer_radius=buffer_radius,
        ),
        "fv_vs_fvt_positive_buffered_overlap_radius2": _positive_pair_overlap(
            candidate_name="fv",
            reference_name="fvt",
            candidate_mask=fv_positive,
            reference_mask=fvt_positive,
            buffer_radius=buffer_radius,
        ),
        "fvt_candidate_to_scanner_ft_distance_p50": fvt_to_scanner_ft_distance[
            "candidate_to_truth_median"
        ],
        "fvt_candidate_to_scanner_ft_distance_p95": fvt_to_scanner_ft_distance[
            "candidate_to_truth_p95"
        ],
        "fvt_candidate_to_fv_distance_p50": fvt_to_fv_distance["candidate_to_truth_median"],
        "fvt_candidate_to_fv_distance_p95": fvt_to_fv_distance["candidate_to_truth_p95"],
        "scanner_ft_positive_edge_shell_fraction": _edge_candidate_fraction(
            scanner_ft_positive,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        ),
        "scanner_fet_positive_edge_shell_fraction": _edge_candidate_fraction(
            scanner_fet_positive,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        ),
        "fv_positive_edge_shell_fraction": _edge_candidate_fraction(
            fv_positive,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        ),
        "fvt_positive_edge_shell_fraction": _edge_candidate_fraction(
            fvt_positive,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        ),
        "fvt_positive_edge_candidate_fraction": _edge_candidate_fraction(
            fvt_positive,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        ),
        "fvt_positive_edge_false_positive_fraction": report["quality"]["edge_false_positive"][
            "fvt_positive_top_truth_count"
        ]["edge_false_positive_fraction_of_candidates"],
        "voter_thin_mode": voter_thin_mode,
        "plateau_tie_breaker_source": plateau_source,
        "scanner_thin_mode": scanner_config.scanner_thin_mode,
    }

    voter = OptimalSurfaceVoter(
        ru=voting_config.ru,
        rv=voting_config.rv,
        rw=voting_config.rw,
    )
    thinning_modes = {}
    for mode in ("reference", "hybrid", "hybrid_v2", "normal_plateau"):
        tie_breaker = (
            scanner_volumes["scanner_fet"] if mode in {"hybrid_v2", "normal_plateau"} else None
        )
        thinning_modes[mode] = _scanner_downstream_thinning_report(
            case=case,
            voter=voter,
            fv=fv,
            vp=vp,
            vt=vt,
            mode=mode,
            plateau_tie_breaker=tie_breaker,
            truth_metric_config=truth_metric_config,
            reference_sigma=voting_config.reference_thin_sigma,
        )
    diagnostic["thinning_modes"] = thinning_modes

    diagnostic["hybrid_v2_tiebreaker_fet"] = _scanner_downstream_thinning_report(
        case=case,
        voter=voter,
        fv=fv,
        vp=vp,
        vt=vt,
        mode="hybrid_v2",
        plateau_tie_breaker=scanner_volumes["scanner_fet"],
        truth_metric_config=truth_metric_config,
        reference_sigma=voting_config.reference_thin_sigma,
    )
    diagnostic["hybrid_v2_tiebreaker_fv"] = _scanner_downstream_thinning_report(
        case=case,
        voter=voter,
        fv=fv,
        vp=vp,
        vt=vt,
        mode="hybrid_v2",
        plateau_tie_breaker=fv,
        truth_metric_config=truth_metric_config,
        reference_sigma=voting_config.reference_thin_sigma,
    )
    diagnostic["hybrid_v2_tiebreaker_scanner_ft"] = _scanner_downstream_thinning_report(
        case=case,
        voter=voter,
        fv=fv,
        vp=vp,
        vt=vt,
        mode="hybrid_v2",
        plateau_tie_breaker=scanner_volumes["scanner_ft"],
        truth_metric_config=truth_metric_config,
        reference_sigma=voting_config.reference_thin_sigma,
    )
    return diagnostic


def _scanner_stage_loss_diagnostics(
    *,
    case: Synthetic3DCase,
    voting_config: SyntheticVotingConfig,
    variant_spec: VariantSpec,
    scanner_volumes: Mapping[str, np.ndarray],
    fv: np.ndarray,
    fvt: np.ndarray,
    skin_mask: np.ndarray,
    truth_metric_config: SyntheticTruthMetricConfig,
) -> dict[str, Any]:
    truth_surface_half_width = _validate_nonnegative_finite_scalar(
        truth_metric_config.truth_surface_half_width,
        "truth_surface_half_width",
    )
    buffer_radius = _validate_nonnegative_finite_scalar(
        truth_metric_config.buffer_radius,
        "buffer_radius",
    )
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(truth_surface_half_width)
    truth_fault_mask = np.asarray(case.truth_fault_mask, dtype=bool)
    seed_selected_mask = _seed_selection_diagnostic(
        shape=case.shape,
        voting_config=voting_config,
        ft=scanner_volumes["scanner_fet"],
        pt=scanner_volumes["scanner_fpt"],
        tt=scanner_volumes["scanner_ftt"],
    )
    boundary_seed_retention: dict[str, Any] | None = None
    if variant_spec.seed_policy == "boundary_seed_retention_v1":
        _, seeds, boundary_seed_retention = _boundary_seed_retention_v1_seeds(
            voting_config=voting_config,
            ft=scanner_volumes["scanner_fet"],
            pt=scanner_volumes["scanner_fpt"],
            tt=scanner_volumes["scanner_ftt"],
            target=scanner_volumes["scanner_fet"],
            target_source="scanner_fet",
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        )
        seed_selected_mask = _seed_mask_from_seeds(case.shape, seeds)

    stage_masks = {
        "scanner_ft_positive": _positive_candidate_mask(scanner_volumes["scanner_ft"]),
        "scanner_fet_positive": _positive_candidate_mask(scanner_volumes["scanner_fet"]),
        "seed_candidate": np.asarray(scanner_volumes["scanner_fet"])
        > np.float32(voting_config.seed_threshold),
        "seed_selected": seed_selected_mask,
        "fv_positive": _positive_candidate_mask(fv),
        "fvt_positive": _positive_candidate_mask(fvt),
        "skin": np.asarray(skin_mask, dtype=bool),
    }
    stages = {
        name: _scanner_stage_metric(
            mask,
            truth_fault_mask=truth_fault_mask,
            truth_surface_mask=truth_surface_mask,
            buffer_radius=buffer_radius,
        )
        for name, mask in stage_masks.items()
    }
    if boundary_seed_retention is not None:
        stages["seed_selected"]["default_candidate_count"] = int(
            boundary_seed_retention["default_seed_count"]
        )
        stages["seed_selected"]["added_candidate_count"] = int(
            boundary_seed_retention["added_seed_count"]
        )

    transition_pairs = (
        ("scanner_ft_positive", "scanner_fet_positive"),
        ("scanner_fet_positive", "seed_candidate"),
        ("seed_candidate", "seed_selected"),
        ("seed_selected", "fv_positive"),
        ("fv_positive", "fvt_positive"),
        ("fvt_positive", "skin"),
        ("scanner_fet_positive", "seed_selected"),
    )
    transitions = {
        f"{source}_to_{target}": _scanner_stage_transition_metric(
            source_mask=stage_masks[source],
            target_mask=stage_masks[target],
            buffer_radius=buffer_radius,
        )
        for source, target in transition_pairs
    }
    return {"stages": stages, "transitions": transitions}


def _seed_selection_diagnostic(
    *,
    shape: tuple[int, int, int],
    voting_config: SyntheticVotingConfig,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
) -> np.ndarray:
    voter = OptimalSurfaceVoter(
        ru=voting_config.ru,
        rv=voting_config.rv,
        rw=voting_config.rw,
    )
    seeds = voter.pick_seeds(
        d=voting_config.seed_distance,
        fm=voting_config.seed_threshold,
        ft=ft,
        pt=pt,
        tt=tt,
    )
    return _seed_mask_from_seeds(shape, seeds)


def _seed_mask_from_seeds(
    shape: tuple[int, int, int],
    seeds: Sequence[FaultCell],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    n3, n2, n1 = shape
    for seed in seeds:
        if 0 <= seed.i3 < n3 and 0 <= seed.i2 < n2 and 0 <= seed.i1 < n1:
            mask[seed.i3, seed.i2, seed.i1] = True
    return mask


def _boundary_seed_retention_v1_seeds(
    *,
    voting_config: SyntheticVotingConfig,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
    target: np.ndarray,
    target_source: str,
    edge_margin: int,
) -> tuple[list[FaultCell], list[FaultCell], dict[str, Any]]:
    ft_array = np.asarray(ft, dtype=np.float32)
    pt_array = np.asarray(pt, dtype=np.float32)
    tt_array = np.asarray(tt, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    if ft_array.ndim != 3:
        raise ValueError("ft must be a 3D array")
    if (
        pt_array.shape != ft_array.shape
        or tt_array.shape != ft_array.shape
        or target_array.shape != ft_array.shape
    ):
        raise ValueError("boundary_seed_retention_v1 input shapes must match")
    if not (
        np.all(np.isfinite(ft_array))
        and np.all(np.isfinite(pt_array))
        and np.all(np.isfinite(tt_array))
        and np.all(np.isfinite(target_array))
    ):
        raise ValueError("boundary_seed_retention_v1 inputs must contain only finite values")

    voter = OptimalSurfaceVoter(
        ru=voting_config.ru,
        rv=voting_config.rv,
        rw=voting_config.rw,
    )
    default_seeds = voter.pick_seeds(
        d=voting_config.seed_distance,
        fm=voting_config.seed_threshold,
        ft=ft_array,
        pt=pt_array,
        tt=tt_array,
    )
    edge_shell = _edge_mask(ft_array.shape, edge_margin)
    edge_ft = ft_array[edge_shell]
    edge_ft_max = float(np.max(edge_ft)) if edge_ft.size else 0.0
    ft_threshold = min(float(voting_config.seed_threshold), 0.5 * edge_ft_max)
    boundary_candidate = (
        edge_shell & _positive_candidate_mask(target_array) & (ft_array > np.float32(ft_threshold))
    )
    boundary_candidate_count = int(np.count_nonzero(boundary_candidate))
    existing_coordinates = {(seed.i1, seed.i2, seed.i3) for seed in default_seeds}
    added_coordinates: set[tuple[int, int, int]] = set()
    added_seeds = []
    distance = max(0, int(voting_config.seed_distance))
    candidate_records = []
    for index in np.argwhere(boundary_candidate):
        i3, i2, i1 = (int(index[0]), int(index[1]), int(index[2]))
        candidate_records.append(
            (
                -float(target_array[i3, i2, i1]),
                -float(ft_array[i3, i2, i1]),
                int(np.ravel_multi_index((i3, i2, i1), ft_array.shape)),
                i1,
                i2,
                i3,
            )
        )
    for _, _, _, i1, i2, i3 in sorted(candidate_records):
        coordinate = (i1, i2, i3)
        if coordinate in existing_coordinates:
            continue
        if any(
            abs(i1 - a1) <= distance and abs(i2 - a2) <= distance and abs(i3 - a3) <= distance
            for a1, a2, a3 in added_coordinates
        ):
            continue
        added_coordinates.add(coordinate)
        added_seeds.append(
            FaultCell(
                i1,
                i2,
                i3,
                ft_array[i3, i2, i1],
                pt_array[i3, i2, i1],
                tt_array[i3, i2, i1],
            )
        )

    retained_seeds = [*default_seeds, *added_seeds]
    added_target_values = np.asarray(
        [target_array[seed.i3, seed.i2, seed.i1] for seed in added_seeds],
        dtype=np.float32,
    )
    diagnostic = {
        "enabled": True,
        "target_source": target_source,
        "edge_margin": int(edge_margin),
        "default_seed_count": int(len(default_seeds)),
        "boundary_candidate_count": boundary_candidate_count,
        "added_seed_count": int(len(added_seeds)),
        "total_seed_count": int(len(retained_seeds)),
        "added_seed_edge_shell_fraction": _edge_candidate_fraction(
            _seed_mask_from_seeds(ft_array.shape, added_seeds),
            edge_margin=edge_margin,
        ),
        "added_seed_target_mean": float(np.mean(added_target_values))
        if added_target_values.size
        else 0.0,
        "added_seed_target_p95": float(np.percentile(added_target_values, 95))
        if added_target_values.size
        else 0.0,
    }
    return default_seeds, retained_seeds, diagnostic


def _scanner_stage_metric(
    candidate_mask: np.ndarray,
    *,
    truth_fault_mask: np.ndarray,
    truth_surface_mask: np.ndarray,
    buffer_radius: float,
) -> dict[str, int | float]:
    return quality_metrics.scanner_stage_metric(
        candidate_mask,
        truth_fault_mask=truth_fault_mask,
        truth_surface_mask=truth_surface_mask,
        buffer_radius=buffer_radius,
    )


def _scanner_stage_transition_metric(
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    buffer_radius: float,
) -> dict[str, int | float]:
    return quality_metrics.scanner_stage_transition_metric(
        source_mask=source_mask,
        target_mask=target_mask,
        buffer_radius=buffer_radius,
    )


def _scanner_downstream_thinning_report(
    *,
    case: Synthetic3DCase,
    voter: OptimalSurfaceVoter,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    mode: str,
    plateau_tie_breaker: np.ndarray | None,
    truth_metric_config: SyntheticTruthMetricConfig,
    reference_sigma: float,
) -> dict[str, Any]:
    fvt = voter.thin(
        fv,
        vp,
        vt,
        mode=mode,
        reference_sigma=reference_sigma,
        plateau_tie_breaker=plateau_tie_breaker,
    )
    truth_surface_half_width = _validate_nonnegative_finite_scalar(
        truth_metric_config.truth_surface_half_width,
        "truth_surface_half_width",
    )
    buffer_radius = _validate_nonnegative_finite_scalar(
        truth_metric_config.buffer_radius,
        "buffer_radius",
    )
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(truth_surface_half_width)
    truth_fault_mask = np.asarray(case.truth_fault_mask, dtype=bool)
    fvt_positive = top_positive_truth_count_mask(fvt, truth_surface_mask)
    quality = _top_truth_count_quality(
        fvt_positive,
        truth_fault_mask=truth_fault_mask,
        truth_surface_mask=truth_surface_mask,
        buffer_radius=buffer_radius,
    )
    return {
        "fvt_positive_candidate_count": int(quality["buffered_overlap_radius2"]["candidate_count"]),
        "fvt_positive_buffered_f1_r2": quality["buffered_overlap_radius2"]["buffered_f1"],
        "fvt_positive_distance_p95": quality["surface_distance"]["candidate_to_truth_p95"],
        "fvt_positive_edge_candidate_fraction": _edge_candidate_fraction(
            fvt > np.float32(NONZERO_EPSILON),
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        ),
        "fvt_positive_edge_false_positive_fraction": edge_false_positive_ratio(
            fvt_positive,
            truth_surface_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        )["edge_false_positive_fraction_of_candidates"],
    }


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


def _recenter_edge_fvt_to_target(
    fvt: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    target: np.ndarray,
    target_source: str,
    max_shift: int,
    edge_margin: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    fvt_array = np.asarray(fvt, dtype=np.float32)
    vp_array = np.asarray(vp, dtype=np.float32)
    vt_array = np.asarray(vt, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    if fvt_array.ndim != 3:
        raise ValueError("fvt must be a 3D array")
    if vp_array.shape != fvt_array.shape or vt_array.shape != fvt_array.shape:
        raise ValueError("fvt, vp, and vt shapes must match")
    if target_array.shape != fvt_array.shape:
        raise ValueError("fvt and target shapes must match")
    if not (
        np.all(np.isfinite(fvt_array))
        and np.all(np.isfinite(vp_array))
        and np.all(np.isfinite(vt_array))
        and np.all(np.isfinite(target_array))
    ):
        raise ValueError("fvt recenter inputs must contain only finite values")
    shift_limit = _validate_nonnegative_int(max_shift, "max_shift")
    before_positive = _positive_candidate_mask(fvt_array)
    edge_candidates = before_positive & _edge_mask(fvt_array.shape, edge_margin)
    edge_candidate_indices = np.argwhere(edge_candidates)
    candidate_count = int(edge_candidate_indices.shape[0])
    recentered = np.zeros_like(fvt_array, dtype=np.float32)
    if candidate_count == 0:
        recentered[before_positive] = fvt_array[before_positive]
        return recentered, _empty_fvt_recenter_diagnostic(
            target_source=target_source,
            candidate_count=0,
            positive_count_before=int(np.count_nonzero(before_positive)),
            positive_count_after=int(np.count_nonzero(recentered > np.float32(NONZERO_EPSILON))),
            before_positive=before_positive,
        )

    destination_to_candidate: dict[
        tuple[int, int, int], tuple[int, float, float, tuple[int, int, int]]
    ] = {}
    shifts: list[float] = []
    moved_count = 0
    collision_count = 0
    for stable_index, (i3, i2, i1) in enumerate(np.argwhere(before_positive)):
        source = (int(i3), int(i2), int(i1))
        if bool(edge_candidates[source]):
            destination, shift = _fvt_recenter_destination(
                source,
                vp_array,
                vt_array,
                target_array,
                max_shift=shift_limit,
            )
            shifts.append(float(shift))
            if shift > 0:
                moved_count += 1
        else:
            destination, shift = source, 0
        value = float(fvt_array[source])
        existing = destination_to_candidate.get(destination)
        candidate_record = (stable_index, value, float(shift), source)
        if existing is None:
            destination_to_candidate[destination] = candidate_record
            continue
        collision_count += 1
        if value > existing[1] or (value == existing[1] and stable_index < existing[0]):
            destination_to_candidate[destination] = candidate_record

    for destination, (_, value, _, _) in destination_to_candidate.items():
        recentered[destination] = np.float32(value)

    shift_values = np.asarray(shifts, dtype=np.float64)
    diagnostic = {
        "fvt_recenter_enabled": True,
        "fvt_recenter_target_source": target_source,
        "fvt_recenter_candidate_count": candidate_count,
        "fvt_recenter_moved_count": int(moved_count),
        "fvt_recenter_collision_count": int(collision_count),
        "fvt_recenter_mean_shift": (float(np.mean(shift_values)) if shift_values.size else 0.0),
        "fvt_recenter_p95_shift": (
            float(np.percentile(shift_values, 95)) if shift_values.size else 0.0
        ),
        "fvt_recenter_max_shift": (float(np.max(shift_values)) if shift_values.size else 0.0),
        "fvt_recenter_edge_shell_only": True,
        "fvt_recenter_positive_count_before": int(np.count_nonzero(before_positive)),
        "fvt_recenter_positive_count_after": int(
            np.count_nonzero(recentered > np.float32(NONZERO_EPSILON))
        ),
        "fvt_recenter_value_source": "original_fvt",
        "_before_positive_mask": before_positive,
    }
    return recentered, diagnostic


def _empty_fvt_recenter_diagnostic(
    *,
    target_source: str,
    candidate_count: int,
    positive_count_before: int,
    positive_count_after: int,
    before_positive: np.ndarray,
) -> dict[str, Any]:
    return {
        "fvt_recenter_enabled": True,
        "fvt_recenter_target_source": target_source,
        "fvt_recenter_candidate_count": int(candidate_count),
        "fvt_recenter_moved_count": 0,
        "fvt_recenter_collision_count": 0,
        "fvt_recenter_mean_shift": 0.0,
        "fvt_recenter_p95_shift": 0.0,
        "fvt_recenter_max_shift": 0.0,
        "fvt_recenter_edge_shell_only": True,
        "fvt_recenter_positive_count_before": int(positive_count_before),
        "fvt_recenter_positive_count_after": int(positive_count_after),
        "fvt_recenter_value_source": "original_fvt",
        "_before_positive_mask": before_positive,
    }


def _fvt_recenter_destination(
    source: tuple[int, int, int],
    vp: np.ndarray,
    vt: np.ndarray,
    target: np.ndarray,
    *,
    max_shift: int,
) -> tuple[tuple[int, int, int], int]:
    if max_shift <= 0:
        return source, 0
    i3, i2, i1 = source
    axis = _dominant_fault_normal_array_axis(float(vp[source]), float(vt[source]))
    current_target = float(target[source])
    best_destination = source
    best_target = current_target
    best_abs_shift = 0
    for offset in range(-max_shift, max_shift + 1):
        if offset == 0:
            continue
        destination = [i3, i2, i1]
        destination[axis] += offset
        if not _inside_shape(tuple(destination), target.shape):
            continue
        destination_tuple = (
            int(destination[0]),
            int(destination[1]),
            int(destination[2]),
        )
        target_value = float(target[destination_tuple])
        abs_shift = abs(offset)
        if target_value > best_target or (
            target_value == best_target and best_abs_shift > 0 and abs_shift < best_abs_shift
        ):
            best_destination = destination_tuple
            best_target = target_value
            best_abs_shift = abs_shift
    if best_target <= current_target:
        return source, 0
    return best_destination, best_abs_shift


def _dominant_fault_normal_array_axis(strike: float, dip: float) -> int:
    p = math.radians(strike)
    t = math.radians(dip)
    components = (
        abs(-math.sin(t) * math.sin(p)),  # i3
        abs(math.sin(t) * math.cos(p)),  # i2
        abs(-math.cos(t)),  # i1
    )
    return int(np.argmax(np.asarray(components, dtype=np.float32)))


def _inside_shape(index: tuple[int, int, int], shape: tuple[int, ...]) -> bool:
    return all(0 <= int(value) < int(size) for value, size in zip(index, shape, strict=True))


def _fvt_recenter_target_distance_diagnostics(
    *,
    before: np.ndarray | None,
    after: np.ndarray | None,
    target: np.ndarray | None,
) -> dict[str, float | None]:
    if before is None or after is None or target is None:
        before_p95 = None
        after_p95 = None
    else:
        before_distance = surface_distance_metrics(
            np.asarray(before, dtype=bool),
            np.asarray(target, dtype=bool),
        )
        after_distance = surface_distance_metrics(
            np.asarray(after, dtype=bool),
            np.asarray(target, dtype=bool),
        )
        before_p95 = before_distance["candidate_to_truth_p95"]
        after_p95 = after_distance["candidate_to_truth_p95"]
    return {
        "fvt_recenter_to_target_distance_p95_before": before_p95,
        "fvt_recenter_to_target_distance_p95_after": after_p95,
    }


def _apply_boundary_edge_thin_v1(
    fvt: np.ndarray,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    voter: OptimalSurfaceVoter,
    target: np.ndarray,
    target_source: str,
    edge_margin: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    fvt_array = np.asarray(fvt, dtype=np.float32)
    fv_array = np.asarray(fv, dtype=np.float32)
    vp_array = np.asarray(vp, dtype=np.float32)
    vt_array = np.asarray(vt, dtype=np.float32)
    target_array = np.asarray(target, dtype=np.float32)
    if fvt_array.ndim != 3:
        raise ValueError("fvt must be a 3D array")
    if (
        fv_array.shape != fvt_array.shape
        or vp_array.shape != fvt_array.shape
        or vt_array.shape != fvt_array.shape
        or target_array.shape != fvt_array.shape
    ):
        raise ValueError("boundary_edge_thin_v1 input shapes must match")
    if not (
        np.all(np.isfinite(fvt_array))
        and np.all(np.isfinite(fv_array))
        and np.all(np.isfinite(vp_array))
        and np.all(np.isfinite(vt_array))
        and np.all(np.isfinite(target_array))
    ):
        raise ValueError("boundary_edge_thin_v1 inputs must contain only finite values")

    edge_mask = _edge_mask(fvt_array.shape, edge_margin)
    before_positive = _positive_candidate_mask(fvt_array)
    edge_positive_before = before_positive & edge_mask
    target_positive = _positive_candidate_mask(target_array)
    target_plateau = voter.thin(
        fv_array,
        vp_array,
        vt_array,
        mode="normal_plateau",
        plateau_tie_breaker=target_array,
    )
    candidate_mask = _positive_candidate_mask(target_plateau) & edge_mask & target_positive
    result = fvt_array.copy()
    collision_count = 0
    adopted_count = 0
    replaced_count = 0

    line_candidates: dict[tuple[int, int, int], tuple[float, float, int, tuple[int, int, int]]] = {}
    for index in np.argwhere(candidate_mask):
        i3, i2, i1 = (int(index[0]), int(index[1]), int(index[2]))
        axis = _dominant_fault_normal_array_axis(
            float(vp_array[i3, i2, i1]), float(vt_array[i3, i2, i1])
        )
        key = _boundary_edge_line_key(axis, i3, i2, i1)
        record = (
            float(target_array[i3, i2, i1]),
            float(fv_array[i3, i2, i1]),
            int(np.ravel_multi_index((i3, i2, i1), fvt_array.shape)),
            (i3, i2, i1),
        )
        existing = line_candidates.get(key)
        if existing is None or _boundary_edge_candidate_precedes(record, existing):
            line_candidates[key] = record

    for key, (_, _, _, destination) in line_candidates.items():
        selector = _boundary_edge_line_selector(key)
        base_line = np.zeros(fvt_array.shape, dtype=bool)
        base_line[selector] = edge_positive_before[selector]
        base_count = int(np.count_nonzero(base_line))
        if base_count > 0:
            collision_count += base_count
        destination_was_positive = bool(before_positive[destination])
        if base_count > 0:
            base_records: list[tuple[float, float, int, tuple[int, int, int]]] = []
            for base_index in np.argwhere(base_line):
                b3, b2, b1 = (
                    int(base_index[0]),
                    int(base_index[1]),
                    int(base_index[2]),
                )
                base_records.append(
                    (
                        float(target_array[b3, b2, b1]),
                        float(fv_array[b3, b2, b1]),
                        int(np.ravel_multi_index((b3, b2, b1), fvt_array.shape)),
                        (b3, b2, b1),
                    )
                )
            best_base = min(base_records, key=lambda item: (-item[0], -item[1], item[2]))
            candidate_record = (
                float(target_array[destination]),
                float(fv_array[destination]),
                int(np.ravel_multi_index(destination, fvt_array.shape)),
                destination,
            )
            if not _boundary_edge_candidate_precedes(candidate_record, best_base):
                continue
            result[base_line] = np.float32(0.0)
            replaced_count += int(
                np.count_nonzero(base_line & ~_single_index_mask(fvt_array.shape, destination))
            )
        if not destination_was_positive:
            adopted_count += 1
        result[destination] = np.float32(fv_array[destination])

    after_positive = _positive_candidate_mask(result)
    distance_before = surface_distance_metrics(before_positive, target_positive)
    distance_after = surface_distance_metrics(after_positive, target_positive)
    diagnostic = {
        "enabled": True,
        "target_source": target_source,
        "edge_margin": int(edge_margin),
        "positive_count_before": int(np.count_nonzero(before_positive)),
        "positive_count_after": int(np.count_nonzero(after_positive)),
        "edge_positive_count_before": int(np.count_nonzero(edge_positive_before)),
        "edge_positive_count_after": int(np.count_nonzero(after_positive & edge_mask)),
        "adopted_candidate_count": int(adopted_count),
        "replaced_candidate_count": int(replaced_count),
        "collision_count": int(collision_count),
        "to_target_distance_p95_before": float(distance_before["candidate_to_truth_p95"]),
        "to_target_distance_p95_after": float(distance_after["candidate_to_truth_p95"]),
    }
    return result.astype(np.float32, copy=False), diagnostic


def _boundary_edge_line_key(axis: int, i3: int, i2: int, i1: int) -> tuple[int, int, int]:
    if axis == 0:
        return (axis, i2, i1)
    if axis == 1:
        return (axis, i3, i1)
    return (axis, i3, i2)


def _boundary_edge_line_selector(key: tuple[int, int, int]) -> tuple[Any, Any, Any]:
    axis, first, second = key
    if axis == 0:
        return (slice(None), first, second)
    if axis == 1:
        return (first, slice(None), second)
    return (first, second, slice(None))


def _boundary_edge_candidate_precedes(
    candidate: tuple[float, float, int, tuple[int, int, int]],
    existing: tuple[float, float, int, tuple[int, int, int]],
) -> bool:
    if candidate[0] != existing[0]:
        return candidate[0] > existing[0]
    if candidate[1] != existing[1]:
        return candidate[1] > existing[1]
    return candidate[2] < existing[2]


def _single_index_mask(shape: tuple[int, int, int], index: tuple[int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[index] = True
    return mask


def _run_voter_thinning_diagnostic(
    *,
    case: Synthetic3DCase,
    voter: OptimalSurfaceVoter,
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    reference_sigma: float,
    truth_metric_config: SyntheticTruthMetricConfig,
    skinning_config: SyntheticSkinningConfig | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    del skinning_config

    truth_surface_half_width = _validate_nonnegative_finite_scalar(
        truth_metric_config.truth_surface_half_width,
        "truth_surface_half_width",
    )
    buffer_radius = _validate_nonnegative_finite_scalar(
        truth_metric_config.buffer_radius,
        "buffer_radius",
    )
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(truth_surface_half_width)
    truth_fault_mask = np.asarray(case.truth_fault_mask, dtype=bool)

    fvt_reference = voter.thin(
        fv,
        vp,
        vt,
        mode="reference",
        reference_sigma=reference_sigma,
    )
    fvt_normal = voter.thin(
        fv,
        vp,
        vt,
        mode="normal",
        reference_sigma=reference_sigma,
    )
    reference_report = _thinning_mode_diagnostic_report(
        fvt_reference,
        vp=vp,
        vt=vt,
        case=case,
        truth_fault_mask=truth_fault_mask,
        truth_surface_mask=truth_surface_mask,
        buffer_radius=buffer_radius,
    )
    normal_report = _thinning_mode_diagnostic_report(
        fvt_normal,
        vp=vp,
        vt=vt,
        case=case,
        truth_fault_mask=truth_fault_mask,
        truth_surface_mask=truth_surface_mask,
        buffer_radius=buffer_radius,
    )
    report = {
        "reference": reference_report,
        "normal": normal_report,
        "delta": {
            "normal_minus_reference": _thinning_quality_delta(
                normal_report["quality"]["fvt_top_truth_count"],
                reference_report["quality"]["fvt_top_truth_count"],
            )
        },
        "keep_mask": _thinning_keep_mask_comparison(
            fvt_reference > 0.0,
            fvt_normal > 0.0,
            truth_fault_mask=truth_fault_mask,
            buffer_radius=buffer_radius,
        ),
    }
    volumes = {
        "fvt_reference_thinning_diagnostic": fvt_reference,
        "fvt_normal_thinning_diagnostic": fvt_normal,
        "keep_reference_thinning_diagnostic": (fvt_reference > 0.0).astype(np.float32),
        "keep_normal_thinning_diagnostic": (fvt_normal > 0.0).astype(np.float32),
        "keep_both_thinning_diagnostic": ((fvt_reference > 0.0) & (fvt_normal > 0.0)).astype(
            np.float32
        ),
        "keep_reference_only_thinning_diagnostic": (
            (fvt_reference > 0.0) & ~(fvt_normal > 0.0)
        ).astype(np.float32),
        "keep_normal_only_thinning_diagnostic": (
            (fvt_normal > 0.0) & ~(fvt_reference > 0.0)
        ).astype(np.float32),
    }
    return report, volumes


def _thinning_mode_diagnostic_report(
    fvt: np.ndarray,
    *,
    vp: np.ndarray,
    vt: np.ndarray,
    case: Synthetic3DCase,
    truth_fault_mask: np.ndarray,
    truth_surface_mask: np.ndarray,
    buffer_radius: float,
) -> dict[str, Any]:
    fvt_top_truth_count = top_truth_count_mask(fvt, truth_surface_mask)
    quality = _top_truth_count_quality(
        fvt_top_truth_count,
        truth_fault_mask=truth_fault_mask,
        truth_surface_mask=truth_surface_mask,
        buffer_radius=buffer_radius,
    )
    quality["orientation_error"] = masked_orientation_error(
        vp,
        vt,
        case.truth_strike,
        case.truth_dip,
        fvt_top_truth_count,
    )
    return {
        "pyosv": {"fvt": _array_summary(fvt)},
        "quality": {"fvt_top_truth_count": quality},
    }


def _thinning_quality_delta(
    normal_quality: Mapping[str, Any],
    reference_quality: Mapping[str, Any],
) -> dict[str, float]:
    return {
        "fvt_buffered_f1_r2": float(
            normal_quality["buffered_overlap_radius2"]["buffered_f1"]
            - reference_quality["buffered_overlap_radius2"]["buffered_f1"]
        ),
        "fvt_candidate_to_truth_p95": float(
            normal_quality["surface_distance"]["candidate_to_truth_p95"]
            - reference_quality["surface_distance"]["candidate_to_truth_p95"]
        ),
        "fvt_strike_median_error": float(
            normal_quality["orientation_error"]["strike_median"]
            - reference_quality["orientation_error"]["strike_median"]
        ),
        "fvt_dip_median_error": float(
            normal_quality["orientation_error"]["dip_median"]
            - reference_quality["orientation_error"]["dip_median"]
        ),
    }


def _thinning_keep_mask_comparison(
    keep_reference: np.ndarray,
    keep_normal: np.ndarray,
    *,
    truth_fault_mask: np.ndarray,
    buffer_radius: float,
) -> dict[str, Any]:
    reference = np.asarray(keep_reference, dtype=bool)
    normal = np.asarray(keep_normal, dtype=bool)
    if reference.shape != normal.shape:
        raise ValueError(f"keep mask shapes must match, got {reference.shape} and {normal.shape}")

    intersection = reference & normal
    union = reference | normal
    reference_only = reference & ~normal
    normal_only = normal & ~reference
    intersection_count = int(np.count_nonzero(intersection))
    union_count = int(np.count_nonzero(union))
    return {
        "reference_count": int(np.count_nonzero(reference)),
        "normal_count": int(np.count_nonzero(normal)),
        "intersection_count": intersection_count,
        "union_count": union_count,
        "reference_only_count": int(np.count_nonzero(reference_only)),
        "normal_only_count": int(np.count_nonzero(normal_only)),
        "jaccard": float(intersection_count / union_count) if union_count else 1.0,
        "reference_only_buffered_overlap_radius2": buffered_surface_overlap(
            reference_only,
            truth_fault_mask,
            radius=buffer_radius,
        ),
        "normal_only_buffered_overlap_radius2": buffered_surface_overlap(
            normal_only,
            truth_fault_mask,
            radius=buffer_radius,
        ),
    }


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
    *,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
    voting_config: SyntheticVotingConfig,
    truth_metric_config: SyntheticTruthMetricConfig,
    skinning_config: SyntheticSkinningConfig,
    variant_spec: VariantSpec,
    include_thinning_diagnostic: bool = False,
    scanner_target_positive_mask: np.ndarray | None = None,
    fvt_recenter_target: np.ndarray | None = None,
    fvt_recenter_target_source: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    voter = OptimalSurfaceVoter(
        ru=voting_config.ru,
        rv=voting_config.rv,
        rw=voting_config.rw,
    )
    voter.set_attribute_smoothing(voting_config.attribute_smoothing)
    voting_patch = variant_spec.voting
    surface_support_min_fraction = (
        voting_config.surface_support_min_fraction
        if voting_patch.support_min_fraction is None
        else voting_patch.support_min_fraction
    )
    surface_support_exponent = (
        voting_config.surface_support_exponent
        if voting_patch.support_exponent is None
        else voting_patch.support_exponent
    )
    voter.set_surface_support_policy(
        min_fraction=surface_support_min_fraction,
        exponent=surface_support_exponent,
    )
    if voting_patch.boundary_policy is not None:
        voter.set_surface_voting_boundary_policy(voting_patch.boundary_policy)
    if voting_patch.orientation_smoothing is not None:
        voter.set_surface_orientation_smoothing(voting_patch.orientation_smoothing)
    if voting_patch.final_normalization_smoothing is not None:
        voter.set_final_normalization_smoothing(voting_patch.final_normalization_smoothing)
    boundary_seed_retention_diagnostic: dict[str, Any] | None = None
    if variant_spec.seed_policy == "boundary_seed_retention_v1":
        boundary_target = ft if fvt_recenter_target is None else fvt_recenter_target
        boundary_target_source = (
            "ft_input" if fvt_recenter_target_source is None else fvt_recenter_target_source
        )
        _, seeds, boundary_seed_retention_diagnostic = _boundary_seed_retention_v1_seeds(
            voting_config=voting_config,
            ft=ft,
            pt=pt,
            tt=tt,
            target=boundary_target,
            target_source=boundary_target_source,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        )
    else:
        seeds = voter.pick_seeds(
            d=voting_config.seed_distance,
            fm=voting_config.seed_threshold,
            ft=ft,
            pt=pt,
            tt=tt,
        )
    fv, vp, vt = voter.apply_voting_from_seeds(
        seeds,
        ft=ft,
        pt=pt,
        tt=tt,
    )
    surface_voting_diagnostic_summary = voter.surface_voting_diagnostic_summary()
    thin_mode = effective_thin_mode(variant_spec, voting_config)
    plateau_tie_breaker = ft if thin_mode in {"hybrid_v2", "normal_plateau"} else None
    fvt = voter.thin(
        fv,
        vp,
        vt,
        mode=thin_mode,
        reference_sigma=voting_config.reference_thin_sigma,
        plateau_tie_breaker=plateau_tie_breaker,
    )
    fvt_recenter_diagnostic: dict[str, Any] | None = None
    boundary_edge_thin_diagnostic: dict[str, Any] | None = None
    if variant_spec.post_thinning_policy == "recenter_scanner_target":
        recenter_target = ft if fvt_recenter_target is None else fvt_recenter_target
        recenter_target_source = (
            "ft_input" if fvt_recenter_target_source is None else fvt_recenter_target_source
        )
        fvt, fvt_recenter_diagnostic = _recenter_edge_fvt_to_target(
            fvt,
            vp,
            vt,
            target=recenter_target,
            target_source=recenter_target_source,
            max_shift=FVT_RECENTER_MAX_SHIFT,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        )
    if variant_spec.post_thinning_policy == "boundary_edge_thin_v1":
        boundary_target = ft if fvt_recenter_target is None else fvt_recenter_target
        boundary_target_source = (
            "ft_input" if fvt_recenter_target_source is None else fvt_recenter_target_source
        )
        fvt, boundary_edge_thin_diagnostic = _apply_boundary_edge_thin_v1(
            fvt,
            fv,
            vp,
            vt,
            voter=voter,
            target=boundary_target,
            target_source=boundary_target_source,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        )

    truth_surface_half_width = _validate_nonnegative_finite_scalar(
        truth_metric_config.truth_surface_half_width,
        "truth_surface_half_width",
    )
    buffer_radius = _validate_nonnegative_finite_scalar(
        truth_metric_config.buffer_radius,
        "buffer_radius",
    )
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(truth_surface_half_width)
    truth_fault_mask = np.asarray(case.truth_fault_mask, dtype=bool)
    fv_top_truth_count = top_truth_count_mask(fv, truth_surface_mask)
    fvt_top_truth_count = top_truth_count_mask(fvt, truth_surface_mask)
    fv_positive_top_truth_count = top_positive_truth_count_mask(fv, truth_surface_mask)
    fvt_positive_top_truth_count = top_positive_truth_count_mask(fvt, truth_surface_mask)
    edge_false_positive_metrics = {
        "fv_top_truth_count": edge_false_positive_ratio(
            fv_top_truth_count,
            truth_surface_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        ),
        "fvt_top_truth_count": edge_false_positive_ratio(
            fvt_top_truth_count,
            truth_surface_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        ),
        "fv_positive_top_truth_count": edge_false_positive_ratio(
            fv_positive_top_truth_count,
            truth_surface_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        ),
        "fvt_positive_top_truth_count": edge_false_positive_ratio(
            fvt_positive_top_truth_count,
            truth_surface_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        ),
    }
    report = {
        "config": {
            "skinning": skinning_config.as_report_dict(),
        },
        "skinning": {"enabled": skinning_config.enabled},
        "pyosv": {
            "fv": _array_summary(fv),
            "fvt": _array_summary(fvt),
            "voting": {
                "surface_voting_boundary_policy": voter.surface_voting_boundary_policy,
                "surface_support_min_fraction": float(surface_support_min_fraction),
                "surface_support_exponent": float(surface_support_exponent),
                "diagnostic_summary": surface_voting_diagnostic_summary,
            },
        },
        "quality": {
            "fv_top_truth_count": {
                "buffered_overlap_radius2": buffered_surface_overlap(
                    fv_top_truth_count,
                    truth_fault_mask,
                    radius=buffer_radius,
                ),
                "surface_distance": surface_distance_metrics(
                    fv_top_truth_count,
                    truth_surface_mask,
                ),
                "orientation_error": masked_orientation_error(
                    vp,
                    vt,
                    case.truth_strike,
                    case.truth_dip,
                    fv_top_truth_count,
                ),
            },
            "fvt_top_truth_count": {
                "buffered_overlap_radius2": buffered_surface_overlap(
                    fvt_top_truth_count,
                    truth_fault_mask,
                    radius=buffer_radius,
                ),
                "surface_distance": surface_distance_metrics(
                    fvt_top_truth_count,
                    truth_surface_mask,
                ),
                "orientation_error": masked_orientation_error(
                    vp,
                    vt,
                    case.truth_strike,
                    case.truth_dip,
                    fvt_top_truth_count,
                ),
            },
            "fv_positive_top_truth_count": {
                "buffered_overlap_radius2": buffered_surface_overlap(
                    fv_positive_top_truth_count,
                    truth_fault_mask,
                    radius=buffer_radius,
                ),
                "surface_distance": surface_distance_metrics(
                    fv_positive_top_truth_count,
                    truth_surface_mask,
                ),
                "orientation_error": masked_orientation_error(
                    vp,
                    vt,
                    case.truth_strike,
                    case.truth_dip,
                    fv_positive_top_truth_count,
                ),
            },
            "fvt_positive_top_truth_count": {
                "buffered_overlap_radius2": buffered_surface_overlap(
                    fvt_positive_top_truth_count,
                    truth_fault_mask,
                    radius=buffer_radius,
                ),
                "surface_distance": surface_distance_metrics(
                    fvt_positive_top_truth_count,
                    truth_surface_mask,
                ),
                "orientation_error": masked_orientation_error(
                    vp,
                    vt,
                    case.truth_strike,
                    case.truth_dip,
                    fvt_positive_top_truth_count,
                ),
            },
            "edge_false_positive": edge_false_positive_metrics,
        },
    }
    if fvt_recenter_diagnostic is not None:
        target_positive = _positive_candidate_mask(recenter_target)
        fvt_recenter_diagnostic.update(
            _fvt_recenter_target_distance_diagnostics(
                before=fvt_recenter_diagnostic.pop("_before_positive_mask"),
                after=_positive_candidate_mask(fvt),
                target=target_positive,
            )
        )
        report["fvt_recenter"] = fvt_recenter_diagnostic
    if boundary_edge_thin_diagnostic is not None:
        report["boundary_edge_thin"] = boundary_edge_thin_diagnostic
    if boundary_seed_retention_diagnostic is not None:
        report["boundary_seed_retention"] = boundary_seed_retention_diagnostic
    diagnostic_volumes: dict[str, np.ndarray] = {}
    if include_thinning_diagnostic:
        thinning_diagnostic, diagnostic_volumes = _run_voter_thinning_diagnostic(
            case=case,
            voter=voter,
            fv=fv,
            vp=vp,
            vt=vt,
            reference_sigma=voting_config.reference_thin_sigma,
            truth_metric_config=truth_metric_config,
            skinning_config=skinning_config,
        )
        report["thinning_diagnostic"] = thinning_diagnostic
    if skinning_config.enabled:
        skin_diagnostics: dict[str, Any] = {}
        skins = _find_synthetic_skins(
            fv,
            fvt,
            vp,
            vt,
            skinning_config=skinning_config,
            diagnostics=skin_diagnostics,
        )
        _add_primary_skin_diagnostics(
            skin_diagnostics,
            skins,
            shape=case.shape,
            fvt_positive_candidate_count=int(np.count_nonzero(fvt_positive_top_truth_count)),
            small_skin_size=skinning_config.small_skin_size,
        )
        _apply_boundary_skinner_fallback(
            skins,
            fvt,
            vp,
            vt,
            skinning_config=skinning_config,
            variant=variant_spec.name,
            diagnostics=skin_diagnostics,
            scanner_target_positive_mask=scanner_target_positive_mask,
        )
        report["skinning"]["diagnostics"] = skin_diagnostics
        skin_metrics = skin_truth_metrics(
            skins,
            shape=case.shape,
            truth_fault_mask=truth_fault_mask,
            truth_surface_mask=truth_surface_mask,
            truth_strike=case.truth_strike,
            truth_dip=case.truth_dip,
            buffer_radius=buffer_radius,
            small_skin_size=skinning_config.small_skin_size,
            truth_fault_id=case.truth_fault_id,
        )
        skin_metrics = _normalize_report_skin_metric_keys(skin_metrics)
        report["pyosv"]["skins"] = skin_metrics["topology"]
        report["quality"]["skin"] = skin_metrics
        skin_mask = skin_mask_from_skins(skins, case.shape)
        report["quality"]["edge_false_positive"]["skin"] = edge_false_positive_ratio(
            skin_mask,
            truth_surface_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
            truth_buffer_radius=buffer_radius,
        )
        skins_output = _skins_json_payload(skins)
    else:
        report["pyosv"]["skins"] = skin_topology_metrics(
            [],
            case.shape,
            small_skin_size=skinning_config.small_skin_size,
        )
        report["quality"]["skin"] = None
        skin_mask = np.zeros(case.shape, dtype=bool)
        skins_output = _disabled_skins_json_payload()

    volumes = {
        "truth_fault_mask": case.truth_fault_mask.astype(np.float32),
        "truth_distance": case.truth_distance,
        "truth_strike": case.truth_strike,
        "truth_dip": case.truth_dip,
        "ft_oracle": case.ft_oracle,
        "pt_oracle": case.pt_oracle,
        "tt_oracle": case.tt_oracle,
        "fv_py": fv,
        "vp_py": vp,
        "vt_py": vt,
        "fvt_py": fvt,
        "skin_mask_py": skin_mask.astype(np.float32),
    }
    volumes.update(diagnostic_volumes)
    return report, volumes, skins_output


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


def _add_primary_skin_diagnostics(
    diagnostics: dict[str, Any],
    skins: Sequence[Any],
    *,
    shape: tuple[int, int, int],
    fvt_positive_candidate_count: int,
    small_skin_size: int,
) -> None:
    topology = skin_topology_metrics(
        skins,
        shape,
        small_skin_size=small_skin_size,
    )
    positive_count = int(fvt_positive_candidate_count)
    if positive_count < 0:
        raise ValueError("fvt_positive_candidate_count must be non-negative")

    unique_cell_count = int(topology["unique_cell_count"])
    largest_size = int(topology["largest_skin_size"])
    cell_coverage = float(unique_cell_count / positive_count) if positive_count else 0.0
    largest_coverage = float(largest_size / positive_count) if positive_count else 0.0
    primary_mask = skin_mask_from_skins(skins, shape)
    primary_edge_shell_fraction = _edge_candidate_fraction(
        primary_mask,
        edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
    )
    reasons = _primary_skin_degraded_reasons(
        fvt_positive_candidate_count=positive_count,
        skin_count=int(topology["skin_count"]),
        cell_coverage_of_fvt_positive=cell_coverage,
        largest_fraction=float(topology["largest_skin_fraction"]),
        small_skin_cell_fraction=float(topology["small_skin_cell_fraction"]),
    )

    diagnostics.update(
        {
            "skin_primary_count": int(topology["skin_count"]),
            "skin_primary_cell_count": int(topology["cell_count"]),
            "skin_primary_unique_cell_count": unique_cell_count,
            "skin_primary_largest_size": largest_size,
            "skin_primary_largest_fraction": float(topology["largest_skin_fraction"]),
            "skin_primary_small_count": int(topology["small_skin_count"]),
            "skin_primary_small_cell_fraction": float(topology["small_skin_cell_fraction"]),
            "skin_primary_cell_coverage_of_fvt_positive": cell_coverage,
            "skin_primary_largest_coverage_of_fvt_positive": largest_coverage,
            "skin_primary_edge_shell_fraction": primary_edge_shell_fraction,
            "skin_primary_degraded_candidate": bool(reasons),
            "skin_primary_degraded_reasons": reasons,
        }
    )


def _primary_skin_degraded_reasons(
    *,
    fvt_positive_candidate_count: int,
    skin_count: int,
    cell_coverage_of_fvt_positive: float,
    largest_fraction: float,
    small_skin_cell_fraction: float,
) -> list[str]:
    if int(fvt_positive_candidate_count) <= 0:
        return []

    reasons: list[str] = []
    if int(skin_count) == 0:
        reasons.append("empty_primary_skin")
    if float(cell_coverage_of_fvt_positive) < SKIN_PRIMARY_DEGRADED_MIN_CELL_COVERAGE:
        reasons.append("low_fvt_positive_coverage")
    if int(skin_count) > 0 and (
        int(skin_count) >= SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_SKIN_COUNT
        or float(largest_fraction) < SKIN_PRIMARY_DEGRADED_FRAGMENTED_MIN_LARGEST_FRACTION
    ):
        reasons.append("fragmented_primary_skins")
    if float(small_skin_cell_fraction) > SKIN_PRIMARY_DEGRADED_MAX_SMALL_CELL_FRACTION:
        reasons.append("high_small_skin_cell_fraction")
    return reasons


def _primary_boundary_degraded_reasons(
    *,
    generic_degraded: bool,
    fvt_positive_candidate_count: int,
    cell_coverage_of_fvt_positive: float,
    fvt_positive_edge_shell_fraction: float,
    primary_edge_shell_fraction: float,
    fvt_to_scanner_target_distance_p95: float | None,
) -> list[str]:
    if not generic_degraded or int(fvt_positive_candidate_count) <= 0:
        return []

    reasons: list[str] = []
    if (
        float(fvt_positive_edge_shell_fraction)
        >= SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_FVT_EDGE_SHELL_FRACTION
    ):
        reasons.append("fvt_positive_edge_shell")
    if (
        fvt_to_scanner_target_distance_p95 is not None
        and float(fvt_to_scanner_target_distance_p95)
        >= SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_SCANNER_TARGET_DISTANCE_P95
    ):
        reasons.append("fvt_far_from_scanner_target")
    if (
        float(cell_coverage_of_fvt_positive) < SKIN_PRIMARY_BOUNDARY_DEGRADED_MAX_CELL_COVERAGE
        and float(fvt_positive_edge_shell_fraction)
        >= SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_EDGE_LOCAL_FVT_FRACTION
        and float(primary_edge_shell_fraction)
        >= SKIN_PRIMARY_BOUNDARY_DEGRADED_MIN_EDGE_LOCAL_PRIMARY_FRACTION
    ):
        reasons.append("low_primary_coverage_with_edge_local_candidates")
    return reasons


def _skeletonized_fallback_boundary_trigger_sufficient(
    *,
    boundary_degraded_reasons: Sequence[str],
    scanner_target_positive_mask: np.ndarray | None,
) -> bool:
    if not boundary_degraded_reasons:
        return False
    if scanner_target_positive_mask is None:
        return False
    return any(
        reason in boundary_degraded_reasons
        for reason in (
            "fvt_far_from_scanner_target",
            "low_primary_coverage_with_edge_local_candidates",
        )
    )


def _fallback_v5_guardrail_report(
    *,
    fallback_topology: Mapping[str, Any],
    fvt_positive_count: int,
    pruned_fraction: float,
) -> dict[str, Any]:
    fallback_skin_count = int(fallback_topology["skin_count"])
    fallback_unique_cell_count = int(fallback_topology["unique_cell_count"])
    coverage_of_fvt_positive = (
        float(fallback_unique_cell_count / int(fvt_positive_count))
        if int(fvt_positive_count) > 0
        else 0.0
    )
    small_skin_cell_fraction = float(fallback_topology["small_skin_cell_fraction"])
    largest_skin_fraction = float(fallback_topology["largest_skin_fraction"])
    pruned_fraction = float(pruned_fraction)

    reasons: list[str] = []
    if fallback_skin_count > SKIN_FALLBACK_V5_MAX_SKIN_COUNT:
        reasons.append("fallback_skin_count_exceeds_max")
    if not math.isfinite(coverage_of_fvt_positive):
        reasons.append("coverage_of_fvt_positive_nonfinite")
    else:
        if coverage_of_fvt_positive < SKIN_FALLBACK_V5_MIN_COVERAGE_OF_FVT_POSITIVE:
            reasons.append("coverage_of_fvt_positive_below_min")
        if coverage_of_fvt_positive > SKIN_FALLBACK_V5_MAX_COVERAGE_OF_FVT_POSITIVE:
            reasons.append("coverage_of_fvt_positive_above_max")
    if (
        not math.isfinite(small_skin_cell_fraction)
        or small_skin_cell_fraction > SKIN_FALLBACK_V5_MAX_SMALL_SKIN_CELL_FRACTION
    ):
        reasons.append("small_skin_cell_fraction_exceeds_max")
    if (
        not math.isfinite(largest_skin_fraction)
        or largest_skin_fraction < SKIN_FALLBACK_V5_MIN_LARGEST_SKIN_FRACTION
    ):
        reasons.append("largest_skin_fraction_below_min")
    if not math.isfinite(pruned_fraction) or pruned_fraction > SKIN_FALLBACK_V5_MAX_PRUNED_FRACTION:
        reasons.append("pruned_fraction_exceeds_max")

    return {
        "enabled": True,
        "passed": not reasons,
        "reasons": reasons,
        "max_skin_count": SKIN_FALLBACK_V5_MAX_SKIN_COUNT,
        "fallback_skin_count": fallback_skin_count,
        "coverage_of_fvt_positive": coverage_of_fvt_positive,
        "min_coverage_of_fvt_positive": SKIN_FALLBACK_V5_MIN_COVERAGE_OF_FVT_POSITIVE,
        "max_coverage_of_fvt_positive": SKIN_FALLBACK_V5_MAX_COVERAGE_OF_FVT_POSITIVE,
        "small_skin_cell_fraction": small_skin_cell_fraction,
        "max_small_skin_cell_fraction": SKIN_FALLBACK_V5_MAX_SMALL_SKIN_CELL_FRACTION,
        "largest_skin_fraction": largest_skin_fraction,
        "min_largest_skin_fraction": SKIN_FALLBACK_V5_MIN_LARGEST_SKIN_FRACTION,
        "pruned_fraction": pruned_fraction,
        "max_pruned_fraction": SKIN_FALLBACK_V5_MAX_PRUNED_FRACTION,
    }


def _fallback_component_diagnostics(
    fvt: np.ndarray,
    *,
    min_skin_size: int | None,
    small_component_size: int,
    connectivity: str,
    component_policy: str = "all",
) -> dict[str, int | float | str]:
    mask = np.asarray(fvt) > np.float32(NONZERO_EPSILON)
    candidate_cell_count = int(np.count_nonzero(mask))
    components = _positive_mask_components(mask, connectivity=connectivity)
    sizes = [len(component) for component in components]
    if component_policy == "all":
        accepted_components = [
            component
            for component in components
            if min_skin_size is None or len(component) >= int(min_skin_size)
        ]
        filter_min_component_size = 0
        filter_min_fraction_of_largest = 0.0
        filter_max_components = 0
    elif component_policy in {
        "degraded_primary_filtered",
        "degraded_primary_skeletonized",
        "degraded_primary_topology_guarded",
    }:
        accepted_components = _filtered_fallback_components(
            components,
            candidate_cell_count=candidate_cell_count,
        )
        filter_min_component_size = _filtered_fallback_min_component_size(candidate_cell_count)
        filter_min_fraction_of_largest = SKIN_FALLBACK_FILTER_MIN_COMPONENT_FRACTION_OF_LARGEST
        filter_max_components = SKIN_FALLBACK_FILTER_MAX_COMPONENTS
    else:
        raise ValueError(f"unknown fallback component policy: {component_policy}")

    accepted_sizes = [len(component) for component in accepted_components]
    discarded_component_count = len(sizes) - len(accepted_sizes)
    small_component_count = sum(1 for size in sizes if size < int(small_component_size))
    largest_component_size = sizes[0] if sizes else 0
    top3_component_cell_count = int(sum(sizes[:3]))
    return {
        "skin_fallback_component_count": int(len(components)),
        "skin_fallback_candidate_cell_count": candidate_cell_count,
        "skin_fallback_largest_component_size": int(largest_component_size),
        "skin_fallback_largest_component_fraction": (
            float(largest_component_size / candidate_cell_count) if candidate_cell_count else 0.0
        ),
        "skin_fallback_top3_component_cell_count": top3_component_cell_count,
        "skin_fallback_top3_component_fraction": (
            float(top3_component_cell_count / candidate_cell_count) if candidate_cell_count else 0.0
        ),
        "skin_fallback_small_component_count": int(small_component_count),
        "skin_fallback_component_policy": component_policy,
        "skin_fallback_accepted_component_count": int(len(accepted_sizes)),
        "skin_fallback_discarded_component_count": int(discarded_component_count),
        "skin_fallback_accepted_component_cell_count": int(sum(accepted_sizes)),
        "skin_fallback_filter_min_component_size": int(filter_min_component_size),
        "skin_fallback_filter_min_component_fraction_of_largest": float(
            filter_min_fraction_of_largest
        ),
        "skin_fallback_filter_max_components": int(filter_max_components),
    }


def _filtered_fallback_min_component_size(candidate_cell_count: int) -> int:
    return max(
        SKIN_FALLBACK_FILTER_MIN_COMPONENT_SIZE_FLOOR,
        int(math.ceil(SKIN_FALLBACK_FILTER_MIN_COMPONENT_FRACTION * int(candidate_cell_count))),
    )


def _filtered_fallback_components(
    components: Sequence[Sequence[tuple[int, int, int]]],
    *,
    candidate_cell_count: int,
) -> list[list[tuple[int, int, int]]]:
    if not components or int(candidate_cell_count) <= 0:
        return []

    largest_component_size = len(components[0])
    min_component_size = _filtered_fallback_min_component_size(candidate_cell_count)
    min_largest_fraction_size = (
        SKIN_FALLBACK_FILTER_MIN_COMPONENT_FRACTION_OF_LARGEST * largest_component_size
    )
    accepted = [
        list(component)
        for component in components
        if len(component) >= min_component_size and len(component) >= min_largest_fraction_size
    ]
    if not accepted:
        accepted = [list(components[0])]
    return accepted[:SKIN_FALLBACK_FILTER_MAX_COMPONENTS]


def _mask_from_components(
    shape: tuple[int, ...],
    components: Sequence[Sequence[tuple[int, int, int]]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for component in components:
        for i3, i2, i1 in component:
            mask[i3, i2, i1] = True
    return mask


def _skeletonize_fallback_components(
    fvt: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    components: Sequence[Sequence[tuple[int, int, int]]],
    *,
    scanner_target_positive_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, int | float | str]]:
    fvt_array = np.asarray(fvt, dtype=np.float32)
    vp_array = np.asarray(vp, dtype=np.float32)
    vt_array = np.asarray(vt, dtype=np.float32)
    if fvt_array.shape != vp_array.shape or fvt_array.shape != vt_array.shape:
        raise ValueError("fvt, vp, and vt shapes must match")
    target_mask = None
    if scanner_target_positive_mask is not None:
        target_mask = np.asarray(scanner_target_positive_mask, dtype=bool)
        if target_mask.shape != fvt_array.shape:
            raise ValueError("scanner_target_positive_mask shape must match fvt")

    retained: set[tuple[int, int, int]] = set()
    axis_counts = [0, 0, 0]
    pruned_component_sizes: list[int] = []
    raw_component_sizes: list[int] = []
    for component in components:
        raw_component_sizes.append(len(component))
        groups: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
        for index in component:
            i3, i2, i1 = index
            axis = _dominant_fault_normal_array_axis(vp_array[i3, i2, i1], vt_array[i3, i2, i1])
            axis_counts[axis] += 1
            groups.setdefault(_skeletonization_line_key(index, axis), []).append(index)

        component_retained: set[tuple[int, int, int]] = set()
        for key, line_indices in groups.items():
            axis = key[0]
            sorted_line = sorted(line_indices, key=lambda item: item[axis])
            run: list[tuple[int, int, int]] = []
            previous_position: int | None = None
            for index in sorted_line:
                position = index[axis]
                if previous_position is None or position == previous_position + 1:
                    run.append(index)
                else:
                    component_retained.add(
                        _select_skeleton_run_sample(
                            run,
                            axis=axis,
                            fvt=fvt_array,
                            scanner_target_positive_mask=target_mask,
                        )
                    )
                    run = [index]
                previous_position = position
            if run:
                component_retained.add(
                    _select_skeleton_run_sample(
                        run,
                        axis=axis,
                        fvt=fvt_array,
                        scanner_target_positive_mask=target_mask,
                    )
                )
        retained.update(component_retained)
        pruned_component_sizes.append(len(component_retained))

    mask = np.zeros(fvt_array.shape, dtype=bool)
    for i3, i2, i1 in retained:
        mask[i3, i2, i1] = True
    raw_count = int(sum(raw_component_sizes))
    pruned_count = int(len(retained))
    return mask, {
        "skin_fallback_pruning_method": "fault_normal_line_collapse",
        "skin_fallback_raw_component_cell_count": raw_count,
        "skin_fallback_pruned_component_cell_count": pruned_count,
        "skin_fallback_pruned_fraction": (float(pruned_count / raw_count) if raw_count else 0.0),
        "skin_fallback_largest_component_size_before_pruning": (
            int(max(raw_component_sizes)) if raw_component_sizes else 0
        ),
        "skin_fallback_largest_component_size_after_pruning": (
            int(max(pruned_component_sizes)) if pruned_component_sizes else 0
        ),
        "skin_fallback_pruning_removed_cell_count": int(raw_count - pruned_count),
        "skin_fallback_skeletonization_axis_mode": _skeletonization_axis_mode(axis_counts),
    }


def _skeletonization_line_key(
    index: tuple[int, int, int],
    axis: int,
) -> tuple[int, int, int]:
    if axis == 0:
        return (axis, index[1], index[2])
    if axis == 1:
        return (axis, index[0], index[2])
    return (axis, index[0], index[1])


def _select_skeleton_run_sample(
    run: Sequence[tuple[int, int, int]],
    *,
    axis: int,
    fvt: np.ndarray,
    scanner_target_positive_mask: np.ndarray | None,
) -> tuple[int, int, int]:
    if not run:
        raise ValueError("run must include at least one sample")
    max_fvt = max(float(fvt[index]) for index in run)
    tied = [index for index in run if float(fvt[index]) == max_fvt]
    if scanner_target_positive_mask is not None:
        max_target = max(int(scanner_target_positive_mask[index]) for index in tied)
        tied = [index for index in tied if int(scanner_target_positive_mask[index]) == max_target]
    center = 0.5 * (run[0][axis] + run[-1][axis])
    return min(tied, key=lambda index: (abs(index[axis] - center), index))


def _skeletonization_axis_mode(axis_counts: Sequence[int]) -> str:
    labels = ("i3", "i2", "i1")
    if not axis_counts or max(axis_counts) == 0:
        return "none"
    max_count = max(axis_counts)
    winners = [labels[index] for index, count in enumerate(axis_counts) if count == max_count]
    return winners[0] if len(winners) == 1 else "mixed"


def _positive_mask_components(
    mask: np.ndarray,
    *,
    connectivity: str,
) -> list[list[tuple[int, int, int]]]:
    mask_array = np.asarray(mask, dtype=bool)
    unvisited = {_index3_tuple(index) for index in np.argwhere(mask_array)}
    offsets = _fallback_connectivity_offsets_i3i2i1(connectivity)
    components: list[list[tuple[int, int, int]]] = []

    while unvisited:
        start = min(unvisited)
        queue: deque[tuple[int, int, int]] = deque([start])
        unvisited.remove(start)
        component: list[tuple[int, int, int]] = []
        while queue:
            i3, i2, i1 = queue.popleft()
            component.append((i3, i2, i1))
            for d3, d2, d1 in offsets:
                neighbor = (i3 + d3, i2 + d2, i1 + d1)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)
        component.sort()
        components.append(component)

    components.sort(key=lambda component: (-len(component), component[0]))
    return components


def _index3_tuple(index: np.ndarray) -> tuple[int, int, int]:
    return (int(index[0]), int(index[1]), int(index[2]))


def _fallback_connectivity_offsets_i3i2i1(
    connectivity: str,
) -> tuple[tuple[int, int, int], ...]:
    max_axis_steps = {
        "face": 1,
        "edge": 2,
        "corner": 3,
    }[connectivity]
    offsets: list[tuple[int, int, int]] = []
    for d3 in (-1, 0, 1):
        for d2 in (-1, 0, 1):
            for d1 in (-1, 0, 1):
                if d3 == 0 and d2 == 0 and d1 == 0:
                    continue
                if abs(d3) + abs(d2) + abs(d1) <= max_axis_steps:
                    offsets.append((d3, d2, d1))
    return tuple(offsets)


def _apply_boundary_skinner_fallback(
    skins: list[Any],
    fvt: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    skinning_config: SyntheticSkinningConfig,
    variant: str,
    diagnostics: dict[str, Any],
    scanner_target_positive_mask: np.ndarray | None = None,
) -> None:
    get_variant_spec(variant)

    fallback_enabled = skinning_config.boundary_skinner_fallback
    fallback_policy = skinning_config.boundary_skinner_fallback_policy
    fallback_connectivity = "edge"
    v5_guardrail_enabled = (
        fallback_enabled and fallback_policy == "degraded_primary_topology_guarded"
    )
    if fallback_policy in {
        "degraded_primary_filtered",
        "degraded_primary_skeletonized",
        "degraded_primary_topology_guarded",
    }:
        component_policy = fallback_policy
    else:
        component_policy = "all"
    component_diagnostics = _fallback_component_diagnostics(
        fvt,
        min_skin_size=skinning_config.min_skin_size,
        small_component_size=skinning_config.small_skin_size,
        connectivity=fallback_connectivity,
        component_policy=component_policy,
    )
    fvt_positive_count = int(component_diagnostics["skin_fallback_candidate_cell_count"])
    primary_skin_count = int(diagnostics.get("skin_primary_count", len(skins)))
    primary_cell_count = int(
        diagnostics.get("skin_primary_cell_count", sum(len(skin) for skin in skins))
    )
    primary_unique_cell_count = int(
        diagnostics.get("skin_primary_unique_cell_count", primary_cell_count)
    )
    coverage_before = (
        float(primary_unique_cell_count / fvt_positive_count) if fvt_positive_count else 0.0
    )
    degraded_reasons = _primary_skin_degraded_reasons(
        fvt_positive_candidate_count=fvt_positive_count,
        skin_count=primary_skin_count,
        cell_coverage_of_fvt_positive=coverage_before,
        largest_fraction=float(diagnostics.get("skin_primary_largest_fraction", 0.0)),
        small_skin_cell_fraction=float(diagnostics.get("skin_primary_small_cell_fraction", 0.0)),
    )
    positive_mask = np.asarray(fvt) > np.float32(NONZERO_EPSILON)
    fvt_positive_edge_shell_fraction = _edge_candidate_fraction(
        positive_mask,
        edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
    )
    primary_edge_shell_fraction = float(diagnostics.get("skin_primary_edge_shell_fraction", 0.0))
    scanner_target_positive_edge_shell_fraction: float | None = None
    fvt_to_scanner_target_distance_p95: float | None = None
    if scanner_target_positive_mask is not None:
        scanner_target_mask = np.asarray(scanner_target_positive_mask, dtype=bool)
        if scanner_target_mask.shape != positive_mask.shape:
            raise ValueError("scanner_target_positive_mask shape must match fvt")
        scanner_target_positive_edge_shell_fraction = _edge_candidate_fraction(
            scanner_target_mask,
            edge_margin=EDGE_FALSE_POSITIVE_MARGIN,
        )
        fvt_to_scanner_target_distance = surface_distance_metrics(
            positive_mask,
            scanner_target_mask,
        )
        fvt_to_scanner_target_distance_p95 = fvt_to_scanner_target_distance[
            "candidate_to_truth_p95"
        ]
    boundary_degraded_reasons = _primary_boundary_degraded_reasons(
        generic_degraded=bool(degraded_reasons),
        fvt_positive_candidate_count=fvt_positive_count,
        cell_coverage_of_fvt_positive=coverage_before,
        fvt_positive_edge_shell_fraction=fvt_positive_edge_shell_fraction,
        primary_edge_shell_fraction=primary_edge_shell_fraction,
        fvt_to_scanner_target_distance_p95=fvt_to_scanner_target_distance_p95,
    )
    diagnostics.update(
        {
            "fallback_enabled": fallback_enabled,
            "fallback_policy": fallback_policy if fallback_enabled else None,
            "fallback_used": False,
            "fallback_reason": None,
            "fallback_method": ("connected_component_on_fvt" if fallback_enabled else None),
            "fallback_input": "fvt" if fallback_enabled else None,
            "fallback_skin_count": 0,
            "fallback_cell_count": 0,
            "fallback_triggered_by_degraded_primary": False,
            "fallback_degraded_reasons": [],
            "fallback_replaced_primary": False,
            "fallback_primary_skin_count": primary_skin_count,
            "fallback_primary_cell_count": primary_cell_count,
            "fallback_candidate_count": fvt_positive_count,
            "fallback_coverage_before": coverage_before,
            "fallback_coverage_after": coverage_before,
            "skin_fvt_positive_edge_shell_fraction": fvt_positive_edge_shell_fraction,
            "skin_primary_edge_shell_fraction": primary_edge_shell_fraction,
            "skin_scanner_target_positive_edge_shell_fraction": (
                scanner_target_positive_edge_shell_fraction
            ),
            "skin_fvt_to_scanner_target_distance_p95": fvt_to_scanner_target_distance_p95,
            "skin_primary_boundary_degraded_candidate": bool(boundary_degraded_reasons),
            "skin_primary_boundary_degraded_reasons": boundary_degraded_reasons,
            "skin_fallback_pruning_method": None,
            "skin_fallback_raw_component_cell_count": 0,
            "skin_fallback_pruned_component_cell_count": 0,
            "skin_fallback_pruned_fraction": 0.0,
            "skin_fallback_largest_component_size_before_pruning": 0,
            "skin_fallback_largest_component_size_after_pruning": 0,
            "skin_fallback_pruning_removed_cell_count": 0,
            "skin_fallback_skeletonization_axis_mode": None,
            "fallback_v5_guardrail": {
                "enabled": v5_guardrail_enabled,
                "passed": False,
                "reasons": [],
                "max_skin_count": SKIN_FALLBACK_V5_MAX_SKIN_COUNT,
                "fallback_skin_count": 0,
                "coverage_of_fvt_positive": 0.0,
                "min_coverage_of_fvt_positive": SKIN_FALLBACK_V5_MIN_COVERAGE_OF_FVT_POSITIVE,
                "max_coverage_of_fvt_positive": SKIN_FALLBACK_V5_MAX_COVERAGE_OF_FVT_POSITIVE,
                "small_skin_cell_fraction": 0.0,
                "max_small_skin_cell_fraction": SKIN_FALLBACK_V5_MAX_SMALL_SKIN_CELL_FRACTION,
                "largest_skin_fraction": 0.0,
                "min_largest_skin_fraction": SKIN_FALLBACK_V5_MIN_LARGEST_SKIN_FRACTION,
                "pruned_fraction": 0.0,
                "max_pruned_fraction": SKIN_FALLBACK_V5_MAX_PRUNED_FRACTION,
            },
            **component_diagnostics,
        }
    )
    if not fallback_enabled:
        return

    if fvt_positive_count == 0:
        diagnostics["fallback_reason"] = "empty_primary_skin_without_positive_fvt"
        return
    if fallback_policy == "degraded_primary_topology_guarded" and primary_skin_count == 0:
        diagnostics["fallback_degraded_reasons"] = degraded_reasons
        diagnostics["fallback_reason"] = "empty_primary_not_supported_by_topology_guarded"
        diagnostics["fallback_v5_guardrail"]["reasons"] = ["empty_primary_not_supported"]
        return
    if fallback_policy == "empty_primary":
        if skins:
            diagnostics["fallback_reason"] = "primary_skin_nonempty"
            return
        fallback_reason = "empty_primary_skin_with_positive_fvt"
    else:
        diagnostics["fallback_degraded_reasons"] = degraded_reasons
        if not degraded_reasons:
            diagnostics["fallback_reason"] = "primary_skin_healthy"
            return
        if not boundary_degraded_reasons:
            diagnostics["fallback_reason"] = "primary_boundary_degraded_not_detected"
            return
        if fallback_policy in {
            "degraded_primary_skeletonized",
            "degraded_primary_topology_guarded",
        } and not _skeletonized_fallback_boundary_trigger_sufficient(
            boundary_degraded_reasons=boundary_degraded_reasons,
            scanner_target_positive_mask=scanner_target_positive_mask,
        ):
            diagnostics["fallback_reason"] = "primary_boundary_degraded_not_sufficient"
            return
        diagnostics["fallback_triggered_by_degraded_primary"] = True
        fallback_reason = "degraded_primary:" + ",".join(
            _fallback_degraded_reason_labels(degraded_reasons)
        )

    fallback_fvt = fvt
    fallback_min_skin_size = skinning_config.min_skin_size
    if component_policy in {
        "degraded_primary_filtered",
        "degraded_primary_skeletonized",
        "degraded_primary_topology_guarded",
    }:
        accepted_components = _filtered_fallback_components(
            _positive_mask_components(positive_mask, connectivity=fallback_connectivity),
            candidate_cell_count=fvt_positive_count,
        )
        accepted_mask = _mask_from_components(fvt.shape, accepted_components)
        if component_policy in {
            "degraded_primary_skeletonized",
            "degraded_primary_topology_guarded",
        }:
            accepted_mask, pruning_diagnostics = _skeletonize_fallback_components(
                fvt,
                vp,
                vt,
                accepted_components,
                scanner_target_positive_mask=scanner_target_positive_mask,
            )
            diagnostics.update(pruning_diagnostics)
        fallback_fvt = np.where(accepted_mask, fvt, np.float32(0.0)).astype(np.float32)
        fallback_min_skin_size = None

    fallback_skins = find_connected_component_skins(
        fallback_fvt,
        vp,
        vt,
        min_likelihood=NONZERO_EPSILON,
        min_skin_size=fallback_min_skin_size,
        connectivity=fallback_connectivity,
    )
    if not fallback_skins:
        diagnostics["fallback_reason"] = "connected_component_fallback_empty"
        return

    fallback_topology = skin_topology_metrics(
        fallback_skins,
        fvt.shape,
        small_skin_size=skinning_config.small_skin_size,
    )
    coverage_after = (
        float(int(fallback_topology["unique_cell_count"]) / fvt_positive_count)
        if fvt_positive_count
        else 0.0
    )
    if fallback_policy == "degraded_primary_topology_guarded":
        guardrail = _fallback_v5_guardrail_report(
            fallback_topology=fallback_topology,
            fvt_positive_count=fvt_positive_count,
            pruned_fraction=(
                float(diagnostics.get("skin_fallback_pruning_removed_cell_count", 0))
                / float(diagnostics.get("skin_fallback_raw_component_cell_count", 0))
                if int(diagnostics.get("skin_fallback_raw_component_cell_count", 0)) > 0
                else 0.0
            ),
        )
        diagnostics["fallback_v5_guardrail"] = guardrail
        if not guardrail["passed"]:
            diagnostics["fallback_reason"] = "fallback_v5_guardrail_failed"
            return

    replaced_primary = bool(skins)
    skins[:] = fallback_skins
    diagnostics.update(
        {
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "fallback_skin_count": int(len(fallback_skins)),
            "fallback_cell_count": int(sum(len(skin) for skin in fallback_skins)),
            "fallback_replaced_primary": replaced_primary,
            "fallback_coverage_after": coverage_after,
        }
    )


def _fallback_degraded_reason_labels(reasons: Sequence[str]) -> list[str]:
    labels = {
        "empty_primary_skin": "empty_primary",
        "low_fvt_positive_coverage": "undercovered",
        "fragmented_primary_skins": "fragmented",
        "high_small_skin_cell_fraction": "small_skin_dominated",
    }
    return [labels.get(reason, reason) for reason in reasons]


def _find_synthetic_skins(
    fv: np.ndarray,
    fvt: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    skinning_config: SyntheticSkinningConfig,
    diagnostics: dict[str, Any] | None = None,
) -> list[Any]:
    skinner_kwargs: dict[str, Any] = {
        "method": skinning_config.method,
        "min_skin_size": skinning_config.min_skin_size,
    }
    if skinning_config.min_likelihood is not None:
        skinner_kwargs["min_likelihood"] = skinning_config.min_likelihood
    skinner = FaultSkinner(**skinner_kwargs)
    grow_volume = fvt
    grow_ft = fvt
    if skinning_config.growth_source == "pre_thin":
        grow_volume = fv
        grow_ft = fv
    return skinner.find_skins(
        grow_volume,
        vp,
        vt,
        min_likelihood=skinning_config.min_likelihood,
        ep=fvt,
        ft=grow_ft,
        pt=vp,
        tt=vt,
        d=skinning_config.d,
        ru=skinning_config.ru,
        rv=skinning_config.rv,
        rw=skinning_config.rw,
        max_steps=skinning_config.max_steps,
        du=skinning_config.du,
        max_delta_strike=skinning_config.max_delta_strike,
        reskin=skinning_config.reskin,
        accepted_occupancy_radius=skinning_config.accepted_occupancy_radius,
        diagnostics=diagnostics,
    )


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
