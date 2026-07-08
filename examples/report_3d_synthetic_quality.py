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
from dataclasses import dataclass
from os import PathLike
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from pyosv.synthetic3d import (
    Synthetic3DCase,
    SyntheticScannerInputConfig,
    make_boundary_plane_case,
    make_crossing_planes_case,
    make_curved_surface_case,
    make_parallel_planes_case,
    make_scanner_input_from_case,
    make_single_dipping_plane_case,
    make_single_vertical_plane_case,
    make_weak_noisy_plane_case,
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
    top_truth_count_mask,
)
from pyosv.orient3d import FaultOrientScanner3
from pyosv.skinner import FaultSkinner
from pyosv.voting3d import OptimalSurfaceVoter

DEFAULT_SHAPE = (33, 33, 33)
FORMAT_VERSION = 1
EDGE_FALSE_POSITIVE_MARGIN = 2
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
NONZERO_EPSILON = 1.0e-6
VARIANT_NAMES = (
    "current_default",
    "no_surface_orientation_smoothing",
    "final_norm_smoothing_1",
    "voter_thin_normal",
    "voter_thin_hybrid",
)
DEFAULT_VARIANTS = ("current_default",)
QUALITY_MATRIX_VARIANTS = (
    "current_default",
    "no_surface_orientation_smoothing",
    "final_norm_smoothing_1",
    "voter_thin_normal",
    "voter_thin_hybrid",
)
VARIANT_PRESETS = {
    "default": DEFAULT_VARIANTS,
    "quality-matrix": QUALITY_MATRIX_VARIANTS,
}
DEFAULT_VARIANT_PRESET = "default"
BASELINE_VARIANT = "current_default"
DEFAULT_THINNING_DIAGNOSTIC_CASES = ("curved_surface",)
WORKFLOW_MODES = ("reference", "quality", "diagnostic")
VARIANT_COMPARISON_METRICS = (
    (
        "fvt_buffered_f1_r2_delta_vs_current",
        ("quality", "fvt_top_truth_count", "buffered_overlap_radius2", "buffered_f1"),
    ),
    (
        "fvt_candidate_to_truth_p95_delta_vs_current",
        (
            "quality",
            "fvt_top_truth_count",
            "surface_distance",
            "candidate_to_truth_p95",
        ),
    ),
    (
        "fvt_strike_median_error_delta_vs_current",
        ("quality", "fvt_top_truth_count", "orientation_error", "strike_median"),
    ),
    (
        "fvt_dip_median_error_delta_vs_current",
        ("quality", "fvt_top_truth_count", "orientation_error", "dip_median"),
    ),
    (
        "fv_buffered_f1_r2_delta_vs_current",
        ("quality", "fv_top_truth_count", "buffered_overlap_radius2", "buffered_f1"),
    ),
    (
        "skin_buffered_f1_r2_delta_vs_current",
        ("quality", "skin", "buffered_overlap_radius2", "buffered_f1"),
    ),
    (
        "skin_candidate_to_truth_p95_delta_vs_current",
        ("quality", "skin", "surface_distance", "candidate_to_truth_p95"),
    ),
    (
        "skin_strike_median_error_delta_vs_current",
        ("quality", "skin", "orientation_error", "strike_median"),
    ),
    (
        "skin_dip_median_error_delta_vs_current",
        ("quality", "skin", "orientation_error", "dip_median"),
    ),
    (
        "skin_count_delta_vs_current",
        ("quality", "skin", "topology", "skin_count"),
    ),
)


def _validate_workflow_mode(value: str) -> str:
    if value not in WORKFLOW_MODES:
        raise ValueError("workflow_mode must be one of: " + ", ".join(WORKFLOW_MODES))
    return value


def parse_workflow_mode(text: str) -> str:
    try:
        return _validate_workflow_mode(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _default_voter_thin_mode_for_workflow(workflow_mode: str) -> str:
    return "normal" if workflow_mode == "quality" else "reference"


def _effective_voter_thin_mode(
    *,
    workflow_mode: str,
    voter_thin_mode: str | None,
) -> str:
    if voter_thin_mode is not None:
        return voter_thin_mode
    return _default_voter_thin_mode_for_workflow(workflow_mode)


def _effective_include_thinning_diagnostic(
    *,
    workflow_mode: str,
    include_thinning_diagnostic: bool,
) -> bool:
    return include_thinning_diagnostic or workflow_mode == "diagnostic"


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


@dataclass(frozen=True, slots=True)
class SyntheticQualityCaseDefinition:
    """A controlled synthetic report case definition."""

    case_id: str
    factory: Callable[[tuple[int, int, int]], Synthetic3DCase]


@dataclass(frozen=True, slots=True)
class SyntheticVotingConfig:
    """Configuration for synthetic oracle voting."""

    ru: int = 1
    rv: int = 2
    rw: int = 2
    seed_distance: int = 3
    seed_threshold: float = 0.5
    attribute_smoothing: int = 0
    voter_thin_mode: str = "reference"
    reference_thin_sigma: float = 1.0

    def as_report_dict(self) -> dict[str, int | float | str]:
        return {
            "ru": int(self.ru),
            "rv": int(self.rv),
            "rw": int(self.rw),
            "seed_distance": int(self.seed_distance),
            "seed_threshold": float(self.seed_threshold),
            "attribute_smoothing": int(self.attribute_smoothing),
            "voter_thin_mode": self.voter_thin_mode,
            "reference_thin_sigma": float(self.reference_thin_sigma),
        }


@dataclass(frozen=True, slots=True)
class SyntheticScannerConfig:
    """Configuration for scanner-inclusive synthetic report inputs."""

    backend: str = "reference-like"
    phi_min: float = 0.0
    phi_max: float = 180.0
    theta_min: float = 45.0
    theta_max: float = 90.0
    sigma1: float = 2.0
    sigma2: float = 2.0
    scanner_thin_mode: str = "reference"
    remove_edge_effects: bool = True
    input_config: SyntheticScannerInputConfig = SyntheticScannerInputConfig()

    def __post_init__(self) -> None:
        if self.backend not in {"reference-like", "fast"}:
            raise ValueError("scanner_backend must be 'reference-like' or 'fast'")
        if self.scanner_thin_mode not in {"none", "reference", "normal"}:
            raise ValueError("scanner_thin_mode must be 'none', 'reference', or 'normal'")
        if not isinstance(self.remove_edge_effects, bool):
            raise ValueError("remove_edge_effects must be a bool")
        if not isinstance(self.input_config, SyntheticScannerInputConfig):
            raise ValueError("input_config must be a SyntheticScannerInputConfig")
        _validate_finite_scalar(self.phi_min, "scanner_phi_min")
        _validate_finite_scalar(self.phi_max, "scanner_phi_max")
        _validate_finite_scalar(self.theta_min, "scanner_theta_min")
        _validate_finite_scalar(self.theta_max, "scanner_theta_max")
        _validate_positive_finite_scalar(self.sigma1, "scanner_sigma1")
        _validate_positive_finite_scalar(self.sigma2, "scanner_sigma2")
        if self.phi_max < self.phi_min:
            raise ValueError("scanner_phi_max must be greater than or equal to scanner_phi_min")
        if self.theta_max < self.theta_min:
            raise ValueError("scanner_theta_max must be greater than or equal to scanner_theta_min")

    def as_report_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "phi_min": float(self.phi_min),
            "phi_max": float(self.phi_max),
            "theta_min": float(self.theta_min),
            "theta_max": float(self.theta_max),
            "sigma1": float(self.sigma1),
            "sigma2": float(self.sigma2),
            "scanner_thin_mode": self.scanner_thin_mode,
            "remove_edge_effects": self.remove_edge_effects,
            "input": {
                "background": float(self.input_config.background),
                "fault_contrast": float(self.input_config.fault_contrast),
                "noise_sigma": float(self.input_config.noise_sigma),
                "seed": int(self.input_config.seed),
                "clip_min": float(self.input_config.clip_min),
                "clip_max": float(self.input_config.clip_max),
            },
        }


@dataclass(frozen=True, slots=True)
class SyntheticTruthMetricConfig:
    """Configuration for controlled truth metrics."""

    truth_surface_half_width: float = 0.5
    buffer_radius: float = 2.0

    def as_report_dict(self) -> dict[str, float]:
        return {
            "truth_surface_half_width": float(self.truth_surface_half_width),
            "buffer_radius": float(self.buffer_radius),
        }


def _validate_finite_scalar(value: float, name: str) -> float:
    if not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_positive_finite_scalar(value: float, name: str) -> float:
    result = _validate_finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_nonnegative_finite_scalar(value: float, name: str) -> float:
    result = _validate_finite_scalar(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _validate_nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _validate_optional_nonnegative_int(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _validate_nonnegative_int(value, name)


def _validate_skinner_radius(value: int, name: str) -> int:
    result = _validate_nonnegative_int(value, name)
    if result < 2:
        raise ValueError(f"{name} must be at least 2")
    return result


def _validate_optional_skinner_radius(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    return _validate_skinner_radius(value, name)


@dataclass(frozen=True, slots=True)
class SyntheticSkinningConfig:
    """Configuration for controlled synthetic skinning."""

    enabled: bool = True
    min_likelihood: float = 0.5
    min_skin_size: int | None = 1
    d: int = 1
    ru: int = 10
    rv: int | None = None
    rw: int | None = None
    max_steps: int = 10
    du: float = 5.0
    max_delta_strike: float = 30.0
    reskin: bool = True
    small_skin_size: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a bool")
        if not isinstance(self.reskin, bool):
            raise ValueError("reskin must be a bool")
        _validate_nonnegative_finite_scalar(self.min_likelihood, "skinner_min_likelihood")
        _validate_optional_nonnegative_int(self.min_skin_size, "skinner_min_skin_size")
        _validate_nonnegative_int(self.d, "skinner_d")
        _validate_skinner_radius(self.ru, "skinner_ru")
        _validate_optional_skinner_radius(self.rv, "skinner_rv")
        _validate_optional_skinner_radius(self.rw, "skinner_rw")
        _validate_nonnegative_int(self.max_steps, "skinner_max_steps")
        _validate_nonnegative_finite_scalar(self.du, "skinner_du")
        _validate_nonnegative_finite_scalar(
            self.max_delta_strike,
            "skinner_max_delta_strike",
        )
        _validate_nonnegative_int(self.small_skin_size, "small_skin_size")

    def as_report_dict(self) -> dict[str, bool | int | float | None]:
        return {
            "enabled": self.enabled,
            "min_likelihood": float(self.min_likelihood),
            "min_skin_size": (None if self.min_skin_size is None else int(self.min_skin_size)),
            "d": int(self.d),
            "ru": int(self.ru),
            "rv": None if self.rv is None else int(self.rv),
            "rw": None if self.rw is None else int(self.rw),
            "max_steps": int(self.max_steps),
            "du": float(self.du),
            "max_delta_strike": float(self.max_delta_strike),
            "reskin": self.reskin,
            "small_skin_size": int(self.small_skin_size),
        }


MINIMAL_CASES = (
    SyntheticQualityCaseDefinition(
        case_id="single_vertical_plane",
        factory=make_single_vertical_plane_case,
    ),
)
GEOMETRY_CASES = (
    *MINIMAL_CASES,
    SyntheticQualityCaseDefinition(
        case_id="single_dipping_plane",
        factory=make_single_dipping_plane_case,
    ),
    SyntheticQualityCaseDefinition(
        case_id="curved_surface",
        factory=make_curved_surface_case,
    ),
)
EXTENDED_CASES = (
    *GEOMETRY_CASES,
    SyntheticQualityCaseDefinition("parallel_planes", make_parallel_planes_case),
    SyntheticQualityCaseDefinition("crossing_planes", make_crossing_planes_case),
    SyntheticQualityCaseDefinition("boundary_plane", make_boundary_plane_case),
    SyntheticQualityCaseDefinition("weak_noisy_plane", make_weak_noisy_plane_case),
)
CASE_SETS = {
    "minimal": MINIMAL_CASES,
    "geometry": GEOMETRY_CASES,
    "extended": EXTENDED_CASES,
}
CASE_IDS = tuple(definition.case_id for definition in EXTENDED_CASES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run controlled 3D synthetic truth-quality reports.",
        epilog=(
            "Example:\n"
            "  PYTHONPATH=src python examples/report_3d_synthetic_quality.py \\\n"
            "    --case-set extended \\\n"
            "    --shape 33,33,33 \\\n"
            "    --variants current_default,no_surface_orientation_smoothing,"
            "final_norm_smoothing_1,voter_thin_normal,voter_thin_hybrid \\\n"
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
            "quality defaults voter thinning to normal, diagnostic keeps reference "
            "thinning and enables reference-vs-normal diagnostics."
        ),
    )
    parser.add_argument(
        "--scanner-backend",
        choices=("reference-like", "fast"),
        default="reference-like",
        help="FaultOrientScanner3 backend for scanner/both mode: reference-like scan or fast scan.",
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
        choices=("reference", "normal", "hybrid"),
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
        default=0.5,
        help="Minimum thinned vote likelihood for FaultSkinner.",
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
    include_thinning_diagnostic: bool = False,
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
        include_thinning_diagnostic=(
            include_thinning_diagnostic and case.case_id in diagnostic_case_ids
        ),
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
    include_thinning_diagnostic: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    if variant not in VARIANT_NAMES:
        raise ValueError(f"unknown variant: {variant}")
    valid_input_mode = _validate_input_mode(input_mode)

    if valid_input_mode == "oracle":
        return _run_oracle_pipeline(
            case,
            voting_config=voting_config,
            truth_metric_config=truth_metric_config,
            skinning_config=skinning_config,
            variant=variant,
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
            variant=variant,
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
        variant=variant,
        include_thinning_diagnostic=include_thinning_diagnostic,
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
    variant: str,
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
        variant=variant,
        include_thinning_diagnostic=include_thinning_diagnostic,
    )


def _run_scanner_pipeline(
    case: Synthetic3DCase,
    *,
    voting_config: SyntheticVotingConfig,
    scanner_config: SyntheticScannerConfig,
    truth_metric_config: SyntheticTruthMetricConfig,
    skinning_config: SyntheticSkinningConfig,
    variant: str,
    include_thinning_diagnostic: bool,
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
        variant=variant,
        include_thinning_diagnostic=include_thinning_diagnostic,
    )
    report["scanner"] = scanner_report
    report["scanner_quality"] = _scanner_truth_quality(
        case,
        scanner_volumes=scanner_volumes,
        truth_metric_config=truth_metric_config,
    )
    volumes.update(scanner_volumes)
    return report, volumes, skins_output


def _scanner_attributes_from_case(
    case: Synthetic3DCase,
    scanner_config: SyntheticScannerConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    scanner_input = make_scanner_input_from_case(case, scanner_config.input_config)
    scanner = FaultOrientScanner3(scanner_config.sigma1, scanner_config.sigma2)
    if scanner_config.backend == "reference-like":
        ft_scan, pt_scan, tt_scan = scanner.scan(
            scanner_config.phi_min,
            scanner_config.phi_max,
            scanner_config.theta_min,
            scanner_config.theta_max,
            scanner_input,
        )
    elif scanner_config.backend == "fast":
        ft_scan, pt_scan, tt_scan = scanner.scan_fast(
            scanner_config.phi_min,
            scanner_config.phi_max,
            scanner_config.theta_min,
            scanner_config.theta_max,
            scanner_input,
        )
    else:
        raise ValueError("scanner_backend must be 'reference-like' or 'fast'")

    if scanner_config.scanner_thin_mode == "none":
        ft_used = ft_scan
        pt_used = pt_scan
        tt_used = tt_scan
    else:
        ft_used, pt_used, tt_used = scanner.thin(
            ft_scan,
            pt_scan,
            tt_scan,
            mode=scanner_config.scanner_thin_mode,
            remove_edge_effects=scanner_config.remove_edge_effects,
        )

    scanner_report = {
        "config": scanner_config.as_report_dict(),
        "input": _array_summary(scanner_input),
        "ft": _array_summary(ft_scan),
        "fet": _array_summary(ft_used),
        "pt": _array_summary(pt_scan),
        "fpt": _array_summary(pt_used),
        "tt": _array_summary(tt_scan),
        "ftt": _array_summary(tt_used),
    }
    scanner_volumes = {
        "scanner_input": scanner_input,
        "scanner_ft": ft_scan,
        "scanner_fet": ft_used,
        "scanner_pt": pt_scan,
        "scanner_fpt": pt_used,
        "scanner_tt": tt_scan,
        "scanner_ftt": tt_used,
    }
    return scanner_report, scanner_volumes


def _scanner_truth_quality(
    case: Synthetic3DCase,
    *,
    scanner_volumes: Mapping[str, np.ndarray],
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
    far_from_truth_mask = np.abs(case.truth_distance) >= np.float32(
        max(3.0, truth_surface_half_width + 2.0)
    )

    raw_ft_top_truth_count = top_truth_count_mask(
        scanner_volumes["scanner_ft"],
        truth_surface_mask,
    )
    used_ft_top_truth_count = top_truth_count_mask(
        scanner_volumes["scanner_fet"],
        truth_surface_mask,
    )

    return {
        "ft_top_truth_count": _top_truth_count_quality(
            raw_ft_top_truth_count,
            truth_fault_mask=truth_fault_mask,
            truth_surface_mask=truth_surface_mask,
            buffer_radius=buffer_radius,
        ),
        "orientation_error": {
            "raw_scan_top_truth_count": masked_orientation_error(
                scanner_volumes["scanner_pt"],
                scanner_volumes["scanner_tt"],
                case.truth_strike,
                case.truth_dip,
                raw_ft_top_truth_count,
            ),
            "used_attributes_top_truth_count": masked_orientation_error(
                scanner_volumes["scanner_fpt"],
                scanner_volumes["scanner_ftt"],
                case.truth_strike,
                case.truth_dip,
                used_ft_top_truth_count,
            ),
        },
        "input_association": _scanner_input_association(
            scanner_volumes["scanner_input"],
            truth_surface_mask=truth_surface_mask,
            far_from_truth_mask=far_from_truth_mask,
        ),
    }


def _top_truth_count_quality(
    candidate_mask: np.ndarray,
    *,
    truth_fault_mask: np.ndarray,
    truth_surface_mask: np.ndarray,
    buffer_radius: float,
) -> dict[str, Any]:
    return {
        "buffered_overlap_radius2": buffered_surface_overlap(
            candidate_mask,
            truth_fault_mask,
            radius=buffer_radius,
        ),
        "surface_distance": surface_distance_metrics(
            candidate_mask,
            truth_surface_mask,
        ),
    }


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
    input_array = np.asarray(scanner_input, dtype=np.float64)
    truth_mean = _masked_mean(input_array, truth_surface_mask)
    far_mean = _masked_mean(input_array, far_from_truth_mask)
    return {
        "truth_surface_mean": truth_mean,
        "far_from_truth_mean": far_mean,
        "contrast": float(far_mean - truth_mean),
    }


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    sample_mask = np.asarray(mask, dtype=bool)
    if values.shape != sample_mask.shape:
        raise ValueError(f"array shapes must match, got {values.shape} and {sample_mask.shape}")
    if not np.any(sample_mask):
        return 0.0
    samples = values[sample_mask]
    if not np.all(np.isfinite(samples)):
        raise ValueError("masked values must contain only finite values")
    return float(np.mean(samples))


def _run_voting_from_attributes(
    case: Synthetic3DCase,
    *,
    ft: np.ndarray,
    pt: np.ndarray,
    tt: np.ndarray,
    voting_config: SyntheticVotingConfig,
    truth_metric_config: SyntheticTruthMetricConfig,
    skinning_config: SyntheticSkinningConfig,
    variant: str,
    include_thinning_diagnostic: bool = False,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    voter = OptimalSurfaceVoter(
        ru=voting_config.ru,
        rv=voting_config.rv,
        rw=voting_config.rw,
    )
    voter.set_attribute_smoothing(voting_config.attribute_smoothing)
    if variant == "no_surface_orientation_smoothing":
        voter.set_surface_orientation_smoothing(0.0)
    if variant == "final_norm_smoothing_1":
        voter.set_final_normalization_smoothing(1.0)
    fv, vp, vt = voter.apply_voting(
        d=voting_config.seed_distance,
        fm=voting_config.seed_threshold,
        ft=ft,
        pt=pt,
        tt=tt,
    )
    variant_thin_modes = {
        "voter_thin_normal": "normal",
        "voter_thin_hybrid": "hybrid",
    }
    thin_mode = variant_thin_modes.get(variant, voting_config.voter_thin_mode)
    fvt = voter.thin(
        fv,
        vp,
        vt,
        mode=thin_mode,
        reference_sigma=voting_config.reference_thin_sigma,
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
    }
    report = {
        "skinning": {"enabled": skinning_config.enabled},
        "pyosv": {
            "fv": _array_summary(fv),
            "fvt": _array_summary(fvt),
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
            "edge_false_positive": edge_false_positive_metrics,
        },
    }
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
        skins = _find_synthetic_skins(
            fvt,
            vp,
            vt,
            skinning_config=skinning_config,
        )
        skin_metrics = skin_truth_metrics(
            skins,
            shape=case.shape,
            truth_fault_mask=truth_fault_mask,
            truth_surface_mask=truth_surface_mask,
            truth_strike=case.truth_strike,
            truth_dip=case.truth_dip,
            buffer_radius=buffer_radius,
            small_skin_size=skinning_config.small_skin_size,
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


def _truth_report(
    case: Synthetic3DCase,
    truth_metric_config: SyntheticTruthMetricConfig,
) -> dict[str, int]:
    truth_surface_half_width = _validate_nonnegative_finite_scalar(
        truth_metric_config.truth_surface_half_width,
        "truth_surface_half_width",
    )
    truth_surface_mask = np.abs(case.truth_distance) <= np.float32(truth_surface_half_width)
    truth_fault_mask = np.asarray(case.truth_fault_mask, dtype=bool)
    return {
        "fault_voxel_count": int(np.count_nonzero(truth_fault_mask)),
        "surface_voxel_count": int(np.count_nonzero(truth_surface_mask)),
    }


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
    workflow_mode: str = "reference",
    include_thinning_diagnostic: bool = False,
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
        workflow_mode=workflow_mode,
        include_thinning_diagnostic=include_thinning_diagnostic,
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
    workflow_mode: str = "reference",
    include_thinning_diagnostic: bool = False,
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
    valid_workflow_mode = _validate_workflow_mode(workflow_mode)
    if voting_config is None:
        voting_config = SyntheticVotingConfig(
            voter_thin_mode=_default_voter_thin_mode_for_workflow(valid_workflow_mode),
        )
    include_thinning_diagnostic = _effective_include_thinning_diagnostic(
        workflow_mode=valid_workflow_mode,
        include_thinning_diagnostic=include_thinning_diagnostic,
    )
    diagnostic_case_ids = set(_validate_thinning_diagnostic_cases(thinning_diagnostic_cases))
    try:
        case_definitions = CASE_SETS[case_set]
    except KeyError as error:
        raise ValueError(f"unknown case_set: {case_set}") from error

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
                include_thinning_diagnostic=(
                    include_thinning_diagnostic and case.case_id in diagnostic_case_ids
                ),
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
                    if key not in {"pipelines"}
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
    valid_variants = tuple(variants)
    if not valid_variants:
        raise ValueError("variants must include at least one variant")
    unknown = sorted(set(valid_variants).difference(VARIANT_NAMES))
    if unknown:
        raise ValueError(f"unknown variant(s): {','.join(unknown)}")
    duplicates = {variant for variant in valid_variants if valid_variants.count(variant) > 1}
    if duplicates:
        raise ValueError(f"duplicate variant(s): {','.join(sorted(duplicates))}")
    return valid_variants


def _validate_variant_preset(variant_preset: str) -> str:
    if variant_preset not in VARIANT_PRESETS:
        raise ValueError("variant_preset must be one of: " + ", ".join(sorted(VARIANT_PRESETS)))
    return variant_preset


def _resolve_variants(
    *,
    variants: Sequence[str] | None,
    variant_preset: str,
) -> tuple[str, ...]:
    if variants is not None:
        return _validate_variants(variants)
    return _validate_variants(VARIANT_PRESETS[_validate_variant_preset(variant_preset)])


def _validate_thinning_diagnostic_cases(case_ids: Sequence[str]) -> tuple[str, ...]:
    valid_case_ids = tuple(case_ids)
    if not valid_case_ids:
        raise ValueError("thinning_diagnostic_cases must include at least one case ID")
    unknown = sorted(set(valid_case_ids).difference(CASE_IDS))
    if unknown:
        raise ValueError(
            f"unknown thinning diagnostic case ID(s): {','.join(unknown)}; "
            f"choices: {','.join(CASE_IDS)}"
        )
    duplicates = {case_id for case_id in valid_case_ids if valid_case_ids.count(case_id) > 1}
    if duplicates:
        raise ValueError(
            f"duplicate thinning diagnostic case ID(s): {','.join(sorted(duplicates))}"
        )
    return valid_case_ids


def _variant_comparison(
    variant_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if BASELINE_VARIANT not in variant_reports:
        return {"baseline_variant": None, "variants": {}}

    baseline_report = variant_reports[BASELINE_VARIANT]
    baseline_values = {
        metric_name: _metric_value(baseline_report, path)
        for metric_name, path in VARIANT_COMPARISON_METRICS
    }
    comparison = {}
    for variant, variant_report in variant_reports.items():
        comparison[variant] = {
            metric_name: _delta_or_none(
                _metric_value(variant_report, path),
                baseline_values[metric_name],
            )
            for metric_name, path in VARIANT_COMPARISON_METRICS
        }
    return {"baseline_variant": BASELINE_VARIANT, "variants": comparison}


def _metric_value(report: Mapping[str, Any], path: Sequence[str]) -> float | None:
    value: Any = report
    for key in path:
        if value is None:
            return None
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    return float(value)


def _delta_or_none(value: float | None, baseline_value: float | None) -> float | None:
    if value is None or baseline_value is None:
        return None
    return float(value - baseline_value)


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
                "scanner_thin_mode",
                "shape_n3",
                "shape_n2",
                "shape_n1",
                "fv_max",
                "fv_mean",
                "fv_nonzero_fraction",
                "fv_buffered_f1_r2",
                "fv_distance_p95",
                "fv_edge_false_positive_fraction",
                "fv_strike_median_error",
                "fv_dip_median_error",
                "fvt_max",
                "fvt_mean",
                "fvt_nonzero_fraction",
                "fvt_buffered_f1_r2",
                "fvt_distance_p95",
                "fvt_edge_false_positive_fraction",
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
                "skin_buffered_f1_r2",
                "skin_buffered_precision_r2",
                "skin_buffered_recall_r2",
                "skin_distance_p95",
                "skin_distance_candidate_to_truth_p95",
                "skin_distance_truth_to_candidate_p95",
                "skin_distance_hausdorff_p95",
                "skin_strike_median_error",
                "skin_dip_median_error",
                "scanner_ft_buffered_f1_r2",
                "scanner_ft_distance_p95",
                "scanner_strike_median_error",
                "scanner_dip_median_error",
                "scanner_input_contrast",
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
                    edge_false_positive = quality["edge_false_positive"]
                    skinning = variant_report["skinning"]
                    skin_quality = quality["skin"]
                    skin_summary = _summary_csv_skin_row(
                        enabled=bool(skinning["enabled"]),
                        quality=skin_quality,
                    )
                    comparison_row = _summary_csv_comparison_row(
                        comparison_variants.get(variant, {}),
                    )
                    scanner_row = _summary_csv_scanner_row(
                        variant_report=variant_report,
                        input_mode=input_mode,
                    )
                    thinning_diagnostic_row = _summary_csv_thinning_diagnostic_row(
                        variant_report.get("thinning_diagnostic"),
                    )
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
                            "fvt_strike_median_error": fvt_quality["orientation_error"][
                                "strike_median"
                            ],
                            "fvt_dip_median_error": fvt_quality["orientation_error"]["dip_median"],
                            **skin_summary,
                            **scanner_row,
                            **thinning_diagnostic_row,
                            **comparison_row,
                        }
                    )
    return output_path


def _summary_csv_scanner_row(
    *,
    variant_report: Mapping[str, Any],
    input_mode: str,
) -> dict[str, str | float | None]:
    empty_row: dict[str, str | float | None] = {
        "scanner_backend": None,
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
    return {
        "scanner_backend": scanner["config"]["backend"],
        "scanner_thin_mode": scanner["config"]["scanner_thin_mode"],
        "scanner_ft_buffered_f1_r2": ft_quality["buffered_overlap_radius2"]["buffered_f1"],
        "scanner_ft_distance_p95": ft_quality["surface_distance"]["candidate_to_truth_p95"],
        "scanner_strike_median_error": orientation_error["strike_median"],
        "scanner_dip_median_error": orientation_error["dip_median"],
        "scanner_input_contrast": input_association["contrast"],
    }


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
) -> dict[str, bool | int | float | None]:
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
            "skin_buffered_f1_r2": None,
            "skin_buffered_precision_r2": None,
            "skin_buffered_recall_r2": None,
            "skin_distance_p95": None,
            "skin_distance_candidate_to_truth_p95": None,
            "skin_distance_truth_to_candidate_p95": None,
            "skin_distance_hausdorff_p95": None,
            "skin_strike_median_error": None,
            "skin_dip_median_error": None,
        }

    topology = quality["topology"]
    overlap = quality["buffered_overlap_radius2"]
    distance = quality["surface_distance"]
    orientation = quality["orientation_error"]
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
        "skin_buffered_f1_r2": overlap["buffered_f1"],
        "skin_buffered_precision_r2": overlap["buffered_precision"],
        "skin_buffered_recall_r2": overlap["buffered_recall"],
        "skin_distance_p95": distance["candidate_to_truth_p95"],
        "skin_distance_candidate_to_truth_p95": distance["candidate_to_truth_p95"],
        "skin_distance_truth_to_candidate_p95": distance["truth_to_candidate_p95"],
        "skin_distance_hausdorff_p95": distance["hausdorff_p95"],
        "skin_strike_median_error": orientation["strike_median"],
        "skin_dip_median_error": orientation["dip_median"],
    }


def _normalize_report_skin_metric_keys(metrics: Mapping[str, Any]) -> dict[str, Any]:
    report_metrics = dict(metrics)
    if "buffered_overlap_radius2" in report_metrics:
        return report_metrics

    buffered_keys = [
        key for key in report_metrics if str(key).startswith("buffered_overlap_radius")
    ]
    if len(buffered_keys) != 1:
        raise ValueError("skin metrics must include exactly one buffered overlap metric")
    report_metrics["buffered_overlap_radius2"] = report_metrics.pop(buffered_keys[0])
    return report_metrics


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


def _find_synthetic_skins(
    fvt: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    skinning_config: SyntheticSkinningConfig,
) -> list[Any]:
    skinner = FaultSkinner(
        method="reference",
        min_likelihood=skinning_config.min_likelihood,
        min_skin_size=skinning_config.min_skin_size,
    )
    return skinner.find_skins(
        fvt,
        vp,
        vt,
        min_likelihood=skinning_config.min_likelihood,
        d=skinning_config.d,
        ru=skinning_config.ru,
        rv=skinning_config.rv,
        rw=skinning_config.rw,
        max_steps=skinning_config.max_steps,
        du=skinning_config.du,
        max_delta_strike=skinning_config.max_delta_strike,
        reskin=skinning_config.reskin,
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
    workflow_mode: str = "reference",
    pretty: bool = False,
    save_volumes: bool = False,
    save_figures: bool = False,
    write_markdown_index: bool = False,
    include_thinning_diagnostic: bool = False,
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
        workflow_mode=workflow_mode,
        include_thinning_diagnostic=include_thinning_diagnostic,
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
    parser = build_parser()
    args = parser.parse_args(argv)
    voter_thin_mode = _effective_voter_thin_mode(
        workflow_mode=args.workflow_mode,
        voter_thin_mode=args.voter_thin_mode,
    )
    include_thinning_diagnostic = _effective_include_thinning_diagnostic(
        workflow_mode=args.workflow_mode,
        include_thinning_diagnostic=args.include_thinning_diagnostic,
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
            ),
            scanner_config=SyntheticScannerConfig(
                backend=args.scanner_backend,
                phi_min=args.scanner_phi_min,
                phi_max=args.scanner_phi_max,
                theta_min=args.scanner_theta_min,
                theta_max=args.scanner_theta_max,
                sigma1=args.scanner_sigma1,
                sigma2=args.scanner_sigma2,
                scanner_thin_mode=args.scanner_thin_mode,
                remove_edge_effects=not args.keep_scanner_edge_effects,
            ),
            truth_metric_config=SyntheticTruthMetricConfig(
                truth_surface_half_width=args.truth_surface_half_width,
                buffer_radius=args.buffer_radius,
            ),
            skinning_config=SyntheticSkinningConfig(
                enabled=not args.skip_skinning,
                min_likelihood=args.skinner_min_likelihood,
                min_skin_size=args.skinner_min_skin_size,
                d=args.skinner_d,
                ru=args.skinner_ru,
                rv=args.skinner_rv,
                rw=args.skinner_rw,
                max_steps=args.skinner_max_steps,
                du=args.skinner_du,
                max_delta_strike=args.skinner_max_delta_strike,
                reskin=not args.no_skinner_reskin,
                small_skin_size=args.small_skin_size,
            ),
            variants=variants,
            variant_preset=args.variant_preset,
            input_mode=args.input_mode,
            workflow_mode=args.workflow_mode,
            include_thinning_diagnostic=include_thinning_diagnostic,
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
