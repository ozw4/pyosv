r"""Report controlled 3D synthetic truth quality metrics.

Example:
    PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
      --case-set geometry \
      --shape 33,33,33 \
      --variants current_default \
      --output-dir outputs/3d/synthetic_quality/geometry_001 \
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
    make_boundary_plane_case,
    make_crossing_planes_case,
    make_curved_surface_case,
    make_parallel_planes_case,
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
NONZERO_EPSILON = 1.0e-6
VARIANT_NAMES = (
    "current_default",
    "no_surface_orientation_smoothing",
    "final_norm_smoothing_1",
    "voter_thin_normal",
)
DEFAULT_VARIANTS = ("current_default",)
BASELINE_VARIANT = "current_default"
VARIANT_COMPARISON_METRICS = (
    (
        "fvt_buffered_f1_r2_delta_vs_current",
        ("quality", "fvt_top_truth_count", "buffered_overlap_radius2", "buffered_f1"),
    ),
    (
        "fvt_candidate_to_truth_p95_delta_vs_current",
        ("quality", "fvt_top_truth_count", "surface_distance", "candidate_to_truth_p95"),
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
class SyntheticTruthMetricConfig:
    """Configuration for controlled truth metrics."""

    truth_surface_half_width: float = 0.5
    buffer_radius: float = 2.0

    def as_report_dict(self) -> dict[str, float]:
        return {
            "truth_surface_half_width": float(self.truth_surface_half_width),
            "buffer_radius": float(self.buffer_radius),
        }


def _validate_nonnegative_finite_scalar(value: float, name: str) -> float:
    if not np.isscalar(value):
        raise ValueError(f"{name} must be a finite scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run controlled 3D synthetic oracle voting and write quality reports.",
        epilog=(
            "Example:\n"
            "  PYTHONPATH=src python examples/report_3d_synthetic_quality.py \\\n"
            "    --case-set geometry \\\n"
            "    --shape 33,33,33 \\\n"
            "    --variants current_default,no_surface_orientation_smoothing,"
            "final_norm_smoothing_1,voter_thin_normal \\\n"
            "    --output-dir outputs/3d/synthetic_quality/geometry_001 \\\n"
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
        default=DEFAULT_VARIANTS,
        help="Comma-separated diagnostic variants to run; see the example below.",
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
        choices=("reference", "normal"),
        default="reference",
        help="Thinning mode passed to OptimalSurfaceVoter.thin().",
    )
    parser.add_argument(
        "--reference-thin-sigma",
        type=float,
        default=1.0,
        help="Smoothing sigma used by reference-like thinning.",
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
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig(),
    skinning_config: SyntheticSkinningConfig = SyntheticSkinningConfig(),
    variant: str = "current_default",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    case = case_definition.factory(shape)
    if case.case_id != case_definition.case_id:
        raise ValueError(
            f"case factory returned {case.case_id!r}, expected {case_definition.case_id!r}"
        )
    variant_report, volumes, _ = _run_case_variant(
        case,
        voting_config=voting_config,
        truth_metric_config=truth_metric_config,
        skinning_config=skinning_config,
        variant=variant,
    )
    report = {
        "case_id": case.case_id,
        "shape": [int(size) for size in case.shape],
        "truth": _truth_report(case, truth_metric_config),
        "variants": {variant: variant_report},
        "variant_comparison": _variant_comparison({variant: variant_report}),
    }
    if variant == BASELINE_VARIANT:
        report.update(variant_report)
    return report, volumes


def _run_case_variant(
    case: Synthetic3DCase,
    *,
    voting_config: SyntheticVotingConfig,
    truth_metric_config: SyntheticTruthMetricConfig,
    skinning_config: SyntheticSkinningConfig,
    variant: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    if variant not in VARIANT_NAMES:
        raise ValueError(f"unknown variant: {variant}")

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
        ft=case.ft_oracle,
        pt=case.pt_oracle,
        tt=case.tt_oracle,
    )
    thin_mode = "normal" if variant == "voter_thin_normal" else voting_config.voter_thin_mode
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
    voting_config: SyntheticVotingConfig = SyntheticVotingConfig(),
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig(),
    variants: Sequence[str] = DEFAULT_VARIANTS,
    skinning_config: SyntheticSkinningConfig = SyntheticSkinningConfig(),
) -> dict[str, Any]:
    report, _, _ = _build_report_and_volumes(
        case_set=case_set,
        shape=shape,
        voting_config=voting_config,
        truth_metric_config=truth_metric_config,
        variants=variants,
        skinning_config=skinning_config,
    )
    return report


def _build_report_and_volumes(
    *,
    case_set: str,
    shape: tuple[int, int, int],
    voting_config: SyntheticVotingConfig = SyntheticVotingConfig(),
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig(),
    variants: Sequence[str] = DEFAULT_VARIANTS,
    skinning_config: SyntheticSkinningConfig = SyntheticSkinningConfig(),
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, dict[str, Any]]],
]:
    valid_shape = validate_shape3(shape)
    valid_variants = _validate_variants(variants)
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
                truth_metric_config=truth_metric_config,
                skinning_config=skinning_config,
                variant=variant,
            )
            variant_reports[variant] = variant_report
            variant_volumes[variant] = volumes
            variant_skins[variant] = skins_output
        case_report = {
            "case_id": case.case_id,
            "shape": [int(size) for size in case.shape],
            "truth": _truth_report(case, truth_metric_config),
            "variants": variant_reports,
            "variant_comparison": _variant_comparison(variant_reports),
        }
        if BASELINE_VARIANT in variant_reports:
            case_report.update(variant_reports[BASELINE_VARIANT])
        cases.append(case_report)
        volume_outputs[case_definition.case_id] = variant_volumes
        skin_outputs[case_definition.case_id] = variant_skins

    report = {
        "format_version": FORMAT_VERSION,
        "config": {
            "case_set": case_set,
            "shape": [int(size) for size in valid_shape],
            "variants": list(valid_variants),
            "voting": voting_config.as_report_dict(),
            "truth_metrics": truth_metric_config.as_report_dict(),
            "skinning": skinning_config.as_report_dict(),
        },
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
                "variant",
                "baseline_variant",
                "shape_n3",
                "shape_n2",
                "shape_n1",
                "fv_max",
                "fv_mean",
                "fv_nonzero_fraction",
                "fv_buffered_f1_r2",
                "fv_distance_p95",
                "fv_edge_false_positive_fraction",
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
        for case in report["cases"]:
            n3, n2, n1 = case["shape"]
            variant_comparison = case["variant_comparison"]
            baseline_variant = variant_comparison["baseline_variant"]
            comparison_variants = variant_comparison["variants"]
            for variant, variant_report in case["variants"].items():
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
                writer.writerow(
                    {
                        "case_id": case["case_id"],
                        "variant": variant,
                        "baseline_variant": baseline_variant,
                        "shape_n3": n3,
                        "shape_n2": n2,
                        "shape_n1": n1,
                        "fv_max": fv["max"],
                        "fv_mean": fv["mean"],
                        "fv_nonzero_fraction": fv["nonzero_fraction"],
                        "fv_buffered_f1_r2": fv_quality["buffered_overlap_radius2"]["buffered_f1"],
                        "fv_distance_p95": fv_quality["surface_distance"]["candidate_to_truth_p95"],
                        "fv_edge_false_positive_fraction": edge_false_positive[
                            "fv_top_truth_count"
                        ]["edge_false_positive_fraction_of_candidates"],
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
                        **comparison_row,
                    }
                )
    return output_path


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
            for name in VOLUME_NAMES:
                written.append(write_dat(output_dir_for_variant / f"{name}.dat", volumes[name]))
    return written


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
            output_path = output_dir_for_variant / "skins.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(skins_output, sort_keys=True) + "\n", encoding="utf-8"
            )
            written.append(output_path)
    return written


def write_case_figures(
    volume_outputs: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    output_dir: str | PathLike[str],
    *,
    buffer_radius: float = 2.0,
) -> list[Path]:
    from pyosv import viz

    written = []
    output_root = Path(output_dir)
    for case_id, variants in volume_outputs.items():
        case_dir = _case_output_dir(output_root, case_id)
        for variant, volumes in variants.items():
            output_dir_for_variant = _variant_output_dir(case_dir, variant, len(variants) == 1)
            figures_dir = output_dir_for_variant / "figures"
            indices = viz.select_center_slices(np.asarray(volumes["fvt_py"]).shape)
            for axis in ("i3", "i2", "i1"):
                index = indices[axis]
                for name in FIGURE_VOLUME_NAMES:
                    figure_path = figures_dir / f"{name}_{axis}_center.png"
                    written.append(
                        viz.save_slice_panel(
                            figure_path,
                            [(name, viz.slice_2d(volumes[name], axis, index))],
                            title=f"{case_id} {variant} {name} {axis}=center",
                        )
                    )
                if axis == "i3":
                    written.append(
                        viz.save_slice_panel(
                            figures_dir / "skin_mask_py_i3_center.png",
                            [("skin_mask_py", viz.slice_2d(volumes["skin_mask_py"], axis, index))],
                            title=f"{case_id} {variant} skin_mask_py {axis}=center",
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
                        title=f"{case_id} {variant} truth vs fvt {axis}=center",
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
                        title=f"{case_id} {variant} truth vs skin {axis}=center",
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
            quality = variant_report["quality"]["fvt_top_truth_count"]
            overlap = quality["buffered_overlap_radius2"]
            distance = quality["surface_distance"]
            orientation = quality["orientation_error"]
            overlay_parts = [case_id]
            if len(variants) > 1:
                overlay_parts.append(variant)
            overlay_parts.extend(("figures", "truth_vs_fvt_overlay_i3_center.png"))
            overlay_path = PurePosixPath(*overlay_parts)
            skin_overlay_parts = [case_id]
            if len(variants) > 1:
                skin_overlay_parts.append(variant)
            skin_overlay_parts.extend(("figures", "truth_vs_skin_overlay_i3_center.png"))
            skin_overlay_path = PurePosixPath(*skin_overlay_parts)
            lines.extend(
                [
                    f"### {variant}",
                    "",
                    f"- buffered_f1_r2: {_format_markdown_metric(overlap['buffered_f1'])}",
                    "- distance_p95: "
                    f"{_format_markdown_metric(distance['candidate_to_truth_p95'])}",
                    "- strike_median_error: "
                    f"{_format_markdown_metric(orientation['strike_median'])}",
                    f"- dip_median_error: {_format_markdown_metric(orientation['dip_median'])}",
                    "",
                    f"![fvt overlay]({overlay_path.as_posix()})",
                    "",
                ]
            )
            if bool(variant_report["skinning"]["enabled"]):
                skin_quality = variant_report["quality"]["skin"]
                skin_topology = skin_quality["topology"]
                skin_overlap = skin_quality["buffered_overlap_radius2"]
                skin_distance = skin_quality["surface_distance"]
                skin_orientation = skin_quality["orientation_error"]
                lines.extend(
                    [
                        f"- skin_count: {_format_markdown_metric(skin_topology['skin_count'])}",
                        f"- skin_cell_count: {_format_markdown_metric(skin_topology['cell_count'])}",
                        "- skin_buffered_f1_r2: "
                        f"{_format_markdown_metric(skin_overlap['buffered_f1'])}",
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
    return "\n".join(lines).rstrip() + "\n"


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
    voting_config: SyntheticVotingConfig = SyntheticVotingConfig(),
    truth_metric_config: SyntheticTruthMetricConfig = SyntheticTruthMetricConfig(),
    skinning_config: SyntheticSkinningConfig = SyntheticSkinningConfig(),
    variants: Sequence[str] = DEFAULT_VARIANTS,
    pretty: bool = False,
    save_volumes: bool = False,
    save_figures: bool = False,
    write_markdown_index: bool = False,
) -> dict[str, Any]:
    report, volume_outputs, skin_outputs = _build_report_and_volumes(
        case_set=case_set,
        shape=shape,
        voting_config=voting_config,
        truth_metric_config=truth_metric_config,
        skinning_config=skinning_config,
        variants=variants,
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
                voter_thin_mode=args.voter_thin_mode,
                reference_thin_sigma=args.reference_thin_sigma,
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
            variants=args.variants,
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
