"""Regional and cross-cell diagnostics for the canonical F3 comparison.

Regions in this module are masks over the one full-volume evaluation unit.
They are not samples, trials, replicates, crops, or independently transformed
datasets.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from pyosv.f3d_reference import interior_slices
from pyosv.metrics import top_percentile_mask

from .data import F3VolumeSource
from .metrics import (
    F3_BUFFERED_PERCENTILE,
    F3_BUFFER_RADIUS,
    F3_PERCENTILES,
    F3_REFERENCE_STAGE_FILES,
    F3_REFERENCE_STAGE_ROLES,
    _validated_extraction_workspace,
)
from .models import F3ModeComparisonPlan
from .runner import F3CellReference

F3_DIAGNOSTIC_SCHEMA_VERSION = 2
F3_REGION_SEMANTICS = "mask_within_full_volume_evaluation_unit"
F3_ORIENTATION_SUPPORT_PERCENTILE = 99.0
F3_DIAGNOSTIC_REGIONS = ("interior", "boundary_shell")
F3_ORIENTATION_PAIRS = (
    ("RL-REF", "RL-QUAL"),
    ("Q-REF", "Q-QUAL"),
    ("RL-REF", "Q-REF"),
    ("RL-QUAL", "Q-QUAL"),
    ("RL-REF", "Q-QUAL"),
)

_CELL_AXES = {
    "RL-REF": ("reference-like", "reference"),
    "RL-QUAL": ("reference-like", "quality"),
    "Q-REF": ("quality", "reference"),
    "Q-QUAL": ("quality", "quality"),
}
_STAGE_FILES = {
    "scanner": ("ft.dat", "pt.dat", "tt.dat"),
    "voting": ("fv.dat", "vp.dat", "vt.dat"),
}
_DAT_DTYPE = np.dtype(">f4")
_REGIONAL_BASIC_CHUNK_VOXELS = 1_048_576
_SHA256_LENGTH = 64


@dataclass(frozen=True, slots=True)
class F3RegionPartition:
    """A complete, disjoint partition of one full-volume array."""

    shape: tuple[int, int, int]
    margin: int
    interior: np.ndarray
    boundary_shell: np.ndarray
    semantics: str = F3_REGION_SEMANTICS

    def __post_init__(self) -> None:
        shape = _shape3(self.shape)
        if isinstance(self.margin, bool) or not isinstance(self.margin, int) or self.margin < 0:
            raise ValueError("margin must be a non-negative integer")
        for name in F3_DIAGNOSTIC_REGIONS:
            mask = np.asarray(getattr(self, name))
            if mask.shape != shape or mask.dtype != np.dtype(bool):
                raise ValueError(f"{name} must be a boolean mask with shape {shape}")
        if np.any(self.interior & self.boundary_shell):
            raise ValueError("interior and boundary_shell must be disjoint")
        if not np.all(self.interior | self.boundary_shell):
            raise ValueError("interior and boundary_shell must cover the full volume")
        if self.semantics != F3_REGION_SEMANTICS:
            raise ValueError(f"semantics must be {F3_REGION_SEMANTICS!r}")

    def mask_for(self, region: str) -> np.ndarray:
        if region not in F3_DIAGNOSTIC_REGIONS:
            raise ValueError(f"unknown diagnostic region: {region!r}")
        return getattr(self, region)

    @property
    def counts(self) -> dict[str, int]:
        return {
            region: int(np.count_nonzero(self.mask_for(region))) for region in F3_DIAGNOSTIC_REGIONS
        }


@dataclass(frozen=True, slots=True)
class RegionalDiagnosticRow:
    """Reference-agreement diagnostics for one regional view of a full volume."""

    schema_version: int
    dataset_id: str
    cell_label: str
    scanner_backend: str
    workflow_mode: str
    stage: str
    source_stage_fingerprint: str
    volume_shape: tuple[int, int, int]
    boundary_margin: int
    region: str
    region_semantics: str
    metrics: Mapping[str, float | int | None]

    def __post_init__(self) -> None:
        if self.schema_version != F3_DIAGNOSTIC_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {F3_DIAGNOSTIC_SCHEMA_VERSION}")
        if _CELL_AXES.get(self.cell_label) != (
            self.scanner_backend,
            self.workflow_mode,
        ):
            raise ValueError("cell label and axes are inconsistent")
        if self.stage not in F3_REFERENCE_STAGE_FILES:
            raise ValueError(f"unknown reference stage: {self.stage!r}")
        _sha256(self.source_stage_fingerprint, "source_stage_fingerprint")
        shape = _shape3(self.volume_shape)
        if (
            isinstance(self.boundary_margin, bool)
            or not isinstance(self.boundary_margin, int)
            or self.boundary_margin < 0
        ):
            raise ValueError("boundary_margin must be a non-negative integer")
        interior_slices(shape, margin=self.boundary_margin)
        object.__setattr__(self, "volume_shape", shape)
        if self.region not in F3_DIAGNOSTIC_REGIONS:
            raise ValueError(f"unknown diagnostic region: {self.region!r}")
        if self.region_semantics != F3_REGION_SEMANTICS:
            raise ValueError(f"region_semantics must be {F3_REGION_SEMANTICS!r}")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "cell_label": self.cell_label,
            "scanner_backend": self.scanner_backend,
            "workflow_mode": self.workflow_mode,
            "stage": self.stage,
            "source_stage_fingerprint": self.source_stage_fingerprint,
            "volume_shape": list(self.volume_shape),
            "boundary_margin": self.boundary_margin,
            "region": self.region,
            "region_semantics": self.region_semantics,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class OrientationDiagnosticRow:
    """Truthless cross-cell orientation disagreement on common ridge support."""

    schema_version: int
    dataset_id: str
    stage: str
    left_cell: str
    right_cell: str
    left_source_stage_fingerprint: str
    right_source_stage_fingerprint: str
    support_contract: str
    support_count: int
    strike_circular_absolute_difference: Mapping[str, float | int | None]
    dip_absolute_difference: Mapping[str, float | int | None]
    normal_vector_angular_difference: Mapping[str, float | int | None]

    def __post_init__(self) -> None:
        if self.schema_version != F3_DIAGNOSTIC_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {F3_DIAGNOSTIC_SCHEMA_VERSION}")
        if (self.left_cell, self.right_cell) not in F3_ORIENTATION_PAIRS:
            raise ValueError("orientation pair is not in the canonical pair contract")
        if self.stage not in _STAGE_FILES:
            raise ValueError("orientation stage must be 'scanner' or 'voting'")
        _sha256(
            self.left_source_stage_fingerprint,
            "left_source_stage_fingerprint",
        )
        _sha256(
            self.right_source_stage_fingerprint,
            "right_source_stage_fingerprint",
        )
        if isinstance(self.support_count, bool) or self.support_count < 0:
            raise ValueError("support_count must be a non-negative integer")
        expected_contract = _orientation_support_contract(self.stage)
        if self.support_contract != expected_contract:
            raise ValueError(f"support_contract must be {expected_contract!r}")
        for name in (
            "strike_circular_absolute_difference",
            "dip_absolute_difference",
            "normal_vector_angular_difference",
        ):
            summary = dict(getattr(self, name))
            if set(summary) != {"count", "mean", "median", "p90", "p95"}:
                raise ValueError(f"{name} has an invalid summary schema")
            if summary["count"] != self.support_count:
                raise ValueError(f"{name} count must equal support_count")
            object.__setattr__(self, name, MappingProxyType(summary))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "stage": self.stage,
            "left_cell": self.left_cell,
            "right_cell": self.right_cell,
            "left_source_stage_fingerprint": self.left_source_stage_fingerprint,
            "right_source_stage_fingerprint": self.right_source_stage_fingerprint,
            "support_contract": self.support_contract,
            "support_count": self.support_count,
            "strike_circular_absolute_difference": dict(self.strike_circular_absolute_difference),
            "dip_absolute_difference": dict(self.dip_absolute_difference),
            "normal_vector_angular_difference": dict(self.normal_vector_angular_difference),
        }


@dataclass(frozen=True, slots=True)
class DiagnosticExtraction:
    """All non-primary diagnostics for one full-volume evaluation unit."""

    dataset_id: str
    evaluation_unit_count: int
    volume_shape: tuple[int, int, int]
    boundary_margin: int
    region_semantics: str
    regional_rows: tuple[RegionalDiagnosticRow, ...]
    orientation_rows: tuple[OrientationDiagnosticRow, ...]

    def __post_init__(self) -> None:
        if self.evaluation_unit_count != 1:
            raise ValueError("F3 diagnostics must describe one evaluation unit")
        shape = _shape3(self.volume_shape)
        if (
            isinstance(self.boundary_margin, bool)
            or not isinstance(self.boundary_margin, int)
            or self.boundary_margin < 0
        ):
            raise ValueError("boundary_margin must be a non-negative integer")
        interior_slices(shape, margin=self.boundary_margin)
        object.__setattr__(self, "volume_shape", shape)
        if self.region_semantics != F3_REGION_SEMANTICS:
            raise ValueError(f"region_semantics must be {F3_REGION_SEMANTICS!r}")
        if any(
            row.dataset_id != self.dataset_id
            or row.volume_shape != shape
            or row.boundary_margin != self.boundary_margin
            or row.region_semantics != self.region_semantics
            for row in self.regional_rows
        ):
            raise ValueError("regional rows must match the extraction identity")

    def as_dict(self) -> dict[str, Any]:
        return {
            "diagnostic_schema_version": F3_DIAGNOSTIC_SCHEMA_VERSION,
            "dataset_id": self.dataset_id,
            "evaluation_unit_count": 1,
            "volume_shape": list(self.volume_shape),
            "boundary_margin": self.boundary_margin,
            "region_semantics": self.region_semantics,
            "regions_are_samples": False,
            "regions_are_trials": False,
            "regions_are_replicates": False,
            "regional_rows": [row.as_dict() for row in self.regional_rows],
            "orientation_rows": [row.as_dict() for row in self.orientation_rows],
            "orientation_interpretation": "cross_cell_disagreement_without_public_truth",
        }


def build_region_partition(shape: tuple[int, int, int], margin: int) -> F3RegionPartition:
    """Partition ``shape`` into interior and boundary-shell masks."""

    valid_shape = _shape3(shape)
    slices = interior_slices(valid_shape, margin=margin)
    interior = np.zeros(valid_shape, dtype=bool)
    interior[slices] = True
    boundary = ~interior
    interior.flags.writeable = False
    boundary.flags.writeable = False
    return F3RegionPartition(valid_shape, margin, interior, boundary)


# Convenient spellings for callers that work directly with masks.
build_region_masks = build_region_partition
regional_masks = build_region_partition


def compute_regional_reference_diagnostics(
    *,
    dataset_id: str,
    cell_label: str,
    scanner_backend: str,
    workflow_mode: str,
    stage: str,
    source_stage_fingerprint: str,
    candidate: np.ndarray,
    reference: np.ndarray,
    margin: int,
    percentiles: Sequence[float] = F3_PERCENTILES,
    buffered_percentile: float = F3_BUFFERED_PERCENTILE,
    buffer_radius: float = F3_BUFFER_RADIUS,
) -> tuple[RegionalDiagnosticRow, ...]:
    """Compute regional views while retaining global ridge/distance semantics."""

    candidate_values, reference_values = _comparable_finite_3d(candidate, reference)
    partition = build_region_partition(candidate_values.shape, margin)
    percentile_values = tuple(float(value) for value in percentiles)
    for percentile in (*percentile_values, float(buffered_percentile)):
        if not 0.0 <= percentile <= 100.0 or not math.isfinite(percentile):
            raise ValueError("percentiles must be finite and in [0, 100]")
    if not math.isfinite(buffer_radius) or buffer_radius < 0.0:
        raise ValueError("buffer_radius must be finite and non-negative")

    metrics_by_region = {
        region: _regional_basic_metrics(
            candidate_values,
            reference_values,
            partition.mask_for(region),
        )
        for region in F3_DIAGNOSTIC_REGIONS
    }
    for percentile in percentile_values:
        reference_mask = top_percentile_mask(reference_values, percentile, positive_only=True)
        candidate_mask = top_percentile_mask(candidate_values, percentile, positive_only=True)
        for region in F3_DIAGNOSTIC_REGIONS:
            metrics_by_region[region].update(
                _regional_overlap_metrics(
                    reference_mask,
                    candidate_mask,
                    partition.mask_for(region),
                    prefix=f"positive_p{_percentile_token(percentile)}",
                )
            )
        reference_mask = None
        candidate_mask = None

    buffered_reference = top_percentile_mask(
        reference_values, float(buffered_percentile), positive_only=True
    )
    buffered_candidate = top_percentile_mask(
        candidate_values, float(buffered_percentile), positive_only=True
    )
    reference_buffer = _dilate_mask(buffered_reference, buffer_radius)
    candidate_buffer = _dilate_mask(buffered_candidate, buffer_radius)
    buffered_prefix = (
        f"positive_p{_percentile_token(float(buffered_percentile))}"
        f"_radius{_number_token(buffer_radius)}"
    )
    distance_prefix = f"positive_p{_percentile_token(float(buffered_percentile))}_distance"
    rows: list[RegionalDiagnosticRow] = []
    try:
        for region in F3_DIAGNOSTIC_REGIONS:
            mask = partition.mask_for(region)
            metrics_by_region[region].update(
                _regional_buffered_metrics(
                    buffered_reference,
                    buffered_candidate,
                    reference_buffer,
                    candidate_buffer,
                    mask,
                    prefix=buffered_prefix,
                )
            )

        have_both_ridge_sets = bool(np.any(buffered_reference) and np.any(buffered_candidate))
        for region in F3_DIAGNOSTIC_REGIONS:
            metrics_by_region[region].update(
                {
                    f"{distance_prefix}_reference_count": int(
                        np.count_nonzero(buffered_reference & partition.mask_for(region))
                    ),
                    f"{distance_prefix}_candidate_count": int(
                        np.count_nonzero(buffered_candidate & partition.mask_for(region))
                    ),
                }
            )

        reference_distance = (
            distance_transform_edt(~buffered_reference) if have_both_ridge_sets else None
        )
        for region in F3_DIAGNOSTIC_REGIONS:
            distances = (
                np.empty(0, dtype=np.float64)
                if reference_distance is None
                else reference_distance[buffered_candidate & partition.mask_for(region)]
            )
            metrics_by_region[region].update(
                _prefixed_summary(
                    f"{distance_prefix}_candidate_to_reference",
                    distances,
                )
            )
        reference_distance = None

        candidate_distance = (
            distance_transform_edt(~buffered_candidate) if have_both_ridge_sets else None
        )
        for region in F3_DIAGNOSTIC_REGIONS:
            distances = (
                np.empty(0, dtype=np.float64)
                if candidate_distance is None
                else candidate_distance[buffered_reference & partition.mask_for(region)]
            )
            metrics_by_region[region].update(
                _prefixed_summary(
                    f"{distance_prefix}_reference_to_candidate",
                    distances,
                )
            )
        candidate_distance = None

        for region in F3_DIAGNOSTIC_REGIONS:
            rows.append(
                RegionalDiagnosticRow(
                    F3_DIAGNOSTIC_SCHEMA_VERSION,
                    dataset_id,
                    cell_label,
                    scanner_backend,
                    workflow_mode,
                    stage,
                    source_stage_fingerprint,
                    partition.shape,
                    partition.margin,
                    region,
                    F3_REGION_SEMANTICS,
                    metrics_by_region[region],
                )
            )
    finally:
        buffered_reference = None
        buffered_candidate = None
        reference_buffer = None
        candidate_buffer = None
    return tuple(rows)


compute_regional_diagnostics = compute_regional_reference_diagnostics


def strike_circular_absolute_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return circular strike disagreement in degrees, wrapped with period 360."""

    left_values, right_values = _comparable_finite(left, right)
    return np.abs((left_values - right_values + 180.0) % 360.0 - 180.0)


circular_strike_difference = strike_circular_absolute_difference


def dip_absolute_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return absolute dip disagreement in degrees."""

    left_values, right_values = _comparable_finite(left, right)
    return np.abs(left_values - right_values)


def normal_vector_angular_difference(
    left_strike: np.ndarray,
    left_dip: np.ndarray,
    right_strike: np.ndarray,
    right_dip: np.ndarray,
) -> np.ndarray:
    """Return axial normal-vector angular disagreement in degrees."""

    ls, ld = _comparable_finite(left_strike, left_dip)
    rs, rd = _comparable_finite(right_strike, right_dip)
    if ls.shape != rs.shape:
        raise ValueError("left and right orientation arrays must share shape")
    left_normal = _normal_vectors(ls, ld)
    right_normal = _normal_vectors(rs, rd)
    cosine = np.sum(left_normal * right_normal, axis=-1, dtype=np.float64)
    # Normals are axes for fault orientation, so opposite normals are equivalent.
    cosine = np.clip(np.abs(cosine), 0.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def compute_orientation_pair_diagnostic(
    *,
    dataset_id: str,
    stage: str,
    left_cell: str,
    right_cell: str,
    left_source_stage_fingerprint: str,
    right_source_stage_fingerprint: str,
    left_likelihood: np.ndarray,
    left_strike: np.ndarray,
    left_dip: np.ndarray,
    right_likelihood: np.ndarray,
    right_strike: np.ndarray,
    right_dip: np.ndarray,
    percentile: float = F3_ORIENTATION_SUPPORT_PERCENTILE,
) -> OrientationDiagnosticRow:
    """Compare two cells on the intersection of their global ridge masks."""

    if percentile != F3_ORIENTATION_SUPPORT_PERCENTILE:
        raise ValueError(
            f"orientation support percentile must be {F3_ORIENTATION_SUPPORT_PERCENTILE}"
        )
    likelihood_left, likelihood_right = _comparable_finite_3d(left_likelihood, right_likelihood)
    for values in (left_strike, left_dip, right_strike, right_dip):
        array = np.asarray(values)
        if array.shape != likelihood_left.shape or not _all_finite(array):
            raise ValueError("orientation fields must be finite and match likelihood shape")
    support = top_percentile_mask(
        likelihood_left, percentile, positive_only=True
    ) & top_percentile_mask(likelihood_right, percentile, positive_only=True)
    count = int(np.count_nonzero(support))
    if count:
        strike = strike_circular_absolute_difference(
            np.asarray(left_strike)[support], np.asarray(right_strike)[support]
        )
        dip = dip_absolute_difference(np.asarray(left_dip)[support], np.asarray(right_dip)[support])
        normal = normal_vector_angular_difference(
            np.asarray(left_strike)[support],
            np.asarray(left_dip)[support],
            np.asarray(right_strike)[support],
            np.asarray(right_dip)[support],
        )
    else:
        strike = dip = normal = np.empty(0, dtype=np.float64)
    return OrientationDiagnosticRow(
        F3_DIAGNOSTIC_SCHEMA_VERSION,
        dataset_id,
        stage,
        left_cell,
        right_cell,
        left_source_stage_fingerprint,
        right_source_stage_fingerprint,
        _orientation_support_contract(stage),
        count,
        _summary(strike),
        _summary(dip),
        _summary(normal),
    )


def compute_orientation_diagnostics(
    *,
    dataset_id: str,
    stage: str,
    source_stage_fingerprints: Mapping[str, str],
    likelihoods: Mapping[str, np.ndarray],
    strikes: Mapping[str, np.ndarray],
    dips: Mapping[str, np.ndarray],
) -> tuple[OrientationDiagnosticRow, ...]:
    """Return all canonical pair rows, including rows with empty support."""

    required = set(_CELL_AXES)
    for name, values in (
        ("source_stage_fingerprints", source_stage_fingerprints),
        ("likelihoods", likelihoods),
        ("strikes", strikes),
        ("dips", dips),
    ):
        if set(values) != required:
            raise ValueError(f"{name} must contain exactly the four canonical cells")
    return tuple(
        compute_orientation_pair_diagnostic(
            dataset_id=dataset_id,
            stage=stage,
            left_cell=left,
            right_cell=right,
            left_source_stage_fingerprint=source_stage_fingerprints[left],
            right_source_stage_fingerprint=source_stage_fingerprints[right],
            left_likelihood=likelihoods[left],
            left_strike=strikes[left],
            left_dip=dips[left],
            right_likelihood=likelihoods[right],
            right_strike=strikes[right],
            right_dip=dips[right],
        )
        for left, right in F3_ORIENTATION_PAIRS
    )


def extract_f3d_diagnostics(
    volume_source: F3VolumeSource,
    cells: Sequence[F3CellReference],
    plan: F3ModeComparisonPlan | None = None,
    *,
    boundary_margin: int | None = None,
) -> DiagnosticExtraction:
    """Extract regional and orientation diagnostics from validated full volumes."""

    if not isinstance(volume_source, F3VolumeSource):
        raise TypeError("volume_source must be an F3VolumeSource")
    if plan is not None:
        if not isinstance(plan, F3ModeComparisonPlan):
            raise TypeError("plan must be an F3ModeComparisonPlan")
        if plan.dataset_spec != volume_source.spec:
            raise ValueError("plan and volume_source dataset specs must match")
        if boundary_margin is not None and boundary_margin != plan.boundary_diagnostic_margin:
            raise ValueError("boundary_margin must match the plan")
        boundary_margin = plan.boundary_diagnostic_margin
    if boundary_margin is None:
        raise ValueError("plan or boundary_margin is required")
    cell_rows = tuple(cells)
    if any(not isinstance(cell, F3CellReference) for cell in cell_rows):
        raise TypeError("cells must contain only F3CellReference values")
    if tuple(cell.label for cell in cell_rows) != tuple(_CELL_AXES):
        raise ValueError("cells must follow canonical F3 cell order")
    workspace = _validated_extraction_workspace(volume_source, cell_rows)
    shape = volume_source.spec.shape
    dataset_id = volume_source.identity.dataset_id

    regional_rows: list[RegionalDiagnosticRow] = []
    for stage, role in F3_REFERENCE_STAGE_ROLES.items():
        reference = volume_source.open_memmap(role)
        try:
            for cell in cell_rows:
                candidate = _open_dat(_candidate_path(workspace, cell, stage), shape)
                try:
                    regional_rows.extend(
                        compute_regional_reference_diagnostics(
                            dataset_id=dataset_id,
                            cell_label=cell.label,
                            scanner_backend=cell.backend,
                            workflow_mode=cell.workflow,
                            stage=stage,
                            source_stage_fingerprint=_cell_stage_fingerprint(cell, stage),
                            candidate=candidate,
                            reference=reference,
                            margin=boundary_margin,
                        )
                    )
                finally:
                    _close_memmap(candidate)
        finally:
            volume_source._close_memmap(reference)

    orientation_rows: list[OrientationDiagnosticRow] = []
    for stage in _STAGE_FILES:
        cells_by_label = {cell.label: cell for cell in cell_rows}
        likelihood_name, strike_name, dip_name = _STAGE_FILES[stage]
        for left_label, right_label in F3_ORIENTATION_PAIRS:
            opened: list[np.memmap] = []
            try:
                fields = []
                for label in (left_label, right_label):
                    path = _orientation_stage_path(workspace, cells_by_label[label], stage)
                    likelihood = _open_dat(path / likelihood_name, shape)
                    strike = _open_dat(path / strike_name, shape)
                    dip = _open_dat(path / dip_name, shape)
                    opened.extend((likelihood, strike, dip))
                    fields.append((likelihood, strike, dip))
                orientation_rows.append(
                    compute_orientation_pair_diagnostic(
                        dataset_id=dataset_id,
                        stage=stage,
                        left_cell=left_label,
                        right_cell=right_label,
                        left_source_stage_fingerprint=_cell_stage_fingerprint(
                            cells_by_label[left_label],
                            stage,
                        ),
                        right_source_stage_fingerprint=_cell_stage_fingerprint(
                            cells_by_label[right_label],
                            stage,
                        ),
                        left_likelihood=fields[0][0],
                        left_strike=fields[0][1],
                        left_dip=fields[0][2],
                        right_likelihood=fields[1][0],
                        right_strike=fields[1][1],
                        right_dip=fields[1][2],
                    )
                )
            finally:
                fields = []
                for array in opened:
                    _close_memmap(array)

    return DiagnosticExtraction(
        dataset_id,
        1,
        shape,
        boundary_margin,
        F3_REGION_SEMANTICS,
        tuple(regional_rows),
        tuple(orientation_rows),
    )


extract_f3d_mode_diagnostics = extract_f3d_diagnostics


def _regional_basic_metrics(
    candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> dict[str, float | int | None]:
    accumulator = _RegionalBasicAccumulator()
    chunks = np.nditer(
        (candidate, reference, mask),
        flags=("buffered", "external_loop", "zerosize_ok"),
        op_flags=(("readonly",), ("readonly",), ("readonly",)),
        op_dtypes=(np.float64, np.float64, np.bool_),
        casting="same_kind",
        order="C",
        buffersize=_REGIONAL_BASIC_CHUNK_VOXELS,
    )
    for candidate_chunk, reference_chunk, mask_chunk in chunks:
        if not np.any(mask_chunk):
            continue
        if np.all(mask_chunk):
            accumulator.add(candidate_chunk, reference_chunk)
        else:
            accumulator.add(candidate_chunk[mask_chunk], reference_chunk[mask_chunk])
    return accumulator.metrics()


class _RegionalBasicAccumulator:
    """Bounded-memory basic statistics for one region mask."""

    def __init__(self) -> None:
        self.count = 0
        self.candidate_nonzero = 0
        self.reference_nonzero = 0
        self.candidate_mean = 0.0
        self.reference_mean = 0.0
        self.candidate_m2 = 0.0
        self.reference_m2 = 0.0
        self.covariance = 0.0
        self.absolute_difference_parts: list[float] = []
        self.squared_difference_parts: list[float] = []

    def add(self, candidate: np.ndarray, reference: np.ndarray) -> None:
        count = int(candidate.size)
        if count == 0:
            return
        self.candidate_nonzero += int(np.count_nonzero(candidate))
        self.reference_nonzero += int(np.count_nonzero(reference))

        difference = candidate - reference
        self.absolute_difference_parts.append(float(np.sum(np.abs(difference), dtype=np.float64)))
        self.squared_difference_parts.append(float(np.dot(difference, difference)))
        del difference

        candidate_mean = float(np.mean(candidate, dtype=np.float64))
        reference_mean = float(np.mean(reference, dtype=np.float64))
        candidate_centered = candidate - candidate_mean
        reference_centered = reference - reference_mean
        candidate_m2 = float(np.dot(candidate_centered, candidate_centered))
        reference_m2 = float(np.dot(reference_centered, reference_centered))
        covariance = float(np.dot(candidate_centered, reference_centered))

        if self.count:
            combined = self.count + count
            candidate_delta = candidate_mean - self.candidate_mean
            reference_delta = reference_mean - self.reference_mean
            factor = self.count * count / combined
            self.candidate_m2 += candidate_m2 + candidate_delta * candidate_delta * factor
            self.reference_m2 += reference_m2 + reference_delta * reference_delta * factor
            self.covariance += covariance + candidate_delta * reference_delta * factor
            self.candidate_mean += candidate_delta * count / combined
            self.reference_mean += reference_delta * count / combined
            self.count = combined
        else:
            self.count = count
            self.candidate_mean = candidate_mean
            self.reference_mean = reference_mean
            self.candidate_m2 = candidate_m2
            self.reference_m2 = reference_m2
            self.covariance = covariance

    def metrics(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "voxel_count": 0,
                "candidate_nonzero_fraction": 0.0,
                "reference_nonzero_fraction": 0.0,
                "normalized_correlation": None,
                "mean_absolute_difference": None,
                "root_mean_square_difference": None,
            }
        denominator = math.sqrt(max(0.0, self.candidate_m2) * max(0.0, self.reference_m2))
        correlation = 0.0 if denominator == 0.0 else self.covariance / denominator
        absolute_difference_sum = math.fsum(self.absolute_difference_parts)
        squared_difference_sum = math.fsum(self.squared_difference_parts)
        return {
            "voxel_count": self.count,
            "candidate_nonzero_fraction": self.candidate_nonzero / self.count,
            "reference_nonzero_fraction": self.reference_nonzero / self.count,
            "normalized_correlation": correlation,
            "mean_absolute_difference": absolute_difference_sum / self.count,
            "root_mean_square_difference": math.sqrt(squared_difference_sum / self.count),
        }


def _regional_overlap_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    region: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float | int]:
    reference_count = int(np.count_nonzero(reference & region))
    candidate_count = int(np.count_nonzero(candidate & region))
    intersection = int(np.count_nonzero(reference & candidate & region))
    union = int(np.count_nonzero((reference | candidate) & region))
    precision = _ratio(intersection, candidate_count)
    recall = _ratio(intersection, reference_count)
    return {
        f"{prefix}_reference_count": reference_count,
        f"{prefix}_candidate_count": candidate_count,
        f"{prefix}_intersection_count": intersection,
        f"{prefix}_union_count": union,
        f"{prefix}_precision": precision,
        f"{prefix}_recall": recall,
        f"{prefix}_f1": _f1(precision, recall),
        f"{prefix}_jaccard": _ratio(intersection, union),
    }


def _regional_buffered_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    reference_buffer: np.ndarray,
    candidate_buffer: np.ndarray,
    region: np.ndarray,
    *,
    prefix: str,
) -> dict[str, float | int]:
    result = _regional_overlap_metrics(reference, candidate, region, prefix=prefix)
    candidate_count = int(result[f"{prefix}_candidate_count"])
    reference_count = int(result[f"{prefix}_reference_count"])
    candidate_in_buffer = int(np.count_nonzero(candidate & reference_buffer & region))
    reference_in_buffer = int(np.count_nonzero(reference & candidate_buffer & region))
    precision = _ratio(candidate_in_buffer, candidate_count)
    recall = _ratio(reference_in_buffer, reference_count)
    result.update(
        {
            f"{prefix}_candidate_in_reference_buffer_count": candidate_in_buffer,
            f"{prefix}_reference_in_candidate_buffer_count": reference_in_buffer,
            f"{prefix}_buffered_precision": precision,
            f"{prefix}_buffered_recall": recall,
            f"{prefix}_buffered_f1": _f1(precision, recall),
        }
    )
    return result


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90.0)),
        "p95": float(np.percentile(array, 95.0)),
    }


def _prefixed_summary(prefix: str, values: np.ndarray) -> dict[str, float | int | None]:
    summary = _summary(values)
    return {f"{prefix}_{name}": value for name, value in summary.items() if name != "count"}


def _normal_vectors(strike: np.ndarray, dip: np.ndarray) -> np.ndarray:
    phi = np.deg2rad(np.asarray(strike, dtype=np.float64))
    theta = np.deg2rad(np.asarray(dip, dtype=np.float64))
    cosine = np.cos(theta)
    sine = np.sin(theta)
    return np.stack(
        (-cosine, sine * np.cos(phi), -sine * np.sin(phi)),
        axis=-1,
    )


def _orientation_support_contract(stage: str) -> str:
    try:
        likelihood = {"scanner": "scanner_likelihood", "voting": "voting_likelihood"}[stage]
    except KeyError as error:
        raise ValueError("orientation stage must be 'scanner' or 'voting'") from error
    return f"intersection_of_global_positive_p99_{likelihood}_masks"


def _dilate_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    if radius == 0.0 or not np.any(mask):
        return mask.copy()
    samples = int(math.ceil(radius))
    axes = np.ogrid[(slice(-samples, samples + 1),) * mask.ndim]
    distance_squared = np.zeros((2 * samples + 1,) * mask.ndim, dtype=np.float64)
    for axis in axes:
        distance_squared += axis.astype(np.float64) ** 2
    return binary_dilation(mask, structure=distance_squared <= radius * radius)


def _shape3(shape: tuple[int, ...]) -> tuple[int, int, int]:
    if (
        not isinstance(shape, tuple)
        or len(shape) != 3
        or any(isinstance(size, bool) or not isinstance(size, int) or size <= 0 for size in shape)
    ):
        raise ValueError("shape must contain exactly three positive integers")
    return shape


def _comparable_finite(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    if left_values.shape != right_values.shape:
        raise ValueError("arrays must share shape")
    if not _all_finite(left_values) or not _all_finite(right_values):
        raise ValueError("arrays must contain only finite values")
    return left_values, right_values


def _all_finite(values: np.ndarray) -> bool:
    chunks = np.nditer(
        values,
        flags=("buffered", "external_loop", "zerosize_ok"),
        op_flags=("readonly",),
        order="K",
        buffersize=1_048_576,
    )
    return all(bool(np.all(np.isfinite(chunk))) for chunk in chunks)


def _comparable_finite_3d(left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_values, right_values = _comparable_finite(left, right)
    _shape3(left_values.shape)
    return left_values, right_values


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def _percentile_token(value: float) -> str:
    return str(value).replace(".", "_").removesuffix("_0")


def _number_token(value: float) -> str:
    return str(float(value)).replace(".", "_").removesuffix("_0")


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _cell_stage_fingerprint(cell: F3CellReference, stage: str) -> str:
    return {
        "ft": cell.stages.scanner,
        "fv": cell.stages.voting,
        "fvt": cell.stages.thinning,
        "scanner": cell.stages.scanner,
        "voting": cell.stages.voting,
    }[stage]


def _candidate_path(workspace: Path, cell: F3CellReference, stage: str) -> Path:
    if stage == "ft":
        return workspace / "stages" / "scanner" / cell.stages.scanner / "ft.dat"
    if stage == "fv":
        return workspace / "stages" / "voting" / cell.stages.voting / "fv.dat"
    if stage == "fvt":
        return workspace / "stages" / "thinning" / cell.stages.thinning / "fvt.dat"
    raise ValueError(f"unknown reference stage: {stage!r}")


def _orientation_stage_path(workspace: Path, cell: F3CellReference, stage: str) -> Path:
    if stage == "scanner":
        return workspace / "stages" / "scanner" / cell.stages.scanner
    if stage == "voting":
        return workspace / "stages" / "voting" / cell.stages.voting
    raise ValueError("orientation stage must be 'scanner' or 'voting'")


def _open_dat(path: Path, shape: tuple[int, int, int]) -> np.memmap:
    expected_bytes = int(np.prod(shape)) * _DAT_DTYPE.itemsize
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError(f"invalid DAT artifact: {path}")
    array = np.memmap(path, dtype=_DAT_DTYPE, mode="r", shape=shape, order="C")
    array.flags.writeable = False
    return array


def _close_memmap(array: np.memmap) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


__all__ = [
    "DiagnosticExtraction",
    "F3_DIAGNOSTIC_REGIONS",
    "F3_DIAGNOSTIC_SCHEMA_VERSION",
    "F3_ORIENTATION_PAIRS",
    "F3_ORIENTATION_SUPPORT_PERCENTILE",
    "F3_REGION_SEMANTICS",
    "F3RegionPartition",
    "OrientationDiagnosticRow",
    "RegionalDiagnosticRow",
    "build_region_masks",
    "build_region_partition",
    "circular_strike_difference",
    "compute_orientation_diagnostics",
    "compute_orientation_pair_diagnostic",
    "compute_regional_diagnostics",
    "compute_regional_reference_diagnostics",
    "dip_absolute_difference",
    "extract_f3d_diagnostics",
    "extract_f3d_mode_diagnostics",
    "normal_vector_angular_difference",
    "regional_masks",
    "strike_circular_absolute_difference",
]
