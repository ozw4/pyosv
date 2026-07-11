"""3D optimal-surface voting entry points."""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence

import numpy as np

from pyosv._accel import NUMBA_AVAILABLE, njit
from pyosv._voting3d.models import (
    _MaskedUVWBoxSamples,
    _SurfaceVotingDiagnostic,
    _TangentialRectangle,
)
from pyosv._voting3d.uvw_numba import (
    _samples_in_uvw_box_masked_numba,
    _samples_in_uvw_box_numba,
)
from pyosv._voting3d.uvw_python import (
    _samples_in_uvw_box_masked_python,
    _samples_in_uvw_box_python,
)
from pyosv._voting3d.uvw_sampling import (
    _crop_masked_uvw_box,
    _select_supported_origin_rectangle,
    _surface_center_lag,
    _validate_uvw_sampling_origin,
)
from pyosv._voting3d.validation import (
    _validate_finite_array2,
    _validate_finite_array3,
    _validate_finite_vector3,
    _validate_fraction_float,
    _validate_int,
    _validate_matching_arrays3,
    _validate_matching_finite_arrays3_many,
    _validate_nonnegative_float,
    _validate_nonnegative_int,
    _validate_positive_int,
    _validate_vector3,
)
from pyosv.cells import FaultCell
from pyosv.dp import (
    _find_surface_3d_masked,
    find_surface_3d,
    shift_range,
    smooth_surface_2d,
    strain_to_bstrain,
    update_shift_ranges_3d,
)
from pyosv.filters import smooth3d
from pyosv.geometry import strike_and_dip_from_local_surface_derivatives
from pyosv.interp import sample3
from pyosv.thinning3d import reference_like_3d_thin_values

__all__ = ["OptimalSurfaceVoter"]


_SURFACE_VOTING_BOUNDARY_POLICIES = ("reference", "masked_in_bounds")


class OptimalSurfaceVoter:
    """Configuration and state holder for 3D optimal-surface voting."""

    def __init__(self, ru: int, rv: int, rw: int) -> None:
        self.ru = _validate_nonnegative_int(ru, "ru")
        self.rv = _validate_nonnegative_int(rv, "rv")
        self.rw = _validate_nonnegative_int(rw, "rw")
        self.lmin, self.lmax, self.nl = shift_range(self.ru)
        self.bstrain1 = 4
        self.bstrain2 = 4
        self.attribute_smoothing = 1
        self.surface_smoothing1 = 2.0
        self.surface_smoothing2 = 2.0
        self.surface_orientation_smoothing = float(max(self.rv, self.rw))
        self.final_normalization_smoothing = 0.0
        self.surface_support_min_fraction = 0.0
        self.surface_support_exponent = 0.0
        self.surface_voting_boundary_policy = "reference"
        self._last_surface_voting_policy = "reference"
        self._last_surface_voting_diagnostics: tuple[_SurfaceVotingDiagnostic, ...] = ()
        self.lmins: np.ndarray
        self.lmaxs: np.ndarray
        self._update_shift_ranges()

    def set_strain_max(self, strain_max1: float, strain_max2: float) -> None:
        """Set the maximum fault-surface strains in the first two dimensions."""

        bstrain1 = strain_to_bstrain(strain_max1)
        bstrain2 = strain_to_bstrain(strain_max2)
        self.bstrain1 = bstrain1
        self.bstrain2 = bstrain2

    def set_attribute_smoothing(self, attribute_smoothing: int) -> None:
        """Set the number of nonlinear smoothings for fault attributes."""

        self.attribute_smoothing = _validate_nonnegative_int(
            attribute_smoothing,
            "attribute_smoothing",
        )

    def set_surface_smoothing(
        self,
        surface_smoothing1: float,
        surface_smoothing2: float,
    ) -> None:
        """Set the smoothing extents used for extracted fault surfaces."""

        smoothing1 = _validate_nonnegative_float(
            surface_smoothing1,
            "surface_smoothing1",
        )
        smoothing2 = _validate_nonnegative_float(
            surface_smoothing2,
            "surface_smoothing2",
        )
        self.surface_smoothing1 = smoothing1
        self.surface_smoothing2 = smoothing2

    def set_surface_orientation_smoothing(
        self,
        surface_orientation_smoothing: float,
    ) -> None:
        """Set smoothing for surface orientation re-estimation."""

        self.surface_orientation_smoothing = _validate_nonnegative_float(
            surface_orientation_smoothing,
            "surface_orientation_smoothing",
        )

    def set_final_normalization_smoothing(self, sigma: float) -> None:
        """Set smoothing for final vote map normalization before power transform."""

        self.final_normalization_smoothing = _validate_nonnegative_float(
            sigma,
            "final_normalization_smoothing",
        )

    def set_surface_support_policy(
        self,
        min_fraction: float = 0.0,
        exponent: float = 0.0,
    ) -> None:
        """Set support-aware skip/down-weighting for extracted surface votes."""

        self.surface_support_min_fraction = _validate_fraction_float(
            min_fraction,
            "surface_support_min_fraction",
        )
        self.surface_support_exponent = _validate_nonnegative_float(
            exponent,
            "surface_support_exponent",
        )

    def set_surface_voting_boundary_policy(self, policy: str) -> None:
        """Select reference clamping or opt-in in-bounds masked surface voting."""

        if policy not in _SURFACE_VOTING_BOUNDARY_POLICIES:
            allowed = ", ".join(repr(value) for value in _SURFACE_VOTING_BOUNDARY_POLICIES)
            raise ValueError(f"policy must be one of: {allowed}")
        self.surface_voting_boundary_policy = policy

    @property
    def surface_voting_diagnostics(self) -> tuple[_SurfaceVotingDiagnostic, ...]:
        """Return immutable per-seed diagnostics from the most recent voting run."""

        return self._last_surface_voting_diagnostics

    @property
    def last_surface_voting_diagnostics(self) -> tuple[_SurfaceVotingDiagnostic, ...]:
        """Alias for :attr:`surface_voting_diagnostics`."""

        return self._last_surface_voting_diagnostics

    def surface_voting_diagnostic_summary(self) -> dict[str, str | int | float]:
        """Return a compact JSON-safe summary of the most recent seed outcomes."""

        diagnostics = self._last_surface_voting_diagnostics
        support_fractions = [diagnostic.support_fraction for diagnostic in diagnostics]
        return {
            "policy": self._last_surface_voting_policy,
            "seed_count": len(diagnostics),
            "boundary_affected_seed_count": sum(
                diagnostic.in_bounds_lag_count < diagnostic.admissible_lag_count
                or diagnostic.selected_tangential_column_count
                < diagnostic.full_tangential_column_count
                for diagnostic in diagnostics
            ),
            "voted_seed_count": sum(not diagnostic.skipped for diagnostic in diagnostics),
            "skipped_seed_count": sum(diagnostic.skipped for diagnostic in diagnostics),
            "support_fraction_min": min(support_fractions, default=1.0),
            "support_fraction_mean": (
                float(sum(support_fractions) / len(support_fractions)) if support_fractions else 1.0
            ),
            "surface_projection_count": sum(
                diagnostic.surface_projection_count for diagnostic in diagnostics
            ),
            "selected_invalid_sample_count": sum(
                diagnostic.selected_invalid_sample_count for diagnostic in diagnostics
            ),
            "face_center_vote_count": sum(
                diagnostic.face_center_vote_count for diagnostic in diagnostics
            ),
        }

    def _update_shift_ranges(self) -> None:
        self.lmins, self.lmaxs = update_shift_ranges_3d(self.ru, self.rv, self.rw)

    def pick_seeds(
        self,
        d: int,
        fm: float,
        ft: np.ndarray,
        pt: np.ndarray,
        tt: np.ndarray,
    ) -> list[FaultCell]:
        """Pick 3D seed cells above a fault-likelihood threshold."""

        distance = _validate_nonnegative_int(d, "d")
        ft_array, pt_array, tt_array = _validate_matching_finite_arrays3_many(
            (ft, pt, tt),
            ("ft", "pt", "tt"),
        )
        threshold = np.float32(fm)
        n3, n2, n1 = ft_array.shape

        candidates = [
            FaultCell(
                i1,
                i2,
                i3,
                ft_array[i3, i2, i1],
                pt_array[i3, i2, i1],
                tt_array[i3, i2, i1],
            )
            for i3 in range(n3)
            for i2 in range(n2)
            for i1 in range(n1)
            if ft_array[i3, i2, i1] > threshold
        ]
        candidates.sort(key=operator.attrgetter("fl"))

        mark = np.zeros((n3, n2, n1), dtype=np.bool_)
        seeds: list[FaultCell] = []
        for cell in reversed(candidates):
            i1 = cell.i1
            i2 = cell.i2
            i3 = cell.i3
            b1 = max(i1 - distance, 0)
            b2 = max(i2 - distance, 0)
            b3 = max(i3 - distance, 0)
            e1 = min(i1 + distance, n1 - 1)
            e2 = min(i2 + distance, n2 - 1)
            e3 = min(i3 + distance, n3 - 1)
            if mark[b3 : e3 + 1, b2 : e2 + 1, b1 : e1 + 1].any():
                continue

            seeds.append(cell)
            mark[i3, i2, i1] = True

        return seeds

    def get_seeds(
        self,
        c1: int,
        c2: int,
        c3: int,
        ft: np.ndarray,
        pt: np.ndarray,
        tt: np.ndarray,
    ) -> list[FaultCell]:
        """Return the seed at one image sample."""

        ft_array, pt_array, tt_array = _validate_matching_finite_arrays3_many(
            (ft, pt, tt),
            ("ft", "pt", "tt"),
        )
        i1 = _validate_int(c1, "c1")
        i2 = _validate_int(c2, "c2")
        i3 = _validate_int(c3, "c3")
        n3, n2, n1 = ft_array.shape
        if not 0 <= i1 < n1:
            raise ValueError("c1 must be inside the image bounds")
        if not 0 <= i2 < n2:
            raise ValueError("c2 must be inside the image bounds")
        if not 0 <= i3 < n3:
            raise ValueError("c3 must be inside the image bounds")

        return [
            FaultCell(
                i1,
                i2,
                i3,
                ft_array[i3, i2, i1],
                pt_array[i3, i2, i1],
                tt_array[i3, i2, i1],
            ),
        ]

    def apply_voting(
        self,
        d: int,
        fm: float,
        ft: np.ndarray,
        pt: np.ndarray,
        tt: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run 3D surface voting for all selected seeds."""

        ft_array, pt_array, tt_array = _validate_matching_finite_arrays3_many(
            (ft, pt, tt),
            ("ft", "pt", "tt"),
        )
        seeds = self.pick_seeds(d, fm, ft_array, pt_array, tt_array)
        return self.apply_voting_from_seeds(seeds, ft_array, pt_array, tt_array)

    def apply_voting_from_seeds(
        self,
        seeds: Sequence[FaultCell],
        ft: np.ndarray,
        pt: np.ndarray,
        tt: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run 3D surface voting for an explicit deterministic seed sequence."""

        self._last_surface_voting_diagnostics = ()
        self._last_surface_voting_policy = self.surface_voting_boundary_policy
        ft_array, pt_array, tt_array = _validate_matching_finite_arrays3_many(
            (ft, pt, tt),
            ("ft", "pt", "tt"),
        )
        n3, n2, n1 = ft_array.shape
        for seed in seeds:
            if not isinstance(seed, FaultCell):
                raise TypeError("seeds must contain FaultCell instances")
            if not (0 <= seed.i1 < n1 and 0 <= seed.i2 < n2 and 0 <= seed.i3 < n3):
                raise ValueError("seed coordinates must be inside the image bounds")
        fs = _smooth_fault_likelihood_3d(ft_array)

        fe = np.zeros_like(ft_array, dtype=np.float32)
        vp = np.zeros_like(ft_array, dtype=np.float32)
        vt = np.zeros_like(ft_array, dtype=np.float32)
        vm = np.zeros_like(ft_array, dtype=np.float32)

        diagnostics: list[_SurfaceVotingDiagnostic] = []
        for seed in seeds:
            diagnostics.append(self._surface_voting(seed, fs, fe, vp, vt, vm))
        self._last_surface_voting_diagnostics = tuple(diagnostics)

        fv = _normalize_and_power_3d(
            fe,
            sigma=self.final_normalization_smoothing,
        )
        return fv, vp, vt

    def thin(
        self,
        fv: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        *,
        mode: str = "reference",
        reference_sigma: float = 1.0,
        hybrid_orientation_gradient_threshold: float = 8.0,
        hybrid_v2_edge_margin: int = 2,
        plateau_tie_breaker: np.ndarray | None = None,
        plateau_tolerance: float = 1.0e-6,
    ) -> np.ndarray:
        """Keep 3D voting-score maxima using the selected thinning mode."""

        fv_array, vp_array, vt_array = _validate_matching_finite_arrays3_many(
            (fv, vp, vt),
            ("fv", "vp", "vt"),
        )
        if plateau_tie_breaker is None:
            tie_breaker_array = fv_array
        else:
            tie_breaker_array = _validate_finite_array3(
                plateau_tie_breaker,
                "plateau_tie_breaker",
            )
            if tie_breaker_array.shape != fv_array.shape:
                raise ValueError("fv and plateau_tie_breaker shapes must match")
        if mode not in {"reference", "normal", "hybrid", "hybrid_v2", "normal_plateau"}:
            raise ValueError(
                "mode must be 'reference', 'normal', 'hybrid', 'hybrid_v2', or 'normal_plateau'"
            )
        threshold = _validate_nonnegative_float(
            hybrid_orientation_gradient_threshold,
            "hybrid_orientation_gradient_threshold",
        )
        edge_margin = _validate_nonnegative_int(
            hybrid_v2_edge_margin,
            "hybrid_v2_edge_margin",
        )
        if mode == "reference":
            thinned, _ = _thin_reference_like_3d(
                fv_array,
                vp_array,
                reference_sigma=reference_sigma,
            )
            return thinned
        if mode == "normal":
            return _thin_fault_normal_3d(fv_array, vp_array, vt_array)
        if mode == "normal_plateau":
            tolerance = _validate_nonnegative_float(
                plateau_tolerance,
                "plateau_tolerance",
            )
            return _thin_fault_normal_plateau_3d(
                fv_array,
                vp_array,
                vt_array,
                plateau_tie_breaker=tie_breaker_array,
                tolerance=tolerance,
            )

        reference, _ = _thin_reference_like_3d(
            fv_array,
            vp_array,
            reference_sigma=reference_sigma,
        )
        normal = _thin_fault_normal_3d(fv_array, vp_array, vt_array)
        roughness_support = fv_array > np.float32(0.0)
        if mode == "hybrid_v2":
            roughness_support = fv_array > np.float32(1.0e-6)
        roughness = _orientation_roughness_3d(
            vp_array,
            vt_array,
            support=roughness_support,
        )
        if mode == "hybrid_v2":
            plateau = _thin_fault_normal_plateau_3d(
                fv_array,
                vp_array,
                vt_array,
                plateau_tie_breaker=tie_breaker_array,
                tolerance=_validate_nonnegative_float(
                    plateau_tolerance,
                    "plateau_tolerance",
                ),
            )
            use_normal = (roughness > np.float32(threshold)) & (normal > np.float32(0.0))
            sparse_normal = _local_candidate_count_3d(normal > np.float32(0.0)) <= 1
            use_plateau = (
                _edge_region_mask_3d(fv_array.shape, edge_margin)
                & (plateau > np.float32(0.0))
                & ((normal <= np.float32(0.0)) | sparse_normal)
            )
            result = reference.copy()
            result[use_normal] = normal[use_normal]
            result[use_plateau] = plateau[use_plateau]
            return result.astype(np.float32, copy=False)

        return np.where(roughness <= threshold, reference, normal).astype(
            np.float32,
            copy=False,
        )

    def update_vector_map(self, radius: int, vector: np.ndarray) -> np.ndarray:
        """Return displacement vectors for offsets ``[-radius, radius]``."""

        radius_int = _validate_nonnegative_int(radius, "radius")
        vector_array = _validate_vector3(vector, "vector")
        offsets = np.arange(-radius_int, radius_int + 1, dtype=np.float32)
        return vector_array[:, np.newaxis] * offsets[np.newaxis, :]

    def samples_in_uvw_box(
        self,
        c1: int,
        c2: int,
        c3: int,
        normal: np.ndarray,
        dip: np.ndarray,
        strike: np.ndarray,
        fx: np.ndarray,
    ) -> np.ndarray:
        """Sample ``1 - fx`` in the seed-centered local ``(w, v, u)`` box."""

        i1, i2, i3, fx_array = _validate_uvw_sampling_origin(c1, c2, c3, fx)

        if NUMBA_AVAILABLE:
            return _samples_in_uvw_box_numba(
                i1,
                i2,
                i3,
                self.ru,
                self.rv,
                self.rw,
                normal,
                dip,
                strike,
                fx_array,
                self.lmins,
                self.lmaxs,
            )
        return _samples_in_uvw_box_python(
            i1,
            i2,
            i3,
            self.ru,
            self.rv,
            self.rw,
            normal,
            dip,
            strike,
            fx_array,
            self.lmins,
            self.lmaxs,
        )

    def _samples_in_uvw_box_masked(
        self,
        c1: int,
        c2: int,
        c3: int,
        normal: np.ndarray,
        dip: np.ndarray,
        strike: np.ndarray,
        fx: np.ndarray,
    ) -> _MaskedUVWBoxSamples:
        """Sample a full UVW box without treating out-of-bounds lags as evidence."""

        i1, i2, i3, fx_array = _validate_uvw_sampling_origin(c1, c2, c3, fx)

        sampler = (
            _samples_in_uvw_box_masked_numba
            if NUMBA_AVAILABLE
            else _samples_in_uvw_box_masked_python
        )
        costs, valid_lag_mask, admissible_count, in_bounds_count = sampler(
            i1,
            i2,
            i3,
            self.ru,
            self.rv,
            self.rw,
            normal,
            dip,
            strike,
            fx_array,
            self.lmins,
            self.lmaxs,
        )
        return _MaskedUVWBoxSamples(
            costs=costs,
            valid_lag_mask=valid_lag_mask,
            w_offset=0,
            v_offset=0,
            full_tangential_shape=costs.shape[:2],
            admissible_lag_count=int(admissible_count),
            in_bounds_lag_count=int(in_bounds_count),
        )

    def _surface_voting(
        self,
        cell: FaultCell,
        ft: np.ndarray,
        fe: np.ndarray,
        vp: np.ndarray,
        vt: np.ndarray,
        vm: np.ndarray,
    ) -> _SurfaceVotingDiagnostic:
        """Accumulate one seed cell's 3D optimal-surface vote in-place."""

        ft_array, fe_array, vp_array, vt_array, vm_array = _validate_matching_arrays3(
            (ft, fe, vp, vt, vm),
            ("ft", "fe", "vp", "vt", "vm"),
        )
        n3, n2, n1 = ft_array.shape
        c1 = cell.i1
        c2 = cell.i2
        c3 = cell.i3
        if not 0 <= c1 < n1:
            raise ValueError("cell.i1 must be inside the image bounds")
        if not 0 <= c2 < n2:
            raise ValueError("cell.i2 must be inside the image bounds")
        if not 0 <= c3 < n3:
            raise ValueError("cell.i3 must be inside the image bounds")

        normal = cell.fault_normal()
        dip = cell.fault_dip_vector()
        strike = cell.fault_strike_vector()
        if self.surface_voting_boundary_policy == "masked_in_bounds":
            return self._surface_voting_masked_in_bounds(
                cell,
                ft_array,
                fe_array,
                vp_array,
                vt_array,
                vm_array,
                normal,
                dip,
                strike,
            )
        return self._surface_voting_reference(
            cell,
            ft_array,
            fe_array,
            vp_array,
            vt_array,
            vm_array,
            normal,
            dip,
            strike,
        )

    def _surface_voting_reference(
        self,
        cell: FaultCell,
        ft_array: np.ndarray,
        fe_array: np.ndarray,
        vp_array: np.ndarray,
        vt_array: np.ndarray,
        vm_array: np.ndarray,
        normal: np.ndarray,
        dip: np.ndarray,
        strike: np.ndarray,
    ) -> _SurfaceVotingDiagnostic:
        """Run the unchanged reference-oriented sampling and source semantics."""

        c1, c2, c3 = cell.index
        support = self._samples_in_uvw_box_masked(
            c1,
            c2,
            c3,
            normal,
            dip,
            strike,
            ft_array,
        )
        full_column_count = int((2 * self.rw + 1) * (2 * self.rv + 1))
        costs = self.samples_in_uvw_box(c1, c2, c3, normal, dip, strike, ft_array)
        surface = find_surface_3d(
            costs,
            lmin=self.lmin,
            bstrain1=self.bstrain1,
            bstrain2=self.bstrain2,
            attribute_smoothing=self.attribute_smoothing,
            surface_smoothing1=self.surface_smoothing1,
            surface_smoothing2=self.surface_smoothing2,
        )

        fa, valid_count = _surface_vote_average(
            c1,
            c2,
            c3,
            self.rv,
            self.rw,
            normal,
            dip,
            strike,
            surface,
            ft_array,
        )
        surface_size = int(surface.size)
        surface_center_lag = _surface_center_lag(surface)
        if valid_count == 0:
            return _SurfaceVotingDiagnostic(
                seed_index=cell.index,
                policy="reference",
                full_tangential_column_count=full_column_count,
                selected_tangential_column_count=surface_size,
                admissible_lag_count=support.admissible_lag_count,
                in_bounds_lag_count=support.in_bounds_lag_count,
                support_fraction=0.0,
                surface_center_lag=surface_center_lag,
                surface_projection_count=0,
                selected_invalid_sample_count=surface_size,
                center_vote_write_count=0,
                face_center_vote_count=0,
                orientation_source=None,
                skipped=True,
                skip_reason="no_valid_surface_samples",
            )

        if surface_size == 0:
            return _SurfaceVotingDiagnostic(
                seed_index=cell.index,
                policy="reference",
                full_tangential_column_count=full_column_count,
                selected_tangential_column_count=0,
                admissible_lag_count=support.admissible_lag_count,
                in_bounds_lag_count=support.in_bounds_lag_count,
                support_fraction=0.0,
                surface_center_lag=None,
                surface_projection_count=0,
                selected_invalid_sample_count=0,
                center_vote_write_count=0,
                face_center_vote_count=0,
                orientation_source=None,
                skipped=True,
                skip_reason="no_feasible_surface",
            )
        support_fraction = float(valid_count) / float(surface_size)
        if support_fraction < self.surface_support_min_fraction:
            return _SurfaceVotingDiagnostic(
                seed_index=cell.index,
                policy="reference",
                full_tangential_column_count=full_column_count,
                selected_tangential_column_count=surface_size,
                admissible_lag_count=support.admissible_lag_count,
                in_bounds_lag_count=support.in_bounds_lag_count,
                support_fraction=support_fraction,
                surface_center_lag=surface_center_lag,
                surface_projection_count=0,
                selected_invalid_sample_count=surface_size - valid_count,
                center_vote_write_count=0,
                face_center_vote_count=0,
                orientation_source=None,
                skipped=True,
                skip_reason="support_below_min_fraction",
            )
        if self.surface_support_exponent > 0.0:
            fa = np.float32(
                fa * np.float32(support_fraction**self.surface_support_exponent),
            )

        strike_angle, dip_angle = _surface_strike_and_dip(
            normal,
            dip,
            strike,
            surface,
            sigma=self.surface_orientation_smoothing,
        )
        vp_value = np.float32(strike_angle)
        vt_value = np.float32(dip_angle)
        align_i3 = abs(normal[2]) > abs(normal[1])

        _accumulate_surface_votes(
            c1,
            c2,
            c3,
            self.rv,
            self.rw,
            fa,
            vp_value,
            vt_value,
            align_i3,
            normal,
            dip,
            strike,
            surface,
            fe_array,
            vp_array,
            vt_array,
            vm_array,
        )

        face_count = _count_reference_face_center_votes(
            c1,
            c2,
            c3,
            self.rv,
            self.rw,
            normal,
            dip,
            strike,
            surface,
            ft_array.shape,
        )
        return _SurfaceVotingDiagnostic(
            seed_index=cell.index,
            policy="reference",
            full_tangential_column_count=full_column_count,
            selected_tangential_column_count=surface_size,
            admissible_lag_count=support.admissible_lag_count,
            in_bounds_lag_count=support.in_bounds_lag_count,
            support_fraction=support_fraction,
            surface_center_lag=surface_center_lag,
            surface_projection_count=0,
            selected_invalid_sample_count=surface_size - valid_count,
            center_vote_write_count=valid_count,
            face_center_vote_count=face_count,
            orientation_source="surface",
            skipped=False,
            skip_reason=None,
        )

    def _surface_voting_masked_in_bounds(
        self,
        cell: FaultCell,
        ft_array: np.ndarray,
        fe_array: np.ndarray,
        vp_array: np.ndarray,
        vt_array: np.ndarray,
        vm_array: np.ndarray,
        normal: np.ndarray,
        dip: np.ndarray,
        strike: np.ndarray,
    ) -> _SurfaceVotingDiagnostic:
        """Run the opt-in masked UVW, masked-DP, and all-face vote path."""

        c1, c2, c3 = cell.index
        full_samples = self._samples_in_uvw_box_masked(
            c1,
            c2,
            c3,
            normal,
            dip,
            strike,
            ft_array,
        )
        full_nw, full_nv = full_samples.full_tangential_shape
        full_column_count = full_nw * full_nv
        supported_columns = np.any(full_samples.valid_lag_mask, axis=2)
        rectangle = _select_supported_origin_rectangle(
            supported_columns,
            origin_w=self.rw,
            origin_v=self.rv,
        )
        if rectangle is None:
            return _SurfaceVotingDiagnostic(
                seed_index=cell.index,
                policy="masked_in_bounds",
                full_tangential_column_count=full_column_count,
                selected_tangential_column_count=0,
                admissible_lag_count=full_samples.admissible_lag_count,
                in_bounds_lag_count=full_samples.in_bounds_lag_count,
                support_fraction=0.0,
                surface_center_lag=None,
                surface_projection_count=0,
                selected_invalid_sample_count=0,
                center_vote_write_count=0,
                face_center_vote_count=0,
                orientation_source=None,
                skipped=True,
                skip_reason="no_supported_origin",
            )

        samples = _crop_masked_uvw_box(full_samples, rectangle)
        surface, projection_count = _find_surface_3d_masked(
            samples.costs,
            samples.valid_lag_mask,
            lmin=self.lmin,
            bstrain1=self.bstrain1,
            bstrain2=self.bstrain2,
            attribute_smoothing=self.attribute_smoothing,
            surface_smoothing1=self.surface_smoothing1,
            surface_smoothing2=self.surface_smoothing2,
        )
        if surface is None:
            return _SurfaceVotingDiagnostic(
                seed_index=cell.index,
                policy="masked_in_bounds",
                full_tangential_column_count=full_column_count,
                selected_tangential_column_count=rectangle.size,
                admissible_lag_count=full_samples.admissible_lag_count,
                in_bounds_lag_count=full_samples.in_bounds_lag_count,
                support_fraction=0.0,
                surface_center_lag=None,
                surface_projection_count=projection_count,
                selected_invalid_sample_count=0,
                center_vote_write_count=0,
                face_center_vote_count=0,
                orientation_source=None,
                skipped=True,
                skip_reason="no_feasible_surface",
            )

        surface_center_lag = float(surface[self.rw - samples.w_offset, self.rv - samples.v_offset])
        fa, valid_count, invalid_count = _surface_vote_average_masked(
            c1,
            c2,
            c3,
            self.rv,
            self.rw,
            samples.w_offset,
            samples.v_offset,
            self.lmin,
            normal,
            dip,
            strike,
            surface,
            samples.valid_lag_mask,
            ft_array,
        )
        support_fraction = float(valid_count) / float(full_column_count)
        if invalid_count > 0:
            return _SurfaceVotingDiagnostic(
                seed_index=cell.index,
                policy="masked_in_bounds",
                full_tangential_column_count=full_column_count,
                selected_tangential_column_count=rectangle.size,
                admissible_lag_count=full_samples.admissible_lag_count,
                in_bounds_lag_count=full_samples.in_bounds_lag_count,
                support_fraction=support_fraction,
                surface_center_lag=surface_center_lag,
                surface_projection_count=projection_count,
                selected_invalid_sample_count=invalid_count,
                center_vote_write_count=0,
                face_center_vote_count=0,
                orientation_source=None,
                skipped=True,
                skip_reason="invalid_selected_sample",
            )
        if valid_count == 0:
            return _SurfaceVotingDiagnostic(
                seed_index=cell.index,
                policy="masked_in_bounds",
                full_tangential_column_count=full_column_count,
                selected_tangential_column_count=rectangle.size,
                admissible_lag_count=full_samples.admissible_lag_count,
                in_bounds_lag_count=full_samples.in_bounds_lag_count,
                support_fraction=0.0,
                surface_center_lag=surface_center_lag,
                surface_projection_count=projection_count,
                selected_invalid_sample_count=0,
                center_vote_write_count=0,
                face_center_vote_count=0,
                orientation_source=None,
                skipped=True,
                skip_reason="no_valid_surface_samples",
            )
        if support_fraction < self.surface_support_min_fraction:
            return _SurfaceVotingDiagnostic(
                seed_index=cell.index,
                policy="masked_in_bounds",
                full_tangential_column_count=full_column_count,
                selected_tangential_column_count=rectangle.size,
                admissible_lag_count=full_samples.admissible_lag_count,
                in_bounds_lag_count=full_samples.in_bounds_lag_count,
                support_fraction=support_fraction,
                surface_center_lag=surface_center_lag,
                surface_projection_count=projection_count,
                selected_invalid_sample_count=0,
                center_vote_write_count=0,
                face_center_vote_count=0,
                orientation_source=None,
                skipped=True,
                skip_reason="support_below_min_fraction",
            )
        if self.surface_support_exponent > 0.0:
            fa = np.float32(
                fa * np.float32(support_fraction**self.surface_support_exponent),
            )

        full_box = rectangle == _TangentialRectangle(0, 0, full_nw, full_nv)
        if full_box and surface.shape[0] >= 3 and surface.shape[1] >= 3:
            strike_angle, dip_angle = _surface_strike_and_dip(
                normal,
                dip,
                strike,
                surface,
                sigma=self.surface_orientation_smoothing,
            )
            orientation_source = "surface"
        else:
            strike_angle, dip_angle = cell.fp, cell.ft
            orientation_source = (
                "seed_boundary_fallback" if not full_box else "seed_small_surface_fallback"
            )

        center_count, face_count, accumulation_invalid_count = _accumulate_surface_votes_masked(
            c1,
            c2,
            c3,
            self.rv,
            self.rw,
            samples.w_offset,
            samples.v_offset,
            self.lmin,
            fa,
            np.float32(strike_angle),
            np.float32(dip_angle),
            abs(normal[2]) > abs(normal[1]),
            normal,
            dip,
            strike,
            surface,
            samples.valid_lag_mask,
            fe_array,
            vp_array,
            vt_array,
            vm_array,
        )
        if accumulation_invalid_count > 0:
            return _SurfaceVotingDiagnostic(
                seed_index=cell.index,
                policy="masked_in_bounds",
                full_tangential_column_count=full_column_count,
                selected_tangential_column_count=rectangle.size,
                admissible_lag_count=full_samples.admissible_lag_count,
                in_bounds_lag_count=full_samples.in_bounds_lag_count,
                support_fraction=support_fraction,
                surface_center_lag=surface_center_lag,
                surface_projection_count=projection_count,
                selected_invalid_sample_count=accumulation_invalid_count,
                center_vote_write_count=0,
                face_center_vote_count=0,
                orientation_source=orientation_source,
                skipped=True,
                skip_reason="invalid_selected_sample",
            )
        return _SurfaceVotingDiagnostic(
            seed_index=cell.index,
            policy="masked_in_bounds",
            full_tangential_column_count=full_column_count,
            selected_tangential_column_count=rectangle.size,
            admissible_lag_count=full_samples.admissible_lag_count,
            in_bounds_lag_count=full_samples.in_bounds_lag_count,
            support_fraction=support_fraction,
            surface_center_lag=surface_center_lag,
            surface_projection_count=projection_count,
            selected_invalid_sample_count=0,
            center_vote_write_count=center_count,
            face_center_vote_count=face_count,
            orientation_source=orientation_source,
            skipped=False,
            skip_reason=None,
        )


def _normalize_and_power_3d(
    x: np.ndarray,
    *,
    sigma: float = 0.0,
    power: int = 8,
) -> np.ndarray:
    """Normalize a final 3D vote map using Java-reference default semantics.

    By default this mirrors ``OptimalSurfaceVoter.normalization``: subtract the
    global minimum, divide by the global maximum when nonzero, then apply
    ``1 - (1 - x) ** power`` without additional smoothing. Set ``sigma > 0`` to
    opt in to the practical smoothed vote-map behavior.
    """

    x_array = _validate_finite_array3(x, "x").astype(np.float32, copy=True)
    sigma_float = _validate_nonnegative_float(sigma, "sigma")
    power_int = _validate_positive_int(power, "power")

    if x_array.size == 0:
        return x_array

    if sigma_float > 0.0:
        x_array = smooth3d(x_array, sigma_float).astype(np.float32, copy=False)

    _normalize_unit_range_in_place(x_array)
    enhanced = np.float32(1.0) - np.power(np.float32(1.0) - x_array, power_int)
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32, copy=False)


def _count_reference_face_center_votes(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    volume_shape: tuple[int, int, int],
) -> int:
    n3, n2, n1 = volume_shape
    face_count = 0
    for kw in range(surface.shape[0]):
        iw = kw - rw
        dw1 = c1 + iw * strike[0]
        dw2 = c2 + iw * strike[1]
        dw3 = c3 + iw * strike[2]
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv - rv
            i1 = math.floor(float(iu * normal[0] + iv * dip[0] + dw1) + 0.5)
            i2 = math.floor(float(iu * normal[1] + iv * dip[1] + dw2) + 0.5)
            i3 = math.floor(float(iu * normal[2] + iv * dip[2] + dw3) + 0.5)
            if not _is_valid_surface_vote_sample(i1, i2, i3, n1, n2, n3):
                continue
            if i1 == 0 or i1 == n1 - 1 or i2 == 0 or i2 == n2 - 1 or i3 == 0 or i3 == n3 - 1:
                face_count += 1
    return face_count


def _surface_vote_average(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    ft: np.ndarray,
) -> tuple[np.float32, int]:
    if NUMBA_AVAILABLE:
        return _surface_vote_average_numba(c1, c2, c3, rv, rw, normal, dip, strike, surface, ft)
    return _surface_vote_average_python(c1, c2, c3, rv, rw, normal, dip, strike, surface, ft)


def _surface_vote_average_python(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    ft: np.ndarray,
) -> tuple[np.float32, int]:
    n3, n2, n1 = ft.shape
    fa = np.float32(0.0)
    valid_count = 0
    for kw in range(surface.shape[0]):
        iw = kw - rw
        dw1 = c1 + iw * strike[0]
        dw2 = c2 + iw * strike[1]
        dw3 = c3 + iw * strike[2]
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv - rv
            x1 = iu * normal[0] + iv * dip[0] + dw1
            x2 = iu * normal[1] + iv * dip[1] + dw2
            x3 = iu * normal[2] + iv * dip[2] + dw3
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            if not _is_valid_surface_vote_sample(i1, i2, i3, n1, n2, n3):
                continue

            fa += ft[i3, i2, i1]
            valid_count += 1

    if valid_count > 0:
        fa /= np.float32(valid_count)
    return fa, valid_count


@njit(cache=True)
def _surface_vote_average_numba(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    ft: np.ndarray,
) -> tuple[np.float32, int]:
    n3, n2, n1 = ft.shape
    fa = np.float32(0.0)
    valid_count = 0
    for kw in range(surface.shape[0]):
        iw = kw - rw
        dw1 = c1 + iw * strike[0]
        dw2 = c2 + iw * strike[1]
        dw3 = c3 + iw * strike[2]
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv - rv
            x1 = iu * normal[0] + iv * dip[0] + dw1
            x2 = iu * normal[1] + iv * dip[1] + dw2
            x3 = iu * normal[2] + iv * dip[2] + dw3
            i1 = math.floor(x1 + 0.5)
            i2 = math.floor(x2 + 0.5)
            i3 = math.floor(x3 + 0.5)
            if not _is_valid_surface_vote_sample(i1, i2, i3, n1, n2, n3):
                continue

            fa += ft[i3, i2, i1]
            valid_count += 1

    if valid_count > 0:
        fa /= np.float32(valid_count)
    return fa, valid_count


def _surface_vote_average_masked(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    w_offset: int,
    v_offset: int,
    lmin: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    valid_lag_mask: np.ndarray,
    ft: np.ndarray,
) -> tuple[np.float32, int, int]:
    if NUMBA_AVAILABLE:
        return _surface_vote_average_masked_numba(
            c1,
            c2,
            c3,
            rv,
            rw,
            w_offset,
            v_offset,
            lmin,
            normal,
            dip,
            strike,
            surface,
            valid_lag_mask,
            ft,
        )
    return _surface_vote_average_masked_python(
        c1,
        c2,
        c3,
        rv,
        rw,
        w_offset,
        v_offset,
        lmin,
        normal,
        dip,
        strike,
        surface,
        valid_lag_mask,
        ft,
    )


def _surface_vote_average_masked_python(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    w_offset: int,
    v_offset: int,
    lmin: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    valid_lag_mask: np.ndarray,
    ft: np.ndarray,
) -> tuple[np.float32, int, int]:
    n3, n2, n1 = ft.shape
    nu = valid_lag_mask.shape[2]
    fa = np.float32(0.0)
    valid_count = 0
    invalid_count = 0
    for kw in range(surface.shape[0]):
        iw = kw + w_offset - rw
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            ku = math.floor(float(iu - lmin) + 0.5)
            if not 0 <= ku < nu or not valid_lag_mask[kw, kv, ku]:
                invalid_count += 1
                continue
            iv = kv + v_offset - rv
            x1 = np.float32(
                float(c1)
                + float(iw) * float(strike[0])
                + float(iv) * float(dip[0])
                + float(iu) * float(normal[0])
            )
            x2 = np.float32(
                float(c2)
                + float(iw) * float(strike[1])
                + float(iv) * float(dip[1])
                + float(iu) * float(normal[1])
            )
            x3 = np.float32(
                float(c3)
                + float(iw) * float(strike[2])
                + float(iv) * float(dip[2])
                + float(iu) * float(normal[2])
            )
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
                invalid_count += 1
                continue
            fa += ft[i3, i2, i1]
            valid_count += 1

    if valid_count > 0:
        fa /= np.float32(valid_count)
    return fa, valid_count, invalid_count


@njit(cache=True)
def _surface_vote_average_masked_numba(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    w_offset: int,
    v_offset: int,
    lmin: int,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    valid_lag_mask: np.ndarray,
    ft: np.ndarray,
) -> tuple[np.float32, int, int]:
    n3, n2, n1 = ft.shape
    nu = valid_lag_mask.shape[2]
    fa = np.float32(0.0)
    valid_count = 0
    invalid_count = 0
    for kw in range(surface.shape[0]):
        iw = kw + w_offset - rw
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            ku = math.floor(float(iu - lmin) + 0.5)
            if not 0 <= ku < nu or not valid_lag_mask[kw, kv, ku]:
                invalid_count += 1
                continue
            iv = kv + v_offset - rv
            x1 = np.float32(
                float(c1)
                + float(iw) * float(strike[0])
                + float(iv) * float(dip[0])
                + float(iu) * float(normal[0])
            )
            x2 = np.float32(
                float(c2)
                + float(iw) * float(strike[1])
                + float(iv) * float(dip[1])
                + float(iu) * float(normal[1])
            )
            x3 = np.float32(
                float(c3)
                + float(iw) * float(strike[2])
                + float(iv) * float(dip[2])
                + float(iu) * float(normal[2])
            )
            i1 = math.floor(x1 + 0.5)
            i2 = math.floor(x2 + 0.5)
            i3 = math.floor(x3 + 0.5)
            if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
                invalid_count += 1
                continue
            fa += ft[i3, i2, i1]
            valid_count += 1

    if valid_count > 0:
        fa /= np.float32(valid_count)
    return fa, valid_count, invalid_count


def _accumulate_surface_votes(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    align_i3: bool,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> None:
    if NUMBA_AVAILABLE:
        _accumulate_surface_votes_numba(
            c1,
            c2,
            c3,
            rv,
            rw,
            fa,
            vp_value,
            vt_value,
            align_i3,
            normal,
            dip,
            strike,
            surface,
            fe,
            vp,
            vt,
            vm,
        )
        return
    _accumulate_surface_votes_python(
        c1,
        c2,
        c3,
        rv,
        rw,
        fa,
        vp_value,
        vt_value,
        align_i3,
        normal,
        dip,
        strike,
        surface,
        fe,
        vp,
        vt,
        vm,
    )


def _accumulate_surface_votes_python(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    align_i3: bool,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> None:
    n3, n2, n1 = fe.shape
    for kw in range(surface.shape[0]):
        iw = kw - rw
        dw1 = c1 + iw * strike[0]
        dw2 = c2 + iw * strike[1]
        dw3 = c3 + iw * strike[2]
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv - rv
            x1 = iu * normal[0] + iv * dip[0] + dw1
            x2 = iu * normal[1] + iv * dip[1] + dw2
            x3 = iu * normal[2] + iv * dip[2] + dw3
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            if not _is_valid_surface_vote_sample(i1, i2, i3, n1, n2, n3):
                continue

            _add_surface_vote(i3, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            if align_i3:
                _add_surface_vote(i3 - 1, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
                _add_surface_vote(i3 + 1, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            else:
                _add_surface_vote(i3, i2 - 1, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
                _add_surface_vote(i3, i2 + 1, i1, fa, vp_value, vt_value, fe, vp, vt, vm)


@njit(cache=True)
def _accumulate_surface_votes_numba(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    align_i3: bool,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> None:
    n3, n2, n1 = fe.shape
    for kw in range(surface.shape[0]):
        iw = kw - rw
        dw1 = c1 + iw * strike[0]
        dw2 = c2 + iw * strike[1]
        dw3 = c3 + iw * strike[2]
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv - rv
            x1 = iu * normal[0] + iv * dip[0] + dw1
            x2 = iu * normal[1] + iv * dip[1] + dw2
            x3 = iu * normal[2] + iv * dip[2] + dw3
            i1 = math.floor(x1 + 0.5)
            i2 = math.floor(x2 + 0.5)
            i3 = math.floor(x3 + 0.5)
            if not _is_valid_surface_vote_sample(i1, i2, i3, n1, n2, n3):
                continue

            _add_surface_vote_numba(i3, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            if align_i3:
                _add_surface_vote_numba(
                    i3 - 1,
                    i2,
                    i1,
                    fa,
                    vp_value,
                    vt_value,
                    fe,
                    vp,
                    vt,
                    vm,
                )
                _add_surface_vote_numba(
                    i3 + 1,
                    i2,
                    i1,
                    fa,
                    vp_value,
                    vt_value,
                    fe,
                    vp,
                    vt,
                    vm,
                )
            else:
                _add_surface_vote_numba(
                    i3,
                    i2 - 1,
                    i1,
                    fa,
                    vp_value,
                    vt_value,
                    fe,
                    vp,
                    vt,
                    vm,
                )
                _add_surface_vote_numba(
                    i3,
                    i2 + 1,
                    i1,
                    fa,
                    vp_value,
                    vt_value,
                    fe,
                    vp,
                    vt,
                    vm,
                )


def _accumulate_surface_votes_masked(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    w_offset: int,
    v_offset: int,
    lmin: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    align_i3: bool,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    valid_lag_mask: np.ndarray,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> tuple[int, int, int]:
    if NUMBA_AVAILABLE:
        return _accumulate_surface_votes_masked_numba(
            c1,
            c2,
            c3,
            rv,
            rw,
            w_offset,
            v_offset,
            lmin,
            fa,
            vp_value,
            vt_value,
            align_i3,
            normal,
            dip,
            strike,
            surface,
            valid_lag_mask,
            fe,
            vp,
            vt,
            vm,
        )
    return _accumulate_surface_votes_masked_python(
        c1,
        c2,
        c3,
        rv,
        rw,
        w_offset,
        v_offset,
        lmin,
        fa,
        vp_value,
        vt_value,
        align_i3,
        normal,
        dip,
        strike,
        surface,
        valid_lag_mask,
        fe,
        vp,
        vt,
        vm,
    )


def _accumulate_surface_votes_masked_python(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    w_offset: int,
    v_offset: int,
    lmin: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    align_i3: bool,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    valid_lag_mask: np.ndarray,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> tuple[int, int, int]:
    n3, n2, n1 = fe.shape
    nu = valid_lag_mask.shape[2]
    center_count = 0
    face_count = 0
    invalid_count = 0
    for kw in range(surface.shape[0]):
        iw = kw + w_offset - rw
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            ku = math.floor(float(iu - lmin) + 0.5)
            if not 0 <= ku < nu or not valid_lag_mask[kw, kv, ku]:
                invalid_count += 1
                continue
            iv = kv + v_offset - rv
            x1 = np.float32(
                float(c1)
                + float(iw) * float(strike[0])
                + float(iv) * float(dip[0])
                + float(iu) * float(normal[0])
            )
            x2 = np.float32(
                float(c2)
                + float(iw) * float(strike[1])
                + float(iv) * float(dip[1])
                + float(iu) * float(normal[1])
            )
            x3 = np.float32(
                float(c3)
                + float(iw) * float(strike[2])
                + float(iv) * float(dip[2])
                + float(iu) * float(normal[2])
            )
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
                invalid_count += 1
                continue
            center_count += 1
            if i1 == 0 or i1 == n1 - 1 or i2 == 0 or i2 == n2 - 1 or i3 == 0 or i3 == n3 - 1:
                face_count += 1

    if invalid_count > 0:
        return 0, 0, invalid_count

    for kw in range(surface.shape[0]):
        iw = kw + w_offset - rw
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv + v_offset - rv
            x1 = np.float32(
                float(c1)
                + float(iw) * float(strike[0])
                + float(iv) * float(dip[0])
                + float(iu) * float(normal[0])
            )
            x2 = np.float32(
                float(c2)
                + float(iw) * float(strike[1])
                + float(iv) * float(dip[1])
                + float(iu) * float(normal[1])
            )
            x3 = np.float32(
                float(c3)
                + float(iw) * float(strike[2])
                + float(iv) * float(dip[2])
                + float(iu) * float(normal[2])
            )
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            _add_surface_vote(i3, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            if align_i3:
                _add_surface_vote(i3 - 1, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
                _add_surface_vote(i3 + 1, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            else:
                _add_surface_vote(i3, i2 - 1, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
                _add_surface_vote(i3, i2 + 1, i1, fa, vp_value, vt_value, fe, vp, vt, vm)

    return center_count, face_count, 0


@njit(cache=True)
def _accumulate_surface_votes_masked_numba(
    c1: int,
    c2: int,
    c3: int,
    rv: int,
    rw: int,
    w_offset: int,
    v_offset: int,
    lmin: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    align_i3: bool,
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    valid_lag_mask: np.ndarray,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> tuple[int, int, int]:
    n3, n2, n1 = fe.shape
    nu = valid_lag_mask.shape[2]
    center_count = 0
    face_count = 0
    invalid_count = 0
    for kw in range(surface.shape[0]):
        iw = kw + w_offset - rw
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            ku = math.floor(float(iu - lmin) + 0.5)
            if not 0 <= ku < nu or not valid_lag_mask[kw, kv, ku]:
                invalid_count += 1
                continue
            iv = kv + v_offset - rv
            x1 = np.float32(
                float(c1)
                + float(iw) * float(strike[0])
                + float(iv) * float(dip[0])
                + float(iu) * float(normal[0])
            )
            x2 = np.float32(
                float(c2)
                + float(iw) * float(strike[1])
                + float(iv) * float(dip[1])
                + float(iu) * float(normal[1])
            )
            x3 = np.float32(
                float(c3)
                + float(iw) * float(strike[2])
                + float(iv) * float(dip[2])
                + float(iu) * float(normal[2])
            )
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
                invalid_count += 1
                continue
            center_count += 1
            if i1 == 0 or i1 == n1 - 1 or i2 == 0 or i2 == n2 - 1 or i3 == 0 or i3 == n3 - 1:
                face_count += 1

    if invalid_count > 0:
        return 0, 0, invalid_count

    for kw in range(surface.shape[0]):
        iw = kw + w_offset - rw
        for kv in range(surface.shape[1]):
            iu = surface[kw, kv]
            iv = kv + v_offset - rv
            x1 = np.float32(
                float(c1)
                + float(iw) * float(strike[0])
                + float(iv) * float(dip[0])
                + float(iu) * float(normal[0])
            )
            x2 = np.float32(
                float(c2)
                + float(iw) * float(strike[1])
                + float(iv) * float(dip[1])
                + float(iu) * float(normal[1])
            )
            x3 = np.float32(
                float(c3)
                + float(iw) * float(strike[2])
                + float(iv) * float(dip[2])
                + float(iu) * float(normal[2])
            )
            i1 = math.floor(float(x1) + 0.5)
            i2 = math.floor(float(x2) + 0.5)
            i3 = math.floor(float(x3) + 0.5)
            _add_surface_vote_numba(i3, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            if align_i3:
                _add_surface_vote_numba(i3 - 1, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
                _add_surface_vote_numba(i3 + 1, i2, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
            else:
                _add_surface_vote_numba(i3, i2 - 1, i1, fa, vp_value, vt_value, fe, vp, vt, vm)
                _add_surface_vote_numba(i3, i2 + 1, i1, fa, vp_value, vt_value, fe, vp, vt, vm)

    return center_count, face_count, 0


@njit(cache=True)
def _is_valid_surface_vote_sample(
    i1: int,
    i2: int,
    i3: int,
    n1: int,
    n2: int,
    n3: int,
) -> bool:
    return 0 <= i1 < n1 and 0 < i2 < n2 - 1 and 0 < i3 < n3 - 1


def _add_surface_vote(
    i3: int,
    i2: int,
    i1: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> None:
    n3, n2, n1 = fe.shape
    if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
        return
    fe[i3, i2, i1] += fa
    _update_orientation_if_stronger(i3, i2, i1, fa, vp_value, vt_value, vp, vt, vm)


@njit(cache=True)
def _add_surface_vote_numba(
    i3: int,
    i2: int,
    i1: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    fe: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> None:
    n3, n2, n1 = fe.shape
    if not (0 <= i1 < n1 and 0 <= i2 < n2 and 0 <= i3 < n3):
        return
    fe[i3, i2, i1] += fa
    if fa > vm[i3, i2, i1]:
        vm[i3, i2, i1] = fa
        vp[i3, i2, i1] = vp_value
        vt[i3, i2, i1] = vt_value


def _update_orientation_if_stronger(
    i3: int,
    i2: int,
    i1: int,
    fa: np.float32,
    vp_value: np.float32,
    vt_value: np.float32,
    vp: np.ndarray,
    vt: np.ndarray,
    vm: np.ndarray,
) -> None:
    if fa > vm[i3, i2, i1]:
        vm[i3, i2, i1] = fa
        vp[i3, i2, i1] = vp_value
        vt[i3, i2, i1] = vt_value


def _fault_normal_components_from_strike_and_dip(
    phi: np.ndarray,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.deg2rad(phi).astype(np.float32, copy=False)
    t = np.deg2rad(theta).astype(np.float32, copy=False)
    cp = np.cos(p)
    sp = np.sin(p)
    ct = np.cos(t)
    st = np.sin(t)
    w1 = -ct
    w2 = st * cp
    w3 = -st * sp
    return (
        w1.astype(np.float32, copy=False),
        w2.astype(np.float32, copy=False),
        w3.astype(np.float32, copy=False),
    )


def _thin_reference_like_3d(
    fv: np.ndarray,
    vp: np.ndarray,
    *,
    reference_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    return reference_like_3d_thin_values(
        fv,
        vp,
        sigma=reference_sigma,
        reinforce_vertical=True,
    )


def _thin_fault_normal_3d(
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
) -> np.ndarray:
    n3, n2, n1 = fv.shape
    thinned = np.zeros((n3, n2, n1), dtype=np.float32)
    if fv.size == 0:
        return thinned

    fs = smooth3d(fv, 1.0).astype(np.float32, copy=False)
    i3, i2, i1 = np.indices((n3, n2, n1), dtype=np.float32)
    w1, w2, w3 = _fault_normal_components_from_strike_and_dip(vp, vt)

    fp = sample3(fs, i1 + w1, i2 + w2, i3 + w3, order=1, mode="nearest")
    fm = sample3(fs, i1 - w1, i2 - w2, i3 - w3, order=1, mode="nearest")
    keep = (fp < fs) & (fm < fs)
    thinned[keep] = fv[keep]
    return thinned


def _thin_fault_normal_plateau_3d(
    fv: np.ndarray,
    vp: np.ndarray,
    vt: np.ndarray,
    *,
    plateau_tie_breaker: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    n3, n2, n1 = fv.shape
    thinned = np.zeros((n3, n2, n1), dtype=np.float32)
    if fv.size == 0:
        return thinned

    fs = smooth3d(fv, 1.0).astype(np.float32, copy=False)
    i3, i2, i1 = np.indices((n3, n2, n1), dtype=np.float32)
    w1, w2, w3 = _fault_normal_components_from_strike_and_dip(vp, vt)

    fp = sample3(fs, i1 + w1, i2 + w2, i3 + w3, order=1, mode="nearest")
    fm = sample3(fs, i1 - w1, i2 - w2, i3 - w3, order=1, mode="nearest")
    eps = np.float32(1.0e-6)
    candidate = (fv > eps) & (fs >= fp - np.float32(tolerance)) & (fs >= fm - np.float32(tolerance))
    if not np.any(candidate):
        return thinned

    dominant_axis = np.argmax(
        np.stack((np.abs(w1), np.abs(w2), np.abs(w3)), axis=0),
        axis=0,
    )
    axis_to_array_axis = (2, 1, 0)
    for normal_axis, array_axis in enumerate(axis_to_array_axis):
        axis_candidates = candidate & (dominant_axis == normal_axis)
        _collapse_candidate_runs_along_axis(
            fv,
            plateau_tie_breaker,
            axis_candidates,
            array_axis,
            thinned,
        )

    return thinned


def _collapse_candidate_runs_along_axis(
    fv: np.ndarray,
    tie_breaker: np.ndarray,
    candidate: np.ndarray,
    axis: int,
    thinned: np.ndarray,
) -> None:
    moved_candidate = np.moveaxis(candidate, axis, -1)
    moved_fv = np.moveaxis(fv, axis, -1)
    moved_tie_breaker = np.moveaxis(tie_breaker, axis, -1)
    moved_thinned = np.moveaxis(thinned, axis, -1)

    line_length = moved_candidate.shape[-1]
    for line_index in np.ndindex(moved_candidate.shape[:-1]):
        line = moved_candidate[line_index]
        start: int | None = None
        for offset in range(line_length + 1):
            in_run = offset < line_length and bool(line[offset])
            if in_run and start is None:
                start = offset
            if (not in_run) and start is not None:
                _retain_plateau_run_sample(
                    moved_fv[line_index],
                    moved_tie_breaker[line_index],
                    moved_thinned[line_index],
                    start,
                    offset,
                )
                start = None


def _retain_plateau_run_sample(
    fv_line: np.ndarray,
    tie_breaker_line: np.ndarray,
    thinned_line: np.ndarray,
    start: int,
    stop: int,
) -> None:
    run_tie_breaker = tie_breaker_line[start:stop]
    if np.all(run_tie_breaker == run_tie_breaker[0]):
        keep_offset = (start + stop - 1) // 2
    else:
        keep_offset = start + int(np.argmax(run_tie_breaker))
    thinned_line[keep_offset] = fv_line[keep_offset]


def _edge_region_mask_3d(shape: tuple[int, int, int], margin: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.bool_)
    if margin == 0 or mask.size == 0:
        return mask

    n3, n2, n1 = shape
    m3 = min(margin, n3)
    m2 = min(margin, n2)
    m1 = min(margin, n1)
    mask[:m3, :, :] = True
    mask[n3 - m3 :, :, :] = True
    mask[:, :m2, :] = True
    mask[:, n2 - m2 :, :] = True
    mask[:, :, :m1] = True
    mask[:, :, n1 - m1 :] = True
    return mask


def _local_candidate_count_3d(candidate: np.ndarray) -> np.ndarray:
    candidate_array = np.asarray(candidate, dtype=np.uint8)
    padded = np.pad(candidate_array, 1, mode="constant", constant_values=0)
    counts = np.zeros(candidate_array.shape, dtype=np.uint8)
    for d3 in range(3):
        for d2 in range(3):
            for d1 in range(3):
                counts += padded[
                    d3 : d3 + candidate_array.shape[0],
                    d2 : d2 + candidate_array.shape[1],
                    d1 : d1 + candidate_array.shape[2],
                ]
    return counts


def _orientation_roughness_3d(
    vp: np.ndarray,
    vt: np.ndarray,
    support: np.ndarray | None = None,
) -> np.ndarray:
    if vp.size == 0:
        return np.zeros(vp.shape, dtype=np.float32)

    if support is None:
        support_array = np.ones(vp.shape, dtype=np.bool_)
    else:
        support_array = np.asarray(support, dtype=np.bool_)
        if support_array.shape != vp.shape:
            raise ValueError("support shape must match vp shape")

    roughness_squared = np.zeros(vp.shape, dtype=np.float32)
    for axis in range(3):
        if vp.shape[axis] < 2:
            continue

        strike_diff = _strike_difference_degrees(
            np.diff(vp, axis=axis).astype(np.float32, copy=False)
        )
        dip_diff = np.diff(vt, axis=axis).astype(np.float32, copy=False)
        diff_squared = strike_diff ** np.float32(2.0) + dip_diff ** np.float32(2.0)

        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(0, -1)
        upper[axis] = slice(1, None)
        pair_support = support_array[tuple(lower)] & support_array[tuple(upper)]
        diff_squared = np.where(pair_support, diff_squared, np.float32(0.0)).astype(
            np.float32,
            copy=False,
        )
        np.maximum(
            roughness_squared[tuple(lower)], diff_squared, out=roughness_squared[tuple(lower)]
        )
        np.maximum(
            roughness_squared[tuple(upper)], diff_squared, out=roughness_squared[tuple(upper)]
        )

    return np.sqrt(roughness_squared).astype(np.float32, copy=False)


def _strike_difference_degrees(delta: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(delta).astype(np.float32, copy=False)
    wrapped = 0.5 * np.rad2deg(np.arctan2(np.sin(2.0 * radians), np.cos(2.0 * radians)))
    return np.abs(wrapped).astype(np.float32, copy=False)


def _smooth_fault_likelihood_3d(
    ft: np.ndarray,
    *,
    sigma: float = 1.0,
) -> np.ndarray:
    ft_array = _validate_finite_array3(ft, "ft").astype(np.float32, copy=True)
    sigma_float = _validate_nonnegative_float(sigma, "sigma")

    if ft_array.size == 0:
        return ft_array

    if sigma_float > 0.0:
        ft_array = smooth3d(ft_array, sigma_float).astype(np.float32, copy=False)

    _normalize_unit_range_in_place(ft_array)
    return ft_array


def _surface_strike_and_dip(
    normal: np.ndarray,
    dip: np.ndarray,
    strike: np.ndarray,
    surface: np.ndarray,
    *,
    sigma: float | None = None,
) -> tuple[float, float]:
    """Return orientation from center differences of a local ``u(w,v)`` surface.

    ``surface`` must be a finite 2D ``(nw, nv)`` array with at least three
    samples along both axes. If ``sigma`` is ``None`` or ``0.0``, derivatives
    are computed from the raw surface. If ``sigma`` is positive, the surface is
    smoothed before computing centered ``du/dv`` and ``du/dw``. The input
    surface is never modified. Strike/dip signs are delegated to
    ``strike_and_dip_from_local_surface_derivatives``.
    """

    normal_array = _validate_finite_vector3(normal, "normal")
    dip_array = _validate_finite_vector3(dip, "dip")
    strike_array = _validate_finite_vector3(strike, "strike")
    surface_array = _smooth_surface_for_orientation(surface, sigma)
    du_dv, du_dw = _surface_center_derivatives(surface_array)
    return strike_and_dip_from_local_surface_derivatives(
        normal_array,
        dip_array,
        strike_array,
        du_dv,
        du_dw,
    )


def _smooth_surface_for_orientation(
    surface: np.ndarray,
    sigma: float | None,
) -> np.ndarray:
    surface_array = _validate_finite_array2(surface, "surface").astype(
        np.float32,
        copy=True,
    )
    if surface_array.shape[0] < 3 or surface_array.shape[1] < 3:
        raise ValueError("surface must have at least three samples along w and v")

    if sigma is None:
        return surface_array

    sigma_float = _validate_nonnegative_float(sigma, "sigma")
    if sigma_float == 0.0:
        return surface_array

    return smooth_surface_2d(
        surface_array,
        sigma1=sigma_float,
        sigma2=sigma_float,
    ).astype(np.float32, copy=False)


def _surface_center_derivatives(surface: np.ndarray) -> tuple[float, float]:
    iw = surface.shape[0] // 2
    iv = surface.shape[1] // 2
    du_dv = float(0.5 * (surface[iw, iv + 1] - surface[iw, iv - 1]))
    du_dw = float(0.5 * (surface[iw + 1, iv] - surface[iw - 1, iv]))
    return du_dv, du_dw


def _normalize_unit_range_in_place(x: np.ndarray) -> None:
    x -= np.min(x)
    max_value = np.max(x)
    if max_value > 0.0:
        x /= max_value
    np.clip(x, 0.0, 1.0, out=x)
