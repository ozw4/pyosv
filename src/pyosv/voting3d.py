"""3D optimal-surface voting entry points."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np

from pyosv._accel import NUMBA_AVAILABLE
from pyosv._voting3d.accumulation import (
    _accumulate_surface_votes as _accumulate_surface_votes_impl,
    _accumulate_surface_votes_masked as _accumulate_surface_votes_masked_impl,
    _add_surface_vote as _add_surface_vote,
    _add_surface_vote_numba as _add_surface_vote_numba,
    _accumulate_surface_votes_masked_numba as _accumulate_surface_votes_masked_numba,
    _accumulate_surface_votes_masked_python as _accumulate_surface_votes_masked_python,
    _accumulate_surface_votes_numba as _accumulate_surface_votes_numba,
    _accumulate_surface_votes_python as _accumulate_surface_votes_python,
    _count_reference_face_center_votes,
    _is_valid_surface_vote_sample as _is_valid_surface_vote_sample,
    _update_orientation_if_stronger as _update_orientation_if_stronger,
)
from pyosv._voting3d.models import (
    _MaskedUVWBoxSamples,
    _SurfaceVotingDiagnostic,
    _TangentialRectangle,
)
from pyosv._voting3d.config import SurfaceVoterConfig
from pyosv._voting3d.context import SurfaceVotingContext
from pyosv._voting3d.normalization import (
    _normalize_and_power_3d,
    _normalize_unit_range_in_place as _normalize_unit_range_in_place,
    _smooth_fault_likelihood_3d,
)
from pyosv._voting3d.orientation import (
    _smooth_surface_for_orientation as _smooth_surface_for_orientation,
    _surface_center_derivatives as _surface_center_derivatives,
    _surface_strike_and_dip,
)
from pyosv._voting3d.scoring_numba import (
    _surface_vote_average_masked_numba as _surface_vote_average_masked_numba,
    _surface_vote_average_numba as _surface_vote_average_numba,
)
from pyosv._voting3d.scoring import (
    _surface_vote_average as _surface_vote_average_impl,
    _surface_vote_average_masked as _surface_vote_average_masked_impl,
)
from pyosv._voting3d.scoring_python import (
    _surface_vote_average_masked_python as _surface_vote_average_masked_python,
    _surface_vote_average_python as _surface_vote_average_python,
)
from pyosv._voting3d.policies import SURFACE_VOTING_POLICY_REGISTRY
from pyosv._voting3d.thinning import (
    _collapse_candidate_runs_along_axis as _collapse_candidate_runs_along_axis,
    _edge_region_mask_3d,
    _fault_normal_components_from_strike_and_dip as _fault_normal_components_from_strike_and_dip,
    _local_candidate_count_3d,
    _orientation_roughness_3d,
    _retain_plateau_run_sample as _retain_plateau_run_sample,
    _strike_difference_degrees as _strike_difference_degrees,
    _thin_fault_normal_3d,
    _thin_fault_normal_plateau_3d,
    _thin_reference_like_3d,
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
    _validate_array3 as _validate_array3,
    _validate_finite_array2 as _validate_finite_array2,
    _validate_finite_array3,
    _validate_finite_vector3 as _validate_finite_vector3,
    _validate_fraction_float,
    _validate_int,
    _validate_matching_arrays3,
    _validate_matching_finite_arrays3_many,
    _validate_nonnegative_float,
    _validate_nonnegative_int,
    _validate_positive_int as _validate_positive_int,
    _validate_vector3,
)
from pyosv.cells import FaultCell
from pyosv.dp import (
    _find_surface_3d_masked,
    find_surface_3d,
    shift_range,
    strain_to_bstrain,
    update_shift_ranges_3d,
)

__all__ = ["OptimalSurfaceVoter"]


_SURFACE_VOTING_BOUNDARY_POLICIES = tuple(SURFACE_VOTING_POLICY_REGISTRY)


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

        context = SurfaceVotingContext(
            config=SurfaceVoterConfig(
                ru=self.ru,
                rv=self.rv,
                rw=self.rw,
                lmin=self.lmin,
                bstrain1=self.bstrain1,
                bstrain2=self.bstrain2,
                attribute_smoothing=self.attribute_smoothing,
                surface_smoothing1=self.surface_smoothing1,
                surface_smoothing2=self.surface_smoothing2,
                surface_orientation_smoothing=self.surface_orientation_smoothing,
                surface_support_min_fraction=self.surface_support_min_fraction,
                surface_support_exponent=self.surface_support_exponent,
            ),
            cell=cell,
            ft=ft_array,
            fe=fe_array,
            vp=vp_array,
            vt=vt_array,
            vm=vm_array,
            normal=cell.fault_normal(),
            dip=cell.fault_dip_vector(),
            strike=cell.fault_strike_vector(),
            sample_reference=self.samples_in_uvw_box,
            sample_masked=self._samples_in_uvw_box_masked,
            find_surface=find_surface_3d,
            find_surface_masked=_find_surface_3d_masked,
            score_reference=_surface_vote_average,
            score_masked=_surface_vote_average_masked,
            accumulate_reference=_accumulate_surface_votes,
            accumulate_masked=_accumulate_surface_votes_masked,
            surface_orientation=_surface_strike_and_dip,
            count_reference_face_votes=_count_reference_face_center_votes,
            select_supported_rectangle=_select_supported_origin_rectangle,
            crop_masked_box=_crop_masked_uvw_box,
            surface_center_lag=_surface_center_lag,
        )
        policy = SURFACE_VOTING_POLICY_REGISTRY.get(self.surface_voting_boundary_policy)
        if policy is None:
            allowed = ", ".join(repr(value) for value in _SURFACE_VOTING_BOUNDARY_POLICIES)
            raise ValueError(f"policy must be one of: {allowed}")
        return policy.vote(context)

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
    return _surface_vote_average_impl(
        c1,
        c2,
        c3,
        rv,
        rw,
        normal,
        dip,
        strike,
        surface,
        ft,
        use_numba=NUMBA_AVAILABLE,
    )


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
    return _surface_vote_average_masked_impl(
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
        use_numba=NUMBA_AVAILABLE,
    )


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
    _accumulate_surface_votes_impl(
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
        use_numba=NUMBA_AVAILABLE,
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
    return _accumulate_surface_votes_masked_impl(
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
        use_numba=NUMBA_AVAILABLE,
    )
