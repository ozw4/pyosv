"""Public facade for 3D fault-orientation scanning."""

# Re-exported private helpers are a compatibility requirement.
# ruff: noqa: F401

from pyosv._orient3d.geometry import (
    _coordinate_grids3,
    _fault_normal_components_from_strike_and_dip,
    _gaussian_derivatives,
)
from pyosv._orient3d.interpolation import (
    _directional_gaussian_smooth,
    _sample2_with_constant,
    _sample3_with_constant,
    _sample_oriented_volume,
    _smooth_oriented_response,
)
from pyosv._orient3d.normalization import (
    _normalize_likelihood,
    _normalize_reference_like_likelihood,
    _normalize_unit_range,
)
from pyosv._orient3d.rotate_shear import (
    _dip_shear_from_theta,
    _rotate3_axis1,
    _rotated_axis1_grid,
    _shear2,
    _shear_rotated_volume,
    _smooth_rotated_strike_axis,
    _smooth_sheared_dip_axis,
    _symmetric_sample_count_covering_radius,
    _unrotate3_axis1,
    _unshear2,
    _unshear_rotated_volume,
)
from pyosv._orient3d.sampling import (
    _angle_sampling,
    _reference_like_dip_sampling,
    _reference_like_strike_sampling,
    _refined_reference_like_sampling,
    _validate_angle,
    _validate_bool,
    _validate_finite_image3,
    _validate_interpolation_order,
    _validate_matching_finite_images3,
    _validate_optional_nonnegative_float,
    _validate_positive_float,
    _validate_reference_like_backend,
    _validate_refinement_factor,
)
from pyosv._orient3d.scanner import FaultOrientScanner3
from pyosv._orient3d.scoring import (
    _orientation_basis_from_strike_and_dip,
    _orientation_confidence_from_scores,
    _reference_like_orientation_score,
    _reference_like_planarity_to_likelihood,
    _update_best_second_orientation,
)

__all__ = ["FaultOrientScanner3"]
