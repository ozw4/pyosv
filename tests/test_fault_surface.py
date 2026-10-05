"""Scientific behavior of the native continuous fault-surface extension."""

from dataclasses import replace

import numpy as np
import pytest

from pyosv.fault_surface import (
    FaultSurfaceFitConfig,
    evaluate_fault_surface,
    fit_fault_surface,
    refit_fault_surface,
    sample_fault_surface,
)


def candidate_points(level=14.0):
    sample, inline = np.meshgrid(np.linspace(3, 28, 8), np.linspace(3, 28, 8), indexing="ij")
    return np.column_stack((sample.ravel(), np.full(sample.size, level), inline.ravel()))


def probability_plane(level=15.0):
    _, crossline, _ = np.indices((32, 32, 32), dtype=np.float32)
    return np.exp(-0.5 * ((crossline - level) / 0.75) ** 2)


def boundary_edges(triangles):
    edges = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.max(counts) == 2
    return unique[counts == 1]


def test_prediction_drives_simple_surface_and_native_sticks_without_mutation():
    probability, candidate = probability_plane(), candidate_points()
    source_probability, source_candidate = probability.copy(), candidate.copy()
    result = fit_fault_surface(probability, candidate)
    assert result.model.dependent_axis == 1
    assert result.model.independent_axes == (0, 2)
    assert result.model.coefficients.shape == (6, 6)
    assert np.max(np.abs(result.vertices[:, 1] - 15)) < 0.1
    assert len(result.triangles) == 2 * 32**2
    assert len(boundary_edges(result.triangles)) == 4 * 32
    assert result.supported.all()
    assert len(result.sticks) == 4
    for stick in result.sticks:
        assert stick.axis == 2
        assert stick.index == int(stick.index)
        assert stick.points.shape == (5, 3)
        np.testing.assert_equal(stick.points[:, stick.axis], stick.index)
    repeated = fit_fault_surface(probability, candidate)
    np.testing.assert_array_equal(result.vertices, repeated.vertices)
    np.testing.assert_array_equal(result.model.coefficients, repeated.model.coefficients)
    np.testing.assert_array_equal(source_probability, probability)
    np.testing.assert_array_equal(source_candidate, candidate)
    assert not result.vertices.flags.writeable
    assert not result.model.coefficients.flags.writeable


def test_smooth_curved_surface_follows_prediction_without_skin_density():
    inline, crossline, sample = np.indices((40, 40, 40), dtype=np.float32)
    truth = 15 + 0.01 * (sample - 20) ** 2 + 0.006 * (inline - 20) ** 2
    probability = np.exp(-0.5 * ((crossline - truth) / 0.75) ** 2)
    sample, inline = np.meshgrid(np.linspace(3, 36, 8), np.linspace(3, 36, 8), indexing="ij")
    candidate = np.column_stack(
        (
            sample.ravel(),
            14 + 0.01 * (sample.ravel() - 20) ** 2 + 0.006 * (inline.ravel() - 20) ** 2,
            inline.ravel(),
        )
    )
    result = fit_fault_surface(probability, candidate)
    sample, crossline, inline = result.vertices.T
    error = crossline - (15 + 0.01 * (sample - 20) ** 2 + 0.006 * (inline - 20) ** 2)
    assert np.sqrt(np.mean(error**2)) < 0.1
    assert np.max(np.abs(error)) < 0.25


def test_missing_prediction_support_keeps_one_complete_surface():
    probability = probability_plane()
    valid = np.ones_like(probability, dtype=bool)
    valid[11:21, :, 11:21] = False
    probability[~valid] = np.nan
    result = fit_fault_surface(probability, candidate_points(), valid_mask=valid)
    assert np.any(~result.supported)
    assert np.any(result.supported)
    assert np.all(np.isfinite(result.vertices))
    assert np.all(np.isfinite(result.probability))
    assert len(result.triangles) == 2 * 32**2
    assert len(boundary_edges(result.triangles)) == 4 * 32
    assert np.max(np.abs(result.vertices[:, 1] - 15)) < 1.1


def test_search_corridor_does_not_jump_to_stronger_neighbor():
    probability = np.maximum(0.4 * probability_plane(12), probability_plane(23))
    result = fit_fault_surface(
        probability,
        candidate_points(12),
        config=FaultSurfaceFitConfig(search_radius=2),
    )
    assert np.max(np.abs(result.vertices[:, 1] - 12)) < 0.1


def test_manual_point_can_move_beyond_search_radius_and_remains_exact():
    baseline = fit_fault_surface(probability_plane(), candidate_points())
    model_bytes = baseline.model.coefficients.tobytes()
    controls = evaluate_fault_surface(baseline.model, np.array([[15.5, 15.5], [7.0, 7.0]]))
    controls = controls.copy()
    controls[0, 1] += 8
    edited = refit_fault_surface(baseline.model, controls)
    actual = evaluate_fault_surface(edited.model, controls[:, edited.model.independent_axes])
    np.testing.assert_allclose(actual, controls, atol=1e-5, rtol=0)
    assert edited.probability is None
    assert edited.supported is None
    assert baseline.model.coefficients.tobytes() == model_bytes
    np.testing.assert_array_equal(edited.triangles, baseline.triangles)
    restored = refit_fault_surface(baseline.model, np.empty((0, 3)))
    np.testing.assert_array_equal(restored.vertices, baseline.vertices)


def test_all_sparse_native_controls_can_be_exact_constraints():
    baseline = fit_fault_surface(probability_plane(), candidate_points())
    controls = np.vstack([stick.points for stick in baseline.sticks])
    controls[7, 1] += 3
    edited = refit_fault_surface(baseline.model, controls)
    actual = evaluate_fault_surface(edited.model, controls[:, edited.model.independent_axes])
    np.testing.assert_allclose(actual, controls, atol=1e-5, rtol=0)


def test_impossible_manual_constraints_reject_without_changing_model():
    baseline = fit_fault_surface(probability_plane(), candidate_points())
    coefficients = baseline.model.coefficients.copy()
    with pytest.raises(ValueError, match="conflict"):
        refit_fault_surface(baseline.model, np.array([[15, 17, 15], [15, 18, 15]]))
    with pytest.raises(ValueError, match="fixed surface domain"):
        refit_fault_surface(baseline.model, np.array([[1, 17, 15]]))
    with pytest.raises(ValueError, match="inside the probability volume"):
        refit_fault_surface(baseline.model, np.array([[15, 40, 15]]))
    np.testing.assert_array_equal(baseline.model.coefficients, coefficients)


def test_multiple_sheets_and_no_prediction_support_fail_explicitly():
    probability = probability_plane()
    candidate = np.vstack((candidate_points(12), candidate_points(18)))
    with pytest.raises(ValueError, match="multiple sheets"):
        fit_fault_surface(probability, candidate)
    with pytest.raises(ValueError, match="No valid prediction support"):
        fit_fault_surface(np.zeros_like(probability), candidate_points())


def test_nonfinite_or_invalid_contract_is_not_silently_accepted():
    probability = probability_plane()
    probability[15, 15, 15] = np.nan
    with pytest.raises(ValueError, match="Sampled valid probability"):
        fit_fault_surface(probability, candidate_points())
    with pytest.raises(ValueError, match="valid_mask"):
        fit_fault_surface(probability_plane(), candidate_points(), valid_mask=np.ones((2, 2, 2)))
    with pytest.raises(ValueError, match="search_step"):
        fit_fault_surface(
            probability_plane(),
            candidate_points(),
            config=FaultSurfaceFitConfig(search_step=0),
        )
    baseline = fit_fault_surface(probability_plane(), candidate_points())
    with pytest.raises(ValueError, match="remaining axes"):
        sample_fault_surface(replace(baseline.model, independent_axes=(2, 0)))
    with pytest.raises(ValueError, match="outside the model domain"):
        evaluate_fault_surface(baseline.model, np.array([[0, 0]]))


def test_volume_boundary_exit_is_rejected_instead_of_clipped():
    baseline = fit_fault_surface(probability_plane(), candidate_points())
    coefficients = baseline.model.coefficients.copy()
    coefficients += 30
    with pytest.raises(ValueError, match="leaves the volume"):
        sample_fault_surface(replace(baseline.model, coefficients=coefficients))


@pytest.mark.parametrize("dependent_axis", [0, 1, 2])
def test_native_graph_axis_and_stick_plane_follow_coordinate_contract(dependent_axis):
    coordinates = np.indices((32, 32, 32), dtype=np.float32)[::-1]
    probability = np.exp(-0.5 * ((coordinates[dependent_axis] - 15) / 0.75) ** 2)
    independent = tuple(axis for axis in range(3) if axis != dependent_axis)
    grid = candidate_points()[:, (0, 2)]
    candidate = np.full((len(grid), 3), 14.0)
    candidate[:, independent] = grid
    result = fit_fault_surface(probability, candidate)
    assert result.model.dependent_axis == dependent_axis
    assert result.model.independent_axes == independent
    assert np.max(np.abs(result.vertices[:, dependent_axis] - 15)) < 0.1
    for stick in result.sticks:
        assert stick.axis == independent[1]
        np.testing.assert_equal(stick.points[:, stick.axis], stick.index)


def test_cropped_model_translation_preserves_full_index_surface():
    cropped = fit_fault_surface(probability_plane(), candidate_points())
    origin_xyz = np.array([100, 200, 300])
    model = cropped.model
    translated_model = replace(
        model,
        bounds=model.bounds + origin_xyz[list(model.independent_axes), None],
        coefficients=model.coefficients + origin_xyz[model.dependent_axis],
        volume_shape=(400, 300, 200),
    )
    translated = sample_fault_surface(translated_model)
    np.testing.assert_allclose(
        translated.vertices, cropped.vertices + origin_xyz, atol=3e-5, rtol=0
    )
    np.testing.assert_array_equal(translated.triangles, cropped.triangles)
    for original, shifted in zip(cropped.sticks, translated.sticks):
        np.testing.assert_allclose(shifted.points, original.points + origin_xyz, atol=1e-10, rtol=0)
        assert shifted.index == original.index + origin_xyz[original.axis]


def test_valid_sample_ignores_zero_weight_masked_nan_neighbor():
    probability = probability_plane()
    valid = np.ones_like(probability, dtype=bool)
    probability[3, 16, 3] = np.nan
    valid[3, 16, 3] = False
    result = fit_fault_surface(probability, candidate_points(), valid_mask=valid)
    assert np.all(np.isfinite(result.vertices))
    assert np.any(result.supported)


def test_stronger_neighbor_inside_corridor_is_reported_as_ambiguous():
    probability = np.maximum(0.4 * probability_plane(12), probability_plane(17))
    with pytest.raises(ValueError, match="Ambiguous prediction bands"):
        fit_fault_surface(probability, candidate_points(12))
    result = fit_fault_surface(
        probability,
        candidate_points(12),
        config=FaultSurfaceFitConfig(search_radius=2),
    )
    assert np.max(np.abs(result.vertices[:, 1] - 12)) < 0.1


@pytest.mark.parametrize("minimum_location, mesh_shape", [(1 / 33, (34, 33)), (1 / 3, (33, 33))])
def test_bounds_check_includes_render_and_stick_points(minimum_location, mesh_shape):
    from pyosv.fault_surface import FaultSurfaceModel

    quadratic = 100.0
    linear = -2 * quadratic * minimum_location
    constant = quadratic * minimum_location**2 - 5e-5
    coefficients = np.array(
        [
            constant,
            constant + linear / 3,
            constant + 2 * linear / 3 + quadratic / 3,
            constant + linear + quadratic,
        ]
    )
    model = FaultSurfaceModel(
        dependent_axis=0,
        independent_axes=(1, 2),
        bounds=np.array([[0.0, 1.0], [0.0, 1.0]]),
        coefficients=np.tile(coefficients[:, None], (1, 4)),
        volume_shape=(2, 2, 129),
        mesh_shape=mesh_shape,
        stick_count=4,
        stick_point_count=4,
    )
    with pytest.raises(ValueError, match="leaves the volume"):
        sample_fault_surface(model)


def test_render_resolution_does_not_change_fitting_or_manual_constraint_solution():
    probability, candidate = probability_plane(), candidate_points()
    coarse = fit_fault_surface(
        probability,
        candidate,
        config=FaultSurfaceFitConfig(mesh_shape=(9, 11)),
    )
    fine = fit_fault_surface(
        probability,
        candidate,
        config=FaultSurfaceFitConfig(mesh_shape=(65, 67)),
    )
    assert len(coarse.vertices) != len(fine.vertices)
    np.testing.assert_array_equal(coarse.model.coefficients, fine.model.coefficients)
    controls = np.array([[15.5, 23.0, 15.5], [7.0, 15.0, 7.0]])
    coarse_edit = refit_fault_surface(coarse.model, controls)
    fine_edit = refit_fault_surface(fine.model, controls)
    np.testing.assert_array_equal(coarse_edit.model.coefficients, fine_edit.model.coefficients)
    for result in (coarse_edit, fine_edit):
        actual = evaluate_fault_surface(result.model, controls[:, result.model.independent_axes])
        np.testing.assert_allclose(actual, controls, atol=1e-5, rtol=0)
