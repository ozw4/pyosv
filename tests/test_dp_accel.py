import numpy as np
import pytest

from pyosv import dp


pytestmark = pytest.mark.skipif(
    not dp.NUMBA_AVAILABLE,
    reason="Numba acceleration is optional",
)


def test_numba_accumulate_2d_matches_python_fallback() -> None:
    rng = np.random.default_rng(7801)
    cost = rng.normal(size=(9, 7)).astype(np.float32)

    for direction in (-1, 1):
        fallback = dp._accumulate_2d_python(cost, 3, direction)
        accelerated = dp._accumulate_2d_numba(cost, 3, direction)

        assert accelerated.dtype == np.float32
        np.testing.assert_allclose(accelerated, fallback, rtol=1e-6, atol=1e-6)


def test_numba_backtrack_reverse_2d_matches_python_fallback() -> None:
    rng = np.random.default_rng(7802)
    cost = rng.normal(size=(11, 9)).astype(np.float32)
    accumulated = dp._accumulate_2d_python(cost, 2, 1)

    fallback = dp._backtrack_2d_python(
        accumulated,
        cost,
        -4,
        2,
        -1,
    )
    accelerated = dp._backtrack_2d_numba(
        accumulated,
        cost,
        -4,
        2,
        -1,
    )

    assert accelerated.dtype == np.float32
    np.testing.assert_allclose(accelerated, fallback, rtol=1e-6, atol=1e-6)


def test_numba_public_path_pipeline_matches_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cost = _cost_image()

    monkeypatch.setattr(dp, "NUMBA_AVAILABLE", False)
    fallback = dp.find_path_2d(
        cost,
        lmin=-3,
        bstrain=2,
        attribute_smoothing=1,
    )

    monkeypatch.setattr(dp, "NUMBA_AVAILABLE", True)
    accelerated = dp.find_path_2d(
        cost,
        lmin=-3,
        bstrain=2,
        attribute_smoothing=1,
    )

    assert accelerated.dtype == np.float32
    np.testing.assert_allclose(accelerated, fallback, rtol=1e-6, atol=1e-6)


def test_numba_public_surface_pipeline_matches_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cost = np.stack(
        [
            _cost_image(offset=0.0),
            _cost_image(offset=0.25),
            _cost_image(offset=-0.25),
        ],
    ).astype(np.float32)

    monkeypatch.setattr(dp, "NUMBA_AVAILABLE", False)
    fallback = dp.find_surface_3d(
        cost,
        lmin=-3,
        bstrain1=2,
        bstrain2=1,
        attribute_smoothing=1,
    )

    monkeypatch.setattr(dp, "NUMBA_AVAILABLE", True)
    accelerated = dp.find_surface_3d(
        cost,
        lmin=-3,
        bstrain1=2,
        bstrain2=1,
        attribute_smoothing=1,
    )

    assert accelerated.dtype == np.float32
    np.testing.assert_allclose(accelerated, fallback, rtol=1e-6, atol=1e-6)


def test_numba_masked_accumulation_and_backtracking_match_python() -> None:
    rng = np.random.default_rng(7803)
    cost = rng.normal(size=(11, 7)).astype(np.float32)
    valid_mask = np.zeros_like(cost, dtype=np.bool_)
    for ii in range(cost.shape[0]):
        lower = 1 + ii % 2
        upper = 5 - (ii + 1) % 2
        valid_mask[ii, lower : upper + 1] = True

    fallback_accumulated = dp._accumulate_2d_masked_python(cost, valid_mask, 2, 1)
    accelerated_accumulated = dp._accumulate_2d_masked_numba(cost, valid_mask, 2, 1)

    np.testing.assert_array_equal(
        np.isfinite(accelerated_accumulated),
        np.isfinite(fallback_accumulated),
    )
    np.testing.assert_allclose(
        accelerated_accumulated,
        fallback_accumulated,
        rtol=1e-6,
        atol=1e-6,
    )

    fallback_path, fallback_feasible = dp._backtrack_2d_masked_python(
        fallback_accumulated,
        cost,
        valid_mask,
        -3,
        2,
        -1,
    )
    accelerated_path, accelerated_feasible = dp._backtrack_2d_masked_numba(
        accelerated_accumulated,
        cost,
        valid_mask,
        -3,
        2,
        -1,
    )

    assert accelerated_feasible is fallback_feasible
    assert accelerated_feasible
    np.testing.assert_allclose(accelerated_path, fallback_path, rtol=1e-6, atol=1e-6)


def test_numba_masked_surface_pipeline_matches_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(7804)
    nw, nv, nu = 3, 9, 7
    lmin = -3
    cost = rng.random((nw, nv, nu), dtype=np.float32)
    valid_mask = np.zeros_like(cost, dtype=np.bool_)
    for iw in range(nw):
        for iv in range(nv):
            lower = 1 + (iw + iv) % 2
            upper = 5 - (iw + iv + 1) % 2
            valid_mask[iw, iv, lower : upper + 1] = True
    cost[~valid_mask] = -1_000.0
    cost[:, :, -lmin] = 0.0

    monkeypatch.setattr(dp, "NUMBA_AVAILABLE", False)
    fallback, fallback_projection_count = dp._find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=lmin,
        bstrain1=2,
        bstrain2=2,
        attribute_smoothing=1,
        surface_smoothing1=0.75,
        surface_smoothing2=0.5,
    )

    monkeypatch.setattr(dp, "NUMBA_AVAILABLE", True)
    accelerated, accelerated_projection_count = dp._find_surface_3d_masked(
        cost,
        valid_mask,
        lmin=lmin,
        bstrain1=2,
        bstrain2=2,
        attribute_smoothing=1,
        surface_smoothing1=0.75,
        surface_smoothing2=0.5,
    )

    assert fallback is not None
    assert accelerated is not None
    assert accelerated_projection_count == fallback_projection_count
    assert accelerated.dtype == np.float32
    np.testing.assert_allclose(accelerated, fallback, rtol=1e-6, atol=1e-6)
    selected_indices = np.floor(accelerated.astype(np.float64) - lmin + 0.5).astype(np.intp)
    selected_valid = np.take_along_axis(valid_mask, selected_indices[:, :, None], axis=2)
    assert selected_valid.all()


def test_numba_masked_surface_post_smoothing_strain_recovery_matches_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cost = np.zeros((1, 4, 5), dtype=np.float32)
    valid_mask = np.zeros_like(cost, dtype=np.bool_)
    for iv, (start, stop) in enumerate(((4, 4), (0, 4), (2, 3), (2, 3))):
        valid_mask[0, iv, start : stop + 1] = True
    kwargs = {
        "lmin": -2,
        "bstrain1": 4,
        "bstrain2": 4,
        "attribute_smoothing": 0,
        "surface_smoothing1": 2.0,
        "surface_smoothing2": 2.0,
    }

    monkeypatch.setattr(dp, "NUMBA_AVAILABLE", False)
    fallback, fallback_projection_count = dp._find_surface_3d_masked(
        cost,
        valid_mask,
        **kwargs,
    )
    monkeypatch.setattr(dp, "NUMBA_AVAILABLE", True)
    accelerated, accelerated_projection_count = dp._find_surface_3d_masked(
        cost,
        valid_mask,
        **kwargs,
    )

    assert fallback is not None
    assert accelerated is not None
    assert accelerated_projection_count == fallback_projection_count
    np.testing.assert_allclose(accelerated, fallback, rtol=1.0e-6, atol=1.0e-6)
    for surface in (fallback, accelerated):
        assert surface.dtype == np.float32
        assert dp._surface_respects_masked_strain(
            surface,
            valid_mask,
            lmin=-2,
            bstrain1=4,
            bstrain2=4,
        )
        assert np.max(np.abs(np.diff(surface, axis=1))) <= np.float32(0.25 + 1.0e-6)


def test_numba_masked_projection_matches_python_at_java_rounding_cells() -> None:
    upper_predecessor = np.nextafter(np.float32(2.5), np.float32(-np.inf))
    cases: list[tuple[np.ndarray, np.ndarray, int]] = []

    isolated = np.zeros((1, 1, 5), dtype=np.bool_)
    isolated[0, 0, 2] = True
    for value in (np.float32(1.5), np.float32(2.5), upper_predecessor):
        cases.append((np.array([[value]], dtype=np.float32), isolated, 0))

    endpoints = np.zeros((1, 2, 5), dtype=np.bool_)
    endpoints[0, 0, 0] = True
    endpoints[0, 1, 4] = True
    cases.append((np.array([[-0.25, 4.25]], dtype=np.float32), endpoints, 0))

    tie_mask = np.zeros((1, 1, 3), dtype=np.bool_)
    tie_mask[0, 0, (0, 2)] = True
    cases.append((np.array([[1.0]], dtype=np.float32), tie_mask, 0))

    for surface, valid_mask, lmin in cases:
        fallback = dp._project_surface_to_valid_mask_python(surface, valid_mask, lmin)
        accelerated = dp._project_surface_to_valid_mask_numba(surface, valid_mask, lmin)

        assert accelerated[1:] == fallback[1:]
        np.testing.assert_array_equal(accelerated[0], fallback[0])


def _cost_image(offset: float = 0.0) -> np.ndarray:
    ni = 13
    nl = 7
    lmin = -3
    path = np.linspace(-1.5 + offset, 1.5 + offset, ni, dtype=np.float32)
    lags = lmin + np.arange(nl, dtype=np.float32)
    trend = 0.05 * np.arange(ni, dtype=np.float32)[:, None]
    return ((lags[None, :] - path[:, None]) ** 2 + trend).astype(np.float32)
