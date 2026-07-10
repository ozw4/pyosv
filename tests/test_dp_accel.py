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


def _cost_image(offset: float = 0.0) -> np.ndarray:
    ni = 13
    nl = 7
    lmin = -3
    path = np.linspace(-1.5 + offset, 1.5 + offset, ni, dtype=np.float32)
    lags = lmin + np.arange(nl, dtype=np.float32)
    trend = 0.05 * np.arange(ni, dtype=np.float32)[:, None]
    return ((lags[None, :] - path[:, None]) ** 2 + trend).astype(np.float32)
