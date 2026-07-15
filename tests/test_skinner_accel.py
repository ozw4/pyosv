import math

import numpy as np
import pytest

from pyosv import skinner
from pyosv._accel import NUMBA_AVAILABLE
from pyosv._skinner.candidate_sampling import (
    _candidate_slice_numba,
    _candidate_slice_python,
)
from pyosv._skinner.models import _LocalTransformMap
from pyosv._skinner.transforms import (
    _local_index_to_world,
    _sample_validated_volume_nearest_java_round,
)


def _identity_transform(radius: int) -> _LocalTransformMap:
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    zeros = np.zeros_like(offsets)
    return _LocalTransformMap(
        us=np.stack((offsets, zeros, zeros)),
        vs=np.stack((zeros, offsets, zeros)),
        ws=np.stack((zeros, zeros, offsets)),
    )


def _candidate_oracle(
    fv: np.ndarray,
    transform_map: _LocalTransformMap,
    origin: tuple[float, float, float],
    ub: int,
    ue: int,
    vc: int,
    wc: int,
    direction: int,
    axis: str,
    max_steps: int | None,
) -> np.ndarray:
    center = vc if axis == "v" else wc
    axis_size = transform_map.vs.shape[1] if axis == "v" else transform_map.ws.shape[1]
    distance_to_edge = center if direction < 0 else axis_size - 1 - center
    row_count = distance_to_edge + 1
    if max_steps is not None:
        row_count = min(row_count, max_steps + 1)
    samples = np.zeros((row_count, ue - ub + 1), dtype=np.float32)
    for row in range(row_count):
        iv = vc + direction * row if axis == "v" else vc
        iw = wc + direction * row if axis == "w" else wc
        for col, iu in enumerate(range(ub, ue + 1)):
            world = _local_index_to_world(iu, iv, iw, origin, transform_map)
            samples[row, col] = _sample_validated_volume_nearest_java_round(fv, *world)
    return samples


@pytest.mark.parametrize("axis", ["v", "w"])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("max_steps", [None, 0, 1, 3])
def test_candidate_slice_dispatch_matches_previous_sampling_order(
    monkeypatch: pytest.MonkeyPatch,
    axis: str,
    direction: int,
    max_steps: int | None,
) -> None:
    fv = np.arange(7 * 7 * 7, dtype=np.float32).reshape(7, 7, 7)
    transform_map = _identity_transform(2)
    args = (fv, transform_map, (3.0, 3.0, 3.0), 0, 4, 2, 2, direction, axis, max_steps)
    expected = _candidate_oracle(*args)

    monkeypatch.setattr("pyosv._skinner.growth.NUMBA_AVAILABLE", False)
    actual = skinner._candidate_slice(*args)

    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is not installed")
def test_candidate_slice_numba_matches_python_at_boundaries_and_half_samples() -> None:
    fv = np.arange(5 * 5 * 5, dtype=np.float32).reshape(5, 5, 5)
    offsets = np.array([-3.0, -1.5, -0.5, 0.5, 1.5, 4.0, 5.0], dtype=np.float32)
    zeros = np.zeros_like(offsets)
    us = np.stack((offsets, zeros, zeros))
    vs = np.stack((zeros, offsets, zeros))
    ws = np.stack((zeros, zeros, offsets))
    args = (fv, us, vs, ws, 0.0, 0.0, 0.0, 0, 6, 3, 3, 1, 0, 3)

    expected = _candidate_slice_python(*args)
    actual = _candidate_slice_numba(*args)

    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous
    assert actual[0, 1] == 0.0
    assert actual[0, 2] == fv[1, 1, 0]
    assert actual[0, 3] == fv[1, 1, 1]
    assert actual[0, 5] == fv[1, 1, 4]
    assert actual[0, 6] == 0.0


def test_candidate_slice_returns_zeros_when_world_grid_is_entirely_outside() -> None:
    fv = np.ones((3, 3, 3), dtype=np.float32)
    transform_map = _identity_transform(1)

    actual = skinner._candidate_slice(
        fv,
        transform_map,
        (-10.0, -10.0, -10.0),
        0,
        2,
        1,
        1,
        1,
        "v",
        None,
    )

    np.testing.assert_array_equal(actual, np.zeros((2, 3), dtype=np.float32))


@pytest.mark.parametrize("use_numba", [False, True])
def test_candidate_slice_dispatch_backend_result_is_identical(
    monkeypatch: pytest.MonkeyPatch,
    use_numba: bool,
) -> None:
    if use_numba and not NUMBA_AVAILABLE:
        pytest.skip("Numba is not installed")
    fv = np.arange(5 * 5 * 5, dtype=np.float32).reshape(5, 5, 5)
    transform_map = _identity_transform(2)
    monkeypatch.setattr("pyosv._skinner.growth.NUMBA_AVAILABLE", use_numba)

    actual = skinner._candidate_slice(fv, transform_map, (2.0, 2.0, 2.0), 0, 4, 2, 2, -1, "w", 2)
    expected = _candidate_oracle(fv, transform_map, (2.0, 2.0, 2.0), 0, 4, 2, 2, -1, "w", 2)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ue": 0, "ub": 1}, "ue must be greater than or equal to ub"),
        ({"direction": 0}, "direction must be -1 or 1"),
        ({"axis": "u"}, "axis must be 'v' or 'w'"),
        ({"max_steps": -1}, "max_steps must be a nonnegative integer"),
        ({"origin": (math.nan, 0.0, 0.0)}, "origin must contain only finite values"),
    ],
)
def test_candidate_slice_validation_is_preserved(
    overrides: dict[str, object],
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "fv": np.ones((3, 3, 3), dtype=np.float32),
        "transform_map": _identity_transform(1),
        "origin": (1.0, 1.0, 1.0),
        "ub": 0,
        "ue": 2,
        "vc": 1,
        "wc": 1,
        "direction": 1,
        "axis": "v",
        "max_steps": 1,
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        skinner._candidate_slice(**kwargs)
