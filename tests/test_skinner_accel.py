import math

import numpy as np
import pytest

from pyosv import skinner
from pyosv._accel import NUMBA_AVAILABLE
from pyosv._skinner.candidate_sampling import (
    _candidate_slice_numba,
    _candidate_slice_python,
)
from pyosv._skinner.candidate_path import (
    _pick_candidate_local_u_path_numba,
    _pick_candidate_local_u_path_python,
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


def _candidate_path_oracle(
    candidate_slice: np.ndarray,
    max_jump: int,
    jump_penalty: float,
) -> np.ndarray:
    nrow, nu = candidate_slice.shape
    if nrow == 0:
        return np.empty(0, dtype=np.int32)
    if nu == 1:
        return np.zeros(nrow, dtype=np.int32)

    max_jump_int = min(max_jump, nu - 1)
    accumulated = np.empty((nrow, nu), dtype=np.float32)
    predecessor = np.full((nrow, nu), -1, dtype=np.int32)
    accumulated[0] = candidate_slice[0]
    for irow in range(1, nrow):
        for iu in range(nu):
            ib = max(0, iu - max_jump_int)
            ie = min(nu, iu + max_jump_int + 1)
            best_score = -math.inf
            best_previous = ib
            best_jump = nu
            for ju in range(ib, ie):
                jump = abs(iu - ju)
                score = float(accumulated[irow - 1, ju]) - float(jump_penalty) * jump
                if score > best_score or (
                    score == best_score
                    and (
                        jump < best_jump
                        or (
                            jump == best_jump
                            and (abs(2 * ju - (nu - 1)), ju)
                            < (abs(2 * best_previous - (nu - 1)), best_previous)
                        )
                    )
                ):
                    best_score = score
                    best_previous = ju
                    best_jump = jump
            accumulated[irow, iu] = candidate_slice[irow, iu] + best_score
            predecessor[irow, iu] = best_previous

    best_u = 0
    best_score = float(accumulated[-1, 0])
    for iu in range(1, nu):
        score = float(accumulated[-1, iu])
        if score > best_score or (
            score == best_score
            and (abs(2 * iu - (nu - 1)), iu) < (abs(2 * best_u - (nu - 1)), best_u)
        ):
            best_u = iu
            best_score = score

    path = np.empty(nrow, dtype=np.int32)
    iu = best_u
    for irow in range(nrow - 1, -1, -1):
        path[irow] = iu
        previous = predecessor[irow, iu]
        if previous >= 0:
            iu = int(previous)
    return path


_PATH_CASES = [
    ("empty", np.empty((0, 4), dtype=np.float32), 2, 0.1),
    ("one_column", np.array([[1.0], [-2.0], [3.0]], dtype=np.float32), 2, 0.1),
    ("one_row_even", np.array([[-2.0, 1.0, 1.0, -2.0]], dtype=np.float32), 2, 0.1),
    ("one_row_odd", np.array([[-2.0, 1.0, 2.0, 1.0, -2.0]], dtype=np.float32), 2, 0.1),
    ("all_equal_even", np.ones((4, 4), dtype=np.float32), 2, 0.1),
    ("all_equal_odd", np.ones((4, 5), dtype=np.float32), 2, 0.1),
    (
        "symmetric",
        np.array([[0.0, 2.0, 0.0, 2.0, 0.0], [1.0, 0.0, 3.0, 0.0, 1.0]], dtype=np.float32),
        2,
        0.1,
    ),
    (
        "negative_scores",
        np.array([[-4.0, -2.0, -3.0], [-5.0, -1.0, -6.0]], dtype=np.float32),
        2,
        0.1,
    ),
    (
        "large_penalty",
        np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 9.0], [5.0, 0.0, 0.0]], dtype=np.float32),
        2,
        1.0e12,
    ),
    (
        "zero_penalty",
        np.array([[1.0, 4.0, 2.0, 4.0], [5.0, 0.0, 5.0, 0.0]], dtype=np.float32),
        2,
        0.0,
    ),
    ("max_jump_zero", np.arange(15, dtype=np.float32).reshape(3, 5), 0, 0.1),
    ("max_jump_one", np.arange(15, dtype=np.float32).reshape(3, 5), 1, 0.1),
    ("max_jump_nu", np.arange(15, dtype=np.float32).reshape(3, 5), 5, 0.1),
]


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


@pytest.mark.parametrize(
    ("candidate_slice", "max_jump", "jump_penalty"),
    [(case[1], case[2], case[3]) for case in _PATH_CASES],
    ids=[case[0] for case in _PATH_CASES],
)
def test_candidate_path_python_matches_previous_implementation(
    candidate_slice: np.ndarray,
    max_jump: int,
    jump_penalty: float,
) -> None:
    expected = _candidate_path_oracle(candidate_slice, max_jump, jump_penalty)
    actual = _pick_candidate_local_u_path_python(candidate_slice, max_jump, jump_penalty)

    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is not installed")
@pytest.mark.parametrize(
    ("candidate_slice", "max_jump", "jump_penalty"),
    [(case[1], case[2], case[3]) for case in _PATH_CASES],
    ids=[case[0] for case in _PATH_CASES],
)
def test_candidate_path_numba_matches_python_and_previous_implementation(
    candidate_slice: np.ndarray,
    max_jump: int,
    jump_penalty: float,
) -> None:
    expected = _candidate_path_oracle(candidate_slice, max_jump, jump_penalty)
    python_path = _pick_candidate_local_u_path_python(
        candidate_slice,
        max_jump,
        jump_penalty,
    )
    numba_path = _pick_candidate_local_u_path_numba(
        candidate_slice,
        max_jump,
        jump_penalty,
    )

    assert numba_path.dtype == np.int32
    np.testing.assert_array_equal(numba_path, expected)
    np.testing.assert_array_equal(numba_path, python_path)


@pytest.mark.parametrize(
    ("candidate_slice", "expected"),
    [
        (
            np.array([[1.0, 1.0, 1.0], [-100.0, 10.0, -100.0]], dtype=np.float32),
            np.array([1, 1], dtype=np.int32),
        ),
        (
            np.array(
                [
                    [-100.0, -100.0, 1.0, -100.0, 1.0, -100.0],
                    [-100.0, -100.0, -100.0, 10.0, -100.0, -100.0],
                ],
                dtype=np.float32,
            ),
            np.array([2, 3], dtype=np.int32),
        ),
        (
            np.array(
                [[-100.0, 1.0, -100.0, 1.0, -100.0], [-100.0, -100.0, 10.0, -100.0, -100.0]],
                dtype=np.float32,
            ),
            np.array([1, 2], dtype=np.int32),
        ),
    ],
    ids=["smaller_jump", "central_predecessor", "smaller_u_predecessor"],
)
def test_candidate_path_predecessor_tie_break_stages(
    candidate_slice: np.ndarray,
    expected: np.ndarray,
) -> None:
    python_path = _pick_candidate_local_u_path_python(candidate_slice, 1, 0.0)
    np.testing.assert_array_equal(python_path, expected)
    if NUMBA_AVAILABLE:
        numba_path = _pick_candidate_local_u_path_numba(candidate_slice, 1, 0.0)
        np.testing.assert_array_equal(numba_path, expected)


@pytest.mark.parametrize("use_numba", [False, True])
def test_candidate_path_dispatch_uses_default_parameters(
    monkeypatch: pytest.MonkeyPatch,
    use_numba: bool,
) -> None:
    if use_numba and not NUMBA_AVAILABLE:
        pytest.skip("Numba is not installed")
    candidate_slice = np.array(
        [[0.0, 2.0, 1.0, 2.0], [3.0, 0.0, 3.0, 0.0], [0.0, 4.0, 0.0, 4.0]],
        dtype=np.float32,
    )
    monkeypatch.setattr("pyosv._skinner.growth.NUMBA_AVAILABLE", use_numba)

    actual = skinner._pick_candidate_local_u_path(candidate_slice)
    expected = _candidate_path_oracle(candidate_slice, 2, 0.1)

    np.testing.assert_array_equal(actual, expected)


def test_pick_candidate_us_preserves_offset_dtype_and_input_conversion() -> None:
    candidate_slice = np.array([[0.0, 3.0, 0.0], [1.0, 0.0, 2.0]], dtype=np.float64)

    actual = skinner._pick_candidate_us(ub=7, candidate_slice=candidate_slice)

    assert actual.dtype == np.int32
    np.testing.assert_array_equal(actual, np.array([8, 9], dtype=np.int32))


@pytest.mark.parametrize(
    ("ub", "candidate_slice", "message"),
    [
        (-1, np.ones((2, 2), dtype=np.float32), "ub must be a nonnegative integer"),
        (0, np.ones(2, dtype=np.float32), "candidate_slice must be a 2D array"),
        (
            0,
            np.empty((2, 0), dtype=np.float32),
            "candidate_slice must contain at least one u sample",
        ),
        (
            0,
            np.array([[1.0, np.nan]], dtype=np.float32),
            "candidate_slice must contain only finite values",
        ),
    ],
)
def test_pick_candidate_us_validation_is_preserved(
    ub: int,
    candidate_slice: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        skinner._pick_candidate_us(ub, candidate_slice)


@pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is not installed")
def test_reference_skinning_candidate_backends_preserve_cells_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fv = np.zeros((13, 13, 13), dtype=np.float32)
    vp = np.zeros_like(fv)
    vt = np.full_like(fv, 90.0)
    fv[3:10, 6, 3:10] = 0.9
    fault_skinner = skinner.FaultSkinner(min_skin_size=1)

    def run(use_numba: bool) -> tuple[list[list[tuple[int, int, int]]], dict[str, object]]:
        monkeypatch.setattr("pyosv._skinner.growth.NUMBA_AVAILABLE", use_numba)
        diagnostics: dict[str, object] = {}
        skins = fault_skinner.find_skins(
            fv,
            vp,
            vt,
            min_likelihood=0.5,
            ru=5,
            rv=6,
            rw=6,
            max_steps=6,
            reskin=False,
            diagnostics=diagnostics,
        )
        return [[cell.index for cell in skin] for skin in skins], diagnostics

    python_cells, python_diagnostics = run(False)
    numba_cells, numba_diagnostics = run(True)

    assert numba_cells == python_cells
    assert numba_diagnostics == python_diagnostics
