"""Reproducible synthetic benchmarks for reference-like 3D skinning."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyosv._accel import NUMBA_AVAILABLE  # noqa: E402
from pyosv._seed_selection import _select_skinner_seed_indices_3d  # noqa: E402
from pyosv._skinner.candidate_sampling import (  # noqa: E402
    _candidate_slice_numba,
    _candidate_slice_python,
)
from pyosv._skinner.candidate_path import (  # noqa: E402
    _pick_candidate_local_u_path_numba,
    _pick_candidate_local_u_path_python,
)
from pyosv.cells import FaultCell  # noqa: E402
from pyosv.skinner import (  # noqa: E402
    FaultSkinner,
    _candidate_slice_above_below,
    _pick_candidate_local_u_path,
    _update_transform_map,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        type=int,
        choices=(25, 49),
        default=25,
        help="Cubic input size (25 or 49).",
    )
    parser.add_argument("--repeat", type=int, default=3, help="Measured repetitions.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup repetitions.")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum local growth steps.")
    parser.add_argument("--seed", type=int, default=20250389, help="Synthetic input RNG seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if NUMBA_AVAILABLE and args.warmup == 0:
        raise ValueError("warmup must be positive when Numba is available")
    fv, vp, vt, seed = synthetic_skinning_inputs(args.shape, seed=args.seed)
    local_radius = min(5, (args.shape - 1) // 2)
    tangent_radius = (args.shape - 1) // 2
    transform_map = _update_transform_map(
        ru=local_radius,
        rv=tangent_radius,
        rw=tangent_radius,
        normal=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        dip=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        strike=np.array([0.0, 0.0, 1.0], dtype=np.float32),
    )
    center = float((args.shape - 1) // 2)
    candidate_args = (
        fv,
        transform_map.us,
        transform_map.vs,
        transform_map.ws,
        center,
        center,
        center,
        0,
        2 * local_radius,
        tangent_radius,
        tangent_radius,
        1,
        0,
        min(tangent_radius + 1, args.max_steps + 1),
    )

    def generate_candidate_slice() -> np.ndarray:
        return _candidate_slice_above_below(
            fv,
            transform_map,
            (center, center, center),
            ub=0,
            ue=2 * local_radius,
            vc=tangent_radius,
            wc=tangent_radius,
            direction=1,
            max_steps=args.max_steps,
        )

    python_candidate_times, python_candidate_slice = time_repeated(
        lambda: _candidate_slice_python(*candidate_args),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    numba_candidate_times, numba_candidate_slice = time_repeated(
        lambda: _candidate_slice_numba(*candidate_args),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    dispatch_candidate_times, candidate_slice = time_repeated(
        generate_candidate_slice,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    if not (
        np.array_equal(python_candidate_slice, numba_candidate_slice)
        and np.array_equal(python_candidate_slice, candidate_slice)
    ):
        raise RuntimeError("candidate slice benchmark paths produced different outputs")
    path_args = (candidate_slice, 2, 0.1)
    python_path_times, python_local_u_path = time_repeated(
        lambda: _pick_candidate_local_u_path_python(*path_args),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    numba_path_times, numba_local_u_path = time_repeated(
        lambda: _pick_candidate_local_u_path_numba(*path_args),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    dispatch_path_times, local_u_path = time_repeated(
        lambda: _pick_candidate_local_u_path(candidate_slice),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    if not (
        np.array_equal(python_local_u_path, numba_local_u_path)
        and np.array_equal(python_local_u_path, local_u_path)
    ):
        raise RuntimeError("local-u path benchmark paths produced different outputs")

    skinner = FaultSkinner(method="reference")

    def grow_one_skin():
        growth_seed = FaultCell(seed.x1, seed.x2, seed.x3, seed.fl, seed.fp, seed.ft)
        return skinner.find_skin(
            growth_seed,
            fv,
            vp,
            vt,
            min_likelihood=0.5,
            ru=local_radius,
            rv=tangent_radius,
            rw=tangent_radius,
            max_steps=args.max_steps,
        )

    growth_times, skin = time_repeated(
        grow_one_skin,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    skin_indices = skin.indices()
    skin_likelihoods = skin.likelihoods()

    seed_planarity = np.ones_like(fv)
    python_seed_times, python_seed_indices = time_repeated(
        lambda: _select_skinner_seed_indices_3d(
            seed_planarity,
            fv,
            np.float32(0.8),
            np.float32(0.5),
            1,
            use_numba=False,
        ),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    numba_seed_times, numba_seed_indices = time_repeated(
        lambda: _select_skinner_seed_indices_3d(
            seed_planarity,
            fv,
            np.float32(0.8),
            np.float32(0.5),
            1,
            use_numba=True,
        ),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    if not np.array_equal(python_seed_indices, numba_seed_indices):
        raise RuntimeError("Python and Numba seed selectors produced different outputs")
    diagnostics: dict[str, object] = {}

    def find_reference_skins():
        return skinner.find_skins(
            fv,
            vp,
            vt,
            min_likelihood=0.5,
            ep=seed_planarity,
            ft=fv,
            pt=vp,
            tt=vt,
            d=1,
            ru=local_radius,
            rv=tangent_radius,
            rw=tangent_radius,
            max_steps=args.max_steps,
            reskin=False,
            diagnostics=diagnostics,
        )

    reference_times, skins = time_repeated(
        find_reference_skins,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    reference_indices = (
        np.concatenate([accepted.indices() for accepted in skins])
        if skins
        else np.empty((0, 3), dtype=np.int32)
    )
    reference_likelihoods = (
        np.concatenate([accepted.likelihoods() for accepted in skins])
        if skins
        else np.empty(0, dtype=np.float32)
    )

    print(f"benchmark=skinning3d numba_available={NUMBA_AVAILABLE}")
    print(
        " ".join(
            [
                f"input_shape={fv.shape}",
                f"dtype={fv.dtype}",
                f"seeds={diagnostics['seed_count_after_spacing']}",
                f"skins={len(skins)}",
                f"rng_seed={args.seed}",
                f"repeat={args.repeat}",
                f"warmup={args.warmup}",
            ],
        ),
    )
    print(
        f"{timing_summary('candidate_slice_python', python_candidate_times)} "
        f"output_count={candidate_slice.size} output_shape={candidate_slice.shape} "
        f"{array_fingerprint('candidate', candidate_slice)}",
    )
    print(
        f"{timing_summary('candidate_slice_numba', numba_candidate_times)} "
        f"output_count={candidate_slice.size} output_shape={candidate_slice.shape} "
        f"{array_fingerprint('candidate', candidate_slice)}",
    )
    print(
        f"{timing_summary('candidate_slice_dispatch', dispatch_candidate_times)} "
        f"output_count={candidate_slice.size} output_shape={candidate_slice.shape} "
        f"{array_fingerprint('candidate', candidate_slice)}",
    )
    print(
        f"{timing_summary('local_u_path_python', python_path_times)} "
        f"output_count={local_u_path.size} {array_fingerprint('local_u', local_u_path)}",
    )
    print(
        f"{timing_summary('local_u_path_numba', numba_path_times)} "
        f"output_count={local_u_path.size} {array_fingerprint('local_u', local_u_path)}",
    )
    print(
        f"{timing_summary('local_u_path_dispatch', dispatch_path_times)} "
        f"output_count={local_u_path.size} {array_fingerprint('local_u', local_u_path)}",
    )
    print(
        f"{timing_summary('single_seed_growth', growth_times)} "
        f"output_count={len(skin)} {array_fingerprint('skin_likelihood', skin_likelihoods)}",
    )
    print(array_fingerprint("skin_indices", skin_indices))
    print(
        f"{timing_summary('seed_selector_python', python_seed_times)} "
        f"candidate_count={diagnostics['seed_candidate_count_before_spacing']} "
        f"accepted_count={python_seed_indices.size}",
    )
    print(
        f"{timing_summary('seed_selector_numba', numba_seed_times)} "
        f"candidate_count={diagnostics['seed_candidate_count_before_spacing']} "
        f"accepted_count={numba_seed_indices.size}",
    )
    print(
        f"{timing_summary('reference_skinning', reference_times)} "
        f"output_count={reference_likelihoods.size} "
        f"{array_fingerprint('reference_likelihood', reference_likelihoods)}",
    )
    print(array_fingerprint("reference_indices", reference_indices))
    print(
        " ".join(
            [
                f"seed_candidates={diagnostics['seed_candidate_count_before_spacing']}",
                f"seeds_after_spacing={diagnostics['seed_count_after_spacing']}",
                f"grow_attempts={diagnostics['grow_attempt_count']}",
                "seed_rejected_by_occupied="
                f"{diagnostics['seed_count_rejected_by_occupied']}",
                f"accepted_skins={diagnostics['accepted_skin_count']}",
                f"accepted_cells={diagnostics['accepted_cell_count']}",
                "accepted_occupancy=dense_bool_mask",
                f"occupancy_bytes={fv.size * np.dtype(np.bool_).itemsize}",
            ],
        ),
    )
    return 0


def synthetic_skinning_inputs(
    shape: int,
    *,
    seed: int = 20250389,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, FaultCell]:
    rng = np.random.default_rng(seed)
    volume_shape = (shape, shape, shape)
    fv = np.zeros(volume_shape, dtype=np.float32)
    vp = np.zeros(volume_shape, dtype=np.float32)
    vt = np.full(volume_shape, 90.0, dtype=np.float32)
    center = (shape - 1) // 2
    margin = max(2, shape // 8)
    patch_shape = (shape - 2 * margin, shape - 2 * margin)
    likelihood = np.float32(0.85) + np.float32(0.1) * rng.random(
        patch_shape,
        dtype=np.float32,
    )
    fv[margin : shape - margin, center, margin : shape - margin] = likelihood
    seed_cell = FaultCell(center, center, center, fv[center, center, center], 0.0, 90.0)
    return fv, vp, vt, seed_cell


def time_repeated(func, *, repeat: int, warmup: int):
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")

    result = None
    for _ in range(warmup):
        result = func()

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - start)

    return times, result


def timing_summary(name: str, times: list[float]) -> str:
    return (
        f"name={name} min_seconds={min(times):.6f} "
        f"median_seconds={float(np.median(times)):.6f}"
    )


def array_fingerprint(name: str, array: np.ndarray) -> str:
    values = np.asarray(array)
    if values.size == 0:
        return f"{name}_sum=0 {name}_min=nan {name}_max=nan"
    return " ".join(
        [
            f"{name}_sum={float(np.sum(values, dtype=np.float64)):.9g}",
            f"{name}_min={float(np.min(values)):.9g}",
            f"{name}_max={float(np.max(values)):.9g}",
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
