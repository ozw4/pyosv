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

    candidate_times, candidate_slice = time_repeated(
        generate_candidate_slice,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    path_times, local_u_path = time_repeated(
        lambda: _pick_candidate_local_u_path(candidate_slice),
        repeat=args.repeat,
        warmup=args.warmup,
    )

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

    print(f"benchmark=skinning3d numba_available={NUMBA_AVAILABLE}")
    print(
        " ".join(
            [
                f"input_shape={fv.shape}",
                f"dtype={fv.dtype}",
                "seeds=1",
                f"skins={int(len(skin) > 0)}",
                f"rng_seed={args.seed}",
                f"repeat={args.repeat}",
                f"warmup={args.warmup}",
            ],
        ),
    )
    print(
        f"{timing_summary('candidate_slice', candidate_times)} "
        f"output_count={candidate_slice.size} output_shape={candidate_slice.shape} "
        f"{array_fingerprint('candidate', candidate_slice)}",
    )
    print(
        f"{timing_summary('local_u_path', path_times)} "
        f"output_count={local_u_path.size} {array_fingerprint('local_u', local_u_path)}",
    )
    print(
        f"{timing_summary('single_seed_growth', growth_times)} "
        f"output_count={len(skin)} {array_fingerprint('skin_likelihood', skin_likelihoods)}",
    )
    print(array_fingerprint("skin_indices", skin_indices))
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
