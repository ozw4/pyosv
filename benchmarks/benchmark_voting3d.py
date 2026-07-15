"""Small synthetic benchmark for 3D optimal-surface voting."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyosv._accel import NUMBA_AVAILABLE  # noqa: E402
from pyosv._seed_selection import _select_voter_seed_indices_3d  # noqa: E402
from pyosv._voting3d.orientation import _surface_strike_and_dip  # noqa: E402
from pyosv.voting3d import OptimalSurfaceVoter  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n3", type=int, default=17, help="Output axis-3 sample count.")
    parser.add_argument("--n2", type=int, default=17, help="Output axis-2 sample count.")
    parser.add_argument("--n1", type=int, default=17, help="Output axis-1 sample count.")
    parser.add_argument("--repeat", type=int, default=3, help="Measured repetitions.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup repetitions.")
    parser.add_argument("--ru", type=int, default=1, help="Voting half-width along local u.")
    parser.add_argument("--rv", type=int, default=2, help="Voting half-width along local v.")
    parser.add_argument("--rw", type=int, default=2, help="Voting half-width along local w.")
    parser.add_argument("--d", type=int, default=5, help="Seed exclusion distance.")
    parser.add_argument("--fm", type=float, default=0.7, help="Seed likelihood threshold.")
    parser.add_argument(
        "--candidate-density",
        type=float,
        default=0.02,
        help="Fraction of samples generated above the seed threshold.",
    )
    parser.add_argument("--seed", type=int, default=20250389, help="Synthetic input RNG seed.")
    parser.add_argument(
        "--orientation-nw",
        type=int,
        default=65,
        help="Orientation microbenchmark surface w sample count.",
    )
    parser.add_argument(
        "--orientation-nv",
        type=int,
        default=65,
        help="Orientation microbenchmark surface v sample count.",
    )
    parser.add_argument(
        "--orientation-sigma",
        type=float,
        default=16.0,
        help="Orientation microbenchmark Gaussian sigma.",
    )
    parser.add_argument(
        "--orientation-iterations",
        type=int,
        default=100,
        help="Orientation calls per measured repetition.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if NUMBA_AVAILABLE and args.warmup == 0:
        raise ValueError("warmup must be positive when Numba is available")
    ft, pt, tt = synthetic_fault_likelihood(
        args.n3,
        args.n2,
        args.n1,
        candidate_density=args.candidate_density,
        seed=args.seed,
    )
    voter = OptimalSurfaceVoter(ru=args.ru, rv=args.rv, rw=args.rw)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)

    orientation_surface = np.random.default_rng(args.seed + 1).normal(
        size=(args.orientation_nw, args.orientation_nv),
    ).astype(np.float32)
    orientation_axes = np.eye(3, dtype=np.float32)
    if args.orientation_iterations <= 0:
        raise ValueError("orientation-iterations must be positive")

    def run_orientation(backend: str) -> tuple[float, float]:
        result = (0.0, 0.0)
        for _ in range(args.orientation_iterations):
            result = _surface_strike_and_dip(
                orientation_axes[0],
                orientation_axes[1],
                orientation_axes[2],
                orientation_surface,
                sigma=args.orientation_sigma,
                backend=backend,
            )
        return result

    full_orientation_times, full_orientation = time_repeated(
        lambda: run_orientation("full_surface"),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    center_orientation_times, center_orientation = time_repeated(
        lambda: run_orientation("center_separable"),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    full_orientation_times = [
        value / args.orientation_iterations for value in full_orientation_times
    ]
    center_orientation_times = [
        value / args.orientation_iterations for value in center_orientation_times
    ]

    seed_times, seeds = time_repeated(
        lambda: voter.pick_seeds(args.d, args.fm, ft, pt, tt),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    threshold = np.float32(args.fm)
    python_seed_times, python_seed_indices = time_repeated(
        lambda: _select_voter_seed_indices_3d(
            ft,
            threshold,
            args.d,
            use_numba=False,
        ),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    numba_seed_times, numba_seed_indices = time_repeated(
        lambda: _select_voter_seed_indices_3d(
            ft,
            threshold,
            args.d,
            use_numba=True,
        ),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    if not np.array_equal(python_seed_indices, numba_seed_indices):
        raise RuntimeError("Python and Numba seed selectors produced different outputs")

    if seeds:
        sample_cell = seeds[0]
        sample_index = sample_cell.index
        sample_normal = sample_cell.fault_normal()
        sample_dip = sample_cell.fault_dip_vector()
        sample_strike = sample_cell.fault_strike_vector()
    else:
        sample_index = (args.n1 // 2, args.n2 // 2, args.n3 // 2)
        sample_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        sample_dip = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sample_strike = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    sampling_times, samples = time_repeated(
        lambda: voter._samples_in_uvw_box_reference_with_support(
            *sample_index,
            sample_normal,
            sample_dip,
            sample_strike,
            ft,
        ),
        repeat=args.repeat,
        warmup=args.warmup,
    )

    def run_once() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return voter.apply_voting(d=args.d, fm=args.fm, ft=ft, pt=pt, tt=tt)

    voting_times, result = time_repeated(run_once, repeat=args.repeat, warmup=args.warmup)
    fv, vp, vt = result
    candidate_count = int(np.count_nonzero(ft > np.float32(args.fm)))
    seed_indices = np.asarray([cell.index for cell in seeds], dtype=np.int32)

    print(f"benchmark=3d_voting numba_available={NUMBA_AVAILABLE}")
    print(
        " ".join(
            [
                f"input_shape={ft.shape}",
                f"output_shapes={(fv.shape, vp.shape, vt.shape)}",
                f"dtype={fv.dtype}",
                f"candidates={candidate_count}",
                f"seeds={len(seeds)}",
                f"candidate_density={args.candidate_density:.6g}",
                f"suppression_distance={args.d}",
                f"rng_seed={args.seed}",
            ],
        ),
    )
    print(f"repeat={args.repeat} warmup={args.warmup}")
    orientation_strike_difference = abs(
        ((center_orientation[0] - full_orientation[0] + 180.0) % 360.0) - 180.0,
    )
    orientation_dip_difference = abs(center_orientation[1] - full_orientation[1])
    orientation_speedup = float(np.median(full_orientation_times)) / float(
        np.median(center_orientation_times),
    )
    print(
        " ".join(
            [
                f"orientation_surface_shape={orientation_surface.shape}",
                f"orientation_sigma={args.orientation_sigma:.6g}",
                f"orientation_iterations={args.orientation_iterations}",
                f"strike_difference_degrees={orientation_strike_difference:.9g}",
                f"dip_difference_degrees={orientation_dip_difference:.9g}",
            ],
        ),
    )
    print(timing_summary("orientation_full_surface_per_call", full_orientation_times))
    print(timing_summary("orientation_center_separable_per_call", center_orientation_times))
    print(f"orientation_center_speedup={orientation_speedup:.3f}")
    print(
        f"{timing_summary('pick_seeds', seed_times)} "
        f"output_count={len(seeds)} {array_fingerprint('seed_indices', seed_indices)}",
    )
    print(
        f"{timing_summary('seed_selector_python', python_seed_times)} "
        f"candidate_count={candidate_count} accepted_count={python_seed_indices.size}",
    )
    print(
        f"{timing_summary('seed_selector_numba', numba_seed_times)} "
        f"candidate_count={candidate_count} accepted_count={numba_seed_indices.size}",
    )
    print(
        f"{timing_summary('reference_uvw_sampling_per_seed', sampling_times)} "
        f"admissible_lag_count={samples.admissible_lag_count} "
        f"in_bounds_lag_count={samples.in_bounds_lag_count} "
        f"{array_fingerprint('cost', samples.cost)}",
    )
    print(
        f"{timing_summary('apply_voting', voting_times)} "
        f"output_count={int(np.count_nonzero(fv))} {array_fingerprint('fv', fv)}",
    )
    print(array_fingerprint("vp", vp))
    print(array_fingerprint("vt", vt))
    return 0


def synthetic_fault_likelihood(
    n3: int,
    n2: int,
    n1: int,
    *,
    candidate_density: float = 0.02,
    seed: int = 20250389,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= candidate_density <= 1.0:
        raise ValueError("candidate_density must be between 0 and 1")

    rng = np.random.default_rng(seed)
    _, i2, _ = np.indices((n3, n2, n1), dtype=np.float32)
    center2 = np.float32(0.5 * (n2 - 1))
    distance = i2 - center2
    structure = np.exp(-0.5 * (distance / np.float32(1.2)) ** 2).astype(np.float32)
    candidates = rng.random((n3, n2, n1)) < candidate_density
    jitter = rng.random((n3, n2, n1), dtype=np.float32)
    ft = np.where(
        candidates,
        np.float32(0.75) + np.float32(0.2) * structure + np.float32(0.05) * jitter,
        np.float32(0.2) * structure,
    ).astype(np.float32)
    pt = np.zeros_like(ft, dtype=np.float32)
    tt = np.full_like(ft, 90.0, dtype=np.float32)
    return ft, pt, tt


def time_repeated(
    func,
    *,
    repeat: int,
    warmup: int,
):
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
