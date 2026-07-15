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

    seed_times, seeds = time_repeated(
        lambda: voter.pick_seeds(args.d, args.fm, ft, pt, tt),
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
    print(
        f"{timing_summary('pick_seeds', seed_times)} "
        f"output_count={len(seeds)} {array_fingerprint('seed_indices', seed_indices)}",
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
