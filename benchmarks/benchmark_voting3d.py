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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ft, pt, tt = synthetic_fault_likelihood(args.n3, args.n2, args.n1)
    voter = OptimalSurfaceVoter(ru=args.ru, rv=args.rv, rw=args.rw)
    voter.set_attribute_smoothing(0)
    voter.set_surface_smoothing(0.0, 0.0)

    seeds = voter.pick_seeds(args.d, args.fm, ft, pt, tt)

    def run_once() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return voter.apply_voting(d=args.d, fm=args.fm, ft=ft, pt=pt, tt=tt)

    times, result = time_repeated(run_once, repeat=args.repeat, warmup=args.warmup)
    fv, vp, vt = result

    print(f"benchmark=3d_voting numba_available={NUMBA_AVAILABLE}")
    print(
        " ".join(
            [
                f"input_shape={ft.shape}",
                f"output_shapes={(fv.shape, vp.shape, vt.shape)}",
                f"dtype={fv.dtype}",
                f"seeds={len(seeds)}",
            ],
        ),
    )
    print(
        " ".join(
            [
                f"repeat={args.repeat}",
                f"warmup={args.warmup}",
                f"best_seconds={min(times):.6f}",
                f"mean_seconds={float(np.mean(times)):.6f}",
                f"fv_max={float(np.max(fv)):.6g}",
                f"fv_nonzero={int(np.count_nonzero(fv))}",
            ],
        ),
    )
    return 0


def synthetic_fault_likelihood(
    n3: int,
    n2: int,
    n1: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, i2, _ = np.indices((n3, n2, n1), dtype=np.float32)
    center2 = np.float32(0.5 * (n2 - 1))
    distance = i2 - center2
    ft = np.exp(-0.5 * (distance / np.float32(1.2)) ** 2).astype(np.float32)
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


if __name__ == "__main__":
    raise SystemExit(main())
