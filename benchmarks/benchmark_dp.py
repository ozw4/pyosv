"""Small synthetic benchmark for dynamic-programming kernels."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyosv._accel import NUMBA_AVAILABLE  # noqa: E402
from pyosv.dp import accumulate_forward_2d, find_path_2d, find_surface_3d  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ni", type=int, default=64, help="2D cost path sample count.")
    parser.add_argument("--nl", type=int, default=17, help="2D cost lag sample count.")
    parser.add_argument("--nw", type=int, default=7, help="3D cost w sample count.")
    parser.add_argument("--nv", type=int, default=33, help="3D cost v sample count.")
    parser.add_argument("--nu", type=int, default=13, help="3D cost u sample count.")
    parser.add_argument("--repeat", type=int, default=5, help="Measured repetitions.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup repetitions.")
    parser.add_argument("--bstrain", type=int, default=4, help="2D DP strain spacing.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cost2d = synthetic_cost_2d(args.ni, args.nl)
    cost3d = synthetic_cost_3d(args.nw, args.nv, args.nu)
    lmin2d = -(args.nl // 2)
    lmin3d = -(args.nu // 2)

    accumulate_times, accumulated = time_repeated(
        lambda: accumulate_forward_2d(cost2d, bstrain=args.bstrain),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    path_times, path = time_repeated(
        lambda: find_path_2d(
            cost2d,
            lmin=lmin2d,
            bstrain=args.bstrain,
            attribute_smoothing=1,
            path_smoothing=0.0,
        ),
        repeat=args.repeat,
        warmup=args.warmup,
    )
    surface_times, surface = time_repeated(
        lambda: find_surface_3d(
            cost3d,
            lmin=lmin3d,
            bstrain1=args.bstrain,
            bstrain2=2,
            attribute_smoothing=1,
            surface_smoothing1=0.0,
            surface_smoothing2=0.0,
        ),
        repeat=args.repeat,
        warmup=args.warmup,
    )

    print(f"benchmark=dp_kernels numba_available={NUMBA_AVAILABLE}")
    print(
        " ".join(
            [
                f"cost2d_shape={cost2d.shape}",
                f"accumulated_shape={accumulated.shape}",
                f"path_shape={path.shape}",
                f"cost3d_shape={cost3d.shape}",
                f"surface_shape={surface.shape}",
                f"dtype={path.dtype}",
            ],
        ),
    )
    print(f"repeat={args.repeat} warmup={args.warmup}")
    print(timing_summary("accumulate_forward_2d", accumulate_times))
    print(timing_summary("find_path_2d", path_times))
    print(timing_summary("find_surface_3d", surface_times))
    return 0


def synthetic_cost_2d(ni: int, nl: int) -> np.ndarray:
    lmin = -(nl // 2)
    i = np.arange(ni, dtype=np.float32)[:, None]
    lags = lmin + np.arange(nl, dtype=np.float32)[None, :]
    trend = np.linspace(-2.0, 2.0, ni, dtype=np.float32)[:, None]
    return ((lags - trend) ** 2 + 0.01 * i).astype(np.float32)


def synthetic_cost_3d(nw: int, nv: int, nu: int) -> np.ndarray:
    lmin = -(nu // 2)
    iw, iv, iu = np.indices((nw, nv, nu), dtype=np.float32)
    lags = lmin + iu
    trend = 1.5 * (iv / max(nv - 1, 1)) - 0.75 + 0.2 * (iw / max(nw - 1, 1))
    return ((lags - trend) ** 2 + 0.02 * iw).astype(np.float32)


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
        f"name={name} "
        f"best_seconds={min(times):.6f} "
        f"mean_seconds={float(np.mean(times)):.6f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
