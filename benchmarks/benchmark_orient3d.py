"""Reproducible synthetic benchmark for the public 3D orientation scanner."""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyosv._accel import NUMBA_AVAILABLE  # noqa: E402
from pyosv.geometry import fault_normal_vector_from_strike_and_dip  # noqa: E402
from pyosv.orient3d import FaultOrientScanner3  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, default=25, help="Cubic input size.")
    parser.add_argument(
        "--phi-samples",
        type=int,
        choices=range(1, 19),
        default=3,
        metavar="1..18",
        help="Number of reference-like strike samples.",
    )
    parser.add_argument(
        "--theta-samples",
        type=int,
        default=3,
        help="Number of approximately five-degree dip samples.",
    )
    parser.add_argument("--repeat", type=int, default=3, help="Measured repetitions.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup repetitions.")
    parser.add_argument("--seed", type=int, default=20250389, help="Synthetic input RNG seed.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shape <= 0:
        raise ValueError("shape must be positive")
    if args.theta_samples <= 0:
        raise ValueError("theta_samples must be positive")
    if NUMBA_AVAILABLE and args.warmup == 0:
        raise ValueError("warmup must be positive when Numba is available")

    true_phi = 40.0
    true_theta = 60.0
    image = synthetic_oriented_structure(
        args.shape,
        phi=true_phi,
        theta=true_theta,
        seed=args.seed,
    )
    phi_min = 0.0
    phi_max = 20.0 * (args.phi_samples - 1)
    theta_half_range = 2.5 * (args.theta_samples - 1)
    theta_min = true_theta - theta_half_range
    theta_max = true_theta + theta_half_range
    scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
    phi_sampling = scanner.reference_like_strike_sampling(phi_min, phi_max)
    theta_sampling = scanner.reference_like_dip_sampling(theta_min, theta_max)

    def scan_without_confidence() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return scanner.scan_reference_like(phi_min, phi_max, theta_min, theta_max, image)

    def scan_with_confidence() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return scanner.scan_with_confidence(phi_min, phi_max, theta_min, theta_max, image)

    without_confidence_times, without_confidence_result = time_repeated(
        scan_without_confidence,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    with_confidence_times, with_confidence_result = time_repeated(
        scan_with_confidence,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    without_confidence_peak_bytes = peak_traced_allocation(scan_without_confidence)
    with_confidence_peak_bytes = peak_traced_allocation(scan_with_confidence)
    ft, pt, tt = without_confidence_result
    confidence = with_confidence_result[3]

    print("benchmark=orient3d scanner=FaultOrientScanner3 backend=reference_like")
    print(
        " ".join(
            [
                f"input_shape={image.shape}",
                f"output_shapes_without_confidence={(ft.shape, pt.shape, tt.shape)}",
                f"output_shape_confidence={confidence.shape}",
                f"dtype={ft.dtype}",
                f"phi_samples={len(phi_sampling)}",
                f"theta_samples={len(theta_sampling)}",
                f"orientations={len(phi_sampling) * len(theta_sampling)}",
                f"rng_seed={args.seed}",
                f"repeat={args.repeat}",
                f"warmup={args.warmup}",
            ],
        ),
    )
    print(
        f"{timing_summary('without_confidence', without_confidence_times)} "
        f"peak_traced_bytes={without_confidence_peak_bytes} "
        f"output_count={ft.size} {array_fingerprint('ft', ft)}",
    )
    print(
        f"{timing_summary('with_confidence', with_confidence_times)} "
        f"peak_traced_bytes={with_confidence_peak_bytes} "
        f"output_count={confidence.size} {array_fingerprint('confidence', confidence)}",
    )
    print(array_fingerprint("pt", pt))
    print(array_fingerprint("tt", tt))
    return 0


def synthetic_oriented_structure(
    shape: int,
    *,
    phi: float,
    theta: float,
    seed: int = 20250389,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    i3, i2, i1 = np.indices((shape, shape, shape), dtype=np.float32)
    center = np.float32(0.5 * (shape - 1))
    w1, w2, w3 = fault_normal_vector_from_strike_and_dip(phi, theta)
    distance = w1 * (i1 - center) + w2 * (i2 - center) + w3 * (i3 - center)
    ridge = np.exp(-0.5 * (distance / np.float32(1.5)) ** 2).astype(np.float32)
    noise = rng.normal(0.0, 0.01, size=ridge.shape).astype(np.float32)
    planarity = np.clip(np.float32(1.0) - np.float32(0.85) * ridge + noise, 0.0, 1.0)
    return planarity.astype(np.float32, copy=False)


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


def peak_traced_allocation(func) -> int:
    """Return peak Python-traced bytes for one scan (not process peak RSS)."""

    tracemalloc.start()
    try:
        func()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak_bytes


def timing_summary(name: str, times: list[float]) -> str:
    return (
        f"name={name} min_seconds={min(times):.6f} "
        f"median_seconds={float(np.median(times)):.6f}"
    )


def array_fingerprint(name: str, array: np.ndarray) -> str:
    values = np.asarray(array)
    return " ".join(
        [
            f"{name}_sum={float(np.sum(values, dtype=np.float64)):.9g}",
            f"{name}_min={float(np.min(values)):.9g}",
            f"{name}_max={float(np.max(values)):.9g}",
        ],
    )


if __name__ == "__main__":
    raise SystemExit(main())
