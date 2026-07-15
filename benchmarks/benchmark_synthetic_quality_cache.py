"""Compare the case-local stage cache across the synthetic quality matrix."""

from __future__ import annotations

import argparse
import gc
from statistics import median
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pyosv.evaluation.synthetic_quality import application  # noqa: E402
from pyosv.evaluation.synthetic_quality.variants import QUALITY_MATRIX_VARIANTS  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, default=9, help="Cubic synthetic case size.")
    parser.add_argument("--case-set", default="minimal", help="Synthetic case-set name.")
    parser.add_argument(
        "--input-mode",
        choices=("oracle", "scanner", "both"),
        default="oracle",
        help="Attribute input mode.",
    )
    parser.add_argument("--repeat", type=int, default=3, help="Measured repetitions.")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup repetitions per path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.shape <= 0:
        raise ValueError("shape must be positive")
    if args.repeat <= 0:
        raise ValueError("repeat must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")

    kwargs = {
        "case_set": args.case_set,
        "shape": (args.shape, args.shape, args.shape),
        "variants": QUALITY_MATRIX_VARIANTS,
        "variant_preset": "quality-matrix",
        "input_mode": args.input_mode,
    }
    uncached_times = _measure_times(
        kwargs,
        use_stage_cache=False,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    cached_times = _measure_times(
        kwargs,
        use_stage_cache=True,
        repeat=args.repeat,
        warmup=args.warmup,
    )
    uncached_peak = _measure_traced_peak_bytes(kwargs, use_stage_cache=False)
    cached_peak = _measure_traced_peak_bytes(kwargs, use_stage_cache=True)

    uncached_median = median(uncached_times)
    cached_median = median(cached_times)
    print("benchmark=synthetic_quality_stage_cache")
    print(
        " ".join(
            (
                f"case_set={args.case_set}",
                f"shape={kwargs['shape']}",
                f"input_mode={args.input_mode}",
                f"variant_count={len(QUALITY_MATRIX_VARIANTS)}",
                f"repeat={args.repeat}",
                f"warmup={args.warmup}",
            )
        )
    )
    print(_timing_summary("uncached_before", uncached_times))
    print(_timing_summary("cached_after", cached_times))
    print(
        " ".join(
            (
                f"elapsed_speedup={uncached_median / cached_median:.3f}",
                f"uncached_before_traced_peak_bytes={uncached_peak}",
                f"cached_after_traced_peak_bytes={cached_peak}",
                f"traced_peak_delta_bytes={cached_peak - uncached_peak}",
                f"traced_peak_ratio={cached_peak / uncached_peak:.3f}",
            )
        )
    )
    return 0


def _measure_times(
    kwargs: dict[str, Any],
    *,
    use_stage_cache: bool,
    repeat: int,
    warmup: int,
) -> list[float]:
    for _ in range(warmup):
        result = application._build_report_outputs(
            **kwargs,
            use_stage_cache=use_stage_cache,
        )
        del result
    times = []
    for _ in range(repeat):
        gc.collect()
        start = time.perf_counter()
        result = application._build_report_outputs(
            **kwargs,
            use_stage_cache=use_stage_cache,
        )
        times.append(time.perf_counter() - start)
        del result
    return times


def _measure_traced_peak_bytes(
    kwargs: dict[str, Any],
    *,
    use_stage_cache: bool,
) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        result = application._build_report_outputs(
            **kwargs,
            use_stage_cache=use_stage_cache,
        )
        _, peak = tracemalloc.get_traced_memory()
        del result
    finally:
        tracemalloc.stop()
    return peak


def _timing_summary(name: str, times: list[float]) -> str:
    return " ".join(
        (
            f"name={name}",
            f"elapsed_s_min={min(times):.6f}",
            f"elapsed_s_median={median(times):.6f}",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
