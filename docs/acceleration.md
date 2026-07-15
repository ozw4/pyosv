# Optional Acceleration

`pyosv` runs with NumPy and SciPy only. Numba is an optional acceleration
dependency for selected dynamic-programming and voting kernels.

## Installation

Install the optional acceleration extra when local benchmarking or repeated
large synthetic runs need it:

```bash
python -m pip install -e ".[accel]"
```

The default package dependencies do not include Numba. Runtime workflows must
not require a JVM, Jython, Gradle, or Mines JTK.

## Import Mode

Set `PYOSV_ACCEL` before importing `pyosv` modules to select how the optional
Numba dependency is handled:

- `auto` (the default) tries to import Numba and uses the Python fallback if
  the import fails;
- `off` does not import Numba and always uses the Python fallback;
- `required` imports Numba and immediately raises `ImportError` if it is not
  available.

Values are case-insensitive and surrounding whitespace is ignored. Any other
value, including an empty value, raises `ValueError` instead of silently using
`auto`. For example:

```bash
PYOSV_ACCEL=off python -m your_workflow
PYOSV_ACCEL=required python -m your_workflow
```

The mode is read once when `pyosv._accel` is imported. Changing the environment
variable afterward does not change the active mode; use a new process or
explicitly reload the module and all consumers that imported its globals.

## Fallback Behavior

The acceleration adapter lives in `pyosv._accel`. In `auto` or `required` mode,
if Numba imports successfully, decorated kernels use `numba.njit(cache=True)`.
In `off` mode, or in `auto` mode when Numba is unavailable, the same decorators
become no-ops and public APIs run the Python and NumPy fallback implementations.

Current accelerated code paths include parts of:

- 2D dynamic-programming accumulation and backtracking;
- 2D voting local sampling and vote accumulation;
- 3D voting local sampling and vote accumulation.
- 3D orientation-scanner structured linear rotation and shear transforms.

Fallback behavior is part of the supported runtime path. Normal tests and user
workflows should not require Numba.

## Determinism Policy

Acceleration must preserve the repository conventions for shape, dtype, finite
values, and practical equivalence:

- 2D arrays use shape `(n2, n1)`.
- Global 3D arrays use shape `(n3, n2, n1)`.
- Local 3D voting boxes use shape `(nw, nv, nu)`.
- Algorithm arrays should use `np.float32` unless a test requires another
  dtype.

Tests compare accelerated kernels with fallback kernels using exact equality or
tight practical tolerances where appropriate. They should not require bitwise
equivalence with Java, Jython, Mines JTK, or Gradle-based reference workflows.

## Benchmarks

Benchmark scripts are local developer tools. They are not performance gates and
are not part of normal pytest collection. Run them from the repository root:

```bash
python benchmarks/benchmark_voting2d.py
python benchmarks/benchmark_voting3d.py
python benchmarks/benchmark_dp.py
python benchmarks/benchmark_skinning3d.py --shape 25 --warmup 1 --repeat 3
python benchmarks/benchmark_orient3d.py --shape 25 --warmup 1 --repeat 3
python benchmarks/benchmark_synthetic_quality_cache.py --shape 9 --repeat 3
```

Each script builds a deterministic synthetic `float32` input, runs one or more
warmup iterations, then prints shape, timing, output-count, and compact numeric
fingerprint summaries. They do not write large outputs by default. The voting
benchmarks report seed selection separately from end-to-end voting and accept
`--candidate-density` and `--d` to control candidate density and suppression
distance. The 3D benchmark also compares the default full-surface orientation
smoothing with the opt-in center-separable backend; use `--orientation-nw`,
`--orientation-nv`, and `--orientation-sigma` to select its surface and sigma.
The orientation benchmark compares the default SciPy interpolation backend with
the opt-in structured linear backend and reports timing, traced allocations,
and output-difference statistics after warmup. The skinning benchmark reports
both single-seed growth and full
reference-like `find_skins` orchestration, including accepted-occupancy seed
rejections and the dense mask's storage size.
The synthetic-quality cache benchmark runs the quality-matrix variants with
and without the case-local cache and reports elapsed time plus traced peak
memory for both paths.

Use `--help` to inspect tunable sizes and repetition counts. With Numba enabled,
the first call to a kernel may include JIT compilation cost; keep at least one
warmup repetition when comparing steady-state timings. Performance values are
environment-dependent measurements for before/after comparison; they are not
pytest pass/fail criteria.
