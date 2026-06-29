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

## Fallback Behavior

The acceleration adapter lives in `pyosv._accel`. If Numba imports
successfully, decorated kernels use `numba.njit(cache=True)`. If Numba is not
available, the same decorators become no-ops and public APIs run the Python and
NumPy fallback implementations.

Current accelerated code paths include parts of:

- 2D dynamic-programming accumulation and backtracking;
- 2D voting local sampling and vote accumulation;
- 3D voting local sampling and vote accumulation.

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
```

Each script builds a small synthetic `float32` input, runs one or more warmup
iterations, then prints shape and timing summaries. They do not write large
outputs by default.

Use `--help` to inspect tunable sizes and repetition counts. With Numba enabled,
the first call to a kernel may include JIT compilation cost; keep at least one
warmup repetition when comparing steady-state timings.
