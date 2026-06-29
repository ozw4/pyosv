# Practical Equivalence Policy

Bitwise equivalence with Java, Jython, or Mines JTK is not a goal for `pyosv`.
Practical equivalence is the goal: Python outputs should preserve the geological
signal and workflow behavior needed for fault interpretation.

Mines JTK interpolation and recursive filtering are intentionally approximated
with Pythonic tools:

- JTK `SincInterpolator` behavior may be approximated with
  `scipy.ndimage.map_coordinates` or another SciPy interpolation primitive.
- JTK `RecursiveExponentialFilter` and `RecursiveGaussianFilterP` behavior may
  be approximated with SciPy Gaussian smoothing, such as
  `scipy.ndimage.gaussian_filter1d`, or with explicit separable smoothing.

These substitutions are allowed to differ in boundary handling, interpolation
kernels, recursive filter response, and floating-point accumulation order.

Tests should prioritize:

- shape correctness
- finite values
- value range sanity
- synthetic localization
- correlation and ridge-overlap metrics for reference comparisons

Tests should not require exact per-sample equality with Java outputs.

All array APIs should use the repository shape convention: 2D arrays are
`(n2, n1)`, and 3D arrays are `(n3, n2, n1)`.

## Current Metrics

The practical-equivalence helpers live in `src/pyosv/metrics.py`. They are
intended for sanity checks, regression tests, and optional reference reports.
They do not define a bitwise-equivalence contract.

`finite_value_report(x)` reports whether an output is numerically usable. The
report includes the input `shape`, total `size`, `finite_count`, `nan_count`,
`posinf_count`, `neginf_count`, `finite_fraction`, and finite-only
`finite_min`, `finite_max`, and `finite_mean` values. Summary statistics ignore
non-finite samples; if there are no finite samples, the finite summary values
are `nan`.

`normalized_correlation(a, b)` computes zero-mean normalized correlation for
same-shape finite arrays. It is useful for checking whether two vote images
carry similar broad signal after implementation differences such as
interpolation kernels and accumulation order. Shape mismatches or non-finite
values raise `ValueError`. Empty arrays raise `ValueError`. If either centered
array is constant and has zero norm, the function returns `0.0` because the
correlation is undefined and carries no localization signal.

`top_percentile_mask(x, percentile)` returns a boolean mask for finite values at
or above the requested percentile threshold. Percentiles must be finite and in
the inclusive range `[0, 100]`; empty arrays and arrays containing non-finite
values raise `ValueError`.

`top_percentile_overlap(a, b, percentile=95.0)` compares the high-value masks
from two same-shape finite arrays. The report includes `percentile`, `a_count`,
`b_count`, `overlap_count`, `union_count`, `a_fraction`, `b_fraction`,
`overlap_fraction`, `overlap_over_a`, `overlap_over_b`, and `jaccard`. This is
the current ridge-overlap metric for 2D voting outputs, where high-percentile
vote samples approximate the strongest interpreted fault ridges.

## Threshold Policy

Default tests should check metric well-formedness and deterministic Python
behavior. They should not require Java/JTK data or strict practical-equivalence
thresholds unless a future issue explicitly defines such thresholds for a
specific feature.

Reference comparisons are report-oriented in this phase. The optional 2D voting
smoke test prints finite-value reports, normalized correlation, and
top-percentile overlap at selected percentiles, but it intentionally avoids
failing on fixed correlation or overlap thresholds while the implementation is
still evolving.

## Optional Reference Checks

The `reference_osv/` directory is a read-only bind mount for the external
reference implementation. It is optional for normal development and default
tests, and it must not be modified or committed.

By default, reference paths resolve under `./reference_osv`. If the reference
checkout or bind mount is elsewhere, set:

```bash
export PYOSV_REFERENCE_OSV=/absolute/path/to/osv-master
```

Default tests skip reference cases clearly when the reference root or required
`.dat` files are absent. To run the optional slow 2D voting reference report,
provide the reference root and opt in explicitly:

```bash
PYOSV_REFERENCE_OSV=/absolute/path/to/osv-master \
PYOSV_RUN_SLOW_REFERENCE_VOTING=1 \
python -m pytest -q tests/test_voting2d_reference_smoke.py
```

The optional report compares `pyosv` output with existing reference `.dat`
files. It does not add a runtime dependency on the JVM, Jython, Mines JTK, or
Gradle, and it does not imply bitwise equivalence.
