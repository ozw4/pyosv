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
