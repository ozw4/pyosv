# Practical Equivalence Policy

Bitwise equivalence with Java, Jython, or Mines JTK is not a goal for `pyosv`.
Practical equivalence is the goal: Python outputs should preserve the geological
signal and workflow behavior needed for fault interpretation.

Mines JTK interpolation and recursive filtering are intentionally approximated
with Pythonic tools. For example, JTK `SincInterpolator` behavior may be replaced
with SciPy interpolation, and JTK recursive exponential filtering may be replaced
with Gaussian-style or other separable smoothing. Floating-point accumulation
order may differ from the reference implementation.

Tests should prioritize:

- shape correctness
- finite values
- value range sanity
- synthetic localization
- correlation and ridge-overlap metrics for reference comparisons

Tests should not require exact per-sample equality with Java outputs.
