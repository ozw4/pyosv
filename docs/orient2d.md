# 2D Orientation Scanning

`pyosv.orient2d.FaultOrientScanner2` is the Python 2D orientation scanner used
to produce fault-likelihood and dip-angle images for `OptimalPathVoter`.

## Practical Equivalence

The scanner targets practical equivalence for fault interpretation workflows,
not exact reproduction of `reference_osv/src/osv/FaultOrientScanner2.java`.
Synthetic tests check deterministic localization and orientation sanity, but
they do not require bitwise equality with Java, Jython, or Mines JTK output.

The implementation uses NumPy and SciPy image derivatives. This is an
intentional approximation of the Java/JTK implementation and can differ in
filter kernels, boundary handling, angle tie-breaking, and floating-point
accumulation.

## Angle Convention

Angles are dip orientations in degrees in the 2D `(i2, i1)` image plane.

- `0` degrees follows a horizontal feature along increasing `i1`.
- `90` and `-90` degrees are equivalent vertical orientations.
- Positive angles dip toward increasing `i2` as `i1` increases.

The scanner evaluates the sampled range from `theta_min` to `theta_max` and
stores the angle with the strongest local score. Because opposite normals are
equivalent for the absolute derivative score, vertical features may be reported
as either `90` or `-90` when both are in the sampled range.

## Outputs

`FaultOrientScanner2.scan(theta_min, theta_max, g)` returns `(ft, pt)`:

- `ft`: normalized fault likelihood, `np.float32`, shape `(n2, n1)`, values in
  `[0, 1]`.
- `pt`: selected dip angle in degrees, `np.float32`, shape `(n2, n1)`.

Input images must be finite 2D numeric arrays and are converted to `float32`.
Constant images return zero likelihood and a finite angle image.

`FaultOrientScanner2.thin(ft, pt)` keeps local likelihood maxima across the
local dip direction. It returns `(thinned_ft, thinned_pt)` as `float32` arrays
with the same `(n2, n1)` shape; non-retained samples use zero for both
likelihood and orientation.

Scanner output can be passed directly to:

```python
from pyosv.orient2d import FaultOrientScanner2
from pyosv.voting2d import OptimalPathVoter

scanner = FaultOrientScanner2(sigma1=2.0)
ft, pt = scanner.scan(-75.0, 75.0, image)

voter = OptimalPathVoter(ru=2, rv=5)
fv, w1, w2 = voter.apply_voting(d=3, fm=0.45, ft=ft, pt=pt)
```

## Limitations

This module does not add a runtime dependency on the JVM, Jython, Mines JTK, or
Gradle. It is not a drop-in numerical clone of the reference scanner. Current
limitations include:

- derivative-based scoring instead of the full Java/JTK filter stack;
- SciPy boundary behavior rather than Mines JTK boundary behavior;
- approximate orientation sampling based on `sigma1`;
- no committed real-data equivalence thresholds against `reference_osv`.

Use the scanner for deterministic Python workflows and synthetic regression
coverage. Treat reference-data comparisons as practical reports unless a future
issue defines feature-specific acceptance thresholds.
