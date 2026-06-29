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

`theta_min` and `theta_max` bound the candidate feature angles scanned in the
2D `(i2, i1)` image plane. Feature angle `0` follows a horizontal feature along
increasing `i1`, and positive feature angles dip toward increasing `i2` as
`i1` increases.

The returned `pt` image uses the same convention consumed by
`pyosv.cells.FaultCell2` and `pyosv.voting2d.OptimalPathVoter`: the local fault
normal is `(sin(pt), cos(pt))`, and the local strike direction is
`(-cos(pt), sin(pt))` in `(i1, i2)` component order. Angles are equivalent
modulo 180 degrees. A horizontal feature therefore has `pt` equivalent to `0`,
a vertical feature has `pt` equivalent to `90`, and a down-right `45` degree
feature has `pt` equivalent to `135`.

The scanner evaluates the sampled feature-angle range and stores the
corresponding voter-compatible `pt` for the strongest local score.

## Outputs

`FaultOrientScanner2.scan(theta_min, theta_max, g)` returns `(ft, pt)`:

- `ft`: normalized fault likelihood, `np.float32`, shape `(n2, n1)`, values in
  `[0, 1]`.
- `pt`: selected voter-compatible orientation angle in degrees, `np.float32`,
  shape `(n2, n1)`.

Input images must be finite 2D numeric arrays and are converted to `float32`.
Constant images return zero likelihood and a finite angle image.

`FaultOrientScanner2.thin(ft, pt)` keeps local likelihood maxima across the
local normal direction implied by `pt`. It returns `(thinned_ft, thinned_pt)` as
`float32` arrays with the same `(n2, n1)` shape; non-retained samples use zero
for both likelihood and orientation.

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
