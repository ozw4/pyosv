# 3D Orientation Scanning

`pyosv.orient3d.FaultOrientScanner3` is the approximate Python 3D orientation
scanner used to produce fault-likelihood, strike, and dip volumes for
`OptimalSurfaceVoter`.

## Shape Convention

Global 3D image volumes use shape `(n3, n2, n1)`. Array indexing is
`g[i3, i2, i1]`, while vector components keep OSV component order
`(x1, x2, x3)`.

All scanner inputs must be finite numeric 3D arrays. They are converted to
`np.float32`. Scanner outputs are also `np.float32` arrays with the same
`(n3, n2, n1)` shape as the input image.

## Strike and Dip Convention

`FaultOrientScanner3.scan(phi_min, phi_max, theta_min, theta_max, g)` scans a
sampled strike range `phi` and dip range `theta`, both in degrees.

The returned tuple is `(ft, pt, tt)`:

- `ft`: normalized fault likelihood in `[0, 1]`.
- `pt`: selected strike angle in degrees.
- `tt`: selected dip angle in degrees.

The returned `pt` and `tt` use the same convention consumed by
`pyosv.cells.FaultCell` and `pyosv.voting3d.OptimalSurfaceVoter`. For strike
`phi` and dip `theta`, the local vectors are:

- fault normal `u = (-cos(theta), sin(theta) cos(phi), -sin(theta) sin(phi))`;
- dip vector `v = (sin(theta), cos(theta) cos(phi), -cos(theta) sin(phi))`;
- strike vector `w = (0, sin(phi), cos(phi))`.

The component order above is `(x1, x2, x3)`, not array indexing order.

## Practical Equivalence

The scanner targets practical equivalence for fault interpretation workflows,
not exact reproduction of `reference_osv/src/osv/FaultOrientScanner3.java`.
It is not a Mines JTK replacement and does not add a runtime dependency on the
JVM, Jython, Gradle, or Mines JTK.

The implementation uses NumPy and SciPy derivative and smoothing operations as
an intentional approximation of the Java/JTK workflow. Outputs may differ from
the reference implementation because of filter kernels, boundary handling,
interpolation behavior, sampled angle density, angle tie-breaking, and
floating-point accumulation order.

Tests and examples should check shape correctness, finite values, value ranges,
synthetic localization, and deterministic Python behavior. They should not
require bitwise equality with Java or Mines JTK outputs.

## Reference-Like Scan Skeleton

`FaultOrientScanner3.scan_reference_like(...)` is an opt-in API skeleton for
future alignment with the Java rotate/shear/smooth scan workflow. It validates
angle ranges, finite 3D input volumes, interpolation order, and smoothing mode,
then raises `NotImplementedError`.

The skeleton is not used by default. Existing examples and F3 validation
scripts continue to call `FaultOrientScanner3.scan(...)`, and the skeleton must
not be used for production interpretation until the orientation sweep is
implemented and validated.

## Integration

Scanner output can be passed directly to 3D optimal-surface voting:

```python
from pyosv.orient3d import FaultOrientScanner3
from pyosv.voting3d import OptimalSurfaceVoter

scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
ft, pt, tt = scanner.scan(
    phi_min=0.0,
    phi_max=90.0,
    theta_min=45.0,
    theta_max=90.0,
    g=image,
)

voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
voter.set_attribute_smoothing(0)
voter.set_surface_smoothing(0.0, 0.0)
fv, vp, vt = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)
fvt = voter.thin(fv, vp, vt)
```

`fv`, `vp`, `vt`, and `fvt` all use the same global `(n3, n2, n1)` shape.
`fv` is a normalized vote volume, and `vp`/`vt` store the strike and dip angles
associated with the strongest local vote at each sample.

`FaultOrientScanner3.thin(ft, pt, tt, mode="normal")` keeps the default
fault-normal local maxima used by existing workflows. The opt-in
`mode="reference"` path instead applies strike-binned non-maximum suppression in
the `i2-i3` plane using `pt` as the strike-angle volume:

```python
fet, fpt, ftt = scanner.thin(ft, pt, tt, mode="reference", reference_sigma=1.0)
```

Both modes return `float32` arrays with the original values retained at kept
samples and zeros elsewhere. `reference_sigma` controls the helper smoothing
used only for the reference-like comparison; output values are copied from the
unsmoothed inputs.

## Limitations

This is a compact Python scanner intended for deterministic local workflows and
synthetic regression coverage. Current limitations include:

- derivative-bank scoring instead of the full Java/JTK scanner;
- SciPy smoothing and interpolation behavior rather than Mines JTK behavior;
- approximate angle sampling controlled by `sigma1` and `sigma2`;
- no committed real-data 3D reference thresholds;
- sequential execution without acceleration-specific dependencies.

Use reference-data comparisons as practical reports unless a future issue
defines feature-specific 3D acceptance thresholds.
