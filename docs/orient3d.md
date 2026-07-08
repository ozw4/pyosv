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

## Reference-First Alignment

The scanner follows the repository reference-first policy for fault
interpretation workflows, but it is not an exact reproduction of
`reference_osv/src/osv/FaultOrientScanner3.java`. It is not a Mines JTK
replacement and does not add a runtime dependency on the JVM, Jython, Gradle,
or Mines JTK.

The default implementation uses NumPy and SciPy interpolation and smoothing
operations as an intentional approximation of the Java/JTK workflow. Outputs
may differ from the reference implementation because of filter kernels,
boundary handling, interpolation behavior, sampled angle density, angle
tie-breaking, and floating-point accumulation order.

Tests and examples should check shape correctness, finite values, value ranges,
synthetic localization, and deterministic Python behavior. They should not
require bitwise equality with Java or Mines JTK outputs.

## Reference-Like Scan

`FaultOrientScanner3.scan(...)` uses the approximate reference-like backend by
default. `FaultOrientScanner3.scan_reference_like(...)` remains as a compatible
explicit alias for callers that want to configure that backend. It validates
angle ranges, finite 3D input volumes, `backend`, interpolation order, optional
smoothing sigma, and normalization mode, then runs a deterministic strike/dip
orientation sweep.

Reference-like mode does not use the legacy derivative-bank scanner's
sigma-derived dense sampling. Strike samples follow the Java scanner's fixed
18-sample grid at 20 degree spacing from 0 degrees, clipped to the requested
range. Dip samples use approximately 5 degree spacing while preserving the
requested endpoints.

The default `backend="rotate_shear"` path approximates the Java scanner's
strike loop more directly: for each strike it rotates the input volume around
axis 1, smooths along the rotated strike axis, shears each rotated slice for
each dip, smooths along the dip axis, unshears, converts planarity to
likelihood, unrotates the candidate likelihood back to global coordinates, clips
it to `[0, 1]`, and keeps the best strike/dip. Dip angles are clipped to the
requested dip range in the returned `tt` volume.

`backend="directional"` keeps the previous practical approximation. It samples
in an orientation-dependent coordinate system and smooths the input planarity
values directly along candidate fault-parallel directions. This backend is
useful for comparisons with older `pyosv` results but is less structurally
aligned with the Java rotate/shear/smooth workflow.

Both reference-like backends clip smoothed planarity responses to `[0, 1]` and
convert them to likelihood with `1 - smoothed**4`. This matches the Java
scanner's smooth-then-semblance-power likelihood semantics more closely than
the older Python ridge/contrast score. They remain Pythonic SciPy
approximations, not bit-exact Mines JTK ports.

`FaultOrientScanner3.scan_with_confidence(...)` runs the same reference-like
scan and returns `(ft, pt, tt, confidence)`. The first three arrays have the
same semantics as `scan_reference_like(...)`. `confidence` is a `float32`
diagnostic volume in `[0, 1]` formed from the normalized response gap between
the best sampled orientation and the second-best sampled orientation. High
confidence means that gap is large; low confidence indicates stronger
orientation ambiguity and can be used as a candidate downstream weighting map
for voting or skinning. This confidence map is pyosv quality/diagnostic
metadata, not an output from the original Java reference implementation.

`FaultOrientScanner3.scan_quality(...)` is an opt-in quality-oriented scanner
path that uses the same reference-like scoring backend but refines the sampled
strike and dip grids. It starts from `reference_like_*_sampling()` and inserts
evenly spaced interior samples between adjacent base samples. The default
`refinement_factor=2` adds midpoints; `refinement_factor=1` is equivalent to
`scan_reference_like(...)` for the same backend and options. Refinement factors
are intentionally limited to the range `1..4` to avoid accidental excessive
orientation sweeps. Pass `return_confidence=True` to receive
`(ft, pt, tt, confidence)`.

`FaultOrientScanner3.scan_fast(...)` exposes the older derivative-bank scanner
as an explicit practical backend for diagnostics or workflows that prefer its
ridge/contrast score and sigma-derived dense angle sampling.

The synthetic quality report also has a report-local diagnostic
`--scanner-backend ensemble` option. It runs `reference-like`, `quality`, and
`fast`, normalizes each `ft` response to unit range, applies fixed reliability
priors, applies the quality confidence map as a small quality score weight, and
selects the best adjusted backend per voxel. This option creates ordinary
`ft/pt/tt` scanner attributes for the existing thinning, voting, and skinning
path; it does not change the `FaultOrientScanner3.scan()`, `scan_quality()`, or
`scan_fast()` API defaults.

Migration note: callers that used the old derivative-bank `scan()` behavior
should call `scan_fast()` explicitly. Callers that used the old scanner
fault-normal thinning default should pass `mode="normal"` explicitly.
Callers that used the old voter fault-normal thinning default should pass
`mode="normal"` to `OptimalSurfaceVoter.thin(...)`.

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
`OptimalSurfaceVoter.thin(fv, vp, vt)` defaults to reference-like strike-binned
voter thinning; pass `mode="normal"` only to reproduce the older fault-normal
voter thinning path.

`FaultOrientScanner3.thin(ft, pt, tt)` defaults to reference-like
strike-binned non-maximum suppression in the `i2-i3` plane using `pt` as the
strike-angle volume. Scanner-style edge-effect removal is applied by default:

```python
fet, fpt, ftt = scanner.thin(ft, pt, tt, reference_sigma=1.0)
```

Disable edge cleanup only for diagnostics:

```python
fet, fpt, ftt = scanner.thin(
    ft,
    pt,
    tt,
    reference_sigma=1.0,
    remove_edge_effects=False,
)
```

The legacy fault-normal local maximum path remains available as an explicit
opt-in:

```python
fet, fpt, ftt = scanner.thin(ft, pt, tt, mode="normal")
```

Both modes return `float32` arrays with zeros outside retained samples. In
`normal` mode, kept likelihood samples retain the original `ft` values. In
`reference` mode, kept likelihood samples use the smoothed comparison values
after scanner edge cleanup; `pt` and `tt` are copied only at retained samples.
`reference_sigma` controls that reference-like smoothing. Voter reference
thinning has separate reinforcement and edge-handling semantics.

## Synthetic Validation

Default scanner tests use small synthetic volumes only. They cover constant
volumes, planar sheets with expected strike/dip recovery, crossing planes with
deterministic maximum-response selection, and finite output from both `scan()`
and `scan_fast()`. Planar localization is checked with near-plane response and
buffered top-percentile overlap metrics. Strike error is measured with
wraparound, while dip error is measured as an absolute angular difference.

## Limitations

This is a compact Python scanner intended for deterministic local workflows and
synthetic regression coverage. Current limitations include:

- SciPy smoothing and interpolation behavior rather than Mines JTK behavior;
- `scan_fast()` remains a derivative-bank approximation, not a Java/JTK
  equivalent;
- no committed real-data 3D reference thresholds;
- sequential execution without acceleration-specific dependencies.

Use reference-data comparisons as practical reports unless a future issue
defines feature-specific 3D acceptance thresholds.
