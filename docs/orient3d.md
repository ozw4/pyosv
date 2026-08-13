# 3D Orientation Scanning

`pyosv.orient3d.FaultOrientScanner3` produces 3D fault-likelihood, strike, and
dip volumes for scanner thinning, optimal-surface voting, and evaluation
workflows.

The implementation follows the transform structure of
`reference_osv/src/osv/FaultOrientScanner3.java` where practical, but it is not
a bit-exact Mines JTK port. PyOSV uses NumPy, SciPy, and optional Numba kernels
and has no JVM, Jython, Gradle, or Mines JTK runtime dependency.

## Array and geometry contract

Global image volumes use shape `(n3, n2, n1)` and array indexing
`array[i3, i2, i1]`. Geometric vectors use component order `(x1, x2, x3)`.

All scanner inputs must be finite numeric three-dimensional arrays. Public scan
methods convert inputs to `float32` and return `float32` arrays with the same
global shape.

Strike `phi` and dip `theta` are expressed in degrees. For one selected
orientation, the local vectors are:

```text
normal = (-cos(theta), sin(theta) cos(phi), -sin(theta) sin(phi))
dip    = ( sin(theta), cos(theta) cos(phi), -cos(theta) sin(phi))
strike = ( 0,          sin(phi),             cos(phi))
```

The returned scanner tuple is `(ft, pt, tt)`:

| Array | Contract |
| --- | --- |
| `ft` | Fault-likelihood response. Reference-like and fast scans return finite values in `[0, 1]`. |
| `pt` | Selected strike sample in degrees. |
| `tt` | Selected dip sample in degrees. |

`pt` and `tt` use the same convention as `pyosv.cells.FaultCell` and
`pyosv.voting3d.OptimalSurfaceVoter`.

## Constructor

```python
from pyosv.orient3d import FaultOrientScanner3

scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
```

Both constructor arguments must be finite positive numbers.

- `sigma1` controls the density of the sigma-derived dip sampling used by
  `dip_sampling()` and `scan_fast()`.
- `sigma2` controls the density of the sigma-derived strike sampling used by
  `strike_sampling()` and `scan_fast()`.
- When `smoothing_sigma` is omitted from a reference-like scan, it resolves to
  `max(1.0, 0.5 * (sigma1 + sigma2))`.

The reference-like scan uses its own Java-inspired base sampling and does not
use the sigma-derived sampling methods.

## Sampling APIs

`strike_sampling()` and `dip_sampling()` return deterministic, monotonic,
finite `float32` samples whose endpoints match the requested range. A
single-angle range returns one sample. These methods supply the angle grids for
`scan_fast()`.

`reference_like_strike_sampling()` uses the fixed strike grid
`0, 20, ..., 340` degrees and keeps samples inside the requested inclusive
range. When a valid range contains no fixed-grid sample, the lower endpoint is
returned as the sole sample.

`reference_like_dip_sampling()` uses approximately five-degree spacing while
preserving the requested endpoints.

The refined sampling methods insert evenly spaced interior samples between
adjacent reference-like base samples:

```python
phis = scanner.refined_reference_like_strike_sampling(
    0.0,
    90.0,
    refinement_factor=2,
)
thetas = scanner.refined_reference_like_dip_sampling(
    45.0,
    90.0,
    refinement_factor=2,
)
```

`refinement_factor` must be an integer from `1` through `4`. A factor of `1`
returns the base sampling; a factor of `2` inserts interval midpoints.

## Reference-like scan

`scan()` and `scan_reference_like()` execute the same reference-like scanner
contract.

```python
ft, pt, tt = scanner.scan(
    phi_min=0.0,
    phi_max=180.0,
    theta_min=45.0,
    theta_max=90.0,
    g=image,
)
```

The configurable options are:

| Option | Default | Contract |
| --- | --- | --- |
| `backend` | `"rotate_shear"` | Selects `rotate_shear` or `directional`. |
| `interpolation_order` | `1` | Integer interpolation order from `0` through `5`. |
| `interpolation_backend` | `"scipy"` | Selects `scipy` or `structured_linear`. |
| `smoothing_sigma` | `None` | Nonnegative smoothing extent; `None` uses the constructor-derived default. |
| `normalize` | `True` | Applies the final reference-like likelihood normalization step. |

A constant input returns zero likelihood, fills `pt` and `tt` with the first
strike and dip samples, and returns zero confidence when confidence is
requested.

Candidate orientations are compared with strict `score > best_score`. Equal
scores retain the first orientation in deterministic sweep order.

### Rotate/shear backend

`backend="rotate_shear"` executes this structure for every strike and dip:

```text
rotate the input around axis 1
  -> smooth along the rotated strike axis
  -> shear by the dip-dependent slope
  -> smooth along the sheared dip axis
  -> unshear
  -> convert planarity to fault likelihood
  -> unrotate to global coordinates
  -> retain the strongest orientation
```

Rotation expands the `(i2, i3)` plane to a symmetric finite rectangular grid.
The SciPy path uses `scipy.ndimage.map_coordinates`. Rotation and shear use a
constant fill value of `1.0`; unrotation uses `0.0` by default.

Strike smoothing uses a one-dimensional Gaussian filter with nearest-edge
handling. Dip smoothing uses effective sigma
`smoothing_sigma * abs(sin(theta))`. The dip shear is
`-cos(theta) / sin(theta)`, with zero shear at numerically vertical dips and a
finite clipped shear for numerically horizontal dips.

### Directional backend

`backend="directional"` evaluates each sampled orientation by sampling and
smoothing directly along candidate strike and dip directions. It shares the
reference-like angle sampling, likelihood conversion, output shapes, and tie
contract, but it does not use the rotate/shear transform structure.

The directional backend uses the SciPy interpolation path. Selecting
`interpolation_backend="structured_linear"` with the directional backend is
rejected.

### Interpolation backends

`interpolation_backend="scipy"` supports interpolation orders `0` through `5`.

`interpolation_backend="structured_linear"` performs direct bilinear
rotation/unrotation and linear slice shear/unshear without allocating full
coordinate grids. It is accepted only with:

```text
backend = rotate_shear
interpolation_order = 1
```

Other combinations raise `ValueError`. The structured kernels use Numba when
available and retain the same API through pure Python kernels otherwise. Small
floating-point differences from the SciPy path are expected.

### Likelihood conversion

Both reference-like backends clip the smoothed planarity response to `[0, 1]`
and compute:

```text
likelihood = 1 - planarity**4
```

The transform structure matches the Java scanner's smooth-and-power likelihood
semantics, while interpolation, smoothing, boundary handling, and floating-point
results follow the Python implementation.

## Orientation confidence

`scan_with_confidence()` runs the base reference-like sampling and returns:

```text
(ft, pt, tt, confidence)
```

`confidence` is the nonnegative gap between the best and second-best sampled
orientation responses, normalized across the volume to `[0, 1]`. A zero dynamic
range produces an all-zero confidence volume. Confidence is diagnostic metadata;
it is not a probability, uncertainty calibration, or geological truth value.

```python
ft, pt, tt, confidence = scanner.scan_with_confidence(
    0.0,
    180.0,
    45.0,
    90.0,
    image,
)
```

## Quality scan

`scan_quality()` uses the reference-like scoring backend with refined strike and
dip sampling.

```python
ft, pt, tt, confidence = scanner.scan_quality(
    0.0,
    180.0,
    45.0,
    90.0,
    image,
    refinement_factor=2,
    return_confidence=True,
)
```

`return_confidence=False` returns `(ft, pt, tt)`. Setting it to `True` returns
`(ft, pt, tt, confidence)`.

`scan_quality()` changes only scanner sampling density. It does not select a
downstream workflow, scanner-thinning policy, voter-thinning policy, or skinning
policy.

## Derivative-bank scan

`scan_fast()` is a distinct derivative-bank scanner. It uses the sigma-derived
strike and dip samples, Gaussian first and second derivatives, and a candidate
score formed from the directional gradient and Hessian responses. The selected
score volume is scaled by its finite 99.5th percentile and clipped to `[0, 1]`.

```python
ft, pt, tt = scanner.scan_fast(
    0.0,
    180.0,
    45.0,
    90.0,
    image,
)
```

`scan_fast()` does not implement the Java rotate/shear workflow and does not
accept the reference-like interpolation or smoothing options.

## Report-level scanner backends

Synthetic evaluation exposes these scanner backend names:

| Backend | Scanner execution |
| --- | --- |
| `reference-like` | `scan()` with the base reference-like sampling. |
| `quality` | `scan_quality()` with the configured refinement factor and confidence output. |
| `fast` | `scan_fast()`. |
| `ensemble` | Per-voxel selection across `reference-like`, `quality`, and `fast`. |

The report-local ensemble unit-range normalizes each component likelihood,
applies component priors `1.00`, `1.05`, and `1.00`, and multiplies the quality
component by `0.75 + 0.25 * confidence`. Per-voxel `argmax` selection uses the
component order `reference-like`, `quality`, `fast`, so an exact adjusted-score
tie selects the first component in that order. The selected component supplies
`ft`, `pt`, and `tt` at that voxel.

The canonical scanner-backend × workflow comparison uses `reference-like` and
`quality` as the scanner axis. Backend selection remains independent of the
downstream workflow. See [Mode Comparison Contract](mode_comparison.md).

## Scanner thinning

`FaultOrientScanner3.thin(ft, pt, tt)` requires finite matching global volumes
and returns `(fet, fpt, ftt)` as `float32` arrays with the same shape. Rejected
samples use zero for likelihood, strike, and dip.

### Reference thinning

The default `mode="reference"` contract is:

1. Smooth `ft` along `i3` and `i2`, but not `i1`.
2. Fold strike into `[0, 180)` and select the corresponding horizontal,
   diagonal, or vertical comparison direction in the `i2-i3` plane.
3. Keep strict local maxima against the two directional neighbors.
4. Write the smoothed likelihood at retained samples.
5. Copy input strike and dip only at retained samples.
6. Apply scanner edge cleanup when `remove_edge_effects=True`.

```python
fet, fpt, ftt = scanner.thin(
    ft,
    pt,
    tt,
    mode="reference",
    reference_sigma=1.0,
    remove_edge_effects=True,
)
```

`reference_sigma` is a nonnegative smoothing extent. Scanner reference thinning
does not apply voter retained-sample reinforcement.

Scanner edge cleanup removes retained samples within five samples of selected
`i2` or `i3` faces when the corresponding squared fault-normal component
exceeds `cos(30 degrees)**2`. It does not remove `i1` face samples. Set
`remove_edge_effects=False` only when that cleanup must be excluded from the
selected scanner contract.

### Fault-normal thinning

`mode="normal"` samples the input likelihood one fault-normal step in both
directions with linear interpolation and nearest-boundary handling. A sample is
retained only when it is positive and strictly greater than both sampled
neighbors. Retained samples keep the original input likelihood.

```python
fet, fpt, ftt = scanner.thin(ft, pt, tt, mode="normal")
```

The scanner backend and scanner-thinning mode are independent settings.

## Integration with 3D voting

Scanner attributes can be thinned and passed directly to
`OptimalSurfaceVoter`:

```python
from pyosv.orient3d import FaultOrientScanner3
from pyosv.voting3d import OptimalSurfaceVoter

scanner = FaultOrientScanner3(sigma1=2.0, sigma2=2.0)
ft, pt, tt = scanner.scan(
    phi_min=0.0,
    phi_max=180.0,
    theta_min=45.0,
    theta_max=90.0,
    g=image,
)
fet, fpt, ftt = scanner.thin(ft, pt, tt)

voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
fv, vp, vt = voter.apply_voting(
    d=3,
    fm=0.5,
    ft=fet,
    pt=fpt,
    tt=ftt,
)
fvt = voter.thin(fv, vp, vt)
```

All global arrays in this sequence use shape `(n3, n2, n1)`. Scanner backend,
scanner thinning, downstream workflow, surface-voting boundary policy, voter
thinning, and skinning are separate configuration axes.

## Verification and equivalence boundary

The Python test suite verifies:

- constructor and input validation;
- sigma-derived, reference-like, and refined angle sampling;
- shape, dtype, finiteness, output range, and constant-input behavior;
- deterministic orientation coding and strict tie handling;
- rotate/unrotate and shear/unshear geometry;
- SciPy and structured-linear agreement within numerical tolerances;
- confidence range and ambiguity behavior;
- planar and crossing-surface localization and orientation;
- reference and fault-normal scanner thinning;
- scanner configuration and evidence consumed by Synthetic and F3 evaluation.

Tests do not require bit-exact Java or Mines JTK arrays. F3 public outputs are
comparison targets rather than geological truth or method-level Java fixtures.
Detailed Java-to-Python mappings are documented in
[3D Scanner Reference Mapping](reference_mapping_orient3d.md).

## Related specifications

- [3D Scanner Reference Mapping](reference_mapping_orient3d.md)
- [Reference-Like 3D Thinning](reference_like_thinning.md)
- [3D Voting Conventions](3d_voting.md)
- [Mode Comparison Contract](mode_comparison.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [F3 3D Reference Data Validation](f3d_validation.md)
