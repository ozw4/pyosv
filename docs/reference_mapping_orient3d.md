# 3D Scanner Reference Mapping

This document maps `reference_osv/src/osv/FaultOrientScanner3.java` concepts to
the Python implementation exposed as
`pyosv.orient3d.FaultOrientScanner3`.

The Python scanner follows the Java scanner's orientation-sweep structure where
practical, but it is not a bit-exact Mines JTK port. PyOSV uses NumPy and SciPy
numerical kernels and has no JVM, Jython, Gradle, or Mines JTK runtime
dependency.

Global image volumes use shape `(n3, n2, n1)` and indexing
`array[i3, i2, i1]`. Geometric vectors use component order `(x1, x2, x3)`.
Scanner inputs and outputs are finite `float32` arrays unless an API explicitly
returns scalar metadata.

## Equivalence boundary

The reference-like scanner preserves these structural elements:

```text
strike sampling
  -> rotate around axis 1
  -> smooth along the rotated strike axis
  -> dip sampling
  -> shear
  -> smooth along the sheared dip axis
  -> unshear
  -> convert planarity to fault likelihood
  -> unrotate to global coordinates
  -> retain the strongest sampled orientation
```

Numerical equality with Java is not required. The principal implementation
boundaries are:

- SciPy interpolation or PyOSV structured-linear interpolation instead of JTK
  sinc interpolation;
- Gaussian filtering instead of JTK recursive exponential filtering;
- Python-defined rotated-grid, fill-value, and boundary behavior;
- `float32` array contracts and explicit Python validation;
- Python-specific scanner confidence, refined sampling, and derivative-bank
  APIs.

Reference agreement is therefore evaluated with deterministic behavior,
synthetic known-truth metrics, and F3 public-reference metrics rather than
byte-level Java equality.

## Method and helper mapping

| Java symbol or operation | Python symbol | Python contract | Equivalence boundary | Test coverage |
| --- | --- | --- | --- | --- |
| `FaultOrientScanner3` constructor | `FaultOrientScanner3.__init__` | Accepts positive finite `sigma1` and `sigma2`. `sigma1` controls dip-sampling density in sigma-derived sampling; `sigma2` controls strike-sampling density. | Python exposes one validated two-parameter constructor and no `power` overload. | Constructor and invalid-input behavior are covered in `tests/test_orient3d.py`. |
| `scan(phiMin, phiMax, thetaMin, thetaMax, g)` | `scan()` and `scan_reference_like()` | Validates the angle ranges and a finite 3-D input, uses Java-inspired strike and dip samples, executes the selected reference-like backend, and returns `(ft, pt, tt)` with the input shape. The default backend is `rotate_shear`. | Interpolation, smoothing, fill values, rotated bounds, edge behavior, and floating-point accumulation use Python/SciPy contracts. | Shape, dtype, range, constant-input, determinism, backend dispatch, and synthetic localization/orientation behavior are covered in `tests/test_orient3d.py`. |
| Strike loop and private `Rotator` | `_rotate3_axis1()`, `_unrotate3_axis1()`, `_rotated_axis1_grid()` | Rotates the `(i2, i3)` plane around axis 1 into a symmetric expanded grid and maps candidate likelihoods back to the original `(n3, n2, n1)` shape. Rotation uses fill value `1.0`; unrotation uses `0.0` unless another fill is supplied. | Python computes an explicit finite rectangular grid and uses SciPy sampling or structured bilinear sampling instead of JTK rotator tables and sinc interpolation. | Rotated-grid size, finite output, boundary behavior, inverse mapping, and SciPy/structured-linear agreement are covered in `tests/test_orient3d.py`. |
| `scanTheta(...)` dip sweep | `_scan_theta_shear_reference_like()` | For every dip sample, computes shear `-cos(theta)/sin(theta)`, shears each rotated slice, smooths along the sheared dip axis, unshears, and converts the response to fault likelihood. Near-vertical dip maps to zero shear; extreme finite shear is clipped to `[-1e4, 1e4]`. | Python keeps each sheared slice at the same shape and uses constant-fill linear or SciPy interpolation. The Java bounds and sinc kernel are not reproduced exactly. | Shear/unshear geometry, near-vertical behavior, finite boundaries, and interpolation-backend agreement are covered in `tests/test_orient3d.py`. |
| Strike-oriented smoothing | `_smooth_rotated_strike_axis()` | Applies `scipy.ndimage.gaussian_filter1d` on rotated axis 0 with nearest-edge handling. A nonpositive sigma returns a `float32` copy. | Java recursive filtering and its edge response are replaced by Gaussian filtering. | Constant, impulse, sigma, dtype, and scanner integration behavior are covered by orientation and filter tests. |
| Dip-oriented smoothing | `_smooth_sheared_dip_axis()` | Applies Gaussian filtering on axis 2 with effective sigma `smoothing_sigma * abs(sin(theta))` and nearest-edge handling. | The filter kernel and boundary response differ from `RecursiveExponentialFilter`. | Dip-dependent smoothing and finite scanner output are covered in `tests/test_orient3d.py`. |
| Semblance power conversion | `_reference_like_planarity_to_likelihood()` and `_reference_like_orientation_score()` | Clips planarity to `[0, 1]` and computes `1 - planarity**4`, returning clipped `float32` likelihood. | The structural power transform is retained; its input differs because Python interpolation and smoothing differ. | Likelihood range, monotonic behavior, and scanner output contracts are covered in `tests/test_orient3d.py`. |
| Best strike/dip retention | `_update_best_orientation()`, `_update_best_second_orientation()`, orientation-code helpers | Uses strict `score > best_score`; equal scores keep the first orientation in sweep order. Strike and dip indices are stored in the smallest supported unsigned code dtype and decoded into independent contiguous `float32` volumes. | Java storage details are not reproduced; deterministic sweep-order tie behavior is explicit in Python. | Code dtype, encode/decode behavior, contiguity, strict ties, and best-orientation selection are covered in `tests/test_orient3d.py`. |
| `getPhiSampling` and strike sampling used by scan | `reference_like_strike_sampling()` | Uses the fixed grid `0, 20, ..., 340` degrees, clipped to the requested inclusive range. A valid range containing no fixed-grid sample returns its lower endpoint. | The public sigma-derived `strike_sampling()` remains a separate Python API and is not used by `scan()`. | Full-range, clipped-range, narrow-range, dtype, ordering, and deterministic sampling are covered in `tests/test_orient3d.py`. |
| `getThetaSampling` and dip sampling used by scan | `reference_like_dip_sampling()` | Uses approximately five-degree spacing while preserving requested endpoints. A single-angle range returns one sample. | Python constructs a `float32` `linspace`; exact Java sampling-object behavior is not required. | Endpoint, spacing, single-angle, dtype, ordering, and deterministic sampling are covered in `tests/test_orient3d.py`. |
| `SincInterpolator` use in rotation and shear | `_sample2_with_constant()`, `_sample3_with_constant()`, structured-linear kernels | The default `scipy` backend accepts interpolation orders `0..5`. `structured_linear` is accepted only with `interpolation_order=1` and `backend="rotate_shear"`. | Kernels, extrapolation, and floating-point results differ from JTK sinc interpolation. | Coordinate order, fill behavior, invalid combinations, and SciPy/structured-linear tolerances are covered in `tests/test_orient3d.py` and `tests/test_interp_filters.py`. |
| `RecursiveExponentialFilter` use | Gaussian filter helpers in `pyosv._orient3d` and `pyosv.filters` | Scanner smoothing uses one-dimensional Gaussian filters on aligned axes. Generic `smooth1d`, `smooth2d`, and `smooth3d` helpers also use SciPy Gaussian filters. | Recursive exponential impulse response and zero-slope edge semantics are not part of the Python contract. | Generic filter shape, dtype, constant, and impulse behavior are covered in `tests/test_interp_filters.py`. |
| `thin(float[][][][] flpt)` | `FaultOrientScanner3.thin()` | `mode="reference"` applies strike-binned suppression through `reference_like_3d_thin_values(..., reinforce_vertical=False)` and applies scanner edge cleanup by default. Nonretained strike and dip samples are zero. | Smoothing and edge cleanup use PyOSV helpers rather than Java/JTK filtering. | Reference thinning, orientation masking, scanner edge cleanup, and validation are covered in `tests/test_orient3d.py` and `tests/test_thinning3d.py`. |
| Fault-normal nonmaximum suppression | `FaultOrientScanner3.thin(..., mode="normal")` | Converts strike/dip to a fault-normal field, samples likelihood one normal step in both directions with linear nearest-boundary interpolation, and keeps strict positive local maxima. | This is a separate Python thinning policy, not the mapping target for Java strike-binned `thin`. | Fault-normal geometry, retained values, sentinels, and boundary behavior are covered in `tests/test_orient3d.py`. |
| `smooth(flstop, sigma, p2, p3, fl, g)` tensor-guided smoothing | No direct scanner equivalent | The scanner API does not expose likelihood-masked local-tensor smoothing. `pyosv.filters.smooth3d()` is an isotropic Gaussian helper and must not be treated as a semantic replacement. | Local tensors, stop masks, and `LocalSmoothingFilter` behavior are outside the Python scanner contract. | Generic Gaussian smoothing is tested; no Java tensor-smoothing equivalence fixture is part of the test suite. |
| Strike/dip to local vectors | `pyosv.geometry` vector helpers | Uses one shared strike/dip convention for scanner scoring, thinning, cells, and voting. Components are ordered `(x1, x2, x3)`, independently of NumPy axis order. | Downstream numerical behavior can differ even when the vector formula agrees. | Formula, normalization, round-trip, and convention behavior are covered in `tests/test_geometry.py` and scanner tests. |

## Python scanner APIs without a direct Java method mapping

These APIs are part of the Python scanner contract but are not representations
of separate `FaultOrientScanner3.java` methods.

| Python API | Contract |
| --- | --- |
| `scan_quality()` | Uses the reference-like scoring path with interval refinement of the base strike and dip samples. `refinement_factor` is an integer from `1` through `4`; `1` preserves the base sampling and `2` inserts midpoints. `return_confidence=True` adds the confidence volume. |
| `scan_with_confidence()` | Returns `(ft, pt, tt, confidence)`. Confidence is the nonnegative best-minus-second-best response gap normalized to `[0, 1]`. It is diagnostic metadata, not geological truth. |
| `scan_fast()` | Uses sigma-derived angle sampling and Gaussian first/second derivatives to score candidate fault normals. Its likelihood is scaled by the finite 99.5th percentile and clipped to `[0, 1]`. It is a distinct derivative-bank backend rather than a Java scan approximation. |
| `scan_reference_like(..., backend="directional")` | Scores orientations by sampling and smoothing directly along candidate strike and dip vectors. It shares the likelihood conversion and output contract with the rotate/shear path but not the Java transform structure. |
| `interpolation_backend="structured_linear"` | Uses direct structured bilinear rotation/unrotation and linear shear/unshear. It retains the rotate/shear geometry and constant-fill contract without allocating full coordinate grids. Numba acceleration is optional. |

The scanner backend, scanner thinning mode, downstream workflow, and voter
thinning mode are independent settings. `scan_quality()` does not select the
quality workflow, and `thin(mode="reference")` does not select a scanner
backend.

## Input, output, and edge contracts

All scan methods require finite numeric three-dimensional input and convert it
to `float32`. Returned arrays have the same global shape as the input.

For `scan()`, `scan_reference_like()`, and `scan_quality()`:

- `ft` is finite and lies in `[0, 1]` when normalization is enabled;
- `pt` contains sampled strike angles in degrees;
- `tt` contains sampled dip angles in degrees;
- a constant input returns zero likelihood and the first strike/dip samples;
- equal candidate scores retain the first orientation in deterministic sweep
  order;
- confidence output, when requested, is zero for a constant input.

The omitted `smoothing_sigma` resolves to:

```text
max(1.0, 0.5 * (sigma1 + sigma2))
```

Scanner reference thinning uses `reference_sigma=1.0` and
`remove_edge_effects=True` by default. Edge cleanup is part of reference
scanner thinning only. The fault-normal thinning mode does not apply that
operation.

## Java utilities outside the Python scanner API

The following Java-side utilities have no public scanner mapping in PyOSV:

- `taper(...)`;
- `getFrequencies(...)`;
- `convertDips(...)`;
- `convertStrikes(...)`;
- Java-side three-dimensional directional derivative helpers;
- null-slice storage helpers internal to the Java `Rotator`.

Their absence is part of the present Python API boundary. Generic utilities
with similar names or numerical ingredients must not be described as semantic
equivalents without an explicit contract.

## Verification boundary

The Python suite verifies:

- constructor, angle, image, backend, and interpolation validation;
- shape, dtype, finiteness, output ranges, and constant-volume behavior;
- Java-inspired base sampling and refined sampling;
- deterministic orientation coding and tie handling;
- rotate/unrotate and shear/unshear geometry;
- SciPy and structured-linear agreement within numerical tolerances;
- confidence range and ambiguity behavior;
- synthetic planar and crossing-surface localization and orientation;
- reference and fault-normal scanner thinning;
- scanner-stage configuration, sampling evidence, and artifact contracts in
  Synthetic and F3 mode-comparison tests.

The suite does not require bit-exact Java or Mines JTK arrays. F3 tests measure
public-reference agreement and bundle consistency; they do not establish
geological truth or Java method-level identity.

## Related specifications

- [3D Orientation Scanning](orient3d.md)
- [Reference-Like 3D Thinning](reference_like_thinning.md)
- [3D Reference Alignment Audit](reference_alignment_3d.md)
- [Mode Comparison Contract](mode_comparison.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [F3 3D Reference Data Validation](f3d_validation.md)
