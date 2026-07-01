# 3D Scanner Reference Mapping

This document maps selected `reference_osv/src/osv/FaultOrientScanner3.java`
methods and helper classes to the current `pyosv.orient3d` implementation.

The current Python scanner is derivative-bank based. It evaluates Gaussian
first- and second-derivative responses for sampled strike/dip normals, then
normalizes the best score. The Java reference instead rotates the volume by
strike, smooths in rotated coordinates, sweeps dip with shear/unshear
interpolation, and unrotates the best response. The entries below therefore
describe approximations and gaps, not exact equivalence.

`SincInterpolator` and `RecursiveExponentialFilter` are Pythonic approximation
targets. They are not hard dependencies for `pyosv`, and no JVM, Jython, Mines
JTK, or Gradle runtime dependency should be introduced.

## Method-Level Mapping

| Java method / class | Python equivalent / status | Reference behavior summary | Current Python behavior | Known difference | Parity tests | Future reference-like implementation required |
| --- | --- | --- | --- | --- | --- | --- |
| `FaultOrientScanner3` constructor | `FaultOrientScanner3.__init__` | Stores strike and dip smoothing half-widths as `_sigmaPhi` and `_sigmaTheta`; the overload with `power` does not currently use that argument in the inspected reference path. | Validates positive finite `sigma1` and `sigma2`, storing them as dip and strike scanning controls. | Parameter names and validation differ; Python has no unused `power` overload. | Python constructor validation regression tests exist; no Java parity test. | No for the current derivative-bank API; revisit only if adding a reference-like scanner facade. |
| `scan(phiMin, phiMax, thetaMin, thetaMax, g)` | `FaultOrientScanner3.scan` | Builds Java sampling objects, rotates input for each strike, applies strike smoothing, scans dips in rotated slices, unrotates results, clips likelihood/dip ranges, and keeps the best strike/dip. | Builds NumPy angle arrays, validates `(n3, n2, n1)` `float32` input, handles constant volumes, then calls `_scan_orientation_bank`. | Different algorithm, interpolation, smoothing, edge handling, angle sampling, and score definition. Current scanner is derivative-bank based. | Python synthetic localization and scan/vote regression tests exist; F3 comparison is report-oriented; no method-level Java parity test. | Yes if a reference-like scanner path is pursued. |
| Reference-like scan entry point | `FaultOrientScanner3.scan_reference_like` | Intended future facade for the Java-style strike rotation, dip shear, smoothing, unrotation, and best-orientation workflow. | Skeleton only: validates the same angle/image inputs as `scan()`, validates `interpolation_order` and `smoothing_mode`, records scan configuration, and raises `NotImplementedError`. It is not used by default. | No orientation sweep, rotate/shear helpers, smoothing approximation, score computation, or outputs are implemented yet. | Python tests cover method presence, validation, explicit `NotImplementedError`, and unchanged F3 validation defaults. | Yes; this skeleton must not be used for production interpretation until implemented and validated. |
| `scanTheta(...)` / dip sweep equivalent | Current `_scan_orientation_bank` | For each rotated `i2` slice and dip angle, shears by `-1/tan(theta)`, smooths along axis 1 with sigma scaled by `sin(theta)`, unshears, applies a semblance power transform, and keeps the best dip. | Loops over strike and dip samples, projects Gaussian derivative responses onto the fault normal, and keeps the largest local score. | Python does not rotate, shear, smooth along dip-aligned coordinates, or compute the same semblance expression. | Python scan regression tests cover output shape/ranges and synthetic behavior; no `scanTheta` parity fixture. | Yes for reference-like scanner work. |
| `thin(float[][][][] flpt)` | `FaultOrientScanner3.thin` | Smooths likelihood in selected horizontal directions, then uses strike-binned 2D neighborhood comparisons in the `i2-i3` plane; zeroes strike/dip where likelihood is not retained. | Default `mode="normal"` samples plus/minus one fault-normal step with `interp.sample3`; opt-in `mode="reference"` uses `reference_like_3d_thin_values` for strike-binned suppression. | Default Python behavior is not the Java strike-binned thinning rule. The reference mode is an approximation and keeps current Python shape/dtype conventions. | Python normal and reference-mode thinning tests exist, including helper-level reference-like bins; no Java bit-exact parity test. | Partly represented by `mode="reference"`; more audit may be needed before changing defaults. |
| `smooth(flstop, sigma, p2, p3, fl, g)` | `filters.smooth3d` / Gaussian derivatives | Builds local tensors from slopes and likelihood mask, then applies `LocalSmoothingFilter` with anisotropic directions. | Generic `smooth3d` wraps SciPy Gaussian filtering; scanner internals use Gaussian derivatives instead of this tensor smoothing method. | No local tensor smoothing, likelihood stop mask, or JTK filter behavior in the scanner. | `smooth3d` has Python regression tests for shape, dtype, constants, and impulse behavior; no tensor-smoothing parity test. | Required only if a future issue needs this Java smoothing behavior. |
| `shear(...)` / `unshear(...)` | Currently missing / future approximation | Uses `SincInterpolator` to shear each 2D slice horizontally by a slope tied to dip, expanding and restoring slice bounds. | No public or private 3D scanner shear helper exists. Similar interpolation adapters live in `interp.sample3` and `interp.warp2d`, but they do not implement this workflow. | Missing coordinate transform stage for the Java scan workflow. | No parity tests. | Yes for reference-like scanner work. |
| `Rotator` private class | Currently missing / future approximation | Rotates and unrotates volumes in the `i2-i3` plane for each strike, computes expanded rotated sampling bounds, and uses sinc interpolation tables. | No 3D scanner rotator exists. `interp.rotate2d` is a SciPy 2D helper and is not a replacement for the Java volume rotator. | Missing strike-aligned coordinate transform and unrotation stage. | No parity tests. | Yes for reference-like scanner work. |
| `SincInterpolator` usage | `interp.sample3` / SciPy approximation | Samples rotated, unrotated, sheared, and local directional values with JTK sinc interpolation and configured extrapolation. | `interp.sample3` wraps `scipy.ndimage.map_coordinates` in Java-style `(i1, i2, i3)` coordinate order; scanner thinning uses linear sampling with nearest boundaries. | Kernel, extrapolation, coordinate bounds, and dtype behavior differ from JTK sinc interpolation. | `sample3` has Python coordinate-order and interpolation regression tests; no JTK parity test. | Yes as an approximation target for reference-like scanner interpolation. |
| `RecursiveExponentialFilter` usage | `filters.smooth*` / SciPy approximation | Applies recursive exponential smoothing for strike and dip-oriented smoothing, with zero-slope input edges. | `filters.smooth1d`, `smooth2d`, and `smooth3d` use SciPy Gaussian filters; `_gaussian_derivatives` uses SciPy Gaussian derivative filters. | Recursive exponential response and edge behavior are not reproduced. | Python smoothing adapter tests exist; no JTK recursive-filter parity test. | Yes as an approximation target if reproducing the Java rotate/shear/smooth workflow. |
| `getPhiSampling` / `getThetaSampling` and private sampling helpers | `strike_sampling` / `dip_sampling` | Public helpers use smoothing-dependent angle sampling; the inspected private scan path uses fixed strike sampling of 18 samples at 20 degrees and dip sampling near 5 degrees. | Uses finite `float32` NumPy linspace arrays. Step density is derived from `degrees(0.5 / sigma)` for both strike and dip controls. | Java public and private scan sampling paths differ from each other; Python follows its documented approximation and endpoint behavior. | Python sampling regression tests exist; no Java sampling parity test. | Audit required before any reference-like scanner chooses sampling semantics. |
| Normal vector conversion | `geometry.fault_normal_vector_from_strike_and_dip` | Java converts selected strike and dip to local fault-normal and related vectors for directional operations. | Python centralizes the same documented normal formula and uses it in scanning, thinning, cells, and voting. | Formula-level behavior is covered, but downstream use differs because scanner algorithms differ. | Geometry vector formula tests exist; no full Java scanner parity test. | No for current formula; verify again in a reference-like scanner audit. |
| Normalization / clipping | `_normalize_likelihood` | Java scan clips interpolated likelihood to `[0, 1]` after unrotation and stores the best raw clipped value. | Python clips nonnegative derivative-bank scores by the 99.5th percentile into `[0, 1]`; zero or invalid high percentiles produce zeros. | Score scale and normalization stage differ. | Python output range and deterministic behavior tests exist; no Java normalization parity test. | Required if a reference-like scanner should compare Java-like likelihood amplitudes. |

## Java Methods Not Currently Represented

The current `pyosv.orient3d` scanner does not represent these Java
`FaultOrientScanner3` utilities as scanner features:

- `taper(...)`
- `getFrequencies(...)`
- `convertDips(...)`
- `convertStrikes(...)`
- 3D directional derivative helpers used by Java-side experiments
- null-slice rotated-array helpers used by `Rotator`

These are not needed by the current derivative-bank scanner. If a future issue
requires one of them, map and test that method independently before changing
scanner behavior.

## Reference-like scanner implementation plan

1. API skeleton.
   Present as `FaultOrientScanner3.scan_reference_like()` without changing
   `FaultOrientScanner3.scan()` defaults. It validates inputs and stores the
   intended sampling/configuration state, but raises `NotImplementedError`
   because the orientation sweep is not implemented. Do not use it for
   production interpretation until implementation and validation are complete.
2. Orientation sweep scaffold.
   Build the strike/dip loop structure and intermediate arrays using small
   synthetic fixtures. Keep Java-like sampling choices explicit and documented.
3. Coordinate transform / interpolation approximation.
   Add Python approximations for the Java `Rotator`, `shear`, and `unshear`
   stages using SciPy interpolation adapters. Test coordinate order and boundary
   behavior separately from scanner scoring.
4. Directional smoothing approximation.
   Approximate `RecursiveExponentialFilter` usage with SciPy or explicit
   separable smoothing. Document sigma mapping and edge handling before using
   F3 metrics.
5. F3 scanner-only comparison.
   Compare scanner likelihood, strike, and dip outputs before voting or
   thinning. Treat the report as diagnostic unless a later issue defines
   acceptance thresholds.
6. Scan-vote-thin validation.
   Run the integrated scan, voting, and thinning workflow only after scanner
   differences have been isolated. Keep any default behavior changes for a
   separate issue with focused regression coverage.
