# 3D Voter Reference Mapping

This document maps selected
`reference_osv/src/osv/OptimalSurfaceVoter.java` methods to the current
`pyosv` 3D voting implementation. It is an audit guide, not an equivalence
claim. The current Python implementation keeps repository conventions:
global arrays are `(n3, n2, n1)`, local voting costs are `(nw, nv, nu)`, and
values are normalized to `np.float32` where practical.

`pyosv` targets practical agreement for fault interpretation workflows. It
does not add runtime dependencies on JVM, Jython, Mines JTK, or Gradle. Java
`SincInterpolator`, `RecursiveExponentialFilter`, and related JTK filters are
therefore approximation targets for Python/SciPy code, not dependencies.

## Method-Level Mapping

| Java method | Python equivalent / status | Reference summary | Current Python summary | Known differences | Audit status | Suggested future parity test |
| --- | --- | --- | --- | --- | --- | --- |
| `OptimalSurfaceVoter(int ru, int rv, int rw)` constructor | `OptimalSurfaceVoter.__init__` | Stores local radii, initializes lag bounds from `ru`, default strain inverse values of 4, one attribute smoothing, surface smoothing extents of 2.0, JTK smoothing filters, and `_lmins` / `_lmaxs`. | Validates nonnegative radii, stores `lmin`, `lmax`, `nl`, default `bstrain1` / `bstrain2`, smoothing settings, and generates `lmins` / `lmaxs` with `dp.update_shift_ranges_3d`. | Python has explicit validation and no JTK filter objects; shift-range generation is Python code and only approximates the reference contract. | `partially covered` | Add a Java-derived fixture for several `(ru, rv, rw)` combinations, including edge radii, and compare `lmin`, `lmax`, `nl`, `lmins`, and `lmaxs`. |
| `setStrainMax(double strainMax1, double strainMax2)` | `set_strain_max` | Converts maximum surface strains to integer inverse strain bounds with ceiling-like behavior. | Uses `dp.strain_to_bstrain`, validating `0 < strain_max <= 1`, and updates only `bstrain1` / `bstrain2`. | Python validation is explicit; no Java parity fixture verifies all rounding cases. | `partially covered` | Compare Java and Python conversion for representative values near reciprocal boundaries, including values that should round up. |
| `setAttributeSmoothing(int esmooth)` / smoothing config | `set_attribute_smoothing` | Sets the number of nonlinear fault-attribute smoothing passes. | Validates a nonnegative integer and stores `attribute_smoothing`. | Java may allow values Python rejects by validation; smoothing implementation differs downstream. | `partially covered` | Check that zero, one, and multiple smoothing passes affect a small cost volume at the same stage as Java. |
| `setSurfaceSmoothing(double usmooth1, double usmooth2)` / smoothing config | `set_surface_smoothing` | Sets surface smoothing extents and rebuilds recursive exponential filters. | Validates nonnegative finite extents and stores `surface_smoothing1` / `surface_smoothing2`; smoothing is applied later through `dp.smooth_surface_2d`. | Python does not maintain JTK filter instances; SciPy smoothing response and edge behavior differ. | `partially covered` | Compare extracted surface smoothing on impulse, step, and sloped synthetic surfaces, focusing on axis order and edge behavior. |
| `pickSeeds(...)` | `pick_seeds` | Selects seed candidates above the thinned fault-attribute threshold, sorts by likelihood, and suppresses nearby lower candidates within a distance box. | Builds `FaultCell` objects from `(n3, n2, n1)` arrays, sorts descending by likelihood, suppresses candidates inside the distance box, and returns a Python list. | Java object ordering and equal-likelihood tie behavior are not audited; Python validates finite matching arrays. | `partially covered` | Build a fixture with equal likelihoods, boundary candidates, and suppression overlaps; compare selected seed coordinates and order. |
| `getSeeds(...)` | `get_seeds` | Reference utility for selecting one seed at a requested sample. | Returns a single `FaultCell` at `(c1, c2, c3)` after bounds and shape validation. | Python exposes a narrow helper; no Java parity fixture confirms reference call semantics. | `partially covered` | Compare one-sample seed extraction for valid and boundary coordinates, including stored strike and dip. |
| `updateVectorMap(...)` | `update_vector_map` | Fills displacement vectors for offsets in `[-radius, radius]` along the supplied local axis vector. | Returns a `(3, 2 * radius + 1)` `float32` array from vector components multiplied by offsets. | Java writes into caller-provided arrays and uses Java float arithmetic; Python allocates and returns a new array. | `partially covered` | Compare vector maps for non-axis-aligned unit and non-unit vectors, including negative components. |
| `samplesInUvwBox(...)` | `samples_in_uvw_box` | Samples `1 - fx` in a seed-centered local UVW box using normal, dip, and strike axes; invalid lag cells stay at default cost and image samples are rounded/clamped. | Returns local costs shaped `(2 * rw + 1, 2 * rv + 1, 2 * ru + 1)`, indexed as `(w, v, u)`, using `lmins` / `lmaxs`, Java-style `floor(x + 0.5)` rounding, and clamping to `(n3, n2, n1)` bounds. | Coordinate sign, axis order, and boundary behavior remain high risk; Numba and Python paths should remain identical; no Java fixture covers non-orthogonal axes. | `partially covered` | Compare a small labeled volume under axis-aligned, oblique, boundary, and out-of-bounds local frames, checking every local `(kw, kv, ku)` cost. |
| `findSurface(float[][][] fx)` | `dp.find_surface_3d` | Repeatedly smooths local costs, solves optimal paths across local slices with strain bounds, and smooths the final surface. | Validates local `(nw, nv, nu)` costs, applies `smooth_fault_attributes_3d`, solves each `w` row with `find_path_2d`, and optionally smooths the `(nw, nv)` surface. | Java DP implementation and smoothing filters differ; Python decomposes the 3D surface into staged 2D helpers. | `partially covered` | Compare flat, sloped, and tie-heavy cost volumes from Java fixtures, including final surface after smoothing. |
| `smoothFaultAttributes(float[][][] fx, float[][][] fs)` | `dp.smooth_fault_attributes_3d` | Applies nonlinear dynamic-programming smoothing in the two surface dimensions and normalizes within the reference workflow. | Smooths local costs along `v` by applying 2D smoothing per `w`, then along `w` by applying 2D smoothing per `v`; returns `float32`. | Java method mutates caller arrays and uses reference accumulation/backtracking helpers; Python returns a new array and does not include all Java normalization side effects. | `partially covered` | Compare constant, impulse, and synthetic surface-valley volumes after one and multiple smoothing passes. |
| `surfaceVoting(...)` | `_surface_voting` | For one seed, builds local axes, samples the UVW cost box, finds an optimal surface, computes average fault attribute and surface strike/dip, then accumulates votes and orientation maps. | Uses `FaultCell` vector helpers, `samples_in_uvw_box`, `dp.find_surface_3d`, `_surface_vote_average`, `_surface_strike_and_dip`, and `_accumulate_surface_votes`. | Several substeps are approximations; Java parallelism and in-place updates can expose ordering differences; full method parity is not established. | `partially covered` | Build a single-seed labeled synthetic plane and compare local costs, picked surface, average value, accumulated vote footprint, and orientation maps. |
| update orientation / vector maps | `_update_orientation_if_stronger` via `_add_surface_vote` | Updates strike/dip maps where the new surface vote is stronger than the stored map value, while accumulating fault evidence. | `_add_surface_vote` adds to `fe`; `_update_orientation_if_stronger` updates `vp`, `vt`, and `vm` only when the new vote is stronger. | Exact Java tie behavior and parallel update ordering are not audited. | `partially covered` | Isolate equal, weaker, and stronger overlapping votes and compare `fe`, `vp`, `vt`, and `vm` transitions. |
| `normalization(float[][][] fx)` / post-vote normalization | `_normalize_and_power_3d` | Normalizes vote evidence and applies the reference post-processing transform before returning fault volume. | Copies the array, optionally smooths with `smooth3d`, scales by the maximum to `[0, 1]`, applies `1 - (1 - x) ** power`, clips, and returns `float32`. | Smoothing kernel, normalization stage, zero-volume handling, and power defaults may differ from Java. | `partially covered` | Compare zero, constant, impulse, and mixed vote volumes before and after smoothing, max scaling, and power transform. |
| `thin(float[][][][] flpt)` | `OptimalSurfaceVoter.thin` / `thinning3d` | Static reference thinning smooths voting likelihoods, keeps strike-binned maxima in the `i2-i3` plane, and handles retained values/orientations according to Java flow. | Default `mode="normal"` keeps maxima along sampled fault normals. Optional `mode="reference"` calls `reference_like_3d_thin_values`, which approximates strike-binned thinning and voter-specific vertical-strike reinforcement. | Default Python behavior is not Java-like; reference mode is an approximation with SciPy smoothing, repository shape conventions, and known reinforcement differences from scanner thinning. | `partially covered` | Compare Java `thin()` against Python reference mode on strike bins, boundary samples, flat regions, and retained-value copying from smoothed versus original arrays. |

## High-risk differences to audit first

- Java rounding versus Python/Numba rounding. `samples_in_uvw_box`,
  `_surface_vote_average`, and vote accumulation use `floor(x + 0.5)`;
  parity tests should verify ties, negative coordinates, and Numba/Python
  agreement.
- `(n3, n2, n1)` global shape versus local `(nw, nv, nu)`. Global volumes
  index samples as `[i3, i2, i1]`, while local costs index `[kw, kv, ku]`.
- Local UVW coordinate sign and axis order. The Python local coordinate formula
  combines `iw * strike + iv * dip + iu * normal`; Java parity should verify
  sign and component order for oblique frames.
- Fault normal / strike / dip vector sign convention. `FaultCell` delegates to
  `geometry` vector helpers; downstream surface orientation and thinning are
  sensitive to sign choices even when scalar angles look plausible.
- `lmins` / `lmaxs` generation. Python uses `update_shift_ranges_3d` with a
  fixed inner radius and Java-style rounding; more reference-derived fixtures
  are needed for multiple radii.
- Boundary handling in `samplesInUvwBox`. Python clamps sampled image indices
  after rounding and leaves disallowed lag positions at cost `1.0`; Java
  behavior should be checked at all volume faces and corners.
- Surface smoothing axes. Python treats local surface arrays as `(nw, nv)` and
  maps `surface_smoothing1` / `surface_smoothing2` to `smooth_surface_2d`
  axes; Java recursive filter axes and edge behavior need direct fixtures.
- Final `thin()` behavior. The default Python normal-mode thinning intentionally
  differs from Java strike-binned thinning; only `mode="reference"` is the
  current approximation target.
- Normalization and power transform. Smoothing, max scaling, zero-volume
  behavior, clipping, dtype conversion, and exponent defaults can all change F3
  metrics without changing seed or surface picking logic.

## Current audit entry points

- `src/pyosv/voting3d.py`: public 3D voter API, UVW sampling, seed voting,
  vote accumulation, normalization, and voter thinning wrapper.
- `src/pyosv/dp.py`: lag ranges, strain conversion, 2D/3D dynamic programming,
  cost smoothing, and surface smoothing.
- `src/pyosv/thinning3d.py`: reference-like strike-binned 3D thinning helpers.
- `src/pyosv/geometry.py` and `src/pyosv/cells.py`: fault-vector and cell
  conventions used by voter seeds.
- `tests/test_voting3d.py`, `tests/test_dp.py`, and `tests/test_thinning3d.py`:
  current Python regression coverage. These are not a substitute for
  method-level Java parity fixtures unless they explicitly compare Java-derived
  expected values.
