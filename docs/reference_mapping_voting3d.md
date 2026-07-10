# 3D Voter Reference Mapping

This document maps selected
`reference_osv/src/osv/OptimalSurfaceVoter.java` methods to the current
`pyosv` 3D voting implementation. It is an audit guide, not an equivalence
claim. The current Python implementation keeps repository conventions:
global arrays are `(n3, n2, n1)`, local voting costs are `(nw, nv, nu)`, and
values are normalized to `np.float32` where practical.

`pyosv` follows a reference-first policy for fault interpretation workflows,
using practical agreement metrics to measure remaining differences. It does not
add runtime dependencies on JVM, Jython, Mines JTK, or Gradle. Java
`SincInterpolator`, `RecursiveExponentialFilter`, and related JTK filters are
therefore approximation targets for Python/SciPy code, not dependencies.

## Current Migration Notes

- `OptimalSurfaceVoter.thin(...)` defaults to `mode="reference"`: a
  reference-like, strike-binned thinning path. Pass `mode="normal"` to reproduce
  the older fault-normal voter thinning behavior.
- Voter reference-like thinning includes voter-specific retained-sample
  reinforcement and does not apply scanner edge-effect cleanup.
- Surface voting defaults to `surface_voting_boundary_policy="reference"`.
  This preserves both reference-style UVW image-coordinate clamping and the
  Java-like target-point condition: `i1` may touch either face, but `i2` and
  `i3` source samples must be interior for averaging and accumulation.
- `surface_voting_boundary_policy="masked_in_bounds"` is an opt-in Python
  quality experiment. It masks out-of-volume UVW lags before surface extraction
  and allows center votes on all six faces. It is not a Java-equivalence mode
  and is not enabled by any default workflow.
- `OptimalSurfaceVoter.apply_voting()` defaults to reference-style final
  normalization with no final vote-map smoothing: subtract min, divide by max
  when `max > 0`, then apply `1 - (1 - x) ** 8`. Leave
  `set_final_normalization_smoothing(...)` unset for reference-first workflows;
  call `set_final_normalization_smoothing(1.0)` only to compare with older
  pyosv runs that smoothed the final vote map before normalization.
- The mapping below is reference-first, not bit-exact. F3 reports and local
  regression tests are comparison evidence, not acceptance thresholds for Java
  equivalence.

## Method-Level Mapping

| Java method | Python equivalent / status | Reference summary | Current Python summary | Known differences | Audit status | Suggested future parity test |
| --- | --- | --- | --- | --- | --- | --- |
| `OptimalSurfaceVoter(int ru, int rv, int rw)` constructor | `OptimalSurfaceVoter.__init__` | Stores local radii, initializes lag bounds from `ru`, default strain inverse values of 4, one attribute smoothing, surface smoothing extents of 2.0, a recursive Gaussian filter sized by `max(rv, rw)` for surface orientation, JTK smoothing filters, and `_lmins` / `_lmaxs`. | Validates nonnegative radii, stores `lmin`, `lmax`, `nl`, default `bstrain1` / `bstrain2`, surface-extraction smoothing settings, `surface_orientation_smoothing=float(max(rv, rw))`, and generates `lmins` / `lmaxs` with `dp.update_shift_ranges_3d`. | Python has explicit validation and no JTK filter objects; surface-orientation smoothing is a stored scalar applied with SciPy-backed surface smoothing when strike/dip are re-estimated; shift-range generation is Python code and only approximates the reference contract. | `partially covered` | Add a Java-derived fixture for several `(ru, rv, rw)` combinations, including edge radii, and compare `lmin`, `lmax`, `nl`, `lmins`, `lmaxs`, and default orientation smoothing. |
| no Java equivalent | `set_surface_voting_boundary_policy` | The reference has one boundary behavior: rounded UVW image samples are clamped, then `i2`/`i3` face source samples are excluded during voting. | Accepts exactly `"reference"` and `"masked_in_bounds"`; the constructor default is `"reference"`. The masked policy selects a separate quality path and does not change `samples_in_uvw_box`. | `masked_in_bounds` deliberately departs from reference behavior, so its tests are Python regression and synthetic-quality evidence rather than Java parity evidence. | `python-regression covered` | Keep constructor/default/report tests proving that reference workflows never select the masked path implicitly. |
| `setStrainMax(double strainMax1, double strainMax2)` | `set_strain_max` | Converts maximum surface strains to integer inverse strain bounds with ceiling-like behavior. | Uses `dp.strain_to_bstrain`, validating `0 < strain_max <= 1`, and updates only `bstrain1` / `bstrain2`. | Python validation is explicit; no Java parity fixture verifies all rounding cases. | `partially covered` | Compare Java and Python conversion for representative values near reciprocal boundaries, including values that should round up. |
| `setAttributeSmoothing(int esmooth)` / smoothing config | `set_attribute_smoothing` | Sets the number of nonlinear fault-attribute smoothing passes. | Validates a nonnegative integer and stores `attribute_smoothing`. | Java may allow values Python rejects by validation; smoothing implementation differs downstream. | `partially covered` | Check that zero, one, and multiple smoothing passes affect a small cost volume at the same stage as Java. |
| `setSurfaceSmoothing(double usmooth1, double usmooth2)` / smoothing config | `set_surface_smoothing` | Sets surface smoothing extents and rebuilds recursive exponential filters. | Validates nonnegative finite extents and stores `surface_smoothing1` / `surface_smoothing2`; smoothing is applied later through `dp.smooth_surface_2d`. | Python does not maintain JTK filter instances; SciPy smoothing response and edge behavior differ. | `partially covered` | Compare extracted surface smoothing on impulse, step, and sloped synthetic surfaces, focusing on axis order and edge behavior. |
| `surfaceStrikeAndDip(...)` / surface-orientation smoothing config | `_surface_strike_and_dip` / `set_surface_orientation_smoothing` | Constructor creates a `RecursiveGaussianFilter(max(rv,rw))` and `surfaceStrikeAndDip` applies it to the extracted surface before strike/dip are recomputed. | Defaults to `float(max(rv, rw))`, validates nonnegative finite values, and passes the value to `_surface_strike_and_dip`; `0.0` disables orientation-only surface smoothing and keeps the older raw-surface diagnostic path. | Python stores a scalar instead of a filter object and approximates the smoothing with `dp.smooth_surface_2d`; SciPy boundary and kernel behavior is not expected to be a bit-exact Mines JTK clone. | `python-regression covered` | Compare noisy and stair-step Java-derived surfaces before/after smoothing to verify practical orientation-jitter reduction and sign conventions. |
| `pickSeeds(...)` | `pick_seeds` | Selects seed candidates above the thinned fault-attribute threshold, sorts by likelihood, and suppresses nearby lower candidates within a distance box. | Builds `FaultCell` objects from `(n3, n2, n1)` arrays, sorts descending by likelihood, suppresses candidates inside the distance box, and returns a Python list. | Java object ordering and equal-likelihood tie behavior are not audited; Python validates finite matching arrays. | `partially covered` | Build a fixture with equal likelihoods, boundary candidates, and suppression overlaps; compare selected seed coordinates and order. |
| `getSeeds(...)` | `get_seeds` | Reference utility for selecting one seed at a requested sample. | Returns a single `FaultCell` at `(c1, c2, c3)` after bounds and shape validation. | Python exposes a narrow helper; no Java parity fixture confirms reference call semantics. | `partially covered` | Compare one-sample seed extraction for valid and boundary coordinates, including stored strike and dip. |
| `updateVectorMap(...)` | `update_vector_map` | Fills displacement vectors for offsets in `[-radius, radius]` along the supplied local axis vector. | Returns a `(3, 2 * radius + 1)` `float32` array from vector components multiplied by offsets. | Java writes into caller-provided arrays and uses Java float arithmetic; Python allocates and returns a new array. | `partially covered` | Compare vector maps for non-axis-aligned unit and non-unit vectors, including negative components. |
| `samplesInUvwBox(...)` | `samples_in_uvw_box` plus the private masked sampler | Samples `1 - fx` in a seed-centered local UVW box using normal, dip, and strike axes; invalid lag cells stay at default cost and image samples are rounded/clamped. | The public method retains the reference path and returns `(nw, nv, nu)` costs using Java-style `floor(x + 0.5)` rounding and clamping. The private masked path returns `float32` costs, a boolean mask requiring both lag-range admission and an in-volume rounded sample, full-box offsets, and lag counts; it never clamps masked samples. | The public path remains the Java approximation target. The masked sampler is a Python quality extension with Python/Numba parity coverage across faces, corners, oblique axes, and rounding boundaries. | `partially covered` | Keep masked-path evidence separate from Java parity fixtures so the opt-in extension cannot weaken reference-clamping audits. |
| `findSurface(float[][][] fx)` | `dp.find_surface_3d` plus a private masked surface extractor | Repeatedly smooths local costs, solves optimal paths across local slices with strain bounds, and smooths the final surface. | The public finite-cost path is unchanged. The masked path excludes invalid states during attribute smoothing, forward/reverse accumulation, and backtracking. After surface smoothing it revalidates mask membership and strain in both tangential directions, then performs deterministic global feasibility recovery when necessary. A value already inside a valid Java-rounding cell is not moved to the integer center unnecessarily; failure to recover a jointly feasible surface is reported safely. `surface_projection_count` is the number of `(w, v)` columns changed between the raw smoothed surface and the final mask-and-strain-feasible surface, counted once per column. | Masked DP is not present in the reference. Python and accelerated results are compared within `float32` tolerance, including infeasible and recovery/projection cases. | `partially covered` | Continue Java fixtures against `find_surface_3d`; treat mask/recovery/projection tests as quality-extension regression coverage. |
| `smoothFaultAttributes(float[][][] fx, float[][][] fs)` | `dp.smooth_fault_attributes_3d` | Applies nonlinear dynamic-programming smoothing in the two surface dimensions and normalizes within the reference workflow. | Smooths local costs along `v` by applying 2D smoothing per `w`, then along `w` by applying 2D smoothing per `v`; returns `float32`. | Java method mutates caller arrays and uses reference accumulation/backtracking helpers; Python returns a new array and does not include all Java normalization side effects. | `partially covered` | Compare constant, impulse, and synthetic surface-valley volumes after one and multiple smoothing passes. |
| `surfaceVoting(...)` | `_surface_voting` | For one seed, builds local axes, samples the UVW cost box, finds an optimal surface, computes average fault attribute and smoothed-surface strike/dip, then accumulates votes and orientation maps. Vote surface points are accepted only when `0 <= i1 < n1`, `0 < i2 < n2 - 1`, and `0 < i3 < n3 - 1`. | The `reference` branch keeps those helpers and predicates unchanged. The masked branch selects the deterministic maximum supported rectangle containing the origin, maps it with explicit full-box offsets, uses only valid selected samples, permits center writes on all six faces, and emits immutable per-seed diagnostics. Full-box orientation uses `_surface_strike_and_dip`; a cropped box uses seed strike/dip with `orientation_source="seed_boundary_fallback"`. | The masked rectangle, mask-aware DP, face-center votes, and crop orientation fallback are quality extensions, not reference parity. Support fraction is measured against the full tangential patch; any selected invalid sample skips the vote. | `partially covered` | Compare the reference branch with Java-derived single-seed fixtures and audit the masked branch with face, padding/crop, axis-permutation, and diagnostic regression tests. |
| update orientation / vector maps | `_update_orientation_if_stronger` via `_add_surface_vote` | Updates strike/dip maps where the new surface vote is stronger than the stored map value, while accumulating fault evidence. | `_add_surface_vote` adds to `fe`; `_update_orientation_if_stronger` updates `vp`, `vt`, and `vm` only when the new vote is stronger. | Exact Java tie behavior and parallel update ordering are not audited. | `reference-audit covered` | Add Java-derived equal-vote fixtures to confirm strict greater-than tie behavior under reference ordering. |
| `normalization(float[][][] fx)` / post-vote normalization | `_normalize_and_power_3d` / `set_final_normalization_smoothing` | Normalizes vote evidence and applies the reference post-processing transform before returning fault volume. | Copies the array, subtracts the global minimum, divides by the global maximum when nonzero, applies `1 - (1 - x) ** power`, clips, and returns `float32`. The default is no final vote-map smoothing; `set_final_normalization_smoothing(sigma)` opts into Python's older practical smoothing before normalization. | Smoothing remains opt-in and uses a SciPy-backed approximation, so smoothed diagnostics are not bit-exact Mines JTK output. Current reference-audit tests cover finite output, `[0, 1]` range, zero dynamic-range input, negative offsets, default versus smoothed synthetic ridges, and setter wiring. | `partially covered` | Compare zero, constant, impulse, and mixed vote volumes from Java before and after min subtraction, max scaling, and power transform. |
| `thin(float[][][][] flpt)` | `OptimalSurfaceVoter.thin` / `thinning3d` | Static reference thinning smooths voting likelihoods, keeps strike-binned maxima in the `i2-i3` plane, and handles retained values/orientations according to Java flow. | Default `mode="reference"` calls `reference_like_3d_thin_values`, which approximates strike-binned thinning and voter-specific vertical-strike reinforcement. Explicit `mode="normal"` keeps the legacy maxima along sampled fault normals. | Default Python behavior now follows the reference-like target, but remains an approximation with SciPy smoothing, repository shape conventions, and known reinforcement differences from scanner thinning. Current reference-audit tests cover folded-strike boundary reinforcement for the voter and verify scanner reference thinning does not apply voter reinforcement. | `partially covered` | Compare Java `thin()` against Python reference mode on strike bins, boundary samples, flat regions, and retained-value copying from smoothed versus original arrays. |

## 3D Thinning And Boundary Audit

| Reference behavior | Python status | Follow-up decision |
| --- | --- | --- |
| `FaultOrientScanner3.thin(...)` smooths only the `i3` and `i2` axes, applies strike-binned strict local maxima in the `i2-i3` plane, writes retained smoothed likelihoods with retained input strike/dip, assigns Java `NO_STRIKE` / `NO_DIP` to rejected orientations, then calls `removeEdgeEffects(...)`. | `FaultOrientScanner3.thin(mode="reference")` uses `reference_like_3d_thin_values(..., reinforce_vertical=False)`, returns zero orientation sentinels at rejected samples, and applies scanner edge-effect removal by default. | Edge-effect removal is scanner-specific in the Java reference. Keep it as a scanner-compatible post-thinning cleanup and do not apply it to voter thinning by default. Keep Python's public zero orientation sentinel for compatibility instead of introducing Java `-0.00001` sentinels into returned arrays. |
| `FaultOrientScanner3.removeEdgeEffects(...)` zeros likelihood, strike, and dip within five samples of the `i3` faces when the squared fault-normal `w3` component exceeds `cos(30 deg)^2`, and within five samples of the `i2` faces when `w2^2` exceeds the same threshold. It does not remove `i1` face samples. | `remove_reference_edge_effects_3d` implements this cleanup, and `FaultOrientScanner3.thin(mode="reference")` applies it by default unless `remove_edge_effects=False` is passed. | Treat this as a post-thinning scanner cleanup, not as part of the shared NMS mask. Tests cover `i2` and `i3` faces separately and leave `i1` faces unaffected. |
| `OptimalSurfaceVoter.thin(...)` uses the same strike-binned NMS structure but does not call `removeEdgeEffects(...)`. It writes Java sentinels only at rejected orientations and includes the voter-only neighbor write for retained near-vertical folded strikes; the audited Java horizontal condition is unreachable as written (`p000 < 30 && p000 > 150`). | `OptimalSurfaceVoter.thin(...)` defaults to `mode="reference"` and calls the shared helper with `reinforce_vertical=True`; explicit `mode="normal"` remains the legacy Python normal-vector thinning path. The public method returns only thinned values, so Java rejected-orientation sentinels are not exposed. | Keep scanner and voter reference-like thinning as separate call sites over the shared helper because the voter has reinforcement behavior and no edge-effect removal. |
| `OptimalSurfaceVoter.surfaceVoting(...)` accepts vote surface points only when `0 <= i1 < n1`, `0 < i2 < n2 - 1`, and `0 < i3 < n3 - 1`; neighbor reinforcement then writes `i2 +/- 1` or `i3 +/- 1` without additional edge checks. `samplesInUvwBox(...)` and `screenPoints(...)` use the same interior `i2`/`i3` condition for their in-box checks. | `_surface_vote_average_python/_numba` and `_accumulate_surface_votes_python/_numba` now use the same Java-style target-point predicate. `_add_surface_vote` still bounds-checks reinforced neighbors as defensive Python behavior. | Keep the average and accumulation point sets narrowed together; do not treat write-side bounds checks as permission to average or accumulate from `i2`/`i3` face source samples. |
| No reference counterpart exists for masked UVW evidence or boundary-face center votes. | `surface_voting_boundary_policy="masked_in_bounds"` uses only lag-range-admissible, in-volume samples, selects the deterministic maximum supported origin-containing rectangle, preserves its full-box crop offsets, extracts a mask-feasible surface, and permits center writes on all six faces. | Keep this path opt-in and separately audited. Do not reinterpret its synthetic-quality results as Java equivalence, and do not replace it with scanner, thinning, or skinner boundary fallback behavior. |

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
- Boundary handling in `samplesInUvwBox`. The public/reference policy clamps
  sampled image indices after rounding and leaves disallowed lag positions at
  cost `1.0`; Java behavior should be checked at all volume faces and corners.
  The separate masked policy must preserve its boolean validity mask through
  sampling, smoothing, accumulation, and backtracking instead of encoding an
  invalid state as an ordinary numeric cost.
- Cropped masked surfaces. Volume mapping must use explicit full-box `w` / `v`
  offsets. Equal-area origin-containing supported rectangles are resolved by
  origin asymmetry and then lexicographic bounds, so Python and accelerated
  paths cannot choose different boundary support.
- Masked surface feasibility and smoothing. A supported column set can still
  be infeasible under strain limits. After smoothing, mask validity and strain
  are checked again in both tangential directions, with global feasibility
  recovery when a per-column choice would not form a valid surface. Values that
  remain feasible within their Java-rounding cells are not snapped to integer
  centers. Failure is a diagnosed `no_feasible_surface` skip.
- Masked surface projection diagnostics. `surface_projection_count` compares
  the raw smoothed surface with the final mask-and-strain-feasible surface and
  counts each `(w, v)` column whose value changed exactly once; it is not a
  count of recovery iterations or nearest-interval operations. It remains zero
  when surface smoothing is disabled.
- Surface smoothing axes. Python treats local surface arrays as `(nw, nv)` and
  maps `surface_smoothing1` / `surface_smoothing2` to `smooth_surface_2d`
  axes; Java recursive filter axes and edge behavior need direct fixtures.
- Surface orientation smoothing. Python smooths extracted `(nw, nv)` surfaces
  before center-difference strike/dip re-estimation when
  `surface_orientation_smoothing > 0.0`; this maps to the reference
  `surfaceStrikeAndDip` use of `RecursiveGaussianFilter(max(rv,rw))`, but
  Java recursive Gaussian behavior and boundary handling need direct fixtures.
- Final normalization smoothing. Python does not smooth the final vote map by
  default. The optional `set_final_normalization_smoothing(sigma)` path smooths
  accumulated `fe` before min/max normalization and the `1 - (1 - x) ** 8`
  transform; it is a practical comparison mode, not the reference default.
- Surface-voting boundaries. Java permits all in-range `i1` samples but
  excludes `i2` and `i3` face samples before averaging and accumulating votes;
  Python voting helpers now use the same target-point predicate.
- Final `thin()` behavior. Scanner and voter defaults now use reference-like
  strike-binned thinning, but the implementation remains an approximation
  target rather than bit-exact Java behavior.
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
