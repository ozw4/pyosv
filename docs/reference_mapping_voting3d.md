# 3D Voter Reference Mapping

This document maps concepts from
`reference_osv/src/osv/OptimalSurfaceVoter.java` to the Python implementation
exposed as `pyosv.voting3d.OptimalSurfaceVoter`.

The Python voter preserves the reference processing structure where practical,
but it is not a bit-exact Mines JTK port. PyOSV uses NumPy, SciPy, and optional
Numba kernels and has no JVM, Jython, Gradle, or Mines JTK runtime dependency.

Global image volumes use shape `(n3, n2, n1)` and indexing
`array[i3, i2, i1]`. Local voting costs use shape `(nw, nv, nu)` and indexing
`cost[kw, kv, ku]`. Geometric vectors use component order `(x1, x2, x3)`:

```text
u = fault normal
v = fault-dip vector
w = fault-strike vector
```

A local point maps to global coordinates as:

```text
seed + iw * strike + iv * dip + iu * normal
```

## Equivalence boundary

The reference-oriented path retains these structural stages:

```text
seed selection
  -> seed-local UVW cost sampling
  -> dynamic-programming surface extraction
  -> surface likelihood and orientation estimation
  -> vote and orientation accumulation
  -> final vote normalization
  -> voter thinning
```

Numerical equality with Java is not required. The principal implementation
boundaries are:

- SciPy Gaussian smoothing instead of JTK recursive filters;
- Python and optional Numba kernels instead of Java array kernels;
- explicit `float32`, shape, finiteness, and scalar validation;
- Python-defined deterministic tie ordering and defensive bounds checks;
- a mask-aware boundary policy and additional thinning policies that have no
  direct Java method counterpart.

Reference agreement is assessed through deterministic regression tests,
controlled Synthetic truth metrics, and F3 public-reference metrics. F3 public
outputs are comparison targets, not geological truth or method-level Java
fixtures.

## Constructor and configuration mapping

| Java symbol or operation | Python symbol | Python contract | Equivalence boundary | Test coverage |
| --- | --- | --- | --- | --- |
| `OptimalSurfaceVoter(int ru, int rv, int rw)` | `OptimalSurfaceVoter.__init__` | Requires nonnegative integer radii. Sets `lmin=-ru`, `lmax=ru`, `nl=2*ru+1`, `bstrain1=bstrain2=4`, `attribute_smoothing=1`, surface smoothing `(2.0, 2.0)`, orientation smoothing `max(rv, rw)`, orientation backend `full_surface`, final-normalization smoothing `0.0`, support policy `(0.0, 0.0)`, and boundary policy `reference`. It builds `(2*rw+1, 2*rv+1)` `lmins` and `lmaxs`. | Python stores scalar configuration rather than JTK filter objects and validates every public setting explicitly. | Constructor defaults, invalid radii, lag bounds, shift arrays, and per-instance state are covered in `tests/test_voting3d.py`. |
| `setStrainMax(double, double)` | `set_strain_max()` | Converts each maximum strain with `ceil(1 / strain_max)`. Each value must satisfy `0 < strain_max <= 1`. Only `bstrain1` and `bstrain2` change. | Python exposes the reciprocal spacing directly and rejects invalid values before execution. | Reciprocal boundaries, defaults, invalid values, and unchanged shift ranges are covered in `tests/test_voting3d.py` and `tests/test_dp.py`. |
| `setAttributeSmoothing(int)` | `set_attribute_smoothing()` | Stores a nonnegative integer number of nonlinear cost-smoothing passes used before surface extraction. | The smoothing kernels are Python/Numba DP accumulation kernels rather than Java/JTK kernels. | Zero, one, multiple passes, invalid values, and cost-volume behavior are covered in `tests/test_voting3d.py` and `tests/test_dp.py`. |
| `setSurfaceSmoothing(double, double)` | `set_surface_smoothing()` | Stores two nonnegative finite surface-smoothing extents. For a surface shaped `(nw, nv)`, `surface_smoothing1` acts on the `v` axis and `surface_smoothing2` acts on the `w` axis through `smooth_surface_2d`. | Smoothing uses SciPy-backed Gaussian filters and nearest-edge behavior. | Setter validation, axis behavior, constant and sloped surfaces, and deterministic output are covered in `tests/test_voting3d.py` and `tests/test_dp.py`. |
| `surfaceStrikeAndDip(...)` and its constructor filter | `_surface_strike_and_dip()` and `set_surface_orientation_smoothing()` | Re-estimates strike and dip from centered `du/dv` and `du/dw` differences on a finite `(nw, nv)` surface. The default smoothing is `float(max(rv, rw))`. `0.0` uses the unsmoothed surface. | Java recursive Gaussian filtering is represented by SciPy-backed smoothing. The Python API also exposes the `full_surface` and `center_separable` computation backends. | Flat, planar, noisy, stair-step, invalid-size, backend-agreement, and nonmutation behavior are covered in `tests/test_voting3d.py`. |
| final vote normalization configuration | `set_final_normalization_smoothing()` | Stores a nonnegative finite sigma applied only to accumulated vote evidence before final unit-range normalization. The default `0.0` performs no smoothing. | This optional smoothing is a Python configuration axis; the reference-oriented default remains unsmoothed. | Setter validation, wiring, bounded output, and unchanged orientation arrays are covered in `tests/test_voting3d.py`. |
| no direct Java setter | `set_surface_support_policy()` | Stores `min_fraction` in `[0, 1]` and a nonnegative finite exponent. A seed is skipped below the minimum; otherwise its average vote is multiplied by `support_fraction ** exponent` when the exponent is positive. Defaults are a no-op. | Support-aware skipping and weighting are Python voting controls. | Validation, no-op defaults, threshold skipping, and down-weighting are covered in `tests/test_voting3d.py`. |
| one reference boundary behavior | `set_surface_voting_boundary_policy()` | Accepts exactly `reference` or `masked_in_bounds`; the default is `reference`. The selection is snapshotted into each seed execution and its diagnostics. | `masked_in_bounds` is a Python-specific policy with a separate sampling, DP, and accumulation contract. | Policy validation, dispatch, state isolation, reference behavior, masked behavior, and diagnostic reporting are covered in `tests/test_voting3d.py`. |

## Method and helper mapping

| Java symbol or operation | Python symbol | Python contract | Equivalence boundary | Test coverage |
| --- | --- | --- | --- | --- |
| `pickSeeds(...)` | `pick_seeds()` | Selects samples with `ft > fm`, orders them by descending likelihood, and applies exact greedy Chebyshev-box suppression with radius `d`. Equal likelihoods use descending flat-index order. Returned `FaultCell` objects retain `(i1, i2, i3)`, likelihood, strike, and dip. | Python makes tie ordering and bounds normalization explicit. The seed loop may use a Numba suppression kernel, with the same accepted-index contract. | Threshold strictness, ordering, suppression, boundaries, dtype conversion, nonmutation, and Python/Numba agreement are covered by seed and voter tests. |
| `getSeeds(...)` | `get_seeds()` | Validates one `(c1, c2, c3)` sample and returns a one-element list containing its `FaultCell`. No likelihood threshold is applied. | The helper is deliberately narrow and follows the Python list-based API. | Coordinate mapping, stored values, shape checks, and bounds errors are covered in `tests/test_voting3d.py`. |
| `updateVectorMap(...)` | `update_vector_map()` | Returns a `float32` array with shape `(3, 2*radius+1)` containing the supplied vector multiplied by offsets from `-radius` through `radius`. | Python allocates and returns the map instead of mutating caller-owned arrays. | Symmetry, shape, dtype, values, and invalid inputs are covered in `tests/test_voting3d.py`. |
| `samplesInUvwBox(...)` | `samples_in_uvw_box()` and reference sampling kernels | Produces `(nw, nv, nu)` costs initialized to `1.0`. Only lags admitted by `lmins/lmaxs` are sampled. Coordinates use `floor(x + 0.5)`, then each global index is clamped to the volume before storing `1 - fx`. | Java-style rounding and clamping are retained; arithmetic is explicitly staged through `float32`. A support-aware internal sampler records pre-clamp in-bounds counts without changing costs. | Axis order, oblique frames, shift masks, face/corner clamping, half-boundary rounding, support counts, and Python/Numba agreement are covered in `tests/test_voting3d.py` and `tests/test_voting_accel.py`. |
| lag-range initialization | `shift_range()` and `update_shift_ranges_3d()` | Uses `[-ru, ru]` normal lags. For tangential offset radius `sqrt(iw**2 + iv**2) <= 2`, the admissible normal lag is zero. Outside that radius, the absolute bound is `floor(radius + 0.5)`, clipped to `ru`. | Python emits explicit `int32` `(nw, nv)` arrays. | Shape, symmetry, radial cutoff, Java rounding, and clipping are covered in `tests/test_voting3d.py` and `tests/test_dp.py`. |
| `findSurface(float[][][] fx)` | `find_surface_3d()` | Accepts finite `(nw, nv, nu)` costs. It applies the configured number of 3-D attribute-smoothing passes, finds one optimal lag path for each `w` row, optionally smooths the `(nw, nv)` surface, and returns `float32` lags. Flat-cost path ties prefer the center lag. | DP accumulation, backtracking, and smoothing are Python/Numba implementations. The unmasked path does not carry a validity mask. | Straight, sloped, noisy, bounded-strain, flat-tie, smoothing, shape, dtype, and Python/Numba behavior are covered in `tests/test_dp.py` and `tests/test_dp_accel.py`. |
| `smoothFaultAttributes(float[][][] fx, float[][][] fs)` | `smooth_fault_attributes_3d()` | Applies forward/reverse DP smoothing along `v`, transposes the volume, then applies the corresponding smoothing along `w`. It returns a new `float32` cost volume. | Java mutation and JTK internals are replaced by explicit returned arrays and Python/Numba batch kernels. | Constant, impulse, path-valley, direction, shape, and acceleration-equivalence behavior are covered in DP tests. |
| `surfaceVoting(...)` | reference policy in `SURFACE_VOTING_POLICY_REGISTRY` | For one seed, samples reference-clamped costs, extracts a surface, averages valid surface likelihoods, applies support controls, re-estimates orientation, and accumulates center and reinforcement votes. It records one immutable diagnostic result. | The seed loop is sequential and deterministic. Numerical kernels and defensive write bounds are Python-specific. | Plane voting, deterministic repeatability, source-boundary exclusion, support weighting, orientation smoothing, accumulation, and diagnostics are covered in `tests/test_voting3d.py`. |
| `screenPoints(...)` and surface-point acceptance | `_is_valid_surface_vote_sample()` in scoring and accumulation | A reference-policy center sample is valid when `0 <= i1 < n1`, `0 < i2 < n2-1`, and `0 < i3 < n3-1`. Thus `i1` faces are permitted while `i2` and `i3` faces are excluded. Scoring and accumulation use the same predicate. | Neighbor reinforcement writes use defensive full-volume bounds checks after an accepted center sample. | Face behavior, zero valid support, scoring/accumulation agreement, and bounded reinforcement are covered in `tests/test_voting3d.py`. |
| strongest-vote orientation update | `_add_surface_vote()` and `_update_orientation_if_stronger()` | Every accepted write adds `fa` to `fe`. `vp`, `vt`, and the internal maximum map `vm` change only when `fa > vm`; equal or weaker votes retain the earlier orientation. | Python makes the strict comparison and write bounds explicit. | Stronger, weaker, equal-threshold, neighbor, and Python/Numba accumulation behavior are covered in voter tests. |
| `normalization(float[][][] fx)` | `_normalize_and_power_3d()` | Optionally smooths when sigma is positive, subtracts the global minimum, divides by the post-subtraction maximum when positive, applies `1 - (1 - x) ** power`, clips to `[0, 1]`, and returns `float32`. `apply_voting()` uses sigma `0.0` and power `8` by default. Constant input becomes zero. | SciPy smoothing is optional and separate from the reference-oriented normalization formula. | Negative offsets, constant input, monotonic ramps, exact formula, optional smoothing, output bounds, and nonmutation are covered in `tests/test_voting3d.py`. |
| `thin(float[][][][] flpt)` | `OptimalSurfaceVoter.thin(..., mode="reference")` | Smooths `fv` only along `i3` and `i2`, applies strict strike-binned nonmaximum suppression in the `i2-i3` plane, and writes the smoothed retained values. Voter thinning enables one-sided retained-sample reinforcement for folded strikes strictly between 60 and 120 degrees. | SciPy Gaussian smoothing replaces JTK filtering. The Python method returns only the thinned value volume and applies no scanner edge cleanup. | Strike bins, retained values, reinforcement, flat regions, input preservation, and separation from scanner cleanup are covered in `tests/test_voting3d.py` and `tests/test_thinning3d.py`. |

## Reference surface-voting contract

The `reference` policy uses the following fixed behavior.

1. `apply_voting()` validates matching finite `ft`, `pt`, and `tt` volumes.
2. Seed selection uses the input `ft`, `pt`, and `tt`.
3. Before seed-local voting, `ft` is smoothed with sigma `1.0` and normalized to
   `[0, 1]`. The smoothed normalized array is the local evidence volume.
4. UVW costs use `1 - evidence` at admitted lags. Rounded image coordinates are
   clamped independently on all three axes.
5. The extracted surface is scored and accumulated only at points satisfying
   the reference target predicate. A surface with no valid point is skipped.
6. The reference support fraction is `valid_surface_point_count / surface.size`.
7. Surface orientation is computed from the extracted surface with the
   configured smoothing and backend.
8. Each accepted center write is reinforced on `i3-1` and `i3+1` when
   `abs(normal[2]) > abs(normal[1])`; otherwise it is reinforced on `i2-1` and
   `i2+1`.
9. Final `fv` normalization uses the configured final-normalization sigma and
   power `8`. `vp` and `vt` are the orientations of the strongest individual
   local vote, not the orientation of accumulated `fe`.

The public `samples_in_uvw_box()` method always implements reference clamping.
Selecting another surface-voting policy does not change that public helper.

## Masked in-bounds policy

`masked_in_bounds` is a Python-specific boundary policy. It has this contract:

- UVW lags are valid only when they satisfy `lmins/lmaxs` and their
  Java-rounded global coordinates are inside the volume.
- Out-of-bounds lags are not clamped. Their cost remains `1.0` and their mask is
  false.
- A tangential column is supported when at least one normal lag is valid.
- The selected domain is the largest all-supported axis-aligned rectangle
  containing the local origin. Ties prefer lower origin asymmetry and then
  lexicographically smaller `(w_start, v_start, w_stop, v_stop)`.
- Cropped rectangles retain explicit full-box `w_offset` and `v_offset`.
- Masked attribute smoothing, accumulation, and backtracking exclude invalid
  states.
- After optional surface smoothing, mask membership and strain feasibility are
  checked in both tangential directions. Deterministic global recovery is used
  when the smoothed surface is not jointly feasible.
- A fractional lag already inside a valid Java-rounding cell is retained when
  it satisfies the mask and strain constraints.
- Scoring and accumulation validate every selected lag. A detected invalid
  sample skips the seed without partially committing votes.
- Support fraction is measured against the full tangential area
  `(2*rw+1)*(2*rv+1)`, not the cropped rectangle.
- Center writes may occur on all six volume faces; reinforcement writes remain
  bounds checked.
- A full surface of at least `3 x 3` uses surface-derived orientation. A cropped
  surface uses the seed orientation and records
  `orientation_source="seed_boundary_fallback"`. A smaller full surface records
  `orientation_source="seed_small_surface_fallback"`.

A seed can be skipped with one of these stable reasons:

```text
no_supported_origin
no_feasible_surface
no_valid_surface_samples
support_below_min_fraction
invalid_selected_sample
```

`surface_projection_count` is the number of `(w, v)` columns whose value differs
between the raw smoothed surface and the final mask-and-strain-feasible surface.
Each changed column is counted once.

The immutable per-seed diagnostic records:

```text
seed_index
policy
full_tangential_column_count
selected_tangential_column_count
admissible_lag_count
in_bounds_lag_count
support_fraction
surface_center_lag
surface_projection_count
selected_invalid_sample_count
center_vote_write_count
face_center_vote_count
orientation_source
skipped
skip_reason
```

`surface_voting_diagnostic_summary()` returns JSON-safe aggregate fields for the
most recent run and uses explicit left-to-right support-fraction accumulation so
its serialized result is stable across supported Python versions.

## Dynamic-programming contract

The unmasked and masked surface extractors share the local shape contract:

```text
cost shape    = (nw, nv, nu)
surface shape = (nw, nv)
surface value = selected u lag
```

`bstrain1` and `bstrain2` are positive integer reciprocal-strain spacings.
Three-dimensional attribute smoothing uses both values. The unmasked extractor
finds each `w` row through the `v-u` cost image and optionally smooths the final
surface.

The masked extractor additionally guarantees that selected Java-rounded lag
states are valid and that the final surface satisfies the configured strain
limits in both tangential directions. It returns `None` when no joint feasible
surface exists. Its projection count describes post-smoothing correction, not
the number of search iterations.

## Python thinning policies without direct Java method mapping

The public voter accepts these additional policies:

| Mode | Contract |
| --- | --- |
| `normal` | Smooths `fv` with sigma `1.0`, samples the smoothed volume one fault-normal step in both directions with linear nearest-boundary interpolation, keeps strict maxima, and writes original `fv` values. |
| `normal_plateau` | Uses non-strict normal comparisons within `plateau_tolerance`, groups candidates along each sample's dominant normal axis, and retains the largest `plateau_tie_breaker` value; an all-equal run retains its center sample. |
| `hybrid` | Computes orientation roughness on positive vote support and selects reference thinning in stable regions and normal-thinning output in rough regions. |
| `hybrid_v2` | Uses support above `1e-6`, starts from reference thinning, adopts positive normal candidates in rough regions, and uses plateau candidates near all volume faces when normal output is absent or locally sparse. |

These modes are explicit voter-thinning settings. They do not select a scanner
backend, scanner thinning mode, surface-voting boundary policy, or workflow by
themselves.

## Scanner and voter thinning distinction

Scanner and voter reference thinning share the strike-binned NMS helper but
have different public contracts.

| Property | Scanner reference thinning | Voter reference thinning |
| --- | --- | --- |
| Input | scanner `ft`, `pt`, `tt` | voter `fv`, `vp`, `vt` |
| Retained value | smoothed scanner likelihood | smoothed vote likelihood |
| Vertical-strike reinforcement | disabled | enabled |
| Edge cleanup | applied by default on selected `i2` and `i3` faces | not applied |
| Returned data | likelihood, strike, and dip volumes | thinned likelihood volume only |
| Rejected orientation value | zero in returned strike/dip arrays | not exposed |

The voter must not inherit scanner edge cleanup implicitly. A comparison that
changes voter thinning must record that setting independently of scanner
thinning.

## Verification boundary

The Python suite verifies:

- constructor defaults, setters, validation, and state isolation;
- lag ranges, Java-style rounding, coordinate order, and UVW costs;
- exact greedy seed ordering and suppression;
- unmasked DP paths, surfaces, smoothing, strain behavior, and tie handling;
- mask-aware DP validity, bidirectional strain feasibility, deterministic
  recovery, and projection accounting;
- reference target-point boundaries, vote reinforcement, and strict
  strongest-orientation updates;
- reference and masked policy routing, support controls, all-face masked writes,
  crop offsets, axis permutations, and immutable diagnostics;
- normalization, surface orientation, and all voter-thinning modes;
- Python and Numba agreement for accelerated kernels;
- Synthetic and F3 configuration, evidence, and bundle contracts that consume
  voter outputs.

The suite does not require bit-exact Java or Mines JTK arrays. F3 tests measure
public-reference agreement and bundle consistency; they do not establish
geological truth or Java method identity.

## Authoritative implementation surfaces

- `src/pyosv/voting3d.py`
- `src/pyosv/_voting3d/`
- `src/pyosv/dp.py`
- `src/pyosv/_dp/`
- `src/pyosv/thinning3d.py`
- `src/pyosv/_seed_selection.py`
- `src/pyosv/geometry.py`
- `src/pyosv/cells.py`

## Related specifications

- [3D Voting Conventions](3d_voting.md)
- [Reference-Like 3D Thinning](reference_like_thinning.md)
- [3D Scanner Reference Mapping](reference_mapping_orient3d.md)
- [3D Reference Alignment Audit](reference_alignment_3d.md)
- [Mode Comparison Contract](mode_comparison.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [F3 3D Reference Data Validation](f3d_validation.md)
