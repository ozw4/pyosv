# 2D Voter Reference Mapping

This document maps the current 2D optimal-path voting implementation to the
corresponding responsibilities in
`reference_osv/src/osv/OptimalPathVoter.java` and its dynamic-programming
helpers.

The mapping covers these Python modules:

- `pyosv.cells.FaultCell2` for 2D fault-cell geometry and attributes;
- `pyosv.voting2d.OptimalPathVoter` for seed selection, local sampling, path
  voting, normalization, and thinning;
- `pyosv.dp` for path-cost accumulation, backtracking, smoothing, and optimal
  path extraction;
- `pyosv.interp` and `pyosv.filters` for SciPy-backed interpolation and
  smoothing.

The implementation follows reference control flow and geometry where practical,
but it does not require bit-exact Java or Mines JTK output and does not add a
JVM, Jython, Mines JTK, or Gradle runtime dependency.

## Coordinate, shape, and dtype contract

Global 2D arrays use shape `(n2, n1)` and are indexed as `array[i2, i1]`.
Vector components use `(x1, x2)` order. `FaultCell2` stores coordinates in
`(i1, i2)` order and supplies the local normal and strike vectors consumed by
the voter.

For radii `ru` and `rv`, `samples_in_uv_box(...)` returns a local cost image
with shape `(2 * rv + 1, 2 * ru + 1)`, corresponding to `(nv, nu)`. The first
axis follows the local strike direction `v`; the second axis stores admissible
fault-normal lags `u`. Admissible samples use cost `1 - fx`; excluded lag cells
remain at cost `1.0`.

Sample coordinates use Java-style nearest rounding, `floor(x + 0.5)`, and are
clamped to the global image bounds. The same local cost image is passed to the
DP API as an `(ni, nl)` image, where `ni = nv` and `nl = nu`.

DP paths have shape `(ni,)`, use `np.float32`, and store selected `u` lag
values. Voting and thinning outputs are `np.float32` arrays with the same
`(n2, n1)` shape as their inputs.

## Voter mapping

| Reference responsibility | Python implementation | Current contract |
| --- | --- | --- |
| Voter construction and lag ranges | `OptimalPathVoter.__init__`, `pyosv.dp.shift_range`, `pyosv.dp.update_shift_ranges` | Validates nonnegative `ru` and `rv`, initializes the normal-lag range, strain bound, attribute smoothing, path smoothing, and per-`v` admissible lag ranges. |
| Strain configuration | `OptimalPathVoter.set_strain_max`, `pyosv.dp.strain_to_bstrain` | Accepts a maximum path strain and stores the corresponding inverse strain bound. |
| Attribute and path smoothing configuration | `set_attribute_smoothing`, `set_path_smoothing` | Stores nonnegative smoothing controls used by `find_path_2d`. |
| Seed selection | `pick_seeds`, `get_seeds` | Builds `FaultCell2` objects from matching `(n2, n1)` likelihood and angle arrays. `pick_seeds` applies the likelihood threshold and deterministic distance suppression; `get_seeds` returns one bounds-checked sample. |
| Local `uv` cost sampling | `samples_in_uv_box` | Samples `1 - fx` in a seed-centered `(v, u)` box using the cell normal and strike vectors, admissible lag ranges, Java-style nearest rounding, and bounds clamping. |
| One-seed path voting | `_path_voting`, `_accumulate_path_votes` | Extracts one optimal lag path, computes its mean valid fault likelihood over interior image samples, accumulates vote evidence, and retains the strongest local vector orientation. |
| Full voting | `apply_voting` | Selects seeds, accumulates every seed vote, normalizes the vote image, and returns `(fv, w1, w2)`. |
| Final vote normalization | `_normalize_and_power_2d` | Applies configured Gaussian smoothing, subtracts the global minimum, divides by the positive global maximum, applies `1 - (1 - x) ** 4`, clips to `[0, 1]`, and returns `float32`. |
| Vote thinning | `thin` | Samples the vote image at plus and minus the local vector through `pyosv.interp.sample2` and retains strict local maxima. |

## Dynamic-programming path mapping

| Reference method | Python implementation | Current contract |
| --- | --- | --- |
| `findPath` | `pyosv.dp.find_path_2d` | Validates an `(ni, nl)` cost image, applies the configured attribute smoothing, accumulates costs, backtracks the optimal path, applies optional path smoothing, and returns `(ni,)` `float32` lag values. |
| `accumulateForward` | `pyosv.dp.accumulate_forward_2d` or `pyosv.dp.accumulate_2d(direction=1)` | Produces the forward accumulated cost image while enforcing the lag-change spacing constraint. |
| `backtrackReverse` | `pyosv.dp.backtrack_reverse_2d` | Backtracks through a forward accumulation and returns physical lag values using the supplied `lmin`. |

`pyosv.dp` dispatches to the optional Numba kernels when they are available.
The public shapes, dtypes, validation, and numerical meaning are the same for
the Python and accelerated paths.

## Approximation boundaries

Mines JTK interpolation is represented by SciPy-backed helpers such as
`pyosv.interp.sample2`. Recursive reference filters are represented by
`pyosv.filters` Gaussian or separable smoothing. These substitutions can differ
from Mines JTK in interpolation kernels, edge behavior, recursive-filter
response, and floating-point accumulation order.

Reference agreement is evaluated with practical numerical and localization
metrics rather than per-sample bit equality. The external `reference_osv/`
directory remains read-only and is not part of the Python package or its
distribution artifacts.

Related specifications:

- [2D Orientation Scanning](orient2d.md)
- [Reference-First Equivalence Policy](equivalence_policy.md)
