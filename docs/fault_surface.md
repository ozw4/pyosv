# Continuous fault-surface fitting

`pyosv.fault_surface` is a PyOSV native numerical extension. It fits one simple
interpretation surface using DL probability and points from a selected FaultSkin
or an equivalent typed point array. It is not a reference-compatible OSV
algorithm. It owns no files, artifact formats, workflow jobs, or Viewer state.

## Coordinate and numerical contract

- Probability volume: floating array `(n_inline, n_xline, n_sample)`.
- Candidate points and constraints: `(N, 3)` in `(sample, xline, inline)` order.
- Coordinates, search distances, and domain bounds: zero-based voxel indices.
- Validity mask: optional Boolean array with the probability volume's shape.
- Model coefficients and exact section samples: float64, retaining hard manual
  constraints to `1e-5` voxel. Render vertices: float32. Triangles: int32.
- Inputs are never mutated. Result arrays are independent read-only snapshots.

Candidate PCA chooses the dependent native coordinate with the largest absolute
normal component. Point-like or collinear candidates obtain a local plane from
connected DL support, as described below. The other two coordinates, in ascending axis order, form the
independent graph coordinates. This permits vertical faults without requiring a
sample-depth graph, and produces editable native inline or crossline sections.
The surface is a tensor-product cubic B-spline graph with open-uniform knots.

One nonfolded, nonbranching patch is supported. Conflicting dependent positions
at identical independent coordinates and candidates that cannot be represented
within the configured corridor are rejected. This check does not segment a
network of intersecting faults: callers must select one patch. All coordinates
must lie inside the same volume, and the fixed independent-coordinate domain
must span two axes and intersect at least one integer native section.

## Public API

```python
from pyosv.fault_surface import (
    FaultSurfaceFitConfig,
    evaluate_fault_surface,
    fit_fault_surface,
    refit_fault_surface,
    sample_fault_surface,
)

result = fit_fault_surface(
    probability,
    selected_skin_points_xyz,
    valid_mask=valid_mask,
    config=FaultSurfaceFitConfig(),
)

# Store the initial model and explicit manual controls separately. Generated
# sticks are suggestions; sampling them does not make them hard constraints.
edited = refit_fault_surface(result.model, manual_points_xyz)

# Reconstruct mesh/sticks from a persisted typed model, without a DL volume.
rendered = sample_fault_surface(edited.model)
points_xyz = evaluate_fault_surface(edited.model, independent_coordinates)
```

`fit_fault_surface` returns `FaultSurfaceFitResult` with `model`, `vertices`,
`triangles`, `sticks`, `probability`, and `supported`. Probability and support
are arrays aligned with render vertices. Both are `None` after probability-free
sampling or refitting; callers must not describe old support as fresh evidence
for moved geometry.

`FaultSurfaceModel` contains:

| Field | Meaning |
| --- | --- |
| `dependent_axis` | Point column 0 = sample, 1 = xline, 2 = inline |
| `independent_axes` | Remaining two point columns, ascending |
| `bounds` | Two `[minimum, maximum]` rows, one per independent axis |
| `coefficients` | Cubic tensor coefficients, one dimension per independent axis |
| `volume_shape` | `(n_inline, n_xline, n_sample)` |
| `mesh_shape` | Number of samples along each independent coordinate |
| `smoothness` | Weight of coefficient curvature penalty |
| `stick_count` | Requested parallel native sections |
| `stick_point_count` | Samples per suggested section line |

Degree 3 and the open-uniform knot construction are part of this API contract,
not inferred from an external artifact. Coefficient dimensions are 4–12 each;
mesh dimensions are 2–257 each. Stick count is 2 through the second coefficient
dimension; point count is 2 through the first. Actual stick count can be lower
when fewer distinct integer sections intersect the domain. A stick's `axis`,
integer `index`, and `points` describe an exact native-plane intersection.

The model supports translations between a cropped and full volume: add the
independent-axis origin to both bounds in each row, add the dependent-axis
origin to every coefficient, and replace `volume_shape`. Other fields are
unchanged. The coordinate origin is external workflow metadata.

## Initial evidence fit

Defaults are a 6 × 6 coefficient grid, a 33 × 33 render grid, 4 section sticks,
and 5 suggested points per stick. The first candidate approximation defines a
rectangular independent-coordinate domain and a dependent-coordinate search
corridor. Default corridor radius is 6 voxels, sampled at intervals no larger
than 0.5 voxel. Search samples outside the volume or touching invalid-mask data
have no evidence. Only sampled probability values are read and validated, so a
memory-mapped survey need not be copied or scanned in full.

Fitting and refitting use a deterministic numerical lattice with
`max(33, 4 * coefficient_count + 1)` samples along each independent coordinate.
`mesh_shape` controls only derived render sampling; changing it does not change
fitted coefficients or the exact manual constraint solution.

At each numerical-grid position, the nearest supported DL band inside the
corridor supplies its highest-probability target. Equal values prefer the smallest absolute displacement,
then the negative displacement. Targets below the default support threshold
0.2 receive no evidence weight. Remaining targets are weighted by squared DL
probability. A coefficient curvature penalty (default weight 0.05) suppresses
small fluctuations. A weak candidate prior anchors unsupported regions.
No valid supported target is an explicit error.

Sparse candidates are usable seeds: one point, duplicate points, or a line do
not need to supply their own plane orientation. The fitter samples integer DL
voxels in a sphere around the seed nearest the candidate median, with radius
`min(search_radius, 16)` voxels. It chooses the nearest 26-connected component
above `support_threshold`; equally near disconnected components are ambiguous.
The probability-squared weighted covariance must have a second eigenvalue of
at least 0.25 voxel squared and at least four times the smallest eigenvalue.
This requires two-dimensional local evidence; a line, isotropic blob, weak
prediction, or absent coverage cannot manufacture a plane.

The resulting plane retains the original seed extent and expands the missing
domain dimension using that local component. Its rectangular domain is bounded
by the union of seed and local evidence extents. Native graph axes are tried in
descending absolute normal-component order, accepting the first rectangle that
agrees with the seeds within `max(search_radius, 1)` dependent-axis voxels and
stays inside the volume. No supported rectangle is an explicit error. Recovery
does not expand the neighborhood until it reaches another fault, and it does
not segment intersecting or folded sheets. The same evidence fitting and
manual editing follow this initialization.

The candidate's vertex density, connectivity, and holes are not retained.
Every parameter-grid cell has two triangles, including unsupported regions.
`probability` is the original DL value sampled at the resulting vertex;
`supported` requires both valid coverage and the configured threshold.
Unsupported values are reported as zero with false support. These values are
not calibrated confidence estimates or combined OSV probabilities.

For candidates that already span a plane, the rectangle's outer extent is the candidate's extent, not an inferred
geological termination. Short missing-support regions within this rectangle
remain continuous and identifiable through `supported`; broad missing regions
require interpretation by the caller. A nearby fault outside the search
corridor cannot attract the initial evidence search. Supported bands are
separated by at least one voxel of below-threshold samples or by an unknown
coverage gap. At each numerical-grid position, the band nearest the candidate
guide is selected, and the highest probability within that band supplies the
target. A stronger neighboring band therefore does not replace a nearer fault.
Equally near bands remain an ambiguity error. Unknown coverage does not prove
that two faults exist, but cannot authorize a jump to distant stronger evidence.

## Manual constraints and derived geometry

`refit_fault_surface(model, constraints)` minimizes departure from the supplied
model with the same curvature penalty, subject to exact 3D point equalities.
Independent coordinates determine where each constraint applies; the dependent
coordinate determines its requested displacement. Constraint points must remain
inside the fixed parameter domain and volume. Displacement is not restricted
by the initial evidence search radius. Up to 4096 constraints are accepted;
conflicting or unrepresentable constraints fail rather than becoming soft
observations. The practical independent constraint count is limited by the
coefficient grid.

An empty constraint set returns the supplied model unchanged. Use a fixed
initial model plus the accumulated accepted manual constraints for repeatable
editing and releasing constraints. Passing successive edited models instead
changes the prior. Generated suggestions remain soft until explicitly edited
or pinned by the caller.

Mesh connectivity is deterministic for a fixed `mesh_shape`, and is derived
from the graph rather than the source FaultSkin. A graph cannot fold across its
independent coordinates. Surface bounds are checked on the render grid, all suggested stick points, and
a grid with at least 65 samples per independent axis; sampled exits reject the
operation instead of clipping or shrinking the requested movement. This is a
numerical sampling check, not an analytic certificate between samples.

## Validation

Run `python -m pytest tests/test_fault_surface.py`. Synthetic cases cover a
plane, a curved ridge, a mask gap without an internal mesh boundary, a stronger
neighbor inside or outside the corridor, equally near ambiguous bands, point
and line recovery on each native axis, weak and isotropic evidence rejection,
boundary recovery, exact large manual moves, full sparse-stick
constraints, deterministic reconstruction, invalid candidates, incompatible
controls, source immutability, and explicit volume-bound rejection. These tests
measure scientific behavior; they do not establish F3 interpretation accuracy.
