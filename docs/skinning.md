# Skinning

`pyosv.skin` and `pyosv.skinner` provide 3D fault-cell grouping, geometry-aware
skin growth, reskinning, and connected-component grouping.

The geometry-aware implementation follows the processing structure of the Java
reference where practical, but it is not a bit-exact Mines JTK port. PyOSV uses
NumPy, SciPy, and optional Numba kernels and has no JVM, Jython, Gradle, or
Mines JTK runtime dependency.

## Array, cell, and skin contracts

Global volumes use shape `(n3, n2, n1)` and array indexing
`array[i3, i2, i1]`. Strike and dip are expressed in degrees. Public skinning
entry points require finite numeric three-dimensional arrays with matching
shapes and convert them to `float32` where required by the numerical kernels.

A `FaultCell` stores continuous world coordinates `(x1, x2, x3)`, likelihood
`fl`, strike `fp`, and dip `ft`. Integer indices use Java-style nearest rounding:

```text
i = floor(x + 0.5)
```

Cells can also carry four in-memory links:

| Field | Meaning |
| --- | --- |
| `ca` | adjacent cell above in the local dip direction |
| `cb` | adjacent cell below in the local dip direction |
| `cl` | adjacent cell to the left in the local strike direction |
| `cr` | adjacent cell to the right in the local strike direction |

`FaultSkin` is an ordered mutable container of `FaultCell` objects. It supports
iteration, `len()`, `append()`, and `add()`.

```python
from pyosv.skin import FaultSkin

skin = FaultSkin.from_cells(cells)
indices = skin.indices()       # (n, 3), int32, order (i1, i2, i3)
likelihoods = skin.likelihoods()  # (n,), float32
```

An empty skin returns an `(0, 3)` index array and an empty likelihood array.

## Public entry points

```python
from pyosv.skinner import (
    ConnectedComponentSkinner,
    FaultSkinner,
    find_connected_component_skins,
    find_skins,
)
```

`FaultSkinner` accepts three methods:

| Method | Contract |
| --- | --- |
| `reference` | Uses the reference-oriented seed gate, local geometry-aware growth, and the selected reskin policy. This is the constructor default. |
| `quality` | Uses the same grower with a lower planarity seed gate and adaptive seed likelihood when no threshold is configured. |
| `connected_component` | Thresholds positive vote samples and groups them by voxel connectivity. Geometry-aware growth and reskinning are not applied. |

Constructor defaults are:

```text
method = reference
min_likelihood = omitted
min_skin_size = None
connectivity = corner
```

When `min_likelihood` is omitted, its stored value is `0.0`. Assigning the
`min_likelihood` property marks the threshold as explicitly configured.
`min_likelihood` must be finite and nonnegative. `min_skin_size` is either
`None` or a nonnegative integer.

The `connectivity` setting is used by connected-component grouping. It is still
validated and stored when another method is selected.

The module-level `find_skins(...)` function constructs `FaultSkinner()` and
therefore uses the `reference` method. Select `quality` or
`connected_component` through an explicit `FaultSkinner` instance.

The following facade methods have fixed responsibilities independent of the
configured method:

- `cells_from_votes(...)` always delegates to connected-component cell
  extraction.
- `find_seeds(...)` always applies the reference seed gate `ep > 0.8`.
- `find_skin(...)` always executes one geometry-aware grow operation from the
  supplied seed. It does not perform method-specific seed selection or adaptive
  threshold resolution.

## Multi-skin input mapping

The main call is:

```python
skins = skinner.find_skins(
    fv,
    vp,
    vt,
    min_likelihood=None,
    ep=ep,
    ft=ft,
    pt=pt,
    tt=tt,
)
```

`fv`, `vp`, and `vt` are the growth likelihood, strike, and dip volumes.
`ep`, `ft`, `pt`, and `tt` are the seed-selection attributes. When they are
omitted, the mapping is:

```text
ep = fv
ft = fv
pt = vp
tt = vt
```

All seven effective arrays must be finite and have the same `(n3, n2, n1)`
shape.

The geometry-aware defaults are:

| Setting | Default | Contract |
| --- | ---: | --- |
| `d` | `1` | Chebyshev seed-spacing radius. |
| `ru` | `150` | Local fault-normal radius; must be at least `2`. |
| `rv` | `None` | Resolves to `max(n2, n3)`; explicit values must be at least `2`. |
| `rw` | `None` | Resolves to `max(n2, n3)`; explicit values must be at least `2`. |
| `max_steps` | `10` | Maximum local rows explored in one directional expansion. |
| `du` | `5.0` | Maximum accepted local-u and world-x1 change between linked candidates. |
| `max_delta_strike` | `30.0` | Maximum circular strike difference between linked candidates. |
| `reskin` | `True` | Applies the selected reskin policy after growth. |
| `reskin_policy` | `existing_cells_v1` | Selects the reskin implementation. |
| `accepted_occupancy_radius` | `None` | Resolves to `5` for accepted-skin occupancy marking. |

Numeric controls must be finite and nonnegative. `reskin` must be boolean.

## Seed selection

The multi-skin geometry-aware driver selects seed candidates with strict tests:

```text
ep > seed_min_ep
ft > seed_threshold
```

The method-specific planarity thresholds are:

| Method | `seed_min_ep` |
| --- | ---: |
| `reference` | `0.8` |
| `quality` | `0.5` |

Candidates are ordered by descending `ft`. Equal-likelihood candidates use
ascending C-order flat index. Exact greedy suppression then accepts a candidate
only when no accepted seed lies in its Chebyshev box of radius `d`.

For `reference`, the seed and grow threshold is the call-site
`min_likelihood`, the configured constructor value, or `0.0` when neither is
provided.

For `quality`, threshold resolution is:

- when no constructor or call-site threshold is configured, the seed threshold
  is the 70th percentile of positive `ft`, clipped to `[0.25, 0.75]`; if no
  positive sample exists, it is `1.0`;
- under that adaptive path, the grow threshold is fixed at `0.5`;
- an explicit constructor or call-site threshold is used for both seed
  selection and growth.

The seed comparison is strict `ft > seed_threshold`. During growth, expansion
stops when the candidate likelihood is below the grow threshold, so a value
equal to the grow threshold is admissible.

Before starting a grow attempt, a seed is rejected when its radius-2 box
intersects occupancy from an accepted skin. Accepted cells mark occupancy boxes
with `accepted_occupancy_radius`, which defaults to `5`. Growth candidates also
reject a radius-2 intersection with that occupancy.

## Geometry-aware growth

`find_skin(...)` and the `reference` and `quality` multi-skin methods use a
seed-local coordinate frame:

```text
u = fault-normal direction
v = fault-dip direction
w = fault-strike direction
```

The seed defines the world origin and the initial normal, dip, and strike basis.
The local grid center is `(ru, rv, rw)`. Accepted local cells are processed by a
priority queue ordered by descending likelihood and stable insertion order.
Each cell explores the four tangential directions:

```text
v - 1
v + 1
w - 1
w + 1
```

For one direction, the grower samples a local likelihood slice over a normal
window from `current_u - 5` through `current_u + 5`, clipped to the transform
bounds. A dynamic-programming path chooses one u sample per explored row with a
maximum index jump of `2` and jump penalty `0.1`. Equal candidate scores prefer
the u sample nearest the center of the available u range, then the lower u
index.

A candidate is accepted only while all applicable conditions hold:

- likelihood is not below the grow threshold;
- its continuous world position satisfies
  `1 < x1 < n1-2`, `1 < x2 < n2-2`, and `1 < x3 < n3-2`;
- its Java-rounded world index has not already been accepted by the current
  skin;
- it does not collide with occupancy from an accepted skin;
- local-u change does not exceed `du`;
- world `x1` change does not exceed `du`;
- circular strike difference does not exceed `max_delta_strike`.

The grower samples candidate strike and dip from `vp` and `vt` with Java-style
nearest rounding. Growth is deterministic for fixed inputs and configuration.
The grow-only cells have `generation="grown"`.

The local grow graph contains above/below and left/right links. Public links are
rebuilt by multi-cell reskin policies. Grow-only and connected-component output
do not require public links to be populated.

### Multi-skin filtering

Each accepted seed is grown independently against the shared occupancy mask.
`min_skin_size` filters the resulting skin before it is added to the output and
before its cells mark occupancy.

When `min_skin_size` is `None`, no size filter is applied. An explicit
`min_skin_size`, including `0`, discards an empty grow result; positive values
also discard nonempty skins below the configured size. Canonical F3 skin
artifacts do not permit empty skins.

## Connected-component grouping

`ConnectedComponentSkinner.cells_from_votes(...)` creates a cell where:

```text
fv > 0
and
fv >= min_likelihood
```

Each cell receives likelihood from `fv`, strike from `vp`, dip from `vt`, and
`generation="connected_component"`.

`find_skins(...)` groups those cells with the selected connectivity:

| Connectivity | Adjacency |
| --- | --- |
| `face` | 6-connected |
| `edge` | 18-connected |
| `corner` | 26-connected |

Each component is internally ordered by `(i1, i2, i3)`. Returned skins are
ordered by descending cell count and then by the first cell index. Components
below `min_skin_size` are omitted.

The convenience function uses the same contract:

```python
fallback_skins = find_connected_component_skins(
    fv,
    vp,
    vt,
    min_likelihood=0.7,
    min_skin_size=20,
    connectivity="corner",
)
```

## Reskin policies

Accepted policy identifiers are:

```text
existing_cells_v1
reference_dense_v1
```

`reskin_policy` is validated even when `reskin=False` or
`method="connected_component"`. The connected-component method does not apply
either policy.

### `existing_cells_v1`

This policy smooths and relinks only cells already accepted during growth.
Empty and single-cell skins are returned without rebuilding cells.

For a multi-cell skin, the policy:

1. chooses the highest-likelihood cell as the local basis origin, with the
   earliest cell winning an equal-likelihood tie;
2. projects cells onto integer local `(v, w)` keys with Java rounding;
3. retains the higher-likelihood cell when multiple cells project to one key;
4. smooths local u values with likelihood weights and smoothing sigma `1.0`;
5. recomputes strike and dip from local surface derivatives;
6. rebuilds world coordinates and above/below and left/right links.

A positive likelihood is used as its smoothing weight. A nonpositive retained
likelihood receives unit weight. The normalized weighted surface divides only
where the smoothed denominator exceeds `1e-6`.

Rebuilt cells retain their source likelihood and receive
`generation="existing_cells_reskinned"`. Projected local duplicates can reduce
the output cell count.

### `reference_dense_v1`

This policy uses the local state retained by the grower and can fill supported
missing `(v, w)` keys inside the observed local bounding box. Empty and
single-cell skins are returned without rebuilding cells.

One observed cell is selected per local key by highest likelihood, with grow
order resolving equal values. The fixed numerical contract is:

```text
surface weight                 = max(fl, 0) ** 2
surface smoothing sigma (w,v)  = (4.0, 4.0)
support smoothing sigma (w,v)  = (8.0, 8.0)
valid denominator              > 1.0e-6
candidate support              > 0.2
candidate local-u change       < 5.0
```

Candidate keys are traversed from the seed through four-neighbor `(v, w)`
adjacency, prioritizing higher support and then lower `(w, v)` indices. A key is
rejected by the first applicable condition in this order:

1. invalid surface denominator or insufficient support;
2. excessive local-u change;
3. continuous or Java-rounded world position outside the volume;
4. false `valid_mask` sample;
5. collision with an accepted prior skin.

Candidates that round to the same world index are resolved after traversal.
Observed keys take priority over generated keys, followed by higher support,
lower `w`, and lower `v`.

Each final cell samples `fl` from the supplied growth likelihood volume at its
Java-rounded world index. The smoothed support controls candidate acceptance but
does not replace that final likelihood. Final orientation is derived from the
smoothed local surface and links are rebuilt on the retained local grid.

Observed and inserted cells receive these generations:

```text
dense_reskin_observed
dense_reskin_generated
```

Both require `reskin_support`, a finite scalar in `[0, 1]` obtained from the
clipped smoothed support field.

## Valid masks

`valid_mask`, when supplied, must be a boolean NumPy array with the same shape
as `fv`.

The mask does not constrain geometry-aware growth. Its numerical effect is
policy-specific:

- `existing_cells_v1` does not use it;
- `reference_dense_v1` rejects a candidate when its final Java-rounded world
  index is false in the mask;
- `connected_component` does not use it.

The mask is still validated before method and reskin dispatch.

## Cell provenance

`FaultCell.generation` accepts exactly:

| Generation | Meaning |
| --- | --- |
| `grown` | Geometry-aware grow output or an unchanged empty/single-cell reskin result. |
| `existing_cells_reskinned` | Cell rebuilt by `existing_cells_v1`. |
| `dense_reskin_observed` | Dense-policy output at a local key observed during growth. |
| `dense_reskin_generated` | Dense-policy output at a local key inserted during reskinning. |
| `connected_component` | Cell created by thresholded component grouping. |

`reskin_support` must be finite and in `[0, 1]` for the two dense generations.
It must be `None` for every other generation.

Generation and support do not participate in `FaultCell` equality. Cell links
are in-memory topology and are not part of the canonical F3 `skins.json` cell
record.

## Diagnostics

### Growth diagnostics

The `diagnostics` mapping applies to multi-skin `reference` and `quality`
execution. It is cleared and populated with:

```text
seed_candidate_count_before_spacing
seed_count_after_spacing
seed_count_rejected_by_occupied
grow_attempt_count
grown_skin_count_before_min_size
discarded_empty_skin_count
discarded_small_skin_count
accepted_skin_count
accepted_cell_count
accepted_occupancy_radius
seed_min_ep
seed_threshold
grow_threshold
```

The connected-component method does not populate this mapping.

### Reskin diagnostics

`reskin_diagnostics` is a separate mutable mapping. It is cleared before
execution and emits:

```text
reskin_diagnostics_contract_version = 2
```

The two diagnostic arguments must be distinct objects. Passing the same mapping
as `diagnostics` and `reskin_diagnostics` raises `ValueError` before either is
changed.

The final namespace describes skins that remain after empty and minimum-size
filtering. It contains:

```text
reskin_policy
reskin_applied
processed_skin_count
input_cell_count
output_cell_count
observed_output_cell_count
generated_cell_count
dropped_input_cell_count
projected_local_duplicate_count
candidate_local_key_count
rejected_support_count
rejected_invalid_mask_count
rejected_prior_skin_collision_count
rejected_out_of_bounds_count
rejected_duplicate_world_index_count
max_generated_chebyshev_distance_from_observed
```

The nested `attempted` mapping contains the same count and state fields before
final filtering. Its `processed_skin_count` counts items that reached the
reskin phase. Multi-skin aggregation sums count fields and takes the maximum of
`max_generated_chebyshev_distance_from_observed`.

A dense candidate rejected by the local-u continuity test contributes to
`candidate_local_key_count` but has no dedicated rejection counter. Each local
key contributes to at most one reported rejection category. When reskinning is
disabled, fields that describe reskin work remain zero. Connected-component
execution reports observed final skin and cell counts with
`reskin_applied=False`.

## Canonical F3 skin artifacts

`canonical_skins_payload(...)` emits `skins.json` with `format_version=2`.
Each cell contains:

```text
x1, x2, x3
i1, i2, i3
fl, fp, ft
generation
reskin_support
```

The writer preserves skin sequence, assigns consecutive `skin_index` values,
sorts cells inside each skin by `(i3, i2, i1)`, and rejects empty skins. Dense
generations require finite support in `[0, 1]`; all other generations require
JSON `null` support.

The parser accepts `format_version` 1 and 2. Format 1 contains the coordinate,
index, likelihood, strike, and dip fields; parsed generation and support are
`None`. Format 2 includes and validates generation and support. Both formats
must use canonical skin indices, cell counts, field sets, bounds, and cell
ordering.

The canonical F3 artifact set cross-checks `skins.json`, `skin_mask.dat`, and
reported topology. Artifact validation does not reconstruct in-memory cell
links.

## Usage

```python
from pyosv.skinner import FaultSkinner, find_connected_component_skins

reference_skinner = FaultSkinner(
    method="reference",
    min_likelihood=0.7,
    min_skin_size=20,
)
reference_skins = reference_skinner.find_skins(
    fvt,
    vp,
    vt,
    ep=fvt,
    ft=fvt,
    pt=vp,
    tt=vt,
)

quality_skinner = FaultSkinner(method="quality", min_skin_size=20)
quality_skins = quality_skinner.find_skins(
    fvt,
    vp,
    vt,
    ep=fvt,
    ft=fvt,
    pt=vp,
    tt=vt,
)

dense_diagnostics: dict[str, object] = {}
dense_skins = reference_skinner.find_skins(
    fvt,
    vp,
    vt,
    ep=fvt,
    ft=fvt,
    pt=vp,
    tt=vt,
    reskin_policy="reference_dense_v1",
    valid_mask=valid_mask,
    reskin_diagnostics=dense_diagnostics,
)

component_skins = find_connected_component_skins(
    fvt,
    vp,
    vt,
    min_likelihood=0.7,
    min_skin_size=20,
    connectivity="corner",
)
```

The self-contained example is:

```bash
python examples/run_3d_synthetic_skinning.py
```

## Verification and equivalence boundary

The test suite verifies:

- array validation, thresholds, seed ordering, and exact greedy spacing;
- local transforms, Java-style sampling, candidate paths, and geometry gates;
- deterministic priority growth, collision handling, and skin-size filtering;
- connected-component connectivity and ordering;
- both reskin policies, generated-cell acceptance, duplicate resolution,
  topology links, provenance, and support validation;
- growth and reskin diagnostics;
- Python and Numba agreement for accelerated candidate operations;
- F3 skin serialization, provenance, mask, topology, and bundle validation.

Tests require deterministic Python behavior and numerical tolerances where SciPy
filtering is involved. They do not require bit-exact Java or Mines JTK arrays.
F3 public outputs are comparison targets rather than geological truth.

The package does not provide throw/slip estimation, the complete set of Java
skin cleanup and pruning utilities, or Java workflow helpers. The
geometry-aware grower and weighted reskin policies define the supported Python
skinning contract.

## Related specifications

- [3D Voting Conventions](3d_voting.md)
- [Reference-Like 3D Thinning](reference_like_thinning.md)
- [Mode Comparison Contract](mode_comparison.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [F3 3D Reference Data Validation](f3d_validation.md)
