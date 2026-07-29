# Skinning

`pyosv.skin` and `pyosv.skinner` provide the Python skinning layer for 3D
voting outputs. `FaultSkinner` defaults to `method="reference"`, the
reference-like primary path for skinning. Use `method="quality"` when a
quality-first extraction should keep the reference-like grower but use adaptive
thresholding and looser seed gating. Use `method="connected_component"` or
`ConnectedComponentSkinner` only when an explicit fallback or diagnostic
connected-component grouping is needed.

## Scope

`FaultSkin` is a small container for grouped `FaultCell` objects. It preserves
cell order, supports iteration and `len()`, and exposes helper arrays:

- `indices()` returns an `(n, 3)` `int32` array in `(i1, i2, i3)` order.
- `likelihoods()` returns an `(n,)` `float32` array of fault likelihood values.

`FaultSkinner(method="reference")` is the default and should be used for normal
fault-interpretation workflows. The module-level
`pyosv.skinner.find_skins(fv, vp, vt, min_likelihood=None)` function is a
convenience wrapper around the same reference-like backend. It does not use
connected-component grouping unless that fallback is requested explicitly. Use
`pyosv.skinner.find_connected_component_skins(...)` only when the legacy
connected-component fallback or a diagnostic comparison is intended.

`ConnectedComponentSkinner.cells_from_votes` extracts one `FaultCell` for
positive `fv` samples where `fv >= min_likelihood`. Zero-valued background
samples are excluded even when `min_likelihood=0.0`. `min_likelihood` must be
finite and nonnegative. Input arrays are global 3D volumes with shape
`(n3, n2, n1)`, and `vp` and `vt` must match `fv`.

`ConnectedComponentSkinner.find_skins` groups extracted cells by voxel
connected components. Connectivity is configured as:

- `face`: 6-connected adjacency.
- `edge`: 18-connected adjacency.
- `corner`: 26-connected adjacency.

Small components can be filtered with `min_skin_size`. Returned skins are
ordered by descending size, then by the first cell index in each component.

`FaultSkinner.find_seeds(d, fm, ep, ft, pt, tt)` selects starting cells from
thinned 3D volumes. All arrays use shape `(n3, n2, n1)` and must have matching
finite values. Candidates must satisfy `ep > 0.8` and `ft > fm`; they are
processed by descending `ft`, with deterministic index ordering for ties. A
candidate is skipped when an already accepted seed falls inside its `d`-sample
axis-aligned exclusion box. Returned seeds are public `FaultCell` objects.

This corresponds to the reference `FaultSkinner.findSeeds(...)` stage. The
Python implementation keeps the same high-level seed-selection role but uses
repository shape conventions and explicit validation.

`FaultSkinner.find_skins` with `method="reference"` runs `find_seeds`, processes
seeds by descending likelihood, skips seeds already occupied by an accepted
skin, grows each candidate with `find_skin`-style local geometry, filters by
`min_skin_size`, and marks accepted skin cells as occupied before continuing.
When `ep`, `ft`, `pt`, and `tt` are not supplied, the convenience argument
mapping is `ep=fv`, `ft=fv`, `pt=vp`, and `tt=vt`.

This is the multi-skin driver corresponding to the reference sequence of
`findSeeds(...)`, repeated `findSkin(...)` growth, accepted-cell occupancy, and
final `reskin(...)` smoothing.

`FaultSkinner.find_skins` with `method="quality"` uses the same multi-skin
driver and reskinning stage as `method="reference"`, but lowers the seed
planarity gate to `ep > 0.5`. If neither a constructor nor call-site
`min_likelihood` is configured, the quality skinner method chooses an adaptive
seed threshold from the positive `ft` values, clipped to a practical
synthetic-report range,
while keeping the grow threshold separately bounded at the quality grow
default. This prevents adaptive seed selection from over-raising the grow gate.
Passing `min_likelihood` explicitly keeps that fixed threshold for both seed
selection and growth. Explicit `ep`, `ft`, `pt`, and `tt` arrays are honored;
otherwise the same convenience mapping as the reference skinner method is used.

After growth, the reference backend reskins each accepted skin by projecting
cells to a seed-local `(v, w)` surface, smoothing local `u` offsets with
likelihood weights, recomputing strike/dip from the smoothed surface
derivatives, and rebuilding local above/below and left/right links. This is a
practical approximation of the Java reference weighted smoothing phase, not the
original conjugate-gradient smoother. Returned `FaultCell` objects expose these
links as `ca`/`cb` for above/below and `cl`/`cr` for left/right neighbors. Pass
`reskin=False` to `find_skins` or `find_skin` to keep the grow-only result.
The keyword-only `reskin_policy` selects the versioned implementation used
when `reskin=True`. Its default value remains `"existing_cells_v1"`, which
smooths and relinks only cells already accepted during growth. Explicit
`"reference_dense_v1"` uses the original grow seed coordinates to smooth the
cropped local surface and regenerate supported missing `(v, w)` cells for the
`reference` and `quality` methods. Dense cells use the original growth volume
sample at their final Java-rounded world position for `fl`; the smoothed
support value is only the dense-growth acceptance signal, so final `fl` may be
below the original grow threshold. The policy is validated even when
reskinning is disabled or the connected-component backend is selected; the
connected-component backend does not apply either reference reskin policy.

Each `FaultCell` has stable, in-memory provenance in `generation`:
`"grown"` for grow-only cells, `"existing_cells_reskinned"` for multi-cell
`existing_cells_v1` output, `"dense_reskin_observed"` and
`"dense_reskin_generated"` for observed and newly filled dense local keys, and
`"connected_component"` for fallback cells. Empty and single-cell reskin fast
paths retain the original generation because no cell was rebuilt. Dense output
also stores its finite `[0, 1]` smoothing support in `reskin_support`; other
generations use `None`. `reskin_support` is the dense acceptance signal and is
independent of the final growth-volume sample in `fl`. These fields are
preserved by in-memory stage-cache snapshots and clones; the canonical
`skins.json` v1 writer remains unchanged and does not serialize them.

`find_skins` and `find_skin` accept a separate mutable
`reskin_diagnostics` mapping. It is cleared at the start and reports the
selected policy, whether reskinning was applied, processed/input/output cell
counts, observed/generated/dropped counts, projected local duplicates,
candidate local keys, rejection counts, and the maximum generated-key
Chebyshev distance from an observed local key. Multi-skin calls sum counts and
take the maximum distance. Dense-v1 candidate rejection uses the fixed order
support or invalid surface, local-u continuity, volume bounds, `valid_mask`,
prior-skin occupancy, then rounded world-index duplicate removal. Local-u
rejections contribute to the candidate count but have no dedicated aggregate
field. A local key is counted in at most one reported rejection category.
This sink is intentionally isolated from the legacy `diagnostics` mapping;
the legacy key set and meanings do not include reskin details.

Both methods also accept a keyword-only `valid_mask`. When supplied, it must be
a three-dimensional boolean NumPy array with the same shape as `fv`. The
`existing_cells_v1` retains this mask without applying it, so supplying a mask
does not change that policy's numerical results. `reference_dense_v1` treats
false mask voxels as hard barriers while rebuilding the dense local surface.

`FaultSkinner.find_skin(seed, fv, vp, vt, ...)` grows one reference-like
`FaultSkin` from a seed and applies the same reskin phase by default. The
grower builds a seed-local `(u, v, w)` coordinate frame where `u` follows the
fault normal, `v` follows the dip vector, and `w` follows the strike vector. It
samples candidate slices with Java-style nearest rounding, uses a priority queue
ordered by likelihood, explores above/below and left/right local directions,
and applies deterministic geometry gates such as minimum likelihood, local `u`
continuity, interior bounds, and accepted-cell collision avoidance.

This corresponds to one reference `findSkin(...)` call. The Python grower is
reference-like, but not a bit-exact Java port.

## Validation Metrics

Default tests should stay small and synthetic. Heavy F3 or full-reference
validation belongs in optional reports, not default CI. Useful skinning
validation summaries are:

- skin count
- largest skin size
- difference from the connected-component fallback
- orientation jitter within accepted skins
- ridge overlap and buffered ridge overlap between expected ridges and skin
  occupancy or thinned likelihood ridges

These metrics are practical-regression signals. They should be interpreted with
the repository equivalence policy, not as exact Java equality checks.

## Minimal Usage

```python
from pyosv.skinner import FaultSkinner, find_connected_component_skins
from pyosv.voting3d import OptimalSurfaceVoter

voter = OptimalSurfaceVoter(ru=1, rv=2, rw=2)
fv, vp, vt = voter.apply_voting(d=3, fm=0.5, ft=ft, pt=pt, tt=tt)
fvt = voter.thin(fv, vp, vt)

skinner = FaultSkinner(min_likelihood=0.7, min_skin_size=20)
skins = skinner.find_skins(fvt, vp, vt, ep=fvt, ft=fvt, pt=vp, tt=vt)

quality_skins = FaultSkinner(method="quality", min_skin_size=20).find_skins(
    fvt, vp, vt, ep=fvt, ft=fvt, pt=vp, tt=vt
)

fallback_skins = find_connected_component_skins(fvt, vp, vt, min_likelihood=0.7)
```

The self-contained example can be run without external data:

```bash
python examples/run_3d_synthetic_skinning.py
```

The example defaults to the reference-like backend. Select the diagnostic
fallback explicitly when comparing behavior:

```bash
python examples/run_3d_synthetic_skinning.py --method connected_component
```

Pass `--output-dir` only when generated DAT outputs and a small text skin
summary should be written.

## Limitations

This implementation does not reproduce the full Java skinning workflow,
including throw/slip estimation, skin clean-up helpers, full Java pruning
behavior, or real-data workflow helpers. The reference backend is a practical
approximation with simplified candidate picking, geometry gates, and weighted
reskin smoothing instead of the original conjugate-gradient smoother.
`ConnectedComponentSkinner` remains available for explicit fallback use.
