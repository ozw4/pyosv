# Reference-Like 3D Thinning

`pyosv` has two 3D thinning modes for scanner and voter outputs:

- `reference`: reference-like behavior. It smooths the comparison volume, bins
  samples by strike angle, and keeps local maxima in the `i2-i3` plane. Kept
  likelihood samples write the smoothed comparison values, matching the current
  Python reference-like helper. Scanner reference thinning also applies
  scanner-style edge-effect removal by default.
- `normal`: existing pyosv behavior. It uses 3D normal-vector interpolation for
  non-maximum suppression.

`FaultOrientScanner3.thin(...)` defaults to `mode="reference"` with edge-effect
removal enabled. The legacy fault-normal scanner thinning path remains
available with `mode="normal"`.
`OptimalSurfaceVoter.thin(...)` still defaults to `mode="normal"`; pass
`mode="reference"` explicitly for reference-like voter thinning reports or
diagnostics.

## Reference Audit Notes

The Java scanner and voter thinning methods share the same broad NMS pattern,
but they are not identical call sites:

- `FaultOrientScanner3.thin(...)` smooths the likelihood volume along `i3` and
  `i2`, keeps strict strike-binned maxima in the `i2-i3` plane, then calls
  `removeEdgeEffects(...)`.
- `FaultOrientScanner3.removeEdgeEffects(...)` is a scanner post-process. It
  zeros retained samples near `i3` faces when the fault normal is nearly
  parallel to those faces, and near `i2` faces when the normal is nearly
  parallel to those faces. It does not remove `i1` face samples.
- `OptimalSurfaceVoter.thin(...)` uses strike-binned NMS but does not call
  `removeEdgeEffects(...)`. Its voter-specific retained-sample reinforcement is
  separate from scanner thinning.

Keep scanner and voter reference-like thinning as separate call sites over the
shared helper. Scanner reference thinning applies edge-effect removal and keeps
voter-specific vertical reinforcement disabled. Voter reference thinning may
use voter-specific retained-sample reinforcement and does not apply scanner
edge cleanup.

Java writes `NO_STRIKE` / `NO_DIP` (`-0.00001`) for rejected orientations before
scanner edge-effect cleanup may zero some retained/rejected samples. Python's
public scanner thinning API already uses zero as the non-retained orientation
sentinel, matching the rest of `pyosv`; keep that sentinel for API compatibility
instead of introducing Java sentinel values into returned arrays.

For surface voting, the Java reference accepts target points with
`0 <= i1 < n1`, `0 < i2 < n2 - 1`, and `0 < i3 < n3 - 1`. Current Python
surface-vote averaging and accumulation helpers accept `i2`/`i3` face samples.
Future reference-mode voting work should narrow those point sets together so
averaging and accumulation use the same Java boundary predicate.

Use scanner reference-like thinning:

```python
fet, fpt, ftt = scanner.thin(
    ft,
    pt,
    tt,
    reference_sigma=1.0,
)
```

For diagnostics, disable scanner edge cleanup explicitly:

```python
fet, fpt, ftt = scanner.thin(
    ft,
    pt,
    tt,
    reference_sigma=1.0,
    remove_edge_effects=False,
)
```

Use the same mode on voter thinning:

```python
fvt = voter.thin(
    fv,
    vp,
    vt,
    mode="reference",
    reference_sigma=1.0,
)
```

For F3 crop, multi-crop, and ablation commands, see
`docs/f3d_validation.md#reference-like-thinning-validation`.

## Interpreting F3 Results

Do not treat the first reference-like thinning runs as proof that pyosv is
equivalent to the Java reference. The first expected improvements are not
necessarily high voxel-wise correlation. Check whether:

- `fvt` `nonzero_fraction` moves closer to the reference.
- `buffered_ridge_overlap.interior.fvt.buffered_f1` improves.
- sparse ridge distance medians decrease.
- ridge overlay figures show fewer far-away candidate-only ridges.
- exact overlap remains plausible even if it is low for sparse ridges.

The previous normal/normal baseline is useful context:

```text
normalized_correlation.interior.fvt.mean ~= 0.224
buffered_ridge_overlap.interior.fvt.buffered_f1.mean ~= 0.075
exact fvt ridge overlap F1/Jaccard = 0.0
```

Do not claim success until an actual ablation report has been generated and
reviewed.
