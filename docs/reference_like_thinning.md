# Reference-Like 3D Thinning

`pyosv` has two 3D thinning modes for scanner and voter outputs:

- `normal`: existing pyosv behavior. It uses 3D normal-vector interpolation for
  non-maximum suppression and remains the default for existing workflows and
  tests.
- `reference`: opt-in reference-like behavior. It smooths the comparison volume,
  bins samples by strike angle, and keeps local maxima in the `i2-i3` plane.
  Kept output samples write the smoothed comparison values, matching the Java
  thinning pattern.

The reference-like mode is diagnostic and opt-in because it is closer to the
reference Java thinning workflow but is still a Python implementation, not a
bit-exact Mines JTK reproduction. Existing pyosv behavior remains unchanged
unless `mode="reference"` is selected explicitly.

Use the mode on scanner thinning:

```python
fet, fpt, ftt = scanner.thin(
    ft,
    pt,
    tt,
    mode="reference",
    reference_sigma=1.0,
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

The previous current/current baseline is useful context:

```text
normalized_correlation.interior.fvt.mean ~= 0.224
buffered_ridge_overlap.interior.fvt.buffered_f1.mean ~= 0.075
exact fvt ridge overlap F1/Jaccard = 0.0
```

Do not claim success until an actual ablation report has been generated and
reviewed.
