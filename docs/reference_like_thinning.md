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
`OptimalSurfaceVoter.thin(...)` also defaults to `mode="reference"`. Its legacy
fault-normal voter thinning path remains available with `mode="normal"`.

The reference-like scanner backend and the `reference` thinning mode are
different concepts. The backend produces scanner likelihood and orientations;
scanner reference thinning and voter reference thinning are two later,
stage-specific operations. A downstream `reference` or `quality` workflow is a
separate choice again. See
[Scanner, Workflow, Thinning, and F3 Reference Comparison](mode_comparison.md)
for the canonical terminology.

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
surface-vote averaging and accumulation helpers use the same reference-like
boundary predicate, so edge-only `i2`/`i3` face samples do not contribute votes.
Compared with older `pyosv` runs that accepted all in-bounds face samples, crop
boundary votes near `i2` and `i3` faces may therefore be weaker.

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

The optional legacy/internal F3 crop, multi-crop, and thinning-ablation scripts
expose the same diagnostic switch as `--keep-scanner-edge-effects`. Leave it
unset for reference-first diagnostics; the reports record whether edge-effect
removal was active. These crop reports are not publication evidence.

Pass `mode="normal"` explicitly for the older scanner fault-normal thinning
path:

```python
fet, fpt, ftt = scanner.thin(ft, pt, tt, mode="normal")
```

Use the same mode on voter thinning:

```python
fvt = voter.thin(
    fv,
    vp,
    vt,
    reference_sigma=1.0,
)
```

Pass `mode="normal"` explicitly for the older fault-normal voter thinning path.

```python
fvt = voter.thin(fv, vp, vt, mode="normal")
```

For the full-volume publication protocol, current baseline runner, and planned
mode-comparison matrix, see [F3 3D Reference Data
Validation](f3d_validation.md). Its
[reference-like thinning validation](f3d_validation.md#reference-like-thinning-validation)
section preserves crop, multi-crop, and ablation commands as optional
legacy/internal diagnostics.

## Interpreting F3 Results

Publication-facing F3 comparison uses the complete volume as one evaluation
unit. Record `scanner_backend`, `scanner_thin_mode`, `workflow_mode`, and
`voter_thin_mode` separately so changes can be attributed stage by stage. Do
not treat crops as publication samples or statistical replicates, or crop-based
reference-like thinning results as publication evidence.

The following checks and baseline values are retained for historical/local crop
diagnostics only. They are not current publication acceptance criteria and do
not prove that pyosv is equivalent to the Java reference. Within such a
diagnostic, inspect whether:

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

Use an actual ablation report to interpret a local diagnostic; do not promote
its crop result to a full-volume publication conclusion.
