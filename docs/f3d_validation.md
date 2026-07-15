# F3 3D Reference Data Validation

This workflow validates the current Python 3D scanner and optimal-surface
voting pipeline against the public F3 reference volumes. The F3 `.dat` files
are external data, not repository files. Do not copy them into git; repository
`.gitignore` rules ignore generated `.dat` files and `outputs/`.

## Data Layout

Use an external data root such as the local shared copy:

```text
/home/dcuser/public_data/field/F3/reference_osv/
  ep.dat
  fl.dat
  fv.dat
  fvt.dat
  xs.dat
```

Point repository commands at that root with:

```bash
export PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv
```

The expected format for each file is:

```text
shape = (420, 400, 100)  # (n3, n2, n1)
dtype = big-endian float32
expected bytes per file = 67,200,000
```

Files:

- `xs.dat`: signed input seismic amplitude image.
- `ep.dat`: planarity attribute used as the scanner input; start OSV validation
  from this file.
- `fl.dat`: public-workflow fault-likelihood attribute.
- `fv.dat`: public-workflow OSV voting result.
- `fvt.dat`: public-workflow thinned OSV result.

Only `xs.dat` is the input seismic image. `ep.dat`, `fl.dat`, `fv.dat`, and
`fvt.dat` are attributes or processing results. In particular, public
`fvt.dat` is useful for comparison but is not independent geological truth.

The current OSV validation starts from `ep.dat`; reproducing `xs.dat -> ep.dat`
is out of scope for this workflow.

## Fast Smoke Validation

Check that the external files are present, have the expected byte size, and can
be read as `(420, 400, 100)` big-endian float32 volumes:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python -m pytest -q tests/test_f3d_reference_data.py -s
```

Generate a summary report for the reference volumes:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/report_3d_f3d_reference.py --pretty
```

On the local shared F3 copy, that report shows `fv.dat` max around `1.0`,
`fvt.dat` max around `0.99`, and `fvt.dat` much sparser than `fv.dat`. Treat
the report output as the source of truth for the exact local values.

## Small Crop Practical-Equivalence Validation

Run one deterministic crop validation and write metrics under `outputs/`:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/run_3d_f3d_crop_validation.py \
  --max-crops 1 \
  --output-dir outputs/3d/f3d/crop_001
```

Run the opt-in pytest wrapper for the crop pipeline:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
PYOSV_RUN_F3D_CROP_PIPELINE=1 \
python -m pytest -q tests/test_f3d_reference_crop_pipeline.py -s
```

By default, the crop example writes `metrics.json` only. Add `--save-volumes`
when crop-level Python `.dat` outputs are needed.

Default crop selection is margin-aware: when a crop shape is used to pick
centers, candidates too close to the volume boundary are skipped instead of
being silently shifted by `crop_slices()`. Pass `--center i3,i2,i1` to validate
a specific manual crop.

Final vote-map normalization defaults to the reference-like path with no final
vote-map smoothing. F3 validation and report scripts record
`final_normalization_smoothing` in their metadata. Pass
`--final-normalization-smoothing 1.0` only when comparing against older pyosv
runs that smoothed the final vote map before the power transform.

Default reference-first comparison:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/run_3d_f3d_crop_validation.py \
  --max-crops 1 \
  --output-dir outputs/3d/f3d/crop_reference_default
```

Older pyosv-style final-normalization comparison:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/run_3d_f3d_crop_validation.py \
  --max-crops 1 \
  --output-dir outputs/3d/f3d/crop_final_norm_smoothing_1 \
  --final-normalization-smoothing 1.0
```

## Reference-Like Thinning Validation

The 3D scanner and voter thinning steps support two modes:

- `normal`: existing pyosv behavior. It uses 3D normal-vector interpolation for
  non-maximum suppression. For scanner and voter thinning this is now the
  legacy opt-in mode.
- `reference`: reference-like behavior. It smooths the comparison volume, bins
  samples by strike angle, and compares local maxima in the `i2-i3` plane.

Scanner and voter thinning default to `reference`; pass
`--scanner-thin-mode normal` or `--voter-thin-mode normal` only when comparing
against older pyosv runs. Both modes are Pythonic approximations, not bit-exact
Mines JTK ports. See `docs/reference_like_thinning.md` for the API-level
details.

Scanner reference thinning removes scanner-style edge effects by default. Pass
`--keep-scanner-edge-effects` only for diagnostics that need to compare the
pre-cleanup retained samples. Reports record this as
`config.scanner.remove_edge_effects` or
`config.scanner.reference_remove_edge_effects` depending on the script.
Reports also record the surface-voting boundary policy as
`reference-like-i2-i3-interior`, meaning vote averaging and accumulation exclude
`i2`/`i3` face-only surface samples. This can suppress votes near crop
boundaries compared with older all-in-bounds boundary handling.

To reproduce older normal/normal thinning behavior in F3 diagnostics, pass both
legacy flags explicitly:

```bash
--scanner-thin-mode normal --voter-thin-mode normal
```

Run one crop with reference-like scanner and voter thinning:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/run_3d_f3d_crop_validation.py \
  --output-dir outputs/3d/f3d/crop_reference_thin_001 \
  --scanner-thin-mode reference \
  --voter-thin-mode reference \
  --reference-thin-sigma 1.0 \
  --pretty \
  --save-figures
```

Run a multi-crop report with reference-like thinning:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/report_3d_f3d_multicrop.py \
  --output-json outputs/3d/f3d/multicrop_reference_thin_001/metrics.json \
  --count 3 \
  --crop-shape 64,64,64 \
  --interior-margin 16 \
  --scanner-thin-mode reference \
  --voter-thin-mode reference \
  --reference-thin-sigma 1.0 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

Run a multi-crop comparison between the reference workflow and the quality
workflow on the same deterministic centers. This is the standard F3 external
smoke command for quality promotion candidates:

```bash
PYTHONPATH=src python examples/report_3d_f3d_multicrop.py \
  --data-root "$PYOSV_F3D_DATA_ROOT" \
  --count 3 \
  --crop-shape 64,64,64 \
  --interior-margin 16 \
  --compare-workflows \
  --save-figures \
  --write-markdown-index \
  --output-json outputs/3d/f3d/quality_external_smoke_001/metrics.json \
  --pretty
```

`--compare-workflows` runs both configured workflows, `reference` and
`quality`, for each selected crop center. F3 crops do not have independent
truth labels. Agreement with the reference workflow therefore does not directly
mean that the quality workflow is better. Use this report to check that the
quality workflow has not removed visible geological signal from the
reference-oriented path, has not introduced excess ridges or boundary
artifacts, and has not become less stable across crops. Support-aware voting is
inactive in both workflow defaults; enable it only with explicit
`--surface-support-*` overrides when running a diagnostic comparison. In compare
mode, metrics and figures are written under workflow-specific directories such
as `volumes/reference/`, `volumes/quality/`, `figures/reference/`, and
`figures/quality/`.

Compare-mode reports also include top-level `quality_validation`. This is a
truthless external smoke summary, not a replacement for the controlled
synthetic promotion gate. The default thresholds are conservative:
`quality_density_not_exploding` fails when quality fvt density exceeds `2.0x`
the reference workflow, `quality_edge_density_not_exploding` fails when the fvt
edge-density proxy delta exceeds `0.10`, and
`quality_sparse_distance_not_worse` fails when sparse distance p95 worsens by
more than `5.0` samples. Finite metric failures always fail the smoke. The
thresholds can be overridden with `--quality-density-max-ratio`,
`--quality-edge-density-max-delta`, and
`--quality-sparse-distance-max-delta`.

F3 data is external. Environments without `PYOSV_F3D_DATA_ROOT` cannot run this
smoke; CI should keep using mock/fixture structure tests instead of requiring
the real F3 volumes.

## Quality-Workflow Scanner-Thinning Policy Validation

The reference-like-backend 49^3 synthetic scanner-thinning gate has passed.
The matching formal F3 64^3-by-3 shared-scan run has also been performed, but
it failed one conservative external-smoke check. Crop 1's candidate
public-FVT sparse-distance p95 was `8.429705` versus baseline `2.236068`, a
`+6.193637`-sample delta above the allowed `+5.0`. The other seven checks
passed. The prerequisite large crop was not run, human geological review is
pending, and no quality-workflow or public scanner-thinning default changes.

Use `examples/report_3d_f3d_scanner_thinning_policy.py` for this comparison.
The existing `report_3d_f3d_multicrop.py --compare-workflows` path changes both
scanner and voter thinning and runs the scanner separately for the reference
and quality workflows. The thinning-ablation report shares the raw scan, but it
does not hold voter thinning at the quality-workflow `hybrid_v2` policy. Neither
therefore isolates scanner thinning in the way required here.

The dedicated comparison profile and policy IDs are:

```text
comparison profile:
  quality-workflow-scanner-thinning-v1

baseline:
  quality_reference_like_scanner_thin_reference_v1

candidate:
  quality_reference_like_scanner_thin_normal_v1
```

Both policies fix the scanner backend to `reference-like`, workflow to
`quality`, voter thinning to `hybrid_v2`, surface support minimum fraction and
exponent to `0.0`, surface-voting boundary policy to `reference`, final
normalization smoothing to the current quality-workflow default, and requested
scanner edge cleanup to `true`. The sole user-configured experiment difference
is scanner thinning, `reference` versus `normal`. The baseline applies the edge
cleanup request and records `effective_remove_edge_effects=true`; normal
thinning does not use that operation and records
`effective_remove_edge_effects=null`.

Each crop must execute `FaultOrientScanner3.scan()` exactly once and share its
raw `ft`, `pt`, and `tt` volumes:

```text
ep crop
  -> shared FaultOrientScanner3.scan()
  -> shared ft / pt / tt
     |-> baseline scanner.thin(mode="reference")
     |     -> independent quality voter
     |     -> voter.thin(mode="hybrid_v2", plateau_tie_breaker=fet_reference)
     |
     `-> candidate scanner.thin(mode="normal")
           -> independent quality voter
           -> voter.thin(mode="hybrid_v2", plateau_tie_breaker=fet_normal)
```

The two downstream voters are separate instances so branch state cannot leak
between policies. Each branch's own scanner-thinned `fet` is mandatory as the
`hybrid_v2` plateau tie-breaker; omitting it is not the current quality workflow.
Every policy branch provides all of these stage outputs:

```text
ft_py.dat   pt_py.dat   tt_py.dat
fet_py.dat  fpt_py.dat  ftt_py.dat
fv_py.dat   vp_py.dat   vt_py.dat
fvt_py.dat
```

### Formal 64^3 Multi-Crop Run

Run three deterministic crops and retain the full local diagnostics:

```bash
PYTHONPATH=src python examples/report_3d_f3d_scanner_thinning_policy.py \
  --data-root "$PYOSV_F3D_DATA_ROOT" \
  --comparison-profile quality-workflow-scanner-thinning-v1 \
  --count 3 \
  --crop-shape 64,64,64 \
  --interior-margin 16 \
  --save-volumes \
  --save-figures \
  --write-markdown-index \
  --fail-on-validation-failure \
  --output-json outputs/3d/f3d/scanner_thinning_policy_64x3/metrics.json \
  --pretty
```

A successful formal run exits with code zero, selects three crops, records
three scanner executions, has finite and nonempty required stages, and reports
`policy_validation.passed=true` with every automatic check passing.

The recorded formal run selected centers `(147, 96, 67)`, `(96, 65, 51)`, and
`(74, 145, 64)` and exited with code 2. Seven of eight checks passed. FVT
density ratios were `1.023707`, `1.025346`, and `1.147162`; the maximum
edge-density increase was `0.005840`, and candidate density CV was `0.039905`.
The sole failure was the crop-1 sparse-distance delta described above. Its
aggregate mean delta was `-0.727719` samples, but the contract deliberately
also gates the worst crop. Preliminary tail analysis reproduced the metric and
found no endian, crop-coordinate, or distance-transform mismatch. It found five
of 57 candidate top-percentile points that promoted formerly weaker public-FVT
ridges and drove the tail. The exceedance points were approximately 19--25
samples inward from the crop faces, but that distance does not rule out a
crop-context effect: the voter uses `ru=10`, `rv=20`, and `rw=30`, and surface
sampling outside a crop is clamped to its edge. Signed seismic amplitude,
adjacent-slice continuity, and recomputation from a larger context are therefore
required before deciding whether the ridges are plausible structure or
crop-dependent disagreement.

### Public-FVT Distance Outlier and Context Diagnostics

The outlier diagnostic reconstructs the existing public-FVT sparse-distance
metric; it does not introduce another validation check. For each base crop it:

1. evaluates the same interior returned by `interior_slices()` using the formal
   run's `interior_margin=16`;
2. constructs public, baseline, and candidate FVT masks with
   `top_percentile_mask(percentile=99.0, positive_only=True)`;
3. computes unit-spacing 3D Euclidean distance from each candidate sparse-ridge
   sample to the nearest public-FVT sparse-ridge sample;
4. sets the allowed candidate distance to baseline candidate-to-public p95 plus
   the unchanged `5.0`-sample allowance; and
5. marks a candidate sparse-ridge sample as an outlier only when its distance is
   strictly greater than that allowed value. Equality is not an outlier.

The JSON records deterministic interior-local, crop-local, global, and nearest
public-FVT coordinates, distances, amplitudes and available stage values. It
also groups the outlier mask into 26-neighbor connected components. These are
candidate-to-public disagreements, not automatically false detections, because
public `fvt.dat` is not truth.

Run the opt-in three-crop amplitude review as follows. The known automatic
failure is expected, so this artifact-producing command deliberately omits
`--fail-on-validation-failure` and exits zero after writing the report:

```bash
PYTHONPATH=src python examples/report_3d_f3d_scanner_thinning_policy.py \
  --data-root "$PYOSV_F3D_DATA_ROOT" \
  --comparison-profile quality-workflow-scanner-thinning-v1 \
  --count 3 \
  --crop-shape 64,64,64 \
  --interior-margin 16 \
  --outlier-diagnostics \
  --save-volumes \
  --save-figures \
  --write-markdown-index \
  --output-json outputs/3d/f3d/scanner_thinning_policy_64x3_outlier_review/metrics.json \
  --pretty
```

`--outlier-diagnostics` reads signed `xs.dat` and adds
`consensus.candidate_minus_baseline.crops[].public_fvt_distance_outliers`.
With `--save-figures`, it writes orthogonal and adjacent-slice amplitude
overlays at the actual component representative coordinates. The metric remains
positive-only 99th percentile for public, baseline, and candidate FVT. Display
masks are deliberately separate: public and baseline remain at the 99th
percentile with `0.8`-point contours, while candidate-policy FVT uses a
positive-only 95th-percentile `2.0`-point yellow contour so a reviewer can trace
more of the candidate surface. The broader yellow line is a display convention,
not the metric ridge, and does not affect points, components, persistence, or
validation. Without this opt-in flag, the existing JSON, Markdown, and output
layout remain unchanged.

To recompute crop 1 using a larger crop while comparing exactly the original
global `64 x 64 x 64` ROI, run:

```bash
PYTHONPATH=src python examples/report_3d_f3d_scanner_thinning_policy.py \
  --data-root "$PYOSV_F3D_DATA_ROOT" \
  --comparison-profile quality-workflow-scanner-thinning-v1 \
  --count 3 \
  --crop-shape 64,64,64 \
  --interior-margin 16 \
  --outlier-diagnostics \
  --context-crop-index 1 \
  --context-crop-shape 128,128,100 \
  --save-figures \
  --write-markdown-index \
  --output-json outputs/3d/f3d/scanner_thinning_policy_64x3_context_review/metrics.json \
  --pretty
```

The context crop uses the same global center, but ROI mapping is derived from
the bounded global slice starts and stops rather than assuming a centered
offset. This remains correct when the larger crop shifts at a source-volume
boundary. The context pipeline scans once, shares that raw scan between its two
branches, and records its execution in
`context_diagnostics.context_scanner_execution_count`; the formal base
`policy_validation.scanner_execution_count` remains three. Context output is
then sliced to the exact base global ROI before comparing shared raw `ft`,
scanner-thinned `fet`, voted `fv`, thinned `fvt`, and the persistence of the
original outlier points. Context sparse percentiles are computed on that same
base ROI, not on the full context crop.

The point-detail limit does not truncate persistence totals: summary counts and
fractions use every reconstructed base outlier. For each stored point,
`context_candidate_to_public_distance` is evaluated at the nearest context
candidate sparse-ridge sample (and is `null` if that sparse mask is empty),
rather than reusing the public distance at the original base coordinate.

This same-global-ROI run is a diagnostic context ablation, not the formal
large-crop acceptance run below. It reports persistence and figures without
automatically calling a ridge geologically valid or an artifact. At the current
recorded state, automatic validation remains failed with seven of eight checks
passing, adding `--fail-on-validation-failure` still exits 2, and manual
geological review remains pending. The `+5.0` distance threshold, scanner and
workflow defaults, and the meaning of `policy_validation.passed` are unchanged.
Do not create a passing F3 evidence fixture from these diagnostics.

### Formal Large-Crop Run

After the 64^3 run passes and its figures have been reviewed, choose one
representative center and replace `I3,I2,I1` below with its coordinates:

```bash
PYTHONPATH=src python examples/report_3d_f3d_scanner_thinning_policy.py \
  --data-root "$PYOSV_F3D_DATA_ROOT" \
  --comparison-profile quality-workflow-scanner-thinning-v1 \
  --center I3,I2,I1 \
  --crop-shape 128,128,100 \
  --interior-margin 40 \
  --save-volumes \
  --save-figures \
  --write-markdown-index \
  --fail-on-validation-failure \
  --output-json outputs/3d/f3d/scanner_thinning_policy_large_001/metrics.json \
  --pretty
```

At least one large crop must be run and reviewed before considering a default
change. Record both the requested center and effective crop bounds in evidence,
especially when an axis spans most or all of the source volume.

### Automatic External-Smoke Checks

The report calls its result `policy_validation`, with role
`truthless_external_smoke`; it must not call this an F3 promotion gate. Public
F3 `fv.dat` and `fvt.dat` are useful spatial smoke references but are not
independent truth labels.

The default validation checks are:

1. The shared scanner execution count equals the crop count. Running the raw
   scanner separately for the two policies fails the contract.
2. `ft`, `pt`, `tt`, `fet`, `fpt`, `ftt`, `fv`, `vp`, `vt`, and `fvt` are finite
   for both policies on every crop.
3. `fet`, `fv`, and `fvt` are nonempty for both policies on every crop. A crop
   where both branches are empty is not comparison evidence.
4. Candidate/baseline FVT nonzero-density ratio is within `[0.5, 2.0]` on every
   crop and in the aggregate. Record the worst crop as well as the aggregate.
5. Candidate minus baseline FVT edge-density proxy is at most `0.10`. Record
   both the largest per-crop increase and the aggregate mean increase.
6. Candidate sparse-distance p95 against public F3 `fvt.dat` is no more than
   `5.0` samples worse than baseline on every crop and in the aggregate. This is
   a conservative extreme-motion smoke, not a truth claim.
7. Candidate FVT-density coefficient of variation across crops is at most
   `2.0`. Also report its change from the baseline coefficient of variation.
8. Recursive effective-config comparison permits only `scanner_thin_mode` and
   the derived `effective_remove_edge_effects` value to differ. At the
   user-specified config level, only `scanner_thin_mode` may differ.

In addition to the existing normalized-correlation, top-percentile-overlap,
buffered-ridge-overlap, sparse-distance, finite, orientation, and edge-density
metrics, report `fet`, `fv`, and `fvt` nonzero fractions explicitly. Direct
policy diagnostics include baseline/candidate FVT buffered overlap and sparse
distance, candidate-only and baseline-only ridge fractions, and the
candidate-only fraction within the edge shell. These locate changes; they do
not by themselves rank geological quality.

### Visual and Manual Review

Generate the existing F3 reference-versus-policy slices, ridge overlays, MIPs,
FV diagnostics, and center slices for each policy. Also generate direct
baseline-versus-candidate FVT slices, a combined ridge overlay,
candidate-only/baseline-only ridge masks, and an edge-shell ridge overlay. The
Markdown index should place baseline and candidate views next to one another
and tabulate FVT density, edge-density proxy, public-FVT distance p95, buffered
precision/recall, candidate/baseline density ratio, and validation result for
each crop.

Automatic checks are necessary but not sufficient. Record a manual decision and
notes for all of the following before considering a default change:

- Major fault-surface continuity is preserved or improved.
- Weak and small faults have not disappeared broadly.
- Candidate-only ridges have not increased as random-looking noise.
- Geologically implausible parallel ridges have not increased.
- Planar artifacts do not appear near crop faces.
- Strike and dip do not become locally discontinuous.
- Large geologically plausible structures present only in the baseline have not
  been removed.

### Compact Evidence

Generated DAT volumes, PNG figures, the full `metrics.json`, and generated
Markdown stay under ignored `outputs/` paths and are not committed. After both
formal runs and manual review are complete, commit only a compact manifest at:

```text
tests/fixtures/f3d_scanner_thinning_policy/
  quality_reference_like_normal_v1_evidence.json
```

It should record policy IDs, source commit and clean/dirty state, Python/NumPy/
SciPy versions, F3 input SHA-256 values or an unambiguous dataset fingerprint,
requested centers and effective crop bounds, crop shapes, automatic validation
results, principal aggregate metrics, explicit manual-review results, and the
SHA-256 values of `metrics.json` and `visual_report.md`. Do not create a passing
F3 evidence fixture before both formal runs and human review have passed. A
compact manifest for the failed 64^3-by-3 run is committed at the path above;
it explicitly records `manual_review=pending_human_review` and
`large_crop=not_run_prerequisite_failed`, and is not a promotion artifact.

Run the thinning ablation report to compare normal/normal, mixed, and
reference/reference thinning cases:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/report_3d_f3d_thinning_ablation.py \
  --output-json outputs/3d/f3d/thinning_ablation_001/metrics.json \
  --count 3 \
  --crop-shape 64,64,64 \
  --interior-margin 16 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

The crop and full F3 reports also include `voting.orientation`, a report-only
summary for `vp_py.dat` and `vt_py.dat`. It records finite and nonzero sample
counts plus strike/dip mean, standard deviation, median, and median absolute
deviation on the high-`fv_py.dat` mask. To inspect the effect of surface
orientation smoothing, run the same crop twice and compare that block. Keep
`--surface-smoothing1` and `--surface-smoothing2` unchanged for this check;
those flags control dynamic-programming surface extraction, not orientation
re-estimation:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/run_3d_f3d_crop_validation.py \
  --output-dir outputs/3d/f3d/orientation_smoothing_default \
  --pretty

PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/run_3d_f3d_crop_validation.py \
  --output-dir outputs/3d/f3d/orientation_smoothing_off \
  --surface-orientation-smoothing 0 \
  --pretty
```

Do not commit full generated reports as fixtures. Generated JSON, PNG, Markdown,
and `.dat` outputs belong under `outputs/` or another ignored working directory.
A compact evidence manifest containing selected metrics, review results, and
artifact hashes is allowed when a validation workflow explicitly requires one;
it is not a substitute for retaining the generated files outside git.

## Large Crop Manual Validation

The `(128, 128, 100)` crop preset is an explicit long-running manual validation,
not part of regular checks or CI. It runs the scanner, thinning, voting, and
voter thinning on a substantially larger crop:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/run_3d_f3d_crop_validation.py \
  --large-crop-preset \
  --max-crops 1 \
  --output-dir outputs/3d/f3d/large_crop_001
```

The opt-in pytest wrapper is also manual:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
PYOSV_RUN_F3D_LARGE_CROP_PIPELINE=1 \
python -m pytest -q tests/test_f3d_large_crop_validation.py -s
```

If a large crop run is interrupted, rerun the command into a fresh output
directory. The crop validation script does not currently reuse partial scanner
or voting stages.

## Full F3 Pipeline

The full F3 run is manual and can be slow because it scans and votes over the
entire `(420, 400, 100)` volume. It is separate from the smoke, small-crop, and
large-crop validations. Run it explicitly with an output directory:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/run_3d_f3d_full.py \
  --output-dir outputs/3d/f3d/full_001
```

The full script writes `run_config.json`, `metrics.json`, and generated Python
volumes such as `fv_py.dat` and `fvt_py.dat` under `--output-dir`. Use
`--skip-save-intermediates` when only final vote volumes and reports are needed.
Use `--reuse-existing` to reuse complete stage outputs already present in that
directory; incomplete stage output sets are rejected with a clear error.

## Output Policy

- Never write into `PYOSV_F3D_DATA_ROOT`.
- Write generated reports and volumes under `outputs/` or another ignored
  working directory.
- Do not commit reference `.dat` files or generated `.dat` outputs.
- Do not commit generated PNGs or full F3 report JSON/Markdown. Commit only an
  explicitly documented compact evidence manifest. A passing manifest requires
  all required runs and review; a failed-run manifest must preserve the failure
  and unfinished-review status without implying promotion.

The F3 scripts reject output paths inside the data root.

## Interpretation

`pyosv` uses practical approximations for Mines JTK interpolation and filtering.
For example, interpolation is based on SciPy primitives rather than JTK sinc
interpolation, and smoothing may use SciPy Gaussian-style filters rather than
JTK recursive filters.

Reference comparisons should use practical metrics and visual review:

- finite-value summaries
- normalized correlation
- top-percentile overlap
- sparsity checks
- visual checks of fault ridges and thinned volumes

For general reference reports, these F3 metrics remain comparison fields rather
than acceptance thresholds. Read them in context with targeted synthetic tests.
The dedicated scanner-thinning policy report is the explicit exception: it
applies the conservative truthless external-smoke limits documented above to
detect extreme density, edge, movement, or stability failures. Passing those
limits is not proof that public F3 volumes are truth and does not replace visual
review.

For an operational figure-first workflow, including crop PNGs, ridge overlays,
MIPs, histograms, and multi-crop markdown indexes, see
`docs/f3d_visual_diagnostics.md`.

For reference-like thinning runs, the first expected improvements are not
necessarily high voxel-wise correlation. The main checks are:

- `fvt` `nonzero_fraction` moving closer to the reference.
- `buffered_ridge_overlap.interior.fvt.buffered_f1` improving.
- sparse ridge distance medians decreasing.
- ridge overlay figures showing fewer far-away candidate-only ridges.
- exact overlap remaining interpretable even when it is low for sparse ridges.

Previous normal/normal baseline context:

```text
normalized_correlation.interior.fvt.mean ~= 0.224
buffered_ridge_overlap.interior.fvt.buffered_f1.mean ~= 0.075
exact fvt ridge overlap F1/Jaccard = 0.0
```

Do not claim success until actual ablation results are generated and reviewed.

No bitwise equality with Java, Jython, or Mines JTK output is expected.
