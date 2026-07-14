# F3 Visual Diagnostics

F3 visualization is a diagnostic workflow for understanding scanner, voting,
and thinning differences before changing numerical parameters. Do not tune
`normalized_correlation` until the figure outputs show which difference mode is
dominant.

## Data Layout

Use an external F3 data root. The local shared copy is:

```bash
export PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv
```

That directory must contain the public reference volumes:

```text
/home/dcuser/public_data/field/F3/reference_osv/
  ep.dat
  fl.dat
  fv.dat
  fvt.dat
  xs.dat
```

The files are read as big-endian `float32` volumes with shape
`(420, 400, 100)` in `(n3, n2, n1)` order. Generated PNGs and generated `.dat`
volumes belong in `outputs/` or another ignored working directory, never in the
data root or `reference_osv/`.

`xs.dat` is the signed input seismic amplitude image. `ep.dat` is the planarity
attribute used as scanner input, while `fl.dat`, `fv.dat`, and `fvt.dat` are
public-workflow attributes or processing results. None of those derived public
volumes is independent geological truth.

## Install Visualization Support

PNG diagnostics require the optional visualization dependency:

```bash
python -m pip install -e ".[dev,viz]"
```

The core package does not require matplotlib unless visualization helpers are
used.

## Small Crop Visual Report

Run one deterministic crop and write metrics, crop volumes, and PNG diagnostics
under `outputs/`:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
PYOSV_RUN_F3D_CROP_PIPELINE=1 \
python examples/run_3d_f3d_crop_validation.py \
  --output-dir outputs/3d/f3d/crop_visual_001 \
  --save-figures \
  --save-volumes \
  --pretty
```

The `PYOSV_RUN_F3D_CROP_PIPELINE=1` flag is only needed for the pytest wrapper,
but keeping it in the environment is harmless for the script. The script writes
`metrics.json` plus per-crop figure directories under `--output-dir`.

## Multi-Crop Visual Report

Run multiple deterministic crops when a single crop is not enough to determine
whether a difference is local or systematic:

```bash
PYOSV_F3D_DATA_ROOT=/home/dcuser/public_data/field/F3/reference_osv \
python examples/report_3d_f3d_multicrop.py \
  --output-json outputs/3d/f3d/multicrop_visual_001/metrics.json \
  --save-figures \
  --write-markdown-index \
  --pretty
```

The multi-crop script requires `--output-json` when figure or markdown output
is requested. It writes metrics to that JSON path, writes per-crop PNGs under
`OUTPUT_JSON.parent/crop_###/figures/`, and writes `visual_report.md` next to
the metrics JSON when `--write-markdown-index` is set. Use the markdown index as
the first browsing surface, then open individual PNGs for detail.

To compare the reference and quality workflows on the same crop centers, add
`--compare-workflows`:

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

In compare mode, the JSON contains `workflows.reference`,
`workflows.quality`, `consensus.workflows`, and
`workflow_delta.quality_vs_reference`, so the markdown index shows both the
reference and quality workflow results. It also contains top-level
`quality_validation`, a truthless external smoke summary for quality promotion
candidates. This smoke can flag obvious density explosion, edge-density
increase, sparse-distance regression, finite metric failures, and extreme
crop-to-crop instability, but it is not a substitute for the synthetic
promotion gate. The quality workflow uses `hybrid_v2` voter thinning unless
`--voter-thin-mode` is passed explicitly. The consensus section summarizes
truthless crop-to-crop stability from the saved crop metrics, including fvt/fv
nonzero density, fvt reference correlation, buffered ridge overlap, sparse
ridge distance p95, finite-check failures, and an fvt edge-density proxy from
full-crop minus interior density. In compare mode,
`consensus.workflow_comparison.quality_minus_reference` reports the matching
quality-minus-reference deltas. Figure directories are split by workflow, for
example `figures/reference/crop_001/` and `figures/quality/crop_001/`, so the
two runs do not overwrite each other. Support-aware voting is not a quality
default in this report; pass explicit `--surface-support-*` overrides only for
a diagnostic comparison.

Default quality smoke thresholds are intentionally loose: fvt density must not
exceed `2.0x` the reference workflow, fvt edge-density proxy delta must not
exceed `0.10`, and sparse distance p95 must not worsen by more than `5.0`
samples. Override them with `--quality-density-max-ratio`,
`--quality-edge-density-max-delta`, and
`--quality-sparse-distance-max-delta` when a diagnostic run needs a different
tolerance.

For reference-like thinning diagnostics, run the same visual reports with
`--scanner-thin-mode reference` and `--voter-thin-mode reference`, or run the
dedicated ablation report. Copy-pastable commands are in
`docs/f3d_validation.md#reference-like-thinning-validation`, and the thinning
mode behavior is summarized in `docs/reference_like_thinning.md`.

## Scanner-Thinning Distance-Outlier Review

The scanner-thinning policy report has an opt-in review for candidate sparse
ridges whose distance to public FVT is strictly greater than the baseline
candidate-to-public p95 plus the unchanged `5.0`-sample allowance. It uses the
same positive-only 99th-percentile masks, interior ROI, and unit-spacing 3D
Euclidean distance as the automatic sparse-distance metric. Public FVT is a
comparison reference here, not a truth label.

Generate the three-crop review with:

```bash
PYTHONPATH=src python examples/report_3d_f3d_scanner_thinning_policy.py \
  --data-root "$PYOSV_F3D_DATA_ROOT" \
  --comparison-profile quality-workflow-scanner-thinning-v1 \
  --count 3 \
  --crop-shape 64,64,64 \
  --interior-margin 16 \
  --outlier-diagnostics \
  --save-figures \
  --write-markdown-index \
  --output-json outputs/3d/f3d/scanner_thinning_policy_64x3_outlier_review/metrics.json \
  --pretty
```

For crop 1, add the exact same-global-ROI context comparison with:

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

Each crop uses one symmetric seismic display range for every outlier and every
panel. From finite samples in `xs_crop`, the report computes
`clip = percentile(abs(xs_crop), amplitude_clip_percentile)` (default `99.0`)
and displays grayscale amplitude with `vmin=-clip` and `vmax=clip`; a safe
fallback is used if the clip is zero or non-finite. Panels are never auto-scaled
independently.

The optional review limits default to 64 stored points, 8 stored connected
components, a 24-sample local window radius, and 3 adjacent slices on either
side. Adjust them with `--outlier-max-points`, `--outlier-max-components`,
`--outlier-window-radius`, and `--outlier-adjacent-slice-radius`. Adjust the
symmetric amplitude percentile with `--amplitude-clip-percentile`; these options
change diagnostic detail only, not validation thresholds.

The orthogonal review passes the representative outlier's actual `i3`, `i2`,
and `i1` slices, rather than only the crop-center slice. Its columns separate
amplitude-only, public-FVT, baseline-FVT, candidate-FVT, and combined overlays.
Thin contours preserve the underlying amplitude, and the legend distinguishes
public, baseline, candidate, the representative outlier, and its nearest public
point. The three adjacent-slice figures show the same overlay from `index-R`
through `index+R` for each axis (default `R=3`), omitting out-of-crop slices and
labelling the global index actually shown. Use them to distinguish a continuous
ridge trend from a single-slice speck.

Context figures use the same seismic amplitude, representative coordinate, and
global base ROI to compare base candidate FVT, context-derived candidate FVT,
their combined overlay, and base-only/context-only sparse masks. A persistent
ridge within two samples is evidence about context sensitivity only; it is not
an automatic geological judgment. Preliminary outliers being 19--25 samples
inside a crop does not itself eliminate context dependence because voting uses
`rw=30` and clamps out-of-crop surface samples to the crop edge.

The generated files live under paths such as:

```text
crop_001/policy_comparison/outlier_diagnostics/component_001/
crop_001/policy_comparison/context_diagnostics/component_001/
```

`visual_report.md` includes a `Public-FVT Distance Outlier Review` section only
when diagnostics are enabled. It embeds the orthogonal amplitude image and
links the adjacent-slice and context images using paths relative to
`metrics.json`.

The currently recorded formal `3 x 64^3` automatic validation still passes
seven of eight checks and fails the crop-1 public-FVT sparse-distance p95 check.
Manual geological review remains pending. These figures do not relax the
threshold, change a scanner/workflow default, create passing evidence, or
constitute formal large-crop acceptance; the `128 x 128 x 100` run above is a
diagnostic context ablation only. In report terms,
`manual_review.status=pending` remains separate from the failed automatic
result.

## Figure Interpretation

Use the figures to localize the mismatch before comparing scalar summary
metrics:

- `scanner_fl_vs_ftpy`: compare `fl.dat` against `ft_py.dat`; this shows scanner
  agreement before voting.
- `fv_ref_vs_py`: compare `fv.dat` against `fv_py.dat`; this shows voting score
  agreement and broad amplitude differences.
- `fvt_ref_vs_py`: compare `fvt.dat` against `fvt_py.dat`; this shows thinned
  sparse ridge agreement.
- `fvt_ridge_overlay`: inspect exact overlap, reference-only samples,
  pyosv-only samples, and buffered matches for shifted ridges.
- `fv_mip.png` and `fvt_mip.png`: compare broad 3D structural trends with
  maximum-intensity projections.
- `fv_hist.png` and `fvt_hist.png`: compare dynamic range, sparsity, and
  near-zero behavior.

For side-by-side slice panels, first look for obvious orientation, crop, or
boundary effects. For ridge overlays, distinguish an actual missing ridge from
a ridge that is consistently shifted by one or two samples.

## Why Correlation Is Not Enough For `fvt`

`normalized_correlation` is useful for dense volumes such as `fv`, but `fvt` is
a sparse thinned ridge volume. In sparse volumes, a small spatial shift can
produce poor sample-wise correlation even when the geological ridge trend is
visually close. The opposite can also happen: background zeros can make summary
statistics look less alarming while ridge placement is still wrong.

For `fvt`, always inspect ridge overlays and sparse-ridge metrics such as
buffered overlap and ridge-distance summaries. Treat correlation as one signal,
not as the tuning target.

When comparing F3 reference and quality workflows, remember that the F3 data has
no independent truth volume. A higher match to the reference workflow is not, by
itself, higher processing quality. Use the side-by-side crops to check that the
quality workflow preserves geological signal, does not add excessive ridges or
boundary artifacts, and remains consistent across crop locations.

F3 visual diagnostics require the external F3 data root. CI should exercise the
JSON and markdown structures with mocks/fixtures only; do not make real F3
volumes mandatory for automated tests.

For reference-like thinning experiments, first look for `fvt` sparsity moving
closer to the reference, better buffered ridge overlap, smaller sparse-ridge
distance medians, and fewer far-away candidate-only ridges. Exact overlap may
remain low for sparse ridges, so do not claim success until the ablation report
has been generated and reviewed.

## Recommended Diagnostic Order

1. Inspect scanner-only `fl_ref` versus `ft_py` figures.
2. Inspect `fv` side-by-side slice panels and MIPs.
3. Inspect `fvt` side-by-side panels and ridge overlays.
4. Read buffered ridge overlap and sparse-ridge distance metrics.
5. Repeat across the multi-crop report for consistency.
6. Tune parameters only after the difference mode is understood.

Visualization is diagnostic. These reports are meant to explain behavior and
guide focused experiments; they are not production pass/fail thresholds.
