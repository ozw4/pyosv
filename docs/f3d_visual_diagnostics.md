# F3 Visual Diagnostics

F3 visualization explains differences between the canonical scanner-backend ×
workflow cells. It does not convert public F3 processing outputs into
independent geological truth, and it does not replace controlled synthetic
known-truth evaluation.

Publication-facing figures use the complete F3 volume with shape
`(420, 400, 100)` in repository order `(n3, n2, n1)`. The full volume is one
evaluation unit. Slices, regional partitions, projections, and crops are views
of that unit, not independent samples or statistical replicates.

## Visualization entry points

The canonical F3 source runner creates and validates the full-volume stage and
scalar bundle. It does not import Matplotlib or write publication PNG files:

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PYOSV_ACCEL=auto \
NUMBA_DISABLE_JIT=0 \
NUMBA_NUM_THREADS=1 \
PYOSV_F3D_DATA_ROOT=/path/to/external/reference_osv \
python -m pyosv.cli.f3d_mode_comparison \
  --output-dir outputs/3d/f3d/mode_comparison_001
```

The derived publication command reads a completed Synthetic bundle, a completed
F3 bundle, and the matching external F3 data root. It generates the fixed
publication tables, figure-data CSV files, PNG figures, and Markdown report
without rerunning scanner, voting, thinning, or skinning stages.

Install the visualization extra before generation:

```bash
python -m pip install -e ".[viz]"
```

Generate the publication bundle with:

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
NUMBA_DISABLE_JIT=0 \
NUMBA_NUM_THREADS=1 \
PYOSV_ACCEL=auto \
PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication \
  --synthetic-bundle <completed-synthetic-bundle> \
  --f3-bundle <completed-f3-bundle> \
  --f3-data-root "$PYOSV_F3D_DATA_ROOT" \
  --environment-lock uv.lock \
  --output-dir outputs/3d/mode_comparison_publication/publication_v1
```

Validate an existing publication directory without source bundles, F3 data, or
Matplotlib:

```bash
PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication \
  --validate-only \
  --output-dir outputs/3d/mode_comparison_publication/publication_v1
```

Validate-only checks the recorded file set, hashes, manifest links, and
publication identity. It does not rerender figures, parse PNG dimensions, or
re-evaluate figure semantics.

## Data contract

The canonical F3 source bundle is derived from these public-reference roles:

| Stage | Public reference | PyOSV stage artifact |
| --- | --- | --- |
| `ft` | `fl.dat` | scanner `ft.dat` |
| `fv` | `fv.dat` | voting `fv.dat` |
| `fvt` | `fvt.dat` | thinning `fvt.dat` |

The public files are big-endian `float32` volumes with shape
`(420, 400, 100)`. The publication generator verifies the dataset identity
against the completed F3 source bundle before reading spatial data.

`xs.dat` is signed seismic amplitude used by selected crop diagnostics. It is
not a public-reference target for the canonical `ft`, `fv`, or `fvt` figures.

Generated files must be written outside `PYOSV_F3D_DATA_ROOT` and outside the
completed source bundles.

## Publication figure artifacts

A publication directory contains:

```text
figure_data/
  <figure_id>.csv
figures/
  <figure_id>.png
report.md
```

Every generated PNG has a matching figure-data CSV with the same `figure_id`.
The CSV records the scalar source rows or spatial rendering metadata used by the
figure, including applicable fields such as:

- dataset and evaluation semantics;
- source metric and processing stage;
- condition and panel labels;
- spatial axis and zero-based slice index;
- slice-selection policy and selection score;
- public-reference and candidate ridge thresholds;
- display minimum, maximum, scale policy, and colormap;
- signed-difference limits.

`experiment.json` records the fixed slice-selection policy for the publication
experiment. `report.md` links the figures and supplies their captions.

`figure_data/*.csv` files are primary publication artifacts and participate in
`publication_id`. PNG files and `report.md` are derived presentation artifacts
and do not participate in `publication_id`. Both tiers are still size- and
SHA-256-validated by `publication_manifest.json`.

## Fixed scalar figures

The publication bundle generates these F3 scalar figures from validated root
CSV tables:

| Figure ID | Meaning |
| --- | --- |
| `f3_normalized_correlation_by_stage` | Full-volume normalized correlation with the public stage reference for `ft`, `fv`, and `fvt`. |
| `f3_buffered_f1_by_stage` | Positive-p99, radius-2 buffered ridge agreement for all three stages. |
| `f3_sparse_distance_p95_by_stage` | Candidate-to-public and public-to-candidate sparse-ridge distance p95. |
| `f3_nonzero_fraction_ratio_by_stage` | Candidate/public nonzero-density ratio by stage. |
| `f3_runtime_breakdown` | Within-experiment attribution of shared and cell-owned F3 stages. |

The four matrix cells use the canonical order:

```text
RL-REF, RL-QUAL, Q-REF, Q-QUAL
```

Sparse-distance null values remain missing; they are not rendered as zero.
Runtime bars describe the recorded stage attribution within the experiment and
are not isolated-process benchmarks.

Regional and orientation diagnostics remain machine-readable tables:

```text
f3_regional_summary.csv
f3_orientation_summary.csv
```

They are not converted into additional fixed PNG families. F3 orientation rows
compare matrix cells with one another because no public F3 strike or dip truth
is available.

## Fixed spatial figure set

Spatial figures are generated for the global axes `i3`, `i2`, and `i1`.
The corresponding 2-D array views are:

| Axis | Fixed index dimension | Displayed plane |
| --- | --- | --- |
| `i3` | axis 0 | `(n2, n1)` |
| `i2` | axis 1 | `(n3, n1)` |
| `i1` | axis 2 | `(n3, n2)` |

Figure IDs use these patterns:

```text
f3_<stage>_comparison_<selection_policy>_<axis>_<index>
f3_fvt_ridge_overlay_<selection_policy>_<axis>_<index>
```

### Stage and selection coverage

The fixed spatial coverage is:

| Stage | `center` | `public_reference_peak` | `end_to_end_difference_peak` |
| --- | --- | --- | --- |
| `ft` | all three axes | all three axes | not generated |
| `fv` | all three axes | all three axes | not generated |
| `fvt` | all three axes | all three axes | all three axes |

FVT ridge overlays are generated for `public_reference_peak` and
`end_to_end_difference_peak` on all three axes. Center-slice ridge overlays are
not part of the fixed publication set.

### `ft` panel layout

Scanner output is shared between workflows for the same scanner backend, so the
`ft` figure shows each scanner backend once:

```text
PUBLIC-REF fl.dat
reference-like scanner ft
quality scanner ft
quality - reference-like signed difference
```

The signed difference is the quality scanner output minus the reference-like
scanner output. The cell metadata is represented by `RL-REF` and `Q-REF`; the
workflow labels do not create additional scanner arrays.

### `fv` and `fvt` panel layout

Voting and thinning figures use:

```text
PUBLIC-REF
RL-REF
RL-QUAL
Q-REF
Q-QUAL
Q-QUAL - RL-REF signed difference
```

The signed panel is a descriptive end-to-end contrast between two aligned
matrix cells. It is not an inferential treatment effect or a comparison of
independent replicates.

## Slice-selection rules

All indices are zero-based. Ties use the smallest index.

### Center

For an axis of length `n`, the center index is `n // 2`. For the official F3
shape this resolves to:

| Axis | Length | Center index |
| --- | ---: | ---: |
| `i3` | 420 | 210 |
| `i2` | 400 | 200 |
| `i1` | 100 | 50 |

### Public-reference peak

For each stage and axis, the generator scans the public-reference slices and
selects the index with the largest count of positive ridge samples at the
validated public-reference p99 threshold.

The selection is deliberately independent of candidate volumes. The
publication generator reuses the threshold recorded in validated F3 metric
evidence and does not recalculate a percentile from the publication-time data.

### End-to-end difference peak

For FVT, the generator selects the slice with the largest:

```text
sum(abs(Q-QUAL - RL-REF))
```

The score is evaluated one 2-D slice at a time. The generator does not
materialize a full-volume candidate-minus-candidate array.

## Ridge threshold and overlay contract

F3 ridge overlays use the validated source selection:

```text
selection = positive_p99_radius2
percentile = 99
buffer radius = 2 voxels
positive epsilon = 1e-6
```

The source metric evidence supplies:

- one public-reference threshold per stage, consistent across the four cells;
- one candidate threshold for each cell and stage.

Candidate-specific thresholds are retained. A threshold from one cell is not
silently reused for another cell, and publication generation does not recompute
thresholds from stage volumes.

Each FVT overlay contains four panels:

```text
PUBLIC-REF vs RL-REF
PUBLIC-REF vs RL-QUAL
PUBLIC-REF vs Q-REF
PUBLIC-REF vs Q-QUAL
```

The categorical colors are:

| Color | Meaning |
| --- | --- |
| red | public-reference ridge only |
| blue | candidate ridge only |
| white | exact ridge overlap |
| cyan | radius-2 buffered match without exact overlap |

Buffering is computed from a radius-sized 3-D slab around the displayed slice,
so the displayed match category respects neighboring samples across the slice
axis without loading a full duplicate mask volume.

## Display-scale contract

Normal `ft`, `fv`, and `fvt` panels in one figure share one `viridis` scale.
The range comes from validated full-volume minimum and maximum evidence across
the public reference and all normal candidate panels displayed in that figure.
Per-panel automatic contrast is not used.

Signed-difference panels use `coolwarm` and a zero-centered symmetric range:

```text
[-max(abs(displayed_difference)), +max(abs(displayed_difference))]
```

The difference limit is calculated from the displayed 2-D difference slice.
Normal and difference panels therefore use separate, explicitly recorded scale
contracts.

Ridge overlays use categorical colors and do not share the scalar likelihood
scale.

## Memory and source-access behavior

Publication spatial generation opens validated public and stage DAT files as
read-only memory maps. Candidate stage volumes are resolved from their
content-addressed scanner, voting, or thinning stage fingerprints.

The generator:

- reads normal spatial data by 2-D slice;
- calculates signed differences after reading the two selected slices;
- searches difference peaks one slice at a time;
- closes opened stage memory maps after figure generation;
- does not modify source bundles or public F3 files.

This keeps figure generation derived-only and avoids allocating a full-volume
signed-difference array.

## Reading the publication figures

A practical review sequence is:

1. Confirm the matrix, dataset identity, and full-volume interpretation in
   `experiment.json` and `report.md`.
2. Read normalized correlation, buffered F1, sparse distance, and density ratio
   together; no single F3 scalar is a geological quality score.
3. Inspect center slices for fixed-position differences.
4. Inspect public-reference-peak slices for high-ridge-density behavior selected
   independently of candidates.
5. Inspect the FVT end-to-end-difference peak for the largest displayed
   `Q-QUAL` versus `RL-REF` slice discrepancy.
6. Use ridge overlays to distinguish exact agreement, buffered displacement,
   public-only ridges, and candidate-only ridges.
7. Check `f3_regional_summary.csv` for boundary/interior context and
   `f3_orientation_summary.csv` for pairwise orientation consistency.
8. Read runtime and storage results as resource diagnostics, not quality
   metrics.

A public-only ridge is not automatically correct, and a candidate-only ridge is
not automatically false. F3 figures require geological interpretation and
should be read with the controlled synthetic known-truth results.

## Current publication scope

The fixed publication generator currently produces scalar plots, orthogonal
stage slices, signed-difference slices, and FVT ridge overlays.

The fixed set does not produce:

- F3 MIP figures;
- axis-profile plots;
- value-distribution or histogram plots;
- full-volume difference DAT files;
- F3 spatial strike/dip accuracy panels;
- F3 skin-truth accuracy figures;
- hand-selected slices chosen after reviewing results.

MIPs, histograms, and other exploratory views remain available through the
optional `pyosv.viz` helpers or crop diagnostics, but they are not part of the
publication figure contract.

## Optional local crop diagnostics

Crop diagnostics are useful for debugging, detailed amplitude review, and
stage isolation. Their selected locations remain views of the same F3 survey
and are not publication samples or replicates.

### One-crop visual report

```bash
PYOSV_F3D_DATA_ROOT=/path/to/external/reference_osv \
python examples/run_3d_f3d_crop_validation.py \
  --max-crops 1 \
  --output-dir outputs/3d/f3d/crop_visual_001 \
  --save-volumes \
  --save-figures \
  --pretty
```

Automatic crop selection is margin-aware. Use `--center i3,i2,i1` for an
explicit global center. Generated files belong under the output directory, not
under the F3 data root.

### Multi-crop workflow comparison

```bash
PYTHONPATH=src python examples/report_3d_f3d_multicrop.py \
  --data-root "$PYOSV_F3D_DATA_ROOT" \
  --count 3 \
  --crop-shape 64,64,64 \
  --interior-margin 16 \
  --compare-workflows \
  --save-figures \
  --write-markdown-index \
  --output-json outputs/3d/f3d/multicrop_workflows/metrics.json \
  --pretty
```

The report keeps reference and quality workflow figures in separate output
directories. Its crop-level density, edge, stability, and public-reference
checks are truthless diagnostics and do not replace the full-volume or
controlled-synthetic contracts.

### Scanner-thinning outlier and context review

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
  --output-json outputs/3d/f3d/scanner_thinning_outliers/metrics.json \
  --pretty
```

Recompute one crop in a larger context while comparing the same global base ROI
with:

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
  --output-json outputs/3d/f3d/scanner_thinning_context/metrics.json \
  --pretty
```

Outlier points measure candidate-to-public-reference displacement. They are not
truth labels. Display contours may be broader than metric ridge masks to make a
surface traceable; display-only thresholds must not alter the metric masks,
outlier coordinates, persistence counts, or validation result.

The larger-context path derives ROI mapping from bounded global slice starts and
stops. It does not assume that the requested crop remains centered after a
source-volume boundary adjustment.

## Optional static helpers

`pyosv.viz` provides reusable Matplotlib helpers for exploratory diagnostics:

- deterministic orthogonal slices;
- reference/candidate/difference panels;
- buffered ridge overlays;
- maximum-intensity projections;
- value histograms.

These helpers follow the project axis and shape conventions. Their outputs are
not publication artifacts unless they are generated through the fixed
publication command and recorded in its manifest.

See [Optional Visualization Helpers](visualization.md) for the API.

## Output policy

- Keep the external F3 data root read-only.
- Keep completed source bundles immutable while deriving figures.
- Write generated figures, CSV files, Markdown, and DAT files under `outputs/`
  or another ignored working directory.
- Do not commit public F3 DAT files or routine generated PNG/DAT report trees.
- Do not copy individual experiment outcomes or artifact hashes into permanent
  documentation.

## Related specifications

- [F3 3D Reference Data Validation](f3d_validation.md)
- [Mode Comparison Contract](mode_comparison.md)
- [Mode Comparison Publication Bundle](mode_comparison_publication.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [Quality Workflow Mode](quality_mode.md)
- [Optional Visualization Helpers](visualization.md)
- [Reference-First Equivalence Policy](equivalence_policy.md)
