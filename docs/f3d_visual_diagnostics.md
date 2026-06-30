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

For reference-like thinning diagnostics, run the same visual reports with
`--scanner-thin-mode reference` and `--voter-thin-mode reference`, or run the
dedicated ablation report. Copy-pastable commands are in
`docs/f3d_validation.md#reference-like-thinning-validation`, and the thinning
mode behavior is summarized in `docs/reference_like_thinning.md`.

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
