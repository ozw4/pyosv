# Quality Mode

This repository separates reference-alignment checks from controlled truth
quality experiments.

## Workflows

`reference` workflow is for origin-aligned regression comparison. Use it when
checking that Python behavior remains close to the current reference-oriented
path. It is not the place to evaluate processing-quality improvements.

`quality` workflow is the entry point for future quality-first implementation
work. It favors controlled synthetic truth quality, but it is not the final
quality workflow yet.

`diagnostic` workflow is for side-by-side review of variants and thinning
diagnostics. Use it when comparing current behavior, diagnostic variants, and
reference-vs-normal thinning on the same synthetic truth.

## Synthetic Truth Benchmark

Use this command as the recommended reproducible benchmark matrix before and
after quality changes:

```bash
PYTHONPATH=src python examples/report_3d_synthetic_quality.py \
  --case-set extended \
  --shape 33,33,33 \
  --variant-preset quality-matrix \
  --input-mode both \
  --workflow-mode diagnostic \
  --output-dir outputs/3d/synthetic_quality/quality_matrix_001 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

Review `summary.csv` first, then use `metrics.json` and visual overlays for
drill-down.

The `quality-matrix` preset includes `current_default`,
`no_surface_orientation_smoothing`, `final_norm_smoothing_1`,
`voter_thin_normal`, `voter_thin_hybrid`, and
`surface_support_weighted`. The hybrid voter thinning variant uses
reference-like thinning in stable-orientation regions and fault-normal thinning
where local orientation changes rapidly.

The `surface_support_weighted` diagnostic variant keeps the default thinning
path but enables support-aware surface voting with
`surface_support_min_fraction=0.5` and `surface_support_exponent=1.0`. This
skips extracted surfaces with low valid support and down-weights the remaining
vote by its valid-support fraction, which is useful for boundary-plane edge
artifact diagnostics. The default support policy is `0.0, 0.0`, so reference
mode behavior is unchanged unless the report CLI flags
`--surface-support-min-fraction` or `--surface-support-exponent` are set, or
the diagnostic variant is selected.

Primary metrics to compare:

- `fvt_buffered_f1_r2`
- `fvt_distance_candidate_to_truth_p95`
- `fvt_strike_median_error`
- `fvt_dip_median_error`
- `skin_buffered_f1_r2`
- `skin_distance_candidate_to_truth_p95`
- `edge_false_positive_fraction` columns
