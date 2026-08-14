# F3 Compact Publication

The F3 compact publication derives a small, self-validating public bundle from
a completed F3 mode-comparison bundle. It reads existing scientific evidence
and stage artifacts without rerunning scanner, voting, thinning, or skinning.
Only `PUBLIC-REF` and the `Q-QUAL` lineage are displayed.

The fixed stage mapping is:

| Stage | PUBLIC-REF | Q-QUAL lineage |
| --- | --- | --- |
| `ft` | `fl.dat` | quality scanner `ft.dat` |
| `fv` | `fv.dat` | quality scanner voting `fv.dat` |
| `fvt` | `fvt.dat` | Q-QUAL `fvt.dat` |

At `ft` and `fv`, quality-workflow-specific processing has not yet acted. The
candidate is therefore described as belonging to the Q-QUAL lineage, not as a
quality-workflow effect.

## Inputs and generation

Generation requires a completed F3 mode-comparison source bundle, the matching
official F3 data root, and an environment lock. The data root contains the
official `ep.dat`, `fl.dat`, `fv.dat`, and `fvt.dat` files. It must also contain
`xs.dat`, a big-endian float32 volume with the same shape, used only as the
seismic-amplitude background for the compact figures.

Install the optional visualization dependencies before generation:

```bash
python -m pip install -e ".[viz]"
```

Run generation with all eight recorded environment controls set before Python
starts:

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
NUMBA_DISABLE_JIT=0 \
NUMBA_NUM_THREADS=1 \
PYOSV_ACCEL=auto \
PYTHONPATH=src python -m pyosv.cli.f3_compact_publication \
  --f3-bundle <completed-f3-bundle> \
  --f3-data-root "$PYOSV_F3D_DATA_ROOT" \
  --environment-lock <environment-lock> \
  --output-dir <new-output-dir> \
  --pretty
```

The output directory must not exist and must be outside the F3 source bundle
and data root. Generation records the current `ozw4/pyosv` Git commit and dirty
state, copies the environment lock byte-for-byte, validates the completed
private directory, and publishes it by an atomic sibling rename.

## Fixed slice and figures

All three stages use one `i2` slice. The generator reads the source-recorded
`positive_p99_radius2` threshold for the public `fvt.dat`, counts positive
samples at or above that threshold in each `i2` slice, and selects the slice
with the largest count. A tie selects the smallest index. The selected index is
not fixed to a dataset-specific constant.

Each stage figure has three panels:

1. gray amplitude with the `PUBLIC-REF` attribute in `magma`;
2. the same gray amplitude with the Q-QUAL-lineage attribute in `magma`;
3. the same gray amplitude with the signed `Q-QUAL - PUBLIC-REF` difference in
   `coolwarm`, centered at zero.

The amplitude range is the symmetric 99th percentile of the absolute selected
`xs.dat` slice. Attribute overlays use their source-recorded stage thresholds,
a shared stage scale from the recorded full-volume maxima, and maximum alpha
`0.75`. The difference is formed only on the selected 2-D slice; its symmetric
range is the 99th percentile of absolute difference and its alpha increases
with absolute difference.

## Summary metrics

`f3_q_qual_vs_public_ref_summary.csv` contains one row each for `ft`, `fv`, and
`fvt`, in that order. Every value is selected from the completed source metric
evidence; publication generation does not recompute metrics.

- `normalized_correlation`: normalized full-volume correlation.
- `mean_absolute_difference`: full-volume mean absolute attribute difference.
- `nonzero_fraction_ratio`: candidate nonzero fraction divided by the public
  reference nonzero fraction.
- `buffered_f1`: positive-p99 ridge F1 after a two-voxel spatial buffer.
- `candidate_to_reference_p95_voxel`: 95th-percentile candidate-to-reference
  ridge distance in voxels.
- `reference_to_candidate_p95_voxel`: 95th-percentile reference-to-candidate
  ridge distance in voxels.

Nullable directional distances are represented by an empty CSV field.

## Output layout

The generated directory has exactly this layout, where `<index>` is the one
selected `i2` index and the lock keeps its input basename:

```text
publication_manifest.json
experiment.json
<environment lock basename>
f3_q_qual_vs_public_ref_summary.csv
figure_data/
  f3_ft_public_ref_vs_q_qual_i2_<index>.csv
  f3_fv_public_ref_vs_q_qual_i2_<index>.csv
  f3_fvt_public_ref_vs_q_qual_i2_<index>.csv
figures/
  f3_ft_public_ref_vs_q_qual_i2_<index>.png
  f3_fv_public_ref_vs_q_qual_i2_<index>.png
  f3_fvt_public_ref_vs_q_qual_i2_<index>.png
report.md
```

The completed source bundle, official DAT files, and `xs.dat` remain external
inputs. They are read-only, are not modified, and are not copied into the
compact bundle.

## Validate only

Validate a completed or archive-extracted directory from its recorded files:

```bash
PYTHONPATH=src python -m pyosv.cli.f3_compact_publication \
  --validate-only \
  --output-dir <completed-output-dir>
```

Validate-only does not need the source bundle, F3 data root, environment lock,
Matplotlib, Numba, Git inspection, or current environment controls. It checks
the manifest schema and publication identity, every recorded artifact's safe
path, regular non-symlink status, size and SHA-256, the lock and experiment
links, and the exact directory file set.

## Reproduction and interpretation boundary

Directory validation proves integrity against the recorded compact manifest;
it does not replay the F3 numerical stages. Reproducing source evidence is a
separate operation that creates a completed F3 mode-comparison bundle.
Scientific generalization requires other surveys or independent geological
evidence.

The F3 public reference is an agreement target, not geological truth or an
accuracy label. The compact publication does not run significance tests,
choose a winner, or establish geological accuracy.
