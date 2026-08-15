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

## Fixed sections and atlases

The volume order is `(n3, n2, n1)`. Time slices use `volume[:, :, i1]` and are
reported as array indices `i1=<index>`. Inline sections use
`volume[i3, :, :]` and are reported as `i3=<index>`; no physical-inline number
conversion is applied.

For each axis, the generator divides its length into five contiguous equal
bins. Within every bin it uses the source-recorded `positive_p99_radius2`
threshold for public `fvt.dat`, counts positive samples at or above the
threshold, and selects the largest ridge count. A tie, including an all-zero
bin, selects the smallest index. The fixed policy is
`public_fvt_positive_p99_peak_per_equal_bin`. Q-QUAL does not affect section
selection. The resulting five time slices and five inline sections are shared
by `ft`, `fv`, and `fvt`.

Each stage and orientation produces one five-row by three-column atlas. Its
columns are:

1. gray signed amplitude with the `PUBLIC-REF` attribute in `inferno`;
2. the same amplitude with the Q-QUAL-lineage attribute in `inferno`;
3. the same amplitude with signed `Q-QUAL - PUBLIC-REF` in `coolwarm`, centered
   at zero.

Amplitude uses a symmetric range from the 99th percentile of absolute values
across the five selected sections. This range is shared by all stages for the
same orientation. Attribute overlays use their source-recorded stage
thresholds and a shared stage scale from the recorded full-volume maxima.
Values below threshold are transparent. At and above threshold, alpha runs
from `0.12` to `0.85` using gamma `2.0`, so threshold ridges remain visible and
high values appear as brighter, denser inferno colors. Signed differences are
formed only on the selected 2-D sections. Their symmetric 99th-percentile range
is shared across the five sections in each stage and orientation, with alpha
linear in absolute difference.

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

The generated directory has exactly this layout; the lock keeps its input
basename:

```text
publication_manifest.json
experiment.json
<environment lock basename>
f3_q_qual_vs_public_ref_summary.csv
figure_data/
  f3_ft_time_slices.csv
  f3_ft_inline_sections.csv
  f3_fv_time_slices.csv
  f3_fv_inline_sections.csv
  f3_fvt_time_slices.csv
  f3_fvt_inline_sections.csv
figures/
  f3_ft_time_slices.png
  f3_ft_inline_sections.png
  f3_fv_time_slices.png
  f3_fv_inline_sections.png
  f3_fvt_time_slices.png
  f3_fvt_inline_sections.png
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
