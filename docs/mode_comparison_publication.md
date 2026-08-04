# Mode Comparison Publication Bundle

The publication command is a derived report generator, not an experiment
runner. It accepts a completed and validated synthetic mode-comparison bundle,
a completed and validated F3 full-volume comparison bundle, and the external F3
data root. It reads recorded scalar evidence and validated stage artifacts; it
does not rerun scanner, voting, thinning, or skinning stages.

Synthetic figures use known-truth metrics and paired contrasts. F3 figures use
public-reference agreement terminology. The public `fl.dat`, `fv.dat`, and
`fvt.dat` files are comparison targets rather than geological ground truth.
The F3 data root is required because the F3 source bundle records checksums and
file-role mappings for those read-only public reference volumes. The generator
checks dataset ID, shape, big-endian float32 storage, required roles, file
sizes, and SHA-256 values before it creates any figure.

## Fixed selection and visualization contracts

The curated metric selection is immutable and is derived from the existing
synthetic and F3 metric registries. Synthetic truth semantics and F3
public-reference-agreement semantics remain separate tables and are never
combined into a score.

The four condition order is always `RL-REF`, `RL-QUAL`, `Q-REF`, `Q-QUAL`.
Synthetic `scanner_raw` evidence keeps the source scanner-only labels
`RL-SCAN` and `Q-SCAN` as auxiliary rows; they are not mixed into the
canonical workflow-cell figures.
F3 stage order is `ft`, `fv`, `fvt`. Spatial slice selection is deterministic:
center index, public-reference positive-p99 ridge-count maximum, or the
positive difference peak for `abs(Q-QUAL - RL-REF)` on `fvt`; ties choose the
smallest index. The p99 threshold and radius 2 are taken from the validated F3
metric contract. Normal panels in one spatial figure share a scale derived from
validated full-volume min/max evidence. Signed difference panels use a separate
finite range centered on zero.

Synthetic source bundles are scalar-only, so synthetic figures are metric,
contrast, and runtime figures. Synthetic spatial replay is not performed.
F3 spatial figures use read-only memory maps of the existing stage DAT files
and public reference files. Public DAT files are never copied to the output.

## Output and validation

The output directory contains `manifest.json`, the publication CSV tables,
`figure_manifest.json`, `report.md`, `figure_data/`, `figures/`, and a final
`completion.json`. The publication artifact, completion, figure, and metric
selection contracts are each versioned independently. The directory is built
privately, validated internally, and renamed atomically only after completion;
an existing output directory is an error. Source bundles and the external F3
data root remain unchanged.

Validate an existing output without source bundles or visualization imports:

```bash
PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication \
  --output-dir outputs/3d/mode_comparison_publication/publication_v1 \
  --validate-only
```

Generate the fixed report:

```bash
PYTHONPATH=src python -m pyosv.cli.mode_comparison_publication \
  --synthetic-bundle <completed-synthetic-bundle> \
  --f3-bundle <completed-f3-bundle> \
  --f3-data-root "$PYOSV_F3D_DATA_ROOT" \
  --output-dir outputs/3d/mode_comparison_publication/publication_v1
```

Figures require the optional visualization extra:

```bash
python -m pip install -e ".[viz]"
```

Related contracts: [mode comparison terminology](mode_comparison.md),
[synthetic comparison](synthetic_mode_comparison.md),
[F3 validation](f3d_validation.md), and
[visualization helpers](visualization.md).
