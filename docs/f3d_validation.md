# F3 3D Reference Data Validation

This document defines the current full-volume F3 comparison contract used by
`pyosv.evaluation.f3d_mode_comparison` and
`pyosv.cli.f3d_mode_comparison`.

The workflow compares PyOSV outputs with public F3 processing outputs. It does
not provide independent geological truth. Terms such as accuracy, fault
recovery, and topology correctness belong to controlled synthetic experiments,
not to F3 public-reference agreement.

## Evaluation scope

Publication-facing F3 evaluation uses one complete volume with shape
`(420, 400, 100)` in repository order `(n3, n2, n1)`. The complete volume is
one dataset and one evaluation unit.

Crops, regional partitions, slices, blocks, and processing tiles are diagnostic
views within that unit. They are not independent samples, replicates, or
repeated experiments. Primary metrics are computed over all voxels of each
complete stage volume. Interior and boundary-shell rows are regional
diagnostics from the same volume.

The public reference-to-stage mapping is:

| Public file | PyOSV stage | Interpretation |
| --- | --- | --- |
| `fl.dat` | scanner likelihood `ft` | Public-reference agreement for scanner likelihood. |
| `fv.dat` | voted likelihood `fv` | Public-reference agreement for vote evidence. |
| `fvt.dat` | thinned voted likelihood `fvt` | Public-reference agreement for thinned vote evidence. |

F3 has no public strike, dip, or skin truth. Orientation rows compare PyOSV
cells with one another, and skin rows describe output structure rather than
truth accuracy.

## Official dataset contract

The official data is external to the repository. Set the data root with either
`--data-root` or `PYOSV_F3D_DATA_ROOT`:

```bash
export PYOSV_F3D_DATA_ROOT=/path/to/external/reference_osv
```

The canonical full-volume comparison requires these files:

```text
ep.dat
fl.dat
fv.dat
fvt.dat
```

Each required file has this storage layout:

```text
shape = (420, 400, 100)
storage dtype = >f4
bytes = 67,200,000
```

The semantic roles are fixed:

```text
input                              -> ep.dat
reference_fault_likelihood         -> fl.dat
reference_fault_votes              -> fv.dat
reference_thinned_fault_votes      -> fvt.dat
```

`ep.dat` is the scanner input. Reconstructing `ep.dat` from seismic amplitude is
outside this workflow.

`xs.dat` is a signed seismic-amplitude volume used by some optional visual and
outlier diagnostics. It is not part of `OFFICIAL_F3_DATASET_SPEC` and is not
required by the canonical four-cell runner.

`F3DatasetSpec` serializes exactly these fields:

```text
dataset_id
shape
storage_dtype
files
expected_bytes
```

`OFFICIAL_F3_DATASET_SPEC` uses:

```text
dataset_id = f3d-official-v1
shape = (420, 400, 100)
storage_dtype = >f4
expected_bytes = 67200000
```

The runner validates every required source as a regular file, checks its byte
size, and records a path-independent SHA-256 identity together with resolved
path provenance. The canonical plan accepts the official dataset specification;
custom specifications are limited to low-level fixture tests.

The output directory must not be equal to or nested under the data root.
Generated files must never be written into the external dataset directory.

## Canonical scanner-backend × workflow matrix

The fixed cell order is:

| Label | Scanner backend | Workflow mode |
| --- | --- | --- |
| `RL-REF` | `reference-like` | `reference` |
| `RL-QUAL` | `reference-like` | `quality` |
| `Q-REF` | `quality` | `reference` |
| `Q-QUAL` | `quality` | `quality` |

Scanner backend and workflow mode are independent axes. The canonical
terminology is defined in [Mode Comparison Contract](mode_comparison.md).

The stage order is:

```text
ep.dat
  -> scanner and scanner thinning
  -> voting
  -> voter thinning
  -> skinning
  -> metrics, contrasts, diagnostics, and resources
```

### Fixed scanner controls

The two scanner configurations differ only in `backend`.

| Setting | Canonical value |
| --- | --- |
| strike range | `0.0` through `360.0` degrees |
| dip range | `65.0` through `80.0` degrees |
| `sigma1`, `sigma2` | `8.0`, `8.0` |
| refinement factor | `2` |
| orientation backend | `rotate_shear` |
| interpolation backend | `scipy` |
| interpolation order | `1` |
| explicit smoothing sigma | `None` |
| normalization | enabled |
| output dtype | `float32` |
| scanner thinning | `reference` |
| scanner reference-thin sigma | `1.0` |
| requested edge cleanup | enabled |
| effective edge cleanup | enabled |

The refinement factor is effective for the `quality` backend. It remains part
of the complete resolved scanner configuration for both backends so the plan
can prove that workflow selection did not alter scanner controls.

### Fixed voting controls

These controls are held constant across all four cells:

| Setting | Canonical value |
| --- | ---: |
| `ru`, `rv`, `rw` | `10`, `20`, `30` |
| seed distance | `4` |
| seed threshold | `0.3` |
| strain maxima | `0.25`, `0.25` |
| attribute smoothing passes | `1` |
| surface smoothing | `2.0`, `2.0` |
| surface-orientation smoothing | `30.0` |
| final normalization smoothing | `0.0` |
| voter reference-thin sigma | `1.0` |
| surface-support minimum fraction | `0.0` |
| surface-support exponent | `0.0` |
| surface-voting boundary policy | `reference` |

Final vote normalization therefore performs minimum subtraction, positive
maximum scaling, and the `1 - (1 - x) ** 8` transform without final vote-map
smoothing.

### Workflow-owned settings

The workflow axis resolves only downstream settings:

| Setting | `reference` workflow | `quality` workflow |
| --- | --- | --- |
| voter thinning | `reference` | `hybrid_v2` |
| skinner method | `reference` | `quality` |
| skinner minimum likelihood | `0.5` | `None` / adaptive |
| seed planarity threshold | `0.8` | `0.5` |
| growth source | `thinned` | `pre_thin` |
| configured accepted-occupancy radius | `None` | `1` |
| effective accepted-occupancy radius | `5` | `1` |
| boundary fallback | disabled | enabled |
| boundary fallback policy | `empty_primary` | `empty_primary` |

The common skinning template enables reskinning. A new CLI run uses
`existing_cells_v1` unless `--skinner-reskin-policy` selects another supported
policy. The selected policy is applied consistently to all cells.

`voter_thin_mode_override` exists in the library configuration for controlled
programmatic comparisons. A canonical plan applies one override identically to
both workflow branches; it never changes scanner configuration.

### Stage sharing

For each scanner backend, the runner computes raw scanner output and scanner
thinning once and shares those attributes between its two workflow cells.
Voting is also shared between workflows for the same scanner backend because
voter thinning and skinning occur in later stages.

With skinning enabled, the canonical stage graph contains:

```text
2 scanner stages
2 voting stages
4 thinning stages
4 skinning stages
```

Every stage is content-addressed. Reuse is permitted only when its parents,
input identities, implementation identity, runtime identity, and resolved
settings produce the exact recorded fingerprint.

## Publication runtime contract

Set numerical controls before Python starts:

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

A publication-valid runtime requires:

- `PYTHONHASHSEED=0`;
- `OMP_NUM_THREADS=1`;
- `OPENBLAS_NUM_THREADS=1`;
- `MKL_NUM_THREADS=1`;
- `NUMEXPR_NUM_THREADS=1`;
- an explicitly selected `PYOSV_ACCEL` mode;
- available Numba with effective JIT state `enabled`;
- `NUMBA_DISABLE_JIT` unset, empty, or `0`;
- `NUMBA_NUM_THREADS=1` when that variable is set;
- available NumPy CPU, NumPy BLAS, and SciPy build identities;
- an effective thread count of one for detected NumPy BLAS libraries.

`run_manifest.json` records the requested acceleration mode, effective Numba
state and version, Python implementation, platform, byte order, thread and
Numba environment controls, CPU dispatch identity, NumPy build identity, BLAS
runtime identity, and SciPy build identity.

The runtime identity prevents incompatible numerical execution paths from being
mixed during resume or deep replay. It is not a cryptographic signature and
does not guarantee bitwise equality across arbitrary hardware.

## Running the full-volume comparison

The command always evaluates all four cells. It has no scanner-axis or
workflow-axis selector.

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PYOSV_ACCEL=auto \
NUMBA_DISABLE_JIT=0 \
NUMBA_NUM_THREADS=1 \
python -m pyosv.cli.f3d_mode_comparison \
  --data-root /path/to/external/reference_osv \
  --output-dir outputs/3d/f3d/mode_comparison_001
```

Without `--resume`, `--output-dir` must not exist. The parent directory is
created as needed.

Current top-level options include:

- `--boundary-margin N`: changes only the regional boundary-shell partition;
- `--no-skinning`: disables skinning identically in all four cells;
- `--skinner-reskin-policy POLICY`: selects the common reskin policy;
- `--pretty`: pretty-prints root completion JSON while fixed report files remain
  canonical;
- `--deep-validate`: performs deep validation during completion;
- `--compare-reskin-policies existing_cells_v1,reference_dense_v1`: creates the
  fixed same-parent Q-QUAL skin-only comparison after the source bundle.

`--validate-only` cannot be combined with `--resume`.
`--no-skinning` cannot be combined with `--compare-reskin-policies`.

The thin entry point
[`examples/run_3d_f3d_mode_comparison.py`](../examples/run_3d_f3d_mode_comparison.py)
invokes the same package CLI.

## Resume

Resume an interrupted workspace with the same dataset, plan, implementation,
and numerical runtime identity:

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PYOSV_ACCEL=auto \
NUMBA_DISABLE_JIT=0 \
NUMBA_NUM_THREADS=1 \
python -m pyosv.cli.f3d_mode_comparison \
  --data-root /path/to/external/reference_osv \
  --output-dir outputs/3d/f3d/mode_comparison_001 \
  --resume
```

An incomplete workspace has no valid root `completion.json`. Resume validates
all existing content-addressed stages, reuses valid exact matches, and computes
only missing stages. A mismatched dataset identity, plan, implementation,
runtime identity, or run fingerprint is rejected.

A complete workspace is validated rather than recomputed. The regular source
resume path still resolves the data root. The comparison-only resume path
described below can reuse a complete source bundle through recorded provenance
without entering source-stage orchestration.

## Bundle validation

### Shallow validation

Validate a completed bundle without running experiment stages:

```bash
python -m pyosv.cli.f3d_mode_comparison \
  --output-dir outputs/3d/f3d/mode_comparison_001 \
  --validate-only
```

Shallow validation checks the recorded publication runtime policy, run and stage
fingerprints, manifests, completion records, artifact paths, regular-file
status, sizes, SHA-256 values, schemas, stage bindings, report field sets, and
cross-file scalar consistency. It does not require the current process to match
the recorded runtime and does not replay numerical stages.

### Deep validation

Run explicit numerical consistency checks with:

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PYOSV_ACCEL=auto \
NUMBA_DISABLE_JIT=0 \
NUMBA_NUM_THREADS=1 \
python -m pyosv.cli.f3d_mode_comparison \
  --output-dir outputs/3d/f3d/mode_comparison_001 \
  --validate-only \
  --deep-validate
```

Deep validation requires the recorded source data to be accessible and the
current numerical runtime to satisfy and match the recorded publication
identity. It:

- rereads public reference and stage volumes;
- recomputes scanner array summaries;
- re-derives scanner sampling evidence from the resolved configuration;
- recomputes metric evidence and published metric rows;
- recomputes regional and orientation diagnostics;
- reruns the final skinning phase from persisted scanner, voting, and thinning
  parents;
- compares complete final skin cells, including subvoxel coordinates, rounded
  indices, likelihood, strike, dip, order, duplicates, generation provenance,
  reskin support, mask, and topology.

It does not rerun scanner computation, voting computation, or base voter
thinning. Deep validation is an internal consistency check, not proof of data
authenticity, geological truth, or tamper prevention.

Only a valid regular root `completion.json`, written after the report and stage
contracts succeed, marks the source bundle complete.

## Bundle layout

A completed source bundle has this root structure:

```text
run_manifest.json
completion.json
reports/
  cells.json
  metrics_long.csv
  metric_evidence.json
  contrasts.csv
  voxel_contrast_summaries.csv
  regional_metrics.csv
  orientation_diagnostics.csv
  runtime.csv
  resources.json
stages/
  scanner/<fingerprint>/
    stage_manifest.json
    complete.json
    ...
  voting/<fingerprint>/
    stage_manifest.json
    complete.json
    ...
  thinning/<fingerprint>/
    stage_manifest.json
    complete.json
    ...
  skinning/<fingerprint>/
    stage_manifest.json
    complete.json
    ...
```

`run_manifest.json` binds:

- the immutable resolved plan;
- path-independent dataset content identity and resolved-path provenance;
- source and implementation identity;
- numerical runtime identity;
- the run fingerprint.

Each stage manifest binds its kind, parent and input fingerprints, resolved
settings, implementation contract, expected artifacts, and stage fingerprint.
Each stage `complete.json` records artifact size and SHA-256 metadata.

The root `completion.json` binds every stage completion and every root report.
A stage directory or report not covered by the completion contract is not valid
bundle evidence.

### Report meanings

- `cells.json` records resolved cell settings and stage fingerprints.
- `metrics_long.csv` contains registry-defined full-volume and skin metrics.
- `metric_evidence.json` stores bounded scalar evidence used to validate metric
  algebra.
- `contrasts.csv` contains declared pairwise matrix contrasts.
- `voxel_contrast_summaries.csv` summarizes voxelwise matrix differences.
- `regional_metrics.csv` contains full, interior, and boundary-shell diagnostic
  rows bound to source stage fingerprints.
- `orientation_diagnostics.csv` contains scanner/cell orientation comparisons
  bound to both source stage fingerprints.
- `runtime.csv` records stage runtime attribution.
- `resources.json` records process RSS snapshots, storage, and resource
  interpretation metadata.

Runtime and resource rows describe this execution. They are not isolated
cross-machine benchmarks.

## External full-volume pytest gate

The opt-in full-data gate exercises the current publication contract:

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
NUMBA_DISABLE_JIT=0 \
NUMBA_NUM_THREADS=1 \
PYOSV_ACCEL=auto \
PYOSV_RUN_F3D_MODE_COMPARISON=1 \
PYOSV_F3D_DATA_ROOT=/path/to/external/reference_osv \
PYOSV_F3D_MODE_COMPARISON_OUTPUT_DIR=outputs/3d/f3d/mode_comparison_official \
python -m pytest -q tests/test_f3d_mode_comparison_full.py -s
```

The gate:

- validates the current publication runtime;
- runs or resumes the four-cell source bundle with deep validation;
- verifies canonical stage sharing and completion coverage;
- validates dataset hashes and runtime identity;
- generates and deep-validates the fixed Q-QUAL reskin-policy comparison;
- verifies that a complete comparison resume does not change its artifact
  paths, sizes, modification times, or hashes.

Default unit tests do not require the external F3 files or full-volume
computation.

## Same-parent reskin-policy comparison

The fixed comparison branches only Q-QUAL skinning from one immutable
`fv`/`fvt`/`vp`/`vt` parent and scanner-target mask. All skinning controls are
held constant except `reskin_policy`.

The policies are:

```text
baseline  = existing_cells_v1
candidate = reference_dense_v1
```

Generate the source bundle and comparison together:

```bash
PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
PYOSV_ACCEL=auto \
NUMBA_DISABLE_JIT=0 \
NUMBA_NUM_THREADS=1 \
python -m pyosv.cli.f3d_mode_comparison \
  --data-root /path/to/external/reference_osv \
  --output-dir outputs/3d/f3d/mode_comparison_reskin \
  --compare-reskin-policies existing_cells_v1,reference_dense_v1 \
  --deep-validate
```

Generate or resume only the comparison from a complete source bundle:

```bash
python -m pyosv.cli.f3d_mode_comparison \
  --output-dir outputs/3d/f3d/mode_comparison_reskin \
  --resume \
  --compare-reskin-policies existing_cells_v1,reference_dense_v1
```

Numerical generation and explicit deep replay require the exact
publication-valid runtime recorded by the source bundle. A comparison-only
resume first validates source provenance and reuses the Q-QUAL parents without
recomputing source stages.

The comparison directory contains:

```text
reskin_policy_comparison/
  reskin_policy_comparison.json
  reskin_policy_metrics.csv
  reskin_policy_comparison.md
  existing_cells_v1_skins.json
  reference_dense_v1_skins.json
  complete.json
```

The comparison report binds:

- source run and runtime identity;
- scanner, voting, and thinning stage fingerprints;
- the shared parent-content fingerprint, including the scanner target mask;
- the complete common skinning configuration;
- a configuration fingerprint proving that the two branches differ only in
  `reskin_policy`;
- canonical skin artifacts, generation diagnostics, link topology, skin
  topology, and correspondence with the positive ridge mask from parent `fvt`.

Parent FVT is comparison evidence, not geological truth.

Validate an existing source bundle and comparison without replay:

```bash
python -m pyosv.cli.f3d_mode_comparison \
  --output-dir outputs/3d/f3d/mode_comparison_reskin \
  --validate-only \
  --compare-reskin-policies existing_cells_v1,reference_dense_v1
```

Add `--deep-validate` to perform the same-runtime skin-only replay. Shallow
validation checks the serialized skin containers, bounded metric evidence,
configuration and parent bindings, completion hashes, and link-topology safety
invariants. Exact unserialized link topology is authoritative only through deep
replay.

A comparison `complete.json` records `validation_level` as `shallow` or `deep`.
Requiring deep completion checks saved deep evidence; explicitly requesting deep
validation replays it in the matching current runtime.

## Fast source-data checks

The general F3 metadata helpers cover five external files, including optional
`xs.dat`. Check their size, readability, ranges, and FVT sparsity with:

```bash
PYOSV_F3D_DATA_ROOT=/path/to/external/reference_osv \
python -m pytest -q tests/test_f3d_reference_data.py -s
```

Generate finite-value summaries for the public volumes with:

```bash
PYOSV_F3D_DATA_ROOT=/path/to/external/reference_osv \
python examples/report_3d_f3d_reference.py --pretty
```

These checks do not create a canonical four-cell bundle.

## Optional local diagnostics

The commands in this section operate on selected crops. They are useful for
local debugging, visualization, and stage isolation. Their rows and automatic
checks are not publication metrics or statistical replicates.

### One-crop pipeline

```bash
PYOSV_F3D_DATA_ROOT=/path/to/external/reference_osv \
python examples/run_3d_f3d_crop_validation.py \
  --max-crops 1 \
  --output-dir outputs/3d/f3d/crop_001 \
  --pretty
```

Add `--save-volumes` or `--save-figures` only when those artifacts are needed.
Automatic crop selection is margin-aware. `--center i3,i2,i1` selects one
explicit center.

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

`--compare-workflows` evaluates the `reference` and `quality` workflows at the
same centers. It invokes the scanner separately in each workflow branch while
holding scanner-side settings constant. It is a workflow comparison, not the
canonical shared-scanner four-cell runner and not a scanner-thinning policy
comparison.

The report's `quality_validation` block is a truthless smoke diagnostic. Its
limits can detect finite-value failure, density expansion, edge-density change,
large public-FVT displacement, or crop-to-crop instability. Passing those
checks does not establish geological quality.

### Scanner-thinning policy diagnostic

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
  --output-json outputs/3d/f3d/scanner_thinning_policy/metrics.json \
  --pretty
```

This diagnostic runs one shared `reference-like` scan per crop, then branches
scanner thinning:

```text
shared ft / pt / tt
  -> reference scanner thinning -> independent quality downstream path
  -> normal scanner thinning    -> independent quality downstream path
```

Both branches use the quality workflow and `hybrid_v2` voter thinning. Each
branch supplies its own scanner-thinned `fet` as the plateau tie-breaker. The
report role is `truthless_external_smoke`; public FVT remains a comparison
reference.

`--outlier-diagnostics` adds signed-amplitude and public-FVT displacement
inspection. `--context-crop-index` with `--context-crop-shape` recomputes the
same global base ROI from a larger context. Display-only ridge thresholds do
not alter metric masks or validation.

### Thinning ablation

```bash
PYOSV_F3D_DATA_ROOT=/path/to/external/reference_osv \
python examples/report_3d_f3d_thinning_ablation.py \
  --output-json outputs/3d/f3d/thinning_ablation/metrics.json \
  --count 3 \
  --crop-shape 64,64,64 \
  --interior-margin 16 \
  --pretty \
  --save-figures \
  --write-markdown-index
```

This isolates scanner and voter thinning combinations. Record scanner thinning
and voter thinning separately when interpreting the result.

### Large-crop diagnostic

```bash
PYOSV_F3D_DATA_ROOT=/path/to/external/reference_osv \
python examples/run_3d_f3d_crop_validation.py \
  --large-crop-preset \
  --max-crops 1 \
  --output-dir outputs/3d/f3d/large_crop_001
```

The crop runner does not provide content-addressed partial-stage resume. Use a
fresh output directory after interruption.

Visualization rules for full-volume and crop outputs are documented in
[F3 Visual Diagnostics](f3d_visual_diagnostics.md).

## Output policy

- Keep `PYOSV_F3D_DATA_ROOT` read-only.
- Write generated bundles, crops, reports, figures, and DAT files under
  `outputs/` or another ignored working directory.
- Do not commit the public F3 DAT files.
- Do not commit routine generated DAT volumes, PNG figures, or full report
  bundles.
- Source-controlled fixtures must be intentionally bounded test contracts; do
  not copy individual run hashes or experiment outcomes into permanent
  documentation.

The CLI rejects output paths inside the recorded or explicitly supplied data
root.

## Interpretation

PyOSV approximates Mines JTK interpolation and recursive filtering with
Python/SciPy numerical kernels. Bitwise equality with Java, Jython, or Mines JTK
is not required.

Use F3 reports to inspect:

- finite-value and range summaries;
- density and sparsity;
- normalized correlation and public-reference agreement;
- exact and buffered ridge overlap;
- directional sparse-ridge distance;
- scanner/workflow contrasts and interaction diagnostics;
- boundary behavior;
- orientation consistency between matrix cells;
- skin count, size, continuity, generation, and link safety;
- runtime, memory, and storage cost;
- fixed, predeclared visual comparisons.

A high public-reference score does not prove geological correctness. A lower
public-reference score does not by itself prove a quality regression. Interpret
F3 with the controlled truth experiments documented in
[Controlled Synthetic Quality](synthetic_quality.md).

The derived read-only publication report built from completed Synthetic and F3
source bundles is documented in
[Mode Comparison Publication Bundle](mode_comparison_publication.md).

## Related specifications

- [Mode Comparison Contract](mode_comparison.md)
- [Mode Comparison Publication Bundle](mode_comparison_publication.md)
- [F3 Visual Diagnostics](f3d_visual_diagnostics.md)
- [Controlled Synthetic Quality](synthetic_quality.md)
- [Quality Workflow Mode](quality_mode.md)
- [3D Orientation Scanning](orient3d.md)
- [3D Voting Conventions](3d_voting.md)
- [Reference-Like 3D Thinning](reference_like_thinning.md)
- [Skinning](skinning.md)
- [Reference-First Equivalence Policy](equivalence_policy.md)
