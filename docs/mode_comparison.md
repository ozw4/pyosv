# Mode Comparison Contract

## 1.1 Purpose and scope

This document defines the mode names and comparison conditions used for public
synthetic/F3 comparisons. It is the canonical terminology and configuration
contract for those comparisons.

The scanner backend and the downstream workflow mode are independent axes. A
scanner backend selects the scanner implementation; a workflow mode resolves
downstream defaults. Scanner thinning and voter thinning are also separate,
stage-specific choices.

The F3 public reference is a comparison target, not a processing mode. The
public `fl.dat`, `fv.dat`, and `fvt.dat` files are not an independent
geological ground truth.

This contract work does not add an algorithm, change a default, or add a new
runner. In particular, there is no single cross-domain runner that combines a
synthetic experiment and an F3 public comparison into one report. Existing
domain-specific comparison APIs, when available, remain separate from this
terminology contract. The current implementation facts and the future public
comparison naming contract are described separately below.

Do not infer that a condition ID means that a particular runner has already
been implemented. The ID fixes the intended effective configuration; section
1.7 records which existing paths can currently produce it.

## 1.2 Terminology

| Term | Definition and contract |
| --- | --- |
| scanner backend | The scanner implementation that produces fault likelihood, strike, and dip attributes. The public comparison values are `reference-like` and `quality`; the existing implementation also has other diagnostic backends. |
| workflow mode | A downstream profile that resolves voting, voter thinning, skinning, and diagnostic defaults. Synthetic code accepts `reference`, `quality`, and `diagnostic`; the F3 crop workflow accepts `reference` and `quality`. |
| scanner thinning mode | The policy applied after scanner output and before voting. The synthetic scanner config accepts `reference`, `normal`, and `none`; the F3 crop CLI exposes `reference` and `normal`. |
| voter thinning mode | The policy applied to the vote volume after voting. The relevant public values are `reference`, `normal`, and `hybrid_v2`; the voter implementation also retains other explicit diagnostic values. |
| F3 public reference | The public F3 output files `fl.dat`, `fv.dat`, and `fvt.dat`, used as external comparison targets. This is not a workflow, scanner backend, or ground truth. |
| synthetic truth | The generated truth surface and truth orientation field owned by a synthetic case. These are independent ground-truth sources for synthetic evaluation. |
| oracle input | Synthetic pipeline input made directly from the case's truth orientation attributes (`ft_oracle`, `pt_oracle`, and `tt_oracle`), so scanner behavior is bypassed. |
| scanner-inclusive input | Synthetic pipeline input generated from the case's scanner input and passed through a selected scanner backend before downstream processing. `input_mode="scanner"` selects this path; the comparison runner can prepare both oracle and scanner inputs. |
| diagnostic workflow | The synthetic `diagnostic` profile, based on reference workflow defaults with additional thinning diagnostics enabled. It is not one of the four publication conditions and is not a primary performance mode. |

Bare `reference mode` and `quality mode` are ambiguous and should not be used
in public reporting. Use qualified terms instead:

- `reference-like scanner backend`
- `quality scanner backend`
- `reference workflow`
- `quality workflow`
- `reference scanner thinning`
- `reference voter thinning`
- `F3 public reference`

Existing machine-facing identifiers are unchanged. In particular, the scanner
backend value remains the hyphenated `reference-like`; this document does not
introduce an identifier such as `reference_like_mode`.

## 1.3 Pipeline stages

The stage order is:

```text
input -> scanner -> scanner thinning -> voting -> voter thinning -> skinning
```

| Stage | Configuration that acts at the stage |
| --- | --- |
| input | `input_mode`, synthetic case/input configuration, or the F3 `ep.dat` crop input |
| scanner | `scanner_backend`; scanner angles, sigma, interpolation, and related scanner controls |
| scanner thinning | `scanner_thin_mode`, scanner reference-thinning sigma, and scanner edge cleanup |
| voting | Voting radii, strain/smoothing settings, and surface-support policy |
| voter thinning | `voter_thin_mode` and voter reference-thinning controls |
| skinning | Skinner method, likelihood, seed, growth, occupancy, and boundary-fallback settings |

`workflow_mode` is a profile that resolves several downstream defaults; it is
not a stage-local scanner switch. `scanner_backend` acts at the scanner stage,
`scanner_thin_mode` at scanner thinning, `voter_thin_mode` at voter thinning,
and skinner settings at skinning.

The current F3 crop workflow stops at `fvt`: it runs through voter thinning and
does not run skinning. Therefore, synthetic skinner differences are not part
of the F3 crop workflow comparison.

## 1.4 Scanner backend contract

### `reference-like`

On the synthetic side, `FaultOrientScanner3.scan()` is used. The current
`scan()` implementation delegates to `scan_reference_like()` with the default
`rotate_shear` reference-like path. That path uses Java-inspired strike/dip
sampling and rotate, shear, smoothing, and unrotate operations approximated in
Python/SciPy.

This is a practical reference-like implementation, not a bit-exact port of
Mines JTK or the Java implementation. Scanner thinning is a later, independent
axis and must not be inferred from the backend name.

`SyntheticScannerConfig` contains `refinement_factor` for reporting and for
the quality backend. The reference-like dispatch does not pass it to
`FaultOrientScanner3.scan()`; it is therefore not effective in a
reference-like scan even though it appears in the report configuration.

### `quality`

On the synthetic side, `FaultOrientScanner3.scan_quality()` is used. It keeps
the reference-like scoring path and refines the strike and dip sampling grids
by `refinement_factor`. The current synthetic scanner configuration default is
`refinement_factor=2`.

The synthetic quality dispatch requests `return_confidence=True`. The scanner
returns a normalized confidence volume derived from the gap between the best
and second-best sampled orientation responses; the report records it as a
scanner diagnostic and the volume is available to scanner-side diagnostic or
ensemble logic. Confidence is not a workflow setting and is not geological
ground truth. The direct scanner API can omit that output by leaving
`return_confidence=False`.

The quality scanner backend is not the quality workflow. Selecting
`scanner_backend=quality` does not select `workflow_mode=quality`, and a
quality workflow does not select the quality scanner backend.

## 1.5 Workflow contract

The same workflow name covers different processing ranges in synthetic and F3
code. Synthetic workflow resolution includes skinning settings. The current
F3 crop resolver only resolves voter thinning and surface-support values; its
pipeline has no skinning stage.

### Synthetic reference workflow

The following are the effective defaults from the synthetic profile resolver
and report dictionaries:

| Setting | Effective default |
| --- | --- |
| voter thinning mode | `reference` |
| skinner method | `reference` |
| skinner minimum likelihood | `0.5` |
| adaptive minimum likelihood | `false` |
| skinner seed planarity threshold | `0.8` (`seed_min_ep`) |
| skin growth source | `thinned` |
| accepted occupancy radius | `None` (configured value) |
| effective accepted occupancy radius | `5` |
| boundary skinner fallback | `false` |
| boundary skinner fallback policy | `empty_primary` |
| surface support policy | minimum fraction `0.0`, exponent `0.0` |

### Synthetic quality workflow

| Setting | Effective default |
| --- | --- |
| voter thinning mode | `hybrid_v2` |
| skinner method | `quality` |
| skinner minimum likelihood | `None`; the quality skinner uses its adaptive threshold when no explicit value is supplied |
| adaptive minimum likelihood | `true` |
| skinner seed planarity threshold | `0.5` (`seed_min_ep`), lower than the reference workflow default |
| skin growth source | `pre_thin` |
| accepted occupancy radius | `1` |
| effective accepted occupancy radius | `1` |
| boundary skinner fallback | `true` |
| boundary skinner fallback policy | `empty_primary` |
| surface support policy | minimum fraction `0.0`, exponent `0.0` |

Explicit configuration values can override these defaults. The scanner backend
and scanner thinning mode are not filled by workflow resolution.

### F3 reference / quality workflow

For the current `run_3d_f3d_crop_validation.py` and
`report_3d_f3d_multicrop.py` crop pipeline, `resolve_workflow_options()` changes
the following effective values:

| Setting | Reference workflow | Quality workflow |
| --- | --- | --- |
| workflow label | `reference` | `quality` |
| default voter thinning | `reference` | `hybrid_v2` |
| default surface-support minimum fraction | `0.0` | `0.0` |
| default surface-support exponent | `0.0` | `0.0` |

An explicit voter-thinning or surface-support value is passed identically to
both branches and takes precedence over the workflow default. The scanner
backend is not selected by this resolver: both branches call
`FaultOrientScanner3.scan()`. `scanner_thin_mode`, scanner sigma, angle range,
edge cleanup, and the other scanner-side arguments are supplied independently
and remain unchanged when only `workflow_mode` changes.

The crop runner executes the scanner separately in each workflow branch; it is
not a shared-scan comparison. It produces `ft`, `fv`, and `fvt`-path outputs,
but it does not execute skinning. Consequently, the synthetic quality
workflow's quality-skinner, adaptive likelihood, occupancy, and boundary
fallback differences are outside this F3 crop comparison.

### Diagnostic workflow

`diagnostic` is not one of the four publication conditions. It is a synthetic
workflow based on reference settings that enables additional thinning
diagnostics. Treat it as an investigation aid, not as a primary performance
mode.

## 1.6 Publication comparison IDs

The public condition IDs are:

| ID | Scanner backend | Workflow | Intended interpretation |
| --- | --- | --- | --- |
| `RL-REF` | `reference-like` | `reference` | Reference-oriented end-to-end condition |
| `RL-QUAL` | `reference-like` | `quality` | Isolates downstream workflow changes relative to `RL-REF` |
| `Q-REF` | `quality` | `reference` | Isolates scanner backend changes relative to `RL-REF` |
| `Q-QUAL` | `quality` | `quality` | Combined quality scanner and quality workflow |
| `PUBLIC-REF` | N/A | N/A | F3 public comparison target, not a processing mode |

The intended contrasts are:

- `RL-REF` versus `RL-QUAL` measures the workflow effect at a fixed
  reference-like scanner backend.
- `RL-REF` versus `Q-REF` measures the scanner effect at a fixed reference
  workflow.
- All four processing conditions provide the scanner/workflow main effects
  and their interaction.
- `PUBLIC-REF` comparisons measure F3 public-output agreement. They are not
  accuracy claims.

`PUBLIC-REF` names the published `fl.dat`, `fv.dat`, and `fvt.dat` outputs as a
single comparison target. It is not a fifth algorithm and must not be encoded
as a workflow or scanner backend.

## 1.7 Current implementation support

The following table distinguishes existing paths from the use of the public
IDs in the different runners:

| Capability | Current support |
| --- | --- |
| Express a synthetic scanner/backend and workflow pairing independently | Yes. `SyntheticScannerConfig` and `resolve_workflow_settings()` are independent; the report configuration records both when scanner input is used. |
| Emit all four synthetic processing conditions from the basic synthetic-quality CLI in one report | No. `pyosv.cli.synthetic_quality` builds one selected configuration per invocation. |
| Run a separate synthetic mode-comparison API/CLI over canonical cells | The repository has a separate `synthetic_mode_comparison` implementation. It is not a cross-domain synthetic+F3 report and is not changed by this contract work. |
| `report_3d_f3d_multicrop.py --compare-workflows` conditions | Two branches: reference-like scanner behavior with the `reference` workflow and the `quality` workflow, corresponding to an RL-REF/RL-QUAL-style crop comparison. |
| Scanner execution in those F3 branches | The scanner is run independently in each branch; scanner output is not shared. |
| Scanner-side settings across those branches | The same scanner thinning mode, sigma, angle range, edge cleanup, and other scanner arguments are passed to both branches. |
| Quality scanner backend selectable through that F3 crop comparison path | No. The crop `run_pipeline()` calls `FaultOrientScanner3.scan()` and has no quality-backend selector. A separate F3 full-volume comparison package has its own backend matrix; it is not the crop `--compare-workflows` path. |
| Skinning in the current F3 crop pipeline | No. The crop pipeline ends at voter thinning and `fvt`; it does not produce skins. |
| One cross-domain synthetic/F3 integrated publication report | Not implemented by this contract work. The IDs define the conditions for future/report-specific use; they do not add a new runner or schema. |

The dedicated scanner-thinning policy comparison is a different experiment:
it holds the intended downstream workflow fixed and varies
`scanner_thin_mode`. A thinning ablation is likewise a stage-specific
experiment. Neither should be described as the generic workflow comparison or
as the canonical scanner-backend × workflow matrix.

## 1.8 Evaluation targets and interpretation

For synthetic data, the truth surface and truth orientation are independent
ground-truth sources. Scanner and downstream metrics may therefore be
interpreted against known synthetic truth, subject to the metric definition.

For F3, `fl.dat`, `fv.dat`, and `fvt.dat` are public workflow outputs. They are
not an independent geological ground truth. Use the following terms for F3
comparisons:

- `public-reference agreement`
- `output difference`
- `ridge displacement`
- `density/stability diagnostic`
- `visual geological review`

Avoid calling these F3 measures accuracy or correctness unless an independent
ground truth is introduced and documented. A mismatch with the public
reference alone does not establish a quality regression. Agreement with the
public reference alone does not establish a geological improvement.

## 1.9 Source-of-truth references

The implementation is authoritative for effective values. The relevant source
files and existing documentation are:

| Area | References |
| --- | --- |
| Scanner and sampling | [`src/pyosv/_orient3d/scanner.py`](../src/pyosv/_orient3d/scanner.py), [`src/pyosv/_orient3d/sampling.py`](../src/pyosv/_orient3d/sampling.py) |
| Synthetic configuration and resolution | [`config.py`](../src/pyosv/evaluation/synthetic_quality/config.py), [`profiles.py`](../src/pyosv/evaluation/synthetic_quality/profiles.py), [`application.py`](../src/pyosv/evaluation/synthetic_quality/application.py), [`scanner.py`](../src/pyosv/evaluation/synthetic_quality/scanner.py) |
| Voting and voter thinning | [`src/pyosv/voting3d.py`](../src/pyosv/voting3d.py) |
| Skinning | [`src/pyosv/_skinner/reference.py`](../src/pyosv/_skinner/reference.py), [`src/pyosv/_skinner/seeds.py`](../src/pyosv/_skinner/seeds.py) |
| F3 crop workflow | [`run_3d_f3d_crop_validation.py`](../examples/run_3d_f3d_crop_validation.py), [`report_3d_f3d_multicrop.py`](../examples/report_3d_f3d_multicrop.py), [`report_3d_f3d_scanner.py`](../examples/report_3d_f3d_scanner.py) |
| Related documentation | [`quality_mode.md`](quality_mode.md), [`f3d_validation.md`](f3d_validation.md), [`f3d_visual_diagnostics.md`](f3d_visual_diagnostics.md), [`orient3d.md`](orient3d.md), [`reference_like_thinning.md`](reference_like_thinning.md), [`reference_mapping_orient3d.md`](reference_mapping_orient3d.md) |

This document intentionally contains no source line numbers so that the
contract remains useful as implementation files evolve.
