# PyOSV Q-QUAL PoC

PyOSV Q-QUAL PoC is a Python proof of concept for detecting and grouping
candidate faults from prepared 3-D seismic-derived attribute volumes.

It runs one fixed, reproducible Q-QUAL workflow and produces scanner
likelihood, voted likelihood, thinned fault ridges, and fault skins.

The optimal surface voting method was introduced by Xinming Wu and Sergey
Fomel. The upstream implementation used as the implementation reference is
[`xinwucwp/osv`](https://github.com/xinwucwp/osv).

The optimal surface voting method and its original implementation are upstream
work; PyOSV provides an independent Python reimplementation and
reproducibility-oriented packaging of the relevant 3-D workflow. It is not a
line-by-line port, is not bit-exact, and is not an API-compatible replacement
for the upstream Java/Jython code.

![F3 inline sections comparing PUBLIC-REF, the Q-QUAL lineage, and their signed difference](docs/images/f3_fvt_inline_sections.png)

*F3 compact-publication example at the FVT stage, comparing PUBLIC-REF, the
Q-QUAL lineage, and their signed difference. PUBLIC-REF is an agreement target,
not geological truth.*

The canonical source repository is
<https://github.com/ozw4/pyosv-qqual-poc>.

## Scope

The distribution contains the Q-QUAL runtime, its required PyOSV dependency
closure, the compact-publication validator, examples, and focused tests. The
fixed runtime uses the quality scanner, quality voter thinning, and quality
skinning. The public CLI runs one fixed Q-QUAL configuration and does not
expose internal comparison modes. It does not include raw F3 DAT volumes, the
internal comparison runner, the full Synthetic evaluation, or the full
publication generator.

## Relationship to upstream OSV

Optimal Surface Voting is not an invention of PyOSV. The method is due to
Xinming Wu and Sergey Fomel, and the implementation reference for this project
is the upstream [`xinwucwp/osv`](https://github.com/xinwucwp/osv) repository.
PyOSV independently reimplements the relevant 3-D processing path in Python;
it does not claim line-by-line, bit-exact, or API compatibility with upstream.

PyOSV's contribution is the reproducibility-oriented Python PoC around that
processing path:

- a numerical implementation using NumPy, SciPy, and optional Numba;
- the fixed public Q-QUAL profile;
- an in-memory API and command-line interface;
- a hashed run-output contract;
- source and artifact provenance; and
- standalone compact-publication validation.

### Main differences

| Aspect | Upstream OSV | PyOSV Q-QUAL PoC |
| --- | --- | --- |
| Origin | Original optimal surface voting method and implementation | Independent Python reimplementation of the relevant 3-D processing |
| Runtime | Java/Jython research code built with Gradle and using Mines JTK | Python 3.10+, NumPy, SciPy, and optional Numba; no JVM, Jython, Mines JTK, or Gradle runtime dependency |
| Scope | Research-oriented 2-D and 3-D voting, scanning, thinning, and fault-surface workflows | One fixed public 3-D Q-QUAL workflow |
| Public configuration | Research programs and examples define their processing parameters | One fixed resolved profile; internal experimental and comparison modes are not part of the public runtime |
| Input | Inputs prepared by upstream research programs and examples | A prepared scanner-input attribute volume; the CLI does not convert raw seismic amplitude into scanner input |
| Outputs | Research voting maps and fault-surface products | `ft.dat`, `fv.dat`, `fvt.dat`, `skin_mask.dat`, `skins.json`, and `run.json` |
| Numerical equivalence | Implementation reference | Practical agreement contract; no bit-exact or API-compatibility claim |
| Reproducibility | Source and research examples | Fixed profile, source identity, artifact SHA-256 records, and a standalone publication validator |

### Upstream method

- Xinming Wu and Sergey Fomel, “Automatic fault interpretation with optimal
  surface voting,” *GEOPHYSICS* 83(5), O67–O82 (2018),
  [doi:10.1190/GEO2018-0115.1](https://doi.org/10.1190/GEO2018-0115.1).
- Upstream repository: [`xinwucwp/osv`](https://github.com/xinwucwp/osv).

## Installation

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Optional Numba acceleration can be installed with `python -m pip install
".[accel]"`.

## Quick start with a prepared input

The CLI expects a prepared scanner-input attribute volume such as `ep.dat`. It
does not convert raw seismic amplitude into the scanner input.

For the F3 PoC data source, dataset identity, and attribution, see
[DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). Raw F3 DAT volumes are not included
in this repository or its Release assets.

Prepare a regular, non-symlink, C-order DAT file containing big-endian float32
samples. Supply its shape in `(n3, n2, n1)` order and choose an output directory
that does not exist:

```bash
pyosv-qqual3d \
  --input /path/to/ep.dat \
  --shape 420,400,100 \
  --output-dir qqual-output \
  --pretty
```

For the F3 PoC files, `(n3, n2, n1)` maps to:

- `n3 = 420`: inline array indices
- `n2 = 400`: crossline array indices
- `n1 = 100`: time-sample array indices

`n1` is the fastest-varying C-order axis. These are array-index dimensions; the
package does not convert them to physical inline, crossline, or time
coordinates.

For other datasets, the physical meaning of each axis is defined by how the
input volume was prepared.

The command prints the new output directory after a successful atomic write.
The input file is not modified.

To validate an extracted compact publication:

```bash
pyosv-validate-compact /path/to/extracted/compact-publication
```

The [v0.1.0-poc Release](https://github.com/ozw4/pyosv-qqual-poc/releases/tag/v0.1.0-poc)
provides `pyosv-qqual-poc-v0.1.0-compact-publication.tar.gz` and its SHA-256
file. The archive is not part of the Git tree.

## Output bundle

The default Q-QUAL run writes:

- `run.json`: input identity, resolved fixed profile, software versions, and
  output size and SHA-256 records.
- `ft.dat`: quality-scanner fault likelihood.
- `fv.dat`: voted likelihood.
- `fvt.dat`: voter-thinned ridge volume.
- `skin_mask.dat`: mask of cells in the returned skins.
- `skins.json`: serialized fault-skin cells.

DAT outputs use big-endian float32 storage and the input shape. See the
[operation manual](docs/manual.md) for the complete file contract.

## Compact publication

The compact publication contains six atlases, their six machine-readable data
tables, a summary table, provenance, and a manifest. It reports agreement with
the F3 public reference; that reference is not geological truth. See
[PUBLIC-REF comparison and interpretation](docs/public_reference_comparison.md)
and [reproducibility boundaries](docs/reproducibility.md).

## Limitations

This is a proof of concept. Runtime and memory use scale with the input volume,
and results require domain interpretation. Only the fixed Q-QUAL workflow is a
public runtime. Users must supply input data they are authorized to use.

## License, attribution, and citation

The source code is provided under the Common Public License Version 1.0
(`CPL-1.0`); see [LICENSE](LICENSE). The CPL-1.0 license matches the upstream OSV
project whose processing structures informed and were adapted by PyOSV; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Data rights are addressed
separately in [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). Citation metadata is
available in [CITATION.cff](CITATION.cff).
