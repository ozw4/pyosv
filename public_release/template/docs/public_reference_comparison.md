# PUBLIC-REF comparison and interpretation

## Evidence identity

The compact evidence has these verified identities:

- Archive SHA-256:
  `872fa183e1016b70ccd41449e689b545adb49e70cb29e563bf6f35133834c13d`
- Manifest schema: `pyosv.f3_compact_publication_manifest.v1`
- Publication ID:
  `c20a3a4195fb5598a9661d16cf368610ba7081c28ece2e63549706fec6a35322`
- Source F3 completion SHA-256:
  `3cc8818b27c9ea68d7fc4f5c9fc8d072aaaeb81cfd672c1c79d54c9fe8c1ae72`
- Dataset ID: `f3d-official-v1`
- Dataset shape and storage: `(420, 400, 100)`, `>f4`
- Publication artifacts: 6 figures and 6 figure-data tables

## Stage mapping

The compact publication compares the public reference with the corresponding
Q-QUAL lineage stage:

| PUBLIC-REF file | Q-QUAL stage | Meaning |
| --- | --- | --- |
| `fl.dat` | `ft` | quality-scanner fault likelihood |
| `fv.dat` | `fv` | voted likelihood |
| `fvt.dat` | `fvt` | voter-thinned ridge volume |

Q-QUAL combines the quality scanner, quality voter thinning, and quality
skinning. The `ft` and `fv` arrays are in the Q-QUAL lineage, but
quality-workflow-specific processing has not acted at those stages. The voter
thinning difference appears at `fvt`. Skinning follows `fvt` and is not one of
the compact summary stages.

## Selected sections

Sections are selected from positive PUBLIC-REF `fvt.dat` values using the
`public_fvt_positive_p99_peak_per_equal_bin` policy. Each axis is divided into
four equal bins and one ridge-count peak is selected per bin. All three stages
share the same sections.

- Time slices: `i1=24` (bin 0, score 127), `i1=49` (bin 1, score 282),
  `i1=74` (bin 2, score 764), and `i1=80` (bin 3, score 1047).
- Inline sections: `i3=101` (bin 0, score 196), `i3=105` (bin 1,
  score 182), `i3=229` (bin 2, score 103), and `i3=362` (bin 3,
  score 123).

The six atlases cover time slices and inline sections for each of `ft`, `fv`,
and `fvt`. Inline panels use crossline (`i2`) horizontally and time (`i1`)
vertically.

## Summary table

These values are transcribed from `f3_q_qual_vs_public_ref_summary.csv` in the
compact archive.

| stage | normalized correlation | mean absolute difference | nonzero fraction ratio | buffered F1 | candidate to reference p95 voxel | reference to candidate p95 voxel |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ft | 0.7881547825676648 | 0.265217443209895 | 1.0 | 0.13669877289423454 | 79.12648102879339 | 27.386127875258307 |
| fv | 0.861874453396084 | 0.09917782813295367 | 0.9479643347246597 | 0.8809190713605013 | 3.7416573867739413 | 6.782329983125268 |
| fvt | 0.5816100600601007 | 0.05416013486248344 | 1.0570606359678965 | 0.5833861441997502 | 45.20287491912169 | 14.422205101855956 |

The metrics mean:

- **Normalized correlation:** normalized full-volume correlation.
- **Mean absolute difference:** full-volume mean absolute attribute difference.
- **Nonzero fraction ratio:** Q-QUAL nonzero fraction divided by the
  PUBLIC-REF nonzero fraction.
- **Buffered F1:** F1 for positive-p99 ridges after a two-voxel spatial buffer.
- **Directional p95 voxel distances:** the 95th-percentile distance from a
  positive-p99 ridge in the named source to the nearest ridge in the other
  source.

The table measures agreement. It is not a significance test or a winner
determination.

## Visualization contract

Source thresholds are positive-value 99th percentiles recorded separately for
PUBLIC-REF and Q-QUAL. Selection and metrics use their specified source
thresholds. Display overlays apply a separate threshold ratio of `0.5` for
`ft`, `fv`, and `fvt`; this display-only policy does not change the metrics.

Signed amplitude from `xs.dat` is clipped at its 99th percentile and shown in gray.
Attribute overlays use `Reds`, alpha range `0.12` to `0.85`, alpha gamma `2.0`,
and nearest-neighbor interpolation. Signed Q-QUAL minus PUBLIC-REF differences
use `coolwarm` with a 99th-percentile symmetric limit.

Only `fvt` overlays receive a fixed one-pixel red cross halo with alpha `0.5`.
The halo is a display aid and is excluded from section selection, metrics, and
the summary table.

## Interpretation boundary

PUBLIC-REF is an agreement target, not geological truth. The reported evidence
does not establish geological accuracy, statistical significance, or a
preferred processing condition. See [data attribution](../DATA_ATTRIBUTION.md)
for the data-rights boundary and [reproducibility](reproducibility.md) for the
public reproduction boundary.
