# Data attribution

## Code and data rights

The PyOSV Q-QUAL PoC source code is licensed under the Common Public License
Version 1.0 (`CPL-1.0`). The raw F3 DAT volumes are not included in the public
Git repository or its Release assets.

## F3 dataset source

The F3 DAT volumes used to generate the compact publication were obtained from
the Google Drive distribution linked by the upstream `xinwucwp/osv`
repository.

- Upstream repository: `https://github.com/xinwucwp/osv`
- Upstream revision:
  `f4e2564fc27b9539edc4caff0944b1ddb94997b8`
- Google Drive file ID:
  `1InfMvCSZWdJclykiTBIXDgV7HYdBj5_K`
- Dataset ID: `f3d-official-v1`
- Dataset shape: `(420, 400, 100)`
- Storage dtype: big-endian float32 (`>f4`)

The upstream repository describes the F3 data as provided by the Dutch
Government through TNO and dGB Earth Sciences.

## Compact publication

The compact publication contains derived figures, agreement metrics,
machine-readable figure data, and provenance records.
It does not contain the raw seismic, planarity, or public-reference DAT volumes.

The F3 public reference is an agreement target and is not geological truth.

## Acknowledgment

We thank dGB Earth Sciences for making the data available as an OpendTect
project via their TerraNubis portal terranubis.com.
