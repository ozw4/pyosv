# Controlled Synthetic Quality

Controlled synthetic cases measure truth quality, not agreement with the
reference implementation. They are for checking whether a known fault geometry
is recovered by the Python workflow without scanner or external-data
confounds.

These validation modes answer different questions:

```text
reference comparison: is the implementation close to the original?
controlled synthetic: is the result correct against known truth?
F3 visual/multicrop: does the workflow avoid obvious failures on real data?
```

## Current PR1 Scope

PR1 includes:

- `SyntheticPlaneSpec`
- `Synthetic3DCase`
- `generate_single_plane_case`
- `make_single_vertical_plane_case`
- `ft` / `pt` / `tt` oracle attributes
- top-k / truth-count masks
- buffered surface overlap
- surface distance metrics
- masked orientation error
- minimal oracle pipeline smoke test

PR1 does not include:

- `report_3d_synthetic_quality.py`
- extended cases: dipping, curved, crossing, boundary, weak noisy
- skin topology metrics
- synthetic seismic generation
- scanner-inclusive synthetic path
- FaultSeg3D loader

## Shape And Convention

Controlled synthetic 3D arrays use the repository 3D shape convention:
`(n3, n2, n1)`.

Coordinates are expressed as `(x1, x2, x3)`, matching the OSV geometry helpers.
Orientation follows the same convention as `pyosv.geometry`, `FaultCell`, and
`OptimalSurfaceVoter`.

The public API is exposed through module imports such as `pyosv.synthetic3d`
and `pyosv.synthetic_metrics`; these helpers are not re-exported from
`pyosv.__init__`.

## Minimal Usage

```python
from pyosv.synthetic3d import make_single_vertical_plane_case
from pyosv.synthetic_metrics import top_truth_count_mask, buffered_surface_overlap

case = make_single_vertical_plane_case(shape=(33, 33, 33))
mask = top_truth_count_mask(case.ft_oracle, case.truth_fault_mask)
metrics = buffered_surface_overlap(mask, case.truth_fault_mask, radius=2.0)
```

## Test Commands

```bash
PYTHONPATH=src python -m pytest -q tests/test_synthetic3d.py tests/test_synthetic_metrics.py tests/test_synthetic_oracle_pipeline.py
```

The broader PR1 acceptance checks are:

```bash
PYTHONPATH=src python -m pytest -q tests/test_voting3d.py
PYTHONPATH=src python -m ruff check src tests
PYTHONPATH=src python -m ruff format --check src tests
```
