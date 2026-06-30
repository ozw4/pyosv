# Optional Visualization Helpers

`pyosv.viz` provides optional, static matplotlib-based helpers for writing PNG
diagnostics from NumPy volumes. The helpers are intended for validation and
inspection workflows, not for core OSV computation.

Install the visualization extra only when figures are needed:

```bash
python -m pip install -e ".[viz]"
```

For development with tests and visualization enabled:

```bash
python -m pip install -e ".[dev,viz]"
```

The core package does not import `matplotlib` at module import time. Helpers
that save figures call `require_matplotlib()` lazily and raise an `ImportError`
with the `pyosv[viz]` installation hint when matplotlib is unavailable.

## Shape Convention

Visualization helpers follow the project-wide array convention:

- 2D arrays: `(n2, n1)`
- 3D arrays: `(n3, n2, n1)`

Axis names map directly to 3D array dimensions:

- `i3` or `0`: slice/projection over `(n2, n1)`
- `i2` or `1`: slice/projection over `(n3, n1)`
- `i1` or `2`: slice/projection over `(n3, n2)`

`select_center_slices(shape)` returns deterministic center indices for all
three axes.

## Slice Panels

Use `slice_2d()` to extract a single 2D slice and `save_slice_panel()` to write
one row of normalized image panels. `save_volume_comparison_slices()` writes
reference, candidate, and absolute-difference panels for each axis with
deterministic filenames:

```text
<name>_i3_<index>.png
<name>_i2_<index>.png
<name>_i1_<index>.png
```

Example:

```python
from pyosv.viz import save_volume_comparison_slices

save_volume_comparison_slices(
    "outputs/diagnostics/fv",
    reference=fv_ref,
    candidate=fv_py,
    name="fv",
)
```

## Ridge Overlays

`ridge_mask()` selects sparse ridge samples by percentile. With the default
`positive_only=True`, zero and negative samples are not selected, so all-zero
volumes do not become all-ridge masks.

`save_ridge_overlay_slice()` writes one RGB ridge overlay, and
`save_buffered_ridge_overlay_slices()` writes one overlay per axis. Overlay
colors are defined by the helper docstring: reference-only, candidate-only,
exact overlap, and buffered matches are shown as distinct categories. Buffered
matches use binary dilation and are useful when thinned ridges are spatially
close but not exactly overlapping.

## MIP And Histogram Diagnostics

`maximum_intensity_projection()` returns one projection for an axis.
`save_mip_comparison()` writes reference, candidate, and absolute-difference
MIP panels for all three axes. `save_histogram_comparison()` writes an overlaid
value histogram for two same-shaped 3D volumes. `save_volume_diagnostics()`
writes both files with deterministic names:

```text
<name>_mip.png
<name>_hist.png
```

Generated diagnostics should be written under an output directory, not into
`reference_osv/` or any external reference-data root.
