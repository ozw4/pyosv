# DAT I/O

`pyosv.io` provides small helpers for the raw binary `.dat` files used by the
reference OSV data. The helpers do not parse headers; callers provide the target
shape and endian convention explicitly or through reference dataset metadata.

## Shape and dtype conventions

- 2D arrays use shape `(n2, n1)`.
- 3D arrays use shape `(n3, n2, n1)`.
- The default dtype is `np.float32`.
- `reference_osv` `.dat` files are treated as big-endian `float32` by default.

## Reading

`read_dat(path, shape, *, endian="big", dtype=np.float32)` reads raw binary
scalar values, validates that `shape` is a non-empty tuple of positive integers,
checks the file size, reshapes the values in C order, and returns a
C-contiguous array with native-byte-order dtype.

Example:

```python
from pyosv.io import read_dat
from pyosv.reference import REFERENCE_DATASETS_2D, resolve_reference_file

dataset = REFERENCE_DATASETS_2D["f3d2d"]
path = resolve_reference_file(dataset, "ft.dat")
ft = read_dat(path, dataset.shape, endian=dataset.endian)
```

If the file size does not exactly match `prod(shape) * dtype.itemsize`,
`read_dat` raises `ValueError` and includes the expected and actual byte counts.
Invalid shapes also raise `ValueError`.

## Memory mapping and region reads

`open_dat_memmap(path, shape, *, endian="big", dtype=np.float32)` validates the
shape and exact file size before returning a read-only `np.memmap`. The mapping
keeps the storage byte order, so callers that use it directly must account for
the requested endian. It does not copy the full volume into native byte order.
The mapping remains usable while the memmap object (or a view derived from it)
is alive; callers should not replace or truncate the backing file during that
lifetime.

`read_dat_region(path, shape, region, *, endian="big", dtype=np.float32)` maps
the file and copies only `region` into an independent, writeable,
C-contiguous array with native byte order. `region` follows array axis order:
`(slice_i2, slice_i1)` for 2D and `(slice_i3, slice_i2, slice_i1)` for 3D.

The region must be a tuple containing exactly one `slice` per axis. Slice steps
may only be `None` or `1`. Open-ended bounds are normalized to the axis
boundaries, and negative bounds are normalized relative to the axis size.
After normalization, every axis must satisfy `0 <= start < stop <= size`;
out-of-range and empty selections raise `ValueError` rather than being clipped.

Example:

```python
from pyosv.io import open_dat_memmap, read_dat_region

volume = open_dat_memmap("volume.dat", (401, 601, 801))
preview = volume[200, ::20, ::20]  # a view backed by the read-only mapping

crop = read_dat_region(
    "volume.dat",
    (401, 601, 801),
    (slice(100, 180), slice(220, 340), slice(300, 460)),
)
crop[0, 0, 0] = 0.0  # independent from volume.dat
```

## Writing

`write_dat(path, array, *, endian="big", dtype=np.float32, create_parents=True)`
writes an array as raw binary scalar values in C order. It converts the output
to the requested storage dtype and endian, creates parent directories by
default, writes the file with `numpy.ndarray.tofile`, and returns the written
`Path`.

## Endian values

Accepted endian values are:

- `"big"` or `">"` for big-endian storage
- `"little"` or `"<"` for little-endian storage

Unknown endian values raise `ValueError`.

## Reference data policy

`reference_osv/` is a read-only bind mount and is not committed. Do not add
reference binary `.dat` files or generated fixtures to this repository.

By default, reference paths resolve under `./reference_osv`. Set
`PYOSV_REFERENCE_OSV=/absolute/path/to/osv-master` when the reference checkout
or bind mount is elsewhere.

Optional smoke tests for reference fixture I/O skip when the reference root does
not exist. Individual cases also skip when a required reference `.dat` file is
missing from an otherwise available mount.
