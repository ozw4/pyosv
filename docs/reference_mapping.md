# Reference Mapping

This table records the initial target modules for the Python reimplementation. It is a planning aid, not an implementation commitment for the bootstrap issue.

| Reference file | Python target | Priority |
| --- | --- | --- |
| `FaultGeometry.java` | `pyosv.geometry` | high |
| `FaultCell2.java` | `pyosv.cells.FaultCell2` | high |
| `FaultCell.java` | `pyosv.cells.FaultCell` | medium |
| `OptimalPathVoter.java` | `pyosv.voting2d` | highest |
| `OptimalSurfaceVoter.java` | `pyosv.voting3d` | high |
| `FaultOrientScanner2.java` | `pyosv.orient2d` | medium |
| `FaultOrientScanner3.java` | `pyosv.orient3d` | later |
| `FaultSkin.java` | `pyosv.skin` | later |
| `FaultSkinner.java` | `pyosv.skinner` | later |
| `SincInterpolator` use sites | `pyosv.interp` | approximate |
| `RecursiveExponentialFilter` use sites | `pyosv.filters` | approximate |
| `RecursiveGaussianFilterP.java` | `pyosv.filters` | approximate/later |

## Approximation policy

`pyosv` does not attempt bit-exact reproduction of Mines JTK interpolation or
recursive filters. Reference uses of `SincInterpolator` in scanner and voting
code are mapped to SciPy interpolation primitives, typically
`scipy.ndimage.map_coordinates`. Reference uses of `RecursiveExponentialFilter`
and `RecursiveGaussianFilterP` are mapped to SciPy Gaussian smoothing or explicit
separable smoothing in `pyosv.filters`.

`reference_osv/` remains a read-only bind mount used for comparison and
inspection only. It is not part of the Python package and must not be included in
distribution artifacts.
