# Implementation Plan

`pyosv` will be implemented in small stages so each issue can validate one functional layer.

1. Repository scaffold and checks.
2. I/O.
3. Geometry and cell dataclasses.
4. Pythonic interpolation and smoothing adapters.
5. 2D dynamic programming path kernel.
6. 2D optimal path voting.
7. 2D thinning and practical-equivalence metrics.
8. 3D dynamic programming surface kernel.
9. 3D optimal surface voting.
10. Orientation scanners.
11. Optional acceleration.
12. Skinning.

## 3D Surface DP Notes

The package-wide 3D image convention remains `(n3, n2, n1)`, matching the
sample order used by DAT I/O and interpolation helpers.

The local dynamic-programming surface cost volume uses a different, local
coordinate system with shape `(nw, nv, nu)`. `u` is the fault-normal lag axis,
`v` is the fault-dip direction, and `w` is the fault-strike direction. The
surface returned by `find_surface_3d` has shape `(nw, nv)` and stores selected
`u` lag values.

Mines JTK recursive smoothing used by the reference implementation is
approximated with the existing Python smoothing helpers in `pyosv.filters`,
which wrap SciPy Gaussian filters. The target is practical behavior for
synthetic and interpretation workflows, not bit-exact Java output.
