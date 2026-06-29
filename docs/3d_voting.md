# 3D Voting Conventions

`pyosv` stores global 3D image volumes with shape `(n3, n2, n1)`. Sample
indices are addressed as `(i3, i2, i1)` in array indexing order, while vector
components follow the OSV component order `(x1, x2, x3)`.

Local 3D voting samples use a seed-centered `(w, v, u)` coordinate system with
array shape `(nw, nv, nu)`. The local axes are:

- `u`: fault-normal lag direction.
- `v`: fault-dip direction.
- `w`: fault-strike direction.

For orientation vectors, `u` is the fault normal, `v` is the dip vector, and
`w` is the strike vector. The component arrays for these vectors use
`(x1, x2, x3)` order even when they are sampled from global volumes stored as
`(n3, n2, n1)`.

`reference_osv/` is a read-only reference implementation. It is not a runtime
dependency of `pyosv`, and generated outputs should not be written under that
directory.
