# Practical Equivalence Policy

pyosv は reference_osv との bitwise equivalence を目標にしない。

## 近似対象

- Mines JTK SincInterpolator
- Mines JTK RecursiveExponentialFilter
- JTK parallel execution behavior
- floating point accumulation order

## 固定するもの

- input/output shape
- value range normalization
- angle convention
- seed threshold semantics
- no NaN/Inf
- deterministic output under the same Python version and dependency set

## 評価するもの

- normalized correlation with reference maps
- top percentile ridge overlap
- synthetic fault localization
- visual sanity check for examples

## 評価しないもの

- per-sample bit exactness
- JTK boundary behavior exactness
- Java parallel accumulation order exactness
