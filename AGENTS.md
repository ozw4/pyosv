# AGENTS.md

このリポジトリは `reference_osv` のOSV機能を Python package `pyosv` として再実装する。

## 基本方針

- `reference_osv/` は読取専用の参照実装として扱う。変更してはいけない。
- `reference_osv/` は bind mount として提供される外部参照であり、コミットしてはいけない。
- `vendor/issue_forge` は外部エンジンとして提供されるため、コミットしてはいけない。
- Python package 名は `pyosv` とする。
- 目標は Java/Mines JTK との bit exact ではなく、地震断層解釈で使える実用一致である。
- JVM、Jython、Mines JTK、Gradle への runtime dependency を追加してはいけない。
- Python実装は NumPy、SciPy を基本依存とし、Numba などの高速化は後続 issue で検討する。
- 配列shapeは 2D: `(n2, n1)`, 3D: `(n3, n2, n1)` に統一する。
- dtypeは原則 `np.float32` とする。
- `reference_osv` の `.dat` は基本 big-endian float32 として読む。
- `SincInterpolator` は `scipy.ndimage.map_coordinates` 等で近似する。
- `RecursiveExponentialFilter` は `scipy.ndimage.gaussian_filter1d` または明示的なseparable smoothingで近似する。
- 完全一致を前提にしたテストを書いてはいけない。実用一致メトリクスまたはPython実装の回帰テストを使う。
- 1 issue では1つの機能単位だけを実装する。大規模な横断リファクタを混ぜない。
- 公開API、shape規約、実用一致基準を変える場合は docs も同じPRで更新する。
- この bootstrap issue ではアルゴリズム実装を追加してはいけない。
