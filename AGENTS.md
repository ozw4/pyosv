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
- orientation、voting、skinningに加え、PyOSV は PyOSV の `FaultSkin` または同等の型付き配列契約を入力とする純粋な数値後処理を所有できる。
- 数値後処理は NumPy、SciPy、および optional Numba だけで動作しなければならない。Atlas、viewer、artifact、I/O、path、manifest、checksum、job管理を PyOSV に取り込んではいけない。
- `reference_osv` に対応物がない PyOSV native extension は許可する。ただし reference-compatible 機能と native extension は文書上で明確に区別する。
- fault-warping のような native extension に対し、reference implementation との同一性を要件として偽ってはならない。
- この拡張は `reference_osv/` の read-only、非 vendoring、非 runtime dependency の方針を一切変更しない。
- 正式な NumPy 対応範囲は NumPy 1.x（依存指定 `numpy<2`）とする。NumPy 2 対応、fixture の更新、または既存回帰テストの緩和は別 issue で扱う。
