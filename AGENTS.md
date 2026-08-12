# AGENTS.md

このリポジトリは `reference_osv` の OSV 機能を Python package `pyosv` として再実装する。

## 基本方針

- `reference_osv/` は読取専用の参照実装として扱う。変更してはいけない。
- `reference_osv/` は bind mount として提供される外部参照であり、コミットしてはいけない。
- `vendor/issue_forge` は外部エンジンとして提供されるため、コミットしてはいけない。
- Python package 名は `pyosv` とする。
- 目標は Java/Mines JTK との bit exact ではなく、地震断層解釈で使える実用一致である。
- JVM、Jython、Mines JTK、Gradle への runtime dependency を追加してはいけない。
- Python 実装は NumPy、SciPy を基本依存とし、Numba は optional acceleration として扱う。
- 配列 shape は 2D: `(n2, n1)`、3D: `(n3, n2, n1)` に統一する。
- dtype は原則 `np.float32` とする。
- `reference_osv` の `.dat` は基本 big-endian float32 として読む。
- `SincInterpolator` は `scipy.ndimage.map_coordinates` 等で近似する。
- `RecursiveExponentialFilter` は `scipy.ndimage.gaussian_filter1d` または明示的な separable smoothing で近似する。
- 完全一致を前提にしたテストを書いてはいけない。実用一致メトリクスまたは Python 実装の回帰テストを使う。
- 変更は一つの機能単位に限定し、大規模な横断リファクタを混ぜない。
- 公開 API、shape 規約、実用一致基準を変更する場合は、関連する文書も同じ変更で更新する。
- orientation、voting、skinning に加え、PyOSV は PyOSV の `FaultSkin` または同等の型付き配列契約を入力とする純粋な数値後処理を所有できる。
- 数値後処理は NumPy、SciPy、および optional Numba だけで動作しなければならない。Atlas、viewer、artifact、I/O、path、manifest、checksum、job 管理を PyOSV に取り込んではいけない。
- `reference_osv` に対応物がない PyOSV native extension は許可する。ただし reference-compatible 機能と native extension は文書上で明確に区別する。
- fault-warping のような native extension に対し、reference implementation との同一性を要件として偽ってはならない。
- native extension は `reference_osv/` の read-only、非 vendoring、非 runtime dependency の方針を変更しない。
- 正式な NumPy 対応範囲は NumPy 1.x（依存指定 `numpy<2`）とする。NumPy 2 の差異を吸収するために fixture を更新したり、既存回帰テストを緩和したりしてはいけない。

## コードコメント

- コードコメントには、コードから復元できない非自明な WHY だけを書く。
- コメントで説明してよいのは、隠れた制約、workaround を採用する理由、読み手が予想しにくい挙動である。
- コードを読めば分かる WHAT を書いてはいけない。
- 変更履歴や旧実装との差分をコメントに書いてはいけない。
- issue、PR、タスク ID をコメントに記載してはいけない。
- docstring は現在の API 契約、入力、出力、例外、数値的制約を記述し、実装経緯や変更履歴を記述しない。

## README と docs

- README と `docs/` は、現在有効な仕様だけを示すスナップショットとして保つ。
- 現在の公開 API、デフォルト、対応範囲、制約、実行方法、検証方法を直接記述する。
- issue、PR、タスク ID、実装計画、進行状況を記載してはいけない。
- 変更経緯、導入経緯、過去のデフォルト、migration history を記載してはいけない。
- 過去の candidate、実験の試行錯誤、個別実行の pass/fail や hash を恒久文書へ転記してはいけない。
- 互換性を記述する必要がある場合は、現在受理する形式や現在提供する挙動だけを記述し、それらが導入された経緯は記述しない。
