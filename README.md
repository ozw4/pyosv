# pyosv

`pyosv` は、OSV（Optimal Surface Voting / Optimal Path Voting）系の断層強調・断層面抽出アルゴリズムを Python パッケージとして再実装するためのプロジェクトです。

本プロジェクトでは、既存の Java/Jython 実装を `reference_osv/` として参照しつつ、Python 実装は NumPy / SciPy / Numba を中心に構成します。Mines JTK への runtime dependency は持ち込みません。

## 目的

このリポジトリの目的は、`reference_osv` の OSV 機能を、実運用で扱いやすい Python パッケージ `pyosv` として段階的に実装することです。

主な対象は次の機能です。

- 2D Optimal Path Voting
- 3D Optimal Surface Voting
- 断層方位・断層属性を扱う幾何ユーティリティ
- Pythonic な補間・平滑化近似
- 2D / 3D thinning
- 実用一致メトリクス
- 後続段階での orientation scanner と fault skinning

## 基本方針

`pyosv` は `reference_osv` の逐語移植ではありません。

Java / Mines JTK 実装との bitwise equivalence は目標にしません。代わりに、断層リッジ位置、主要構造、値域、NaN/Inf の有無、合成データでの断層面復元などを基準にした「実用一致」を目標にします。

特に、次の Mines JTK 依存は Pythonic な近似に置換します。

- `SincInterpolator`
  → `scipy.ndimage.map_coordinates` などによる補間近似

- `RecursiveExponentialFilter`
  → `scipy.ndimage.gaussian_filter` / `gaussian_filter1d` などによる平滑化近似

- JTK の並列実行・浮動小数点加算順序
  → Python 側では決定的な逐次実装または chunk reduce による実装を優先

## 重要な配列規約

Java 実装では配列添字が `i1`, `i2`, `i3` の順で扱われますが、Python 実装では NumPy 配列として次の shape 規約を固定します。

```text
2D: array.shape == (n2, n1)
3D: array.shape == (n3, n2, n1)
```

データ型は原則として `np.float32` を使用します。

## `reference_osv` の扱い

`reference_osv/` は read-only bind mount として配置します。Git 管理対象には含めません。

想定される作業ディレクトリは次の形です。

```text
pyosv/
  README.md
  pyproject.toml
  AGENTS.md
  .gitignore

  reference_osv/        # read-only bind mount; not tracked by git

  vendor/
    issue_forge/        # symlink or bind mount; not tracked by git

  src/
    pyosv/
      __init__.py
      io.py
      geometry.py
      cells.py
      interp.py
      filters.py
      dp.py
      voting2d.py
      voting3d.py
      orient2d.py
      orient3d.py
      skin.py
      skinner.py
      metrics.py

  tests/
  examples/
  docs/
```

Docker などで作業する場合の bind mount 例です。

```bash
docker run --rm -it \
  --mount type=bind,src=/path/to/osv-master,dst=/workspace/reference_osv,readonly \
  --mount type=bind,src=/path/to/pyosv,dst=/workspace \
  -w /workspace \
  python:3.12 bash
```

ローカル作業でも、`reference_osv/` は変更しないでください。参照実装・fixture・メソッド対応確認のためだけに使います。

## 推奨 issue 分割

初期実装は、次の順序で進めます。

```text
1. repository scaffold / pyproject / checks / AGENTS.md
2. binary DAT I/O
3. geometry primitives and cell dataclasses
4. interpolation and smoothing adapters
5. 2D dynamic programming path kernel
6. 2D seed picking
7. 2D path voting core
8. practical-equivalence metrics
9. 2D thinning
10. 2D examples
11. 3D dynamic programming surface kernel
12. 3D seed picking and local UVW sampling
13. 3D surface voting core
14. 3D thinning
15. FaultOrientScanner2 approximation
16. FaultOrientScanner3 approximation
17. optional Numba acceleration
18. minimal FaultSkin / FaultSkinner
```

1 issue では 1 つの機能単位だけを実装します。大規模な横断リファクタ、API 変更、性能最適化を同じ issue に混ぜないでください。

## 実用一致ポリシー

`pyosv` では、Java 実装との完全一致ではなく、以下を重視します。

- shape 規約が一致していること
- 出力に NaN / Inf がないこと
- voting map の値域が妥当であること
- 合成データで断層リッジまたは断層面が期待位置に出ること
- 参照出力に対する相関・上位 percentile ridge overlap が大きく破綻しないこと
- 同じ Python 環境で決定的な出力が得られること

逆に、以下は合格条件にしません。

- Java / JTK との bit exact 一致
- JTK の境界条件の完全再現
- Java 並列実行時の浮動小数点加算順序の再現
- Sinc 補間の完全再現
- Recursive exponential filter の完全再現

## インストール

開発環境では editable install を使います。

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

想定する主要依存関係です。

```text
numpy
scipy
numba
pytest
ruff
```

## テストとチェック

通常のチェックは次で実行します。

```bash
python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
```

issue_forge 経由では `.issue_forge/checks/run_changed.sh` から同等のチェックを実行します。

```bash
./.issue_forge/checks/run_changed.sh
```

## 参照データ

`reference_osv/data/` 配下の `.dat` ファイルは、参照実装の入力・出力 fixture として使います。

基本的な読み込み規約は次の通りです。

```text
format: float32
endian: big-endian を基本とする
2D shape: (n2, n1)
3D shape: (n3, n2, n1)
```

Python 側では、shape を明示して読み込む API を提供します。

```python
from pyosv.io import read_dat

ft = read_dat("reference_osv/data/2d/f3d2d/ft.dat", shape=(n2, n1), dtype=">f4")
```

## 想定 API

2D voting の想定 API です。

```python
from pyosv.voting2d import OptimalPathVoter

voter = OptimalPathVoter(ru=15, rv=30)
voter.set_strain_max(0.25)
voter.set_path_smoothing(2.0)

fv, w1, w2 = voter.apply_voting(
    d=4,
    fm=0.3,
    ft=ft,
    pt=pt,
)
```

3D voting の想定 API です。

```python
from pyosv.voting3d import OptimalSurfaceVoter

voter = OptimalSurfaceVoter(ru=10, rv=20, rw=30)
voter.set_strain_max(0.25, 0.25)
voter.set_surface_smoothing(2.0, 2.0)

fv, vp, vt = voter.apply_voting(
    d=4,
    fm=0.3,
    ft=ft,
    pt=pt,
    tt=tt,
)
```

## 開発上の注意

`reference_osv/` は変更禁止です。必要な情報は読み取るだけにしてください。

Python 実装では、Java のクラス名・メソッド名を必要に応じて対応表に残しつつ、公開 API は Python らしい snake_case を基本にします。

性能最適化は、正しさと実用一致メトリクスが安定した後に行います。初期段階では、Numba 化よりも読みやすく検証しやすい実装を優先します。

大きな `.dat`, `.bin`, `.npy`, `.npz`, `.segy`, `.sgy` ファイルは原則として Git に含めません。小さなテスト fixture だけを `tests/fixtures/` に明示的に配置します。

## ライセンス

`reference_osv` のライセンスと、Python 再実装としての `pyosv` の配布ライセンスは別途確認・決定してください。

`reference_osv` に由来する実装を移植する場合は、元実装のライセンス条件を確認したうえで、README、ソースヘッダ、配布設定に反映してください。
