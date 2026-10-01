# 日本語の利用案内

インストール、最短例、配列形状、停止理由、メモリの注意、再測定コマンドは [README.md](README.md) にまとめています。

- 初めて使う: `python -m pip install .` → `from ubukit_rmcm import fit_rmcm`
- 速度と直接実装を比較する: 同じ `init` を渡し、`backend='numpy'`, `'csr'`, `'adjoint'` を比較する
- 同じデータ・δで繰り返す: `prepare_rmcm(...).fit(...)` で近傍と前計算を再利用する
- 大きいデータで R が不要: `return_memberships=False`。ただし δ グラフ自体は必要
- `cycle` / `max_iter`: 収束したという意味ではない
- RMCM2: この候補には含まれない
- 数値境界に近い入力: バックエンド間で微小な中心差からラベル経路が変わり得る。`THEORY.md` の実例を確認する

実測結果と配布検証は [REPORT.md](REPORT.md) に記載しています。
