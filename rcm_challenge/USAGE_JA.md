# ExRCM / RCM の独立実装

ExRCM を次の更新として実装しています。

- 距離 d は Euclidean distance（通常のユークリッド距離）
- over_u[c,i] = 1{ d[c,i]^p <= alpha^p * dmin[i]^p + beta^p }
- u[c,i] = over_u[c,i] / sum_c over_u[c,i]
- center[c] = sum_i u[c,i] * x[i] / sum_i u[c,i]
- p=1 が RCM。p=2 は二乗距離のまま判定できます
- beta は p にかかわらず「距離」の単位です

メンバーシップに追加の fuzzy exponent を掛けたり、一般的な rough c-means の lower/upper-region 重み付けに置き換えたりはしていません。

## 使い方

```python
from rough_cmeans import fit_exrcm

result = fit_exrcm(
    X, n_clusters=8,
    alpha=1.15, beta=0.5, p=2,
    seed=42, backend="auto",
)

centers = result.centers
u = result.memberships       # shape: クラスタ数 × データ数
print(result.stop_reason, result.n_iter)
```

`backend="auto"` は SciPy があれば SciPy、なければ NumPy を使います。`"numba"` は任意依存で、初回コンパイルがあります。`"naive"` は比較用の素朴な broadcast 実装です。

初期中心はデータのインデックスを重複なしでランダムサンプリングします。異なる実装の比較では seed だけでなく同一の `init` 配列を渡してください。空クラスタは前の中心を保持します。

`return_memberships=False` では最終の密な u / over_u を返しません。Numba 版は反復途中も距離・判定・集計をまとめて行い、密な membership 行列を作りません。ただし収束・周期検出用の bit-packed mask は保持します。

## 数値上の注意

- beta=0 なら p>0 を代数的に消去して比較します。p が極端に小さいときに全ての累乗値が 1 に丸められる問題を避けます
- 通常範囲では高速経路、極端な p や小さな比では log / log1p / logaddexp を使う安定経路に切り替えます
- 小さい比が 0 になった場合だけでなく subnormal の場合も保護しています
- 境界付近の再計算に使うガードは許容領域を広げる epsilon ではありません。判定そのものは <= のままです
- 距離の計算から x² の項を落としていません。このモデルでは落とすと別の判定になります
- 二乗距離が overflow / subnormal / underflow する極端な入力には、誤ったゼロ距離を返さず明示的な例外を出します。X と beta を同じ倍率でスケーリングしてください
- 演算順序が違う実装間の、全ての浮動小数点境界での bit 単位一致は保証しません

## 検証・計測

[README](README.md) に実行コマンドと詳細仕様、[性能要約](../docs/PERFORMANCE_JA.md) に比較条件があります。再計測スクリプトは初期中心・反復回数・各回の時間・mask 一致等を新しいファイルへ保存します。
Python と JavaScript の比較に使う小さなテストデータは [exrcm-python.json](../javascript/fixtures/exrcm-python.json) です。過去の全実験ログは利用に必要ありません。

本ディレクトリは独立した Python パッケージです。プロジェクト全体の公開ライセンス・配布名の統合はまだ確定していません。
