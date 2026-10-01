# PythonとJavaScriptの同一条件比較

2026-10-01、Linux x86-64、Node.js 24/Python 3.12。1 CPU/1スレッド、同じfloat64データ・初期値、6反復、5回ウォームアップ後の5回中央値。言語一般の比較ではなく具体的なUbuKit実装の比較です。

| 手法 | JavaScript ms | Python標準構成 ms | Python別構成 ms |
|---|---:|---:|---:|
| k-means | 33.22 | 41.17 | 4.29 |
| FCM | 67.31 | 16.60 | 20.96 |
| RCM | 47.06 | 23.31 | 17.71 |
| ExRCM | 46.32 | 22.26 | 17.27 |
| SOM-OLP | 36.22 | 8.25 | 6.40 |
| trustworthiness + continuity | 293.61 | 212.01 | 80.95 |

クラスタリングはN=6000,D=32,K=16。SOMはN=2000,D=16,36ノードで共通の初期W/Pを与え、初期化時間を両方から除外。近傍評価はN=1800,D=16→2,k=[5,15]。公開APIの検証・バッファ確保・最終出力処理を含み、読み込み・入力作成・Worker転送は除外しています。

Python標準構成はk-meansがnumpy（sklearn finalizer）、FCM/RCM/ExRCMがscipy、SOMがcdist_optimized、近傍評価がsqrt_numpy。別構成は主にnumba、SOMのみgemm_guardedです。FCMのNumbaはこの条件ではSciPyより遅いため、別構成が常に最速ではありません。FCM比較にO(NK²)基準は使っていません。

12ケース・32バックエンド比較でラベル・マスク・反復数・近傍ペナルティが一致し、浮動小数値は検証許容誤差内でした。近傍評価は同距離のない入力を使用。Pythonは全距離行列、JSは行単位処理のため、メモリ使用量は同一ではありません。ブラウザ実測ではありません。

小規模条件ではJSが標準Pythonのk-means/RCM/ExRCM/近傍評価を上回る一方、選んだ別構成のPythonはいずれもJSより速い結果でした。初回実行は遅延importとNumba JITの影響が大きく、ウォーム値と混同しないでください。詳細な再現用データと記録は別途保存した比較ZIPに収録しています。
