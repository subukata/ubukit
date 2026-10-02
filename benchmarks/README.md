# Benchmarks

`python/smoke/` は小規模な探索・品質smoke、`python/efficiency/` はSOMと既存失敗条件の再現、`javascript/` はJavaScript実装の比較スクリプトです。

先にローカル配布物をビルドしてインストールしてください。JavaScriptスクリプトは `javascript/consumer/node_modules/ubukit-js` のインストール済み配布物を読みます。一部の比較は `javascript/validation/` の固定参照実装も使います。結果は手動実行時に生成され、Gitには追加しません。

古いdev3インストールだけを対象にするドライバは完全保存版にあります。過去の時間・メモリ結果は履歴であり、現行ソースの再測定結果ではありません。CIはベンチマークを自動実行しません。追加の大規模計測・GPU計測には別途実行判断が必要です。
