# UbuKit preview

現在のPython / JavaScriptパッケージのソースとドキュメントです。導入と最小例は[リポジトリのREADME](../README.md)を参照してください。

- [Python API](python/staging/README.md): `import ubukit`、クラスタリング、SOM、評価、探索
- [JavaScript API](javascript/package/README.md): ES Modules、Session、Worker、非同期実行
- [ビルド・テスト手順](REPRODUCE.md): ローカルwheel / sdist / tgzの作成と検証
- [数値上の制限](NUMERICAL_LIMITS.md): 極端な入力や浮動小数点の注意
- [性能とbackendの選択](EFFICIENCY.md): 測定条件、保持された既存経路、明示的な選択肢
- [Python旧版からの移行](NAMESPACE_MIGRATION.md): namespace、pickle、Numba cache

Pythonのソースは `python/staging`、JavaScriptは `javascript/package` にあります。どちらもprivate previewで、レジストリには未公開です。

検証の詳細は[VERIFICATION.md](VERIFICATION.md)、開発用WASM資料は[wasm-rebuild](javascript/wasm-rebuild/README.md)にあります。
