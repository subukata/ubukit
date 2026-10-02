# サンプル

- `python/all_methods.py`: インストール済みwheelで公開APIの小さな例
- `python/som_variants.py`: SOM / BatchSOM
- `javascript/`: 既存のrealtime / RMCM例
- `browser/`: 既存のWorkerデモとブラウザテスト

```sh
.venv/bin/python -I -B examples/python/all_methods.py
node examples/browser/serve.js
```

ブラウザで http://localhost:8765/ を開きます。開発サーバーは127.0.0.1だけで待ち受けます。ブラウザ例は同梱ソース用です。配布物の検証は `tools/ci/` のインストール済みパッケージ用ゲートを使います。別途作成した公開Siteのソースはこのリポジトリに含まれるとは限らず、ここには追加していません。
