# UbuKit

k-means、SOM-OLP、Trustworthiness / Continuity の計算を対象とした研究用高速化パッケージです。
保存済みのコード・テスト・測定記録を、UbuKit の初期版として登録しています。

この初期登録では、既存の Python import 名 `portable_accel`、配布名
`portable-accel-restart`、バージョン `0.2.0a2` を維持しています。
数値コードや過去の測定値は変更していません。

## はじめに

- [日本語の使い方](restart_v2/USAGE_JA.md)
- [API・数値計算の契約と制約](restart_v2/README.md)
- [結果と根拠の索引](restart_v2/RESULT_INDEX.md)
- [最終結果レポート PDF](restart_v2/evidence/report_stable/restart_v2_final_report_ja.pdf)
- [パッケージソース](restart_v2/portable_accel/)
- [統合テスト](restart_v2/tests/)
- [測定・配布検証の記録](restart_v2/evidence/)

## インストールとテスト

```sh
cd restart_v2
python -m pip install .
python -m unittest discover -s tests -v
```

Numba を使う追加の経路を有効にする場合:

```sh
python -m pip install '.[numba]'
```

[requirements-foundation.txt](restart_v2/requirements-foundation.txt) と
[requirements-tested.txt](restart_v2/requirements-tested.txt) に、記録時の依存バージョンがあります。

## 記録の範囲

- 実機での動作・速度検証は Linux x86-64 / Python 3.12 の記録環境のみです。Windows、macOS、ARM、本番運用での信頼性は未検証です
- 保存済みの a2 統合テストは 14 件成功。独立した no-JIT 配布検証は 12 件成功・Numba 専用 2 件スキップです。この登録作業で性能測定やテストを再実行したものではありません
- 各測定には固有のデータ、スレッド数、初期化、計時範囲があります。異なる条件の倍率を合算したり、過去の候補測定を a2 配布版の新規測定として読み替えたりしないでください
- `evidence/measured_public_a1/` と `evidence/distribution/` は過去の a1 スナップショットです。a2 の配布検証は `evidence/distribution_a2/` に分けて保存しています
- 実験記録に含まれる絶対パスは測定時の環境を示します。実験用スクリプトの再実行には、データの再生成やパスの調整が必要な場合があります

## 保存元と整合性

`restart_v2/` の 303 ファイルは、最終チェックポイントの内容をパス・バイト列ともに維持しています。
元の文書にある「GitHub push は未実施」などの記述は、チェックポイント作成時点の履歴です。
今回の初期登録は、この非公開リポジトリへの保存です。

- 保存元: `restart_v2_checkpoint.zip`（748,902 bytes）
- SHA-256: `d04316dd1bb513416aae72a78a95b6230b319a09d24f6c7259194550d91aa859`
- [チェックポイント内の SHA-256 一覧](restart_v2/CHECKPOINT_MANIFEST.json)（マニフェスト自身を除く 302 ファイル）

データセット本体、仮想環境、ネイティブビルド生成物、JIT キャッシュは含めていません。
配布検証に用いた a1 / a2 の pure-Python wheel と最終 PDF は、検証対象の成果物として保存しています。

## ライセンス

プロジェクト全体の公開ライセンスは未選定です。
第三者由来のコードの表示は [NOTICE.txt](restart_v2/NOTICE.txt) と
[LICENSE-SOM.txt](restart_v2/LICENSE-SOM.txt) に保持しています。
これらの表示は、プロジェクト全体への新たなライセンス付与を意味しません。
