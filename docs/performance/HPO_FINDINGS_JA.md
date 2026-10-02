> Historical investigation: results, package identities and command blocks below describe their original recorded snapshot. Commands are archival instructions, not the current checkout workflow. Use [current build instructions](../getting-started.md) and [benchmark layout](../../benchmarks/README.md) for this product tree. No new performance measurement is claimed.

# 実データHPO: 2026-10-02 の探索的所見

Python dev3 をインストールし、FCM / ExRCM / RMCM × Iris・Wine・WDBC、SOM-OLP × Iris・Wineの11課題を評価しました。
UbuKit TPE・Optuna 5.0.0 の既定TPE・randomを、各5探索seed・36試行枠で比較。最初の12点は共通のLatin hypercube観測です。
クラスタリングの選択目的は2初期化seedの平均ARI、SOMはk=10の(T+C)/2です。反復上限100、1 thread、各fitに2秒の共通制限を設けました。

- UbuKitの平均best observedはRMCM IrisとSOMで上、OptunaはExRCM 3データとFCM WDBCで上でした。残りは同点または微差を含み、一般的優位・有意差・SOTAを主張しません
- SOMの840件の物理評価のうち53件は2秒以内に完了しませんでした。未完了設定の品質は不明です。表現安全性のfallbackに時間がかかることが主要な制限です
- ExRCM WDBCの選択時ARI約0.696–0.698は、未使用の3初期化seedで約0.464–0.466へ低下しました。初期化への頑健性は課題です
- UbuKit提案APIは平均0.645ms、Optunaは1.177ms、randomは0.024msでした。Optunaのstudy/storageを含むAPI呼び出しの比較であり、TPE単体の速度比較でも総探索時間の比較でもありません

ARIにラベルを使うため、このクラスタリングの設定選択はラベル補助型です。ラベルを知らない教師なし設定選択ではありません。
未使用seed評価は同じデータの初期化検証で、未知データへの汎化検証ではありません。
共有CPU、5seed、手選択の探索空間という限定的な検証です。SOMの失敗も試行枠を消費し、等試行数は等wall-timeを意味しません。
元の実行結果を要約した記録であり、このGitHub同期では追加のベンチマークを実施していません。
