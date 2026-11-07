# 相同實際90%剪枝率：importance criterion 公平比較

啟動日期：2026-10-27。本批專門隔離 importance criterion 的影響。

## 設定

- Dataset：CIFAR-10，固定 45000 / 5000 / 10000 train / validation / test split。
- Models：ResNet34、ResNet50、GoogLeNet、DenseNet121、MobileNetV2。
- Seeds：0、1、2；各方法從同 model/seed 的固定100 epochs Dense checkpoint 分支。
- Tm=8、連續 filter 分組、global_cluster sparsity reference、包含1×1與depthwise convolution。
- Target=0.9；記錄實際 sparsity，並逐 seed 驗證與 l1_sum 基準相同（容許1e-6比例的浮點/計數差距）。
- 無敏感層保護、無 early-layer protection、無全層或敏感層 cap。
- Fine-tune：10 epochs、SGD、LR=0.001、momentum=0.9、weight decay=0.0005、batch=128。
- AMP float16、固定 RNG seed、deterministic CUDA 設定與舊正式實驗一致。
- 每個方法 fine-tune 前重設相同 RNG，沿用原主程式流程。
- GPU1，預設並行3個任務；每任務 CPU BLAS/OMP threads=4、DataLoader workers=2。

## 方法與工作量

`l1_sum` 的15組有效正式基準直接沿用；新增 `max_abs`、`top2_mean_abs`、`top4_mean_abs`，共5模型×3seeds×3criteria=45個十輪fine-tune任務。

舊 l1_sum 基準來源是 `../fixed_checkpoint_all_ratio_summary/all_results_combined.csv` 中 Tm8 / ratio0.9 / ratio_source=global_cluster 的 rows。啟動前交叉核對逐次原始CSV、checkpoint metadata與SHA-256。

新增方法使用 `--cross-only`，略過重複訓練 Dense extra / Global，但不改 cross-filter 分支的固定起點與 fine-tune RNG。原始新CSV的 Dense extra 欄位因此為空；合併表才以同 model/seed/checkpoint 的既有 Dense extra 補入，並在 source_file 和 reused_baseline 保留來源。

## 檔案

- `experiment_manifest.json`：固定參數、所有 commands、環境版本、checkpoint與核心程式SHA-256。
- `source_snapshot/`：實際使用的核心程式與排程腳本快照。
- `reused_l1_baselines.csv`：15組既有 l1_sum 原始基準。
- `per_run_csv/`：45個新增方法的逐次原始結果；完成一個任務才會有完整CSV。
- `logs/`：各任務訓練紀錄、排程紀錄與每5分鐘GPU資源快照。
- `status.json`：目前active PID、完成數、排隊數與失敗任務。
- `all_results_combined.csv`：已完成結果加既有 l1_sum，逐 seed 配對。
- `paired_vs_l1.csv`：各方法對同 seed l1_sum 的FT accuracy提升，正值代表改善。
- `criterion_summary.csv`：三seed mean / sample std / sparsity / MAC reduction；complete_3_seeds=false 的資料不可當正式三seed結論。
- `accuracy_loss_comparison.png`：自動更新精度損失比較圖；未滿三seed的群組淡色呈現。

圖上的accuracy loss誤差棒為逐seed loss的sample std；它不一定等於FT accuracy的sample std。

## 啟動、恢復與更新

從專案根目錄執行：

```bash
python scripts/run_fair_criterion_r0p9.py --prepare
python scripts/run_fair_criterion_r0p9.py --run --gpu 1 --max-jobs 3
python scripts/run_fair_criterion_r0p9.py --summarize
```

排程會自動略過已完成且通過設定檢查的CSV，透過lock防止同一資料夾重複啟動。失敗任務記錄於status.json；其他任務繼續，全部排完後不會無限重試。可檢查失敗原因，再用相同run指令恢復未完成任務。

這次先回答「相同實際 sparsity 下，importance 定義是否改善精度」。敏感層保護加ratio redistribution屬於下一階段，未混入本批。
