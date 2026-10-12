# ResNet34 固定 checkpoint：85% 與 90% 剪枝

本實驗延續 80% 正式流程，測試 ResNet34 在更高 Cross-Filter Structured Pruning ratio 下，fine-tune 是否仍能將精確度損失控制在 1 percentage point（pp）內。

## 實驗設定

- 資料集：CIFAR-10
- 模型：ResNet34
- Target pruning ratio：0.85、0.90
- Seeds：0、1、2
- `Tm`：8、16
- Dense checkpoint：固定使用各 seed 已訓練 100 epochs 的 checkpoint
- Dense extra / pruning fine-tune：10 epochs，learning rate `0.001`
- Batch size：128
- Ratio source：`global_cluster`
- 原始方法：`l1_sum`
- 改良方法：`top2_mean_abs` + sensitivity-aware protection
- 改良版設定：`top_k=4`、`skip_top_k=1`、敏感層 cap `0.2`、全層 cap `0.75`
- Accuracy loss 基準：同 seed 的 Dense extra test accuracy

共完成 2 ratios x 3 seeds x 2 Tm = 12 個工作，全部成功且無 OOM。6 個工作同時在 GPU1 執行，總時間約 47 分鐘。

## 三個 seed 平均結果

| Target | 方法 | Tm | FT test accuracy | Accuracy loss | Sample std | 實際 sparsity | 低於 1 pp |
|---:|---|---:|---:|---:|---:|---:|:---:|
| 85% | l1_sum | 8 | 94.183% | 0.607 pp | 0.108 pp | 85.000% | 是 |
| 85% | l1_sum | 16 | 93.923% | 0.867 pp | 0.305 pp | 85.000% | 是 |
| 90% | l1_sum | 8 | 94.007% | 0.783 pp | 0.195 pp | 90.000% | 是 |
| 90% | l1_sum | 16 | 93.413% | **1.377 pp** | 0.263 pp | 90.000% | **否** |
| 85% | top2 + 敏感層 | 8 | 94.443% | 0.347 pp | 0.122 pp | 69.637% | 是 |
| 85% | top2 + 敏感層 | 16 | 94.390% | 0.400 pp | 0.225 pp | 69.510% | 是 |
| 90% | top2 + 敏感層 | 8 | 94.267% | 0.523 pp | 0.067 pp | 71.637% | 是 |
| 90% | top2 + 敏感層 | 16 | 94.147% | 0.643 pp | 0.121 pp | 71.502% | 是 |

## 與 80% l1_sum 比較

| Target | Tm8 loss | Tm16 loss |
|---:|---:|---:|
| 80% | 0.350 pp | 0.647 pp |
| 85% | 0.607 pp | 0.867 pp |
| 90% | 0.783 pp | **1.377 pp** |

## 結論

1. ResNet34 的 fine-tune 恢復能力有上限，不是任何 pruning ratio 都能恢復。
2. `Tm=8` 在真正 90% sparsity 下仍維持平均損失 0.783 pp，代表較小 cluster 保留了更細緻的 K-position 選擇能力。
3. `Tm=16` 在 90% 時平均損失達 1.377 pp，已超出原先的 1 pp 目標；ResNet34 的臨界區大約落在 85% 至 90% 之間。
4. top2+sensitivity 雖然精度較好，但實際 sparsity 只有約 69.5% 至 71.6%，不能當作相同 85%/90% sparsity 下優於 l1_sum。
5. 若硬體允許 `Tm=8`，目前結果支持以真正 90% sparsity 作為高壓縮選項；若使用 `Tm=16`，85% 較符合低於 1 pp 的精度要求。

## 檔案

- `all_results_combined.csv`：12 個工作的完整結果
- `summary/all_methods_summary.csv`：Global、l1_sum、top2+sensitivity 三 seed統計
- `summary/cross_filter_summary.csv`：Cross-filter 方法摘要
- `summary/accuracy_loss_r0p85_r0p90.png`：85%/90% 精確度損失圖
- `summary/accuracy_sparsity_tradeoff.png`：實際 sparsity 與精確度損失圖

執行腳本：`scripts/run_fixed_checkpoint_resnet34_r0p85_r0p90.sh`

彙整腳本：`scripts/summarize_fixed_checkpoint_resnet34_r0p85_r0p90.py`
