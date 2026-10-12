# 固定 Dense checkpoint：目標剪枝率 80%

本資料夾比較 Cross-Filter Structured Pruning 在目標剪枝率 `0.8` 時的精確度、實際 sparsity 與硬體估算結果。

## 實驗設定

- 資料集：CIFAR-10
- 模型：ResNet34、ResNet50、GoogLeNet、DenseNet121、MobileNetV2
- Dense checkpoint：每個 seed 固定使用先前訓練 100 epochs 的權重
- Seeds：0、1、2
- `Tm`：8、16
- Fine-tune：10 epochs，learning rate `0.001`
- Batch size：128
- Cross-filter ratio source：`global_cluster`
- 原始方法：`l1_sum`
- 改良方法：`top2_mean_abs` + sensitivity-aware protection
- 敏感層設定：`top_k=4`、`skip_top_k=1`、敏感層 cap `0.2`、全層 cap `0.75`
- 精確度損失基準：同一個 seed 的 Dense extra（相同 checkpoint 再訓練 10 epochs）

## 三個 seed 平均結果

表內精確度與 loss 的單位均為百分點（pp）。括號內是實際權重剪枝率。

| 模型 | Dense extra | l1 Tm8：FT acc / loss | l1 Tm16：FT acc / loss | top2+敏感層 Tm8：FT acc / loss | top2+敏感層 Tm16：FT acc / loss |
|---|---:|---:|---:|---:|---:|
| ResNet34 | 94.790 | 94.440 / 0.350 (80.00%) | 94.143 / 0.647 (80.00%) | 94.507 / 0.283 (67.04%) | 94.420 / 0.370 (66.90%) |
| ResNet50 | 94.263 | 93.810 / 0.453 (80.00%) | 93.473 / 0.790 (80.00%) | 93.900 / 0.363 (68.54%) | 93.827 / 0.437 (68.62%) |
| GoogLeNet | 94.097 | 93.390 / 0.707 (80.00%) | 93.340 / 0.757 (80.00%) | 93.663 / 0.433 (65.10%) | 93.413 / 0.683 (65.12%) |
| DenseNet121 | 95.217 | 94.477 / 0.740 (80.00%) | 94.353 / 0.863 (80.00%) | 94.763 / 0.453 (68.24%) | 94.613 / 0.603 (66.76%) |
| MobileNetV2 | 85.323 | 82.597 / 2.727 (79.99%) | 81.440 / 3.883 (79.99%) | 83.440 / 1.883 (67.04%) | 82.580 / 2.743 (67.19%) |

## 解讀

1. `l1_sum` 的實際剪枝率確實約為 80%。ResNet34、ResNet50、GoogLeNet、DenseNet121 的平均精確度損失都低於 1 pp。
2. MobileNetV2 在 80% 時明顯失敗於「低於 1 pp」目標，Tm8 損失 2.727 pp，Tm16 損失 3.883 pp；其 depthwise/pointwise 結構可用冗餘較少。
3. 五個模型皆呈現 Tm8 優於 Tm16，代表較小 cluster 能更細緻地選擇共同保留的 K 位置。
4. `top2_mean_abs + sensitivity-aware` 的精確度普遍較好，但因 layer cap 與敏感層保護，實際只剪掉約 65% 至 69%。它不能直接宣稱在相同 80% sparsity 下優於 l1。
5. 若要做同 sparsity 的方法比較，下一步應調高 top2 的未保護層 ratio 或重新分配 pruning budget，讓其實際 sparsity 也到 80%。

## 檔案

- `all_results_combined.csv`：30 個模型/seed/Tm 實驗的原始結果合併
- `summary/all_methods_r0p8_summary.csv`：各方法的三 seed 平均與 sample standard deviation
- `summary/cross_filter_under_1pp_check.csv`：Cross-filter 是否低於 1 pp 的檢查表
- `summary/dense_summary.csv`：Dense 與 Dense extra 基準
- `summary/accuracy_loss_by_model_tm.png`：各模型與 Tm 的精確度損失圖
- `summary/accuracy_sparsity_tradeoff.png`：精確度與實際 sparsity 的 trade-off 圖

執行腳本：`scripts/run_fixed_checkpoint_r0p8_multi_model.sh`

彙整腳本：`scripts/summarize_fixed_checkpoint_r0p8.py`
