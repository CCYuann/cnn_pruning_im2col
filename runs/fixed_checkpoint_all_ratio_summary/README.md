# 固定 Dense Checkpoint：跨模型剪枝率總整理

本資料夾統整 CIFAR-10、固定 Dense checkpoint、3 seeds、Tm=8/16、fine-tune 10 epochs 的正式實驗。

五個模型共同比較的 target pruning ratio 為 70%、80%、85%、90%、95%。ResNet34 另外保留先前 50%、60% 結果於完整 CSV 與曲線圖中。

## l1_sum 精確度損失

下表為相對同 seed `Dense extra` 的 test accuracy 平均損失，單位為 percentage point（pp）。粗體代表三 seed 平均已達或超過 1 pp。

| 模型 | Tm | 70% | 80% | 85% | 90% | 95% |
|---|---:|---:|---:|---:|---:|---:|
| ResNet34 | 8 | 0.260 | 0.350 | 0.607 | 0.783 | **2.157** |
| ResNet34 | 16 | 0.397 | 0.647 | 0.867 | **1.377** | **2.973** |
| ResNet50 | 8 | 0.207 | 0.453 | 0.780 | **1.330** | **2.153** |
| ResNet50 | 16 | 0.417 | 0.790 | **1.170** | **1.703** | **3.080** |
| GoogLeNet | 8 | 0.293 | 0.707 | 0.887 | **1.637** | **3.583** |
| GoogLeNet | 16 | 0.430 | 0.757 | **1.113** | **2.257** | **4.607** |
| DenseNet121 | 8 | 0.417 | 0.740 | 0.873 | **1.630** | **3.823** |
| DenseNet121 | 16 | 0.447 | 0.863 | **1.293** | **2.297** | **5.017** |
| MobileNetV2 | 8 | **1.977** | **2.727** | **2.980** | **3.363** | **4.337** |
| MobileNetV2 | 16 | **3.017** | **3.883** | **4.250** | **4.920** | **6.170** |

## 低於 1 pp 的最高測試剪枝率

| 模型 | Tm8 | Tm16 |
|---|---:|---:|
| ResNet34 | 90% | 85% |
| ResNet50 | 85% | 80% |
| GoogLeNet | 85% | 80% |
| DenseNet121 | 85% | 80% |
| MobileNetV2 | 無；70% 已超過 | 無；70% 已超過 |

這是「目前測試過的離散 ratio」結論，不代表精確臨界點。例如 ResNet34/Tm8 的臨界點位於 90% 到 95% 之間，但尚未做更細的 91%–94% 搜尋。

## 主要觀察

1. Fine-tune 並非可以救回任意剪枝率。所有模型在 ratio 繼續增加後都會跨過 1 pp，95% 時下降尤其明顯。
2. Tm8 幾乎全面優於 Tm16。較小 cluster 只要求 8 個 filters 共用 mask，能保留更適合各群組的 K positions；Tm16 的共同限制較強。
3. ResNet34 對 cross-filter pruning 最有韌性，Tm8 到真正 90% sparsity 仍只有 0.783 pp 平均損失。
4. MobileNetV2 最敏感，70% 時已超過 1 pp；其 depthwise/pointwise 結構原本就較精簡，可替代冗餘較少。
5. `top2_mean_abs + sensitivity-aware` 在高 target ratio 下實際只達約 59%–74% sparsity，因為敏感層與全層 cap 限制了剪枝量。其精度較好，但不能宣稱是在相同 85%–95% sparsity 下勝過 l1_sum。

## 檔案

- `all_results_combined.csv`：所有固定 checkpoint 實驗的原始結果合併
- `all_methods_summary.csv`：Global、l1_sum、top2+sensitivity 的三 seed 平均與 sample standard deviation
- `cross_filter_summary.csv`：Cross-filter 方法摘要
- `l1_accuracy_loss_pivot.csv`：本 README 主表的機器可讀版本
- `l1_under_1pp_thresholds.csv`：各模型低於 1 pp 的最高測試 ratio
- `l1_accuracy_loss_curves.png`：不同 target ratio 的精確度損失曲線
- `l1_accuracy_loss_heatmap.png`：模型、Tm、ratio 的精確度損失 heatmap
- `accuracy_sparsity_tradeoffs.png`：l1 與 top2 保護版的實際 sparsity/accuracy trade-off

高剪枝率原始結果位於 `runs/fixed_checkpoint_high_ratio_sweep/`，共 78 份 CSV，所有工作成功完成且日誌未發現 OOM 或 traceback。
