# sensitivity-aware l1_sum 實驗整理

這個資料夾是把論文原始 cross-filter `l1_sum` 也套上 sensitivity-aware 保護策略後的 CIFAR-10 實驗。目的是跟前一批新方法 `max_abs / top2_mean_abs / top4_mean_abs` 的 sensitivity-aware 版本做公平比較。

## 實驗設定

- Dataset：CIFAR-10
- Models：ResNet34、ResNet50、GoogLeNet、DenseNet121、MobileNetV2
- Target pruning ratio：`0.6`、`0.7`
- Cross-filter criterion：`l1_sum`
- Ratio source：`global_cluster`
- Sensitivity CSV：`runs/layer_sensitivity_summary/layer_sensitivity_ranked.csv`
- Sensitive-layer rule：`top_k=4`、`skip_top_k=1`、`cap=0.2`
- Global layer cap：`--max-layer-prune-ratio 0.75`
- Training：`--epochs 5 --finetune-epochs 2 --batch-size 64`

注意：目前 `layer_sensitivity_ranked.csv` 沒有專門的 `l1_sum` sensitivity ranking，所以程式會用 `filter_l1` ranking 當保護層依據。也就是說，這批是在測「論文 l1_sum 剪枝準則 + 由 layer sensitivity 找出的保護層」。

## 輸出檔案

- `all_sensitivity_aware_l1_sum_results_combined.csv`：10 個 run 的完整合併表，包含 Dense、Global、Cross-filter rows。
- `l1_sum_sensitivity_aware_summary.csv`：每個 model / ratio 的 l1_sum cross-filter 摘要。
- `comparison_l1_sum_vs_newbest_sensitivity_aware.csv`：和 `runs/sensitivity_aware_topk_cifar10_split/best_sensitivity_aware_summary.csv` 的最佳新方法比較。
- `summaries/`：上述 CSV 的備份整理版。
- `charts/`：準確率、差距與 speedup 圖表。

## l1_sum 結果摘要

| Ratio | Model | Acc (%) | vs Dense Extra (pp) | vs Global (pp) | Prune (%) | MAC Red. (%) | Speedup |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0.6 | DenseNet121 | 87.28 | -1.13 | -1.72 | 55.78 | 47.88 | 1.92x |
| 0.6 | GoogLeNet | 84.33 | +0.25 | -0.74 | 58.48 | 40.42 | 1.68x |
| 0.6 | MobileNetV2 | 54.65 | -0.88 | -0.86 | 57.95 | 37.11 | 2.15x |
| 0.6 | ResNet34 | 86.29 | -0.46 | -0.38 | 59.89 | 41.79 | 1.72x |
| 0.6 | ResNet50 | 74.40 | -1.77 | -0.61 | 57.40 | 41.93 | 1.72x |
| 0.7 | DenseNet121 | 86.60 | -1.81 | -2.15 | 63.65 | 55.90 | 2.27x |
| 0.7 | GoogLeNet | 84.41 | +0.56 | -0.64 | 64.38 | 46.91 | 1.88x |
| 0.7 | MobileNetV2 | 55.17 | -0.36 | -0.66 | 64.22 | 43.37 | 2.47x |
| 0.7 | ResNet34 | 86.01 | +0.34 | -0.23 | 66.95 | 48.85 | 1.95x |
| 0.7 | ResNet50 | 72.99 | -0.70 | +0.02 | 63.80 | 48.09 | 1.93x |

## 和新方法最佳版比較

整體看，sensitivity-aware 保護讓 `l1_sum` 明顯比沒有保護時合理，但大多數情況仍輸給 `top2_mean_abs / top4_mean_abs`。例外是：

- ResNet50 ratio 0.6：`l1_sum` 比新方法最佳高 `+2.09pp`。
- ResNet34 ratio 0.6：`l1_sum` 比新方法最佳高 `+0.46pp`。

新方法明顯比較好的地方：

- ResNet50 ratio 0.7：新方法最佳比 `l1_sum` 高 `+4.20pp`。
- MobileNetV2 ratio 0.6：新方法最佳比 `l1_sum` 高 `+1.61pp`。
- DenseNet121 ratio 0.7：新方法最佳比 `l1_sum` 高 `+0.89pp`。

目前結論：如果要主張改良方法，`top2_mean_abs / top4_mean_abs + sensitivity-aware layer protection` 比單純 `l1_sum + protection` 更穩，尤其在高 pruning ratio 和 bottleneck / depthwise-heavy 模型上比較有優勢。
