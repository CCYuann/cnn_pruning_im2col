# Layer-wise pruning sensitivity

這個資料夾是單層剪枝敏感度分析：每次只剪一個 convolution layer，直接看 validation/test accuracy 掉多少。

## 設定

- dataset: `cifar10`
- models: `resnet34`
- mode: `filter_l1`
- ratios: `0.1`
- epochs for dense model: `0`
- CSV: `layer_sensitivity.csv`

## 怎麼看

- `acc_drop_pp` 越大，代表該層越敏感，剪那一層 accuracy 掉越多。
- 曲線圖是不同 prune ratio 下，每一層的 accuracy drop。
- heatmap 顏色越深，代表該 layer/ratio 組合越危險。
- 如果要模仿 *Pruning Filters for Efficient ConvNets*，可用 `--mode filter_l1`。如果要對應目前 cross-filter 實驗，用 `--mode cross_filter`。

## 圖表

- `resnet34_layer_sensitivity.png`
- `resnet34_layer_sensitivity_heatmap.png`
