# Layer-wise pruning sensitivity 總整理

這裡整理的是仿照 *Pruning Filters for Efficient ConvNets* 的單層敏感度實驗：每次只剪一個 convolution layer，直接看 accuracy drop。

## 已完成

- L1 filter pruning：`filter_l1`，剪 filter L1 norm 最小的整個 filters。
- Cross-filter top-k：`top2_mean_abs`、`top4_mean_abs`，使用目前 cross-filter mask 方式做單層敏感度。
- 模型：ResNet34、ResNet50、GoogLeNet、DenseNet121、MobileNetV2。
- Ratios：10%、30%、50%、70%、90%。
- Dataset：CIFAR-10。

## 主要檔案

- `all_layer_sensitivity_combined.csv`：全部 sensitivity row 合併。
- `layer_sensitivity_ranked.csv`：依每層平均 accuracy drop 排名，最適合找敏感層。
- `top10_sensitive_layers_by_ratio.csv`：每個 model/criterion/ratio 最敏感前 10 層。
- `stage_sensitivity_summary.csv`：依 stage 聚合的平均/最大 drop。

## 圖表位置

- L1 filter pruning 圖：`../layer_sensitivity_l1_filter/*/*.png`
- cross-filter top2/top4 圖：`../layer_sensitivity_cross_filter_topk/*/*.png`

## 最敏感層摘要

### densenet121 / filter_l1

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `features.transition2.conv` | `transition2` | `1x1` | 42.34 | 75.95 |
| 2 | `features.transition1.conv` | `transition1` | `1x1` | 39.48 | 75.32 |
| 3 | `features.transition3.conv` | `transition3` | `1x1` | 23.91 | 64.27 |
| 4 | `features.conv0` | `other` | `3x3` | 17.89 | 49.91 |
| 5 | `features.denseblock1.denselayer4.conv2` | `denseblock1` | `3x3` | 9.05 | 28.89 |

### densenet121 / top2_mean_abs

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `features.conv0` | `other` | `3x3` | 41.69 | 70.22 |
| 2 | `features.transition2.conv` | `transition2` | `1x1` | 27.34 | 75.91 |
| 3 | `features.transition1.conv` | `transition1` | `1x1` | 24.69 | 74.33 |
| 4 | `features.transition3.conv` | `transition3` | `1x1` | 13.90 | 57.60 |
| 5 | `features.denseblock1.denselayer4.conv2` | `denseblock1` | `3x3` | 4.93 | 15.97 |

### densenet121 / top4_mean_abs

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `features.conv0` | `other` | `3x3` | 41.85 | 73.29 |
| 2 | `features.transition2.conv` | `transition2` | `1x1` | 25.24 | 73.29 |
| 3 | `features.transition1.conv` | `transition1` | `1x1` | 23.69 | 73.61 |
| 4 | `features.transition3.conv` | `transition3` | `1x1` | 13.48 | 55.23 |
| 5 | `features.denseblock1.denselayer4.conv2` | `denseblock1` | `3x3` | 5.46 | 19.22 |

### googlenet / filter_l1

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `inception4a.branch4.1.conv` | `inception4a` | `1x1` | 48.84 | 66.48 |
| 2 | `conv2.conv` | `conv2` | `1x1` | 43.26 | 73.56 |
| 3 | `inception3a.branch4.1.conv` | `inception3a` | `1x1` | 39.35 | 65.56 |
| 4 | `inception3a.branch1.conv` | `inception3a` | `1x1` | 30.77 | 59.18 |
| 5 | `conv3.conv` | `conv3` | `3x3` | 25.78 | 66.37 |

### googlenet / top2_mean_abs

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `inception3a.branch4.1.conv` | `inception3a` | `1x1` | 22.34 | 56.74 |
| 2 | `conv3.conv` | `conv3` | `3x3` | 21.82 | 68.09 |
| 3 | `conv2.conv` | `conv2` | `1x1` | 20.75 | 71.21 |
| 4 | `inception4a.branch4.1.conv` | `inception4a` | `1x1` | 20.69 | 61.89 |
| 5 | `conv1.conv` | `conv1` | `3x3` | 18.95 | 52.33 |

### googlenet / top4_mean_abs

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `conv2.conv` | `conv2` | `1x1` | 22.23 | 71.83 |
| 2 | `conv3.conv` | `conv3` | `3x3` | 20.61 | 68.51 |
| 3 | `inception3a.branch4.1.conv` | `inception3a` | `1x1` | 19.73 | 54.88 |
| 4 | `conv1.conv` | `conv1` | `3x3` | 19.18 | 52.32 |
| 5 | `inception4a.branch4.1.conv` | `inception4a` | `1x1` | 18.03 | 60.26 |

### mobilenet_v2 / filter_l1

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `features.1.conv.1` | `features.1` | `1x1` | 35.68 | 45.18 |
| 2 | `features.1.conv.0.0` | `features.1` | `3x3` | 26.49 | 44.07 |
| 3 | `features.2.conv.1.0` | `features.2` | `3x3` | 24.24 | 45.04 |
| 4 | `features.2.conv.2` | `features.2` | `1x1` | 22.31 | 44.02 |
| 5 | `features.4.conv.2` | `features.4` | `1x1` | 20.94 | 46.42 |

### mobilenet_v2 / top2_mean_abs

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `features.0.0` | `features.0` | `3x3` | 21.99 | 41.18 |
| 2 | `features.1.conv.0.0` | `features.1` | `3x3` | 21.97 | 46.32 |
| 3 | `features.2.conv.1.0` | `features.2` | `3x3` | 18.34 | 42.83 |
| 4 | `features.2.conv.2` | `features.2` | `1x1` | 14.99 | 40.40 |
| 5 | `features.2.conv.0.0` | `features.2` | `1x1` | 14.11 | 41.50 |

### mobilenet_v2 / top4_mean_abs

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `features.0.0` | `features.0` | `3x3` | 23.41 | 40.10 |
| 2 | `features.1.conv.0.0` | `features.1` | `3x3` | 21.36 | 45.60 |
| 3 | `features.2.conv.1.0` | `features.2` | `3x3` | 17.19 | 40.21 |
| 4 | `features.2.conv.0.0` | `features.2` | `1x1` | 13.43 | 39.19 |
| 5 | `features.2.conv.2` | `features.2` | `1x1` | 12.93 | 33.93 |

### resnet34 / filter_l1

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `layer2.0.downsample.0` | `layer2` | `1x1` | 19.00 | 44.91 |
| 2 | `layer2.3.conv1` | `layer2` | `3x3` | 14.06 | 30.73 |
| 3 | `layer2.0.conv1` | `layer2` | `3x3` | 12.83 | 39.27 |
| 4 | `conv1` | `stem` | `3x3` | 10.53 | 46.45 |
| 5 | `layer1.0.conv1` | `layer1` | `3x3` | 9.85 | 33.05 |

### resnet34 / top2_mean_abs

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `conv1` | `stem` | `3x3` | 33.25 | 53.90 |
| 2 | `layer2.0.downsample.0` | `layer2` | `1x1` | 13.72 | 43.46 |
| 3 | `layer2.0.conv1` | `layer2` | `3x3` | 7.60 | 31.28 |
| 4 | `layer2.1.conv1` | `layer2` | `3x3` | 7.16 | 28.64 |
| 5 | `layer2.3.conv1` | `layer2` | `3x3` | 7.05 | 23.61 |

### resnet34 / top4_mean_abs

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `conv1` | `stem` | `3x3` | 30.74 | 48.36 |
| 2 | `layer2.0.downsample.0` | `layer2` | `1x1` | 14.39 | 44.95 |
| 3 | `layer2.0.conv1` | `layer2` | `3x3` | 8.19 | 34.18 |
| 4 | `layer2.3.conv1` | `layer2` | `3x3` | 7.84 | 26.99 |
| 5 | `layer2.1.conv1` | `layer2` | `3x3` | 7.07 | 27.72 |

### resnet50 / filter_l1

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `conv1` | `stem` | `3x3` | 34.25 | 51.56 |
| 2 | `layer1.1.conv2` | `layer1` | `3x3` | 28.61 | 38.71 |
| 3 | `layer4.1.conv1` | `layer4` | `1x1` | 27.83 | 52.02 |
| 4 | `layer1.0.downsample.0` | `layer1` | `1x1` | 27.18 | 48.79 |
| 5 | `layer2.0.conv2` | `layer2` | `3x3` | 24.92 | 38.34 |

### resnet50 / top2_mean_abs

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `conv1` | `stem` | `3x3` | 35.41 | 48.89 |
| 2 | `layer1.0.conv3` | `layer1` | `1x1` | 17.05 | 35.40 |
| 3 | `layer2.0.conv1` | `layer2` | `1x1` | 16.82 | 35.35 |
| 4 | `layer1.0.conv2` | `layer1` | `3x3` | 15.65 | 34.88 |
| 5 | `layer1.0.downsample.0` | `layer1` | `1x1` | 14.29 | 34.10 |

### resnet50 / top4_mean_abs

| rank | layer | stage | kernel | mean drop pp | max drop pp |
|---:|---|---|---|---:|---:|
| 1 | `conv1` | `stem` | `3x3` | 38.79 | 48.45 |
| 2 | `layer1.0.conv3` | `layer1` | `1x1` | 17.92 | 37.37 |
| 3 | `layer2.0.conv1` | `layer2` | `1x1` | 15.24 | 30.11 |
| 4 | `layer1.0.conv1` | `layer1` | `1x1` | 14.94 | 27.08 |
| 5 | `layer1.0.conv2` | `layer1` | `3x3` | 14.32 | 33.80 |

## 解讀方式

- `acc_drop_pp` 越大，代表單獨剪該層越傷 accuracy。
- `layer_sensitivity_ranked.csv` 的 `mean_drop_pp` 可用來挑要 protected/skip 的層。
- `stage_sensitivity_summary.csv` 可用來設定 stage-wise pruning ratio，例如敏感 stage 給較低 ratio。
- `is_1x1=1` 的層如果排名很前面，代表 1x1 pruning 可能是目前 accuracy 掉的原因之一。
