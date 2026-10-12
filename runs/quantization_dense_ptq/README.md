# Dense 模型 PTQ 位元寬度測試

本測試使用固定 Dense checkpoints，在 CIFAR-10 完整測試集比較 FP32、INT8、
INT9 與 INT10。每個模型使用 3 個 seeds，activation range 使用 2,048 張 validation
影像校正。

量化設定：

- Weight：per-output-channel signed symmetric
- Activation：per-layer signed symmetric static PTQ
- Weight/Activation 使用相同位元寬度
- 模擬 INT32 accumulator
- 這是 fake-quantized inference，尚未包含剪枝後模型與 QAT

## 三個 seeds 平均結果

| 模型 | FP32 | INT8（損失） | INT9（損失） | INT10（損失） |
|---|---:|---:|---:|---:|
| ResNet34 | 94.753% | 94.703%（0.050 pp） | 94.733%（0.020 pp） | 94.727%（0.027 pp） |
| ResNet50 | 94.253% | 94.227%（0.027 pp） | 94.207%（0.047 pp） | 94.217%（0.037 pp） |
| GoogLeNet | 94.013% | 94.013%（0.000 pp） | 93.980%（0.033 pp） | 94.013%（0.000 pp） |
| DenseNet121 | 95.267% | 95.130%（0.137 pp） | 95.210%（0.057 pp） | 95.273%（-0.007 pp） |
| MobileNetV2 | 85.500% | 85.220%（0.280 pp） | 85.487%（0.013 pp） | 85.450%（0.050 pp） |

五模型平均 accuracy loss：INT8 `0.099 pp`、INT9 `0.034 pp`、INT10
`0.021 pp`。INT8 最大單次 seed loss 是 MobileNetV2 的 `0.38 pp`。

## 固定乘 32/64 的對照

ResNet34 seed 0 的 FP32 accuracy 為 94.82%。直接使用單一固定 scale 量化權重：

| Weight scale | Weight-only | 加上 INT8 activation |
|---|---:|---:|
| x32 | 21.57% | 21.36% |
| x64 | 89.42% | 89.25% |
| x128 | 94.45% | 94.40% |
| x256 | 94.76% | 94.64% |
| 校正式 per-channel INT8 | - | 94.80% |

固定 x32/x64 對目前 checkpoints 的解析度不足，會將大量小權重變成零。若硬體希望
使用 shift，應為每層或每個 output channel 選擇 power-of-two scale，而不是全模型
固定使用同一個 x32/x64。

原始結果在 `results_combined.csv`，平均結果在 `summary.csv`。
