# ResNet34 剪枝加 INT8 模擬

本實驗使用相同的固定 Dense checkpoints，先做 80% cross-filter structured
pruning，再 Fine-tune 10 epochs，最後執行 FX static INT8 PTQ。

設定：

- 模型：ResNet34 / CIFAR-10
- Seeds：0、1、2
- Filter grouping：similarity Jaccard
- Criterion：l1_sum
- Tm：8
- Target pruning ratio：80%
- Fine-tune：10 epochs，learning rate 0.001
- Calibration：2,048 images
- INT8 backend：ONEDNN
- Conv+BN fusion、per-channel signed INT8 weights
- Affine 8-bit activations、INT32 accumulation、quantized residual add

## 結果

| Seed | Dense FP32 | 剪枝當下 | 剪枝 FT 後 FP32 | 剪枝 FT 後 INT8 | INT8 相對損失 |
|---:|---:|---:|---:|---:|---:|
| 0 | 94.81% | 88.59% | 94.53% | 94.57% | -0.04 pp |
| 1 | 94.88% | 79.14% | 94.48% | 94.54% | -0.06 pp |
| 2 | 94.58% | 80.39% | 94.49% | 94.50% | -0.01 pp |
| 平均 | 94.757% | 82.707% | 94.500% | 94.537% | -0.037 pp |

負數表示量化結果略高於 FP32。差距小於 CIFAR-10 單一樣本所代表的
0.01 percentage point 尺度累積，應視為沒有可辨識的量化損失，而不是量化能提升模型。

## 解讀限制

ONEDNN 使用 signed INT8 weights，但 activation 採 affine 8-bit（通常為 uint8
加 zero point），因此已涵蓋實際量化 Conv、融合與 residual add，但還不是目前
硬體文件中「所有 activation 都是 signed INT8、zero point=0」的 RTL bit-exact
結果。正式 RTL 驗證仍需使用相同 layer scale、rounding、saturation 與
requantization 參數逐層比對。

原始三個 seed 的資料收錄於 `results_combined.csv`。
