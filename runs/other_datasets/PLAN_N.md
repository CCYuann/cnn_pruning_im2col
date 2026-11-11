# 計畫 N：更多模型、結構化代價、KD 的通用性（2026-11-09 凍結，下列組別皆尚未產生）

背景：計畫 M 中 GoogLeNet、DenseNet121 的 gradual+floor 對 rewind 全部成立（+1.4 至 +12.3pp），KD 在 ResNet34 以正確 Dense 配方重現。
使用者要求多跑幾個模型確認可行性。本計畫的組別與配方全部固定，不調參，test 於每條結束時讀一次。

## N1：VGG16-BN、ResNet-56（CIFAR-100）
兩者為 Li et al. 2017（Pruning Filters for Efficient ConvNets）在 CIFAR 上使用的模型，定義於 cifar_models.py。
ResNet-56 的捷徑改用 1×1 卷積（option B），兩者全部為 3×3/1×1、groups=1，激活值不超過 65,536。
組別同計畫 M3：Dense 100 epochs；rewind 與 gradual+floor 在 90/98/99%；unstructured 在 98/99%；seeds 0/1/2。
判定：gradual+floor 對 rewind 配對平均 >= +0.3pp 且 3/3 為正，逐比例判定。

## N2：結構化的真正代價
gradual_unstructured：與 gradual 相同的逐步排程，但每次用全域 magnitude 的 unstructured mask。
在 ResNet34、GoogLeNet、DenseNet121、VGG16-BN、ResNet-56 的 98/99%，seeds 0/1/2。
報告 gradual+floor − gradual_unstructured，這是排除「逐步 vs 一次」之後的結構化代價。不設門檻，只描述。

## N3：KD 是否通用
gradual+floor+kd（T=4，alpha=0.9）98%，GoogLeNet、DenseNet121、VGG16-BN、ResNet-56，seeds 0/1/2。
判定：對 gradual+floor 98% 配對平均 >= +0.3pp 且 3/3 為正。

共 96 條。週期為 V3 公式預測；Q15 只支援 ResNet34/50，新模型只報 FP32。

## 結果（2026-11-09 13:07 完成，96 條，0 失敗；CIFAR-100 FP32 test，seeds 0/1/2；腳本 scripts/decide_plan_n.py，原始值 PLAN_N_RESULTS.json）

### N1：gradual+floor 對 rewind
| 模型（Dense） | 比例 | rewind | gradual+floor | 配對差 | 判定 | V3 ms rewind → floor（Dense） |
|---|---:|---:|---:|---|---|---|
| VGG16-BN（72.10） | 90% | 71.79 | 71.90 | +0.02/+0.10/+0.20（+0.11） | 未過 | 34.27 → 31.09（154.23） |
| | 98% | 1.00 | 68.20 | +67.08/+67.05/+67.47 | 成立，但見下 | 10.92 → 8.90 |
| | 99% | 1.00 | 64.22 | +62.99/+63.71/+62.96 | 成立，但見下 | 6.93 → 5.28 |
| ResNet-56（70.93） | 90% | 65.14 | 67.40 | +2.97/+2.22/+1.61（+2.27） | 成立 | 4.63 → 4.42（39.46） |
| | 98% | 48.34 | 55.01 | +6.88/+6.26/+6.87（+6.67） | 成立 | 2.76 → 2.74 |
| | 99% | 40.87 | 46.18 | +5.81/+5.18/+4.93（+5.31） | 成立 | 2.72 → 2.72 |

- VGG16-BN 的 rewind 在 98/99% 三個 seeds 全部停在 1%（train 也是 1%）：一次剪讓 118–127 個 cluster 歸零，網路沒有捷徑，訊號斷掉。一次剪的 unstructured 99% 也有 seed 崩到 1%（平均 16.58）。這兩列的差距是基準崩潰，不是方法增益，論文中不能當作 +67pp 報告。計畫 O2 補一次剪加下限的對照。
- ResNet-56 在 98% 與 99% 的 V3 週期幾乎相同，多數 cluster 已低於 q=15，99% 只多掉精度、不省週期。

### N2：結構化代價（gradual+floor − gradual_unstructured）
| 模型 | 98% | 99% |
|---|---|---|
| ResNet34 | 72.42 − 74.67 = -2.25（-2.58/-2.44/-1.74） | 70.31 − 73.35 = -3.04 |
| GoogLeNet | 69.67 − 72.54 = -2.88 | 64.63 − 70.42 = -5.79 |
| DenseNet121 | 70.88 − 75.38 = -4.50 | 64.78 − 73.39 = -8.62 |
| VGG16-BN | 68.20 − 70.59 = -2.39 | 64.22 − 68.75 = -4.53 |
| ResNet-56 | 55.01 − 63.60 = -8.58 | 46.18 − 57.63 = -11.45 |

- 逐步的 unstructured 在每個模型、每個比例都比一次剪的 unstructured 好（例如 DenseNet121 99%：59.78 → 73.39）。計畫 M3 中「DenseNet121 的 floor 贏過 unstructured」只是因為對手是一次剪；排除排程差異後，結構化在同稀疏度下一律付出 2–11pp。
- 代價隨比例變大，在參數少、層窄的 ResNet-56 最大。

### N3：KD（T=4，alpha=0.9）於 98%
| 模型 | gradual+floor | +KD | 配對差 | 判定 |
|---|---:|---:|---|---|
| ResNet34（對照，計畫 M 已判定） | 72.42 | 73.74 | +1.47/+1.50/+1.00（+1.32） | 成立 |
| VGG16-BN | 68.20 | 69.45 | +1.58/+1.12/+1.04（+1.25） | 成立 |
| GoogLeNet | 69.67 | 69.70 | +0.18/-0.03/-0.06（+0.03） | 未過 |
| DenseNet121 | 70.88 | 70.73 | +0.04/-0.48/+0.00（-0.15） | 未過 |
| ResNet-56 | 55.01 | 53.77 | -1.11/-1.71/-0.90（-1.24） | 未過，有害 |

KD 不是通用增益：在 ResNet34、VGG16-BN 有效，在 GoogLeNet、DenseNet121 無效，在 ResNet-56 有害。alpha 未調整；為各模型另調 alpha 屬事後調參，本計畫不做。

### 等 V3 週期比較（事後分析，計畫 O3 的一次剪近似）
unstructured 在 V3 上要跑 8 個 filter 位置的聯集，98% 的預測週期為 ResNet34 90.14、GoogLeNet 27.76、DenseNet121 40.87、VGG16-BN 43.50、ResNet-56 7.33 ms，是 gradual+floor 98% 的 2.7–7.3 倍。
在不多於 unstructured 98% 的週期下，gradual+floor 90% 在 5 個模型都比 gradual_unstructured 98% 準：
ResNet34 75.98（48.61 ms）對 74.67（90.14）、GoogLeNet 74.93（22.65）對 72.54（27.76）、DenseNet121 77.14（37.81）對 75.38（40.87）、VGG16-BN 71.90（31.09）對 70.59（43.50）、ResNet-56 67.40（4.42）對 63.60（7.33）。
此比較是事後提出，unstructured 的週期為一次剪 mask 的近似，O3 補充以實際逐步 mask 檢查。
