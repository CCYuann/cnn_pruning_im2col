# 軟體端測試結果：S1–S5（2026-11-11）

寫給硬體端。所有數字都來自定案的 Q15 流程（MSE scale、分支內 CLE k≤8、bias correction）。

## 量測的模型

- 輸入：每個模型用 5,000 張 validation 影像（從訓練集切出），沒有用 test。
- 每個模型共 8,512 個輸出 channel，36 個卷積層。

| 代號 | 模型 | 備註 |
|---|---|---|
| c10_fs98_seed6 | ResNet34 CIFAR-10，98%（fs） | V2.3 RTL 36/36 層逐位元驗證過 |
| c100_kd98_seed3 | ResNet34 CIFAR-100，98%（KD） | V2.3 RTL 36/36 層逐位元驗證過 |
| c100_floor90_seed0/1/2 | ResNet34 CIFAR-100，90%（gradual+floor） | 論文主線；以前沒有 Q15 |

## 結論

| 測試 | 用途 | 結論 |
|---|---|---|
| S1 | 縮小移位器（D） | 照現在的編碼，原始 right_shift 到 53，**不能縮**。軟體把「輸出固定」的 channel 改寫後，最大降到 35。 |
| S2 | 縮小量化輸入（D）、bias 預載進 28-bit 累加器（F） | 照現在的編碼，acc+bias 最大 31 bit，5 個模型都有 1–7 層超過 28 bit，**F 不安全**。改寫後最大 22 bit，可以預載。 |
| S3 | 確認 package 是否最終版 | `exports/layer_json_r0p8_contiguous` **不是最終版**。另外提供 90% 主線每個 cluster 的 q。 |
| S4 | zero gating 省電 | ReLU 後的層平均 47–56% 的 MAC 乘到 0。 |
| S5 | 論文精度比較 | 50% 時 cluster 比非結構化低 0.3–0.5pp、比 2:4 低 0.1–0.2pp；90% 時比非結構化低 0.9–2.0pp。V3 週期是另兩種的 1/2–1/5。 |

**D 和 F 能不能做，取決於軟體端是否採用「改寫固定輸出 channel」。** 改寫後的逐位元驗證有限制，見第 2 節。

## 1. 照現在的編碼量到的範圍（S1、S2）

明細：[summary.csv](summary.csv)（每個模型一列）、[layers.csv](layers.csv)（每層一列）。

| 項目 | 範圍（5 個模型合計） |
|---|---|
| conv multiplier | 16384–32767（Q15，正規化，15 bit 無號） |
| conv right_shift（原始） | 18–53；超過 48 的 channel 有 0–17 個，目前夾到 48 |
| identity requant right_shift／multiplier | 14–16／16433–32198 |
| classifier right_shift | 23–25 |
| acc（卷積和，不含 bias） | 最大 18–19 bit（有號） |
| bias | 最大 30–31 bit |
| acc+bias | 最大 30–31 bit；超過 28 bit 的層：1–7 層 |
| (acc+bias)×multiplier | 最大 44–45 bit |

原始 right_shift 的分布（每個 channel；細節見 [clamp_test.txt](clamp_test.txt)）：
- 大部分落在 21–26。
- 超過 32 的有 124–657 個 channel，其中 95% 以上在 validation 上輸出固定。
- 超過 28 bit 的大 bias 全部出現在輸出固定的 channel，例如 BN gain 幾乎為 0 的 channel。

**直接夾值的結果：** 不改寫，只把數值夾小，結果如下（logits 是否逐位元相同；括號內是 5,000 張中 Top-1 改變的張數）：

| 模型 | bias 夾到 28 bit | right_shift 上限 40 | 36 | 32 |
|---|---|---|---|---|
| c10_fs98_seed6 | 否（3） | 否（56） | 否（1350） | 否（2969） |
| c100_kd98_seed3 | 是 | 是 | 是 | 否（155） |
| c100_floor90_seed0/1/2 | 是 | 是 | 是 | 否（92–207） |

直接夾值在 CIFAR-100 的 4 個模型可以做到 bias 28 bit、right_shift 上限 36，但 CIFAR-10 的模型不行，所以不能當成通用規則。

## 2. 建議：軟體把輸出固定的 channel 改寫

**做法：** 若某個 channel 在 validation 上輸出永遠是同一個值 v，就把它改寫成：
- 權重全部設 0，
- bias = v，
- multiplier = 2^14，
- right_shift = 14。

這樣輸出一定是 v。每個模型要改寫的 channel 有 193–956 個，占 8,512 個的 2–11%。

**改寫後的範圍：**

| 項目 | 改寫後 |
|---|---|
| right_shift | 14–35（只算輸出不固定的 channel 是 18–35） |
| bias | 最大 20–22 bit |
| acc+bias | 最大 20–22 bit，**可以預載進 28-bit 累加器（F）** |
| (acc+bias)×multiplier | 約 37–38 bit（推算，未逐筆量） |

**逐位元驗證（[reencode_test.txt](reencode_test.txt)）：**
- validation 5,000 張：5 個模型的 logits 全部逐位元相同。
- test 10,000 張：
  - 3 個模型逐位元相同。
  - c100_floor90 的 seed1、seed2 有部分影像的 logits 不同，但 Top-1 都沒有改變（0/10,000）。

**限制：** 「輸出固定」是在 validation 上觀察的，不是數學上的保證，所以在沒看過的影像上可能有很小的差異。若硬體需要保證逐位元相同，軟體端可以改用保守的判定：用權重的最大可能累加值去證明輸出不會變，只改寫證明得了的 channel。這還沒做。

## 3. S3：package 版本

- `mask_aware_conv_accelerator/exports/layer_json_r0p8_contiguous`：
  - 內容是 ResNet34 CIFAR-10、80%、舊的 contiguous 分組，seeds 0/1/2，2026-11-05 commit。
  - **不是最終版**，用它算出的週期代價不代表論文主線。
- **論文主線：** ResNet34 CIFAR-100，90%，gradual+floor。這版還沒有輸出完整的 package（權重 .mem 檔）。
  - 如果只是估算週期代價（A），用每層幾何和每個 cluster 的 q： [seed 0](q_per_cluster_r90/cifar100_resnet34_seed0.json)、[seed 1](q_per_cluster_r90/cifar100_resnet34_seed1.json)、[seed 2](q_per_cluster_r90/cifar100_resnet34_seed2.json)。
  - 如果需要完整 package，請告訴軟體端。

## 4. S4：ReLU 後的零值比例

每層輸入為 0 的比例，以「保留權重的 MAC 中，乘到 0 的比例」計算，不含 padding。明細在 [layers.csv](layers.csv) 的 `mac_zero_share` 欄。

| 模型 | ReLU 後 35 層的平均 | 最小–最大 |
|---|---|---|
| c10_fs98_seed6 | 53.2% | 16.3–80.7% |
| c100_kd98_seed3 | 46.8% | 13.1–82.0% |
| c100_floor90_seed0 | 54.2% | 8.6–83.3% |
| c100_floor90_seed1 | 54.5% | 12.2–84.3% |
| c100_floor90_seed2 | 55.8% | 12.5–83.1% |

- 這是 MAC 數的平均比例，不是週期。V3 每個 tile 是固定拍數，所以 zero gating 能省的是電，不是時間。
- conv1 的輸入是有號的影像，乘到 0 的比例只有 0.5%。

## 5. S5：精度比較（完成）

[計畫 Y](../other_datasets/PLAN_Y.md) 在讀結果前凍結。CIFAR-100、seeds 0/1/2、每組 n=3，三種剪枝用相同的逐步剪枝流程。

| 模型、剪枝率 | Dense | cluster 共用 | 非結構化 | 2:4 | V3 ms：cluster／非結構化／2:4 |
|---|---:|---:|---:|---:|---|
| ResNet34 50% | 76.79 | 76.76 | 77.23 | 76.86 | 221.8／418.8／420.0 |
| ResNet-56 50% | 70.93 | 70.57 | 70.90 | 70.77 | 18.3／38.8／38.9 |
| ResNet34 90% | 76.79 | 75.98 | 76.84 | 做不到 | 48.6／250.4／— |
| ResNet-56 90% | 70.93 | 67.40 | 69.39 | 做不到 | 4.4／20.8／— |

- 50%：cluster 比非結構化低 0.3–0.5pp，比 2:4 低 0.1–0.2pp（n=3 下和 2:4 分不出差異）。
- 週期：在 V3 上，cluster 約是非結構化和 2:4 的 1/2（50%）到 1/5（90%）。非結構化和 2:4 要跑 8 個 filter 的位置聯集，所以幾乎沒有加速。
- 2:4 最多只能剪 50%。它在支援 2:4 的硬體上可加速 2 倍，但 V3 不是那類硬體。
- 只有浮點精度（AMP float16）。週期是 V3 公式預測，不是 RTL 實測。

## 檔案

| 檔案 | 內容 |
|---|---|
| [summary.csv](summary.csv)、[layers.csv](layers.csv) | S1、S2、S4 的量測，由 [hw_range_report.py](../../scripts/hw_range_report.py) 產生 |
| [clamp_test.txt](clamp_test.txt) | 直接夾值的逐位元測試 |
| [reencode_test.txt](reencode_test.txt) | 改寫固定輸出 channel 的逐位元測試 |
| [scripts/](scripts/) | 上面兩個測試的腳本 |
| [q_per_cluster_r90/](q_per_cluster_r90/) | 90% 主線每層的幾何和每個 cluster 的 q |
