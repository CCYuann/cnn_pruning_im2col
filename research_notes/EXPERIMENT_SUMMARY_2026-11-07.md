# 高剪枝率實驗統整（截至 2026-11-07）

本文件統整 Cross-Filter Structured Pruning 在 98–99% 高剪枝率的實驗，以及在 mask-aware 加速器 V3 上的週期分析。
細節與逐條判定見文末的檔案索引；本文件只列結論與主要數字。

所有週期皆為 V3 公式預測（50 MHz，僅卷積層），尚未在 V3 RTL 上量測；RTL 逐位元驗證是在 V2.3 RTL 上做的。

## 1. 結論摘要

1. **方法在極端剪枝率下有效，且剪得越多優勢越大。** 90–95% 時各方法差距在雜訊內，98–99% 才拉開。
2. **CIFAR-10 的主方法是 fs**（逐步剪枝 + q 下限 + conv1 全保留）。99% 對 rewind +1.06pp（6 seeds，6/6 為正），週期少約 27%。
3. **CIFAR-10 會掩蓋精度損失，方法證據要看 CIFAR-100。** CIFAR-100 上隨機 mask 輸 2–3pp，能分辨方法；99% 對 Dense 掉約 7pp，CIFAR-10 只掉約 2pp。
4. **KD 在 CIFAR-100 的 90–98% 有效**，98% 約 +1pp，兩批 seeds 皆重現；在 CIFAR-10 有害，在 99% 無效。
5. **主線應是 ResNet34。** V3 每 tile 至少 15 拍，ResNet50 最快 18.95 ms，ResNet34 最快 5.73 ms，且 ResNet50 在 CIFAR-100 全面較差。
6. **軟硬體一致。** Q15 整數流程只掉 0.0–0.4pp；兩個定案模型在 RTL 36/36 層逐位元一致，週期估算與 RTL 量測差 0.15–0.35%。

## 2. 名詞

| 名稱 | 意義 |
|---|---|
| cluster | 8 個 filter 一組（Tm=8），共用同一個 K-position mask |
| q | 每個 cluster 保留的 K 位置數；K = Cin × Kh × Kw |
| rewind | 由 Dense 一次產生 mask（全域 cluster 配額，flat L1 加總），再以 LR rewinding 重訓 60 epochs |
| gradual | 前 2/3 epochs 以三次曲線逐步提高剪枝率，每 2 epochs 更新 mask |
| floor（q 下限） | 每個 cluster 至少保留 1 個位置，總權重數不變 |
| fs | gradual + floor + conv1 全保留 |
| fill15 / budget15 | 每個 cluster 至少 15 個位置；fill15 額外加權重，budget15 總權重不變 |
| KD | Dense 模型當 teacher，T=4；alpha 為 soft target 權重，預設 0.9 |
| Q15 | 加速器的整數流程：per-tensor MSE scale、分支內 CLE（k≤8）、逐層 bias correction（256 張），right_shift 夾到 48 |

## 3. 方法紀律

- 每個計畫在跑之前寫下組別與門檻，判定只用 validation，判定寫下後才讀 test，每個模型只讀一次。
- 開發 seeds 上通過的方法，必須在沒看過的新 seeds 上再通過一次，才算成立。
- 門檻：CIFAR-10 為配對平均 ≥ +0.3pp 且所有 seed 為正；CIFAR-100 validation 雜訊較大（單模型標準誤約 0.65pp），自計畫 I 起改為 ≥ +0.7pp。
- 事後才提出的想法、讀錯的資料、判定與 test 不一致的情形都照實揭露，負面結果照報。

## 4. CIFAR-10 ResNet34（硬體主線）

### 4.1 G3：seeds 6/7/8（Dense FP32 94.84%）

| 方法 | 95% | 98% | 99% |
|---|---:|---:|---:|
| rewind | 94.37 | 93.49 | 91.60 |
| gradual | 94.21 | 93.00 | 91.88 |
| gradual + floor | | 93.38 | |
| **fs** | **94.54** | **93.66** | **92.58** |

Q15 test（%）。V3：fs 98% 20.74 ms、99% 12.53 ms；rewind 98% 26.97 ms、99% 16.81 ms。

### 4.2 G4：seeds 9/10/11（Dense FP32 94.98%）

| 方法 | Q15 test | V3 ms | 加速 |
|---|---:|---:|---:|
| rewind 98% | 93.37 | 27.14 | 15.8× |
| **fs 98%** | **93.83** | **20.64** | **20.8×** |
| fill15 98% | 93.73 | 18.73 | 22.9× |
| budget15 98% | 93.73 | 20.04 | 21.4× |
| rewind 99% | 91.95 | 16.97 | 25.2× |
| **fs 99%** | **92.98** | **12.34** | **34.7×** |
| fill15 99% | 92.77 | 10.90 | 39.3× |
| budget15 99% | 92.36 | 10.94 | 39.2× |

- fill15 98% 開發時 +0.42pp，新 seeds -0.09pp，未重現。
- budget15 99% 依規則以 validation 通過（週期 -11.4%），但 Q15 test 對 fs -0.48/-0.68/-0.68pp，不採用。

### 4.3 conv1 色盲

逐步剪枝的全域 L1 排序會把 conv1（K=27）剪到 3–5 個位置，有時整個顏色通道消失（例如 G2 seed4 只剩 G 通道）。
fs 保留 conv1 全部位置後修好這個問題。G3 中增益集中在缺色的配對：缺色 3 對平均 +0.99pp，顏色齊全 6 對平均 +0.10pp。
CIFAR-100 的 conv1 都保有 RGB，fs 在那裡沒有增益（計畫 H1）。

## 5. 其他資料集

### 5.1 ImageNette、ImageWoof（ResNet34，160px，seeds 0/1/2）

隨機 mask 在 ImageNette 差距小於 1pp，在 ImageWoof 反而比學到的 mask 好（98%：79.81 對 78.69）。
兩個資料集都無法區分剪枝方法，精度不能當證據。它們的 stem 是 7×7，V3 也無法實跑。

### 5.2 CIFAR-100 ResNet34（seeds 0/1/2，FP32 test，Dense 76.79%）

| 方法 | 90% | 95% | 98% | 99% |
|---|---:|---:|---:|---:|
| 隨機 mask | | | 68.96 | 65.31 |
| rewind | 75.91 | 74.59 | 71.32 | 67.97 |
| **gradual + floor** | 75.98 | 74.54 | **72.42** | **70.31** |
| unstructured（參考） | 77.11 | 76.01 | 74.31 | 71.77 |
| V3 ms：rewind → gradual | 59.60 → 48.61 | 34.57 → 25.71 | 17.87 → 12.38 | 11.53 → 8.27 |

- gradual + floor 對 rewind：98% +0.63/+1.35/+1.32pp，99% +2.72/+2.71/+1.60pp；90%、95% 未過門檻。
- Q15 test 的配對增益：98% +1.10pp，99% +2.11pp。
- 新 seeds 3/4/5 的 FP32 test 重現：98% +1.48pp，99% +2.47pp，3/3。

### 5.3 CIFAR-100 ResNet50（seeds 0/1/2，Dense 75.73%）

四個比例的 gradual 增益都不是 3/3。99% 為 64.50%，比 ResNet34 低 5.8pp；對 unstructured 的結構化代價 -6.19pp（ResNet34 -1.46pp）；週期 20.77 ms，接近 18.95 ms 的下限。

## 6. 訓練配方（CIFAR-100 ResNet34，gradual + floor 為基準）

| 計畫 | 內容 | 結果 |
|---|---|---|
| H | fs、budget15、週期感知配額（|w|/(Hout·Wout)^β） | 皆未通過；週期感知配額同週期下 -1 至 -4pp |
| I | 降低 weight decay、延長到 120 epochs、KD | 只有 KD 98% 通過（+1.67pp）；欠擬合假設不成立 |
| J | KD 新 seeds 3/4/5 確認 | 98% 通過，Q15 test +1.05pp |
| K | KD 用在 CIFAR-10、alpha=0.5、90/95% | CIFAR-10 有害；alpha 0.5 99% 與 KD 90/95% 開發通過 |
| L | K 的新 seeds 確認 | 90% 通過；95% validation 差一個 seed 未過、test 3/3 支持；99% 未重現 |

KD 最終結果（seeds 3/4/5，Q15 test，Dense 76.70%）：

| 剪枝率 | 無 KD（FP32） | KD（FP32） | KD（Q15） | V3 ms | 加速 |
|---:|---:|---:|---:|---:|---:|
| 90% | 75.44 | 76.30 | 76.29 | 50.48 | 8.5× |
| 95% | 74.19 | 75.40 | 75.32 | 26.99 | 15.9× |
| 98% | 72.20 | 73.16 | **73.12** | **12.85** | **33.3×** |
| 99% | 69.87 | 69.76 | 69.65 | 8.35 | 51.3× |

KD 只改訓練，不動硬體與資料包格式。CIFAR-10 的 fs + KD 為 98% -0.47pp、99% -0.57pp（3/3 為負），所以 CIFAR-10 主線不加 KD。

## 7. 硬體相關

- **V3 週期模型：** 每個 cluster 31 + Hout × ceil(Wout/8) × max(q, 15) 拍。q 低於 15 不省週期，只多佔權重與載入頻寬。
- **V3 下限（32×32）：** ResNet34 5.73 ms（最多 74.8×），ResNet50 18.95 ms（最多 25.3×）。
- **mask 落點影響很大：** 同樣約 53,200 個位置，放在 layer4 便宜、放在 layer1/2 貴，週期可差 25%。開發 seeds 1/2 的 ms 因此不能和 seeds 3–11 比。
- **Q15：** right_shift 可能出現 49–55（近乎死掉的 channel），夾到 48 後 5000 張 logits 完全相同。peak_acc 約 2^29 來自這些 channel 的 bias，真實卷積和峰值約 1.5–1.8e5。
- **RTL 逐位元驗證（V2.3）：**
  - CIFAR-10 fs 98% seed6，兩張影像皆 36/36，量測 1,160,791 拍，估算差 0.35%。
  - CIFAR-100 KD 98% seed3，一張影像 36/36，量測 686,221 拍，估算差 0.15%。
  - 涵蓋 q=27 的 conv1、q 最大 250 以上的 cluster、stride 2、1×1 downsample、2^29 bias 與 right_shift 48。

## 8. 試過但無效

| 方法 | 結果 |
|---|---|
| 95% 階段的 DPF、LAMP、MAC 配額 | 未通過（runs/breakthrough_r0p95/RESULT_ROUND1.md） |
| fill15、budget15 | 週期少約 10%，精度沒有穩定增益 |
| 週期感知配額 | 同週期下精度更差 |
| 降低 weight decay、延長訓練 | 無增益 |
| fs 用在 CIFAR-100 | 無增益，只修色盲 |
| KD 用在 CIFAR-10、99% | 有害或無效 |
| ImageNette、ImageWoof | 不能區分方法 |

## 9. 統計

配對 test 差值，FP32，明細見 runs/other_datasets/BOOTSTRAP_FINAL.json。
影像區間只含 test 抽樣變異，seed 區間（t 分布）只含 seed 變異。

| 比較 | seeds | 差值（pp） | 影像 95% | seed 95% |
|---|---:|---:|---|---|
| CIFAR-10 fs − rewind 98% | 6 | +0.39 | 0.19 至 0.59 | 0.08 至 0.70 |
| CIFAR-10 fs − rewind 99% | 6 | +1.06 | 0.82 至 1.30 | 0.76 至 1.36 |
| CIFAR-100 gradual+floor − rewind 98% | 3 | +1.09 | 0.62 至 1.55 | 0.18 至 2.01 |
| CIFAR-100 gradual+floor − rewind 99% | 3 | +2.32 | 1.80 至 2.81 | 0.73 至 3.90 |
| CIFAR-100 KD − gradual+floor 98% | 3 | +0.96 | 0.53 至 1.41 | -0.55 至 2.47 |

KD 的 seed 區間含 0，論文應寫「兩批 seeds 皆重現」，不寫「顯著」。

## 10. 與原論文的差異與限制

原論文（Pham et al., IEEE Access 2025）只測 50–90% 的精度，加速比較用約 60%，資料集為 CIFAR-10、Animals-10、ImageNette，流程是一次剪完再 fine-tune。
本研究延伸到 98–99%，找出原流程在極端剪枝率下的失效點（cluster q=0、conv1 色盲、一次剪枝崩潰）並修正，再用自己的加速器量化週期。

限制：
- 本研究的 rewind 基準與原論文流程不完全相同（配額為全域 cluster 排序，並有 LR rewinding），還不能寫「原方法在 99% 掉多少」。
- V3 週期為公式預測；RTL 驗證在 V2.3 上做，tb 副本放大了陣列與 MEMORY_DEPTH，未跑 backpressure 與 Reload。
- residual add、pooling、FC 仍在軟體；不是端到端硬體推論。
- 計畫 H 彙整時曾意外印出 fs 組 seeds 0/1/2 的 test，判定已先寫下，未受影響（揭露於 PLAN_H.md）。

## 11. 下一步

1. 照原論文流程原樣實作基準，在 CIFAR-100 跑 90–99%，與原論文直接比較。
2. 取得 v3-datapath 分支後，在 V3 RTL 重跑 36 層驗證，週期改為實測。
3. 回覆硬體文件：不需評估 ResNet50；q 補到 15 不建議當主流程；right_shift 夾 48 已驗證。

## 12. 檔案索引

| 檔案 | 內容 |
|---|---|
| runs/confirmation_r0p95/RESULT_TEST.md | CIFAR-10 G2–G4、Q15、V3 週期、RTL |
| runs/breakthrough_r0p95/PLAN_STEM.md | conv1 色盲診斷、G3 計畫與判定 |
| runs/breakthrough_r0p95/PLAN_V3.md | fill15、budget15、G4 計畫與判定 |
| runs/other_datasets/PLAN.md | ImageNette、ImageWoof、CIFAR-100 ResNet34/50 |
| runs/other_datasets/PLAN_H.md | 計畫 H–L、統計、V3 下限 |
| research_notes/RESEARCH_FINDINGS_MASTER_2026-11-05.md | 全部研究的總檔 |
| ../mask_aware_conv_accelerator/results/v3_cycles/ | V3 週期預測 |
| ../mask_aware_conv_accelerator/results/rtl_full_model/README.md | RTL 逐位元驗證 |
