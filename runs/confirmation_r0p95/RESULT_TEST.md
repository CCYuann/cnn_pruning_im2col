# 結果：新Dense seeds的test（只評估一次）

ResNet50/CIFAR-10完整10000張test，seed3/4/5，95%。方法與Q15校準方式都在看test之前定案。Q15為依硬體數值規格的整數模擬，含整數分類頭，非RTL逐週期實測。

| 組別 | FP32 seed3/4/5 (%) | FP32平均 | 對rewind (pp) | Q15平均 | Q15損失 (pp) |
|---|---|---:|---:|---:|---:|
| Dense | 94.22/94.18/94.06 | 94.153 | +0.563 | - | - |
| unstructured | 94.47/94.06/94.47 | 94.333 | +0.743 | - | - |
| rewind | 93.82/93.33/93.62 | 93.590 | 基準 | 93.530 | 0.060 |
| group | 93.58/93.45/93.83 | 93.620 | +0.030（-0.24/+0.12/+0.21） | 93.573 | 0.047 |
| group_gradual | 94.09/93.20/93.76 | 93.683 | +0.093（+0.27/-0.13/+0.14） | 93.507 | 0.177 |

結論：
- validation上的+0.27pp在test沒有重現，不能宣稱Group提高精度。
- 95%非共享mask在test與Dense相當（+0.18pp）；共享mask與Dense差約0.56pp。
- group_gradual精度與rewind相當，但MAC減少91.48%對88.66%，剩餘MAC少約25%。這是目前站得住的好處。
- 本批validation比test約高1pp，論文精度應引用test。

# 98%確認與DPF（2026-11-06）

依PLAN.md凍結條件，validation上對rewind平均 >= +0.5pp 且3/3為正才通過；不論通過與否各評估一次test。

| 組別 | validation seed3/4/5 (%) | val對rewind (pp) | 判定 | test seed3/4/5 (%) | test平均 | test對rewind (pp) | Q15 test | 卷積ms | MAC減少 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| rewind_r98 | 92.04/92.48/92.40 | 基準 | - | 91.99/91.55/91.96 | 91.833 | 基準 | 91.737 | 32.87 | 94.26% |
| group_r98 | 93.00/92.58/92.78 | +0.480（+0.96/+0.10/+0.38） | 未過（差0.02pp） | 92.81/92.01/92.48 | 92.433 | +0.600（+0.82/+0.46/+0.52） | 92.330 | 31.99 | 94.58% |
| group_gradual_r98 | 94.10/93.08/93.26 | +1.173（+2.06/+0.60/+0.86） | 通過 | 93.40/92.06/92.96 | 92.807 | +0.973（+1.41/+0.51/+1.00） | 92.680 | 28.47 | 96.31% |

配對檢查（`PAIRED_BOOTSTRAP.json`，影像層級bootstrap 20000次，重新推論與表內差異<=3張）：
- group_gradual_r98對rewind_r98：+0.98pp，95% CI [+0.67,+1.28]；McNemar seed3 p=1.5e-8、seed4 p=0.056、seed5 p=1.2e-4。
- group_r98對rewind_r98：+0.61pp，95% CI [+0.32,+0.90]。
- 此CI只涵蓋測試影像抽樣，不涵蓋seed變異（n=3）；seed層級差值標準差約0.45pp。

結論：
- 98%下group_gradual對rewind有確認過的精度增益，且剩餘MAC少約36%、卷積時間少約13%。
- group在98%未過validation門檻，test雖為正，只能當描述，不算通過。
- group_gradual seed4有3層、seed5有1層全q=0（皆在layer4.2）；必須靠bias-only路徑與residual寬位元相加才能完整上硬體。

## DPF（探索性，seed3/4/5，95%）
validation 94.66/93.98/93.94，平均94.193（對rewind +0.047）；test 93.91/93.67/93.83，平均93.803（+0.213，3/3為正）；Q15 93.743；50.72 ms；MAC減少90.59%。
配對bootstrap +0.21pp，95% CI [-0.04,+0.46]，未排除0。DPF是事後加入的探索組，不算主判定。

# 第E輪確認（2026-11-06，seed3/4/5，test只評估一次）

| 組別 | validation平均 | test seed3/4/5 (%) | test平均 | 對同率rewind (pp) | Q15 test平均（MSE scale） | ms | MAC減少 |
|---|---:|---|---:|---:|---:|---:|---:|
| unstructured_r98 | 94.633 | 94.21/93.82/93.97 | 94.000 | 參考線 | - | - | 94.26% |
| rewind_r98 | 92.307 | 91.99/91.55/91.96 | 91.833 | 基準 | 91.737 | 32.87 | 94.26% |
| gradual_r98 | 93.300 | 93.22/92.46/92.69 | 92.790 | +0.957（+1.23/+0.91/+0.73） | 92.623 | 27.86 | 96.41% |
| group_gradual_r98 | 93.480 | 93.40/92.06/92.96 | 92.807 | +0.973 | 92.680 | 28.47 | 96.31% |
| ggrad_r98 lam1e-4 | 93.347 | 93.12/92.21/93.11 | 92.813 | +0.980 | 92.483 | 29.73 | 96.11% |
| unstructured_r99 | 93.813 | 93.21/92.90/93.12 | 93.077 | 參考線 | - | - | 96.51% |
| rewind_r99 | 89.873 | 89.07/89.95/89.81 | 89.610 | 基準 | 89.470 | 24.08 | 96.51% |
| gradual_r99 | 91.900 | 92.32/90.51/91.21 | 91.347 | +1.737（+3.25/+0.56/+1.40） | 90.753（診斷中） | 20.71 | 98.09% |
| group_gradual_r99 | 91.640 | 91.86/88.76/91.30 | 90.640 | +1.030（+2.79/-1.19/+1.49） | 異常（seed4僅19.96%，診斷中） | 21.00 | 98.04% |

配對bootstrap（不含seed變異）：gradual_r98對rewind_r98 +0.97pp [+0.66,+1.27]；gradual_r99對rewind_r99 +1.73pp [+1.39,+2.08]；group_gradual_r99對rewind_r99 +1.03pp [+0.68,+1.39]；group_gradual_r98對gradual_r98 +0.02pp [-0.25,+0.28]。

結論：
- gradual是主方法。98%與99%都通過validation門檻，test重現，三seed皆正；group懲罰在98%沒有貢獻（+0.02pp），在99%反而較差。
- lambda 1e-4未過。
- 共享mask代價（test）：98% gradual距unstructured 1.21pp，rewind距2.17pp；99% gradual距1.73pp，rewind距3.47pp。
- 表中Q15欄是MSE scale的第一次數字，99%有失效，定案數字見下節。seed4 group_gradual_r99的FP32為88.76%、Q15為19.96%；原先「輸入被裁切」的解釋已被否定（MinMax不裁切但更差）。實際原因是BN gain極大的channel依賴大bias與大權重互相抵消，8-bit權重捨入造成系統性輸出偏移。

# Q15定案流程（2026-11-06）

流程：per-tensor MSE scale + 分支內cross-layer equalization（k<=8）+ 逐層bias correction（256張校準影像，取自validation之外）。只改權重、bias與scale數值，mask不變，不需改RTL；RTL尚未以此流程實測。
流程依validation 5個模型選定：MSE損失平均14.1pp（被一個失效模型主導）、equalize k=8為7.05pp（4模型）、bias correction 0.34pp、兩者合用0.27pp。
TEST_Q15_FINAL.csv是修正校準方式後的第二次test評估，揭露於此；未用來選模型或調參。peak accumulator約5.4e8，未超過INT32。

見 `TEST_Q15_FINAL.csv`。

# q下限判定（PLAN F，seed3/4/5）

| 組別 | validation | test FP32 | Q15 test | ms | q=0 cluster | 全q=0層 |
|---|---:|---:|---:|---:|---:|---:|
| gradual_r98 | 93.300 | 92.790 | 92.750 | 27.86 | 1032 | 4 |
| gradual_r98_floor | 93.300 | 92.940 | 92.663 | 27.82 | 0 | 0 |
| gradual_r99 | 91.900 | 91.347 | 90.897 | 20.71 | 1350 | 6 |
| gradual_r99_floor | 92.147 | 91.313 | 91.280 | 20.64 | 0 | 0 |

判定：98%與99%皆達非劣（validation +0.000與+0.247pp、q=0為0、週期未增加），硬體版主方法改用floor。兩者都未達精度增益門檻（需 >= +0.3pp 且3/3），不宣稱floor提高精度。
對同率rewind，floor的Q15 test：98% +0.917pp（+1.05/+1.15/+0.55），99% +1.913pp（+2.37/+1.37/+2.00）。

## ResNet34 確認（G2，seed3/4/5，test 一次）

| 組別 | FP32 test（3 seeds） | 平均 | Q15 test 平均 | ms（50 MHz） | MAC 減少 |
|---|---|---:|---:|---:|---:|
| Dense | 95.08/94.73/94.96 | 94.923 | | 454.6 | 0 |
| rewind 80% | 95.00/94.73/94.78 | 94.837 | 94.810 | 153.82 | 63.27% |
| rewind 95% | 94.48/93.79/94.53 | 94.267 | 94.303 | 55.62 | 86.76% |
| rewind 98% | 93.37/93.00/93.53 | 93.300 | 93.250 | 28.77 | 93.31% |
| gradual 98% | 93.64/91.33/93.05 | 92.673 | 92.573 | 20.96 | 95.52% |
| unstructured 98% | 94.37/94.11/94.58 | 94.353 | | | |

既定判定（ResNet34 rewind 98% 的 Q15 test 較高且週期較低，才可說優於 ResNet50 gradual 98%）：**不成立**。
Q15 93.25 對 92.75（+0.50pp），但週期 28.77 對 27.86 ms（+3.3%）；非配對，兩者無支配關係，主模型不由此定案。
共享 mask 代價（對 unstructured，FP32 test）：rewind 1.05pp、gradual 1.68pp。

gradual 對 rewind（98%，FP32 test）：+0.27/-1.67/-0.48，平均 -0.627pp。seed4 是色盲模型（見下），依計畫不重跑、不排除。

## Stem 飢餓（事後診斷，非事前假設）

ResNet34 gradual 98% seed4 的 conv1 只剩 4 個位置，全在 G 通道；R、B 換成雜訊後 validation 90.94% 與乾淨資料相同，seed3 同樣處理掉到 10.14%。
181 個模型掃描：ResNet34 gradual 在 80–99% 的 conv1 都只剩 3–5 個位置（rewind 98% 約 77）；另有 7 個 gradual 類模型缺 R 或 B。
這是 gradual 與 group_gradual 的共同風險，全域 L1 排序不知道輸入層資訊不可取代。後續見 runs/breakthrough_r0p95/PLAN_STEM.md。

## peak_acc 更正

TEST_Q15_FINAL.csv 的 peak_acc 約 2^29，並非卷積累加峰值，而是近乎死掉 channel 的 bias（設計上限 2^28，bias correction 最多再加一倍）。
分開量測（一批 validation，ResNet34 rewind 98%、ResNet50 gradual 98%）：卷積和峰值 1.54e5、1.78e5。加上 bias 仍低於 INT32，不溢位的結論不變。

## G3：ResNet34 色彩覆蓋確認（seeds 6/7/8，全新 Dense，test 一次）

Dense FP32 test 94.68/94.97/94.88，平均 94.843%，454.6 ms。fs = gradual + q 下限 + stem 全保留。

| 比例 | 方法 | FP32 test | Q15 test | ms |
|---|---|---:|---:|---:|
| 95% | rewind | 94.373 | 94.367 | 58.33 |
| 95% | gradual | 94.217 | 94.210 | 48.53 |
| 95% | fs | 94.617 | 94.543 | 48.61 |
| 98% | rewind | 93.480 | 93.487 | 29.99 |
| 98% | gradual | 93.067 | 93.000 | 22.79 |
| 98% | gradual+floor | 93.533 | 93.377 | 22.79 |
| 98% | fs | 93.793 | 93.660 | 22.98 |
| 99% | rewind | 91.670 | 91.597 | 18.51 |
| 99% | gradual | 91.950 | 91.877 | 13.22 |
| 99% | fs | 92.587 | 92.577 | 13.54 |

配對（Q15 test，seeds 6/7/8）：
- fs 對 gradual：95% +0.88/+0.09/+0.03，98% +0.63/+1.16/+0.19，99% +0.65/-0.05/+1.50。
- fs 對 rewind：95% +0.38/+0.20/-0.05，98% +0.41/+0.24/-0.13，99% +1.05/+0.61/+1.28。

判定：C1、C2、C3 於 validation 通過（PLAN_STEM.md）。C4 通過：ResNet34 fs 98% Q15 93.660%、22.98 ms，對 ResNet50 gradual+floor 98% 92.663%、27.82 ms，高 0.997pp 且週期少 17.4%。
限制：C4 為不同 seeds 的非配對比較；fs 對 rewind 的 98% 增益不是 3/3；stem 假設由 seed4 異常事後提出，G3 才是獨立檢驗。fs 對 floor 的差距在 98% 為 validation +0.133pp、Q15 test +0.283pp，不足以說全保留比 q 下限準，採用 fs 的理由是保證 RGB 齊全。

## V3 週期重算（2026-11-06，取代本檔先前的 ms）

本檔先前的 ms 皆為 V2.3 模型。V3：cluster = 31 + tiles×max(q,15)，重現硬體文件 ResNet34 Dense 21,425,368 拍。CSV：mask_aware_conv_accelerator/results/v3_cycles/。
| 模型 | 方法 | V3 ms（50 MHz） | 對 Dense 倍數 | q<15 |
|---|---|---:|---:|---:|
| ResNet34 | Dense | 428.51 | 1 | 0 |
| ResNet34 | rewind 80%（seeds 3-5） | 139.33 | 3.08 | 11.1% |
| ResNet34 | rewind 98% | 26.97 | 15.89 | 48.2% |
| ResNet34 | fs 98% | 20.74 | 20.66 | 50.3% |
| ResNet34 | rewind 99% | 16.81 | 25.48 | 61.8% |
| ResNet34 | fs 99% | 12.53 | 34.21 | 64.8% |
| ResNet50 | Dense | 479.71 | 1 | 0 |
| ResNet50 | gradual+floor 98%（seeds 3-5） | 27.96 | 17.15 | 73.7% |
| ResNet50 | gradual+floor 99% | 22.72 | 21.11 | 85.7% |
C4 在 V3 下仍成立且差距更大：fs 98% 對 ResNet50 gradual+floor 98% 週期少 25.8%。
V3 RTL（v3-datapath）不在本機與 origin，週期為公式預測，未在 V3 RTL 實測。

## RTL 逐位元驗證（2026-11-07）
fs 98% seed6，兩張 validation 影像，36 個卷積層 36/36 與軟體 Q15 逐位元相同；RTL 量測 1,160,791 拍，V2.3 估算差 0.35%。
範圍與限制見 mask_aware_conv_accelerator/results/rtl_full_model/README.md：RTL 為 V2.3，residual、pooling、FC 仍在軟體。
right_shift 49–55 的 channel 已夾到 48，3 個模型 5000 張 validation logits 完全相同。

## G4 確認（seeds 9/10/11，V3 週期，Q15 test）
fs 98% 93.83%/20.64 ms，99% 92.98%/12.34 ms；rewind 98% 93.37%/27.14 ms，99% 91.95%/16.97 ms。
fs 對 rewind 第三批 seeds 再次成立，99% 三個 seed 皆高 0.78 至 1.19pp。fill15 98% 未過 D1；budget15 99% 過 D2 但 Q15 test -0.61pp（3/3），不採用。明細：runs/breakthrough_r0p95/PLAN_V3.md。
