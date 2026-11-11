# 計畫 M：原論文流程基準、KD 重現、新模型（2026-11-08 凍結，下列組別皆尚未產生）

## 更正
先前文件說 rewind 基準「與原論文流程不完全相同（配額為全域 cluster 排序）」，這是錯的。
rewind 的 mask 來自 `compensation_ratio_worker.make_masks`：先做 global unstructured 剪枝，每個 cluster 照抄其稀疏度，cluster 內以 L1 加總選位置，與原論文第 3 節的規則相同。
與原論文唯一的差別是重訓：rewind 用 LR rewinding（60 epochs，cosine 後段），原論文只寫「fine-tune」，未給 LR。

## M1：原論文流程（照原文的 fine-tune）
ft_const：mask 與 rewind 相同，固定 LR 0.001 微調 60 epochs。LR 0.001 是本研究的假設（原文未給），沿用 2026-11-05 的舊微調對照。
- CIFAR-100 ResNet34，90/95/98/99%，seeds 0/1/2，不存模型。
- CIFAR-10 ResNet34，98/99%，seeds 6/7/8，保留模型做 Q15。
組別固定、沒有選擇，test 於結束時讀一次。
主張成立條件：gradual+floor（CIFAR-100）或 fs（CIFAR-10）對 ft_const 配對平均 >= +0.3pp 且 3/3 為正，逐比例判定。

## M2：KD 重現（正確 Dense 配方）
CIFAR-100 ResNet34，新 Dense seeds 6/7/8（100 epochs，完整 cosine）。
組別：gradual+floor 與 gradual+floor+kd（T=4，alpha=0.9）在 90/95/98%，rewind 98%，共 21 條；只保留 98% 的兩組模型做 Q15。
判定（validation，配對）：各比例 kd 對 gradual+floor 平均 >= +0.7pp 且 3/3 為正。判定寫下後才讀 test 與 Q15。
報告：98% 合併 seeds 3–8（n=6）的配對 test 差與 bootstrap 區間；seeds 3/4/5 的 Dense 配方不同（見 PLAN_H.md 揭露），分開列出。

## M3：新模型（GoogLeNet、DenseNet121，CIFAR-100）
兩者皆為原論文用過的模型，卷積層全部符合 V3（1×1/3×3、groups=1、輸出 channel 為 8 的倍數）。
配方完全沿用 ResNet34，不調參：Dense 100 epochs；rewind 與 gradual+floor 在 90/98/99%，unstructured 在 98/99%；seeds 0/1/2。每個模型 27 條。
組別固定，test 於結束時讀一次。判定：gradual+floor 對 rewind 配對平均 >= +0.3pp 且 3/3 為正，逐比例判定。
報告：對 Dense、對 unstructured 的掉幅，V3 週期。只有 FP32，Q15 整數模擬目前只支援 ResNet。
限制：DenseNet121 單層輸入最多 262,144 個 INT8，超過 Activation SRAM 65,536，實際上機需要空間切塊。

## M2 判定（2026-11-09 09:25，validation，seeds 6/7/8，正確 Dense 配方，test 與 Q15 尚未讀）
- 98% 通過：kd 對 gradual+floor +0.94/+1.50/+0.94，平均 +1.13pp。
- 90% 通過：+0.44/+1.16/+1.06，平均 +0.89pp。
- 95% 未通過：+1.30/+0.36/+0.40，平均 +0.687pp（門檻 +0.7），3/3 為正。

## M1 結果：CIFAR-100 ResNet34（test，seeds 0/1/2，Dense 76.79%）
ft_const 與 rewind 的 mask 相同，V3 週期完全一致（90% 59.60、95% 34.57、98% 17.87、99% 11.53 ms），差異只來自重訓。
| 比例 | ft_const（原文微調） | rewind | gradual+floor | gradual+floor − ft_const | 判定 |
|---|---:|---:|---:|---|---|
| 90% | 74.53 | 75.91 | 75.98 | +1.55/+1.49/+1.31（+1.45） | 成立 |
| 95% | 71.71 | 74.59 | 74.54 | +2.88/+2.49/+3.12（+2.83） | 成立 |
| 98% | 66.57 | 71.32 | 72.42 | +5.01/+6.44/+6.10（+5.85） | 成立 |
| 99% | 59.99 | 67.97 | 70.31 | +10.18/+11.12/+9.67（+10.32） | 成立 |
照原文微調時，高剪枝率的崩潰很明顯：99% 對 Dense 掉 16.8pp。LR rewinding 救回約 8pp，gradual+floor 再多 2.3pp。
ft_const 的 LR 0.001 是假設，原文未給 LR；換別的微調 LR 可能較好，這個對照代表「固定小 LR 微調」，不是原作者實際的設定。

## M3 結果：新模型（CIFAR-100，test，seeds 0/1/2，FP32）
| 模型 | Dense | 比例 | rewind | gradual+floor | 增益 | unstructured | V3 ms：rewind → gradual |
|---|---:|---|---:|---:|---|---:|---|
| GoogLeNet | 76.62 | 90% | 73.52 | 74.93 | +1.03/+1.26/+1.93（+1.41），成立 | | 22.41 → 22.65 |
| | | 98% | 64.15 | 69.67 | +4.97/+5.46/+6.13（+5.52），成立 | 70.31 | 6.64 → 6.99 |
| | | 99% | 56.23 | 64.63 | +8.08/+8.58/+8.54（+8.40），成立 | 64.85 | 5.11 → 5.20 |
| DenseNet121 | 79.39 | 90% | 73.06 | 77.14 | +4.10/+4.67/+3.47（+4.08），成立 | | 32.19 → 37.81 |
| | | 98% | 61.32 | 70.88 | +9.19/+9.59/+9.89（+9.56），成立 | 69.02 | 13.67 → 14.71 |
| | | 99% | 52.49 | 64.78 | +10.89/+12.94/+13.03（+12.29），成立 | 59.78 | 12.00 → 11.87 |
V3 週期：GoogLeNet Dense 153.47 ms、下限 4.75 ms；DenseNet121 Dense 289.05 ms、下限 10.62 ms。
- 六個比較全部成立，增益隨剪枝率變大，與 ResNet34 一致。
- GoogLeNet rewind 99% 每個 seed 都有一整層被剪光（zero layer），q=0 cluster 約 177 個；gradual+floor 沒有。
- 與 ResNet34 不同，這兩個模型的 gradual 沒有省週期（V3 下 q 多已接近 15 的下限）。
- DenseNet121 的 gradual+floor 高於 unstructured（98% +1.86pp、99% +5.00pp）。unstructured 是一次剪完 + rewind，所以這表示增益主要來自逐步剪枝，不是結構；沒有跑逐步的 unstructured，不能據此說結構化優於非結構化。
- 配方沿用 ResNet34、未針對新模型調參；rewind 在 DenseNet121 90% 已比 Dense 低 6.3pp，基準對這個模型可能偏弱。

## M2 test（判定寫下後才讀，seeds 6/7/8，正確 Dense 配方，Dense 76.95%）
| 比例 | gradual+floor | +kd | kd 配對差（FP32） | V3 ms：floor → kd | 判定 |
|---|---:|---:|---|---|---|
| 90% | 75.83 | 76.65 | +0.69/+0.34/+1.41（+0.81） | 45.73 → 43.36 | validation 通過，test 3/3 |
| 95% | 74.36 | 75.52 | +1.76/+0.50/+1.23（+1.16） | 24.47 → 23.46 | validation 差 0.013pp 未過，test 3/3 |
| 98% | 72.08 | 73.24 | +0.72/+1.07/+1.68（+1.16） | 11.88 → 11.59 | validation 通過，test 3/3 |
rewind 98% 70.94%、16.75 ms；gradual+floor 對 rewind +1.14pp，kd 對 rewind +2.30pp，皆 3/3。
KD 98% 合併 seeds 3–8（n=6）：+1.04pp，seed 區間 0.51 至 1.56，影像區間 0.71 至 1.36（BOOTSTRAP_PLAN_M.json）。區間不再含 0。
95% 在兩批 seeds 都是 validation 差一點未過、test 3/3 為正，依規則維持「未通過」，但兩批 test 一致支持。

## M1 結果：CIFAR-10 ResNet34（seeds 6/7/8，Dense FP32 94.84%，Q15 test）
| 比例 | ft_const（原文微調） | rewind | fs | fs − ft_const | V3 ms：ft/rewind → fs |
|---|---:|---:|---:|---|---|
| 98% | 91.58 | 93.49 | 93.66 | +2.80/+1.97/+1.47（+2.08），成立 | 26.97 → 20.74 |
| 99% | 89.61 | 91.60 | 92.58 | +3.33/+2.92/+2.64（+2.96），成立 | 16.81 → 12.53 |
CIFAR-10 上照原文微調在 99% 掉 5.2pp（CIFAR-100 掉 16.8pp）。fs 在更少週期下高 2 至 3pp。

## M2 Q15（seeds 6/7/8）
kd 98% 72.97%，gradual+floor 98% 71.97%，配對 +0.71/+0.87/+1.44（+1.00pp），3/3。
