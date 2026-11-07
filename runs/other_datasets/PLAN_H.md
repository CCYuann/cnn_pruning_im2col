# 計畫 H：CIFAR-100 上的主方法與 V3 週期感知配額（2026-11-07 凍結，新組別跑之前）

## 動機
- CIFAR-10 的方法差距接近雜訊：fill15 98% 開發 +0.42pp，新 seeds -0.09pp。
- CIFAR-100 ResNet34 能分辨方法：隨機 mask 明顯輸，gradual+q 下限 99% 對 rewind +2.34pp（3/3）。
- 同樣位置數下 V3 週期可差 25%（seed1 fs 98% 16.5 ms；seed9 20.6 ms），來自位置落在哪一層。

## 揭露
寫下時已讀：CIFAR-100 seeds 0/1/2 的 rewind、gradual+floor、unstructured、random test。
新組別在這三個 seeds 上只用 validation 判定；判定後的正式數字必須來自新 seeds 3/4/5。

## 組別（ResNet34，CIFAR-100，seeds 0/1/2，gradual，60 epochs）
- floor（已完成）：90/95/98/99。
- fs：floor + conv1 全保留，95/98/99。
- b15：每 cluster 至少 15 位置 + conv1 全保留，總權重不變，98/99。
- mac0.25、mac0.5：fs，配額改用 |w|/(Hout*Wout)^beta，98/99。
共 27 條。開發組不存模型，省磁碟。

## 判定（只用 validation，配對，週期為 V3 公式預測）
- H1 fs 對 floor：98%、99% 各自平均 >= +0.3pp 且 3/3 為正。
- H2 b15 對 fs：99% 週期至少少 5%，validation 平均 >= -0.1pp。98% 只描述。
- H3 同週期增益：每個 seed 用 fs 95/98/99 在 log(ms) 上線性內插，取 mac 組同週期的 validation 當基準。
  平均 >= +0.3pp 且 3/3 為正才算通過。超出 fs 週期範圍者不外插，只描述。
通過者才進新 seeds 確認，未通過者照報。

## H 判定（2026-11-07 05:00，validation，seeds 0/1/2，test 未讀）
| 組別 | 95% | 98% | 99% |
|---|---|---|---|
| floor | 73.90 / 25.71 ms | 71.56 / 12.38 ms | 69.19 / 8.27 ms |
| fs | 74.33 / 26.00 | 71.98 / 12.53 | 69.25 / 8.49 |
| b15 | | 72.15 / 12.34 | 69.93 / 8.12 |
| mac0.25 | | 69.33 / 10.05 | 66.03 / 7.57 |
| mac0.5 | | 66.35 / 9.65 | 61.67 / 7.50 |
- H1 未通過：fs 對 floor 98% -0.48/+0.36/+1.38，99% +0.80/-0.88/+0.28。CIFAR-100 的 conv1 本來就保有 RGB，fs 沒有額外增益，只修色盲。
- H2 未通過：b15 99% 週期 -4.40%（門檻 -5%），validation +0.68pp（-0.12/+1.12/+1.04）。與 CIFAR-10 test 的 -0.61pp 方向相反，兩邊都接近雜訊。
- H3 未通過：mac0.25 98% 同週期 -1.34/-1.08/-0.86，mac0.5 98% -4.33/-3.27/-3.76。99% 兩組低於 fs 週期範圍，不外插，但精度已掉 3 至 7.5pp。結論：把位置從前段搬到後段的週期收益，抵不過精度損失。
validation 5000 張、精度約 70% 時單模型標準誤約 0.65pp，差值 1pp 內不應過度解讀。

## V3 週期下限（公式：31 + 15 × tiles，kept_count 不影響）
ResNet34 5.73 ms（最多 74.8×），ResNet50 18.95 ms（最多 25.3×），32×32 輸入，50 MHz。
ResNet50 的 1×1 層多，下限高於 ResNet34 99% 的 12.34 ms，在 V3 上再剪也追不上。

# 計畫 I：高剪枝率的訓練配方（2026-11-07 凍結，新組別跑之前）
觀察：CIFAR-100 gradual+floor 99% 的 train accuracy 僅約 85%，訓練端有欠擬合跡象，WD 5e-4 與 60 epochs 都沿用 Dense 配方。
基準：gradual+floor，98/99%，seeds 0/1/2。新組別只改一項，不存模型：
- kd：Dense teacher，T=4，alpha=0.9（沿用既有 kd_loss，CIFAR-10 上 rewind_kd 未過，此處測 100 類）。
- wd1e-4：weight decay 1e-4。
- e120：120 epochs，cosine 曲線等比例拉長，漸進區間仍為前 2/3。
判定（validation，配對）：平均 >= +0.7pp 且 3/3 為正。門檻較 H 高，因 CIFAR-100 validation 雜訊較大，且 fill15 在 CIFAR-10 的開發增益 +0.42pp 未在新 seeds 重現。
通過者進新 seeds 3/4/5（需訓練新 Dense）確認，才讀 test。

## I 判定（2026-11-07 07:35，validation，seeds 0/1/2，test 未讀）
基準 gradual+floor：98% 71.56（train 95.5），99% 69.19（train 84.7）。門檻：平均 >= +0.7pp 且 3/3 為正。
| 組別 | 98% 差值 | 平均 | 99% 差值 | 平均 |
|---|---|---:|---|---:|
| kd | +1.36/+1.78/+1.88 | **+1.67，通過** | -0.24/-0.04/+0.94 | +0.22，未過 |
| wd1e-4 | -0.84/-0.54/-0.34 | -0.57，未過 | -0.88/-1.44/-0.68 | -1.00，未過 |
| e120 | -0.20/-0.56/+0.90 | +0.05，未過 | +0.28/-0.38/+1.04 | +0.31，未過 |
欠擬合假設不成立：e120 99% train 91.1%、validation 僅 +0.31pp；降低 WD 更差。KD 的 train accuracy 反而較低（98% 90.9%），效果較接近正則化。

# 計畫 J：KD 新 seeds 確認（2026-11-07 凍結，seeds 3/4/5 尚未產生）
CIFAR-100 ResNet34，新 Dense seeds 3/4/5（100 epochs，配方不變）。組別：rewind、gradual+floor、gradual+floor+kd，98/99%，共 18 條。
kd 參數不變（T=4，alpha=0.9）。先看 validation、寫下判定，才讀 test 與 Q15，每模型一次。
- J1（主判定）：kd 98% 對 gradual+floor 98%，validation 平均 >= +0.7pp 且 3/3 為正。
- 探索（不判定）：kd 99%。開發時未過，納入是為完整報告曲線。
- 報告：kd 98% 的 Q15 test 對 Dense、rewind 的配對差。KD 不改 mask 的產生方式，mask 仍由權重大小決定，週期照實計算。

## J 判定（2026-11-07 09:50，validation，新 seeds 3/4/5，test 與 Q15 尚未讀）
- J1 通過：kd 98% 對 gradual+floor 98% +1.12/+0.18/+1.94，平均 +1.08pp（門檻 >= +0.7 且 3/3）。seed4 的差值小，3/3 成立但不穩。
- 探索：kd 99% -0.14/-0.34/+0.18，平均 -0.10pp，99% 無增益，與開發結果一致。
- gradual+floor 對 rewind（validation）：98% +1.85pp，99% +2.23pp，3/3，再次成立。

揭露（2026-11-07 10:00）：彙整 ResNet50 時，summarize_datasets.py 同時印出 fs 組 seeds 0/1/2 的 test，違反「新組別在這三個 seeds 上只用 validation」。H1 判定早已依 validation 寫下，未更動。數字：98% 72.50（floor 72.42），99% 69.77（floor 70.31），與未通過的結論一致。

## J test 與 Q15（2026-11-07 10:00，判定寫下後才讀，每模型一次，新 seeds 3/4/5）
Dense FP32 test 76.42/76.86/76.83，平均 76.70。週期為 V3 公式預測，Dense 428.51 ms。rewind 模型未保留，僅有 FP32。
| 組別 | FP32 | Q15 | ms | 倍數 |
|---|---:|---:|---:|---:|
| rewind 98% | 70.71 | 無 | 20.14 | 21.3 |
| gradual+floor 98% | 72.19 | 72.07 | 13.68 | 31.3 |
| **gradual+floor+kd 98%** | **73.15** | **73.12** | **12.85** | **33.3** |
| rewind 99% | 67.42 | 無 | 12.87 | 33.3 |
| gradual+floor 99% | 69.89 | 69.68 | 8.75 | 49.0 |
| gradual+floor+kd 99% | 69.79 | 69.65 | 8.35 | 51.3 |
配對：kd 對 gradual+floor 98% Q15 +1.64/+0.61/+0.89（+1.05pp），FP32 +1.44/+0.29/+1.13（+0.95pp）；99% Q15 +0.52/-0.12/-0.50（-0.03pp）。
kd 對 rewind 98% FP32 +2.71/+2.05/+2.56（+2.44pp）。gradual+floor 對 rewind FP32：98% +1.48pp，99% +2.47pp，3/3。
結論：KD 的增益在 98% 成立（開發與新 seeds、validation 與 test 皆為 3/3），週期少 6%；99% 無增益。KD 只改訓練，不動硬體與資料包。

# 計畫 K：KD 的適用範圍（2026-11-07 凍結，新組別跑之前）
已知：CIFAR-100 98% KD 有效、99% 無效；CIFAR-10 95% rewind_kd 曾低 1.23pp。尚不知 KD 能否用於 CIFAR-10 主線，也不知 99% 無效是否因 alpha 太高。
- K1 CIFAR-10：fs+kd（T=4，alpha=0.9），98/99%，seeds 6/7/8，對既有 fs。validation 平均 >= +0.3pp 且 3/3 為正（CIFAR-10 validation 標準誤約 0.33pp）。
  通過的比例才跑 seeds 9/10/11 確認，新 seeds 的 validation 同門檻，判定後才讀 Q15 test。
- K2 CIFAR-100 alpha：kd alpha=0.5，98/99%，seeds 0/1/2。98% 對 kd alpha=0.9，99% 對 gradual+floor，validation 平均 >= +0.7pp 且 3/3。
- K3 CIFAR-100 曲線：kd alpha=0.9，90/95%，seeds 0/1/2，對 gradual+floor，同上門檻。
K2、K3 通過者進新 seeds 3/4/5 確認。全部開發組不存模型。共 18 條開發，K1 通過時再加確認。

## K 判定（2026-11-07 11:10，validation，test 未讀）
- K1 未通過，不跑確認：CIFAR-10 fs+kd 對 fs，98% -0.16/-0.54/-0.70（-0.47pp），99% -0.42/-1.16/-0.14（-0.57pp），seeds 6/7/8，3/3 為負。與 rewind_kd 95% 的結果一致。KD 的增益不適用於 CIFAR-10，CIFAR-10 主線維持 fs。
- K2：kd alpha=0.5 99% 對 gradual+floor +0.76/+0.26/+1.42（+0.81pp），**通過**。98% 對 alpha=0.9 +0.04/-0.28/-0.24（-0.16pp），未過。
- K3：kd 90% 對 gradual+floor +1.26/+0.50/+1.18（+0.98pp），**通過**；95% +1.20/+1.08/+1.18（+1.15pp），**通過**。

# 計畫 L：KD 新 seeds 確認（2026-11-07 凍結，下列組別在 seeds 3/4/5 尚未產生）
CIFAR-100 ResNet34，seeds 3/4/5，既有 Dense。先看 validation、寫下判定，才讀 test 與 Q15，每模型一次。
- L1：kd alpha=0.5 99% 對既有 gradual+floor 99%。
- L2：kd alpha=0.9 95% 對新跑的 gradual+floor 95%。
- L3：kd alpha=0.9 90% 對新跑的 gradual+floor 90%。
門檻皆為 validation 平均 >= +0.7pp 且 3/3 為正。
- 探索（不判定）：kd alpha=0.5 98%，看能否用單一 alpha 涵蓋全部比例。
gradual+floor 90/95% 不存模型（只比 FP32，省磁碟），共 18 條，保留 12 個模型約 1G。

## L 判定（2026-11-07 12:10，validation，新 seeds 3/4/5，test 與 Q15 尚未讀）
- L1 未通過：kd alpha=0.5 99% +0.02/+0.16/+0.76，平均 +0.31pp（門檻 +0.7），開發的 +0.81pp 未重現。
- L2 未通過：kd 95% +2.36/-0.08/+1.94，平均 +1.41pp，但 seed4 為負，不滿足 3/3。
- L3 通過：kd 90% +1.94/+0.40/+1.04，平均 +1.13pp。
- 探索：alpha=0.5 98% 對 gradual+floor +0.52/+0.40/+0.92（+0.61pp），對 alpha=0.9 -0.60/+0.22/-1.02（-0.47pp），單一 alpha=0.5 不如 0.9。

## L test 與 Q15（2026-11-07 12:15，判定寫下後才讀，每模型一次，seeds 3/4/5）
Dense 76.70%，428.51 ms。週期為 V3 公式預測。
| 組別 | 基準 FP32 | KD FP32 | 配對差 | KD Q15 | ms：基準 → KD | 判定 |
|---|---:|---:|---|---:|---|---|
| 90%，alpha 0.9 | 75.44 | 76.30 | +0.53/+1.11/+0.96（+0.87） | 76.29 | 54.41 → 50.48 | L3 通過 |
| 95%，alpha 0.9 | 74.19 | 75.40 | +1.38/+0.95/+1.29（+1.21） | 75.32 | 28.92 → 26.99 | L2 未過 |
| 98%，alpha 0.9 | 72.20 | 73.16 | +1.49/+0.29/+1.10（+0.96） | 73.12 | 13.68 → 12.85 | J1 通過 |
| 98%，alpha 0.5 | 72.20 | 72.93 | +1.00/+0.43/+0.76（+0.73） | 72.83 | 13.68 → 12.80 | 探索 |
| 99%，alpha 0.9 | 69.87 | 69.76 | +0.40/-0.31/-0.41（-0.11） | 69.65 | 8.75 → 8.35 | 探索 |
| 99%，alpha 0.5 | 69.87 | 69.83 | +0.24/-0.53/+0.18（-0.04） | 69.64 | 8.75 → 8.38 | L1 未過 |
L2 依規則未過，判定不改，但 test 3/3 為正且 +1.21pp，應報為「validation 差一個 seed、test 支持」。
99% 兩種 alpha 的 test 增益皆為零，開發時的 +0.81pp 是雜訊。KD 只在 90 至 98% 有效。
基準欄與 KD FP32 取自 result.json 的 final_test，與 J 表用 final_q15_cifar100 重算的 FP32 有 0.01pp 差異（AMP）。

## 統計與 RTL（2026-11-07 12:30）
配對 test 差值，FP32，明細 BOOTSTRAP_FINAL.json。影像區間只含 test 抽樣，seed 區間只含 seed 變異，n 小時 seed 區間很寬。
| 比較 | n | 差值 | 影像 95% | seed 95% |
|---|---:|---:|---|---|
| CIFAR-10 fs－rewind 98% | 6 | +0.39 | 0.19 至 0.59 | 0.08 至 0.70 |
| CIFAR-10 fs－rewind 99% | 6 | +1.06 | 0.82 至 1.30 | 0.76 至 1.36 |
| CIFAR-100 gradual+floor－rewind 98% | 3 | +1.09 | 0.62 至 1.55 | 0.18 至 2.01 |
| CIFAR-100 gradual+floor－rewind 99% | 3 | +2.32 | 1.80 至 2.81 | 0.73 至 3.90 |
| CIFAR-100 kd－gradual+floor 98% | 3 | +0.96 | 0.53 至 1.41 | -0.55 至 2.47 |
KD 的 seed 區間含 0。支持證據是兩批 seeds（開發 +1.67pp、確認 +1.08pp，validation）皆 3/3，不是單批的顯著性。
RTL：KD 98% seed3 一張影像 36/36 逐位元通過，見加速器 repo results/rtl_full_model/README.md。
