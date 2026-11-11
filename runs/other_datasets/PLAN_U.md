# 計畫 U：為什麼 gradual+floor 在高剪枝率較好（因素拆解，2026-11-10 凍結，下列新組別皆尚未產生）

背景：使用者問「為什麼我們的方法在高剪枝率精度相對較好」。gradual+floor 與 rewind 用同一個 mask 規則與同一條 LR 曲線（`--arm rewind` 與 `--arm gradual` 都用 rewind LR），差別只有三點：
1. **配額分配**：每層、每個 cluster 留幾個位置（q）。rewind 由 Dense 權重的 global unstructured 稀疏度一次決定；gradual 每 2 epochs 在訓練中的權重上重算，且每個 cluster 至少 1 個。
2. **位置選擇**：在 q 相同時，留哪些位置。rewind 依 Dense 權重的 L1；gradual 依剪枝過程中逐步調整後的權重。
3. **訓練過程**：rewind 一開始就固定 mask 訓練 60 epochs；gradual 前 40 epochs 逐步剪、最後 20 epochs 固定 mask。

事前已看過的描述性資料（`scratchpad` 分析，非判定）：兩者最終 mask 的位置重疊（Jaccard）只有 0.1–0.3；各層預算份額的差距（total variation）0.1–0.28，比例越高越大；gradual 把 conv1 和前段層的預算移到後段層；凡 gradual 贏的組別，train 精度也較高。這些是相關，不是因果，所以做以下對調。

## 新增的兩種對照（都從同一個 Dense checkpoint 出發，用 rewind 的 LR 與 60 epochs、固定 mask）
- **A1 quota_from**（`--arm rewind --quota-from <同 seed 的 gradual+floor 目錄>`）：只借 gradual 最終 mask 的每個 cluster 的 q，位置仍依 Dense L1 選。實作已驗證：以 rewind 自己的 mask 為來源時，重建結果與 rewind 完全相同（0 個位置不同）。
- **A2 mask_from**（`--arm rewind --mask-from <同 seed 的 gradual+floor 目錄>`）：直接用 gradual 最終 mask。
與既有的 A0 rewind、A3 gradual+floor 組成拆解：
- A1 − A0 = 配額分配的效果
- A2 − A1 = 位置選擇的效果
- A3 − A2 = 逐步剪枝訓練過程的效果

## 組別（CIFAR-100，seeds 0/1/2，各 2 條新工作 × 3 seeds，共 42 條，前綴 `u_`）
- 95%：ResNet-56、VGG16-BN、GoogLeNet、DenseNet121（gradual 贏 2.1–6.9pp），ResNet-110（gradual 輸 1.10pp）
- 98%：ResNet34（贏 1.10pp）
- 90%：DenseNet121（贏 4.08pp，90% 為使用者的基準比例）

## 判定（test 直接判定，每條讀一次）
每個組別、每個步驟（A1−A0、A2−A1、A3−A2）：配對平均 >= +0.3pp 且 3/3 為正 → 該因素「有正效果」；配對平均 <= −0.3pp 且 3/3 為負 → 「有負效果」；其餘為「不確定」。
另報每步佔總差（A3−A0）的比例、train 精度、V3 週期（A1、A2 與 A3 的 q 相同，週期相同）。
不設跨模型的總判定；結論以「幾個組別中哪個因素為正」描述。只有浮點精度（AMP float16 評估）。

## 指令
```
/usr/local/anaconda3/bin/python3 runs/breakthrough_r0p95/enqueue_u.py
/usr/local/anaconda3/bin/python3 scripts/decide_plan_u.py
```

## 結果（2026-11-10 完成，42/42 條成功；CIFAR-100，seeds 0/1/2，每組 n=3，浮點 test）

判定輸出：`PLAN_U_DECISION.txt`；數字：`PLAN_U_RESULTS.json`。括號內為配對平均（pp）與判定。

| 組別 | A0 rewind | A1 借配額 | A2 借整個 mask | A3 gradual+floor | 總差 A3−A0 | 配額分配 A1−A0 | 位置選擇 A2−A1 | 逐步過程 A3−A2 |
|---|---:|---:|---:|---:|---:|---|---|---|
| ResNet-56 95% | 59.13 | 56.80 | 60.76 | 63.29 | +4.16 | −2.33 負 | +3.96 正 | +2.53 正 |
| VGG16-BN 95% | 68.88 | 69.04 | 70.62 | 71.02 | +2.14 | +0.17 不確定 | +1.57 正 | +0.40 正 |
| GoogLeNet 95% | 70.70 | 69.64 | 71.84 | 73.37 | +2.67 | −1.07 負 | +2.20 正 | +1.53 正 |
| DenseNet121 95% | 68.51 | 66.63 | 69.26 | 75.43 | +6.91 | −1.88 負 | +2.63 正 | +6.16 正 |
| ResNet-110 95% | 64.06 | 60.84 | 61.62 | 62.96 | −1.10 | −3.23 負 | +0.78 不確定 | +1.34 正 |
| ResNet34 98% | 71.32 | 68.51 | 71.25 | 72.42 | +1.10 | −2.80 負 | +2.74 正 | +1.17 正 |
| DenseNet121 90% | 73.06 | 70.06 | 73.84 | 77.14 | +4.08 | −3.00 負 | +3.78 正 | +3.29 正 |

- **逐步剪枝的訓練過程 7/7 為正**（+0.40 至 +6.16pp），是唯一在每個組別都成立的因素；在 DenseNet121 95% 佔總差的 89%、ResNet34 98% 佔 106%。
- **位置選擇 6/7 為正**（+1.57 至 +3.96pp）：同樣的 q，依剪枝過程中調整後的權重選位置，比依 Dense 權重選好。
- **配額分配單獨拿出來 6/7 為負**（−1.07 至 −3.23pp），VGG16-BN 不確定。gradual 的每組配額只有搭配它自己選的位置才有用；套用 Dense L1 選位置反而比 rewind 差。這推翻了事前「預算分配是主因」的猜測。
- A2（只換成 gradual 的最終 mask、從 Dense 重新訓練）對 rewind：ResNet-56 +1.63、VGG16-BN +1.74、GoogLeNet +1.14、DenseNet121 95% +0.75、DenseNet121 90% +0.78、ResNet34 98% −0.07、ResNet-110 −2.45（事後由表相減，未判定）。也就是說 mask 本身只解釋一部分，其餘來自逐步剪枝期間權重與 mask 一起調整的過程。
- ResNet-110 95% 輸 rewind 的原因：配額分配的負效果（−3.23）大於過程的正效果（+1.34），位置選擇不穩定（+0.68/−2.24/+3.90）。
- train 精度的順序在 7 組中幾乎都是 A1 < A0 < A2 < A3，增益伴隨更好的擬合，不是正則化。

### 保留
- 拆解的順序（先換配額、再換位置、再換過程）是一種選擇；換一個順序，各因素的數值可能不同，交互作用沒有單獨估計。
- A2 從 Dense 權重重新開始；A3 的最後 20 epochs 是從逐步剪枝後的權重接續，兩者的訓練長度相同（60 epochs）但起點不同，「過程」包含了這個起點差異。
- 只有 seeds 0/1/2；未在新 seeds 重現。週期：A1、A2 與 A3 的 q 相同，V3 週期相同。
