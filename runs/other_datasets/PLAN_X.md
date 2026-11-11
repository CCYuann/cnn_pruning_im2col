# 計畫 X：封閉原論文流程 LR 網格的下界（2026-11-11 凍結，下列新組別皆尚未產生）

背景：計畫 W 在 {0.001, 0.003, 0.01, 0.03} 中用 validation 為原論文流程（ft_const）選固定微調 LR。8 個模型中 7 個選到最小的 0.001，validation 隨 LR 增大單調下降，所以網格下界沒有封閉：更小的 LR 可能更好。本計畫補 LR 0.0003，讓「公平 LR」的主張完整。

## 組別（CIFAR-100，90%，seeds 0/1/2，8 個模型）
ResNet18、ResNet34、ResNet50、ResNet-56、ResNet-110、VGG16-BN、GoogLeNet、DenseNet121。
新增 ft_const `--ft-lr .0003`：8 模型 × 3 seeds = 24 條，前綴 `x_`，結果目錄後綴 `ft_const_r90_lr0.0003`。其餘 LR、gradual+floor、rewind 都已存在，不重跑。

## 選 LR（只看 validation）
候選改為 {0.0003, 0.001, 0.003, 0.01, 0.03}。每個模型取 seeds 0/1/2 的 `final_validation` 平均最高者，平手取較小的 LR。
- 先執行選擇並存檔（`PLAN_X_LR_SELECTION.json`），之後才讀 test；選定後不得更改。
- 若某模型選到 0.0003（新的下界），報告要註明「下界仍未封閉」，**不再追加 LR**。

## 判定（test）
- **X1（主判定）**：gradual+floor 對「選定 LR 的 ft_const」，配對平均 >= +0.3pp 且 3/3 為正，逐模型。X1 取代計畫 W 的 W1，成為論文中「對原論文流程」的主數字。
- 選定 LR 與計畫 W 相同的模型，X1 就等於 W1（test 已讀過），要標明「沿用」。
- **X2（描述）**：rewind 對選定 LR 的 ft_const。
- 0.0003 的 test 對未選中它的模型只列附表，不能用來改選。

## 指令
```
/usr/local/anaconda3/bin/python3 runs/breakthrough_r0p95/enqueue_x.py
/usr/local/anaconda3/bin/python3 scripts/decide_plan_x.py --select   # 只讀 validation，寫 PLAN_X_LR_SELECTION.json
/usr/local/anaconda3/bin/python3 scripts/decide_plan_x.py            # 讀 test，判定 X1/X2
```

## 結果（2026-11-11 13:01，新增24/24成功，0失敗）

CIFAR-100、90%、seeds 0/1/2、每模型 n=3。X1 為 8/8 模型通過；通過：ResNet18、ResNet34、ResNet50、ResNet-56、ResNet-110、VGG16-BN、GoogLeNet、DenseNet121；未過：無。X1 取代 W1 成為論文中「對原論文流程」的主數字；W1／S1 保留為歷史對照。

沿用 W1：ResNet18、ResNet-56、ResNet-110、VGG16-BN、GoogLeNet（5/8）；新增主對照：ResNet34、ResNet50、DenseNet121（3/8）。改選模型的主對照使用新增的 LR 0.0003；其餘模型直接引用已讀過的 W1。

選到 0.0003：ResNet34、ResNet50、DenseNet121；下界仍未封閉，可能有更小的最佳 LR，依凍結計畫不再追加。

### 主表：validation 選 LR 後的 test 比較

精度為 %；配對差依 seeds 0/1/2 排列。顯示值取兩位，判定使用未四捨五入值。

| 模型 | val .0003 | val .001 | val .003 | val .01 | val .03 | 選定 LR | 沿用 W | ft test | floor test | X1 floor−ft（pp） | X2 rewind−ft（pp，只描述） |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|---:|
| ResNet18 | 71.72 | 71.83 | 71.15 | 68.67 | 65.68 | 0.001 | 沿用 W1 | 71.93 | 74.16 | +2.21/+1.92/+2.54（平均 +2.22；通過） | +2.08 |
| ResNet34 | 73.87 | 73.73 | 73.15 | 69.34 | 65.49 | 0.0003（下界） | NEW | 74.22 | 75.98 | +1.78/+1.80/+1.70（平均 +1.76；通過） | +1.69 |
| ResNet50 | 72.25 | 72.15 | 70.91 | 66.97 | 62.01 | 0.0003（下界） | NEW | 72.52 | 73.97 | +0.71/+2.38/+1.25（平均 +1.45；通過） | +1.28 |
| ResNet-56 | 58.06 | 60.83 | 61.01 | 58.47 | 54.45 | 0.003 | 沿用 W1 | 61.42 | 67.40 | +6.23/+5.88/+5.85（平均 +5.99；通過） | +3.72 |
| ResNet-110 | 62.79 | 64.57 | 63.09 | 61.64 | 56.34 | 0.001 | 沿用 W1 | 64.89 | 68.22 | +3.06/+2.63/+4.32（平均 +3.34；通過） | +2.74 |
| VGG16-BN | 68.71 | 68.88 | 67.91 | 65.48 | 60.23 | 0.001 | 沿用 W1 | 70.21 | 71.90 | +2.29/+1.25/+1.52（平均 +1.69；通過） | +1.58 |
| GoogLeNet | 69.55 | 69.84 | 68.53 | 66.65 | 61.53 | 0.001 | 沿用 W1 | 70.71 | 74.93 | +3.61/+4.45/+4.61（平均 +4.22；通過） | +2.82 |
| DenseNet121 | 73.29 | 73.18 | 72.14 | 70.00 | 65.81 | 0.0003（下界） | NEW | 73.97 | 77.14 | +3.47/+3.03/+3.00（平均 +3.17；通過） | -0.91 |

### 附表：五個 LR 的 test（不參與選擇）

| 模型 | test .0003 | test .001 | test .003 | test .01 | test .03 |
|---|---:|---:|---:|---:|---:|
| ResNet18 | 72.35 | 71.93 | 71.34 | 68.91 | 65.94 |
| ResNet34 | 74.22 | 74.53 | 73.63 | 69.64 | 66.25 |
| ResNet50 | 72.52 | 72.64 | 71.52 | 67.49 | 62.62 |
| ResNet-56 | 58.37 | 61.71 | 61.42 | 59.23 | 54.37 |
| ResNet-110 | 63.41 | 64.89 | 63.26 | 61.49 | 55.98 |
| VGG16-BN | 69.90 | 70.21 | 68.66 | 65.91 | 61.58 |
| GoogLeNet | 70.51 | 70.71 | 69.37 | 67.76 | 61.85 |
| DenseNet121 | 73.97 | 74.08 | 72.81 | 70.35 | 66.41 |

### 保留與來源

- LR 0.0003 是看過計畫 W 的 validation 後才事後新增；選擇規則在讀新增 test 前已凍結，並非完全事前的五點網格研究。
- 五個候選為 0.0003、0.001、0.003、0.01、0.03；只用 seeds 0/1/2 的 final_validation 平均選，平手取較小 LR。先單獨執行 --select、保存不可覆寫的 PLAN_X_LR_SELECTION.json，再單獨執行 test 判定；未依 test 重選。
- 原論文未提供 LR；結果只限本研究候選範圍，不代表作者實際配方。選到 0.0003：ResNet34、ResNet50、DenseNet121；下界仍未封閉，可能有更小的最佳 LR，依凍結計畫不再追加。
- 沿用 W1：ResNet18、ResNet-56、ResNet-110、VGG16-BN、GoogLeNet（5/8）；新增主對照：ResNet34、ResNet50、DenseNet121（3/8）。改選模型的主對照使用新增的 LR 0.0003；其餘模型直接引用已讀過的 W1。 floor／rewind 均引用 S；沿用 W1 的 test 先前已讀過，不把它們算成新結果。X 使用相同 seeds，並非新 seeds 的獨立重現。
- X1 門檻維持配對平均 ≥ +0.3pp 且 3/3 為正；每模型 n=3，門檻不是統計顯著性檢定。X2 只描述；附表各 LR test 不參與選擇，未選 0.0003 者的該 test 僅供附表。
- 僅浮點結果，worker 在 CUDA 以 AMP float16 評估；新模型沒有 Q15 流程。
- V3 週期沿用 S 的公式預測：每 cluster 31 + Hout × ceil(Wout/8) × max(q,15) 拍，50 MHz、僅卷積層，非 RTL 實測。GoogLeNet／DenseNet121 的 floor 比 ft/rewind 多約 1.1%／17.5% 週期，不能稱為同週期增益。

選擇輸出：PLAN_X_SELECTION.txt；凍結選擇：PLAN_X_LR_SELECTION.json；選擇與輸入雜湊證據：PLAN_X_SELECTION_AUDIT.json；判定快照：PLAN_X_RESULTS.json；原始輸出：PLAN_X_DECISION.txt。

**事後分析（2026-11-11，看過 test 後才做，不屬於凍結判定）：**
- 改選 0.0003 的三個模型（ResNet34、ResNet50、DenseNet121），0.0003 和 0.001 的 validation 只差 0.10–0.14pp，test 則相反，是 0.001 較高（差 0.11–0.31pp）。所以這三個模型的 X1 比 W1 大（ResNet34 +1.45→+1.76、ResNet50 +1.33→+1.45、DenseNet121 +3.05→+3.17），是選 LR 時的雜訊，不代表我們的方法變好。
- 保守下限：每個 seed 各自取五個 LR 中 test 最高的 ft_const 當對手（等於用 test 挑 LR，是對手最有利的情況），gradual+floor 仍然 8/8 模型 3/3 為正。平均差：ResNet18 +1.81、ResNet34 +1.45、ResNet50 +1.33、ResNet-56 +5.69、ResNet-110 +3.34、VGG16-BN +1.66、GoogLeNet +4.22、DenseNet121 +3.04（seeds 0/1/2，n=3）。
- 0.0003–0.001 之間的 validation 曲線幾乎是平的。下界在形式上仍未封閉，但目前看不出再降 LR 能讓對手多拿 1pp 以上。這是推測，沒有實測。
- 建議論文正文報 X1，並附上這個保守下限，回應「對手 LR 沒調好」的質疑。
