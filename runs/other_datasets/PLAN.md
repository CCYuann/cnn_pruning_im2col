# 其他資料集驗證（凍結計畫）

凍結日期：2026-11-06，於任何其他資料集結果產生之前寫下。

動機：CIFAR-10上98至99%剪枝仍有91至93%，懷疑是任務太簡單。本批回答：相同配方換資料集後，精度掉多少、gradual相對rewind的增益是否還在。

資料集：
- ImageNette（imagenette2-160）：與Cross-Filter論文同一資料集，可對照其Figure 6/7的50至90%。
- ImageWoof（imagewoof2-160）：10種狗，同尺寸但細粒度，較難。
- CIFAR-100：100類，32x32。
Animals-10需Kaggle憑證，本機沒有，不做。

模型：ResNet34（硬體主線）跑全部資料集；ResNet50只跑CIFAR-100。
160px資料集沿用torchvision ImageNet stem（7x7 stride2+maxpool）；CIFAR-100沿用small stem。
訓練：Dense 100 epochs，剪枝60 epochs，LR、WD、batch與CIFAR-10完全相同，不調參。160px資料增強為RandomResizedCrop(0.35,1)+flip。
組別（seed0/1/2）：Dense；90/95/98/99%各跑rewind、gradual+q下限、unstructured。

因組別與參數全部預先固定、沒有任何選擇，test於每條結束時直接評估一次，報test平均。
主要報告：
1. 各剪枝率對Dense的掉幅、對unstructured的掉幅。
2. gradual+q下限對rewind的配對增益，判定：平均 >= +0.3pp 且3/3為正才算該資料集有增益。
3. 週期（用該資料集幾何重新估算）。
預先說明：若高剪枝率掉幅遠大於CIFAR-10，CIFAR-10上的結果不得外推。

## 判定（ImageNette、ImageWoof，2026-11-06 22:05，test，seeds 0/1/2）
gradual+q下限對rewind：**兩資料集四個比例皆未達增益門檻**（>= +0.3pp 且3/3）。ImageWoof 95%為-1.30/-2.42/-1.20，gradual反而較差。
週期增益仍在：98% 90.77→54.83 ms（ImageNette）、93.21→53.34 ms（ImageWoof）。
掉幅遠小於預期：98%對未剪枝對照 ImageNette +0.08/+0.13pp，ImageWoof -0.87/-0.44pp（負值代表剪枝較高）。
未剪枝對照：ImageNette 87.55%（Dense 86.91%），ImageWoof 77.81%（Dense 75.32%），證實Dense未收斂、Dense不是合適基準。
test僅3925張，單模型標準誤約0.4至0.5pp，ImageWoof seed間差距約2至3pp。

## 隨機mask檢查（後加，看過上述結果才提出，非原計畫）
問題：掉幅這麼小，是因為方法好，還是模型容量根本不是限制？這兩個資料集各只有約8500張訓練影像。
方法：random組沿用rewind 同比例每cluster的位置數，位置隨機選，60 epochs，LR同rewind。seeds 0/1/2，98%、99%。
解讀（事前寫定）：
- random與rewind/gradual差距 < 1pp：該資料集無法區分剪枝方法，精度不得當作方法強度的證據。
- random明顯較低（>= 1pp且3/3）：mask選擇確實有效。
CIFAR-100 ResNet34同樣跑，對照組已排定其餘工作之前。

## 隨機mask結果（test，seeds 0/1/2，先前寫定的解讀）
| 資料集 | 比例 | rewind | gradual+floor | random | learned-random（rewind） |
|---|---|---:|---:|---:|---:|
| ImageNette | 98% | 87.47 | 87.42 | 86.98 | +0.00/+0.87/+0.61 |
| ImageNette | 99% | 86.58 | 86.62 | 86.60 | -0.03/-0.43/+0.41 |
| ImageWoof | 98% | 78.69 | 78.26 | 79.81 | -2.01/-0.28/-1.07 |
| ImageWoof | 99% | 77.31 | 77.49 | 78.57 | -2.11/+0.15/-1.83 |
ImageNette差距 < 1pp；ImageWoof隨機mask高於學到的mask，未剪枝對照77.81%也低於隨機mask 79.81%。
結論：這兩個資料集無法區分剪枝方法，精度不得當作方法強度的證據，也不足以檢驗容量極限。解釋（資料量受限與正則化）符合觀察，但本批未直接驗證。

## 判定（CIFAR-100 ResNet34，2026-11-07 03:00，test，seeds 0/1/2，事前寫定的門檻）
Dense 76.79%，未剪枝對照 77.27%。週期為 V3 公式預測。
| 比例 | rewind | gradual+floor | 增益 | unstructured | random | ms：rewind → gradual |
|---|---:|---:|---|---:|---:|---|
| 90% | 75.91 | 75.98 | +0.37/+0.51/-0.66，未過 | 77.11 | 未跑 | 59.60 → 48.61 |
| 95% | 74.59 | 74.54 | +0.08/-0.52/+0.29，未過 | 76.01 | 未跑 | 34.57 → 25.71 |
| 98% | 71.32 | 72.42 | +0.63/+1.35/+1.32，平均 +1.10，**通過** | 74.31 | 68.96 | 17.87 → 12.38 |
| 99% | 67.97 | 70.31 | +2.72/+2.71/+1.60，平均 +2.34，**通過** | 71.77 | 65.31 | 11.53 → 8.27 |
隨機 mask：rewind 對 random 98% +2.74/+2.21/+2.12，99% +1.85/+2.78/+3.34，皆 >= 1pp 且 3/3，CIFAR-100 能區分剪枝方法。
掉幅遠大於 CIFAR-10：gradual+floor 對 Dense 98% -4.38pp、99% -6.48pp，CIFAR-10 上的結果不得外推。
對 unstructured 的結構化代價：gradual+floor 98% -1.89pp、99% -1.46pp。
三個 seeds 的 conv1 都保有 RGB，這批沒有色盲現象。

## CIFAR-100 ResNet34 Q15（test，seeds 0/1/2，流程沿用 CIFAR-10 定案，未調參）
| 比例 | rewind FP32 → Q15 | gradual+floor FP32 → Q15 | Q15 配對增益 |
|---|---|---|---|
| 98% | 71.33 → 71.18 | 72.42 → 72.29 | +0.87/+1.28/+1.16，平均 +1.10 |
| 99% | 67.98 → 67.75 | 70.30 → 69.86 | +2.65/+2.41/+1.28，平均 +2.11 |
peak_acc 約 2^29，同 CIFAR-10，來自近乎死掉 channel 的 bias。明細：TEST_Q15_CIFAR100.csv。

## 判定（CIFAR-100 ResNet50，2026-11-07 10:00，test，seeds 0/1/2，事前寫定的門檻）
Dense 75.73%，未剪枝對照 76.12%。週期為 V3 公式預測，下限 18.95 ms。
| 比例 | rewind | gradual+floor | 增益 | unstructured | ms：rewind → gradual |
|---|---:|---:|---|---:|---|
| 90% | 73.80 | 73.97 | -0.21/+0.50/+0.22，未過 | 75.46 | 50.96 → 47.58 |
| 95% | 71.95 | 72.04 | -0.63/+0.72/+0.16，未過 | 74.47 | 33.02 → 31.82 |
| 98% | 68.22 | 69.22 | -0.33/+2.68/+0.65，未過 | 72.37 | 23.28 → 23.46 |
| 99% | 63.56 | 64.50 | -0.71/+2.58/+0.94，未過 | 70.68 | 20.55 → 20.77 |
ResNet50 全面低於 ResNet34：99% 64.50 對 70.31，結構化對 unstructured 的代價 -6.19pp（ResNet34 -1.46pp），週期 20.77 ms 對 8.27 ms。
V3 上 q 已接近 15 的下限，ResNet50 的 gradual 省不到週期。主線維持 ResNet34。
