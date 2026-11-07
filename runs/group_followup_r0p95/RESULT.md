# 結果：Group lambda精掃與Group+gradual

ResNet50/CIFAR-10、Tm8、95%、60 epochs、5000張validation最後一epoch；未評估test。n=3不是顯著性檢定。

seed0 lambda：group 1e-5 94.12、2e-5 94.40、3e-5 94.76（前批1e-4 94.12、3e-4 93.98）；group_gradual 1e-5 94.28、3e-5 94.44。兩方法皆選3e-5。

| 組別 | seed0/1/2 (%) | 平均 | 對rewind (pp) | MAC減少 | q=0 |
|---|---|---:|---:|---:|---|
| group 3e-5（前批） | 94.76/94.38/94.58 | 94.573 | +0.427 (3/3) | 88.44% | 353/285/616 |
| group_gradual 3e-5 | 94.44/94.62/94.50 | 94.520 | +0.373 (3/3) | 90.57% | 373/471/656 |

判定：
- 精度升級（>=94.773%）：未達成。
- MAC升級（>=94.573%且MAC減少>=89.6%）：未達成，MAC達標但精度差0.053pp。
- 依凍結規則：採用group lambda3e-5，不再調參。

事後觀察（非判定依據）：seed1/2平均group 94.48%、group_gradual 94.56%；group的seed0帶選擇偏差。group_gradual剩餘MAC 9.43%，較group 11.56%少約18%。
限制：group_gradual實際sparsity 94.99%，在容許範圍內；seed2的layer4.2.conv2與conv3全部q=0。
