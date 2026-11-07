# 新Dense seeds確認（凍結計畫）

凍結日期：2026-11-06，於任何seed3/4/5剪枝結果產生之前寫下。ResNet50/CIFAR-10、Tm8、95%、60 epochs，lambda固定3e-5。

第一輪篩選（seed1/2）沒有方法達晉級門檻（group_e100待seed2），因此組別依原計畫固定為：
- 主方法 group；次要 group_gradual；基準 rewind；上限 unstructured。

通過條件（對rewind，n=3，不是顯著性檢定）：平均 >= +0.3pp 且3/3為正。
報validation最後一epoch；通過與否都接著各評估一次test，不得再依test調整任何東西。
同時報Dense新seeds的validation與「unstructured-方法」差距。

## 修訂（2026-11-06，seed3/4/5尚無任何剪枝結果）
追加探索性組別 dpf。它未達第一輪MAC門檻，因RTL週期模型顯示其週期在seed1/2比group_gradual少約6%才加入，屬事後動機，不算主判定，僅作為下一輪假說。

## 98%確認（2026-11-06，於seed3/4/5任何98%結果產生之前寫下）
動機：seed1/2 validation上98% group比rewind高0.80pp（+0.28/+1.32），達原計畫門檻，且rewind出現全q=0層。
組別：rewind、group、group_gradual，98%，lambda固定3e-5，seed3/4/5。
通過條件：validation上對rewind平均 >= +0.5pp 且3/3為正。不論是否通過，完成後對這些模型各評估一次test，不再依test調整。

## 98%上限參考（2026-11-06，已看過98% test之後追加）
unstructured 98% seed3/4/5，只作為共享mask代價的參考線，不參與任何方法選擇或判定。
