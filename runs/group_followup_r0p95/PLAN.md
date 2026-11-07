# Group lambda精掃與Group+gradual（凍結計畫）

凍結日期：2026-11-05，於讀取本批結果之前寫下。設定、預算、LR、validation規則與retrain_recovery_r0p95完全相同，不評估test。

新組別：
- group：lambda 1e-5、2e-5（3e-5沿用前批）。
- group_gradual：前40ep同時做L2,1 prox與gradual mask更新，lambda 1e-5、3e-5；mask與上一輪取交集以避免零值平手造成regrowth。

選擇：兩方法各自以seed0 validation取最高lambda，再跑seed1/2；前批已有者不重跑。seed0帶選擇偏差，seed1/2另報。

判定（比較對象group lambda3e-5：94.573%、MAC減少88.44%）：
- 精度升級：平均 >= 94.773%（+0.2pp）。
- MAC升級：平均 >= 94.573% 且平均MAC減少 >= 89.6%（剩餘MAC少至少10%）。
- 都未達成：採用group lambda3e-5，不再調參。

必報：實際sparsity（須在95%±0.5pp內，否則該分支不合格）、kept、MAC、q=0、全q=0層。
定案後另以新Dense seeds確認，通過才評估test。
