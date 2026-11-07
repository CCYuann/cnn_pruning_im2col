# 95%突破測試：第一輪篩選（凍結計畫）

凍結日期：2026-11-06，於任何本批結果產生之前寫下。ResNet50/CIFAR-10、Tm8、95%、contiguous、與前批相同訓練設定，5000張validation最後一epoch，不評估test。
群組lambda一律沿用3e-5，不重新調參；MAC的beta只試0.25與0.5。

只跑seed1、seed2（seed0帶lambda選擇偏差）。最終判定不在這裡，而在全新Dense seeds 3/4/5。

| 組別 | 內容 |
|---|---|
| dpf | Dynamic Pruning with Feedback（Lin et al. ICLR 2020）的共享mask版：前40ep每2ep依cubic排程重算mask，以遮罩後網路求梯度、更新完整權重，位置可回復；第40ep硬剪 |
| group_lamp | group lambda3e-5，第40ep改用LAMP（Lee et al. ICLR 2021）全域分配 |
| ggrad_lamp | group_gradual改用LAMP |
| ggrad_mac0.5 / 0.25 | group_gradual的配額改由|w|/(hw)^beta全域排序決定，cluster內仍用flat l1 |
| group_e100 | group，100 epochs完整cosine，前67ep prox（預算不同，另列） |
| rewind_mac0.5 | 固定Dense mask的MAC配額對照，判斷group是否必要 |

參考值（seed1/2平均，事後已知）：group 94.48%、MAC減少約88.4%；group_gradual 94.56%、MAC減少約90.6%。

晉級規則：
- 精度候選：平均 >= 94.76%（group_gradual +0.2pp）。
- MAC候選：平均 >= 94.46%（group_gradual -0.1pp）且平均MAC減少 >= 92%。
- group_e100僅在 >= 94.76%才算，並註明訓練預算為100ep。
- 兩項最多各取一個（精度取最高，MAC取MAC減少最大者）。sparsity須在95%±0.5pp內。

確認階段（新seeds 3/4/5，預先宣告）：主方法group；次要group_gradual；加上晉級者；基準rewind；上限unstructured。
通過條件沿用H2：對rewind平均 >= +0.3pp 且3/3為正。MAC候選另列：對rewind平均 >= -0.1pp。
完成後才對確認組模型各評估一次test。必報：實際sparsity、kept、MAC、q=0、全q=0層。
n=2或3都不是顯著性檢定。
