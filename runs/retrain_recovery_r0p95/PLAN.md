# 95%共享mask：訓練端救回研究（凍結計畫）

凍結日期：2026-11-05，於讀取任何本研究結果之前寫下。

共同設定：ResNet50/CIFAR-10、Tm8、contiguous groups、flat l1、global_cluster配額；
3 Dense seeds；batch128、SGD m0.9、wd5e-4、AMP float16、相同FT RNG；僅5000張validation，不評估test。
總預算60 epochs。LR為Dense 100ep cosine(0.1->1e-4)的最後60 epochs（起始約0.0655），不調參。報最後一epoch，不挑最佳epoch。

| 組別 | 內容 |
|---|---|
| dense | 無mask，同LR，參考上限 |
| unstructured | 95%全域非共享mask，固定，同LR |
| ft_const | A的95% mask，固定lr0.001，60ep（舊微調對照） |
| rewind | A的95% mask，LR rewinding |
| rewind_r90 | A的90% mask，LR rewinding（公平比較「權重減半」） |
| rewind_B | 95% mask，B補償權重為初始，LR rewinding |
| rewind_kd | rewind + Dense teacher KD（T=4，alpha=0.9） |
| gradual | 前40ep每2ep依cubic排程由0升到95%更新mask，後20ep固定 |
| group | 前40ep對每cluster每K位置加L2,1 prox，第40ep依權重剪到95%，後20ep固定 |

group的lambda：seed0試{3e-5,1e-4,3e-4}，取seed0 validation最高者，再跑seed1/2。seed0因此帶選擇偏差，判讀以seed1/2為準並另報。

判定（皆為投入門檻，不是顯著性檢定；n=3）：
- H1 訓練端是否重要：rewind-ft_const 平均>=+0.5pp 且3/3為正。
- H2 方法是否有增益：gradual/group/rewind_kd/rewind_B 對 rewind 平均>=+0.3pp 且3/3為正。
- H3 95%能否追上90%：最佳方法@95% >= rewind_r90@90%。
- H4 共享差距是否縮小：unstructured-方法 < unstructured-rewind，且需與H2同時成立。

gradual/group的層配額會隨權重改變，kept相近不代表MAC相同，必報MAC與q=0。
若通過，下一步才疊加（如group+KD）；不在本批加seed或改設定救結果。
