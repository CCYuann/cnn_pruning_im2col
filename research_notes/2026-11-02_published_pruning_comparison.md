# 已發表CNN剪枝數據與本專案對照

核對日期：2026-11-02。這是原論文報告值與本專案既有實驗的並列，不是相同checkpoint、訓練及評估實作下的重跑排名。未新增訓練、未對最新搜尋模型讀取test。

## 1. Cross-filter原論文：CIFAR-10、90%標示剪枝率

原文：Pham et al., Cross-Filter Structured Pruning for Efficient Sparse CNN Acceleration, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3587027。

來源：本機PDF第7頁（印刷頁129467）Figure6(b)，及第8頁（129468）Figure7(b)。數字是圖內原有表格的精確標示，不是從柱高估計。ResNet34出自Figure7；其他三模型出自Figure6。

| 模型 | 原文cross-filter accuracy (%) | 本專案l1_sum (%) | 本專案其他criterion (%) |
|---|---:|---:|---:|
| ResNet34 | 82.81 | 94.007 | max_abs 94.123 |
| ResNet50 | 88.42 | 92.933 | top2_mean_abs 93.313 |
| GoogLeNet | 91.08 | 92.460 | top2_mean_abs 92.517 |
| DenseNet121 | 89.93 | 93.587 | top4_mean_abs 93.727 |

本專案：Tm8、連續分組、無敏感層保護或cap、實際conv權重sparsity約90%、三seed test平均、Dense100epochs後fine-tune10epochs。本表列出的其他criterion依模型不同，不能當成一個統一演算法，也不能當成未經選擇的整體平均優勢。

判斷：本專案報告accuracy確實高於這些原文數字；但連本專案l1基準也高出許多，因此差距不能歸功於max/top2/top4改良。需核對模型stem、影像處理、訓練、cluster大小、剪枝分母及精度等設定。MobileNet原文是V1，本專案正式比較為V2，不直接並列。

原文ResNet34 CIFAR-10另列：

| 標示剪枝率 | 原文cross-filter (%) | 本專案l1 Tm8三seed (%) |
|---|---:|---:|
| 70% | 89.12 | 94.530 |
| 80% | 87.39 | 94.440 |
| 90% | 82.81 | 94.007 |

[論文與DOI資訊](https://pure.korea.ac.kr/en/publications/cross-filter-structured-pruning-for-efficient-sparse-cnn-accelera/)。精確表格以本機PDF Figure6/7為依據；機構摘要頁本身不包含這些數字。

## 2. GraNet論文：ResNet50、CIFAR-10、90%權重剪枝

原文：Liu et al., Sparse Training via Boosting Pruning Plasticity with Neuroregeneration, NeurIPS2021。Table2提供作者三seed重跑的RigL/GMP及GraNet結果。

| 方法 | 稀疏結構 | Test accuracy mean ± std (%) |
|---|---|---:|
| RigL（GraNet論文重跑） | 非結構化 | 94.45 ± 0.43 |
| GMP（GraNet論文重跑） | 非結構化 | 94.34 ± 0.09 |
| GraNet，initial sparsity=0 | 非結構化 | 94.49 ± 0.08 |
| GraNet，initial sparsity=0.5 | 非結構化 | 94.64 ± 0.27 |
| 本專案l1_sum | Tm8共享K mask | 92.933 ± 0.465 |
| 本專案top2_mean_abs | Tm8共享K mask | 93.313 ± 0.460 |

差距：本專案top2比上述GraNet兩版本低1.177／1.327pp，比RigL低1.137pp。這是報告值差距，不是控制其他變因後的演算法因果差異。

條件：GraNet Table1列ResNet50 CIFAR-10訓練160epochs，Dense為94.75±0.01%；本專案為45000train/5000validation、Dense100epochs及10epochs FT。稀疏結構、訓練流程、layer allocation及prunable tensor範圍也未對齊。非結構化不強制8filters共享同一support，因此此表用作精度參考，不作硬體效率排名。

[原論文Table1/2](https://proceedings.neurips.cc/paper/2021/file/5227b6aaf294f5f027273aebf16015f2-Paper.pdf)。本段不是已獨立重現RigL或GraNet。

## 3. HRank：GoogLeNet、CIFAR-10、接近運算量縮減

原文：Lin et al., HRank: Filter Pruning Using High-Rank Feature Map, CVPR2020, Table2。

| 方法 | 參數移除／權重sparsity (%) | 運算量縮減 (%) | Accuracy (%) |
|---|---:|---:|---:|
| HRank較低壓縮點 | 55.4，整體parameters | 54.9，FLOPs | 94.53 |
| 本專案l1、target80% | 80.001，conv權重 | 55.458，conv MAC | 93.390 |
| HRank較高壓縮點 | 69.8，整體parameters | 70.4，FLOPs | 94.07 |
| 本專案l1、target90% | 90.001，conv權重 | 74.842，conv MAC | 92.460 |

第一組運算縮減百分比相近，HRank accuracy高1.14pp；第二組本專案運算縮減更多，但accuracy低1.61pp。HRank是完整filter刪除，本專案是保留channel形狀的共享K稀疏，且運算統計分母並非完全一致；只以比例作近似參考，不宣稱相同絕對MAC。

HRank原始Dense GoogLeNet accuracy95.05%；本專案Dense-extra三seed94.097%，兩者baseline及額外訓練預算也不一致。原文Section4.1寫每剪一層後retrain30epochs，與本專案一次剪枝後整體FT10epochs不同，因此不能把raw accuracy差直接歸因於importance。

[原論文Table2與Section4.1](https://arxiv.org/pdf/2002.10179)。HRank學術會議版本亦已交叉核對，Table2一致。

## 4. 本專案數據來源

- `runs/fixed_checkpoint_all_ratio_summary/cross_filter_summary.csv`：三seed l1、不同ratio的精度、MAC及實際權重剪枝率。
- `runs/fair_criterion_r0p9_tm8/criterion_summary.csv`：無保護、實際90%的importance-only比較及sample std。
- `runs/FAIR_PRUNING_R0P9_FINAL.md`：公平預算實驗總結。
- `runs/task_loss_mask_study/FINAL_REPORT.md`：最新mask搜尋只有seed0 validation，不混進本文件三seed test表。

## 5. 結論與下一步比較規格

本專案不是accuracy很低到不能研究，但也沒有證據說勝過成熟剪枝方法。與基底Cross-filter論文報告值相比較高，與GraNet/RigL非結構化相比仍有約1pp以上差距；與HRank相近運算縮減點相比，GoogLeNet仍有精度差距。

論文若要宣稱方法更好，至少在本專案相同checkpoint、split、實際sparsity或MAC、訓練步數與Tm下重跑比較法，再報三seed配對差異。硬體比較另需實際cycle、SRAM accesses與資源；此處無法由90%權重剪枝推論90%加速。
