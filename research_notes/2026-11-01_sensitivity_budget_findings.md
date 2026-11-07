# 研究紀錄：敏感層保護與固定剪枝預算的取捨

紀錄日期：2026-11-01。實驗於2026-10-27完成。本文件保存本次討論的想法、正式結果與後續研究判斷，供論文整理及交接使用。

## 核心想法與目前結論

保護敏感層後，如果沒有把少剪的權重數分配到其他層，實際剪枝率通常低於設定的target。這時較高的accuracy同時受到「保護位置」與「保留更多權重」影響，不能直接當成相同剪枝預算下的方法改良。

把缺額重新分配到其他層，可以維持指定的整體weight sparsity；但其他層承擔更高剪枝量後，也可能破壞模型。原本單層測試較不敏感，不表示能承受多層聯合、極高比例的剪枝。

**本批實驗顯示：目前這個重新分配方法在MobileNetV2有幫助，在ResNet34、ResNet50、GoogLeNet退步，DenseNet121退步尤其明顯。不能將結果概括為所有重新分配方法都會變差，也不能宣稱敏感層保護普遍有效。**

## 三種流程需要分開解讀

| 流程 | 實際剪枝率 | 能回答的問題 |
|---|---|---|
| 原reference，無敏感層保護與cap | 約達target，存在K-position取整誤差 | 原始配置下，不同importance的效果 |
| 敏感層保護＋cap，未重新分配 | 通常低於target | 較保守配置的accuracy / sparsity取捨 |
| 敏感層保護＋缺額重新分配 | 本批匹配基準的實際約90% | 在相同weight budget下，該保護與分配方案是否有幫助 |

目前不補剪的舊版本設定為最敏感第1層不剪、其後3層cap=0.2、其他層cap=0.75。這些限制會降低可達的總剪枝量；程式不會自動把差額補到其他層。

## 本次正式比較條件

- Dataset：CIFAR-10，train / validation / test為45000 / 5000 / 10000。
- Models：ResNet34、ResNet50、GoogLeNet、DenseNet121、MobileNetV2。
- 每個model使用seed 0、1、2的固定100-epoch Dense checkpoints。
- Tm=8，連續filter分組；同cluster共用K-position mask。
- Fine-tune 10 epochs、LR=0.001、batch size=128。
- 第一階段新增45個importance-only任務，另沿用15個l1_sum正式基準。
- 第二階段新增30個l1_sum / top2敏感層保護與重新分配任務。
- 每組結果均有3 seeds，全部75個新增任務完成。
- 報告的loss是相對同model/seed的Dense extra accuracy差距，單位為percentage point (pp)，正值表示下降。

## 本次重新分配方法

使用各model / seed / Tm的正式validation top2 sensitivity ranking。排名來自每次只剪一層、測試50% / 60% / 70%比例、不做fine-tune的平均validation accuracy drop。

本次保護規則為：

```text
最敏感第1層：完全不剪
其後3個敏感層：prune_k <= floor(0.2*K)
其他層：移除原75% cap，但每cluster至少保留1個K position
```

先以global_cluster reference建立初始配置，再按以下固定規則補剪：

```text
s = 該層平均validation accuracy drop，單位pp
mobility = 1 / (1 + max(0, s))
r(lambda) = min(max_prune_k/K, initial_prune_k/K + lambda*mobility)
```

透過二分搜尋lambda與整數餘額修正，精確匹配同model/seed原始l1_sum移除的權重數。單位為整個共享K position；本批所有cluster恰好有8個filters，每移除1個K便移除8個權重。

這是根據既有敏感度計算的啟發式配置，**沒有每補剪一次就重新量測validation accuracy**，也沒有驗證多層聯合剪枝的交互影響。兩個保護版都固定使用top2排名，並非分別尋找各criterion的最佳保護配置。

## 固定約90%預算的實際結果

以下固定比較top2，避免把importance切換與保護策略混在一起。數值為三seed平均accuracy loss。

| Model | Top2無保護 loss (pp) | Top2保護＋重新分配 loss (pp) | 額外損失：後者−前者 (pp) |
|---|---:|---:|---:|
| ResNet34 | 0.853 | 0.947 | +0.093 |
| ResNet50 | 0.950 | 1.210 | +0.260 |
| GoogLeNet | 1.580 | 1.717 | +0.137 |
| DenseNet121 | 1.607 | 8.207 | +6.600 |
| MobileNetV2 | 3.153 | 2.310 | −0.843 |

差值由未取整平均值計算，因此可能與顯示到三位小數後的相減結果相差0.001 pp。正值代表加入本次保護與分配後更差；負值代表改善。

DenseNet121的重新分配版變異也很大：top2 loss的sample std約5.758 pp；l1_sum重新分配版平均loss為12.390 pp，sample std約9.337 pp。因此不能只引用單一均值，而應回查逐seed配置與原始結果。

## Importance本身的觀察

相同無保護reference配置下，各模型的三seed平均最佳方法如下：

| Model | l1_sum loss (pp) | 觀察到的最佳criterion | 最佳loss (pp) | 相對l1_sum的accuracy提升 (pp) |
|---|---:|---|---:|---:|
| ResNet34 | 0.783 | max_abs | 0.667 | +0.117 |
| ResNet50 | 1.330 | top2_mean_abs | 0.950 | +0.380 |
| GoogLeNet | 1.637 | top2_mean_abs | 1.580 | +0.057 |
| DenseNet121 | 1.630 | top4_mean_abs | 1.490 | +0.140 |
| MobileNetV2 | 3.363 | top2_mean_abs | 3.153 | +0.210 |

這是事後對各模型選最佳方法的描述，不是一套預先固定的criterion在所有模型全面勝出。三seed的小幅平均提升不等於已證明統計顯著。

## 如何寫進論文

可使用的保守敘述：

> 敏感層保護在未重新分配預算時會降低實際稀疏度，因此其精度優勢不能直接視為等預算改良。在實際約90%權重稀疏度下，本研究的敏感度導向重新分配於MobileNetV2改善精度，但在其他受測模型未呈現一致優勢，DenseNet121尤其出現明顯退化。此結果顯示，單層敏感度不足以單獨決定高稀疏度下的全網路剪枝配置。

最後一句是由現有結果支持的研究判斷，仍應保留演算法設定、模型與測試比例的適用範圍。退步的確切成因尚未確認。

不能主張：

- 保護敏感層必然提高同剪枝率下的精度。
- 所有重新分配方法都比不保護差。
- 舊保護版target=90%就是實際90%。
- 相同weight sparsity代表相同MAC reduction、memory traffic或硬體latency。
- 已經確認DenseNet121失敗完全由某一層或某一結構造成。

## 後續研究方向與決策

1. 保留importance-only結果作目前最直接的公平比較。
2. ResNet50的top2、MobileNetV2的保護與重新分配，是值得繼續核對穩定性的候選方向。
3. 先檢查DenseNet121逐seed的cluster配置，找出哪些層被補剪到極高比例，以及其與精度退化的關聯。
4. 未來可測試限制單層額外剪枝量、重新量測validation反應，或更溫和的保護規則；這些目前是待驗證想法，尚無成果。
5. 若保護條件使指定預算不可達，應回報不可行；若保護條件雖可達但補剪代價太大，應呈現其accuracy / sparsity取捨。

記錄本想法不代表啟動新的訓練；本次只保存已完成的成果與討論。

## 數據與程式來源

- [兩階段完整比較表及圖表](../runs/FAIR_PRUNING_R0P9_FINAL.md)。
- [Importance-only三seed統計](../runs/fair_criterion_r0p9_tm8/criterion_summary.csv)。
- [Importance-only逐seed配對](../runs/fair_criterion_r0p9_tm8/paired_vs_l1.csv)。
- [保護與重新分配三seed統計](../runs/sensitivity_redistributed_r0p9_tm8/method_summary.csv)。
- [同criterion加入保護的逐seed效果](../runs/sensitivity_redistributed_r0p9_tm8/paired_protection_effect.csv)。
- [重新分配方法、設定與檔案說明](../runs/sensitivity_redistributed_r0p9_tm8/README.md)。
- [重新分配器](../scripts/sensitivity_budget_pruning.py)。
- [已完成實驗資料ZIP](../runs/fair_pruning_r0p9_complete_20261027.zip)。

ZIP是2026-10-27完成時的實驗快照；本份2026-11-01研究紀錄另外保存於專案，未修改原ZIP與其checksum。
