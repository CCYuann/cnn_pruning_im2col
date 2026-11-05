# 研究提案：固定配額的共享Mask修補

日期：2026-11-01。執行更新：已完成五模型seed0診斷與兩模型16個新fine-tune分支，並測試mean-safe調整。重建誤差可改善，但修補版未在這兩個seed0 validation pilot勝過l1，因此未補三seed、未評估test。不是已成立的分類精度貢獻；新穎性仍待比對。見 [最終報告](../runs/shared_mask_repair_study/FINAL_REPORT.md) 與 [完整問題紀錄](../runs/shared_mask_repair_study/EXECUTION_LOG.md)。

本提案是從現有cross-filter結構與實驗限制推導的待驗證方向。它借用既有重建、相對誤差與離散最佳化概念，不主張這些基礎概念由本研究首創。能否形成論文貢獻，取決於具體方法、相關工作差異與實驗證據。

## 1. 從我們自己的問題出發

已確認的現象：

- 只改l1 / max / top2 / top4，改善幅度與最佳方法因模型而異。
- 硬性保護層再補足90%，部分模型反而更差。
- 固定cluster共用mask是硬體友善性的主要約束；使用者目前不採用相似filter分組。

待驗證的機制假說：同cluster可能存在保留位置需求不同的filters。用總分或少數最大權重排序，可能讓部分filters在最終mask下損失特別嚴重。此機制尚未透過現有數據直接確認，不能當成已查明的退步原因。

研究問題：**不改每層或每cluster剪枝數量，能否只調整共享mask的位置，減少對個別filter的極端破壞，改善最終精度？**

## 2. 核心約束

對每個cluster g，沿用正式l1_sum reference生成的kept_count q_g。新方法只選擇另一個集合S_g：

```text
|S_g| = q_g
cluster成員、Tm、output channels保持相同
同cluster內所有filters共用S_g
```

因此各cluster移除權重數與理論MAC數維持相同。推論仍輸出kept_k、kept_count與packed weights；不同位置可能影響實際讀取或bank衝突，不能因此保證量測latency及energy完全一致。

q_g=0或q_g=K時沒有交換空間，保留原mask並回報該情況。

## 3. 可檢驗的例子

兩個filters、四個K位置，只允許保留兩個位置：

| Filter | K0 | K1 | K2 | K3 |
|---|---:|---:|---:|---:|
| F0 | 10 | 9 | 0 | 0 |
| F1 | 0 | 0 | 1 | 1 |

l1_sum選K0、K1，F0全部保留，F1的有效權重全部刪掉。換成K0、K2，兩個filters都保留部分連接，kept_count仍為2。

若校正輸入的各K為單位二階矩且互不相關，無bias輸出重建的相對平方誤差可由移除權重能量比例精確得到：

| Shared mask | F0相對誤差 | F1相對誤差 | 最壞filter誤差 |
|---|---:|---:|---:|
| K0、K1 | 0 | 1 | 1 |
| K0、K2 | 81/181 ≈ 0.448 | 1/2 = 0.5 | 0.5 |

這個例子只說明可能的總量排序盲點，不是CNN實驗成果，也不保證平均任務精度改善。

## 4. 以最終Mask的誤差決定交換

先從原始mask S0開始，以固定的訓練集校正patches取得Dense線性卷積輸出Y_m，以及同輸入下使用mask S的輸出Yhat_m(S)。在第一版不調整保留權重，隔離mask選擇本身的影響。

```text
E_m(S) = ||Y_m - Yhat_m(S)||² / (||Y_m||² + epsilon)
```

第一個研究目標是：

```text
minimize   max_m E_m(S)
subject to |S| = q
           mean_m E_m(S) <= mean_m E_m(S0)
```

也就是在不增加校正集平均相對重建誤差的限制下，降低受損最嚴重filter的誤差。原始mask本身永遠是可行候選；若沒有符合條件的改進，就保留原mask，不強迫交換。

這裡的epsilon、近零能量輸出的處理與候選搜尋設定，必須在看test前固定。近零輸出的filter可能因分母過小而主導目標，第一版須定義可靠的能量下限或排除規則，並報告涉及數量。

這項約束只約束校正集重建誤差，不保證validation或test accuracy不下降，也不保證fine-tune後的精度排序相同。

## 5. 固定配額的一換一修補

```text
從l1_sum mask S0開始
  -> 找相對重建誤差最大的filter
  -> 從未保留位置中找可能補償它的候選b
  -> 從已保留位置中找可移除的候選a
  -> 試算 S' = S - {a} + {b}
  -> 僅接受改善最壞誤差且符合平均誤差約束的交換
  -> 到固定交換次數上限，或無可接受候選時停止
```

候選可先限制為8個移除位置與8個加入位置，每輪最多64對。候選數是計算量控制的初始設計，不是已選定的最佳超參數。所有方法要在validation上決定設定，再用test做最終比較。

可用殘差更新減少重新forward的成本。若R_m=Y_m-Yhat_m，移除a並加入b後：

```text
R'_m = R_m + W[m,a]*X_a - W[m,b]*X_b
```

因此在固定校正輸入及原始權重下，可直接評估候選，不必每次跑整個網路。一般卷積使用對應patch；depthwise需用各filter實際對應的input channel，不能假設所有filter共用同一X。

## 6. 與參考方法要比較的差異

| 方向 | 已有概念 | 本提案希望驗證的具體部分 |
|---|---|---|
| Wanda | activation加權importance | 根據目前整個共享mask的逐filter誤差決定交換，而非一次性固定分數排序 |
| Taylor pruning | 移除結構的loss近似 | 以固定cluster配額下的實測線性輸出殘差作局部修補 |
| OBC / SparseGPT | layer reconstruction及權重補償 | 第一版只換共享位置、不調權重，關注逐filter相對誤差的尾端 |
| LAMP | magnitude縮放與distortion | 不分配layer sparsity，固定每cluster的q |
| 多觀測共同support選擇 | 多輸出聯合選擇 | 需深入比對其目標與求解法，不能僅靠名稱差異宣稱新穎 |

這張表是待查證的定位，不代表已完成優先權或新穎性審查。平均誤差約束、minimax目標、shared support與mask交換各自都可能有相關先例。

## 7. 先驗證機制，再花時間訓練

第一步只重新產生既有baseline mask並分析校正誤差，不啟動fine-tune：

- 每cluster的mean / max相對重建誤差。
- 有多少filters的相對誤差接近1，是否集中在少數cluster。
- Tm8與Tm16是否出現不同的誤差尾端。
- 是否有足夠多cluster能透過一換一同時改善最壞與平均誤差。

若此假說在真實模型中沒有支持，或可接受交換非常少，就應調整或放棄本方向，而不是先跑完大量seed再找故事。

初步fine-tune候選為ResNet50（importance改良已有改善空間）及DenseNet121（重新分配不穩定，需要避免額外提高其他層剪枝量）。這是研究優先順序，不代表保證這兩個模型會成功。

## 8. 必要消融

1. 原始l1_sum，固定原q。
2. 正規化後的靜態importance，固定原q，檢查只做正規化能解釋多少。
3. 只最小化平均重建誤差的mask交換。
4. 最壞相對誤差＋平均約束的mask交換，即本提案。
5. 同次數的隨機交換對照，避免把任何mask擾動都視為方法效果。

各方法固定checkpoint、split、Tm、q、fine-tune步數與RNG，校正樣本從train取，validation選超參數，test最後評估。記錄交換時間、交換數、誤差改善及逐seedaccuracy差異。

## 9. 第二個待探索方向：跨Cluster的通道覆蓋修補

另一個假說是：各cluster獨立選mask，可能讓某個input channel在整層的所有clusters都失去連接。對1×1卷積而言，這可解讀為整層完全不再讀取該channel。

可在固定每cluster q的條件下，透過cluster內一換一修補具有高校正重要性、卻在全層被完全移除的channel；此時目標改成重要input channel的全層覆蓋，以及總重建誤差控制。

這需要先量測是否真的存在重要channel的整層斷接。不能假設channel覆蓋越廣越好；有些channel本來就冗餘。同樣不能要求全部channel都保留：若所有clusters的總保留位置不足，這個要求在數量上不可行。

此方向也維持cluster內共享mask、固定q與原硬體資料格式，但其新穎性及與既有connectivity/diversity-aware pruning的重疊需另查證。它與「保護整層後跨層補剪」是不同的待測問題。

## 10. 目前決策

先保存提案與參考論文，接著應以少量校正資料驗證共享mask的誤差分布及交換可行性。未獲實驗證據前，不能將本提案列為有效成果或論文既定創新。

本文最初整理的是研究設計。其後使用者授權自主執行，已用獨立腳本完成診斷、小規模訓練與有限調整，未更動原剪枝核心。依預先設定的validation門檻停止本輪；不能將原提案的預期效果當成已完成的分類精度成果。

參考來源：[文獻紀錄](2026-11-01_pruning_reference_papers.md)。既有證據：[敏感層與固定預算研究紀錄](2026-11-01_sensitivity_budget_findings.md)。
