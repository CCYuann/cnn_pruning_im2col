# 補償方法：固定共享Mask權重補償

## 名稱與定義

2026-11-02依使用者指定，這個研究方向統一簡稱「補償方法」。後續提到此名稱，指固定剪枝mask、kept_k、kept_count與filter分組，只用校正資料重算保留權重，再依相同預算fine-tune；不是改importance、換剪枝位置或把連接加回來。

- 補償方法B／簡單版：Dense-input ridge補償。
- 補償方法C／逐層版：student-prefix sequential ridge補償。
- A原權重為控制組，不稱為補償方法。名稱是專案簡稱，不代表已確認學術新穎性。

## 完成範圍

ResNet50/CIFAR-10、Tm8、連續分組、flat l1 mask、每seed實際約90%conv-weight sparsity。每seed A/B/C的mask逐bit相同，FT均10epochs，計畫在看結果前凍結。完成9個分支，沒有讀test，也沒有做INT8或RTL變更。

## Validation數據

| 方法 | FT0 mean (%) | FT0共同BN校正副本 (%) | FT2 mean (%) | FT10 mean ± sample std (%) |
|---|---:|---:|---:|---:|
| A 原保留權重 | 10.707 | 75.453 | 93.300 | 93.633 ± 0.541 |
| B Dense-input ridge | 81.540 | 90.120 | 93.747 | 94.133 ± 0.270 |
| C student-prefix逐層ridge | 91.467 | 90.453 | 93.907 | 94.160 ± 0.299 |

C-A逐seed為+0.48/+0.84/+0.26pp，平均+0.527pp，配對sample std0.293pp。B-A為+0.40/+0.94/+0.16pp，平均+0.500pp。C-B為+0.08/-0.10/+0.10pp，平均只有+0.027pp，配對sample std0.110pp。

## 能說與不能說

- 本設定中，兩種固定support補償的FT10 validation都在三seeds高於A，較先前只換mask位置的試驗更值得追查。
- 不能由C-A正值宣稱C一定優於B；FT10兩者接近且逐seed不一致。若主打最終accuracy，較簡單B應是重要強控制，而非被省略的消融。
- C在不FT時較B高，C的FT2也高於A的FT10。這是先前已指定的低FT預算線索；要證明效率還需算校正、求解與資料取得的額外成本，不以epochs直接換算wall-time speedup。
- 原始A直接accuracy很低有BN不匹配因素；共同BN副本校正後A約75.45%，B/C約90%，不能只用10.7→91.5的原始差距包裝全部效果。BN副本從未帶回主FT。
- 同一validation曾用於多輪開發，三seeds仍非完全獨立測試。不能據此宣稱統計顯著、跨模型普遍有效或勝過其他論文。
- Ridge重建、保留權重補償與二階方法已有文獻，這輪證據是cross-filter共享support約束下的具體改編，不把概念本身稱為原創。

## 已驗證條件

39項tests、teacher/patch座標、資料隔離、mask與q、禁止位置、非卷積參數及來源SHA通過；seed0 A最終模型逐bit重現舊控制。最大q的校正benchmark與rank-deficient toy均通過。43個seed0 q=0 clusters未被恢復。

## 量化與硬體界線

本機找到Q15 scalar primitive及ResNet34完整參考，已檢查的mapping/forward僅兩卷積block，不支援ResNet50 Bottleneck conv3。必須先取得或建立已驗證的ResNet50參考，不能把oneDNN當硬體bit-exact，也不能把浮點改善直接當INT8改善。

權重範圍、離線時間與峰值記憶體已保存。共享mask不變只固定規劃的權重數與MAC，不提供實測能耗或SRAM讀取收益證明。

## 建議下一階段

先保留A/B/C三個方法，不繼續調lambda尋找validation更高值。分開規劃：鎖定評估規則後的獨立test確認、較簡單B與C的低FT預算比較、ResNet50 Q15相容參考，以及第二個模型的有限驗證。每一項需另有控制與停止條件，不一次把全部自由度混在一起。

詳見 [完整報告](../runs/fixed_mask_weight_compensation/FINAL_REPORT.md) 與 [工作紀錄](../runs/fixed_mask_weight_compensation/EXECUTION_LOG.md)。
