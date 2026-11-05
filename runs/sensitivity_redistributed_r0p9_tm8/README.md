# 後續排程：Sensitivity Protection + 精確剪枝預算重新分配

建立日期：2026-10-27。已獲使用者授權，第一階段完成後自動執行，不需要再次確認。

## 執行順序

1. 等 `../fair_criterion_r0p9_tm8/` 的45個新任務全部完成並通過驗證。
2. 以既有正式 validation top2 sensitivity ranking 作配置，進行本批30個任務。
3. 每個任務完成就驗證實際 sparsity、保護條件與整數預算，更新CSV、比較圖與RESULTS.md。
4. 全部成功後產生 `../FAIR_PRUNING_R0P9_FINAL.md`，並將兩階段有效CSV、圖表、方法說明與程式快照打包為 `../fair_pruning_r0p9_complete_20261027.zip`。

第一階段失敗且已結束時，會降低為單工重試一次；若仍失敗或狀態為error，記錄原因，不會將缺資料的階段當成已完成。

## 這一批回答什麼

固定實際約90%的 convolution weight sparsity，比較以下四個實驗臂：

| Importance | 不保護 | 敏感層保護＋重新分配 |
|---|---|---|
| l1_sum | 沿用既有15個正式結果 | 新增15個 |
| top2_mean_abs | 沿用第一階段15個結果 | 新增15個 |

這可分別觀察同一配置下 importance 的差異，以及同一 importance 下 protection/redistribution 的差異。不依 test accuracy 挑選要跑的 criterion；預先固定為l1與top2。

兩個保護版都使用同一份top2 validation sensitivity排名與相同配置規則。這有利於控制比較條件，但不表示是l1本身的最佳sensitivity配置。

## 固定設定

- 五模型：ResNet34、ResNet50、GoogLeNet、DenseNet121、MobileNetV2。
- Seeds=0/1/2，Tm8，連續分組，沿用固定100-epoch Dense checkpoint。
- Fine-tune=10 epochs，LR=0.001，batch=128，AMP float16。
- 最敏感第1層完全不剪，其後3層最多剪20%。以 `floor(0.2*K)` 實施真正的比例上限；depthwise小K也不會因四捨五入而超過20%。
- 其他層移除原本75% cap，但每cluster至少保留1個K position。
- 實際權重剪枝數對齊同 model/seed 的無保護 l1_sum；使用整數K-position數量配額。
- GPU1，並行3個任務。等待第一階段時不使用GPU。

本次15個 model/seed 均已通過理論容量檢查；最大可達比例見 `allocation_feasibility.csv`。可行表示數量上放得下，不代表準確率一定能維持。部分DenseNet種子因保護比重大，其他層必須承擔極高剪枝率，精度可能下降。

## 重新分配演算法

先產生與基準相同的global unstructured cluster reference，對各cluster設定：

```text
initial_prune_k = min(round(reference_ratio*K), max_prune_k)

最敏感層：max_prune_k = 0
其餘3個敏感層：max_prune_k = floor(0.2*K)
其他層：max_prune_k = K-1
```

令該layer的正式validation平均accuracy drop為s，分配的可調程度為：

```text
mobility = 1/(1+max(0,s))
r(lambda) = min(max_prune_k/K, initial_prune_k/K + lambda*mobility)
```

以二分搜尋找lambda，使總移除權重達到預算；最後依小數餘額作整數修正，確保總數精確吻合。低敏感層通常會承擔較多額外剪枝，但也受原reference與各層K大小影響。

本批所有convolution Cout均可被8整除，每個cluster移除1個K恰好移除8個權重；因此能精確匹配原l1_sum預算。程式對partial cluster會拒絕套用這個整數求解器，避免假裝能任意表示預算。

這是預先固定的啟發式分配，不是已證明最優的validation搜尋。未加入反覆試剪/微調，也未根據test選參數。

## 公平性與限制

本批匹配的是實際 **weight sparsity**。不同layer配置會改變MAC reduction，因為高解析度層每個權重使用次數不同。因此不能宣稱同時匹配MAC或硬體延遲；CSV與圖表會同時列出MAC reduction。

原始模型載入、訓練、mask enforcement及評估直接沿用核心程式。另以worker wrapper替換reference配置與方法標籤，不修改正在執行的第一階段核心程式。

## 檔案

- `status.json`：等待、執行、完成或失敗狀態。
- `experiment_manifest.json`：設定、checkpoint/ranking/程式hash。
- `allocation_feasibility.csv`：15個精確目標與保護條件下的容量。
- `allocations/<run>/cluster_allocation.csv`：每個cluster的reference、cap、分配數量。
- `allocations/<run>/budget.json`：精確權重預算與保護層清單。
- `per_run_csv/`、`logs/`：原始實驗與資源紀錄。
- `all_results_combined.csv`：四個實驗臂的逐seed數據。
- `paired_protection_effect.csv`：同seed、同criterion，加保護與重新分配的FT精度差。
- `method_summary.csv`、`RESULTS.md`、`matched_budget_accuracy.png`：自動更新總表與圖表。

若需手動恢復，從專案根目錄執行 `python scripts/run_pruning_followup.py --gpu 1 --max-jobs 3`。排程有lock，避免重複啟動。
