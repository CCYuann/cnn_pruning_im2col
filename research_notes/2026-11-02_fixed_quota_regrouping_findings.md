# 固定剪枝量重新分組：seed0可行性結果

## 問題與實驗

原相似分組實驗同時依新組員重新計算global_cluster配額，不能隔離組員與預算分配。這次以ResNet50/CIFAR-10、Tm8、同一Dense seed0，從global 90% reference取得每層密度，換算固定q_l，同層所有cluster保留相同q_l。所有7分支實際卷積權重稀疏度89.9987171194%、保留2,345,024權重、233,504,768 MAC/image，含1x1一般卷積，不用保護/cap、補償或新importance。

這是新的layer-uniform quota控制，不是原global_cluster配置的重跑；不能把不同配置下舊分組增益的差額直接全部歸因於配額。

## 結果

| 方法 | 共同BN副本 FT0 (%) | FT2 (%) | FT10 (%) | 對連續分組 (pp) | L1缺口閉合 (%) |
|---|---:|---:|---:|---:|---:|
| 連續 | 68.02 | 92.78 | 93.20 | 0.00 | 0.00 |
| 隨機0 | 65.82 | 92.96 | 93.20 | 0.00 | -0.03 |
| 隨機1 | 70.24 | 92.60 | 93.52 | +0.32 | +0.04 |
| 隨機2 | 70.14 | 92.42 | 93.30 | +0.10 | +0.02 |
| L1 norm排序 | 52.98 | 91.74 | 92.78 | -0.42 | -2.91 |
| Jaccard | 72.18 | 92.54 | 93.04 | -0.16 | +9.74 |
| Jaccard +交換 | 72.88 | 92.64 | 93.38 | +0.18 | +13.47 |

三個random初始化FT10平均93.34%、初始化sample std0.1637pp；不是三個獨立Dense seeds。交換版僅高於random平均0.04pp，且低於random1，不支持目前有穩定精度優勢。沒有正式顯著性結論。

## 決策

依事前計畫，只有交換版FT10對連續>=0.3pp且不低於Jaccard/L1排序，才擴seeds1/2。實際只有+0.18pp，故完成全部seed0七分支後停止，不改門檻或調搜尋次數追高。

交換版第9輪93.72%、第10輪93.38%；報固定第10輪，不拿第9輪代替。這是觀察，未證明下降原因，不據此改成best epoch或增加epochs。

## 能說與不能說

- 固定q下，Jaccard及交換搜尋確實降低共享mask刪除的L1質量；交換版的代理代價不高於起點，partition及mask散佈檢查通過。
- 代理目標改善沒有一致轉成FT10精度改善。Jaccard在共同BN副本FT0較高，FT10反而低於連續，不能只以無微調診斷作結論。
- 這個有界L1搜尋不足以支持升為論文主要精度貢獻，但不等於所有分組或activation/task-aware目標都無效。
- 不能由單seed推論其他模型、高稀疏度或其他訓練預算；本validation亦曾用於其他研究開發，沒有獨立test證據。
- 所有分組離線完成，模型channel意義不變；未評估非連續filter ID的實際SRAM/參數查表/寫回代價，不能宣稱硬體零成本或加速。
- 補償方法既有3seed結果仍為獨立研究線，本輪未混入，不用不同quota配置的accuracy直接排序兩條方法。

## 實作與問題處理

獨立helper/worker/controller，不改核心或RTL。4輪有界交換，依L1目標接受互不共用cluster的交換，不看validation選分組。48項單元及integration tests通過，另以Dense reference重新產生quota、對實際mask與MAC budget核對，檢查原filter ID散佈及FT後禁止位置維持0。

覆核發現原controller缺少擴展前完整驗證、zero-gap aggregate處理及optimizer/resume integration tests。10:12以SIGINT停止本研究controller，其finally只停止自己的兩個未完成workers；保留epoch6/4 checkpoint。未改worker、groups或訓練規則，只修controller驗證與新增3項tests。10:14明確接受來源amendment後續跑；原始與修訂來源、SHA、事件均保留。已有兩個完整分支未重跑。未干涉其他程序，無OOM或數值錯誤。

訓練seconds欄累計已保存epoch及續跑時間，不完整包含被中斷且重做的部分epoch，不能當作全研究總成本或獨占GPU benchmark。所有原始時間見execution.jsonl。

## 資料入口

- [完整結果及圖](../runs/fixed_quota_filter_regrouping/README.md)
- [逐分支數據](../runs/fixed_quota_filter_regrouping/validation_results.csv)
- [配對統計](../runs/fixed_quota_filter_regrouping/paired_summary.csv)
- [配額稽核](../runs/fixed_quota_filter_regrouping/BUDGET_VALIDATION.json)
- [執行及修訂紀錄](../runs/fixed_quota_filter_regrouping/EXECUTION_LOG.md)
- [交接ZIP](../runs/fixed_quota_filter_regrouping.zip)，不含大型最終模型、訓練resume及BN影像；它們保留於本機。
