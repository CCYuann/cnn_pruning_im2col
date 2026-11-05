# 實驗結果總覽

## 研究總檔

[截至2026-11-05的研究發現與完整實驗紀錄](../research_notes/RESEARCH_FINDINGS_MASTER_2026-11-05.md)：整合目前主線亮點、所有主要已測方向、改善條件、未奏效的數據、已確認問題與可能原因，以及量化/硬體界線。先讀此檔可避免把舊test、近期validation或不同剪枝配額混在一起。

## 已完成：共享Mask補償的便宜控制與FT30

`compensation_controls_ft30/`：2026-11-05完成三個獨立ResNet50 Dense seeds、Tm8、約90%原共享mask的24條軌跡。FT10平均：A/BN93.633%、scalar93.740%、B94.133%、C94.160%；FT30：A/BN94.020%、scalar94.040%、B94.253%、C94.080%。B對scalar的配對提升由FT10的+0.393pp縮為FT30的+0.213pp，長微調對A只2/3seeds為正，不能宣稱普遍或顯著。兩種非共享參考FT10為global94.600%、quota-matched94.580%；B補回約52.8%的同配額經驗差距。63項tests與42個checkpoint稽核通過；沒有新test forward或RTL變更。達預先5seed投入資格，但5seeds/跨模型/稀疏度掃描尚未做。[完整表格與圖](compensation_controls_ft30/README.md)、[研究判斷](compensation_controls_ft30/FINDINGS.md)、[資料ZIP](compensation_controls_ft30.zip)、[凍結計畫](../research_notes/2026-11-05_compensation_controls_plan.md)。

## 已完成：Activation-aware固定配額分組

`activation_aware_fixed_quota_grouping/`：2026-11-02完成ResNet50/CIFAR-10 seed0、Tm8，完全沿用上一輪逐層q。四格FT10 validation：L1連續93.20%、L1最佳化93.38%、activation連續92.48%、activation最佳化92.90%。新指標兩種配置均較對應L1低（-0.72/-0.48pp）；雖降低activation proxy，未轉成精度改善，本pilot不擴跑或調參。53項測試與53層的quota/score/mask稽核通過。單seed有限負結果，不代表所有activation-aware方法無效；無補償/保護/量化，不改RTL、不跑test。[結果與圖表](activation_aware_fixed_quota_grouping/README.md)、[研究判斷](activation_aware_fixed_quota_grouping/FINDINGS.md)、[資料ZIP](activation_aware_fixed_quota_grouping.zip)、[凍結計畫](../research_notes/2026-11-02_activation_aware_grouping_plan.md)。

## 已完成：固定剪枝量的 Filter 重新分組

`fixed_quota_filter_regrouping/`：2026-11-02完成ResNet50/CIFAR-10、Tm8、實際89.9987%卷積權重剪枝的seed0七分支。每層所有cluster固定相同保留數，所有方法的weight/MAC預算相同。FT10 validation：連續93.20%、三次隨機平均93.34%、L1排序92.78%、Jaccard93.04%、Jaccard加交換93.38%。交換版關閉13.47% L1缺口，但只比連續高0.18pp，未達預設0.3pp擴展門檻，因此未擴三Dense seeds。48項測試及reference/實際mask稽核通過。不能宣稱穩定精度優勢；無補償/保護/量化，不改RTL、不跑test。[結果與圖表](fixed_quota_filter_regrouping/README.md)、[資料ZIP](fixed_quota_filter_regrouping.zip)、[研究判斷](../research_notes/2026-11-02_fixed_quota_regrouping_findings.md)、[凍結計畫](../research_notes/2026-11-02_fixed_quota_regrouping_plan.md)。

## 已完成：補償方法（三seed初步正結果）

「補償方法」為使用者指定簡稱，指固定共享mask後重算保留權重；包含B Dense-input簡單版與C student-prefix逐層版，不更動既有實驗目錄或凍結設定。

`fixed_mask_weight_compensation/`：2026-11-02完成3seeds、9個新FT分支。ResNet50/Tm8/同mask/實際約90%剪枝，FT10 validation平均：原權重93.633%、Dense-input補償94.133%、逐層補償94.160%。逐層版相對原權重三seed均改善，平均+0.527pp；但僅比Dense-input平均高0.027pp，未證明額外複雜度有優勢。39項測試與資料/mask稽核通過；未讀test、未做INT8，不能宣稱獨立test、硬體能效或新穎性已成立。[結果與圖表](fixed_mask_weight_compensation/FINAL_REPORT.md)、[資料ZIP](fixed_mask_weight_compensation.zip)、[研究判斷](../research_notes/2026-11-02_weight_compensation_findings.md)、[工作紀錄](fixed_mask_weight_compensation/EXECUTION_LOG.md)。

## 最新總結：基準審核與適應後梯度

2026-11-02已完成兩階段12個新訓練分支，沒有使用test。[整體總結與圖表](BASELINE_AND_GRADIENT_20261102.md)、[證據ZIP](baseline_and_gradient_20261102.zip)。

`adapted_gradient_mask_study/`：完成ResNet50 seed0四分支2+8epoch測試。無交換93.46%、梯度全部交換93.52%、雙CE門檻93.54%、匹配隨機93.58%，均未超過本輪最佳控制93.68%。未擴三seed。無交換完整最終模型與原10epoch l1逐bit相同；保留optimizer／AMP／RNG，不增加訓練epochs。[結果](adapted_gradient_mask_study/FINAL_REPORT.md)、[凍結方法](../research_notes/2026-11-02_adapted_gradient_plan.md)、[問題處理紀錄](adapted_gradient_mask_study/EXECUTION_LOG.md)。

## 已完成：Cross-filter配額粒度審核

`spatial_quota_baseline_study/`：2026-11-02完成ResNet34/50、l1/top2的8個seed0分支。固定Tm8、每cluster保留數與10epoch FT；spatial-flat依序+0.14/-0.04/-0.06/+0.26pp，未達預設0.3pp門檻，未補三seed。這不是等效或顯著性結論。spatial是原文歧義下的解讀，不稱已確認原始實作。[結果與圖表](spatial_quota_baseline_study/BASELINE_REPORT.md)、[原文核對](spatial_quota_baseline_study/PAPER_AUDIT.md)、[工作紀錄](spatial_quota_baseline_study/EXECUTION_LOG.md)。

## 已完成：全模型分類Loss篩選共享Mask

`task_loss_mask_study/`：2026-11-02完成ResNet50 seed0四個搜尋＋10epoch fine-tune分支。保持每cluster配額、Tm8及實際89.9994%剪枝，分別測frozen BN／候選BN校正、search-only／雙資料loss門檻。audit CE改善，但主方法dual為93.24%／93.20%，低於l1的93.46%；search-only最高93.54%，只有+0.08pp，尚非可信優勢。兩版未過預設門檻，未補三seed、未使用test。[結果與圖表](task_loss_mask_study/FINAL_REPORT.md)、[計畫](../research_notes/2026-11-02_task_loss_mask_plan.md)、[工作紀錄](task_loss_mask_study/EXECUTION_LOG.md)、[資料ZIP](task_loss_mask_study.zip)。

## 已完成：固定配額共享Mask修補初步驗證

`shared_mask_repair_study/`：2026-11-01完成五模型seed0診斷、兩模型16個新fine-tune分支及一次mean-safe調整。固定Tm8各cluster保留數，重建誤差改善但分類pilot未支持修補優於l1，未擴大三seed、未評估test。[完整結果](shared_mask_repair_study/FINAL_REPORT.md)、[問題與處理紀錄](shared_mask_repair_study/EXECUTION_LOG.md)、[資料ZIP](shared_mask_repair_study.zip)。這是初步負結果，不代表所有mask修補都無效。

## 已完成成果與研究判斷：2026-11-01

下述45個importance-only及30個重新分配任務均於2026-10-27完成。結果顯示現有保護與重新分配方案在MobileNetV2有效，但並非所有模型受益。完整結論、相同實際剪枝預算的比較限制與待驗證原因，已保存於 [研究紀錄](../research_notes/2026-11-01_sensitivity_budget_findings.md)；實驗總表見 [最終結果](FAIR_PRUNING_R0P9_FINAL.md)。

## 最新公平比較：實際約90%剪枝

`fair_criterion_r0p9_tm8/`：2026-10-27 啟動。固定 Dense checkpoint、Tm8、連續分組、關閉全部敏感層保護與 cap，比較 `l1_sum / max_abs / top2_mean_abs / top4_mean_abs`。沿用15組 l1_sum 基準，新增45個 fine-tune 任務。進度見 `fair_criterion_r0p9_tm8/status.json`，實驗設定與自動更新圖表見 [該批說明](fair_criterion_r0p9_tm8/README.md)。未滿三個 seed 的結果只供觀察進度。

`sensitivity_redistributed_r0p9_tm8/`：等待上述45個任務完成後自動接續。保留top4敏感層規則，移除其他層75% cap，將缺額重新分配並精確匹配既有約90%權重剪枝預算；比較l1/top2，共30個新任務。[方法與排程說明](sensitivity_redistributed_r0p9_tm8/README.md)。

這個資料夾存放 CNN pruning 實驗輸出的 CSV、圖表與說明文件。主要目標是比較論文原始 cross-filter structured pruning 的 `l1_sum`，以及後續改良的 `max_abs`、`top2_mean_abs`、`top4_mean_abs`、early-layer protected/cap 版本。

## 目前主要結論

- `max_abs` 在多模型、多 pruning ratio 平均上通常優於 `l1_sum`，但不是每個模型都贏。
- `top2_mean_abs` / `top4_mean_abs` 對部分模型有效，尤其 GoogLeNet 表現穩定較好。
- early-layer protected + ratio cap 對 ResNet 系列有幫助，但對 DenseNet121 不一定好。
- 10-way GPU 並行雖然能跑完，但 wall time 很長；後續建議每張 GPU 同時跑 4-5 個 job。

## 資料夾分類

- `fixed_checkpoint_resnet34/`：正式 ResNet34 / CIFAR-10 固定 Dense 基準，以及已完成的 18 組公平剪枝比較、validation-only sensitivity ranking、三 seed 統計與圖表。
- `paper_like_bs128/`：ResNet34 + CIFAR-10、batch size 128、較接近論文流程的 seed 實驗。重點是比較 `l1_sum` 與 `max_abs`。
- `multi_model_cifar10_split/`：CIFAR-10 多模型比較，包含 ResNet34、ResNet50、GoogLeNet、DenseNet121、MobileNetV2、EfficientNetV2-M。
- `ratio_sweep_cifar10_gpu1_split/`：CIFAR-10 不同 pruning ratio 比較，包含 `0.5`、`0.6`、`0.7`、`0.8`。
- `improvement_topk_protect_cifar10_split/`：後續改良實驗。包含 `top2_mean_abs`、`top4_mean_abs`，以及 early-layer protected + max layer prune ratio cap。已完成 `r0p6` 與 `r0p7`。
- `layer_sensitivity_summary/`：單層 pruning sensitivity 總整理，用來找每個模型哪些 layer 對剪枝最敏感。
- `sensitivity_aware_topk_cifar10_split/`：使用 layer sensitivity ranking 保護敏感層後的 top-k/protected-cap 實驗。
- `sensitivity_aware_l1_sum_cifar10_split/`：把論文原始 `l1_sum` 也套用 sensitivity-aware 保護策略，用來和新方法最佳版公平對照。
- `next_experiments_plan/`：後續實驗規劃。
- `smoke_tests/`：小型流程測試，只用來確認程式可跑、CSV 欄位正常，不用來做 accuracy 結論。

## 優先閱讀順序

如果要看目前最完整的改良方法結果，先看：

```text
runs/improvement_topk_protect_cifar10_split/README.md
```

如果要看 `max_abs` 對原始 `l1_sum` 的主要對照，先看：

```text
runs/multi_model_cifar10_split/README.md
runs/ratio_sweep_cifar10_gpu1_split/README.md
```

如果要看最接近論文流程的 ResNet34 seed 結果，看：

```text
runs/paper_like_bs128/README.md
```

## 重要 CSV

- `improvement_topk_protect_cifar10_split/summaries/all_r0p6_r0p7_results_combined.csv`：改良方法 `r0p6/r0p7` 全部結果合併。
- `improvement_topk_protect_cifar10_split/summaries/variant_summary_r0p6_r0p7.csv`：每個 cross-filter variant 對照同 run 的 dense/global baseline。
- `improvement_topk_protect_cifar10_split/summaries/best_cross_summary_r0p6_r0p7.csv`：每個 model/ratio 最佳 cross-filter variant。
- `multi_model_cifar10_split/all_models_combined_results.csv`：多模型 `l1_sum/max_abs` 對照。
- `ratio_sweep_cifar10_gpu1_split/all_results_combined.csv`：不同 pruning ratio 對照。
- `layer_sensitivity_summary/layer_sensitivity_ranked.csv`：每個模型/criterion 的敏感層排名。
- `layer_sensitivity_summary/top10_sensitive_layers_by_ratio.csv`：每個 ratio 下最敏感的前 10 層。
- `sensitivity_aware_topk_cifar10_split/best_sensitivity_aware_summary.csv`：sensitivity-aware 每個 model/ratio 最佳 cross-filter 結果。
- `sensitivity_aware_l1_sum_cifar10_split/l1_sum_sensitivity_aware_summary.csv`：`l1_sum + sensitivity-aware` 每個 model/ratio 摘要。
- `sensitivity_aware_l1_sum_cifar10_split/comparison_l1_sum_vs_newbest_sensitivity_aware.csv`：`l1_sum + sensitivity-aware` 對照新方法最佳 sensitivity-aware 版本。

## CSV 欄位說明

- `dataset`：資料集，例如 CIFAR-10。
- `model`：模型名稱。
- `method`：實驗方法，例如 dense baseline、dense extra training、global unstructured pruning、cross-filter pruning。
- `criterion`：cross-filter pruning 的 importance 計算方式，例如 `l1_sum`、`max_abs`、`top2_mean_abs`、`top4_mean_abs`。
- `ratio_source`：cross-filter pruning 的 sparsity ratio 來源。`global_cluster` 代表先做 global unstructured pruning，再用對應 cluster 的 sparsity 當 reference。
- `direct_acc`：剪枝後、fine-tuning 前的測試 accuracy。
- `after_extra_or_ft_acc`：dense extra training 後的 accuracy，或 pruned model fine-tuning 後的 accuracy。
- `pruning_ratio`：實際權重剪枝比例。
- `mac_reduction`：根據 mask 估算的有效 MAC 減少比例。
- `pe_utilization`：根據 mask workload 估算的 PE 使用率。
- `cycle_pe_utilization`：更接近論文 PE cluster 同步執行假設的 cycle-based PE 使用率。
- `cycle_reduction`：在 paper-like PE cluster 模型下估算的 cycle 減少比例。
- `estimated_speedup`：估算 speedup，計算方式是 dense cycles / pruned cycles。

## 整理備註

- `improvement_topk_protect_cifar10_split/` 內部已整理成 `summaries/`、`per_run_csv/`、`logs/`、`scripts/`、`charts/`。
- 失敗的 GPU0 r0p7 OOM log 已移到 `logs/failed/gpu0_r0p7_oom/`，成功補跑結果在 `logs/successful/r0p7_rerun_gpu1/`。

## 後續建議

目前 sensitivity-aware 對照已補上 `l1_sum`。整體看，新方法 `top2_mean_abs / top4_mean_abs + sensitivity-aware layer protection` 比 `l1_sum + sensitivity-aware` 更穩，但 ResNet50 ratio 0.6、ResNet34 ratio 0.6 是 `l1_sum` 較好。下一步不要全部方法都補 seed，建議只挑每個模型或整體平均較好的 variant，再補 3 個 seed 做穩定性檢查。
