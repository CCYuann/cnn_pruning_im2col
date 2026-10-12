# 實驗結果總覽

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
