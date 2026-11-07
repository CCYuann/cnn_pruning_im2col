# Cross-Filter Structured Pruning 研究改良方向

## 目前主線與完整紀錄（2026-11-05）

請先讀[研究發現與完整實驗總檔](research_notes/RESEARCH_FINDINGS_MASTER_2026-11-05.md)。其中區分test/validation、實際剪枝預算、三Dense seeds與單seed探索，逐項記錄做過什麼、數據、改善條件、已證實限制與尚未證實的失敗原因。

目前以B Dense-input ridge的共享mask權重補償為主線。三seed FT10對BN-only平均+0.500pp、對per-filter scalar +0.393pp；FT30縮為+0.233/+0.213pp。C在FT30三seeds都低於B，不再因早期FT0或FT10的小優勢當成更好的最終主線。[最新控制/FT30結果](runs/compensation_controls_ft30/FINDINGS.md)。

停止新增分組搜尋與RMS activation加權實驗。補償感知mask僅為尚未實作的延伸；5seeds、跨模型、80/95%與lambda/校正量消融也尚未完成。下文是歷史結果及早期提案，不能將「預期優點」或舊的「目前結論」當成最新已驗證優勢。

## 歷史結果：補償方法初次FT10（2026-11-02）

使用者指定簡稱為「補償方法」：不改mask與kept_count，只重算保留權重。B為Dense-input簡單版，C為student-prefix逐層版；後續討論沿用此名稱。

2026-11-02完成ResNet50三seeds、9個10epoch分支。相同共享mask與實際約90%剪枝，validation FT10：原權重93.633%、Dense-input補償94.133%、student-prefix逐層補償94.160%。主方法C相對A逐seed+0.48/+0.84/+0.26pp，平均+0.527pp；但C-B只有+0.027pp且seed1為負，不支持逐層版本一定更好。

這是目前比前幾輪更一致的初步正面訊號，仍不是獨立test、跨模型或學術新穎性證明。39項測試及mask/資料/非卷積參數稽核通過；本機Q15完整路徑未覆蓋ResNet50 Bottleneck，因此INT8尚未驗證。[完整結果](runs/fixed_mask_weight_compensation/FINAL_REPORT.md)、[判斷與限制](research_notes/2026-11-02_weight_compensation_findings.md)。

## 最新追加：配額粒度與適應後梯度

2026-11-02又完成12個新訓練分支。原文Figure4/5及公式未唯一指定spatial subcluster配額，因此不能把目前flat K直接判為錯誤。固定每cluster q的兩種解讀，在ResNet34/50 seed0的差異為-0.06至+0.26pp，未達預設擴大門檻；不是等效證明。

接續的ResNet50適應2epochs後共享gate梯度候選，總訓練仍10epochs，主方法93.54%、隨機控制93.58%，都低於本輪最佳控制93.68%。未支持此版本有優勢，未使用test。[總結與完整證據](runs/BASELINE_AND_GRADIENT_20261102.md)。不把單seed小提升或數學導數正確當成已成立分類貢獻。

## 最新追加：2026-11-02

已完成固定配額的全模型分類loss接受／否決mask搜尋，ResNet50 seed0共四個10epoch分支，含一次BN proxy有界調整。主方法dual均未勝過l1；search-only最高+0.08pp，未達預設擴大門檻。這輪再次顯示訓練前proxy改善不保證fine-tune後精度提升，但不能由單模型單seed斷言所有task-aware mask方法無效。[完整結果](runs/task_loss_mask_study/FINAL_REPORT.md)、[預先計畫](research_notes/2026-11-02_task_loss_mask_plan.md)。

## 最新研究紀錄：2026-11-01

已完成實際約90%剪枝的importance-only比較與敏感層保護／預算重新分配實驗。保護但不補剪會降低實際sparsity；補足預算的現有方法在MobileNetV2有幫助，但在其他受測模型退步，DenseNet121尤其明顯。請優先閱讀 [正式結果與研究判斷](research_notes/2026-11-01_sensitivity_budget_findings.md)。

下文保留早期規劃與假設，不能直接當成已完成驗證的結論；正式數據及限制以該研究紀錄與其連結的CSV為準。

新增 [剪枝參考論文紀錄](research_notes/2026-11-01_pruning_reference_papers.md) 與 [固定配額共享mask修補提案](research_notes/2026-11-01_shared_mask_repair_proposal.md)。截至2026-11-01已實作並完成五模型診斷及兩模型16個訓練分支；重建誤差下降，但原版與mean-safe調整均未在seed0 validation勝過l1，未擴大三seed或使用test。見 [完整負結果及限制](runs/shared_mask_repair_study/FINAL_REPORT.md)。學術新穎性仍未確認。

## 目前專題定位

目前這個專題是 CNN 剪枝與推論成本分析的 Python 軟體模擬。程式目前已經支援：

- Dense CNN baseline。
- Global unstructured pruning baseline。
- Cross-filter structured pruning。
- 剪枝後 fine-tuning。
- Accuracy 比較。
- Weight pruning ratio 統計。
- 理論 MAC reduction 統計。
- PE workload utilization 估算。
- ResNet34 + CIFAR-10 實驗。
- 偵測與剪枝 1x1 convolution layer。

這個方向比較接近論文中的「演算法層級實驗」，但還不是完整的硬體加速器實作。目前我們是用軟體估算 MAC reduction 與 PE workload balance，來分析剪枝方法是否適合硬體加速。

## 目前版本的主要限制

目前 cross-filter pruning 主要採用論文的 weight-based importance：

```text
G[c, r, s] = sum_m |W[m, c, r, s]|
```

這個方法可以讓同一個 filter cluster 裡面的 filters 共用相同 mask，因此 PE 工作量會比較平衡。

但是它只考慮 weight magnitude，沒有考慮：

- input activation 分布。
- 不同 layer 對 accuracy 的敏感度。
- 逐步剪枝與恢復。
- 1x1 conv 與 3x3 conv 的不同特性。
- 更完整的 fine-tuning 策略。

因此，這些地方都可以成為後續改良與創新的方向。

## 可改良方向

### 1. 改良 Importance Algorithm

原論文使用 L1-based cross-filter importance。我們可以比較或提出不同的 importance score：

```text
l1_sum    = sum_m |W[m, c, r, s]|
l2_sum    = sqrt(sum_m W[m, c, r, s]^2)
max_abs   = max_m |W[m, c, r, s]|
min_abs   = min_m |W[m, c, r, s]|
random    = 隨機 structured mask baseline
```

可以探討的研究問題：

```text
在維持相同 cross-filter structured mask 與 PE workload balance 的前提下，
不同 importance criterion 是否能改善剪枝後 accuracy？
```

這是最容易實作與比較的改良方向，因為剪枝結構不變，只改變「如何判斷重要性」。

### 2. Activation-Aware Cross-Filter Structured Pruning

論文主要只看 weight magnitude。可以進一步加入 activation 統計資訊：

```text
G[c, r, s] = sum_m |W[m, c, r, s]| * A[c]
```

其中：

```text
A[c] = input activation channel c 的平均絕對值
```

直覺是：

```text
如果某個 input channel 經常被強烈啟動，
那麼與它相連的 weight 可能比單純 weight magnitude 顯示得更重要。
```

這對 1x1 convolution 特別有意義，因為 1x1 conv 本質上是在做 channel mixing。

預期優點：

```text
在保持 cross-filter structured sparsity 與 PE balance 的同時，
比單純 weight-based L1 pruning 更能維持 accuracy。
```

可使用的題目名稱：

```text
Activation-Aware Cross-Filter Structured Pruning for Efficient CNN Inference
```

中文題目：

```text
基於啟動值感知之跨濾波器結構化剪枝方法於 CNN 推論加速之研究
```

### 3. Layer-Adaptive Pruning Ratio

目前常見做法是所有 layers 使用相同 pruning ratio，例如全部剪 60%。

但實際上不同 layer 對 accuracy 的敏感度不同。可以設計 layer-adaptive pruning ratio：

```text
前面較敏感的 layers：剪少一點
後面較不敏感的 layers：剪多一點
1x1 conv layers：可考慮剪多一點
3x3 conv layers：可考慮剪保守一點
```

可能方法：

```text
1. 對每一層暫時剪掉少量 weights。
2. 測量 validation accuracy drop。
3. accuracy drop 大的 layer 視為敏感層。
4. 敏感層給較低 pruning ratio。
5. 不敏感層給較高 pruning ratio。
6. 維持整體目標 sparsity 不變。
```

可以探討的研究問題：

```text
在相同總 sparsity 或相同 MAC reduction 下，
layer-adaptive cross-filter pruning 是否能得到更高 accuracy？
```

### 4. 1x1-Prioritized Pruning

論文特別強調 1x1 convolution，因為現代 CNN 中 1x1 conv 佔了大量計算，而且 unstructured pruning 在 1x1 conv 上容易造成 PE load imbalance。

因此可以設計針對 1x1 conv 的剪枝策略：

```text
1x1 conv：使用較積極的 cross-filter pruning
3x3 conv：使用較保守的 pruning
```

例如：

```text
1x1 conv prune ratio = 70%
3x3 conv prune ratio = 40%
```

這個方向比較適合 1x1 conv 較多的模型，例如：

```text
MobileNetV1
ResNet50
GoogLeNet
DenseNet121
```

可以探討的研究問題：

```text
優先剪枝 1x1 convolution 是否能在維持 accuracy 的同時，
提升硬體友善程度與推論效率？
```

### 5. Hybrid Pruning Strategy

Global unstructured pruning 通常 accuracy 較好，但 sparsity pattern 不規則，容易造成 PE workload imbalance。

Cross-filter structured pruning 的 PE balance 較好，但 accuracy 可能略低。

因此可以設計 hybrid pruning：

```text
1x1 layers：使用 cross-filter structured pruning
3x3 layers：使用 global unstructured pruning
```

或：

```text
前面 layers：使用較保守的 pruning
後面 layers：使用 cross-filter structured pruning
```

可以探討的研究問題：

```text
Hybrid pruning 是否能在 accuracy 與 PE utilization 之間取得更好的 trade-off？
```

### 6. Gradual Cross-Filter Pruning

目前常見方式是一次剪到目標 sparsity，例如直接剪 60%。

可以改成逐步剪枝：

```text
20% pruning -> fine-tune
40% pruning -> fine-tune
60% pruning -> fine-tune
```

預期優點：

```text
模型有更多時間適應剪枝造成的結構變化，
通常可以得到比 one-shot pruning 更好的 accuracy。
```

可以探討的研究問題：

```text
在相同最終 sparsity 下，
gradual cross-filter pruning 是否能比 one-shot cross-filter pruning 保留更高 accuracy？
```

### 7. Fine-Tuning Policy 改良

剪枝後 accuracy 能不能恢復，很大程度取決於 fine-tuning 設定。

可以改良的方向：

```text
調整 learning rate
增加 fine-tuning epochs
加入 cosine learning rate schedule
加入 multi-step learning rate decay
調整 optimizer
調整 weight decay
加入 knowledge distillation
```

其中 knowledge distillation 很適合做為改良：

```text
Dense model 作為 teacher。
Pruned model 作為 student。
Student 同時學習 label 與 teacher logits。
```

可以探討的研究問題：

```text
Knowledge distillation 是否能讓 cross-filter structured pruned model 恢復更多 accuracy？
```

### 8. 更完整的軟體硬體模擬

目前程式已經估算：

```text
MAC reduction
PE workload utilization
```

可以進一步加入：

```text
estimated cycle count
estimated speedup vs dense
estimated speedup vs global unstructured pruning
load imbalance ratio
1x1-only statistics
3x3-only statistics
per-layer PE utilization
```

這樣可以讓軟體模擬更接近論文的硬體加速器分析，但不需要真的寫 Verilog 或 FPGA。

## 最推薦的創新方向

目前最實際、最容易完成、也最有研究味道的方向是：

```text
Activation-Aware Cross-Filter Structured Pruning
```

建議公式：

```text
G[c, r, s] = sum_m |W[m, c, r, s]| * A[c]
```

其中：

```text
A[c] = mean(|input activation channel c|)
```

推薦原因：

- 保留 cross-filter structured mask。
- 保留 PE workload balance。
- 相比原論文，只看 weight 的方法多加入 activation 資訊。
- 容易用 Python 實作。
- 特別適合 1x1 convolution。
- 可以直接和原論文的 L1 方法比較。

## 建議實驗設計

### 模型

先從：

```text
ResNet34 on CIFAR-10
```

再擴展到更貼近論文重點的模型：

```text
MobileNetV1 on CIFAR-10
Animals-10 if available
ImageNette if available
```

### 比較方法

建議比較：

```text
Dense baseline
Dense extra-training baseline
Global unstructured pruning
Cross-filter L1 pruning
Cross-filter max_abs pruning
Activation-aware cross-filter pruning
Random cross-filter pruning
```

### 評估指標

建議記錄：

```text
Top-1 accuracy
Accuracy drop
Weight pruning ratio
MAC reduction
PE workload utilization
1x1-only MAC reduction
1x1-only PE utilization
3x3-only MAC reduction
3x3-only PE utilization
```

## 預期主張

如果 activation-aware 方法有效，可以這樣主張：

```text
本文提出一種 activation-aware cross-filter structured pruning 方法。
相較於原始 L1-based cross-filter pruning，本方法在維持相同 structured sparsity pattern
與 PE workload balance 的前提下，利用 activation statistics 改善剪枝後 accuracy。
```

## 可能論文題目

英文：

```text
Activation-Aware Cross-Filter Structured Pruning for Efficient CNN Inference
```

中文：

```text
基於啟動值感知之跨濾波器結構化剪枝方法於 CNN 推論加速之研究
```

