# Cross-Filter Structured Pruning 研究改良方向

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

