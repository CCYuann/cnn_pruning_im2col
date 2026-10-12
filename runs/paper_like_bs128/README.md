# Paper-Like BS128 正式實驗

這個資料夾存放主要實驗結果，用來比較兩種 cross-filter structured pruning criterion：

1. `Cross-filter l1_sum`：論文原始方法，importance = sum(abs(W)) across filters in a cluster。
2. `Cross-filter max_abs`：本實驗提出的改良方法，importance = max(abs(W)) across filters in a cluster。

## 共同實驗設定

- Model：ResNet34，使用 CIFAR-style stem。
- Dataset：CIFAR-10。
- Dense training：5 epochs。
- Fine-tuning / dense extra training：2 epochs。
- Batch size：128。
- Pruning ratio：60%。
- Cross-filter ratio source：`global_cluster`。
- PE cluster size：`tm = 8`。
- Paper-style cycle estimate 的 MACs per PE：`8`。
- cuDNN benchmark：disabled。

## 檔案說明

- `l1_maxabs_paper_like_bs128_results.csv`：第一次主要實驗，還沒有額外 seed sweep 前的結果。
- `l1_maxabs_paper_like_bs128_seed1_results.csv`：seed 1 的結果。
- `l1_maxabs_paper_like_bs128_seed2_results.csv`：seed 2 的結果。
- `l1_maxabs_paper_like_bs128_seed3_results.csv`：seed 3 的結果。

## Seeds 1-3 的主要結果

三個 seed 的平均 fine-tuned accuracy：

- Dense extra training：84.63%。
- Global unstructured：85.50%。
- Cross-filter `l1_sum`：84.46%。
- Cross-filter `max_abs`：84.80%。

`max_abs` 相對 `l1_sum` 的平均提升：

```text
+0.35 percentage points
```

Speedup 估算：

- Global unstructured：約 1.01x。
- Cross-filter structured pruning：約 1.80x。

## 初步解讀

在這組 CIFAR-10 / ResNet34 / 60% sparsity 設定下，`max_abs` 平均 accuracy 高於論文原始的 `l1_sum`，而且不改變 cross-filter structured mask，因此保留相同的硬體友善 pattern 和相同的 estimated speedup。

簡單說，這批結果支持下面這個觀察：

```text
max_abs 是 cross-filter pruning criterion 的一個可行改良方向。
它在保留硬體友善結構的同時，平均 accuracy 比 l1_sum 更好。
```

不過目前 seed 數仍然不多，所以比較嚴謹的說法應該是「promising」而不是已經完全定論。
