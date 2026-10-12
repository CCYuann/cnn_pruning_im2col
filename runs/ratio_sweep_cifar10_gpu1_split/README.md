# CIFAR-10 Pruning Ratio Sweep GPU1 Split

這批實驗用同一張 GPU 1 跑 CIFAR-10 的 pruning ratio sweep，用來觀察 `max_abs` 和論文原始 `l1_sum` 在不同 sparsity 下的差異。

## 實驗設定

- Dataset：CIFAR-10
- Models：resnet34、resnet50、googlenet、densenet121、mobilenet_v2
- Pruning ratios：0.5、0.6、0.7、0.8
- Seed：0
- Dense training：5 epochs
- Fine-tuning / dense extra training：2 epochs
- Batch size：128
- Cross-filter criteria：`l1_sum,max_abs`
- Cross-filter ratio source：`global_cluster`
- PE cluster size：`tm = 8`
- MACs per PE：8

## 檔案結構

- `r0p5_seed0/`：50% pruning ratio 結果
- `r0p6_seed0/`：60% pruning ratio 結果
- `r0p7_seed0/`：70% pruning ratio 結果
- `r0p8_seed0/`：80% pruning ratio 結果
- `logs/`：每個模型/ratio 的訓練 log
- `all_results_combined.csv`：全部結果合併表
- `aggregate_summary.txt`：聚合後的簡要統計

## 聚合結果

這批一共完成 20 組 model/ratio 組合。

整體平均：

```text
max_abs - l1_sum = +0.52 percentage points
```

代表在這批 pruning ratio sweep 中，`max_abs` 平均 fine-tuned accuracy 高於 `l1_sum`，同時保留相同 cross-filter structured mask 與相同 estimated speedup。

## 各 ratio 的觀察方向

這批資料可以用來回答：

- `max_abs` 是否只在 60% sparsity 有效？
- sparsity 提高到 70% / 80% 時，`max_abs` 是否更穩？
- Cross-filter structured pruning 的 estimated speedup 是否隨 pruning ratio 上升？

目前從 aggregate summary 看，`max_abs` 在多數 model/ratio 組合中優於 `l1_sum`，但不是每一組都贏。這很適合寫成：

```text
max_abs 在多模型、多 sparsity ratio 設定下平均優於 l1_sum，但仍存在模型與 sparsity 依賴性。
```

## 注意

這批曾經把 GPU1 塞到約 10 個 concurrent training jobs。雖然最後成功完成，但 wall time 很長。之後建議使用 `MAX_PARALLEL=5` 或 `MAX_PARALLEL=6`，通常會比過度併發更穩。
