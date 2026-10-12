# Filter 相似度分組實驗

固定 CIFAR-10 Dense checkpoints，`Tm=8`、Cross-filter `l1_sum`、fine-tune 10 epochs。
相似分組使用 global unstructured keep masks 的 K-position Jaccard similarity。
模型 channel 順序沒有改動；filter group 僅決定哪些 filters 共用 cross-filter mask。

完成進度：`45/45` 組配對結果（完成）。

## 平均結果

| 模型 | Ratio | 連續分組損失 (pp) | 相似分組損失 (pp) | FT 提升 (pp) |
|---|---:|---:|---:|---:|
| densenet121 | 80% | 0.740 | 0.593 | +0.147 |
| densenet121 | 85% | 0.873 | 0.950 | -0.077 |
| densenet121 | 90% | 1.630 | 1.470 | +0.160 |
| googlenet | 80% | 0.707 | 0.607 | +0.100 |
| googlenet | 85% | 0.887 | 0.820 | +0.067 |
| googlenet | 90% | 1.637 | 1.287 | +0.350 |
| mobilenet_v2 | 80% | 2.727 | 0.213 | +2.513 |
| mobilenet_v2 | 85% | 2.980 | 0.427 | +2.553 |
| mobilenet_v2 | 90% | 3.363 | 0.580 | +2.783 |
| resnet34 | 80% | 0.350 | 0.290 | +0.060 |
| resnet34 | 85% | 0.607 | 0.417 | +0.190 |
| resnet34 | 90% | 0.783 | 0.720 | +0.063 |
| resnet50 | 80% | 0.453 | 0.350 | +0.103 |
| resnet50 | 85% | 0.780 | 0.573 | +0.207 |
| resnet50 | 90% | 1.330 | 0.977 | +0.353 |

正值表示相似分組的 fine-tuned accuracy 較高。比較使用相同 model、seed、
target ratio、Tm 與 Dense checkpoint 配對。
