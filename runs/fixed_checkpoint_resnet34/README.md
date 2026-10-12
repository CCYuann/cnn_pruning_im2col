# ResNet34 固定 Dense checkpoint

這個資料夾保存後續正式剪枝比較共用的 Dense 基準。三個 seed 使用相同 CIFAR-10 train/validation split 與相同訓練設定，避免不同 pruning job 各自重訓 Dense，造成基準不一致。

## 設定

- 模型與資料集：ResNet34 / CIFAR-10
- seed：`0, 1, 2`
- split seed：`0`
- train / validation / test：`45000 / 5000 / 10000`
- split ID：`42fbb7440cc27c71`
- 訓練：100 epochs、batch size 128、SGD、初始 LR 0.1
- scheduler：cosine，minimum LR 0.0001
- weight decay：0.0005
- AMP：啟用
- deterministic CUDA：啟用

## Dense 結果

| Seed | Validation accuracy | Test accuracy | Checkpoint |
|---:|---:|---:|---|
| 0 | 95.10% | 94.81% | `checkpoints/resnet34_seed0_dense.pt` |
| 1 | 95.32% | 94.88% | `checkpoints/resnet34_seed1_dense.pt` |
| 2 | 94.88% | 94.58% | `checkpoints/resnet34_seed2_dense.pt` |
| Mean | 95.10% | 94.76% | - |

Test accuracy 的三個 seed sample standard deviation 約為 0.16 percentage point。

## 檔案

- `checkpoints/`：三個固定 Dense checkpoint，後續 pruning job 應以 `--dense-checkpoint-in` 載入。
- `dense_seed*_results.csv`：各 seed 原始結果。
- `dense_all_seeds_combined.csv`：三個 seed 的合併結果。

## 正式剪枝比較

`pruning_comparison/` 已完成 `3 seeds × 3 ratios × 2 Tm = 18` 組正式實驗，所有方法都從上述固定 checkpoint 分支。結果、圖表與限制說明見：

```text
pruning_comparison/README.md
```

`sensitivity_validation/` 保存每個 seed / Tm 使用 validation set 重建的敏感層結果；舊的 test-based sensitivity ranking 沒有用在這批正式比較。
