# Smoke Tests 小型測試

這個資料夾裡的 CSV 是快速 sanity check，不是正式實驗結果。

這些檔案通常使用很小的 CIFAR-10 subset，或使用 `epochs = 0` / `finetune_epochs = 0`，目的是確認程式流程可以正常跑完。

## 測試目的

這些 smoke tests 用來確認：

- 主程式可以 end-to-end 執行。
- CSV 會輸出預期欄位。
- `l1_sum` 和 `max_abs` 都能正常執行。
- `global_cluster` sparsity reference 可以正常被 cross-filter pruning 使用。
- paper-style cycle、cycle reduction、estimated speedup 欄位可以正常產生。

## 注意

請不要用這些 smoke test CSV 來做 accuracy 結論，因為它們不是完整訓練結果。

正式結果請看：

```text
../paper_like_bs128/
```
