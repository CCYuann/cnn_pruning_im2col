# 固定 Dense checkpoint 的多模型 70% 剪枝比較

這組實驗用來確認：在相同且充分訓練的 Dense 起點下，cross-filter pruning 後的 accuracy loss 是否能控制在 1 個百分點以內。

## 實驗流程

1. CIFAR-10，每個模型以 seed 0、1、2 分別訓練 100 epochs，儲存固定 Dense checkpoint。
2. Dense 與所有剪枝方法都從同一個 seed 的 checkpoint 開始。
3. 以固定 5,000 張 validation set 執行單層 sensitivity 測試；測試資料不參與模型更新。
4. 目標 pruning ratio 為 `0.7`，測試 `Tm=8` 與 `Tm=16`。
5. 剪枝後 fine-tune 10 epochs，`lr=0.001`。
6. accuracy loss 以同 checkpoint 再訓練 10 epochs 的 `Dense extra training` 為基準。

比較方法：

- `Cross l1_sum`：原始 cross-filter criterion，未保護敏感層。
- `Cross top2 + sensitivity`：`top2_mean_abs`，設定 `top_k=4`、`skip_top_k=1`、敏感層 cap `0.2`、全層 cap `0.75`。
- `Global unstructured`：作為 sparsity reference 與額外對照，不是可直接利用的規則 cross-filter 結構。

## 主要結果

下表為三個 seed 的平均 accuracy loss，單位是 percentage point（pp）；正值代表比 Dense extra 差。

| 模型 | l1 Tm8 | l1 Tm16 | top2+sens Tm8 | top2+sens Tm16 | 是否全部低於 1 pp |
|---|---:|---:|---:|---:|:---:|
| ResNet34 | 0.260 | 0.397 | 0.127 | 0.210 | 是 |
| ResNet50 | 0.207 | 0.417 | 0.263 | 0.257 | 是 |
| GoogLeNet | 0.293 | 0.430 | 0.253 | 0.333 | 是 |
| DenseNet121 | 0.417 | 0.447 | 0.220 | 0.317 | 是 |
| MobileNetV2 | 1.977 | 3.017 | 1.387 | 1.987 | **否** |

結論是 **5 個模型中有 4 個符合低於 1 pp，MobileNetV2 不符合**。因此目前不能宣稱此流程對所有 CNN 都能把誤差控制在 1 pp 內。

MobileNetV2 的 depthwise convolution 每個 filter 只有很少的 channel 自由度，共用 cross-filter mask 時更容易刪除不可替代的連接；`Tm=16` 的群組也比 `Tm=8` 更難保留每個 filter 的重要位置。這與它明顯較大的 accuracy loss 一致，但仍屬於由實驗與架構做出的解釋，後續需用 depthwise 專用規則驗證。

## 公平性提醒

- `l1_sum` 的實際 sparsity 約為 70%。
- sensitivity-aware top2 的實際 sparsity只有約 59% 至 64%，因為敏感層與全層 cap 會把部分權重保留下來。
- 所以 top2 accuracy 較高不能單獨解讀成 criterion 一定較好；它同時用了較低的實際剪枝量。
- 正式比較時應同時報告 accuracy、實際 sparsity、MAC reduction 與 estimated speedup。

## 圖表與 CSV

- `summary/accuracy_loss_by_model_tm.png`：不同模型、Tm 與方法的平均 accuracy loss；誤差棒為三 seed sample standard deviation。
- `summary/accuracy_sparsity_tradeoff.png`：實際 sparsity 與 accuracy loss 的關係。
- `summary/cross_filter_under_1pp_check.csv`：cross-filter 的完整三 seed 統計與 `< 1 pp` 判定。
- `summary/all_methods_r0p7_summary.csv`：包含 global unstructured 的所有方法摘要。
- `summary/dense_summary.csv`：各模型 Dense 與 Dense extra 統計。
- `all_results_combined.csv`：四個新模型的逐次原始結果；ResNet34 原始結果位於 `fixed_checkpoint_resnet34/pruning_comparison/`。

## 尚未完成

EfficientNetV2-M 已完成三個 100-epoch Dense checkpoint，但尚未加入本表。它有約 245 個 convolution layers；目前逐層、三 ratio、三 seed、兩個 Tm 的完整 sensitivity 流程成本很高，應先縮減候選層或做分階段 sensitivity screening，再執行正式剪枝。
