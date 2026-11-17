# r90 checkpoint（給硬體端）

CIFAR-100、ResNet34、gradual+floor、剪枝率 0.9、seeds 0/1/2 的 fine-tuned 權重。
它們就是 branch `research-2026-11`、commit `59df000` 底下
`runs/hw_ranges_2026-11-11/q_per_cluster_r90/` 三份 JSON 所對應的模型（每個 cluster 的 q 已逐一比對相同）。

這個分支只放權重，和研究分支沒有共同歷史。

## 檔案

每個 `seed{0,1,2}/` 底下：

| 檔案 | 內容 |
|---|---|
| `final_model.pt.gz` | 權重。gzip 壓縮的 `torch.save(state_dict)`，解壓後 85 MB |
| `masks.pt` | 遮罩，另外存一份 |
| `q_per_cluster.json` | 和 `q_per_cluster_r90/cifar100_resnet34_seed{s}.json` 相同 |
| `result.json` | 訓練結束時的紀錄 |

## 確認載入正確

```
python3 load_checkpoint.py seed0 --data <放 CIFAR-100 的目錄> --q-json seed0/q_per_cluster.json
```

`load_checkpoint.py` 只需要 torch 和 torchvision，會檢查遮罩外的權重全為 0、q 和 JSON 相同，再量 test top-1。應該得到：

| seed | FP32 top-1（test 10,000 張） | 解壓後 `final_model.pt` 的 sha256 |
|---|---|---|
| 0 | 76.02%（7602） | `5b69d3d49ec51adb013cb1d4e70a95d3fb09380f7b1618f2be59925c74d6d427` |
| 1 | 76.27%（7627） | `1812843ec50626a82d7120dfcc45cc521935529f2c0528f149e65929cf3c149a` |
| 2 | 75.64%（7564） | `a5b6ed7b53ab27f7ef16232738cefe7b2fc85033073bc5f524306788740f3835` |

`result.json` 的 `final_test` 是用 AMP float16 量的，seed 2 是 75.65%（7565），和 FP32 差一張。
量測環境：torch 2.6.0、torchvision 0.21.0。

## 載入方式

- **模型定義：** torchvision 的 `resnet34(num_classes=100)`，再改兩處：
  - `conv1` 換成 `nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)`；
  - `maxpool` 換成 `nn.Identity()`。
  - 研究分支裡是 `resnet34_pruning_experiments.py` 的 `make_model("resnet34", 100, 32)`；`load_checkpoint.py` 的 `make_model()` 是同一個模型的獨立寫法。
- **checkpoint 的 key：** 就是 `state_dict` 本身，沒有外層包裝，共 218 個 key，名稱和 torchvision 相同（`conv1.weight`、`bn1.*`、`layer1.0.conv1.weight` … `fc.weight`、`fc.bias`）。直接 `model.load_state_dict(state, strict=True)`。
- **遮罩：** 已經乘進權重，被剪掉的位置在 `final_model.pt` 裡就是 0。`masks.pt` 是另外存的一份：
  - `{卷積層名稱: bool tensor，形狀 (cluster 數, K)}`，共 36 層、1,064 個 cluster；
  - cluster c 是第 8c 到 8c+7 個 filter，8 個 filter 共用同一列遮罩；
  - K = Cin × kh × kw，位置順序是 (Cin, kh, kw)，也就是 `weight[f].flatten()` 的順序；
  - 還原成權重形狀：`mask.repeat_interleave(8, 0).reshape_as(conv.weight)`。
- **輸入前處理：** `ToTensor()` 後用 mean (0.5071, 0.4867, 0.4408)、std (0.2675, 0.2565, 0.2761) 正規化，沒有其他處理。
- **校正用的影像：** 軟體端的 Q15 流程用訓練集切出的 5,000 張 validation，不用 test。切法是
  `torch.randperm(50000, generator=torch.Generator().manual_seed(0))` 的前 5,000 個 index（三個 seed 共用）。
