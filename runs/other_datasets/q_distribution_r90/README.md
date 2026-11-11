# 90% 目標剪枝率的 q 分布

資料集：CIFAR-100；seeds 0/1/2。直接讀取各模型已完成結果目錄中的 masks.pt，未重新訓練或讀取 test。

每 8 個輸出 filters 為一個 cluster，q 為共用 mask 保留的位置數，K = Cin × Kh × Kw。每個 filter 保留 q 個卷積權重位置，每個 cluster 保留 8q 個純量權重；不計 bias。

下表按 cluster 等權統計，將三個 seeds 的所有 cluster 合併；範圍也涵蓋三個 seeds。不同層的 K 不同，不能僅以 q 大小判斷保留比例。

| 模型 | 方法 | 每 seed cluster 數 | 平均 q | 中位數 q | 最小 q | 最大 q | q < 15 比例 | 實際剪枝率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| resnet34 | rewind_r90 | 1064 | 249.77 | 160.0 | 0 | 838 | 6.14% | 89.9997% |
| resnet34 | gradual_r90_floor | 1064 | 249.86 | 167.0 | 1 | 935 | 7.52% | 89.9964% |
| resnet50 | rewind_r90 | 3320 | 88.28 | 36.0 | 0 | 1331 | 19.25% | 90.0002% |
| resnet50 | gradual_r90_floor | 3320 | 88.35 | 25.0 | 1 | 1330 | 38.73% | 89.9918% |
| resnet56 | rewind_r90 | 266 | 39.98 | 34.0 | 4 | 114 | 11.40% | 90.0011% |
| resnet56 | gradual_r90_floor | 266 | 40.09 | 37.0 | 1 | 110 | 22.81% | 89.9725% |
| vgg16_bn | rewind_r90 | 528 | 348.26 | 196.5 | 2 | 1076 | 8.21% | 90.0001% |
| vgg16_bn | gradual_r90_floor | 528 | 348.34 | 256.0 | 1 | 911 | 6.25% | 89.9976% |
| googlenet | rewind_r90 | 910 | 76.62 | 77.0 | 0 | 376 | 8.13% | 89.9999% |
| googlenet | gradual_r90_floor | 910 | 76.72 | 61.5 | 1 | 310 | 15.53% | 89.9868% |
| densenet121 | rewind_r90 | 1280 | 67.02 | 48.0 | 1 | 289 | 17.40% | 90.0001% |
| densenet121 | gradual_r90_floor | 1280 | 67.11 | 32.0 | 1 | 304 | 31.09% | 89.9857% |

90% 是全模型目標，不是每層或每個 cluster 都精確保留 10%。實際剪枝率的微小差異來自離散配額與逐步更新，不應假定兩方法的保留量完全一致。

V3 週期公式使用 max(q,15)；q < 15 的 cluster 已低於此計算項的下限。全模型平均 q 不能直接換算推論週期，還需各層輸出尺寸與逐 cluster q。

明細：summary.csv 為模型彙總；per_layer.csv 為各 seed／各層分布；per_cluster.csv 為各 seed／各層／各 cluster 的實際整數 q。

來源檔案格式：../cifar100_<model>_seed<seed>_<arm>/masks.pt；幾何資訊：../geometry_cifar100_<model>.json。
