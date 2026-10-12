# Multi-Model CIFAR-10 Split Run
這個資料夾存放 CIFAR-10 多模型剪枝實驗結果。每個模型獨立跑一個 process，部分模型在 GPU 1，較重模型在 GPU 0。
## 實驗設定
- Dataset：CIFAR-10。
- Models：resnet34、resnet50、googlenet、densenet121、mobilenet_v2、efficientnet_v2_m。
- Dense training：5 epochs。
- Fine-tuning / dense extra training：2 epochs。
- Batch size：128。
- Pruning ratio：60%。
- Cross-filter criteria：`l1_sum,max_abs`。
- Cross-filter ratio source：`global_cluster`。
- PE cluster size：`tm = 8`。
- MACs per PE：8。

## 檔案
- `gpu1_resnet34_results.csv`
- `gpu0_resnet50_results.csv`
- `gpu1_googlenet_results.csv`
- `gpu1_densenet121_results.csv`
- `gpu1_mobilenet_v2_results.csv`
- `gpu0_efficientnet_v2_m_results.csv`
- `all_models_combined_results.csv`：上述六個模型合併後的總表。

## Accuracy / Speedup 摘要
| Model | Dense extra FT | Global FT | l1_sum FT | max_abs FT | max_abs - l1_sum | Cross speedup |
|---|---:|---:|---:|---:|---:|---:|
| resnet34 | 83.23% | 84.09% | 83.04% | 83.07% | +0.03 pp | 1.81x |
| resnet50 | 71.09% | 69.88% | 69.95% | 70.80% | +0.85 pp | 1.72x |
| googlenet | 83.89% | 84.36% | 84.58% | 84.97% | +0.39 pp | 1.78x |
| densenet121 | 87.78% | 88.14% | 84.30% | 85.32% | +1.02 pp | 2.16x |
| mobilenet_v2 | 49.36% | 48.93% | 48.39% | 48.76% | +0.37 pp | 2.25x |
| efficientnet_v2_m | 21.34% | 21.84% | 23.08% | 22.65% | -0.43 pp | 2.24x |

## 初步觀察
- 六個模型中，`max_abs - l1_sum` 的平均差值是 `+0.37 percentage points`。
- Cross-filter 方法相對 global unstructured 的主要優勢仍然是 paper-style estimated speedup 和 cycle-based PE utilization。
- `efficientnet_v2_m` 在 5 epochs 下 accuracy 很低，代表這個模型目前訓練不充分；它的 accuracy 結果不適合和已較快收斂的模型直接下強結論。
- 後續若要正式比較 EfficientNet，建議單獨跑、增加 epochs、開 AMP，或先調整 learning rate / image size。
