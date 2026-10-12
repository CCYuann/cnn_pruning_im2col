# CNN Cross-Filter Pruning 與 INT8 模擬

本專案研究 Cross-Filter Structured Pruning，並分析剪枝後 CNN 在
mask-aware accelerator 上的準確率、MAC reduction、PE utilization 與 INT8 量化行為。

目前主要實驗包含：

- CIFAR-10 與 ImageFolder/Animals-10 資料集。
- ResNet34、ResNet50、GoogLeNet、DenseNet121、MobileNetV2 等模型。
- Global unstructured sparsity reference。
- Cross-filter `l1_sum`、`max_abs`、`top2_mean_abs`、`top4_mean_abs`。
- Sensitivity-aware layer protection 與 pruning ratio cap。
- 多 seed、不同 pruning ratio 與 filter grouping 實驗。
- Layer sensitivity 分析及圖表。
- 剪枝模型的 INT8/Q15 軟體模擬與硬體匯出。

## 主要程式

- `resnet34_pruning_experiments.py`：訓練、剪枝、fine-tuning 與主要指標輸出。
- `pruned_int8_simulation.py`：固定 checkpoint 的剪枝與 INT8 模擬流程。
- `quantization_experiments.py`：量化相關實驗。
- `scripts/layer_sensitivity.py`：逐層敏感度測試。
- `scripts/`：批次執行、結果彙整與繪圖工具。
- `runs/README.md`：現有實驗資料夾索引。
- `research_improvement_ideas.md`：研究方向與改良構想。

## 安裝

建議使用具備 CUDA 的 PyTorch 環境：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch/CUDA 的實際版本應依執行機器與 NVIDIA driver 選擇。

## 基本執行範例

```bash
python resnet34_pruning_experiments.py \
  --dataset cifar10 \
  --device cuda \
  --epochs 5 \
  --finetune-epochs 2 \
  --batch-size 128 \
  --num-workers 0 \
  --prune-ratio 0.6 \
  --cross-criteria l1_sum,max_abs \
  --cross-ratio-source global_cluster \
  --no-cudnn-benchmark \
  --save-dir runs \
  --run-name example
```

各批次實驗指令位於 `scripts/*.sh`，實驗結果與中文說明位於 `runs/`。

## Repository 資料政策

Git repository 保留程式碼、CSV、JSON、Markdown 與結果圖表。以下大型或可重建資料不納入 Git：

- CIFAR-10、MNIST、Animals-10 等 datasets。
- `.pt`、`.pth`、`.ckpt` model checkpoints。
- 壓縮檔、cache 與 runtime logs。

因此 clone 後需要重新下載資料集，或自行放入 `data/`。需要重現固定 checkpoint
實驗時，也必須另外取得對應 checkpoint。

