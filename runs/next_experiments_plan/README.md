# 下一批實驗準備：Pruning Ratio Sweep 與其他資料集

這個資料夾是下一批實驗的執行計畫說明。實際執行腳本放在專案根目錄的 `scripts/`。

## 1. 不同 pruning ratio：CIFAR-10

建議先跑：

```text
0.5, 0.7, 0.8
```

暫時不把 `efficientnet_v2_m` 放進預設 sweep，原因是前一批 5 epochs 下 accuracy 很低且執行時間非常長。

直接執行：

```bash
bash scripts/run_ratio_sweep_cifar10.sh
```

預設會跑：

- GPU 1：`resnet34,googlenet,mobilenet_v2`
- GPU 0：`resnet50,densenet121`
- Dataset：CIFAR-10
- Epochs：5
- Fine-tune epochs：2
- Batch size：128
- Ratio：0.5、0.7、0.8

輸出位置：

```text
runs/ratio_sweep_cifar10/
```

自訂範例：

```bash
RATIOS="0.5 0.6 0.7 0.8 0.9" SEED=1 bash scripts/run_ratio_sweep_cifar10.sh
```

## 2. Animals-10 / Imagenette / 自訂 ImageFolder

目前專案資料夾裡尚未看到 Animals-10 或 ImageFolder 資料。準備好資料後可以跑：

```bash
DATASET=animals10 DATA_DIR=/path/to/animals10 bash scripts/run_imagefolder_dataset_sweep.sh
```

或：

```bash
DATASET=imagenette DATA_DIR=/path/to/imagenette2-160 bash scripts/run_imagefolder_dataset_sweep.sh
```

通用 ImageFolder：

```bash
DATASET=imagefolder DATA_DIR=/path/to/custom_dataset bash scripts/run_imagefolder_dataset_sweep.sh
```

支援資料格式：

```text
root/train/class_x/*.jpg
root/val/class_x/*.jpg
```

或：

```text
root/train/class_x/*.jpg
root/test/class_x/*.jpg
```

如果沒有 `train/val` 或 `train/test`，程式會自動做 80/20 split。

## 3. 結果彙總

跑完後可以用：

```bash
python scripts/aggregate_pruning_results.py runs/ratio_sweep_cifar10
```

或：

```bash
python scripts/aggregate_pruning_results.py runs/animals10_sweep
```

這會產生：

```text
all_results_combined.csv
```

並在 terminal 顯示每個 model/file 的 `max_abs - l1_sum` 差值。
