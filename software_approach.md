# Mask-Aware CNN Accelerator 軟體做法

更新日期：2026-11-05。範圍：剪枝方法、mask 格式、稀疏度、軟體資料流、輸出給硬體的介面。

本文件整理目前的討論結論，並已對照程式、`masks.pt`、`result.json` 和實際匯出的 package。三個 seeds 的 package 都通過 `validate_pruned_layer_package.py`；RTL 部分只讀遠端 `main`（`de02bd4`）的程式與文件，沒有重跑模擬。

> **資料狀態：** `retrain_recovery_r0p95` 的 29 條實驗已全部完成，第 1.5 節為最終結果。ResNet50 的結果都只有 validation，沒有 test 或 INT8 驗證。

## 0. 目前採用的設定

| 項目 | 交付硬體的 package | 精度研究線 |
|---|---|---|
| 模型 / 資料集 | ResNet34 / CIFAR-10 | ResNet50 / CIFAR-10 |
| 目標稀疏度 | 80% | 90%、95% |
| 分組 | contiguous，`Tm=8` | contiguous，`Tm=8` |
| 重要性 | `l1_sum` | `l1_sum` |
| 配額 | `global_cluster` | `global_cluster` |
| 訓練 | FT10，固定 lr 0.001 | 比較 FT、LR rewinding 等方法 |
| 數值 | INT8 weight/activation，INT32 bias，Q15 multiplier | 尚無補償或重訓後模型的 INT8 驗證 |
| Dense seeds | 0、1、2 | 0、1、2 |

幾個已定案的判斷：

- **分組用 contiguous：** 不需要 output remap，zero-kept clusters 也從平均 94.3 降到 7.0。代價是三 seed 端到端 Q15 精度從 94.39% 降到 94.30%。
- **B ridge 補償改為低成本附加步驟：** 95% 判定未通過（B−A 在 FT10 為 +0.08pp，門檻 +0.8pp）。總檔第 0、16 節仍把它列為主線，內容已過時。
- **分組搜尋、activation 加權、mask 修補都停止投入：** 代理指標改善，但分類精度沒有勝過控制組。
- **訓練流程是目前最明顯的施力點：** 高 LR 重訓明顯勝過舊微調，`group` 懲罰是目前唯一通過方法增益門檻的做法，見第 1.5 節。但兩者都還沒進硬體 package。

## 1. 剪枝方法

### 1.1 剪的單位

卷積權重形狀為 `[Cout, Cin, Kh, Kw]`，每個 filter 有 `K = Cin × Kh × Kw` 個位置。每 `Tm=8` 個 filters 組成一個 cluster，共用一份 K mask：同一位置一起保留或一起移除，保留下來的權重值仍各自不同。

- 3×3、`Cin=3` 時 `K=27`。
- 1×1、`Cin=64` 時 `K=64`。
- `q`（`kept_count`）是一個 cluster 保留的位置數。`q=0` 代表該 cluster 全部位置都被剪掉，不等於輸出為 0（見第 5.5 節）。

硬體因此可以只讀保留位置的 activation 和權重，被剪位置不進 PE。

### 1.2 Mask 產生流程

實作在 [resnet34_pruning_experiments.py](resnet34_pruning_experiments.py) 的 `global_unstructured_reference` 與 `cross_filter_prune_`：

1. **全域參考：** 把所有納入的卷積權重依絕對值排序，取最小的 `target_sparsity` 比例，得到非共享的參考 mask。
2. **換算配額：** 統計參考 mask 在每個 cluster 的零比例 `r`，得到要移除的位置數 `round(r × K)`。
3. **計算重要性：** cluster 內沿 filters 對每個 K 位置加總，`l1_sum = Σ|W|`。
4. **產生共享 mask：** 移除重要性最低的 `round(r × K)` 個位置。位置是從 Dense 權重重新選的，不是在參考 mask 上再剪。
5. **寫回：** `weight[group] = cluster × mask`，同時保存 `full_mask` 和各 cluster 的 mask。

配額由全域決定，所以各層、各 cluster 的 `q` 不同，這也是 scheduler 必須逐 cluster 讀 `kept_count` 的原因。

### 1.3 訓練時維持 mask

所有訓練分支都用相同規則，避免被剪位置長回來：

- `backward` 後把梯度乘上 mask，`optimizer.step()` 後再把權重乘上 mask。
- 結束時檢查所有禁止位置皆為 0。
- `kept` 位置的值允許剛好是 0，所以**不能從非零權重反推 mask**。

### 1.4 訓練與補償的選項

| 方法 | 改權重 | 改 mask / q | 硬體影響 | 狀態 |
|---|---|---:|---|---|
| FT（固定 lr 0.001） | 是 | 否 | 無 | 目前 package 使用 |
| LR rewinding（Dense cosine 的最後 60 epochs） | 是 | 否 | metadata 不變，權重與 scale 要重新匯出 | H1 已通過 |
| B Dense-input ridge | 是 | 否 | 同上 | 降為附加步驟，重訓後也未過 H2 |
| Gradual 剪枝 | 是 | **是** | 整份 package 重新匯出 | 未過 H2 |
| Group L2,1 懲罰 | 是 | **是** | 同上，且 `q=0` 較多 | **通過 H2、H4，未過 H3** |

B 的定義：在校正資料上固定 kept 集合 `S`，解下式，只調整保留權重。

```text
min_U ||U X_S - Y||_F^2 / N + lambda ||U - W_S||_F^2
lambda = max(1e-8, 0.001 * mean(diag(G)))
```

若 check patches 的重建誤差變差，整個 cluster 退回原權重；`q=0` 不會加回連接。

### 1.5 95% 救回實驗

ResNet50、5,000 張 validation、最後一個 epoch，總預算 60 epochs。判定標準已事先凍結在 [PLAN.md](runs/retrain_recovery_r0p95/PLAN.md)，屬於投入門檻，不是顯著性檢定。

| 組別 | seed0 | seed1 | seed2 | 平均 | 對 `rewind`（pp） |
|---|---:|---:|---:|---:|---:|
| `ft_const`（舊微調） | 93.40 | 92.96 | 93.94 | 93.433 | −0.713 |
| **`rewind`（95%）** | 94.28 | 93.88 | 94.28 | **94.147** | 基準 |
| `rewind_B` | 94.10 | 94.26 | 94.66 | 94.340 | +0.193 |
| `rewind_kd` | 92.94 | 92.42 | 93.38 | 92.913 | −1.233 |
| `gradual` | 94.28 | 94.44 | 94.48 | 94.400 | +0.253 |
| **`group`（λ=3e-5）** | 94.76 | 94.38 | 94.58 | **94.573** | **+0.427** |
| `rewind_r90`（90%） | 94.96 | 94.86 | 95.12 | 94.980 | +0.833 |
| `unstructured`（95%） | 95.02 | 95.20 | 95.24 | 95.153 | +1.007 |
| `dense` | 94.96 | 95.04 | 95.44 | 95.147 | +1.000 |

判定結果：

| 假設 | 內容 | 結果 | 依據 |
|---|---|---|---|
| H1 | `rewind` 比舊微調好（≥+0.5pp，3/3） | **通過** | +0.713pp（+0.88 / +0.92 / +0.34） |
| H2 | 方法比 `rewind` 好（≥+0.3pp，3/3） | **僅 `group` 通過** | +0.427pp（+0.48 / +0.50 / +0.30） |
| H3 | 95% 追上 90%（≥ `rewind_r90`） | **未通過** | 最佳的 `group` 仍低 0.407pp，三個 seeds 都較低 |
| H4 | 共享差距縮小，且 H2 成立 | **`group` 通過** | 對非共享 mask 的差距從 1.007pp 降到 0.580pp，縮小約 42% |

其餘方法的 H2：`rewind_B` +0.193pp（2/3），`gradual` +0.253pp（2/3，seed0 為 0.00），`rewind_kd` −1.233pp（0/3），都未通過。

解讀：

- **訓練流程確實重要：** 舊 FT 的精度和方法間差距可能都要重新看。
- **共享限制是主要差距：** 95% 非共享 mask 已與 Dense 同水準（95.153% 對 95.147%），`rewind` 還差約 1pp。
- **`group` 是目前最有希望的方法：** 它直接讓同 cluster 的 filters 在訓練中收斂到共同位置，而不是事後彌補。
- **「權重減半、精度不變」仍不成立：** 95% 最佳結果比 90% 低 0.407pp。
- **`rewind_kd` 明顯變差：** 本批只測一組 T=4、alpha=0.9，不代表 KD 整類無效。

`group` 結果的限制：

- λ 是在 seed0 上挑的，不帶偏差的 seed1、seed2 平均為 +0.40pp，仍高於門檻。
- 選到的 λ=3e-5 是試驗範圍的最小值，真正最佳值可能更小。要再搜尋需另開計畫，不能算進這批結果。
- n=3、只有 validation，門檻不是顯著性檢定。
- 只在 ResNet50 / 95% 測過，ResNet34 / 80% 沒有數據。

### 1.6 其他已結案的方法

| 方法 | 結果 | 決策 |
|---|---|---|
| 換重要性準則（max、top2、top4） | 小幅且依模型而異，沒有統一贏家 | 保留 `l1_sum` |
| 敏感層保護 + 補剪 | 四個模型退步，DenseNet121 惡化 6.6pp | 不採用 |
| Jaccard 相似分組 | 舊正訊號混有配額變化；固定 q 後未勝過 random | 不採用 |
| Activation 加權 | 自身 proxy 下降，精度反而較差 | 停止 |
| 局部重建 / CE / 梯度修補 mask | 未勝過 random 或控制組 | 停止 |

## 2. Mask 格式

### 2.1 軟體端

| 檔案 / 欄位 | 形狀 | 型別 | 用途 |
|---|---|---|---|
| checkpoint `pruning_layers[layer].full_mask` | `[Cout,Cin,Kh,Kw]` | float32 0/1 | 訓練時套用，匯出時檢查 |
| checkpoint `pruning_layers[layer].cluster_masks` | list of `[Cin,Kh,Kw]` | float32 0/1 | 匯出來源 |
| checkpoint `pruning_layers[layer].filter_groups` | list of filter index lists | int | cluster 的成員 |
| `masks.pt`（研究線）`masks[layer]` | `[clusters, K]` | bool | ResNet50 實驗 |

Checkpoint 也記錄 `prune_ratio`、`tm`、`criterion`、`ratio_source`、`filter_grouping`、`finetune_epochs`。

K 的攤平順序固定為 `(channel, kernel_row, kernel_col)`：

```text
k = channel * Kh * Kw + kernel_row * Kw + kernel_col
```

`True` / `1` 一律代表**保留**。

### 2.2 硬體端

硬體不接收 bitmap，而是 kept 位置清單。每個 kept 位置對應一筆 metadata 和一筆 packed weight，兩者都按上述順序遞增排列，第 `q` 筆彼此對應。

```text
metadata（16-bit）
[15:4] input_channel   上限 4095
[3:2]  kernel_row
[1:0]  kernel_column

packed weight（64-bit）
{w7, w6, w5, w4, w3, w2, w1, w0}，w0 在 [7:0]，對應 PE column 0
```

範例：3×3、kept 位置 `channel=5, row=1, col=2`。

```text
metadata = (5 << 4) | (1 << 2) | 2 = 0x0056
```

1×1 卷積的 `kernel_row` 和 `kernel_column` 恆為 0。

### 2.3 規則

- 同一 cluster 內所有 filters 的 mask 必須完全相同。
- Cluster 的 `kept_count` 可能為 0，此時沒有 metadata 和 weight rows，但 descriptor 與 post-process 參數都還在。
- 量化後保留位置的值可能變成 0，這筆資料仍要保留。
- 被剪位置在 BN folding 和量化後必須仍為 0，匯出器會檢查。

## 3. 稀疏度

### 3.1 定義

```text
sparsity = 1 - kept_weights / total_conv_weights
MAC reduction = 1 - Σ(kept_weights × Hout × Wout) / Σ(total_weights × Hout × Wout)
```

- 分母是納入剪枝的卷積權重，不含 FC 和 BN。
- `kept_weights = kept_count × 有效 filters 數`。
- 要回報實際值，不能只寫目標值。

### 3.2 實測值

| 配置 | seed | Kept 權重 | 實際稀疏度 | MAC reduction | `q=0` clusters | `q≤4` 比例 | `q` 中位數 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet34 / 80% | 0 | 4,252,008 | 79.9999% | 58.07% | 4 | 3.2% | 294.5 |
| | 1 | 4,251,880 | 80.0005% | 66.03% | 9 | 5.0% | 517.5 |
| | 2 | 4,252,088 | 79.9996% | 66.89% | 8 | 3.5% | 431 |
| ResNet50 / 90% | 0 | 2,344,856 | 89.9994% | 82.01% | 43 | 9.9% | 44 |
| | 1 | 2,344,552 | 90.0007% | 79.28% | 26 | 7.0% | 39 |
| | 2 | 2,344,352 | 90.0016% | 78.39% | 209 | 18.3% | 43 |
| ResNet50 / 95% | 0 | 1,171,976 | 95.0016% | 89.62% | 247 | 20.9% | 22 |
| | 1 | 1,172,160 | 95.0009% | 87.14% | 172 | 21.9% | 20 |
| | 2 | 1,171,896 | 95.0020% | 87.41% | 510 | 29.1% | 20 |

卷積權重總數：ResNet34 為 21,259,968（1,064 clusters），ResNet50 為 23,447,232（3,320 clusters）。ResNet34 的 MAC reduction 由 package 的 `layer.json` 計算，ResNet50 取自 `result.json`。

### 3.3 ResNet34 / 80% 各 stage 分佈

來自三個 seeds 的 `layer.json`。層形狀為 stem 3×3 stride 1，輸出寬度依序 32、32、16、8、4。Vector 指 `q × Tile 數`，Tile 數為 `Hout × ceil(Wout/8)`。

| Stage | 輸出寬度 | 稀疏度（seed 0/1/2） | Dense MAC 占比 | 剪枝後 MAC 占比 | Dense vector 占比 | 剪枝後 vector 占比 |
|---|---:|---:|---:|---:|---:|---:|
| stem | 32 | 50.5 / 55.1 / 60.2% | 0.2% | 0.2% | 0.1% | 0.2% |
| layer1 | 32 | 46.9 / 64.4 / 63.9% | 19.5% | 20.4–24.7% | 16.5% | 19.5–24.1% |
| layer2 | 16 | 34.7 / 51.7 / 57.6% | 24.6% | 31.5–38.3% | 20.8% | 30.0–37.3% |
| layer3 | 8 | 62.0 / 64.4 / 62.9% | 37.6% | 34.1–42.1% | 31.9% | 33.2–40.2% |
| layer4 | 4 | **93.8 / 90.8 / 91.1%** | 18.1% | **2.7–4.9%** | 30.6% | **5.2–9.3%** |

全域配額把 layer4 剪得最多，這對硬體優化的排序有直接影響：

- **「寬度 < 8 塞兩列」的收益大幅縮小：** Dense 時約省 15.3% vectors，真實 kept 分佈下只省 2.6 / 4.7 / 4.7%。這是一階估計，未計 stride 2 和 writeback。
- **第三個 bank 在這裡幾乎沒有施力點：** `q<21` 的 clusters 占 10.4–11.3%，但只占 0.19–0.24% 的 vectors，而且幾乎都在 layer4。把每個 Tile 計成至少 8 個 writeback cycles 後，占比也只有 0.20–0.26%。

正式結論以硬體端的週期模型為準。

### 3.4 解讀

- **MAC 降幅低於權重稀疏度：** MAC 依輸出解析度加權，高解析度層的權重比例會直接影響結果。同樣約 80%，ResNet34 的 MAC reduction 就有 58% 到 67% 的差異。
- **MAC 不等於 cycle：** Writeback、stride 2 gather、partial Tile 都有額外成本。V1 已量到 `kept_count` 小時 writeback 會成為瓶頸。
- **加速預期要用 MAC，不能用權重稀疏度：** ResNet34 / 80% 的 MAC 比例上限是 2.39 / 2.94 / 3.02 倍，不是 5 倍。但 dense 在 layer4 本來就有空 lane，以 issued vectors 計算的比例是 2.74 / 3.31 / 3.40 倍。兩者都未計 stride 2、writeback 和載入成本。
- **`q=0` 隨稀疏度快速增加：** ResNet50 平均占比從 90% 的 2.8% 升到 95% 的 9.3%，zero-kept bypass 不能省。上表的 mask 沒有整層全部 `q=0`。
- **97% 以上先不考慮：** `q≤4` 平均約 32.8%（先前離線統計，本次未重算）。
- **換訓練方法會改變 `q` 分布：** 下表都是 ResNet50 / 95%，改變 mask 的方法要重算硬體成本。

| 方法 | `q=0`（seed 0/1/2） | `q≤4` 比例 | `q` 中位數 | 平均 MAC reduction |
|---|---:|---:|---:|---:|
| `rewind`（固定 mask） | 247 / 172 / 510 | 20.9–29.1% | 20–22 | 88.06% |
| `group`（λ=3e-5） | 353 / 285 / 616 | 24.3–30.4% | 17–20 | 88.44% |
| `gradual` | 309 / 403 / 574 | 37.0–40.6% | 8–10 | 90.91% |

- **`group` 的硬體代價不大：** `q` 分布接近固定 mask，`q=0` 多約 100 個。λ 增大時 `q=0` 會快速增加（seed0 在 3e-4 時為 1,003）。
- **`gradual` 的 MAC 較低，但小 `q` 比例高得多：** 這會放大 writeback 瓶頸，MAC 優勢不一定能轉成 cycle，且為事後觀察。seed2 的 `layer4.2.conv2` 整層 `q=0`，該層輸出只剩 bias。

## 4. 資料流

### 4.1 離線軟體流程

```text
Dense 訓練（100 epochs，cosine 0.1→1e-4，SGD m0.9，wd 5e-4，batch 128）
  │  固定 checkpoint，5,000 張 validation 從 train 切出
  ▼
全域參考 + cross-filter mask（第 1.2 節）
  ▼
維持 mask 訓練（FT10；研究線比較 rewinding）          → pruned checkpoint
  ▼
BN folding（Conv + BN 併成 weight 與 bias）
  ▼
校正（2,048 張）：記錄各層輸入/輸出與 logit 的最大絕對值
  ▼
量化：per-channel weight、INT32 bias、Q15 multiplier + shift
  ▼
匯出 package（kept 清單、packed weight、descriptor、參數、schedule）
  ▼
驗證：結構與 checksum → 整數模型精度 → RTL 逐位元比對
```

### 4.2 BN folding 與量化公式

實作在 [analyze_resnet34_quantization.py](../mask_aware_conv_accelerator/scripts/analyze_resnet34_quantization.py)。硬體沒有 BN 路徑，全部折進權重和 bias：

```text
factor        = bn.weight / sqrt(bn.running_var + bn.eps)
folded_weight = weight * factor                                  （per output channel）
folded_bias   = bn.bias + (conv_bias - bn.running_mean) * factor

input_scale   = 校正 max|input|  / 127                           （per layer）
weight_scale  = max|folded_weight[ch]| / 127                     （per output channel）
bias_int32    = round(folded_bias / (input_scale * weight_scale))
real_mult     = input_scale * weight_scale / output_scale
real_mult    ≈ multiplier * 2^shift / 2^15
```

全部 zero-point 為 0，數值範圍 `[-127, 127]`，不使用 `-128`。Rounding 一律 round-to-nearest、ties-away-from-zero，算術右移不能取代。

Contiguous seed0 的 `acc + bias` 上界約 2.61 億，INT32 餘裕約 8.24 倍。BN folding 驗證的最大 logit 誤差為 0.0024。

### 4.3 推論時的整數資料流

```text
ifmap INT8（CHW，bank = x mod 8）
  ▼  只依 metadata 讀 kept 位置；padding 位置補 0，不讀 SRAM
INT8 × INT8 → INT16 → INT32 accumulate（中途不飽和）
  ▼
accumulator + bias（INT32，溢位視為錯誤）
  ▼
× multiplier（INT48），round，>> right_shift
  ▼
一般層：ReLU → clamp → INT8
Residual：conv 與 residual 分別 rescale，寬位元相加 → ReLU → clamp → INT8
  ▼
下一層 ifmap
```

Residual branch **不可先各自飽和成 INT8 再相加**。

結尾的 global average pooling 除以 16，再經 INT8 classifier 輸出 INT8 logits，以 `argmax` 分類。

### 4.4 目前精度

ResNet34、80%、contiguous、Q15、10,000 張 test。這份 test 歷史上用過多次，也不能與第 1.5 節的 validation 直接比較。

| seed | Dense FP32 | Pruned FP32 | Q15 端到端 |
|---:|---:|---:|---:|
| 0 | 94.81% | 94.50% | 94.44% |
| 1 | 94.88% | 94.56% | 94.41% |
| 2 | 94.58% | 94.26% | 94.05% |
| 平均 | 94.757% | 94.443% | 94.300% |

## 5. 輸出給硬體的介面

### 5.1 Package 結構

交付位置：`mask_aware_conv_accelerator/exports/resnet34_seed{0,1,2}_r0p8_l1_contiguous_q15/`

```text
manifest.json             模型與總量、相容性旗標
network_schedule.json     stem、residual blocks、identity requant、head
NN_<layer_name>/          36 個卷積層，每層一個目錄
  layer.json
  metadata.mem
  packed_weights.mem
  cluster_descriptors.mem
  filter_indices.mem
  bias_int32.mem
  multiplier_q15.mem
  shift_q15.mem
head/
  head.json、weights_int8.mem、bias_int32.mem、multiplier_q15.mem、shift_q15.mem
```

### 5.2 檔案格式

`.mem` 為文字檔：每行一個 hex word，無位址，最高位數字在前，負數用固定寬度二補數。

| 檔案 | 位寬 | 行數 | 內容 |
|---|---:|---:|---|
| `metadata.mem` | 16 | 全層 kept 總數 | 第 2.2 節 |
| `packed_weights.mem` | 64 | 全層 kept 總數 | 第 2.2 節 |
| `cluster_descriptors.mem` | 64 | cluster 數 | 第 5.3 節 |
| `filter_indices.mem` | 16 | 8 × cluster 數 | 每 lane 的 output channel，空 lane 填 `0xFFFF` |
| `bias_int32.mem` | 32 | 8 × cluster 數 | signed INT32 |
| `multiplier_q15.mem` | 16 | 8 × cluster 數 | Q15，有效值落在 `[2^14, 2^15)` |
| `shift_q15.mem` | 8 | 8 × cluster 數 | signed `shift_int8`（**尚不是** `right_shift`） |

### 5.3 Cluster descriptor（64-bit）

```text
[15:0]  filter_base         該 cluster 第一個 output channel，對齊 8
[31:16] kept_count          可為 0
[47:32] data_offset         本層 metadata/weight 的累積起點
[55:48] valid_filter_mask   有效 PE columns
[63:56] reserved            0
```

- **Reload 模式：** 每個 cluster 開始前，把 `data_offset .. data_offset+kept_count-1` 載入 SRAM 位址 `0 .. kept_count-1`，並給 `cluster_data_offset_i=0`。
- **Resident 模式：** 整層一次載入，`cluster_data_offset_i=data_offset`，每個 Tile 從該位址重播 q。
- 每個 Tile 都重播同一組 q，host 不必重新載入權重。

### 5.4 層級資訊

`layer.json` 提供 RTL 命令所需欄位：

| 欄位 | 對應 RTL 輸入 |
|---|---|
| `input_shape_chw` | `cin_i`、`hin_i`、`win_i` |
| `output_shape_chw` | `cout_i`、`hout_i`、`wout_i` |
| `kernel`、`stride`、`padding` | `kernel_i`、`stride_i`、`padding_i` |
| `clusters[i].kept_count`、`valid_filter_mask`、`data_offset` | 逐 cluster 命令 |

另有 `input_scale`、`output_scale`、`fraction_bits` 和各檔 SHA-256。

硬體規格內的限制：`groups=1`、kernel 1×1 或 3×3、stride 1 或 2、padding 0 或 1、batch 1、`Tm=8`。

`network_schedule.json` 的每個 block 記錄 `conv1`、`conv2`、`downsample`、兩端 scale 和 `identity_requantization`，規定順序為 `conv2_requant + residual_requant, then ReLU and INT8 saturation`。

### 5.5 `kept_count=0` 的語意

Cluster 不跑 MAC，但輸出仍要存在：

```text
accumulator = 0
一般層    ：output = post_process(bias)
Residual  ：output = requantize(bias) + residual，再 ReLU 與飽和
```

直接輸出 0 是錯的。Data Feeder、weight fetch 和 PE array 都不需要啟動。

### 5.6 V2 量化參數介面

V2 的 host 介面以 output channel 為位址：

```text
host_quant_param_addr_i = output_channel
內部 54-bit word = {right_shift[5:0], multiplier[15:0], bias[31:0]}
right_shift = 15 - shift_int8       合法範圍 0..48
```

三個 seeds 的 Q15 package 共 25,605 個參數，`shift_int8` 為 −33 到 +4，`right_shift` 為 11 到 48，沒有左移案例。硬體應拒絕 `shift_int8 > 15`。

### 5.7 驗證方式

```bash
cd ../mask_aware_conv_accelerator
python scripts/validate_pruned_layer_package.py exports/resnet34_seed0_r0p8_l1_contiguous_q15
python scripts/validate_real_layer_golden.py testdata/resnet34_seed0_r0p8_q23_sample0
```

現有逐層 golden vectors 是 **Q23 + similarity** 的舊資料，不適用於 Q15 contiguous package，需重新產生。RTL 驗收標準是與 Python 參考逐位元相同，mismatch 為 0。

## 6. 接上真實模型前的缺口

硬體現況依 `main`（`de02bd4`，2026-09-21）。軟體匯出成果在分支 `software-q15-package`，已合併 `de02bd4`，Python 測試通過，RTL 模擬未重跑。

| # | 缺口 | 狀態 | 依據與剩餘工作 |
|---:|---|---|---|
| 1 | `kept_count=0` | **大致完成** | V2.3 已有 bias-only bypass，`tb_activation_ab_bias_only` 涵蓋。`q=0` 都在 layer4.2：`conv1` 的 3/7/4 個可直接用；`conv2` 的 1/2/4 個在 residual branch，要等 #5 |
| 2 | 參數檔不一致 | **軟體完成** | [convert_package_to_v2_params.py](../mask_aware_conv_accelerator/scripts/convert_package_to_v2_params.py)，結果在 `exports/v2_params_r0p8_contiguous/`。identity requant 尚未轉換，RTL 載入尚未測試 |
| 3 | SRAM 深度 | **待查證** | 硬體端回報已有 Host 空間分塊原型（ResNet-20、`MEMORY_DEPTH=1024`、210 次 CONV 全部 bit-exact），但 `de02bd4` 找不到相關程式，TASK_BOARD 的 tiling 也未勾選。36 層中有 8 層每 bank 需 8,192 words |
| 4 | Resident 容量 | 未解決 | 單一 cluster 最大 1,809 筆放得進 4096，單層最多 60,762 筆放不下。大層先用 reload |
| 5 | Residual add | **未完成** | TASK_BOARD 未勾選。後處理目前直接飽和成 INT8，不符合第 4.3 節。ResNet-20 hybrid 的接法要查，但 `de02bd4` 找不到相關檔案 |
| 6 | 多層串接 | **部分完成** | Activation A/B 已完成，兩層串接 bit-exact。Layer descriptor RAM 與 multi-layer sequencer 未完成 |
| 7 | Pooling / classifier | 未完成 | 硬體實作，或先由軟體處理 |
| 8 | 匯出器僅支援 ResNet34 | 未解決 | `export_pruned_layer_package.py` 明確拒絕其他模型。硬體若要跑 ResNet50，需支援 Bottleneck 並重做 bit-exact 驗證 |
| 9 | Q15 golden | 未完成 | 重新產生 contiguous 逐層向量（第 5.7 節） |

`filter_indices` remap **不需要**：contiguous 的 lane `j` 就是 `filter_base + j`。[SOFTWARE_HARDWARE_INTERFACE_GAPS.md](../mask_aware_conv_accelerator/docs/SOFTWARE_HARDWARE_INTERFACE_GAPS.md) 結尾建議「改用 contiguous」的段落已過時。`similarity_q15_backbone` 的 README 也還列著 remap，那是舊方案。

所有欄位目前都放得下：`data_offset` 最大 60,762（上限 65,535），ResNet50 的 `Cin=2048` 也在 12-bit `input_channel` 內。

## 7. 待決定事項

1. **是否採用 `group`：** 它通過 H2、H4，但硬體 package 尚未使用，也沒有 ResNet34 或 INT8 數據。若要採用，mask、`q`、`layer.json` 都要重新產生，週期模型也要重跑。
2. **是否把 rewinding 帶進硬體 package：** ResNet34 的 mask 和 metadata 不變，但要重訓、重新校正並重新匯出權重與參數。ResNet34 上還沒有 rewinding 數據。
3. **論文評估主體（依教授建議，已採用）：** 硬體評估以 ResNet34 / 80% 為主；ResNet50 / 95% 作為「高稀疏度下的架構延伸」，第三個 bank 的論證放在該段。ResNet50 / 95% 沒有 INT8 驗證，只能報架構與週期模型的結果。
4. **新模型的 INT8 驗證：** B、rewinding、`group` 都會改變權重範圍，現有量化結果不能外推。

下一步：硬體端用真實 `kept_count` 跑週期模型，軟體端補 identity requant 轉換和 Q15 golden。目前還沒有任何真實模型的 RTL cycle、SRAM 存取或功耗數據。

## 8. 來源

| 主題 | 文件 |
|---|---|
| 研究總檔 | [RESEARCH_FINDINGS_MASTER_2026-11-05.md](research_notes/RESEARCH_FINDINGS_MASTER_2026-11-05.md) |
| 95% B 判定 | [RESULT.md](runs/compensation_r0p95/RESULT.md) |
| 救回實驗 | [PLAN.md](runs/retrain_recovery_r0p95/PLAN.md)、[RESULT.md](runs/retrain_recovery_r0p95/RESULT.md) |
| 匯出器 | [export_pruned_layer_package.py](../mask_aware_conv_accelerator/scripts/export_pruned_layer_package.py) |
| 記憶體格式 | [memory_layout.json](../mask_aware_conv_accelerator/spec/memory_layout.json)、[MEMORY_LAYOUT_V1.md](../mask_aware_conv_accelerator/docs/MEMORY_LAYOUT_V1.md) |
| 數值契約 | [QUANTIZATION_V1.md](../mask_aware_conv_accelerator/docs/QUANTIZATION_V1.md)、[QUANTIZATION_V2.md](../mask_aware_conv_accelerator/docs/QUANTIZATION_V2.md) |
| Shift 範圍 | [results/contiguous_q15_shift_range/README.md](../mask_aware_conv_accelerator/results/contiguous_q15_shift_range/README.md) |
