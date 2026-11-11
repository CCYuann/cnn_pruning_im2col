# 交接：計畫 T（95% 的跨模型驗證）

寫給接手的 AI 代理。本文件可單獨閱讀，不需要先前的對話。
交接時間：2026-11-10（CST）。上一份交接是 `research_notes/HANDOFF_2026-11-09.md`（計畫 S），環境與規則相同，細節可參考該檔第 3、8 節。

## 1. 背景（一段話）

CNN Cross-Filter Structured Pruning 與自製 mask-aware 卷積加速器（V3）的軟硬體協同研究。
每 8 個 filter 一組（cluster）共用一個輸入位置 mask，每組保留 q 個位置。
原論文（Pham et al., IEEE Access 2025）一次剪完再微調，精度只報到 90%。
我們的方法是 **gradual + floor**：前 2/3 epochs 以三次曲線逐步剪、每 2 epochs 更新 mask、每個 cluster 至少保留 1 個位置。

使用者的主線：**以 90% 為基準，看做法能否推廣到其他模型；現在往上看 95%。** 98–99% 只是延伸，不是主線。

計畫 S（90%，CIFAR-100，8 個模型，seeds 0/1/2）的結果：
- 對原論文流程 ft_const：8/8 通過（+1.3 到 +5.7pp）。
- 對 rewind：4/8 通過（ResNet-56、ResNet-110、GoogLeNet、DenseNet121）；ResNet18、ResNet34、ResNet50、VGG16-BN 未過。
- 詳見 `runs/other_datasets/PLAN_S.md`。

## 2. 你的任務

1. 讀 `runs/other_datasets/PLAN_T.md`。計畫與門檻已凍結，**不要修改**。
2. 確認佇列 daemon 還活著（第 3 節），然後排入工作：
   ```
   cd /home/yenmh3/11457019/cnn_pruning_im2col
   /usr/local/anaconda3/bin/python3 runs/breakthrough_r0p95/enqueue_t.py
   ```
   應輸出 `39 queued`。重跑是安全的，已存在的名稱不會重複排入。
3. 等 39 條跑完（第 4 節）。
4. 執行判定：`/usr/local/anaconda3/bin/python3 scripts/decide_plan_t.py`，把輸出另存成 `runs/other_datasets/PLAN_T_DECISION.txt`。
5. 把結果寫進 `PLAN_T.md` 檔尾的「## 結果」，並更新統整文件（第 6 節）。
6. 用繁體中文向使用者回報（第 7 節）。
7. 不要 commit 或 push，除非使用者要求。

## 3. 環境與佇列

- 軟體 repo：`/home/yenmh3/11457019/cnn_pruning_im2col`（branch `research-2026-11`），所有指令都在這裡執行。
- Python：`/usr/local/anaconda3/bin/python3`。兩張 L40S，每張同時跑 4 條。
- 佇列 daemon：`scripts/gpu_queue.py`（PID 254618），**2026-11-11 01:25 會自動結束**。檢查：`ps -p 254618`。
  若已結束，重啟：
  ```
  nohup /usr/local/anaconda3/bin/python3 scripts/gpu_queue.py --queue runs/breakthrough_r0p95/queue.txt --logs runs/breakthrough_r0p95/logs --per-gpu 4 --hours 40 --cwd . > runs/breakthrough_r0p95/queue5.out 2>&1 &
  ```
  已有 `.rc` 的工作不會重跑，所以重啟不會重複訓練。
- 佇列檔 `runs/breakthrough_r0p95/queue.txt`；日誌 `runs/breakthrough_r0p95/logs/<名稱>.log`，完成後寫 `<名稱>.rc`（0 = 成功）。
- 磁碟：共用，約 40G 可用。低於 5G 時依 `HANDOFF_2026-11-09.md` 第 8 節的規則清理，並記錄在 `runs/DELETED_FILES_2026-11-08.md`。

## 4. 進度與預估

```
echo "$(ls runs/breakthrough_r0p95/logs/t_*.rc 2>/dev/null | wc -l)/39"
grep -L "^0$" runs/breakthrough_r0p95/logs/t_*.rc     # 列出失敗的工作
```
單條耗時約 8–42 分鐘（DenseNet121、ResNet50 最慢）。8 條並行時，全部約 3–4 小時。
失敗的工作：先看 log 找原因；若是暫時性錯誤（OOM、磁碟滿），刪掉該 `.rc` 讓 daemon 重跑。不要改任何訓練參數。

結果目錄：`runs/other_datasets/cifar100_<模型>_seed<n>_<ft_const_r95|rewind_r95|gradual_r95_floor>/result.json`，`final_test` 是 test 精度（0–1）。

## 5. 判定重點（已寫在腳本中）

- **T1（主判定）**：gradual+floor 對 ft_const。門檻：配對平均 >= +0.3pp 且 3/3 為正。
- **T2**：gradual+floor 對 rewind。**只有 ResNet18、ResNet-110 是新的 test**；其餘 6 個模型的結果之前已讀過（腳本標 `[already read]`），報告時要說明它們是引用，不是新的驗證。
- **T3**：ResNet-56、VGG16-BN seeds 3/4/5 的 T1 重現。
- **T4**：90% → 95% 的變化，只描述。腳本每行會附 `[90%: ...]`。

交接時已知（test 已讀過，可直接引用）：
- ResNet34 T1 +2.83（通過，90% 時 +1.45）；T2 −0.05（未過）。
- T2：ResNet50 +0.08（未過）、ResNet-56 +4.16、VGG16-BN +2.14、GoogLeNet +2.67、DenseNet121 +6.91（皆通過）。
- seeds 3/4/5 的 T2：ResNet-56 +4.07、VGG16-BN +1.41（皆通過）。
- 注意：VGG16-BN 在 90% 時 T2 未過，在 95% 通過，因此「對 rewind 的優勢隨剪枝率變大」是事後觀察，報告時要標明。

### 一定要寫進報告的注意事項
- **週期不一定相同。** 95% 時 GoogLeNet 的 gradual+floor 比 rewind 多約 3%（12.55 → 12.98 ms），DenseNet121 多約 16%（20.06 → 23.33 ms）。這兩個模型的增益不是在同週期下取得的。ResNet18、ResNet-110 跑完後也要檢查。
- **ft_const 的 LR 0.001 是我們的假設**，原論文沒給 LR。
- **週期都是 V3 公式預測**（每個 cluster 31 + Hout·ceil(Wout/8)·max(q,15) 拍，50 MHz，只算卷積層），不是 RTL 實測。
- **只有浮點精度。** 新模型沒有 Q15 流程；worker 在 CUDA 用 AMP float16 評估。
- 負面結果照實報告，事後的想法要標明。

## 6. 要更新的文件

1. `runs/other_datasets/PLAN_T.md`：檔尾加「## 結果」。格式比照 `PLAN_S.md` 的結果段落：
   - 完成時間、條數、失敗數；
   - 主表：模型、Dense、ft_const、rewind、gradual+floor、對 Dense 掉點、T1、T2（標出哪些是已讀過的）、V3 ms（rewind → floor）；
   - seeds 3/4/5 重現表；
   - 90% → 95% 對照表（T4）；
   - 第 5 節的注意事項。
2. `research_notes/EXPERIMENT_SUMMARY_2026-11-07.md`：在計畫 S 的 5.5 節之後新增 5.6「95% 跨模型（計畫 T）」，第 1 節結論摘要補一條，第 12 節檔案索引加 `PLAN_T.md`。
3. `research_notes/RESEARCH_FINDINGS_MASTER_2026-11-05.md`：在計畫 S 那一條之後新增一條計畫 T。
4. 若計畫 S 的文件更新（上一份交接的第 6 節）尚未完成，先完成那部分。

## 7. 回報使用者

繁體中文，結論先講：T1 幾個模型通過、T2 的新結果（ResNet18、ResNet-110），以及 90% → 95% 的差距變化。表格精簡，數字附 seeds 與 n，不確定的事不要寫成確定。

## 8. 工作規則（摘要，完整版見 `HANDOFF_2026-11-09.md` 第 8 節）

- 計畫與門檻已凍結；每個模型的 test 只讀一次；不為了過門檻調參或重跑。
- 不要 commit 或 push，除非使用者明確要求。
- 不要修改硬體設計文件（claude.ai 上的 V3 文件）。
- 不要執行使用者 2026-11-06 誤貼的 Phase 2 / FIFO / Quartus 指示。
- 不可刪：Dense checkpoint、`result.json`、`masks.pt`、計畫檔、報告與 Q15／RTL 會用到的模型。

## 9. 本次新增、尚未 commit 的檔案

- `runs/other_datasets/PLAN_T.md`（凍結的計畫）
- `runs/breakthrough_r0p95/enqueue_t.py`（排入 39 條）
- `scripts/decide_plan_t.py`（判定；已用現有資料試跑，新組別顯示 pending，運作正常）
- 本交接文件
